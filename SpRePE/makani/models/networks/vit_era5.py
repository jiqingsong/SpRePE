import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
import itertools

import torch.utils.checkpoint as checkpoint
# mp stuff
from makani.utils import comm
from makani.models.common import DropPath, MLP, PatchEmbed
from makani.mpu.layers import DistributedMatmul, DistributedMLP, DistributedAttention
from makani.models.common import SphericRope_vit, AxialRope, MixRope

def coslat_weights(H: int, W: int, device=None, top_is_north: bool=True):
    if H == 1:
        phi = torch.zeros(1, device=device) 
    else:
        # ϕ_i ∈ [+π/2, -π/2]
        rows = torch.arange(H, device=device)
        if top_is_north:
            phi = math.pi/2 - rows * (math.pi/(H-1))
        else:
            phi = -math.pi/2 + rows * (math.pi/(H-1))
    w_lat = torch.cos(phi).clamp_min(1e-6)     
    w_tokens = w_lat.repeat_interleave(W)      
    return w_tokens

def build_mask_prefix_on_D(H: int, W: int, D: int, *,
                           top_is_north: bool = True,
                           dtype=torch.float64):
    if H == 1:
        phi = torch.zeros(1, dtype=dtype)
    else:
        rows = torch.arange(H, dtype=dtype)
        step = math.pi / (H - 1)
        phi = (math.pi/2 - rows * step) if top_is_north \
              else (-math.pi/2 + rows * step)

    p = torch.cos(phi).clamp(0.0, 1.0)  # [H]

    D1 = D // 2
    D2 = D - D1

    first_half = torch.ones(H, D1, dtype=dtype)

    k2 = torch.floor(p * D2 + 1e-8).to(torch.long)  # [H]
    ar2 = torch.arange(D2).view(1, D2)              # [1, D2] (int64)
    second_half = (ar2 < k2.view(H, 1)).to(dtype)   # [H, D2]

    M_hD = torch.cat([first_half, second_half], dim=1)  # [H, D]

    M_hWD = M_hD[:, None, :].expand(H, W, D)            # [H, W, D]
    M = M_hWD.reshape(1, 1, H * W, D).contiguous().to(dtype)
    return M

def get_alibi(attention_heads, patch_l, patch_w):
    coords_h = torch.arange(patch_l)
    coords_w = torch.arange(patch_w)
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    points = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # shape: [N, 2], where N = patch_l * patch_w

    diff = points.unsqueeze(1) - points.unsqueeze(0)  # [N, N, 2]
    dist_matrix = torch.norm(diff.float(), dim=-1)  # [N, N]

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(attention_heads)).view(attention_heads, 1, 1)  # [H, 1, 1]

    bias = -slopes * dist_matrix  # [H, N, N]

    return bias.unsqueeze(0)  # [1, H, N, N]

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        input_format="traditional",
        num_heads=8,
        qkv_bias=False,
        qk_norm=True,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        num_patch=None,
        pos_embed_scheme='ape'
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.last_attn = None
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()
        
        self.if_rope = pos_embed_scheme=='sprepe'
        self.if_axialrope = pos_embed_scheme=='axial_rope'
        self.if_mixrope = pos_embed_scheme=='mix_rope'
        self.if_relative_pe = pos_embed_scheme=='rpe'
        self.if_alibi = pos_embed_scheme=='alibi'
        self.if_rope_axial =  pos_embed_scheme=='sprepe_axial_rope'
        self.if_sprepe_ape =  pos_embed_scheme=='sprepe_ape'
        if self.if_sprepe_ape:
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, (128, 256), (2, 2), (2, 2))
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # const mask
            mask = torch.ones(self.head_dim)
            mask[self.head_dim//8:] = 0.
            mask = mask.view(1,1,1,self.head_dim)
            
            # coslat mask
            # mask = coslat_weights(64, 128)
            # # mask = mask.clamp_min(1e-3).view(1, 1, 64*128, 1)
            # mask = build_mask_prefix_on_D(64, 128, 48)
            self.register_buffer("mask", mask)

        if self.if_rope_axial:
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim // 2, (128, 256), (2, 2), (2, 2))
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.axial_rope = AxialRope((64, 128), self.head_dim // 2)

        if self.if_rope:
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, (128, 256), (2, 2), (2, 2))
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # print(self.spheric_rope.midpoint_theta._version)
        
        if self.if_axialrope:
            self.softmax = nn.Softmax(dim=-1)
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.axial_rope = AxialRope((64, 128), self.head_dim)

        if self.if_mixrope:
            self.mix_rope = MixRope(dim = dim, num_heads=num_heads, shape=[64, 128])
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)

        if self.if_relative_pe:
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * 64 - 1) * (2 * 128 - 1), self.num_heads))
            coords_h = torch.arange(64)  
            coords_w = torch.arange(128) 
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  
            relative_coords[:, :, 0] += 64 - 1  
            relative_coords[:, :, 1] += 128 - 1  
            relative_coords[:, :, 0] *= 2 * 128 - 1  
            relative_position_index = relative_coords.sum(-1)  
            self.register_buffer("relative_position_index", relative_position_index)  

        if self.if_alibi:
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)


    def forward(self, x, alibi=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        flag = 0
        if self.if_sprepe_ape:
            sprepe_q, sprepe_k = self.spheric_rope(q, k) 
            q = sprepe_q * self.mask + q * (1 - self.mask)
            k = sprepe_k * self.mask + k * (1 - self.mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  
            attn = self.softmax(attn)
            self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype) 
            x = (attn @ v)
            flag = 1

        if self.if_rope_axial:
            q1 = q[:, :, :, :self.head_dim // 2]
            q2 = q[:, :, :, self.head_dim // 2:]
            k1 = k[:, :, :, :self.head_dim // 2]
            k2 = k[:, :, :, self.head_dim // 2:]
            q1, k1 = self.spheric_rope(q1, k1)
            q2 = self.axial_rope(q2)
            k2 = self.axial_rope(k2)
            q  = torch.cat([q1, q2], dim=-1).to(dtype=v.dtype)
            k  = torch.cat([k1, k2], dim=-1).to(dtype=v.dtype)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)
            flag = 1

        if self.if_rope:
            q, k = self.spheric_rope(q, k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  
            attn = self.softmax(attn)
            self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  
            x = (attn @ v)
            flag = 1

        if self.if_axialrope:
            q = self.axial_rope(q)
            k = self.axial_rope(k)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  
            attn = self.softmax(attn)
            self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)
            x = (attn @ v)
            flag = 1
        
        if self.if_mixrope:
            q, k = self.mix_rope(q, k)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            flag = 1

        if self.if_relative_pe:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                64 * 128, 64 * 128, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            flag = 1
        if self.if_alibi:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn + alibi
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v)
            flag = 1

        if flag == 0:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        comm_inp_name="fin",
        comm_hidden_name="fout",
        pos_embed_scheme='ape'
    ):
        super().__init__()

        if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
            self.attn = DistributedAttention(
                dim,
                input_format="traditional",
                comm_inp_name=comm_inp_name,
                comm_hidden_name=comm_hidden_name,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=mlp_drop_rate,
                norm_layer=norm_layer,
            )
        else:
            self.attn = Attention(
                dim, input_format="traditional", num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rate=mlp_drop_rate, norm_layer=norm_layer,pos_embed_scheme=pos_embed_scheme
            )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
            self.mlp = DistributedMLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                input_format="traditional",
                comm_inp_name=comm_inp_name,
                comm_hidden_name=comm_hidden_name,
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop_rate=mlp_drop_rate, input_format="traditional")

    def forward(self, x, alibi=None):
        y = self.attn(self.norm1(x), alibi)
        x = x + self.drop_path(y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        inp_shape=[224, 224],
        patch_size=(16, 16),
        inp_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        norm_layer="layer_norm",
        comm_inp_name="fin",
        comm_hidden_name="fout",
        pos_embed_scheme='ape',
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = inp_shape
        self.out_ch = out_chans
        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=inp_chans, embed_dim=self.embed_dim)
        self.pos_embed_scheme=pos_embed_scheme
        num_patches = self.patch_embed.num_patches

        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_embed.is_shared_mp = []
        
        
        if self.pos_embed_scheme == 'alibi':
            self.register_buffer("alibi", get_alibi(num_heads, 64, 128))
        else:
            self.alibi = None
        self.pos_drop = nn.Dropout(p=path_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]

        if norm_layer == "layer_norm":
            norm_layer_handle = nn.LayerNorm
        else:
            raise NotImplementedError(f"Error, normalization layer type {norm_layer} not implemented for ViT.")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop_rate=mlp_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=dpr[i],
                    norm_layer=norm_layer_handle,
                    comm_inp_name=comm_inp_name,
                    comm_hidden_name=comm_hidden_name,
                    pos_embed_scheme=pos_embed_scheme
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer_handle(embed_dim)

        self.out_size = self.out_ch * self.patch_size[0] * self.patch_size[1]

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)
        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape':
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).transpose(1, 2)  # patch linear embedding

        # add positional encoding to each token
        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape':
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        B, _, _ = x.shape  # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.red_img_size[0], self.patch_embed.red_img_size[1], self.embed_dim)
        B, h, w, _ = x.shape

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size[0], self.patch_size[1], self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], self.img_size[1]))

        return x

    def forward(self, x):
        # res = x[:, :70]
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            # x = checkpoint.checkpoint(blk, x, self.alibi)
            x = blk(x, self.alibi)
        x = self.norm(x)
        x = self.forward_head(x)
        # x = res + x
        return x
