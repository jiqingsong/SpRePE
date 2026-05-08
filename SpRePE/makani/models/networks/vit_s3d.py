import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
import itertools

import numpy as np
import torch.utils.checkpoint as checkpoint
# mp stuff
from makani.utils import comm
from makani.models.common import DropPath, MLP, PatchEmbed
from makani.mpu.layers import DistributedMatmul, DistributedMLP, DistributedAttention
from makani.models.common import SphericRope_vit, AxialRope, MixRope, SphereGridMixScaleSpatialRelationEncoder
imgs = (256, 512)
ps = (8, 8)
pn = (imgs[0] // ps[0], imgs[1] // ps[1])
@torch.no_grad()
def resize_pos_embed(
    pos_embed: torch.Tensor,      # (1, T_extra + H0*W0, C)
    old_hw: tuple[int, int],
    new_hw: tuple[int, int],
    num_extra_tokens: int = 0,
    mode: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    assert pos_embed.dim() == 3 and pos_embed.size(0) == 1
    C = pos_embed.size(-1)
    H0, W0 = old_hw
    H1, W1 = new_hw

    if num_extra_tokens > 0:
        extra = pos_embed[:, :num_extra_tokens, :]      # (1, T_extra, C)
        pos_tokens = pos_embed[:, num_extra_tokens:, :] # (1, H0*W0, C)
    else:
        extra = None
        pos_tokens = pos_embed

    assert pos_tokens.size(1) == H0 * W0

    pos_2d = pos_tokens.reshape(1, H0, W0, C).permute(0, 3, 1, 2)  # (1,C,H0,W0)
    pos_2d = F.interpolate(
        pos_2d,
        size=(H1, W1),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
        antialias=antialias if mode in ("bilinear", "bicubic") else False,
    )
    pos_new = pos_2d.permute(0, 2, 3, 1).reshape(1, H1 * W1, C)    # (1,H1W1,C)

    if extra is not None:
        pos_new = torch.cat([extra, pos_new], dim=1)

    return pos_new

def build_2d_ring_sincos_pos_embed(
    embed_dim: int,
    grid_size_h: int,
    grid_size_w: int,
    num_prefix_tokens: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Literal 2D Ring PE from the formula in your screenshot.

    Formula implemented (0-based code indexing):
        PE[k, 4j + 0] = sin((p_x / w) * 2pi * i) * w / (4i)
        PE[k, 4j + 1] = cos((p_x / w) * 2pi * i) * w / (4i)
        PE[k, 4j + 2] = sin((p_y / w) * 2pi * i) * w / (4i)
        PE[k, 4j + 3] = cos((p_y / w) * 2pi * i) * w / (4i)

    where:
        i = 1, 2, ..., embed_dim // 4
        j = i - 1 in 0-based code
        p_x in {0, ..., w-1}, p_y in {0, ..., h-1}

    Returns:
        pos_embed: [1, num_prefix_tokens + H*W, embed_dim]
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")

    device = device if device is not None else torch.device("cpu")

    H = grid_size_h
    W = grid_size_w
    N = H * W
    n_freq = embed_dim // 4

    # i = 1, 2, ..., D/4
    i = torch.arange(1, n_freq + 1, device=device, dtype=dtype)  # [D/4]

    # token grid positions, row-major flatten
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(y, x)

    px = grid_x.reshape(-1)  # [N]
    py = grid_y.reshape(-1)  # [N]

    # Literal amplitude in the formula: w / (4i)
    amp = (W / (4.0 * i)).unsqueeze(0)  # [1, D/4]

    # Literal angles in the formula
    ang_x = (px.unsqueeze(1) / W) * (2.0 * math.pi * i.unsqueeze(0))  # [N, D/4]
    ang_y = (py.unsqueeze(1) / W) * (2.0 * math.pi * i.unsqueeze(0))  # [N, D/4]

    pos = torch.empty(N, embed_dim, device=device, dtype=dtype)
    pos[:, 0::4] = torch.sin(ang_x) * amp
    pos[:, 1::4] = torch.cos(ang_x) * amp
    pos[:, 2::4] = torch.sin(ang_y) * amp
    pos[:, 3::4] = torch.cos(ang_y) * amp

    pos = pos.unsqueeze(0)  # [1, N, C]

    if num_prefix_tokens > 0:
        prefix = torch.zeros(
            1, num_prefix_tokens, embed_dim,
            device=device, dtype=dtype
        )
        pos = torch.cat([prefix, pos], dim=1)

    return pos

def coslat_weights(H: int, W: int, device=None, top_is_north: bool=True):
    # 生成纬度（弧度）。假设 i=0 在图像顶部为北极，i=H-1 为南极
    if H == 1:
        phi = torch.zeros(1, device=device)  # 退化情形
    else:
        # ϕ_i ∈ [+π/2, -π/2]
        rows = torch.arange(H, device=device)
        if top_is_north:
            phi = math.pi/2 - rows * (math.pi/(H-1))
        else:
            phi = -math.pi/2 + rows * (math.pi/(H-1))
    w_lat = torch.cos(phi).clamp_min(1e-6)     # 防止在极点为 0
    w_tokens = w_lat.repeat_interleave(W)      # 展平成长度 N=H*W 的权重
    return w_tokens

def build_mask_prefix_on_D(H: int, W: int, D: int, *,
                           top_is_north: bool = True,
                           dtype=torch.float64):
    """
    为每个纬度 i 计算 p_i = cos(phi_i)。在最后一维 D 上：
      - 前一半通道（D//2）恒为 1；
      - 后一半通道按比例置 1：前 floor(p_i * (D - D//2)) 个通道为 1，其余为 0。
    同一纬度的所有经度共享同一条 D 维掩码。
    返回: mask [1, 1, H*W, D]，0/1 浮点。
    """
    # 计算各纬度的纬度角（弧度）：北极 +π/2 → 南极 -π/2
    if H == 1:
        phi = torch.zeros(1, dtype=dtype)
    else:
        rows = torch.arange(H, dtype=dtype)
        step = math.pi / (H - 1)
        phi = (math.pi/2 - rows * step) if top_is_north \
              else (-math.pi/2 + rows * step)

    # p = cos(phi) ∈ [0,1]；赤道=1，极点=0
    p = torch.cos(phi).clamp(0.0, 1.0)  # [H]

    # 拆分为前半固定1，后半按比例
    D1 = D // 2
    D2 = D - D1

    # 前半：[H, D1] 全 1
    first_half = torch.ones(H, D1, dtype=dtype)

    # 后半：[H, D2]，每行前 k2[i] 个为 1
    k2 = torch.floor(p * D2 + 1e-8).to(torch.long)  # [H]
    ar2 = torch.arange(D2).view(1, D2)              # [1, D2] (int64)
    second_half = (ar2 < k2.view(H, 1)).to(dtype)   # [H, D2]

    # 拼接得到 [H, D]
    M_hD = torch.cat([first_half, second_half], dim=1)  # [H, D]

    # 扩展到所有经度并重排到 [1,1,H*W,D]
    M_hWD = M_hD[:, None, :].expand(H, W, D)            # [H, W, D]
    M = M_hWD.reshape(1, 1, H * W, D).contiguous().to(dtype)
    return M

def get_alibi(attention_heads, patch_l, patch_w):
    # 1. 生成 patch grid 坐标 (H*W, 2)
    coords_h = torch.arange(patch_l)
    coords_w = torch.arange(patch_w)
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    points = torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)  # shape: [N, 2], where N = patch_l * patch_w

    # 2. 计算所有点对之间的欧几里得距离矩阵
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # [N, N, 2]
    dist_matrix = torch.norm(diff.float(), dim=-1)  # [N, N]

    # 3. 生成 slopes
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

    # 4. 扩展 distance 矩阵到 [H, N, N] 并加上 slope 衰减
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
        self.if_rope_fix_grid = pos_embed_scheme=='sprepe_fix_grid'
        self.if_rope_fix_same = pos_embed_scheme=='sprepe_fix_same'
        self.if_rope_fix_random = pos_embed_scheme=='sprepe_fix_random'
        self.if_axialrope = pos_embed_scheme=='axial_rope'
        self.if_axialrope_ape = pos_embed_scheme=='axial_rope_ape'
        self.if_mixrope = pos_embed_scheme=='mix_rope'
        self.if_relative_pe = pos_embed_scheme=='rpe'
        self.if_alibi = pos_embed_scheme=='alibi'
        self.if_rope_axial =  pos_embed_scheme=='sprepe_axial_rope'
        self.if_sprepe_ape =  pos_embed_scheme=='sprepe_ape'
        self.if_sprepe_ape_learnable = pos_embed_scheme=='sprepe_ape_learnable'
        self.if_sprepe_sphere2vec_learnable = pos_embed_scheme=='sprepe_sphere2vec_learnable'
        self.if_sprepe_learnable = pos_embed_scheme=='sprepe_learnable'
        self.if_sprepe_share_learnable = pos_embed_scheme=='sprepe_share'
        self.if_sprepe_share_ape_learnable = pos_embed_scheme=='sprepe_share_ape_learnable'
        self.if_sprepe_share_sphere2vec = pos_embed_scheme == 'sprepe_share_sphere2vec'

        if self.if_axialrope_ape:
            print('with ape + rope mask same, init with 0.875', flush=True)
            # self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, (128, 256), ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.axial_rope = AxialRope((imgs[0] // ps[0], imgs[1] // ps[1]), self.head_dim, origin_shape=[pn[0], pn[1]])
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.875) 

        if self.if_sprepe_share_sphere2vec:
            print('with sphere2vec + sprepe *share* learnable mask same, init with 0.95', flush=True)
            # self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, (128, 256), ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.95)    

        if self.if_sprepe_share_ape_learnable:
            print('with ape + sprepe *share* learnable mask same, init with 0.875', flush=True)
            # self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, (128, 256), ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.875)        


        if self.if_sprepe_share_learnable:
            print('with sprepe share', flush=True)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)

        if self.if_rope_fix_grid:
            print('with rope fix grid', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps, init_strategy='grid', trainable=False)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
        
        if self.if_rope_fix_same:
            print('with rope fix same', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps, init_strategy='same', trainable=False)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
        
        if self.if_rope_fix_random:
            print('with rope fix random', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps, init_strategy='random', trainable=False)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)

        if self.if_sprepe_ape_learnable:
            print('with ape + sprepe learnable mask patchwise, init with 0.9', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.9)
        
        if self.if_sprepe_sphere2vec_learnable:
            print('with sphere2vec + sprepe learnable mask patchwise, init with 0.9', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.9)

        if self.if_sprepe_learnable:
            print('with sprepe learnable mask patchwise, init with 0.9', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # self.mask_logit = nn.Parameter(torch.zeros(1, 1, 64*128, self.head_dim))
            self.mask_logit = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
            self._init_mask(0.9)


        if self.if_sprepe_ape:
            print('with ape + sprepe', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # const mask
            mask = torch.ones(self.head_dim)
            mask[self.head_dim//8:] = 0.
            mask = mask.view(1,1,1,self.head_dim)
            self.register_buffer("mask", mask)

        if self.if_rope_axial:
            print('with rope + sprepe', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim // 2, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.axial_rope = AxialRope((imgs[0] // ps[0], imgs[1] // ps[1]), self.head_dim // 2)

        if self.if_rope:
            print('with rope', flush=True)
            self.spheric_rope = SphericRope_vit(self.num_heads, self.head_dim, imgs, ps, ps)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            # print(self.spheric_rope.midpoint_theta._version)

        if self.if_axialrope:
            print('with axial rope', flush=True)
            self.softmax = nn.Softmax(dim=-1)
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.axial_rope = AxialRope((imgs[0] // ps[0], imgs[1] // ps[1]), self.head_dim, origin_shape=[pn[0], pn[1]])

        if self.if_mixrope:
            print('with mix rope', flush=True)
            self.mix_rope = MixRope(dim = dim, num_heads=num_heads, shape=[imgs[0] // ps[0], imgs[1] // ps[1]])
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)

        if self.if_relative_pe:
            print('with relative pe', flush=True)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * pn[0] - 1) * (2 * pn[1] - 1), self.num_heads))
            coords_h = torch.arange(pn[0])  # 
            coords_w = torch.arange(pn[1])  # 
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 坐标，结构为 [2, Wh, Ww]
            coords_flatten = torch.flatten(coords, 1)  # 重构张量结构为 [2, Wh*Ww]
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
            relative_coords[:, :, 0] += pn[0] - 1  # 第1个维度移位
            relative_coords[:, :, 1] += pn[1] - 1  # 第1个维度移位
            relative_coords[:, :, 0] *= 2 * pn[1] - 1  # 第1个维度的值乘以 2倍的 Ww，再减 1
            relative_position_index = relative_coords.sum(-1)  # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
            self.register_buffer("relative_position_index", relative_position_index, persistent=False)  # 保存数据，不再更新

            self.base_Wh, self.base_Ww = pn[0], pn[1]  # 训练时 patch 网格（用于 bias table 插值）
            self._cached_hw = None

            # self.register_buffer("relative_position_index", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("relative_position_bias_table_resized", torch.empty(0), persistent=False)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
            # 额外：缓存插值表，避免重复插值（如果你会在多 dtype/device 上跑）
            self._rpe_table_cache = {}

        if self.if_alibi:
            print('with alibi', flush=True)
            self.softmax = nn.Softmax(dim=-1) 
            self.attn_drop = nn.Dropout(self.attn_drop_rate)
            print('with alibi init done', flush=True)

    def _init_mask(self, p: float):
        """
        p: 初始时希望的 mask 值（0~1 之间），
        p 越接近 1 越偏向 sprepe_q/k，p 越接近 0 越偏向原始 q/k。
        """
        # 避免 log(0) 或 log(1)
        eps = 1e-6
        p = min(max(p, eps), 1.0 - eps)
        logit = math.log(p / (1.0 - p))
        with torch.no_grad():
            self.mask_logit.fill_(logit)

    @torch.no_grad()
    def _update_rpe(self, Wh: int, Ww: int, device, dtype):
        if self._cached_hw == (Wh, Ww) and self.relative_position_index.numel() > 0:
            # 已经是当前分辨率，直接复用
            if self.relative_position_bias_table_resized.numel() > 0 and self.relative_position_bias_table_resized.dtype == dtype:
                return
            # 若 dtype 变化，bias 表也要对应 dtype（否则会触发隐式 cast 或报错）
            # 继续往下更新 bias 表

        # 1) 更新 index（注意：非常大）
        coords_h = torch.arange(Wh, device=device)
        coords_w = torch.arange(Ww, device=device)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, Wh, Ww]
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # [2, N]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]   # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()             # [N, N, 2]
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= (2 * Ww - 1)
        index = relative_coords.sum(-1).to(torch.long)                               # [N, N]

        # 替换 buffer（不会重复注册）
        self.relative_position_index = index

        # 2) 更新/缓存插值后的 bias table（只做一次）
        if (Wh, Ww) == (self.base_Wh, self.base_Ww):
            table = self.relative_position_bias_table.to(device=device, dtype=dtype)
        else:
            key = (Wh, Ww, device.type, device.index, str(dtype))
            if key in self._rpe_table_cache:
                table = self._rpe_table_cache[key]
            else:
                old_hw = (2 * self.base_Wh - 1, 2 * self.base_Ww - 1)
                new_hw = (2 * Wh - 1, 2 * Ww - 1)
                base_table = self.relative_position_bias_table.to(device=device, dtype=dtype)[None, :, :]  # (1,L,heads)
                table = resize_pos_embed(
                    pos_embed=base_table,
                    old_hw=old_hw,
                    new_hw=new_hw,
                    num_extra_tokens=0,
                    mode="bicubic",
                    antialias=True,
                )[0]  # (L_new, heads)
                self._rpe_table_cache[key] = table

        self.relative_position_bias_table_resized = table
        self._cached_hw = (Wh, Ww)

    def forward(self, x, alibi=None, share_sprepe=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # print(q.shape)
        q, k = self.q_norm(q), self.k_norm(k)
        # print(q.shape)
        flag = 0
        
        if self.if_axialrope_ape:
            # assert share_sprepe is not None

            q1 = self.axial_rope(q)
            k1 = self.axial_rope(k)
            mask = self.mask_logit.sigmoid()
            q = q1 * mask + q * (1 - mask)
            k = k1 * mask + k * (1 - mask)
            # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_share_sphere2vec:
            assert share_sprepe is not None

            q, k = share_sprepe(q, k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_share_learnable:
            assert share_sprepe is not None

            q, k = share_sprepe(q, k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            # x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_share_ape_learnable:
            assert share_sprepe is not None
            sprepe_q, sprepe_k = share_sprepe(q, k)
            mask = self.mask_logit.sigmoid()
            q = sprepe_q * mask + q * (1 - mask)
            k = sprepe_k * mask + k * (1 - mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_ape_learnable:
            sprepe_q, sprepe_k = self.spheric_rope(q, k)
            mask = self.mask_logit.sigmoid()
            q = sprepe_q * mask + q * (1 - mask)
            k = sprepe_k * mask + k * (1 - mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1
        
        if self.if_sprepe_sphere2vec_learnable:
            sprepe_q, sprepe_k = self.spheric_rope(q, k)
            mask = self.mask_logit.sigmoid()
            q = sprepe_q * mask + q * (1 - mask)
            k = sprepe_k * mask + k * (1 - mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_learnable:
            sprepe_q, sprepe_k = self.spheric_rope(q, k)
            mask = self.mask_logit.sigmoid()
            q = sprepe_q * mask + q * (1 - mask)
            k = sprepe_k * mask + k * (1 - mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_sprepe_ape:
            sprepe_q, sprepe_k = self.spheric_rope(q, k) 
            q = sprepe_q * self.mask + q * (1 - self.mask)
            k = sprepe_k * self.mask + k * (1 - self.mask)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
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

        if self.if_rope or self.if_rope_fix_grid or self.if_rope_fix_random or self.if_rope_fix_same:
            # print(self.spheric_rope.midpoint_theta._version)
            q, k = self.spheric_rope(q, k)
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_axialrope:
            q = self.axial_rope(q)
            k = self.axial_rope(k)
            
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            attn = self.softmax(attn)
            # self.last_attn = attn
            attn = self.attn_drop(attn).to(dtype=v.dtype)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1
        
        if self.if_mixrope:
            q, k = self.mix_rope(q, k)
            
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            flag = 1

        if self.if_relative_pe:
            # print('relative_position_bias_table', self.relative_position_bias_table[0, 0], flush=True)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度
            
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                (imgs[0]//ps[0]) * (imgs[1]//ps[1]), (imgs[0]//ps[0]) * (imgs[1]//ps[1]), -1)
            # print(relative_position_bias.shape)
            print(relative_position_bias[0,0])
            # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)  # 带相对位置偏移的注意力图
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            flag = 1
        if self.if_alibi:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn + alibi
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
            # 注意力图与 v 相乘得到新的注意力图
            x = (attn @ v)
            # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
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

        # distribute MLP for model parallelism
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

    def forward(self, x, alibi=None, share_sprepe=None):
        # flatten transpose:
        # y = checkpoint.checkpoint(self.attn, self.norm1(x), alibi, share_sprepe)
        y = self.attn(self.norm1(x), alibi, share_sprepe)
        x = x + self.drop_path(y)
        y2 = self.norm2(x)
        x = x + self.drop_path(self.mlp(y2))

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

        # annotate for distributed
        if self.pos_embed_scheme == 'sphere2vec' or self.pos_embed_scheme == 'sprepe_share_sphere2vec' or self.pos_embed_scheme == 'sprepe_sphere2vec_learnable':
            self.pos_embed = SphereGridMixScaleSpatialRelationEncoder(spa_embed_dim=embed_dim)
            self.patch_centers_lat, self.patch_centers_lon = self.pos_embed.compute_patch_center_coordinates(lat_lon=(180,360), origin_shape=self.img_size, patch_size=ps, stride=ps)
            self.patch_centers = np.stack([self.patch_centers_lon, self.patch_centers_lat], axis=1)[np.newaxis, :, :]
        
        if self.pos_embed_scheme == 'ringpe':
            print('init ringpe', flush=True)
            pos_embed = build_2d_ring_sincos_pos_embed(
                        embed_dim=embed_dim,
                        grid_size_h=imgs[0] // patch_size[0],
                        grid_size_w=imgs[1] // patch_size[1],
                        num_prefix_tokens=0,)
            self.register_buffer("pos_embed", pos_embed)

        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape' or self.pos_embed_scheme=='sprepe_ape_learnable' or self.pos_embed_scheme=='axial_rope_ape' or self.pos_embed_scheme == 'sprepe_share_ape_learnable':
            #  
            # self.pos_embed = nn.Parameter(torch.zeros(1, 8192, self.embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            self.pos_embed.is_shared_mp = []
        self.sprepe = None
        if self.pos_embed_scheme == 'sprepe_share' or self.pos_embed_scheme == 'sprepe_share_ape_learnable' or self.pos_embed_scheme == 'sprepe_share_sphere2vec':
            self.sprepe = SphericRope_vit(num_heads, embed_dim // num_heads, self.img_size, ps, ps, init_strategy='random', trainable=True)

        if self.pos_embed_scheme == 'alibi':
            # self.alibi = get_alibi(num_heads, 64, 128)
            self.register_buffer("alibi", get_alibi(num_heads, imgs[0] // ps[0], imgs[1] // ps[1]))
        else:
            self.alibi = None
        self.pos_drop = nn.Dropout(p=path_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]  # stochastic depth decay rule

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
        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape' or self.pos_embed_scheme=='sprepe_ape_learnable' or self.pos_embed_scheme=='axial_rope_ape' or self.pos_embed_scheme == 'sprepe_share_ape_learnable':
            #  '
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
        if self.pos_embed_scheme == 'ape' or self.pos_embed_scheme == 'sprepe_ape' or self.pos_embed_scheme=='sprepe_ape_learnable' or self.pos_embed_scheme=='axial_rope_ape' or self.pos_embed_scheme == 'ringpe' or self.pos_embed_scheme == 'sprepe_share_ape_learnable':
            # 
            # print(self.pos_embed.shape)
            if (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]) != (pn[0], pn[1]):
                pos = resize_pos_embed(
                    self.pos_embed.to(device=x.device, dtype=x.dtype),
                    old_hw=(pn[0], pn[1]),     # e.g. (64,128)
                    new_hw=(self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]),
                    num_extra_tokens=0,
                )
                x = x + pos
            else:
                x = x + self.pos_embed
    
        if self.pos_embed_scheme == 'sphere2vec' or self.pos_embed_scheme == 'sprepe_share_sphere2vec' or self.pos_embed_scheme == 'sprepe_sphere2vec_learnable':
            # print(x.shape, self.pos_embed(self.patch_centers).shape)
            x = x + self.pos_embed(self.patch_centers)

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
            x = checkpoint.checkpoint(blk, x, self.alibi, self.sprepe)
            # x = blk(x, self.alibi)
        x = self.norm(x)
        x = self.forward_head(x)
        # x = checkpoint.checkpoint(self.forward_head, x)
        return x
