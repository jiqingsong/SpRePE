import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.utils.checkpoint as checkpoint

def compute_patch_center_coordinates(lat, lon, patch_size, stride=None):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    
    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    a, b = lat.shape

    n_patches_h = (a - patch_size[0]) // stride[0] + 1
    n_patches_w = (b - patch_size[1]) // stride[1] + 1
    
    patch_centers_lat = np.zeros((n_patches_h, n_patches_w))
    patch_centers_lon = np.zeros((n_patches_h, n_patches_w))
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            start_h = i * stride[0]
            start_w = j * stride[1]

            patch_lat = lat[start_h:start_h+patch_size[0], 
                           start_w:start_w+patch_size[1]]
            patch_lon = lon[start_h:start_h+patch_size[0], 
                           start_w:start_w+patch_size[1]]
            

            patch_centers_lat[i, j] = np.mean(patch_lat)
            patch_centers_lon[i, j] = np.mean(patch_lon)
    
    return patch_centers_lat, patch_centers_lon

def compute_cartesian_coordinates(theta, phi):
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)
import numpy as np

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


class SphericRope_vit(nn.Module):
    def __init__(
        self,
        head_num,
        head_dim,
        origin_shape,
        patch_size,
        stride,
        lat_lon=(180, 360),
        offset=(0,0),
        with_pos = False
    ):
        super().__init__()
        self.head_num = head_num
        self.embed_dim = head_dim
        assert head_dim % 3 == 0, 
        self.origin_shape = origin_shape
        self.patch_size = patch_size
        self.stride = stride
        self.lat_lon = lat_lon
        self.offset = offset
        self.with_pos = False
        resolution_lat = lat_lon[0] / origin_shape[0]
        resolution_lon = lat_lon[1] / origin_shape[1]
        lat = np.arange(offset[0] + resolution_lat / 2, offset[0] + lat_lon[0], resolution_lat)[::-1]
        lon = np.arange(offset[1], offset[1] + lat_lon[1], resolution_lon)
        self.lon_grid, self.lat_grid = np.meshgrid(lon, lat)

        self.patch_centers_lat, self.patch_centers_lon = compute_patch_center_coordinates(self.lat_grid, self.lon_grid, self.patch_size, self.stride)
        self.patch_centers_lat_rad = torch.tensor(np.radians(self.patch_centers_lat), requires_grad=False)
        self.patch_centers_lon_rad = torch.tensor(np.radians(self.patch_centers_lon), requires_grad=False)
        patch_centers = compute_cartesian_coordinates(self.patch_centers_lat_rad, self.patch_centers_lon_rad)

        self.register_buffer("patch_centers", patch_centers)
        num_points = self.embed_dim // 3
       
        self.midpoint_theta = nn.Parameter(torch.arccos(1 - 2 * torch.rand((self.head_num, num_points))), requires_grad=False) 
        self.midpoint_phi = nn.Parameter(2 * torch.pi * torch.rand(self.head_num, num_points), requires_grad=False)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'midpoint_theta, midpoint_phi'}

    def forward(self, q, k):       
        with torch.cuda.amp.autocast(enabled=False):
            num_points = self.embed_dim // 3
            midpoint = compute_cartesian_coordinates(self.midpoint_theta, self.midpoint_phi).unsqueeze(1).unsqueeze(1)
            patch_centers_repeat = self.patch_centers.unsqueeze(2).unsqueeze(0).repeat(self.head_num, 1, 1, num_points, 1)
            if self.with_pos:
                patch_padding = self.slopes * patch_centers_repeat.flatten(-2, -1).flatten(1, 2).to(q.device)
            if midpoint.device != patch_centers_repeat.device:
                midpoint = midpoint.to(patch_centers_repeat.device)
            v = patch_centers_repeat - midpoint
            v = v / (torch.norm(v, p=2, dim=-1, keepdim=True) + 1e-7)
            
            if v.device != self.midpoint_theta.device:
                v = v.to(self.midpoint_theta.device)
            bs, head_num, patch_num, embed_dim = q.shape
            
            indices = torch.arange(self.embed_dim).reshape(-1, 3)
            
            indices_231 = indices[:, [1, 2, 0]].reshape(-1)
            
            q_231 = q[:, :, :, indices_231]
            k_231 = k[:, :, :, indices_231]

            # q''的索引排列：[2,0,1]
            indices_312 = indices[:, [2, 0, 1]].reshape(-1)
            q_312 = q[:, :, :, indices_312]
            k_312 = k[:, :, :, indices_312]
            
            v_231 = v[:, :, :, :, [1, 2, 0]]
            v_312 = v[:, :, :, :, [2, 0, 1]]
            rope_123 = 1 - 2 * v * v
            rope_123 = rope_123.flatten(-2, -1).flatten(1, 2)
            rope_231 = -2 * v * v_231
            rope_231 = rope_231.flatten(-2, -1).flatten(1, 2)
            rope_312 = -2 * v * v_312
            rope_312 = rope_312.flatten(-2, -1).flatten(1, 2)
            
            re_q = q * rope_123 + q_231 * rope_231 + q_312 * rope_312 
            
            re_k = k * rope_123 + k_231 * rope_231 + k_312 * rope_312 

        return re_q, re_k


class AxialRope(nn.Module):
    def __init__(self, shape, dim, origin_shape=[0,0]) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        
        if origin_shape[0] > 0:
            coords_0 = coords_0 / (shape[0] - 1) * (origin_shape[0] - 1)
            coords_1 = coords_1 / (shape[1] - 1) * (origin_shape[1] - 1)
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 1000000 ** -freq_seq1
        inv_freq2 = 1000000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2     

        self.sin1 = torch.sin(sinusoid1).reshape(shape[0] * shape[1], sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(shape[0] * shape[1], sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(shape[0] * shape[1], sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(shape[0] * shape[1], sinusoid2.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)
        # print(x11.shape, self.cos1.shape)
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)

        return res
