# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import math
from makani.models.preprocessor import Preprocessor2D
from torch.utils.checkpoint import checkpoint

def rotate_latitude_by_angle(x: torch.Tensor, delta_deg: float) -> torch.Tensor:
    # B, _, C, H, W =  x.shape
    print(x.shape)
    B = 1
    H = 128
    W = 256
    device = x.device
    r = torch.arange(H, device=device).float()
    c = torch.arange(W, device=device).float()
    phi = (0.5 - r/(H-1)) * math.pi              # [-pi/2, pi/2]
    lam = (-1.0 + 2.0*c/(W-1)) * math.pi         # [-pi, pi]
    PHI, LAM = torch.meshgrid(phi, lam, indexing='ij')  # [H,W]

    cosp, sinp = torch.cos(PHI), torch.sin(PHI)
    cosl, sinl = torch.cos(LAM), torch.sin(LAM)
    X = cosp*cosl; Y = sinp; Z = cosp*sinl

    ang = math.radians(delta_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    Xp = X
    Yp = ca*Y - sa*Z
    Zp = sa*Y + ca*Z

    PHI_src = torch.asin(torch.clamp(Yp, -1, 1))               # φ' ∈ [-pi/2, pi/2]
    LAM_src = torch.atan2(Zp, Xp)                              # λ' ∈ [-pi, pi]

    grid_x = (LAM_src / math.pi).clamp(-1, 1)                  # W 方向
    grid_y = (-2*PHI_src / math.pi).clamp(-1, 1)               # H 方向（向下为正）
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B,1,1,1)  # [B,H,W,2]

    out = torch.nn.functional.grid_sample(
        x, grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    print(out.shape)
    return out

class SingleStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

    def forward(self, inp, shift=0, rotate=0):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)
        if shift != 0:
            inpans = torch.roll(inpt, shifts=shift, dims=-1)
        if rotate != 0:
            inpans = rotate_latitude_by_angle(inpans, rotate)
        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y


class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()
        self.residual_mode = True if (params.target == "target") else False

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):
        result = []
        inpt = inp
        for step in range(self.n_future + 1):
            # add unpredicted features
            inpa = self.preprocessor.append_unpredicted_features(inpt)

            # do history normalization
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)

            # add static features
            inpans = self.preprocessor.add_static_features(inpan)

            # prediction
            predn = checkpoint(self.model, inpans, use_reentrant=False)
            # append the denormalized result to output list
            # important to do that here, otherwise normalization stats
            # will have been updated later:
            pred = self.preprocessor.history_denormalize(predn, target=True)
            # add residual (for residual learning, no-op for direct learning
            pred = self.preprocessor.add_residual(inpt, pred)
    
            # append output
            result.append(pred)

            if step == self.n_future:
                break

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)

        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)

        return result

    def _forward_eval(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # do history normalization
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # add static features
        inpans = self.preprocessor.add_static_features(inpan)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        yn = self.model(inpans)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        y = self.preprocessor.history_denormalize(yn, target=True)

        # add residual (for residual learning, no-op for direct learning
        y = self.preprocessor.add_residual(inp, y)

        return y

    def forward(self, inp):
        # decide which routine to call
        if self.training:
            y = self._forward_train(inp)
        else:
            y = self._forward_eval(inp)

        return y