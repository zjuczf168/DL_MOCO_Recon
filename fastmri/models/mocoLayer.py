"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import fastmri
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MocoSimLayer(nn.Module):
    """
    Motion Simulation layer 
    """

    def __init__(
        self,
        maxTrans: int = 10, maxRot: int = 10, N_tp_Max: int = 16, half_width_no_motion: int = 12, noMotionFrac: float = 0.2
    ):
        """
        Args:
            maxTrans: Maximum amount of translation in pixels
            maxRot: Maximum amount of rotation in degrees
            N_tp_Max: Maximum Number of motion events
            half_width_no_motion: Half width of center fraction without any motion event
            noMotionFrac: Fraction of samples without any motion simulation
        """
        super().__init__()
        self.maxTrans = maxTrans
        self.maxRot = maxRot
        self.N_tp_Max = N_tp_Max
        self.half_width_no_motion = half_width_no_motion
        self.noMotionFrac = noMotionFrac

    def generate_motion_params(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        # --- Number of motion events before centre k-space----- #
        N_tp_half = int(torch.randperm(self.N_tp_Max)[0])
        # --- Permissible number of time points for motion event ---- #
        indx_motion = int(w / 2 - self.half_width_no_motion)
        # --- Actual time points of motion events ---- #
        indx_motion_pts_fh = (torch.sort(1 + torch.randperm(indx_motion)[0:N_tp_half]))
        indx_motion_pts_sh = (torch.sort(int(w / 2 + self.half_width_no_motion) + torch.randperm(indx_motion)[0:self.N_tp_Max - N_tp_half]))
        # ----- append first and last time point ---- #
        indx_motion_pts = torch.cat((torch.tensor([0]), indx_motion_pts_fh[0], indx_motion_pts_sh[0], torch.tensor([h])), dim=0)
        # --- Generate Discrete Motion Parameters at motion events----- #
        mask = (0.5+1.49*torch.rand(self.N_tp_Max, 1)).int()
        motion_param_trans = mask * self.maxTrans * (2 * torch.rand(self.N_tp_Max, 2) - 1)
        motion_param_rot = mask * self.maxRot * (2 * torch.rand(self.N_tp_Max, 1) - 1) * math.pi / 180.0
        motion_param = torch.cat((motion_param_rot, motion_param_trans), dim=-1)
        # ---- make centre of k-space without any moition ---- #
        motion_param1 = torch.cat((motion_param[0:N_tp_half, :], torch.zeros(1, 3), motion_param[N_tp_half:self.N_tp_Max, :]), dim=0)
        if torch.cuda.is_available():
            return motion_param1.cuda(), indx_motion_pts.cuda()
        else:
            return motion_param1, indx_motion_pts
        # motion_param1 (N_tp_Max, 3), indx_motion_pts (N_tp_Max+1)
    
    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
        
    def get_rot_trans_kspace(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # params --> (1, 3)
        b, c, h, w, two = x.shape
        x = fastmri.ifft2c(x) # image
        #params = torch.tile(params, (b, 1))
        if torch.cuda.is_available():
            params = torch.repeat_interleave(params, torch.tensor([b]).cuda(), dim=0).reshape(b, 3)
        else:
            params = torch.repeat_interleave(params, torch.tensor([b]), dim=0).reshape(b, 3)
        x = self.complex_to_chan_dim(x) # (b, 2 * c, h, w)
        x = kornia.geometry.transform.translate(x, params[:, 0:2], mode='nearest', padding_mode='zeros', align_corners=True)
        x = kornia.geometry.transform.rotate(x, params[:, -1], center=None, mode='nearest', padding_mode='zeros', align_corners=True)
        x = self.chan_complex_to_last_dim(x)
        x = fastmri.fft2c(x) # kspace
        return x
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- generate motion parameters ---- #
        motion_param, indx_motion_pts = self.generate_motion_params(x)
        # ---- get rotated k-space for the first motion event ---- #
        kspace = self.get_rot_trans_kspace(x, motion_param[0, :])
        # ---- Fill a partial kspace acquired upto first motion event --- #
        kspace_txrx = kspace[:, :, :, indx_motion_pts[0]: indx_motion_pts[1], :]
        # ---- Fill a motion corrupted kspace for rest of the motion event --- #
        for k in range(1, int(self.N_tp_Max)+1):
            kspace = self.get_rot_trans_kspace(x, motion_param[k, :])
            kspace_txrx = torch.cat([kspace_txrx, kspace[:, :, :, indx_motion_pts[k]: indx_motion_pts[k+1], :]], dim=-2)
        if torch.rand(1) < self.noMotionFrac:
            return x
        else:
            return kspace_txrx 

class UndersampleLayer(nn.Module):
    """
    Equidistant Undersamling  Layer
    """

    def __init__(
        self,
        center_fraction=0.08, acceleration=4.0, center_lines=0, variable_center=0.0, 
    ):
        """
        Args:

        """
        super().__init__()
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.center_lines = center_lines
        self.variable_center=variable_center
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        num_cols = w
        if self.center_lines != 0:
            num_low_freqs = self.center_lines 
        else:
            num_low_freqs = int(round(num_cols * self.center_fraction * np.random.uniform(low=1.0-self.variable_center, high=1.0+self.variable_center)))
        
        # create the mask
        if torch.cuda.is_available():
            mask = torch.zeros(1, 1, 1, w, 1, dtype=torch.float32).cuda()
        else:
            mask = torch.zeros(1, 1, 1, w, 1, dtype=torch.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[:, :, :, pad : pad + num_low_freqs, :] = 1.0

        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (self.acceleration * (num_low_freqs - num_cols)) / (
            num_low_freqs * self.acceleration - num_cols
        )
        offset = torch.randperm(round(adjusted_accel))[0]

        accel_samples = torch.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = accel_samples.long()
        mask[:, :, :, accel_samples, :] = 1.0
        # --- apply mask --- #
        masked_x = x * mask + 0.0  # the + 0.0 removes the sign of the zeros
        
        return masked_x, mask
        
        

        
        