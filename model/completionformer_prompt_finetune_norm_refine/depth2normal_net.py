import torch
import numpy as np
import sys
from depth2norm import *

class DepthToNormalNet(torch.nn.Module):
    def __init__(self, camera_matrix, neighborhood_size=5, computation_mode="row"):
        super(DepthToNormalNet, self).__init__()
        
        self.neighborhood_size = neighborhood_size
        self.camera_matrix = camera_matrix
        self.computation_mode = computation_mode
        
        self.residual_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
                torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
                torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
                )
        
        self.one_by_one_conv = torch.nn.Conv2d(in_channels=22, out_channels=3, kernel_size=1, padding=0)
        
    def forward(self, init_depth, init_norm):
        rough_norm = depth2norm(depth=init_depth, camera_matrix=self.camera_matrix, neighborhood_size=self.neighborhood_size, computation_mode=self.computation_mode, step_size=110)
        # print("rough norm shape", rough_norm.size())
        res_out = self.residual_block(rough_norm)
        # print("residual output shape", res_out.size())
        refined_norm = torch.cat([res_out, rough_norm], dim=1) # (B, 16, H, W) concatenate with (B, 3, H, W) resulting in (B, 19, H, W), i.e. skip connection
        # print("3x3 conv + skip connection shape", refined_norm.size())
        refined_norm = torch.cat([refined_norm, init_norm], dim=1) # (B, 22, H, W)
        # print("before 1x1 conv shape", refined_norm.size())
        refined_norm = self.one_by_one_conv(refined_norm) # (B, 3, H, W)
        # print("refined norm shape", refined_norm.size())
        
        return refined_norm, rough_norm