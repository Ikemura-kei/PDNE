import torch
import torch.nn as nn

from .completionformer_finetune import CompletionFormerFinetune
from .pvt import *
from .backbone import *

class NormalBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        # -- define initial encoding layer --
        self.conv1 = conv_bn_relu(11, 64, kernel=3, stride=1, padding=1, bn=False)
        
        # -- define JCAT-based encoder --
        self.former = PVT(in_chans=64, patch_size=4, pretrained='./model/completionformer_original/pretrained/pvt.pth', num_stages=2)
        
        # -- decoder layers --
        channels = [64, 128, 64, 128]
        
        self.dec4 = nn.Sequential(
            convt_bn_relu(channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/2
        self.dec3 = nn.Sequential(
            convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.dec2 = nn.Sequential(
            convt_bn_relu(64 + channels[1], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        
        # -- surface normal decoding --
        self.norm_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.norm_dec0 = conv_bn_relu(64+64, 3, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)
        
    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, x):
        rgb = x["rgb"]
        pol = x["pol"]
        dep = x["dep"]
        fe1 = self.conv1(torch.cat([rgb, pol, dep], dim=1))
            
        fe2, fe3, fe4, fe5 = self.former(fe1)
        
        # -- surface normal feature decoding --
        fd4 = self.dec4(fe5)
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # -- surface normal decoding --
        norm_fd1 = self.norm_dec1(self._concat(fd2, fe2))
        normal = self.norm_dec0(self._concat(norm_fd1, fe1))
        normal = nn.functional.normalize(normal, dim=1)
        
        return {'normal':normal}
    
    
class NormalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.normal_embed = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1, bn=False)
    
        self.former = PVT(in_chans=64, patch_size=4, pretrained='./model/completionformer_original/pretrained/pvt.pth', num_stages=2)
    
    def forward(self, normal):
        fe1 = self.normal_embed(normal)
        fe2, fe3, fe4, fe5 = self.former(fe1)
        
        return fe1, fe2, fe3, fe4, fe5
        

class NormalDepthBranching(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.depth_branch = CompletionFormerFinetune(args)
        self.normal_branch = NormalBranch(args)
        self.normal_encoder = NormalEncoder(args)
    
    def forward(self, x):
        normal_out = self.normal_branch(x)
        normal_out['normal'].register_hook(lambda grad: print('normal_out', grad)) 
        
        norm_fe1, norm_fe2, norm_fe3, norm_fe4, norm_fe5 = self.normal_encoder(normal_out['normal'])
        
        depth_out = self.depth_branch(x, [norm_fe1, norm_fe2, norm_fe3, norm_fe4, norm_fe5])
        depth_out['pred'].register_hook(lambda grad: print('depth_out', grad)) 
        depth_out['norm'] = normal_out['normal']
        
        return depth_out
        