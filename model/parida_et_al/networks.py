import torch
import torch.nn as nn
from .pvt import PVT
from .backbone import conv_bn_relu, convt_bn_relu
from .resnet_cbam import BasicBlock
import torch.nn.functional as F

class DepthBackbone(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        
        # -- define initial encoding layer --
        self.conv1 = conv_bn_relu(input_ch, 64, kernel=3, stride=1, padding=1, bn=False)
        
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
        
        # -- depth normal decoding --
        self.depth_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.depth_dec0 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1, bn=False)
        self.out = conv_bn_relu(64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)
        
    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f
    
    def forward(self, x):
        fe1 = self.conv1(x)
        
        fe2, fe3, fe4, fe5 = self.former(fe1)
        
        # -- surface normal feature decoding --
        # print("fe5 shape {}".format(fe5.shape))
        fd4 = self.dec4(fe5)
        # print("fd4 shape {}".format(fd4.shape))
        fd3 = self.dec3(self._concat(fd4, fe4))
        # print("fd3 shape {}".format(fd3.shape))
        # print("fe3 shape {}".format(fe3.shape))
        fd2 = self.dec2(self._concat(fd3, fe3))
        # print("fd2 shape {}".format(fd2.shape))

        # -- surface normal decoding --
        depth_fd1 = self.depth_dec1(self._concat(fd2, fe2))
        # print("depth_fd1 shape {}".format(depth_fd1.shape))
        depth_feat = self.depth_dec0(self._concat(depth_fd1, fe1))
        
        return self.out(depth_feat), fe5

class PolarizationBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.net = DepthBackbone(7)
        
    def forward(self, pol):
        return self.net(pol)
    
class RgbBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.net = DepthBackbone(3)
        
    def forward(self, rgb):
        return self.net(rgb)
    
class RawDepthBranch(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        # -- define initial encoding layer --
        self.conv1 = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1, bn=False)
        
        # -- define JCAT-based encoder --
        self.former = PVT(in_chans=64, patch_size=4, pretrained='./model/completionformer_original/pretrained/pvt.pth', num_stages=2)
        
    def forward(self, dep):
        fe1 = self.conv1(dep)
        fe2, fe3, fe4, fe5 = self.former(fe1)
        
        return fe5
    
class MultiModalFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.attention_rgb = nn.Bilinear(128, 128, 128)
        self.attention_dep = nn.Bilinear(128, 128, 128)
        
        self.attention_dec3 = nn.Sequential(
            convt_bn_relu(128+128, 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/2
        self.attention_dec2 = nn.Sequential(
            convt_bn_relu(64, 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.attention_dec1 = nn.Sequential(
            convt_bn_relu(64, 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        
        self.attention_dec0 = nn.Sequential(
            convt_bn_relu(64, 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )
        
        self.attention_dec_fin = conv_bn_relu(64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, pol_feat, dep_feat, rgb_feat):
        pol_feat = pol_feat.permute(0,2,3,1).contiguous()
        dep_feat = dep_feat.permute(0,2,3,1).contiguous()
        rgb_feat = rgb_feat.permute(0,2,3,1).contiguous()
        
        # print("pol_feat size {}".format(pol_feat.shape))
        # print("dep_feat size {}".format(dep_feat.shape))
        # print("rgb_feat size {}".format(rgb_feat.shape))
        
        attn_rgb = self.attention_rgb(rgb_feat, pol_feat)
        attn_dep = self.attention_dep(dep_feat, pol_feat)
        
        attn_rgb = attn_rgb.permute(0,3,1,2).contiguous()
        attn_dep = attn_dep.permute(0,3,1,2).contiguous()
        
        attn_feat = torch.cat([attn_rgb, attn_dep], dim=1)
        # print("attn_feat shape {}".format(attn_feat.shape))
        
        attn_feat = self.attention_dec3(attn_feat)
        # print("attn_feat shape dec3 {}".format(attn_feat.shape))
        
        attn_feat = self.attention_dec2(attn_feat)
        # print("attn_feat shape dec2 {}".format(attn_feat.shape))
        
        attn_feat = self.attention_dec1(attn_feat)
        # print("attn_feat shape dec1 {}".format(attn_feat.shape))
        
        attn_feat = self.attention_dec0(attn_feat)
        # print("attn_feat shape dec0 {}".format(attn_feat.shape))
        
        attn_feat = self.attention_dec_fin(attn_feat)
        # print("attn_feat shape dec fin {}".format(attn_feat.shape))
        
        attention_map = self.sigmoid(attn_feat)
        
        return attention_map
    
class ParidaEtAl(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.pol_branch = PolarizationBranch(args)
        self.rgb_branch = RgbBranch(args)
        self.dep_branch = RawDepthBranch(args)
        self.fusion_module = MultiModalFusion(args)
        
    def forward(self, x):
        rgb = x["rgb"]
        pol = x["pol"]
        dep = x["dep"]
        
        pol_depth, pol_feat = self.pol_branch(pol)
        rgb_depth, rgb_feat = self.rgb_branch(rgb)
        dep_feat = self.dep_branch(dep)
        
        attention_map = self.fusion_module(pol_feat, dep_feat, rgb_feat)
        # print('attention_map shape {}'.format(attention_map.shape))
        # print('pol_depth shape {}'.format(pol_depth.shape))
        # print('rgb_depth shape {}'.format(rgb_depth.shape))
        depth = ((attention_map * pol_depth)+((1-attention_map) * rgb_depth)) 
        
        print('attention_map range {} {}'.format(torch.min(attention_map), torch.max(attention_map)))
        print('pol_depth range {} {}'.format(torch.min(pol_depth), torch.max(pol_depth)))
        print('rgb_depth range {} {}'.format(torch.min(rgb_depth), torch.max(rgb_depth)))
        
        return {'pred': depth, 'attn': attention_map, 'pol_depth' :pol_depth, 'rgb_depth': rgb_depth}