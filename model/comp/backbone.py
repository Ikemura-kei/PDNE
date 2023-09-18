import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cbam import BasicBlock
import torchvision.transforms as T
from .pvt import PVT


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.args = args
        self.mode = args.completionformer_mode
        self.use_pvt = args.pre_pvt
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1
        self.use_prior = args.prior
        self.pre_res = args.pre_res
        self.direct_cat = args.direct_cat
        if not args.resume:
            self.align = args.align
            self.direct_align = args.direct_align
        else:
            self.align = False
            self.direct_align = False
        self.with_norm = args.use_norm
        print("Use norm:", self.with_norm)
        print("Use prior:", self.use_prior)
        print("Use direct cat:", self.direct_cat)
        print("Use pretrained pvt:", self.use_pvt)
        print("Use align:", self.align)
        print("Use direct align:", self.direct_align)



        # Encoder
        if self.mode == 'rgbd':
            if self.use_prior:
                in_ch = 4
            else:
                in_ch = 3
            self.conv1_rgb = conv_bn_relu(in_ch, 48, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                      bn=False)

        elif self.mode == 'grayd':
            if not self.direct_cat:
                if not self.use_prior:
                    in_ch = 3
                else:
                    in_ch = 4
                self.conv1_gray = conv_bn_relu(in_ch, 48, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                        bn=False)
            else:
                if not self.use_prior:
                    in_ch = 4
                else:
                    in_ch = 5
                self.conv1 = conv_bn_relu(in_ch, 64, kernel=3, stride=1, padding=1,
                                        bn=False)
            

        elif self.mode == 'pd':
            if self.direct_cat:
                if not self.use_prior:
                    in_ch = 8
                else:
                    in_ch = 9
                self.conv1 = conv_bn_relu(in_ch, 64, kernel=3, stride=1, padding=1,
                                        bn=False)
                
            elif self.align:
                if not self.use_prior:
                    in_ch = 7
                else:
                    in_ch = 8
                self.conv_align = conv_bn_relu(in_ch, 3, kernel=1, stride=1, bn=False)
                self.norm = T.Compose([
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                self.conv1_polar = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                        bn=False)

            elif self.direct_align:
                if not self.use_prior:
                    in_ch = 7
                else:
                    in_ch = 8
                self.conv_align = conv_bn_relu(in_ch, 3, kernel=1, stride=1, bn=False)
                self.norm = T.Compose([
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
                self.conv1 = conv_bn_relu(4, 64, kernel=3, stride=1, padding=1,
                                        bn=False)
            else:
                if not self.use_prior:
                    in_ch = 7
                else:
                    in_ch = 8
                self.conv1_polar = conv_bn_relu(in_ch, 48, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                              bn=False)
                self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                        bn=False)

        elif self.mode == 'rgb':
            self.conv = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        elif self.mode == 'd':
            self.conv = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        else:
            raise TypeError(self.mode)
        
        if self.use_pvt:
            self.former = PVT(in_chans=64, patch_size=2,
                          pretrained='./model/comp/pretrained/pvt.pth', pre_res=self.pre_res, layer0=args.layer0)
        else:
            self.former = PVT(in_chans=64, patch_size=2,
                          pretrained=None, pre_res=self.pre_res, layer0=args.layer0)



        channels = [64, 128, 64, 128, 320, 512]
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(channels[5], 256, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(256+channels[4], 128, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(128, 128, stride=1, downsample=None, ratio=8),

        )
        # 1/4
        self.dec4 = nn.Sequential(
            convt_bn_relu(128 + channels[3], 64, kernel=3, stride=2,
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

        # Init Depth Branch
        # 1/1
        self.dep_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                     padding=1)
        self.dep_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                     padding=1, bn=False, relu=True)
        
        # Init Norm Branch
        if self.with_norm:
            self.norm_dec0 = conv_bn_relu(64+64, 3, kernel=3, stride=1,
                                        padding=1, bn=False, relu=True)
        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+channels[0], 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=(3, 3),
                          stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
            )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We),
                           mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb=None, depth=None):
        # Encoding
        if self.mode == 'rgbd':
            fe1_rgb = self.conv1_rgb(rgb)
            fe1_dep = self.conv1_dep(depth)
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
            fe1 = self.conv1(fe1)
        elif self.mode == 'grayd':
            if not self.direct_cat:
                fe1_rgb = self.conv1_gray(rgb)
                fe1_dep = self.conv1_dep(depth)
                fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
                fe1 = self.conv1(fe1)
            else:
                fe1 = torch.cat((rgb, depth), dim=1)
                fe1 = self.conv1(fe1)
        elif self.mode == 'pd':
            if self.direct_cat:
                fe1 = torch.cat((rgb, depth), dim=1)
                fe1 = self.conv1(fe1)
            elif self.align:
                rgb = self.conv_align(rgb)
                rgb = self.norm(rgb)
                fe1_rgb = self.conv1_polar(rgb)
                fe1_dep = self.conv1_dep(depth)
                fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
                fe1 = self.conv1(fe1)

            elif self.direct_align:
                rgb = self.conv_align(rgb)
                rgb = self.norm(rgb)
                fe1 = torch.cat((rgb, depth), dim=1)
                fe1 = self.conv1(fe1)
            else:
                fe1_rgb = self.conv1_polar(rgb)
                fe1_dep = self.conv1_dep(depth)
                fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
                fe1 = self.conv1(fe1)
                
        elif self.mode == 'rgb':
            fe1 = self.conv(rgb)
        elif self.mode == 'd':
            fe1 = self.conv(depth)
        else:
            raise TypeError(self.mode)

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
        init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        # Init Norm Decoding
        if self.with_norm:
            init_norm = torch.nn.functional.normalize(self.norm_dec0(self._concat(dep_fd1, fe1)), dim=1)

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        if self.with_norm:
            return init_depth, guide, confidence, init_norm
        else:
            return init_depth, guide, confidence
