import torch
import torch.nn as nn
from .backbone import conv_bn_relu

class FoveaMask(nn.Module):
    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, qk):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = qk.shape
        qk = qk.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(qk * self.smooth)
        else:
            mask = self.softmax(qk)
        # output = mask * qk
        # output = output.contiguous().view(b, c, h, w)

        return mask, qk
    
class CrossFovea(nn.Module):
    def __init__(self, smooth=False):
        super().__init__()
        self.fovea_mask_q1 = FoveaMask(smooth)
        self.fovea_mask_q2 = FoveaMask(smooth)
    
    def forward(self, q1, q2):
        b, c, h, w = q1.shape
        
        mask_q1, q1 = self.fovea_mask_q1(q1)
        mask_q2, q2 = self.fovea_mask_q2(q2)
        
        out_q1 = mask_q2 * q1
        out_q2 = mask_q1 * q2
        
        out_q1 = out_q1.contiguous().view(b, c, h, w)
        out_q2 = out_q2.contiguous().view(b, c, h, w)
        
        return out_q1, out_q2
    
class DepthFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_1 = conv_bn_relu(1+1, 64, kernel=3, stride=1,
                                     padding=1)
        self.decoder_2 = conv_bn_relu(64, 1, kernel=3, stride=1,
                                     padding=1, bn=False, relu=True)
        self.cross_fovea = CrossFovea()
    
    def forward(self, d1, d2):
        d1, d2 = self.cross_fovea(d1, d2)
        d_fin = self.decoder_2(self.decoder_1(torch.cat([d1, d2], dim=1)))
        
        return d_fin