from .completionformer import CompletionFormer
import torch
import torch.nn as nn
from .backbone_polar_norm import BackbonePolarNorm
from .nlspn_module import NLSPN

class CompletionFormerPolarNorm(nn.Module):
    def __init__(self, args):
        super(CompletionFormerPolarNorm, self).__init__()

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1
        
        self.backbone = BackbonePolarNorm(args, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                    self.args.prop_kernel)
    
    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        pol = sample['pol']

        pred_init, guide, confidence, norm = self.backbone(rgb, dep, pol)
        pred_init = pred_init + dep
        
        # -- difussion --
        y_inter = [pred_init, ]
        conf_inter = [confidence, ]
        if self.prop_time > 0:
            y, y_inter, offset, aff, aff_const = \
                self.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            y = pred_init
            offset, aff, aff_const = torch.zeros_like(y), torch.zeros_like(y), torch.zeros_like(y).mean()

        # Remove negative depth
        y = torch.clamp(y, min=0)
        
        # best at first
        y_inter.reverse()
        conf_inter.reverse()
        if not self.args.conf_prop:
            conf_inter = None

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': conf_inter, 'norm': norm}

        return output
