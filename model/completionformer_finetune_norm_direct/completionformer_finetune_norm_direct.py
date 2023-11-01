from .completionformer import CompletionFormer
import torch
import torch.nn as nn
from .backbone_finetune_norm_direct import BackboneFinetuneNormDirect
from model.depth2norm.depth2norm import depth2norm

class CompletionFormerFinetuneNormDirect(nn.Module):
    def __init__(self, args):
        super(CompletionFormerFinetuneNormDirect, self).__init__()

        self.args = args
        self.camera_matrix = args.camera_matrix
        print("Camera matrix is \n{}".format(self.camera_matrix))

        self.foundation = CompletionFormer(args)
        self.foundation.load_state_dict(torch.load(args.pretrained_completionformer, map_location='cpu')['net'])

        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1
        self.backbone = BackboneFinetuneNormDirect(args, self.foundation.backbone, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = self.foundation.prop_layer
    
    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        pol = sample['pol']

        pred_init, guide, confidence = self.backbone(rgb, dep, pol)
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
            
        norm = depth2norm(y, self.camera_matrix, computation_mode='vectorized')

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': conf_inter, 'norm': norm}

        return output
