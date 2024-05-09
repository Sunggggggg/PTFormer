import torch
import torch.nn as nn

from lib.models.SFormer import SFormer
from lib.models.TFormer import TFormer

class PTFormer(nn.Module):
    def __init__(self, 
                 seqlen,
                 num_joint,
                 d_model,
                 num_head,
                 s_n_layer,
                 ) :
        super().__init__()
        self.sformer = SFormer(num_joint=num_joint, d_model=d_model, num_head=num_head, num_layer=s_n_layer)

    def forward(self, input, vitpose_j2d=None, is_train=False, J_regressor=None) :
        global_joint_feat, local_joint_feat = self.sformer(vitpose_j2d) # [B, T, J, D], [B, T, J, D/2]
        

        return global_joint_feat, local_joint_feat
