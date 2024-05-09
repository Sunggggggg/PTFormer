import torch
import torch.nn as nn

from lib.models.trans_operator import CrossAttention
from lib.models.spin import Regressor
from lib.models.SFormer import SFormer
from lib.models.TFormer import TFormer

class PTFormer(nn.Module):
    def __init__(self, 
                 seqlen,
                 stride,
                 num_joint,
                 d_model,
                 num_head,
                 s_n_layer,
                 t_n_layer,
                 mask_ratio,
                 dropout=0., 
                 drop_path_r=0., 
                 atten_drop=0.,
                 ) :
        super().__init__()
        self.s_former = SFormer(num_joint=num_joint, d_model=d_model, num_head=num_head,
                                 num_layer=s_n_layer, dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop)
        self.t_former = TFormer(seqlen=seqlen, stride=stride, d_model=d_model, num_head=num_head,
                                num_layer=t_n_layer, dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop, mask_ratio=mask_ratio)
        
        self.global_regressor = Regressor()

    def cat_pelvis_neck(self, vitpose_j2d) :
        vitpose_j2d_pelvis = vitpose_j2d[:,:,[11,12],:2].mean(dim=2, keepdim=True)   # [B, T, 1, 2]
        vitpose_j2d_neck = vitpose_j2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)       # [B, T, 1, 2]
        joint_2d_feats = torch.cat([vitpose_j2d[... ,:2], vitpose_j2d_pelvis, vitpose_j2d_neck], dim=2)   # [B, T, J, 2]

        return joint_2d_feats

    def forward(self, input, vitpose_j2d=None, is_train=False, J_regressor=None) :
        """
        input           : [B, T, 2048]
        vitpose_j2d     : [B, T, J, 2]
        """
        vitpose_j2d = self.cat_pelvis_neck(vitpose_j2d)

        global_joint_feat, local_joint_feat = self.s_former(vitpose_j2d)                # [B, T, J, D], [B, T, J, D/2]
        global_temp_feat, local_temp_feat, mask_ids = self.t_former(input, is_train)    # [B, T, D], [B, t, D/2]
        global_temp_feat = global_temp_feat.unsqueeze(2)        # [B, T, 1, D]
        local_temp_feat = local_temp_feat.unsqueeze(2)          # [B, T, 1, D]

        global_feat = torch.cat([global_joint_feat, global_temp_feat], dim=2)   # [B, T, J+1, D]

        return 
