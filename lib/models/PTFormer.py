import torch
import torch.nn as nn

from lib.models.trans_operator import CrossAttention
from lib.models.spin import Regressor
from lib.models.HSCR import HSCR
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
                 drop_reg_short=0.
                 ) :
        super().__init__()
        self.seqlen = seqlen
        self.stride = stride
        self.d_model = d_model

        self.s_former = SFormer(num_joint=num_joint, d_model=d_model, num_head=num_head,
                                 num_layer=s_n_layer, dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop)
        self.t_former = TFormer(seqlen=seqlen, stride=stride, d_model=d_model, num_head=num_head,
                                num_layer=t_n_layer, dropout=dropout, drop_path_r=drop_path_r, atten_drop=atten_drop, mask_ratio=mask_ratio)
        self.joint_weight_proj = nn.Linear(2048, num_joint)
        
        
        self.global_alpha_proj = nn.Linear(d_model*2, d_model*2)
        self.local_alpha_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 2048)

        self.global_regressor = Regressor()
        self.local_regressor = HSCR(drop=drop_reg_short)

    def cat_pelvis_neck(self, vitpose_j2d) :
        vitpose_j2d_pelvis = vitpose_j2d[:,:,[11,12],:2].mean(dim=2, keepdim=True)   # [B, T, 1, 2]
        vitpose_j2d_neck = vitpose_j2d[:,:,[5,6],:2].mean(dim=2, keepdim=True)       # [B, T, 1, 2]
        joint_2d_feats = torch.cat([vitpose_j2d[... ,:2], vitpose_j2d_pelvis, vitpose_j2d_neck], dim=2)   # [B, T, J, 2]

        return joint_2d_feats

    def joint_weighted_sum(self, input, global_joint_feat, local_joint_feat) :
        """
        input           : [B, T, 2048]
        global_joint_feat, local_joint_feat : [B, T, J, D]

        global_joint_feat, local_joint_feat : [B, T, 1, D]
        """
        B = input.shape[0]
        joint_weight = self.joint_weight_proj(input)                    # [B, T, J]
        joint_weight = joint_weight.softmax(dim=-1).unsqueeze(2)        # [B, T, 1, J]

        global_joint_feat = (joint_weight @ global_joint_feat).view(B, self.seqlen, -1)
        local_joint_feat = (joint_weight @ local_joint_feat).view(B, self.seqlen, -1)
        
        return global_joint_feat, local_joint_feat

    def global_prediction(self, global_joint_feat, global_temp_feat, is_train=False, J_regressor=None):
        """
        global_joint_feat   : [B, T, D]
        global_temp_feat    : [B, T, D]
        """
        B = global_joint_feat.shape[0]

        # Aggregation
        global_feat = torch.cat([global_joint_feat, global_temp_feat], dim=-1)           # [B, T, 2D]
        alpha = self.global_alpha_proj(global_feat).reshape(B, self.seqlen, self.d_model, 2)
        alpha = alpha.softmax(dim=-1)   # [B, T, D, 2]

        global_feat = global_joint_feat*alpha[..., 0] + global_temp_feat*alpha[..., 1]    # [B, T, D]

        # Global regressor
        if is_train :
            global_feat = self.out_proj(global_feat)                                      # [B, T, 2048]
        else :
            global_feat = self.out_proj(global_feat)[:, self.seqlen // 2][:, None, :]     # [B, 1, 2048]
        
        smpl_output_global, pred_global = self.global_regressor(global_feat, is_train=is_train, J_regressor=J_regressor, n_iter=3)
        
        scores = None
        if is_train:
            size = self.seqlen
        else:
            size = 1

        for s in smpl_output_global:
            s['theta'] = s['theta'].reshape(B, size, -1)           
            s['verts'] = s['verts'].reshape(B, size, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)
            s['scores'] = scores
        
        return smpl_output_global, pred_global
    
    def local_prediction(self, local_joint_feat, local_temp_feat, pred_global, is_train=False, J_regressor=None):
        """
        local_joint_feat   : [B, 3, D/2] D/2 = 256
        local_temp_feat    : [B, T, D/2]

        """
        B = local_joint_feat.shape[0]
        mid_frame = self.seqlen // 2
        local_joint_feat = local_joint_feat[:, mid_frame - 1 : mid_frame + 2]

        # Aggregation
        local_feat = torch.cat([local_joint_feat, local_temp_feat], dim=-1)         
        alpha = self.local_alpha_proj(local_feat).reshape(B, 3, self.d_model//2, 2)
        alpha = alpha.softmax(dim=-1)  

        local_feat = local_joint_feat*alpha[..., 0] + local_temp_feat*alpha[..., 1] 

        if is_train:
            local_feat = local_feat
        else:
            local_feat = local_feat[:, 1][:, None, :]                           # [B, 1, 256]

        smpl_output = self.local_regressor(local_feat, init_pose=pred_global[0], init_shape=pred_global[1], init_cam=pred_global[2], is_train=is_train, J_regressor=J_regressor)
        
        scores = None
        if not is_train:    # Eval
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, -1)         
                s['verts'] = s['verts'].reshape(B, -1, 3)      
                s['kp_2d'] = s['kp_2d'].reshape(B, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(B, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(B, -1, 3, 3)
                s['scores'] = scores

        else:
            size = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(B, size, -1)           # [B, 3, 10]
                s['verts'] = s['verts'].reshape(B, size, -1, 3)        # [B, 3, 6980]
                s['kp_2d'] = s['kp_2d'].reshape(B, size, -1, 2)        # [B, 3, 2]
                s['kp_3d'] = s['kp_3d'].reshape(B, size, -1, 3)        # [B, 3, 3]
                s['rotmat'] = s['rotmat'].reshape(B, size, -1, 3, 3)   # [B, 3, 3, 3]
                s['scores'] = scores

        return smpl_output

    def forward(self, input, vitpose_j2d=None, is_train=False, J_regressor=None) :
        """
        input           : [B, T, 2048]
        vitpose_j2d     : [B, T, J, 2]
        """
        self.is_train = is_train
        vitpose_j2d = self.cat_pelvis_neck(vitpose_j2d)
        B, T, J = vitpose_j2d.shape[:-1]

        global_joint_feat, local_joint_feat = self.s_former(vitpose_j2d)                # [B, T, J, D], [B, T, J, D/2]
        global_joint_feat, local_joint_feat = self.joint_weighted_sum(input, global_joint_feat, local_joint_feat)   # [B, T, D], [B, T, D/2]
        global_temp_feat, local_temp_feat, mask_ids = self.t_former(input, is_train)    # [B, T, D], [B, t, D/2]

        # 
        smpl_output_global, pred_global = self.global_prediction(global_joint_feat, global_temp_feat, is_train, J_regressor)
        smpl_output = self.local_prediction(local_joint_feat, local_temp_feat, pred_global, is_train, J_regressor)
        
        return smpl_output, mask_ids, smpl_output_global
