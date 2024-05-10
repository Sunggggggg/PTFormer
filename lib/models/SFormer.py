import torch
import torch.nn as nn

from lib.models.transformer import Transformer
from lib.models.trans_operator import CrossAttention

LOCAL_JOINT_INDEX = [
    [17, 18], # 0
    [17, 18], # 1
    [17, 18], # 2
    [17, 18], # 3
    [17, 18], # 4
    # UPPER
    [17],       # 5
    [17],       # 6
    [17, 5],    # 7
    [17, 6],    # 8
    [17, 5, 7], # 9
    [17, 6, 8], # 10
    # LOWER
    [17],           # 11
    [17],           # 12
    [17, 11],       # 13
    [17, 12],       # 14
    [17, 11, 13],   # 15
    [17, 12, 14],   # 16
    [],             # 17
    [17]            # 18
]

class SFormer(nn.Module):
    def __init__(self, 
                 num_joint=19, 
                 d_model=512, 
                 num_head=8, 
                 num_layer=3,
                 dropout=0., 
                 drop_path_r=0., 
                 atten_drop=0.) :
        super().__init__()
        assert len(LOCAL_JOINT_INDEX) == num_joint, "Check num_joint"
        d_local_model = d_model // 2

        self.global_joint_embed = nn.Linear(2, d_model)
        self.global_encoder = Transformer(depth=num_layer, embed_dim=d_model, mlp_hidden_dim=d_model*2, h=num_head, 
                                          drop_rate=dropout, drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=num_joint)
        
        self.local_joint_embed = nn.ModuleList()
        for idx, local_joint_idx in enumerate(LOCAL_JOINT_INDEX):
            proj = nn.Linear(2 * (len(local_joint_idx) + 1), d_local_model)
            nn.init.xavier_uniform_(proj.weight, gain=0.01)
            self.local_joint_embed.append(proj)
        self.local_proj = nn.Linear(d_model, d_local_model)
        self.local_encoder = Transformer(depth=num_layer, embed_dim=d_local_model, mlp_hidden_dim=d_local_model*2, h=num_head, 
                                          drop_rate=dropout, drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=num_joint)
        self.local_decoder = CrossAttention(d_local_model, num_heads=num_head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

    def foward_global(self, x) :
        joint_feat = self.global_joint_embed(x)      # [BT, J, D]
        global_joint_feat = self.global_encoder(joint_feat)
        return global_joint_feat

    def foward_local(self, x, global_joint_feat) :
        """
        joint_feat          : [BT, J, 2]
        global_joint_feat   : [BT, J, D]
        """
        local_joint_feat = []
        for idx, (local_joint_idx, proj) in enumerate(zip(LOCAL_JOINT_INDEX, self.local_joint_embed)):
            acc_joint = torch.cat([x[:, i] for i in local_joint_idx] + [x[:, idx]], dim=-1)  #[BT, 2*(n+1)]
            local_joint_feat.append(proj(acc_joint))
        local_joint_feat = torch.stack(local_joint_feat, dim=1) # [BT, J, D//2]
        g2l_joint_feat = self.local_proj(global_joint_feat)
        local_joint_feat = self.local_encoder(local_joint_feat)
        local_joint_feat = self.local_decoder(local_joint_feat, g2l_joint_feat)
        return local_joint_feat

    def forward(self, x):
        """
        x : [B, T, J, 2]
        """
        B, T, J, D = x.shape

        x = x.view(-1, J, D)
        
        global_joint_feat = self.foward_global(x)
        local_joint_feat = self.foward_local(x, global_joint_feat)
        
        global_joint_feat = global_joint_feat.reshape(B, T, J, -1)
        local_joint_feat = local_joint_feat.reshape(B, T, J, -1)

        return global_joint_feat, local_joint_feat