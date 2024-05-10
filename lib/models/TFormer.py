import torch
import torch.nn as nn

from lib.models.trans_operator import CrossAttention
from lib.models.transformer import MaskTransformer, Transformer

class TFormer(nn.Module):
    def __init__(self, 
                 seqlen=16, 
                 stride=4, 
                 d_model=2048, 
                 num_head=8, 
                 num_layer=3,
                 dropout=0., 
                 drop_path_r=0., 
                 atten_drop=0., 
                 mask_ratio=0.,) :
        super().__init__()
        self.mid_frame = seqlen // 2
        self.stride_short = stride
        self.mask_ratio = mask_ratio
        d_local_model = d_model // 2

        self.global_proj = nn.Linear(2048, d_model)
        self.global_encoder = MaskTransformer(depth=num_layer, embed_dim=d_model, mlp_hidden_dim=d_model*2, h=num_head,
                                       drop_rate=dropout, drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=seqlen)
        
        self.local_proj = nn.Linear(d_model, d_local_model)
        self.g2l_proj = nn.Linear(d_model, d_local_model)
        self.local_encoder = Transformer(depth=num_layer, embed_dim=d_local_model, mlp_hidden_dim=d_local_model*2, h=num_head,
                                       drop_rate=dropout, drop_path_rate=drop_path_r, attn_drop_rate=atten_drop, length=stride*2+1)
        self.local_decoder = CrossAttention(d_local_model, num_heads=num_head, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)

    def forward_global(self, x, is_train) :
        temp_feat = self.global_proj(x)
        if is_train:
            mem, mask_ids, ids_restore = self.global_encoder.forward_encoder(temp_feat, mask_flag=True, mask_ratio=self.mask_ratio)
        else:
            mem, mask_ids, ids_restore = self.global_encoder.forward_encoder(temp_feat, mask_flag=False, mask_ratio=0.)
        global_temp_feat = self.global_encoder.forward_decoder(mem, ids_restore)

        return global_temp_feat, mask_ids
    
    def forward_local(self, x, global_temp_feat):
        local_temp_feat = x[:, self.mid_frame - self.stride_short:self.mid_frame + self.stride_short + 1]
        local_temp_feat = self.local_proj(local_temp_feat)
        local_temp_feat = self.local_encoder(local_temp_feat)
        g2l_temp_feat = self.g2l_proj(global_temp_feat)
        local_temp_feat = local_temp_feat[:, self.stride_short - 1: self.stride_short + 2]
        local_temp_feat = self.local_decoder(local_temp_feat, g2l_temp_feat)

        return local_temp_feat

    def forward(self, x, is_train=False):
        """
        x : [B, T, 2048]
        """
        B, T, D = x.shape
        
        global_temp_feat, mask_ids = self.forward_global(x, is_train)
        local_temp_feat = self.forward_local(x, global_temp_feat)

        return global_temp_feat, local_temp_feat, mask_ids