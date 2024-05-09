import torch
import torch.nn as nn

from lib.models.transformer import Transformer

class TFormer(nn.Module):
    def __init__(self, seqlen=16, d_model=2048, num_head=8, spatial_n_layer=3) :
        super().__init__()
        self.transformer = Transformer(depth=spatial_n_layer, embed_dim=seqlen, mlp_hidden_dim=seqlen*2, h=num_head, length=d_model)
        

    def forward(self, x):
        """
        x : [B, T, D]
        """
        self.transformer(x)

        return 