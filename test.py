import torch
from lib.models.PTFormer import PTFormer

def main(x, y):
    model = PTFormer(
        seqlen=16,
        num_joint=19,
        d_model=512,
        num_head=8,
        s_n_layer=3,
    )

    res = model(x, y)
    print(res.shape)

if __name__ == "__main__" :
    x = torch.rand((1, 16, 2048))
    y = torch.rand((1, 16, 19, 2))

    main(x, y)