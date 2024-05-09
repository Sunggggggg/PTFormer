import torch
from lib.models.SFormer import SFormer

if __name__ == "__main__" :
    x = torch.rand((1, 16, 19, 2))
    model = SFormer(num_joint=19, d_model=256)

    y = model(x)

    print(y.shape)