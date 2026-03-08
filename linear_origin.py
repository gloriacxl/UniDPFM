from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ULIP' in model:
            self.fc = nn.ModuleList([nn.ModuleList([nn.Linear(384, 512), nn.Linear(512, 512)]) for _ in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens,cls):
        bs = tokens[0].shape[0]
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i]= tokens[i][:, 1:, :]
                tokens[i] =  F.relu(self.fc[i][0](tokens[i]))
                tokens[i] = self.fc[i][1](tokens[i])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
