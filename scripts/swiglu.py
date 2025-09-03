import torch
import torch.nn as nn
import torch.nn.init as init

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device = device, dtype = dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device = device, dtype = dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device = device, dtype = dtype))
        self.init_weight()

    def init_weight(self):
        init.kaiming_uniform_(self.w1)
        init.kaiming_uniform_(self.w2)
        init.kaiming_uniform_(self.w3)

    def feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        W_x = x @ self.w1.T
        swish_output = torch.sigmoid(W_x) * W_x
        glu_output = swish_output * (x @ self.w3.T)
        return glu_output @ self.w2.T