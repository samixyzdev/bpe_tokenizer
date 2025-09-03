import torch
import torch.nn as nn
from scripts.rmsnorm import RMSNorm
from scripts.multihead_self_attention import Multihead_Self_Attention
from scripts.swiglu import SwiGLU

class Transformer_Block(nn.Module):
    def __init__(self,
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 theta: int = 10000,
                 max_seq_len: int = 1024,
                 device: torch.device = None, 
                 dtype: torch.dtype = None
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.swiglu = SwiGLU(d_model = d_model, d_ff = d_ff, device = device, dtype = dtype)
        self.rmsnorm1 = RMSNorm(d_model = d_model, device = device, dtype = dtype)
        self.rmsnorm2 = RMSNorm(d_model = d_model, device = device, dtype = dtype)
        self.MSA = Multihead_Self_Attention(d_model = d_model,
                                                                 num_heads = num_heads, 
                                                                 theta = theta, 
                                                                 max_seq_len = max_seq_len, 
                                                                 device = device, 
                                                                 dtype = dtype
                                                                 )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed_x = self.rmsnorm1.forward(x)
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device = self.device, dtype = self.dtype)
        x2 = x + self.MSA.forward_w_rope(normed_x, token_positions)
        normed_x2 = self.rmsnorm2.forward(x2)
        y = x2 + self.swiglu.feed_forward(normed_x2)
        return y