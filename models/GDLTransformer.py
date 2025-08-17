import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d
from .GDLAttention import GDLAttention

    
class GDLTransformer(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden = None, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_hidden = ff_hidden
        
        self.attn = GDLAttention(d_model, n_heads, dropout=dropout)
        
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        ff_hidden = ff_hidden or 4*d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, T, D = x.size()
        
        x_norm = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        attn_out = self.attn(x_norm)
        x1 = x + attn_out
        
        x_norm2 = self.bn2(x1.permute(0, 2, 1)).permute(0, 2, 1)
        ff_out = self.ffn(x_norm2)
        
        out = x1 + ff_out
        return out
        
        
        
    
    
        
        
        