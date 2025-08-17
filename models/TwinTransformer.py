import torch
import torch.nn as nn
from .GDLTransformer import GDLTransformer
from .positional_encoding import PositionslEncoding  # your existing positional encoding

class TwinGDLTransformer(nn.Module):
    def __init__(self, input_dim=52, seq_len=500, d_model=64, n_heads=4, n_classes=21, n_layers=3, dropout=0.1):
        super().__init__()
        
        # Input projection + positional encoding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionslEncoding(seq_len, d_model)
        
        # Twin branches
        self.branch1 = nn.ModuleList([GDLTransformer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.branch2 = nn.ModuleList([GDLTransformer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        
        # Final classifier after concatenation
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
    def forward(self, x):
        x = self.input_proj(x)          # [B, T, D]
        x = self.pos_enc(x)
        
        # Branch 1
        out1 = x
        for layer in self.branch1:
            out1 = layer(out1)
        out1 = out1.mean(dim=1)         
        
        # Branch 2
        out2 = x
        for layer in self.branch2:
            out2 = layer(out2)
        out2 = out2.mean(dim=1)
        
        # Concatenate and classify
        out = torch.cat([out1, out2], dim=-1)
        out = self.classifier(out)
        
        return out