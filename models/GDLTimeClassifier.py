import torch
import torch.nn as nn
from GDLTransformer import GDLTransformer 
from positional_encoding import PositionslEncoding

class GDLTimeClassifier(nn.Module):
    def __init__(self, input_dim= 52, seq_len = 500, d_model = 64, n_heads = 64, n_classes=21, n_layers = 4, dropout = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionslEncoding(d_model, max_len=seq_len)
        self.transformer = GDLTransformer(d_model, n_heads, n_layers = n_layers, dropout=dropout)
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim = 1)
        out = self.classifier(x)
        return out