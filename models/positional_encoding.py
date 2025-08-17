import torch
import torch.nn as nn
import math

class PositionslEncoding(nn.Module):
    def __init__(self, timesteps, d_model):
        super(PositionslEncoding, self).__init__()
        self.timesteps = timesteps
        self.d_model = d_model
        
        pe = torch.zeros(timesteps, d_model)
        position = torch.arange(0, timesteps, dtype=torch.float).unsqueeze(1) # [timesteps] -> [timesteps, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
        