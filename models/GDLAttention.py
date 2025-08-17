# models/GDLAttention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GDLAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # integer division -> int

        # Linear projections for Q, K, V (map d_model -> d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # GDL static head gates (one scalar per head)
        self.g = nn.Parameter(torch.zeros(n_heads))  # shape: [n_heads]

        # Gate MLP: produce a dynamic gate score per head (output dim = n_heads)
        hidden = max(8, d_model // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_heads)   # IMPORTANT: output = n_heads
        )

    def forward(self, x):
        """
        x: [B, T, D] where D == d_model
        returns: [B, T, D]
        """
        B, T, D = x.size()
        assert D == self.d_model, f"Input dim ({D}) must equal d_model ({self.d_model})"

        # Project and reshape into heads
        # After linear: [B, T, D] -> reshape -> [B, T, n_heads, d_k] -> transpose -> [B, n_heads, T, d_k]
        Q = self.W_q(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = self.W_k(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = self.W_v(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        # Optionally normalize Q/K (you already had this)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)

        # Attention scores: [B, n_heads, T, T]
        scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum -> [B, n_heads, T, d_k]
        head_outs = torch.matmul(attn, V)

        # Static gate: shape [1, n_heads, 1, 1]
        gate_static = torch.sigmoid(self.g).view(1, self.n_heads, 1, 1)

        # Dynamic gate: compute per-head logits, aggregated over time
        # gate_mlp(x) -> [B, T, n_heads]; mean(dim=1) -> [B, n_heads]
        gate_logits = self.gate_mlp(x).mean(dim=1)               # [B, n_heads]
        gate_dynamic = torch.sigmoid(gate_logits).view(B, self.n_heads, 1, 1)

        # Combined gate: broadcast to head_outs shape
        gate = gate_static * gate_dynamic                        # [B, n_heads, 1, 1]
        head_outs = head_outs * gate                             # gated head outputs

        # Merge heads: [B, n_heads, T, d_k] -> transpose -> [B, T, n_heads, d_k] -> reshape -> [B, T, D]
        out = head_outs.transpose(1, 2).contiguous().reshape(B, T, D)

        return self.W_o(out)
