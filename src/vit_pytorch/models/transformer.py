import torch.nn as nn

from .attention import Attention


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.0):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.dropout(self.fc2(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=3.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        mlp_hidden_size = int(mlp_ratio * dim)
        self.mlp = MLP(dim, mlp_hidden_size, dim, p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))
