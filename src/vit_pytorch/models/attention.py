import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=12,
        qkv_bias=True,
        attn_dropout_p=0.0,
        proj_dropout_p=0.0,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.QK_scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_dropout_p = nn.Dropout(p=attn_dropout_p)
        self.proj_dropout_p = nn.Dropout(p=proj_dropout_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q @ k.transpose(-2, -1)) * self.QK_scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_dropout_p(attention)

        output = attention @ v
        output = output.transpose(1, 2).flatten(2)
        output = self.proj(output)
        return self.proj_dropout_p(output)
