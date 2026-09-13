import torch
import torch.nn as nn

from .attention import Attention
from .patch_embedding import PatchEmbed
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0., ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_p)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=3.0, qkv_bias=True,
                p=0., attn_p=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads,
            qkv_bias,
            attn_p,
            p
        )

        self.norm2 = nn.LayerNorm(dim, 1e-6)

        mlp_hidden_size = int(mlp_ratio * dim) # caso essa multiplicação n dê um int, force um

        self.mlp = MLP(
            dim,
            mlp_hidden_size,
            dim,
            p
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # esse '+x' é devido às conexões residuais no paper.
        x = x + self.mlp(self.norm2(x))

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, canais, n_classes, dim, depth,
                 n_heads, mlp_ratio, qkv_bias, p, attn_p):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            canais,
            dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1,1,dim)) # inicializamos o class token com zeros
        self.pos_embed = nn.Parameter(                      # embeddings the posição. Temos um para cada patch. Esse embedding, obviamente, tem dimensão 'dim'.
            torch.zeros(1,self.patch_embed.n_patches + 1, dim) 
        )

        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(    # cria vários blocos de transformer. Cada um vai ter sua própria quantidade de parâmetros aprendíveis
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    mlp_ratio,
                    qkv_bias,
                    p,
                    attn_p
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):

        # X: imagem. F: torch.Tensor(n_samples, n_channels, img_size, img_size)
        # essa função deve retornar logits. 

        n_samples = x.shape[0]
        
        x = self.patch_embed(x) # gera patch embeddings
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for attn_block in self.blocks:
            x = attn_block(x)

        x = self.norm(x)

        cls_token_logit = x[:,0]
        x = self.head(cls_token_logit)  # logits !!!!

        return x