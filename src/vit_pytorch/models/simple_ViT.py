import torch
import torch.nn as nn

class PathEmbed(nn.Module):
    def __init__(self, img_size, patch_size, canais=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(  # quebra a imagem original em patches
            canais, 
            embed_dim,
            kernel_size=patch_size, # faz com que os patches não sofram modificações
            stride=patch_size
        )

    def forward(self, x):
        # formato de x: (batch_size, channels, img_size_h, img_size_w)

        x = self.proj(x)    # (batch_size, embed_dim, n_patches**0.5, ...)
        x = x.flatten(x)
        x = x.transpose(1,2)

        return x 
    
class Attention(nn.Module):
    def __init__(self, dim,  n_heads=12, qkv_bias=True, attn_dropout_p=0., proj_dropout_p=0.):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads # depois vamos concatenar a previsão de todas as heads para conseguir o valor final da SA
        self.QK_scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) # permite que o modelo misture informações de diferentes heads.

        self.attn_dropout_p = nn.Dropout(p=attn_dropout_p)
        self.proj_dropout_p = nn.Dropout(p=proj_dropout_p)
    def forward(self, x):

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )
        qkv = qkv.permute(2,0,3,1,4)

        q,k,v = qkv[0], qkv[1], qkv[2]

        kt = k.transpose(-2,-1) # fazemos o swap das duas últimas dimensões de k, ou seja, o número de patches / tokens e a dimensão de cada token

        dp = (q @ kt) * self.scale

        attn = dp.softmax(dim=-1) # o cos sim score está na row -1, conforme definido no nosso swap / transpoe
        attn = self.attn_dropout_p(attn)

        final = attn @ v
        final = final.transpose(1,2)   # esses transpose são chatos. Eles geralmente são porque queremos deixar algo em um formato bonitinho.
        final = final.flatten(2)       # junta as previsões das heads

        x = self.proj(x)               # combina (de verdade) a previsão das heads.
        x = self.proj_dropout_p(x)

        return x
    
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
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True,
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

        mlp_hidden_size = int(3 * dim) # caso essa multiplicação n dê um int, force um

        self.mlp = MLP(
            dim,
            mlp_hidden_size,
            dim,
            p
        )

    def forward(self, x):
        x = x + self.attn(self.nomr1(x))    # esse '+x' é devido às conexões residuais no paper.
        x = x + self.mlp(self.norm2(x))

        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, canais, n_classes, dim, depth,
                 n_heads, mlp_ratio, qkv_bias, p, attn_p):
        super().__init__()

        self.patch_embed = PathEmbed(
            img_size,
            patch_size,
            canais,
            dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1,1,dim)) # inicializamos o class token com zeros
        self.pos_embed = nn.Parameter(                      # embeddings the posição. Temos um para cada patch. Esse embedding, obviamente, tem dimensão 'dim'.
            torch.zeros(1,1,self.patch_embed.n_patches, dim) 
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
        x = self.head(cls_token_logit)

        return x