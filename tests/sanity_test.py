import torch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vit_pytorch.models.simple_ViT import VisionTransformer


def test_vit_forward():
    batch_size = 4
    img_size = 224
    patch_size = 16
    channels = 3
    n_classes = 10

    dim = 768
    depth = 12
    n_heads = 12
    mlp_ratio = 3.0
    qkv_bias = True
    p = 0.0
    attn_p = 0.0

    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        canais=channels,
        n_classes=n_classes,
        dim=dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        p=p,
        attn_p=attn_p
    )

    x = torch.randn(
        batch_size,
        channels,
        img_size,
        img_size
    )

    y = model(x)

    assert y.shape == (batch_size, n_classes)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
    print("Sanity test passed!")


if __name__ == "__main__":
    test_vit_forward()