# Vision Transformer Architecture

## Overview

The current model follows the basic Vision Transformer design for image classification. An image is divided into non-overlapping patches, each patch is projected into an embedding vector, and the resulting sequence is processed by Transformer blocks.

For an input image with spatial size $H \times W$ and patch size $P$, the number of patches is:

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

The model adds one classification token to this sequence, resulting in $N + 1$ tokens.

## Main Components

### Patch Embedding

`PatchEmbed` uses a convolution with kernel size and stride equal to the patch size. This converts the image into a sequence of patch embeddings while preserving the embedding dimension required by the Transformer.

### Classification Token

A learned classification token is prepended to the patch sequence. After the Transformer blocks process the sequence, the final representation of this token is used for image classification.

### Positional Embeddings

Learned positional embeddings are added to the patch and classification-token sequence. They provide information about the original position of each patch.

### Transformer Block

Each block contains:

1. Layer normalization.
2. Multi-head self-attention.
3. A residual connection.
4. A second layer normalization.
5. A GELU-based MLP.
6. A second residual connection.

The current implementation uses pre-normalization, where normalization is applied before attention and the MLP.

## Current Model Configuration

The CIFAR-100 example uses:

| Parameter | Value |
| --- | ---: |
| Image size | 32 x 32 |
| Patch size | 4 x 4 |
| Number of channels | 3 |
| Embedding dimension | 192 |
| Transformer depth | 6 |
| Attention heads | 3 |
| Number of classes | 100 |

The complete reference is available in [configs/vit_cifar100.yaml](../configs/vit_cifar100.yaml).

## Scope

This implementation is intended for learning and experimentation. It does not currently aim to reproduce the full training recipes or performance of large production Vision Transformer models.
