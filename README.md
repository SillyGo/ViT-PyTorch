# ViT-PyTorch

ViT-PyTorch is an educational and experimental repository for implementing Vision Transformer models with PyTorch. The current work focuses on understanding the main components of a ViT and applying them to image classification tasks.

The repository is intentionally developed in small steps. The current model includes patch embeddings, multi-head self-attention, Transformer blocks, positional embeddings, a classification token, and a classification head.

## Current Status

The current training example uses the CIFAR-100 dataset and a compact Vision Transformer configuration. The implementation is still evolving, and the training script is primarily intended for experimentation and validation of the model components.

The CIFAR-100 configuration is available at [configs/vit_cifar100.yaml](configs/vit_cifar100.yaml). The script currently defines its training procedure directly, so the configuration file serves as a documented reference while configuration loading is developed further.

## Repository Structure

```text
configs/                    Model and experiment configurations
docs/                       Architecture notes and experiment documentation
experiments/                Space for experiment outputs and notes
scripts/                    Executable training scripts
src/vit_pytorch/models/     Vision Transformer implementation
src/vit_pytorch/data/       Dataset-related code
src/vit_pytorch/training/   Training-related code
src/vit_pytorch/utils/      Shared utilities as they are introduced
tests/                      Sanity checks and model tests
```

## Running the Current Example

Install the required PyTorch and torchvision dependencies, then run:

```bash
python3 scripts/train_ciphar100.py
```

The script downloads CIFAR-100 into the repository's `data/` directory when necessary. Training may require a CUDA-capable device for practical execution times.

## Planned Work

Future development may include:

- Additional Vision Transformer model sizes and training configurations.
- Swin Transformer implementations.
- Encoder-decoder Transformer architectures.
- Image segmentation models built with Transformer backbones.
- Better configuration loading, experiment tracking, checkpointing, and evaluation utilities.

These directions are planned extensions and are not yet part of the current implementation.

## Documentation

- [ViT architecture](docs/ViT_Architecture.md)
- [Experiments and roadmap](docs/experiments.md)
