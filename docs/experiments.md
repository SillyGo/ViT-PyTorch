# Experiments and Roadmap

## Current Experiment

The current experiment trains a compact Vision Transformer on CIFAR-100 for image classification.

The training script currently uses:

- Random crops and horizontal flips for training augmentation.
- CIFAR-100 dataset normalization.
- An 80/20 training and validation split.
- AdamW optimization.
- Cosine annealing learning-rate scheduling.
- Gradient clipping.
- Early stopping based on validation loss.

The experiment configuration is recorded in [configs/vit_cifar100.yaml](../configs/vit_cifar100.yaml). Results, checkpoints, and downloaded data should remain outside the source package and should not be committed to the repository.

## Recommended Experiment Records

Each completed experiment should record:

- Model architecture and parameter values.
- Dataset and preprocessing pipeline.
- Optimizer, learning rate, scheduler, and number of epochs.
- Hardware and software environment.
- Training and validation metrics.
- Checkpoint or output location.
- Observations and possible follow-up changes.

This information makes comparisons between experiments easier and reduces dependence on undocumented settings.

## Planned Directions

The repository may expand beyond the initial image-classification example in several directions:

### Additional Vision Transformers

Implement and compare different model sizes and design choices, including changes to depth, embedding dimension, attention heads, regularization, and positional embeddings.

### Swin Transformers

Swin Transformer models are a planned future direction. Their shifted-window attention and hierarchical feature representations are especially relevant for vision tasks that require multi-scale features.

### Encoder-Decoder Architectures

Encoder-decoder models may be added to support tasks that require a structured output rather than a single class prediction. Possible applications include image reconstruction, dense prediction, and sequence-based experiments.

### Segmentation Models

Transformer backbones may eventually be used in image segmentation models. This will require dataset-specific masks, spatial feature decoding, suitable segmentation losses, and metrics such as intersection over union.

## Project Status

The items above are a roadmap, not a list of completed features. New experiments should be added only after the relevant implementation and evaluation code is available.
