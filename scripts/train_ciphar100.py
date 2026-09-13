# 1. importando o modelo

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vit_pytorch.models.simple_ViT import VisionTransformer

# 2. baixando os datasets

import torchvision.transforms as transforms
from torch import Generator
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

og_set = CIFAR100(
    root='./data', train=True, download=True, transform=train_transform
)

train_size = int(0.8 * len(og_set))
val_size   = len(og_set) - train_size

generator = Generator().manual_seed(42)

train_set, validation_set = random_split(og_set,[train_size, val_size],generator=generator)

test_set = CIFAR100(
    root='./data', train=False, download=True, transform=test_transform
)

# 3. dataloaders, etc

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

validation_loader = DataLoader(
    validation_set,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader  = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 4. pegando o modelo

model = VisionTransformer(
    img_size=32,
    patch_size=4,
    canais=3,
    n_classes=100,
    dim=192,
    depth=6,
    n_heads=3,
    mlp_ratio=3.0,
    qkv_bias=True,
    p=0.1,
    attn_p=0.1
)

# 5. transferindo pro dispositivo

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Using:", device)

# 6. loss, optimizer, scheduler, etc:

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.05
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100
)

# 7. loop de treinamneto

epocas = 100

for ep in range(epocas):
    model.train() # queremos treinar nosso modelo

    train_loss      = 0.0
    train_corretas  = 0
    train_total     = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_corretas += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)
    
    scheduler.step()

    train_accuracy = train_corretas / train_total

    print(
        f"Epoch [{ep+1}/{epocas}] "
        f"Loss: {train_loss / len(train_loader):.4f} "
        f"Accuracy: {train_accuracy:.4f}"
    )

    model.eval()

    validation_loss     = 0.0
    validation_total    = 0
    validation_correct  = 0

    with torch.no_grad():
        
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            validation_correct += (
                predictions == labels
            ).sum().item()

            validation_total += labels.size(0)

    validation_accuracy = validation_correct / validation_total

    print(
        f"loss de validação: {(validation_loss / len(validation_loader)):.4f} "
        f"acurácia de validação: {validation_accuracy:.4f} "
    )

        # roda o modelo em cima dos dados de validation