"""PyTorch model definitions."""

from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.flatten(start_dim=1))


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str, dropout: float = 0.0) -> nn.Module:
    """Build a supported model by name."""
    normalized = name.lower()
    if normalized == "linear":
        return LinearClassifier()
    if normalized == "mlp":
        return MLP()
    if normalized == "cnn":
        return SmallCNN(dropout=dropout)
    raise ValueError(f"Unknown model '{name}'. Choose from: linear, mlp, cnn")
