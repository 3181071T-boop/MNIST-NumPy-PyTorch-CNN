"""Preprocess personal handwritten digit images to the MNIST format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .data import MNIST_MEAN, MNIST_STD


def preprocess_custom_image(path: str | Path) -> torch.Tensor:
    """Return a personal image as a normalized tensor of shape (1, 1, 28, 28)."""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0

    border = np.concatenate(
        [array[0, :], array[-1, :], array[:, 0], array[:, -1]]
    )
    if float(border.mean()) > 0.5:
        array = 1.0 - array

    foreground = array > max(0.08, float(array.max()) * 0.15)
    coordinates = np.argwhere(foreground)
    if coordinates.size:
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0) + 1
        array = array[top:bottom, left:right]

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    image = ImageOps.contain(image, (20, 20), method=Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))

    tensor = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0)
    tensor = (tensor - MNIST_MEAN) / MNIST_STD
    return tensor.unsqueeze(0).unsqueeze(0)
