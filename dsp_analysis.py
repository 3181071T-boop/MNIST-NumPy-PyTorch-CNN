"""Optional DSP analysis for MNIST images."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_mnist_numpy
from .utils import ensure_dir


def fft_magnitude(image: np.ndarray) -> np.ndarray:
    """Return centered log-magnitude spectrum of a 2D image."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(spectrum))


def frequency_filter(image: np.ndarray, cutoff: float = 0.25, mode: str = "low") -> np.ndarray:
    """Apply a circular low-pass or high-pass filter in the frequency domain."""
    rows, columns = image.shape
    y, x = np.ogrid[-0.5 : 0.5 : complex(rows), -0.5 : 0.5 : complex(columns)]
    radius = np.sqrt(x * x + y * y)
    mask = radius <= cutoff
    if mode == "high":
        mask = ~mask
    if mode not in {"low", "high"}:
        raise ValueError("mode must be 'low' or 'high'")
    shifted = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(shifted * mask)).real
    return np.clip(filtered, 0.0, 1.0)


def save_dsp_examples(data_dir: str, output_dir: str, count: int = 6) -> None:
    images, labels = load_mnist_numpy(data_dir=data_dir, train=False, max_samples=count)
    output = ensure_dir(output_dir)
    figure, axes = plt.subplots(count, 4, figsize=(10, 2.2 * count))
    axes = np.asarray(axes).reshape(count, 4)
    for row, (image, label) in enumerate(zip(images, labels)):
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"digit={label}")
        axes[row, 1].imshow(fft_magnitude(image), cmap="magma")
        axes[row, 1].set_title("log spectrum")
        axes[row, 2].imshow(frequency_filter(image, mode="low"), cmap="gray")
        axes[row, 2].set_title("low-pass")
        axes[row, 3].imshow(frequency_filter(image, mode="high"), cmap="gray")
        axes[row, 3].set_title("high-pass")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(Path(output) / "dsp_examples.png", dpi=150)
    plt.close(figure)
    print(f"Saved DSP examples to: {Path(output).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/dsp")
    parser.add_argument("--count", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_dsp_examples(args.data_dir, args.output_dir, args.count)
