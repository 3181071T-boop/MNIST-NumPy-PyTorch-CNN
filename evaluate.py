"""Evaluate a trained checkpoint and produce report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from .data import load_mnist
from .models import build_model
from .train import evaluate
from .utils import choose_device, ensure_dir, save_json, set_seed
from .visualize import plot_confusion_matrix, plot_misclassified


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    loaders = load_mnist(data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed)
    model = build_model(args.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate(
        model,
        loaders.test,
        nn.CrossEntropyLoss(),
        device,
        args.model,
        return_predictions=True,
    )
    labels = result["labels"].numpy()
    predictions = result["predictions"].numpy()
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    report = classification_report(
        labels, predictions, labels=list(range(10)), output_dict=True, zero_division=0
    )
    save_json(
        {"loss": result["loss"], "accuracy": result["accuracy"], "classification_report": report},
        output_dir / "test_metrics.json",
    )
    np.save(output_dir / "confusion_matrix.npy", matrix)
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    plot_misclassified(
        result["images"], labels, predictions, output_dir / "misclassified.png"
    )
    print(f"Test loss: {result['loss']:.4f}")
    print(f"Test accuracy: {result['accuracy']:.4f}")
    print(f"Saved evaluation artifacts to: {output_dir.resolve()}")
    return {"loss": result["loss"], "accuracy": result["accuracy"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["linear", "mlp", "cnn"], default="cnn")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
