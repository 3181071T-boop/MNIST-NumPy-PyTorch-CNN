"""Train linear, MLP, or CNN MNIST models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .data import load_mnist
from .models import build_model
from .utils import accuracy_from_logits, choose_device, count_parameters, ensure_dir, save_json, set_seed
from .visualize import plot_history, save_image_grid


def _forward(model: nn.Module, images: torch.Tensor, model_name: str) -> torch.Tensor:
    return model(images)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str = "cnn",
) -> dict[str, float]:
    """Train one epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, images, model_name)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += labels.size(0)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "cnn",
    return_predictions: bool = False,
) -> dict:
    """Evaluate a model; optionally return all labels, predictions, and images."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_images, all_labels, all_predictions = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = _forward(model, images, model_name)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += labels.size(0)
        if return_predictions:
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
    result = {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
    if return_predictions:
        result.update(
            {
                "images": torch.cat(all_images),
                "labels": torch.cat(all_labels),
                "predictions": torch.cat(all_predictions),
            }
        )
    return result


def save_checkpoint(model, optimizer, epoch, metrics, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        target,
    )


def run_training(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    loaders = load_mnist(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_augmentation=args.augment,
        max_train=args.max_train,
    )
    first_images, first_labels = next(iter(loaders.train))
    save_image_grid(first_images, first_labels, output_dir / "sample_grid.png")

    model = build_model(args.model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_accuracy = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders.train, criterion, optimizer, device, args.model
        )
        validation_metrics = evaluate(
            model, loaders.validation, criterion, device, args.model
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"acc={validation_metrics['accuracy']:.4f}"
        )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, validation_metrics, output_dir / "best.pt")

    elapsed = time.perf_counter() - started
    plot_history(history, output_dir / "training_curves.png")
    save_json(history, output_dir / "history.json")
    metadata = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "augmentation": args.augment,
        "seed": args.seed,
        "device": str(device),
        "parameters": count_parameters(model),
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
    }
    save_json(metadata, output_dir / "metadata.json")
    print(f"Saved checkpoint and reports to: {output_dir.resolve()}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "mlp", "cnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/cnn")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
