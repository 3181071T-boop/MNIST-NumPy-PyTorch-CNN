"""Educational NumPy implementations of a linear classifier and a two-layer MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_cross_entropy(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return mean cross-entropy and d(loss)/d(logits)."""
    probabilities = softmax(logits)
    batch_size = logits.shape[0]
    loss = -np.log(probabilities[np.arange(batch_size), labels] + 1e-12).mean()
    d_logits = probabilities.copy()
    d_logits[np.arange(batch_size), labels] -= 1.0
    d_logits /= batch_size
    return float(loss), d_logits


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.maximum(1e-12, np.abs(a) + np.abs(b))
    return float(np.max(np.abs(a - b) / denominator))


def numerical_gradient(
    function: Callable[[np.ndarray], float], values: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Compute a central-difference numerical gradient."""
    gradient = np.zeros_like(values)
    iterator = np.nditer(values, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = values[index]
        values[index] = original + epsilon
        positive = function(values)
        values[index] = original - epsilon
        negative = function(values)
        values[index] = original
        gradient[index] = (positive - negative) / (2 * epsilon)
        iterator.iternext()
    return gradient


@dataclass
class NumpyHistory:
    train_loss: list[float]
    train_accuracy: list[float]


class LinearSoftmaxClassifier:
    """Linear classifier with mini-batch SGD."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((input_dim, num_classes))
        self.b = np.zeros(num_classes, dtype=np.float64)

    def loss_and_grads(self, x: np.ndarray, labels: np.ndarray, l2: float = 0.0):
        scores = x @ self.W + self.b
        data_loss, d_scores = softmax_cross_entropy(scores, labels)
        loss = data_loss + l2 * np.sum(self.W * self.W)
        grads = {
            "W": x.T @ d_scores + 2.0 * l2 * self.W,
            "b": np.sum(d_scores, axis=0),
        }
        return loss, grads

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(x @ self.W + self.b, axis=1)

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> NumpyHistory:
        rng = np.random.default_rng(seed)
        history = NumpyHistory([], [])
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch = order[start : start + batch_size]
                _, grads = self.loss_and_grads(x[batch], labels[batch], l2=l2)
                self.W -= learning_rate * grads["W"]
                self.b -= learning_rate * grads["b"]
            loss, _ = self.loss_and_grads(x, labels, l2=l2)
            accuracy = float(np.mean(self.predict(x) == labels))
            history.train_loss.append(float(loss))
            history.train_accuracy.append(accuracy)
        return history


class TwoLayerMLP:
    """Two-layer ReLU MLP with explicit forward and backward propagation."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.W1 = np.sqrt(2.0 / input_dim) * rng.standard_normal((input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = np.sqrt(2.0 / hidden_dim) * rng.standard_normal((hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float64)

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(0.0, z1)
        scores = h @ self.W2 + self.b2
        return scores, (x, z1, h)

    def loss_and_grads(self, x: np.ndarray, labels: np.ndarray, l2: float = 0.0):
        scores, cache = self.forward(x)
        loss, d_scores = softmax_cross_entropy(scores, labels)
        x_cached, z1, h = cache

        dW2 = h.T @ d_scores + 2.0 * l2 * self.W2
        db2 = np.sum(d_scores, axis=0)
        dh = d_scores @ self.W2.T
        dz1 = dh * (z1 > 0.0)
        dW1 = x_cached.T @ dz1 + 2.0 * l2 * self.W1
        db1 = np.sum(dz1, axis=0)
        loss += l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores, _ = self.forward(x)
        return np.argmax(scores, axis=1)

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> NumpyHistory:
        rng = np.random.default_rng(seed)
        history = NumpyHistory([], [])
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch = order[start : start + batch_size]
                _, grads = self.loss_and_grads(x[batch], labels[batch], l2=l2)
                self.W1 -= learning_rate * grads["W1"]
                self.b1 -= learning_rate * grads["b1"]
                self.W2 -= learning_rate * grads["W2"]
                self.b2 -= learning_rate * grads["b2"]
            loss, _ = self.loss_and_grads(x, labels, l2=l2)
            accuracy = float(np.mean(self.predict(x) == labels))
            history.train_loss.append(float(loss))
            history.train_accuracy.append(accuracy)
        return history
