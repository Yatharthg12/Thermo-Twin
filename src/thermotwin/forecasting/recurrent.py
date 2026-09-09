"""Compact genuinely recurrent GRU with deterministic training and early stopping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GRUForecaster(nn.Module):
    """Shared rack-level GRU producing direct forecasts for all requested horizons."""

    def __init__(self, n_features: int, hidden_size: int, n_horizons: int):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        return self.head(output[:, -1, :])


@dataclass
class TrainedGRU:
    """Model plus training-only normalization and observed training history."""

    model: GRUForecaster
    mean: np.ndarray
    scale: np.ndarray
    history: list[dict[str, float]]
    trained_epochs: int

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        self.model.eval()
        normalized = (X - self.mean) / self.scale
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                tensor = torch.from_numpy(normalized[start:start + batch_size].astype(np.float32))
                outputs.append(self.model(tensor).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.model.head.out_features))


def train_gru(X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray, y_validation: np.ndarray, config: dict[str, Any], seed: int) -> TrainedGRU:
    """Train with validation early stopping and restore the best checkpoint."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    np.random.seed(seed)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    scale = X_train.std(axis=(0, 1), keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_x = torch.from_numpy(((X_train - mean) / scale).astype(np.float32))
    train_y = torch.from_numpy(y_train.astype(np.float32))
    val_x = torch.from_numpy(((X_validation - mean) / scale).astype(np.float32))
    val_y = torch.from_numpy(y_validation.astype(np.float32))
    model = GRUForecaster(X_train.shape[-1], int(config["forecasting"]["recurrent_hidden"]), y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(config["forecasting"]["batch_size"]), shuffle=True, generator=generator)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    epochs = int(config["forecasting"]["recurrent_epochs"])
    patience = int(config["forecasting"]["recurrent_patience"])
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("GRU training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainedGRU(model, mean, scale, history, len(history))

