from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.25,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 300
    patience: int = 30
    device: str = "cpu"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
) -> Dict[str, float]:
    device = torch.device(config.device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            if X_batch.size(0) < 2:
                continue  # BatchNorm requires batch size >= 2
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_losses.append(criterion(model(X_batch), y_batch).item())

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss}


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
