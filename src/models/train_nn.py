from __future__ import annotations

"""
Unified training script for all NBA award models.

Uses temporal train/val splitting and the shared feature engineering pipeline.
Trains one MLPRegressor per award and saves model + scaler.
"""

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ..data.feature_engineering import (
    build_features,
    temporal_split,
    save_scaler,
)
from .pytorch_base import MLPRegressor, TabularDataset, TrainConfig, train_model, save_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

AWARD_TYPES = ["MVP", "DPOY", "ROTY", "MIP", "6MOY"]


def train_award(
    full_df: pd.DataFrame,
    award_type: str,
    val_season: str,
    config: TrainConfig,
) -> None:
    award_df = full_df[full_df["AWARD_TYPE"] == award_type].copy()
    if award_df.empty:
        print(f"  No data for {award_type}, skipping.")
        return

    train_df, val_df = temporal_split(award_df, val_season)
    if train_df.empty:
        print(f"  No training data for {award_type} before {val_season}, skipping.")
        return

    X_train, y_train, _, scaler, feature_cols = build_features(train_df, scaler=None)
    if X_train.empty:
        print(f"  No features for {award_type}, skipping.")
        return

    if not val_df.empty:
        X_val, y_val, _, _, _ = build_features(val_df, scaler=scaler)
    else:
        X_val, y_val = X_train.iloc[:1], y_train.iloc[:1]

    train_ds = TabularDataset(X_train.to_numpy(), y_train.to_numpy())
    val_ds = TabularDataset(X_val.to_numpy(), y_val.to_numpy())
    batch_sz = min(config.batch_size, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, drop_last=len(train_ds) > batch_sz)
    val_batch = min(config.batch_size, len(val_ds))
    val_loader = DataLoader(val_ds, batch_size=val_batch, drop_last=False)

    model = MLPRegressor(input_dim=len(feature_cols))
    metrics = train_model(model, train_loader, val_loader, config)

    award_dir = MODELS_DIR / award_type.lower()
    save_model(model, award_dir / f"{award_type.lower()}_model.pt")
    save_scaler(scaler, award_type, feature_cols)

    print(f"  {award_type}: val_loss={metrics['best_val_loss']:.4f}, features={len(feature_cols)}")


def train_all(training_csv: Path, val_season: str = "2024-25") -> None:
    print(f"Loading training data from {training_csv}")
    full_df = pd.read_csv(training_csv)

    seasons = sorted(full_df["season"].unique())
    print(f"Seasons in dataset: {seasons[0]} to {seasons[-1]} ({len(seasons)} total)")
    print(f"Validation season: {val_season}")
    print()

    config = TrainConfig()

    for award in AWARD_TYPES:
        print(f"Training {award}...")
        train_award(full_df, award, val_season, config)

    print("\nAll models trained.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch models for NBA awards.")
    parser.add_argument("--training-csv", type=str, default="")
    parser.add_argument("--val-season", type=str, default="2024-25",
                        help="Season to hold out for validation (default: 2024-25).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.training_csv:
        training_csv = Path(args.training_csv)
    else:
        candidates = sorted(PROCESSED_DIR.glob("training_dataset_*.csv"))
        if not candidates:
            raise SystemExit("No processed training dataset found. Run preprocess first.")
        training_csv = candidates[0]
    train_all(training_csv, val_season=args.val_season)


if __name__ == "__main__":
    main()
