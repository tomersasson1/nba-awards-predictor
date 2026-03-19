from __future__ import annotations

"""MVP model -- delegates to shared training pipeline."""

from pathlib import Path
import pandas as pd

from ..data.feature_engineering import build_features, temporal_split, save_scaler
from .pytorch_base import MLPRegressor, TabularDataset, TrainConfig, train_model, save_model

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def train_mvp_model(training_csv: Path, val_season: str = "2024-25") -> None:
    df = pd.read_csv(training_csv)
    award_df = df[df["AWARD_TYPE"] == "MVP"].copy()
    if award_df.empty:
        print("  No MVP data, skipping.")
        return

    train_df, val_df = temporal_split(award_df, val_season)
    X_train, y_train, _, scaler, feature_cols = build_features(train_df)
    X_val, y_val, _, _, _ = build_features(val_df, scaler=scaler) if not val_df.empty else (X_train.iloc[:1], y_train.iloc[:1], None, None, [])

    train_loader = DataLoader(TabularDataset(X_train.to_numpy(), y_train.to_numpy()), batch_size=64, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val.to_numpy(), y_val.to_numpy()), batch_size=64)

    model = MLPRegressor(input_dim=len(feature_cols))
    train_model(model, train_loader, val_loader, TrainConfig())
    save_model(model, MODELS_DIR / "mvp" / "mvp_model.pt")
    save_scaler(scaler, "MVP")
