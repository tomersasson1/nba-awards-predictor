from __future__ import annotations

from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from ..data.feature_engineering import build_features, temporal_split, save_scaler
from .pytorch_base import MLPRegressor, TabularDataset, TrainConfig, train_model, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def train_dpoy_model(training_csv: Path, val_season: str = "2024-25") -> None:
    df = pd.read_csv(training_csv)
    award_df = df[df["AWARD_TYPE"] == "DPOY"].copy()
    if award_df.empty:
        return
    train_df, val_df = temporal_split(award_df, val_season)
    X_train, y_train, _, scaler, fcols = build_features(train_df)
    X_val, y_val, _, _, _ = build_features(val_df, scaler=scaler) if not val_df.empty else (X_train.iloc[:1], y_train.iloc[:1], None, None, [])
    model = MLPRegressor(input_dim=len(fcols))
    train_model(model,
                DataLoader(TabularDataset(X_train.to_numpy(), y_train.to_numpy()), batch_size=64, shuffle=True),
                DataLoader(TabularDataset(X_val.to_numpy(), y_val.to_numpy()), batch_size=64),
                TrainConfig())
    save_model(model, MODELS_DIR / "dpoy" / "dpoy_model.pt")
    save_scaler(scaler, "DPOY")
