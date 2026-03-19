from __future__ import annotations

"""
Simple baseline models (linear regression / random forest) for comparison
with the PyTorch neural networks.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ..data.feature_engineering import build_features


def train_baseline_for_award(training_csv: Path, award_type: str) -> None:
    df = pd.read_csv(training_csv)
    df_award = df[df["AWARD_TYPE"] == award_type].copy()
    if df_award.empty:
        print(f"No data for award {award_type} in baseline training.")
        return

    tmp_path = training_csv.parent / f"_tmp_{award_type}_baseline.csv"
    df_award.to_csv(tmp_path, index=False)

    X_df, y_series, _meta, _scaler = build_features(tmp_path)
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    out_dir = Path("models") / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_dir / f"{award_type}_rf.joblib")

    print(f"Trained baseline RF for {award_type}, validation MAE={mae:.4f}")


