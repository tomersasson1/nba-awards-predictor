from __future__ import annotations

"""
Evaluation utilities for trained models.

Metrics:
- MAE / RMSE on vote share
- Per-season Spearman rank correlation
- Top-1 and Top-3 accuracy (did the actual winner appear in predicted top-K?)
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from ..data.feature_engineering import build_features, load_scaler
from .pytorch_base import MLPRegressor


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    if len(actual) < 2:
        return float("nan")
    corr, _ = spearmanr(actual, predicted)
    return float(corr) if not np.isnan(corr) else float("nan")


def evaluate_award(
    df: pd.DataFrame,
    award_type: str,
    model_path: Path,
    feature_cols: List[str],
) -> Dict[str, object]:
    """
    Comprehensive evaluation for one award on the given DataFrame (expected to
    contain actual vote_share for comparison).
    """
    adf = df[df["AWARD_TYPE"] == award_type].copy()
    if adf.empty:
        return {"award": award_type, "n_seasons": 0}

    scaler = load_scaler(award_type)
    X_raw = adf[feature_cols].astype(float).fillna(0.0)
    X_scaled = scaler.transform(X_raw)

    model = MLPRegressor(input_dim=len(feature_cols))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.as_tensor(X_scaled, dtype=torch.float32)).squeeze().numpy()

    adf["pred_vote_share"] = preds

    actual = adf["vote_share"].values
    mae = float(np.mean(np.abs(actual - preds)))
    rmse = float(np.sqrt(np.mean((actual - preds) ** 2)))

    per_season_spearman: List[float] = []
    top1_correct = 0
    top3_correct = 0
    n_seasons = 0

    for season, grp in adf.groupby("season"):
        if grp.shape[0] < 2:
            continue
        n_seasons += 1

        sp = _spearman(grp["vote_share"], grp["pred_vote_share"])
        if not np.isnan(sp):
            per_season_spearman.append(sp)

        actual_top1 = grp.sort_values("vote_share", ascending=False).iloc[0]["player_name"]
        pred_top1 = grp.sort_values("pred_vote_share", ascending=False).iloc[0]["player_name"]
        if actual_top1 == pred_top1:
            top1_correct += 1

        actual_top3 = set(grp.sort_values("vote_share", ascending=False).head(3)["player_name"])
        pred_top3 = set(grp.sort_values("pred_vote_share", ascending=False).head(3)["player_name"])
        if actual_top3 & pred_top3:
            top3_correct += 1

    avg_spearman = float(np.mean(per_season_spearman)) if per_season_spearman else float("nan")

    return {
        "award": award_type,
        "n_seasons": n_seasons,
        "mae": mae,
        "rmse": rmse,
        "avg_spearman": avg_spearman,
        "per_season_spearman": per_season_spearman,
        "top1_accuracy": top1_correct / max(n_seasons, 1),
        "top3_accuracy": top3_correct / max(n_seasons, 1),
    }


def load_trained_model(model_path: Path, input_dim: int) -> MLPRegressor:
    model = MLPRegressor(input_dim=input_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
