from __future__ import annotations

"""
Feature engineering for NBA award vote-share prediction.

Builds ~40 features from the merged training dataset, organized in tiers:
  Tier 1 -- Advanced stats (OFF_RATING, DEF_RATING, USG_PCT, TS_PCT, PIE, ...)
  Tier 2 -- Engineered context (market size, eligibility, ranks, efficiency)
  Tier 3 -- Year-over-year deltas (critical for MIP)

Also provides temporal train/val/predict splitting and a StandardScaler pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Columns we pull from the merged CSV as raw features
BASE_STAT_COLS = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "GP", "MIN", "PLUS_MINUS", "AGE",
]

ADVANCED_STAT_COLS = [
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "USG_PCT", "TS_PCT", "EFG_PCT", "PIE", "PACE",
    "AST_PCT", "AST_RATIO",
    "OREB_PCT", "DREB_PCT", "REB_PCT",
]

DELTA_SOURCE_COLS = ["PTS", "REB", "AST", "TS_PCT", "NET_RATING"]


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(0.0, index=df.index)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add all engineered feature columns to *df* in-place and return
    (df_with_features, feature_column_names).
    """
    out = df.copy()

    # --- Tier 1: collect raw stat columns that exist ---
    feature_cols: List[str] = []
    for col in BASE_STAT_COLS + ADVANCED_STAT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            feature_cols.append(col)

    # --- Tier 2: Engineered features ---

    # Team win pct (may already exist from preprocess)
    if "TEAM_WIN_PCT" not in out.columns:
        if "W" in out.columns and "L" in out.columns:
            total = out["W"] + out["L"]
            out["TEAM_WIN_PCT"] = np.where(total > 0, out["W"] / total, 0.0)
        else:
            out["TEAM_WIN_PCT"] = 0.0
    feature_cols.append("TEAM_WIN_PCT")

    # Games played fraction and 65-game eligibility
    if "GP" in out.columns:
        team_gp = out.groupby(["TEAM_ABBREVIATION", "season"])["GP"].transform("max")
        out["games_pct"] = np.where(team_gp > 0, out["GP"] / team_gp, 0.0)
        out["eligible_65"] = (out["GP"] >= 65).astype(float)
    else:
        out["games_pct"] = 0.0
        out["eligible_65"] = 0.0
    feature_cols.extend(["games_pct", "eligible_65"])

    # Per-season stat ranks (lower = better)
    for stat, rank_name in [("PTS", "ppg_rank"), ("REB", "rpg_rank"), ("AST", "apg_rank")]:
        if stat in out.columns:
            out[rank_name] = out.groupby("season")[stat].rank(ascending=False, method="min")
            feature_cols.append(rank_name)

    # Team conference seed proxy (rank teams by W_PCT within season, 1=best)
    out["team_conf_seed"] = out.groupby("season")["TEAM_WIN_PCT"].rank(
        ascending=False, method="min"
    )
    out["top3_seed"] = (out["team_conf_seed"] <= 3).astype(float)
    feature_cols.extend(["team_conf_seed", "top3_seed"])

    # Stats above season average
    for stat, col_name in [("PTS", "pts_above_avg"), ("REB", "reb_above_avg"), ("AST", "ast_above_avg")]:
        if stat in out.columns:
            season_avg = out.groupby("season")[stat].transform("mean")
            out[col_name] = out[stat] - season_avg
            feature_cols.append(col_name)

    # Starter flag (for 6MOY -- bench players have < 24 min)
    if "MIN" in out.columns:
        out["is_starter"] = (out["MIN"] >= 24.0).astype(float)
        feature_cols.append("is_starter")

    # Scoring efficiency: points per true shooting attempt
    if "PTS" in out.columns and "FGA" in out.columns and "FTA" in out.columns:
        tsa = out["FGA"] * 2 + out["FTA"] * 0.44
        out["scoring_efficiency"] = np.where(tsa > 0, out["PTS"] / tsa, 0.0)
        feature_cols.append("scoring_efficiency")

    # Double-double and triple-double rates
    if "DD2" in out.columns and "GP" in out.columns:
        out["dd_rate"] = np.where(out["GP"] > 0, out["DD2"] / out["GP"], 0.0)
        feature_cols.append("dd_rate")
    if "TD3" in out.columns and "GP" in out.columns:
        out["td_rate"] = np.where(out["GP"] > 0, out["TD3"] / out["GP"], 0.0)
        feature_cols.append("td_rate")

    # --- Tier 3: Year-over-year deltas ---
    if "PLAYER_ID" in out.columns:
        out = out.sort_values(["PLAYER_ID", "season"]).reset_index(drop=True)
        for stat in DELTA_SOURCE_COLS:
            if stat not in out.columns:
                continue
            delta_col = f"{stat.lower()}_delta"
            out[delta_col] = out.groupby("PLAYER_ID")[stat].diff()
            out[delta_col] = out[delta_col].fillna(0.0)
            feature_cols.append(delta_col)

        # Team win pct delta
        out["win_pct_delta"] = out.groupby("PLAYER_ID")["TEAM_WIN_PCT"].diff().fillna(0.0)
        feature_cols.append("win_pct_delta")

    # De-duplicate feature list (in case any col was added twice)
    feature_cols = list(dict.fromkeys(feature_cols))

    # Fill remaining NaN
    for col in feature_cols:
        out[col] = out[col].fillna(0.0)

    return out, feature_cols


def build_features(
    df: pd.DataFrame,
    award_type: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, StandardScaler, List[str]]:
    """
    Build feature matrix X, target y, metadata, and scaler.

    Parameters
    ----------
    df : merged training DataFrame (from preprocess CSV or in-memory)
    award_type : if set, filter to a single award before feature building
    scaler : if provided, use this scaler (transform only); otherwise fit a new one

    Returns
    -------
    X, y, meta_df, scaler, feature_cols
    """
    if award_type:
        df = df[df["AWARD_TYPE"] == award_type].copy()
    if df.empty:
        empty_x = pd.DataFrame()
        return empty_x, pd.Series(dtype=float), pd.DataFrame(), scaler or StandardScaler(), []

    df, feature_cols = engineer_features(df)

    # Target
    if "vote_share" in df.columns:
        y = df["vote_share"].fillna(0.0)
    else:
        df["vote_points"] = pd.to_numeric(df.get("vote_points", 0), errors="coerce").fillna(0)
        max_pts = df.groupby(["AWARD_TYPE", "season"])["vote_points"].transform("max").replace(0, np.nan)
        y = (df["vote_points"] / max_pts).fillna(0.0)

    # Metadata for evaluation / display
    meta_cols = ["AWARD_TYPE", "season", "player_name"]
    if "TEAM_ABBREVIATION" in df.columns:
        meta_cols.append("TEAM_ABBREVIATION")
    meta_df = df[meta_cols].copy()

    # Scale features
    X_raw = df[feature_cols].astype(float)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)

    X = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    return X, y, meta_df, scaler, feature_cols


def temporal_split(
    df: pd.DataFrame,
    val_season: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a merged training DataFrame by season.

    Returns (train_df, val_df) where val_df contains only `val_season`.
    """
    train_df = df[df["season"] < val_season].copy()
    val_df = df[df["season"] == val_season].copy()
    return train_df, val_df


def save_scaler(scaler: StandardScaler, award_type: str, feature_cols: Optional[List[str]] = None) -> Path:
    out_dir = MODELS_DIR / award_type.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scaler.joblib"
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols or []}, path)
    return path


def load_scaler(award_type: str) -> StandardScaler:
    path = MODELS_DIR / award_type.lower() / "scaler.joblib"
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["scaler"]
    return obj


def load_scaler_and_features(award_type: str) -> tuple[StandardScaler, List[str]]:
    path = MODELS_DIR / award_type.lower() / "scaler.joblib"
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj["scaler"], obj["feature_cols"]
    return obj, []
