from __future__ import annotations

"""
Generate predictions for the current in-progress season.

Loads trained models + scalers, reads current-season player stats,
computes features, applies per-award eligibility filters, runs inference,
and saves ranked predictions per award.

COTY is handled separately since it operates on team/coach data, not player stats.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from ..data.feature_engineering import engineer_features, load_scaler_and_features
from ..data.eligibility import enrich_with_metadata, AWARD_FILTERS
from .pytorch_base import MLPRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PLAYER_AWARD_TYPES = ["MVP", "DPOY", "ROTY", "MIP", "6MOY"]
ALL_AWARD_TYPES = ["MVP", "DPOY", "ROTY", "MIP", "6MOY", "COTY"]


def _load_raw_players() -> pd.DataFrame:
    candidates = sorted(RAW_DIR.glob("players_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError("No raw player stats found. Run ingest first.")
    df = pd.read_csv(candidates[0]).copy()
    df["player_name"] = df["PLAYER_NAME"]
    df["season"] = df["SEASON"]
    return df


def _load_raw_teams() -> pd.DataFrame:
    candidates = sorted(RAW_DIR.glob("teams_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return pd.DataFrame()
    df = pd.read_csv(candidates[0])
    df["season"] = df["SEASON"]
    return df


def _prepare_prediction_data() -> tuple[pd.DataFrame, str]:
    """
    Prepare feature-engineered data for the current season.
    Enriches with authoritative metadata (DRAFT_YEAR, EXP, IS_ROOKIE, etc.).
    """
    players = _load_raw_players()
    teams = _load_raw_teams()

    if not teams.empty and "W_PCT" in teams.columns:
        team_wpct = teams[["TEAM_ID", "season", "W_PCT"]].drop_duplicates()
        team_wpct = team_wpct.rename(columns={"W_PCT": "TEAM_WIN_PCT"})
        players = players.merge(team_wpct, on=["TEAM_ID", "season"], how="left")

    if "TEAM_WIN_PCT" not in players.columns:
        if "W" in players.columns and "L" in players.columns:
            total = players["W"] + players["L"]
            players["TEAM_WIN_PCT"] = np.where(total > 0, players["W"] / total, 0.0)
        else:
            players["TEAM_WIN_PCT"] = 0.0

    # Enrich with authoritative metadata (draft year, experience, rookie flag)
    players = enrich_with_metadata(players)

    seasons = sorted(players["season"].unique())
    current_season = seasons[-1]

    last_two = seasons[-2:] if len(seasons) >= 2 else seasons
    subset = players[players["season"].isin(last_two)].copy()
    subset["AWARD_TYPE"] = "PREDICT"

    subset, _ = engineer_features(subset)

    current_featured = subset[subset["season"] == current_season].copy()
    return current_featured, current_season


def predict_award(
    current_df: pd.DataFrame,
    award_type: str,
) -> pd.DataFrame:
    model_path = MODELS_DIR / award_type.lower() / f"{award_type.lower()}_model.pt"
    if not model_path.exists():
        print(f"  No trained model for {award_type}, skipping.")
        return pd.DataFrame()

    filter_fn = AWARD_FILTERS.get(award_type)
    if filter_fn:
        eligible = filter_fn(current_df)
        n_filtered = len(current_df) - len(eligible)
        print(f"    Eligible candidates: {len(eligible)} (filtered {n_filtered})")
    else:
        eligible = current_df

    if eligible.empty:
        print(f"    No eligible candidates for {award_type}.")
        return pd.DataFrame()

    scaler, train_feature_cols = load_scaler_and_features(award_type)
    if not train_feature_cols:
        print(f"  No feature column list saved for {award_type}, skipping.")
        return pd.DataFrame()

    X_raw = pd.DataFrame(index=eligible.index)
    for col in train_feature_cols:
        if col in eligible.columns:
            X_raw[col] = eligible[col].astype(float)
        else:
            X_raw[col] = 0.0
    X_raw = X_raw.fillna(0.0)

    X_scaled = scaler.transform(X_raw)

    n_features = len(train_feature_cols)
    model = MLPRegressor(input_dim=n_features)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.as_tensor(X_scaled, dtype=torch.float32)).squeeze().numpy()
    if preds.ndim == 0:
        preds = np.array([preds.item()])

    result = eligible[["player_name", "TEAM_ABBREVIATION", "season"]].copy()
    result["AWARD_TYPE"] = award_type
    result["predicted_vote_share"] = preds

    for col in ["PTS", "REB", "AST", "STL", "BLK", "GP", "MIN",
                 "TEAM_WIN_PCT", "AGE", "NBA_EXPERIENCE", "IS_ROOKIE"]:
        if col in eligible.columns:
            result[col] = eligible[col].values

    result = result.sort_values("predicted_vote_share", ascending=False).reset_index(drop=True)
    result["predicted_rank"] = range(1, len(result) + 1)

    # Normalize within-pool: rescale so top candidate = ~1.0, gives relative ranking
    raw_max = result["predicted_vote_share"].max()
    if raw_max > 0:
        result["predicted_vote_share_raw"] = result["predicted_vote_share"]
        result["predicted_vote_share"] = result["predicted_vote_share"] / raw_max
    else:
        # Model gives all zeros -- fall back to a simple stat-based heuristic
        if award_type == "ROTY" and "PTS" in result.columns:
            score = (
                result["PTS"].fillna(0) * 0.4
                + result.get("REB", pd.Series(0, index=result.index)).fillna(0) * 0.15
                + result.get("AST", pd.Series(0, index=result.index)).fillna(0) * 0.15
                + result.get("MIN", pd.Series(0, index=result.index)).fillna(0) * 0.01
                + result.get("GP", pd.Series(0, index=result.index)).fillna(0) * 0.005
            )
            score_max = score.max()
            result["predicted_vote_share"] = score / score_max if score_max > 0 else 0
            result = result.sort_values("predicted_vote_share", ascending=False).reset_index(drop=True)
            result["predicted_rank"] = range(1, len(result) + 1)

    return result


def predict_coty(current_season: str) -> pd.DataFrame:
    """
    Predict Coach of the Year based on team performance.
    Uses team win %, improvement vs prior season, and model if available.
    """
    model_path = MODELS_DIR / "coty" / "coty_model.pt"

    coaches_path = RAW_DIR / "coaches.csv"
    if not coaches_path.exists():
        print("  No coaches data found. Run ingest with --include-current first.")
        return pd.DataFrame()

    coaches = pd.read_csv(coaches_path)
    head_coaches = coaches[coaches["IS_ASSISTANT"] == 1].copy()
    if head_coaches.empty:
        head_coaches = coaches[coaches["COACH_TYPE"].str.contains("Head", case=False, na=False)].copy()
    if head_coaches.empty:
        print("  No head coaches found in data.")
        return pd.DataFrame()

    # Get current team standings
    teams = _load_raw_teams()
    if teams.empty:
        print("  No team stats for COTY prediction.")
        return pd.DataFrame()

    current_teams = teams[teams["season"] == current_season].copy()
    if current_teams.empty:
        current_teams = teams[teams["SEASON"] == current_season].copy()

    result_rows = []
    for _, coach in head_coaches.iterrows():
        team_id = coach["TEAM_ID"]
        team_stats = current_teams[current_teams["TEAM_ID"] == team_id]
        if team_stats.empty:
            continue

        ts = team_stats.iloc[0]
        w_pct = ts.get("W_PCT", 0.0)
        wins = ts.get("W", 0)
        losses = ts.get("L", 0)

        # Prior season comparison
        prior_teams = teams[teams["season"] < current_season]
        prior = prior_teams[prior_teams["TEAM_ID"] == team_id]
        prev_wpct = prior.groupby("season")["W_PCT"].first().iloc[-1] if not prior.empty else 0.5
        wpct_delta = w_pct - prev_wpct

        result_rows.append({
            "player_name": coach["COACH_NAME"],
            "TEAM_ABBREVIATION": coach.get("TEAM_ABBREVIATION", "?"),
            "season": current_season,
            "AWARD_TYPE": "COTY",
            "W": int(wins),
            "L": int(losses),
            "TEAM_WIN_PCT": round(w_pct, 3),
            "WIN_PCT_IMPROVEMENT": round(wpct_delta, 3),
        })

    if not result_rows:
        return pd.DataFrame()

    result = pd.DataFrame(result_rows)

    # COTY uses a heuristic based on historical patterns:
    # COTY winners typically have high W% AND significant improvement over last season
    result["predicted_vote_share"] = (
        result["TEAM_WIN_PCT"] * 0.5
        + result["WIN_PCT_IMPROVEMENT"].clip(lower=0) * 0.5
    )

    result = result.sort_values("predicted_vote_share", ascending=False).reset_index(drop=True)
    result["predicted_rank"] = range(1, len(result) + 1)
    return result


def predict_all(with_trends: bool = False) -> pd.DataFrame:
    print("Preparing prediction features...")
    current_featured, current_season = _prepare_prediction_data()
    print(f"  {len(current_featured)} players for {current_season}")

    if "IS_ROOKIE" in current_featured.columns:
        n_rookies = current_featured["IS_ROOKIE"].sum()
        print(f"  Rookies identified: {n_rookies}")

    all_predictions = []
    for award in PLAYER_AWARD_TYPES:
        print(f"  Predicting {award}...")
        preds = predict_award(current_featured, award)
        if not preds.empty:
            all_predictions.append(preds)
            for _, row in preds.head(5).iterrows():
                extra = ""
                if "AGE" in row.index and pd.notna(row.get("AGE")):
                    extra += f"age={int(row['AGE'])} "
                if "NBA_EXPERIENCE" in row.index and pd.notna(row.get("NBA_EXPERIENCE")):
                    extra += f"exp={int(row['NBA_EXPERIENCE'])}yr "
                if "IS_ROOKIE" in row.index and row.get("IS_ROOKIE"):
                    extra += "[ROOKIE] "
                print(f"    {int(row['predicted_rank'])}. {row['player_name']} "
                      f"({row.get('TEAM_ABBREVIATION', '?')}) "
                      f"-- {row.get('PTS', 0):.1f}ppg {row.get('MIN', 0):.0f}min {extra}"
                      f"pred: {row['predicted_vote_share']:.3f}")

    # COTY prediction
    print(f"  Predicting COTY...")
    coty_preds = predict_coty(current_season)
    if not coty_preds.empty:
        all_predictions.append(coty_preds)
        for _, row in coty_preds.head(5).iterrows():
            print(f"    {int(row['predicted_rank'])}. {row['player_name']} "
                  f"({row.get('TEAM_ABBREVIATION', '?')}) "
                  f"-- {row.get('W', 0)}-{row.get('L', 0)} "
                  f"({row.get('TEAM_WIN_PCT', 0):.3f}) "
                  f"delta={row.get('WIN_PCT_IMPROVEMENT', 0):+.3f} "
                  f"pred: {row['predicted_vote_share']:.3f}")

    if all_predictions:
        result = pd.concat(all_predictions, ignore_index=True)

        if with_trends:
            print("\n  Fetching Google Trends (media hype)...")
            try:
                from ..data.google_trends import (
                    fetch_trends_for_predictions,
                    merge_trends_into_predictions,
                )
                trends = fetch_trends_for_predictions(result, top_n_per_award=15)
                result = merge_trends_into_predictions(result, trends)
                print("  Trends data merged (80% model + 20% media hype).")
            except Exception as exc:
                print(f"  Warning: Trends fetch failed ({exc}). Using model-only predictions.")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "predictions_current.csv"
        result.to_csv(out_path, index=False)
        print(f"\nSaved predictions to {out_path}")
        return result

    print("No predictions generated.")
    return pd.DataFrame()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-trends", action="store_true",
        help="Fetch Google Trends data to factor in media hype (adds ~5 min).",
    )
    args = parser.parse_args()
    predict_all(with_trends=args.with_trends)


if __name__ == "__main__":
    main()
