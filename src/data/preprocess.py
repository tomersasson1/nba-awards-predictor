from __future__ import annotations

"""
Preprocessing script for raw NBA stats and award voting data.

Responsibilities:
- Read raw CSVs produced by `src.data.ingest`.
- Align player/team identifiers across nba_api stats and Basketball Reference awards.
- Apply basic cleaning and eligibility filters.
- Produce a unified training dataset CSV in `data/processed/`.
"""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw(
    start_season: int,
    end_season: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players_path = RAW_DIR / f"players_{start_season}_{end_season}.csv"
    teams_path = RAW_DIR / f"teams_{start_season}_{end_season}.csv"
    awards_path = RAW_DIR / f"awards_{start_season}_{end_season}.csv"

    players_df = pd.read_csv(players_path)
    teams_df = pd.read_csv(teams_path)
    awards_df = pd.read_csv(awards_path)
    return players_df, teams_df, awards_df


def _normalize_player_name(name: str) -> str:
    return str(name).strip().lower()


def _prepare_players(players_df: pd.DataFrame) -> pd.DataFrame:
    df = players_df.copy()
    # nba_api LeagueDashPlayerStats typically exposes these identifiers
    df["player_name_norm"] = df["PLAYER_NAME"].apply(_normalize_player_name)
    df["season"] = df["SEASON"]
    return df


def _prepare_teams(teams_df: pd.DataFrame) -> pd.DataFrame:
    df = teams_df.copy()
    df["season"] = df["SEASON"]
    return df


def _prepare_awards(awards_df: pd.DataFrame) -> pd.DataFrame:
    df = awards_df.copy()

    # Standardize column names across seasons
    df.columns = [c.strip() for c in df.columns]

    # Basketball Reference uses 'Player' for player awards, 'Coach' for COTY
    if "Player" in df.columns:
        df["player_name"] = df["Player"]
    if "Coach" in df.columns:
        df["player_name"] = df["player_name"].fillna(df["Coach"]) if "player_name" in df.columns else df["Coach"]
    if "Name" in df.columns:
        df["player_name"] = df["player_name"].fillna(df["Name"]) if "player_name" in df.columns else df["Name"]
    if "player_name" not in df.columns:
        raise ValueError("Could not find player/coach name column in awards data.")

    # Voting points column may be 'Pts Won' or 'Vote Pts' depending on era
    if "Pts Won" in df.columns:
        df["vote_points"] = df["Pts Won"]
    elif "Vote Pts" in df.columns:
        df["vote_points"] = df["Vote Pts"]
    else:
        raise ValueError("Could not find vote points column in awards data.")

    # First-place votes column may be 'First' or similar
    if "First" in df.columns:
        df["first_place_votes"] = df["First"]
    else:
        df["first_place_votes"] = 0

    if "Share" in df.columns:
        df["vote_share"] = pd.to_numeric(df["Share"], errors="coerce")
    elif "Pts Max" in df.columns:
        pts_max = pd.to_numeric(df["Pts Max"], errors="coerce")
        df["vote_share"] = df["vote_points"] / pts_max.replace(0, pd.NA)
    else:
        df["vote_share"] = pd.NA

    df["player_name_norm"] = df["player_name"].apply(_normalize_player_name)
    df["season"] = df["SEASON"]

    keep_cols = [
        "AWARD_TYPE",
        "season",
        "player_name",
        "player_name_norm",
        "vote_points",
        "vote_share",
        "first_place_votes",
        "Rank",
    ]
    df = df[keep_cols]
    return df


def _join_stats_and_awards(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    awards_df: pd.DataFrame,
) -> pd.DataFrame:
    players = _prepare_players(players_df)
    teams = _prepare_teams(teams_df)
    awards = _prepare_awards(awards_df)

    # Separate COTY (coach-level) from player-level awards
    player_awards = awards[awards["AWARD_TYPE"] != "COTY"].copy()
    coty_awards = awards[awards["AWARD_TYPE"] == "COTY"].copy()

    # --- Player awards: join on (season, player_name_norm) ---
    merged = player_awards.merge(
        players,
        left_on=["season", "player_name_norm"],
        right_on=["season", "player_name_norm"],
        how="left",
        suffixes=("", "_player"),
    )
    merged = merged.merge(
        teams,
        left_on=["TEAM_ID", "season"],
        right_on=["TEAM_ID", "season"],
        how="left",
        suffixes=("", "_team"),
    )
    if "W_PCT_team" in merged.columns:
        merged["TEAM_WIN_PCT"] = merged["W_PCT_team"]
    elif "W_PCT" in merged.columns:
        merged["TEAM_WIN_PCT"] = merged["W_PCT"]

    # --- COTY: uses team-level data directly from awards page ---
    if not coty_awards.empty:
        coty_merged = coty_awards.copy()
        if "Tm" in coty_merged.columns:
            coty_merged["TEAM_ABBREVIATION"] = coty_merged["Tm"]

        # COTY records carry their own W, L, W/L% from Basketball Reference
        if "W/L%" in coty_merged.columns:
            coty_merged["TEAM_WIN_PCT"] = pd.to_numeric(coty_merged["W/L%"], errors="coerce")
        elif "W" in coty_merged.columns and "L" in coty_merged.columns:
            w = pd.to_numeric(coty_merged["W"], errors="coerce").fillna(0)
            l = pd.to_numeric(coty_merged["L"], errors="coerce").fillna(0)
            total = w + l
            coty_merged["TEAM_WIN_PCT"] = (w / total.replace(0, pd.NA)).fillna(0)

        merged = pd.concat([merged, coty_merged], ignore_index=True)

    return merged


def _apply_eligibility_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    # COTY records don't have GP/MIN -- exempt them from player-level filters
    is_coty = filtered["AWARD_TYPE"] == "COTY" if "AWARD_TYPE" in filtered.columns else False
    player_mask = ~is_coty

    if "GP" in filtered.columns:
        invalid = player_mask & (filtered["GP"].fillna(0) <= 0)
        filtered = filtered[~invalid]

    return filtered


def preprocess(start_season: int, end_season: int) -> None:
    ensure_dirs()
    players_df, teams_df, awards_df = _load_raw(start_season, end_season)
    merged = _join_stats_and_awards(players_df, teams_df, awards_df)
    merged = _apply_eligibility_filters(merged)

    output_path = PROCESSED_DIR / f"training_dataset_{start_season}_{end_season}.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved processed training dataset to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw NBA data into a training dataset.")
    parser.add_argument("--start-season", type=int, required=True, help="First season start year (e.g., 2010).")
    parser.add_argument("--end-season", type=int, required=True, help="Last season start year (e.g., 2024).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.start_season, args.end_season)


if __name__ == "__main__":
    main()

