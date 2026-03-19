from __future__ import annotations

"""
Entry point for data ingestion.

Fetches:
- Player stats (base + advanced) and team stats from nba_api
- Award voting data from Basketball Reference
- Optionally, current in-progress season stats for prediction
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .nba_api_client import (
    fetch_seasons_player_team_stats,
    fetch_player_season_stats,
    fetch_team_season_stats,
    fetch_player_bio_stats,
    fetch_player_index,
    fetch_all_team_rosters_and_coaches,
)
from .basketball_reference_scraper import fetch_award_voting_for_seasons


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _print(msg: str) -> None:
    print(msg, flush=True)


def ingest(start_season: int, end_season: int, include_current: bool = False) -> None:
    """
    Ingest NBA data.

    Season year convention in this project: end_season=2025 refers to the
    season ending in 2025 (i.e. "2024-25"). nba_api uses the start year,
    so we translate: API year = end_season - 1 for "2024-25".
    """
    ensure_dirs()

    api_start = start_season
    api_end = end_season - 1  # 2025 -> API year 2024 -> "2024-25"

    _print(f"Fetching player and team stats from nba_api ({api_start}-{api_end})...")
    stats = fetch_seasons_player_team_stats(api_start, api_end)
    players_df: pd.DataFrame = stats["players"]
    teams_df: pd.DataFrame = stats["teams"]

    players_path = RAW_DIR / f"players_{start_season}_{end_season}.csv"
    teams_path = RAW_DIR / f"teams_{start_season}_{end_season}.csv"
    awards_path = RAW_DIR / f"awards_{start_season}_{end_season}.csv"

    _print("Fetching award voting data from Basketball Reference...")
    try:
        awards_df = fetch_award_voting_for_seasons(start_season, end_season)
    except Exception as exc:
        if awards_path.exists():
            _print(f"Warning: scraping failed ({exc}). Using existing {awards_path}.")
            awards_df = pd.read_csv(awards_path)
        else:
            raise RuntimeError(
                "Failed to scrape award voting data and no local CSV found.\n"
                "Save awards HTML pages under 'data/raw' or assemble the CSV manually."
            ) from exc

    players_df.to_csv(players_path, index=False)
    teams_df.to_csv(teams_path, index=False)
    awards_df.to_csv(awards_path, index=False)

    _print(f"Saved player stats to {players_path}")
    _print(f"Saved team stats to   {teams_path}")
    _print(f"Saved awards data to  {awards_path}")

    if include_current:
        cur_api_year = end_season  # 2025 -> API year 2025 -> "2025-26"
        _print(f"\nFetching current season {cur_api_year}-{(cur_api_year+1)%100:02d} stats...")
        cur_players = fetch_player_season_stats(cur_api_year)
        cur_teams = fetch_team_season_stats(cur_api_year)

        cur_players_path = RAW_DIR / "current_season_players.csv"
        cur_teams_path = RAW_DIR / "current_season_teams.csv"
        cur_players.to_csv(cur_players_path, index=False)
        cur_teams.to_csv(cur_teams_path, index=False)
        _print(f"Saved current-season player stats to {cur_players_path}")
        _print(f"Saved current-season team stats to   {cur_teams_path}")

        _print("\nFetching player bio stats (DRAFT_YEAR, etc.)...")
        try:
            bio = fetch_player_bio_stats(cur_api_year)
            bio.to_csv(RAW_DIR / "player_bio.csv", index=False)
            _print(f"  Saved {len(bio)} player bio records.")
        except Exception as exc:
            _print(f"  Warning: bio stats fetch failed: {exc}")

        _print("Fetching player index (FROM_YEAR, POSITION, etc.)...")
        try:
            pidx = fetch_player_index(cur_api_year)
            pidx.to_csv(RAW_DIR / "player_index.csv", index=False)
            _print(f"  Saved {len(pidx)} player index records.")
        except Exception as exc:
            _print(f"  Warning: player index fetch failed: {exc}")

        _print("Fetching team rosters + coaches (30 teams)...")
        try:
            rosters, coaches = fetch_all_team_rosters_and_coaches(cur_api_year)
            rosters.to_csv(RAW_DIR / "rosters.csv", index=False)
            coaches.to_csv(RAW_DIR / "coaches.csv", index=False)
            _print(f"  Saved {len(rosters)} roster entries, {len(coaches)} coach entries.")
        except Exception as exc:
            _print(f"  Warning: roster/coach fetch failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest NBA stats and award voting data.")
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    parser.add_argument(
        "--include-current", action="store_true",
        help="Also fetch stats for the season after end-season (the in-progress season).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(args.start_season, args.end_season, include_current=args.include_current)


if __name__ == "__main__":
    main()
