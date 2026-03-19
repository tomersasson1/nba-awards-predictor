from __future__ import annotations

"""
Script to scrape NBA award voting history from Basketball Reference
for seasons 1990–2024 and export to a single CSV file.

It uses the helper utilities in `basketball_reference_scraper.py`
to fetch and parse the awards pages.
"""

import time
from pathlib import Path

import pandas as pd

from .basketball_reference_scraper import (
    ScrapingError,
    fetch_award_voting_for_season,
)


def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names across seasons and awards.

    - Strip whitespace from column names.
    - Create a unified `VOTE_POINTS` column from either `Pts Won` or `Vote Pts` if present.
    - Ensure at least these columns exist (some may be completely NaN for certain seasons):
        ['AWARD_TYPE', 'SEASON', 'Player', 'Rank', 'First', 'VOTE_POINTS']
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Create unified vote-points column
    vote_cols = [c for c in df.columns if c.lower() in {"pts won", "vote pts", "points"}]
    if vote_cols:
        main_vote_col = vote_cols[0]
        df["VOTE_POINTS"] = df[main_vote_col]
    else:
        df["VOTE_POINTS"] = pd.NA

    # Ensure required columns exist; fill missing with NA
    required_cols = ["AWARD_TYPE", "SEASON", "Player", "Rank", "First", "VOTE_POINTS"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def build_voting_history(
    start_season: int = 1990,
    end_season: int = 2024,
    sleep_seconds: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch and combine award voting tables for a range of seasons.

    A `time.sleep(sleep_seconds)` is performed between requests to avoid
    hitting Basketball Reference too aggressively.
    """
    frames: list[pd.DataFrame] = []
    failed_seasons: list[int] = []
    for year in range(start_season, end_season + 1):
        print(f"Fetching awards voting for season {year}...")
        try:
            season_df = fetch_award_voting_for_season(year)
        except ScrapingError as exc:
            print(f"  !! Skipping season {year} due to scraping error: {exc}")
            failed_seasons.append(year)
            continue

        season_df = _standardize_schema(season_df)
        frames.append(season_df)

        # Respectful pause between network requests
        time.sleep(sleep_seconds)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Sort for convenience
    combined = combined.sort_values(by=["SEASON", "AWARD_TYPE", "Rank", "Player"]).reset_index(drop=True)
    return combined


def main() -> None:
    df = build_voting_history(start_season=1997, end_season=2025, sleep_seconds=3.0)

    # Save under project root (same directory as README) as requested
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "nba_voting_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved combined voting history to {csv_path}")


if __name__ == "__main__":
    main()

