from __future__ import annotations

"""
Award eligibility rules applied at prediction time.

Uses authoritative NBA data:
- PlayerIndex: DRAFT_YEAR, FROM_YEAR → real rookie/experience status
- CommonTeamRoster: EXP column ("R" = rookie) → confirmation
- Player stats: GP, MIN, PTS → minimum contribution thresholds

Award rules:
- MVP:  GP >= 55% of team schedule, MIN >= 28, PTS >= 15
- DPOY: GP >= 55% of team schedule, MIN >= 25
- ROTY: DRAFT_YEAR == current draft year (true rookies only)
- MIP:  >= 3 years experience, NOT a former top-tier player, meaningful delta
- 6MOY: bench player (not regular starter by minutes/role)
- COTY: coaches only (separate pipeline)
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _load_player_index() -> pd.DataFrame:
    path = RAW_DIR / "player_index.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_player_bio() -> pd.DataFrame:
    path = RAW_DIR / "player_bio.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_rosters() -> pd.DataFrame:
    path = RAW_DIR / "rosters.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge authoritative draft/experience/position data into the player DataFrame.
    Adds: DRAFT_YEAR, FROM_YEAR, NBA_EXPERIENCE, IS_ROOKIE, POSITION.
    """
    out = df.copy()

    pidx = _load_player_index()
    bio = _load_player_bio()
    rosters = _load_rosters()

    # PlayerIndex has PERSON_ID (=PLAYER_ID), FROM_YEAR, DRAFT_YEAR, POSITION
    if not pidx.empty and "PERSON_ID" in pidx.columns:
        pidx_sub = pidx[["PERSON_ID", "FROM_YEAR", "DRAFT_YEAR", "POSITION"]].copy()
        pidx_sub = pidx_sub.rename(columns={"PERSON_ID": "PLAYER_ID"})
        pidx_sub = pidx_sub.drop_duplicates(subset="PLAYER_ID", keep="first")
        out = out.merge(pidx_sub, on="PLAYER_ID", how="left", suffixes=("", "_pidx"))

    # Bio stats has DRAFT_YEAR too (fallback)
    if not bio.empty and "DRAFT_YEAR" in bio.columns and "PLAYER_ID" in bio.columns:
        bio_sub = bio[["PLAYER_ID", "DRAFT_YEAR"]].copy()
        bio_sub = bio_sub.rename(columns={"DRAFT_YEAR": "DRAFT_YEAR_BIO"})
        bio_sub = bio_sub.drop_duplicates(subset="PLAYER_ID", keep="first")
        out = out.merge(bio_sub, on="PLAYER_ID", how="left", suffixes=("", "_bio"))

        if "DRAFT_YEAR" not in out.columns:
            out["DRAFT_YEAR"] = out.get("DRAFT_YEAR_BIO")
        else:
            out["DRAFT_YEAR"] = out["DRAFT_YEAR"].fillna(out.get("DRAFT_YEAR_BIO"))

    # Roster has EXP ("R" for rookies)
    if not rosters.empty and "PLAYER_ID" in rosters.columns and "EXP" in rosters.columns:
        roster_sub = rosters[["PLAYER_ID", "EXP"]].copy()
        roster_sub = roster_sub.rename(columns={"EXP": "ROSTER_EXP"})
        roster_sub = roster_sub.drop_duplicates(subset="PLAYER_ID", keep="first")
        out = out.merge(roster_sub, on="PLAYER_ID", how="left", suffixes=("", "_roster"))

    # Compute authoritative experience and rookie flag
    out["IS_ROOKIE"] = False

    if "ROSTER_EXP" in out.columns:
        out["IS_ROOKIE"] = out["IS_ROOKIE"] | (out["ROSTER_EXP"].astype(str).str.strip() == "R")

    if "DRAFT_YEAR" in out.columns:
        # Current season 2025-26: rookies are drafted in 2025
        # Determine current draft year from the season column
        if "season" in out.columns:
            latest_season = out["season"].max()
            current_draft_year = int(latest_season.split("-")[0]) + 1  # "2025-26" -> 2026? No, draft 2025
            # NBA draft year for 2025-26 season is 2025
            current_draft_year = int(latest_season.split("-")[0])
        else:
            current_draft_year = 2025

        draft_yr = pd.to_numeric(out["DRAFT_YEAR"], errors="coerce")
        out["IS_ROOKIE"] = out["IS_ROOKIE"] | (draft_yr == current_draft_year)

    if "FROM_YEAR" in out.columns and "season" in out.columns:
        season_start = out["season"].apply(lambda s: int(s.split("-")[0]))
        from_year = pd.to_numeric(out["FROM_YEAR"], errors="coerce")
        out["NBA_EXPERIENCE"] = (season_start - from_year).clip(lower=0)
    elif "ROSTER_EXP" in out.columns:
        out["NBA_EXPERIENCE"] = pd.to_numeric(
            out["ROSTER_EXP"].replace("R", "0"), errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["NBA_EXPERIENCE"] = 99

    return out


def _estimate_team_games(df: pd.DataFrame) -> float:
    """Estimate how many games teams have played this season."""
    if "GP" in df.columns:
        return df.groupby("TEAM_ABBREVIATION")["GP"].max().median()
    return 82.0


def filter_mvp_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """MVP: significant contributors on winning teams."""
    mask = pd.Series(True, index=df.index)
    team_gp = _estimate_team_games(df)
    gp_threshold = max(35, team_gp * 0.55)

    if "GP" in df.columns:
        mask &= df["GP"] >= gp_threshold
    if "MIN" in df.columns:
        mask &= df["MIN"] >= 28.0
    if "PTS" in df.columns:
        mask &= df["PTS"] >= 15.0
    return df[mask].copy()


def filter_dpoy_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """DPOY: must play significant minutes and games."""
    mask = pd.Series(True, index=df.index)
    team_gp = _estimate_team_games(df)
    gp_threshold = max(35, team_gp * 0.55)

    if "GP" in df.columns:
        mask &= df["GP"] >= gp_threshold
    if "MIN" in df.columns:
        mask &= df["MIN"] >= 25.0
    return df[mask].copy()


def filter_roty_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """ROTY: true NBA rookies only (drafted in the current year)."""
    mask = pd.Series(True, index=df.index)

    if "IS_ROOKIE" in df.columns:
        mask &= df["IS_ROOKIE"] == True
    elif "NBA_EXPERIENCE" in df.columns:
        mask &= df["NBA_EXPERIENCE"] == 0
    else:
        mask &= False

    if "GP" in df.columns:
        mask &= df["GP"] >= 20
    if "MIN" in df.columns:
        mask &= df["MIN"] >= 15.0
    return df[mask].copy()


def filter_mip_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    MIP: Must have meaningful prior history, NOT already a superstar.
    Historical MIP winners profile:
    - 2-7 years experience typically
    - Previous season PPG usually < 20 (breakout, not continuation)
    - Age typically 22-28
    - Must show positive improvement
    """
    mask = pd.Series(True, index=df.index)

    if "NBA_EXPERIENCE" in df.columns:
        mask &= df["NBA_EXPERIENCE"] >= 2
    if "IS_ROOKIE" in df.columns:
        mask &= df["IS_ROOKIE"] == False

    if "GP" in df.columns:
        mask &= df["GP"] >= 30
    if "MIN" in df.columns:
        mask &= df["MIN"] >= 20.0

    # Exclude established superstars: if current PTS > 25 AND pts_delta < 5,
    # the player was already elite. Real MIP winners break out from mid-tier.
    if "PTS" in df.columns and "pts_delta" in df.columns:
        already_star = (df["PTS"] - df["pts_delta"]) > 20
        mask &= ~already_star

    if "pts_delta" in df.columns:
        mask &= df["pts_delta"] > 1.0

    if "AGE" in df.columns:
        mask &= df["AGE"] <= 28

    return df[mask].copy()


def filter_6moy_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    6MOY: bench players -- NOT regular starters.
    Historical 6MOY winners typically:
    - Play 24-28 minutes (significant role but off the bench)
    - Score 12-20 PPG
    - Not the team's leading scorer
    """
    mask = pd.Series(True, index=df.index)

    if "GP" in df.columns:
        mask &= df["GP"] >= 30

    # Real 6th men play < 28 min/game; starters average 30+
    if "MIN" in df.columns:
        mask &= df["MIN"] < 28.0
        mask &= df["MIN"] >= 15.0

    if "PTS" in df.columns:
        mask &= df["PTS"] >= 8.0

    return df[mask].copy()


AWARD_FILTERS = {
    "MVP": filter_mvp_candidates,
    "DPOY": filter_dpoy_candidates,
    "ROTY": filter_roty_candidates,
    "MIP": filter_mip_candidates,
    "6MOY": filter_6moy_candidates,
}
