from __future__ import annotations

"""
Thin wrapper around `nba_api` for fetching player and team stats.

Fetches both Base and Advanced stats per season and merges them
into a single player DataFrame. Includes retry logic for API timeouts.
"""

import time
from typing import Dict, List

import pandas as pd

try:
    from nba_api.stats.endpoints import (
        leaguedashplayerstats,
        leaguedashteamstats,
        leaguedashplayerbiostats,
        playerindex,
        commonteamroster,
    )
    from nba_api.stats.static import teams as nba_teams
except ImportError as exc:
    raise ImportError(
        "nba_api is required for src.data.nba_api_client. "
        "Install it with `pip install nba_api`."
    ) from exc


DEFAULT_SLEEP_SECONDS = 2.0
MAX_RETRIES = 3
TIMEOUT = 60


def _season_str(year_start: int) -> str:
    year_end_two = (year_start + 1) % 100
    return f"{year_start}-{year_end_two:02d}"


def _retry(fn, retries=MAX_RETRIES, backoff=5):
    """Retry a callable up to `retries` times with exponential backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"    Retry {attempt+1}/{retries} after error: {exc}. Waiting {wait}s...")
            time.sleep(wait)


def _fetch_player_stats(
    season: str,
    per_mode: str = "PerGame",
    measure_type: str = "Base",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    time.sleep(sleep_seconds)

    def _call():
        ep = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure_type,
            timeout=TIMEOUT,
        )
        return ep.get_data_frames()[0]

    return _retry(_call)


def fetch_player_season_stats(
    season_start_year: int,
    per_mode: str = "PerGame",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """
    Fetch Base + Advanced player stats for one season and merge them.
    """
    season = _season_str(season_start_year)

    base = _fetch_player_stats(season, per_mode, "Base", sleep_seconds)

    try:
        adv = _fetch_player_stats(season, per_mode, "Advanced", sleep_seconds)
        shared_keys = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
        adv_only_cols = [c for c in adv.columns if c not in base.columns]
        merge_cols = [k for k in shared_keys if k in adv.columns and k in base.columns]
        if merge_cols and adv_only_cols:
            base = base.merge(adv[merge_cols + adv_only_cols], on=merge_cols, how="left")
    except Exception as exc:
        print(f"  Warning: could not fetch Advanced stats for {season}: {exc}")

    base["SEASON"] = season
    return base


def fetch_team_season_stats(
    season_start_year: int,
    per_mode: str = "PerGame",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    season = _season_str(season_start_year)
    time.sleep(sleep_seconds)

    def _call():
        ep = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense="Base",
            timeout=TIMEOUT,
        )
        return ep.get_data_frames()[0]

    df = _retry(_call)
    df["SEASON"] = season
    return df


def fetch_player_bio_stats(
    season_start_year: int,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Fetch player bio stats including DRAFT_YEAR, DRAFT_ROUND, DRAFT_NUMBER."""
    season = _season_str(season_start_year)
    time.sleep(sleep_seconds)

    def _call():
        ep = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season,
            per_mode_simple="PerGame",
            timeout=TIMEOUT,
        )
        return ep.get_data_frames()[0]

    df = _retry(_call)
    df["SEASON"] = season
    return df


def fetch_player_index(
    season_start_year: int,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> pd.DataFrame:
    """Fetch player index with FROM_YEAR, TO_YEAR, DRAFT_YEAR, POSITION."""
    season = _season_str(season_start_year)
    time.sleep(sleep_seconds)

    def _call():
        ep = playerindex.PlayerIndex(season=season, timeout=TIMEOUT)
        return ep.get_data_frames()[0]

    df = _retry(_call)
    df["SEASON"] = season
    return df


def fetch_all_team_rosters_and_coaches(
    season_start_year: int,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch roster (with EXP column) and coach info for all 30 teams.
    Returns (roster_df, coaches_df).
    """
    season = _season_str(season_start_year)
    all_teams_list = nba_teams.get_teams()
    roster_frames: List[pd.DataFrame] = []
    coach_frames: List[pd.DataFrame] = []

    for team in all_teams_list:
        team_id = team["id"]
        time.sleep(sleep_seconds)
        try:
            def _call(tid=team_id):
                ep = commonteamroster.CommonTeamRoster(
                    team_id=tid, season=season, timeout=TIMEOUT
                )
                return ep.get_data_frames()

            dfs = _retry(_call)
            roster_df = dfs[0]
            coaches_df = dfs[1]

            roster_df["TEAM_ID"] = team_id
            roster_df["TEAM_ABBREVIATION"] = team["abbreviation"]
            coaches_df["TEAM_ID"] = team_id
            coaches_df["TEAM_ABBREVIATION"] = team["abbreviation"]

            roster_frames.append(roster_df)
            coach_frames.append(coaches_df)
        except Exception as exc:
            print(f"  Warning: roster fetch failed for {team['abbreviation']}: {exc}")

    rosters = pd.concat(roster_frames, ignore_index=True) if roster_frames else pd.DataFrame()
    coaches = pd.concat(coach_frames, ignore_index=True) if coach_frames else pd.DataFrame()

    rosters["SEASON"] = season
    coaches["SEASON"] = season
    return rosters, coaches


def fetch_seasons_player_team_stats(
    start_season: int,
    end_season: int,
    per_mode: str = "PerGame",
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, pd.DataFrame]:
    """Fetch player (base+advanced) and team stats for a range of seasons."""
    player_frames: List[pd.DataFrame] = []
    team_frames: List[pd.DataFrame] = []

    for year in range(start_season, end_season + 1):
        print(f"  Fetching stats for {_season_str(year)}...", flush=True)
        player_frames.append(
            fetch_player_season_stats(year, per_mode=per_mode, sleep_seconds=sleep_seconds)
        )
        team_frames.append(
            fetch_team_season_stats(year, per_mode=per_mode, sleep_seconds=sleep_seconds)
        )

    players_all = pd.concat(player_frames, ignore_index=True) if player_frames else pd.DataFrame()
    teams_all = pd.concat(team_frames, ignore_index=True) if team_frames else pd.DataFrame()

    return {"players": players_all, "teams": teams_all}
