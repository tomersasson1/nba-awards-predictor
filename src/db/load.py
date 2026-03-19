from __future__ import annotations

"""
Load processed data into a SQLite database according to `schema.sql`.

For simplicity, player and team IDs are derived from the processed CSV
using pandas categorical codes. In a more advanced version, you may want
to maintain stable IDs across runs.
"""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "nba_awards.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _init_db(conn: sqlite3.Connection) -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        conn.executescript(f.read())


def load_from_processed(training_csv_path: Path) -> None:
    df = pd.read_csv(training_csv_path)

    conn = sqlite3.connect(DB_PATH)
    try:
        _init_db(conn)

        # Seasons table
        seasons = (
            df[["season"]]
            .drop_duplicates()
            .assign(
                year_start=lambda d: d["season"].str.slice(0, 4).astype(int),
                year_end=lambda d: d["season"].str.slice(0, 4).astype(int) + 1,
            )
        )
        seasons["season_id"] = seasons.reset_index().index + 1

        # Players table
        players = df[["player_name"]].drop_duplicates().rename(columns={"player_name": "name"})
        players["player_id"] = players.reset_index().index + 1

        # Teams table
        team_cols = []
        if "TEAM_ID" in df.columns:
            team_cols.append("TEAM_ID")
        if "TEAM_NAME" in df.columns:
            team_cols.append("TEAM_NAME")
        if "TEAM_ABBREVIATION" in df.columns:
            team_cols.append("TEAM_ABBREVIATION")

        teams = df[team_cols].drop_duplicates()
        if "TEAM_ID" in teams.columns:
            teams = teams.rename(
                columns={
                    "TEAM_ID": "team_id",
                    "TEAM_NAME": "name",
                    "TEAM_ABBREVIATION": "abbreviation",
                }
            )
        else:
            teams["team_id"] = teams.reset_index().index + 1

        # Merge IDs back into main dataframe
        df = df.merge(players[["player_id", "name"]], left_on="player_name", right_on="name", how="left")
        df = df.merge(seasons[["season_id", "season"]], on="season", how="left")
        if "team_id" in teams.columns and "TEAM_ID" in df.columns:
            df = df.merge(teams[["team_id"]].join(teams.set_index("team_id")), left_on="TEAM_ID", right_index=True, how="left")

        # Player season stats
        pstats_cols = {
            "PTS": "ppg",
            "REB": "rpg",
            "AST": "apg",
            "STL": "stl",
            "BLK": "blk",
            "GP": "games_played",
            "MIN": "minutes",
            "PLUS_MINUS": "plus_minus",
            "TEAM_WIN_PCT": "team_win_pct",
        }
        pstats = df[
            ["player_id", "season_id", "TEAM_ID"]
            + [c for c in pstats_cols.keys() if c in df.columns]
        ].copy()
        pstats = pstats.rename(columns=pstats_cols)
        pstats = pstats.rename(columns={"TEAM_ID": "team_id"})

        # Team season stats (aggregate from team-level columns)
        w_col = "W_team" if "W_team" in df.columns else ("W" if "W" in df.columns else None)
        l_col = "L_team" if "L_team" in df.columns else ("L" if "L" in df.columns else None)
        wp_col = "TEAM_WIN_PCT" if "TEAM_WIN_PCT" in df.columns else (
            "W_PCT_team" if "W_PCT_team" in df.columns else (
                "W_PCT" if "W_PCT" in df.columns else None
            )
        )

        agg_dict = {}
        if w_col:
            agg_dict["wins"] = (w_col, "max")
        if l_col:
            agg_dict["losses"] = (l_col, "max")
        if wp_col:
            agg_dict["team_win_pct"] = (wp_col, "max")

        if agg_dict:
            tstats = df.groupby(["TEAM_ID", "season"], as_index=False).agg(**agg_dict)
        else:
            tstats = df[["TEAM_ID", "season"]].drop_duplicates()

        tstats = tstats.merge(seasons[["season_id", "season"]], on="season", how="left")
        if "TEAM_ID" in tstats.columns:
            tstats = tstats.rename(columns={"TEAM_ID": "team_id"})

        # Award voting
        avoting_cols = ["AWARD_TYPE", "season_id", "player_id", "vote_points",
                        "vote_share", "first_place_votes", "Rank"]
        for col in avoting_cols:
            if col not in df.columns:
                df[col] = pd.NA
        avoting = df[avoting_cols].copy()
        avoting = avoting.rename(
            columns={
                "AWARD_TYPE": "award_type",
                "Rank": "rank",
            }
        )

        # Write to DB
        seasons[["season_id", "season", "year_start", "year_end"]].to_sql(
            "seasons", conn, if_exists="replace", index=False
        )
        players[["player_id", "name"]].to_sql("players", conn, if_exists="replace", index=False)
        teams[["team_id", "name", "abbreviation"]].to_sql("teams", conn, if_exists="replace", index=False)
        pstats.to_sql("player_season_stats", conn, if_exists="replace", index=False)
        tstats_cols = [c for c in ["team_id", "season_id", "wins", "losses", "team_win_pct"] if c in tstats.columns]
        tstats[tstats_cols].to_sql(
            "team_season_stats", conn, if_exists="replace", index=False
        )
        avoting.to_sql("award_voting", conn, if_exists="replace", index=False)

        conn.commit()
        print(f"Loaded processed data into SQLite at {DB_PATH}")
    finally:
        conn.close()


def main() -> None:
    # For now, use the first matching training_dataset file if more than one exists
    candidates = sorted(PROCESSED_DIR.glob("training_dataset_*.csv"))
    if not candidates:
        raise SystemExit("No processed training_dataset_*.csv found in data/processed/. Run preprocess first.")
    load_from_processed(candidates[0])


if __name__ == "__main__":
    main()

