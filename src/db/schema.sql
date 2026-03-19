CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    birth_year INTEGER
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seasons (
    season_id INTEGER PRIMARY KEY,
    season TEXT NOT NULL UNIQUE,
    year_start INTEGER NOT NULL,
    year_end INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    team_id INTEGER,
    ppg REAL,
    rpg REAL,
    apg REAL,
    stl REAL,
    blk REAL,
    games_played INTEGER,
    minutes REAL,
    plus_minus REAL,
    team_win_pct REAL,
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (season_id) REFERENCES seasons (season_id),
    FOREIGN KEY (team_id) REFERENCES teams (team_id)
);

CREATE TABLE IF NOT EXISTS team_season_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    wins INTEGER,
    losses INTEGER,
    conf_rank INTEGER,
    team_win_pct REAL,
    FOREIGN KEY (team_id) REFERENCES teams (team_id),
    FOREIGN KEY (season_id) REFERENCES seasons (season_id)
);

CREATE TABLE IF NOT EXISTS award_voting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    award_type TEXT NOT NULL,
    season_id INTEGER NOT NULL,
    player_id INTEGER,
    vote_points REAL,
    vote_share REAL,
    first_place_votes INTEGER,
    rank INTEGER,
    FOREIGN KEY (player_id) REFERENCES players (player_id),
    FOREIGN KEY (season_id) REFERENCES seasons (season_id)
);

