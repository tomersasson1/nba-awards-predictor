# NBA Individual Awards Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/dash/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

**Predict MVP, DPOY, ROTY, MIP, 6MOY, and Coach of the Year before the votes are in.**

I built this project to predict NBA individual award voting using 30 years of player and team data. The pipeline scrapes award history from Basketball Reference, ingests stats from the NBA API, engineers 47 features, and trains per-award neural networks with PyTorch. I added award-specific eligibility filters so predictions only include realistic candidates, and built a Plotly Dash dashboard to explore historical results and live picks.

---

## Results

> *Add a screenshot:* Run `python -m src.dashboard.app`, open http://127.0.0.1:8050/, go to "2025-26 Predictions", take a screenshot, save as `docs/dashboard_screenshot.png`, then add: `![Dashboard](docs/dashboard_screenshot.png)`

---

## What I Built

- **30 years of award voting data** — Scraped from Basketball Reference (1996–2025). The site uses Cloudflare anti-bot protection, so I implemented a tiered fallback: local cache, then curl_cffi (TLS fingerprint impersonation), then Playwright headless browser.
- **47 engineered features** — Advanced stats (NET_RATING, TS_PCT, USG_PCT), per-season ranks, year-over-year deltas, and scoring efficiency. The model uses a temporal split to avoid data leakage.
- **Real NBA eligibility rules** — ROTY uses authoritative draft-year data from the NBA API, 6MOY filters to bench players (MIN < 28), MIP excludes established superstars, and COTY scores coaches by team win% plus improvement.
- **Optional Google Trends integration** — I blend 80% model prediction with 20% media hype based on search interest, so the narrative factor is reflected without overpowering the stats.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | nba_api, Basketball Reference, SQLite |
| **ML** | PyTorch, scikit-learn, joblib |
| **Viz** | Plotly Dash, Dash Bootstrap |
| **Scraping** | curl_cffi, Playwright, BeautifulSoup |

---

## Running It Locally

Python 3.10+ and pip required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m playwright install chromium

# Run the pipeline (from project root)
python -m src.data.build_voting_history
python -m src.data.ingest --start-season 1996 --end-season 2025 --include-current
python -m src.data.preprocess --start-season 1996 --end-season 2025
python -m src.db.load
python -m src.models.train_nn
python -m src.models.predict

python -m src.dashboard.app
```

Then open **http://127.0.0.1:8050/** — "Historical Analysis" for past results, "2025-26 Predictions" for live picks.

Add `--with-trends` to the predict step to factor in Google Trends (rate-limited; may fall back to model-only).

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── nba_voting_history.csv
├── data/
│   ├── raw/              # cached HTML, raw CSVs, player bios, rosters, coaches
│   ├── processed/        # training-ready datasets + predictions
│   └── nba_awards.db
├── models/               # trained model weights + scalers
└── src/
    ├── data/
    │   ├── basketball_reference_scraper.py
    │   ├── build_voting_history.py
    │   ├── nba_api_client.py
    │   ├── ingest.py
    │   ├── preprocess.py
    │   ├── feature_engineering.py
    │   └── eligibility.py
    ├── db/
    ├── models/
    │   ├── train_nn.py
    │   ├── predict.py
    │   ├── evaluate.py
    │   └── pytorch_base.py
    └── dashboard/
        └── app.py
```

---

## Implementation Details

### Award Eligibility Filters

| Award | Criteria |
|-------|----------|
| **MVP** | GP >= 55% of team schedule, MIN >= 28, PTS >= 15 |
| **DPOY** | GP >= 55% of team schedule, MIN >= 25 |
| **ROTY** | True rookies only (verified via NBA DRAFT_YEAR from PlayerIndex/roster data) |
| **MIP** | 2+ years experience, age <= 28, prior season PTS < 20, positive improvement |
| **6MOY** | Bench players: MIN < 28, GP >= 30 |
| **COTY** | All head coaches, scored by team W% + improvement over prior season |

### Scraping (Cloudflare Bypass)

1. **Local cache** — Uses `data/raw/awards_YYYY.html` if present
2. **curl_cffi** — Impersonates Chrome TLS fingerprint
3. **Playwright** — Falls back to headless Chromium if blocked

### Feature Engineering (~47 features)

| Tier | Features | Purpose |
|------|----------|---------|
| **Advanced Stats** | OFF/DEF/NET_RATING, USG_PCT, TS_PCT, EFG_PCT, PIE | Player impact beyond box scores |
| **Context** | team_conf_seed, top3_seed, eligible_65, is_starter | Team success & eligibility |
| **Ranks** | ppg_rank, rpg_rank, apg_rank | Era-adjusted dominance |
| **YoY Deltas** | pts_delta, reb_delta, ast_delta, ts_pct_delta | Improvement (critical for MIP) |

### Temporal Split

- **Train:** 1996-97 through 2023-24  
- **Validate:** 2024-25  
- **Predict:** 2025-26 (in progress)
