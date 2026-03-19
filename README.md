# NBA Individual Awards Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/dash/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

**Predict MVP, DPOY, ROTY, MIP, 6MOY, and Coach of the Year before the votes are in.**

An end-to-end ML pipeline that ingests 30 years of NBA data, engineers 47 features, trains per-award neural networks, and predicts current-season award winners. Built to demonstrate production-style data engineering, web scraping with anti-bot bypass, and interpretable modeling—perfect for a data science portfolio.

---

## Results

> **Add a screenshot (recommended):** Run `python -m src.dashboard.app`, open http://127.0.0.1:8050/, go to "2025-26 Predictions", take a screenshot, save as `docs/dashboard_screenshot.png`, then add below:  
> `![Dashboard](docs/dashboard_screenshot.png)`

---

## Key Highlights

- **Scraped 30 years of award voting** from Basketball Reference (1996–2025), bypassing Cloudflare protection with TLS fingerprint impersonation and Playwright fallback
- **Engineered 47 features** including advanced stats (NET_RATING, TS_PCT, USG_PCT), per-season ranks, year-over-year deltas, and scoring efficiency
- **Real NBA eligibility rules** — ROTY uses authoritative draft-year data, 6MOY filters to bench players (MIN &lt; 28), MIP excludes established superstars
- **PyTorch MLP regressors** per award with temporal train/validation split (no data leakage)
- **Optional Google Trends integration** — blends 80% model + 20% media hype for narrative-aware predictions

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Data** | nba_api, Basketball Reference, SQLite |
| **ML** | PyTorch, scikit-learn, joblib |
| **Viz** | Plotly Dash, Dash Bootstrap |
| **Scraping** | curl_cffi, Playwright, BeautifulSoup |

---

## Getting Started

**Prerequisites:** Python 3.10+, pip

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Dependencies
pip install -r requirements.txt
python -m playwright install chromium

# 3. Run the pipeline (from project root)
python -m src.data.build_voting_history
python -m src.data.ingest --start-season 1996 --end-season 2025 --include-current
python -m src.data.preprocess --start-season 1996 --end-season 2025
python -m src.db.load
python -m src.models.train_nn
python -m src.models.predict

# 4. Launch dashboard
python -m src.dashboard.app
```

Then open **http://127.0.0.1:8050/** — "Historical Analysis" for past results, "2025-26 Predictions" for live picks.

Optional: `python -m src.models.predict --with-trends` to factor in Google Trends media hype (rate-limited; may fall back to model-only).

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
    │   ├── nba_api_client.py       # base + advanced + bio + roster + coach
    │   ├── ingest.py               # --include-current flag
    │   ├── preprocess.py
    │   ├── feature_engineering.py   # ~47 features + temporal split
    │   └── eligibility.py          # per-award candidate filters
    ├── db/
    ├── models/
    │   ├── train_nn.py             # unified training for all awards
    │   ├── predict.py              # current-season inference + COTY heuristic
    │   ├── evaluate.py             # Spearman, top-1, top-3 accuracy
    │   └── pytorch_base.py         # MLP (256,128,64) + LR scheduler
    └── dashboard/
        └── app.py                  # historical + predictions tabs
```

---

## How It Works

### Award Eligibility Filters

| Award | Eligibility Criteria |
|-------|---------------------|
| **MVP** | GP >= 55% of team schedule, MIN >= 28, PTS >= 15 |
| **DPOY** | GP >= 55% of team schedule, MIN >= 25 |
| **ROTY** | True rookies only (verified via NBA DRAFT_YEAR from PlayerIndex/roster data) |
| **MIP** | 2+ years experience, age <= 28, prior season PTS < 20, positive improvement |
| **6MOY** | Bench players: MIN < 28, GP >= 30 |
| **COTY** | All head coaches, scored by team W% + improvement over prior season |

### Scraping (Cloudflare Bypass)

1. **Local cache** — If `data/raw/awards_YYYY.html` exists, uses it
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
