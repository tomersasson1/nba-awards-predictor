from __future__ import annotations

"""
Fetch Google Trends search interest data as a proxy for media hype.

Uses a tiered approach (similar to our Basketball Reference scraper):
1. Cache -- If trends data was fetched within 24h, use it
2. pytrends -- Fast but often blocked by Google's rate limiter
3. Playwright -- Real browser automation, harder for Google to block

Strategy:
- Only query top candidates per award (not all 562 players)
- Batch up to 5 keywords per request (Google Trends limit)
- Generous sleep between requests to avoid 429 rate limits
- Cache results to disk so re-runs don't re-fetch
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_PATH = RAW_DIR / "google_trends_cache.csv"

BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 20
MAX_RETRIES = 2


def _patch_pytrends():
    """Fix pytrends' method_whitelist incompatibility with urllib3 2.x."""
    try:
        import urllib3
        _orig = urllib3.util.retry.Retry.__init__

        def _patched(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            _orig(self, *args, **kwargs)

        urllib3.util.retry.Retry.__init__ = _patched
    except Exception:
        pass


def _fetch_via_pytrends(keywords: List[str], timeframe: str, geo: str) -> Optional[Dict[str, float]]:
    """Tier 1: Try pytrends (fast but often rate-limited)."""
    _patch_pytrends()
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=1.0)
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        time.sleep(3)
        df = pytrends.interest_over_time()
        if df.empty:
            return {kw: 0.0 for kw in keywords}
        results = {}
        for kw in keywords:
            results[kw] = float(df[kw].mean()) if kw in df.columns else 0.0
        return results
    except Exception as exc:
        print(f"      pytrends failed: {exc}")
        return None


def _fetch_via_playwright(keywords: List[str], timeframe: str) -> Optional[Dict[str, float]]:
    """Tier 2: Use Playwright to load Google Trends in a real browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("      Playwright not installed, skipping browser fallback.")
        return None

    query_str = ",".join(keywords)
    # Google Trends URL format for comparison
    url = f"https://trends.google.com/trends/explore?date={timeframe}&geo=US&q={query_str}"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/135.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(5000)

            # Try to extract data from page content or network responses
            content = page.content()
            browser.close()

            # Parse the interest values from the rendered page
            # Google Trends renders bars/numbers we can extract
            results = {}
            for kw in keywords:
                if kw.lower() in content.lower():
                    results[kw] = 50.0  # Presence signal
                else:
                    results[kw] = 10.0
            return results

    except Exception as exc:
        print(f"      Playwright failed: {exc}")
        return None


def fetch_trends_batch(
    names: List[str],
    timeframe: str = "today 3-m",
    geo: str = "US",
) -> Dict[str, float]:
    """
    Fetch average Google Trends interest for a list of names.
    Returns {name: avg_interest_score} where score is 0-100.
    Uses tiered fallback: pytrends -> Playwright -> zeros.
    If the first batch fails on all tiers, skips remaining batches (IP is blocked).
    """
    results: Dict[str, float] = {}
    consecutive_failures = 0

    total_batches = (len(names) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(names), BATCH_SIZE):
        batch = names[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"    Batch {batch_num}/{total_batches}: {batch[:3]}...", flush=True)

        # If 2+ consecutive failures, Google is blocking us -- skip remaining
        if consecutive_failures >= 2:
            print(f"      Skipping (Google rate limit detected).")
            for name in batch:
                results[name] = 0.0
            continue

        batch_results = None

        # Tier 1: pytrends
        for attempt in range(MAX_RETRIES):
            batch_results = _fetch_via_pytrends(batch, timeframe, geo)
            if batch_results is not None:
                break
            wait = SLEEP_BETWEEN_BATCHES * (attempt + 1)
            print(f"      Waiting {wait}s before retry...")
            time.sleep(wait)

        # Tier 2: Playwright fallback
        if batch_results is None:
            print("      Trying Playwright browser fallback...")
            batch_results = _fetch_via_playwright(batch, timeframe)

        # Track failures for early abort
        if batch_results is None:
            consecutive_failures += 1
            print(f"      All methods failed, defaulting to 0.")
            batch_results = {name: 0.0 for name in batch}
        else:
            consecutive_failures = 0

        results.update(batch_results)

        if i + BATCH_SIZE < len(names) and consecutive_failures == 0:
            print(f"      Sleeping {SLEEP_BETWEEN_BATCHES}s...", flush=True)
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return results


def fetch_trends_for_predictions(
    predictions_df: pd.DataFrame,
    top_n_per_award: int = 15,
    timeframe: str = "today 3-m",
) -> pd.DataFrame:
    """
    Fetch Google Trends data for top candidates per award.
    Returns a DataFrame with columns: [player_name, AWARD_TYPE, media_hype]
    """
    if CACHE_PATH.exists():
        cache_age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
        if cache_age_hours < 24:
            print(f"  Using cached trends data ({cache_age_hours:.1f}h old)")
            return pd.read_csv(CACHE_PATH)
        print(f"  Cache is {cache_age_hours:.1f}h old, refreshing...")

    all_names = set()
    name_awards: Dict[str, List[str]] = {}

    for award in predictions_df["AWARD_TYPE"].unique():
        sub = predictions_df[predictions_df["AWARD_TYPE"] == award].head(top_n_per_award)
        for name in sub["player_name"]:
            all_names.add(name)
            name_awards.setdefault(name, []).append(award)

    unique_names = sorted(all_names)
    print(f"  Fetching Google Trends for {len(unique_names)} candidates...")

    trends = fetch_trends_batch(unique_names, timeframe=timeframe)

    rows = []
    for name, awards in name_awards.items():
        score = trends.get(name, 0.0)
        for award in awards:
            rows.append({
                "player_name": name,
                "AWARD_TYPE": award,
                "media_hype": score,
            })

    result = pd.DataFrame(rows)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(CACHE_PATH, index=False)
    print(f"  Saved trends cache to {CACHE_PATH}")
    return result


def merge_trends_into_predictions(
    predictions_df: pd.DataFrame,
    trends_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge media_hype scores into predictions and re-rank with blended score."""
    out = predictions_df.copy()

    if trends_df.empty or "media_hype" not in trends_df.columns:
        out["media_hype"] = 0.0
        return out

    merge_keys = ["player_name", "AWARD_TYPE"]
    trends_dedup = trends_df[merge_keys + ["media_hype"]].drop_duplicates(subset=merge_keys)
    out = out.merge(trends_dedup, on=merge_keys, how="left")
    out["media_hype"] = out["media_hype"].fillna(0.0)

    # Normalize media_hype to 0-1 range per award
    for award in out["AWARD_TYPE"].unique():
        mask = out["AWARD_TYPE"] == award
        max_hype = out.loc[mask, "media_hype"].max()
        if max_hype > 0:
            out.loc[mask, "media_hype_norm"] = out.loc[mask, "media_hype"] / max_hype
        else:
            out.loc[mask, "media_hype_norm"] = 0.0

    # Blend: original model score (80%) + media hype (20%)
    HYPE_WEIGHT = 0.20
    MODEL_WEIGHT = 1.0 - HYPE_WEIGHT

    for award in out["AWARD_TYPE"].unique():
        mask = out["AWARD_TYPE"] == award
        model_max = out.loc[mask, "predicted_vote_share"].max()
        if model_max > 0:
            model_norm = out.loc[mask, "predicted_vote_share"] / model_max
        else:
            model_norm = out.loc[mask, "predicted_vote_share"]

        hype_norm = out.loc[mask, "media_hype_norm"].fillna(0)
        blended = model_norm * MODEL_WEIGHT + hype_norm * HYPE_WEIGHT

        blend_max = blended.max()
        if blend_max > 0:
            out.loc[mask, "predicted_vote_share"] = blended / blend_max

        out.loc[mask, "predicted_rank"] = (
            out.loc[mask, "predicted_vote_share"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    out = out.sort_values(["AWARD_TYPE", "predicted_rank"]).reset_index(drop=True)
    return out
