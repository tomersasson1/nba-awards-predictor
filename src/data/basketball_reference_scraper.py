from __future__ import annotations

"""
Utilities for scraping award voting results from Basketball Reference.

This focuses on pages like:
https://www.basketball-reference.com/awards/awards_2024.html
which contain voting tables for MVP, DPOY, ROTY, MIP, and 6MOY.

Fetching strategy (tiered fallback):
  1. Local cached HTML (data/raw/awards_YYYY.html)
  2. curl_cffi with Chrome TLS impersonation (fast, no browser needed)
  3. Playwright headless Chromium (heavy but defeats JS challenges)
"""

from typing import List

import json
import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

BASE_URL = "https://www.basketball-reference.com"


class ScrapingError(RuntimeError):
    """Raised when an expected table cannot be found or parsed."""


def _awards_page_url(season_start_year: int) -> str:
    """Build the awards page URL for a given season start year."""
    return f"{BASE_URL}/awards/awards_{season_start_year}.html"


def _awards_html_path(season_start_year: int, base_dir: Path | None = None) -> Path:
    """
    Local path for cached awards HTML, e.g. data/raw/awards_2015.html.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"awards_{season_start_year}.html"


def _build_headers(url: str) -> dict[str, str]:
    """Assemble browser-like request headers, optionally enriched via env vars."""
    headers = {
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": BASE_URL + "/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1",
    }

    cookie = os.getenv("BBR_COOKIE")
    if cookie:
        headers["Cookie"] = cookie

    extra = os.getenv("BBR_EXTRA_HEADERS")
    if extra:
        try:
            headers.update(json.loads(extra))
        except Exception:
            pass

    return headers


def _fetch_with_curl_cffi(url: str) -> str:
    """
    Tier 1: Use curl_cffi to impersonate Chrome's TLS fingerprint.

    This bypasses Cloudflare/Akamai bot detection that blocks Python's
    default TLS stack (urllib3/OpenSSL).
    """
    headers = _build_headers(url)
    resp = cffi_requests.get(
        url,
        headers=headers,
        impersonate="chrome",
        timeout=15,
    )
    if resp.status_code == 403:
        raise ScrapingError(
            f"curl_cffi received 403 for {url}; falling back to Playwright."
        )
    resp.raise_for_status()
    return resp.text


def _fetch_with_playwright(url: str) -> str:
    """
    Tier 2: Launch a real headless Chromium via Playwright.

    Handles JS challenges and advanced fingerprinting that even
    curl_cffi cannot bypass.  Requires `playwright install chromium`.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ScrapingError(
            "Playwright is not installed.  Run:\n"
            "  pip install playwright && playwright install chromium"
        ) from exc

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            ),
        )

        cookie = os.getenv("BBR_COOKIE")
        if cookie:
            cookie_objects = []
            for pair in cookie.split("; "):
                if "=" in pair:
                    name, value = pair.split("=", 1)
                    cookie_objects.append({
                        "name": name,
                        "value": value,
                        "domain": ".basketball-reference.com",
                        "path": "/",
                    })
            if cookie_objects:
                context.add_cookies(cookie_objects)

        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("table", timeout=15_000)
        html = page.content()
        browser.close()

    return html


def _fetch_html_from_web(url: str) -> str:
    """
    Fetch raw HTML from Basketball Reference with tiered fallback:
      1. curl_cffi (Chrome TLS impersonation) -- fast, lightweight
      2. Playwright headless Chromium           -- heavy, reliable
    """
    time.sleep(1.0)

    try:
        return _fetch_with_curl_cffi(url)
    except Exception as curl_err:
        print(f"  curl_cffi failed ({curl_err}); trying Playwright...")

    try:
        return _fetch_with_playwright(url)
    except Exception as pw_err:
        raise ScrapingError(
            f"All fetch strategies failed for {url}.\n"
            f"  curl_cffi error : {curl_err}\n"
            f"  Playwright error: {pw_err}\n"
            "You can manually download the page HTML and save it as "
            f"'data/raw/awards_{url.rsplit('_', 1)[-1]}' so the scraper "
            "can read it from disk."
        ) from pw_err


def _find_table_in_comments(soup: BeautifulSoup, table_id: str) -> "BeautifulSoup | None":
    """Basketball Reference hides some tables inside HTML comments."""
    import re
    from bs4 import Comment

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if f'id="{table_id}"' in comment:
            inner = BeautifulSoup(comment, "lxml")
            tbl = inner.find("table", id=table_id)
            if tbl:
                return tbl
    return None


def _parse_voting_table(soup: BeautifulSoup, table_id: str, award_type: str, season: str) -> pd.DataFrame:
    table = soup.find("table", id=table_id)
    if table is None:
        table = _find_table_in_comments(soup, table_id)
    if table is None:
        raise ScrapingError(f"Could not find table with id={table_id} for award {award_type} and season {season}")

    df = pd.read_html(StringIO(str(table)))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[-1].strip() if isinstance(c, tuple) else str(c).strip()
            for c in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    if "Rank" in df.columns:
        df = df[df["Rank"].notna()]
    df["AWARD_TYPE"] = award_type
    df["SEASON"] = season
    return df


AWARD_TABLE_IDS = {
    "MVP": "mvp",
    "DPOY": "dpoy",
    "ROTY": "roy",
    "MIP": "mip",
    "6MOY": "smoy",
    "COTY": "coy",
}


def fetch_award_voting_for_season(season_start_year: int) -> pd.DataFrame:
    """
    Fetch voting results for all major awards for a given season start year.

    Returns a DataFrame with at least:
    - 'AWARD_TYPE'
    - 'SEASON'
    - 'Player'
    - 'Rank'
    - 'Pts Won' / 'Vote Pts' (depends on season)
    - 'First' (first-place votes)
    plus any other columns BR provides.
    """
    url = _awards_page_url(season_start_year)
    html_path = _awards_html_path(season_start_year)

    # 1) Try local cached HTML first (user may have downloaded via browser/curl).
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
    else:
        # 2) Fallback: fetch from web, then cache to local HTML file.
        html = _fetch_html_from_web(url)
        html_path.write_text(html, encoding="utf-8")

    soup = BeautifulSoup(html, "lxml")
    # BBR awards_YYYY.html uses the ENDING year. The NBA season that ends in
    # 2025 is "2024-25". So start_year = season_start_year - 1.
    start_yr = season_start_year - 1
    season_str = f"{start_yr}-{season_start_year % 100:02d}"

    frames: List[pd.DataFrame] = []
    for award_type, table_id in AWARD_TABLE_IDS.items():
        try:
            frames.append(_parse_voting_table(soup, table_id, award_type, season_str))
        except ScrapingError:
            # Some seasons may not have all awards; skip gracefully
            continue

    if not frames:
        raise ScrapingError(f"No award tables parsed for season {season_start_year}")

    return pd.concat(frames, ignore_index=True)


def fetch_award_voting_for_seasons(start_season: int, end_season: int) -> pd.DataFrame:
    """Fetch award voting for a range of seasons."""
    frames: List[pd.DataFrame] = []
    for year in range(start_season, end_season + 1):
        frames.append(fetch_award_voting_for_season(year))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


