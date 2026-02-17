"""
=============================================================================
Data Pipeline - Phase 1
=============================================================================
Downloads and stores all data required for model training and live trading:
  1. Historical 1-min OHLCV via Zerodha Kite (60-day chunks, 9-month aggregate)
  2. OI data via nsepython bhavcopy (Kite API does NOT include OI in candles)
  3. India VIX historical via nsepython
  4. News sentiment via Moneycontrol RSS (live) + GDELT (historical backtest)
  5. GIFT Nifty pre-market proxy (scraped from broker/TradingView)

Storage: ./data/ on local SSD. SQLite for structured data.

Usage:
    python scripts/data_pipeline.py --mode full      # Full 9-month download
    python scripts/data_pipeline.py --mode update     # Daily incremental update
    python scripts/data_pipeline.py --mode news       # News only
=============================================================================
"""

import os
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import feedparser
from dotenv import load_dotenv
from loguru import logger

# Kite Connect for historical data
from kiteconnect import KiteConnect

# NSE data (replaces broken NSEPy)
try:
    from nsepython import nse_optionchain_scrapper, nse_get_fii_dii
except ImportError:
    logger.warning("nsepython not installed. OI features will use fallback.")

from scripts.kimi_client import kimi_sentiment

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "market_data.db"

# Kite historical API: max 60 days per call for 1-minute data
KITE_1MIN_MAX_DAYS = 60
BACKTEST_MONTHS = 9  # ~270 days of trading data

# Instruments config
CONFIG_PATH = Path(__file__).parent.parent / "config" / "instruments.json"
with open(CONFIG_PATH) as f:
    INSTRUMENTS = json.load(f)

# Moneycontrol RSS feeds for live news
RSS_FEEDS = {
    "moneycontrol_latest": "https://www.moneycontrol.com/rss/latestnews.xml",
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}


# =============================================================================
# Database Setup
# =============================================================================
def init_db():
    """Create SQLite tables for market data storage."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1-minute OHLCV candles
    c.execute("""
        CREATE TABLE IF NOT EXISTS candles_1min (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, timestamp)
        )
    """)

    # Daily OI data (from bhavcopy / nsepython)
    c.execute("""
        CREATE TABLE IF NOT EXISTS oi_daily (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open_interest INTEGER,
            oi_change INTEGER,
            oi_change_pct REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # India VIX daily
    c.execute("""
        CREATE TABLE IF NOT EXISTS vix_daily (
            date TEXT PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL
        )
    """)

    # News sentiment (timestamped for backtesting)
    c.execute("""
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT,
            headline TEXT,
            sentiment_score REAL,
            sentiment_label TEXT,
            market_impact TEXT,
            raw_json TEXT
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


# =============================================================================
# 1. Kite Historical Data (1-min OHLCV)
# =============================================================================
def get_kite_client() -> KiteConnect:
    """Initialize Kite Connect client with credentials from .env."""
    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise ValueError("KITE_API_KEY and KITE_ACCESS_TOKEN must be set in .env")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    logger.info("Kite Connect client initialized")
    return kite


def fetch_kite_historical(
    kite: KiteConnect,
    instrument_token: int,
    symbol: str,
    from_date: datetime,
    to_date: datetime,
    interval: str = "minute",
) -> pd.DataFrame:
    """
    Fetch historical candles from Kite API.

    RULE: Kite allows max 60 days per API call for 1-min data.
    We loop in 60-day chunks to build the full 9-month dataset.
    Respects API rate limits with 0.5s sleep between calls.
    """
    all_data = []
    current_start = from_date

    while current_start < to_date:
        chunk_end = min(current_start + timedelta(days=KITE_1MIN_MAX_DAYS), to_date)

        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=current_start.strftime("%Y-%m-%d"),
                to_date=chunk_end.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if data:
                all_data.extend(data)
                logger.info(
                    f"  {symbol}: fetched {len(data)} candles "
                    f"({current_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')})"
                )
        except Exception as e:
            logger.error(f"  {symbol}: Kite API error for {current_start} - {chunk_end}: {e}")

        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Rate limit: ~2 calls/sec

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["symbol"] = symbol
    df.rename(columns={"date": "timestamp"}, inplace=True)
    return df


def download_all_historical(kite: KiteConnect):
    """
    Download 9 months of 1-min OHLCV for all instruments.
    Stores to SQLite candles_1min table.

    RULE: Loop in 60-day chunks. Max 60 days per Kite API call for minute data.
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=BACKTEST_MONTHS * 30)

    conn = sqlite3.connect(DB_PATH)

    # Core indices
    for name, info in INSTRUMENTS["core_indices"].items():
        logger.info(f"Downloading {name} (token: {info['kite_token']})...")
        df = fetch_kite_historical(
            kite, info["kite_token"], name, from_date, to_date
        )
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype(str)
            df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].to_sql(
                "candles_1min", conn, if_exists="append", index=False
            )
            logger.info(f"  {name}: saved {len(df)} candles")

    # Satellite stocks (top 50 Nifty stocks)
    # First, get instrument list to map symbols -> tokens
    instruments = kite.instruments("NSE")
    symbol_to_token = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}

    for symbol in INSTRUMENTS["satellite_universe"]:
        token = symbol_to_token.get(symbol)
        if not token:
            logger.warning(f"  {symbol}: not found in Kite instruments, skipping")
            continue

        logger.info(f"Downloading {symbol} (token: {token})...")
        df = fetch_kite_historical(kite, token, symbol, from_date, to_date)
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype(str)
            df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].to_sql(
                "candles_1min", conn, if_exists="append", index=False
            )
            logger.info(f"  {symbol}: saved {len(df)} candles")

        time.sleep(0.3)  # Extra rate limiting for batch downloads

    conn.close()
    logger.info("Historical OHLCV download complete")


# =============================================================================
# 2. OI Data via nsepython (bhavcopy)
# =============================================================================
def download_oi_data():
    """
    Download Open Interest data from NSE bhavcopy archives.

    RULE: Kite historical API does NOT include OI in OHLCV candles.
    We use nsepython for daily OI snapshots from bhavcopy.
    """
    conn = sqlite3.connect(DB_PATH)

    try:
        # Fetch option chain for Nifty and BankNifty (gives current OI)
        for index_name in ["NIFTY", "BANKNIFTY"]:
            try:
                oc = nse_optionchain_scrapper(index_name)
                if oc and "records" in oc and "data" in oc["records"]:
                    today = datetime.now().strftime("%Y-%m-%d")

                    # Aggregate total call OI and put OI
                    total_ce_oi = sum(
                        r.get("CE", {}).get("openInterest", 0)
                        for r in oc["records"]["data"]
                    )
                    total_pe_oi = sum(
                        r.get("PE", {}).get("openInterest", 0)
                        for r in oc["records"]["data"]
                    )

                    conn.execute(
                        """INSERT OR REPLACE INTO oi_daily
                           (symbol, date, open_interest, oi_change, oi_change_pct)
                           VALUES (?, ?, ?, ?, ?)""",
                        (f"{index_name}_CE", today, total_ce_oi, 0, 0.0),
                    )
                    conn.execute(
                        """INSERT OR REPLACE INTO oi_daily
                           (symbol, date, open_interest, oi_change, oi_change_pct)
                           VALUES (?, ?, ?, ?, ?)""",
                        (f"{index_name}_PE", today, total_pe_oi, 0, 0.0),
                    )
                    logger.info(f"  {index_name} OI: CE={total_ce_oi:,} PE={total_pe_oi:,}")
            except Exception as e:
                logger.error(f"  {index_name} OI fetch failed: {e}")

    except Exception as e:
        logger.error(f"OI data download failed: {e}")

    conn.commit()
    conn.close()


def download_vix_data(kite: KiteConnect):
    """Download India VIX historical data via Kite."""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=BACKTEST_MONTHS * 30)

    vix_token = INSTRUMENTS.get("vix_kite_token", 264969)

    try:
        data = kite.historical_data(
            instrument_token=vix_token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval="day",
        )

        if data:
            conn = sqlite3.connect(DB_PATH)
            for row in data:
                conn.execute(
                    """INSERT OR REPLACE INTO vix_daily
                       (date, open, high, low, close)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        str(row["date"])[:10],
                        row["open"], row["high"], row["low"], row["close"],
                    ),
                )
            conn.commit()
            conn.close()
            logger.info(f"VIX data: saved {len(data)} days")
    except Exception as e:
        logger.error(f"VIX download failed: {e}")


# =============================================================================
# 3. News Sentiment (RSS live + GDELT historical)
# =============================================================================
def fetch_rss_news() -> list[dict]:
    """
    Fetch latest news from Moneycontrol + ET RSS feeds.
    Returns list of dicts with 'title', 'published', 'source', 'link'.
    """
    all_articles = []

    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:  # Latest 20 per feed
                published = entry.get("published", "")
                all_articles.append({
                    "title": entry.get("title", ""),
                    "published": published,
                    "source": source_name,
                    "link": entry.get("link", ""),
                })
            logger.info(f"  RSS {source_name}: {len(feed.entries)} articles")
        except Exception as e:
            logger.error(f"  RSS {source_name} failed: {e}")

    return all_articles


def fetch_gdelt_news(query: str = "India stock market NSE", days_back: int = 1) -> list[dict]:
    """
    Fetch historical timestamped news from GDELT for backtesting.
    Uses GDELT DOC API (free, no auth required).

    RULE: For backtesting, only use news published BEFORE each bar's timestamp.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 50,
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "format": "json",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "published": article.get("seendate", ""),
                "source": "gdelt_" + article.get("domain", "unknown"),
                "link": article.get("url", ""),
            })
        logger.info(f"  GDELT: {len(articles)} articles for '{query}'")
        return articles
    except Exception as e:
        logger.error(f"  GDELT fetch failed: {e}")
        return []


def process_news_sentiment(articles: list[dict]) -> dict:
    """
    Process news articles through Kimi 2.5 for sentiment analysis.
    Stores results in SQLite for backtesting.

    RULE: Kimi 2.5 is the ONLY LLM for all intelligence tasks.
    """
    if not articles:
        return {"score": 0.0, "label": "neutral", "key_themes": [], "market_impact": "none"}

    headlines = [a["title"] for a in articles if a.get("title")]
    sentiment = kimi_sentiment(headlines)

    # Store in database
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    conn.execute(
        """INSERT INTO news_sentiment
           (timestamp, source, headline, sentiment_score, sentiment_label,
            market_impact, raw_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            now, "aggregated",
            "; ".join(headlines[:5]),
            sentiment.get("score", 0.0),
            sentiment.get("label", "neutral"),
            sentiment.get("market_impact", "none"),
            json.dumps(sentiment),
        ),
    )
    conn.commit()
    conn.close()

    logger.info(
        f"Sentiment: score={sentiment.get('score', 0):.2f} "
        f"label={sentiment.get('label')} impact={sentiment.get('market_impact')}"
    )
    return sentiment


# =============================================================================
# 4. Pre-Market Data (GIFT Nifty, VIX)
# =============================================================================
def get_premarket_data() -> dict:
    """
    Gather pre-market indicators for 8:30 AM daily job.

    RULE: GIFT Nifty replaced SGX Nifty (July 2023). Use as pre-market indicator.
    Available from broker feed or TradingView proxy.
    """
    premarket = {
        "gift_nifty_change_pct": 0.0,
        "vix_current": 0.0,
        "vix_prev_close": 0.0,
        "oi_nifty_change": 0,
        "oi_banknifty_change": 0,
        "timestamp": datetime.now().isoformat(),
    }

    # GIFT Nifty: best effort from available sources
    # In production, this comes from Kite WebSocket pre-market or broker API
    # Fallback: we log a warning and use 0.0 (flat assumption)
    logger.info("Pre-market: GIFT Nifty data requires live broker feed or TradingView proxy")

    # India VIX from database (latest)
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT close FROM vix_daily ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            premarket["vix_current"] = row[0]

        row2 = conn.execute(
            "SELECT close FROM vix_daily ORDER BY date DESC LIMIT 1 OFFSET 1"
        ).fetchone()
        if row2:
            premarket["vix_prev_close"] = row2[0]

        conn.close()
    except Exception as e:
        logger.error(f"VIX read failed: {e}")

    # OI from latest bhavcopy
    try:
        conn = sqlite3.connect(DB_PATH)
        for idx in ["NIFTY", "BANKNIFTY"]:
            ce_row = conn.execute(
                "SELECT open_interest FROM oi_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (f"{idx}_CE",),
            ).fetchone()
            pe_row = conn.execute(
                "SELECT open_interest FROM oi_daily WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (f"{idx}_PE",),
            ).fetchone()
            if ce_row and pe_row:
                premarket[f"oi_{idx.lower()}_change"] = ce_row[0] - pe_row[0]
        conn.close()
    except Exception as e:
        logger.error(f"OI read failed: {e}")

    logger.info(f"Pre-market data: VIX={premarket['vix_current']:.1f}")
    return premarket


# =============================================================================
# Feature Engineering (for model training)
# =============================================================================
def compute_features(symbol: str, date: str = None) -> pd.DataFrame:
    """
    Compute all features for a given symbol from stored data.
    Features are defined in config/strategy_params.json -> model.feature_list.

    Returns DataFrame with one row per 5-min bar, with all features computed.
    """
    conn = sqlite3.connect(DB_PATH)

    query = "SELECT * FROM candles_1min WHERE symbol = ?"
    params = [symbol]
    if date:
        query += " AND timestamp LIKE ?"
        params.append(f"{date}%")
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Resample to 5-minute bars for feature computation
    df_5min = df.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(df_5min) < 30:
        return pd.DataFrame()

    # --- Technical Features ---
    import ta

    # RSI (14-period on 5-min bars)
    df_5min["rsi_14"] = ta.momentum.RSIIndicator(df_5min["close"], window=14).rsi()

    # ATR (14-period)
    atr = ta.volatility.AverageTrueRange(
        df_5min["high"], df_5min["low"], df_5min["close"], window=14
    )
    df_5min["atr_14"] = atr.average_true_range()

    # VWAP (cumulative intraday, RESET DAILY)
    # VWAP must reset each trading day — cumulative across months is meaningless
    df_5min["_date"] = df_5min.index.date
    df_5min["_cum_vol_price"] = (df_5min["close"] * df_5min["volume"]).groupby(df_5min["_date"]).cumsum()
    df_5min["_cum_vol"] = df_5min["volume"].groupby(df_5min["_date"]).cumsum()
    df_5min["vwap"] = df_5min["_cum_vol_price"] / df_5min["_cum_vol"].replace(0, np.nan)
    df_5min.drop(columns=["_date", "_cum_vol_price", "_cum_vol"], inplace=True)
    df_5min["vwap_distance_pct"] = (
        (df_5min["close"] - df_5min["vwap"]) / df_5min["vwap"] * 100
    )

    # Volume ratio (current / 20-period average)
    df_5min["volume_ratio_20"] = df_5min["volume"] / df_5min["volume"].rolling(20).mean()

    # Bollinger Band width
    bb = ta.volatility.BollingerBands(df_5min["close"], window=20)
    df_5min["bb_width_20"] = bb.bollinger_wband()

    # MACD histogram
    macd = ta.trend.MACD(df_5min["close"])
    df_5min["macd_histogram"] = macd.macd_diff()

    # ADX (14-period)
    adx = ta.trend.ADXIndicator(df_5min["high"], df_5min["low"], df_5min["close"], window=14)
    df_5min["adx_14"] = adx.adx()

    # --- Time features ---
    df_5min["hour_of_day"] = df_5min.index.hour
    df_5min["day_of_week"] = df_5min.index.dayofweek

    # --- Daily-level features (computed once per day, forward-filled) ---
    df_5min["prev_close_return"] = df_5min["close"].pct_change()

    # Gap % (first bar of day vs prev day close)
    df_5min["gap_pct"] = 0.0  # Placeholder, computed in daily context

    # Open = Low / Open = High (first 5-min candle of day)
    df_5min["open_eq_low"] = (df_5min["open"] == df_5min["low"]).astype(int)
    df_5min["open_eq_high"] = (df_5min["open"] == df_5min["high"]).astype(int)

    # --- External features from database (VIX, OI) ---
    # These were previously hardcoded to 0.0, causing the model to train on zeros.
    df_5min["gift_nifty_gap_pct"] = 0.0  # Only available at live inference time

    # VIX: merge daily VIX close into each 5-min bar by date
    try:
        vix_conn = sqlite3.connect(DB_PATH)
        vix_df = pd.read_sql_query("SELECT date, close as vix_close FROM vix_daily", vix_conn)
        vix_conn.close()
        if not vix_df.empty:
            vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
            df_5min["_bar_date"] = df_5min.index.date
            vix_map = vix_df.set_index("date")["vix_close"].to_dict()
            df_5min["vix_level"] = df_5min["_bar_date"].map(vix_map).ffill().fillna(15.0)
            df_5min["vix_change_pct"] = df_5min["vix_level"].pct_change().fillna(0.0) * 100
            df_5min.drop(columns=["_bar_date"], inplace=True)
        else:
            df_5min["vix_level"] = 15.0
            df_5min["vix_change_pct"] = 0.0
    except Exception:
        df_5min["vix_level"] = 15.0
        df_5min["vix_change_pct"] = 0.0

    # OI change: merge daily OI into 5-min bars by date
    try:
        oi_conn = sqlite3.connect(DB_PATH)
        oi_df = pd.read_sql_query(
            "SELECT date, oi_change_pct FROM oi_daily WHERE symbol LIKE '%_CE'", oi_conn
        )
        oi_conn.close()
        if not oi_df.empty:
            oi_df["date"] = pd.to_datetime(oi_df["date"]).dt.date
            df_5min["_bar_date"] = df_5min.index.date
            oi_map = oi_df.groupby("date")["oi_change_pct"].mean().to_dict()
            df_5min["oi_change_pct"] = df_5min["_bar_date"].map(oi_map).fillna(0.0)
            df_5min.drop(columns=["_bar_date"], inplace=True)
        else:
            df_5min["oi_change_pct"] = 0.0
    except Exception:
        df_5min["oi_change_pct"] = 0.0

    # News sentiment: merge daily sentiment score by date
    try:
        news_conn = sqlite3.connect(DB_PATH)
        news_df = pd.read_sql_query(
            "SELECT substr(timestamp, 1, 10) as date, sentiment_score FROM news_sentiment", news_conn
        )
        news_conn.close()
        if not news_df.empty:
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
            df_5min["_bar_date"] = df_5min.index.date
            news_map = news_df.groupby("date")["sentiment_score"].mean().to_dict()
            df_5min["news_sentiment_score"] = df_5min["_bar_date"].map(news_map).fillna(0.0)
            df_5min.drop(columns=["_bar_date"], inplace=True)
        else:
            df_5min["news_sentiment_score"] = 0.0
    except Exception:
        df_5min["news_sentiment_score"] = 0.0

    df_5min["regime_label"] = 0  # 0=neutral, 1=trending, 2=ranging (set by model at inference)

    df_5min.dropna(inplace=True)
    return df_5min


# =============================================================================
# Main CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSE Paper Trading - Data Pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "update", "news", "features"],
        default="update",
        help="full=9-month download, update=daily incremental, news=news only, features=compute features",
    )
    args = parser.parse_args()

    init_db()

    if args.mode in ("full", "update"):
        try:
            kite = get_kite_client()

            if args.mode == "full":
                logger.info("=== FULL 9-MONTH HISTORICAL DOWNLOAD ===")
                download_all_historical(kite)

            download_vix_data(kite)
            download_oi_data()

        except Exception as e:
            logger.error(f"Kite data pipeline failed: {e}")
            logger.info("Continuing with news pipeline...")

    if args.mode in ("full", "update", "news"):
        logger.info("=== NEWS SENTIMENT PIPELINE ===")
        articles = fetch_rss_news()
        if args.mode == "full":
            gdelt_articles = fetch_gdelt_news(days_back=30)
            articles.extend(gdelt_articles)
        sentiment = process_news_sentiment(articles)
        logger.info(f"Final sentiment: {sentiment}")

    if args.mode == "features":
        logger.info("=== FEATURE COMPUTATION ===")
        for symbol in ["NIFTY", "BANKNIFTY"] + INSTRUMENTS["satellite_universe"][:10]:
            df = compute_features(symbol)
            if not df.empty:
                out_path = DATA_DIR / f"features_{symbol}.parquet"
                df.to_parquet(out_path)
                logger.info(f"  {symbol}: {len(df)} feature rows -> {out_path}")

    logger.info("Data pipeline complete.")


if __name__ == "__main__":
    main()
