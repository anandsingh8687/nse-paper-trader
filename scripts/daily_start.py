"""
=============================================================================
Daily Start - Phase 5 Orchestrator
=============================================================================
Main entry point. User runs this at 8:00 AM IST (or via OpenClaw cron).

Sequence:
  1. 8:00 AM - Data update (incremental OHLCV + OI + VIX)
  2. 8:30 AM - Kimi 2.5 daily brief (pre-market + news + model inference)
  3. 9:15 AM - Market open → scan candidates → execute top 2-4
  4. 9:15 - 3:20 PM - Monitor exits (SL, scale-out, trail, time stop)
  5. 3:20 PM - Auto square-off all positions
  6. 3:30 PM - Daily summary + log

Usage:
    python scripts/daily_start.py                 # Full day (wait for market)
    python scripts/daily_start.py --backtest      # Run 9-month backtest
    python scripts/daily_start.py --brief-only    # Just generate morning brief
    python scripts/daily_start.py --setup-data    # First-time data download
=============================================================================
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import schedule
import pytz
from loguru import logger
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_pipeline import (
    init_db, download_all_historical, download_oi_data, download_vix_data,
    get_kite_client, fetch_rss_news, process_news_sentiment, get_premarket_data,
    compute_features, DATA_DIR, INSTRUMENTS,
)
from scripts.model_trainer import train_and_evaluate, load_model, predict
from scripts.strategy_engine import StrategyEngine, run_backtest
from scripts.kimi_client import kimi_daily_brief, kimi_chat

load_dotenv()

IST = pytz.timezone("Asia/Kolkata")
CAPITAL = float(os.getenv("INITIAL_CAPITAL", 1_000_000))

# Configure logging
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "daily_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
)


# =============================================================================
# Telegram Alerts (optional)
# =============================================================================
def send_telegram(message: str):
    """Send alert to Telegram if configured."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        return

    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# =============================================================================
# Phase Functions
# =============================================================================
def phase_data_update():
    """Phase 1: Incremental data update."""
    logger.info("=== PHASE 1: DATA UPDATE ===")
    try:
        init_db()
        kite = get_kite_client()
        download_vix_data(kite)
        download_oi_data()

        # Compute features for today's date
        today = datetime.now().strftime("%Y-%m-%d")
        for symbol in ["NIFTY", "BANKNIFTY"] + INSTRUMENTS["satellite_universe"][:10]:
            df = compute_features(symbol, date=today)
            if not df.empty:
                out_path = DATA_DIR / f"features_{symbol}_live.parquet"
                df.to_parquet(out_path)

        logger.info("Data update complete")
    except Exception as e:
        logger.error(f"Data update failed: {e}")
        send_telegram(f"DATA UPDATE FAILED: {e}")


def phase_morning_brief() -> str:
    """Phase 2: Generate Kimi 2.5 morning brief."""
    logger.info("=== PHASE 2: MORNING BRIEF (8:30 AM) ===")

    engine = StrategyEngine(capital=CAPITAL, mode="sandbox")
    brief = engine.morning_routine()

    send_telegram(f"*Daily Trading Brief*\n\n{brief}")
    return brief


def phase_market_open(engine: StrategyEngine):
    """Phase 3: Market open - scan and execute."""
    logger.info("=== PHASE 3: MARKET OPEN (9:15 AM) ===")

    candidates = engine.scan_candidates()
    if candidates:
        engine.execute_trades(candidates)
        send_telegram(
            f"*Trades Executed:* {len(engine.active_trades)}\n"
            + "\n".join(
                f"- {t.candidate.symbol} {t.candidate.direction} "
                f"P={t.candidate.p_win:.0%} EV={t.candidate.ev:.2f}R"
                for t in engine.active_trades
            )
        )
    else:
        logger.info("No valid candidates found. Sitting out today.")
        send_telegram("No trades today - no candidates met entry criteria.")


def phase_monitor_exits(engine: StrategyEngine):
    """Phase 4: Monitor positions for exit conditions."""
    if not engine.active_trades:
        return

    # In paper trading, simulate current prices
    # In live trading, this would use Kite WebSocket
    current_prices = {}
    for trade in engine.active_trades:
        if trade.status == "open":
            # Placeholder: in production, fetch real-time price from Kite WebSocket
            # For paper trading, use last known price + small random walk
            import random
            noise = random.gauss(0, trade.candidate.atr * 0.1)
            current_prices[trade.candidate.symbol] = trade.entry_price + noise

    engine.manage_exits(current_prices)


def phase_squareoff(engine: StrategyEngine):
    """Phase 5: Auto square-off at 3:20 PM IST."""
    logger.info("=== PHASE 5: AUTO SQUARE-OFF (3:20 PM) ===")

    for trade in engine.active_trades:
        if trade.status != "closed":
            engine._close_trade(trade, trade.entry_price, "auto_squareoff_320pm")

    engine._save_daily_summary()

    total_pnl = engine.daily_pnl
    send_telegram(
        f"*End of Day Summary*\n"
        f"Trades: {engine.trades_today}\n"
        f"P&L: {total_pnl:+,.0f} ({total_pnl/CAPITAL:+.2%})\n"
        f"Capital: {CAPITAL + total_pnl:,.0f}"
    )
    logger.info(f"Day complete. P&L: {total_pnl:+,.0f}")


# =============================================================================
# Main Scheduler
# =============================================================================
def run_full_day():
    """
    Full trading day scheduler.
    Runs 8:00 AM to 3:30 PM IST, executing each phase at the right time.
    """
    logger.info("=" * 60)
    logger.info("NSE PAPER TRADING SYSTEM - DAILY START")
    logger.info(f"Date: {datetime.now(IST).strftime('%Y-%m-%d %A')}")
    logger.info(f"Capital: {CAPITAL:,.0f}")
    logger.info("=" * 60)

    engine = StrategyEngine(capital=CAPITAL, mode="sandbox")

    # Phase 1: Data update (runs immediately)
    phase_data_update()

    # Phase 2: Morning brief at 8:30 AM
    now_ist = datetime.now(IST)
    target_830 = now_ist.replace(hour=8, minute=30, second=0, microsecond=0)
    if now_ist < target_830:
        wait_secs = (target_830 - now_ist).total_seconds()
        logger.info(f"Waiting {wait_secs/60:.0f} minutes until 8:30 AM brief...")
        time.sleep(max(0, wait_secs))

    brief = phase_morning_brief()

    # Phase 3: Market open at 9:15 AM
    now_ist = datetime.now(IST)
    target_915 = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    if now_ist < target_915:
        wait_secs = (target_915 - now_ist).total_seconds()
        logger.info(f"Waiting {wait_secs/60:.0f} minutes until 9:15 AM market open...")
        time.sleep(max(0, wait_secs))

    phase_market_open(engine)

    # Phase 4: Monitor exits every 5 minutes until 3:20 PM
    logger.info("Entering exit monitoring loop (every 5 min until 3:20 PM)...")
    while True:
        now_ist = datetime.now(IST)
        if now_ist.hour >= 15 and now_ist.minute >= 20:
            break
        if engine.shutdown:
            logger.warning("System shutdown triggered. Exiting loop.")
            break

        phase_monitor_exits(engine)
        time.sleep(300)  # Check every 5 minutes

    # Phase 5: Square-off
    phase_squareoff(engine)

    logger.info("Daily trading session complete.")


def run_setup_data():
    """First-time setup: download full 9-month historical data."""
    logger.info("=== FIRST-TIME DATA SETUP ===")
    init_db()

    try:
        kite = get_kite_client()
        download_all_historical(kite)
        download_vix_data(kite)
        download_oi_data()
    except Exception as e:
        logger.error(f"Kite data download failed: {e}")
        logger.info("You can still proceed with model training on available data.")

    # Compute features
    logger.info("Computing features...")
    for symbol in ["NIFTY", "BANKNIFTY"] + INSTRUMENTS["satellite_universe"][:10]:
        df = compute_features(symbol)
        if not df.empty:
            out_path = DATA_DIR / f"features_{symbol}.parquet"
            df.to_parquet(out_path)
            logger.info(f"  {symbol}: {len(df)} rows -> {out_path}")

    # Train model
    logger.info("Training models...")
    result = train_and_evaluate()
    if "error" not in result:
        logger.info(f"Best model: {result['best_model']} (Sharpe: {result['all_results'][result['best_model']]['sharpe']})")
    else:
        logger.error(f"Training failed: {result['error']}")

    logger.info("Setup complete. Run: python scripts/daily_start.py")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSE Paper Trading - Daily Orchestrator")
    parser.add_argument("--backtest", action="store_true", help="Run 9-month backtest")
    parser.add_argument("--brief-only", action="store_true", help="Generate morning brief only")
    parser.add_argument("--setup-data", action="store_true", help="First-time data download")
    parser.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.setup_data:
        run_setup_data()
    elif args.backtest:
        results = run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            capital=CAPITAL,
        )
        print(json.dumps(results, indent=2))
    elif args.brief_only:
        brief = phase_morning_brief()
        print(brief)
    else:
        run_full_day()


if __name__ == "__main__":
    main()
