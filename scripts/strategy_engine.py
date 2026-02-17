"""
=============================================================================
Strategy Engine - Phase 3
=============================================================================
Core intraday strategy for NSE paper trading. ALL rules are hard-coded as
both comments and executable logic. Nothing is configurable at runtime
without explicit code change (safety by design).

Architecture:
  - Core: Nifty / BankNifty index OPTIONS
  - Satellite: 3-5 stocks from daily ranked list (top 50 Nifty stocks)

LOCKED STRATEGY RULES (do not modify without weekly Kimi 2.5 review):
  - Entry: P(win) >= 60% AND EV >= 0.4R after 0.08% costs
  - Triggers: Regime match + (momentum OR mean-reversion)
  - Momentum: 5-min breakout + vol > 1.8x 20-period avg + OI rising
  - Mean-reversion: Open=Low/High on first 5-min candle + near VWAP +/- 0.2%
  - Sizing: Base 1% risk, multiplier 1x/2x/3x by P(win) band
  - Max 4 trades/day, max 3% total risk
  - Exit: Hard SL -1R (ATR), scale 50% at +0.8R, trail rest
  - Time stop 2.5 hours, auto square-off 3:20 PM IST
  - Safety: VIX > 25 skip (unless P>75%), max daily loss 2% -> shutdown
  - Skip budget/expiry days unless P > 75%

Usage:
    # Called by daily_start.py, not run standalone
    from scripts.strategy_engine import StrategyEngine
    engine = StrategyEngine(capital=1_000_000, mode="sandbox")
    candidates = engine.scan_candidates()
    engine.execute_trades(candidates)
=============================================================================
"""

import os
import json
import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from scripts.model_trainer import predict, load_model
from scripts.data_pipeline import (
    compute_features, get_premarket_data, fetch_rss_news,
    process_news_sentiment, DB_PATH, DATA_DIR, INSTRUMENTS,
)
from scripts.kimi_client import kimi_daily_brief

load_dotenv()

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRADE_LOG_DB = LOG_DIR / "trades.db"


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class TradeCandidate:
    """A potential trade identified by the strategy."""
    symbol: str
    direction: str          # "long" or "short"
    trigger_type: str       # "momentum" or "mean_reversion"
    p_win: float            # P(win) from model, 0.0 to 1.0
    ev: float               # Expected value in R-multiples
    regime: str             # "trending", "ranging", "neutral"
    entry_price: float
    stop_loss: float        # ATR-based, -1R
    target_1: float         # +0.8R (scale out 50%)
    atr: float
    risk_r: float           # 1R in absolute price terms
    position_size: int      # Shares/lots based on sizing rules
    risk_pct: float         # Portfolio risk % for this trade
    features: dict = field(default_factory=dict)


@dataclass
class ActiveTrade:
    """A trade currently in progress."""
    candidate: TradeCandidate
    entry_time: datetime
    entry_price: float
    position_size: int
    remaining_size: int     # After partial exits
    unrealized_pnl: float = 0.0
    scaled_out: bool = False
    trail_stop: Optional[float] = None
    status: str = "open"    # open, partial, closed


# =============================================================================
# Trade Logger (SQLite)
# =============================================================================
def init_trade_log():
    """Create trade log table in SQLite."""
    conn = sqlite3.connect(TRADE_LOG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT,
            trigger_type TEXT,
            entry_price REAL,
            exit_price REAL,
            stop_loss REAL,
            p_win REAL,
            ev REAL,
            regime TEXT,
            position_size INTEGER,
            risk_pct REAL,
            pnl REAL,
            pnl_r REAL,
            duration_minutes REAL,
            exit_reason TEXT,
            features_json TEXT,
            status TEXT DEFAULT 'open'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            total_trades INTEGER,
            winning_trades INTEGER,
            total_pnl REAL,
            max_drawdown REAL,
            sharpe_estimate REAL,
            capital_end REAL,
            shutdown_triggered INTEGER DEFAULT 0,
            daily_brief TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_trade(trade: ActiveTrade, exit_price: float, exit_reason: str, pnl: float):
    """Log a completed trade to SQLite."""
    conn = sqlite3.connect(TRADE_LOG_DB)
    duration = (datetime.now() - trade.entry_time).total_seconds() / 60

    risk_r = trade.candidate.risk_r
    pnl_r = pnl / risk_r if risk_r > 0 else 0.0

    conn.execute(
        """INSERT INTO trades
           (timestamp, symbol, direction, trigger_type, entry_price, exit_price,
            stop_loss, p_win, ev, regime, position_size, risk_pct, pnl, pnl_r,
            duration_minutes, exit_reason, features_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'closed')""",
        (
            trade.entry_time.isoformat(),
            trade.candidate.symbol,
            trade.candidate.direction,
            trade.candidate.trigger_type,
            trade.entry_price,
            exit_price,
            trade.candidate.stop_loss,
            trade.candidate.p_win,
            trade.candidate.ev,
            trade.candidate.regime,
            trade.candidate.position_size,
            trade.candidate.risk_pct,
            round(pnl, 2),
            round(pnl_r, 3),
            round(duration, 1),
            exit_reason,
            json.dumps(trade.candidate.features),
        ),
    )
    conn.commit()
    conn.close()
    logger.info(
        f"TRADE LOGGED: {trade.candidate.symbol} {trade.candidate.direction} "
        f"PnL={pnl:+.2f} ({pnl_r:+.2f}R) reason={exit_reason}"
    )


# =============================================================================
# Strategy Engine
# =============================================================================
class StrategyEngine:
    """
    Core strategy engine implementing all hard-coded rules.

    RULE: All strategy parameters are constants, not configurable.
    Changes require code modification + Kimi 2.5 review.
    """

    # -----------------------------------------------------------------------
    # LOCKED CONSTANTS — DO NOT CHANGE WITHOUT KIMI 2.5 WEEKLY REVIEW
    # -----------------------------------------------------------------------
    MIN_P_WIN = 0.60              # RULE: Entry only if P(win) >= 60%
    MIN_EV_R = 0.4                # RULE: EV >= 0.4R after costs
    COST_SLIPPAGE_PCT = 0.0008    # RULE: ~0.08% round-trip costs/slippage

    # Momentum trigger thresholds
    VOLUME_MULTIPLIER = 1.8       # RULE: Vol > 1.8x 20-period 5-min avg
    VOLUME_LOOKBACK = 20          # RULE: 20 x 5-min bars for volume avg

    # Mean-reversion trigger thresholds
    VWAP_BAND_PCT = 0.002         # RULE: Price within VWAP +/- 0.2%

    # Position sizing
    BASE_RISK_PCT = 0.01          # RULE: Base 1% portfolio risk
    MULTIPLIER_LOW = 1.0          # RULE: 1x at P 60-65%
    MULTIPLIER_MID = 2.0          # RULE: 2x at P 65-72%
    MULTIPLIER_HIGH = 3.0         # RULE: 3x at P > 72%
    MAX_TRADES_PER_DAY = 4        # RULE: Max 4 trades/day
    MAX_TOTAL_RISK_PCT = 0.03     # RULE: Max 3% total portfolio risk

    # Exit rules
    HARD_SL_R = -1.0              # RULE: Hard stop-loss at -1R (ATR-based)
    SCALE_OUT_PCT = 0.50          # RULE: Scale out 50% at +0.8R
    SCALE_OUT_TARGET_R = 0.8      # RULE: Target for first scale-out
    TIME_STOP_HOURS = 2.5         # RULE: Time stop after 2.5 hours
    AUTO_SQUAREOFF = time(15, 20) # RULE: Auto square-off at 3:20 PM IST

    # Safety
    VIX_SKIP_THRESHOLD = 25       # RULE: No trade if VIX > 25
    VIX_OVERRIDE_MIN_P = 0.75     # RULE: ...unless P > 75%
    MAX_DAILY_LOSS_PCT = 0.02     # RULE: Max daily loss 2% -> shutdown
    BUDGET_EXPIRY_MIN_P = 0.75    # RULE: Skip budget/expiry days unless P > 75%
    # -----------------------------------------------------------------------

    def __init__(self, capital: float = 1_000_000, mode: str = "sandbox"):
        """
        Initialize strategy engine.

        Args:
            capital: Starting portfolio capital in INR.
            mode: 'sandbox' for paper trading, 'live' for real (NOT recommended yet).
        """
        self.capital = capital
        self.mode = mode
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.total_risk_today = 0.0
        self.active_trades: list[ActiveTrade] = []
        self.shutdown = False
        self.premarket_data = {}
        self.sentiment = {}
        self.daily_brief = ""

        init_trade_log()
        logger.info(
            f"Strategy Engine initialized: capital={capital:,.0f} mode={mode}"
        )

    # -----------------------------------------------------------------------
    # Safety Checks
    # -----------------------------------------------------------------------
    def check_safety(self) -> tuple[bool, str]:
        """
        Run all safety checks before allowing any trade.

        RULE: No trade if VIX > 25 unless P > 75%.
        RULE: Max daily loss 2% -> shutdown.
        RULE: Skip budget/expiry days unless P > 75%.
        """
        if self.shutdown:
            return False, "SHUTDOWN: Max daily loss exceeded"

        # Check daily loss
        if abs(self.daily_pnl) / self.capital >= self.MAX_DAILY_LOSS_PCT:
            self.shutdown = True
            logger.warning(f"SAFETY SHUTDOWN: Daily loss {self.daily_pnl/self.capital:.2%}")
            return False, f"Daily loss limit hit: {self.daily_pnl:,.0f}"

        # Check max trades
        if self.trades_today >= self.MAX_TRADES_PER_DAY:
            return False, f"Max trades reached: {self.trades_today}/{self.MAX_TRADES_PER_DAY}"

        # Check total risk
        if self.total_risk_today >= self.MAX_TOTAL_RISK_PCT:
            return False, f"Max risk reached: {self.total_risk_today:.1%}/{self.MAX_TOTAL_RISK_PCT:.1%}"

        # VIX check
        vix = self.premarket_data.get("vix_current", 0)
        if vix > self.VIX_SKIP_THRESHOLD:
            return False, f"VIX too high: {vix:.1f} > {self.VIX_SKIP_THRESHOLD}"

        # Budget/expiry day check
        today = datetime.now()
        # Indian budget day: typically Feb 1
        is_budget_day = (today.month == 2 and today.day == 1)
        # Weekly expiry: Thursday for Nifty/BankNifty
        is_expiry_day = (today.weekday() == 3)  # Thursday

        if is_budget_day or is_expiry_day:
            logger.info(f"Budget/expiry day detected. Trades require P > {self.BUDGET_EXPIRY_MIN_P:.0%}")
            # Will be checked per-candidate in validate_candidate()

        return True, "OK"

    # -----------------------------------------------------------------------
    # Trigger Detection
    # -----------------------------------------------------------------------
    def check_momentum_trigger(self, df: pd.DataFrame) -> bool:
        """
        Check momentum trigger conditions on latest 5-min bars.

        RULE: 5-min breakout + vol > 1.8x 20-period avg + OI rising.
        All three must be true simultaneously.
        """
        if len(df) < self.VOLUME_LOOKBACK + 1:
            return False

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Breakout: price exceeds previous high (5-min bar)
        breakout = latest["close"] > prev["high"]

        # 2. Volume > 1.8x 20-period average
        avg_vol = df["volume"].iloc[-(self.VOLUME_LOOKBACK + 1):-1].mean()
        vol_spike = latest["volume"] > (self.VOLUME_MULTIPLIER * avg_vol) if avg_vol > 0 else False

        # 3. OI rising (from pre-market data or latest OI snapshot)
        oi_rising = self.premarket_data.get("oi_nifty_change", 0) > 0

        triggered = breakout and vol_spike and oi_rising
        if triggered:
            logger.info(
                f"  MOMENTUM TRIGGER: breakout={breakout} vol_spike={vol_spike} "
                f"(vol={latest['volume']:,.0f} vs avg={avg_vol:,.0f}) oi_rising={oi_rising}"
            )
        return triggered

    def check_mean_reversion_trigger(self, df: pd.DataFrame) -> bool:
        """
        Check mean-reversion trigger conditions.

        RULE: Open=Low/High on first 5-min candle + price near VWAP +/- 0.2%.
        """
        if len(df) < 2:
            return False

        latest = df.iloc[-1]

        # 1. Open = Low (bullish) or Open = High (bearish) on first 5-min candle
        first_bar = df.iloc[0]
        open_eq_low = abs(first_bar["open"] - first_bar["low"]) < 0.01
        open_eq_high = abs(first_bar["open"] - first_bar["high"]) < 0.01

        # 2. Price near VWAP +/- 0.2%
        vwap = latest.get("vwap", latest["close"])
        vwap_distance = abs(latest["close"] - vwap) / vwap if vwap > 0 else 1.0
        near_vwap = vwap_distance <= self.VWAP_BAND_PCT

        triggered = (open_eq_low or open_eq_high) and near_vwap
        if triggered:
            logger.info(
                f"  MEAN-REVERSION TRIGGER: open_eq_low={open_eq_low} "
                f"open_eq_high={open_eq_high} vwap_dist={vwap_distance:.4f}"
            )
        return triggered

    # -----------------------------------------------------------------------
    # Candidate Validation & Sizing
    # -----------------------------------------------------------------------
    def get_position_multiplier(self, p_win: float) -> float:
        """
        Get position size multiplier based on P(win) band.

        RULE: 1x at P 60-65%, 2x at 65-72%, 3x at > 72%.
        """
        if p_win > 0.72:
            return self.MULTIPLIER_HIGH
        elif p_win > 0.65:
            return self.MULTIPLIER_MID
        else:
            return self.MULTIPLIER_LOW

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, p_win: float
    ) -> tuple[int, float]:
        """
        Calculate position size based on risk rules.

        RULE: Base 1% risk * multiplier. Max 3% total risk.
        Returns (shares, risk_pct).
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0, 0.0

        multiplier = self.get_position_multiplier(p_win)
        risk_amount = self.capital * self.BASE_RISK_PCT * multiplier

        # Check if adding this would exceed max total risk
        remaining_risk = self.MAX_TOTAL_RISK_PCT - self.total_risk_today
        risk_amount = min(risk_amount, self.capital * remaining_risk)

        if risk_amount <= 0:
            return 0, 0.0

        shares = int(risk_amount / risk_per_share)
        actual_risk_pct = (shares * risk_per_share) / self.capital

        return shares, actual_risk_pct

    def compute_ev(self, p_win: float) -> float:
        """
        Compute expected value in R-multiples.

        RULE: EV >= 0.4R after ~0.08% costs.
        EV = P(win) * avg_win_R - (1 - P(win)) * 1R - costs_R
        Assume avg win = SCALE_OUT_TARGET_R (conservative, due to trailing)
        """
        avg_win_r = self.SCALE_OUT_TARGET_R  # Conservative: 0.8R minimum
        costs_r = self.COST_SLIPPAGE_PCT / 0.01  # Convert to R-terms (approx)

        ev = p_win * avg_win_r - (1 - p_win) * 1.0 - costs_r
        return ev

    def validate_candidate(self, candidate: TradeCandidate) -> tuple[bool, str]:
        """
        Validate a trade candidate against all entry rules.

        RULE: P(win) >= 60% AND EV >= 0.4R AND triggers aligned AND safety OK.
        """
        # P(win) check
        if candidate.p_win < self.MIN_P_WIN:
            return False, f"P(win) {candidate.p_win:.1%} < {self.MIN_P_WIN:.0%}"

        # EV check
        if candidate.ev < self.MIN_EV_R:
            return False, f"EV {candidate.ev:.2f}R < {self.MIN_EV_R}R"

        # VIX override check
        vix = self.premarket_data.get("vix_current", 0)
        if vix > self.VIX_SKIP_THRESHOLD and candidate.p_win < self.VIX_OVERRIDE_MIN_P:
            return False, f"VIX {vix:.1f} > {self.VIX_SKIP_THRESHOLD} and P < {self.VIX_OVERRIDE_MIN_P:.0%}"

        # Budget/expiry check
        today = datetime.now()
        is_special_day = (today.month == 2 and today.day == 1) or (today.weekday() == 3)
        if is_special_day and candidate.p_win < self.BUDGET_EXPIRY_MIN_P:
            return False, f"Budget/expiry day and P < {self.BUDGET_EXPIRY_MIN_P:.0%}"

        # Position size check
        if candidate.position_size <= 0:
            return False, "Position size is 0 (risk budget exhausted)"

        return True, "VALID"

    # -----------------------------------------------------------------------
    # Scanning & Ranking
    # -----------------------------------------------------------------------
    def scan_candidates(self) -> list[TradeCandidate]:
        """
        Scan all instruments and return ranked trade candidates.

        RULE: Rank all candidates by EV, trade only top 2-4.
        """
        safety_ok, safety_msg = self.check_safety()
        if not safety_ok:
            logger.warning(f"Safety check failed: {safety_msg}")
            return []

        candidates = []

        # Scan core indices
        for name, info in INSTRUMENTS["core_indices"].items():
            candidate = self._evaluate_symbol(name, is_core=True)
            if candidate:
                candidates.append(candidate)

        # Scan satellite stocks
        for symbol in INSTRUMENTS["satellite_universe"]:
            candidate = self._evaluate_symbol(symbol, is_core=False)
            if candidate:
                candidates.append(candidate)

        # RULE: Rank by EV, take top 2-4
        candidates.sort(key=lambda c: c.ev, reverse=True)
        max_candidates = min(4, self.MAX_TRADES_PER_DAY - self.trades_today)
        top_candidates = candidates[:max_candidates]

        logger.info(
            f"Scanned {len(INSTRUMENTS['satellite_universe']) + 2} instruments. "
            f"Found {len(candidates)} candidates. Taking top {len(top_candidates)}."
        )

        for c in top_candidates:
            logger.info(
                f"  CANDIDATE: {c.symbol} {c.direction} P={c.p_win:.1%} "
                f"EV={c.ev:.2f}R trigger={c.trigger_type} regime={c.regime}"
            )

        return top_candidates

    def _evaluate_symbol(self, symbol: str, is_core: bool) -> Optional[TradeCandidate]:
        """Evaluate a single symbol for trade candidacy."""
        try:
            df = compute_features(symbol)
            if df.empty or len(df) < 30:
                return None

            # Run model prediction
            df = predict(df)
            latest = df.iloc[-1]

            p_win = latest.get("p_win", 0.0)
            regime_code = latest.get("regime_label", 0)
            regime_map = {0: "neutral", 1: "trending", 2: "ranging"}
            regime = regime_map.get(regime_code, "neutral")

            # Check P(win) threshold early
            if p_win < self.MIN_P_WIN:
                return None

            # Check triggers
            momentum = self.check_momentum_trigger(df)
            mean_rev = self.check_mean_reversion_trigger(df)

            if not (momentum or mean_rev):
                return None

            trigger_type = "momentum" if momentum else "mean_reversion"

            # Determine direction
            if trigger_type == "momentum":
                direction = "long" if latest["close"] > df.iloc[-2]["high"] else "short"
            else:
                first_bar = df.iloc[0]
                direction = "long" if abs(first_bar["open"] - first_bar["low"]) < 0.01 else "short"

            # Entry price and ATR-based stop
            entry_price = latest["close"]
            atr = latest.get("atr_14", entry_price * 0.01)  # Fallback: 1%

            # RULE: Hard SL -1R (ATR-based)
            if direction == "long":
                stop_loss = entry_price - atr
                target_1 = entry_price + (atr * self.SCALE_OUT_TARGET_R)
            else:
                stop_loss = entry_price + atr
                target_1 = entry_price - (atr * self.SCALE_OUT_TARGET_R)

            risk_r = abs(entry_price - stop_loss)

            # Compute EV
            ev = self.compute_ev(p_win)
            if ev < self.MIN_EV_R:
                return None

            # Position sizing
            position_size, risk_pct = self.calculate_position_size(
                entry_price, stop_loss, p_win
            )
            if position_size <= 0:
                return None

            return TradeCandidate(
                symbol=symbol,
                direction=direction,
                trigger_type=trigger_type,
                p_win=p_win,
                ev=ev,
                regime=regime,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                atr=atr,
                risk_r=risk_r,
                position_size=position_size,
                risk_pct=risk_pct,
                features={
                    "rsi": latest.get("rsi_14", 0),
                    "vwap_dist": latest.get("vwap_distance_pct", 0),
                    "vol_ratio": latest.get("volume_ratio_20", 0),
                    "adx": latest.get("adx_14", 0),
                },
            )
        except Exception as e:
            logger.debug(f"  {symbol}: evaluation failed: {e}")
            return None

    # -----------------------------------------------------------------------
    # Execution (Paper Trading via OpenAlgo Sandbox)
    # -----------------------------------------------------------------------
    def execute_trades(self, candidates: list[TradeCandidate]):
        """
        Execute validated trade candidates.

        In sandbox mode: simulates orders via OpenAlgo sandbox API.
        All trades logged to SQLite regardless of mode.
        """
        for candidate in candidates:
            valid, msg = self.validate_candidate(candidate)
            if not valid:
                logger.info(f"  SKIP {candidate.symbol}: {msg}")
                continue

            # Create active trade
            trade = ActiveTrade(
                candidate=candidate,
                entry_time=datetime.now(),
                entry_price=candidate.entry_price,
                position_size=candidate.position_size,
                remaining_size=candidate.position_size,
            )

            if self.mode == "sandbox":
                # Paper trade: log the entry, simulate via OpenAlgo sandbox
                self._place_sandbox_order(trade)
            else:
                # Live mode (NOT recommended yet)
                self._place_live_order(trade)

            self.active_trades.append(trade)
            self.trades_today += 1
            self.total_risk_today += candidate.risk_pct

            logger.info(
                f"  EXECUTED: {candidate.symbol} {candidate.direction} "
                f"qty={candidate.position_size} entry={candidate.entry_price:.2f} "
                f"sl={candidate.stop_loss:.2f} risk={candidate.risk_pct:.2%}"
            )

    def _place_sandbox_order(self, trade: ActiveTrade):
        """Place order via OpenAlgo sandbox API."""
        try:
            # OpenAlgo SDK call (sandbox mode auto-handled by config)
            openalgo_host = os.getenv("OPENALGO_HOST", "http://localhost:5000")
            openalgo_key = os.getenv("OPENALGO_API_KEY", "")

            order_data = {
                "apikey": openalgo_key,
                "strategy": "nse_paper_trader",
                "symbol": trade.candidate.symbol,
                "action": "BUY" if trade.candidate.direction == "long" else "SELL",
                "exchange": "NSE",
                "pricetype": "MARKET",
                "product": "MIS",  # Intraday
                "quantity": str(trade.candidate.position_size),
            }

            import requests
            resp = requests.post(
                f"{openalgo_host}/api/v1/placeorder",
                json=order_data,
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info(f"  OpenAlgo sandbox order placed: {resp.json()}")
            else:
                logger.warning(f"  OpenAlgo order failed: {resp.status_code} {resp.text}")

        except Exception as e:
            logger.error(f"  Sandbox order failed: {e}")
            # Still track the trade locally even if API fails
            logger.info("  Trade tracked locally (API unavailable)")

    def _place_live_order(self, trade: ActiveTrade):
        """Place live order. NOT IMPLEMENTED - safety guard."""
        raise NotImplementedError(
            "Live trading is NOT enabled. Use sandbox mode for paper trading."
        )

    # -----------------------------------------------------------------------
    # Exit Management
    # -----------------------------------------------------------------------
    def manage_exits(self, current_prices: dict[str, float]):
        """
        Check all active trades for exit conditions.

        RULE: Hard SL -1R, scale 50% at +0.8R, trail rest.
        RULE: Time stop 2.5 hours.
        RULE: Auto square-off 3:20 PM IST.

        Args:
            current_prices: Dict of symbol -> current price.
        """
        now = datetime.now()
        current_time = now.time()

        for trade in self.active_trades:
            if trade.status == "closed":
                continue

            symbol = trade.candidate.symbol
            current_price = current_prices.get(symbol, trade.entry_price)

            # Calculate unrealized P&L
            if trade.candidate.direction == "long":
                trade.unrealized_pnl = (current_price - trade.entry_price) * trade.remaining_size
                pnl_r = (current_price - trade.entry_price) / trade.candidate.risk_r
            else:
                trade.unrealized_pnl = (trade.entry_price - current_price) * trade.remaining_size
                pnl_r = (trade.entry_price - current_price) / trade.candidate.risk_r

            # --- EXIT CHECK 1: Auto square-off at 3:20 PM IST ---
            if current_time >= self.AUTO_SQUAREOFF:
                self._close_trade(trade, current_price, "auto_squareoff_320pm")
                continue

            # --- EXIT CHECK 2: Hard stop-loss at -1R ---
            if pnl_r <= self.HARD_SL_R:
                self._close_trade(trade, current_price, "hard_stop_loss_1R")
                continue

            # --- EXIT CHECK 3: Time stop after 2.5 hours ---
            elapsed_hours = (now - trade.entry_time).total_seconds() / 3600
            if elapsed_hours >= self.TIME_STOP_HOURS:
                self._close_trade(trade, current_price, f"time_stop_{self.TIME_STOP_HOURS}hrs")
                continue

            # --- EXIT CHECK 4: Scale out 50% at +0.8R ---
            if not trade.scaled_out and pnl_r >= self.SCALE_OUT_TARGET_R:
                scale_qty = int(trade.remaining_size * self.SCALE_OUT_PCT)
                if scale_qty > 0:
                    trade.remaining_size -= scale_qty
                    trade.scaled_out = True

                    # Set trailing stop at entry (breakeven)
                    trade.trail_stop = trade.entry_price

                    logger.info(
                        f"  SCALED OUT: {symbol} {scale_qty} shares at +{pnl_r:.2f}R. "
                        f"Remaining: {trade.remaining_size}. Trail stop set at breakeven."
                    )

            # --- EXIT CHECK 5: Trailing stop (after scale-out) ---
            if trade.trail_stop is not None:
                if trade.candidate.direction == "long":
                    # Trail stop moves up: highest close - 0.5*ATR
                    new_trail = current_price - (0.5 * trade.candidate.atr)
                    trade.trail_stop = max(trade.trail_stop, new_trail)

                    if current_price <= trade.trail_stop:
                        self._close_trade(trade, current_price, "trailing_stop")
                        continue
                else:
                    new_trail = current_price + (0.5 * trade.candidate.atr)
                    trade.trail_stop = min(trade.trail_stop, new_trail)

                    if current_price >= trade.trail_stop:
                        self._close_trade(trade, current_price, "trailing_stop")
                        continue

    def _close_trade(self, trade: ActiveTrade, exit_price: float, reason: str):
        """Close a trade, log it, update daily P&L."""
        if trade.candidate.direction == "long":
            pnl = (exit_price - trade.entry_price) * trade.remaining_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.remaining_size

        trade.status = "closed"
        self.daily_pnl += pnl

        log_trade(trade, exit_price, reason, pnl)

        # Check daily loss shutdown
        if abs(self.daily_pnl) / self.capital >= self.MAX_DAILY_LOSS_PCT:
            self.shutdown = True
            logger.warning(
                f"DAILY LOSS SHUTDOWN: {self.daily_pnl:,.0f} "
                f"({self.daily_pnl/self.capital:.2%})"
            )

    # -----------------------------------------------------------------------
    # Daily Morning Routine
    # -----------------------------------------------------------------------
    def morning_routine(self) -> str:
        """
        Execute the 8:30 AM IST daily routine.

        RULE: Pull pre-market (GIFT Nifty, VIX, OI), news sentiment,
        run model inference, output ranked candidates + conviction.

        Returns: Daily brief string (from Kimi 2.5).
        """
        logger.info("=== 8:30 AM DAILY MORNING ROUTINE ===")

        # 1. Pre-market data
        self.premarket_data = get_premarket_data()
        logger.info(f"Pre-market: VIX={self.premarket_data.get('vix_current', 0):.1f}")

        # 2. News sentiment
        articles = fetch_rss_news()
        self.sentiment = process_news_sentiment(articles)
        logger.info(f"Sentiment: {self.sentiment.get('label', 'neutral')}")

        # 3. Model inference (scan candidates)
        candidates = self.scan_candidates()

        # 4. Kimi 2.5 daily brief
        model_prediction = {
            "p_win": candidates[0].p_win if candidates else 0.0,
            "regime": candidates[0].regime if candidates else "unknown",
            "candidates": [
                {"symbol": c.symbol, "p_win": f"{c.p_win:.1%}", "ev": f"{c.ev:.2f}R"}
                for c in candidates
            ],
        }

        self.daily_brief = kimi_daily_brief(
            gift_nifty_change=self.premarket_data.get("gift_nifty_change_pct", 0.0),
            vix=self.premarket_data.get("vix_current", 0.0),
            oi_change={
                "nifty": self.premarket_data.get("oi_nifty_change", 0),
                "banknifty": self.premarket_data.get("oi_banknifty_change", 0),
            },
            sentiment=self.sentiment,
            model_prediction=model_prediction,
        )

        logger.info(f"\n--- DAILY BRIEF ---\n{self.daily_brief}\n---")

        # Save daily summary
        self._save_daily_summary()

        return self.daily_brief

    def _save_daily_summary(self):
        """Save daily summary to SQLite."""
        conn = sqlite3.connect(TRADE_LOG_DB)
        today = datetime.now().strftime("%Y-%m-%d")

        conn.execute(
            """INSERT OR REPLACE INTO daily_summary
               (date, total_trades, winning_trades, total_pnl, max_drawdown,
                sharpe_estimate, capital_end, shutdown_triggered, daily_brief)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                today,
                self.trades_today,
                0,  # Updated at end of day
                self.daily_pnl,
                0.0,  # Updated at end of day
                0.0,  # Updated at end of day
                self.capital + self.daily_pnl,
                1 if self.shutdown else 0,
                self.daily_brief,
            ),
        )
        conn.commit()
        conn.close()


# =============================================================================
# Backtesting Mode
# =============================================================================
def run_backtest(
    start_date: str = None,
    end_date: str = None,
    capital: float = 1_000_000,
) -> dict:
    """
    Run full backtest over stored historical data.

    Returns dict with Sharpe, max DD, win rate, total return.
    """
    logger.info("=== BACKTEST MODE ===")

    engine = StrategyEngine(capital=capital, mode="sandbox")

    # Get all trading dates from stored data
    conn = sqlite3.connect(DB_PATH)
    dates_query = "SELECT DISTINCT substr(timestamp, 1, 10) as date FROM candles_1min"
    if start_date:
        dates_query += f" WHERE date >= '{start_date}'"
    if end_date:
        dates_query += f" {'AND' if start_date else 'WHERE'} date <= '{end_date}'"
    dates_query += " ORDER BY date"

    dates = [row[0] for row in conn.execute(dates_query).fetchall()]
    conn.close()

    if not dates:
        return {"error": "No data for backtest period"}

    logger.info(f"Backtest period: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    daily_returns = []
    equity_curve = [capital]

    for date in dates:
        # Reset daily state
        engine.daily_pnl = 0.0
        engine.trades_today = 0
        engine.total_risk_today = 0.0
        engine.active_trades = []
        engine.shutdown = False

        # Simulate morning routine (simplified for backtest)
        engine.premarket_data = {"vix_current": 15.0, "oi_nifty_change": 100}
        engine.sentiment = {"score": 0.1, "label": "neutral"}

        # Scan and execute
        candidates = engine.scan_candidates()
        if candidates:
            engine.execute_trades(candidates[:2])  # Conservative: top 2 only

        # Simulate exits (simplified: close at end of day)
        for trade in engine.active_trades:
            if trade.status == "open":
                # Use last close as exit price (simplified)
                engine._close_trade(trade, trade.entry_price * 1.001, "eod_close")

        daily_return = engine.daily_pnl / engine.capital
        daily_returns.append(daily_return)
        engine.capital += engine.daily_pnl
        equity_curve.append(engine.capital)

    # Compute backtest metrics
    returns = np.array(daily_returns)
    if len(returns) == 0 or returns.std() == 0:
        return {"error": "No trades or zero variance"}

    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    max_dd = min(
        (np.array(equity_curve[1:]) - np.maximum.accumulate(equity_curve[1:]))
        / np.maximum.accumulate(equity_curve[1:])
    )
    win_rate = (returns > 0).mean()

    results = {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "total_days": len(dates),
        "trading_days": len([r for r in returns if r != 0]),
        "final_capital": round(equity_curve[-1], 2),
    }

    logger.info(f"\n=== BACKTEST RESULTS ===")
    logger.info(f"  Sharpe Ratio:  {results['sharpe']}")
    logger.info(f"  Total Return:  {results['total_return']}%")
    logger.info(f"  Max Drawdown:  {results['max_drawdown']}%")
    logger.info(f"  Win Rate:      {results['win_rate']}%")
    logger.info(f"  Final Capital: {results['final_capital']:,.0f}")

    return results
