# NSE Intraday Paper Trading System

Self-hosted paper trading system for Indian NSE markets. Hybrid strategy: Nifty/BankNifty index options (core) + 3-5 daily stock picks (satellite). Built on OpenAlgo + OpenClaw + Kimi 2.5.

## Prerequisites

- Docker Desktop
- Python 3.11+
- Zerodha Kite Connect subscription (₹500/month) — sign up at [developers.kite.trade](https://developers.kite.trade)
- Moonshot AI (Kimi 2.5) API key — get at [platform.moonshot.ai](https://platform.moonshot.ai)

Kite Connect base plan includes: live WebSocket data, historical candles (bundled free), execution APIs (free), sandbox/paper mode.

## Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd nse-paper-trader

# 2. Configure
cp .env.example .env
# Edit .env: add MOONSHOT_API_KEY, KITE_API_KEY, KITE_API_SECRET

# 3. Generate Kite access token (daily requirement)
pip install kiteconnect python-dotenv
python scripts/generate_access_token.py

# 4. Start services
docker-compose up -d

# 5. First-time data setup (downloads 9 months of data + trains models)
pip install -r requirements.txt
python scripts/daily_start.py --setup-data

# 6. Run backtest
python scripts/daily_start.py --backtest

# 7. Start paper trading (run at 8:00 AM IST on trading days)
python scripts/daily_start.py
```

## Architecture

```
daily_start.py (orchestrator)
  ├── data_pipeline.py      → Kite historical + nsepython OI + RSS news
  ├── model_trainer.py       → Walk-forward ML (LR/RF/LGBM/XGB)
  ├── strategy_engine.py     → All rules hard-coded + OpenAlgo execution
  ├── kimi_client.py         → Kimi 2.5 for sentiment + briefs + analysis
  └── squareoff.py           → 3:20 PM safety square-off

OpenAlgo (Docker)           → Unified broker API + sandbox mode
OpenClaw (Docker)           → AI assistant + scheduled jobs + Telegram
Kimi 2.5 (Moonshot API)    → ALL LLM intelligence (no other LLM used)
```

## Strategy Rules (locked in code)

Entry: P(win) ≥ 60%, EV ≥ 0.4R after 0.08% costs. Triggers must align: regime match + momentum (5-min breakout + vol >1.8x + OI rising) or mean-reversion (Open=Low/High + near VWAP). Sizing: 1% base risk, 1-3x multiplier by conviction. Max 4 trades, 3% total risk. Exit: -1R hard stop (ATR), scale 50% at +0.8R, trail rest, 2.5hr time stop, 3:20 PM auto close. Safety: VIX >25 skip, 2% daily loss shutdown.

## Daily Schedule (IST)

| Time | Action |
|------|--------|
| 8:00 AM | `daily_start.py` — data update |
| 8:30 AM | Kimi 2.5 morning brief (GIFT Nifty, VIX, sentiment) |
| 9:15 AM | Market open — scan + execute top candidates |
| 9:15-3:15 PM | Monitor exits every 5 minutes |
| 3:20 PM | Auto square-off all positions |
| Sunday 10 AM | Weekly Kimi 2.5 review + retrain suggestion |

## Key Files

| File | Purpose |
|------|---------|
| `.env.example` | All configuration (copy to `.env`) |
| `docker-compose.yml` | OpenAlgo + OpenClaw + strategy services |
| `config/strategy_params.json` | Strategy parameters (reference only — logic is in code) |
| `config/instruments.json` | Nifty 50 universe + index tokens |
| `scripts/daily_start.py` | Main entry point |
| `scripts/kimi_client.py` | Kimi 2.5 wrapper (all LLM calls) |
| `openclaw-config/openclaw.json` | OpenClaw + Kimi 2.5 + scheduled jobs |

## Notes

- Access tokens expire daily. Run `generate_access_token.py` each morning, or use OpenAlgo's callback flow.
- All LLM calls use Kimi 2.5 via OpenAI-compatible SDK. No other LLM.
- Paper trading uses OpenAlgo sandbox mode. No real money at risk.
- Kite historical API returns max 60 days per call for 1-minute data — the pipeline loops in chunks.
- OI data comes from nsepython (bhavcopy), not Kite candles (which don't include OI).
