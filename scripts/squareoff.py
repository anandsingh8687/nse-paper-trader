"""
=============================================================================
Emergency Square-Off Script
=============================================================================
Called by OpenClaw at 3:15 PM IST as safety net.
Closes all open positions via OpenAlgo API.

RULE: Auto square-off at 3:20 PM IST. This runs at 3:15 as early warning.
=============================================================================
"""

import os
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def squareoff_all():
    """Close all open positions via OpenAlgo API."""
    host = os.getenv("OPENALGO_HOST", "http://localhost:5000")
    api_key = os.getenv("OPENALGO_API_KEY", "")

    try:
        # Get open positions
        resp = requests.post(
            f"{host}/api/v1/positionbook",
            json={"apikey": api_key},
            timeout=10,
        )

        if resp.status_code != 200:
            logger.error(f"Failed to get positions: {resp.status_code}")
            return

        positions = resp.json().get("data", [])
        open_positions = [p for p in positions if int(p.get("quantity", 0)) != 0]

        if not open_positions:
            logger.info("No open positions to square off.")
            return

        logger.warning(f"SQUARING OFF {len(open_positions)} open positions!")

        for pos in open_positions:
            qty = abs(int(pos.get("quantity", 0)))
            symbol = pos.get("symbol", "")
            action = "SELL" if int(pos.get("quantity", 0)) > 0 else "BUY"

            order = {
                "apikey": api_key,
                "strategy": "nse_paper_trader_squareoff",
                "symbol": symbol,
                "action": action,
                "exchange": pos.get("exchange", "NSE"),
                "pricetype": "MARKET",
                "product": "MIS",
                "quantity": str(qty),
            }

            resp = requests.post(
                f"{host}/api/v1/placeorder",
                json=order,
                timeout=10,
            )
            logger.info(f"  {symbol} {action} {qty}: {resp.json()}")

    except Exception as e:
        logger.error(f"Square-off failed: {e}")


if __name__ == "__main__":
    squareoff_all()
