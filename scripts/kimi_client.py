"""
=============================================================================
Kimi 2.5 (Moonshot AI) Client Wrapper
=============================================================================
All LLM calls in the system go through this single module.
Uses OpenAI-compatible SDK pointed at Moonshot's endpoint.

Usage:
    from scripts.kimi_client import kimi_chat, kimi_sentiment, kimi_interpret
=============================================================================
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# RULE: Kimi 2.5 is the ONLY LLM used in this system. No OpenAI, no local
# models, no other providers. All intelligence routes through this client.
# ---------------------------------------------------------------------------

_client = None


def _get_client() -> OpenAI:
    """Lazy-init the Moonshot OpenAI-compatible client."""
    global _client
    if _client is None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY not set in .env")
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
        )
        logger.info("Kimi 2.5 client initialized (Moonshot API)")
    return _client


def kimi_chat(
    prompt: str,
    system_prompt: str = "You are a quantitative finance assistant for Indian NSE markets.",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    thinking: bool = False,
) -> str:
    """
    General-purpose Kimi 2.5 chat completion.

    Args:
        prompt: User message / question.
        system_prompt: System context.
        temperature: 0.6 for instant mode (default), 1.0 for thinking mode.
        max_tokens: Max response length.
        thinking: If True, uses thinking mode (temp 1.0, deeper reasoning).

    Returns:
        String response from Kimi 2.5.
    """
    client = _get_client()

    if thinking:
        temperature = 1.0

    kwargs = {
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Instant mode: disable thinking for faster responses
    if not thinking:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"thinking": False}
        }

    try:
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content or ""

        # Kimi K2.5 thinking mode may return content in reasoning_content
        # while the main content field is empty
        if not content.strip() and hasattr(msg, "reasoning_content"):
            content = msg.reasoning_content or ""

        return content.strip()
    except Exception as e:
        logger.error(f"Kimi 2.5 API error: {e}")
        return f"[KIMI_ERROR] {str(e)}"


def kimi_sentiment(headlines: list[str]) -> dict:
    """
    Analyze news sentiment for trading decisions using Kimi 2.5.

    Args:
        headlines: List of news headline strings with timestamps.

    Returns:
        Dict with 'score' (-1.0 to 1.0), 'label' (bullish/bearish/neutral),
        'key_themes' (list), and 'market_impact' (str).
    """
    if not headlines:
        return {"score": 0.0, "label": "neutral", "key_themes": [], "market_impact": "none"}

    headlines_text = "\n".join(f"- {h}" for h in headlines[:30])  # Cap at 30

    prompt = f"""Analyze these Indian market news headlines for trading sentiment.
Return ONLY valid JSON with these fields:
- "score": float from -1.0 (very bearish) to 1.0 (very bullish)
- "label": one of "bullish", "bearish", "neutral"
- "key_themes": list of 3-5 key market themes
- "market_impact": one of "high", "medium", "low", "none"
- "nifty_bias": one of "up", "down", "flat"
- "sectors_bullish": list of sector names
- "sectors_bearish": list of sector names

Headlines:
{headlines_text}

Return ONLY the JSON object, no markdown or explanation."""

    response = kimi_chat(prompt, temperature=0.3, max_tokens=1024)

    try:
        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Failed to parse Kimi sentiment JSON: {response[:200]}")
        return {"score": 0.0, "label": "neutral", "key_themes": [], "market_impact": "none"}


def kimi_interpret(results: dict) -> str:
    """
    Use Kimi 2.5 to interpret model training or backtest results.

    Args:
        results: Dict containing model metrics, feature importances, etc.

    Returns:
        Human-readable interpretation string.
    """
    prompt = f"""Interpret these trading model results for an Indian NSE intraday system.
Focus on: which model is best, key risk factors, overfitting signs, and actionable suggestions.

Results:
{json.dumps(results, indent=2, default=str)}

Be concise (max 300 words). Focus on actionable insights."""

    return kimi_chat(prompt, thinking=False, max_tokens=1024)


def kimi_daily_brief(
    gift_nifty_change: float,
    vix: float,
    oi_change: dict,
    sentiment: dict,
    model_prediction: dict,
) -> str:
    """
    Generate the daily 8:30 AM trading brief using Kimi 2.5.

    Args:
        gift_nifty_change: GIFT Nifty % change from prev close.
        vix: Current India VIX level.
        oi_change: Dict with Nifty/BankNifty OI changes.
        sentiment: Output from kimi_sentiment().
        model_prediction: Dict with P(win), regime, ranked candidates.

    Returns:
        Formatted daily brief string.
    """
    prompt = f"""Generate a concise daily trading brief for Indian NSE intraday trading.

Pre-Market Data:
- GIFT Nifty change: {gift_nifty_change:+.2f}%
- India VIX: {vix:.1f}
- Nifty OI change: {oi_change.get('nifty', 'N/A')}
- BankNifty OI change: {oi_change.get('banknifty', 'N/A')}

News Sentiment:
- Score: {sentiment.get('score', 0):.2f} ({sentiment.get('label', 'neutral')})
- Impact: {sentiment.get('market_impact', 'unknown')}
- Themes: {', '.join(sentiment.get('key_themes', []))}

Model Output:
- P(win): {model_prediction.get('p_win', 0):.1%}
- Regime: {model_prediction.get('regime', 'unknown')}
- Top candidates: {model_prediction.get('candidates', [])}

Format as:
1. MARKET OUTLOOK (1 sentence)
2. RISK LEVEL (low/medium/high + reason)
3. TRADE PLAN (which candidates to watch, entry conditions)
4. AVOID (what to skip today)

Be direct. No fluff."""

    return kimi_chat(prompt, temperature=0.4, max_tokens=1024)
