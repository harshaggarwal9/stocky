"""
market_analyst_node.py
-----------------------
Market Analyst AI Node

Inputs  (from MarketState):
  - stock_a_price, stock_b_price
  - financial_reports
  - market_events
  - forum_messages
  - is_season_report_day, season_report_index

Outputs (written into MarketState):
  - market_analysis   : str  — detailed reasoning text
  - market_sentiment  : str  — "bullish" | "bearish" | "neutral"
"""

import json
import re
from shared_state import MarketState
from log.custom_logger import log


# ─── Prompt ──────────────────────────────────────────────────────────────────

MARKET_ANALYST_SYSTEM = """You are an expert stock market analyst with deep knowledge of financial markets.
Your role is to analyze current market conditions and provide a concise market assessment.
Always respond with valid JSON only — no preamble, no markdown fences."""

MARKET_ANALYST_PROMPT_TEMPLATE = """
Analyze the current stock market situation and determine the overall market sentiment.

## Market Data
- Stock A current price: ${stock_a_price}
- Stock B current price: ${stock_b_price}

## Market Events Today
{market_events}

## Trader Forum Messages (from yesterday)
{forum_messages}

{financial_section}

## Task
Based on the above information:
1. Analyze the market sentiment (bullish / bearish / neutral)
2. Provide your reasoning in 2-3 sentences

Respond ONLY with this JSON (no extra text):
{{"market_sentiment": "bullish"|"bearish"|"neutral", "market_reasoning": "<your analysis here>"}}
"""


def _build_prompt(state: MarketState) -> str:
    """Build the analyst prompt from the current MarketState."""
    # Market events section
    if state["market_events"]:
        events_text = "\n".join(f"- {e}" for e in state["market_events"])
    else:
        events_text = "No special market events today."

    # Forum messages section
    if state["forum_messages"]:
        forum_text = "\n".join(
            f"- Trader {m['name']}: {m['message']}"
            for m in state["forum_messages"]
        )
    else:
        forum_text = "No forum messages."

    # Financial reports section (only on season report days)
    if state["is_season_report_day"] and state.get("financial_reports"):
        rpt = state["financial_reports"]
        financial_section = (
            "## Quarterly Financial Reports\n"
            f"- Company A: {rpt.get('A', 'N/A')}\n"
            f"- Company B: {rpt.get('B', 'N/A')}"
        )
    else:
        financial_section = ""

    return MARKET_ANALYST_PROMPT_TEMPLATE.format(
        stock_a_price=state["stock_a_price"],
        stock_b_price=state["stock_b_price"],
        market_events=events_text,
        forum_messages=forum_text,
        financial_section=financial_section,
    )


def _call_llm(prompt: str, model_runner) -> str:
    """Call the LLM via the provided runner callable and return raw text."""
    try:
        return model_runner(prompt)
    except Exception as e:
        log.logger.warning(f"[MarketAnalyst] LLM call failed: {e}")
        return ""


def _parse_response(raw: str) -> dict:
    """
    Parse the JSON from the LLM response.
    Returns a dict with keys market_sentiment and market_reasoning,
    or safe defaults on failure.
    """
    defaults = {
        "market_sentiment": "neutral",
        "market_reasoning": "Could not determine market sentiment due to parse error.",
    }
    if not raw:
        return defaults

    # Extract first JSON object
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        return defaults

    try:
        parsed = json.loads(match.group())
        sentiment = parsed.get("market_sentiment", "neutral").lower()
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"
        return {
            "market_sentiment": sentiment,
            "market_reasoning": parsed.get("market_reasoning", defaults["market_reasoning"]),
        }
    except (json.JSONDecodeError, AttributeError):
        return defaults


# ─── LangGraph Node ──────────────────────────────────────────────────────────

def market_analyst_node(state: MarketState, model_runner) -> MarketState:
    """
    LangGraph node: Market Analyst AI.

    Parameters
    ----------
    state        : MarketState — the current shared graph state
    model_runner : callable(prompt: str) -> str — wraps your LLM call

    Returns
    -------
    Updated MarketState with market_analysis and market_sentiment filled in.
    """
    log.logger.info(
        f"[MarketAnalyst] Day={state['date']} Session={state['session']} "
        f"Agent={state['agent_order']} — running market analysis"
    )

    prompt = _build_prompt(state)
    raw_response = _call_llm(prompt, model_runner)
    result = _parse_response(raw_response)

    analysis_text = (
        f"Sentiment: {result['market_sentiment'].upper()}. "
        f"{result['market_reasoning']}"
    )

    log.logger.info(
        f"[MarketAnalyst] Result -> sentiment={result['market_sentiment']} | "
        f"reasoning={result['market_reasoning'][:80]}..."
    )

    return {
        **state,
        "market_sentiment": result["market_sentiment"],
        "market_analysis": analysis_text,
    }

