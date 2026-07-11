import json
import re
from shared_state import MarketState
from log.custom_logger import log


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
    if state["market_events"]:
        events_text = "\n".join(f"- {e}" for e in state["market_events"])
    else:
        events_text = "No special market events today."

    if state["forum_messages"]:
        forum_text = "\n".join(
            f"- Trader {m['name']}: {m['message']}"
            for m in state["forum_messages"]
        )
    else:
        forum_text = "No forum messages."

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
    try:
        return model_runner(prompt)
    except Exception as e:
        log.logger.warning(f"[MarketAnalyst] LLM call failed: {e}")
        return ""


def _parse_response(raw: str) -> dict:
    defaults = {
        "market_sentiment": "neutral",
        "market_reasoning": "Could not determine market sentiment due to parse error.",
    }
    if not raw:
        return defaults

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


def market_analyst_node(state: MarketState, model_runner) -> MarketState:
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