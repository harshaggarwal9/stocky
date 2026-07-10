"""
risk_manager_node.py
---------------------
Risk Management AI Node

Inputs  (from MarketState):
  - market_analysis, market_sentiment (from MarketAnalystNode)
  - portfolio, cash, loans
  - stock_a_price, stock_b_price
  - agent_character

Outputs (written into MarketState):
  - risk_analysis  : str  — detailed reasoning
  - risk_level     : str  — "low" | "medium" | "high"
"""

import json
import re
from shared_state import MarketState
from log.custom_logger import log


# ─── Prompt ──────────────────────────────────────────────────────────────────

RISK_MANAGER_SYSTEM = """You are a quantitative risk manager for a stock trading firm.
Your role is to assess the financial risk of an investor's current portfolio given market conditions.
Always respond with valid JSON only — no preamble, no markdown fences."""

RISK_MANAGER_PROMPT_TEMPLATE = """
Assess the investment risk for an investor given the following context.

## Market Analysis
{market_analysis}

## Investor Profile
- Character: {agent_character}

## Current Portfolio
- Stock A holdings: {stock_a} shares × ${stock_a_price}/share = ${stock_a_value:.2f}
- Stock B holdings: {stock_b} shares × ${stock_b_price}/share = ${stock_b_value:.2f}
- Cash: ${cash:.2f}
- Total assets: ${total_assets:.2f}

## Outstanding Loans
{loans_text}

## Risk Assessment Task
Classify the overall risk as "low", "medium", or "high" based on:
1. Leverage ratio (loans / total assets)
2. Portfolio concentration
3. Market sentiment alignment with character
4. Cash buffer adequacy

Respond ONLY with this JSON (no extra text):
{{"risk_level": "low"|"medium"|"high", "risk_reasoning": "<your 2-3 sentence reasoning here>"}}
"""


def _compute_loan_total(loans: list) -> float:
    return sum(loan.get("amount", 0) for loan in loans if loan.get("loan") == "yes")


def _build_prompt(state: MarketState) -> str:
    """Build risk manager prompt from MarketState."""
    portfolio = state["portfolio"]
    stock_a = portfolio.get("stock_a", 0)
    stock_b = portfolio.get("stock_b", 0)
    stock_a_price = state["stock_a_price"]
    stock_b_price = state["stock_b_price"]
    cash = state["cash"]

    stock_a_value = stock_a * stock_a_price
    stock_b_value = stock_b * stock_b_price
    total_assets = stock_a_value + stock_b_value + cash

    # Loans summary
    loans = state.get("loans", [])
    total_loans = _compute_loan_total(loans)
    if loans and total_loans > 0:
        loans_text = f"Total outstanding loan principal: ${total_loans:,.2f}\n"
        for loan in loans:
            if loan.get("loan") == "yes":
                loans_text += (
                    f"  • Amount: ${loan.get('amount', 0):,.2f}, "
                    f"Type: {loan.get('loan_type', '?')}, "
                    f"Due day: {loan.get('repayment_date', '?')}\n"
                )
    else:
        loans_text = "No outstanding loans."

    leverage = total_loans / total_assets if total_assets > 0 else 0
    loans_text += f"\nLeverage ratio: {leverage:.1%}"

    return RISK_MANAGER_PROMPT_TEMPLATE.format(
        market_analysis=state.get("market_analysis", "No analysis available."),
        agent_character=state.get("agent_character", "Balanced"),
        stock_a=stock_a,
        stock_a_price=stock_a_price,
        stock_a_value=stock_a_value,
        stock_b=stock_b,
        stock_b_price=stock_b_price,
        stock_b_value=stock_b_value,
        cash=cash,
        total_assets=total_assets,
        loans_text=loans_text,
    )


def _call_llm(prompt: str, model_runner) -> str:
    try:
        return model_runner(prompt)
    except Exception as e:
        log.logger.warning(f"[RiskManager] LLM call failed: {e}")
        return ""


def _parse_response(raw: str) -> dict:
    """Parse risk level and reasoning from LLM output."""
    defaults = {
        "risk_level": "medium",
        "risk_reasoning": "Could not determine risk level; defaulting to medium.",
    }
    if not raw:
        return defaults

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        return defaults

    try:
        parsed = json.loads(match.group())
        risk_level = parsed.get("risk_level", "medium").lower()
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"
        return {
            "risk_level": risk_level,
            "risk_reasoning": parsed.get("risk_reasoning", defaults["risk_reasoning"]),
        }
    except (json.JSONDecodeError, AttributeError):
        return defaults


# ─── LangGraph Node ──────────────────────────────────────────────────────────

def risk_manager_node(state: MarketState, model_runner) -> MarketState:
    """
    LangGraph node: Risk Management AI.

    Parameters
    ----------
    state        : MarketState — includes market_analysis from previous node
    model_runner : callable(prompt: str) -> str

    Returns
    -------
    Updated MarketState with risk_analysis and risk_level.
    """
    log.logger.info(
        f"[RiskManager] Day={state['date']} Session={state['session']} "
        f"Agent={state['agent_order']} — running risk assessment"
    )

    prompt = _build_prompt(state)
    raw_response = _call_llm(prompt, model_runner)
    result = _parse_response(raw_response)

    risk_text = (
        f"Risk Level: {result['risk_level'].upper()}. "
        f"{result['risk_reasoning']}"
    )

    log.logger.info(
        f"[RiskManager] Result -> risk_level={result['risk_level']} | "
        f"reasoning={result['risk_reasoning'][:80]}..."
    )

    return {
        **state,
        "risk_level": result["risk_level"],
        "risk_analysis": risk_text,
    }

