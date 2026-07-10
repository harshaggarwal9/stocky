import json
import re
from shared_state import MarketState
from secretary import Secretary        # original rule-based secretary for format checks
from log.custom_logger import log


# ─── Prompt ──────────────────────────────────────────────────────────────────

AI_SECRETARY_SYSTEM = """You are a professional trade compliance officer at a stock brokerage.
Your role is to review investor trade orders and decide whether to approve or reject them.
You must protect the investor from imprudent decisions while allowing reasonable trades.
Always respond with valid JSON only — no preamble, no markdown fences."""

AI_SECRETARY_PROMPT_TEMPLATE = """
Review the following trade order and decide whether to approve it.

## Proposed Trade Order
- Action: {action_type}
- Stock: {stock}
- Quantity: {amount} shares
- Price: ${price}
- Total Trade Value: ${trade_value:.2f}

## Investor Financial Position
- Cash available: ${cash:.2f}
- Stock A holdings: {stock_a} shares (current price: ${stock_a_price})
- Stock B holdings: {stock_b} shares (current price: ${stock_b_price})
- Total portfolio value: ${total_portfolio:.2f}

## Risk Context
- Current Risk Level: {risk_level}
- Market Sentiment: {market_sentiment}

## Approval Guidelines
APPROVE the trade if:
- It is a "no" action (always approve)
- The trade value is ≤ 80% of available cash (for buys)
- The quantity sold is ≤ holdings (for sells) — this is a hard rule
- The trade makes sense given risk level and market sentiment

REJECT the trade if:
- It would leave the investor with less than 10% cash buffer relative to portfolio
- Risk level is "high" AND trade is aggressive (buying large quantities in bearish market)
- The investor character is "Conservative" but is making very large speculative bets

Respond ONLY with this JSON:
{{"approved": true|false, "reason": "<one sentence explanation>"}}
"""


def _build_prompt(state: MarketState) -> str:
    """Build AI secretary prompt."""
    decision = state.get("investor_decision", {"action_type": "no"})
    action_type = decision.get("action_type", "no")
    portfolio = state.get("portfolio", {})
    cash = state.get("cash", 0.0)
    stock_a_price = state["stock_a_price"]
    stock_b_price = state["stock_b_price"]
    stock_a = portfolio.get("stock_a", 0)
    stock_b = portfolio.get("stock_b", 0)

    if action_type == "no":
        stock = "-"
        amount = 0
        price = 0.0
        trade_value = 0.0
    else:
        stock = decision.get("stock", "?")
        amount = decision.get("amount", 0)
        price = decision.get("price", 0.0)
        trade_value = amount * price

    total_portfolio = (
        stock_a * stock_a_price
        + stock_b * stock_b_price
        + cash
    )

    return AI_SECRETARY_PROMPT_TEMPLATE.format(
        action_type=action_type,
        stock=stock,
        amount=amount,
        price=price,
        trade_value=trade_value,
        cash=cash,
        stock_a=stock_a,
        stock_a_price=stock_a_price,
        stock_b=stock_b,
        stock_b_price=stock_b_price,
        total_portfolio=total_portfolio,
        risk_level=state.get("risk_level", "medium"),
        market_sentiment=state.get("market_sentiment", "neutral"),
    )


def _call_llm(prompt: str, model_runner) -> str:
    try:
        return model_runner(prompt)
    except Exception as e:
        log.logger.warning(f"[AISecretary] LLM call failed: {e}")
        return ""


def _parse_approval(raw: str) -> dict:
    """Parse approval decision from LLM response."""
    defaults = {
        "approved": False,
        "reason": "Rejected because the AI secretary response could not be parsed.",
    }
    if not raw:
        return defaults

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        return defaults

    try:
        parsed = json.loads(match.group())
        approved = parsed.get("approved")
        if not isinstance(approved, bool):
            return defaults
        return {
            "approved": approved,
            "reason": parsed.get("reason", defaults["reason"]),
        }
    except (json.JSONDecodeError, AttributeError):
        return defaults


# ─── LangGraph Node ──────────────────────────────────────────────────────────

def secretary_node(state: MarketState, model_runner) -> MarketState:

    decision = state.get("investor_decision", {"action_type": "no"})
    action_type = decision.get("action_type", "no")

    # ── Fast path: "no action" is always approved ─────────────────────────
    if action_type == "no":
        sec_decision = {"approved": True, "reason": "No trade action — auto-approved."}
        log.logger.info(f"[AISecretary] Agent={state['agent_order']} -> no-action, auto-approved.")
        return {**state, "secretary_decision": sec_decision}

    log.logger.info(
        f"[AISecretary] Day={state['date']} Session={state['session']} "
        f"Agent={state['agent_order']} — reviewing trade: {decision}"
    )

    # ── Hard constraint check via original Secretary ──────────────────────
    # (ensures format and financial feasibility — these rules are inviolable)
    portfolio = state.get("portfolio", {})
    orig_secretary = Secretary.__new__(Secretary)  # lightweight instance
    orig_secretary.model = "dummy"                 # not used for check_action

    format_ok, fail_reason, _ = orig_secretary.check_action(
        resp=json.dumps(decision),
        cash=state.get("cash", 0.0),
        stock_a_amount=portfolio.get("stock_a", 0),
        stock_b_amount=portfolio.get("stock_b", 0),
        stock_a_price=state["stock_a_price"],
        stock_b_price=state["stock_b_price"],
    )

    if not format_ok:
        # Hard constraint violated — reject immediately, no LLM needed
        sec_decision = {
            "approved": False,
            "reason": f"Hard constraint violation: {fail_reason}",
        }
        log.logger.warning(
            f"[AISecretary] Agent={state['agent_order']} trade REJECTED (hard constraint): {fail_reason}"
        )
        return {**state, "secretary_decision": sec_decision}

    # ── AI review for soft/strategic approval ─────────────────────────────
    prompt = _build_prompt(state)
    raw_response = _call_llm(prompt, model_runner)
    sec_decision = _parse_approval(raw_response)

    status = "APPROVED" if sec_decision["approved"] else "REJECTED"
    log.logger.info(
        f"[AISecretary] Agent={state['agent_order']} trade {status}: {sec_decision['reason']}"
    )

    return {
        **state,
        "secretary_decision": sec_decision,
    }

