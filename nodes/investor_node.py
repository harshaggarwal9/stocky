"""
investor_node.py
-----------------
Investor AI Node — wraps the existing Agent.plan_stock() logic.

This node PRESERVES the original Agent implementation completely.
It only:
  1. Injects market_analysis and risk_analysis into the prompt context
  2. Calls the existing plan_stock / plan_loan methods
  3. Writes the result into MarketState.investor_decision

The Agent class in agent.py is NOT modified.

Inputs  (from MarketState):
  - market_analysis, market_sentiment  (from MarketAnalystNode)
  - risk_analysis, risk_level          (from RiskManagerNode)
  - portfolio, cash, loans
  - stock_a_price, stock_b_price
  - stock_a_deals, stock_b_deals
  - date, session, agent_character

Outputs (written into MarketState):
  - investor_decision : dict
    e.g. {"action_type": "buy", "stock": "A", "amount": 100, "price": 30}
"""

from shared_state import MarketState
from log.custom_logger import log


# ─── Prompt injection ────────────────────────────────────────────────────────

def _build_enriched_context(state: MarketState) -> str:
    """
    Build the market + risk context string that gets prepended to the
    investor's decision prompt so the existing Agent benefits from the
    upstream AI analysis.
    """
    return (
        f"\n## Market Intelligence Briefing\n"
        f"- Market Sentiment: {state.get('market_sentiment', 'neutral').upper()}\n"
        f"- Market Analysis: {state.get('market_analysis', 'No analysis available.')}\n"
        f"\n## Risk Assessment\n"
        f"- Risk Level: {state.get('risk_level', 'medium').upper()}\n"
        f"- Risk Analysis: {state.get('risk_analysis', 'No risk assessment available.')}\n"
        f"\nPlease incorporate the above intelligence into your trading decision.\n"
    )


# ─── LangGraph Node ──────────────────────────────────────────────────────────

def investor_node(state: MarketState, agent, stock_a, stock_b) -> MarketState:
    """
    LangGraph node: Investor AI (wraps existing Agent).

    Parameters
    ----------
    state   : MarketState — includes upstream AI analysis
    agent   : Agent instance (from agent.py — unchanged)
    stock_a : Stock instance for stock A
    stock_b : Stock instance for stock B

    Returns
    -------
    Updated MarketState with investor_decision populated.
    """
    log.logger.info(
        f"[InvestorNode] Day={state['date']} Session={state['session']} "
        f"Agent={state['agent_order']} — making investment decision"
    )

    date = state["date"]
    session = state["session"]
    stock_a_deals = state["stock_a_deals"]
    stock_b_deals = state["stock_b_deals"]

    # ── Inject market + risk intelligence into the agent's prompt system ──
    # We do this by temporarily prepending context to the agent's chat_history.
    # This is a non-invasive hook that doesn't change agent.py at all.
    enriched_context = _build_enriched_context(state)
    context_message = {
        "role": "user",
        "content": (
            "Before making your stock trading decision, here is the latest "
            "market intelligence from our analysis team:\n" + enriched_context
        )
    }
    ack_message = {
        "role": "assistant",
        "content": (
            "Understood. I will incorporate the market sentiment and risk "
            "assessment into my trading decision."
        )
    }
    # Prepend context to the agent's running chat history
    agent.chat_history.insert(0, context_message)
    agent.chat_history.insert(1, ack_message)

    # ── Call the ORIGINAL plan_stock method (completely unchanged) ────────
    action = agent.plan_stock(date, session, stock_a, stock_b, stock_a_deals, stock_b_deals)

    log.logger.info(
        f"[InvestorNode] Decision -> {action}"
    )

    return {
        **state,
        "investor_decision": action,
    }

