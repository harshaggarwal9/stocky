from shared_state import MarketState
from log.custom_logger import log


def _build_enriched_context(state: MarketState) -> str:
    return (
        f"\n## Market Intelligence Briefing\n"
        f"- Market Sentiment: {state.get('market_sentiment', 'neutral').upper()}\n"
        f"- Market Analysis: {state.get('market_analysis', 'No analysis available.')}\n"
        f"\n## Risk Assessment\n"
        f"- Risk Level: {state.get('risk_level', 'medium').upper()}\n"
        f"- Risk Analysis: {state.get('risk_analysis', 'No risk assessment available.')}\n"
        f"\nPlease incorporate the above intelligence into your trading decision.\n"
    )


def investor_node(state: MarketState, agent, stock_a, stock_b) -> MarketState:
    log.logger.info(
        f"[InvestorNode] Day={state['date']} Session={state['session']} "
        f"Agent={state['agent_order']} — making investment decision"
    )

    date = state["date"]
    session = state["session"]
    stock_a_deals = state["stock_a_deals"]
    stock_b_deals = state["stock_b_deals"]

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
    agent.chat_history.insert(0, context_message)
    agent.chat_history.insert(1, ack_message)

    action = agent.plan_stock(date, session, stock_a, stock_b, stock_a_deals, stock_b_deals)

    log.logger.info(
        f"[InvestorNode] Decision -> {action}"
    )

    return {
        **state,
        "investor_decision": action,
    }