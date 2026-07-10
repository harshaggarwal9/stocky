from typing import Callable
from langgraph.graph import StateGraph, END

from shared_state import MarketState
from nodes.market_analyst_node import market_analyst_node
from nodes.risk_manager_node import risk_manager_node
from nodes.investor_node import investor_node
from nodes.secretary_node import secretary_node
from log.custom_logger import log


# ─── Node factory functions ────────────────────────────────────────────────

def make_market_analyst(model_runner: Callable) -> Callable:
    """Return a LangGraph-compatible node for the Market Analyst."""
    def _node(state: MarketState) -> MarketState:
        return market_analyst_node(state, model_runner)
    return _node


def make_risk_manager(model_runner: Callable) -> Callable:
    """Return a LangGraph-compatible node for the Risk Manager."""
    def _node(state: MarketState) -> MarketState:
        return risk_manager_node(state, model_runner)
    return _node


def make_investor(agent, stock_a, stock_b) -> Callable:
    """Return a LangGraph-compatible node for the Investor Agent."""
    def _node(state: MarketState) -> MarketState:
        return investor_node(state, agent, stock_a, stock_b)
    return _node


def make_secretary(model_runner: Callable) -> Callable:
    """Return a LangGraph-compatible node for the AI Secretary."""
    def _node(state: MarketState) -> MarketState:
        return secretary_node(state, model_runner)
    return _node


# ─── Conditional edge ──────────────────────────────────────────────────────

def route_after_secretary(state: MarketState) -> str:

    approved = state.get("secretary_decision", {}).get("approved", True)
    if approved:
        return "approved"
    else:
        return "rejected"


def rejection_override_node(state: MarketState) -> MarketState:
    """
    If the AI Secretary rejected the trade, override investor_decision to
    a safe no-action so main.py never executes the rejected trade.
    """
    original = state.get("investor_decision", {})
    reason = state.get("secretary_decision", {}).get("reason", "Unknown rejection reason.")
    log.logger.warning(
        f"[Graph] Agent={state['agent_order']} trade OVERRIDDEN to no-action. "
        f"Reason: {reason}"
    )
    return {
        **state,
        "investor_decision": {"action_type": "no"},
    }


# ─── Graph builder ─────────────────────────────────────────────────────────

def build_pipeline(
    model_runner: Callable,
    agent,
    stock_a,
    stock_b,
) -> StateGraph:

    graph = StateGraph(MarketState)

    # ── Register nodes ──────────────────────────────────────────────────
    graph.add_node("market_analyst",  make_market_analyst(model_runner))
    graph.add_node("risk_manager",    make_risk_manager(model_runner))
    graph.add_node("investor",        make_investor(agent, stock_a, stock_b))
    graph.add_node("ai_secretary",    make_secretary(model_runner))
    graph.add_node("rejection_override", rejection_override_node)

    # ── Entry point ─────────────────────────────────────────────────────
    graph.set_entry_point("market_analyst")

    # ── Sequential edges ────────────────────────────────────────────────
    graph.add_edge("market_analyst", "risk_manager")
    graph.add_edge("risk_manager",   "investor")
    graph.add_edge("investor",       "ai_secretary")

    # ── Conditional edge after Secretary ────────────────────────────────
    graph.add_conditional_edges(
        "ai_secretary",
        route_after_secretary,
        {
            "approved": END,                 # Trade proceeds to main.py execution
            "rejected": "rejection_override", # Trade is cancelled
        }
    )
    graph.add_edge("rejection_override", END)

    return graph.compile()


# ─── Convenience runner ────────────────────────────────────────────────────

def run_pipeline(compiled_graph, initial_state: MarketState) -> MarketState:

    log.logger.info(
        f"[Graph] Running pipeline — Day={initial_state['date']} "
        f"Session={initial_state['session']} Agent={initial_state['agent_order']}"
    )
    result = compiled_graph.invoke(initial_state)
    log.logger.info(
        f"[Graph] Pipeline complete — Decision: {result.get('investor_decision')} | "
        f"Approved: {result.get('secretary_decision', {}).get('approved', True)}"
    )
    return result

if __name__ == "__main__":
    print("GRAPH COMPILED SUCCESSFULLY")
