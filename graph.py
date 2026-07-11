from typing import Callable
from langgraph.graph import StateGraph, END

from shared_state import MarketState
from nodes.market_analyst_node import market_analyst_node
from nodes.risk_manager_node import risk_manager_node
from nodes.investor_node import investor_node
from nodes.secretary_node import secretary_node
from log.custom_logger import log


def make_market_analyst(model_runner: Callable) -> Callable:
    def _node(state: MarketState) -> MarketState:
        return market_analyst_node(state, model_runner)
    return _node


def make_risk_manager(model_runner: Callable) -> Callable:
    def _node(state: MarketState) -> MarketState:
        return risk_manager_node(state, model_runner)
    return _node


def make_investor(agent, stock_a, stock_b) -> Callable:
    def _node(state: MarketState) -> MarketState:
        return investor_node(state, agent, stock_a, stock_b)
    return _node


def make_secretary(model_runner: Callable) -> Callable:
    def _node(state: MarketState) -> MarketState:
        return secretary_node(state, model_runner)
    return _node


def route_after_secretary(state: MarketState) -> str:

    approved = state.get("secretary_decision", {}).get("approved", True)
    if approved:
        return "approved"
    else:
        return "rejected"


def rejection_override_node(state: MarketState) -> MarketState:
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


def build_pipeline(
    model_runner: Callable,
    agent,
    stock_a,
    stock_b,
) -> StateGraph:

    graph = StateGraph(MarketState)

    graph.add_node("market_analyst",  make_market_analyst(model_runner))
    graph.add_node("risk_manager",    make_risk_manager(model_runner))
    graph.add_node("investor",        make_investor(agent, stock_a, stock_b))
    graph.add_node("ai_secretary",    make_secretary(model_runner))
    graph.add_node("rejection_override", rejection_override_node)

    graph.set_entry_point("market_analyst")

    graph.add_edge("market_analyst", "risk_manager")
    graph.add_edge("risk_manager",   "investor")
    graph.add_edge("investor",       "ai_secretary")

    graph.add_conditional_edges(
        "ai_secretary",
        route_after_secretary,
        {
            "approved": END,
            "rejected": "rejection_override",
        }
    )
    graph.add_edge("rejection_override", END)

    return graph.compile()


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