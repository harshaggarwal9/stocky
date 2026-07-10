"""
shared_state.py
---------------
Defines the shared LangGraph state that flows through all agent nodes
in the Multi-AI-Agent pipeline for each investor decision cycle.

Pipeline order:
  MarketAnalystNode → RiskManagerNode → InvestorNode → SecretaryNode
  → (Stock Exchange, Order Matching, Portfolio Manager, Price Discovery, Recorder)
"""

from typing import TypedDict, Any


class MarketState(TypedDict):
    # ── Simulation context ──────────────────────────────────────────────────
    date: int                        # Current simulation day
    session: int                     # Current trading session within the day
    agent_order: int                 # Which investor agent is being processed

    # ── Market prices ────────────────────────────────────────────────────────
    stock_a_price: float
    stock_b_price: float

    # ── Investor financial state ─────────────────────────────────────────────
    portfolio: dict                  # {"stock_a": int, "stock_b": int}
    cash: float
    loans: list                      # list of loan dicts from agent.loans

    # ── Market information fed into analyst ─────────────────────────────────
    financial_reports: dict          # {"A": str, "B": str} — quarterly reports
    market_events: list              # list of event message strings
    forum_messages: list             # [{name, message}, ...] from last day

    # ── Order book snapshot (read-only for AI nodes) ─────────────────────────
    stock_a_deals: dict              # {"buy": [...], "sell": [...]}
    stock_b_deals: dict

    # ── AI node outputs ──────────────────────────────────────────────────────
    market_analysis: str             # from MarketAnalystNode
    market_sentiment: str            # "bullish" | "bearish" | "neutral"
    risk_analysis: str               # from RiskManagerNode
    risk_level: str                  # "low" | "medium" | "high"

    investor_decision: dict          # from InvestorNode
    # e.g. {"action_type": "buy", "stock": "A", "amount": 100, "price": 30}

    secretary_decision: dict         # from SecretaryNode (AI secretary)
    # e.g. {"approved": True, "reason": "..."}

    # ── Extra context passed down from main simulation ───────────────────────
    agent_character: str             # "Conservative" | "Aggressive" | ...
    is_season_report_day: bool       # whether to include financial reports today
    season_report_index: int         # which quarter report index to use
