from typing import TypedDict


class MarketState(TypedDict):
    date: int
    session: int
    agent_order: int

    stock_a_price: float
    stock_b_price: float

    portfolio: dict
    cash: float
    loans: list

    financial_reports: dict
    market_events: list
    forum_messages: list

    stock_a_deals: dict
    stock_b_deals: dict

    market_analysis: str
    market_sentiment: str
    risk_analysis: str
    risk_level: str

    investor_decision: dict

    secretary_decision: dict

    agent_character: str
    is_season_report_day: bool
    season_report_index: int