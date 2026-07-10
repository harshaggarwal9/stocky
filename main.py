"""
main.py  (Quantix LangGraph Edition)
--------------------------------------
Simulation Controller — migrated to Multi-AI-Agent architecture via LangGraph.

Architecture:
  Shared Market State
       │
       ▼
  MarketAnalystNode (AI)    ← financial reports, prices, events, forum
       │
       ▼
  RiskManagerNode (AI)      ← portfolio, loans, market analysis
       │
       ▼
  InvestorNode (AI)         ← existing Agent + market/risk context
       │
       ▼
  AISecretaryNode (AI)      ← replaces rule-based Secretary
       │
  [approved/rejected?]
       │
       ▼
  Stock Exchange            ← UNCHANGED from original
  Order Matching            ← UNCHANGED
  Portfolio Manager         ← UNCHANGED
  Price Discovery           ← UNCHANGED
  Recorder                  ← UNCHANGED

PRESERVED UNCHANGED: stock.py, record.py, util.py
"""

import argparse
import random
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

import util
from agent import Agent
from stock import Stock
from log.custom_logger import log
from record import (
    create_stock_record,
    create_trade_record,
    AgentRecordDaily,
    create_agentses_record,
)

# ── NEW: LangGraph imports ──────────────────────────────────────────────────
from shared_state import MarketState
from graph import build_pipeline, run_pipeline

load_dotenv()


# ════════════════════════════════════════════════════════════════════════════
#  MODEL RUNNER — unified LLM interface for all AI nodes
# ════════════════════════════════════════════════════════════════════════════

def make_model_runner(model: str):
    """
    Returns a callable(prompt: str) -> str that routes to the correct LLM.

    All four AI nodes (Market Analyst, Risk Manager, Investor context,
    AI Secretary) share this single runner.  The existing Agent class
    has its own runner for its internal calls.
    """
    def run(prompt: str) -> str:
        if "gemini" in model:
            return _run_gemini(prompt, model)
        if "gpt" in model:
            return _run_gpt(prompt, model)
        elif "gemini" in model:
            return _run_gemini_stub(prompt)   # stub — swap with real Gemini call
        else:
            return _offline_gemini_response(prompt)

    return run


def _run_gpt(prompt: str, model: str) -> str:
    import openai
    if not util.OPENAI_API_KEY:
        log.logger.warning("[ModelRunner] OPENAI_API_KEY is not configured; using a safe no-trade response.")
        return '{"action_type": "no"}'

    client = openai.OpenAI(api_key=util.OPENAI_API_KEY)
    max_retry = 2
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.logger.warning(f"[ModelRunner] GPT retry {attempt + 1}: {e}")
            time.sleep(1)
    return ""


def _run_gemini(prompt: str, model: str) -> str:
    """Run Gemini and return the JSON text expected by the graph nodes."""
    if not util.GOOGLE_API_KEY:
        log.logger.error("[ModelRunner] GOOGLE_API_KEY is not configured.")
        return ""

    client = genai.Client(api_key=util.GOOGLE_API_KEY)
    max_retry = 2
    for attempt in range(max_retry):
        time.sleep(7)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                ),
            )
            if response.text:
                return response.text
            raise ValueError("Gemini returned an empty response")
        except Exception as e:
            log.logger.warning(f"[ModelRunner] Gemini retry {attempt + 1}: {e}")
            time.sleep(15)
    return ""


def _offline_gemini_response(prompt: str) -> str:
    """
    Stub Gemini runner — returns deterministic JSON responses for each node.
    Replace with real Google Gemini API call when deploying.
    """
    p = prompt.lower()

    if "market_sentiment" in p or "market analyst" in p or "market intelligence" in p:
        # Market Analyst response
        return '{"market_sentiment": "neutral", "market_reasoning": "Market conditions are stable with no strong directional signals. Forum messages show mixed sentiment among traders."}'

    if "risk_level" in p or "risk assessment" in p or "leverage ratio" in p:
        # Risk Manager response
        return '{"risk_level": "medium", "risk_reasoning": "Portfolio is moderately leveraged. Current market conditions do not suggest extreme risk."}'

    if "approved" in p or "trade compliance" in p or "approval guidelines" in p:
        # AI Secretary response
        return '{"approved": true, "reason": "Trade is within acceptable parameters given current risk profile."}'

    # Investor Agent response. A no-trade decision is intentionally used as
    # the offline fallback: it conforms to the order schema and cannot create
    # an invalid or unaffordable order when no LLM provider is configured.
    return '{"action_type": "no"}'


# ════════════════════════════════════════════════════════════════════════════
#  ORDER MATCHING — preserved exactly from original main.py
# ════════════════════════════════════════════════════════════════════════════

def get_agent(all_agents, order):
    for agent in all_agents:
        if agent.order == order:
            return agent
    return None


def handle_action(action, stock_deals, all_agents, stock, session):
    """Identical to original main.py handle_action — zero changes."""
    try:
        if action["action_type"] == "buy":
            for sell_action in stock_deals["sell"][:]:
                if action["price"] == sell_action["price"]:
                    close_amount = min(action["amount"], sell_action["amount"])
                    get_agent(all_agents, action["agent"]).buy_stock(stock.name, action["price"], close_amount)
                    if not sell_action["agent"] == -1:
                        get_agent(all_agents, sell_action["agent"]).sell_stock(stock.name, action["price"], close_amount)
                    stock.add_session_deal({"price": action["price"], "amount": close_amount})
                    create_trade_record(action["date"], session, stock.name, action["agent"],
                                        sell_action["agent"], close_amount, action["price"])
                    if action["amount"] > close_amount:
                        log.logger.info(
                            f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                            f"STOCK:{stock.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                        )
                        stock_deals["sell"].remove(sell_action)
                        action["amount"] -= close_amount
                    else:
                        log.logger.info(
                            f"ACTION - BUY:{action['agent']}, SELL:{sell_action['agent']}, "
                            f"STOCK:{stock.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                        )
                        sell_action["amount"] -= close_amount
                        return
            stock_deals["buy"].append(action)

        else:
            for buy_action in stock_deals["buy"][:]:
                if action["price"] == buy_action["price"]:
                    close_amount = min(action["amount"], buy_action["amount"])
                    get_agent(all_agents, action["agent"]).sell_stock(stock.name, action["price"], close_amount)
                    get_agent(all_agents, buy_action["agent"]).buy_stock(stock.name, action["price"], close_amount)
                    stock.add_session_deal({"price": action["price"], "amount": close_amount})
                    create_trade_record(action["date"], session, stock.name, buy_action["agent"],
                                        action["agent"], close_amount, action["price"])
                    if action["amount"] > close_amount:
                        log.logger.info(
                            f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                            f"STOCK:{stock.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                        )
                        stock_deals["buy"].remove(buy_action)
                        action["amount"] -= close_amount
                    else:
                        log.logger.info(
                            f"ACTION - BUY:{buy_action['agent']}, SELL:{action['agent']}, "
                            f"STOCK:{stock.name}, PRICE:{action['price']}, AMOUNT:{close_amount}"
                        )
                        buy_action["amount"] -= close_amount
                        return
            stock_deals["sell"].append(action)

    except Exception as e:
        log.logger.error(f"handle_action error: {e}")
        return


# ════════════════════════════════════════════════════════════════════════════
#  STATE BUILDER — creates the initial MarketState for each agent per session
# ════════════════════════════════════════════════════════════════════════════

def build_initial_state(
    agent: Agent,
    date: int,
    session: int,
    stock_a: Stock,
    stock_b: Stock,
    stock_a_deals: dict,
    stock_b_deals: dict,
    last_day_forum_message: list,
    current_market_events: list,
) -> MarketState:
    """
    Construct the MarketState dict that seeds the LangGraph pipeline
    for a single agent × session invocation.
    """
    is_season_report = date in util.SEASON_REPORT_DAYS
    season_idx = util.SEASON_REPORT_DAYS.index(date) if is_season_report else 0

    financial_reports = {}
    if is_season_report:
        financial_reports = {
            "A": stock_a.gen_financial_report(season_idx),
            "B": stock_b.gen_financial_report(season_idx),
        }

    return MarketState(
        date=date,
        session=session,
        agent_order=agent.order,
        stock_a_price=stock_a.get_price(),
        stock_b_price=stock_b.get_price(),
        portfolio={
            "stock_a": agent.stock_a_amount,
            "stock_b": agent.stock_b_amount,
        },
        cash=agent.cash,
        loans=agent.loans,
        financial_reports=financial_reports,
        market_events=current_market_events,
        forum_messages=last_day_forum_message,
        stock_a_deals=dict(stock_a_deals),   # snapshot; pipeline is read-only
        stock_b_deals=dict(stock_b_deals),
        market_analysis="",
        market_sentiment="neutral",
        risk_analysis="",
        risk_level="medium",
        investor_decision={},
        secretary_decision={},
        agent_character=agent.character,
        is_season_report_day=is_season_report,
        season_report_index=season_idx,
    )


# ════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def simulation(args):
    # ── Init (same as original) ─────────────────────────────────────────
    from secretary import Secretary   # kept for agent.__init__ compatibility
    secretary = Secretary(args.model)

    stock_a = Stock("A", util.STOCK_A_INITIAL_PRICE, 0, is_new=False)
    stock_b = Stock("B", util.STOCK_B_INITIAL_PRICE, 0, is_new=False)

    all_agents = []
    log.logger.debug("Agents initial...")
    for i in range(util.AGENTS_NUM):
        agent = Agent(i, stock_a.get_price(), stock_b.get_price(), secretary, args.model)
        all_agents.append(agent)
        log.logger.debug(
            f"cash: {agent.cash}, stock a: {agent.stock_a_amount}, "
            f"stock b: {agent.stock_b_amount}, debt: {agent.loans}"
        )

    # ── Create shared model runner for the AI nodes ─────────────────────
    model_runner = make_model_runner(args.model)

    last_day_forum_message = []
    stock_a_deals = {"sell": [], "buy": []}
    stock_b_deals = {"sell": [], "buy": []}

    log.logger.debug("--------Simulation Start! (LangGraph Edition)--------")

    for date in range(1, util.TOTAL_DATE + 1):
        log.logger.debug(f"--------DAY {date}---------")

        stock_a_deals["sell"].clear()
        stock_a_deals["buy"].clear()
        stock_b_deals["buy"].clear()
        stock_b_deals["sell"].clear()

        # ── Daily pre-session bookkeeping (unchanged) ──────────────────
        for agent in all_agents[:]:
            agent.chat_history.clear()
            agent.loan_repayment(date)

        if date in util.REPAYMENT_DAYS:
            for agent in all_agents[:]:
                agent.interest_payment()

        for agent in all_agents[:]:
            if agent.is_bankrupt:
                quit_sig = agent.bankrupt_process(stock_a.get_price(), stock_b.get_price())
                if quit_sig:
                    agent.quit = True
                    all_agents.remove(agent)

        # ── Market events (unchanged) ──────────────────────────────────
        current_market_events = []
        if date == util.EVENT_1_DAY:
            util.LOAN_RATE = util.EVENT_1_LOAN_RATE
            last_day_forum_message.append({"name": -1, "message": util.EVENT_1_MESSAGE})
            current_market_events.append(util.EVENT_1_MESSAGE)
        if date == util.EVENT_2_DAY:
            util.LOAN_RATE = util.EVENT_2_LOAN_RATE
            last_day_forum_message.append({"name": -1, "message": util.EVENT_2_MESSAGE})
            current_market_events.append(util.EVENT_2_MESSAGE)

        # ── Loan planning (uses existing Agent.plan_loan — unchanged) ──
        daily_agent_records = []
        for agent in all_agents:
            loan = agent.plan_loan(date, stock_a.get_price(), stock_b.get_price(), last_day_forum_message)
            daily_agent_records.append(AgentRecordDaily(date, agent.order, loan))

        # ── Trading sessions ───────────────────────────────────────────
        for session in range(1, util.TOTAL_SESSION + 1):
            log.logger.debug(f"SESSION {session}")
            sequence = list(range(len(all_agents)))
            random.shuffle(sequence)

            for i in sequence:
                agent = all_agents[i]

                # ── BUILD LangGraph pipeline for this agent ────────────
                # Pipeline is rebuilt each invocation so per-agent
                # state (stock objects) is correctly captured.
                pipeline = build_pipeline(
                    model_runner=model_runner,
                    agent=agent,
                    stock_a=stock_a,
                    stock_b=stock_b,
                )

                # ── BUILD initial state ────────────────────────────────
                initial_state = build_initial_state(
                    agent=agent,
                    date=date,
                    session=session,
                    stock_a=stock_a,
                    stock_b=stock_b,
                    stock_a_deals=stock_a_deals,
                    stock_b_deals=stock_b_deals,
                    last_day_forum_message=last_day_forum_message,
                    current_market_events=current_market_events,
                )

                # ══ RUN the LangGraph pipeline ═════════════════════════
                #   MarketAnalyst → RiskManager → Investor → AISecretary
                final_state = run_pipeline(pipeline, initial_state)
                # ══════════════════════════════════════════════════════

                # ── Extract the (possibly overridden) action ───────────
                action = final_state["investor_decision"]

                # ── Record agent session stats (unchanged) ─────────────
                proper, cash, valua_a, value_b = agent.get_proper_cash_value(
                    stock_a.get_price(), stock_b.get_price()
                )
                create_agentses_record(
                    agent.order, date, session, proper, cash, valua_a, value_b, action
                )

                # ── Submit to Stock Exchange (unchanged) ───────────────
                action["agent"] = agent.order
                action["date"] = date
                if action["action_type"] != "no":
                    if action.get("stock") == "A":
                        handle_action(action, stock_a_deals, all_agents, stock_a, session)
                    else:
                        handle_action(action, stock_b_deals, all_agents, stock_b, session)

            # ── Price Discovery (unchanged) ────────────────────────────
            stock_a.update_price(date)
            stock_b.update_price(date)
            create_stock_record(date, session, stock_a.get_price(), stock_b.get_price())

        # ── End-of-day estimation (unchanged) ─────────────────────────
        for idx, agent in enumerate(all_agents):
            estimation = agent.next_day_estimate()
            log.logger.info(f"Agent {agent.order} tomorrow estimation: {estimation}")
            if idx >= len(daily_agent_records):
                break
            daily_agent_records[idx].add_estimate(estimation)
            daily_agent_records[idx].write_to_excel()
        daily_agent_records.clear()

        # ── Forum messages (unchanged) ─────────────────────────────────
        last_day_forum_message.clear()
        log.logger.debug(f"DAY {date} ends, display forum messages...")
        for agent in all_agents:
            message = agent.post_message()
            log.logger.info(f"Agent {agent.order} says: {message}")
            last_day_forum_message.append({"name": agent.order, "message": message})

    log.logger.debug("--------Simulation finished! (LangGraph Edition)--------")


# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantix LangGraph Multi-Agent Simulation")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite`", help="LLM model name")
    args = parser.parse_args()
    simulation(args)
