import math
import time
import openai
import random
from google import genai
from google.genai import types
import util
from log.custom_logger import log

from prompt.agent_prompt import (
    build_prompt,
    format_prompt,
    SECTION_BACKGROUND,
    SECTION_LASTDAY_FORUM,
    SECTION_LOAN_TYPE,
    SECTION_LOAN_INSTRUCTION,
    SECTION_3Y_FINANCIAL,
    SECTION_BACKGROUND_KNOWLEDGE,
    SECTION_SEASONAL_REPORT,
    SECTION_STOCK_INSTRUCTION,
    LOAN_REF_MAP,
    FINANCIAL_REF_MAP,
    DECIDE_BUY_STOCK_PROMPT,
    LOAN_RETRY_PROMPT,
    BUY_STOCK_RETRY_PROMPT,
    POST_MESSAGE_PROMPT,
    NEXT_DAY_ESTIMATE_PROMPT,
    NEXT_DAY_ESTIMATE_RETRY,
)

from secretary import Secretary
from stock import Stock
from dotenv import load_dotenv
import os

load_dotenv()


def random_init(stock_a_initial, stock_b_initial):
    stock_a, stock_b, cash, debt_amount = 0.0, 0.0, 0.0, 0.0
    while stock_a * stock_a_initial + stock_b * stock_b_initial + cash < util.MIN_INITIAL_PROPERTY \
            or stock_a * stock_a_initial + stock_b * stock_b_initial + cash > util.MAX_INITIAL_PROPERTY \
            or debt_amount > stock_a * stock_a_initial + stock_b * stock_b_initial + cash:
        stock_a = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_a_initial))
        stock_b = int(random.uniform(0, util.MAX_INITIAL_PROPERTY / stock_b_initial))
        cash = random.uniform(0, util.MAX_INITIAL_PROPERTY)
        debt_amount = random.uniform(0, util.MAX_INITIAL_PROPERTY)
    debt = {
        "loan": "yes",
        "amount": debt_amount,
        "loan_type": random.randint(0, len(util.LOAN_TYPE) - 1),
        "repayment_date": random.choice(util.REPAYMENT_DAYS)
    }
    return stock_a, stock_b, cash, debt


class Agent:
    def __init__(self, i, stock_a_price, stock_b_price, secretary, model):
        self.order = i
        self.secretary = secretary
        self.model = model
        self.character = random.choice(["Conservative", "Aggressive", "Balanced", "Growth-Oriented"])

        self.stock_a_amount, self.stock_b_amount, self.cash, init_debt = random_init(stock_a_price, stock_b_price)
        self.init_proper = self.get_total_proper(stock_a_price, stock_b_price)

        self.action_history = [[] for _ in range(util.TOTAL_DATE)]
        self.chat_history = []
        self.loans = [init_debt]
        self.is_bankrupt = False
        self.quit = False

    def run_api(self, prompt, temperature: float = 1):
        if 'gpt' in self.model:
            return self.run_api_gpt(prompt, temperature)
        elif 'gemini' in self.model:
            return self.run_api_gemini(prompt, temperature)
        log.logger.warning("Unsupported model '%s'; skipping the LLM interaction.", self.model)
        return ""

    def run_api_gemini(self, prompt, temperature: float = 1):
        if not util.GOOGLE_API_KEY:
            log.logger.error("GOOGLE_API_KEY is not configured. Skip this Gemini interaction.")
            return ""

        client = genai.Client(api_key=util.GOOGLE_API_KEY)
        max_retry = 2
        retry = 0

        while retry < max_retry:
            time.sleep(7)
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )
                if response.text:
                    return response.text
                raise ValueError("Gemini returned an empty response")
            except Exception as e:
                log.logger.warning(f"Gemini API retry... {e}")
                retry += 1
                time.sleep(15)

        log.logger.error("ERROR: GEMINI API FAILED. SKIP THIS INTERACTION.")
        return ""

    def run_api_gpt(self, prompt, temperature: float = 1):
        openai.api_key = util.OPENAI_API_KEY
        client = openai.OpenAI(api_key=openai.api_key)
        self.chat_history.append({"role": "user", "content": prompt})
        max_retry = 2
        retry = 0

        while retry < max_retry:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.chat_history,
                    temperature=temperature,
                )
                new_message_dict = {"role": response.choices[0].message.role,
                                    "content": response.choices[0].message.content}
                self.chat_history.append(new_message_dict)
                resp = response.choices[0].message.content
                return resp
            except openai.OpenAIError as e:
                log.logger.warning("OpenAI api retry...{}".format(e))
                retry += 1
                time.sleep(1)
        log.logger.error("ERROR: OPENAI API FAILED. SKIP THIS INTERACTION.")
        return ""

    def get_total_proper(self, stock_a_price, stock_b_price):
        return self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash

    def get_proper_cash_value(self, stock_a_price, stock_b_price):
        proper = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price + self.cash
        a_value = self.stock_a_amount * stock_a_price
        b_value = self.stock_b_amount * stock_b_price
        return proper, self.cash, a_value, b_value

    def get_total_loan(self):
        debt = 0
        for loan in self.loans:
            debt += loan["amount"]
        return debt

    def plan_loan(self, date, stock_a_price, stock_b_price, lastday_forum_message):
        if self.quit:
            return {"loan": "no"}

        max_loan = self.init_proper - self.get_total_loan()
        if max_loan <= 0:
            return {"loan": "no"}

        if date == 1:
            prompt = build_prompt(
                SECTION_BACKGROUND,
                SECTION_LOAN_TYPE,
                SECTION_LOAN_INSTRUCTION,
                ref_map=LOAN_REF_MAP,
            )
            inputs = {
                'date': date,
                'character': self.character,
                'stock_a': self.stock_a_amount,
                'stock_b': self.stock_b_amount,
                'cash': self.cash,
                'debt': self.loans,
                'max_loan': max_loan,
                'loan_rate1': util.LOAN_RATE[0],
                'loan_rate2': util.LOAN_RATE[1],
                'loan_rate3': util.LOAN_RATE[2],
            }
        else:
            prompt = build_prompt(
                SECTION_BACKGROUND,
                SECTION_LASTDAY_FORUM,
                SECTION_LOAN_TYPE,
                SECTION_LOAN_INSTRUCTION,
                ref_map=LOAN_REF_MAP,
            )
            inputs = {
                "date": date,
                "character": self.character,
                "stock_a": self.stock_a_amount,
                "stock_b": self.stock_b_amount,
                "cash": self.cash,
                "debt": self.loans,
                "max_loan": max_loan,
                "stock_a_price": stock_a_price,
                "stock_b_price": stock_b_price,
                "lastday_forum_message": lastday_forum_message,
                'loan_rate1': util.LOAN_RATE[0],
                'loan_rate2': util.LOAN_RATE[1],
                'loan_rate3': util.LOAN_RATE[2],
            }

        try_times = 0
        MAX_TRY_TIMES = 3
        resp = self.run_api(format_prompt(prompt, inputs))

        if resp == "":
            return {"loan": "no"}

        loan_format_check, fail_response, loan = self.secretary.check_loan(resp, max_loan)
        while not loan_format_check:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Loan format try times > MAX_TRY_TIMES. Skip as no loan today.")
                loan = {"loan": "no"}
                break

            resp = self.run_api(format_prompt(LOAN_RETRY_PROMPT, {"fail_response": fail_response}))
            if resp == "":
                return {"loan": "no"}
            loan_format_check, fail_response, loan = self.secretary.check_loan(resp, max_loan)

        if loan["loan"] == "yes":
            loan["repayment_date"] = date + util.LOAN_TYPE_DATE[loan["loan_type"]]
            self.loans.append(loan)
            self.cash += loan["amount"]
            log.logger.info("INFO: Agent {} decide to loan: {}".format(self.order, loan))
        else:
            log.logger.info("INFO: Agent {} decide not to loan".format(self.order))
        return loan

    def plan_stock(self, date, time, stock_a, stock_b, stock_a_deals, stock_b_deals):
        if self.quit:
            return {"action_type": "no"}

        if date in util.SEASON_REPORT_DAYS and time == 1:
            index = util.SEASON_REPORT_DAYS.index(date)
            prompt = build_prompt(
                SECTION_3Y_FINANCIAL,
                SECTION_BACKGROUND_KNOWLEDGE,
                SECTION_SEASONAL_REPORT,
                SECTION_STOCK_INSTRUCTION,
                ref_map=FINANCIAL_REF_MAP,
            )
            inputs = {
                "date": date,
                "time": time,
                "stock_a": self.stock_a_amount,
                "stock_b": self.stock_b_amount,
                "stock_a_price": stock_a.get_price(),
                "stock_b_price": stock_b.get_price(),
                "stock_a_deals": stock_a_deals,
                "stock_b_deals": stock_b_deals,
                "cash": self.cash,
                "stock_a_report": stock_a.gen_financial_report(index),
                "stock_b_report": stock_b.gen_financial_report(index),
            }

        elif time == 1:
            prompt = build_prompt(
                SECTION_3Y_FINANCIAL,
                SECTION_BACKGROUND_KNOWLEDGE,
                SECTION_STOCK_INSTRUCTION,
                ref_map=FINANCIAL_REF_MAP,
            )
            inputs = {
                "date": date,
                "time": time,
                "stock_a": self.stock_a_amount,
                "stock_b": self.stock_b_amount,
                "stock_a_price": stock_a.get_price(),
                "stock_b_price": stock_b.get_price(),
                "stock_a_deals": stock_a_deals,
                "stock_b_deals": stock_b_deals,
                "cash": self.cash,
            }

        else:
            prompt = DECIDE_BUY_STOCK_PROMPT
            inputs = {
                "date": date,
                "time": time,
                "stock_a": self.stock_a_amount,
                "stock_b": self.stock_b_amount,
                "stock_a_price": stock_a.get_price(),
                "stock_b_price": stock_b.get_price(),
                "stock_a_deals": stock_a_deals,
                "stock_b_deals": stock_b_deals,
                "cash": self.cash,
            }

        try_times = 0
        MAX_TRY_TIMES = 3
        resp = self.run_api(format_prompt(prompt, inputs))

        if resp == "":
            return {"action_type": "no"}

        action_format_check, fail_response, action = self.secretary.check_action(
            resp, self.cash, self.stock_a_amount, self.stock_b_amount,
            stock_a.get_price(), stock_b.get_price())

        while not action_format_check:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Action format try times > MAX_TRY_TIMES. Skip as no loan today.")
                action = {"action_type": "no"}
                break

            resp = self.run_api(format_prompt(BUY_STOCK_RETRY_PROMPT, {"fail_response": fail_response}))
            if resp == "":
                return {"action_type": "no"}
            action_format_check, fail_response, action = self.secretary.check_action(
                resp, self.cash, self.stock_a_amount, self.stock_b_amount,
                stock_a.get_price(), stock_b.get_price())

        if action["action_type"] == "buy":
            log.logger.info("INFO: Agent {} decide to action: {}".format(self.order, action))
            return action
        elif action["action_type"] == "sell":
            log.logger.info("INFO: Agent {} decide to action: {}".format(self.order, action))
            return action
        elif action["action_type"] == "no":
            log.logger.info("INFO: Agent {} decide not to action".format(self.order))
            return action

        log.logger.error("ERROR: WRONG ACTION: {}".format(action))
        return {"action_type": "no"}

    def buy_stock(self, stock_name, price, amount):
        if self.quit:
            return False
        if self.cash < price * amount or stock_name not in ['A', 'B']:
            log.logger.warning("ILLEGAL STOCK BUY BEHAVIOR: remain cash {}".format(self.cash))
            return False
        self.cash -= price * amount
        if stock_name == 'A':
            self.stock_a_amount += amount
        elif stock_name == 'B':
            self.stock_b_amount += amount
        return True

    def sell_stock(self, stock_name, price, amount):
        if self.quit:
            return False
        if stock_name == 'B' and self.stock_b_amount < amount:
            log.logger.warning("ILLEGAL STOCK SELL BEHAVIOR: remain stock_b {}, amount {}".format(
                self.stock_b_amount, amount))
            return False
        elif stock_name == 'A' and self.stock_a_amount < amount:
            log.logger.warning("ILLEGAL STOCK SELL BEHAVIOR: remain stock_a {}, amount {}".format(
                self.stock_a_amount, amount))
            return False
        if stock_name == 'A':
            self.stock_a_amount -= amount
        elif stock_name == 'B':
            self.stock_b_amount -= amount
        self.cash += price * amount
        return True

    def loan_repayment(self, date):
        if self.quit:
            return
        for loan in self.loans[:]:
            if loan["repayment_date"] == date:
                self.cash -= loan["amount"] * (1 + util.LOAN_RATE[loan["loan_type"]])
                self.loans.remove(loan)
        if self.cash < 0:
            self.is_bankrupt = True

    def interest_payment(self):
        if self.quit:
            return
        for loan in self.loans:
            self.cash -= loan["amount"] * util.LOAN_RATE[loan["loan_type"]] / 12
            if self.cash < 0:
                self.is_bankrupt = True

    def bankrupt_process(self, stock_a_price, stock_b_price):
        if self.quit:
            return False
        total_value_of_stock = self.stock_a_amount * stock_a_price + self.stock_b_amount * stock_b_price
        if total_value_of_stock + self.cash < 0:
            log.logger.warning(f"Agent {self.order} bankrupt. ")
            return True
        if stock_a_price * self.stock_a_amount >= -self.cash:
            sell_a = math.ceil(-self.cash / stock_a_price)
            self.stock_a_amount -= sell_a
            self.cash += sell_a * stock_a_price
        else:
            self.cash += stock_a_price * self.stock_a_amount
            self.stock_a_amount = 0
            sell_b = math.ceil(-self.cash / stock_b_price)
            self.stock_b_amount -= sell_b
            self.cash += sell_b * stock_b_price

        if self.stock_a_amount < 0 or self.stock_b_amount < 0 or self.cash < 0:
            raise RuntimeError("ERROR: WRONG BANKRUPT PROCESS")
        self.is_bankrupt = False
        return False

    def post_message(self):
        if self.quit:
            return ""
        prompt = format_prompt(POST_MESSAGE_PROMPT, inputs={})
        resp = self.run_api(prompt)
        return resp

    def next_day_estimate(self):
        if self.quit:
            return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
        prompt = format_prompt(NEXT_DAY_ESTIMATE_PROMPT, inputs={})
        resp = self.run_api(prompt)
        if resp == "":
            return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
        format_check, fail_response, estimate = self.secretary.check_estimate(resp)
        try_times = 0
        MAX_TRY_TIMES = 3
        while not format_check:
            try_times += 1
            if try_times > MAX_TRY_TIMES:
                log.logger.warning("WARNING: Estimation format try times > MAX_TRY_TIMES. Skip as all 'no' today.")
                estimate = {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
                break
            resp = self.run_api(format_prompt(NEXT_DAY_ESTIMATE_RETRY, {"fail_response": fail_response}))
            if resp == "":
                return {"buy_A": "no", "buy_B": "no", "sell_A": "no", "sell_B": "no", "loan": "no"}
            format_check, fail_response, estimate = self.secretary.check_estimate(resp)
        return estimate