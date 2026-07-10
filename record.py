import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "res"

class TradeRecord:
    def __init__(self, date, session, stock_type, buyer, seller, quantity, price):
        self.date = date
        self.session = session
        self.stock_type = stock_type
        self.buyer = buyer
        self.seller = seller
        self.quantity = quantity
        self.price = price

    def write_to_excel(self, file_name=None):
        file_name = Path(file_name) if file_name else DATA_DIR / "trades.xlsx"
        file_name.parent.mkdir(parents=True, exist_ok=True)
        if file_name.is_file():
            existing_df = pd.read_excel(file_name)
        else:
            existing_df = pd.DataFrame(columns=[
    "Trading Day",
    "Trading Session",
    "Stock Type",
    "Buyer Trader",
    "Seller Trader",
    "Trade Quantity",
    "Trade Price"
])

        new_records = [[self.date, self.session, self.stock_type, self.buyer, self.seller, self.quantity, self.price]]
        new_df = pd.DataFrame(new_records, columns=existing_df.columns)
        all_records_df = pd.concat([existing_df, new_df], ignore_index=True)

        all_records_df.to_excel(file_name, index=False)

def create_trade_record(date, stage, stock, buy_trader, sell_trader, amount, price):
    record = TradeRecord(date, stage, stock, buy_trader, sell_trader, amount, price)
    record.write_to_excel()
    record = None


class StockRecord:
    def __init__(self, date, session, stock_a_price, stock_b_price):
        self.date = date
        self.session = session
        self.stock_a_price = stock_a_price
        self.stock_b_price = stock_b_price

    def write_to_excel(self, file_name=None):
        file_name = Path(file_name) if file_name else DATA_DIR / "stocks.xlsx"
        file_name.parent.mkdir(parents=True, exist_ok=True)
        if file_name.is_file():
            existing_df = pd.read_excel(file_name)
        else:
            existing_df = pd.DataFrame(columns=[
    "Trading Day",
    "Trading Session Number",
    "Stock A Price After Session",
    "Stock B Price After Session"
])

        new_records = [[self.date, self.session, self.stock_a_price, self.stock_b_price]]
        new_df = pd.DataFrame(new_records, columns=existing_df.columns)
        all_records_df = pd.concat([existing_df, new_df], ignore_index=True)


        all_records_df.to_excel(file_name, index=False)

def create_stock_record(date, session, stock_a_price, stock_b_price):
    record = StockRecord(date, session, stock_a_price, stock_b_price)
    record.write_to_excel()
    record = None


class AgentRecordDaily:
    def __init__(self, agent, date, loan_json):
        self.agent = agent
        self.date = date
        self.if_loan = loan_json["loan"]
        self.loan_type = 0
        self.loan_amount = 0
        if self.if_loan == "yes":
            self.loan_type = loan_json["loan_type"]
            self.loan_amount = loan_json["amount"]
        self.will_loan = "no"
        self.will_buy_a = "no"
        self.will_sell_a = "no"
        self.will_buy_b = "no"
        self.will_sell_b = "no"

    def add_estimate(self, js):
        self.will_loan = js["loan"]
        self.will_buy_a = js["buy_A"]
        self.will_sell_a = js["sell_A"]
        self.will_buy_b = js["buy_B"]
        self.will_sell_b = js["sell_B"]

    def write_to_excel(self, file_name=None):
        file_name = Path(file_name) if file_name else DATA_DIR / "agent_day_record.xlsx"
        file_name.parent.mkdir(parents=True, exist_ok=True)
        if file_name.is_file():
            existing_df = pd.read_excel(file_name)
        else:
            existing_df = pd.DataFrame(columns=[
    "Buyer Trader",
    "Trading Day",
    "Whether to Loan",
    "Loan Type",
    "Loan Amount",
    "Will Loan Tomorrow",
    "Will Buy A Tomorrow",
    "Will Sell A Tomorrow",
    "Will Buy B Tomorrow",
    "Will Sell B Tomorrow"
])


        new_records = [[self.agent, self.date, self.if_loan, self.loan_type, self.loan_amount,
                        self.will_loan, self.will_buy_a, self.will_sell_a, self.will_buy_b, self.will_sell_b]]
        new_df = pd.DataFrame(new_records, columns=existing_df.columns)
        all_records_df = pd.concat([existing_df, new_df], ignore_index=True)

        
        all_records_df.to_excel(file_name, index=False)

class AgentRecordSession:
    def __init__(self, agent, date, session, proper, cash, stock_a_value, stock_b_value, action_json):
        self.agent = agent
        self.date = date
        self.session = session
        self.proper = proper
        self.cash = cash
        self.stock_a_value = stock_a_value
        self.stock_b_value = stock_b_value
        self.action_stock = "-"
        self.amount = 0
        self.price = 0
        self.action_type = action_json["action_type"]
        if not self.action_type == "no":
            self.action_stock = action_json["stock"]
            self.amount = action_json["amount"]
            self.price = action_json["price"]

    def write_to_excel(self, file_name=None):
        file_name = Path(file_name) if file_name else DATA_DIR / "agent_session_record.xlsx"
        file_name.parent.mkdir(parents=True, exist_ok=True)
        if file_name.is_file():
            existing_df = pd.read_excel(file_name)
        else:
            existing_df = pd.DataFrame(columns=[
    "Buyer Trader",
    "Trading Day",
    "Trading Session",
    "Total Asset Before Trade",
    "Cash Before Trade",
    "Value of Stock A Before Trade",
    "Value of Stock B Before Trade",
    "Order Type",
    "Stock Category",
    "Order Quantity",
    "Order Price"
])

        new_records = [[self.agent, self.date, self.session, self.proper, self.cash,
                        self.stock_a_value, self.stock_b_value, self.action_type, self.action_stock,
                        self.amount, self.price]]
        new_df = pd.DataFrame(new_records, columns=existing_df.columns)
        all_records_df = pd.concat([existing_df, new_df], ignore_index=True)

        all_records_df.to_excel(file_name, index=False)

def create_agentses_record(agent, date, session, proper, cash, stock_a_value, stock_b_value, action_json):
    record = AgentRecordSession(agent, date, session, proper, cash, stock_a_value, stock_b_value, action_json)
    record.write_to_excel()
    record = None

