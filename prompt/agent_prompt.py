"""
prompt/agent_prompt.py
-----------------------
All Investor Agent prompt templates — Procoder-free.

Replaces these Procoder constructs with plain Python:
  (name, content) block         → (title, content) tuple  +  build_prompt()
  Named variable (refname, name, content) → (title, content) tuple  +  ref_map dict
  Block collection                → build_prompt(*sections, ref_map=...)
    produces ## N. Title headers with double-newline separation
  format_prompt(template, inputs)   → format_prompt(str, dict)

Public API used by agent.py (same names where possible):
  build_prompt(*sections, ref_map)   — assemble a numbered prompt
  format_prompt(template, inputs)    — substitute {placeholders} from inputs
  SECTION_*                          — (title, content) tuples for each block
  LOAN_REF_MAP, FINANCIAL_REF_MAP    — pre-built cross-reference dicts
  DECIDE_BUY_STOCK_PROMPT            — pre-rendered standalone block (string)
  LOAN_RETRY_PROMPT                  — pre-rendered standalone block (string)
  BUY_STOCK_RETRY_PROMPT             — pre-rendered standalone block (string)
  POST_MESSAGE_PROMPT                — pre-rendered standalone block (string)
  NEXT_DAY_ESTIMATE_PROMPT           — pre-rendered standalone block (string)
  NEXT_DAY_ESTIMATE_RETRY            — pre-rendered standalone block (string)
"""


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_refs(content: str, ref_map: dict) -> str:
    """
    Substitute {refname} cross-references using simple str.replace.

    Why str.replace and NOT str.format_map:
      format_map (and format) consume {{ and }} immediately, turning them into
      { and }.  That breaks the JSON-example escape sequences (e.g. {{"loan":"yes"}})
      before the final format_prompt call can handle them correctly.
      str.replace only touches the exact token "{refname}" and leaves every
      other {…} and {{…}} sequence completely intact.
    """
    for refname, value in ref_map.items():
        content = content.replace("{" + refname + "}", value)
    return content


def build_prompt(*sections: tuple, ref_map: dict = None) -> str:
    """
    Replaces:
      numbered multi-section prompt with sharp2-style ## N. Title headers

    Parameters
    ----------
    *sections : (title: str, content: str)  — one tuple per named block
    ref_map   : {refname: content_str}      — resolves cross-references
                 e.g. {loan_type_prompt} inside one block is replaced with
                 the content of the Loan Type block before final formatting.

    Returns
    -------
    A numbered, double-newline-separated string:
      ## 1. Title\\ncontent\\n\\n## 2. Title\\ncontent ...
    """
    ref_map = ref_map or {}
    parts = []
    for i, (title, content) in enumerate(sections, 1):
        resolved = _resolve_refs(content, ref_map)
        parts.append(f"## {i}. {title}\n{resolved.strip()}")
    return "\n\n".join(parts)


def format_prompt(template: str, inputs: dict) -> str:
    """
    Drop-in replacement for the original format_prompt(template, inputs)

    Substitutes {key} placeholders in template with values from inputs.
    Also converts {{ → { and }} → } (Python str.format standard behaviour),
    which is how JSON examples in the prompts are safely embedded.
    """
    return template.format(**inputs)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION TUPLES  —  (title, content)
#  Each tuple is one named block from the original prompt library.
#  {placeholders} are user inputs filled by format_prompt().
#  {{…}} are literal brace characters (JSON examples) filled by format_prompt().
# ══════════════════════════════════════════════════════════════════════════════

# ── BACKGROUND ────────────────────────────────────────────────────────────────
SECTION_BACKGROUND = (
    "Background",
    "You are a stock trader, and next you will simulate interactions with other traders in the market.\n"
    "There are two stocks in the market, A and B, where B is the newly listed stock. \n"
    "Next, please complete your trading actions according to the order.",
)

# ── LAST DAY FORUM AND STOCK ──────────────────────────────────────────────────
SECTION_LASTDAY_FORUM = (
    "Last Day Forum and Stock",
    "After the close of trading yesterday, the stock prices of Company A and Company B \n"
    "were {stock_a_price} dollars per share and {stock_b_price} dollars per share, respectively. \n"
    "Posts by other traders on the forum are as follows: {lastday_forum_message}",
)

# ── LOAN TYPE  (refname: loan_type_prompt) ───────────────────────────────────
# The content is also exported as _LOAN_TYPE_BODY so it can be used as a
# cross-reference value inside SECTION_LOAN_INSTRUCTION via LOAN_REF_MAP.
_LOAN_TYPE_BODY = (
    "0. 22days, the benchmark interest rate {loan_rate1}\n"
    "1. 44days, the benchmark interest rate {loan_rate2}\n"
    "2. 66days, the benchmark interest rate {loan_rate3}"
)

SECTION_LOAN_TYPE = ("Loan Type", _LOAN_TYPE_BODY)

# ── DECIDE IF LOAN  (references {loan_type_prompt} → resolved via LOAN_REF_MAP)
SECTION_LOAN_INSTRUCTION = (
    "Instruction",
    "It is the {date} day, and your current character is {character}. \n"
    "You hold {stock_a} shares of Company A, {stock_b} shares of Company B,\n"
    "Now you have {cash} dollars in cash and {debt} in your loan situation.\n"
    "You need to decide whether to continue the loan and the amount of the loan.\n"
    "The alternative type is {loan_type_prompt}, and you should use the number to select a loan type. \n"
    "The loan amount shall not exceed {max_loan}.\n\n"
    "Return the result as json, for example:\n"
    '{{"loan": "yes", "loan_type": 3, "amount": 1000}}\n\n'
    "If no loan is required, return:\n"
    '{{"loan" : "no"}}',
)

# ── 3-YEAR HISTORICAL FINANCIAL DATA  (refname: first_day_financial_prompt) ──
# All numbers — no {placeholders}.  Inlined into SECTION_BACKGROUND_KNOWLEDGE
# via FINANCIAL_REF_MAP when build_prompt is called.
_FINANCIAL_3Y_BODY = (
    "The following lists the financial data for the past three years, "
    "covering a total of twelve quarters.\n"
    "Stock A:\n"
    "Revenue million: 3696.19, 3578.00, 3595.49, 3215.64, 3973.40, 3810.57, "
    "3840.70, 3433.02, 4344.52, 4095.22, 4114.16, 3717.96\n"
    "Net profit million: 127.711441, 217.9586418, 360.756337, 358.08228, "
    "650.8868033, 693.3022798, 433.2338757, 517.0593354, "
    "712.7358875, 628.310145, 250.5046675, 325.5147258\n"
    "Cash flow million: 30.0950631, 135.4141818, 344.3249477, 279.5563512, "
    "564.624197, 642.8122273, 350.3899245, 493.4058465, "
    "650.6526937, 579.0037013, 185.7066407, 273.1287018\n"
    "Stock B:\n"
    "Revenue million: 570.00, 774.00, 643.00, 995.00, 684.46, 934.37, "
    "782.08, 1204.05, 788.29, 1100.32, 914.96, 1418.37\n"
    "Net profit million: 85.9691, 142.086, 87.5419224, 135.7643678, "
    "132.7973368, 169.6505746, 194.9436163, 272.1084953, "
    "225.1707811, 356.7201332\n"
    "Cash flow million: 68.97, 90.171, 82.1754, 124.773, 75.4954968, "
    "123.5240842, 132.7191287, 153.7571212, 194.9436163, "
    "261.1053212, 216.3871992, 345.6568448"
)

SECTION_3Y_FINANCIAL = (
    "The last 3 years financial report of Stock A and B",
    _FINANCIAL_3Y_BODY,
)

# ── BACKGROUND KNOWLEDGE  (references {first_day_financial_prompt}) ───────────
SECTION_BACKGROUND_KNOWLEDGE = (
    "The initial financial situation of Stock A and B",
    "Company A has been listed for 10 years, deeply rooted in the chemical industry. "
    "However, the company's operations have encountered bottlenecks, with revenues declining "
    "over the past three years. Although Company A's performance has declined over the past "
    "five years, the overall trend is stable. With the recent CEO change and the exploration "
    "of new business avenues, the new CEO appears more proactive compared to the previous one. "
    "The future operational outlook is expected to improve. \n\n"
    "Company B, as a technology company, has just been listed for three years and is in a "
    "period of business growth. Last year, its revenue declined due to the overall tech "
    "environment, but the company's operations remain robust. According to the latest "
    "corporate news, it is expected that the future revenue growth rate will return to over 20%. "
    "In the short term, the stock price is expected to continue rising.\n"
    "While Company B's operations are good, there is a history of concealing critical data "
    "before its IPO, casting doubt on the reliability of its revenue. "
    "Company B recently received government inquiries regarding recent operational and stock "
    "price fluctuations, and it provided explanations while committing to allocate more "
    "resources to social services. \n\n"
    "The government recently held talks with both Company A and Company B, actively encouraging "
    "their contributions to society. Subsequently, agreements on government subsidies were "
    "signed with both companies. \n\n"
    "The last 3 years financial report of stock A and B is listed in {first_day_financial_prompt}.",
)

# ── SEASONAL FINANCIAL REPORT  (refname: seasonal_financial_report) ──────────
# {stock_a_report} and {stock_b_report} are filled by format_prompt() from inputs.
_SEASONAL_BODY = "Stock A: {stock_a_report}\nStock B: {stock_b_report}"

SECTION_SEASONAL_REPORT = (
    "The Seasonal financial report of Stock A and B",
    _SEASONAL_BODY,
)

# ── DECIDE BUY / SELL STOCK  (used standalone AND in multi-section prompts) ─────────────
_STOCK_INSTRUCTION_BODY = (
    "It is the {time} trading session on the {date} day, and after the previous session, \n"
    "the stock price of Company A is {stock_a_price} and the stock price of Company B is {stock_b_price}.\n"
    "In the current session, the buy and sell order of stock A is {stock_a_deals}, \n"
    "and the buy and sell order of stock B is {stock_b_deals}\n"
    "You currently hold {stock_a} shares of Company A, {stock_b} shares of Company B, "
    "and {cash} yuan in cash.\n"
    "You need to decide whether to buy/sell shares of Company A or Company B, "
    "and how much to buy/sell and at what price.\n"
    "You can refer to the current share price and the market to determine the price yourself, "
    "not the current share price. \n"
    "The quantity must be an integer.\n"
    "We encourage you to buy and sell more. You can only answer one json action.\n"
    "Return the result as json, for example:\n"
    '{{"action_type":"buy"|"sell", "stock":"A"|"B", amount: 100, price : 30.1}}\n'
    "If neither buy nor sell, return:\n"
    '{{"action_type" : "no"}}'
)

SECTION_STOCK_INSTRUCTION = ("Instruction", _STOCK_INSTRUCTION_BODY)

# Pre-rendered standalone version (matches original: prompt = DECIDE_BUY_STOCK_PROMPT).
# Used for time > 1 in plan_stock where no multi-section wrapper is needed.
DECIDE_BUY_STOCK_PROMPT = f"## Instruction\n{_STOCK_INSTRUCTION_BODY}"


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-REFERENCE MAPS
#  These replace Procoder's refname mechanism.  Each key is the refname string
#  that appears as {refname} inside a section's content; the value is the raw
#  content string that should be substituted in its place.
# ══════════════════════════════════════════════════════════════════════════════

# Resolves {loan_type_prompt} inside SECTION_LOAN_INSTRUCTION
LOAN_REF_MAP = {
    "loan_type_prompt": _LOAN_TYPE_BODY,
}

# Resolves {first_day_financial_prompt} inside SECTION_BACKGROUND_KNOWLEDGE
FINANCIAL_REF_MAP = {
    "first_day_financial_prompt": _FINANCIAL_3Y_BODY,
}


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE PROMPT STRINGS
#  These are used directly with format_prompt(prompt, inputs) in agent.py.
#  Each is the pre-rendered equivalent of a standalone prompt block:
#    ## Name\ncontent
# ══════════════════════════════════════════════════════════════════════════════

LOAN_RETRY_PROMPT = (
    "## Instruction\n"
    "The following questions appeared in the loan format you last answered: {fail_response}.\n"
    "You should return the results as json, for example:\n"
    '{{"loan": "yes", "loan_type": 2, "amount": 1000}}\n'
    "If no loan is required, return:\n"
    '{{"loan" : "no"}}\n'
    "Please answer again."
)

BUY_STOCK_RETRY_PROMPT = (
    "## Instruction\n"
    "The following questions appeared in the action format you last answered: {fail_response}.\n"
    "You should return the result as json, for example:\n"
    '{{"action_type":"buy"|"sell", "stock":"A"|"B", amount: 100, price: 30.1}}\n'
    "If neither buy nor sell, return:\n"
    '{{"action_type" : "no"}}\n'
    "Please answer again. You can only answer one json action."
)

POST_MESSAGE_PROMPT = (
    "## Instruction\n"
    "The current trading day is over, please briefly post your trading tips on the forum "
    "and post them on the forum.\n"
    "What you post will be publicly visible to all traders. "
    "The responses contain only what needs to be posted."
)

NEXT_DAY_ESTIMATE_PROMPT = (
    "## Instruction\n"
    "Based on the market information and forum information of the current trading day, \n"
    "please estimate whether you will buy and sell stock A and stock B tomorrow, "
    "and whether you will choose loan.\n"
    "Actions that are expected to take place are marked yes, "
    "and actions that will not take place are marked no. \n"
    "Return the result in json format, for example:\n"
    '{{"buy_A":"yes", "buy_B":"no", "sell_A":"yes", "sell_B": "no", "loan": "yes"}}'
)

NEXT_DAY_ESTIMATE_RETRY = (
    "## Instruction\n"
    "The following questions appeared in the JSON format you last answered: {fail_response}.\n"
    "Return the result in json format, for example:\n"
    '{{"buy_A":"yes", "buy_B":"no", "sell_A":"yes", "sell_B": "no", "loan": "yes"}}'
)
