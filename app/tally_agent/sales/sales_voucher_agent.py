"""
sales_voucher_agent.py
LangGraph multi-node agent for Tally Sales Voucher creation.

Pipeline:
  START
    → [N1] Resolve Customer   (LLM + tally.get_all_ledgers_list, retry ≤4)
    → [N2] Parse Date         (LLM + format validation,            retry ≤4)
    → [N3] Resolve Items      (LLM + tally.get_all_ledgers_list,  retry ≤4)
    → [N4] Prepare XML        (LLM + Tally XML template)
    → [N5] Execute & Recover  (tally.execute_xml_query + LLM error diagnosis)
  END  |  FAILED
"""

import json
import re
from datetime import datetime
from typing import Any, List, Literal, Optional

# from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rapidfuzz import fuzz, process

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

MAX_RETRIES = 4

# Sample Tally XML template shown to the LLM in Node 4
TALLY_SALES_XML_TEMPLATE = """
<ENVELOPE>
  <HEADER>
    <VERSION>1</VERSION>
    <TALLYREQUEST>Import</TALLYREQUEST>
    <TYPE>Data</TYPE>
    <ID>Vouchers</ID>
  </HEADER>
  <BODY>
    <DESC><STATICVARIABLES/></DESC>
    <REQUESTDESC>
      <REPORTNAME>Vouchers</REPORTNAME>
    </REQUESTDESC>
    <REQUESTDATA>
      <TALLYMESSAGE xmlns:UDF="TallyUDF">
          <VOUCHER VCHTYPE="Sales" ACTION="Create">
          <DATE>{DATE}</DATE>
          <VOUCHERTYPENAME>Sales</VOUCHERTYPENAME>
          <PARTYLEDGERNAME>{PARTY_NAME}</PARTYLEDGERNAME>
          <VOUCHERNUMBER>ARH/25-26/0654</VOUCHERNUMBER>
                                            
          <PERSISTEDVIEW>Invoice Voucher View</PERSISTEDVIEW>
          <ISINVOICE>Yes</ISINVOICE>
          <FBTPAYMENTTYPE>Default</FBTPAYMENTTYPE>

          <!-- Debit the party (receivable) -->
          <LEDGERENTRIES.LIST>
              <LEDGERNAME>{PARTY_NAME}</LEDGERNAME>
              <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
              <AMOUNT>-{TOTAL_AMOUNT}</AMOUNT>
          </LEDGERENTRIES.LIST>

          <!-- One block per inventory item -->
          <ALLINVENTORYENTRIES.LIST>
              <STOCKITEMNAME>{ITEM_NAME}</STOCKITEMNAME>
              <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
              <RATE>{RATE}</RATE>
              <AMOUNT>{ITEM_AMOUNT}</AMOUNT>
              <ACTUALQTY>{QTY}</ACTUALQTY>
              <BILLEDQTY>{QTY}</BILLEDQTY>

              <!-- Credit the sales ledger -->
              <ACCOUNTINGALLOCATIONS.LIST>
                  <LEDGERNAME>Sales</LEDGERNAME>
                  <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
                  <AMOUNT>{TOTAL_AMOUNT}</AMOUNT>
              </ACCOUNTINGALLOCATIONS.LIST>
          </ALLINVENTORYENTRIES.LIST>

          </VOUCHER>
      </TALLYMESSAGE>
    </REQUESTDATA>
  </BODY>
</ENVELOPE>
""".strip()


# ──────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────
llm_client: Any

class SalesVoucherState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────
    user_input: str
    llm_client: Any

    # ── Node 1: Customer ───────────────────────────────────
    customer_name: Optional[str]          # matched Tally ledger name
    customer_candidates: List[str]        # LLM-extracted names across retries
    customer_confidence: str              # "pending" | "confirmed" | "failed"
    customer_retry_count: int

    # ── Node 2: Date ───────────────────────────────────────
    date_str: Optional[str]               # YYYYMMDD
    date_confidence: str                  # "pending" | "confirmed"
    date_retry_count: int

    # ── Node 3: Items ──────────────────────────────────────
    items: Optional[List[dict]]           # [{extracted_name, tally_name, qty, rate}]
    item_confidence: str                  # "pending" | "confirmed" | "failed"
    item_retry_count: int

    # ── Node 4: XML ────────────────────────────────────────
    xml_payload: Optional[str]
    xml_retry_count: int

    # ── Node 5: Execution ──────────────────────────────────
    execution_result: Optional[dict]
    execution_error: Optional[str]
    error_retry_count: int

    # ── Overall ────────────────────────────────────────────
    status: str                           # "running" | "success" | "error:<cause>" | "failed"
    failed_reason: Optional[str]


# ──────────────────────────────────────────────────────────
# NODE 1 – CUSTOMER RESOLUTION
# ──────────────────────────────────────────────────────────

def node_resolve_customer(state: SalesVoucherState, tally) -> SalesVoucherState:
    retry         = state.get("customer_retry_count", 0)
    candidates    = state.get("customer_candidates", [])
    tally_ledgers = tally.get_all_ledgers_list()

    error_hint = ""
    if retry > 0:
        error_hint = (
            f"\n\nAttempt #{retry + 1}. Previously extracted names that had NO match: "
            f"{candidates}. Try alternate spellings, abbreviations, or partial names."
        )

    system = (
        "You are a Tally ERP data extraction assistant.\n"
        "Extract the customer/party name from the user message and match it "
        "against the Tally ledger list.\n\n"
        "Rules:\n"
        "1. Try exact match first, then case-insensitive, then partial/fuzzy.\n"
        "2. Return ONLY valid JSON (no markdown):\n"
        '   {"matched": "<exact tally ledger name or null>", '
        '"extracted": "<name as written in user message>"}'
        + error_hint
    )
    human = (
        f"User message:\n{state['user_input']}\n\n"
        f"Tally ledger list:\n{json.dumps(tally_ledgers, indent=2)}"
    )

    try:
        result    = state["llm_client"].chat(system, human)
        result    = json.loads(result) if isinstance(result, str) else result
        matched   = result.get("matched")
        extracted = result.get("extracted", "")
    except (ValueError, json.JSONDecodeError):
        matched, extracted = None, ""

    if matched:
        if matched not in tally_ledgers:
            best = process.extractOne(
                matched,
                tally_ledgers,
                scorer=fuzz.WRatio,
                score_cutoff=70,
            )
            matched = best[0] if best else None

    if extracted and extracted not in candidates:
        candidates = candidates + [extracted]

    if matched:
        return {
            **state,
            "customer_name":        matched,        # ← guaranteed to be exact Tally string
            "customer_confidence":  "confirmed",
            "customer_candidates":  candidates,
            "customer_retry_count": retry + 1,
        }

    return {
        **state,
        "customer_name":        None,
        "customer_confidence":  "pending" if retry < MAX_RETRIES - 1 else "failed",
        "customer_candidates":  candidates,
        "customer_retry_count": retry + 1,
    }


def route_customer(
    state: SalesVoucherState,
) -> Literal["node_parse_date", "node_resolve_customer", "node_failed"]:
    conf = state.get("customer_confidence", "pending")
    if conf == "confirmed":
        return "node_parse_date"
    if conf == "failed":
        return "node_failed"
    return "node_resolve_customer"


# ──────────────────────────────────────────────────────────
# NODE 2 – DATE PARSING
# ──────────────────────────────────────────────────────────

def node_parse_date(state: SalesVoucherState) -> SalesVoucherState:
    retry = state.get("date_retry_count", 0)
    today = datetime.now().strftime("%Y%m%d")

    error_hint = (
        f"\n\nAttempt #{retry + 1}. Previous extraction returned an invalid format. "
        "Ensure YYYYMMDD with no dashes or slashes."
        if retry > 0 else ""
    )

    system = (
        "You are a date extraction assistant.\n"
        "Extract the date from the user message and return it in strict YYYYMMDD format "
        "(e.g., '20240401'). No dashes, no slashes.\n\n"
        "Rules:\n"
        "1. Handle natural language: 'yesterday', 'last Monday', '5th March 2024', etc.\n"
        f"2. If no date is mentioned, default to today: {today}.\n"
        "3. Return ONLY valid JSON (no markdown):\n"
        '   {"date": "YYYYMMDD", "source": "extracted|defaulted"}'
        + error_hint
    )
    human = f"User message:\n{state['user_input']}"

    try:
        result   = state["llm_client"].chat(system, human)
        result   = json.loads(result) if isinstance(result, str) else result
        date_str = result.get("date", "")
        is_valid = bool(re.fullmatch(r"\d{8}", date_str))
    except (ValueError, json.JSONDecodeError):
        date_str, is_valid = "", False

    if is_valid:
        return {
            **state,
            "date_str":         date_str,
            "date_confidence":  "confirmed",
            "date_retry_count": retry + 1,
        }

    if retry >= MAX_RETRIES - 1:
        return {
            **state,
            "date_str":         today,
            "date_confidence":  "confirmed",
            "date_retry_count": retry + 1,
        }

    return {
        **state,
        "date_str":         None,
        "date_confidence":  "pending",
        "date_retry_count": retry + 1,
    }


def route_date(
    state: SalesVoucherState,
) -> Literal["node_resolve_items", "node_parse_date"]:
    if state.get("date_confidence") == "confirmed":
        return "node_resolve_items"
    return "node_parse_date"


# ──────────────────────────────────────────────────────────
# NODE 3 – ITEM EXTRACTION & RESOLUTION
# ──────────────────────────────────────────────────────────

def _multi_score(query: str, choice: str) -> float:
    """
    Max across 4 strategies so no single scorer's blind spot causes a wrong pick.

    ratio           → strict character-level similarity
    partial_ratio   → substring match (catches truncated extractions like '50-50 (114p)')
    token_sort_ratio → handles word-order differences
    token_set_ratio  → handles subset tokens (brand name matching)
    """
    return max(
        fuzz.ratio(query, choice),
        fuzz.partial_ratio(query, choice),       # ← the one that fixes THIS bug
        fuzz.token_sort_ratio(query, choice),
        fuzz.token_set_ratio(query, choice),
    )


def node_resolve_items(state: SalesVoucherState, tally) -> SalesVoucherState:
    retry       = state.get("item_retry_count", 0)
    prev_items  = state.get("items", [])
    tally_items = tally.get_all_stock_items()

    # ── PASS 1: Extract names/qty/rate only ──────────────────────────────────
    extract_system = (
        "You are a Tally ERP item extraction assistant.\n"
        "Extract ALL line items from the user message.\n\n"
        "CRITICAL Rules for extracted_name:\n"
        "1. Preserve the FULL item name exactly as mentioned — including pack size like '(114p)', "
        "price variants like '5/-' or '30/-', and any alphanumeric suffixes.\n"
        "   Example: '50-50 (114p) 5/-' must NOT be shortened to '50-50 (114p)'.\n"
        "2. Do NOT interpret price variants (like '5/-') as the rate — they are part of the name.\n"
        "3. Rate is only the separately stated selling/billing rate (e.g., 'at rate 1179.55').\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"items": [{"extracted_name": "...", "qty": <number>, "rate": <number or 0>}]}'
    )
    extract_human = f"User message:\n{state['user_input']}"

    try:
        raw = state["llm_client"].chat(extract_system, extract_human)
        raw = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(raw, dict):
            raise ValueError(
                f"PASS1 LLM returned {type(raw).__name__} instead of dict: {raw}"
            )
        extracted_items = raw.get("items", [])
    except (ValueError, json.JSONDecodeError, AttributeError):
        extracted_items = [
            {"extracted_name": i["extracted_name"], "qty": i.get("qty", 1), "rate": i.get("rate", 0)}
            for i in (prev_items or [])
        ]

    # ── PASS 2: rapidfuzz shortlist with multi-scorer ────────────────────────
    FUZZY_TOP_K     = 8
    FUZZY_THRESHOLD = max(20, 50 - (retry * 10))

    shortlisted = []
    for item in extracted_items:
        name = item["extracted_name"]
        scored = [
            (tally_name, _multi_score(name, tally_name))
            for tally_name in tally_items
        ]
        candidates = [
            tally_name
            for tally_name, score in sorted(scored, key=lambda x: x[1], reverse=True)
            if score >= FUZZY_THRESHOLD
        ][:FUZZY_TOP_K]

        shortlisted.append({
            "extracted_name": name,
            "qty":            item.get("qty", 1),
            "rate":           item.get("rate", 0),
            "candidates":     candidates,
        })

    # ── PASS 3: LLM picks best from tiny shortlist ───────────────────────────
    error_hint = ""
    if retry > 0:
        unmatched = [i["extracted_name"] for i in (prev_items or []) if not i.get("tally_name")]
        error_hint = (
            f"\n\nAttempt #{retry + 1}. Previously unmatched: {unmatched}. "
            "Be more lenient — minor spelling/spacing differences are acceptable."
        )

    match_system = (
        "You are a Tally ERP item matching assistant.\n"
        "Each item has an extracted name and a shortlist of fuzzy-matched Tally stock candidates.\n\n"
        "Rules:\n"
        "1. Pick the best tally_name from candidates — prefer the one sharing the most "
        "tokens (brand name, pack size like '114p', price variant like '5/-').\n"
        "2. If candidates is empty or none are reasonable, set tally_name to null.\n"
        "3. Preserve qty and rate values exactly from input.\n"
        "4. Set all_matched=true only if every item has a non-null tally_name.\n"
        "5. Return ONLY valid JSON (no markdown):\n"
        '   {"items": [{"extracted_name": "...", "tally_name": "...", "qty": 1, "rate": 100}], '
        '"all_matched": true/false}'
        + error_hint
    )
    match_human = f"Items to match:\n{json.dumps(shortlisted, indent=2)}"

    try:
        result = state["llm_client"].chat(match_system, match_human)
        result = json.loads(result) if isinstance(result, str) else result
        if not isinstance(result, dict):
            raise ValueError(
                f"PASS3 LLM returned {type(result).__name__} instead of dict: {result}"
            )
        items       = result.get("items", [])
        all_matched = result.get("all_matched", False)
    except (ValueError, json.JSONDecodeError, AttributeError):
        items, all_matched = prev_items or [], False

    if all_matched and items:
        return {
            **state,
            "items":            items,
            "item_confidence":  "confirmed",
            "item_retry_count": retry + 1,
        }

    return {
        **state,
        "items":            items,
        "item_confidence":  "pending" if retry < MAX_RETRIES - 1 else "failed",
        "item_retry_count": retry + 1,
    }

def route_items(
    state: SalesVoucherState,
) -> Literal["node_prepare_xml", "node_resolve_items", "node_failed"]:
    conf = state.get("item_confidence", "pending")
    if conf == "confirmed":
        return "node_prepare_xml"
    if conf == "failed":
        return "node_failed"
    return "node_resolve_items"


# ──────────────────────────────────────────────────────────
# NODE 4 – XML PREPARATION
# ──────────────────────────────────────────────────────────

def node_prepare_xml(state: SalesVoucherState) -> SalesVoucherState:
    retry = state.get("xml_retry_count", 0)

    error_hint = ""
    if state.get("execution_error"):
        error_hint = (
            f"\n\nThe previous XML caused this Tally error:\n"
            f"{state['execution_error']}\n"
            "Fix the XML accordingly."
        )

    items_for_xml = [
        {
            "name": i["tally_name"],        # ← the ONLY name the LLM will see
            "qty":  i.get("qty", 1),
            "rate": i.get("rate", 0),
        }
        for i in (state.get("items") or [])
    ]

    total = sum(i["qty"] * i["rate"] for i in items_for_xml)

    system = (
        "You are a Tally ERP XML generation expert.\n"
        "Generate a valid Tally XML payload to import a Sales Voucher.\n\n"
        "Use this template as the exact structure (substitute all placeholders):\n"
        f"{TALLY_SALES_XML_TEMPLATE}\n\n"
        "Rules:\n"
        "1. DATE must be YYYYMMDD (no dashes).\n"
        "2. Add one <ALLINVENTORYENTRIES.LIST> block per item.\n"
        "3. Party ledger AMOUNT is negative (Dr party / Cr sales).\n"
        "4. Sales ledger AMOUNT equals total of all item amounts (positive).\n"
        "5. STRICTLY OUTPUT ONLY: {\"xml\": \"<raw xml string>\"} — "
        "valid JSON, no markdown, no extra keys, no explanation."
        + error_hint
    )

    human = (
        f"Customer (PARTY_NAME): {state.get('customer_name')}\n"
        f"Date (YYYYMMDD):        {state.get('date_str')}\n"
        f"Total amount:           {total}\n"
        f"Items:\n{json.dumps(items_for_xml, indent=2)}"
    )

    # ── Inner parse-repair loop ───────────────────────────────────────────────
    MAX_PARSE_RETRIES = 3
    raw_output        = None
    xml_raw           = None

    for parse_attempt in range(MAX_PARSE_RETRIES):
        if parse_attempt == 0:
            # First attempt: normal generation call
            raw_output = state["llm_client"].chat(system, human)
        else:
            # Subsequent attempts: feed bad output + exact error back to LLM
            repair_system = (
                "You are a JSON repair assistant.\n"
                "The previous response failed to parse. Fix it and return ONLY valid JSON.\n\n"
                f"Parse error encountered:\n{parse_error}\n\n"                      # ← exact Python error
                f"Your bad previous output was:\n{raw_output}\n\n"                  # ← what went wrong
                "Return STRICTLY: {\"xml\": \"<raw xml string>\"}\n"
                "Rules:\n"
                "- No markdown fences, no extra keys, no explanation.\n"
                "- The xml value must be a valid XML string with all quotes escaped as \\\"."
            )
            raw_output = state["llm_client"].chat(repair_system, "Fix the output above.")

        # ── Attempt parse ─────────────────────────────────────────────────────
        try:
            # Strip markdown fences before JSON parse (LLM sometimes adds them)
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.DOTALL).strip()
            result  = json.loads(cleaned)
            xml_raw = result.get("xml", "").strip()

            if not xml_raw:
                raise ValueError(f"'xml' key is missing or empty in parsed JSON: {result}")

            # Strip any fences the LLM may have wrapped around the XML value itself
            xml_raw = re.sub(r"^```(?:xml)?\s*|\s*```$", "", xml_raw, flags=re.DOTALL).strip()
            break   # ← success, exit inner loop

        except (json.JSONDecodeError, ValueError) as e:
            parse_error = f"{type(e).__name__}: {e}"   # ← captured for next repair prompt
            xml_raw     = None
            if parse_attempt == MAX_PARSE_RETRIES - 1:
                # All inner retries exhausted — return failure into the graph
                return {
                    **state,
                    "xml_payload":     None,
                    "execution_error": f"XML generation failed after {MAX_PARSE_RETRIES} parse attempts. Last error: {parse_error}",
                    "status":          "error:xml",     # ← graph routes back to this node
                    "xml_retry_count": retry + 1,
                }

    return {
        **state,
        "xml_payload":     xml_raw,
        "xml_retry_count": retry + 1,
    }


# ──────────────────────────────────────────────────────────
# NODE 5 – EXECUTE & ERROR RECOVERY
# ──────────────────────────────────────────────────────────

def node_execute(state: SalesVoucherState, tally) -> SalesVoucherState:
    retry = state.get("error_retry_count", 0)

    try:
        result = tally.execute_xml_query(state["xml_payload"])

        if isinstance(result, dict) and result.get("status") == "error":
            raise ValueError(result.get("message", "Tally returned an application error"))

        return {
            **state,
            "execution_result":  result,
            "execution_error":   None,
            "status":            "success",
            "error_retry_count": retry + 1,
        }

    except Exception as exc:
        error_msg = str(exc)

        system = (
            "You are a Tally ERP error diagnosis assistant.\n"
            "Given a Tally execution error and the XML/data used, "
            "identify the root cause field.\n\n"
            "Return ONLY valid JSON:\n"
            '{"root_cause": "customer|date|items|xml|unknown", "explanation": "..."}'
        )
        human = (
            f"Error message:\n{error_msg}\n\n"
            f"Customer: {state.get('customer_name')}\n"
            f"Date:     {state.get('date_str')}\n"
            f"Items:    {json.dumps(state.get('items', []))}\n\n"
            f"XML sent:\n{state.get('xml_payload', '')}"
        )

        try:
            diagnosis  = state["llm_client"].chat(system, human)   # ← CHANGED
            root_cause = diagnosis.get("root_cause", "xml")
        except (ValueError, json.JSONDecodeError):
            root_cause = "xml"

        return {
            **state,
            "execution_result":  None,
            "execution_error":   error_msg,
            "status":            f"error:{root_cause}",
            "error_retry_count": retry + 1,
        }


def route_execution(state: SalesVoucherState) -> Literal[
    END,                        # type: ignore[valid-type]
    "node_resolve_customer",
    "node_parse_date",
    "node_resolve_items",
    "node_prepare_xml",
    "node_failed",
]:
    if state.get("status") == "success":
        return END
    if state.get("error_retry_count", 0) >= MAX_RETRIES:
        return "node_failed"
    cause = state.get("status", "")
    if "customer" in cause:
        state["customer_confidence"] = "pending"
        return "node_resolve_customer"
    if "date" in cause:
        state["date_confidence"] = "pending"
        return "node_parse_date"
    if "items" in cause:
        state["item_confidence"] = "pending"
        return "node_resolve_items"
    return "node_prepare_xml"


# ──────────────────────────────────────────────────────────
# FAILED TERMINAL NODE
# ──────────────────────────────────────────────────────────

def node_failed(state: SalesVoucherState) -> SalesVoucherState:
    reason = (
        state.get("execution_error")
        or (
            "Customer not found after max retries"
            if state.get("customer_confidence") == "failed"
            else "Item(s) not found in Tally after max retries"
            if state.get("item_confidence") == "failed"
            else "Max retries exceeded"
        )
    )
    return {**state, "status": "failed", "failed_reason": reason}


# ──────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ──────────────────────────────────────────────────────────

def build_sales_voucher_graph(tally_interface):
    def _resolve_customer(s): return node_resolve_customer(s, tally_interface)
    def _resolve_items(s):    return node_resolve_items(s, tally_interface)
    def _execute(s):          return node_execute(s, tally_interface)

    builder = StateGraph(SalesVoucherState)

    builder.add_node("node_resolve_customer", _resolve_customer)
    builder.add_node("node_parse_date",       node_parse_date)
    builder.add_node("node_resolve_items",    _resolve_items)
    builder.add_node("node_prepare_xml",      node_prepare_xml)
    builder.add_node("node_execute",          _execute)
    builder.add_node("node_failed",           node_failed)

    builder.add_edge(START, "node_resolve_customer")

    builder.add_conditional_edges("node_resolve_customer", route_customer, {
        "node_parse_date":       "node_parse_date",
        "node_resolve_customer": "node_resolve_customer",
        "node_failed":           "node_failed",
    })
    builder.add_conditional_edges("node_parse_date", route_date, {
        "node_resolve_items": "node_resolve_items",
        "node_parse_date":    "node_parse_date",
    })
    builder.add_conditional_edges("node_resolve_items", route_items, {
        "node_prepare_xml":   "node_prepare_xml",
        "node_resolve_items": "node_resolve_items",
        "node_failed":        "node_failed",
    })
    builder.add_edge("node_prepare_xml", "node_execute")
    builder.add_conditional_edges("node_execute", route_execution, {
        END:                     END,
        "node_resolve_customer": "node_resolve_customer",
        "node_parse_date":       "node_parse_date",
        "node_resolve_items":    "node_resolve_items",
        "node_prepare_xml":      "node_prepare_xml",
        "node_failed":           "node_failed",
    })
    builder.add_edge("node_failed", END)

    return builder.compile()


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

class SalesVoucherAgent:
    def __init__(self, tally_interface, llm_client):   # ← CHANGED: added llm_client param
        self.tally      = tally_interface
        self.llm_client = llm_client                   # ← CHANGED
        self.graph      = build_sales_voucher_graph(tally_interface)

    def run(self, user_input: str, context: dict = None) -> dict:
        initial_state: SalesVoucherState = {
            "user_input":           user_input,
            "llm_client":           self.llm_client,   # ← CHANGED: injected into state
            "customer_name":        None,
            "customer_candidates":  [],
            "customer_confidence":  "pending",
            "customer_retry_count": 0,
            "date_str":             None,
            "date_confidence":      "pending",
            "date_retry_count":     0,
            "items":                None,
            "item_confidence":      "pending",
            "item_retry_count":     0,
            "xml_payload":          None,
            "xml_retry_count":      0,
            "execution_result":     None,
            "execution_error":      None,
            "error_retry_count":    0,
            "status":               "running",
            "failed_reason":        None,
        }
        return self.graph.invoke(initial_state)
