# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SEC EDGAR XBRL API utilities for financial benchmark.

Provides functions to fetch company financial data from SEC EDGAR's free XBRL API,
with local JSON caching and rate limiting.

EDGAR API docs: https://www.sec.gov/edgar/sec-api-documentation
Rate limit: 10 requests/second (we use 0.12s between requests to be safe).
"""

import json
import os
import random
import re
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://data.sec.gov"
EDGAR_HEADERS = {"User-Agent": "DrBencher research@example.com", "Accept": "application/json"}
EDGAR_CACHE_DIR = "cache/edgar_cache"

# Minimum delay between EDGAR API requests (seconds)
_RATE_LIMIT_DELAY = 0.12
_last_request_time = 0.0


# ---------------------------------------------------------------------------
# XBRL Tag Fallback Lists
# ---------------------------------------------------------------------------

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",  # ASC 606
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",  # gaming/hospitality
    "InterestAndDividendIncomeOperating",  # banks/financial services
    "RevenuesNetOfInterestExpense",  # banks
]
COST_OF_REVENUE_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
]
NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",
]
ASSETS_TAGS = ["Assets"]
LIABILITIES_TAGS = ["Liabilities"]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue"]
EPS_TAGS = ["EarningsPerShareBasic"]
CURRENT_ASSETS_TAGS = ["AssetsCurrent"]
CURRENT_LIABILITIES_TAGS = ["LiabilitiesCurrent"]
RD_EXPENSE_TAGS = ["ResearchAndDevelopmentExpense"]
OPERATING_CASH_FLOW_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]
INTEREST_EXPENSE_TAGS = ["InterestExpense", "InterestExpenseDebt"]
INVENTORY_TAGS = ["InventoryNet", "InventoryGross"]
ACCOUNTS_RECEIVABLE_TAGS = ["AccountsReceivableNetCurrent", "AccountsReceivableNet"]
ACCOUNTS_PAYABLE_TAGS = ["AccountsPayableCurrent", "AccountsPayable"]
DEPRECIATION_AMORTIZATION_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
]
DIVIDENDS_PER_SHARE_TAGS = [
    "CommonStockDividendsPerShareDeclared",
    "CommonStockDividendsPerShareCashPaid",
]

FINANCIAL_CONCEPTS: Dict[str, List[str]] = {
    "revenue": REVENUE_TAGS,
    "cost_of_revenue": COST_OF_REVENUE_TAGS,
    "net_income": NET_INCOME_TAGS,
    "total_assets": ASSETS_TAGS,
    "total_liabilities": LIABILITIES_TAGS,
    "equity": EQUITY_TAGS,
    "operating_income": OPERATING_INCOME_TAGS,
    "cash": CASH_TAGS,
    "eps": EPS_TAGS,
    "current_assets": CURRENT_ASSETS_TAGS,
    "current_liabilities": CURRENT_LIABILITIES_TAGS,
    "rd_expense": RD_EXPENSE_TAGS,
    "operating_cash_flow": OPERATING_CASH_FLOW_TAGS,
    "capex": CAPEX_TAGS,
    "interest_expense": INTEREST_EXPENSE_TAGS,
    "inventory": INVENTORY_TAGS,
    "accounts_receivable": ACCOUNTS_RECEIVABLE_TAGS,
    "accounts_payable": ACCOUNTS_PAYABLE_TAGS,
    "depreciation_amortization": DEPRECIATION_AMORTIZATION_TAGS,
    "dividends_per_share": DIVIDENDS_PER_SHARE_TAGS,
}


# ---------------------------------------------------------------------------
# Company Universe (top companies per sector by market cap)
# ---------------------------------------------------------------------------

COMPANY_UNIVERSE_SECTORS: Dict[str, List[str]] = {
    "construction": [
        "DHI", "LEN", "NVR", "PHM", "TOL", "MDC", "MTH", "KBH",
        "MHO", "CCS", "BLDR", "BECN", "TREX", "AZEK", "AWI",
        "OC", "UFPI", "BLD", "JELD", "SUM",
    ],
    "chemicals": [
        "DOW", "DD", "LYB", "CE", "EMN", "HUN", "OLN", "RPM",
        "AVNT", "FUL", "ASH", "CBT", "NEU", "WDFC", "KWR",
        "GCP", "SCL", "MTX", "CC", "TROX",
    ],
    "mining_metals": [
        "FCX", "NEM", "GOLD", "AEM", "WPM", "FNV", "RGLD", "HL",
        "CDE", "PAAS", "SCCO", "AA", "CLF", "X", "STLD",
        "RS", "ATI", "HCC", "ARCH", "BTU",
    ],
    "hospitality_travel": [
        "MAR", "HLT", "H", "WH", "ABNB", "EXPE", "BKNG", "CHH",
        "RHP", "PK", "APLE", "SHO", "DRH", "PEB", "RCL",
        "CCL", "NCLH", "PLYA", "TNL", "VAC",
    ],
    "freight_logistics": [
        "XPO", "ODFL", "SAIA", "JBHT", "KNX", "WERN", "SNDR", "LSTR",
        "CHRW", "ARCB", "GXO", "EXPD", "FWRD", "ECHO", "HTLD",
        "MRTN", "HUBG", "RLGT", "UHAL", "ABG",
    ],
    "medical_devices": [
        "MDT", "SYK", "BSX", "EW", "ISRG", "HOLX", "ALGN", "DXCM",
        "PODD", "INSP", "TFX", "GKOS", "IRTC", "NARI", "TNDM",
        "LIVN", "MMSI", "SWAV", "AORT", "ATEC",
    ],
    "restaurants_dining": [
        "MCD", "SBUX", "YUM", "CMG", "DPZ", "DINE", "CAKE", "TXRH",
        "BJRI", "JACK", "SHAK", "WING", "LOCO", "EAT", "QSR",
        "WEN", "ARCO", "NDLS", "KRUS", "RRGB",
    ],
    "payments_fintech": [
        "V", "MA", "PYPL", "SQ", "GPN", "FIS", "FISV", "JKHY",
        "WEX", "FLT", "EVTC", "DFS", "FOUR", "PAGS", "STNE",
        "FLYW", "BILL", "AFRM", "RPAY", "SOFI",
    ],
    "clean_energy": [
        "ENPH", "SEDG", "FSLR", "RUN", "CSIQ", "JKS", "DQ", "ARRY",
        "SHLS", "STEM", "BE", "PLUG", "BLDP", "FCEL", "CHPT",
        "EVGO", "QS", "NEP", "CWEN", "ORA",
    ],
    "enterprise_software": [
        "CRM", "ADBE", "NOW", "WDAY", "ZM", "TEAM", "SNOW", "DDOG",
        "MDB", "HUBS", "VEEV", "ANSS", "PAYC", "PCTY", "TYL",
        "GWRE", "ZI", "PLAN", "FROG", "ESTC",
    ],
    "ecommerce": [
        "SHOP", "EBAY", "ETSY", "W", "CHWY", "CARG", "RVLV", "REAL",
        "CPNG", "MELI", "SE", "PDD", "JD", "BABA", "GLBE",
        "VIPS", "FTCH", "POSH", "CVNA", "OSTK",
    ],
    "regional_banks": [
        "USB", "PNC", "TFC", "KEY", "CFG", "FITB", "HBAN", "MTB",
        "RF", "ZION", "CMA", "WAL", "FHN", "BOKF", "SNV",
        "VLY", "EWBC", "SBCF", "IBKR", "SIVB",
    ],
    "asset_management": [
        "BLK", "SCHW", "BEN", "TROW", "IVZ", "AMG", "AB", "APAM",
        "VCTR", "CG", "KKR", "APO", "ARES", "OWL", "BAM",
        "BX", "TPG", "HLNE", "STEP", "VRTS",
    ],
    "healthcare_providers": [
        "UNH", "CVS", "CI", "ELV", "HUM", "CNC", "MOH", "HCA",
        "THC", "UHS", "DVA", "ENSG", "AMED", "SGRY", "SEM",
        "USPH", "OPCH", "PINC", "CHE", "NHC",
    ],
    "agribusiness": [
        "ADM", "BG", "CTVA", "FMC", "NTR", "SMG", "ANDE", "CALM",
        "PPC", "INGR", "DAR", "VITL", "LMNR", "FDP", "HAIN",
        "THS", "LANC", "SEB", "CASY", "UNFI",
    ],
    "cybersecurity": [
        "PANW", "CRWD", "ZS", "FTNT", "S", "CYBR", "NET", "QLYS",
        "TENB", "RPD", "VRNS", "OKTA", "CHKP", "RDWR", "OSPN",
        "BB", "SCWX", "PING", "TMCI", "FFIV",
    ],
    "gaming_casinos": [
        "MGM", "LVS", "WYNN", "CZR", "PENN", "DKNG", "RSI", "GENI",
        "GLPI", "RRR", "IGT", "EVRI", "SGMS", "AGS", "BALY",
        "BYD", "MLCO", "FLUT", "GAN", "SRAD",
    ],
    "specialty_finance": [
        "COF", "SYF", "DFS", "ALLY", "CACC", "ENVA", "OMF", "SLM",
        "NAVI", "TREE", "WRLD", "QFIN", "OPRT", "UPST", "LC",
        "SOFI", "AFRM", "FCFS", "EEFT", "PRAA",
    ],
    "apparel_footwear": [
        "NKE", "LULU", "VFC", "HBI", "PVH", "RL", "TPR", "CPRI",
        "UAA", "GPS", "ANF", "AEO", "GIII", "OXM", "COLM",
        "SKX", "CROX", "DECK", "WWW", "EXPR",
    ],
    "oil_field_services": [
        "SLB", "HAL", "BKR", "NOV", "FTI", "CHX", "HP", "PTEN",
        "RIG", "VAL", "NE", "LBRT", "OII", "DRQ", "WTTR",
        "MRC", "CLB", "PUMP", "AESI", "XPRO",
    ],
}


# ---------------------------------------------------------------------------
# Caching & Rate Limiting
# ---------------------------------------------------------------------------

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(EDGAR_CACHE_DIR, exist_ok=True)


def _cached_get(url: str, cache_key: str, max_age_hours: int = 24) -> Optional[dict]:
    """Fetch URL with local JSON cache and rate limiting.

    Args:
        url: URL to fetch.
        cache_key: Key for local cache filename (sanitized).
        max_age_hours: Maximum age of cached data in hours.

    Returns:
        Parsed JSON dict, or None on failure.
    """
    global _last_request_time
    _ensure_cache_dir()

    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(EDGAR_CACHE_DIR, f"{safe_key}.json")

    # Check cache
    if os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < max_age_hours:
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass  # Cache corrupted, re-fetch

    # Rate limit
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            _last_request_time = time.time()

            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            elif response.status_code in (429, 503):
                wait = (2 ** attempt) * 5 + random.random() * 5
                print(f"EDGAR rate-limited ({response.status_code}) for {cache_key}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"EDGAR API error {response.status_code} for {cache_key}", flush=True)
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.random() * 5
            print(f"EDGAR request error for {cache_key}: {e}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"EDGAR unexpected error for {cache_key}: {e}", flush=True)
            return None

    print(f"EDGAR request failed after {max_retries} retries for {cache_key}", flush=True)
    return None


# ---------------------------------------------------------------------------
# Core API Functions
# ---------------------------------------------------------------------------

def get_company_tickers() -> Dict[str, dict]:
    """Fetch ticker -> CIK mapping from SEC.

    Returns:
        Dict mapping ticker to {"cik": int, "title": str}.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    data = _cached_get(url, "company_tickers", max_age_hours=168)  # 1 week cache
    if not data:
        return {}

    result = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = entry.get("cik_str", 0)
        title = entry.get("title", "")
        if ticker and cik:
            result[ticker] = {"cik": int(cik), "title": title}
    return result


def get_cik_for_ticker(ticker: str) -> Optional[int]:
    """Look up CIK number for a ticker symbol."""
    tickers = get_company_tickers()
    entry = tickers.get(ticker.upper())
    return entry["cik"] if entry else None


def get_company_name(ticker: str) -> Optional[str]:
    """Look up company name for a ticker symbol."""
    tickers = get_company_tickers()
    entry = tickers.get(ticker.upper())
    return entry["title"] if entry else None


def fetch_company_facts(cik: int) -> Optional[dict]:
    """Fetch ALL XBRL facts for a company.

    Endpoint: data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json

    Returns:
        Nested dict: {taxonomy: {tag: {units: {unit: [fact_entries]}}}}
    """
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json"
    return _cached_get(url, f"facts_{cik:010d}", max_age_hours=72)


def fetch_company_concept(cik: int, taxonomy: str, tag: str) -> Optional[dict]:
    """Fetch one XBRL concept for a company.

    Endpoint: data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json
    """
    url = f"{EDGAR_BASE}/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
    return _cached_get(url, f"concept_{cik:010d}_{taxonomy}_{tag}", max_age_hours=72)


def fetch_frame(taxonomy: str, tag: str, unit: str, period: str) -> Optional[dict]:
    """Fetch one concept across ALL companies for one period.

    Endpoint: data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json
    Period examples: CY2023 (annual), CY2023Q1 (quarterly), CY2023Q4I (instantaneous)
    """
    url = f"{EDGAR_BASE}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
    return _cached_get(url, f"frame_{taxonomy}_{tag}_{unit}_{period}", max_age_hours=168)


# ---------------------------------------------------------------------------
# Financial Data Extraction
# ---------------------------------------------------------------------------

def extract_financial_metric(company_facts: dict, concept: str,
                             fiscal_year: int, form: str = "10-K") -> Optional[float]:
    """Extract a financial metric from company facts using tag fallback.

    Tries each tag in FINANCIAL_CONCEPTS[concept] until one matches.
    Filters by: form=10-K (or 10-K/A), fy=fiscal_year, fp=FY.
    Deduplicates by preferring most recent filing date.

    Args:
        company_facts: Raw company facts JSON from fetch_company_facts().
        concept: Key from FINANCIAL_CONCEPTS (e.g., "revenue").
        fiscal_year: Fiscal year to extract (e.g., 2023).
        form: SEC filing form (default "10-K").

    Returns:
        Float value in USD, or None if not found.
    """
    if not company_facts or concept not in FINANCIAL_CONCEPTS:
        return None

    tags = FINANCIAL_CONCEPTS[concept]
    facts_data = company_facts.get("facts", {})

    for tag in tags:
        # Try us-gaap taxonomy first, then ifrs-full
        for taxonomy in ["us-gaap", "ifrs-full"]:
            tag_data = facts_data.get(taxonomy, {}).get(tag, {})
            if not tag_data:
                continue

            units = tag_data.get("units", {})

            # For most metrics, look in USD; for per-share metrics use USD/shares
            if concept in ("eps", "dividends_per_share"):
                unit_key = "USD/shares"
            else:
                unit_key = "USD"

            entries = units.get(unit_key, [])
            if not entries:
                continue

            # Filter by form and fiscal year
            candidates = []
            for entry in entries:
                entry_form = entry.get("form", "")
                entry_fy = entry.get("fy", 0)
                entry_fp = entry.get("fp", "")

                if entry_form not in (form, f"{form}/A"):
                    continue
                if entry_fy != fiscal_year:
                    continue
                if entry_fp != "FY":
                    continue

                candidates.append(entry)

            if not candidates:
                continue

            # Prefer the entry whose period-end date is latest.
            # This selects current-year data over comparative (prior-year)
            # data that appears in the same filing (all share the same fy).
            # Secondary: prefer annual (longer-duration) entries over
            # quarterly breakdowns that may share the same end date.
            def _sort_key(e):
                end = e.get("end", "")
                start = e.get("start")
                if start and end:
                    try:
                        from datetime import date as _d
                        dur = (_d.fromisoformat(end) - _d.fromisoformat(start)).days
                    except (ValueError, TypeError):
                        dur = 9999
                else:
                    dur = 9999          # balance-sheet instant → no quarterly dups
                return (end, dur)

            candidates.sort(key=_sort_key, reverse=True)
            value = candidates[0].get("val")
            if value is not None:
                return float(value)

    return None


def get_company_financials(ticker: str, fiscal_year: int) -> Optional[Dict[str, Any]]:
    """Fetch all key financial metrics for a company and year.

    Returns:
        Dict with ticker, name, fiscal_year, and all available metrics + computed ratios.
        None if company not found.
    """
    cik = get_cik_for_ticker(ticker)
    if cik is None:
        return None

    company_facts = fetch_company_facts(cik)
    if not company_facts:
        return None

    name = get_company_name(ticker) or company_facts.get("entityName", ticker)

    result: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "name": name,
        "cik": cik,
        "fiscal_year": fiscal_year,
    }

    # Extract all base metrics
    for concept in FINANCIAL_CONCEPTS:
        val = extract_financial_metric(company_facts, concept, fiscal_year)
        result[concept] = val

    # Prior-year total assets for averaging (used by asset_turnover template)
    total_assets = result.get("total_assets")
    total_assets_prior = extract_financial_metric(company_facts, "total_assets", fiscal_year - 1)
    result["total_assets_prior"] = total_assets_prior
    if total_assets and total_assets_prior:
        result["total_assets_avg"] = (total_assets + total_assets_prior) / 2
    else:
        result["total_assets_avg"] = total_assets  # fallback to single year

    # Compute derived ratios
    revenue = result.get("revenue")
    cost_of_revenue = result.get("cost_of_revenue")
    net_income = result.get("net_income")
    total_liabilities = result.get("total_liabilities")
    equity = result.get("equity")
    operating_income = result.get("operating_income")
    current_assets = result.get("current_assets")
    current_liabilities = result.get("current_liabilities")

    # Gross margin
    if revenue and cost_of_revenue and revenue != 0:
        result["gross_margin"] = (revenue - cost_of_revenue) / revenue
    else:
        result["gross_margin"] = None

    # Operating margin
    if revenue and operating_income and revenue != 0:
        result["operating_margin"] = operating_income / revenue
    else:
        result["operating_margin"] = None

    # Net margin
    if revenue and net_income and revenue != 0:
        result["net_margin"] = net_income / revenue
    else:
        result["net_margin"] = None

    # ROE
    if net_income is not None and equity and equity != 0:
        result["roe"] = net_income / equity
    else:
        result["roe"] = None

    # ROA
    if net_income is not None and total_assets and total_assets != 0:
        result["roa"] = net_income / total_assets
    else:
        result["roa"] = None

    # Debt-to-equity
    if total_liabilities is not None and equity and equity != 0:
        result["debt_to_equity"] = total_liabilities / equity
    else:
        result["debt_to_equity"] = None

    # Current ratio
    if current_assets and current_liabilities and current_liabilities != 0:
        result["current_ratio"] = current_assets / current_liabilities
    else:
        result["current_ratio"] = None

    return result


def get_metric_history(ticker: str, concept: str,
                       start_year: int, end_year: int) -> Optional[Dict[int, float]]:
    """Fetch a financial metric across multiple years for a company.

    Args:
        ticker: Stock ticker symbol.
        concept: Key from FINANCIAL_CONCEPTS.
        start_year: First fiscal year.
        end_year: Last fiscal year (inclusive).

    Returns:
        Dict mapping year to value, or None if company not found.
    """
    cik = get_cik_for_ticker(ticker)
    if cik is None:
        return None

    company_facts = fetch_company_facts(cik)
    if not company_facts:
        return None

    result = {}
    for year in range(start_year, end_year + 1):
        val = extract_financial_metric(company_facts, concept, year)
        if val is not None:
            result[year] = val

    return result if result else None


def compare_companies_metric(tickers: List[str], concept: str,
                              fiscal_year: int) -> List[Dict[str, Any]]:
    """Fetch one metric for multiple companies and return ranked list.

    Returns:
        List of {"ticker", "name", "value"} dicts, sorted descending by value.
    """
    results = []
    for ticker in tickers:
        cik = get_cik_for_ticker(ticker)
        if cik is None:
            continue
        company_facts = fetch_company_facts(cik)
        if not company_facts:
            continue
        val = extract_financial_metric(company_facts, concept, fiscal_year)
        if val is not None:
            name = get_company_name(ticker) or ticker
            results.append({"ticker": ticker.upper(), "name": name, "value": val})

    results.sort(key=lambda x: x["value"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Wikipedia Company Clue Fetching
# ---------------------------------------------------------------------------

def fetch_company_clues(ticker: str) -> List[dict]:
    """Fetch descriptive clue facts from Wikipedia for a company.

    Extracts non-financial facts such as:
    - Founding year and location
    - Key products/services
    - Industry/sector description
    - Headquarters location
    - Notable events (acquisitions, IPO)
    - Number of employees (approximate)

    Returns:
        List of fact dicts: {"fact_id", "fact", "topic", "source"}.
    """
    name = get_company_name(ticker)
    if not name:
        return []

    # Search Wikipedia for the company
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": name,
        "srlimit": 3,
        "utf8": 1,
    }
    headers = {
        "User-Agent": "DrBencher/1.0 (research@example.com)",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(search_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        search_data = resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return []
    except Exception:
        return []

    # Fetch the top article extract
    page_title = search_results[0].get("title", "")
    extract_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "utf8": 1,
    }

    try:
        resp = requests.get(extract_url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        pages = resp.json().get("query", {}).get("pages", {})
        extract_text = ""
        for page in pages.values():
            extract_text = page.get("extract", "")
            break
    except Exception:
        return []

    if not extract_text:
        return []

    # Parse sentences into clue facts
    sentences = re.split(r'(?<=[.!?])\s+', extract_text.strip())
    clues = []
    topics = ["description", "founding", "products", "headquarters", "industry", "history"]

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if len(clues) >= 10:
            break

        # Assign a topic based on keywords
        sent_lower = sent.lower()
        if any(w in sent_lower for w in ["founded", "established", "incorporated"]):
            topic = "founding"
        elif any(w in sent_lower for w in ["headquartered", "headquarters", "based in"]):
            topic = "headquarters"
        elif any(w in sent_lower for w in ["product", "service", "manufacture", "develop", "design"]):
            topic = "products"
        elif any(w in sent_lower for w in ["industry", "sector", "technology", "financial"]):
            topic = "industry"
        elif any(w in sent_lower for w in ["acquired", "merged", "ipo", "listed"]):
            topic = "history"
        else:
            topic = topics[min(i, len(topics) - 1)]

        clues.append({
            "fact_id": f"W{i+1}",
            "fact": sent,
            "topic": topic,
            "source": f"Wikipedia:{page_title}",
        })

    return clues


# ---------------------------------------------------------------------------
# Sector / SIC Code Helpers
# ---------------------------------------------------------------------------

FINANCIAL_THEMES: Dict[str, Dict[str, Any]] = {
    "technology": {"sic_codes": ["7371", "7372", "3674"], "min_companies": 20},
    "healthcare": {"sic_codes": ["2834", "2836", "8000"], "min_companies": 15},
    "financials": {"sic_codes": ["6020", "6022", "6211"], "min_companies": 15},
    "energy": {"sic_codes": ["1311", "2911", "4911"], "min_companies": 10},
    "consumer": {"sic_codes": ["5411", "5912", "5812"], "min_companies": 10},
    "industrials": {"sic_codes": ["3711", "3721", "3724"], "min_companies": 10},
    "utilities": {"sic_codes": ["4911", "4931", "4941"], "min_companies": 8},
    "real_estate": {"sic_codes": ["6500", "6512", "6798"], "min_companies": 8},
    # ---- NEW TOPICS (2026-03-16) ----
    "construction": {"sic_codes": ["1521", "1522", "2431"], "min_companies": 10},
    "chemicals": {"sic_codes": ["2810", "2820", "2860"], "min_companies": 10},
    "mining_metals": {"sic_codes": ["1040", "1090", "3312"], "min_companies": 10},
    "hospitality_travel": {"sic_codes": ["7011", "7941", "4512"], "min_companies": 10},
    "freight_logistics": {"sic_codes": ["4213", "4215", "4731"], "min_companies": 10},
    "medical_devices": {"sic_codes": ["3841", "3845", "3827"], "min_companies": 10},
    "restaurants_dining": {"sic_codes": ["5812", "5813"], "min_companies": 10},
    "payments_fintech": {"sic_codes": ["7372", "6153", "6159"], "min_companies": 10},
    "clean_energy": {"sic_codes": ["3674", "3679", "4911"], "min_companies": 8},
    "enterprise_software": {"sic_codes": ["7372", "7371"], "min_companies": 10},
    "ecommerce": {"sic_codes": ["5961", "5947", "7374"], "min_companies": 10},
    "regional_banks": {"sic_codes": ["6022", "6020"], "min_companies": 10},
    "asset_management": {"sic_codes": ["6282", "6726", "6211"], "min_companies": 10},
    "healthcare_providers": {"sic_codes": ["8062", "8011", "6324"], "min_companies": 10},
    "agribusiness": {"sic_codes": ["2040", "2099", "5150"], "min_companies": 10},
    "cybersecurity": {"sic_codes": ["7372", "7371"], "min_companies": 8},
    "gaming_casinos": {"sic_codes": ["7011", "7993", "7999"], "min_companies": 8},
    "specialty_finance": {"sic_codes": ["6141", "6153", "6159"], "min_companies": 10},
    "apparel_footwear": {"sic_codes": ["2300", "5651", "3021"], "min_companies": 10},
    "oil_field_services": {"sic_codes": ["1381", "1382", "1389"], "min_companies": 10},
}


def get_sector_companies(sector: str) -> List[str]:
    """Return list of tickers in the pre-built company universe for a sector."""
    return COMPANY_UNIVERSE_SECTORS.get(sector, [])


# ---------------------------------------------------------------------------
# Wikidata QID Resolution for Companies
# ---------------------------------------------------------------------------

# Known Wikidata P31 (instance-of) types that indicate a company/organization
_COMPANY_P31_TYPES = {
    "Q783794",   # company
    "Q4830453",  # business
    "Q6881511",  # enterprise
    "Q43229",    # organization
    "Q891723",   # public company
    "Q5225895",  # public limited company
    "Q167037",   # corporation
    "Q219577",   # holding company
    "Q163740",   # nonprofit organization
    "Q161726",   # multinational corporation
    "Q778575",   # conglomerate
}

# Module-level cache for resolved Wikidata IDs
_company_wikidata_cache: Dict[str, Optional[str]] = {}


def resolve_company_wikidata_id(ticker: str) -> Optional[str]:
    """Resolve the Wikidata QID for a company given its ticker symbol.

    Uses get_company_name() to find the company name, then searches Wikidata
    and validates the top result is a company/organization by checking P31 claims.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").

    Returns:
        Wikidata QID string (e.g., "Q312") or None if not found/validated.
    """
    ticker = ticker.upper()
    if ticker in _company_wikidata_cache:
        return _company_wikidata_cache[ticker]

    name = get_company_name(ticker)
    if not name:
        _company_wikidata_cache[ticker] = None
        return None

    # Lazy import to avoid circular dependency at module load time
    from .tool_util import search_wikidata_entities

    results = search_wikidata_entities(name, limit=5)
    if not results:
        _company_wikidata_cache[ticker] = None
        return None

    # Check top results for company/organization type via P31
    for candidate in results:
        qid = candidate.get("id", "")
        if not qid.startswith("Q"):
            continue

        if _is_company_entity(qid):
            _company_wikidata_cache[ticker] = qid
            return qid

    # If none validated, fall back to top result (Wikidata search is usually good)
    top_qid = results[0].get("id", "")
    if top_qid.startswith("Q"):
        _company_wikidata_cache[ticker] = top_qid
        return top_qid

    _company_wikidata_cache[ticker] = None
    return None


def _is_company_entity(qid: str) -> bool:
    """Check if a Wikidata entity is a company/organization by checking P31 claims."""
    global _last_request_time

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {"User-Agent": "DrBencher research@example.com", "Accept": "application/json"}

    # Rate limit
    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        _last_request_time = time.time()
        if resp.status_code != 200:
            return False

        data = resp.json()
        entity_data = data.get("entities", {}).get(qid, {})
        claims = entity_data.get("claims", {})

        # Check P31 (instance of) claims
        p31_claims = claims.get("P31", [])
        for claim in p31_claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            target_id = value.get("id", "")
            if target_id in _COMPANY_P31_TYPES:
                return True

    except Exception as e:
        print(f"Wikidata P31 check error for {qid}: {e}", flush=True)

    return False
