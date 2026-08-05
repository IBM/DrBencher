# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""World Bank WDI API utilities for economics benchmark.

Provides functions to fetch country economic data from the World Bank's free
Indicators API, with local JSON caching and rate limiting.

World Bank API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
Rate limit: None (public), but we add 0.1s delay to be polite.
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

WB_BASE = "https://api.worldbank.org/v2"
WB_HEADERS = {"User-Agent": "DrBencher research@example.com", "Accept": "application/json"}
WB_CACHE_DIR = "cache/worldbank_cache"

# Minimum delay between WB API requests (seconds)
_RATE_LIMIT_DELAY = 0.10
_last_request_time = 0.0


# ---------------------------------------------------------------------------
# Economic Indicator Definitions
# ---------------------------------------------------------------------------

ECONOMICS_INDICATORS: Dict[str, Dict[str, str]] = {
    "gdp": {
        "code": "NY.GDP.MKTP.CD",
        "label": "GDP (current US$)",
        "unit": "US$",
    },
    "gdp_per_capita": {
        "code": "NY.GDP.PCAP.CD",
        "label": "GDP per capita (current US$)",
        "unit": "US$",
    },
    "population": {
        "code": "SP.POP.TOTL",
        "label": "Population, total",
        "unit": "people",
    },
    "life_expectancy": {
        "code": "SP.DYN.LE00.IN",
        "label": "Life expectancy at birth (years)",
        "unit": "years",
    },
    "unemployment": {
        "code": "SL.UEM.TOTL.ZS",
        "label": "Unemployment (% of total labor force)",
        "unit": "%",
    },
    "inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "label": "Inflation, consumer prices (annual %)",
        "unit": "%",
    },
    "trade_pct_gdp": {
        "code": "NE.TRD.GNFS.ZS",
        "label": "Trade (% of GDP)",
        "unit": "%",
    },
    "health_expenditure_pct_gdp": {
        "code": "SH.XPD.CHEX.GD.ZS",
        "label": "Current health expenditure (% of GDP)",
        "unit": "%",
    },
    "education_expenditure_pct_gdp": {
        "code": "SE.XPD.TOTL.GD.ZS",
        "label": "Government expenditure on education (% of GDP)",
        "unit": "%",
    },
    "population_density": {
        "code": "EN.POP.DNST",
        "label": "Population density (people per sq. km)",
        "unit": "people/km²",
    },
    "surface_area": {
        "code": "AG.SRF.TOTL.K2",
        "label": "Surface area (sq. km)",
        "unit": "km²",
    },
    "gini": {
        "code": "SI.POV.GINI",
        "label": "Gini index",
        "unit": "index",
    },
    "fdi_pct_gdp": {
        "code": "BX.KLT.DINV.WD.GD.ZS",
        "label": "Foreign direct investment, net inflows (% of GDP)",
        "unit": "%",
    },
    "external_debt_pct_gni": {
        "code": "DT.DOD.DECT.GN.ZS",
        "label": "External debt stocks (% of GNI)",
        "unit": "%",
    },
    "co2_per_capita": {
        "code": "EN.ATM.CO2E.PC",
        "label": "CO2 emissions (metric tons per capita)",
        "unit": "metric tons",
    },
}

# Reverse lookup: WB indicator code -> our short key
_CODE_TO_KEY = {v["code"]: k for k, v in ECONOMICS_INDICATORS.items()}


# ---------------------------------------------------------------------------
# Country Universe (grouped by World Bank region)
# ---------------------------------------------------------------------------

COUNTRY_UNIVERSE_REGIONS: Dict[str, List[str]] = {
    "oecd": [
        "USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS", "KOR", "ITA", "ESP",
        "NLD", "CHE", "SWE", "NOR", "DNK", "FIN", "AUT", "BEL", "IRL", "NZL",
    ],
    "g20": [
        "USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS", "KOR", "ITA", "BRA",
        "MEX", "ARG", "IND", "CHN", "RUS", "ZAF", "SAU", "TUR", "IDN", "ARE",
    ],
    "brics_plus": [
        "BRA", "RUS", "IND", "CHN", "ZAF", "SAU", "ARE", "EGY", "ETH", "IRN",
        "ARG", "KAZ", "NGA", "THA", "VNM", "BGD", "IDN", "MEX", "TUR", "MYS",
    ],
    "opec_plus": [
        "SAU", "IRQ", "IRN", "KWT", "ARE", "VEN", "NGA", "DZA", "LBY", "AGO",
        "GAB", "GNQ", "COG", "RUS", "KAZ", "AZE", "BHR", "BRN", "MYS", "OMN",
    ],
    "eu_core": [
        "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "SWE", "DNK", "FIN",
        "IRL", "PRT", "CZE", "GRC", "ROU", "HUN", "POL", "BGR", "HRV", "SVK",
    ],
    "asean_plus": [
        "IDN", "THA", "MYS", "SGP", "PHL", "VNM", "MMR", "KHM", "LAO", "BRN",
        "JPN", "KOR", "CHN", "AUS", "NZL", "IND", "PAK", "BGD", "LKA", "NPL",
    ],
    # --- Income-based ---
    "high_income": [
        "USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS", "KOR", "SGP", "CHE",
        "NOR", "SWE", "DNK", "IRL", "NLD", "AUT", "BEL", "FIN", "ISR", "NZL",
    ],
    "upper_middle_income": [
        "CHN", "BRA", "MEX", "TUR", "THA", "MYS", "COL", "ZAF", "PER", "DOM",
        "ECU", "GTM", "JAM", "JOR", "BGR", "ROU", "KAZ", "ARG", "CRI", "MUS",
    ],
    "lower_middle_income": [
        "IND", "IDN", "BGD", "PAK", "VNM", "PHL", "NGA", "KEN", "GHA", "UKR",
        "EGY", "MAR", "TUN", "UZB", "KHM", "LKA", "SEN", "CIV", "CMR", "BOL",
    ],
    "low_income": [
        "ETH", "TZA", "UGA", "MOZ", "MDG", "ZMB", "ZWE", "RWA", "COD", "AFG",
        "NPL", "MMR", "MLI", "BFA", "NER", "TCD", "SLE", "LBR", "MWI", "BDI",
    ],
    # --- Sub-regional ---
    "west_africa": [
        "NGA", "GHA", "SEN", "CIV", "MLI", "BFA", "NER", "TGO", "BEN", "GIN",
        "SLE", "LBR", "GMB", "GNB", "CPV", "MRT", "CMR", "TCD", "GAB", "COG",
    ],
    "east_africa": [
        "KEN", "ETH", "TZA", "UGA", "RWA", "BDI", "SSD", "COD", "SOM", "ERI",
        "DJI", "SDN", "MOZ", "MDG", "MWI", "ZMB", "ZWE", "COM", "MUS", "SYC",
    ],
    "central_america_caribbean": [
        "GTM", "HND", "SLV", "NIC", "CRI", "PAN", "BLZ", "DOM", "HTI", "CUB",
        "JAM", "TTO", "BHS", "BRB", "GUY", "SUR", "ATG", "DMA", "GRD", "KNA",
    ],
    "central_eastern_europe": [
        "POL", "CZE", "HUN", "SVK", "SVN", "HRV", "SRB", "BIH", "MNE", "MKD",
        "ALB", "ROU", "BGR", "EST", "LVA", "LTU", "MDA", "BLR", "UKR", "GEO",
    ],
    "southern_africa": [
        "ZAF", "BWA", "NAM", "LSO", "SWZ", "MOZ", "ZWE", "ZMB", "MWI", "AGO",
        "COD", "TZA", "MDG", "COM", "MUS", "SYC", "RWA", "BDI", "GAB", "COG",
    ],
    "nordic_western_europe": [
        "NOR", "SWE", "DNK", "FIN", "ISL", "GBR", "IRL", "NLD", "BEL", "LUX",
        "CHE", "FRA", "DEU", "AUT", "PRT", "ESP", "ITA", "GRC", "CYP", "MLT",
    ],
    # --- Thematic ---
    "small_island_states": [
        "FJI", "PNG", "WSM", "TON", "VUT", "SLB", "KIR", "MHL", "MDV", "MUS",
        "SYC", "COM", "CPV", "TTO", "JAM", "BHS", "BRB", "ATG", "DMA", "GRD",
    ],
    "landlocked_developing": [
        "AFG", "BTN", "NPL", "KAZ", "UZB", "MNG", "LAO", "ETH", "UGA", "RWA",
        "BDI", "MLI", "BFA", "NER", "TCD", "ZMB", "ZWE", "MWI", "BOL", "PRY",
    ],
    "commodity_exporters": [
        "SAU", "RUS", "NGA", "AGO", "IRQ", "KWT", "QAT", "ARE", "VEN", "CHL",
        "PER", "ZMB", "BWA", "COD", "GHA", "CIV", "BRA", "AUS", "NOR", "KAZ",
    ],
    "emerging_markets": [
        "CHN", "IND", "BRA", "MEX", "IDN", "TUR", "ZAF", "THA", "MYS", "PHL",
        "COL", "PER", "CHL", "EGY", "PAK", "VNM", "BGD", "NGA", "KEN", "ARG",
    ],
}


# ---------------------------------------------------------------------------
# Wikidata Validation Types
# ---------------------------------------------------------------------------

_COUNTRY_P31_TYPES = {
    "Q3624078",  # sovereign state
    "Q6256",     # country
    "Q7275",     # state
    "Q1763527",  # constituent country
    "Q15634554", # partially recognized state
    "Q1520223",  # island country
    "Q46395",    # British Overseas Territory
    "Q112099",   # island state
}

# Module-level cache for resolved Wikidata IDs
_country_wikidata_cache: Dict[str, Optional[str]] = {}


# ---------------------------------------------------------------------------
# Caching & Rate Limiting
# ---------------------------------------------------------------------------

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(WB_CACHE_DIR, exist_ok=True)


def _cached_get(url: str, cache_key: str, max_age_hours: int = 168,
                params: Optional[dict] = None) -> Optional[Any]:
    """Fetch URL with local JSON cache and rate limiting.

    The World Bank API returns ``[metadata, data]`` 2-element arrays for
    indicator queries.  This function returns the raw parsed JSON (caller
    handles the structure).

    Args:
        url: URL to fetch.
        cache_key: Key for local cache filename (sanitized).
        max_age_hours: Maximum age of cached data in hours (default 1 week).
        params: Optional query parameters.

    Returns:
        Parsed JSON (list or dict), or None on failure.
    """
    global _last_request_time
    _ensure_cache_dir()

    safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', cache_key)
    cache_path = os.path.join(WB_CACHE_DIR, f"{safe_key}.json")

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
            response = requests.get(url, headers=WB_HEADERS, params=params, timeout=30)
            _last_request_time = time.time()

            if response.status_code == 200:
                data = response.json()
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                return data
            elif response.status_code in (429, 503):
                wait = (2 ** attempt) * 2 + random.random() * 2
                print(f"World Bank rate-limited ({response.status_code}) for {cache_key}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"World Bank API error {response.status_code} for {cache_key}", flush=True)
                return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.random() * 5
            print(f"World Bank request error for {cache_key}: {e}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"World Bank unexpected error for {cache_key}: {e}", flush=True)
            return None

    print(f"World Bank request failed after {max_retries} retries for {cache_key}", flush=True)
    return None


def _wb_paginated_get(url: str, cache_key: str, max_age_hours: int = 168,
                      params: Optional[dict] = None) -> Optional[List[dict]]:
    """Fetch all pages of a World Bank indicator query.

    WB responses are ``[{page, pages, total, ...}, [data_records]]``.
    This function handles pagination and returns the concatenated data records.
    """
    if params is None:
        params = {}
    params.setdefault("format", "json")
    params.setdefault("per_page", "500")

    page = 1
    all_data = []
    while True:
        params["page"] = str(page)
        page_key = f"{cache_key}__p{page}"
        raw = _cached_get(url, page_key, max_age_hours, params)
        if raw is None:
            break

        # WB API returns [metadata, data] or a message dict on error
        if isinstance(raw, list) and len(raw) == 2:
            meta = raw[0]
            data = raw[1]
            if data is None:
                break
            all_data.extend(data)
            total_pages = meta.get("pages", 1)
            if page >= total_pages:
                break
            page += 1
        elif isinstance(raw, list) and len(raw) == 1:
            # Error response
            break
        else:
            break

    return all_data if all_data else None


# ---------------------------------------------------------------------------
# Core API Functions
# ---------------------------------------------------------------------------

def get_country_info(iso3: str) -> Optional[dict]:
    """Fetch basic country info from WB API.

    Returns:
        Dict with id, name, region, incomeLevel, capitalCity, longitude, latitude.
    """
    url = f"{WB_BASE}/country/{iso3}"
    data = _wb_paginated_get(url, f"country_info_{iso3}")
    if data and len(data) > 0:
        return data[0]
    return None


def get_country_name(iso3: str) -> Optional[str]:
    """Look up country name for an ISO-3 code."""
    info = get_country_info(iso3)
    return info.get("name") if info else None


def fetch_country_indicators(iso3: str, year: int) -> Optional[Dict[str, Any]]:
    """Fetch all 15 economic indicators for a country and year.

    Args:
        iso3: ISO 3166-1 alpha-3 code (e.g., "USA").
        year: Data year (e.g., 2022).

    Returns:
        Dict with iso3, name, year, and all available indicator values.
        None if country not found.
    """
    info = get_country_info(iso3)
    if info is None:
        return None

    name = info.get("name", iso3)
    region = info.get("region", {}).get("value", "")
    capital = info.get("capitalCity", "")

    result: Dict[str, Any] = {
        "iso3": iso3.upper(),
        "name": name,
        "region": region,
        "capital": capital,
        "data_year": year,
    }

    # Fetch each indicator
    for key, ind_info in ECONOMICS_INDICATORS.items():
        code = ind_info["code"]
        url = f"{WB_BASE}/country/{iso3}/indicator/{code}"
        params = {"date": str(year), "format": "json"}
        data = _wb_paginated_get(url, f"ind_{iso3}_{code}_{year}", params=params)
        if data:
            for record in data:
                if record.get("value") is not None:
                    result[key] = float(record["value"])
                    break
            else:
                result[key] = None
        else:
            result[key] = None

    return result


def get_indicator_value(iso3: str, indicator_key: str, year: int) -> Optional[float]:
    """Fetch a single indicator value for a country and year.

    Args:
        iso3: ISO 3166-1 alpha-3 code.
        indicator_key: Key from ECONOMICS_INDICATORS (e.g., "gdp").
        year: Data year.

    Returns:
        Float value, or None if not available.
    """
    if indicator_key not in ECONOMICS_INDICATORS:
        return None

    code = ECONOMICS_INDICATORS[indicator_key]["code"]
    url = f"{WB_BASE}/country/{iso3}/indicator/{code}"
    params = {"date": str(year), "format": "json"}
    data = _wb_paginated_get(url, f"ind_{iso3}_{code}_{year}", params=params)
    if data:
        for record in data:
            if record.get("value") is not None:
                return float(record["value"])
    return None


def get_indicator_history(iso3: str, indicator_key: str,
                          start_year: int, end_year: int) -> Optional[Dict[int, float]]:
    """Fetch a time series of one indicator for a country.

    Args:
        iso3: ISO 3166-1 alpha-3 code.
        indicator_key: Key from ECONOMICS_INDICATORS.
        start_year: First year.
        end_year: Last year (inclusive).

    Returns:
        Dict mapping year to value, or None if not found.
    """
    if indicator_key not in ECONOMICS_INDICATORS:
        return None

    code = ECONOMICS_INDICATORS[indicator_key]["code"]
    url = f"{WB_BASE}/country/{iso3}/indicator/{code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json"}
    data = _wb_paginated_get(url, f"hist_{iso3}_{code}_{start_year}_{end_year}",
                             params=params)
    if not data:
        return None

    result = {}
    for record in data:
        yr = record.get("date")
        val = record.get("value")
        if yr is not None and val is not None:
            try:
                result[int(yr)] = float(val)
            except (ValueError, TypeError):
                pass

    return result if result else None


def compare_countries_indicator(iso3_list: List[str], indicator_key: str,
                                year: int) -> List[Dict[str, Any]]:
    """Fetch one indicator for multiple countries and return ranked list.

    Returns:
        List of {"iso3", "name", "value"} dicts, sorted descending by value.
    """
    results = []
    for iso3 in iso3_list:
        val = get_indicator_value(iso3, indicator_key, year)
        if val is not None:
            name = get_country_name(iso3) or iso3
            results.append({"iso3": iso3.upper(), "name": name, "value": val})

    results.sort(key=lambda x: x["value"], reverse=True)
    return results


def find_best_year(iso3: str, required_indicators: List[str],
                   preferred: int = 2022) -> Optional[int]:
    """Find the most recent year where all required indicators are available.

    Tries preferred year first, then falls back to year-1, year-2.

    Args:
        iso3: ISO 3166-1 alpha-3 code.
        required_indicators: List of indicator keys (e.g., ["gdp", "population"]).
        preferred: Preferred data year.

    Returns:
        Best year (int), or None if no year has all indicators.
    """
    for year in [preferred, preferred - 1, preferred - 2]:
        all_available = True
        for key in required_indicators:
            val = get_indicator_value(iso3, key, year)
            if val is None:
                all_available = False
                break
        if all_available:
            return year
    return None


# ---------------------------------------------------------------------------
# Wikipedia Country Clue Fetching
# ---------------------------------------------------------------------------

def fetch_country_clues(iso3: str) -> List[dict]:
    """Fetch descriptive clue facts from Wikipedia for a country.

    Extracts non-economic facts such as:
    - Geographic location and neighbors
    - Official language(s)
    - Capital city
    - Form of government
    - Major rivers/mountains/landmarks
    - Historical events

    Returns:
        List of fact dicts: {"fact_id", "fact", "topic", "source"}.
    """
    name = get_country_name(iso3)
    if not name:
        return []

    # Search Wikipedia for the country
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
    topics = ["description", "geography", "government", "culture", "history", "economy"]

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue
        if len(clues) >= 10:
            break

        # Assign a topic based on keywords
        sent_lower = sent.lower()
        if any(w in sent_lower for w in ["border", "coast", "island", "continent", "ocean", "river", "mountain"]):
            topic = "geography"
        elif any(w in sent_lower for w in ["government", "president", "parliament", "republic", "monarchy", "democratic"]):
            topic = "government"
        elif any(w in sent_lower for w in ["language", "religion", "culture", "ethnic"]):
            topic = "culture"
        elif any(w in sent_lower for w in ["independence", "colonial", "war", "founded", "empire", "century"]):
            topic = "history"
        elif any(w in sent_lower for w in ["economy", "gdp", "export", "trade", "industry", "agriculture"]):
            topic = "economy"
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
# Region Helpers
# ---------------------------------------------------------------------------

ECONOMICS_THEMES: Dict[str, Dict[str, Any]] = {
    "east_asia_pacific": {"min_countries": 15},
    "south_asia": {"min_countries": 6},
    "europe_central_asia": {"min_countries": 20},
    "sub_saharan_africa": {"min_countries": 15},
    "latin_america": {"min_countries": 15},
    "middle_east_north_africa": {"min_countries": 12},
    "north_america": {"min_countries": 2},
    # ---- NEW TOPICS (2026-03-16) ----
    "oecd": {"min_countries": 15},
    "g20": {"min_countries": 15},
    "brics_plus": {"min_countries": 15},
    "opec_plus": {"min_countries": 12},
    "eu_core": {"min_countries": 15},
    "asean_plus": {"min_countries": 15},
    "high_income": {"min_countries": 15},
    "upper_middle_income": {"min_countries": 15},
    "lower_middle_income": {"min_countries": 15},
    "low_income": {"min_countries": 12},
    "west_africa": {"min_countries": 12},
    "east_africa": {"min_countries": 12},
    "central_america_caribbean": {"min_countries": 12},
    "central_eastern_europe": {"min_countries": 15},
    "southern_africa": {"min_countries": 12},
    "nordic_western_europe": {"min_countries": 15},
    "small_island_states": {"min_countries": 10},
    "landlocked_developing": {"min_countries": 12},
    "commodity_exporters": {"min_countries": 15},
    "emerging_markets": {"min_countries": 15},
}


def get_region_countries(region: str) -> List[str]:
    """Return list of ISO-3 codes in the pre-built country universe for a region."""
    return COUNTRY_UNIVERSE_REGIONS.get(region, [])


# ---------------------------------------------------------------------------
# Wikidata QID Resolution for Countries
# ---------------------------------------------------------------------------

def resolve_country_wikidata_id(iso3: str) -> Optional[str]:
    """Resolve the Wikidata QID for a country given its ISO 3166-1 alpha-3 code.

    Uses SPARQL P298 (ISO 3166-1 alpha-3 code) lookup, then validates P31.

    Args:
        iso3: ISO 3166-1 alpha-3 code (e.g., "USA").

    Returns:
        Wikidata QID string (e.g., "Q30") or None if not found/validated.
    """
    iso3 = iso3.upper()
    if iso3 in _country_wikidata_cache:
        return _country_wikidata_cache[iso3]

    global _last_request_time

    # SPARQL query: find entity with P298 = iso3
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?item WHERE {{
      ?item wdt:P298 "{iso3}" .
    }} LIMIT 5
    """
    headers = {
        "User-Agent": "DrBencher research@example.com",
        "Accept": "application/json",
    }

    elapsed = time.time() - _last_request_time
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)

    try:
        resp = requests.get(sparql_url, params={"query": query}, headers=headers, timeout=15)
        _last_request_time = time.time()

        if resp.status_code != 200:
            _country_wikidata_cache[iso3] = None
            return None

        results = resp.json().get("results", {}).get("bindings", [])
        if not results:
            _country_wikidata_cache[iso3] = None
            return None

        # Check each candidate for country P31 type
        for r in results:
            uri = r.get("item", {}).get("value", "")
            qid = uri.split("/")[-1] if "/" in uri else ""
            if not qid.startswith("Q"):
                continue

            if _is_country_entity(qid):
                _country_wikidata_cache[iso3] = qid
                return qid

        # Fall back to first result
        uri = results[0].get("item", {}).get("value", "")
        qid = uri.split("/")[-1] if "/" in uri else ""
        if qid.startswith("Q"):
            _country_wikidata_cache[iso3] = qid
            return qid

    except Exception as e:
        print(f"Wikidata SPARQL error for {iso3}: {e}", flush=True)

    _country_wikidata_cache[iso3] = None
    return None


def _is_country_entity(qid: str) -> bool:
    """Check if a Wikidata entity is a country by checking P31 claims."""
    global _last_request_time

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    headers = {"User-Agent": "DrBencher research@example.com", "Accept": "application/json"}

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

        p31_claims = claims.get("P31", [])
        for claim in p31_claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            target_id = value.get("id", "")
            if target_id in _COUNTRY_P31_TYPES:
                return True

    except Exception as e:
        print(f"Wikidata P31 check error for {qid}: {e}", flush=True)

    return False
