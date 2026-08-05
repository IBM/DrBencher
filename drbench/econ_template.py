# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Economics reasoning templates: template definitions and selection functions.

Mirrors ``financial_template.py`` — extracted from ``economics_drbencher.py``
to keep the main module focused on QA generation, verification, and CLI logic.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from .multiskill_template import _execute_computation_code
from .econ_util import get_indicator_history


# ===========================================================================
# Economics Templates
# ===========================================================================

ECONOMICS_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # --- Single-metric (8) ---

    "gdp_per_capita_verify": {
        "level": 1,
        "type": "single",
        "label": "GDP per Capita Verification",
        "required_indicators": ["gdp", "population"],
        "code_template": "print(round({gdp} / {population}, 2))",
        "question_hint": "What is this country's GDP per capita (US$) for {data_year}?",
        "answer_unit": "US$",
        "reasoning_depth": 2,
    },
    "population_density_verify": {
        "level": 1,
        "type": "single",
        "label": "Population Density Verification",
        "required_indicators": ["population", "surface_area"],
        "code_template": "print(round({population} / {surface_area}, 2))",
        "question_hint": "What is this country's population density (people per km²) for {data_year}?",
        "answer_unit": "people/km²",
        "reasoning_depth": 2,
    },
    "health_spending_per_capita": {
        "level": 1,
        "type": "single",
        "label": "Health Spending per Capita",
        "required_indicators": ["health_expenditure_pct_gdp", "gdp", "population"],
        "code_template": "print(round({health_expenditure_pct_gdp} / 100 * {gdp} / {population}, 2))",
        "question_hint": "What is this country's health expenditure per capita (US$) for {data_year}?",
        "answer_unit": "US$",
        "reasoning_depth": 3,
    },
    "education_spending_per_capita": {
        "level": 1,
        "type": "single",
        "label": "Education Spending per Capita",
        "required_indicators": ["education_expenditure_pct_gdp", "gdp", "population"],
        "code_template": "print(round({education_expenditure_pct_gdp} / 100 * {gdp} / {population}, 2))",
        "question_hint": "What is this country's government education expenditure per capita (US$) for {data_year}?",
        "answer_unit": "US$",
        "reasoning_depth": 3,
    },
    "trade_to_gdp": {
        "level": 1,
        "type": "single",
        "label": "Trade-to-GDP Ratio",
        "required_indicators": ["trade_pct_gdp"],
        "code_template": "print(round({trade_pct_gdp}, 2))",
        "question_hint": "What is this country's trade as a percentage of GDP for {data_year}?",
        "answer_unit": "%",
        "reasoning_depth": 1,
    },
    "fdi_pct_verify": {
        "level": 1,
        "type": "single",
        "label": "FDI as % of GDP",
        "required_indicators": ["fdi_pct_gdp"],
        "code_template": "print(round({fdi_pct_gdp}, 2))",
        "question_hint": "What is the net FDI inflow as a percentage of GDP for this country in {data_year}?",
        "answer_unit": "%",
        "reasoning_depth": 1,
    },
    "co2_per_unit_gdp": {
        "level": 1,
        "type": "single",
        "label": "CO2 per Unit GDP",
        "required_indicators": ["co2_per_capita", "gdp_per_capita"],
        "code_template": "print(round({co2_per_capita} / {gdp_per_capita} * 1000, 4))",
        "question_hint": "What is this country's CO2 emissions per $1000 of GDP per capita for {data_year}?",
        "answer_unit": "metric tons per $1000 GDP/capita",
        "reasoning_depth": 2,
    },
    "debt_to_gdp": {
        "level": 1,
        "type": "single",
        "label": "External Debt to GNI",
        "required_indicators": ["external_debt_pct_gni"],
        "code_template": "print(round({external_debt_pct_gni}, 2))",
        "question_hint": "What is this country's external debt as a percentage of GNI for {data_year}?",
        "answer_unit": "%",
        "reasoning_depth": 1,
    },

    # --- Comparative (6) ---

    "gdp_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "GDP Ratio (B / A)",
        "required_indicators": ["gdp"],
        "code_template": "print(round({gdp_b} / {gdp_a}, 4))",
        "question_hint": "What is the ratio of the second country's GDP to the first's for {data_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "population_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "Population Ratio (B / A)",
        "required_indicators": ["population"],
        "code_template": "print(round({population_b} / {population_a}, 4))",
        "question_hint": "What is the ratio of the second country's population to the first's for {data_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "life_expectancy_diff": {
        "level": 1,
        "type": "comparative",
        "label": "Life Expectancy Difference",
        "required_indicators": ["life_expectancy"],
        "code_template": "print(round(abs({life_expectancy_b} - {life_expectancy_a}), 2))",
        "question_hint": "What is the difference in life expectancy (years) between these two countries for {data_year}?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "gdp_per_capita_diff": {
        "level": 1,
        "type": "comparative",
        "label": "GDP per Capita Difference",
        "required_indicators": ["gdp", "population"],
        "code_template": (
            "gpc_a = {gdp_a} / {population_a}\n"
            "gpc_b = {gdp_b} / {population_b}\n"
            "print(round(abs(gpc_b - gpc_a), 2))"
        ),
        "question_hint": "What is the difference in GDP per capita (US$) between these two countries for {data_year}?",
        "answer_unit": "US$",
        "reasoning_depth": 3,
    },
    "co2_per_capita_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "CO2 per Capita Ratio (B / A)",
        "required_indicators": ["co2_per_capita"],
        "code_template": "print(round({co2_per_capita_b} / {co2_per_capita_a}, 4))",
        "question_hint": "What is the ratio of the second country's CO2 per capita to the first's for {data_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "trade_openness_diff": {
        "level": 1,
        "type": "comparative",
        "label": "Trade Openness Difference",
        "required_indicators": ["trade_pct_gdp"],
        "code_template": "print(round(abs({trade_pct_gdp_b} - {trade_pct_gdp_a}), 2))",
        "question_hint": "What is the difference in trade openness (% of GDP) between these two countries for {data_year}?",
        "answer_unit": "percentage points",
        "reasoning_depth": 2,
    },

    # --- Temporal (5) ---

    "gdp_cagr_5yr": {
        "level": 2,
        "type": "temporal",
        "label": "5-Year GDP CAGR",
        "required_indicators": ["gdp"],
        "years_needed": 6,
        "code_template": (
            "gdp_start = {gdp_start}\n"
            "gdp_end = {gdp_end}\n"
            "years = {num_years}\n"
            "cagr = ((gdp_end / gdp_start) ** (1 / years) - 1) * 100\n"
            "print(round(cagr, 2))"
        ),
        "question_hint": "What is this country's {num_years}-year GDP CAGR (%) from {start_year} to {end_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "population_growth_10yr": {
        "level": 2,
        "type": "temporal",
        "label": "10-Year Population Growth Rate",
        "required_indicators": ["population"],
        "years_needed": 11,
        "code_template": (
            "pop_start = {population_start}\n"
            "pop_end = {population_end}\n"
            "growth = (pop_end - pop_start) / pop_start * 100\n"
            "print(round(growth, 2))"
        ),
        "question_hint": "What was this country's total population percentage change (%) from {start_year} to {end_year}, computed as (pop_end − pop_start) / pop_start × 100?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "life_expectancy_change_decade": {
        "level": 2,
        "type": "temporal",
        "label": "Decade Life Expectancy Change",
        "required_indicators": ["life_expectancy"],
        "years_needed": 11,
        "code_template": (
            "le_start = {life_expectancy_start}\n"
            "le_end = {life_expectancy_end}\n"
            "change = le_end - le_start\n"
            "print(round(change, 2))"
        ),
        "question_hint": "How many years did life expectancy change from {start_year} to {end_year} for this country?",
        "answer_unit": "years",
        "reasoning_depth": 2,
    },
    "average_inflation_5yr": {
        "level": 2,
        "type": "temporal",
        "label": "5-Year Average Inflation",
        "required_indicators": ["inflation"],
        "years_needed": 5,
        "code_template": (
            "values = [{inflation_values}]\n"
            "avg = sum(values) / len(values)\n"
            "print(round(avg, 2))"
        ),
        "question_hint": "What was the average annual inflation rate (%) for this country from {start_year} to {end_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "trade_openness_change_5yr": {
        "level": 2,
        "type": "temporal",
        "label": "5-Year Trade Openness Change",
        "required_indicators": ["trade_pct_gdp"],
        "years_needed": 6,
        "code_template": (
            "trade_start = {trade_pct_gdp_start}\n"
            "trade_end = {trade_pct_gdp_end}\n"
            "change = trade_end - trade_start\n"
            "print(round(change, 2))"
        ),
        "question_hint": "How many percentage points did trade openness change from {start_year} to {end_year}?",
        "answer_unit": "percentage points",
        "reasoning_depth": 2,
    },

    # --- Composite (6) ---

    "economic_complexity_proxy": {
        "level": 2,
        "type": "composite",
        "label": "Economic Complexity Proxy",
        "required_indicators": ["trade_pct_gdp", "gdp_per_capita", "fdi_pct_gdp"],
        "code_template": (
            "trade = {trade_pct_gdp}\n"
            "gpc = {gdp_per_capita}\n"
            "fdi = {fdi_pct_gdp}\n"
            "# Proxy: trade_openness * log10(gdp_pc) + fdi_share\n"
            "import math\n"
            "proxy = trade * math.log10(max(gpc, 1)) + fdi\n"
            "print(round(proxy, 2))"
        ),
        "question_hint": "What is this country's economic complexity proxy score (trade_openness * log10(GDP_per_capita) + FDI_pct) for {data_year}?",
        "answer_unit": "index",
        "reasoning_depth": 3,
    },
    "health_efficiency": {
        "level": 2,
        "type": "composite",
        "label": "Health Spending Efficiency",
        "required_indicators": ["life_expectancy", "health_expenditure_pct_gdp", "gdp_per_capita"],
        "code_template": (
            "le = {life_expectancy}\n"
            "health_pct = {health_expenditure_pct_gdp}\n"
            "gpc = {gdp_per_capita}\n"
            "health_pc = health_pct / 100 * gpc\n"
            "efficiency = le / (health_pc / 1000) if health_pc > 0 else 0\n"
            "print(round(efficiency, 2))"
        ),
        "question_hint": "What is this country's health spending efficiency (life expectancy per $1000 health spending per capita) for {data_year}?",
        "answer_unit": "years per $1000",
        "reasoning_depth": 4,
    },
    "fiscal_pressure": {
        "level": 2,
        "type": "composite",
        "label": "Fiscal Pressure Index",
        "required_indicators": ["health_expenditure_pct_gdp", "education_expenditure_pct_gdp", "unemployment"],
        "code_template": (
            "health = {health_expenditure_pct_gdp}\n"
            "edu = {education_expenditure_pct_gdp}\n"
            "unemp = {unemployment}\n"
            "fiscal_pressure = health + edu + unemp\n"
            "print(round(fiscal_pressure, 2))"
        ),
        "question_hint": "What is this country's fiscal pressure index (health + education expenditure % GDP + unemployment rate) for {data_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "human_capital_proxy": {
        "level": 2,
        "type": "composite",
        "label": "Human Capital Proxy",
        "required_indicators": ["life_expectancy", "education_expenditure_pct_gdp", "gdp_per_capita"],
        "code_template": (
            "le = {life_expectancy}\n"
            "edu = {education_expenditure_pct_gdp}\n"
            "gpc = {gdp_per_capita}\n"
            "import math\n"
            "hc = le * edu * math.log10(max(gpc, 1))\n"
            "print(round(hc, 2))"
        ),
        "question_hint": "What is this country's human capital proxy (life_expectancy × government_education_expenditure_pct_of_GDP × log10(GDP_per_capita)) for {data_year}?",
        "answer_unit": "index",
        "reasoning_depth": 3,
    },
    "carbon_intensity": {
        "level": 2,
        "type": "composite",
        "label": "Carbon Intensity of GDP",
        "required_indicators": ["co2_per_capita", "gdp_per_capita"],
        "code_template": (
            "co2 = {co2_per_capita}\n"
            "gpc = {gdp_per_capita}\n"
            "intensity = co2 / gpc * 1e6 if gpc > 0 else 0\n"
            "print(round(intensity, 2))"
        ),
        "question_hint": "What is this country's carbon intensity (metric tons CO2 per million US$ GDP per capita) for {data_year}?",
        "answer_unit": "metric tons per M$ GDP/cap",
        "reasoning_depth": 2,
    },
    "investment_attractiveness": {
        "level": 2,
        "type": "composite",
        "label": "Investment Attractiveness Score",
        "required_indicators": ["fdi_pct_gdp", "trade_pct_gdp", "inflation", "gdp_per_capita"],
        "code_template": (
            "fdi = {fdi_pct_gdp}\n"
            "trade = {trade_pct_gdp}\n"
            "inf = {inflation}\n"
            "gpc = {gdp_per_capita}\n"
            "import math\n"
            "# Score = FDI + trade/10 + log10(GPC) - abs(inflation)/5\n"
            "score = fdi + trade / 10 + math.log10(max(gpc, 1)) - abs(inf) / 5\n"
            "print(round(score, 2))"
        ),
        "question_hint": "What is this country's investment attractiveness score for {data_year}? The formula is: FDI net inflows (% of GDP) + trade openness (% of GDP) / 10 + log₁₀(GDP per capita in current US$) − |inflation rate (%)| / 5. All inputs are percentages or per-capita US$ from World Bank data.",
        "answer_unit": "index",
        "reasoning_depth": 3,
    },
}


# ===========================================================================
# Template Selection Functions
# ===========================================================================

def select_economics_template(indicators: dict, used_templates: set,
                              region: str = None) -> Optional[Dict[str, Any]]:
    """Select a single/composite template that the country's data can support.

    Args:
        indicators: Country indicators dict from fetch_country_indicators().
        used_templates: Set of template IDs already used.
        region: Current region (for potential weighting).

    Returns:
        Dict with template info and computed gold answer, or None.
    """
    template_ids = list(ECONOMICS_TEMPLATES.keys())
    random.shuffle(template_ids)

    data_year = indicators.get("data_year", 2022)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = ECONOMICS_TEMPLATES[tmpl_id]

        if tmpl["type"] in ("comparative", "temporal"):
            continue

        required = tmpl["required_indicators"]
        all_available = True
        params = {}
        for ind in required:
            val = indicators.get(ind)
            if val is None or val == 0:
                all_available = False
                break
            params[ind] = val

        if not all_available:
            continue

        params["data_year"] = data_year
        code = tmpl["code_template"].format(**params)
        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "data_year": data_year,
            "question_hint": tmpl["question_hint"].format(data_year=data_year),
        }

    return None


def select_comparative_template(indicators_a: dict, indicators_b: dict,
                                used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a comparative template for two countries.

    Both countries must share the same data_year (enforced by caller).

    Returns:
        Template context dict with gold answer, or None.
    """
    template_ids = [tid for tid, t in ECONOMICS_TEMPLATES.items()
                    if t["type"] == "comparative"]
    random.shuffle(template_ids)

    data_year_a = indicators_a.get("data_year", 2022)
    data_year_b = indicators_b.get("data_year", 2022)
    if data_year_a != data_year_b:
        return None
    data_year = data_year_a

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = ECONOMICS_TEMPLATES[tmpl_id]
        required = tmpl["required_indicators"]

        params = {}
        all_available = True
        for ind in required:
            val_a = indicators_a.get(ind)
            val_b = indicators_b.get(ind)
            if val_a is None or val_b is None or val_a == 0 or val_b == 0:
                all_available = False
                break
            params[f"{ind}_a"] = val_a
            params[f"{ind}_b"] = val_b

        if not all_available:
            continue

        code = tmpl["code_template"].format(**params)
        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
        except ValueError:
            continue

        # Skip ratio templates that produce negative values
        if tmpl_id in ("gdp_ratio", "population_ratio", "co2_per_capita_ratio") and gold_float < 0:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "data_year": data_year,
            "question_hint": tmpl["question_hint"].format(data_year=data_year),
        }

    return None


def select_temporal_template(iso3: str, indicators: dict,
                             used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a temporal trend template requiring multi-year data.

    Returns:
        Template context dict with gold answer, or None.
    """
    template_ids = [tid for tid, t in ECONOMICS_TEMPLATES.items()
                    if t["type"] == "temporal"]
    random.shuffle(template_ids)

    data_year = indicators.get("data_year", 2022)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = ECONOMICS_TEMPLATES[tmpl_id]
        years_needed = tmpl.get("years_needed", 6)
        required = tmpl["required_indicators"]

        start_year = data_year - years_needed + 1
        end_year = data_year
        num_years = end_year - start_year

        # Fetch multi-year data
        params = {}
        all_available = True
        for ind in required:
            history = get_indicator_history(iso3, ind, start_year, end_year)
            if history is None or start_year not in history or end_year not in history:
                all_available = False
                break
            params[f"{ind}_start"] = history[start_year]
            params[f"{ind}_end"] = history[end_year]

            # For average templates, collect all values
            if "average" in tmpl_id:
                sorted_years = sorted(y for y in history.keys()
                                      if start_year <= y <= end_year)
                values = [history[y] for y in sorted_years]
                if len(values) < 3:
                    all_available = False
                    break
                params[f"{ind}_values"] = ", ".join(str(v) for v in values)

        if not all_available:
            continue

        params["start_year"] = start_year
        params["end_year"] = end_year
        params["num_years"] = num_years

        code = tmpl["code_template"].format(**params)
        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        try:
            gold_float = float(gold_answer)
            if math.isnan(gold_float) or math.isinf(gold_float):
                continue
            # For CAGR, reject extreme values
            if "cagr" in tmpl_id and (gold_float < -50 or gold_float > 100):
                continue
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "data_year": data_year,
            "start_year": start_year,
            "end_year": end_year,
            "num_years": num_years,
            "question_hint": tmpl["question_hint"].format(
                data_year=data_year, start_year=start_year,
                end_year=end_year, num_years=num_years,
            ),
        }

    return None
