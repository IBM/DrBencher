"""Financial reasoning templates: template definitions and selection functions.

Extracted from ``financial_drbencher.py`` to keep the main module focused on
QA generation, verification, and CLI logic.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

from .multiskill_template import _execute_computation_code
from .edgar_util import get_metric_history


# ===========================================================================
# Financial Templates
# ===========================================================================

FINANCIAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --- Level 1: Single Metric ---
    "gross_margin": {
        "level": 1,
        "type": "single",
        "label": "Gross Profit Margin",
        "required_metrics": ["revenue", "cost_of_revenue"],
        "code_template": "print(round(({revenue} - {cost_of_revenue}) / {revenue} * 100, 2))",
        "question_hint": "What is this company's gross profit margin (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "operating_margin": {
        "level": 1,
        "type": "single",
        "label": "Operating Margin",
        "required_metrics": ["revenue", "operating_income"],
        "code_template": "print(round({operating_income} / {revenue} * 100, 2))",
        "question_hint": "What is this company's operating margin (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "net_margin": {
        "level": 1,
        "type": "single",
        "label": "Net Profit Margin",
        "required_metrics": ["revenue", "net_income"],
        "code_template": "print(round({net_income} / {revenue} * 100, 2))",
        "question_hint": "What is this company's net profit margin (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "return_on_equity": {
        "level": 1,
        "type": "single",
        "label": "Return on Equity (ROE)",
        "required_metrics": ["net_income", "equity"],
        "code_template": "print(round({net_income} / {equity} * 100, 2))",
        "question_hint": "What is this company's ROE (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "return_on_assets": {
        "level": 1,
        "type": "single",
        "label": "Return on Assets (ROA)",
        "required_metrics": ["net_income", "total_assets"],
        "code_template": "print(round({net_income} / {total_assets} * 100, 2))",
        "question_hint": "What is this company's ROA (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "debt_to_equity": {
        "level": 1,
        "type": "single",
        "label": "Debt-to-Equity Ratio",
        "required_metrics": ["total_liabilities", "equity"],
        "code_template": "print(round({total_liabilities} / {equity}, 4))",
        "question_hint": "What is this company's debt-to-equity ratio for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "current_ratio": {
        "level": 1,
        "type": "single",
        "label": "Current Ratio",
        "required_metrics": ["current_assets", "current_liabilities"],
        "code_template": "print(round({current_assets} / {current_liabilities}, 4))",
        "question_hint": "What is this company's current ratio for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "asset_turnover": {
        "level": 1,
        "type": "single",
        "label": "Asset Turnover Ratio",
        "required_metrics": ["revenue", "total_assets_avg"],
        "code_template": "print(round({revenue} / {total_assets_avg}, 4))",
        "question_hint": "What is this company's asset turnover ratio (total revenue divided by average total assets) for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "equity_multiplier": {
        "level": 1,
        "type": "single",
        "label": "Equity Multiplier",
        "required_metrics": ["total_assets", "equity"],
        "code_template": "print(round({total_assets} / {equity}, 4))",
        "question_hint": "What is this company's equity multiplier for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "operating_expense_ratio": {
        "level": 1,
        "type": "single",
        "label": "Operating Expense Ratio",
        "required_metrics": ["revenue", "cost_of_revenue", "operating_income"],
        "code_template": (
            "opex = {revenue} - {cost_of_revenue} - {operating_income}\n"
            "print(round(opex / {revenue} * 100, 2))"
        ),
        "question_hint": "What is this company's operating expense ratio (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },

    # --- Level 1: Comparative ---
    "revenue_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "Revenue Ratio (B / A)",
        "required_metrics": ["revenue"],
        "code_template": "print(round({revenue_b} / {revenue_a}, 4))",
        "question_hint": "What is the ratio of the second company's revenue to the first's for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "margin_difference": {
        "level": 1,
        "type": "comparative",
        "label": "Gross Margin Difference",
        "required_metrics": ["revenue", "cost_of_revenue"],
        "code_template": (
            "gm_a = ({revenue_a} - {cost_of_revenue_a}) / {revenue_a} * 100\n"
            "gm_b = ({revenue_b} - {cost_of_revenue_b}) / {revenue_b} * 100\n"
            "print(round(abs(gm_b - gm_a), 2))"
        ),
        "question_hint": "What is the difference in gross margin (percentage points) between these two companies for fiscal year {fiscal_year}?",
        "answer_unit": "percentage points",
        "reasoning_depth": 3,
    },
    "roe_difference": {
        "level": 1,
        "type": "comparative",
        "label": "ROE Difference",
        "required_metrics": ["net_income", "equity"],
        "code_template": (
            "roe_a = {net_income_a} / {equity_a} * 100\n"
            "roe_b = {net_income_b} / {equity_b} * 100\n"
            "print(round(abs(roe_b - roe_a), 2))"
        ),
        "question_hint": "What is the difference in ROE (percentage points) between these two companies for fiscal year {fiscal_year}?",
        "answer_unit": "percentage points",
        "reasoning_depth": 3,
    },
    "asset_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "Total Asset Ratio (B / A)",
        "required_metrics": ["total_assets"],
        "code_template": "print(round({total_assets_b} / {total_assets_a}, 4))",
        "question_hint": "What is the ratio of the second company's total assets to the first's for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "eps_ratio": {
        "level": 1,
        "type": "comparative",
        "label": "EPS Ratio (B / A)",
        "required_metrics": ["eps"],
        "code_template": "print(round({eps_b} / {eps_a}, 4))",
        "question_hint": "What is the ratio of the second company's basic EPS to the first's for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },

    # --- Level 2: Temporal Trend ---
    "revenue_cagr_3yr": {
        "level": 2,
        "type": "temporal",
        "label": "3-Year Revenue CAGR",
        "required_metrics": ["revenue"],
        "years_needed": 4,
        "code_template": (
            "rev_start = {revenue_start}\n"
            "rev_end = {revenue_end}\n"
            "years = {num_years}\n"
            "cagr = ((rev_end / rev_start) ** (1 / years) - 1) * 100\n"
            "print(round(cagr, 2))"
        ),
        "question_hint": "What is this company's {num_years}-year revenue CAGR (%) from {start_year} to {end_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "net_income_cagr_3yr": {
        "level": 2,
        "type": "temporal",
        "label": "3-Year Net Income CAGR",
        "required_metrics": ["net_income"],
        "years_needed": 4,
        "code_template": (
            "ni_start = {net_income_start}\n"
            "ni_end = {net_income_end}\n"
            "years = {num_years}\n"
            "cagr = ((ni_end / ni_start) ** (1 / years) - 1) * 100\n"
            "print(round(cagr, 2))"
        ),
        "question_hint": "What is this company's {num_years}-year net income CAGR (%) from {start_year} to {end_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "margin_trend_direction": {
        "level": 2,
        "type": "temporal",
        "label": "Operating Margin Trend",
        "required_metrics": ["revenue", "operating_income"],
        "years_needed": 3,
        "code_template": (
            "margin_start = {operating_income_start} / {revenue_start} * 100\n"
            "margin_end = {operating_income_end} / {revenue_end} * 100\n"
            "change = margin_end - margin_start\n"
            "print(round(change, 2))"
        ),
        "question_hint": "How many percentage points did the operating margin change from {start_year} to {end_year}?",
        "answer_unit": "percentage points",
        "reasoning_depth": 3,
    },
    "asset_growth_rate": {
        "level": 2,
        "type": "temporal",
        "label": "Total Asset Growth Rate",
        "required_metrics": ["total_assets"],
        "years_needed": 3,
        "code_template": (
            "assets_start = {total_assets_start}\n"
            "assets_end = {total_assets_end}\n"
            "growth = (assets_end - assets_start) / assets_start * 100\n"
            "print(round(growth, 2))"
        ),
        "question_hint": "What was the total asset growth rate (%) from {start_year} to {end_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },

    # --- Level 2: Cross-Metric Composite ---
    "dupont_roe": {
        "level": 2,
        "type": "composite",
        "label": "DuPont ROE Decomposition",
        "required_metrics": ["net_income", "revenue", "total_assets", "equity"],
        "code_template": (
            "net_margin = {net_income} / {revenue}\n"
            "asset_turnover = {revenue} / {total_assets}\n"
            "equity_multiplier = {total_assets} / {equity}\n"
            "dupont_roe = net_margin * asset_turnover * equity_multiplier * 100\n"
            "print(round(dupont_roe, 2))"
        ),
        "question_hint": "Using the DuPont decomposition, what is this company's ROE (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 4,
    },
    "enterprise_value_proxy": {
        "level": 2,
        "type": "composite",
        "label": "Enterprise Value / Revenue",
        "required_metrics": ["total_assets", "cash", "revenue"],
        "code_template": (
            "ev = {total_assets} - {cash}\n"
            "ev_to_rev = ev / {revenue}\n"
            "print(round(ev_to_rev, 2))"
        ),
        "question_hint": "What is this company's enterprise value-to-revenue ratio (simplified EV proxy) for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 3,
    },
    "working_capital_to_revenue": {
        "level": 2,
        "type": "composite",
        "label": "Working Capital / Revenue",
        "required_metrics": ["current_assets", "current_liabilities", "revenue"],
        "code_template": (
            "wc = {current_assets} - {current_liabilities}\n"
            "wc_to_rev = wc / {revenue} * 100\n"
            "print(round(wc_to_rev, 2))"
        ),
        "question_hint": "What is this company's working capital as a percentage of revenue for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "cash_to_debt_ratio": {
        "level": 2,
        "type": "composite",
        "label": "Cash to Total Liabilities Ratio",
        "required_metrics": ["cash", "total_liabilities"],
        "code_template": (
            "ratio = {cash} / {total_liabilities}\n"
            "print(round(ratio, 4))"
        ),
        "question_hint": "What is this company's cash-to-total-liabilities ratio for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "rd_intensity": {
        "level": 2,
        "type": "composite",
        "label": "R&D Intensity (R&D / Revenue)",
        "required_metrics": ["rd_expense", "revenue"],
        "code_template": (
            "rd_intensity = {rd_expense} / {revenue} * 100\n"
            "print(round(rd_intensity, 2))"
        ),
        "question_hint": "What is this company's R&D intensity (R&D expense as % of revenue) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },

    # --- Level 1: Additional Single ---
    "interest_coverage_ratio": {
        "level": 1,
        "type": "single",
        "label": "Interest Coverage Ratio",
        "required_metrics": ["operating_income", "interest_expense"],
        "code_template": (
            "oi = {operating_income}\n"
            "ie = abs({interest_expense})\n"
            "icr = oi / ie if ie > 0 else 0\n"
            "print(round(icr, 2))"
        ),
        "question_hint": "What is this company's interest coverage ratio (operating income / interest expense) for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "inventory_turnover": {
        "level": 1,
        "type": "single",
        "label": "Inventory Turnover",
        "required_metrics": ["cost_of_revenue", "inventory"],
        "code_template": (
            "cogs = {cost_of_revenue}\n"
            "inv = {inventory}\n"
            "turnover = cogs / inv if inv > 0 else 0\n"
            "print(round(turnover, 2))"
        ),
        "question_hint": "What is this company's inventory turnover ratio (COGS / inventory) for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },

    # --- Level 2: Additional Composite ---
    "receivables_turnover": {
        "level": 1,
        "type": "single",
        "label": "Receivables Turnover",
        "required_metrics": ["revenue", "accounts_receivable"],
        "code_template": (
            "rev = {revenue}\n"
            "ar = {accounts_receivable}\n"
            "turnover = rev / ar if ar > 0 else 0\n"
            "print(round(turnover, 2))"
        ),
        "question_hint": "What is this company's receivables turnover ratio for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    "payables_turnover": {
        "level": 1,
        "type": "single",
        "label": "Payables Turnover",
        "required_metrics": ["cost_of_revenue", "accounts_payable"],
        "code_template": (
            "cogs = {cost_of_revenue}\n"
            "ap = {accounts_payable}\n"
            "turnover = cogs / ap if ap > 0 else 0\n"
            "print(round(turnover, 2))"
        ),
        "question_hint": "What is this company's payables turnover ratio for fiscal year {fiscal_year}?",
        "answer_unit": "times",
        "reasoning_depth": 2,
    },
    # --- New Level 2: Cash Flow & Efficiency ---
    "free_cash_flow_margin": {
        "level": 2,
        "type": "single",
        "label": "Free Cash Flow Margin",
        "required_metrics": ["operating_cash_flow", "capex", "revenue"],
        "code_template": (
            "ocf = {operating_cash_flow}\n"
            "capex = abs({capex})\n"
            "fcf = ocf - capex\n"
            "fcf_margin = fcf / {revenue} * 100\n"
            "print(round(fcf_margin, 2))"
        ),
        "question_hint": "What is this company's free cash flow margin (%) for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 3,
    },
    "capex_to_revenue": {
        "level": 1,
        "type": "single",
        "label": "CapEx-to-Revenue Ratio",
        "required_metrics": ["capex", "revenue"],
        "code_template": (
            "capex = abs({capex})\n"
            "ratio = capex / {revenue} * 100\n"
            "print(round(ratio, 2))"
        ),
        "question_hint": "What is this company's capital expenditure as a percentage of revenue for fiscal year {fiscal_year}?",
        "answer_unit": "%",
        "reasoning_depth": 2,
    },
    "operating_cash_flow_ratio": {
        "level": 1,
        "type": "single",
        "label": "Operating Cash Flow Ratio",
        "required_metrics": ["operating_cash_flow", "current_liabilities"],
        "code_template": (
            "ratio = {operating_cash_flow} / {current_liabilities}\n"
            "print(round(ratio, 4))"
        ),
        "question_hint": "What is this company's operating cash flow ratio (OCF / current liabilities) for fiscal year {fiscal_year}?",
        "answer_unit": "ratio",
        "reasoning_depth": 2,
    },
    "cash_conversion_cycle": {
        "level": 2,
        "type": "composite",
        "label": "Cash Conversion Cycle",
        "required_metrics": ["accounts_receivable", "inventory", "accounts_payable", "revenue", "cost_of_revenue"],
        "code_template": (
            "dso = {accounts_receivable} / ({revenue} / 365)\n"
            "dio = {inventory} / ({cost_of_revenue} / 365)\n"
            "dpo = {accounts_payable} / ({cost_of_revenue} / 365)\n"
            "ccc = dso + dio - dpo\n"
            "print(round(ccc, 2))"
        ),
        "question_hint": "What is this company's cash conversion cycle (in days) for fiscal year {fiscal_year}?",
        "answer_unit": "days",
        "reasoning_depth": 4,
    },
}


# Templates that produce nonsensical results when equity is negative
_NEGATIVE_EQUITY_SKIP_TEMPLATES = {
    "return_on_equity", "debt_to_equity", "equity_multiplier", "dupont_roe",
}


# ===========================================================================
# Template Selection Functions
# ===========================================================================

def select_financial_template(financials: dict, used_templates: set,
                              sector: str = None) -> Optional[Dict[str, Any]]:
    """Select a template that the company's data can support.

    Args:
        financials: Company financials dict from get_company_financials().
        used_templates: Set of template IDs already used.
        sector: Current sector (for weighting template selection).

    Returns:
        Dict with template info and computed gold answer, or None.
    """
    # Shuffle templates for variety
    template_ids = list(FINANCIAL_TEMPLATES.keys())
    random.shuffle(template_ids)

    fiscal_year = financials.get("fiscal_year", 2023)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = FINANCIAL_TEMPLATES[tmpl_id]

        # Check all required metrics are available
        required = tmpl["required_metrics"]
        if tmpl["type"] == "comparative":
            # Comparative needs two companies — skip here, handle separately
            continue
        if tmpl["type"] == "temporal":
            # Temporal needs multi-year data — skip here, handle separately
            continue

        all_available = True
        params = {}
        for metric in required:
            val = financials.get(metric)
            if val is None or val == 0:
                all_available = False
                break
            params[metric] = val

        if not all_available:
            continue

        # Skip equity-based templates when equity is negative
        if tmpl_id in _NEGATIVE_EQUITY_SKIP_TEMPLATES and params.get("equity", 0) < 0:
            continue

        # Skip templates when COGS / revenue < 0.15 (likely mis-reported data)
        if "cost_of_revenue" in required and params.get("revenue", 0) > 0:
            if params["cost_of_revenue"] / params["revenue"] < 0.15:
                continue

        # Fill code template and compute gold answer
        params["fiscal_year"] = fiscal_year
        code = tmpl["code_template"].format(**params)
        gold_answer = _execute_computation_code(code)
        if gold_answer is None:
            continue

        # Validate gold answer is a reasonable number
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
            "fiscal_year": fiscal_year,
            "question_hint": tmpl["question_hint"].format(fiscal_year=fiscal_year),
        }

    return None


def select_comparative_template(financials_a: dict, financials_b: dict,
                                 used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a comparative template for two companies.

    Returns:
        Template context dict with gold answer, or None.
    """
    template_ids = [tid for tid, t in FINANCIAL_TEMPLATES.items()
                    if t["type"] == "comparative"]
    random.shuffle(template_ids)

    fiscal_year = financials_a.get("fiscal_year", 2023)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = FINANCIAL_TEMPLATES[tmpl_id]
        required = tmpl["required_metrics"]

        # Check both companies have required metrics
        params = {}
        all_available = True
        for metric in required:
            val_a = financials_a.get(metric)
            val_b = financials_b.get(metric)
            if val_a is None or val_b is None or val_a == 0 or val_b == 0:
                all_available = False
                break
            params[f"{metric}_a"] = val_a
            params[f"{metric}_b"] = val_b

        if not all_available:
            continue

        # Skip ROE difference when either company has negative equity
        if tmpl_id == "roe_difference":
            if params.get("equity_a", 0) < 0 or params.get("equity_b", 0) < 0:
                continue

        # Skip templates when COGS / revenue < 0.15 for either company
        if "cost_of_revenue" in required:
            rev_a = params.get("revenue_a", 0)
            rev_b = params.get("revenue_b", 0)
            cor_a = params.get("cost_of_revenue_a", 0)
            cor_b = params.get("cost_of_revenue_b", 0)
            if (rev_a > 0 and cor_a / rev_a < 0.15) or \
               (rev_b > 0 and cor_b / rev_b < 0.15):
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

        # Skip ratio templates that produce negative values (misleading wording)
        if tmpl_id in ("revenue_ratio", "asset_ratio", "eps_ratio") and gold_float < 0:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "fiscal_year": fiscal_year,
            "question_hint": tmpl["question_hint"].format(fiscal_year=fiscal_year),
        }

    return None


def select_temporal_template(ticker: str, financials: dict,
                              used_templates: set) -> Optional[Dict[str, Any]]:
    """Select a temporal trend template requiring multi-year data.

    Returns:
        Template context dict with gold answer, or None.
    """
    template_ids = [tid for tid, t in FINANCIAL_TEMPLATES.items()
                    if t["type"] == "temporal"]
    random.shuffle(template_ids)

    fiscal_year = financials.get("fiscal_year", 2023)

    for tmpl_id in template_ids:
        if tmpl_id in used_templates:
            continue

        tmpl = FINANCIAL_TEMPLATES[tmpl_id]
        years_needed = tmpl.get("years_needed", 3)
        required = tmpl["required_metrics"]

        start_year = fiscal_year - years_needed + 1
        end_year = fiscal_year
        num_years = end_year - start_year

        # Fetch multi-year data
        params = {}
        all_available = True
        for metric in required:
            history = get_metric_history(ticker, metric, start_year, end_year)
            if history is None or start_year not in history or end_year not in history:
                all_available = False
                break
            params[f"{metric}_start"] = history[start_year]
            params[f"{metric}_end"] = history[end_year]

            # For margin trend, also need intermediate years
            if "margin" in tmpl_id:
                for y in range(start_year, end_year + 1):
                    if y in history:
                        params[f"{metric}_y{y - start_year + 1}"] = history[y]

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
            # For CAGR, require positive start value (negative makes no sense)
            if "cagr" in tmpl_id and gold_float < -90:
                continue
        except ValueError:
            continue

        return {
            "template_id": tmpl_id,
            "template": tmpl,
            "params": params,
            "code": code,
            "gold_answer": gold_answer,
            "fiscal_year": fiscal_year,
            "start_year": start_year,
            "end_year": end_year,
            "num_years": num_years,
            "question_hint": tmpl["question_hint"].format(
                fiscal_year=fiscal_year, start_year=start_year,
                end_year=end_year, num_years=num_years,
            ),
        }

    return None
