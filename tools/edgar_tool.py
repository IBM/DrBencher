"""SEC EDGAR Financial Tool for V2 agentic verification.

Provides a Harmony-compatible tool that wraps the SEC EDGAR XBRL API,
allowing V2 verification agents to look up real financial data.

Actions:
    edgar.search_company(query)              -> list of {ticker, name, cik}
    edgar.get_financials(ticker, year)        -> dict of financial metrics
    edgar.get_metric(ticker, metric, year)    -> float value
    edgar.compare_companies(tickers, metric, year) -> ranked list
    edgar.get_metric_history(ticker, metric, start_year, end_year) -> time series
"""

import json
import logging
from typing import AsyncIterator, Optional, Dict, Any

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
    ToolDescription,
    ToolNamespaceConfig,
)

try:
    from gpt_oss.tools.simple_browser.simple_browser_tool import (
        SimpleBrowserTool,
        maybe_get_function_args,
    )
    _HAS_BROWSER_BASE = True
except ImportError:
    _HAS_BROWSER_BASE = False

from drbench.edgar_util import (
    get_company_tickers,
    get_cik_for_ticker,
    get_company_name,
    get_company_financials,
    extract_financial_metric,
    fetch_company_facts,
    get_metric_history,
    compare_companies_metric,
    FINANCIAL_CONCEPTS,
)

logger = logging.getLogger(__name__)


class EdgarFinancialTool:
    """Tool for V2 agentic verification — wraps SEC EDGAR XBRL API.

    Provides financial data lookup capabilities for the verification agent.
    """

    name = "edgar"

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Harmony tool configuration for the EDGAR financial tool."""
        return ToolNamespaceConfig(
            name="edgar",
            description=(
                "SEC EDGAR financial data tool. Look up company financial data "
                "from SEC XBRL filings."
            ),
            tools=[
                ToolDescription(
                    name="search_company",
                    description="Search for a company by name or ticker",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (company name or ticker symbol)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_financials",
                    description="Get all financial metrics for a company and year",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL')",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Fiscal year (e.g., 2023)",
                            },
                        },
                        "required": ["ticker"],
                    },
                ),
                ToolDescription(
                    name="get_metric",
                    description=(
                        "Get a specific financial metric for a company and year. "
                        "Available metrics: revenue, cost_of_revenue, net_income, "
                        "total_assets, total_liabilities, equity, operating_income, "
                        "cash, eps, current_assets, current_liabilities, rd_expense"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL')",
                            },
                            "metric": {
                                "type": "string",
                                "description": "Financial metric name",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Fiscal year (e.g., 2023)",
                            },
                        },
                        "required": ["ticker", "metric"],
                    },
                ),
                ToolDescription(
                    name="compare_companies",
                    description="Compare a financial metric across multiple companies",
                    parameters={
                        "type": "object",
                        "properties": {
                            "tickers": {
                                "type": "string",
                                "description": "Comma-separated ticker symbols",
                            },
                            "metric": {
                                "type": "string",
                                "description": "Financial metric to compare",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Fiscal year (e.g., 2023)",
                            },
                        },
                        "required": ["tickers", "metric"],
                    },
                ),
                ToolDescription(
                    name="get_metric_history",
                    description="Get a financial metric's history across a range of years",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL')",
                            },
                            "metric": {
                                "type": "string",
                                "description": "Financial metric name",
                            },
                            "start_year": {
                                "type": "integer",
                                "description": "Start year",
                            },
                            "end_year": {
                                "type": "integer",
                                "description": "End year",
                            },
                        },
                        "required": ["ticker", "metric", "start_year", "end_year"],
                    },
                ),
            ],
        )

    def _make_response(self, content: str, recipient: str) -> Message:
        """Create a tool response message."""
        return Message(
            author=Author(role=Role.TOOL, name=self.name),
            content=[TextContent(text=content)],
        ).with_recipient("assistant")

    async def process(self, msg: Message) -> AsyncIterator[Message]:
        """Route tool calls to appropriate handler."""
        recipient = str(getattr(msg, 'recipient', ''))

        # Parse function name from recipient (e.g., "edgar.get_financials")
        parts = recipient.split(".", 1)
        if len(parts) == 2:
            func_name = parts[1]
        else:
            func_name = None

        # Extract text content from Message (content is a list of TextContent objects)
        raw_content = getattr(msg, 'content', None)
        content_text = ''
        if isinstance(raw_content, list):
            for item in raw_content:
                if hasattr(item, 'text'):
                    content_text = item.text
                    break
        elif isinstance(raw_content, str):
            content_text = raw_content
        else:
            content_text = str(raw_content) if raw_content else '{}'

        # Also try to get function from args
        try:
            args = json.loads(content_text)
            if isinstance(args, dict):
                func_name = func_name or args.get("function", "")
            else:
                args = {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        try:
            if func_name == "search_company":
                result = self._search_company(args.get("query", ""))
            elif func_name == "get_financials":
                result = self._get_financials(
                    args.get("ticker", ""),
                    int(args.get("year", 2023)),
                )
            elif func_name == "get_metric":
                result = self._get_metric(
                    args.get("ticker", ""),
                    args.get("metric", ""),
                    int(args.get("year", 2023)),
                )
            elif func_name == "compare_companies":
                tickers_str = args.get("tickers", "")
                tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
                result = self._compare_companies(
                    tickers,
                    args.get("metric", "revenue"),
                    int(args.get("year", 2023)),
                )
            elif func_name == "get_metric_history":
                result = self._get_metric_history(
                    args.get("ticker", ""),
                    args.get("metric", "revenue"),
                    int(args.get("start_year", 2020)),
                    int(args.get("end_year", 2023)),
                )
            else:
                result = {
                    "error": f"Unknown function: {func_name}. "
                    "Available: search_company, get_financials, get_metric, "
                    "compare_companies, get_metric_history"
                }
        except Exception as e:
            result = {"error": f"Error in {func_name}: {type(e).__name__}: {e}"}

        yield self._make_response(json.dumps(result, indent=2), recipient)

    def _search_company(self, query: str) -> Dict[str, Any]:
        """Search for companies by name or ticker."""
        if not query:
            return {"error": "query parameter required"}

        tickers = get_company_tickers()
        query_upper = query.upper().strip()
        query_lower = query.lower().strip()

        matches = []

        # Exact ticker match
        if query_upper in tickers:
            entry = tickers[query_upper]
            matches.append({
                "ticker": query_upper,
                "name": entry["title"],
                "cik": entry["cik"],
            })

        # Partial name/ticker match
        for ticker, entry in tickers.items():
            if len(matches) >= 10:
                break
            if ticker == query_upper:
                continue  # Already added
            if (query_lower in entry["title"].lower() or
                    query_upper in ticker):
                matches.append({
                    "ticker": ticker,
                    "name": entry["title"],
                    "cik": entry["cik"],
                })

        return {"results": matches[:10], "count": len(matches)}

    def _get_financials(self, ticker: str, year: int) -> Dict[str, Any]:
        """Get all financial metrics for a company and year."""
        if not ticker:
            return {"error": "ticker parameter required"}

        financials = get_company_financials(ticker, year)
        if financials is None:
            return {"error": f"Could not fetch financials for {ticker} FY{year}"}

        # Format large numbers for readability
        formatted = {}
        for key, val in financials.items():
            if isinstance(val, float) and abs(val) > 1e6 and key not in ("eps", "gross_margin",
                    "operating_margin", "net_margin", "roe", "roa", "debt_to_equity",
                    "current_ratio"):
                formatted[key] = val
                formatted[f"{key}_formatted"] = _format_usd(val)
            else:
                formatted[key] = val

        return formatted

    def _get_metric(self, ticker: str, metric: str, year: int) -> Dict[str, Any]:
        """Get a specific metric for a company and year."""
        if not ticker or not metric:
            return {"error": "ticker and metric parameters required"}

        if metric not in FINANCIAL_CONCEPTS:
            return {
                "error": f"Unknown metric: {metric}. Available: {list(FINANCIAL_CONCEPTS.keys())}"
            }

        cik = get_cik_for_ticker(ticker)
        if cik is None:
            return {"error": f"Ticker not found: {ticker}"}

        facts = fetch_company_facts(cik)
        if not facts:
            return {"error": f"Could not fetch facts for {ticker}"}

        val = extract_financial_metric(facts, metric, year)
        if val is None:
            return {"error": f"Metric {metric} not available for {ticker} FY{year}"}

        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "year": year,
            "value": val,
            "formatted": _format_usd(val) if metric != "eps" else f"${val:.2f}",
        }

    def _compare_companies(self, tickers: list, metric: str, year: int) -> Dict[str, Any]:
        """Compare a metric across multiple companies."""
        if not tickers:
            return {"error": "tickers parameter required (comma-separated)"}
        if metric not in FINANCIAL_CONCEPTS:
            return {
                "error": f"Unknown metric: {metric}. Available: {list(FINANCIAL_CONCEPTS.keys())}"
            }

        ranked = compare_companies_metric(tickers, metric, year)
        return {
            "metric": metric,
            "year": year,
            "ranking": ranked,
            "count": len(ranked),
        }

    def _get_metric_history(self, ticker: str, metric: str,
                            start_year: int, end_year: int) -> Dict[str, Any]:
        """Get metric history across years."""
        if not ticker or not metric:
            return {"error": "ticker and metric parameters required"}
        if metric not in FINANCIAL_CONCEPTS:
            return {
                "error": f"Unknown metric: {metric}. Available: {list(FINANCIAL_CONCEPTS.keys())}"
            }

        history = get_metric_history(ticker, metric, start_year, end_year)
        if history is None:
            return {"error": f"Could not fetch history for {ticker} {metric}"}

        return {
            "ticker": ticker.upper(),
            "metric": metric,
            "start_year": start_year,
            "end_year": end_year,
            "values": {str(y): v for y, v in sorted(history.items())},
            "count": len(history),
        }


def _format_usd(value: float) -> str:
    """Format a USD value with appropriate suffix."""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.2f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.2f}K"
    else:
        return f"{sign}${abs_val:.2f}"
