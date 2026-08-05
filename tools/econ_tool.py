"""World Bank Economics Tool for V2 agentic verification.

Provides a Harmony-compatible tool that wraps the World Bank WDI API,
allowing V2 verification agents to look up real economic data.

Actions:
    worldbank.search_country(query)                          -> list of {iso3, name, region}
    worldbank.get_indicators(iso3, year)                     -> dict of economic indicators
    worldbank.get_indicator(iso3, indicator, year)            -> float value
    worldbank.compare_countries(iso3_list, indicator, year)   -> ranked list
    worldbank.get_indicator_history(iso3, indicator, start_year, end_year) -> time series
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

from drbench.econ_util import (
    get_country_info,
    get_country_name,
    fetch_country_indicators,
    get_indicator_value,
    get_indicator_history,
    compare_countries_indicator,
    ECONOMICS_INDICATORS,
    COUNTRY_UNIVERSE_REGIONS,
)

logger = logging.getLogger(__name__)


class WorldBankEconomicsTool:
    """Tool for V2 agentic verification — wraps World Bank WDI API.

    Provides economic data lookup capabilities for the verification agent.
    """

    name = "worldbank"

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Harmony tool configuration for the World Bank economics tool."""
        indicator_names = ", ".join(ECONOMICS_INDICATORS.keys())
        return ToolNamespaceConfig(
            name="worldbank",
            description=(
                "World Bank economic data tool. Look up country economic "
                "indicators from the World Development Indicators (WDI) database."
            ),
            tools=[
                ToolDescription(
                    name="search_country",
                    description="Search for a country by name or ISO code",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (country name or ISO-3 code)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_indicators",
                    description="Get all economic indicators for a country and year",
                    parameters={
                        "type": "object",
                        "properties": {
                            "iso3": {
                                "type": "string",
                                "description": "ISO 3166-1 alpha-3 code (e.g., 'USA')",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Data year (e.g., 2022)",
                            },
                        },
                        "required": ["iso3"],
                    },
                ),
                ToolDescription(
                    name="get_indicator",
                    description=(
                        f"Get a specific economic indicator for a country and year. "
                        f"Available indicators: {indicator_names}"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "iso3": {
                                "type": "string",
                                "description": "ISO 3166-1 alpha-3 code (e.g., 'USA')",
                            },
                            "indicator": {
                                "type": "string",
                                "description": "Economic indicator name",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Data year (e.g., 2022)",
                            },
                        },
                        "required": ["iso3", "indicator"],
                    },
                ),
                ToolDescription(
                    name="compare_countries",
                    description="Compare an economic indicator across multiple countries",
                    parameters={
                        "type": "object",
                        "properties": {
                            "iso3_list": {
                                "type": "string",
                                "description": "Comma-separated ISO-3 codes (e.g., 'USA,CHN,DEU')",
                            },
                            "indicator": {
                                "type": "string",
                                "description": "Economic indicator to compare",
                            },
                            "year": {
                                "type": "integer",
                                "description": "Data year (e.g., 2022)",
                            },
                        },
                        "required": ["iso3_list", "indicator"],
                    },
                ),
                ToolDescription(
                    name="get_indicator_history",
                    description="Get an indicator's history across a range of years",
                    parameters={
                        "type": "object",
                        "properties": {
                            "iso3": {
                                "type": "string",
                                "description": "ISO 3166-1 alpha-3 code (e.g., 'USA')",
                            },
                            "indicator": {
                                "type": "string",
                                "description": "Economic indicator name",
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
                        "required": ["iso3", "indicator", "start_year", "end_year"],
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

        # Parse function name from recipient (e.g., "worldbank.get_indicators")
        parts = recipient.split(".", 1)
        if len(parts) == 2:
            func_name = parts[1]
        else:
            func_name = None

        # Extract text content from Message
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

        try:
            args = json.loads(content_text)
            if isinstance(args, dict):
                func_name = func_name or args.get("function", "")
            else:
                args = {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        try:
            if func_name == "search_country":
                result = self._search_country(args.get("query", ""))
            elif func_name == "get_indicators":
                result = self._get_indicators(
                    args.get("iso3", ""),
                    int(args.get("year", 2022)),
                )
            elif func_name == "get_indicator":
                result = self._get_indicator(
                    args.get("iso3", ""),
                    args.get("indicator", ""),
                    int(args.get("year", 2022)),
                )
            elif func_name == "compare_countries":
                iso3_str = args.get("iso3_list", "")
                iso3_list = [c.strip().upper() for c in iso3_str.split(",") if c.strip()]
                result = self._compare_countries(
                    iso3_list,
                    args.get("indicator", "gdp"),
                    int(args.get("year", 2022)),
                )
            elif func_name == "get_indicator_history":
                result = self._get_indicator_history(
                    args.get("iso3", ""),
                    args.get("indicator", "gdp"),
                    int(args.get("start_year", 2017)),
                    int(args.get("end_year", 2022)),
                )
            else:
                result = {
                    "error": f"Unknown function: {func_name}. "
                    "Available: search_country, get_indicators, get_indicator, "
                    "compare_countries, get_indicator_history"
                }
        except Exception as e:
            result = {"error": f"Error in {func_name}: {type(e).__name__}: {e}"}

        yield self._make_response(json.dumps(result, indent=2), recipient)

    def _search_country(self, query: str) -> Dict[str, Any]:
        """Search for countries by name or ISO code."""
        if not query:
            return {"error": "query parameter required"}

        query_upper = query.upper().strip()
        query_lower = query.lower().strip()

        matches = []

        # Check all regions for matching countries
        for region, codes in COUNTRY_UNIVERSE_REGIONS.items():
            for iso3 in codes:
                if iso3 == query_upper:
                    name = get_country_name(iso3) or iso3
                    matches.insert(0, {
                        "iso3": iso3,
                        "name": name,
                        "region": region,
                    })
                elif len(matches) < 10:
                    name = get_country_name(iso3) or iso3
                    if query_lower in name.lower():
                        matches.append({
                            "iso3": iso3,
                            "name": name,
                            "region": region,
                        })

        return {"results": matches[:10], "count": len(matches)}

    def _get_indicators(self, iso3: str, year: int) -> Dict[str, Any]:
        """Get all economic indicators for a country and year."""
        if not iso3:
            return {"error": "iso3 parameter required"}

        indicators = fetch_country_indicators(iso3.upper(), year)
        if indicators is None:
            return {"error": f"Could not fetch indicators for {iso3} ({year})"}

        return indicators

    def _get_indicator(self, iso3: str, indicator: str, year: int) -> Dict[str, Any]:
        """Get a specific indicator for a country and year."""
        if not iso3 or not indicator:
            return {"error": "iso3 and indicator parameters required"}

        if indicator not in ECONOMICS_INDICATORS:
            return {
                "error": f"Unknown indicator: {indicator}. "
                f"Available: {list(ECONOMICS_INDICATORS.keys())}"
            }

        val = get_indicator_value(iso3.upper(), indicator, year)
        if val is None:
            return {"error": f"Indicator {indicator} not available for {iso3} ({year})"}

        ind_info = ECONOMICS_INDICATORS[indicator]
        return {
            "iso3": iso3.upper(),
            "indicator": indicator,
            "label": ind_info["label"],
            "year": year,
            "value": val,
            "unit": ind_info["unit"],
        }

    def _compare_countries(self, iso3_list: list, indicator: str,
                           year: int) -> Dict[str, Any]:
        """Compare an indicator across multiple countries."""
        if not iso3_list:
            return {"error": "iso3_list parameter required (comma-separated)"}
        if indicator not in ECONOMICS_INDICATORS:
            return {
                "error": f"Unknown indicator: {indicator}. "
                f"Available: {list(ECONOMICS_INDICATORS.keys())}"
            }

        ranked = compare_countries_indicator(iso3_list, indicator, year)
        return {
            "indicator": indicator,
            "year": year,
            "ranking": ranked,
            "count": len(ranked),
        }

    def _get_indicator_history(self, iso3: str, indicator: str,
                               start_year: int, end_year: int) -> Dict[str, Any]:
        """Get indicator history across years."""
        if not iso3 or not indicator:
            return {"error": "iso3 and indicator parameters required"}
        if indicator not in ECONOMICS_INDICATORS:
            return {
                "error": f"Unknown indicator: {indicator}. "
                f"Available: {list(ECONOMICS_INDICATORS.keys())}"
            }

        history = get_indicator_history(iso3.upper(), indicator, start_year, end_year)
        if history is None:
            return {"error": f"Could not fetch history for {iso3} {indicator}"}

        return {
            "iso3": iso3.upper(),
            "indicator": indicator,
            "start_year": start_year,
            "end_year": end_year,
            "values": {str(y): v for y, v in sorted(history.items())},
            "count": len(history),
        }
