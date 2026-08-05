# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""History Tool for V2 agentic verification.

Provides a Harmony-compatible tool that wraps Wikidata temporal data
APIs, allowing V2 verification agents to look up historical dates
and compute durations.

Actions:
    history.search_entity(query)         -> list of {id, label, description}
    history.get_entity_dates(qid)        -> dict of temporal data
    history.get_entity_info(qid)         -> dict of label, description, dates
    history.compare_entities(qid_a, qid_b) -> side-by-side temporal data
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

from drbench.history_util import (
    search_entity,
    get_entity_temporal_data,
    get_entity_info,
    compare_entities,
    normalize_entity_dates,
)

logger = logging.getLogger(__name__)


class HistoryTool:
    """Tool for V2 agentic verification — wraps Wikidata temporal APIs.

    Provides historical date lookup capabilities for the verification agent.
    """

    name = "history"

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Harmony tool configuration for the history tool."""
        return ToolNamespaceConfig(
            name="history",
            description=(
                "Historical data tool. Look up dates (birth, death, founding, "
                "start, end) from Wikidata for historical entities (people, "
                "conflicts, organizations, milestones). Calculate durations "
                "and compare entities."
            ),
            tools=[
                ToolDescription(
                    name="search_entity",
                    description="Search Wikidata for historical entities by name or description",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (person name, war name, organization, event)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_entity_dates",
                    description=(
                        "Get temporal data for a Wikidata entity. Returns: "
                        "birth/death dates, start/end dates, inception date, "
                        "point-in-time date (depending on entity type)"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "qid": {
                                "type": "string",
                                "description": "Wikidata entity ID (e.g., 'Q362' for WWII)",
                            },
                        },
                        "required": ["qid"],
                    },
                ),
                ToolDescription(
                    name="get_entity_info",
                    description=(
                        "Get entity information including label, description, "
                        "and temporal data from Wikidata"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "qid": {
                                "type": "string",
                                "description": "Wikidata entity ID (e.g., 'Q937' for Einstein)",
                            },
                        },
                        "required": ["qid"],
                    },
                ),
                ToolDescription(
                    name="compare_entities",
                    description="Compare temporal data between two historical entities",
                    parameters={
                        "type": "object",
                        "properties": {
                            "qid_a": {
                                "type": "string",
                                "description": "First entity QID",
                            },
                            "qid_b": {
                                "type": "string",
                                "description": "Second entity QID",
                            },
                        },
                        "required": ["qid_a", "qid_b"],
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

        parts = recipient.split(".", 1)
        if len(parts) == 2:
            func_name = parts[1]
        else:
            func_name = None

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
            if func_name == "search_entity":
                result = self._search_entity(args.get("query", ""))
            elif func_name == "get_entity_dates":
                result = self._get_entity_dates(args.get("qid", ""))
            elif func_name == "get_entity_info":
                result = self._get_entity_info(args.get("qid", ""))
            elif func_name == "compare_entities":
                result = self._compare_entities(
                    args.get("qid_a", ""), args.get("qid_b", ""),
                )
            else:
                result = {
                    "error": f"Unknown function: {func_name}. "
                    "Available: search_entity, get_entity_dates, get_entity_info, "
                    "compare_entities"
                }
        except Exception as e:
            result = {"error": f"Error in {func_name}: {type(e).__name__}: {e}"}

        yield self._make_response(json.dumps(result, indent=2), recipient)

    def _search_entity(self, query: str) -> Dict[str, Any]:
        """Search Wikidata for historical entities."""
        if not query:
            return {"error": "query parameter required"}
        results = search_entity(query)
        return {"results": results[:10], "count": len(results)}

    def _get_entity_dates(self, qid: str) -> Dict[str, Any]:
        """Get temporal data for an entity."""
        if not qid:
            return {"error": "qid parameter required"}
        data = get_entity_temporal_data(qid)
        if data is None:
            return {"error": f"Could not fetch temporal data for {qid}"}
        # Return a cleaner view with just years
        clean = {"qid": qid, "dates": {}}
        for prop_id, entry in data.get("raw_dates", {}).items():
            clean["dates"][entry["label"]] = {
                "year": entry["year"],
                "property": prop_id,
            }
        return clean

    def _get_entity_info(self, qid: str) -> Dict[str, Any]:
        """Get entity info including label and dates."""
        if not qid:
            return {"error": "qid parameter required"}
        data = get_entity_info(qid)
        if data is None:
            return {"error": f"Could not fetch entity info for {qid}"}
        return data

    def _compare_entities(self, qid_a: str, qid_b: str) -> Dict[str, Any]:
        """Compare temporal data between two entities."""
        if not qid_a or not qid_b:
            return {"error": "Both qid_a and qid_b parameters required"}
        data = compare_entities(qid_a, qid_b)
        if data is None:
            return {"error": f"Could not compare {qid_a} and {qid_b}"}
        return data
