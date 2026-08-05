# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Cryptography + ATT&CK Tool for V2 agentic verification.

Provides a Harmony-compatible tool that wraps hardcoded reference data for
cryptographic algorithms and MITRE ATT&CK entities, allowing V2 verification
agents to look up security data.

Actions:
    security.get_algorithm(name)                 -> crypto algorithm parameters
    security.search_algorithms(query, type)      -> search crypto algorithms
    security.get_attack_entity(name)             -> ATT&CK group/technique/software
    security.search_attack(query, type)          -> search ATT&CK entities
    security.compare_algorithms(names)           -> side-by-side crypto params
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

from drbench.security_util import (
    get_crypto_algorithm_data,
    get_attack_entity_data,
    search_crypto_algorithms,
    search_attack_entities,
)

logger = logging.getLogger(__name__)


class SecurityTool:
    """Tool for V2 agentic verification — wraps crypto + ATT&CK reference data.

    Provides cryptographic algorithm parameters and MITRE ATT&CK metrics
    lookup capabilities for the verification agent.
    """

    name = "security"

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Harmony tool configuration for the security tool."""
        return ToolNamespaceConfig(
            name="security",
            description=(
                "Cybersecurity data tool. Look up cryptographic algorithm parameters "
                "(key size, block size, rounds, security strength) and MITRE ATT&CK "
                "entity metrics (technique counts, tactic coverage, software arsenal)."
            ),
            tools=[
                ToolDescription(
                    name="get_algorithm",
                    description=(
                        "Get cryptographic algorithm parameters. Returns: type, family, "
                        "key_size, block_size, rounds, security_strength, year, and more "
                        "depending on algorithm type (nonce_size, tag_size, output_size, state_size)"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Algorithm name (e.g., 'AES-256', 'SHA-256', 'ChaCha20')",
                            },
                        },
                        "required": ["name"],
                    },
                ),
                ToolDescription(
                    name="search_algorithms",
                    description="Search cryptographic algorithms by name, family, or type",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (name, family, or type keyword)",
                            },
                            "algo_type": {
                                "type": "string",
                                "description": (
                                    "Optional type filter: block_cipher, stream_cipher, hash, "
                                    "public_key, key_exchange, aead, mac"
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_attack_entity",
                    description=(
                        "Get MITRE ATT&CK entity metrics. For groups: technique_count, "
                        "sub_technique_count, tactic_count, software_count, first_seen. "
                        "For techniques: sub_technique_count, group_count, software_count, "
                        "mitigation_count, data_source_count, tactic. "
                        "For software: technique_count, group_count."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Entity name (e.g., 'APT28') or ATT&CK ID (e.g., 'G0007', 'T1566')",
                            },
                        },
                        "required": ["name"],
                    },
                ),
                ToolDescription(
                    name="search_attack",
                    description="Search MITRE ATT&CK entities by name, ID, or tactic",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (name, ATT&CK ID, or tactic name)",
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "Optional type filter: group, technique, software",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="compare_algorithms",
                    description="Compare parameters across multiple cryptographic algorithms side-by-side",
                    parameters={
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "string",
                                "description": "Comma-separated algorithm names (e.g., 'AES-128,AES-256,ChaCha20')",
                            },
                        },
                        "required": ["names"],
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
            if func_name == "get_algorithm":
                result = self._get_algorithm(args.get("name", ""))
            elif func_name == "search_algorithms":
                result = self._search_algorithms(
                    args.get("query", ""), args.get("algo_type"),
                )
            elif func_name == "get_attack_entity":
                result = self._get_attack_entity(args.get("name", ""))
            elif func_name == "search_attack":
                result = self._search_attack(
                    args.get("query", ""), args.get("entity_type"),
                )
            elif func_name == "compare_algorithms":
                names_str = args.get("names", "")
                names = [n.strip() for n in names_str.split(",") if n.strip()]
                result = self._compare_algorithms(names)
            else:
                result = {
                    "error": f"Unknown function: {func_name}. "
                    "Available: get_algorithm, search_algorithms, get_attack_entity, "
                    "search_attack, compare_algorithms"
                }
        except Exception as e:
            result = {"error": f"Error in {func_name}: {type(e).__name__}: {e}"}

        yield self._make_response(json.dumps(result, indent=2), recipient)

    def _get_algorithm(self, name: str) -> Dict[str, Any]:
        """Get cryptographic algorithm parameters."""
        if not name:
            return {"error": "name parameter required"}
        data = get_crypto_algorithm_data(name)
        if data is None:
            return {"error": f"Algorithm not found: {name}"}
        return data

    def _search_algorithms(self, query: str,
                           algo_type: Optional[str] = None) -> Dict[str, Any]:
        """Search cryptographic algorithms."""
        if not query:
            return {"error": "query parameter required"}
        matches = search_crypto_algorithms(query, algo_type=algo_type)
        return {"results": matches[:20], "count": len(matches)}

    def _get_attack_entity(self, name: str) -> Dict[str, Any]:
        """Get ATT&CK entity metrics."""
        if not name:
            return {"error": "name parameter required"}
        data = get_attack_entity_data(name)
        if data is None:
            return {"error": f"ATT&CK entity not found: {name}"}
        return data

    def _search_attack(self, query: str,
                       entity_type: Optional[str] = None) -> Dict[str, Any]:
        """Search ATT&CK entities."""
        if not query:
            return {"error": "query parameter required"}
        matches = search_attack_entities(query, entity_type=entity_type)
        return {"results": matches[:20], "count": len(matches)}

    def _compare_algorithms(self, names: list) -> Dict[str, Any]:
        """Compare parameters across multiple algorithms."""
        if not names:
            return {"error": "names parameter required (comma-separated)"}

        results = []
        for name in names:
            data = get_crypto_algorithm_data(name)
            if data is not None:
                results.append(data)

        return {
            "comparison": results,
            "count": len(results),
        }
