"""Biochemistry Tool for V2 agentic verification.

Provides a Harmony-compatible tool that wraps PubChem, UniProt, PDB,
and ChEMBL APIs, allowing V2 verification agents to look up real
biochemical data.

Actions:
    bio.search_protein(query)              -> list of {uniprot_id, name, organism}
    bio.get_protein(uniprot_id)            -> dict of protein properties
    bio.search_compound(query)             -> list of {cid, name, formula, MW}
    bio.get_compound(name_or_cid)          -> dict of compound properties
    bio.get_structure(pdb_id)              -> dict of PDB structure data
    bio.compare_compounds(names, property) -> ranked comparison
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

from drbench.bio_util import (
    get_protein_data,
    get_compound_data,
    get_pdb_data,
    search_protein,
    search_compound,
)

logger = logging.getLogger(__name__)


class BiochemTool:
    """Tool for V2 agentic verification — wraps PubChem, UniProt, PDB APIs.

    Provides biochemical data lookup capabilities for the verification agent.
    """

    name = "bio"

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Harmony tool configuration for the biochemistry tool."""
        return ToolNamespaceConfig(
            name="bio",
            description=(
                "Biochemistry data tool. Look up protein data from UniProt, "
                "compound data from PubChem, and structure data from RCSB PDB."
            ),
            tools=[
                ToolDescription(
                    name="search_protein",
                    description="Search for a protein by name, gene, or function",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (protein name, gene name, etc.)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_protein",
                    description=(
                        "Get protein properties from UniProt. Returns: name, organism, "
                        "molecular weight (Da), sequence length, amino acid counts, "
                        "function, subcellular location, EC numbers, PDB IDs"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "uniprot_id": {
                                "type": "string",
                                "description": "UniProt accession (e.g., 'P01308')",
                            },
                        },
                        "required": ["uniprot_id"],
                    },
                ),
                ToolDescription(
                    name="search_compound",
                    description="Search for a compound/drug by name",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (compound name, drug name, etc.)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                ToolDescription(
                    name="get_compound",
                    description=(
                        "Get compound properties from PubChem. Returns: molecular weight, "
                        "formula, XLogP, TPSA, H-bond donors/acceptors, rotatable bonds, "
                        "heavy atom count, complexity, SMILES, IUPAC name"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "name_or_cid": {
                                "type": "string",
                                "description": "Compound name (e.g., 'aspirin') or CID (e.g., '2244')",
                            },
                        },
                        "required": ["name_or_cid"],
                    },
                ),
                ToolDescription(
                    name="get_structure",
                    description=(
                        "Get crystal structure data from RCSB PDB. Returns: resolution, "
                        "R-factor, cell dimensions, space group, atom count"
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "pdb_id": {
                                "type": "string",
                                "description": "PDB ID (e.g., '1HHO')",
                            },
                        },
                        "required": ["pdb_id"],
                    },
                ),
                ToolDescription(
                    name="compare_compounds",
                    description="Compare a property across multiple compounds (ranked)",
                    parameters={
                        "type": "object",
                        "properties": {
                            "names": {
                                "type": "string",
                                "description": "Comma-separated compound names",
                            },
                            "property": {
                                "type": "string",
                                "description": (
                                    "Property to compare. Available: molecular_weight, "
                                    "xlogp, tpsa, hbond_donor_count, hbond_acceptor_count, "
                                    "heavy_atom_count, complexity, rotatable_bond_count"
                                ),
                            },
                        },
                        "required": ["names", "property"],
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

        # Parse function name from recipient (e.g., "bio.get_protein")
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
            if func_name == "search_protein":
                result = self._search_protein(args.get("query", ""))
            elif func_name == "get_protein":
                result = self._get_protein(args.get("uniprot_id", ""))
            elif func_name == "search_compound":
                result = self._search_compound(args.get("query", ""))
            elif func_name == "get_compound":
                result = self._get_compound(args.get("name_or_cid", ""))
            elif func_name == "get_structure":
                result = self._get_structure(args.get("pdb_id", ""))
            elif func_name == "compare_compounds":
                names_str = args.get("names", "")
                names = [n.strip() for n in names_str.split(",") if n.strip()]
                result = self._compare_compounds(
                    names, args.get("property", "molecular_weight"),
                )
            else:
                result = {
                    "error": f"Unknown function: {func_name}. "
                    "Available: search_protein, get_protein, search_compound, "
                    "get_compound, get_structure, compare_compounds"
                }
        except Exception as e:
            result = {"error": f"Error in {func_name}: {type(e).__name__}: {e}"}

        yield self._make_response(json.dumps(result, indent=2), recipient)

    def _search_protein(self, query: str) -> Dict[str, Any]:
        """Search for proteins by name or function."""
        if not query:
            return {"error": "query parameter required"}
        matches = search_protein(query)
        return {"results": matches[:10], "count": len(matches)}

    def _get_protein(self, uniprot_id: str) -> Dict[str, Any]:
        """Get protein properties from UniProt."""
        if not uniprot_id:
            return {"error": "uniprot_id parameter required"}
        data = get_protein_data(uniprot_id)
        if data is None:
            return {"error": f"Could not fetch protein data for {uniprot_id}"}

        # Format for readability — exclude raw sequence
        formatted = dict(data)
        if formatted.get("sequence") and len(formatted["sequence"]) > 100:
            formatted["sequence_preview"] = formatted["sequence"][:50] + "..."
            formatted["sequence_length"] = len(formatted["sequence"])
            del formatted["sequence"]

        # Format MW
        if formatted.get("molecular_weight"):
            mw = formatted["molecular_weight"]
            formatted["molecular_weight_da"] = mw
            formatted["molecular_weight_kda"] = round(mw / 1000, 2)

        return formatted

    def _search_compound(self, query: str) -> Dict[str, Any]:
        """Search for compounds by name."""
        if not query:
            return {"error": "query parameter required"}
        matches = search_compound(query)
        return {"results": matches[:10], "count": len(matches)}

    def _get_compound(self, name_or_cid: str) -> Dict[str, Any]:
        """Get compound properties from PubChem."""
        if not name_or_cid:
            return {"error": "name_or_cid parameter required"}
        data = get_compound_data(name_or_cid)
        if data is None:
            return {"error": f"Could not fetch compound data for {name_or_cid}"}
        return data

    def _get_structure(self, pdb_id: str) -> Dict[str, Any]:
        """Get crystal structure data from RCSB PDB."""
        if not pdb_id:
            return {"error": "pdb_id parameter required"}
        data = get_pdb_data(pdb_id)
        if data is None:
            return {"error": f"Could not fetch structure data for {pdb_id}"}
        return data

    def _compare_compounds(self, names: list, prop: str) -> Dict[str, Any]:
        """Compare a property across multiple compounds."""
        if not names:
            return {"error": "names parameter required (comma-separated)"}

        valid_props = [
            "molecular_weight", "xlogp", "tpsa", "hbond_donor_count",
            "hbond_acceptor_count", "heavy_atom_count", "complexity",
            "rotatable_bond_count",
        ]
        if prop not in valid_props:
            return {"error": f"Unknown property: {prop}. Available: {valid_props}"}

        results = []
        for name in names:
            data = get_compound_data(name)
            if data is None:
                continue
            val = data.get(prop)
            if val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue
                results.append({"name": name, "value": val})

        results.sort(key=lambda x: x["value"], reverse=True)
        return {
            "property": prop,
            "ranking": results,
            "count": len(results),
        }
