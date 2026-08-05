# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import logging
import os
import random
import time
import html
import json
import re
import asyncio
from typing import AsyncIterator, Optional, List, Tuple, Dict, Any
from aiohttp import ClientSession
from urllib.parse import quote, unquote, urlencode

from gpt_oss.tools.simple_browser.page_contents import (
    process_html,
)
from gpt_oss.tools.simple_browser.backend import (
    VIEW_SOURCE_PREFIX,
    BackendError,
    maybe_truncate,
)

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from gpt_oss.tools.simple_browser.simple_browser_tool import (
    SimpleBrowserTool,
    maybe_get_function_args,
)

logger = logging.getLogger(__name__)

KB_CACHE_DIR = "cache/kb_cache"
KB_CACHE_MAX_AGE_HOURS = 168  # 1 week


def _ensure_cache_dir():
    os.makedirs(KB_CACHE_DIR, exist_ok=True)


class _FrozenResponse:
    """Snapshot of an aiohttp response so it can be used after the context manager closes."""
    __slots__ = ("status", "headers", "_body", "content_type")

    def __init__(self, status, headers, body, content_type):
        self.status = status
        self.headers = headers
        self._body = body
        self.content_type = content_type

    async def json(self):
        return json.loads(self._body)

    async def text(self):
        return self._body.decode("utf-8", errors="replace")

    async def read(self):
        return self._body

    def to_disk(self) -> dict:
        return {
            "status": self.status,
            "body_b64": base64.b64encode(self._body).decode("ascii"),
            "content_type": self.content_type,
        }

    @classmethod
    def from_disk(cls, data: dict) -> "_FrozenResponse":
        return cls(
            status=data["status"],
            headers={},
            body=base64.b64decode(data["body_b64"]),
            content_type=data.get("content_type", "application/json"),
        )


class MultiSourceKnowledgeBackend:
    """Backend for multi-source knowledge base search (Wikidata, Wikimedia)."""

    source = "wikipedia_enhanced"
    
    def __init__(self, language: str = "en", primary_source: str = "wikidata"):
        """
        Initialize multi-source knowledge backend.

        Args:
            language: Language code (default: "en")
            primary_source: Primary source to use ("wikidata" or "wikimedia")
        """
        self.language = language
        self.primary_source = primary_source

        # API endpoints
        self.wikidata_sparql = "https://query.wikidata.org/sparql"
        self.wikimedia_rest = f"https://{language}.wikipedia.org/api/rest_v1"
        self.wikimedia_action = f"https://{language}.wikipedia.org/w/api.php"
        
        # Wikimedia requires a descriptive User-Agent with a reachable contact
        # (policy T400119); a UA-less request is hard-blocked with 403. We do NOT
        # hard-code a personal contact here: whoever runs this code is responsible
        # for their own API traffic, so operators may identify themselves by
        # setting DRBENCHER_USER_AGENT (e.g. "MyTool/1.0 (you@example.org)"). The
        # default is descriptive enough to work out of the box and points only at
        # the public project, never at an individual.
        self.headers = {
            "User-Agent": os.environ.get(
                "DRBENCHER_USER_AGENT",
                "DrBencher/1.0 (https://github.com/IBM/DrBencher)",
            ),
            "Accept": "application/json",
        }
        self._session: Optional[ClientSession] = None
        self._max_retries = 5
        self._base_backoff = 5.0  # seconds
        self._pre_request_delay = 3.0  # minimum delay before each API call
    
    async def _get_session(self, provided_session: Optional[ClientSession] = None) -> ClientSession:
        """Get or create aiohttp session."""
        if provided_session:
            return provided_session
        
        if self._session is None or self._session.closed:
            self._session = ClientSession(headers=self.headers)
        
        return self._session
    
    async def close(self):
        """Close the internal aiohttp session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    def clear_cache():
        """Remove all disk-cached responses."""
        import shutil
        if os.path.isdir(KB_CACHE_DIR):
            shutil.rmtree(KB_CACHE_DIR)

    @staticmethod
    def _cache_path_for(url, params) -> str:
        """Derive a deterministic file path for this request."""
        raw = url
        if params:
            raw += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return os.path.join(KB_CACHE_DIR, f"{digest}.json")

    async def _get_with_retry(self, session, url, **kwargs):
        """GET request with exponential backoff on 403/429.  Results are disk-cached."""
        timeout = kwargs.pop("timeout", 15)
        params = kwargs.get("params")

        # ALWAYS send our User-Agent per-request. The gpt-oss SimpleBrowserTool
        # drives this backend with ITS OWN aiohttp session (created without our
        # headers), so relying on the session's default headers is not enough —
        # a UA-less request is hard-blocked by Wikimedia (403 "Please set a
        # user-agent", policy T400119). Per-request headers override the
        # session's and guarantee the UA regardless of who owns the session.
        merged_headers = dict(self.headers)
        merged_headers.update(kwargs.get("headers") or {})
        kwargs["headers"] = merged_headers

        # Check disk cache
        cache_path = self._cache_path_for(url, params)
        if os.path.exists(cache_path):
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours < KB_CACHE_MAX_AGE_HOURS:
                try:
                    with open(cache_path, "r") as f:
                        return _FrozenResponse.from_disk(json.load(f))
                except (json.JSONDecodeError, KeyError, IOError):
                    pass  # cache corrupted, re-fetch

        # Pre-request throttle to avoid rate limiting (3-8s jitter)
        await asyncio.sleep(self._pre_request_delay + random.random() * 5)

        last_exc = None
        for attempt in range(self._max_retries):
            try:
                async with session.get(url, timeout=timeout, **kwargs) as resp:
                    if resp.status in (403, 429):
                        retry_after = resp.headers.get("Retry-After")
                        wait = float(retry_after) if retry_after else (2 ** attempt) * self._base_backoff + random.random() * 5
                        logger.warning(f"Got {resp.status} from {url[:80]}, retry {attempt+1}/{self._max_retries} in {wait:.1f}s")
                        # On the first block, log WHY: Wikimedia's 403/429 body states the
                        # reason (IP block vs UA policy vs rate limit) and the key headers.
                        if attempt == 0:
                            try:
                                body_preview = (await resp.text())[:400]
                            except Exception:
                                body_preview = "<unreadable>"
                            hdrs = {k: resp.headers.get(k) for k in
                                    ("Retry-After", "X-RateLimit-Remaining", "Server", "CF-RAY", "X-Cache")}
                            logger.warning(f"[{resp.status} detail] headers={hdrs} body={body_preview!r}")
                        await asyncio.sleep(wait)
                        continue
                    # Read the body while the context manager is alive
                    body = await resp.read()
                    frozen = _FrozenResponse(resp.status, resp.headers, body, resp.content_type)
                    # Persist successful (2xx) responses to disk
                    if 200 <= resp.status < 300:
                        try:
                            _ensure_cache_dir()
                            with open(cache_path, "w") as f:
                                json.dump(frozen.to_disk(), f)
                        except IOError:
                            pass  # non-fatal
                    return frozen
            except asyncio.TimeoutError:
                wait = (2 ** attempt) * self._base_backoff + random.random() * 5
                logger.warning(f"Timeout from {url[:80]}, retry {attempt+1}/{self._max_retries} in {wait:.1f}s")
                last_exc = asyncio.TimeoutError()
                await asyncio.sleep(wait)
        # All retries exhausted
        if last_exc:
            raise last_exc
        raise Exception(f"All {self._max_retries} retries returned 403/429 for {url[:80]}")
    
    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize query before processing for SPARQL.
        Removes problematic characters that cause SPARQL errors.
        """
        if not query:
            return ""
        
        # Remove all quotes (single and double) - they cause SPARQL regex issues
        query = query.replace('"', '').replace("'", '')
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # For heavily non-ASCII queries (like Japanese), try to extract ASCII parts
        # or simplify to avoid SPARQL errors
        ascii_chars = sum(1 for c in query if ord(c) < 128)
        total_chars = len(query)
        
        if total_chars > 0 and ascii_chars / total_chars < 0.3:
            # Query is >70% non-ASCII, extract ASCII keywords if any
            ascii_words = re.findall(r'[a-zA-Z0-9]+', query)
            if ascii_words and len(ascii_words) >= 2:
                # Use ASCII words if we have enough
                query = ' '.join(ascii_words)
                logger.debug(f"Extracted ASCII keywords from non-ASCII query: {query}")
            elif len(query) < 50:
                # Keep short non-ASCII queries as-is
                pass
            else:
                # For long non-ASCII queries with no ASCII parts, this will likely fail
                # Log and let it try
                logger.warning(f"Query is heavily non-ASCII and may fail in SPARQL: {query[:50]}")
        
        # Remove parentheses and brackets that can interfere with SPARQL
        query = query.replace('(', '').replace(')', '')
        query = query.replace('[', '').replace(']', '')
        query = query.replace('{', '').replace('}', '')
        
        # Limit length to avoid timeout
        if len(query) > 100:
            query = query[:100]
            logger.debug("Query truncated to 100 characters")
        
        return query.strip()
    
    def _escape_sparql_string(self, s: str) -> str:
        """
        Escape string for safe use in SPARQL query.
        This is applied AFTER sanitization.
        """
        # Escape backslashes first
        s = s.replace('\\', '\\\\')
        # Escape any remaining quotes (though sanitize should remove them)
        s = s.replace('"', '\\"')
        s = s.replace("'", "\\'")
        # Escape newlines and control characters
        s = s.replace('\n', ' ')
        s = s.replace('\r', ' ')
        s = s.replace('\t', ' ')
        return s
    
    async def search(
        self, 
        query: str, 
        topn: int = 5,
        session: ClientSession = None,
        source_preference: Optional[str] = None
    ):
        """
        Search across knowledge bases with fallback support.
        
        Args:
            query: Search query string
            topn: Number of results to return
            session: aiohttp ClientSession
            source_preference: Override primary source for this search
        
        Returns:
            Processed HTML page with search results
        """
        sess = await self._get_session(session)
        source = source_preference or self.primary_source
        
        # Sanitize query once at the start
        sanitized_query = self._sanitize_query(query)
        
        # Check if query is valid after sanitization
        if not sanitized_query or len(sanitized_query) < 2:
            logger.warning(f"Query too short or empty after sanitization: '{query}' -> '{sanitized_query}'")
            # Fall back to wikimedia immediately for invalid queries
            source = "wikimedia"
            sources_to_try = ["wikimedia"]
        else:
            # Only use the configured primary source (no cross-fallback to avoid
            # SPARQL 403s when primary is wikimedia, and vice versa)
            sources_to_try = [source]
        
        last_error = None
        
        for attempt_source in sources_to_try:
            try:
                if attempt_source == "wikidata":
                    results = await self._search_wikidata(sanitized_query, topn, sess)
                else:  # wikimedia
                    results = await self._search_wikimedia(sanitized_query, topn, sess)
                
                if results:
                    html_page = self._format_search_results(query, results, attempt_source)
                    pseudo_url = f"kb-search://{attempt_source}/{query}?ts={int(time.time())}"
                    
                    return process_html(
                        html=html_page,
                        url=pseudo_url,
                        title=f"Knowledge Search: {query}",
                        display_urls=True,
                        session=sess,
                    )
            except Exception as e:
                last_error = e
                logger.warning(f"Search failed on {attempt_source} for '{query}': {e}")
                continue
        
        # If all sources failed, raise error with last exception
        error_msg = f"No results found for query: '{query}' across all sources"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        raise BackendError(error_msg)
    
    async def _search_wikidata(
        self,
        query: str,
        limit: int,
        session: ClientSession
    ) -> List[Dict[str, Any]]:
        """
        Search Wikidata using SPARQL.
        Query is expected to already be sanitized.
        """
        escaped_query = self._escape_sparql_string(query)
        
        sparql_query = f"""
        SELECT DISTINCT ?item ?itemLabel ?itemDescription
        WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search "{escaped_query}";
                            mwapi:language "en".
            ?item wikibase:apiOutputItem mwapi:item.
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT {limit}
        """
        
        params = {
            "query": sparql_query,
            "format": "json"
        }
        
        try:
            response = await self._get_with_retry(
                session, self.wikidata_sparql, params=params, timeout=20,
            )
            if response.status != 200:
                raise Exception(f"Wikidata returned status {response.status}")

            data = await response.json()
            results = []

            for binding in data.get("results", {}).get("bindings", []):
                item_uri = binding.get("item", {}).get("value", "")
                item_id = item_uri.split("/")[-1]
                label = binding.get("itemLabel", {}).get("value", "")
                description = binding.get("itemDescription", {}).get("value", "No description available")

                results.append({
                    "title": label,
                    "url": f"https://www.wikidata.org/wiki/{item_id}",
                    "summary": description,
                    "source": "Wikidata",
                    "item_id": item_id
                })

            return results
        except asyncio.TimeoutError:
            logger.error(f"Wikidata search timeout for query: {query}")
            raise Exception("Wikidata search timed out")
        except Exception as e:
            logger.error(f"Wikidata search error: {e}")
            raise
    
    async def _search_wikimedia(
        self,
        query: str,
        limit: int,
        session: ClientSession
    ) -> List[Dict[str, Any]]:
        """
        Search using Wikimedia REST API.
        Query is expected to already be sanitized (quotes removed).
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,  # Already sanitized
            "srlimit": limit,
            "srprop": "snippet",
            "utf8": 1,
            "formatversion": 2
        }
        
        try:
            response = await self._get_with_retry(
                session, self.wikimedia_action, params=params, timeout=15,
            )
            if response.status != 200:
                raise Exception(f"Wikimedia returned status {response.status}")

            data = await response.json()
            results = []

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")

                # Get summary using REST API
                summary = await self._get_wikimedia_summary(title, session)

                results.append({
                    "title": title,
                    "url": f"https://{self.language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    "summary": summary or snippet,
                    "source": "Wikimedia"
                })

            return results
        except asyncio.TimeoutError:
            logger.error(f"Wikimedia search timeout for query: {query}")
            raise Exception("Wikimedia search timed out")
        except Exception as e:
            logger.error(f"Wikimedia search error: {e}")
            raise
    
    async def _search_wikimedia_rest(
        self,
        query: str,
        limit: int,
        session: ClientSession
    ) -> List[Dict[str, Any]]:
        """Alternative search using Wikimedia REST API search endpoint."""
        try:
            search_url = f"{self.wikimedia_rest}/page/search/{quote(query)}"

            response = await self._get_with_retry(session, search_url, timeout=15)
            if response.status != 200:
                raise Exception(f"REST search returned status {response.status}")

            data = await response.json()
            results = []

            for item in data.get("pages", [])[:limit]:
                title = item.get("title", "")
                description = item.get("description", "")
                excerpt = item.get("excerpt", "")

                summary = description or excerpt or "No description available"

                results.append({
                    "title": title,
                    "url": f"https://{self.language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    "summary": summary,
                    "source": "Wikimedia"
                })

            return results
        except Exception as e:
            logger.error(f"Wikimedia REST search error: {e}")
            return []
    
    async def _get_wikimedia_summary(
        self,
        title: str,
        session: ClientSession
    ) -> Optional[str]:
        """Get summary using Wikimedia REST API."""
        try:
            url = f"{self.wikimedia_rest}/page/summary/{quote(title.replace(' ', '_'))}"
            response = await self._get_with_retry(session, url, timeout=10)
            if response.status == 200:
                data = await response.json()
                return data.get("extract", "")
        except Exception as e:
            logger.debug(f"Failed to get Wikimedia summary for '{title}': {e}")
        return None
    
    def _format_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str
    ) -> str:
        """Format search results as HTML."""
        items_html = ""
        for result in results:
            title = html.escape(result["title"])
            url = html.escape(result["url"])
            summary = html.escape(result["summary"])
            source_badge = html.escape(result.get("source", source))
            
            items_html += f"""
            <li>
                <div style="margin-bottom: 15px;">
                    <strong><a href='{url}'>{title}</a></strong>
                    <span style="background: #e0e0e0; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; margin-left: 8px;">{source_badge}</span>
                    <p style="margin-top: 5px;">{summary}</p>
                </div>
            </li>
            """
        
        html_page = f"""
<html>
<head><title>Knowledge Search: {html.escape(query)}</title></head>
<body>
<h1>Knowledge Base Search Results: {html.escape(query)}</h1>
<p>Found {len(results)} results from {html.escape(source.upper())}</p>
<ul style="list-style-type: none; padding-left: 0;">
{items_html}
</ul>
</body>
</html>
"""
        return html_page
    
    async def fetch(self, url: str, session: ClientSession = None):
        """
        Fetch content from knowledge base by URL.
        
        Args:
            url: Resource URL or identifier
            session: aiohttp ClientSession
        
        Returns:
            Processed HTML page with content
        """
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]
        
        sess = await self._get_session(session)
        
        try:
            # Determine source from URL
            if "wikidata.org" in url:
                content = await self._fetch_wikidata(url, sess)
            elif "wikipedia.org" in url:
                content = await self._fetch_wikimedia(url, sess)
            else:
                raise BackendError(f"Unsupported URL format: {url}")
            
            return process_html(
                html=content["html"],
                url=content["url"],
                title=content["title"],
                display_urls=True,
                session=sess,
            )
            
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"Error fetching content from '{url}': {str(e)}") from e
    
    async def _fetch_wikidata(self, url: str, session: ClientSession) -> Dict[str, str]:
        """Fetch Wikidata entity information."""
        # Extract entity ID from URL
        entity_id = url.split("/")[-1]

        api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

        response = await self._get_with_retry(session, api_url, timeout=15)
        if response.status != 200:
            raise Exception(f"Wikidata returned status {response.status}")

        data = await response.json()
        entity = data.get("entities", {}).get(entity_id, {})

        if not entity:
            raise BackendError(f"Wikidata entity not found: {entity_id}")

        # Extract label and description
        labels = entity.get("labels", {})
        descriptions = entity.get("descriptions", {})

        label = labels.get("en", {}).get("value", entity_id)
        description = descriptions.get("en", {}).get("value", "No description available")

        # Extract claims/properties
        claims = entity.get("claims", {})
        properties_html = "<ul>"
        for prop_id, prop_values in list(claims.items())[:10]:
            properties_html += f"<li><strong>{html.escape(prop_id)}:</strong> {len(prop_values)} value(s)</li>"
        properties_html += "</ul>"

        html_content = f"""
<html>
<head><title>{html.escape(label)}</title></head>
<body>
<h1>{html.escape(label)}</h1>
<p><strong>Source:</strong> Wikidata</p>
<p><strong>Entity ID:</strong> <a href='{html.escape(url)}'>{html.escape(entity_id)}</a></p>
<hr/>
<h2>Description</h2>
<p>{html.escape(description)}</p>
<h2>Properties</h2>
{properties_html}
</body>
</html>
"""

        return {"html": html_content, "url": url, "title": label}
    
    async def _fetch_wikimedia(self, url: str, session: ClientSession) -> Dict[str, str]:
        """Fetch Wikipedia article using REST API."""
        # Extract title from URL
        title = unquote(url.split("/wiki/")[-1].replace("_", " "))

        rest_url = f"{self.wikimedia_rest}/page/html/{quote(title.replace(' ', '_'))}"

        response = await self._get_with_retry(session, rest_url, timeout=15)
        if response.status != 200:
            raise Exception(f"Wikimedia returned status {response.status}")

        html_content = await response.text()

        return {"html": html_content, "url": url, "title": title}


class MultiSourceKnowledgeBrowserTool(SimpleBrowserTool):
    """Browser tool for multi-source knowledge base search and navigation."""
    
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return
        
        try:
            if function_name == "search":
                async for msg in self.search(**function_args):
                    yield msg
            elif function_name == "open":
                async for msg in self.open(**function_args):
                    yield msg
            elif function_name == "find":
                async for msg in self.find(**function_args):
                    yield msg
        except TypeError as e:
            error_text = f"Error: Invalid arguments for function '{function_name}'. Details: {e}"
            yield self.make_response(
                content=TextContent(text=error_text),
                author=Author(role=Role.TOOL, name=message.recipient)
            )
        except Exception as e:
            error_text = f"Error executing '{function_name}': {e}"
            yield self.make_response(
                content=TextContent(text=error_text),
                author=Author(role=Role.TOOL, name=message.recipient)
            )
    
    async def _open_url(self, url: str, direct_url_open: bool):
        """Open knowledge base page with caching support."""
        backend = self.backend
        
        # Check cache if not forcing refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            page = await backend.fetch(url, session=None)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(
                f"Error fetching knowledge base page `{maybe_truncate(url)}`: {msg}"
            ) from e
        
