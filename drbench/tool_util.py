import copy
import random
import re
import time
import requests
import os, argparse, ast, json, tqdm
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import Counter, defaultdict
try:
    from autogen.code_utils import extract_code, execute_code
except ImportError:
    extract_code = None
    execute_code = None
import numpy as np
from bs4 import BeautifulSoup
from .util import gen_from_prompt

import hashlib

# ---------------------------------------------------------------------------
# Generation-side disk caches for shared knowledge fetches.
#
# Wikipedia article text (search_step) and Wikidata query/entity results
# (sparql_query, get_entity_data) were re-fetched on every run and every entity
# — the dominant wall-clock cost of benchmark generation. These caches persist
# them to disk (SHA-256(key) -> JSON, 1-week TTL) so repeated lookups of the
# same article / query are free. Two separate directories keep the two source
# types distinct. Benefits every domain that fetches via these shared helpers
# (geophysical, biochem's Wikipedia articles, security, etc.).
# ---------------------------------------------------------------------------
WIKIPEDIA_CACHE_DIR = "cache/wikipedia_cache"
WIKIDATA_CACHE_DIR = "cache/wikidata_cache"
_KG_CACHE_MAX_AGE_HOURS = 168  # 1 week


def _kg_cache_get(cache_dir, key):
    """Return the cached JSON value for *key* under *cache_dir*, or None."""
    try:
        digest = hashlib.sha256(key.encode()).hexdigest()
        path = os.path.join(cache_dir, f"{digest}.json")
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours < _KG_CACHE_MAX_AGE_HOURS:
                with open(path) as f:
                    return json.load(f)
    except Exception:
        pass
    return None


def _kg_cache_set(cache_dir, key, value):
    """Persist *value* (JSON-serializable) for *key* under *cache_dir*."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha256(key.encode()).hexdigest()
        with open(os.path.join(cache_dir, f"{digest}.json"), "w") as f:
            json.dump(value, f)
    except Exception:
        pass  # cache write is best-effort, never fatal


DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."



def extract_json_v2(json_text, outfilename):
    response = json_text.replace("TERMINATE", "")
    json_dict = None

    # First try: look for ```json blocks
    if "```json" in response:
        try:
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, response)

            if matches:
                for match in reversed(matches):
                    json_str = match.strip()
                    json_str = json_str.replace('...', '')
                    if json_str and (json_str.startswith('[') or json_str.startswith('{')):
                        try:
                            json_dict = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            try:
                                json_dict = ast.literal_eval(json_str)
                                break
                            except (SyntaxError, ValueError):
                                continue
        except Exception:
            pass

    # Second try: look for ``` blocks (without json specifier)
    if json_dict is None and "```" in response:
        try:
            code_pattern = r'```\s*([\s\S]*?)\s*```'
            matches = re.findall(code_pattern, response)
            for match in reversed(matches):
                json_str = match.strip()
                json_str = json_str.replace('...', '')
                if json_str and (json_str.startswith('[') or json_str.startswith('{')):
                    try:
                        json_dict = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        try:
                            json_dict = ast.literal_eval(json_str)
                            break
                        except (SyntaxError, ValueError):
                            continue
        except Exception:
            pass

    # Third try: find JSON arrays using bracket-counting (O(n), no backtracking)
    if json_dict is None:
        try:
            idx = 0
            candidates = []
            while idx < len(response):
                start = response.find('[', idx)
                if start == -1:
                    break
                # Check if next non-ws char is '{'
                peek = start + 1
                while peek < len(response) and response[peek] in ' \t\n\r':
                    peek += 1
                if peek >= len(response) or response[peek] != '{':
                    idx = start + 1
                    continue
                # Bracket-count to find matching ']'
                depth = 0
                in_str = False
                esc = False
                end = -1
                for i in range(start, len(response)):
                    c = response[i]
                    if esc:
                        esc = False
                        continue
                    if c == '\\' and in_str:
                        esc = True
                        continue
                    if c == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == '[':
                        depth += 1
                    elif c == ']':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > start:
                    candidates.append(response[start:end])
                    idx = end
                else:
                    break  # unclosed bracket — no point continuing
            for cand in reversed(candidates):
                try:
                    json_dict = json.loads(cand)
                    break
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    # Try 3.5: Clean up common thinking patterns and try again
    if json_dict is None:
        try:
            # Skip common thinking prefixes that models output before JSON
            thinking_patterns = [
                r'^We need to produce.*?\n',
                r'^Let me.*?\n',
                r'^I will.*?\n',
                r'^First,.*?\n',
                r'^To generate.*?\n',
                r'^Here are.*?\n',
                r'.*?produce \d+ QA pairs.*?\n',
                r'.*?We must output.*?\n',
                r'.*?Let\'s examine.*?\n',
            ]
            cleaned_response = response
            for pattern in thinking_patterns:
                cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE | re.MULTILINE)

            # Try to find JSON in cleaned response
            if '[' in cleaned_response and '{' in cleaned_response:
                start_idx = cleaned_response.find('[')
                bracket_count = 0
                end_idx = start_idx
                in_string = False
                escape_next = False
                for i, char in enumerate(cleaned_response[start_idx:], start_idx):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    json_str = cleaned_response[start_idx:end_idx]
                    try:
                        json_dict = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try fixing common issues
                        json_str_fixed = json_str.replace('...', '').replace('\n', ' ')
                        try:
                            json_dict = json.loads(json_str_fixed)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    # Fourth try: look for any [ ... ] that might be JSON
    if json_dict is None:
        try:
            # Find the first [ and try to parse from there
            start_idx = response.find('[')
            if start_idx != -1:
                # Try to find matching ]
                bracket_count = 0
                end_idx = start_idx
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
                if end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    try:
                        json_dict = json.loads(json_str)
                    except json.JSONDecodeError:
                        try:
                            json_dict = ast.literal_eval(json_str)
                        except (SyntaxError, ValueError):
                            pass
        except Exception:
            pass

    # Fifth try: use extract_code as fallback
    if json_dict is None and extract_code is not None:
        try:
            if '...' in response:
                response = response.replace('...', '')
            extracted_json = extract_code(response)
            combined_json = []
            for xx in extracted_json:
                try:
                    parsed = ast.literal_eval(xx[1])
                    if isinstance(parsed, list):
                        combined_json.extend(parsed)
                    else:
                        combined_json.append(parsed)
                except (SyntaxError, ValueError):
                    continue
            if combined_json:
                json_dict = combined_json
        except Exception:
            pass

    # If still nothing found, return empty list with warning
    if json_dict is None:
        json_dict = []
        print(f"Warning: Could not extract valid JSON from response, returning empty list. Response preview: {response[:200]}...", flush=True)

    if outfilename is not None:
        out_dir = os.path.dirname(outfilename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(outfilename, "w") as f:
            json.dump(json_dict, f)

    return json_dict

def test_taker_inference(test_model_info, problem_json, outfile, bsz=1, temperature=0.01, max_length=50):
    if len(test_model_info) == 3:
        model_choice, tokenizer_choice, client_choice = test_model_info
        auth = None
        use_helm = False
    elif len(test_model_info) == 4:
        model_choice, tokenizer_choice, client_choice, auth = test_model_info
        use_helm = True

    print(f'writing to {outfile}')
    out_handle = open(outfile, 'w')
    full_result_lst = []
    batch_lst, line_lst = [], []
    for line in tqdm.tqdm(problem_json):
        line['prompt'] = "Output just with the final answer to the question.\nQuestion:" + line[
            'question'] + "\n" + "Answer:"
        line_lst.append(line)
        batch_lst.append(line['prompt'])
        if len(batch_lst) < bsz:
            continue  # batch not full yet
        request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                         echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                         service=client_choice,
                                         terminate_by_linebreak='no', use_helm=use_helm, auth=auth,
                                         verbose=False)

        for line, xx in zip(line_lst, request_result.completions):
            # print(line['prompt'])
            # print('-' * 100)
            # print(xx.text)
            line['test_taker_response'] = xx.text
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
        batch_lst, line_lst = [], []
    if len(batch_lst) > 0:
        request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                         echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                         service=args.model_auth, terminate_by_linebreak='no', use_helm=use_helm,
                                         auth=auth, verbose=False)
        for line, xx in zip(line_lst, request_result.completions):
            line['test_taker_response'] = xx.text
            print(json.dumps(line), file=out_handle)
            full_result_lst.append(line)
    out_handle.close()
    return full_result_lst


def _generate_lm_answers(question_inputs, test_model_info, agent_model_info, outfile_prefix='att1'):
    if os.path.exists(f"{outfile_prefix}.test_taker_inference.json"):
        full_result_lst = []
        with open(f"{outfile_prefix}.test_taker_inference.json", 'r') as in_handle:
            for line in in_handle:
                line = json.loads(line.strip())
                full_result_lst.append(line)
        return full_result_lst

    # test_taker_lm, test_taker_tokenizer, test_taker_client = test_model_info
    if isinstance(question_inputs, list) or isinstance(question_inputs, dict):
        question_inputs_str = json.dumps(question_inputs, indent=2)
    else:
        assert False

    if isinstance(question_inputs, list):
        if len(question_inputs) == 0:
            print('question_inputs is empty.')
            return []
        if isinstance(question_inputs[0], list):
            json_dict = question_inputs[0]
        else:
            json_dict = question_inputs
    else:
        print('question_inputs should be a list.')
        assert False

    full_result_lst = test_taker_inference(test_model_info, json_dict,
                                           outfile=f"{outfile_prefix}.test_taker_inference.json")

    return full_result_lst


def test_taker_inference_harmony(harmony_generator, problem_json, outfile, temperature=0.01, max_length=50):
    """
    Test taker inference using Harmony format for gpt-oss-120b.

    Args:
        harmony_generator: HarmonyVLLMGenerator instance
        problem_json: List of questions to answer
        outfile: Output file path
        temperature: Sampling temperature
        max_length: Maximum tokens to generate

    Returns:
        List of results with test_taker_response field
    """
    print(f'[Harmony] writing to {outfile}')
    out_handle = open(outfile, 'w')
    full_result_lst = []

    developer_content = "You are answering knowledge questions. Provide ONLY the direct answer in a few words. Do not explain or elaborate."

    for line in tqdm.tqdm(problem_json):
        user_content = f"Question: {line['question']}\nAnswer:"

        try:
            response = harmony_generator.generate_response_sync(
                developer_content=developer_content,
                user_content=user_content,
                temperature=temperature,
                max_tokens=max_length,
                reasoning_effort="low",
            )
            line['test_taker_response'] = response.strip()
        except Exception as e:
            print(f"[Harmony] Error generating answer: {e}")
            line['test_taker_response'] = ""

        print(json.dumps(line), file=out_handle)
        full_result_lst.append(line)

    out_handle.close()
    return full_result_lst


def _generate_lm_answers_harmony(question_inputs, harmony_generator, outfile_prefix='att1'):
    """
    Generate LM answers using Harmony format for gpt-oss-120b.

    Args:
        question_inputs: List of questions
        harmony_generator: HarmonyVLLMGenerator instance
        outfile_prefix: Output file prefix

    Returns:
        List of results with test_taker_response field
    """
    if os.path.exists(f"{outfile_prefix}.test_taker_inference.json"):
        full_result_lst = []
        with open(f"{outfile_prefix}.test_taker_inference.json", 'r') as in_handle:
            for line in in_handle:
                line = json.loads(line.strip())
                full_result_lst.append(line)
        return full_result_lst

    if isinstance(question_inputs, list):
        if len(question_inputs) == 0:
            print('question_inputs is empty.')
            return []
        if isinstance(question_inputs[0], list):
            json_dict = question_inputs[0]
        else:
            json_dict = question_inputs
    else:
        print('question_inputs should be a list.')
        return []

    full_result_lst = test_taker_inference_harmony(
        harmony_generator,
        json_dict,
        outfile=f"{outfile_prefix}.test_taker_inference.json"
    )

    return full_result_lst



def search_related_pages(search_query):
    # URL for Wikipedia API search action
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json&cmlimit=max"

    # Headers required by Wikipedia API
    headers = {
        "User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
        "Api-User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Making the request
    try:
        response = requests.get(url, headers=headers, timeout=30)

        # Checking if request was successful
        if response.status_code == 200:
            data = response.json()
            search_results = data['query']['search']

            # Extracting titles of the search results
            related_pages = [result['title'] for result in search_results]
            return related_pages
        else:
            print(f"Failed to retrieve data from Wikipedia API. Status: {response.status_code}")
            return []
    except Exception as e:
        print(f"Failed to retrieve data from Wikipedia API. Error: {e}")
        return []


def get_pageviews(page_title, start_date="2020040100", end_date="2023040700"):
    headers = {
        'User-Agent': 'ResearchBot/1.0 (IBM Research; Academic/Research Purpose)',
        'Api-User-Agent': 'ResearchBot/1.0 (IBM Research; Academic/Research Purpose)',
        'Accept': 'application/json',
    }

    # Construct the API URL with the appropriate parameters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    # Make the HTTP GET request to the API
    try:
        response = requests.get(url, headers=headers, timeout=30)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            # Extract the pageview data
            print('retrieved for ', page_title)
            views = sum(item['views'] for item in data['items'])
            return views
        else:
            print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Failed to retrieve pageviews data for {page_title}. Error: {e}")
        return 0

def clean_str(p):
  try:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
  except:
    return ''

WIKIPEDIA_BOILERPLATE_PHRASES = [
    "This page was last edited on",
    "Text is available under the Creative Commons",
    "Wikipedia® is a registered trademark",
    "Terms of Use and Privacy Policy",
    "Wikimedia Foundation",
    "By using this site, you agree to the",
    "additional terms may apply",
]


def _is_boilerplate(text):
    """Check if a paragraph is Wikipedia boilerplate/metadata text."""
    return any(phrase in text for phrase in WIKIPEDIA_BOILERPLATE_PHRASES)


def filter_paragraph(paragraph_lst):
    return [p for p in paragraph_lst
            if len(p.split(" ")) > 2 and len(p.split(".")) > 1
            and not _is_boilerplate(p)]


def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def search_step(entity, output_more=False, _visited=None):
    """Fetch a Wikipedia article for *entity*, with a disk cache.

    Only top-level lookups (``_visited is None``) are cached — recursive
    disambiguation calls pass a ``_visited`` set and go straight to the impl.
    The cached value is the resolved ``(paragraphs, title)`` pair, so a repeated
    lookup of the same entity/title skips the network round trip entirely.
    """
    if _visited is not None:
        return _search_step_impl(entity, output_more, _visited)
    key = f"{output_more}|{entity.strip().lower()}"
    hit = _kg_cache_get(WIKIPEDIA_CACHE_DIR, key)
    if hit is not None:
        return hit["obs"], hit["entity"]
    obs, resolved = _search_step_impl(entity, output_more, set())
    _kg_cache_set(WIKIPEDIA_CACHE_DIR, key, {"obs": obs, "entity": resolved})
    return obs, resolved


def _search_step_impl(entity, output_more=False, _visited=None):
    # Cycle guard: a disambiguation page ("X may refer to:") recurses into the
    # bracketed query "[X]", whose search results point back to "X" — an
    # infinite loop (e.g. "UCG"). Track normalized entities we've already tried
    # and fall back to returning the current page instead of recursing again.
    if _visited is None:
        _visited = set()
    norm = entity.strip().strip("[]").lower()
    cycle = norm in _visited
    _visited.add(norm)

    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    headers = {
        "User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
        "Api-User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
        "Accept": "text/html",
    }
    response_text = requests.get(search_url, headers=headers, timeout=30).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
    if result_divs:  # mismatch
      result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
      # obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
      if cycle or result_titles[0].strip().strip("[]").lower() in _visited:
        # Would loop back to an entity we've already searched — stop here.
        print(f"Could not find {entity}. Similar entities: {result_titles[:5]}")
        obs = [f"Could not find {entity}. Similar: {result_titles[:5]}."]
      else:
        print(f"Could not find {entity}. Search for similar entities, {result_titles[0]}, instead")
        obs, entity = search_step(result_titles[0], output_more=output_more, _visited=_visited)
    else:
      print('found entity', entity)
      page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
      if any("may refer to:" in p for p in page) and not cycle:
        obs, entity = search_step("[" + entity + "]", output_more=output_more, _visited=_visited)
      else:
        page_ = ""
        for p in page:
          if len(p.split(" ")) > 2:
              page_ += clean_str(p)
              if not p.endswith("\n"):
                  page_ += "\n"
        obs = get_page_obs(page_)
        if output_more:
            obs = filter_paragraph(obs)
        else:
            obs = filter_paragraph(obs[:10])

    return obs, entity


# ===== Wikidata Utility Functions =====

_label_cache = {}

WIKIDATA_HEADERS = {
    "User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
    "Accept": "application/json",
}

# Boring/structural properties to filter out from Wikidata triples
WIKIDATA_PROPERTY_BLACKLIST = {
    'P31',    # instance of
    'P279',   # subclass of
    'P910',   # topic's main category
    'P1343',  # described by source
    'P373',   # Commons category
    'P18',    # image
    'P935',   # Commons gallery
    'P1424',  # topic's main template
    'P1151',  # topic's main Wikimedia portal
    'P2959',  # permanent duplicated item
    # Wikipedia/Wikimedia metadata properties
    'P5008',  # on focus list of Wikimedia project
    'P6104',  # maintained by WikiProject
    'P1740',  # category for films shot at this location
    'P1792',  # category of associated people
    'P163',   # flag (image)
    'P1464',  # category for people born here
    'P1465',  # category for people who died here
    'P1791',  # category of people buried here
    'P2354',  # has list
    'P1456',  # list of monuments
    'P1659',  # see also
    'P460',   # said to be the same as
    'P1889',  # different from
    'P2184',  # history of topic
    'P2579',  # studied by
    'P2633',  # geography of topic
}

# High-frequency generic predicates that produce ambiguous questions
WIKIDATA_PROPERTY_BLACKLIST_GENERIC = {
    'P1542',  # has effect
    'P710',   # participant
    'P527',   # has part(s)
    'P361',   # part of
    'P828',   # has cause
    'P1536',  # immediate cause of
    'P1478',  # has immediate cause
    'P2283',  # uses
    'P1382',  # partially coincident with
}

# Properties blacklisted only on the FIRST hop.
# These funnel through hubs, making every chain "Entity -> Hub -> random fact
# about the hub" and producing trivial or off-domain answers.
#   - Geographic hubs (country/region): "Event -> Country -> fact about country".
#   - Taxon hubs (found in taxon): a compound like GABA has ~90 "found in taxon"
#     edges, so nearly every 2-hop chain becomes "Compound -> plant/fungus ->
#     taxonomy fact (spore print color, CITES status, ...)". Those are unrelated
#     to the compound and get stripped by the semantic-fidelity filter, leaving
#     0 facts. A domain that genuinely wants P703 can re-enable it via
#     allowed_properties (see fetch_multihop_triples).
WIKIDATA_PROPERTY_BLACKLIST_HOP1 = {
    'P17',    # country
    'P276',   # location
    'P131',   # located in the administrative territorial entity
    'P703',   # found in taxon (compound -> organism; funnels into taxonomy)
}


def _extract_label_from_response(entity_id, data, language='en'):
    """Extract label from wbgetentities API response data."""
    entities = data.get('entities', {})
    if entity_id in entities:
        labels = entities[entity_id].get('labels', {})
        if language in labels:
            return labels[language]['value']
        if language != 'en' and 'en' in labels:
            return labels['en']['value']
        if labels:
            return next(iter(labels.values()))['value']
    return None


def get_entity_label(entity_id, language='en'):
    """
    Fetch human-readable label for a Wikidata entity or property ID.
    e.g., Q25188 -> "Inception", P57 -> "director"
    Uses module-level cache to avoid redundant API calls.
    Retries with exponential backoff on 429/503.
    """
    if entity_id in _label_cache:
        return _label_cache[entity_id]

    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&props=labels&format=json"
    _MAX_RETRIES = 6
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(url, headers=WIKIDATA_HEADERS, timeout=30)
            if response.status_code == 200:
                data = response.json()
                label = _extract_label_from_response(entity_id, data, language)
                if label:
                    _label_cache[entity_id] = label
                    return label
                break  # 200 but no label found — don't retry
            elif response.status_code in (429, 503):
                wait = min((2 ** attempt) * 5, 60) + random.random() * 5
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                    continue
            else:
                break  # non-retryable error
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(min((2 ** attempt) * 5, 60))
                continue
            print(f"Error fetching label for {entity_id}: {e}")

    _label_cache[entity_id] = entity_id  # fallback to raw ID
    print(f"Warning: no human-readable label found for {entity_id}, using raw ID")
    return entity_id


def get_entity_labels_batch(entity_ids, language='en'):
    """Batch-resolve labels for multiple Wikidata entities.

    Uses wbgetentities API with up to 50 IDs per call.
    Updates _label_cache and returns {entity_id: label} dict.
    Retries with exponential backoff on 429/503.
    """
    uncached = [eid for eid in set(entity_ids) if eid not in _label_cache]
    if not uncached:
        return {eid: _label_cache[eid] for eid in entity_ids}

    for i in range(0, len(uncached), 50):
        batch = uncached[i:i + 50]
        ids_param = "|".join(batch)
        url = (f"https://www.wikidata.org/w/api.php?action=wbgetentities"
               f"&ids={ids_param}&props=labels&format=json")

        success = False
        _MAX_RETRIES = 6
        for attempt in range(_MAX_RETRIES):
            try:
                response = requests.get(url, headers=WIKIDATA_HEADERS, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    for eid in batch:
                        label = _extract_label_from_response(eid, data, language)
                        _label_cache[eid] = label if label else eid
                    success = True
                    break
                elif response.status_code in (429, 503):
                    wait = min((2 ** attempt) * 5, 60) + random.random() * 5
                    print(f"Batch label rate-limited (HTTP {response.status_code}), "
                          f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                else:
                    break
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(min((2 ** attempt) * 5, 60))
                    continue
                print(f"Error in batch label fetch: {e}", flush=True)

        if not success:
            for eid in batch:
                _label_cache.setdefault(eid, eid)

        # Rate limit between batches
        if i + 50 < len(uncached):
            time.sleep(1)

    return {eid: _label_cache.get(eid, eid) for eid in entity_ids}


def fetch_wikipedia_for_entities(chains):
    """Fetch Wikipedia articles for all unique entities in Wikidata triple chains.

    Args:
        chains: List of chain dicts from fetch_multihop_triples()

    Returns:
        List of {'qid': str, 'title': str, 'paragraph': str} dicts
    """
    # Collect unique entity IDs from chains, tracking which chain(s) each belongs to
    entity_chain_map = {}  # qid -> set of 1-based chain indices
    for chain_idx, chain in enumerate(chains, 1):
        for hop in chain.get('chain', []):
            eid = hop.get('entity', {}).get('id', '')
            vid = hop.get('value', {}).get('id', '')
            if eid.startswith('Q'):
                entity_chain_map.setdefault(eid, set()).add(chain_idx)
            if vid.startswith('Q'):
                entity_chain_map.setdefault(vid, set()).add(chain_idx)

    entity_ids = set(entity_chain_map.keys())

    if not entity_ids:
        return []

    # Batch-resolve Q-IDs to enwiki titles via wbgetentities sitelinks
    wiki_titles = {}  # qid -> title
    id_list = sorted(entity_ids)
    for i in range(0, len(id_list), 50):
        batch = id_list[i:i+50]
        ids_param = "|".join(batch)
        url = (
            f"https://www.wikidata.org/w/api.php?action=wbgetentities"
            f"&ids={ids_param}&props=sitelinks&sitefilter=enwiki&format=json"
        )
        _MAX_RETRIES = 6
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, headers=WIKIDATA_HEADERS, timeout=30)
                if resp.status_code == 200:
                    entities = resp.json().get('entities', {})
                    for qid, ent in entities.items():
                        title = ent.get('sitelinks', {}).get('enwiki', {}).get('title')
                        if title:
                            wiki_titles[qid] = title
                    break  # success
                elif resp.status_code in (429, 503):
                    wait = min((2 ** attempt) * 5, 60) + random.random() * 5
                    print(f"Sitelinks rate-limited (HTTP {resp.status_code}), "
                          f"retry {attempt+1}/{_MAX_RETRIES} in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                else:
                    print(f"Sitelinks batch returned HTTP {resp.status_code}, skipping",
                          flush=True)
                    break
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(min((2 ** attempt) * 5, 60))
                    continue
                print(f"Error fetching sitelinks for batch: {e}")
        time.sleep(1)  # rate limit

    print(f"Resolved {len(wiki_titles)}/{len(entity_ids)} entities to Wikipedia titles", flush=True)

    # Fetch Wikipedia article content for each title
    articles = []
    for qid, title in wiki_titles.items():
        try:
            paragraphs, _ = search_step(title, output_more=True)
            content = "\n".join(paragraphs) if isinstance(paragraphs, list) else str(paragraphs)
            if len(content.strip()) > 100:
                articles.append({
                    'qid': qid, 'title': title, 'paragraph': content,
                    'chain_indices': sorted(entity_chain_map.get(qid, [])),
                })
                print(f"  Fetched Wikipedia article: {title} ({len(content)} chars)", flush=True)
            else:
                print(f"  Skipped {title}: too short ({len(content)} chars)", flush=True)
        except Exception as e:
            print(f"  Error fetching Wikipedia article for {title}: {e}")
        time.sleep(0.5)  # rate limit

    print(f"Fetched {len(articles)} Wikipedia articles for {len(entity_ids)} chain entities", flush=True)
    return articles


def search_wikidata_entities(query, language='en', limit=10):
    """
    Search Wikidata entities by text query.
    Returns list of {'id', 'label', 'description'} dicts.
    """
    url = (
        f"https://www.wikidata.org/w/api.php?action=wbsearchentities"
        f"&search={requests.utils.quote(query)}&language={language}"
        f"&limit={limit}&format=json"
    )
    try:
        response = requests.get(url, headers=WIKIDATA_HEADERS, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get('search', []):
                results.append({
                    'id': item.get('id', ''),
                    'label': item.get('label', ''),
                    'description': item.get('description', ''),
                })
            return results
    except Exception as e:
        print(f"Error searching Wikidata for '{query}': {e}")
    return []


def get_entity_data(entity_id, language='en'):
    """
    Fetch full entity data including claims/properties from Wikidata.
    Parses snak types: wikibase-entityid, time, string, quantity, monolingualtext.
    Returns simplified {'id', 'label', 'description', 'claims': {property_id: [values]}}.
    Retries with exponential backoff on 429/503.
    Uses batch label resolution for entity-reference claims.
    """
    cache_key = f"entity|{entity_id}|{language}"
    cached = _kg_cache_get(WIKIDATA_CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = (
        f"https://www.wikidata.org/w/api.php?action=wbgetentities"
        f"&ids={entity_id}&props=labels|descriptions|claims"
        f"&languages={language}&format=json"
    )
    entity = None
    _MAX_RETRIES = 6
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(url, headers=WIKIDATA_HEADERS, timeout=30)
            if response.status_code == 200:
                data = response.json()
                entity = data.get('entities', {}).get(entity_id, {})
                break
            elif response.status_code in (429, 503):
                wait = min((2 ** attempt) * 5, 60) + random.random() * 5
                print(f"get_entity_data rate-limited (HTTP {response.status_code}) for {entity_id}, "
                      f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                return None
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(min((2 ** attempt) * 5, 60))
                continue
            print(f"Error fetching entity data for {entity_id}: {e}")
            return None

    if entity is None:
        return None

    label = entity.get('labels', {}).get(language, {}).get('value', None)
    if not label:
        label = get_entity_label(entity_id, language)
    description = entity.get('descriptions', {}).get(language, {}).get('value', '')

    # Collect all entity-reference QIDs for batch label resolution
    ref_ids = set()
    for prop_id, claim_list in entity.get('claims', {}).items():
        for claim in claim_list:
            mainsnak = claim.get('mainsnak', {})
            datavalue = mainsnak.get('datavalue', {})
            if datavalue.get('type') == 'wikibase-entityid':
                ref_id = datavalue.get('value', {}).get('id', '')
                if ref_id:
                    ref_ids.add(ref_id)

    # Batch-resolve all referenced entity labels at once (instead of N individual calls)
    if ref_ids:
        get_entity_labels_batch(list(ref_ids), language)

    # Build claims using cached labels
    claims = {}
    for prop_id, claim_list in entity.get('claims', {}).items():
        values = []
        for claim in claim_list:
            mainsnak = claim.get('mainsnak', {})
            datavalue = mainsnak.get('datavalue', {})
            dtype = datavalue.get('type', '')
            val = datavalue.get('value', {})

            if dtype == 'wikibase-entityid':
                ref_id = val.get('id', '')
                ref_label = _label_cache.get(ref_id, ref_id)
                values.append({'type': 'entity', 'id': ref_id, 'label': ref_label})
            elif dtype == 'time':
                time_str = val.get('time', '')
                values.append({'type': 'time', 'value': time_str})
            elif dtype == 'string':
                values.append({'type': 'string', 'value': val})
            elif dtype == 'quantity':
                amount = val.get('amount', '')
                unit = val.get('unit', '')
                values.append({'type': 'quantity', 'amount': amount, 'unit': unit})
            elif dtype == 'monolingualtext':
                values.append({'type': 'text', 'value': val.get('text', ''), 'language': val.get('language', '')})
            else:
                values.append({'type': dtype, 'raw': val})

        if values:
            claims[prop_id] = values

    _label_cache[entity_id] = label

    result = {
        'id': entity_id,
        'label': label,
        'description': description,
        'claims': claims,
    }
    _kg_cache_set(WIKIDATA_CACHE_DIR, cache_key, result)
    return result


def sparql_query(query_string, max_retries=5):
    """
    Execute a SPARQL query against Wikidata Query Service.
    Returns response['results']['bindings'] or empty list on failure.
    Retries with exponential backoff on 429/503/timeout errors.
    """
    cache_key = "sparql|" + query_string
    cached = _kg_cache_get(WIKIDATA_CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "ResearchBot/1.0 (IBM Research; Academic/Research Purpose)",
        "Accept": "application/sparql-results+json",
    }
    # Add jitter to initial delay to avoid thundering herd from parallel jobs
    time.sleep(1 + random.random() * 3)
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                params={'query': query_string, 'format': 'json'},
                headers=headers,
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                _kg_cache_set(WIKIDATA_CACHE_DIR, cache_key, bindings)
                return bindings
            elif response.status_code in (429, 503):
                wait = (2 ** attempt) * 5 + random.random() * 5
                print(f"SPARQL rate-limited (HTTP {response.status_code}), retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            else:
                print(f"SPARQL query failed with status {response.status_code}: {response.text[:200]}")
                return []
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = (2 ** attempt) * 5 + random.random() * 5
            print(f"SPARQL request error: {e}, retry {attempt+1}/{max_retries} in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            return []
    print(f"SPARQL query failed after {max_retries} retries", flush=True)
    return []


# Borderline predicates: not blocked but deprioritized in diversification scoring
PENALIZED_PROPERTIES = {
    'P150',   # contains administrative territorial entity
    'P47',    # shares border with
    'P530',   # diplomatic relation
    'P737',   # influenced by
    'P941',   # inspired by
}
PENALTY_FACTOR = 0.1  # 10x less likely to be selected


def _diversify_chains(chains, limit):
    """Select a diverse subset of chains, preferring rare intermediates and property pairs.

    Scores each chain by rarity (1 / (intermediate_freq * property_pair_freq)) and
    greedily selects chains with caps per intermediate and per property pair to avoid
    hub-dominated results.

    Args:
        chains: List of chain dicts from SPARQL results
        limit: Maximum number of chains to return

    Returns:
        Diversified list of chains (up to limit)
    """
    if len(chains) <= limit:
        return chains

    # Count frequency of each intermediate entity and each (p1, p2) property pair
    intermediate_counts = Counter()
    property_pair_counts = Counter()
    chain_keys = []

    for chain in chains:
        hops = chain.get('chain', [])
        if len(hops) >= 2:
            intermediate = hops[0].get('value', {}).get('id', '')
            p1 = hops[0].get('property', {}).get('id', '')
            p2 = hops[1].get('property', {}).get('id', '')
        else:
            intermediate = ''
            p1 = ''
            p2 = ''
        intermediate_counts[intermediate] += 1
        property_pair_counts[(p1, p2)] += 1
        chain_keys.append((intermediate, (p1, p2)))

    # Score each chain by rarity: prefer less common intermediates and properties
    scored = []
    for i, (inter, pp) in enumerate(chain_keys):
        inter_freq = intermediate_counts[inter]
        pp_freq = property_pair_counts[pp]
        rarity = 1.0 / (inter_freq * pp_freq)
        # Penalize chains with generic/high-frequency properties
        p1, p2 = pp
        if p1 in PENALIZED_PROPERTIES or p2 in PENALIZED_PROPERTIES:
            rarity *= PENALTY_FACTOR
        scored.append((rarity, i))

    # Sort by rarity descending (rarest first)
    scored.sort(key=lambda x: -x[0])

    # Greedy selection with caps
    max_per_intermediate = max(2, limit // 5)
    max_per_property_pair = max(2, limit // 4)

    selected = []
    inter_used = Counter()
    pp_used = Counter()

    for _rarity, idx in scored:
        if len(selected) >= limit:
            break
        inter, pp = chain_keys[idx]
        if inter_used[inter] < max_per_intermediate and pp_used[pp] < max_per_property_pair:
            selected.append(idx)
            inter_used[inter] += 1
            pp_used[pp] += 1

    # Second pass with relaxed caps to fill remaining slots
    if len(selected) < limit:
        selected_set = set(selected)
        for _rarity, idx in scored:
            if len(selected) >= limit:
                break
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    result = [chains[i] for i in sorted(selected)]

    # Log diversity stats
    result_intermediates = set()
    result_pp = set()
    for chain in result:
        hops = chain.get('chain', [])
        if len(hops) >= 2:
            result_intermediates.add(hops[0].get('value', {}).get('label', ''))
            result_pp.add((hops[0].get('property', {}).get('label', ''),
                           hops[1].get('property', {}).get('label', '')))
    print(f"  Diversified {len(chains)} -> {len(result)} chains: "
          f"{len(result_intermediates)} unique intermediates, "
          f"{len(result_pp)} unique property pairs", flush=True)

    return result


def fetch_multihop_triples(entity_id, num_hops=2, limit=50,
                           min_sitelinks=10, allowed_properties=None):
    """
    Fetch multi-hop relationship chains from Wikidata via SPARQL.
    For 2-hop: Entity1 -[property]-> Entity2 -[property]-> Entity3

    Filters out boring properties (P31, P279, etc.), circular chains,
    and entities with fewer than ``min_sitelinks`` sitelinks.

    Over-fetches raw results and then diversifies to avoid hub-dominated chains
    (e.g., 19/20 chains going through "Japan").

    Args:
        entity_id: Wikidata QID (e.g., "Q123").
        num_hops: Number of hops (currently only 2 supported).
        limit: Max chains to return after diversification.
        min_sitelinks: Minimum sitelinks for intermediate/end entities (default 10).
            Lower values (e.g., 3) help niche domains like security.
        allowed_properties: Optional set of property IDs (e.g., {"P710", "P828"})
            that bypass ALL blacklists.  Useful for domain-specific properties.

    Returns list of chain dicts:
    [{'chain': [{'entity': ..., 'property': ..., 'value': ...}, ...],
      'start_entity': {'id': ..., 'label': ...},
      'end_entity': {'id': ..., 'label': ...},
      'path_description': '...'}]

    Falls back to 1-hop + Python combination if 2-hop query times out.
    """
    fetch_limit = min(limit * 10, 500)

    # Compute effective blacklists — allowed_properties bypass all blacklists
    _allowed = allowed_properties or set()
    all_blacklisted = (WIKIDATA_PROPERTY_BLACKLIST | WIKIDATA_PROPERTY_BLACKLIST_GENERIC) - _allowed
    blacklist_filter = " ".join(
        f"FILTER(?p1prop != wd:{p}) ." for p in all_blacklisted
    )
    blacklist_filter2 = " ".join(
        f"FILTER(?p2prop != wd:{p}) ." for p in all_blacklisted
    )
    # Block geographic hub properties on the first hop only
    _hop1_blacklist = WIKIDATA_PROPERTY_BLACKLIST_HOP1 - _allowed
    blacklist_filter_hop1 = " ".join(
        f"FILTER(?p1prop != wd:{p}) ." for p in _hop1_blacklist
    )

    # 2-hop SPARQL query
    query_2hop = f"""
    SELECT ?mid ?midLabel ?end ?endLabel ?p1 ?p1Label ?p2 ?p2Label ?p1prop ?p1propLabel ?p2prop ?p2propLabel WHERE {{
      wd:{entity_id} ?p1 ?mid .
      ?mid ?p2 ?end .

      # Resolve property IDs
      ?p1prop wikibase:directClaim ?p1 .
      ?p2prop wikibase:directClaim ?p2 .

      # Only entities (not literals)
      FILTER(STRSTARTS(STR(?mid), "http://www.wikidata.org/entity/Q"))
      FILTER(STRSTARTS(STR(?end), "http://www.wikidata.org/entity/Q"))

      # No circular chains
      FILTER(?mid != wd:{entity_id})
      FILTER(?end != wd:{entity_id})
      FILTER(?mid != ?end)

      # Filter boring properties
      {blacklist_filter}
      {blacklist_filter2}

      # Block geographic-hub properties on first hop
      {blacklist_filter_hop1}

      # Only notable entities (with sitelinks)
      ?mid wikibase:sitelinks ?midSitelinks .
      FILTER(?midSitelinks > {min_sitelinks})
      ?end wikibase:sitelinks ?endSitelinks .
      FILTER(?endSitelinks > {min_sitelinks})

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY MD5(CONCAT(STR(?mid), STR(?end), STR(?p1), STR(?p2)))
    LIMIT {fetch_limit}
    """

    start_label = get_entity_label(entity_id)
    results = sparql_query(query_2hop)

    chains = []
    if results:
        # Batch-resolve unresolved labels from SPARQL results (instead of N individual calls)
        unresolved_ids = set()
        for row in results:
            mid_id = row.get('mid', {}).get('value', '').split('/')[-1]
            mid_label = row.get('midLabel', {}).get('value', mid_id)
            if re.match(r'^Q\d+$', mid_label):
                unresolved_ids.add(mid_id)
            end_id = row.get('end', {}).get('value', '').split('/')[-1]
            end_label = row.get('endLabel', {}).get('value', end_id)
            if re.match(r'^Q\d+$', end_label):
                unresolved_ids.add(end_id)
        if unresolved_ids:
            get_entity_labels_batch(list(unresolved_ids))

        for row in results:
            mid_id = row.get('mid', {}).get('value', '').split('/')[-1]
            mid_label = row.get('midLabel', {}).get('value', mid_id)
            if re.match(r'^Q\d+$', mid_label):
                mid_label = _label_cache.get(mid_id, mid_id)
            end_id = row.get('end', {}).get('value', '').split('/')[-1]
            end_label = row.get('endLabel', {}).get('value', end_id)
            if re.match(r'^Q\d+$', end_label):
                end_label = _label_cache.get(end_id, end_id)
            p1_label = row.get('p1propLabel', {}).get('value', row.get('p1Label', {}).get('value', ''))
            p2_label = row.get('p2propLabel', {}).get('value', row.get('p2Label', {}).get('value', ''))
            p1_id = row.get('p1prop', {}).get('value', '').split('/')[-1]
            p2_id = row.get('p2prop', {}).get('value', '').split('/')[-1]

            chain = {
                'chain': [
                    {'entity': {'id': entity_id, 'label': start_label},
                     'property': {'id': p1_id, 'label': p1_label},
                     'value': {'id': mid_id, 'label': mid_label}},
                    {'entity': {'id': mid_id, 'label': mid_label},
                     'property': {'id': p2_id, 'label': p2_label},
                     'value': {'id': end_id, 'label': end_label}},
                ],
                'start_entity': {'id': entity_id, 'label': start_label},
                'end_entity': {'id': end_id, 'label': end_label},
                'path_description': f'"{start_label}" -[{p1_label}]-> "{mid_label}" -[{p2_label}]-> "{end_label}"',
            }
            chains.append(chain)

        print(f"Fetched {len(chains)} raw 2-hop chains for {entity_id} ({start_label})", flush=True)
        chains = _diversify_chains(chains, limit)
        return chains

    # Fallback: 1-hop query and combine in Python
    print(f"2-hop query returned no results for {entity_id}, trying 1-hop fallback...", flush=True)

    all_blacklisted_1hop = (WIKIDATA_PROPERTY_BLACKLIST | WIKIDATA_PROPERTY_BLACKLIST_GENERIC) - _allowed
    blacklist_filter_1hop = " ".join(
        f"FILTER(?pprop != wd:{p}) ." for p in all_blacklisted_1hop
    )
    # Also block geographic-hub properties on the first hop in fallback
    _hop1_blacklist_1hop = WIKIDATA_PROPERTY_BLACKLIST_HOP1 - _allowed
    blacklist_filter_1hop_geo = " ".join(
        f"FILTER(?pprop != wd:{p}) ." for p in _hop1_blacklist_1hop
    )

    query_1hop = f"""
    SELECT ?obj ?objLabel ?p ?pLabel ?pprop ?ppropLabel WHERE {{
      wd:{entity_id} ?p ?obj .
      ?pprop wikibase:directClaim ?p .
      FILTER(STRSTARTS(STR(?obj), "http://www.wikidata.org/entity/Q"))
      FILTER(?obj != wd:{entity_id})
      {blacklist_filter_1hop}
      {blacklist_filter_1hop_geo}
      ?obj wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > {min_sitelinks})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    hop1_results = sparql_query(query_1hop)
    if not hop1_results:
        print(f"1-hop fallback also returned no results for {entity_id}", flush=True)
        return []

    # For each 1-hop result, try to get a second hop
    import random
    random.shuffle(hop1_results)

    for row in hop1_results[:20]:  # Limit second-hop attempts
        mid_id = row.get('obj', {}).get('value', '').split('/')[-1]
        mid_label = row.get('objLabel', {}).get('value', mid_id)
        p1_label = row.get('ppropLabel', {}).get('value', row.get('pLabel', {}).get('value', ''))
        p1_id = row.get('pprop', {}).get('value', '').split('/')[-1]

        # Second hop from mid entity
        query_2nd_hop = f"""
        SELECT ?obj ?objLabel ?p ?pLabel ?pprop ?ppropLabel WHERE {{
          wd:{mid_id} ?p ?obj .
          ?pprop wikibase:directClaim ?p .
          FILTER(STRSTARTS(STR(?obj), "http://www.wikidata.org/entity/Q"))
          FILTER(?obj != wd:{mid_id})
          FILTER(?obj != wd:{entity_id})
          {blacklist_filter_1hop}
          ?obj wikibase:sitelinks ?sitelinks .
          FILTER(?sitelinks > {min_sitelinks})
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 5
        """

        hop2_results = sparql_query(query_2nd_hop)
        for row2 in hop2_results:
            end_id = row2.get('obj', {}).get('value', '').split('/')[-1]
            end_label = row2.get('objLabel', {}).get('value', end_id)
            p2_label = row2.get('ppropLabel', {}).get('value', row2.get('pLabel', {}).get('value', ''))
            p2_id = row2.get('pprop', {}).get('value', '').split('/')[-1]

            chain = {
                'chain': [
                    {'entity': {'id': entity_id, 'label': start_label},
                     'property': {'id': p1_id, 'label': p1_label},
                     'value': {'id': mid_id, 'label': mid_label}},
                    {'entity': {'id': mid_id, 'label': mid_label},
                     'property': {'id': p2_id, 'label': p2_label},
                     'value': {'id': end_id, 'label': end_label}},
                ],
                'start_entity': {'id': entity_id, 'label': start_label},
                'end_entity': {'id': end_id, 'label': end_label},
                'path_description': f'"{start_label}" -[{p1_label}]-> "{mid_label}" -[{p2_label}]-> "{end_label}"',
            }
            chains.append(chain)

        if len(chains) >= limit:
            break

    print(f"Fetched {len(chains)} raw 2-hop chains (via fallback) for {entity_id} ({start_label})", flush=True)
    chains = _diversify_chains(chains, limit)
    return chains