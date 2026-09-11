# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

from drbench.tool_util import extract_json_v2


def test_extract_json_v2_parses_non_json_fenced_dict():
    response = """Result:\n```python\n{'name': 'example', 'score': 1}\n```"""

    assert extract_json_v2(response, None) == [{"name": "example", "score": 1}]
