"""Release review of 2026-11-18 (whole change, B): the record masks meet a
tiled read. A vision call that fails inside a tiled read leaves its
exception's text one level down in the tool's result (a tile's ``error``,
``tiles_not_read``, ``overview_not_read``). The guard on a tool's error
result and the run record mask a TOP-LEVEL ``error`` only, so a key quoted
in that text reached the model and the record. The text is now masked where
it is made.

Made-up values only.
"""

import json

from aec_agent_core import vision_tools
from aec_agent_core.activity_log import mask_payload

SECRET = "sk-madeup1234567890abcd"


def _failed():
    return RuntimeError(
        f"401 from https://gateway.example/v1?api_key={SECRET}")


def test_a_failed_vision_calls_text_is_masked_where_it_is_made():
    text = vision_tools._vision_error(_failed())
    assert SECRET not in text
    assert "401" in text            # the rest of the error is still there


def test_a_tiles_error_is_masked_in_the_result_and_the_record():
    ok = {"tile": "r1c1", "view": [0, 0, 10, 10], "analysis": "text"}
    bad = {"tile": "r1c2", "view": [10, 0, 20, 10],
           "error": vision_tools._vision_error(_failed())}
    raw = json.dumps({
        "page": 0, "analysis": "layout", "tiles": [ok, bad],
        "tiles_not_read": vision_tools._tiles_not_read_note([ok, bad])})
    assert SECRET not in raw                              # the model
    assert SECRET not in mask_payload("tool_end", "result", raw)  # record


def test_an_error_with_no_secret_is_kept_word_for_word():
    exc = ValueError("page 7 could not be rendered at 150 dpi")
    from aec_agent_core.deep.vision_engine import describe_error
    assert vision_tools._vision_error(exc) == describe_error(exc)
