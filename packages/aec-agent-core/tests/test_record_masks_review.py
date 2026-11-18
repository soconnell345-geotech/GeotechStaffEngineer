"""The review of the record's masker (``module_work/record_masks/REVIEW.md``):
one test per finding, each failing before its fix.

Every value here is made up (``Made0Up...``, ``example.test``), never a
live key of any vendor.
"""

from __future__ import annotations

import json
import time

import pytest

from aec_agent_core.error_text import (error_line, mask_error_result,
                                       mask_error_secrets)

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"


# ---------------------------------------------------------------------------
# a private-key block with no END line
# ---------------------------------------------------------------------------

def test_many_unfinished_key_blocks_do_not_hold_the_turn():
    # 8,000 BEGIN lines and no END: about 8 s before the fix (each BEGIN
    # searched to the end of the text), 268 s for a 2 MB text.
    text = "-----BEGIN PRIVATE KEY-----" * 8000
    t0 = time.perf_counter()
    out = mask_error_secrets(text)
    assert time.perf_counter() - t0 < 1.0
    assert out == "…"


def test_a_key_block_cut_short_is_masked_to_the_end():
    text = ("bad key\n-----BEGIN RSA PRIVATE KEY-----\nMADEUPMADEUPMADEUP\n"
            "MOREMADEUP")
    assert mask_error_secrets(text) == "bad key\n…"
    whole = ("bad key\n-----BEGIN PRIVATE KEY-----\nMADEUPMADEUP\n"
             "-----END PRIVATE KEY-----\nend")
    assert mask_error_secrets(whole) == "bad key\n…\nend"


# ---------------------------------------------------------------------------
# a secret's name at the end of a longer one (a setting's name)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,masked", [
    (f"GRAPH_CLIENT_SECRET={VALUE} was refused",
     "GRAPH_CLIENT_SECRET=… was refused"),
    (f"env PROMPTER_API_KEY='{VALUE}'", "env PROMPTER_API_KEY='…'"),
    (f"GRAPH_CLIENT_KEY_PASSWORD: {VALUE}", "GRAPH_CLIENT_KEY_PASSWORD: …"),
    (f"PROMPTER-API-KEY--J.DOE={VALUE}", "PROMPTER-API-KEY--J.DOE=…"),
    (f"MADE_UP_ACCESS_TOKEN={VALUE}; retry", "MADE_UP_ACCESS_TOKEN=…; retry"),
    (f"made_up_secret_access_key={VALUE}", "made_up_secret_access_key=…"),
    ('{"private_key": "' + VALUE + '", "client_email": "a@example.test"}',
     '{"private_key": "…", "client_email": "a@example.test"}'),
    (f"X-Auth-Token: {VALUE}", "X-Auth-Token: …"),
    (f"passphrase={VALUE}", "passphrase=…"),
])
def test_a_secret_named_as_a_setting_is_masked(text, masked):
    assert mask_error_secrets(text) == masked
    assert mask_error_secrets(masked) == masked


def test_names_that_only_end_like_a_secret_keep_ordinary_values():
    for text in ("GRAPH_CLIENT_SECRET is not set", "PROMPTER_API_KEY=None",
                 "no_password=true", "max_tokens=4096",
                 "the private key file was not found",
                 "set GRAPH_CLIENT_SECRET and GRAPH_CLIENT_ID"):
        assert mask_error_secrets(text) == text


# ---------------------------------------------------------------------------
# key / token / secret outside a URL's query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,masked", [
    (json.dumps({"key": VALUE, "model": "m"}),
     '{"key": "…", "model": "m"}'),
    (json.dumps({"token": VALUE}), '{"token": "…"}'),
    (json.dumps({"secret": VALUE}), '{"secret": "…"}'),
    ("sent {'key': '" + VALUE + "', 'token': '" + VALUE + "'}",
     "sent {'key': '…', 'token': '…'}"),
    (f"body: token={VALUE}&grant_type=x", "body: token=…&grant_type=x"),
    (f"body: code={VALUE}&redirect_uri=x", "body: code=…&redirect_uri=x"),
    (f"connection refused: secret={VALUE} was sent",
     "connection refused: secret=… was sent"),
    # inside a tool's JSON error result the quotes are escaped
    (json.dumps({"error": 'body {"token": "' + VALUE + '"}'}),
     json.dumps({"error": 'body {"token": "…"}'}, ensure_ascii=False)),
])
def test_a_key_or_token_outside_a_query_is_masked(text, masked):
    assert mask_error_secrets(text) == masked
    assert mask_error_secrets(masked) == masked


def test_an_ordinary_value_under_key_token_or_code_is_kept():
    for text in (
            '{"error": "page 7 of 12: no text layer", "key": "sheet_index", '
            '"token": "page_count"}',
            "the token limit was reached: max_tokens=4096, "
            "code=context_length_exceeded",
            '{"code": "invalid_value", "param": "max_tokens"}',
            "DXF handle 2B7: key=LAYER code=8",
            "KeyError: 'token'",
            "key 'A1' on sheet S-101; the key of schedule mark W-12",
            "signature: required", "token: page 12 of 40"):
        assert mask_error_secrets(text) == text


# ---------------------------------------------------------------------------
# a cookie header
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,masked", [
    (f"sent Cookie: sessionid={VALUE}; theme=dark and was refused",
     "sent Cookie: … and was refused"),
    (f"Set-Cookie: SESSION={VALUE}; Path=/; HttpOnly",
     "Set-Cookie: …; HttpOnly"),
    ("headers {'Cookie': 'sid=" + VALUE + "', 'Accept': 'text/html'}",
     "headers {'Cookie': '…', 'Accept': 'text/html'}"),
])
def test_a_cookie_header_is_masked(text, masked):
    assert mask_error_secrets(text) == masked
    assert mask_error_secrets(masked) == masked


def test_the_word_cookie_in_prose_is_kept():
    for text in ("the cookie was rejected", "Cookie: none sent",
                 "cookie banner on page 3"):
        assert mask_error_secrets(text) == text
