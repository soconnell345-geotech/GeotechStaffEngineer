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


# ---------------------------------------------------------------------------
# a secret in a URL that is itself a parameter's value (percent-encoded)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,masked", [
    ("302 to https://login.example.test/?redirect_uri=https%3A%2F%2F"
     f"app.example.test%2Fcb%3Faccess_token%3D{VALUE}%26state%3D1",
     "302 to https://login.example.test/?redirect_uri=https%3A%2F%2F"
     "app.example.test%2Fcb%3Faccess_token%3D…%26state%3D1"),
    ("refused next=https%3A%2F%2Fstore.example.test%2Ff.pdf%3Fsv%3D1"
     f"%26sig%3D{VALUE}%26se%3D2",
     "refused next=https%3A%2F%2Fstore.example.test%2Ff.pdf%3Fsv%3D1"
     "%26sig%3D…%26se%3D2"),
])
def test_a_secret_in_an_encoded_url_is_masked(text, masked):
    assert mask_error_secrets(text) == masked
    assert mask_error_secrets(masked) == masked


def test_an_encoded_url_with_no_secret_is_kept():
    text = ("404 for url: https://files.example.test/open?src=https%3A%2F%2F"
            "files.example.test%2FShared%2520Documents%2Freport.pdf"
            "%3Fpage%3D3%26version%3D12")
    assert mask_error_secrets(text) == text


# ---------------------------------------------------------------------------
# a URL's password with an ``@`` in it
# ---------------------------------------------------------------------------

def test_a_url_password_with_an_at_sign_is_masked_whole():
    text = (f"ProxyError: cannot connect to http://someone:made@up{VALUE}@"
            "proxy.example.test:8080/path")
    assert mask_error_secrets(text) == (
        "ProxyError: cannot connect to http://someone:…@"
        "proxy.example.test:8080/path")
    for plain in ("cannot open https://host.example.test:8443/a/b@c.pdf",
                  "see http://host.example.test:8080 or mail a@example.test"):
        assert mask_error_secrets(plain) == plain


# ---------------------------------------------------------------------------
# a tool's OWN error text, on its way to the model (ToolErrorGuard)
# ---------------------------------------------------------------------------

SIGNED = f"https://files.example.test/a/d.pdf?download=1&tempauth={VALUE}"


def _through_the_guard(content, **kw):
    import asyncio
    from types import SimpleNamespace

    from langchain_core.messages import ToolMessage

    from aec_agent_core.deep.tool_guards import ToolErrorGuard
    request = SimpleNamespace(tool_call={"name": "fetch", "id": "c1",
                                         "args": {}})
    result = (ToolMessage(content=content, name="fetch", tool_call_id="c1",
                          **kw) if isinstance(content, (str, list))
              else content)
    out = ToolErrorGuard().wrap_tool_call(request, lambda _r: result)

    async def handler(_r):
        return result
    again = asyncio.run(ToolErrorGuard().awrap_tool_call(request, handler))
    assert getattr(again, "content", again) == getattr(out, "content", out)
    return result, out


def test_a_tools_own_error_text_is_masked_before_the_model_reads_it():
    content = json.dumps({
        "error": f"HTTPError: 403 Client Error for url: {SIGNED}",
        "hint": "ask for a new link", "link": SIGNED})
    result, out = _through_the_guard(content, status="error")
    data = json.loads(out.content)
    assert VALUE not in data["error"]
    assert data["error"] == ("HTTPError: 403 Client Error for url: https://"
                             "files.example.test/a/d.pdf?download=1"
                             "&tempauth=…")
    # only the error's text: the other keys, the name, the id, the status
    assert data["hint"] == "ask for a new link" and data["link"] == SIGNED
    assert list(data) == ["error", "hint", "link"]
    assert (out.name, out.tool_call_id, out.status) == ("fetch", "c1",
                                                        "error")
    assert result.content == content            # the tool's own, untouched


@pytest.mark.parametrize("content", [
    # an error with no secret, spaced and escaped its own way
    '{"error":"page 7 of 12: no text layer",  "page":7, "note":"caf\\u00e9"}',
    json.dumps({"error": "FileNotFoundError: 'Site_Report_Rev2_Final.pdf'",
                "request_id": "req_0123456789abcdef0123456789abcdef"}),
    # not an error: a result that holds a signed link, and the word "error"
    json.dumps({"link": SIGNED, "note": 'no "error" here'}),
    json.dumps({"error": None, "link": SIGNED}),
    json.dumps({"error": {"message": "nested"}, "link": SIGNED}),
    json.dumps([{"error": f"tempauth={VALUE}"}]),
    f'plain text with "error" and tempauth={VALUE}',
    '{"error": "cut off JSON, tempauth=' + VALUE,
    "",
])
def test_every_other_result_passes_the_guard_as_the_same_object(content):
    result, out = _through_the_guard(content)
    assert out is result and out.content == content


def test_a_result_that_is_no_text_passes_the_guard_untouched():
    blocks = [{"type": "text", "text": '{"error": "tempauth=' + VALUE + '"}'}]
    result, out = _through_the_guard(blocks)
    assert out is result
    other = object()
    assert _through_the_guard(other)[1] is other


def test_an_error_text_that_cannot_be_checked_is_not_handed_on(monkeypatch):
    from aec_agent_core import error_text
    from aec_agent_core.deep import tool_guards

    def broken(_text):
        raise RuntimeError("the masker broke")
    monkeypatch.setattr(error_text, "_mask", broken)
    content = json.dumps({"error": f"refused tempauth={VALUE}", "page": 3})
    _result, out = _through_the_guard(content, status="error")
    data = json.loads(out.content)
    assert data == {"error": tool_guards.ERROR_NOT_CHECKED, "page": 3}
    assert out.status == "error"
    # ... and a tool that raises still hands on an error, its type alone
    assert error_line(ValueError(f"tempauth={VALUE}")) == "ValueError"
