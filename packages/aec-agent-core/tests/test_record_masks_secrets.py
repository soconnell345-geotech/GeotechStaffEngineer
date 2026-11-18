"""The run record keeps the text of an error with its secrets masked (a
repair, no switch).

An exception from a model client or an HTTP library can carry a key, a
bearer token or a signed URL, and the run record (``activity.jsonl``, the
review suite's ``run.json`` / ``results.json``, the run report) is what a
tester downloads and sends on. The failed-tool and failed-model handlers
wrote the exception's text as it came. They now mask it with
``error_text.mask_error_secrets``, which masks a secret where its name or
its form says it is one and keeps a request id, a run id or a hash however
long; an error with no secret in it is written as before, character for
character, and a tool's ordinary output is never touched.
``error_text.mask_secrets`` (the page's shown errors, the sign-in refusal,
the side prompt) keeps its own wider rule, byte for byte.

Every value here is a made-up shape (``fake``, ``example.test``), never a
live key of any vendor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid

import pytest

from aec_agent_core import activity_log as al
from aec_agent_core.error_text import (mask_error_result,
                                       mask_error_secrets, mask_secrets)

#: A random-looking token: upper and lower case and digits, mixed.
FAKE_TOKEN = "Fake0aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0u"

SHA256 = hashlib.sha256(b"made-up drawing set").hexdigest()
SHA1 = hashlib.sha1(b"made-up drawing set").hexdigest()
RUN_ID = str(uuid.UUID(int=0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0))
#: A made-up request id of 36 characters, as mixed as a random token.
REQUEST_ID = "req_Fk0aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT"

#: ``mask_secrets``' rule before this change, copied here to prove it is
#: kept byte for byte for the places that used it.
OLD_RULE = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_*\-]{6,}|\bBearer\s+\S+|\b[A-Za-z0-9_\-]{32,}\b")

#: An ordinary error: hashes, a UUID, a request id, page numbers, file
#: names, a long word, and words that only sound like secrets.
ORDINARY = (
    "page 12 of 40 could not be read from "
    "'Site_Investigation_Report_Rev2_Final_20261118.pdf' "
    "(ReviewSetMarkedPage12FinalVersion.pdf, "
    "geotechnical_investigation_report_final_rev2.pdf); "
    f"sha256 {SHA256}, sha1 {SHA1}, run {RUN_ID}, "
    "request req_0123456789abcdef0123456789abcdef; "
    "pneumonoultramicroscopicsilicovolcanoconiosis; pages 3-7; "
    "max_tokens=4096; the token limit was reached; api_key=None; "
    "password: ***; Authorization failed; key 'A1' on sheet S-101; "
    "see https://docs.example.test/errors?page=2#limits")

#: (text with a secret, what must not survive, what the masked text is).
SECRETS = [
    ("Error code: 401 - Incorrect API key provided: "
     "sk-fake-00000000-not-real.",
     "sk-fake", "Error code: 401 - Incorrect API key provided: ….",),
    ("403 Client Error: Forbidden for url: https://files.example.test/a/"
     "doc.pdf?download=1&tempauth=v1.FakeTempauthValue000",
     "FakeTempauthValue000",
     "403 Client Error: Forbidden for url: https://files.example.test/a/"
     "doc.pdf?download=1&tempauth=…"),
    ("blob refused: https://store.example.test/c/d.pdf?sv=2024-01-01&sp=r"
     "&sig=FAKEsig%2Bvalue%3D",
     "FAKEsig",
     "blob refused: https://store.example.test/c/d.pdf?sv=2024-01-01&sp=r"
     "&sig=…"),
    ("GET https://bucket.example.test/k?X-Amz-Credential=FAKECRED%2F2026"
     "&X-Amz-Signature=0000fake1111beef&X-Amz-Expires=60",
     "fake1111",
     "GET https://bucket.example.test/k?X-Amz-Credential=…"
     "&X-Amz-Signature=…&X-Amz-Expires=60"),
    ("ProxyError: cannot connect to http://someone:fake-pass-000@"
     "proxy.example.test:8080",
     "fake-pass",
     "ProxyError: cannot connect to http://someone:…@proxy.example.test:8080"),
    ("sent headers {'Authorization': 'Bearer FakeBearer.Token-000'}",
     "FakeBearer", "sent headers {'Authorization': '…'}"),
    ("Authorization: Basic ZmFrZTpmYWtlMDAw",
     "ZmFrZTpm", "Authorization: …"),
    ("body: grant_type=client_credentials&client_secret=Fake~Secret.Val-0"
     "&scope=x",
     "Fake~Secret",
     "body: grant_type=client_credentials&client_secret=…&scope=x"),
    ('request {"api_key": "fake-key-value-000", "model": "m"}',
     "fake-key-value", 'request {"api_key": "…", "model": "m"}'),
    ("token expired: eyJmYWtlIjoxfQ.eyJub3QiOiJyZWFsIn0.ZmFrZS1zaWc",
     "eyJub3Qi", "token expired: …"),
    ("sent Bearer FakeBearerValue000 and was refused",
     "FakeBearerValue000", "sent … and was refused"),
    ("Invalid API Key = fake-unprefixed-key-000. Check the settings.",
     "fake-unprefixed", "Invalid API Key = …. Check the settings."),
    ("bad key\n-----BEGIN PRIVATE KEY-----\nFAKEFAKEFAKE\n"
     "-----END PRIVATE KEY-----\nend",
     "FAKEFAKEFAKE", "bad key\n…\nend"),
    ("callback https://app.example.test/cb?code=FakeCode000&state=1&"
     "key=FakeKey000",
     "FakeCode000",
     "callback https://app.example.test/cb?code=…&state=1&key=…"),
    ("x-api-key: fake-key-000 was refused", "fake-key-000",
     "x-api-key: … was refused"),
    ("ocp-apim-subscription-key=0123456789abcdef0123456789abcdef",
     "0123456789abcdef", "ocp-apim-subscription-key=…"),
]


# ---------------------------------------------------------------------------
# the masker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,secret,masked", SECRETS)
def test_each_shape_of_secret_is_masked(text, secret, masked):
    out = mask_error_secrets(text)
    assert secret not in out
    assert out == masked


def test_an_ordinary_error_comes_back_character_for_character():
    assert mask_error_secrets(ORDINARY) == ORDINARY
    for word in (SHA256, SHA1, RUN_ID, REQUEST_ID, "page 12 of 40",
                 "Site_Investigation_Report_Rev2_Final_20261118.pdf",
                 "ReviewSetMarkedPage12FinalVersion.pdf",
                 "pneumonoultramicroscopicsilicovolcanoconiosis"):
        assert word in mask_error_secrets(f"failed: {word}")


def test_a_key_is_masked_and_a_request_id_and_a_hash_beside_it_survive():
    assert len(REQUEST_ID) == 36 and len(SHA1) == 40
    raised = (f"401 Unauthorized (request id {REQUEST_ID}, document sha1 "
              f"{SHA1}): api-key: fake-key-value-000 was refused")
    assert mask_error_secrets(raised) == raised.replace(
        "fake-key-value-000", "…")
    # the wider rule, kept for the places that used it, would hide both
    assert REQUEST_ID not in mask_secrets(raised)
    assert SHA1 not in mask_secrets(raised)


def test_a_run_with_no_name_beside_it_is_kept_however_random():
    # It cannot be told from a request id; only its name or its form
    # (sk-, Bearer, a JSON web token) says a run of characters is a key.
    text = f"refused the key {FAKE_TOKEN} for this model"
    assert mask_error_secrets(text) == text


def test_mask_secrets_keeps_its_rule_byte_for_byte():
    texts = [t for t, _s, _m in SECRETS] + [
        ORDINARY, FAKE_TOKEN, REQUEST_ID, SHA1, RUN_ID, "",
        f"refused the key {FAKE_TOKEN} (request {REQUEST_ID})"]
    for text in texts:
        assert mask_secrets(text) == OLD_RULE.sub("…", text), text
    assert mask_secrets(None) == ""


def test_a_masked_text_masks_to_itself_and_none_is_empty():
    for text, _secret, masked in SECRETS:
        assert mask_error_secrets(masked) == masked
    assert mask_error_secrets(None) == ""


def test_a_tools_error_result_is_masked_and_its_ordinary_output_is_not():
    err = json.dumps({"error": f"read failed: Bearer {FAKE_TOKEN}",
                      "hint": "try again"})
    assert FAKE_TOKEN not in mask_error_result(err)
    assert json.loads(mask_error_result(err))["hint"] == "try again"
    ordinary = json.dumps({"text": f"the report quotes {FAKE_TOKEN}"})
    assert mask_error_result(ordinary) == ordinary
    plain = f"page text: Bearer {FAKE_TOKEN}"
    assert mask_error_result(plain) == plain
    clean = json.dumps({"error": ORDINARY})
    assert mask_error_result(clean) == clean


# ---------------------------------------------------------------------------
# the activity log
# ---------------------------------------------------------------------------

SECRET_ERROR = " | ".join(t for t, _s, _m in SECRETS)
GONE = [s for _t, s, _m in SECRETS]


def _clean(text: str) -> None:
    for secret in GONE:
        assert secret not in text, secret


def _tool(log, name, *, error=None, result=None, call_id=None):
    rid = uuid.uuid4()
    log.on_tool_start({"name": name}, "{}", run_id=rid, inputs={},
                      tool_call_id=call_id)
    if error is not None:
        log.on_tool_error(error, run_id=rid)
    else:
        log.on_tool_end(result, run_id=rid)
    return rid


def _events(conv, event):
    return [r for r in al.load(conv) if r.get("event") == event]


def test_a_failed_tool_and_a_failed_model_call_are_kept_masked(tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1)
    _tool(log, "open_document", error=RuntimeError(SECRET_ERROR))
    log.on_chat_model_start({}, [[]], run_id="m1")
    log.on_llm_error(PermissionError(SECRET_ERROR), run_id="m1")
    log.turn_end(error=f"The turn stopped: {SECRET_ERROR}",
                 error_detail=f"provider said: {SECRET_ERROR}")
    (tool,) = _events(conv, "tool_error")
    (model,) = _events(conv, "model_error")
    (end,) = _events(conv, "turn_end")
    assert tool["error"] == mask_error_secrets(f"RuntimeError: {SECRET_ERROR}")
    assert model["error"] == mask_error_secrets(
        f"PermissionError: {SECRET_ERROR}")
    assert end["error"] == mask_error_secrets(
        f"The turn stopped: {SECRET_ERROR}")
    assert end["error_detail"] == mask_error_secrets(
        f"provider said: {SECRET_ERROR}")
    with open(al.activity_path(conv), encoding="utf-8") as fh:
        _clean(fh.read())


def test_an_error_with_no_secret_is_written_as_before(tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1)
    _tool(log, "open_document", error=ValueError(ORDINARY))
    log.on_chat_model_start({}, [[]], run_id="m1")
    log.on_llm_error(TimeoutError(ORDINARY), run_id="m1")
    log.turn_end(error=ORDINARY, error_detail=ORDINARY)
    al.append_event(conv, "page_error", error=ORDINARY, where="x.py:1")
    assert _events(conv, "tool_error")[0]["error"] == f"ValueError: {ORDINARY}"
    assert _events(conv, "model_error")[0]["error"] == \
        f"TimeoutError: {ORDINARY}"
    end = _events(conv, "turn_end")[0]
    assert end["error"] == ORDINARY and end["error_detail"] == ORDINARY
    assert _events(conv, "page_error")[0]["error"] == ORDINARY


def test_a_raised_errors_key_is_masked_and_its_ids_are_kept(tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1)
    raised = (f"401 Unauthorized (request id {REQUEST_ID}, document sha1 "
              f"{SHA1}): api-key: fake-key-value-000 was refused")
    _tool(log, "open_document", error=PermissionError(raised))
    log.on_chat_model_start({}, [[]], run_id="m1")
    log.on_llm_error(PermissionError(raised), run_id="m1")
    log.turn_end(error=raised)
    for event in ("tool_error", "model_error", "turn_end"):
        (rec,) = _events(conv, event)
        assert "fake-key-value-000" not in rec["error"], event
        assert "api-key: … was refused" in rec["error"], event
        assert REQUEST_ID in rec["error"] and SHA1 in rec["error"], event


def test_the_records_keep_their_shape(tmp_path):
    def shapes(conv, message):
        log = al.ActivityLogger(conv, turn=1)
        _tool(log, "t", error=RuntimeError(message))
        log.on_chat_model_start({}, [[]], run_id="m1")
        log.on_llm_error(RuntimeError(message), run_id="m1")
        log.turn_end(error=message, error_detail=message)
        return [{k: type(v).__name__ for k, v in r.items()
                 if k not in ("ts", "t", "duration_s")}
                for r in al.load(conv)]

    assert shapes(str(tmp_path / "a"), SECRET_ERROR) == \
        shapes(str(tmp_path / "b"), ORDINARY)


def test_a_long_error_is_masked_before_its_cut_and_in_its_whole_copy(
        tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1, max_chars=200)
    long = ("x " * 300) + SECRET_ERROR
    _tool(log, "open_document", error=RuntimeError(long))
    (rec,) = _events(conv, "tool_error")
    assert rec["truncated"] is True and rec["full"]
    with open(os.path.join(conv, rec["full"]), encoding="utf-8") as fh:
        whole = fh.read()
    assert whole == mask_error_secrets(f"RuntimeError: {long}")
    _clean(whole)
    assert rec["chars"] == len(whole)


def test_a_tools_error_text_is_masked_and_its_ordinary_output_kept(
        tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1)
    err = json.dumps({"error": f"failed: {SECRET_ERROR}", "hint": "again"})
    out = json.dumps({"text": f"the sheet says api_key=quoted-in-doc "
                              f"{FAKE_TOKEN}"})
    _tool(log, "look", result=err)
    _tool(log, "read", result=out, call_id="c2")
    # the error message the model is given, and a result a guard changed
    log.on_custom_event("tool_delivered", {
        "tool_call_id": "c9", "name": "x", "refused": True,
        "content": f"Error: {SECRET_ERROR}"}, run_id=uuid.uuid4())
    log.on_custom_event("tool_delivered", {
        "tool_call_id": "c2", "name": "read",
        "content": json.dumps({"error": f"cut: {SECRET_ERROR}"})},
        run_id=uuid.uuid4())
    ends = _events(conv, "tool_end")
    assert ends[0]["result"] == mask_error_secrets(err)
    assert ends[1]["result"] == out                  # as the model got it
    (refused,) = _events(conv, "tool_refused")
    (delivered,) = _events(conv, "tool_delivered")
    _clean(ends[0]["result"] + refused["text"] + delivered["text"])
    assert refused["text"].startswith("Error: ")


def test_app_events_mask_the_account_of_a_failure(tmp_path):
    conv = str(tmp_path)
    log = al.ActivityLogger(conv, turn=1)
    log.on_custom_event("helper_kept", {
        "path": "kept/helper_1.md",
        "why": f"it stopped on an error ({SECRET_ERROR})"},
        run_id=uuid.uuid4())
    al.append_event(conv, "page_error", error=SECRET_ERROR,
                    traceback=f"Traceback ...\n{SECRET_ERROR}", where="a:1")
    (kept,) = _events(conv, "helper_kept")
    (page,) = _events(conv, "page_error")
    assert kept["path"] == "kept/helper_1.md"
    _clean(kept["why"] + page["error"] + page["traceback"])


# ---------------------------------------------------------------------------
# the run report and the review suite's results
# ---------------------------------------------------------------------------

def test_the_run_report_masks_a_record_written_before_the_masking(
        tmp_path):
    from aec_agent_core import run_report
    conv = tmp_path / "run"
    (conv / "activity_full").mkdir(parents=True)
    (conv / "activity_full" / "whole.txt").write_text(
        f"RuntimeError: {SECRET_ERROR}", encoding="utf-8")
    recs = [
        {"turn": 1, "ts": 1.0, "t": 0.0, "agent": "primary",
         "event": "turn_start", "prompt": "q", "model": "m"},
        {"turn": 1, "ts": 1.1, "t": 0.1, "agent": "primary",
         "event": "tool_start", "name": "look", "run_id": "r1",
         "args": {}},
        {"turn": 1, "ts": 1.2, "t": 0.2, "agent": "primary",
         "event": "tool_error", "name": "look", "run_id": "r1",
         "error": f"RuntimeError: {SECRET_ERROR}"[:100], "truncated": True,
         "chars": 9999, "full": "activity_full/whole.txt"},
        {"turn": 1, "ts": 1.3, "t": 0.3, "agent": "primary",
         "event": "model_start", "run_id": "m1"},
        {"turn": 1, "ts": 1.4, "t": 0.4, "agent": "primary",
         "event": "model_error", "run_id": "m1",
         "error": f"PermissionError: {SECRET_ERROR}"},
        {"turn": 1, "ts": 1.5, "t": 0.5, "agent": "primary",
         "event": "turn_end", "error": f"stopped: {SECRET_ERROR}",
         "error_detail": SECRET_ERROR},
    ]
    (conv / "activity.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    page = run_report.build_report(str(conv))
    assert "Incorrect API key provided: …" in page   # shown, masked
    for secret in GONE:
        assert secret not in page, secret


def test_the_suites_result_keeps_its_errors_masked():
    from aec_agent_core.review_eval import runner
    res = {"error": f"RuntimeError: {SECRET_ERROR}",
           "outcome_error": SECRET_ERROR, "traceback": SECRET_ERROR,
           "turns": [{"error": SECRET_ERROR, "answer": "a"},
                     {"error": None, "answer": "b"}],
           "vision_profile": {"error": SECRET_ERROR, "detail": "high"},
           "answer": "b", "seconds": 1.0}
    out = runner._masked_errors(json.loads(json.dumps(res)))
    assert set(out) == set(res) and out["turns"][1]["error"] is None
    assert out["vision_profile"]["detail"] == "high"
    _clean(json.dumps(out))
    clean = {"error": ORDINARY, "turns": [{"error": ORDINARY}],
             "vision_profile": {"error": ORDINARY}}
    assert runner._masked_errors(json.loads(json.dumps(clean))) == clean
