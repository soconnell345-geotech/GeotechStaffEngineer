"""Live smoke wave 2c (module_work/live_smoke/runs/w2c-review-sonnet/
REVIEW.md), the agent-side fixes.

* D2  a spent AI budget is told apart from a busy model: never retried,
      said in plain words (Anthropic's 400 "usage limits", OpenAI's 429
      insufficient_quota, the Funhouse SDK's BudgetExceededError); a plain
      429 rate limit is still retried.
* D4  the "already looked at" record survives a restart and a restore.
* D5  the atomic markup save uses a short temporary name, and a save that
      fails says so plainly (retrying will not help).
* Pages: a page the document does not have is a JSON error, never a raise,
      with "pages are 0-based here; PDF page 5 is page 4" when it is one
      past the end; pdf_page / pdf_pages (1-based) are taken everywhere.

Fakes and synthetic PDFs only: no model, no network.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio, document_tools, vision_tools  # noqa: E402
from funhouse_agent import page_numbers  # noqa: E402
from funhouse_agent.deep import vision_engine  # noqa: E402
from funhouse_agent.deep.vision_engine import (  # noqa: E402
    LangChainVisionEngine, busy_kind, describe_error)
from funhouse_agent.error_text import (  # noqa: E402
    BUDGET_HINT, budget_exhausted, budget_message, tool_error)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class APIStatusError(Exception):
    """Shaped like the OpenAI / Anthropic SDK errors."""

    def __init__(self, message, status_code, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(status_code=status_code,
                                        headers=headers or {})


#: Anthropic's own words in F43/F44/F46 t3 (the request id is invented).
ANTHROPIC_BODY = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": ("You have reached your specified API usage limits. You will "
                "regain access on 2026-11-01 at 00:00 UTC.")},
    "request_id": "req_011CTest"}


def anthropic_usage_limit():
    return APIStatusError(f"Error code: 400 - {ANTHROPIC_BODY}", 400,
                          body=ANTHROPIC_BODY)


def openai_insufficient_quota():
    inner = {"message": "You exceeded your current quota, please check your "
                        "plan and billing details.",
             "type": "insufficient_quota", "param": None,
             "code": "insufficient_quota"}
    return APIStatusError(f"Error code: 429 - {{'error': {inner}}}", 429,
                          body=inner)


class BudgetExceededError(RuntimeError):
    """The Funhouse SDK's metered client raises this."""


def plain_rate_limit():
    return APIStatusError(
        "Error code: 429 - Rate limit reached for requests", 429,
        body={"message": "Rate limit reached for gpt on tokens per min",
              "type": "tokens", "code": "rate_limit_exceeded"},
        headers={"retry-after": "1"})


# ---------------------------------------------------------------------------
# D2: a spent budget is not a busy model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make, until", [
    (anthropic_usage_limit, "2026-11-01 at 00:00 UTC"),
    (openai_insufficient_quota, None),
    (lambda: BudgetExceededError("Monthly AI budget exceeded: $50.12 of "
                                 "$50.00"), None),
])
def test_a_spent_budget_is_recognised_by_status_body_and_type(make, until):
    exc = make()
    stop = budget_exhausted(exc)
    assert stop is not None and stop.until == until
    assert busy_kind(exc) is None                  # never "wait a minute"
    msg = budget_message(stop)
    assert msg.startswith("Your AI budget is used up")
    assert "Tell the app owner" in msg and "ask again" not in msg
    assert "{" not in msg and "req_" not in msg


def test_a_plain_rate_limit_stays_a_retry():
    exc = plain_rate_limit()
    assert budget_exhausted(exc) is None
    assert busy_kind(exc) == "rate limit"
    assert budget_exhausted(APIStatusError("Too Many Requests", 429)) is None
    # Azure's token-rate 429 talks of a pricing tier, not a budget.
    assert budget_exhausted(APIStatusError(
        "Requests have exceeded token rate limit of your current OpenAI S0 "
        "pricing tier. Please retry after 6 seconds.", 429)) is None
    assert budget_exhausted(ValueError("boom")) is None


def test_other_spent_budget_shapes():
    apim = APIStatusError("x", 403, body={
        "statusCode": 403,
        "message": "Out of call volume quota. Quota will be replenished in "
                   "2.04:12:23."})
    stop = budget_exhausted(apim)
    assert stop is not None and re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC",
                                         stop.until)
    assert budget_exhausted(APIStatusError("Payment Required", 402))
    wrapped = RuntimeError("model call failed")
    wrapped.__cause__ = openai_insufficient_quota()
    assert budget_exhausted(wrapped) is not None
    # the SDK's text alone (no body kept) still says so
    assert budget_exhausted(Exception(
        f"Error code: 400 - {ANTHROPIC_BODY}")).until == \
        "2026-11-01 at 00:00 UTC"


class _Scripted:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return AIMessage(content=step)


@pytest.fixture
def waits(monkeypatch):
    monkeypatch.delenv(vision_engine.BUSY_TRIES_ENV, raising=False)
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    seen = []
    monkeypatch.setattr(vision_engine, "_sleep", seen.append)
    return seen


@pytest.mark.parametrize("make", [anthropic_usage_limit,
                                  openai_insufficient_quota])
def test_a_vision_side_call_on_a_spent_budget_is_asked_once(waits, make):
    pytest.importorskip("langchain_core")
    model = _Scripted([make(), "never reached"])
    with pytest.raises(APIStatusError) as caught:
        LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert model.calls == 1 and waits == []
    text = describe_error(caught.value)
    assert "AI budget is used up" in text
    assert "NOT read" in text and "do NOT retry" in text
    assert "429" not in text and "{" not in text


def test_a_plain_429_on_a_side_call_is_still_asked_again(waits):
    pytest.importorskip("langchain_core")
    model = _Scripted([plain_rate_limit(), "ok"])
    assert LangChainVisionEngine(model, detail="").analyze_image(PNG, "x") \
        == "ok"
    assert model.calls == 2 and waits == [1.0]


def test_a_tool_error_on_a_spent_budget_says_do_not_retry():
    out = tool_error("annotate_document", anthropic_usage_limit())
    assert "AI budget is used up" in out["error"]
    assert out["hint"] == BUDGET_HINT
    assert "request_id" not in json.dumps(out)


class _Spent:
    def analyze_image(self, image, prompt=""):
        raise openai_insufficient_quota()


def _pdf(n=5) -> bytes:
    d = fitz.open()
    for i in range(n):
        page = d.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Sheet {i + 1}", fontsize=14)
    data = d.tobytes()
    d.close()
    return data


@pytest.fixture(autouse=True)
def _fresh():
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _look(engine, pdf, tool="analyze_pdf_page", **args):
    base = {"attachment_key": "set.pdf", "prompt": "What sheet is this?"}
    if tool == "analyze_pdf_page":
        base["tiles"] = "off"
    else:
        base["bbox"] = [50, 60, 300, 130]
    return json.loads(vision_tools.dispatch_extended_tool(
        tool, {**base, **args}, engine=engine, attachments={"set.pdf": pdf}))


def test_a_page_read_on_a_spent_budget_says_so_plainly(monkeypatch):
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    out = _look(_Spent(), _pdf(), page=0)
    assert "AI budget is used up" in out["error"]
    assert "insufficient_quota" not in out["error"]


# ---------------------------------------------------------------------------
# Pages: out of range is an error, not a raise; pdf_page is taken
# ---------------------------------------------------------------------------

class Reader:
    def __init__(self):
        self.calls = 0

    def analyze_image(self, image, prompt=""):
        self.calls += 1
        return "I see the sheet."


@pytest.mark.parametrize("tool", ["analyze_pdf_page", "render_region"])
def test_one_past_the_end_is_a_clear_error_with_the_hint(tool):
    """F47 re-run: analyze_pdf_page RAISED IndexError on PDF page 5 of 5."""
    engine = Reader()
    out = _look(engine, _pdf(5), tool=tool, page=5)
    assert "out of range" in out["error"] and "5 pages" in out["error"]
    assert out["hint"] == "pages are 0-based here; PDF page 5 is page 4"
    assert engine.calls == 0
    far = _look(engine, _pdf(5), tool=tool, page=9)
    assert "pass page 0-4, or pdf_page 1-5" in far["hint"]


@pytest.mark.parametrize("tool", ["analyze_pdf_page", "render_region"])
def test_pdf_page_is_one_based_and_never_guessed(tool):
    engine = Reader()
    out = _look(engine, _pdf(5), tool=tool, pdf_page=5)
    assert "error" not in out, out
    assert out["page"] == 4 and out["pdf_page"] == 5
    # both given and different: refused, naming both
    both = _look(engine, _pdf(5), tool=tool, page=1, pdf_page=5)
    assert "different pages" in both["error"]
    # both given and the same page: fine
    same = _look(engine, _pdf(5), tool=tool, page=4, pdf_page=5)
    assert same["page"] == 4
    # out of range in the caller's own numbers
    past = _look(engine, _pdf(5), tool=tool, pdf_page=6)
    assert "pdf_page 6 is out of range" in past["error"]
    assert "pdf_page 1-5" in past["error"]


def test_a_raising_render_is_still_a_json_error(monkeypatch):
    """Belt and braces: whatever the renderer raises for a page, it comes
    back as the tool's error."""
    from funhouse_agent import vision_view

    def boom(*_a, **_k):
        raise IndexError("page(s) [5] out of range: this document has 5 "
                         "pages, numbered 0-4")

    monkeypatch.setattr(vision_view, "render_view", boom)
    monkeypatch.setattr(vision_tools, "_page_count", lambda data: None)
    out = _look(Reader(), _pdf(5), page=5)
    assert "out of range" in out["error"]
    assert out["hint"] == "pages are 0-based here; PDF page 5 is page 4"


def test_the_agent_tools_take_pdf_page(monkeypatch):
    pytest.importorskip("langchain_core")
    from funhouse_agent.deep.tools import make_vision_tools
    tools = {t.name: t for t in make_vision_tools(
        engine=Reader(), attachments={"set.pdf": _pdf(5)})}
    out = json.loads(tools["analyze_pdf_page"].invoke(
        {"attachment_key": "set.pdf", "pdf_page": 2, "tiles": "off"}))
    assert out["page"] == 1 and out["pdf_page"] == 2
    out = json.loads(tools["render_region"].invoke(
        {"attachment_key": "set.pdf", "pdf_page": 3,
         "bbox": [50, 60, 300, 130]}))
    assert out["page"] == 2
    bad = json.loads(tools["analyze_pdf_page"].invoke(
        {"attachment_key": "set.pdf", "page": 5}))
    assert bad["hint"] == "pages are 0-based here; PDF page 5 is page 4"
    assert "pdf_page" in tools["analyze_pdf_page"].description


def test_page_number_helpers():
    assert page_numbers.pdf_pages_to_pages("1-3,6") == ("0-2,5", None)
    assert page_numbers.pdf_pages_to_pages(5) == (4, None)
    assert page_numbers.pdf_pages_to_pages([1, 2]) == ([0, 1], None)
    assert page_numbers.pdf_pages_to_pages("0-2")[1] is not None
    assert page_numbers.resolve_page(None, None) == (0, None)
    assert page_numbers.resolve_page("3", None) == (3, None)
    assert page_numbers.resolve_page(None, 0)[1] is not None
    assert page_numbers.range_hint(
        "page(s) [5] out of range: this document has 5 pages, numbered "
        "0-4") == "pages are 0-based here; PDF page 5 is page 4"
    assert page_numbers.range_hint(
        "page 5 is outside this document, which has pages 0-4") == \
        "pages are 0-based here; PDF page 5 is page 4"
    assert page_numbers.range_hint("some other error") is None


def test_coverage_counts_the_page_a_one_based_look_read():
    from funhouse_agent import coverage
    ref, rows = coverage.pages_from_call(
        "analyze_pdf_page", {"attachment_key": "set.pdf", "pdf_page": 5},
        json.dumps({"page": 4, "pdf_page": 5, "analysis": "x"}))
    assert rows == [(4, "look")]
    ref, rows = coverage.pages_from_call(
        "render_region", {"attachment_key": "set.pdf", "pdf_page": 3}, "x")
    assert rows == [(2, "look")]


# -- the document tools -------------------------------------------------------

@pytest.fixture
def folder(tmp_path, monkeypatch):
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    monkeypatch.delenv("GEOTECH_MARKUP_AUTHOR", raising=False)
    return work


def _doc_tools(pdf):
    pytest.importorskip("planlens.tools")
    pytest.importorskip("langchain_core")
    from funhouse_agent.deep.tools import make_vision_tools
    return {t.name: t for t in make_vision_tools(
        engine=None, attachments={"set.pdf": pdf})}


def _call(tool, **kw):
    return json.loads(tool.invoke(kw))


def test_document_tools_say_pages_are_0_based_and_take_pdf_pages(folder):
    tools = _doc_tools(_pdf(5))
    handle = _call(tools["open_document"], source="set.pdf")["handle"]
    bad = _call(tools["read_document"], handle=handle, pages="5")
    assert "out of range" in bad["error"]
    assert bad["hint"].startswith("pages are 0-based here; PDF page 5 is "
                                  "page 4")
    good = _call(tools["read_document"], handle=handle, pdf_pages="5")
    assert "error" not in good, good
    assert good["pages_returned"] == "4" and "Sheet 5" in json.dumps(good)
    past = _call(tools["read_document"], handle=handle, pdf_pages="6")
    assert "pdf_page 1-5" in past["hint"]
    both = _call(tools["search_document"], handle=handle, pattern="Sheet",
                 pages="0", pdf_pages="1")
    assert "not both" in both["error"]
    hits = _call(tools["search_document"], handle=handle, pattern="Sheet",
                 pdf_pages="2-3")
    assert hits["n_hits"] == 2


def test_a_markup_may_give_pdf_page(folder):
    if not document_tools.has_tool("annotate_document"):
        pytest.skip("installed planlens predates annotate_document")
    tools = _doc_tools(_pdf(5))
    handle = _call(tools["open_document"], source="set.pdf")["handle"]
    out = _call(tools["annotate_document"], handle=handle,
                output_path="set_marked.pdf", check=False,
                markups=[{"kind": "note", "pdf_page": 5,
                          "comment": "last sheet", "point": [80, 80]}])
    assert "error" not in out, out
    assert out["written"][0]["page"] == 4
    clash = _call(tools["annotate_document"], handle=handle,
                  output_path="set_marked.pdf", check=False,
                  markups=[{"kind": "note", "page": 0, "pdf_page": 5,
                            "comment": "x", "point": [80, 80]}])
    assert "different pages" in clash["error"]
    past = _call(tools["annotate_document"], handle=handle,
                 output_path="set_marked2.pdf", check=False,
                 markups=[{"kind": "note", "page": 5, "comment": "x",
                           "point": [80, 80]}])
    skipped = past.get("skipped") or []
    assert skipped and skipped[0]["hint"] == \
        "pages are 0-based here; PDF page 5 is page 4"


# ---------------------------------------------------------------------------
# D4: the read record outlives the process
# ---------------------------------------------------------------------------

def _conversation(tmp_path, name):
    """A conversation laid out as the web app lays it out: meta.json
    beside the working folder files/."""
    conv = tmp_path / name
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text(json.dumps({"thread_id": name}),
                                    encoding="utf-8")
    return conv


def test_the_read_record_survives_a_restart_and_a_restore(tmp_path):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf(3)
    with _fileio.working_dir_bound(str(conv / "files")):
        _look(Reader(), pdf, page=0)
        _look(Reader(), pdf, tool="render_region", pdf_page=2)
        before = vision_tools.reads_for_conversation()
    assert [(r["page"], r["tool"]) for r in before] == [
        (0, "analyze_pdf_page"), (1, "render_region")]
    # Kept in the CONVERSATION folder (mirrored, restored, never a card).
    record = conv / vision_tools.READS_FILE
    assert record.is_file()
    assert not (conv / "files" / vision_tools.READS_FILE).exists()
    # A restart forgets memory; the record is read back.
    vision_tools.clear_read_log()
    assert vision_tools.reads_for_conversation(conv / "files") == before
    # ...and the next read carries on from it rather than starting over.
    with _fileio.working_dir_bound(str(conv / "files")):
        _look(Reader(), pdf, page=2)
    assert len(vision_tools.reads_for_conversation(conv / "files")) == 3
    # A conversation restored elsewhere (the mirror brings the record back).
    restored = _conversation(tmp_path / "restored", "alice")
    shutil.copy(record, restored / vision_tools.READS_FILE)
    vision_tools.clear_read_log()
    assert len(vision_tools.reads_for_conversation(restored / "files")) == 3
    # Another conversation sees none of it.
    bob = _conversation(tmp_path, "bob")
    assert vision_tools.reads_for_conversation(bob / "files") == []


def test_a_bare_working_folder_keeps_the_record_in_its_scratch(tmp_path):
    work = tmp_path / "run" / "work"
    work.mkdir(parents=True)
    with _fileio.working_dir_bound(str(work)):
        _look(Reader(), _pdf(2), page=1)
    assert (work / vision_tools.SCRATCH_DIR / vision_tools.READS_FILE).is_file()
    vision_tools.clear_read_log()
    assert [r["page"] for r in
            vision_tools.reads_for_conversation(work)] == [1]


# ---------------------------------------------------------------------------
# D5: a short temporary name, and a failed save said plainly
# ---------------------------------------------------------------------------

def test_the_temporary_name_is_short(folder):
    with _fileio.working_dir_bound(str(folder)):
        tmp = document_tools._temp_beside(str(
            folder / ("Riverside_Civil_Set_IFC_with_a_long_descriptive_name"
                      "_marked.pdf")))
    assert os.path.dirname(tmp) == str(folder / vision_tools.SCRATCH_DIR)
    assert re.fullmatch(r"[0-9a-f]{8}\.part\.pdf", os.path.basename(tmp))


def test_an_unwritable_temporary_copy_is_said_plainly(folder, monkeypatch):
    blocker = folder.parent / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setattr(document_tools, "_temp_beside",
                        lambda final: str(blocker / "abcd1234.part.pdf"))
    calls = []
    out = document_tools.write_marked_copy(
        "set.pdf", [{"kind": "note", "page": 0, "comment": "x",
                     "point": [1, 1]}],
        "set_marked.pdf", True, "me", lambda args: calls.append(args))
    assert out["error"].startswith(
        "could not write the marked copy on the server")
    assert "will fail the same way" in out["hint"]
    assert calls == []                         # planlens never asked
    assert str(folder.parent) not in json.dumps(out)


def test_mupdfs_save_failure_is_said_plainly(folder):
    def planlens(args):
        return json.dumps({
            "error": "annotate_document failed: FzErrorSystem: code=2: "
                     f"cannot open file '.scratch/"
                     f"{os.path.basename(args['output_path'])}': No such "
                     "file or directory",
            "hint": "The tool raised an error instead of answering; ... Try "
                    "it again once if the cause looks passing"})
    out = document_tools.write_marked_copy(
        "set.pdf", [{"kind": "note", "page": 0, "comment": "x",
                     "point": [1, 1]}], "set_marked.pdf", True, "me", planlens)
    assert out["error"].startswith(
        "could not write the marked copy on the server")
    assert "FzErrorSystem" in out["error"]
    assert "will fail the same way" in out["hint"]
    assert "Try it again once" not in out["hint"]
    leftovers = [n for _r, _d, names in os.walk(folder) for n in names
                 if ".part" in n]
    assert leftovers == []


@pytest.mark.skipif(os.name != "nt", reason="the 260-character limit is "
                                            "Windows'")
def test_a_path_too_long_for_windows_is_named_as_the_cause(tmp_path):
    long_path = str(tmp_path / ("x" * (300 - len(str(tmp_path)))))
    exc = document_tools._write_failure(long_path, OSError(2, "No such file"))
    assert "characters long, over Windows' 260-character limit" in str(exc)
