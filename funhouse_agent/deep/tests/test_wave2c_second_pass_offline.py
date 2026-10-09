"""Live smoke wave 2c, second pass (module_work/live_smoke/runs/
w2c-review-sonnet/REVIEW.md, "Second pass"): the agent-side fixes.

* E1  each kind of vision call has its own output cap, and with tiles the
      whole-page call is asked for the layout only (F44: a whole-page answer
      ran 72 s and 10,778 tokens while its tiles finished in 15 s). The
      prompt and the cap actually SENT are checked here, with fake engines
      and a fake chat model; the tile prompt is unchanged.
* E2  each page's full reading is kept for the conversation, and
      ``analyze_pdf_page(reuse=true)`` hands it back without a new look
      (F44: one sheet transcribed whole in all four turns, 206 s).
* E4  a DXF read lists its geometry with the text it sits by; annotate_dxf
      writes notes on a copy with a CAD library; a retyped DXF is refused
      (F40: the model retyped a 19 KB DXF twice and changed a group code).
* E6  read_text_file says what it returned of how much, and takes
      ``limit`` / ``offset`` (it silently capped 20,000 at 6,000).

Fakes, synthetic PDFs and drawings made here: no model, no network.
"""

from __future__ import annotations

import json
import os
import re
import threading

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio, vision_tools  # noqa: E402
from funhouse_agent.deep.vision_engine import (  # noqa: E402
    LangChainVisionEngine, VisionAnswer, capped_model)
from funhouse_agent.vision_tools import dispatch_extended_tool  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
TILE = re.compile(r"tile row (\d) of \d, column (\d)")
QUESTION = "Transcribe ALL text on this sheet exactly."


def _pdf(n=1) -> bytes:
    d = fitz.open()
    for i in range(n):
        page = d.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Sheet {i + 1}  B-1  B-2", fontsize=12)
        page.draw_rect(fitz.Rect(300, 300, 500, 500), color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


class CapReader:
    """Takes an output cap, as the app's engine does, and records the prompt
    and the cap of every call."""

    accepts_output_cap = True

    def __init__(self, cut=()):
        self.lock = threading.Lock()
        self.calls = []                    # (who, prompt, cap)
        self.cut = set(cut)

    def analyze_image(self, image, prompt="", max_output_tokens=None):
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "page"
        with self.lock:
            self.calls.append((who, prompt, max_output_tokens))
        text = f"I see {who}."
        if who in self.cut:
            text = VisionAnswer(text)
            text.cut_off = True
        return text

    def of(self, who):
        return [c for c in self.calls if c[0] == who]


class OldReader:
    """An engine with the old two-argument surface (no cap)."""

    def __init__(self):
        self.calls = 0

    def analyze_image(self, image, prompt=""):
        self.calls += 1
        return "I see the sheet."


def _read(engine, pdf, attachments=None, **args):
    base = {"attachment_key": "sheet.pdf", "page": 0, "prompt": QUESTION}
    return json.loads(dispatch_extended_tool(
        "analyze_pdf_page", {**base, **args}, engine=engine,
        attachments=attachments or {"sheet.pdf": pdf}))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(vision_tools.VISION_CAP_ENV, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _conversation(tmp_path, name):
    """A conversation laid out as the web app lays it out."""
    conv = tmp_path / name
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text(json.dumps({"thread_id": name}),
                                    encoding="utf-8")
    return conv


# ---------------------------------------------------------------------------
# E1: the prompt and the cap each vision call is sent
# ---------------------------------------------------------------------------

def test_with_tiles_the_whole_page_is_asked_for_the_layout_with_a_small_cap():
    engine = CapReader()
    out = _read(engine, _pdf(), tiles="3")
    (page,) = engine.of("page")
    _who, prompt, cap = page
    assert cap == vision_tools.VISION_OUTPUT_CAPS["overview"] == 4000
    assert prompt.startswith("This image is the WHOLE sheet")
    assert "give only the LAYOUT" in prompt and "3x3 close-up tiles" in prompt
    assert "do not transcribe" in prompt
    # The agent's own question rides along, framed, not as the instruction.
    assert f"What is being asked of this sheet: {QUESTION}" in prompt
    assert not prompt.startswith(QUESTION)
    # The result says what the overview is.
    assert "LAYOUT only" in out["tiling"]


def test_the_tile_prompt_is_unchanged_and_tiles_get_their_own_cap():
    engine = CapReader()
    _read(engine, _pdf(), tiles="2")
    tiles = [c for c in engine.calls if c[0] != "page"]
    assert len(tiles) == 4
    for who, prompt, cap in tiles:
        assert cap == vision_tools.VISION_OUTPUT_CAPS["tile"] == 6000
        r, c = int(who[1]), int(who[3])
        # Exactly the prompt tiles were sent before this change.
        assert prompt.startswith(
            f"{QUESTION}\n\nThis image is tile row {r} of 2, column {c} of "
            f"2 of the sheet (tiles overlap slightly). Report only what is "
            f"IN this tile, briefly.")
        assert "LAYOUT" not in prompt


def test_a_page_read_whole_is_the_reading_and_keeps_the_question():
    engine = CapReader()
    _read(engine, _pdf(), tiles="off")
    ((who, prompt, cap),) = engine.calls
    assert prompt.startswith(QUESTION) and "LAYOUT" not in prompt
    # 12,000 until live smoke wave 3 (F1): see test_wave3_latency_offline
    assert cap == vision_tools.VISION_OUTPUT_CAPS["page"] == 8000


def test_zoom_and_image_calls_have_their_caps():
    engine = CapReader()
    out = json.loads(dispatch_extended_tool(
        "render_region", {"attachment_key": "sheet.pdf", "page": 0,
                          "bbox": [60, 80, 300, 120], "prompt": "Read it."},
        engine=engine, attachments={"sheet.pdf": _pdf()}))
    assert "error" not in out, out
    assert engine.calls[-1][2] == vision_tools.VISION_OUTPUT_CAPS["region"]
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="PNG")
    out = json.loads(dispatch_extended_tool(
        "analyze_image", {"attachment_key": "pic.png", "prompt": "What?"},
        engine=engine, attachments={"pic.png": buf.getvalue()}))
    assert "error" not in out, out
    assert engine.calls[-1][2] == vision_tools.VISION_OUTPUT_CAPS["image"]


def test_the_markup_check_look_has_its_cap():
    from funhouse_agent import markup_check
    engine = CapReader()
    out = markup_check._check_one(_pdf(), {
        "index": 0, "row": {"kind": "box", "page": 0,
                            "bbox": [60, 80, 200, 120]},
        "spec": {"comment": "check", "target": "Sheet 1"}}, engine)
    assert out["verdict"] in ("unsure", "not_checked", "confirmed",
                              "misplaced")
    assert engine.calls[-1][2] == vision_tools.VISION_OUTPUT_CAPS["check"]


def test_an_engine_without_a_cap_is_called_as_before():
    engine = OldReader()
    out = _read(engine, _pdf(), tiles="2")
    assert engine.calls == 5 and "error" not in out


def test_the_caps_can_be_switched_off_or_set_without_a_release(monkeypatch):
    monkeypatch.setenv(vision_tools.VISION_CAP_ENV, "off")
    engine = CapReader()
    _read(engine, _pdf(), tiles="2")
    assert {c[2] for c in engine.calls} == {None}
    monkeypatch.setenv(vision_tools.VISION_CAP_ENV, "5000")
    engine = CapReader()
    _read(engine, _pdf(), tiles="2", prompt="again")
    assert {c[2] for c in engine.calls} == {5000}


def test_a_cut_layout_answer_leaves_the_tiles_as_the_reading():
    out = _read(CapReader(cut={"page"}), _pdf(), tiles="2")
    assert "cut_off" not in out
    assert "the tiles below read the lettering" in out["overview_cut"]


class _Response:
    def __init__(self, text, meta=None):
        self.content = text
        self.response_metadata = meta or {}


class FakeChatModel:
    """A LangChain-shaped chat model: an output cap in ``max_tokens`` and a
    ``model_copy`` that changes it, as ChatOpenAI / ChatAnthropic have."""

    model_fields = {"max_tokens": None, "model_name": None}

    def __init__(self, max_tokens=32000, log=None):
        self.max_tokens = max_tokens
        self.model_name = "fake"
        self.log = log if log is not None else []

    def model_copy(self, update=None):
        twin = FakeChatModel(self.max_tokens, self.log)
        for k, v in (update or {}).items():
            setattr(twin, k, v)
        return twin

    def invoke(self, messages, **_kw):
        self.log.append(self.max_tokens)
        stop = "length" if self.max_tokens <= 100 else "stop"
        return _Response("seen", {"finish_reason": stop})


def test_the_cap_reaches_the_model_and_never_raises_its_own(monkeypatch):
    from funhouse_agent.deep import vision_engine
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    log = []
    model = FakeChatModel(32000, log)
    engine = LangChainVisionEngine(model, detail="")
    engine.analyze_image(PNG, "x", max_output_tokens=4000)
    engine.analyze_image(PNG, "x")
    assert log == [4000, 32000]
    assert model.max_tokens == 32000              # the original is untouched
    low = LangChainVisionEngine(FakeChatModel(2000, log), detail="")
    low.analyze_image(PNG, "x", max_output_tokens=4000)
    assert log[-1] == 2000                        # never raised
    cut = engine.analyze_image(PNG, "x", max_output_tokens=50)
    assert getattr(cut, "cut_off", False) is True   # still flagged


def test_a_model_with_no_cap_field_is_asked_as_before():
    class Plain:
        def invoke(self, messages):
            return _Response("ok")
    m = Plain()
    assert capped_model(m, 4000) is m


def test_a_real_openai_and_anthropic_request_carries_the_cap():
    """No network: the request payload the client WOULD send."""
    pytest.importorskip("langchain_openai")
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    gpt = ChatOpenAI(model="gpt-5.1", api_key="x",
                     max_completion_tokens=32000)
    capped = capped_model(gpt, 4000)
    payload = capped._get_request_payload([HumanMessage("hi")])
    assert payload["max_completion_tokens"] == 4000
    assert "max_tokens" not in payload
    assert gpt._get_request_payload(
        [HumanMessage("hi")])["max_completion_tokens"] == 32000
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic
    claude = ChatAnthropic(model="claude-sonnet-4-6", api_key="x",
                           max_tokens=32000)
    assert capped_model(claude, 6000)._get_request_payload(
        [HumanMessage("hi")])["max_tokens"] == 6000


# ---------------------------------------------------------------------------
# E2: a page's full reading is kept and can be had back without a new look
# ---------------------------------------------------------------------------

def test_a_reading_is_kept_beside_the_read_record_never_as_a_card(tmp_path):
    conv = _conversation(tmp_path, "alice")
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _read(CapReader(), _pdf(), tiles="2")
        kept = vision_tools.readings_for_conversation()
    record = conv / vision_tools.READINGS_FILE
    assert record.is_file()
    assert not (conv / "files" / vision_tools.READINGS_FILE).exists()
    (entry,) = kept
    assert entry["document"] == "sheet.pdf" and entry["page"] == 0
    assert entry["pdf_page"] == 1 and entry["view"] == "page+tiles 2x2"
    assert entry["prompt"] == QUESTION and "result" not in entry
    full = json.loads(record.read_text(encoding="utf-8"))["readings"][0]
    assert json.loads(full["result"]) == first


def test_reuse_hands_the_reading_back_without_a_new_look(tmp_path):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _read(CapReader(), pdf, tiles="2")
        engine = CapReader()
        again = _read(engine, pdf, prompt="What size is the pipe?",
                      reuse=True)
    assert engine.calls == []                      # no new look
    assert "page+tiles 2x2" in again["reused"]
    assert QUESTION in again["reused"]
    assert "without reuse" in again["reused"]
    assert again["tiles"] == first["tiles"]
    assert again["analysis"] == first["analysis"]


def test_reuse_is_never_forced(tmp_path):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        _read(CapReader(), pdf, tiles="2")
        engine = CapReader()
        fresh = _read(engine, pdf, tiles="2", prompt="A new question")
    assert len(engine.calls) == 5 and "reused" not in fresh


def test_reuse_of_a_page_never_read_reads_it_now(tmp_path):
    conv = _conversation(tmp_path, "alice")
    with _fileio.working_dir_bound(str(conv / "files")):
        engine = CapReader()
        out = _read(engine, _pdf(), tiles="off", reuse=True)
    assert len(engine.calls) == 1
    assert out["reuse_note"] == vision_tools.NO_EARLIER_READING


def test_reuse_prefers_the_view_asked_for(tmp_path):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        _read(CapReader(), pdf, tiles="2")
        _read(CapReader(), pdf, tiles="off", prompt="title block?")
        whole = _read(CapReader(), pdf, tiles="off", reuse=True)
        tiled = _read(CapReader(), pdf, tiles="2", reuse=True)
        newest = _read(CapReader(), pdf, reuse=True)      # auto: newest
    assert whole["reused"].startswith(
        "An earlier reading of this page in this conversation (page;")
    assert "(page+tiles 2x2;" in tiled["reused"]
    assert "(page;" in newest["reused"]


def test_readings_survive_a_restart_and_stay_in_their_conversation(
        tmp_path):
    alice, bob = _conversation(tmp_path, "alice"), _conversation(tmp_path,
                                                                 "bob")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(alice / "files")):
        _read(CapReader(), pdf, tiles="2")
    vision_tools.clear_read_log()                  # a restart
    vision_tools.clear_repeat_reads()
    with _fileio.working_dir_bound(str(alice / "files")):
        engine = CapReader()
        out = _read(engine, pdf, reuse=True)
    assert engine.calls == [] and "reused" in out
    with _fileio.working_dir_bound(str(bob / "files")):
        engine = CapReader()
        out = _read(engine, pdf, reuse=True)
    assert len(engine.calls) == 5 and "reused" not in out
    assert vision_tools.readings_for_conversation(bob / "files")[0][
        "document"] == "sheet.pdf"


def test_the_same_name_with_other_pages_is_not_reused(tmp_path):
    """A reading belongs to the file's CONTENT, not its name (F48)."""
    conv = _conversation(tmp_path, "alice")
    with _fileio.working_dir_bound(str(conv / "files")):
        _read(CapReader(), _pdf(1), tiles="off")
        engine = CapReader()
        out = _read(engine, _pdf(2), tiles="off", reuse=True)
    assert len(engine.calls) == 1 and "reused" not in out


def test_the_kept_readings_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(vision_tools, "READINGS_MAX", 3)
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf(5)
    with _fileio.working_dir_bound(str(conv / "files")):
        for p in range(5):
            _read(CapReader(), pdf, page=p, tiles="off")
        kept = vision_tools.readings_for_conversation()
    assert [r["page"] for r in kept] == [2, 3, 4]


def test_the_page_tool_takes_reuse_and_says_so(tmp_path):
    from funhouse_agent.deep.review_agent import REVIEW_DESCRIPTIONS
    from funhouse_agent.deep.tools import make_vision_tools
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    tools = {t.name: t for t in make_vision_tools(
        engine=CapReader(), attachments={"sheet.pdf": pdf})}
    tool = tools["analyze_pdf_page"]
    assert "reuse" in tool.args and "reuse=true" in tool.description
    assert "reuse=true" in REVIEW_DESCRIPTIONS["analyze_pdf_page"]
    with _fileio.working_dir_bound(str(conv / "files")):
        tool.invoke({"attachment_key": "sheet.pdf", "tiles": "off",
                     "prompt": QUESTION})
        out = json.loads(tool.invoke({"attachment_key": "sheet.pdf",
                                      "tiles": "off", "reuse": True,
                                      "prompt": "other"}))
    assert "reused" in out


# ---------------------------------------------------------------------------
# E4: DXF geometry listed; notes written by a CAD library; no retyping
# ---------------------------------------------------------------------------

ezdxf = pytest.importorskip("ezdxf")


def _boring_plan(path, crlf=False):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 2                          # feet
    msp = doc.modelspace()
    for name in ("BORINGS", "BUILDING", "TITLE"):
        doc.layers.add(name)
    for label, (x, y) in (("B-1", (70, 50)), ("B-2", (210, 50))):
        msp.add_circle((x, y), 2.5, dxfattribs={"layer": "BORINGS"})
        msp.add_text(label, height=5, dxfattribs={
            "layer": "BORINGS", "insert": (x + 4, y + 2)})
    msp.add_lwpolyline([(80, 60), (200, 60), (200, 140), (80, 140)],
                       close=True, dxfattribs={"layer": "BUILDING"})
    msp.add_text("PROPOSED BUILDING", height=5, dxfattribs={
        "layer": "BUILDING", "insert": (100, 98)})
    msp.add_line((280, 170), (280, 190), dxfattribs={"layer": "TITLE"})
    msp.add_text("C-2 BORING PLAN", height=8, dxfattribs={
        "layer": "TITLE", "insert": (10, 380)})
    doc.saveas(path)
    # ezdxf writes the platform's line endings; set the ones under test.
    data = open(path, "rb").read().replace(b"\r\n", b"\n")
    open(path, "wb").write(data.replace(b"\n", b"\r\n") if crlf else data)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    return work


def _tools(**kw):
    from funhouse_agent.deep.tools import make_vision_tools
    return {t.name: t for t in make_vision_tools(engine=None, **kw)}


def _call(tool, **kw):
    return json.loads(tool.invoke(kw))


def test_a_dxf_read_lists_its_geometry_with_the_text_it_sits_by(folder):
    pytest.importorskip("planlens.tools")
    _boring_plan(folder / "plan.dxf")
    tools = _tools()
    out = _call(tools["open_document"], source="plan.dxf")
    assert "error" not in out, out
    text = out["text"]
    assert 'CIRCLE centre (210, 50) radius 2.5 [layer BORINGS] near "B-2"' \
        in text
    assert ("POLYLINE closed, 4 vertices: (80, 60) (200, 60) (200, 140) "
            "(80, 140) [layer BUILDING] encloses \"PROPOSED BUILDING\"") \
        in text
    assert "LINE (280, 170) to (280, 190), length 20 [layer TITLE]" in text
    assert out["entities_by_layer"]["BORINGS"] == {"circle": 2, "text": 2}
    assert out["n_geometry_lines"] == 4
    assert "annotate_dxf" in out["note"] and "PDF plot" in out["note"]
    hits = _call(tools["search_document"], handle="plan.dxf",
                 pattern="CIRCLE")
    assert hits["n_hits"] == 2


def _entities(path):
    doc = ezdxf.readfile(path)
    return [(e.dxftype(), e.dxf.layer,
             tuple(round(v, 6) for v in e.dxf.center) if e.dxftype() ==
             "CIRCLE" else None) for e in doc.modelspace()]


@pytest.mark.parametrize("crlf", [False, True])
def test_annotate_dxf_writes_notes_on_a_checked_copy(folder, crlf):
    src = folder / "plan.dxf"
    _boring_plan(src, crlf=crlf)
    before_bytes = src.read_bytes()
    before = _entities(src)
    tools = _tools()
    out = _call(tools["annotate_dxf"], source="plan.dxf", notes=[
        {"text": "Checked - B-2 location OK", "x": 220, "y": 35,
         "leader_to": [210, 50]},
        {"text": "Two\nlines", "at": [150, 120]}])
    assert "error" not in out, out
    assert out["saved"] == "plan_marked.dxf"
    assert out["notes_added"] == 2 and out["layer"] == "REVIEW"
    assert "all 8 original entities are kept" in out["check"]
    assert src.read_bytes() == before_bytes          # original untouched
    copy = folder / "plan_marked.dxf"
    after = _entities(copy)
    assert after[:len(before)] == before             # every entity, as was
    doc = ezdxf.readfile(copy)
    notes = [e for e in doc.modelspace() if e.dxf.layer == "REVIEW"]
    kinds = sorted(e.dxftype() for e in notes)
    assert kinds == ["LEADER", "MTEXT", "TEXT"]
    text = next(e for e in notes if e.dxftype() == "TEXT")
    assert text.dxf.text == "Checked - B-2 location OK"
    assert tuple(text.dxf.insert)[:2] == (220, 35)
    assert doc.layers.get("REVIEW").color == 1
    data = copy.read_bytes()
    assert (b"\r\n" in data) == crlf                 # its own line endings
    if not crlf:
        assert b"\r\n" not in data


def test_annotate_dxf_refuses_what_it_cannot_do(folder):
    _boring_plan(folder / "plan.dxf")
    tools = _tools()
    assert "notes must be" in _call(tools["annotate_dxf"], source="plan.dxf",
                                    notes=[])["error"]
    assert "x must be a number" in _call(
        tools["annotate_dxf"], source="plan.dxf",
        notes=[{"text": "a", "y": 3}])["error"]
    (folder / "sheet.pdf").write_bytes(_pdf())
    assert "not a DXF" in _call(tools["annotate_dxf"], source="sheet.pdf",
                                notes=[{"text": "a", "x": 1, "y": 2}])["error"]
    # A second round on the marked copy updates that copy, never the source.
    _call(tools["annotate_dxf"], source="plan.dxf",
          notes=[{"text": "one", "x": 1, "y": 2}])
    again = _call(tools["annotate_dxf"], source="plan_marked.dxf",
                  notes=[{"text": "two", "x": 3, "y": 4}])
    assert again["saved"] == "plan_marked.dxf"
    assert again["note"].startswith("'plan_marked.dxf' updated in place")
    over = _call(tools["annotate_dxf"], source="plan.dxf",
                 output_path="plan.dxf",
                 notes=[{"text": "x", "x": 1, "y": 2}])
    assert over["saved"] == "plan_marked.dxf"


def test_a_retyped_dxf_is_refused_and_a_new_one_is_checked(folder):
    _boring_plan(folder / "plan.dxf")
    tools = _tools()
    retyped = (folder / "plan.dxf").read_text(encoding="latin-1").replace(
        "B-2", "B-2 checked")
    out = _call(tools["save_file"], path="plan_v2.dxf", content=retyped)
    assert "retyped copy of 'plan.dxf'" in out["error"]
    assert "annotate_dxf(source='plan.dxf'" in out["hint"]
    assert not (folder / "plan_v2.dxf").exists()
    # A drawing made from scratch is saved, after a CAD library reads it.
    import io
    doc = ezdxf.new()
    doc.modelspace().add_circle((0, 0), 1)
    sio = io.StringIO()
    doc.write(sio)
    out = _call(tools["save_file"], path="new.dxf", content=sio.getvalue())
    assert "error" not in out, out
    assert "1 entities" in out["dxf_check"]
    # Text that is not a DXF at all is not saved as one.
    out = _call(tools["save_file"], path="broken.dxf",
                content="  0\nSECTION\n  2\nENTITIES\n  0\nCIRCLE\n 10\n")
    assert "Not saved" in out["error"]
    # Any other file is saved as before.
    out = _call(tools["save_file"], path="notes.txt", content="hello")
    assert "error" not in out and "dxf_check" not in out


# ---------------------------------------------------------------------------
# E6: read_text_file says what it returned, and takes limit / offset
# ---------------------------------------------------------------------------

def _text(tmp_path, n=48213, body="abcdefghij"):
    p = tmp_path / "long.txt"
    p.write_text((body * (n // len(body) + 1))[:n], encoding="utf-8")
    return str(p)


def _read_text(**kw):
    return json.loads(dispatch_extended_tool(
        "read_text_file", kw, engine=None, attachments={}))


def test_a_capped_read_says_so(tmp_path):
    out = _read_text(path=_text(tmp_path), max_chars=20000)
    assert out["returned_chars"] == 12000
    assert out["showing"] == (
        "chars 0-12,000 of 48,213; pass offset=12000 for more (20,000 were "
        "asked for; 12,000 came back, the most one call returns)")
    assert out["next_offset"] == 12000 and out["truncated"] is True


def test_limit_and_offset_page_through_the_file(tmp_path):
    path = _text(tmp_path)
    out = _read_text(path=path, limit=3000)
    assert out["returned_chars"] == 3000
    assert out["showing"] == "chars 0-3,000 of 48,213; pass offset=3000 " \
                             "for more"
    end = _read_text(path=path, offset=45000, limit=6000)
    assert end["showing"] == "chars 45,000-48,213 of 48,213"
    assert "next_offset" not in end and "truncated" not in end
    # the default is unchanged: 6,000
    assert _read_text(path=path)["returned_chars"] == 6000


def test_heavily_escaped_text_comes_back_in_a_piece_that_fits(tmp_path):
    path = _text(tmp_path, n=30000, body='<a href="x">\\</a>\n')
    raw = dispatch_extended_tool("read_text_file",
                                 {"path": path, "limit": 12000},
                                 engine=None, attachments={})
    out = json.loads(raw)
    assert len(raw) <= vision_tools._TEXT_READ_JSON_BUDGET
    assert out["returned_chars"] < 12000
    assert "the most that fits one result" in out["showing"]
    assert out["next_offset"] == out["returned_chars"]


def test_the_text_tool_takes_limit():
    from funhouse_agent.deep.tools import make_vision_tools
    tool = next(t for t in make_vision_tools(engine=None)
                if t.name == "read_text_file")
    assert "limit" in tool.args and "showing" in tool.description
