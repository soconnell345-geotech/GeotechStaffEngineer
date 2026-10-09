"""Live smoke wave 3 (module_work/live_smoke/runs/w3-confirm-sonnet/
REVIEW.md), the agent-side polish.

* F3  a markup call whose every mark is skipped (and that removes nothing)
      writes no file and reports no ``output_path``, so no "marked" copy
      without marks becomes a download card; it says why.
* F6  zoom readings are kept for the conversation, keyed by page and
      region, listed in the turn note with the box to pass back, and
      ``render_region(reuse=true)`` hands one back without a new look; a
      page reuse never returns a zoom.

(F4 and F7 are tested beside the markup check, in
test_markup_check_offline.py; F1 in test_wave3_latency_offline.py.)

Fakes and synthetic PDFs only: no model, no network.
"""

from __future__ import annotations

import json
import os

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio, document_tools, vision_tools  # noqa: E402


@pytest.fixture
def folder(tmp_path, monkeypatch):
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_MARKUP_AUTHOR", raising=False)
    return work


def _pdf(n=2) -> bytes:
    d = fitz.open()
    for i in range(n):
        page = d.new_page(width=612, height=792)
        page.insert_text((72, 100), f"Sheet {i + 1} STD. NO. 21.01",
                         fontsize=12)
    data = d.tobytes()
    d.close()
    return data


def _leftovers(root) -> list:
    return [n for _r, _d, names in os.walk(str(root)) for n in names
            if ".part" in n]


# ---------------------------------------------------------------------------
# F3: every mark skipped -> nothing written
# ---------------------------------------------------------------------------

def _all_skipped(args):
    """planlens as F04's first call answered: the temporary copy written,
    nothing on it, every row skipped."""
    with open(args["output_path"], "wb") as fh:
        fh.write(b"%PDF-1.7 zero marks")
    return json.dumps({
        "handle": args["handle"], "output_path": args["output_path"],
        "author": args["author"], "appended_to_existing": False,
        "n_written": 0, "n_skipped": 1,
        "note": "a NEW file: the document you opened is unchanged. ...",
        "written": [],
        "skipped": [{"index": 0, "reason": "a box read off a view 792 pt "
                                           "across is refused: zoom"}]})


def test_an_all_skipped_call_writes_nothing_and_says_why(folder):
    out = document_tools.write_marked_copy(
        "set.pdf", [{"kind": "box", "page": 0, "comment": "x",
                     "bbox": [1, 1, 5, 5]}],
        "set_marked.pdf", True, "me", _all_skipped)
    assert "error" not in out, out
    assert "output_path" not in out and "appended_to_existing" not in out
    assert out["nothing_written"] is True and out["n_written"] == 0
    assert out["skipped"][0]["index"] == 0           # the reasons are kept
    assert out["note"].startswith("Nothing was written: every mark was "
                                  "skipped")
    assert "'set_marked.pdf' was not created" in out["note"]
    assert not (folder / "set_marked.pdf").exists()
    assert _leftovers(folder.parent) == []


def test_an_all_skipped_append_leaves_the_existing_copy_as_it_was(folder):
    (folder / "set_marked.pdf").write_bytes(b"%PDF earlier marks")
    out = document_tools.write_marked_copy(
        "set.pdf", [{"kind": "box", "page": 0, "comment": "x",
                     "bbox": [1, 1, 5, 5]}],
        "set_marked.pdf", True, "me", _all_skipped)
    assert "output_path" not in out
    assert "'set_marked.pdf' is unchanged" in out["note"]
    assert (folder / "set_marked.pdf").read_bytes() == b"%PDF earlier marks"
    assert _leftovers(folder.parent) == []


def test_a_call_that_places_one_mark_still_writes(folder):
    def one(args):
        with open(args["output_path"], "wb") as fh:
            fh.write(b"%PDF one mark")
        return json.dumps({"handle": args["handle"],
                           "output_path": args["output_path"],
                           "n_written": 1, "n_skipped": 0, "written": [{}]})
    out = document_tools.write_marked_copy(
        "set.pdf", [{"kind": "note", "page": 0, "comment": "x",
                     "point": [1, 1]}], "set_marked.pdf", True, "me", one)
    assert out["output_path"].endswith("set_marked.pdf")
    assert "nothing_written" not in out
    assert (folder / "set_marked.pdf").read_bytes() == b"%PDF one mark"


def _tools(pdf, engine=None):
    pytest.importorskip("planlens.tools")
    pytest.importorskip("langchain_core")
    if not document_tools.has_tool("annotate_document"):
        pytest.skip("installed planlens has no annotate_document")
    from funhouse_agent.deep.tools import make_vision_tools
    tools = {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"set.pdf": pdf})}
    handle = json.loads(tools["open_document"].invoke(
        {"source": "set.pdf"}))["handle"]
    return tools, handle


def test_through_the_agent_tool_no_card_is_offered(folder):
    """End to end with planlens: a quote on no page is skipped, nothing is
    written, and the result carries no output_path for the app to turn into
    a download card (webapp/output_capture)."""
    tools, handle = _tools(_pdf())
    raw = tools["annotate_document"].invoke({
        "handle": handle, "output_path": "set_marked.pdf", "markups": [
            {"kind": "highlight", "page": 0, "comment": "not here",
             "quote": "WORDS PRINTED ON NO PAGE AT ALL"}]})
    out = json.loads(raw)
    assert out["n_written"] == 0 and out["skipped"], out
    assert out.get("nothing_written") is True
    assert '"output_path"' not in raw
    assert "check" not in out and "in_file" not in out
    assert not (folder / "set_marked.pdf").exists()
    output_capture = pytest.importorskip("webapp.output_capture")
    assert output_capture.paths_in(raw)[0] == []


# ---------------------------------------------------------------------------
# F6: zoom readings kept for reuse, keyed by page and region
# ---------------------------------------------------------------------------

class Looker:
    """Records every vision call; answers with what it was asked."""

    def __init__(self):
        self.calls = []

    def analyze_image(self, image, prompt=""):
        self.calls.append(prompt)
        return f"I see the region ({len(self.calls)})."


ZOOM = "Read every label in this box."


def _conversation(tmp_path, name):
    conv = tmp_path / name
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text(json.dumps({"thread_id": name}),
                                    encoding="utf-8")
    return conv


def _zoom(engine, pdf, **args):
    base = {"attachment_key": "sheet.pdf", "page": 0, "prompt": ZOOM,
            "bbox": [300, 300, 400, 380]}
    return json.loads(vision_tools.dispatch_extended_tool(
        "render_region", {**base, **args}, engine=engine,
        attachments={"sheet.pdf": pdf}))


def _page(engine, pdf, **args):
    base = {"attachment_key": "sheet.pdf", "page": 0, "prompt": "What?",
            "tiles": "off"}
    return json.loads(vision_tools.dispatch_extended_tool(
        "analyze_pdf_page", {**base, **args}, engine=engine,
        attachments={"sheet.pdf": pdf}))


@pytest.fixture
def fresh_reads():
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def test_a_zoom_is_kept_with_its_page_and_region(tmp_path, fresh_reads):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _zoom(Looker(), pdf)
        _zoom(Looker(), pdf, bbox=[50, 60, 200, 130])     # another region
        kept = vision_tools.readings_for_conversation()
    zooms = [r for r in kept if r["view"] == vision_tools.ZOOM_VIEW]
    assert len(zooms) == 2                       # one per region, both kept
    assert zooms[0]["region"] == first["view"]
    assert zooms[0]["page"] == 0 and zooms[0]["pdf_page"] == 1
    assert zooms[0]["prompt"] == ZOOM and "result" not in zooms[0]
    record = conv / vision_tools.READINGS_FILE
    assert record.is_file()                      # beside the read record
    assert not (conv / "files" / vision_tools.READINGS_FILE).exists()


@pytest.mark.parametrize("again", [
    {},                                      # the same bbox as before
    "kept",                                  # the bbox the turn note gives
])
def test_reuse_hands_a_kept_zoom_back_without_a_new_look(tmp_path,
                                                         fresh_reads, again):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _zoom(Looker(), pdf)
        args = {"bbox": first["view"]} if again == "kept" else {}
        engine = Looker()
        out = _zoom(engine, pdf, prompt="What size is the pipe?",
                    reuse=True, **args)
    assert engine.calls == []                     # no new look
    assert out["analysis"] == first["analysis"]
    assert out["reused"].startswith("An earlier zoom on this region")
    assert ZOOM in out["reused"] and "without reuse" in out["reused"]


def test_reuse_survives_a_restart(tmp_path, fresh_reads):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _zoom(Looker(), pdf)
    vision_tools.clear_read_log()                 # a restart
    vision_tools.clear_repeat_reads()
    with _fileio.working_dir_bound(str(conv / "files")):
        engine = Looker()
        out = _zoom(engine, pdf, reuse=True)
    assert engine.calls == [] and out["analysis"] == first["analysis"]


@pytest.mark.parametrize("bbox", [
    [50, 60, 200, 130],          # elsewhere on the page
    [320, 320, 340, 340],        # inside, but the kept zoom is far wider
    [250, 250, 450, 450],        # wider than the kept zoom
])
def test_a_zoom_of_another_region_is_looked_at_now(tmp_path, fresh_reads,
                                                    bbox):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        _zoom(Looker(), pdf)
        engine = Looker()
        out = _zoom(engine, pdf, bbox=bbox, reuse=True)
    assert len(engine.calls) == 1 and "reused" not in out
    assert out["reuse_note"] == vision_tools.NO_EARLIER_ZOOM


def test_a_page_reuse_never_hands_back_a_zoom(tmp_path, fresh_reads):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        _zoom(Looker(), pdf)
        engine = Looker()
        out = _page(engine, pdf, reuse=True)
    assert len(engine.calls) == 1 and "reused" not in out
    assert out["reuse_note"] == vision_tools.NO_EARLIER_READING


def test_reuse_is_never_forced_on_a_zoom(tmp_path, fresh_reads):
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        _zoom(Looker(), pdf)
        engine = Looker()
        out = _zoom(engine, pdf, prompt="A new question")
    assert len(engine.calls) == 1 and "reused" not in out


def test_the_agent_tool_takes_reuse_and_says_so():
    pytest.importorskip("langchain_core")
    from funhouse_agent.deep.tools import make_vision_tools
    tool = next(t for t in make_vision_tools(engine=Looker(), attachments={})
                if t.name == "render_region")
    assert "reuse" in tool.args
    assert "kept zoom of the same region" in tool.description


def test_the_turn_note_lists_a_kept_zoom_with_its_box(tmp_path, fresh_reads):
    core = pytest.importorskip("webapp.core")
    conv = _conversation(tmp_path, "alice")
    pdf = _pdf()
    with _fileio.working_dir_bound(str(conv / "files")):
        first = _zoom(Looker(), pdf)
    note = core.readings_note(str(conv / "files"))
    box = ", ".join(f"{float(v):g}" for v in first["view"])
    assert f"'sheet.pdf' p. 1 (zoom bbox=[{box}], asked \"{ZOOM}\")" in note
    assert "render_region(..., pdf_page=N, bbox=<the zoom's bbox>, " \
           "reuse=true)" in note
    assert len(note) <= core.READINGS_NOTE_MAX_CHARS
