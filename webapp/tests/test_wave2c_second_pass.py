"""Live smoke wave 2c, second pass (module_work/live_smoke/runs/
w2c-review-sonnet/REVIEW.md, "Second pass"): the web-app side.

* E2  the turn note names the page readings kept for the conversation and
      how to have one back without a new look.
* E5  two SharePoint downloads at once no longer lose a ledger entry (F41:
      two downloads 3 ms apart, one entry lost).
"""

import json
import os
import threading

import pytest

import webapp.core as core


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))


# ---------------------------------------------------------------------------
# E5: the download ledger under parallel downloads
# ---------------------------------------------------------------------------

def test_parallel_downloads_keep_every_ledger_entry(tmp_path):
    conv = tmp_path / "conv"
    conv.mkdir()
    n = 24
    gate = threading.Barrier(n)

    def fetch(i):
        gate.wait(timeout=10)                # all at once
        core.record_download(str(conv), f"Shared/Docs/file_{i}.pdf",
                             str(conv / "files" / f"file_{i}.pdf"), 100 + i)

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    entries = core.load_downloads(str(conv))
    assert sorted(e["remote"] for e in entries) == sorted(
        f"Shared/Docs/file_{i}.pdf" for i in range(n))
    # no half-written ledger left behind, and the in-flight name is one the
    # SharePoint mirror never uploads
    assert [p for p in os.listdir(conv) if p.endswith(".tmp")] == []
    assert core.is_in_flight_file(
        f"{core.DOWNLOADS_LEDGER}.{os.getpid()}-1234.tmp")


def test_a_repeat_download_still_replaces_its_entry(tmp_path):
    conv = tmp_path / "conv"
    core.record_download(str(conv), "Shared/a.pdf", str(tmp_path / "a.pdf"))
    core.record_download(str(conv), "shared/A.PDF", str(tmp_path / "a2.pdf"),
                         7)
    (entry,) = core.load_downloads(str(conv))
    assert entry["local"].endswith("a2.pdf") and entry["bytes"] == 7


# ---------------------------------------------------------------------------
# E2: the turn note names the readings kept
# ---------------------------------------------------------------------------

def _reads():
    return [{"document": "21.01.pdf", "page": 0, "pdf_page": 1,
             "view": "page+tiles 3x3", "tool": "analyze_pdf_page"}]


def test_the_note_names_the_readings_and_how_to_reuse_one(tmp_path,
                                                          monkeypatch):
    from funhouse_agent import vision_tools
    files = tmp_path / "c" / "files"
    files.mkdir(parents=True)
    monkeypatch.setattr(vision_tools, "reads_for_conversation",
                        lambda folder=None: _reads())
    monkeypatch.setattr(vision_tools, "readings_for_conversation",
                        lambda folder=None: [
                            {"document": "21.01.pdf", "page": 0,
                             "pdf_page": 1, "view": "page+tiles 3x3",
                             "prompt": "Transcribe ALL text on this sheet "
                                       "exactly (it is rotated 90 degrees). "
                                       "Include the title block."}])
    note = core.reads_note(str(files))
    assert "Already LOOKED AT" in note
    assert "Readings already taken and kept" in note
    assert "analyze_pdf_page(..., pdf_page=N, reuse=true)" in note
    assert ("'21.01.pdf' p. 1 (page+tiles 3x3, asked \"Transcribe ALL text "
            "on this sheet exactly (it is rotated...\")") in note


def test_many_readings_are_listed_newest_first_within_a_budget(
        tmp_path, monkeypatch):
    from funhouse_agent import vision_tools
    files = tmp_path / "c" / "files"
    files.mkdir(parents=True)
    kept = [{"document": f"sheet_{i:02d}.pdf", "page": 0, "pdf_page": 1,
             "view": "page", "prompt": "What is on this sheet?"}
            for i in range(40)]
    monkeypatch.setattr(vision_tools, "reads_for_conversation",
                        lambda folder=None: _reads())
    monkeypatch.setattr(vision_tools, "readings_for_conversation",
                        lambda folder=None: kept)
    line = core.readings_note(str(files))
    assert len(line) <= core.READINGS_NOTE_MAX_CHARS
    assert line.index("sheet_39.pdf") < line.index("sheet_38.pdf")
    assert "more." in line and "sheet_00.pdf" not in line


def test_no_readings_adds_nothing(tmp_path, monkeypatch):
    from funhouse_agent import vision_tools
    files = tmp_path / "c" / "files"
    files.mkdir(parents=True)
    monkeypatch.setattr(vision_tools, "readings_for_conversation",
                        lambda folder=None: [])
    assert core.readings_note(str(files)) == ""
    monkeypatch.setattr(vision_tools, "readings_for_conversation",
                        lambda folder=None: (_ for _ in ()).throw(
                            RuntimeError("no record")))
    assert core.readings_note(str(files)) == ""


def test_the_note_reads_the_real_record(tmp_path):
    """End to end on disk: a page read in a conversation folder shows in
    that conversation's next turn note."""
    fitz = pytest.importorskip("fitz")
    from funhouse_agent import _fileio, vision_tools
    conv = tmp_path / "alice"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text(json.dumps({"thread_id": "alice"}),
                                    encoding="utf-8")
    d = fitz.open()
    d.new_page().insert_text((72, 100), "Sheet 1")
    pdf = d.tobytes()
    d.close()

    class Eye:
        def analyze_image(self, image, prompt=""):
            return "a sheet"

    with _fileio.working_dir_bound(str(conv / "files")):
        vision_tools.dispatch_extended_tool(
            "analyze_pdf_page", {"attachment_key": "set.pdf", "page": 0,
                                 "tiles": "off", "prompt": "Read the title"},
            engine=Eye(), attachments={"set.pdf": pdf})
    note = core.reads_note(str(conv / "files"))
    assert "'set.pdf' p. 1 (page, asked \"Read the title\")" in note
    vision_tools.clear_read_log()
    vision_tools.clear_repeat_reads()
