"""Photograph pages nothing looked at (BACKLOG DR-48).

At turn end the core lists, per attached document, the photograph pages
(planlens' own page role) that no tool or helper looked at and that were
not on a contact sheet a later call looked at. Behind
``GEOTECH_PHOTO_PAGES``: off (the default) nothing of it runs; ``record``
keeps the list on the turn's record at no model call; ``retry`` also tells
the model once. :func:`photo_coverage` reads a saved folder with no switch.
Made-up PDFs built here with PyMuPDF.
"""

from __future__ import annotations

import itertools
import json

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("planlens.document.roles")

from aec_agent_core import cited_pages as cp  # noqa: E402
from aec_agent_core import review_flags  # noqa: E402

PROSE = ("The made-up site lies on a gentle slope above a small creek, and "
         "the made-up borings met stiff clay over weathered rock. ") * 14
BAD = "The clay is described on PDF pages 40-41."
_IDS = itertools.count(1)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (review_flags.ALL_ENVS + review_flags.LATER_ENVS
                + review_flags.SETTINGS_ENVS):
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


def _png() -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 240, 160), False)
    pix.clear_with(150)
    return pix.tobytes("png")


def make_pdf(path, kinds) -> None:
    """A made-up report: ``"text"`` pages of prose, ``"photo"`` pages of
    one picture under a photograph title."""
    doc = fitz.open()
    png = _png()
    n = 0
    for kind in kinds:
        page = doc.new_page(width=612, height=792)
        if kind == "photo":
            n += 1
            page.insert_text((72, 90), f"PHOTOGRAPH {n}: VIEW LOOKING NORTH",
                             fontsize=20)
            page.insert_image(fitz.Rect(72, 130, 540, 560), stream=png)
            page.insert_text((72, 600), f"Photo {n}. Made-up test pit area.",
                             fontsize=10)
        else:
            page.insert_text((72, 90), "SITE DESCRIPTION", fontsize=16)
            page.insert_textbox(fitz.Rect(72, 120, 540, 720), PROSE,
                                fontsize=10)
    doc.save(str(path))
    doc.close()


def write_calls(conv, calls, turn=1) -> None:
    """Tool calls on the conversation's record, as the activity log keeps
    them."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for name, args, result in calls:
            rid = f"r{next(_IDS)}"
            fh.write(json.dumps({"turn": turn, "event": "tool_start",
                                 "name": name, "run_id": rid,
                                 "args": args}) + "\n")
            fh.write(json.dumps({
                "turn": turn, "event": "tool_end", "name": name,
                "run_id": rid,
                "result": (result if isinstance(result, str)
                           else json.dumps(result))}) + "\n")


def opened(name, handle, n):
    return ("open_document", {"source": name},
            {"handle": handle, "name": name, "n_pages": n})


def look(name, page):
    return ("analyze_pdf_page", {"attachment_key": name, "page": page},
            "A made-up photograph of a test pit.")


A_LOOKED_B_READ = [
    opened("a.pdf", "doc_aaa", 2), look("a.pdf", 1),
    opened("b.pdf", "doc_bbb", 3),
    ("read_document", {"handle": "doc_bbb"},
     {"handle": "doc_bbb", "pages_returned": "0-2"})]


def conversation(tmp_path, attach=("a.pdf", "b.pdf"), calls=()):
    """a.pdf: prose, then one photograph page (PDF page 2); b.pdf: prose,
    then two photograph pages (PDF pages 2-3)."""
    conv = tmp_path / "conv"
    files = conv / "files"
    files.mkdir(parents=True)
    make_pdf(files / "a.pdf", ["text", "photo"])
    make_pdf(files / "b.pdf", ["text", "photo", "photo"])
    if attach is not None:
        (conv / "attachments.json").write_text(json.dumps(list(attach)),
                                               encoding="utf-8")
    (conv / "activity.jsonl").write_text("", encoding="utf-8")
    write_calls(conv, calls)
    return conv


def logger_for(conv, logged):
    """A log that keeps each event and appends it to the record, as the
    pages' ActivityLogger does."""
    def log(data):
        logged.append(data)
        with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"agent": "primary",
                                 "event": cp.RECORD_EVENT, **data}) + "\n")
    return log


# -- the measure ---------------------------------------------------------------

def test_planlens_names_the_made_up_photograph_pages(tmp_path):
    make_pdf(tmp_path / "p.pdf", ["text", "photo", "photo", "text"])
    make_pdf(tmp_path / "q.pdf", ["text", "text"])
    assert cp.photograph_pages(str(tmp_path / "p.pdf")) == ([1, 2], "")
    assert cp.photograph_pages(str(tmp_path / "q.pdf")) == ([], "")


# -- the measure on a saved folder (no switch) ------------------------------------

def test_the_record_names_each_documents_photo_pages_nothing_looked_at(
        tmp_path):
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    got = cp.photo_coverage(str(conv))
    assert got["checked"] is True and got["attached"] == "attachment list"
    assert got["documents"] == ["a.pdf", "b.pdf"]
    assert got["photos"] == {"a.pdf": "2", "b.pdf": "2-3"}
    assert got["unseen"] == {"b.pdf": "2-3"}     # read as text is not seen
    assert "left_out" not in got and "told_before" not in got


def test_unseen_is_empty_when_every_photo_page_was_looked_at(tmp_path):
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ + [
        look("b.pdf", 1), look("b.pdf", 2)])
    assert cp.photo_coverage(str(conv))["unseen"] == {}


def test_a_contact_sheet_counts_once_a_later_call_looked_at_it(tmp_path):
    sheet = ("render_page_thumbnails", {"handle": "doc_bbb"},
             {"handle": "doc_bbb", "sheets": [
                 {"image_path": "/made/up/files/thumbs/doc_bbb_1.png",
                  "pages": "0-2", "n_pages": 3}]})
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ + [sheet])
    assert cp.photo_coverage(str(conv))["unseen"] == {"b.pdf": "2-3"}
    write_calls(conv, [("analyze_image",
                        {"image_path": "thumbs/doc_bbb_1.png"},
                        "Three made-up page thumbnails.")])
    assert cp.photo_coverage(str(conv))["unseen"] == {}
    # ...kept apart: what the cited-pages check reads is unchanged
    rec = cp.pages_opened(str(conv))
    assert rec.hows(2, ["b.pdf"]) == {"text"}
    assert rec.sheet_seen == {"b.pdf": {0, 1, 2}}


def test_only_a_looking_tool_naming_the_sheet_counts_as_seeing_it(tmp_path):
    sheet = ("render_page_thumbnails", {"handle": "doc_bbb"},
             {"handle": "doc_bbb", "sheets": [
                 {"image_path": "/made/up/files/thumbs/doc_bbb_1.png",
                  "pages": "0-2", "n_pages": 3}]})
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ + [sheet])
    write_calls(conv, [
        ("list_files", {"path": "thumbs/doc_bbb_1.png"},
         {"files": ["thumbs/doc_bbb_1.png"]}),
        ("task", {"description": "thumbs/doc_bbb_1.png"},
         "A made-up helper reply.")])
    assert cp.photo_coverage(str(conv))["unseen"] == {"b.pdf": "2-3"}
    write_calls(conv, [("analyze_image",
                        {"attachment_key": "/made/up/files/thumbs/"
                                           "doc_bbb_1.png",
                         "prompt": "Which pages are photographs?"},
                        "Three made-up page thumbnails.")])
    assert cp.photo_coverage(str(conv))["unseen"] == {}


def test_a_document_whose_page_kinds_cannot_be_told_is_left_out(tmp_path):
    conv = conversation(tmp_path, attach=("a.pdf", "notes.docx", "gone.pdf"),
                        calls=A_LOOKED_B_READ)
    (conv / "files" / "notes.docx").write_bytes(b"made-up")
    got = cp.photo_coverage(str(conv))
    assert got["documents"] == ["a.pdf"]
    assert got["unseen"] == {}
    assert set(got["left_out"]) == {"notes.docx", "gone.pdf"}
    assert got["left_out"]["notes.docx"].startswith("not a PDF")
    assert "not in the conversation's folder" in got["left_out"]["gone.pdf"]


def test_without_an_upload_list_the_call_logs_documents_are_used(tmp_path):
    conv = conversation(tmp_path, attach=None, calls=[
        opened("b.pdf", "doc_bbb", 3),
        ("read_pdf_text", {"source": "/elsewhere/reference.pdf"},
         {"pages": [{"page": 0, "has_text_layer": True}]})])
    got = cp.photo_coverage(str(conv))
    assert got["attached"] == "call log"
    assert got["documents"] == ["b.pdf"]         # a.pdf was never opened;
    assert got["unseen"] == {"b.pdf": "2-3"}     # a file elsewhere is no
    assert "left_out" not in got                 # attachment


def test_no_attached_document_means_no_record(tmp_path):
    assert cp.photo_coverage(str(conversation(tmp_path, attach=None))) is None
    conv2 = tmp_path / "two"
    (conv2 / "files").mkdir(parents=True)
    (conv2 / "attachments.json").write_text("[]", encoding="utf-8")
    (conv2 / "activity.jsonl").write_text("", encoding="utf-8")
    assert cp.photo_coverage(str(conv2)) is None


def _counted(monkeypatch):
    """Count the calls of the measure and of planlens' page pass."""
    calls = {"photo_coverage": 0, "photograph_pages": 0}
    for name in calls:
        real = getattr(cp, name)

        def wrapper(*a, _real=real, _name=name, **k):
            calls[_name] += 1
            return _real(*a, **k)
        monkeypatch.setattr(cp, name, wrapper)
    return calls


def test_switch_off_nothing_of_it_runs_and_the_record_is_todays(
        tmp_path, monkeypatch):
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    calls = _counted(monkeypatch)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    assert check.photo_mode == "off"
    answer = "The site is on a gentle slope."
    assert check.feedback(answer) is None
    assert check.finish(answer) == answer
    assert logged == []              # an answer citing no page: no event
    # an answer citing pages: the event as it was, with no photo field
    check = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    assert check.finish(BAD) == BAD
    expected = {**cp.check_answer(BAD, record_folder=str(conv)),
                "mode": "off", "action": "none"}
    assert logged == [expected]
    assert cp.PHOTO_KEY not in logged[0] and "photo_mode" not in logged[0]
    assert calls == {"photo_coverage": 0, "photograph_pages": 0}


def test_record_keeps_it_on_the_record_and_hands_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "record")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    calls = _counted(monkeypatch)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    assert check.photo_mode == "record"
    assert check.feedback("The site is on a gentle slope.") is None
    assert logged == []                          # nothing told, no event
    answer = "The site is on a gentle slope."
    assert check.finish(answer) == answer
    assert len(logged) == 1
    ev = logged[0]
    assert ev["action"] == "none" and ev["photo_mode"] == "record"
    assert ev[cp.PHOTO_KEY]["unseen"] == {"b.pdf": "2-3"}
    assert "told" not in ev[cp.PHOTO_KEY]
    assert calls["photo_coverage"] == 1 and calls["photograph_pages"] == 2


def test_an_error_in_the_check_is_recorded_and_never_fails_the_turn(
        tmp_path, monkeypatch):
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)

    def broken(_path):
        raise RuntimeError("made-up failure")

    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv), log=logged.append,
                             photo_mode="retry", photo_source=broken)
    assert check.feedback("An answer.") is None
    assert check.finish("An answer.") == "An answer."
    got = logged[-1][cp.PHOTO_KEY]
    assert got["documents"] == [] and got["unseen"] == {}
    assert got["left_out"]["b.pdf"].endswith("RuntimeError")

    def boom(*_a, **_k):
        raise RuntimeError("made-up failure")

    monkeypatch.setattr(cp, "photo_coverage", boom)
    logged.clear()
    check = cp.TurnPageCheck(record_folder=str(conv), log=logged.append,
                             photo_mode="retry")
    assert check.feedback("An answer.") is None
    assert check.finish("An answer.") == "An answer."
    assert logged[-1][cp.PHOTO_KEY] == {
        "checked": False,
        "why": "the check failed: RuntimeError: made-up failure"}


# -- the switch -----------------------------------------------------------------

def test_the_switch_is_off_unless_set(monkeypatch):
    assert review_flags.photo_pages() == "off"
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "Record")
    assert review_flags.photo_pages() == "record"
    for value in ("retry", "1", "on"):
        monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, value)
        assert review_flags.photo_pages() == "retry"
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "0")
    assert review_flags.photo_pages() == "off"
    assert review_flags.PHOTO_PAGES_ENV in review_flags.LATER_ENVS
    # an arm left unnamed clears it, so `baseline` stays the page as it is
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    assert "GEOTECH_PHOTO_PAGES=retry" in review_flags.describe()
    with review_flags.switches({}):
        assert review_flags.photo_pages() == "off"
    assert review_flags.photo_pages() == "retry"     # restored after


def test_switch_off_the_page_checks_note_is_exactly_todays(tmp_path):
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    check = cp.TurnPageCheck(record_folder=str(conv), mode="retry")
    said = check.feedback(BAD)
    assert said == cp.note_for_model(cp.check_answer(BAD,
                                                     record_folder=str(conv)))
    assert cp.PHOTO_NOTE_PREFIX not in said
    assert check.retry_status == cp.RETRY_STATUS


def test_switch_on_tells_the_model_once_naming_the_document_and_pages(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    said = check.feedback("The site is on a gentle slope.")
    assert said.startswith(cp.PHOTO_NOTE_PREFIX)
    assert "- b.pdf: PDF pages 2-3" in said and "a.pdf" not in said
    assert check.retry_status == cp.PHOTO_RETRY_STATUS
    assert check.feedback("The site is on a gentle slope.") is None   # once
    assert logged[0]["action"] == "photos_fed_back"
    assert logged[0][cp.PHOTO_KEY]["told"] == {"b.pdf": "2-3"}
    # a later turn of the conversation is not told the same pages again,
    # and its record says they were told before
    later = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    assert later.feedback("Another answer.") is None
    later.finish("Another answer.")
    assert logged[-1][cp.PHOTO_KEY]["unseen"] == {"b.pdf": "2-3"}
    assert logged[-1][cp.PHOTO_KEY]["told_before"] == {"b.pdf": "2-3"}


def test_switch_on_nothing_is_told_when_every_photo_page_was_seen(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ + [
        look("b.pdf", 1), look("b.pdf", 2)])
    check = cp.TurnPageCheck(record_folder=str(conv))
    assert check.feedback("The site is on a gentle slope.") is None


def test_both_checks_speak_in_one_note(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv), mode="retry",
                             log=logger_for(conv, logged))
    said = check.feedback(BAD)
    assert said.startswith(cp.NOTE_PREFIX) and cp.PHOTO_NOTE_PREFIX in said
    assert check.retry_status == (f"{cp.RETRY_STATUS}; "
                                  f"{cp.PHOTO_RETRY_STATUS}")
    # the page check's event as it was, then the photograph pages told
    assert [e["action"] for e in logged] == ["fed_back", "photos_fed_back"]
    assert cp.PHOTO_KEY not in logged[0]
    assert logged[1][cp.PHOTO_KEY]["told"] == {"b.pdf": "2-3"}


def test_pages_count_as_told_only_once_the_note_is_handed(
        tmp_path, monkeypatch):
    """A page that may drop a note (``confirms_notes``) records the pages
    as told only when it says the note went to the model."""
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    logged = []
    dropped = cp.TurnPageCheck(record_folder=str(conv), confirms_notes=True,
                               log=logger_for(conv, logged))
    assert dropped.feedback("An answer.").startswith(cp.PHOTO_NOTE_PREFIX)
    dropped.finish("An answer.")                 # the note never went out
    assert [e["action"] for e in logged] == ["none"]
    assert "told" not in logged[0][cp.PHOTO_KEY]
    # so the next turn tells them...
    handed = cp.TurnPageCheck(record_folder=str(conv), confirms_notes=True,
                              log=logger_for(conv, logged))
    said = handed.feedback("Another answer.")
    assert "- b.pdf: PDF pages 2-3" in said
    assert [e["action"] for e in logged] == ["none"]
    handed.note_handed()                         # ...and this one went out
    handed.note_handed()                         # (once is enough)
    assert [e["action"] for e in logged] == ["none", "photos_fed_back"]
    assert logged[1][cp.PHOTO_KEY]["told"] == {"b.pdf": "2-3"}
    later = cp.TurnPageCheck(record_folder=str(conv), confirms_notes=True,
                             log=logger_for(conv, logged))
    assert later.feedback("A third answer.") is None


def test_at_a_met_image_ceiling_nothing_is_told_and_nothing_marked_told(
        tmp_path, monkeypatch):
    """Release review: with the turn's image ceiling met (BACKLOG TA-43) a
    note asking for a look at the photograph pages would only have the
    looks refused, and the pages would count as told for the rest of the
    conversation. So the note waits: nothing is said, nothing is recorded
    as told, and the next turn (its ceiling new) is told."""
    from aec_agent_core.deep import keep_work as kw
    from aec_agent_core.deep import limits
    monkeypatch.setenv(review_flags.PHOTO_PAGES_ENV, "retry")
    monkeypatch.setenv(limits.TURN_MAX_IMAGE_CALLS_ENV, "4")
    conv = conversation(tmp_path, calls=A_LOOKED_B_READ)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    with kw.turn_scope(kw.TurnScope()) as scope:
        scope.image_calls = 4
        assert check.feedback("The site is on a gentle slope.") is None
        check.finish("The site is on a gentle slope.")
    assert not any(e.get("action") == "photos_fed_back" for e in logged)
    assert logged[-1][cp.PHOTO_KEY]["unseen"] == {"b.pdf": "2-3"}
    assert "told_before" not in logged[-1][cp.PHOTO_KEY]
    later = cp.TurnPageCheck(record_folder=str(conv),
                             log=logger_for(conv, logged))
    with kw.turn_scope(kw.TurnScope()) as scope:
        scope.image_calls = 3                      # one look left
        said = later.feedback("Another answer.")
    assert said.startswith(cp.PHOTO_NOTE_PREFIX)
    assert "- b.pdf: PDF pages 2-3" in said
