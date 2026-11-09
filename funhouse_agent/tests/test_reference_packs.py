"""The app's side of local reference packs (BACKLOG REF-64 part (A); plan
``module_work/REFERENCE_PACKS_PLAN.md`` sections 2.4 to 2.8).

Every value here is made up: a fictitious "Example Practice Manual" about a
fictitious fill, zorblite, built in ``tmp_path``. No real manual is used.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from funhouse_agent import reference_packs, reference_vision

pytest.importorskip("geotech_references")
from geotech_references import _retrieval_db  # noqa: E402

if not hasattr(_retrieval_db, "pack_dirs"):          # refs < 1.6.0
    pytest.skip("geotech-references without local packs (< 1.6.0)",
                allow_module_level=True)

from geotech_references import _packs  # noqa: E402

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None

PID = "local_example_manual"
TITLE = "Example Practice Manual"
PDF_NAME = "example practice manual.pdf"
BBOX = [72.0, 100.0, 400.0, 400.0]
NOTES = "The x axis is logarithmic; curves are labelled by fill height."


def _section(sid, title, body):
    return {"section_id": sid, "title": title, "body": body,
            "key_points": [], "equations": [], "figures": [], "tables": [],
            "applicability": "Made-up test text."}


def make_pack(root: Path) -> Path:
    folder = root / PID
    (folder / "text").mkdir(parents=True)
    (folder / "pack.json").write_text(json.dumps({
        "id": PID, "reference_id": "EPM-01", "reference_title": TITLE,
        "edition": "2026 (made up)", "pdf_files": [PDF_NAME],
        "prepared_by": "test", "prepared_on": "2026-11-09",
        "method": "printed", "distribution": "test fixture"}),
        encoding="utf-8")
    (folder / "text" / "chapter01.json").write_text(json.dumps({
        "reference_id": "EPM-01", "reference_title": TITLE, "volume": None,
        "chapter": 1, "chapter_title": "Zorblite Embankments",
        "sections": [
            _section("1.1", "Scope", "[p. 1-1] Embankments of zorblite."),
            _section("1.2", "Zorblite Settlement",
                     "[p. 1-2] Zorblite settles; see Figure 1-1."),
        ]}), encoding="utf-8")
    (folder / "figures_catalog.json").write_text(json.dumps({
        "reference_id": "EPM-01", "reference_title": TITLE, "package": PID,
        "volume": None, "pdf_path": PDF_NAME,
        "figures": [
            {"figure_number": "1-1", "caption": "Zorblite settlement chart",
             "chapter": 1, "pdf_page_index": 0, "printed_page": 2,
             "page_estimated": False, "description": "Zorblite settlement.",
             "bbox": BBOX, "read_notes": NOTES},
            {"figure_number": "1-2", "caption": "Wet zorblite settlement",
             "chapter": 1, "pdf_page_index": 0, "printed_page": 3,
             "page_estimated": False, "description": "Wet zorblite."},
        ]}), encoding="utf-8")
    if pymupdf is not None:
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 80), "Figure 1-1 Zorblite settlement chart")
        page.insert_text((72, 500), "Figure 1-2 Wet zorblite settlement")
        doc.save(str(folder / PDF_NAME))
        doc.close()
    return folder


@pytest.fixture
def no_packs(monkeypatch):
    monkeypatch.delenv(_packs.PACKS_ENV, raising=False)


@pytest.fixture(scope="module")
def pack_root(tmp_path_factory):
    """A pack set, with the search databases in a private temp folder."""
    root = tmp_path_factory.mktemp("packs")
    make_pack(root)
    dbdir = tmp_path_factory.mktemp("db")
    old_env = os.environ.get(_packs.PACKS_ENV)
    old_tmp = tempfile.tempdir
    tempfile.tempdir = str(dbdir)
    try:
        yield root
    finally:
        tempfile.tempdir = old_tmp
        if old_env is None:
            os.environ.pop(_packs.PACKS_ENV, None)
        else:
            os.environ[_packs.PACKS_ENV] = old_env


@pytest.fixture
def with_packs(pack_root, monkeypatch):
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    return pack_root


# ---------------------------------------------------------------------------
# read_reference_figure: bbox to the render, read_notes to the prompt
# ---------------------------------------------------------------------------

class _FakeVision:
    """Captures what the reader renders and asks; no model."""

    def __init__(self, monkeypatch):
        from aec_agent_core import vision_view
        self.renders, self.prompts = [], []
        self.analysis = "Following the 5 m curve, s ~ 0.05 m (made up)."

        def render_view(source, page=0, **kwargs):
            self.renders.append({"source": str(source), "page": page,
                                 **kwargs})
            return b"\x89PNG fake", {
                "page": page, "clip": kwargs.get("bbox") or [0, 0, 612, 792],
                "dpi": 100, "width_px": 800, "height_px": 600}

        def ask_vision(engine, image, prompt, kind):
            self.prompts.append(prompt)
            return self.analysis

        monkeypatch.setattr(vision_view, "render_view", render_view)
        monkeypatch.setattr(vision_view, "with_location",
                            lambda prompt, size: prompt)
        monkeypatch.setattr(vision_view, "view_payload",
                            lambda info, engine: {})
        monkeypatch.setattr(reference_vision._vt, "ask_vision", ask_vision)
        monkeypatch.setattr(reference_vision._vt, "_finish_answer",
                            lambda out, info, tagged_only=True: None)
        monkeypatch.setenv("GEOTECH_CHART_CODE_VOTER", "0")


def _read(reference, figure, prompt="Read s for H = 5 m."):
    return json.loads(reference_vision._dispatch_read_reference_figure_at_budget(
        {"reference": reference, "figure_number": figure, "prompt": prompt},
        object()))


def test_pack_figure_bbox_reaches_render_and_notes_reach_prompt(
        with_packs, monkeypatch):
    fake = _FakeVision(monkeypatch)
    out = _read(PID, "1-1")
    assert "error" not in out, out
    assert out["reference"] == PID and out["figure_number"] == "1-1"
    assert fake.renders[0]["bbox"] == BBOX
    assert Path(fake.renders[0]["source"]).name == PDF_NAME
    prompt = fake.prompts[0]
    assert prompt.endswith(
        f"\n\n{reference_vision.READ_NOTES_HEADING}:\n{NOTES}")


def test_read_notes_capped_in_the_prompt(with_packs, monkeypatch):
    """A catalog never checked may carry longer notes than ``packs check``
    allows: the prompt takes the first READ_NOTES_MAX characters."""
    from geotech_references import _figures_db
    cap = reference_vision.READ_NOTES_MAX
    assert cap == 500
    long_notes = "Read at the curve's centreline; made-up note. " * 60
    assert len(long_notes) > 2000
    real = _figures_db.figure_get
    monkeypatch.setattr(_figures_db, "figure_get", lambda ref, n, **kw: {
        **real(ref, n, **kw), "read_notes": long_notes})
    fake = _FakeVision(monkeypatch)
    out = _read(PID, "1-1")
    assert "error" not in out, out
    head = f"\n\n{reference_vision.READ_NOTES_HEADING}:\n"
    tail = fake.prompts[0].split(head, 1)[1]
    assert tail == long_notes[:cap].rstrip()
    assert len(tail) <= cap


def test_pack_figure_without_bbox_or_notes_renders_whole_page(
        with_packs, monkeypatch):
    fake = _FakeVision(monkeypatch)
    out = _read(PID, "1-2")
    assert "error" not in out, out
    assert "bbox" not in fake.renders[0]
    assert reference_vision.READ_NOTES_HEADING not in fake.prompts[0]


#: dm7_2 Figure 4-12's read prompt as the reader built it before REF-64
#: (the code voter off). A change here changes every built-in chart read.
_BUILTIN_PROMPT = (
    'This image is a rendered page from dm7_2 containing Figure 4-12: '
    '"{caption}".\n\n'
    "Read the requested value(s) off this engineering chart:\n"
    "1. Locate Figure 4-12 on the page; ignore other figures and body text.\n"
    "2. Identify the axes (note any logarithmic scales) and the family of "
    "curves and what parameter distinguishes them.\n"
    "3. For the requested inputs, select the correct curve (interpolating "
    "between curves where needed) and read the value at the right axis "
    "position.\n"
    "4. Report the value(s) clearly, state which curve/axis you used, and "
    "flag that this is a chart read-off estimate.\n\n"
    "Request: Read Kp for phi'=35 deg."
)


@pytest.mark.parametrize("packs_set", [False, True])
def test_builtin_figure_prompt_and_render_unchanged(
        packs_set, pack_root, monkeypatch, tmp_path):
    if packs_set:
        monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    else:
        monkeypatch.delenv(_packs.PACKS_ENV, raising=False)
    from geotech_references import _figures_db
    rec = _figures_db.figure_get("dm7_2", "4-12")
    assert "bbox" not in rec and "read_notes" not in rec
    fake_pdf = tmp_path / "builtin.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(_figures_db, "resolve_pdf",
                        lambda ref, n: (fake_pdf, int(rec["pdf_page_index"])))
    fake = _FakeVision(monkeypatch)
    out = _read("dm7_2", "4-12", prompt="Read Kp for phi'=35 deg.")
    assert "error" not in out, out
    assert fake.prompts[0] == _BUILTIN_PROMPT.format(caption=rec["caption"])
    assert set(fake.renders[0]) == {"source", "page", "allow_jpeg", "engine"}


@pytest.mark.skipif(pymupdf is None, reason="PyMuPDF not installed")
def test_pack_figure_crop_renders_for_real(with_packs, monkeypatch):
    """The real render on the pack's own PDF: the crop is the box (padded)."""
    from aec_agent_core import vision_view
    seen = {}
    real = vision_view.render_view

    def spy(source, page=0, **kwargs):
        img, info = real(source, page=page, **kwargs)
        seen.update(info)
        return img, info

    monkeypatch.setattr(vision_view, "render_view", spy)
    monkeypatch.setattr(reference_vision._vt, "ask_vision",
                        lambda engine, image, prompt, kind: "made up")
    monkeypatch.setenv("GEOTECH_CHART_CODE_VOTER", "0")
    out = _read(PID, "1-1")
    assert "error" not in out, out
    x0, y0, x1, y1 = seen["clip"]
    assert x1 - x0 < 612 and y1 - y0 < 792        # not the whole page
    assert x0 <= BBOX[0] and y0 <= BBOX[1] and x1 >= BBOX[2] and y1 >= BBOX[3]


def test_pack_pdf_missing_falls_back_to_host_fetcher(
        tmp_path, monkeypatch):
    """No PDF in the pack or GEOTECH_REFERENCES_DOCS: the host's fetcher is
    asked by the catalog's bare file name, as for a built-in figure."""
    root = tmp_path / "packs"
    folder = make_pack(root)
    (folder / PDF_NAME).unlink(missing_ok=True)
    monkeypatch.setenv(_packs.PACKS_ENV, str(root))
    monkeypatch.delenv("GEOTECH_REFERENCES_DOCS", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    fetched = tmp_path / "fetched.pdf"
    fetched.write_bytes(b"%PDF-1.4\n")
    asked = []
    from funhouse_agent import reference_docs
    monkeypatch.setattr(reference_docs, "fetch",
                        lambda name: asked.append(name) or str(fetched))
    fake = _FakeVision(monkeypatch)
    out = _read(PID, "1-1")
    assert "error" not in out, out
    assert asked == [PDF_NAME]
    assert fake.renders[0]["source"] == str(fetched)
    assert fake.renders[0]["bbox"] == BBOX


@pytest.mark.skipif(pymupdf is None, reason="PyMuPDF not installed")
def test_pack_pdf_opens_by_the_name_the_read_gives(
        with_packs, monkeypatch, tmp_path):
    """With a working folder bound (as in the web app), the read-off names a
    pack's PDF by its file name, and render_region / read_document open it
    from the pack's folder by that name, as the result's zoom_hint says."""
    from aec_agent_core import vision_tools as vt
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    (tmp_path / "work").mkdir()
    _FakeVision(monkeypatch)
    out = _read(PID, "1-1")
    assert "error" not in out, out
    assert out["source"] == PDF_NAME                  # no host path
    folder = with_packs / PID
    assert str(folder) in reference_packs.pack_folders()
    found = vt.find_readable_file(out["source"])
    assert found and Path(found) == folder / PDF_NAME
    data, _src = vt._resolve_attachment_or_path(out["source"], {})
    assert bytes(data[:5]) == b"%PDF-"


def test_no_pack_folders_without_packs(no_packs):
    """No packs: the read folders are the three they always were (the
    fetched-PDF cache, the library's docs, a checkout's docs)."""
    assert reference_packs.pack_folders() == []
    assert len(reference_vision.reference_read_roots()) == 3


# ---------------------------------------------------------------------------
# The packs line and the build record
# ---------------------------------------------------------------------------

def test_packs_line_only_with_packs(no_packs, pack_root, monkeypatch):
    assert reference_packs.packs_line() == ""
    assert reference_packs.pack_records() == []
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    assert reference_packs.packs_line() == (
        f"This deployment adds: {TITLE} (`{PID}`).")


def test_references_spec_carries_line_only_with_packs(
        no_packs, pack_root, monkeypatch):
    from funhouse_agent.deep.agent import (
        _REFERENCES_DESCRIPTION, build_references_subagent)
    plain = build_references_subagent()
    assert plain["description"] == _REFERENCES_DESCRIPTION
    assert "This deployment adds" not in plain["system_prompt"]
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    spec = build_references_subagent()
    line = f"This deployment adds: {TITLE} (`{PID}`)."
    assert spec["description"] == _REFERENCES_DESCRIPTION + " " + line
    assert spec["system_prompt"] == plain["system_prompt"] + "\n\n" + line


def test_no_packs_on_older_refs(pack_root, monkeypatch):
    """refs 1.5.0 has no ``pack_dirs``: no line, no records, no problems."""
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    monkeypatch.delattr(_retrieval_db, "pack_dirs")
    assert reference_packs.packs_line() == ""
    assert reference_packs.pack_records() == []
    assert reference_packs.pack_problems() == []


class _Agent:
    pass


def test_reference_packs_in_build_record_only_with_packs(
        no_packs, pack_root, monkeypatch):
    from funhouse_agent.deep import agent as deep_agent
    a = _Agent()
    a.geotech_build = {"tools": []}
    deep_agent._record_reference_packs(a)
    assert "reference_packs" not in a.geotech_build
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    deep_agent._record_reference_packs(a)
    (rec,) = a.geotech_build["reference_packs"]
    assert rec["id"] == PID and rec["title"] == TITLE
    assert "text/chapter01.json" in rec["files"]
    assert len(rec["sha"]) == 64


def test_reference_packs_through_build_deep_agent(
        no_packs, pack_root, monkeypatch):
    """The real build: with packs, ``reference_packs`` is in
    ``geotech_build`` and in the host's engine record (what each
    ``turn_start`` writes); without, it is absent. Guards the record against
    a move of ``record_build`` (app split step 11 (c))."""
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel)
    from langchain_core.messages import AIMessage
    from funhouse_agent.deep.agent import build_deep_agent
    from webapp import core

    def build():
        return build_deep_agent(model=GenericFakeChatModel(
            messages=iter([AIMessage(content="ok")])))

    plain = build()
    assert isinstance(plain.geotech_build, dict) and plain.geotech_build
    assert "reference_packs" not in plain.geotech_build
    assert "reference_packs" not in core.engine_record(plain).get("build", {})
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    agent = build()
    (rec,) = agent.geotech_build["reference_packs"]
    assert rec["id"] == PID and rec["title"] == TITLE and len(rec["sha"]) == 64
    assert core.engine_record(agent)["build"]["reference_packs"] == [rec]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_list_a_pack_pdf(no_packs, pack_root, monkeypatch):
    from webapp import geotech_page
    assert PDF_NAME not in geotech_page.reference_pdf_names()
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    names = geotech_page.reference_pdf_names()
    assert names[PDF_NAME] == [PID]


def test_diagnostics_find_a_pack_pdf_in_its_folder(
        tmp_path, pack_root, monkeypatch):
    from webapp import geotech_page
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    monkeypatch.setenv(geotech_page.REFERENCE_DOCS_ENV, str(tmp_path))
    monkeypatch.setattr(geotech_page, "reference_pdf_names",
                        lambda: {PDF_NAME: [PID], "builtin.pdf": ["dm7_2"]})
    from funhouse_agent import reference_docs
    monkeypatch.setattr(reference_docs, "cache_dir",
                        lambda: str(tmp_path / "no_cache"))
    c = geotech_page.reference_docs_check()
    if pymupdf is not None:   # the fixture wrote the pack's PDF
        assert "1 of 2 PDFs found" in c["detail"]
        assert PDF_NAME not in c["detail"].split("Missing:")[1]
    assert "builtin.pdf" in c["detail"]


def test_diagnostics_name_a_skipped_pack(tmp_path, monkeypatch):
    from webapp import geotech_page
    bad = tmp_path / "packs" / "no_prefix"
    bad.mkdir(parents=True)
    (bad / "pack.json").write_text(json.dumps(
        {"id": "no_prefix", "reference_id": "X", "reference_title": "X"}),
        encoding="utf-8")
    monkeypatch.setenv(_packs.PACKS_ENV, str(tmp_path / "packs"))
    monkeypatch.setenv(geotech_page.REFERENCE_DOCS_ENV, str(tmp_path))
    monkeypatch.setattr(geotech_page, "reference_pdf_names",
                        lambda: {"builtin.pdf": ["dm7_2"]})
    c = geotech_page.reference_docs_check()
    assert "reference pack skipped:" in c["detail"]
    assert "no_prefix" in c["detail"]


_OLD_REFS_NOTE = "reference packs need geotech-references 1.6.0 or later; installed"


@pytest.mark.parametrize("docs_set", [True, False])
def test_diagnostics_say_packs_need_a_newer_refs(
        docs_set, pack_root, tmp_path, monkeypatch):
    """GEOTECH_REFERENCES_PACKS set on a refs without packs (as 1.5.0; a
    Foundry wheel set goes in with --no-deps): the row says so plainly,
    whether or not the reference PDFs are found."""
    from webapp import geotech_page
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))
    monkeypatch.delattr(_retrieval_db, "pack_dirs")
    monkeypatch.setattr(reference_packs, "refs_version", lambda: "1.5.0")
    if docs_set:
        monkeypatch.setenv(geotech_page.REFERENCE_DOCS_ENV, str(tmp_path))
    else:
        monkeypatch.delenv(geotech_page.REFERENCE_DOCS_ENV, raising=False)
        from geotech_references import _figures_db
        monkeypatch.setattr(_figures_db, "_REPO_ROOT", tmp_path / "no_repo")
    monkeypatch.setattr(geotech_page, "reference_pdf_names",
                        lambda: {"builtin.pdf": ["dm7_2"]})
    from funhouse_agent import reference_docs
    monkeypatch.setattr(reference_docs, "cache_dir",
                        lambda: str(tmp_path / "no_cache"))
    c = geotech_page.reference_docs_check()
    assert f"{_OLD_REFS_NOTE} 1.5.0" in c["detail"]


def test_diagnostics_no_refs_note_without_the_setting_or_with_packs(
        pack_root, tmp_path, monkeypatch):
    from webapp import geotech_page
    monkeypatch.setenv(geotech_page.REFERENCE_DOCS_ENV, str(tmp_path))
    monkeypatch.setattr(geotech_page, "reference_pdf_names",
                        lambda: {"builtin.pdf": ["dm7_2"]})
    monkeypatch.delenv(_packs.PACKS_ENV, raising=False)
    assert reference_packs.unsupported_note() == ""
    assert _OLD_REFS_NOTE not in geotech_page.reference_docs_check()["detail"]
    monkeypatch.setenv(_packs.PACKS_ENV, str(pack_root))   # refs 1.6.0 here
    assert reference_packs.unsupported_note() == ""
    assert _OLD_REFS_NOTE not in geotech_page.reference_docs_check()["detail"]
