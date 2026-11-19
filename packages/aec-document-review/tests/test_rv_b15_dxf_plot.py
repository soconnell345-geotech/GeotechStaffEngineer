"""Review of B15 (``GEOTECH_REVIEW_DXF_PLOTS``, BACKLOG DR-19): what
``open_document`` says of a DXF agrees with itself. Where the plot was made
the note no longer says there are no pages to look at; where the plot is
refused (an empty model space) there is no plot line and the note stands.
The drawings are made up in the test.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("pymupdf")
ezdxf = pytest.importorskip("ezdxf")

from aec_agent_core import document_tools, review_flags  # noqa: E402
from aec_agent_core.deep import vision_kit  # noqa: E402
from tests.test_review_harness_offline import FakeEngine  # noqa: E402

pytestmark = pytest.mark.skipif(
    not document_tools.dxf_plots_available(),
    reason="the installed planlens plots no DXF (0.16.0)")

ASK = "ask the user for a PDF plot"
NO_PAGES = "There are no pages here"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    from planlens.review import conversation
    conversation._SPACES.clear()


def _dxf(tmp_path, name, lines):
    doc = ezdxf.new()
    for a, b in lines:
        doc.modelspace().add_line(a, b)
    path = tmp_path / name
    doc.saveas(str(path))
    return path.read_bytes()


def _open(attachments, name, **kw):
    kit = {t.name: t for t in vision_kit.make_vision_tools(
        engine=FakeEngine(), attachments=attachments, **kw)}
    return json.loads(kit["open_document"].invoke({"source": name}))


def test_a_plotted_dxf_is_not_also_said_to_have_no_pages(tmp_path):
    att = {"shed.dxf": _dxf(tmp_path, "shed.dxf", [
        ((0, 0), (30, 0)), ((30, 0), (30, 12)), ((30, 12), (0, 0))])}
    out = _open(att, "shed.dxf", dxf_plots=True)
    assert "analyze_pdf_page" in out["plot"]
    assert NO_PAGES not in out["note"] and ASK not in out["note"]
    assert "use the plot made here" in out["note"]
    # the rest of planlens' note is kept
    assert "read_document(handle='shed.dxf'" in out["note"]


def test_a_dxf_that_is_not_plotted_keeps_its_note_and_has_no_plot_line(
        tmp_path):
    att = {"none.dxf": _dxf(tmp_path, "none.dxf", [])}
    out = _open(att, "none.dxf", dxf_plots=True)
    assert "plot" not in out
    assert NO_PAGES in out["note"] and ASK in out["note"]


def test_unset_the_note_is_planlens_own(tmp_path):
    att = {"shed.dxf": _dxf(tmp_path, "shed.dxf", [((0, 0), (30, 0))])}
    out = _open(att, "shed.dxf")
    assert "plot" not in out
    assert NO_PAGES in out["note"] and ASK in out["note"]
