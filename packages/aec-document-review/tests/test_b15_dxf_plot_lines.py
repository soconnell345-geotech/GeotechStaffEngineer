"""What the page says of a DXF's plot (``GEOTECH_REVIEW_DXF_PLOTS``, BACKLOG
DR-19), with the setting on: ``open_document``'s plot line names only tools
the build it is in offers, and the page map's row for the plot's page says
it is a plot of the DXF's model space, never ``blank``.

Here, not in the core's tests, because the first half needs the Document
Review page and its reading helper as they are built. The drawing is made
up in the test.
"""

import json
import re

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("pymupdf")

from aec_agent_core import document_tools, review_flags  # noqa: E402
from aec_agent_core.deep import vision_kit  # noqa: E402
from tests.test_review_harness_offline import FakeEngine  # noqa: E402

pytestmark = pytest.mark.skipif(
    not document_tools.dxf_plots_available(),
    reason="the installed planlens plots no DXF (0.16.0)")

DXF = "yard.dxf"
DXF_ENV = "GEOTECH_REVIEW_DXF_PLOTS"
EXTENT = [0.0, 0.0, 48.0, 20.0]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    from planlens.review import conversation
    conversation._SPACES.clear()


@pytest.fixture(scope="module")
def dxf(tmp_path_factory):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (48, 0), (48, 20), (0, 20)], close=True)
    msp.add_line((0, 10), (48, 10))
    msp.add_text("BAY 3", height=2.0).set_placement((5, 14))
    path = tmp_path_factory.mktemp("b15") / DXF
    doc.saveas(str(path))
    return path.read_bytes()


def _kit(attachments, **kw):
    return {t.name: t for t in vision_kit.make_vision_tools(
        engine=FakeEngine(), attachments=attachments, **kw)}


def _call(kit, name, **args):
    return json.loads(kit[name].invoke(args))


def _named(line):
    """The tool names a plot line gives before it says what they do."""
    return re.findall(r"[a-z]+(?:_[a-z]+)+", line.split(" take this file")[0])


# -- (1) the plot line names only tools the build offers --------------------------

def test_the_plot_line_names_only_tools_of_the_kit(dxf):
    from planlens.dxf.plot import PLOT_NOTE
    kit = _kit({DXF: dxf}, dxf_plots=True)
    line = _call(kit, "open_document", source=DXF)["plot"]
    named = _named(line)
    assert named and set(named) <= set(kit), named
    assert "render_page," not in line        # planlens' own tool, not ours
    assert line.endswith(PLOT_NOTE)


def test_the_plot_line_names_only_tools_of_the_page_and_its_reader(dxf):
    from tests.test_b14_wiring_offline import _page
    tools, reader = _page(**{DXF_ENV: "1"})
    line = _call(_kit({DXF: dxf}, dxf_plots=True),
                 "open_document", source=DXF)["plot"]
    named = set(_named(line))
    assert named and named <= set(tools), named - set(tools)
    # the helper is handed the page's own tool objects, so the same line
    assert named <= set(reader), named - set(reader)


def test_a_build_with_fewer_page_tools_is_told_of_those_only(dxf):
    few = _kit({DXF: dxf}, dxf_plots=True,
               include={"open_document", "read_document", "render_region"})
    assert _named(_call(few, "open_document", source=DXF)["plot"]) == [
        "render_region"]
    from planlens.review import conversation
    conversation._SPACES.clear()
    none = _kit({DXF: dxf}, dxf_plots=True,
                include={"open_document", "read_document"})
    assert "plot" not in _call(none, "open_document", source=DXF)


# -- (2) the plot's page is mapped as a plot --------------------------------------

def test_the_page_map_says_the_page_is_a_plot_of_the_model_space(dxf):
    from planlens.dxf.plot import PLOT_NOTE
    kit = _kit({DXF: dxf}, dxf_plots=True)
    mapped = _call(kit, "document_page_map", handle=DXF)
    (row,) = mapped["rows"]
    assert row["page"] == 0 and row["kind"] == "dxf_plot"
    assert row["plot"] == {"of": DXF, "n_entities": 3, "extent": EXTENT,
                           "drawing_units": "m", "what": PLOT_NOTE}
    # by the document handle too
    again = _call(kit, "document_page_map", handle=mapped["handle"])
    assert again["rows"] == mapped["rows"]
    assert "blank" not in json.dumps(mapped)


def test_the_structure_says_so_too(dxf):
    kit = _kit({DXF: dxf}, dxf_plots=True)
    (seg,) = _call(kit, "document_structure", handle=DXF)["segments"]
    assert seg["title"] == "DXF plot" and seg["kinds"] == {"dxf_plot": 1}
    assert seg["plot"]["of"] == DXF and seg["plot"]["extent"] == EXTENT


# -- nothing else moves ------------------------------------------------------------

def test_a_pdf_is_mapped_as_before_and_unset_nothing_is_said(dxf):
    import pymupdf
    pdf = pymupdf.open()
    pdf.new_page()                      # a page with nothing on it
    att = {"empty.pdf": pdf.tobytes(), DXF: dxf}
    rows = []
    for kw in ({}, {"dxf_plots": True}):
        from planlens.review import conversation
        conversation._SPACES.clear()
        got = _call(_kit(att, **kw), "document_page_map", handle="empty.pdf")
        got.pop("handle", None), got.pop("opened", None)
        rows.append(got)
    assert rows[0] == rows[1] and rows[0]["rows"][0]["kind"] == "blank"
    conversation._SPACES.clear()
    unset = _kit(att)
    assert "plot" not in _call(unset, "open_document", source=DXF)
    assert "error" in _call(unset, "document_page_map", handle=DXF)
