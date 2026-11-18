"""Brief 14's wiring, the points its review repaired (offline).

* A host without Pillow is not told a DXF is also a plot: planlens refuses
  every plot there (``PlotUnavailable``), so the page stays the build it
  was.
* A DXF that will not plot is plotted once per build, not again at every
  look.

Every drawing here is made up in the test.
"""

import importlib.util
import json

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("pymupdf")

from aec_agent_core import document_tools  # noqa: E402
from aec_agent_core.deep import vision_kit  # noqa: E402
from tests.test_review_harness_offline import FakeEngine  # noqa: E402

DXF = "site.dxf"
needs_dxf_plots = pytest.mark.skipif(
    not document_tools.dxf_plots_available(),
    reason="the installed planlens plots no DXF (0.16.0)")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    from planlens.review import conversation
    conversation._SPACES.clear()


@pytest.fixture(scope="module")
def dxf(tmp_path_factory):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (40, 0), (40, 20), (0, 20)], close=True)
    msp.add_text("BAY 3", height=2.0).set_placement((4, 12))
    path = tmp_path_factory.mktemp("b14rv") / DXF
    doc.saveas(str(path))
    return path.read_bytes()


def _kit(attachments, **kw):
    return {t.name: t for t in vision_kit.make_vision_tools(
        engine=FakeEngine(), attachments=attachments, **kw)}


@needs_dxf_plots
def test_without_pillow_a_dxf_is_not_offered_as_a_plot(dxf, monkeypatch):
    real = importlib.util.find_spec

    def no_pillow(name, *args, **kwargs):
        return None if name == "PIL" else real(name, *args, **kwargs)

    plain = _kit({DXF: dxf})
    monkeypatch.setattr(importlib.util, "find_spec", no_pillow)
    assert not document_tools.dxf_plots_available()
    options = document_tools.host_for(dxf_plots=True).toolkit_options or {}
    assert "dxf_plots" not in options
    asked = _kit({DXF: dxf}, dxf_plots=True)
    # the page does not say a DXF is a plot where none can be made
    assert asked["open_document"].description == \
        plain["open_document"].description
    opened = json.loads(asked["open_document"].invoke({"source": DXF}))
    assert "plot" not in opened
