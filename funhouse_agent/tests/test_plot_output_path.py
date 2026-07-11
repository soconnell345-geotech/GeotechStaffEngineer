"""Tests for the plot `output_path` feature — save the figure HTML to a file
instead of returning a ~MB HTML blob inline.

Covers the shared adapter helpers (figure_output_format, save_html_output) and
the end-to-end wiring on the subsurface plot methods (all 5 route through the
same two helpers). Default behavior (no output_path) must be byte-identical.
"""

import json
import os

import pytest

from funhouse_agent.adapters import figure_output_format, save_html_output
from funhouse_agent.dispatch import call_agent


# ---------------------------------------------------------------------------
# Shared helpers (no plotly / site needed)
# ---------------------------------------------------------------------------

class TestFigureOutputFormat:
    def test_output_path_forces_html(self):
        assert figure_output_format({"output_path": "/tmp/x.html"}) == "html"

    def test_output_path_overrides_explicit_format(self):
        # output_path means "save the figure" — the HTML must be produced even
        # if the caller left output_format at metadata.
        assert figure_output_format(
            {"output_path": "/tmp/x.html", "output_format": "metadata"}) == "html"

    def test_no_output_path_uses_explicit_format(self):
        assert figure_output_format({"output_format": "json"}) == "json"

    def test_default_is_metadata(self):
        assert figure_output_format({}) == "metadata"


class TestSaveHtmlOutput:
    def test_saves_and_replaces_blob(self, tmp_path):
        out = tmp_path / "fig.html"
        big = "<html>" + "z" * 5000 + "</html>"
        res = save_html_output(
            {"plot_type": "scatter", "title": "T", "html": big}, {"output_path": str(out)})
        assert "html" not in res                       # blob dropped
        assert res["file_exists"] is True
        assert res["file_size_bytes"] == out.stat().st_size > 0
        assert os.path.isabs(res["output_path"])        # calc_package-shaped
        assert "saved" not in res
        assert "renderer_note" in res
        assert res["plot_type"] == "scatter"           # metadata retained
        assert out.read_text().startswith("<html>")

    def test_noop_without_output_path(self):
        res = save_html_output({"html": "X", "plot_type": "p"}, {})
        assert res["html"] == "X"                       # unchanged
        assert "renderer_note" not in res

    def test_noop_when_no_html_key(self, tmp_path):
        # metadata-only result + output_path: nothing to save, returned as-is.
        res = save_html_output(
            {"plot_type": "p", "n_data_points": 3}, {"output_path": str(tmp_path / "x.html")})
        assert "renderer_note" not in res
        assert res["n_data_points"] == 3

    def test_custom_html_key(self, tmp_path):
        out = tmp_path / "f.html"
        res = save_html_output(
            {"figure_html": "<html>Y</html>"}, {"output_path": str(out)},
            html_key="figure_html")
        assert "figure_html" not in res
        assert res["file_exists"] is True

    def test_figure_json_writes_plotly_sidecar(self, tmp_path):
        out = tmp_path / "fig.html"
        res = save_html_output(
            {"html": "<html>FIG</html>"}, {"output_path": str(out)},
            figure_json='{"data": [], "layout": {}}')
        assert res["file_exists"] is True
        sidecar = tmp_path / "fig.plotly.json"
        assert res["plotly_json_path"] == str(sidecar) or \
            os.path.abspath(res["plotly_json_path"]) == str(sidecar)
        assert sidecar.read_text() == '{"data": [], "layout": {}}'

    def test_no_figure_json_no_sidecar(self, tmp_path):
        out = tmp_path / "fig.html"
        res = save_html_output({"html": "<html>FIG</html>"},
                               {"output_path": str(out)})
        assert res["file_exists"] is True
        assert "plotly_json_path" not in res
        assert not (tmp_path / "fig.plotly.json").exists()

    def test_save_failure_surfaces_rescue(self, tmp_path, monkeypatch):
        import funhouse_agent._fileio as fio

        def bad_verified(path, content):
            return {"error": "target did not store the content",
                    "rescue_path": str(tmp_path / "rescue.html"),
                    "file_exists": False}

        monkeypatch.setattr(fio, "save_verified", bad_verified)
        res = save_html_output({"html": "<html>Z</html>"},
                               {"output_path": "/Workspace/x/f.html"})
        assert res["file_exists"] is False
        assert "rescue_path" in res
        assert "rescue" in res["renderer_note"].lower()


# ---------------------------------------------------------------------------
# End-to-end on the subsurface plot methods (needs plotly for html rendering)
# ---------------------------------------------------------------------------

_SITE = {
    "project_name": "OutputPath Test",
    "investigations": [
        {"investigation_id": "B-1", "investigation_type": "boring",
         "x": 0, "y": 0, "elevation_m": 10, "total_depth_m": 15,
         "measurements": [
             {"depth_m": 1.5, "parameter": "N_spt", "value": 8},
             {"depth_m": 4.5, "parameter": "N_spt", "value": 12},
             {"depth_m": 1.5, "parameter": "wn_pct", "value": 22},
             {"depth_m": 4.5, "parameter": "wn_pct", "value": 26},
         ], "lithology": []},
        {"investigation_id": "B-2", "investigation_type": "boring",
         "x": 10, "y": 0, "elevation_m": 11, "total_depth_m": 15,
         "measurements": [
             {"depth_m": 2.0, "parameter": "N_spt", "value": 10},
             {"depth_m": 5.0, "parameter": "N_spt", "value": 15},
         ], "lithology": []},
    ],
}


@pytest.fixture
def site_key():
    pytest.importorskip("plotly")
    res = call_agent("subsurface", "load_site", {"site_data": _SITE})
    assert "site_key" in res, res
    return res["site_key"]


_PLOT_CASES = [
    ("plot_parameter_vs_depth", {"parameter": "N_spt"}),
    ("plot_multi_parameter", {"parameters": ["N_spt", "wn_pct"]}),
    ("plot_plan_view", {}),
    ("plot_cross_section", {"investigation_ids": ["B-1", "B-2"]}),
]


@pytest.mark.parametrize("method,extra", _PLOT_CASES)
def test_plot_output_path_saves_file_and_drops_html(method, extra, site_key, tmp_path):
    out = tmp_path / f"{method}.html"
    params = {"site_key": site_key, "output_path": str(out)}
    params.update(extra)
    res = call_agent("subsurface", method, params)
    assert "error" not in res, res
    assert "html" not in res              # the ~MB blob is NOT returned inline
    assert res["file_exists"] is True
    assert res["file_size_bytes"] > 0
    assert os.path.abspath(res["output_path"]) == os.path.abspath(str(out))
    assert "renderer_note" in res
    assert out.is_file()
    # a self-contained Plotly HTML doc
    assert out.read_text(encoding="utf-8").lstrip().lower().startswith("<html")


def test_plot_output_path_emits_plotly_sidecar(site_key, tmp_path):
    """With output_path, the subsurface plot also writes a <name>.plotly.json
    sidecar (for native st.plotly_chart rendering) that round-trips."""
    out = tmp_path / "pvd.html"
    res = call_agent("subsurface", "plot_parameter_vs_depth",
                     {"site_key": site_key, "parameter": "N_spt",
                      "output_path": str(out)})
    assert "error" not in res, res
    sidecar = res.get("plotly_json_path")
    assert sidecar and os.path.isfile(sidecar)
    assert sidecar.endswith(".plotly.json")
    import plotly.io as pio
    fig = pio.from_json(open(sidecar, encoding="utf-8").read())
    assert len(fig.data) >= 1


def test_plot_default_unchanged_no_html_no_output_path(site_key):
    """Default (no output_path, no output_format) is byte-identical: metadata
    only, no figure blob, no save-confirmation keys."""
    res = call_agent("subsurface", "plot_parameter_vs_depth",
                     {"site_key": site_key, "parameter": "N_spt"})
    assert "html" not in res
    assert "output_path" not in res and "file_exists" not in res
    assert "renderer_note" not in res
    assert res["n_data_points"] == 4


def test_plot_output_format_html_still_returns_blob_inline(site_key):
    """Explicit output_format=html WITHOUT output_path keeps the old behavior:
    the HTML is returned inline."""
    res = call_agent("subsurface", "plot_parameter_vs_depth",
                     {"site_key": site_key, "parameter": "N_spt",
                      "output_format": "html"})
    assert "html" in res and res["html"].lstrip().lower().startswith("<html")
    assert "renderer_note" not in res
