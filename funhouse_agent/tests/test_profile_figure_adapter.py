"""Profile-figure adapter — the subsurface schematic tool the agent calls.

Covers the 2026-09-04 field feedback (Praia downdrag): asked for "a figure
showing the pile and a subsurface profile", the agent had no drawing tool at
all and shipped a coloured HTML table instead.
"""

import base64
import json
import os

import pytest

from funhouse_agent.adapters.profile_figure_adapter import (
    METHOD_INFO, METHOD_REGISTRY, _run_subsurface_profile)
from funhouse_agent.dispatch import call_agent, describe_method, list_agents

LAYERS = [
    {"name": "Fill", "thickness": 3.0, "description": "N=5"},
    {"name": "Soft clay", "thickness": 9.0, "settling": True},
    {"name": "Dense sand", "thickness": 8.0},
]


# ---------------------------------------------------------------------------
# Adapter surface
# ---------------------------------------------------------------------------

class TestProfileFigureAdapter:
    def test_saves_a_png_and_echoes_the_geometry(self, tmp_path):
        out = tmp_path / "profile.png"
        res = _run_subsurface_profile({
            "layers": LAYERS, "water_depth": 2.0,
            "title": "Boring B-1", "output_path": str(out)})
        assert res["status"] == "success"
        assert res["file_exists"] is True
        assert res["file_size_bytes"] > 10_000
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert res["width_px"] > 300 and res["height_px"] > 300

        profile = res["profile"]
        assert [lay["name"] for lay in profile["layers"]] == \
            ["Fill", "Soft clay", "Dense sand"]
        assert [lay["top"] for lay in profile["layers"]] == [0.0, -3.0, -12.0]
        assert profile["water_elevation"] == pytest.approx(-2.0)

    def test_response_is_json_safe_and_omits_the_base64_blob(self, tmp_path):
        res = _run_subsurface_profile({"layers": LAYERS,
                                       "output_path": str(tmp_path / "p.png")})
        # The base64 PNG is ~100k chars — it must not land in tool output.
        assert "image_base64" not in res
        assert len(json.dumps(res)) < 8000

    def test_include_base64_returns_the_image(self, tmp_path):
        res = _run_subsurface_profile({"layers": LAYERS, "include_base64": True,
                                       "output_path": str(tmp_path / "p.png")})
        raw = base64.b64decode(res["image_base64"])
        assert raw == (tmp_path / "p.png").read_bytes()

    def test_html_img_tag_points_at_the_saved_file(self, tmp_path):
        out = tmp_path / "p.png"
        res = _run_subsurface_profile({"layers": LAYERS,
                                       "output_path": str(out)})
        assert res["html_img_tag"].startswith('<img src="')
        assert os.path.abspath(str(out)) in res["html_img_tag"]

    def test_extension_is_added_when_missing(self, tmp_path):
        res = _run_subsurface_profile({"layers": LAYERS,
                                       "output_path": str(tmp_path / "fig")})
        assert res["output_path"].endswith(".png")
        assert os.path.isfile(res["output_path"])

    def test_bare_filename_lands_in_the_working_folder(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        res = _run_subsurface_profile({"layers": LAYERS,
                                       "output_path": "section.png"})
        assert res["output_path"] == str(tmp_path / "section.png")
        assert os.path.isfile(res["output_path"])

    def test_warnings_surface_a_tip_below_the_profile(self, tmp_path):
        res = _run_subsurface_profile({
            "layers": LAYERS,
            "foundation": {"type": "pile", "diameter": 0.4, "tip_depth": 30.0},
            "output_path": str(tmp_path / "p.png")})
        assert any("below the deepest layer" in w for w in res["warnings"])

    def test_missing_layers_is_an_actionable_error(self):
        with pytest.raises(ValueError, match="layers"):
            _run_subsurface_profile({"title": "no layers"})

    def test_unknown_param_is_rejected(self):
        with pytest.raises(ValueError, match="subsurface_profile"):
            _run_subsurface_profile({"layers": LAYERS, "colour_scheme": "blue"})

    def test_binary_save_verifies_clean(self, tmp_path):
        """PNG bytes contain \\r\\n (the signature does) — the verified-save
        head check must not newline-normalize them into a false failure."""
        from funhouse_agent._fileio import save_verified
        out = tmp_path / "x.png"
        saved = save_verified(str(out), _tiny_png())
        assert saved["file_exists"] is True
        assert "error" not in saved

    def test_method_info_documents_every_registered_method(self):
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)
        info = METHOD_INFO["subsurface_profile"]
        assert info["parameters"]["layers"]["required"] is True
        assert "output_path" in info["parameters"]


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------

class TestDispatchWiring:
    def test_module_is_in_the_catalog_within_budget(self):
        catalog = list_agents()
        assert "profile_figure" in catalog
        assert len(json.dumps(catalog)) < 8000

    def test_call_agent_runs_the_method(self, tmp_path):
        res = call_agent("profile_figure", "subsurface_profile",
                         {"layers": LAYERS,
                          "output_path": str(tmp_path / "p.png")})
        assert res["status"] == "success"

    def test_module_name_as_method_is_aliased(self, tmp_path):
        res = call_agent("profile_figure", "profile_figure",
                         {"layers": LAYERS,
                          "output_path": str(tmp_path / "p.png")})
        assert res["status"] == "success"

    def test_guess_on_the_wrong_module_is_redirected(self):
        res = call_agent("downdrag", "subsurface_profile", {"layers": LAYERS})
        assert "profile_figure" in res["error"]
        assert "subsurface_profile" in res["error"]

    def test_describe_method_reaches_the_adapter(self):
        info = describe_method("profile_figure", "subsurface_profile")
        assert "layers" in info["parameters"]


def _tiny_png() -> bytes:
    """A 1x1 PNG — enough for the embed path without a plotting round-trip."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
        "hQGAhKmMIQAAAABJRU5ErkJggg==")


# ---------------------------------------------------------------------------
# plot_data — the generic data plot (owner feedback 2026-09-11: calc packages
# lacked figures and the agent had no general plotting tool to make one)
# ---------------------------------------------------------------------------

SERIES = [{"x": [4, 8, 12, 9], "y": [1.5, 3.0, 4.5, 6.0], "label": "B-1 SPT N"},
          {"x": [6, 10, 15, 11], "y": [1.5, 3.0, 4.5, 6.0], "label": "B-2",
           "style": "markers"}]


class TestPlotData:
    def test_saves_a_png_and_echoes_the_series(self, tmp_path):
        from funhouse_agent.adapters.profile_figure_adapter import _run_plot_data
        out = tmp_path / "spt.png"
        res = _run_plot_data({
            "series": SERIES, "depth_axis": True, "title": "SPT vs depth",
            "xlabel": "N (blows/0.3 m)", "ylabel": "Depth (m)",
            "hlines": [{"value": 2.0, "label": "GWT"}],
            "output_path": str(out)})
        assert res["status"] == "success" and res["file_exists"] is True
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert res["file_size_bytes"] > 5_000
        assert [s["label"] for s in res["series"]] == ["B-1 SPT N", "B-2"]
        assert res["series"][0]["n"] == 4
        assert res["series"][0]["y_range"] == [1.5, 6.0]
        assert res["output_path"] in res["html_img_tag"]
        assert "image_base64" not in res
        assert len(json.dumps(res)) < 4000

    def test_bare_filename_lands_in_the_working_folder(self, tmp_path,
                                                       monkeypatch):
        from funhouse_agent.adapters.profile_figure_adapter import _run_plot_data
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        res = _run_plot_data({"series": SERIES[:1], "output_path": "sweep"})
        assert res["output_path"].endswith("sweep.png")
        assert os.path.dirname(res["output_path"]) == str(tmp_path)

    def test_validation_is_actionable(self, tmp_path):
        from funhouse_agent.adapters.profile_figure_adapter import _run_plot_data
        with pytest.raises(ValueError, match="x has 3 values, y has 2"):
            _run_plot_data({"series": [{"x": [1, 2, 3], "y": [1, 2]}],
                            "output_path": str(tmp_path / "a.png")})
        with pytest.raises(ValueError, match="non-empty list"):
            _run_plot_data({"series": [], "output_path": str(tmp_path / "b.png")})
        with pytest.raises(ValueError):                # unknown param
            _run_plot_data({"series": SERIES, "colour": "red",
                            "output_path": str(tmp_path / "c.png")})

    def test_non_finite_points_are_dropped_with_a_warning(self, tmp_path):
        from funhouse_agent.adapters.profile_figure_adapter import _run_plot_data
        res = _run_plot_data({
            "series": [{"x": [1, 2, float("nan"), 4], "y": [1, 2, 3, 4]}],
            "output_path": str(tmp_path / "n.png")})
        assert res["status"] == "success"
        assert res["series"][0]["n"] == 3
        assert any("non-finite" in w for w in res["warnings"])

    def test_html_to_pdf_embeds_the_saved_plot(self, tmp_path):
        """The whole point: a plot_data PNG goes straight into a bespoke
        report via html_img_tag and html_to_pdf does not refuse it."""
        pytest.importorskip("fitz")
        from funhouse_agent.adapters.profile_figure_adapter import _run_plot_data
        from funhouse_agent.adapters.calc_package import _generate_html_to_pdf
        res = _run_plot_data({"series": SERIES,
                              "output_path": str(tmp_path / "p.png")})
        pdf = _generate_html_to_pdf({
            "html": f"<html><body><h1>R</h1>{res['html_img_tag']}</body></html>",
            "output_path": str(tmp_path / "r.pdf")})
        assert pdf["status"] == "success", pdf.get("error")
        assert pdf["file_size_bytes"] > 10_000

    def test_dispatch_aliases_reach_plot_data(self, tmp_path):
        for verb in ("plot_data", "plot_xy", "plot", "data_plot"):
            res = call_agent("profile_figure", verb,
                             {"series": SERIES[:1],
                              "output_path": str(tmp_path / f"{verb}.png")})
            assert res["status"] == "success", (verb, res)
        info = describe_method("profile_figure", "plot_data")
        assert "series" in json.dumps(info)
        assert len(json.dumps(list_agents())) < 8000
