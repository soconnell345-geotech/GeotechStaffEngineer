"""
The kriging map and the plan-map path it shares with the probability map
(GM-96): plans drawn as drawings (owner, 2026-10-15) -- to scale, no axes,
a scale bar and a north arrow, the property line dash-dot, the observations
named, every quantity named with its unit -- and the option to hold the
nugget while the variogram is fitted.
"""

import json

import numpy as np
import pytest

from gstools_agent.gstools_utils import has_gstools

pytestmark = pytest.mark.skipif(not has_gstools(),
                                reason="gstools not installed")

from gstools_agent.tests.test_conditional_simulation import (  # noqa: E402
    BX, BY, BZ, FOOT, NAMES,
)

PNG = b"\x89PNG\r\n\x1a\n"
PROP = [[-25, -10], [85, -10], [85, 62], [-25, 62]]
OUTLINES = [{"name": "Property line", "polygon": PROP,
             "kind": "property_line"},
            {"name": "Retaining wall", "line": [[-20, 58], [80, 58]]}]


def _krige(**kw):
    from gstools_agent import analyze_kriging
    args = dict(model_type="Exponential", fit_variogram=False, variance=2.0,
                len_scale=40.0, grid_x_min=-25, grid_x_max=85,
                grid_y_min=-10, grid_y_max=62, n_grid_x=56, n_grid_y=37,
                regions=[{"name": "Building footprint", "polygon": FOOT,
                          "level": 97.0}])
    args.update(kw)
    return analyze_kriging(BX, BY, BZ, **args)


def _traces(pfig, name):
    return [t for t in pfig.data if t.name == name]


class TestKrigingMap:

    def test_mean_and_std_side_by_side_as_a_plan(self):
        from gstools_agent.plan_map import kriging_map_figure
        png, pfig, meta = kriging_map_figure(
            _krige(), names=NAMES, outlines=OUTLINES,
            value_name="top of rock elevation", value_unit="m",
            north_deg=20)
        assert png[:8] == PNG
        assert meta == {"value_name": "top of rock elevation",
                        "north_note": None, "observations_beyond_map": 0}
        lay = json.loads(pfig.to_json())["layout"]
        # two plans, each to scale (1 m across = 1 m up), no axes
        for sx in ("", "2"):
            assert lay[f"yaxis{sx}"]["scaleanchor"] == f"x{sx}"
            assert lay[f"xaxis{sx}"]["visible"] is False
        contours = [t for t in pfig.data if t.type == "contour"]
        assert len(contours) == 2
        assert all(t.contours.showlabels for t in contours)
        titles = [a.text for a in pfig.layout.annotations]
        assert "Kriged mean (m)" in titles
        assert "Kriging standard deviation (m)" in titles
        assert "Kriging of the top of rock elevation" in pfig.layout.title.text
        obs = _traces(pfig, "Observations")
        assert len(obs) == 2 and list(obs[0].text) == NAMES
        assert "top of rock elevation 96 m" in obs[0].hovertext[0]
        prop = _traces(pfig, "Property line")
        assert prop and prop[0].line.dash == "dashdot"
        assert _traces(pfig, "Retaining wall")[0].line.dash == "solid"
        assert _traces(pfig, "Building footprint")[0].line.dash == "solid"
        # one legend entry per thing drawn on both plans
        assert sum(bool(t.showlegend) for t in prop) == 1
        # the scale bar and the north arrow sit on the second plan
        refs = {(s.xref, s.yref) for s in pfig.layout.shapes}
        assert refs == {("x2", "y2")}
        caption = [a.text for a in pfig.layout.annotations
                   if "covariance model" in (a.text or "")]
        assert caption and "correlation length 40 m" in caption[0]
        assert "nugget 0 m²" in caption[0]

    def test_unnamed_observations_are_marked_not_named(self):
        from gstools_agent.plan_map import kriging_map_figure
        _png, pfig, meta = kriging_map_figure(_krige())
        obs = _traces(pfig, "Observations")[0]
        assert set(obs.text) == {""}
        assert obs.hovertext[0].startswith("value 96")
        assert meta["value_name"] == "value"
        assert "north_deg" in meta["north_note"]

    def test_a_long_site_stacks_its_plans(self):
        from gstools_agent import analyze_kriging
        from gstools_agent.plan_map import kriging_map_figure
        k = analyze_kriging([0, 40, 80, 120, 160, 200], [0, 8, -4, 6, 2, 10],
                            [12.0, 10.5, 9.8, 11.2, 13.0, 12.1],
                            model_type="Gaussian", fit_variogram=False,
                            variance=1.5, len_scale=60.0, n_grid_x=40,
                            n_grid_y=10)
        png, pfig, _meta = kriging_map_figure(k)
        assert png[:8] == PNG
        lay = json.loads(pfig.to_json())["layout"]
        # one above the other: the second plan's x domain is the first's
        assert lay["xaxis"]["domain"] == lay["xaxis2"]["domain"]
        assert lay["yaxis2"]["domain"][1] < lay["yaxis"]["domain"][0]

    def test_a_grid_of_one_line_has_no_map(self):
        from gstools_agent.plan_map import kriging_map_figure
        k = _krige(grid_y_min=20, grid_y_max=20, n_grid_y=1, regions=None)
        with pytest.raises(ValueError, match="2 x 2"):
            kriging_map_figure(k)

    def test_names_must_match_the_observations(self):
        from gstools_agent.plan_map import kriging_map_figure
        with pytest.raises(ValueError, match="one name per observation"):
            kriging_map_figure(_krige(), names=NAMES[:3])

    @pytest.mark.parametrize("bad, match", [
        ([{"name": "x", "polygon": [[0, 0], [1, 1]]}], "3 points"),
        ([{"name": "x", "line": [[0, 0], [1, 1]], "kind": "fence"}],
         "kind must be"),
        ([{"name": "x", "polygon": PROP, "colour": "red"}], "unknown key"),
    ])
    def test_outlines_are_checked(self, bad, match):
        from gstools_agent.plan_map import parse_outlines
        with pytest.raises(ValueError, match=match):
            parse_outlines(bad)

    def test_contour_levels_are_round(self):
        from gstools_agent.plan_map import contour_levels
        levels, step = contour_levels(np.array([[94.53, 99.71]]))
        assert step == 1.0 and levels[0] == 94.0 and levels[-1] == 100.0
        flat, step0 = contour_levels(np.full((3, 3), 5.0))
        assert step0 == 0.0 and len(flat) == 2


class TestProbabilityMapOutlines:

    def test_the_property_line_is_drawn_on_the_probability_map(self):
        from gstools_agent import analyze_conditional_simulation
        from gstools_agent.conditional_figures import probability_map_figure
        r = analyze_conditional_simulation(
            BX, BY, BZ, [{"name": "b", "polygon": FOOT, "level": 97.0}],
            model_type="Exponential", variance=2.0, len_scale=40.0,
            cell_size=5.0, n_realizations=20, seed=3, names=NAMES)
        png, pfig, meta = probability_map_figure(
            r, outlines=OUTLINES[:1], value_name="top of rock elevation",
            value_unit="m")
        assert png[:8] == PNG and meta["region"] == "b"
        assert _traces(pfig, "Property line")[0].line.dash == "dashdot"
        heat = [t for t in pfig.data if t.type == "heatmap"][0]
        assert (heat.zmin, heat.zmax) == (0.0, 1.0)
        # the view takes in the property line, which is wider than the cells
        lay = json.loads(pfig.to_json())["layout"]
        assert lay["xaxis"]["range"][0] < -25 and lay["xaxis"]["range"][1] > 85


class TestNugget:
    """fit_variogram fits the nugget by default (unchanged); fit_nugget=false
    holds it at the value given -- GSTools' fit_variogram(nugget=False)."""

    def _data(self):
        rng = np.random.RandomState(42)
        x, y = rng.uniform(0, 100, 30), rng.uniform(0, 100, 30)
        return x, y, 10 + 0.1 * x + rng.randn(30) * 3

    def test_the_nugget_is_held_when_asked(self):
        from gstools_agent import analyze_kriging
        x, y, v = self._data()
        held = analyze_kriging(x, y, v, model_type="Exponential", nugget=0.4,
                               fit_nugget=False, n_grid_x=5, n_grid_y=5)
        assert held.nugget == pytest.approx(0.4)
        assert held.variogram_fitted and not held.nugget_fitted
        assert any("nugget held at 0.4" in n for n in held.notes)
        fitted = analyze_kriging(x, y, v, model_type="Exponential",
                                 nugget=0.4, n_grid_x=5, n_grid_y=5)
        assert fitted.nugget_fitted
        assert fitted.nugget != pytest.approx(0.4)

    def test_the_caption_says_the_nugget_was_held(self):
        from gstools_agent import analyze_kriging
        from gstools_agent.plan_map import kriging_map_figure
        x, y, v = self._data()
        k = analyze_kriging(x, y, v, model_type="Exponential", nugget=0.4,
                            fit_nugget=False, n_grid_x=8, n_grid_y=8)
        _png, pfig, _meta = kriging_map_figure(k)
        text = " ".join(a.text or "" for a in pfig.layout.annotations)
        assert "fitted to the empirical variogram with the nugget held" \
            in text


class TestAdapter:

    BASE = dict(x=BX, y=BY, values=BZ, fit_variogram=False, variance=2.0,
                len_scale=40.0, n_grid_x=12, n_grid_y=10, names=NAMES,
                value_name="top of rock elevation", value_unit="m",
                outlines=OUTLINES[:1], output_prefix="km")

    def test_the_map_lands_in_the_working_folder(self, tmp_path,
                                                 monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "kriging", dict(self.BASE))
        assert "error" not in r, r.get("error")
        assert (tmp_path / "km_kriging_map.png").is_file()
        assert (tmp_path / "km_kriging_map.plotly.json").is_file()
        fig = r["figures"][0]
        assert fig["figure"] == "kriging_map" and fig["file_exists"]
        assert fig["html_img_tag"].startswith("<img src=")
        assert "html_img_tag" in r["embed_note"]
        assert any("north_deg" in n for n in r["notes"])

    def test_figures_false_draws_nothing(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "kriging", dict(self.BASE, figures=False))
        assert "error" not in r and "figures" not in r
        assert not list(tmp_path.iterdir())

    def test_one_point_has_no_map_and_no_error(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "kriging", dict(
            self.BASE, grid_x_min=5, grid_x_max=5, n_grid_x=1,
            grid_y_min=5, grid_y_max=5, n_grid_y=1))
        assert "error" not in r, r.get("error")
        assert "figures" not in r
        assert not any("No kriging map" in n for n in r.get("notes", []))

    def test_a_bad_outline_is_refused_before_the_run(self, tmp_path,
                                                     monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        bad = [{"name": "x", "polygon": [[0, 0]]}]
        for method, extra in (
                ("kriging", {}),
                ("conditional_simulation",
                 {"regions": [{"name": "b", "polygon": FOOT,
                               "level": 97.0}], "n_realizations": 10})):
            params = dict(self.BASE, outlines=bad, **extra)
            if method == "conditional_simulation":
                params.pop("fit_variogram")
                params.pop("n_grid_x")
                params.pop("n_grid_y")
            r = call_agent("gstools", method, params)
            assert "outlines[0]" in r.get("error", ""), (method, r)

    def test_fit_nugget_through_the_adapter(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        rng = np.random.RandomState(42)
        x, y = rng.uniform(0, 100, 30), rng.uniform(0, 100, 30)
        v = 10 + 0.1 * x + rng.randn(30) * 3
        r = call_agent("gstools", "kriging", dict(
            x=x.tolist(), y=y.tolist(), values=v.tolist(),
            model_type="Exponential", nugget=0.4, fit_nugget=False,
            n_grid_x=5, n_grid_y=5, figures=False))
        assert "error" not in r, r.get("error")
        assert r["nugget"] == pytest.approx(0.4)
