"""Tests for calc_package.interactive (Phase 5 stretch, calc-viz)."""

import builtins

import pytest


@pytest.fixture(scope="module")
def slope_setup():
    from slope_stability import (
        SlopeGeometry, SlopeSoilLayer, Anchor, analyze_slope,
        search_critical_surface,
    )
    geom = SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Fill", top_elevation=20,
                           bottom_elevation=12, gamma=19, phi=28,
                           c_prime=5),
            SlopeSoilLayer("Clay", top_elevation=12,
                           bottom_elevation=0, gamma=18, phi=22,
                           c_prime=12),
        ],
        gwt_points=[(0, 16), (30, 9), (50, 9)],
        anchors=[Anchor(x_head=24, z_head=13, length=12, T_allow=80)],
    )
    result = analyze_slope(geom, xc=22, yc=28, radius=20,
                           method="spencer", include_slice_data=True)
    search = search_critical_surface(geom, nx=3, ny=3,
                                     method="bishop", n_slices=20)
    return result, geom, search


@pytest.fixture(scope="module")
def srm_result():
    from fem2d import analyze_slope_srm
    layers = [dict(name="Soft clay fill", bottom_elevation=-6,
                   E=20000, nu=0.3, c=8, phi=15, gamma=19)]
    return analyze_slope_srm([(0, 10), (10, 10), (22, 4), (32, 4)],
                             layers, nx=10, ny=5, srf_tol=0.15)


class TestSlopeViewer:
    def test_file_offline_viewable(self, slope_setup, tmp_path):
        pytest.importorskip("plotly")
        from calc_package.interactive import save_interactive_report
        result, geom, search = slope_setup
        path = save_interactive_report(
            result, tmp_path / "slope.html", geom=geom, search=search)
        html = open(path, encoding="utf-8").read()
        # plotly.js embedded inline -> no network needed
        assert "plotly.js v" in html
        assert 'src="https://cdn.plot.ly' not in html
        assert len(html) > 1_000_000  # plotly.js is inline
        # expected traces
        for trace in ["Slip surface", "Line of thrust",
                      "Trial surfaces", "Slices", "Ground surface",
                      "GWT", "Anchors"]:
            assert trace in html, trace
        # slice hover payload carries the rigorous forces
        assert "S_mob" in html
        assert "E = " in html

    def test_geom_required(self, slope_setup, tmp_path):
        pytest.importorskip("plotly")
        from calc_package.interactive import save_interactive_report
        result, _, _ = slope_setup
        with pytest.raises(ValueError, match="geom"):
            save_interactive_report(result, tmp_path / "x.html")


class TestFemViewer:
    def test_file_with_controls(self, srm_result, tmp_path):
        pytest.importorskip("plotly")
        from calc_package.interactive import save_interactive_report
        path = save_interactive_report(srm_result,
                                       tmp_path / "fem.html")
        html = open(path, encoding="utf-8").read()
        assert "plotly.js v" in html
        # contour dropdown + toggles + SRF subplot
        assert "updatemenus" in html
        for trace in ["|u| (m)", "sigma_yy (kPa)", "Deformed mesh",
                      "Mesh", "Plastic points", "SRF trials"]:
            assert trace in html, trace

    def test_unsupported_type_raises(self, tmp_path):
        pytest.importorskip("plotly")
        from calc_package.interactive import save_interactive_report
        with pytest.raises(TypeError, match="Unsupported"):
            save_interactive_report(object(), tmp_path / "x.html")


class TestNoPlotlyDegradation:
    def test_clear_import_error(self, slope_setup, tmp_path,
                                monkeypatch):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "plotly" or name.startswith("plotly."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded)
        from calc_package.interactive import save_interactive_report
        result, geom, _ = slope_setup
        with pytest.raises(ImportError, match="plotly is required"):
            save_interactive_report(result, tmp_path / "x.html",
                                    geom=geom)

