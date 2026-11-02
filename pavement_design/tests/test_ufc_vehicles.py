"""UFC 3-250-01 Figures E-2 to E-31 read off the print (ufc_vehicles).

Anchors: Appendix G Table G-1 (pdf_page 276, printed 257) at CBR 3. The
other values are made up and test the reading rules only.
"""

import math

import pytest

from pavement_design.ufc_vehicles import (_FIGURES,
                                          figure_e_vehicle_allowable_passes,
                                          figure_e_vehicle_thickness,
                                          resolve_vehicle)


def test_table_g1_reads():
    # 5-axle truck 100,000 passes 15.8 in, 3-axle truck 500,000 passes
    # 12.8 in, passenger car 20,000,000 passes 6.1 in (PCASE solutions,
    # which "differ slightly" from the charts, p. 256).
    assert figure_e_vehicle_thickness("E-28", 3, 1e5)["thickness_in"] == \
        pytest.approx(15.8, abs=0.25)
    assert figure_e_vehicle_thickness("E-26", 3, 5e5)["thickness_in"] == \
        pytest.approx(12.8, abs=0.15)
    car = figure_e_vehicle_thickness("E-2", 3, 2e7)
    assert car["thickness_in"] == pytest.approx(6.1, abs=0.05)
    assert "extrapolated" in car["note"]
    assert car["reference"] == \
        "UFC 3-250-01, Figure E-2 (pdf_page 212, printed 193)"


def test_legends_and_frames_as_printed():
    assert len(_FIGURES) == 30
    for fig, data in _FIGURES.items():
        levels = sorted(data["curves"])
        if fig == "E-2":
            assert levels == [10_000, 100_000, 1_000_000, 10_000_000]
        elif fig in ("E-3", "E-12", "E-15"):
            assert levels == [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
        elif fig == "E-31":
            assert levels == [100, 1_000, 10_000, 100_000, 500_000]
        else:
            assert levels == [1_000, 10_000, 100_000, 1_000_000,
                              10_000_000]
    # E-15's frame top is 15 in; the 14 in gridline sits below it.
    assert _FIGURES["E-15"]["y_max_in"] == 15


def test_curves_never_cross_or_rise():
    for fig, data in _FIGURES.items():
        levels = sorted(data["curves"])
        for i in range(60):
            cbr = 10 ** (2 * i / 59)
            last = -1.0
            for p in levels:
                try:
                    t = figure_e_vehicle_thickness(fig, cbr, p)["thickness_in"]
                except ValueError:
                    continue
                assert t >= last - 1e-9, (fig, cbr, p)
                last = t
        for p, pts in data["curves"].items():
            assert all(b[0] > a[0] and b[1] <= a[1]
                       for a, b in zip(pts, pts[1:])), (fig, p)


def test_between_curves_log_passes():
    lo = figure_e_vehicle_thickness("E-27", 6, 1e4)["thickness_in"]
    hi = figure_e_vehicle_thickness("E-27", 6, 1e5)["thickness_in"]
    mid = figure_e_vehicle_thickness("E-27", 6, 10 ** 4.5)["thickness_in"]
    assert mid == pytest.approx((lo + hi) / 2, abs=0.06)


def test_off_the_chart():
    with pytest.raises(ValueError, match="below Figure E-28's CBR axis"):
        figure_e_vehicle_thickness("E-28", 0.9, 1e4)
    # E-16's curves enter at the top frame at CBR 2 and more.
    with pytest.raises(ValueError, match="above the chart"):
        figure_e_vehicle_thickness("E-16", 1.5, 1e3)
    # E-30's curves stop 2 in above the axis.
    with pytest.raises(ValueError, match="ends at CBR"):
        figure_e_vehicle_thickness("E-30", 95, 1e3)
    # Past the point where a curve meets the axis: no cover needed.
    assert figure_e_vehicle_thickness("E-28", 80, 1e3)["thickness_in"] == 0.0
    few = figure_e_vehicle_thickness("E-28", 5, 500)
    assert few["thickness_in"] == \
        figure_e_vehicle_thickness("E-28", 5, 1e3)["thickness_in"]
    assert "below" in few["note"]


def test_allowable_passes_reverses_the_chart():
    t = figure_e_vehicle_thickness("E-27", 5, 3e4)["thickness_in"]
    r = figure_e_vehicle_allowable_passes("E-27", 5, t)
    assert "clamped" not in r
    assert math.log10(r["allowable_passes"]) == pytest.approx(
        math.log10(3e4), abs=0.05)
    above = figure_e_vehicle_allowable_passes("E-26", 3, 16.4)
    assert above["clamped"] == "above_highest_curve"
    below = figure_e_vehicle_allowable_passes("E-26", 3, 1.0)
    assert below["clamped"] == "below_lowest_curve"
    # Every E-16 curve is above the chart at CBR 1.5.
    assert figure_e_vehicle_allowable_passes(
        "E-16", 1.5, 45)["clamped"] == "below_lowest_curve"
    with pytest.raises(ValueError, match="above Figure E-16's"):
        figure_e_vehicle_allowable_passes("E-16", 1.5, 60)


def test_vehicle_names():
    for name in ("E-28", "e28", "Figure E-28", "truck_5_axle",
                 "Truck 5 Axle"):
        assert resolve_vehicle(name) == "E-28"
    with pytest.raises(ValueError, match="Unknown vehicle"):
        resolve_vehicle("E-1")


def test_same_names_as_the_reference_layer():
    utb = pytest.importorskip("geotech_references.ufc_pavement.tables")
    registry = getattr(utb, "_VEHICLES_E", None)
    if registry is None:
        pytest.skip("no vehicle registry in this refs install")
    for key, data in registry.items():
        assert resolve_vehicle(key) == data["figure"]
        assert resolve_vehicle(data["figure"]) == data["figure"]
