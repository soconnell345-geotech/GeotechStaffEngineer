"""GM-160, then GM-299: no unprinted cushion, COR or helmet.

GM-160 made cushion_cor reach the default cushion (500,000 kN/m). GM-299
took that default away: no held reference prints it, nor the COR 0.80 or
the 5 kN helmet. A run now takes the hammer's printed driving system
(GEC-12 Vol 2 Table 12-2, p. 183: the Vulcan 014), one named by
driving_system, or a cushion, COR and helmet given; otherwise it is
refused with the printed options. Made-up piles."""

import pytest

from funhouse_agent import dispatch as D
from wave_equation import (Cushion, PRINTED_DRIVING_SYSTEMS, SoilSetup,
                           discretize_pile, get_driving_system, get_hammer,
                           make_cushion_from_properties, simulate_blow)

_PROBE = {"R_total": 1000.0, "pile_area": 0.01, "pile_length": 20.0,
          "ram_weight": 30.0, "stroke": 2.5}
_GIVEN = {"cushion_stiffness": 500000.0, "cushion_cor": 0.80,
          "helmet_weight": 5.0}
KIP, IN, KSI = 4.4482216152605, 0.0254, 6894.757293168
_WHAT = "hammer cushion and helmet"


def _error(params, method="single_blow"):
    r = D.call_agent("wave_equation", method, params)
    assert "error" in r, r
    return r["error"]


class TestRefused:
    def test_no_cushion_is_refused_with_the_printed_options(self):
        err = _error(dict(_PROBE))
        assert "A cushion is required" in err
        for printed in ("Table 12-2 (p. 183)", "p. 156", "p. 226",
                        "0.90 to 0.92", "1.0 to 4.07 kips"):
            assert printed in err

    def test_a_listed_hammer_with_no_printed_system_is_refused(self):
        err = _error({**_PROBE, "hammer_name": "Vulcan 010"})
        assert "Vulcan 010" in err and "Table 12-2" in err

    def test_a_cushion_needs_its_cor(self):
        p = {**_PROBE, "cushion_stiffness": 500000.0, "helmet_weight": 5.0}
        assert "cushion_cor is required" in _error(p)

    def test_a_cushion_needs_the_helmet(self):
        p = {**_PROBE, "cushion_stiffness": 500000.0, "cushion_cor": 0.8}
        assert "helmet_weight is required" in _error(p)

    def test_a_cushion_and_a_driving_system_both(self):
        p = {**_PROBE, **_GIVEN, "driving_system": "Vulcan 014"}
        assert "not both" in _error(p)

    def test_every_tool_refuses(self):
        assert "A cushion is required" in _error(
            {k: v for k, v in _PROBE.items() if k != "R_total"}
            | {"R_min": 500.0, "R_max": 500.0}, "bearing_graph")
        assert "A cushion is required" in _error(
            {"pile_area": 0.01, "ram_weight": 30.0, "stroke": 2.5,
             "depths": [10.0], "R_at_depth": [500.0]}, "drivability")

    def test_the_library_needs_cor_and_helmet(self):
        with pytest.raises(TypeError):
            Cushion(stiffness=500000.0)
        with pytest.raises(TypeError):
            make_cushion_from_properties(0.1, 0.05, 3.5e6)
        h = get_hammer("Vulcan 010")
        p = discretize_pile(10.0, 0.01, 200e6)
        with pytest.raises(ValueError, match="helmet_weight is required"):
            simulate_blow(h, Cushion(500000.0, 0.8), p, SoilSetup(500.0))


class TestPrinted:
    def test_table_12_2_as_printed(self):
        v = get_driving_system("Vulcan 014")
        assert (v.helmet_kips, v.E_ksi, v.area_in2, v.thickness_in, v.cor) \
            == (1.67, 208.0, 234.0, 6.0, 0.91)
        assert v.stiffness == pytest.approx(208 * KSI * 234 * IN**2 / (6 * IN))
        assert v.stiffness == pytest.approx(1.4206e6, rel=1e-4)
        i = get_driving_system("ice 42s")
        assert (i.helmet_kips, i.E_ksi, i.area_in2, i.thickness_in, i.cor) \
            == (2.05, 175.0, 398.0, 2.0, 0.92)
        assert set(PRINTED_DRIVING_SYSTEMS) == {"Vulcan 014", "ICE 42S"}

    def test_the_vulcan_014_takes_its_printed_system(self):
        r = D.call_agent("wave_equation", "single_blow",
                         {**_PROBE, "hammer_name": "Vulcan 014"})
        assert "error" not in r, r
        ch = [c for c in r["choices"] if c["what"] == _WHAT][0]
        assert ch["how"] == "default" and "Table 12-2" in ch["basis"]
        assert ch["chosen"] == "k = 1,420,629 kN/m, COR 0.91, helmet 7.43 kN"
        used = r["defaults_used"]
        assert used["cushion_stiffness"] == pytest.approx(1420628.8875)
        assert used["cushion_cor"] == 0.91
        assert used["helmet_weight"] == pytest.approx(1.67 * KIP, abs=1e-3)
        same = D.call_agent("wave_equation", "single_blow", {
            **_PROBE, "hammer_name": "Vulcan 014",
            "cushion_stiffness": used["cushion_stiffness"],
            "cushion_cor": 0.91, "helmet_weight": 1.67 * KIP})
        assert same["blow_count_per_m"] == r["blow_count_per_m"]
        given = [c for c in same["choices"] if c["what"] == _WHAT][0]
        assert given["how"] == "given"
        assert {**given, "how": "default"} == ch
        assert "cushion_stiffness" not in same["defaults_used"]

    def test_a_named_system_with_another_hammer(self):
        r = D.call_agent("wave_equation", "single_blow",
                         {**_PROBE, "driving_system": "ICE 42S"})
        assert "error" not in r, r
        ch = [c for c in r["choices"] if c["what"] == _WHAT][0]
        assert ch["how"] == "given"
        assert "here run with the Custom" in ch["basis"]
        assert "cushion_stiffness" not in r.get("defaults_used", {})

    def test_cor_and_helmet_given_replace_the_systems(self):
        a = D.call_agent("wave_equation", "single_blow",
                         {**_PROBE, "driving_system": "Vulcan 014"})
        b = D.call_agent("wave_equation", "single_blow",
                         {**_PROBE, "driving_system": "Vulcan 014",
                          "cushion_cor": 0.5, "helmet_weight": 15.0})
        assert a["blow_count_per_m"] != b["blow_count_per_m"]
        ch = [c for c in b["choices"] if c["what"] == _WHAT][0]
        assert "COR 0.5" in ch["chosen"] and "15.00 kN" in ch["chosen"]
        assert "1,420,629 kN/m" in ch["chosen"]

    def test_given_values_reproduce_the_old_default_run(self):
        """The values GM-299 removed, given: the run before GM-299."""
        r = D.call_agent("wave_equation", "single_blow", {**_PROBE, **_GIVEN})
        assert "error" not in r, r
        c = D.call_agent("wave_equation", "single_blow",
                         {**_PROBE, **_GIVEN, "cushion_cor": 0.5})
        assert c["blow_count_per_m"] != r["blow_count_per_m"]
