"""Tests for pynite_agent — anchored to closed-form beam solutions."""

import pytest

pytest.importorskip("Pynite")

from pynite_agent import analyze_frame, analyze_continuous_beam, has_pynite

E = 200e6          # kPa (steel)
I = 100e-6         # m^4


def _ss_beam(**kwargs):
    return analyze_frame(
        nodes=[{"name": "N1", "x": 0, "y": 0}, {"name": "N2", "x": 6, "y": 0}],
        members=[{"name": "M1", "i": "N1", "j": "N2",
                  "E": E, "A": 0.01, "Iz": I}],
        supports=[{"node": "N1", "type": "pinned"},
                  {"node": "N2", "type": "roller_y"}],
        **kwargs)


class TestSimplySupportedUDL:
    """L=6 m, w=10 kN/m: R=wL/2=30, |M|=wL^2/8=45, d=5wL^4/384EI=8.4375mm."""

    @pytest.fixture(scope="class")
    def res(self):
        return _ss_beam(member_dist_loads=[
            {"member": "M1", "direction": "FY", "w": -10}])

    def test_reactions(self, res):
        assert res.reactions["N1"]["FY_kN"] == pytest.approx(30.0, rel=1e-6)
        assert res.reactions["N2"]["FY_kN"] == pytest.approx(30.0, rel=1e-6)

    def test_moment(self, res):
        assert res.members[0].moment_abs_kNm == pytest.approx(45.0, rel=1e-6)

    def test_deflection(self, res):
        d_exact = 5 * 10 * 6**4 / (384 * E * I) * 1000    # 8.4375 mm
        assert res.members[0].deflection_abs_mm == pytest.approx(
            d_exact, rel=0.002)

    def test_shear(self, res):
        assert res.members[0].shear_abs_kN == pytest.approx(30.0, rel=1e-6)


class TestPointLoad:
    def test_midspan_point_load(self):
        # P=40 at midspan: R=20 each, M=PL/4=60, d=PL^3/48EI
        res = _ss_beam(member_point_loads=[
            {"member": "M1", "direction": "FY", "value": -40, "x": 3.0}])
        assert res.reactions["N1"]["FY_kN"] == pytest.approx(20.0, rel=1e-6)
        assert res.members[0].moment_abs_kNm == pytest.approx(60.0, rel=1e-6)
        d_exact = 40 * 6**3 / (48 * E * I) * 1000
        assert res.members[0].deflection_abs_mm == pytest.approx(
            d_exact, rel=0.01)


class TestProppedCantilever:
    def test_prop_reaction_and_fixed_moment(self):
        # Fixed at N1, roller at N2, UDL w: R_prop = 3wL/8, M_fix = wL^2/8
        res = analyze_frame(
            nodes=[{"name": "N1", "x": 0, "y": 0},
                   {"name": "N2", "x": 4, "y": 0}],
            members=[{"name": "M1", "i": "N1", "j": "N2",
                      "E": E, "A": 0.01, "Iz": I}],
            supports=[{"node": "N1", "type": "fixed"},
                      {"node": "N2", "type": "roller_y"}],
            member_dist_loads=[{"member": "M1", "direction": "FY", "w": -10}])
        assert res.reactions["N2"]["FY_kN"] == pytest.approx(
            3 * 10 * 4 / 8, rel=1e-6)                       # 15 kN
        assert abs(res.reactions["N1"]["MZ_kNm"]) == pytest.approx(
            10 * 4**2 / 8, rel=1e-6)                        # 20 kN*m


class TestPortalFrame:
    def test_symmetric_portal_reactions(self):
        # 2D portal, vertical UDL on the beam: vertical reactions share wL/2
        res = analyze_frame(
            nodes=[{"name": "A", "x": 0, "y": 0}, {"name": "B", "x": 0, "y": 3},
                   {"name": "C", "x": 5, "y": 3}, {"name": "D", "x": 5, "y": 0}],
            members=[
                {"name": "COL1", "i": "A", "j": "B", "E": E, "A": 0.01, "Iz": I},
                {"name": "BEAM", "i": "B", "j": "C", "E": E, "A": 0.01, "Iz": I},
                {"name": "COL2", "i": "D", "j": "C", "E": E, "A": 0.01, "Iz": I}],
            supports=[{"node": "A", "type": "pinned"},
                      {"node": "D", "type": "pinned"}],
            member_dist_loads=[{"member": "BEAM", "direction": "FY", "w": -12}])
        assert res.reactions["A"]["FY_kN"] == pytest.approx(30.0, rel=1e-4)
        assert res.reactions["D"]["FY_kN"] == pytest.approx(30.0, rel=1e-4)
        # horizontal thrusts equal & opposite
        assert res.reactions["A"]["FX_kN"] == pytest.approx(
            -res.reactions["D"]["FX_kN"], rel=1e-4)


class TestContinuousBeam:
    """Two equal spans, UDL: textbook R = 0.375wL / 1.25wL / 0.375wL,
    M_support = -wL^2/8, span sagging = 9wL^2/128."""

    @pytest.fixture(scope="class")
    def res(self):
        return analyze_continuous_beam([5, 5], E_kPa=E, I_m4=I, udl_kN_m=10)

    def test_reactions(self, res):
        assert res.support_reactions_kN[0] == pytest.approx(18.75, rel=1e-4)
        assert res.support_reactions_kN[1] == pytest.approx(62.50, rel=1e-4)
        assert res.support_reactions_kN[2] == pytest.approx(18.75, rel=1e-4)

    def test_support_moment(self, res):
        assert res.support_moments_kNm[1] == pytest.approx(-31.25, rel=1e-4)
        assert res.max_hogging_kNm == pytest.approx(-31.25, rel=1e-4)

    def test_span_sagging(self, res):
        assert res.span_max_sagging_kNm[0] == pytest.approx(
            9 * 10 * 5**2 / 128, rel=1e-3)                  # 17.578
        assert res.max_sagging_kNm > 0

    def test_summary_and_dict(self, res):
        assert "CONTINUOUS BEAM" in res.summary()
        d = res.to_dict()
        assert len(d["support_reactions_kN"]) == 3

    def test_point_load_and_custom_supports(self):
        r = analyze_continuous_beam(
            [4, 4], E_kPa=E, I_m4=I,
            point_loads=[{"span": 1, "x": 2.0, "P": 50}],
            support_types=["fixed", "roller_y", "roller_y"])
        assert sum(r.support_reactions_kN) == pytest.approx(50.0, rel=1e-6)


class TestValidation:
    def test_missing_pieces(self):
        with pytest.raises(ValueError):
            analyze_frame(nodes=[], members=[], supports=[])
        with pytest.raises(ValueError, match="unknown node"):
            analyze_frame(
                nodes=[{"name": "N1", "x": 0, "y": 0}],
                members=[{"name": "M1", "i": "N1", "j": "NX",
                          "E": E, "A": 0.01, "Iz": I}],
                supports=[{"node": "N1", "type": "fixed"}])

    def test_continuous_beam_validation(self):
        with pytest.raises(ValueError):
            analyze_continuous_beam([], E_kPa=E, I_m4=I)
        with pytest.raises(ValueError, match="support_types"):
            analyze_continuous_beam([5], E_kPa=E, I_m4=I,
                                    support_types=["pinned"])
        with pytest.raises(ValueError, match="outside span"):
            analyze_continuous_beam([5], E_kPa=E, I_m4=I,
                                    point_loads=[{"span": 1, "x": 9, "P": 10}])

    def test_has_flag(self):
        assert has_pynite()
