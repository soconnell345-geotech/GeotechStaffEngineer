"""V-059: GEC-12 Vol 3 Appendix D, Pier 2, HP 12x74 (pre-scour), a layered
profile: the Nordlund method in sand and gravel, the alpha method in clay
(GM-282).

Source: GEC-12 (FHWA-NHI-16-009) Vol 3, Sec. D.22.1, Table D-51 (p. 100),
with the profile of Tables D-49 and D-50 and Figs. D-37 and D-39 (pp. 95-98),
read from ``geotech-references/docs/GEC 12 Vol 3.pdf``. The bottom of the
pile cap is 10 ft below the ground surface; the water table is at the
ground surface (Fig. D-37). Layer 1 loose silty fine sand 0-7 ft, gamma_sat
95 pcf, phi' 32; Layer 2 extremely dense gravel 7-20 ft, 130 pcf, phi'
shaft 36, toe 43; Layer 3 very stiff silty clay 20-50 ft, 128 pcf, in six
5 ft sublayers 3a-3f with su 3.10, 3.21, 3.25, 3.30, 3.34 and 3.40 ksf;
Layer 4 dense gravel with sand 50-70 ft, 125 pcf, phi' shaft 36, toe 38;
limestone below. Table D-51's depths are below the pile cap.

The example ran DrivenPiles with the choices GEC-12's V-002 example also
made: Fig. 7-17's smooth steel curves for the H-pile, Ca held at the
curve's end above su 3.05 ksf, and no limit on sigma'p at the toe. With
those choices the module gives the shaft within +1.1 % to +1.8 % at every
depth, the clay toe (9 su) within +0.5 %, the Layer 2 toe (phi 43) 2.9 %
low and the Layer 4 toe (phi 38) 2.1 % low rising to its q_L plateau
within 0.1 %. Table D-51's clay toe is 9 su on 1.025 ft2, the module's box
area 1.029 ft2.

With the module's defaults the clay shaft is higher (Fig. 7-17's
corrugated curve for H-piles, GEC-12 p. 247: 178.5 against 161.0 kips at
39.99 ft) and the Layer 4 toe lower (Step 8's 3 ksf limit on sigma'p:
240.3 against 275.4 kips at 59.99 ft).

GEC-12 factors this example layer by layer (Vol 3 p. 111: 0.45 on the
Nordlund layers, 0.35 on the alpha-method layers); it prints the factored
resistance only as a plot (Fig. D-54), so the factoring is checked by
its parts.
"""

import pytest

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_h_pile,
)

FT, PCF, KSF, KIP = 0.3048, 0.157087, 47.880259, 4.448222

_EXAMPLE_CHOICES = {"toe_stress_limit": None,
                    "adhesion_curve": "smooth_steel",
                    "adhesion_beyond_chart": "chart_end"}

# Table D-51: depth below the pile cap (ft): (nominal shaft, nominal toe),
# kips
_TABLE_D51 = {
    0.01: (0.0, 106.5), 2: (2.4, 139.7), 4.99: (7.2, 189.6),
    9.99: (18.5, 273.0), 11: (22.4, 28.6), 14.99: (38.9, 28.6),
    19.99: (61.0, 29.6), 24.99: (84.7, 30.0), 29.99: (109.9, 30.5),
    34.99: (135.5, 30.8), 39.99: (161.0, 31.4), 41: (168.5, 256.4),
    43: (184.0, 266.6), 45: (200.0, 275.4), 50: (242.8, 275.4),
    55: (289.4, 275.4), 59.99: (339.6, 275.4),
}


def _profile():
    clay = [AxialSoilLayer(5 * FT, "cohesive", 128 * PCF, cohesion=su * KSF,
                           description=f"Layer 3{name}")
            for name, su in zip("abcdef",
                                (3.10, 3.21, 3.25, 3.30, 3.34, 3.40))]
    return AxialSoilProfile(layers=[
        AxialSoilLayer(7 * FT, "cohesionless", 95 * PCF, friction_angle=32,
                       description="Layer 1"),
        AxialSoilLayer(13 * FT, "cohesionless", 130 * PCF, friction_angle=36,
                       toe_friction_angle=43, description="Layer 2"),
        *clay,
        AxialSoilLayer(20 * FT, "cohesionless", 125 * PCF, friction_angle=36,
                       toe_friction_angle=38, description="Layer 4"),
    ], gwt_depth=0.0)


def _run(D_ft, **kw):
    return AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_profile(),
                             pile_length=D_ft * FT, head_depth=10 * FT,
                             **kw).compute()


def test_v059_shaft_matches_table_d51():
    for D_ft, (Rs_pub, _) in _TABLE_D51.items():
        r = _run(D_ft, **_EXAMPLE_CHOICES)
        if Rs_pub == 0:                     # 0.01 ft of shaft
            assert r.Q_skin / KIP < 0.05
            continue
        assert r.Q_skin / KIP == pytest.approx(Rs_pub, rel=0.02), (
            f"D={D_ft} ft: shaft {r.Q_skin / KIP:.1f} vs published {Rs_pub}")


def test_v059_toe_matches_table_d51():
    for D_ft, (_, Rp_pub) in _TABLE_D51.items():
        r = _run(D_ft, **_EXAMPLE_CHOICES)
        tol = 0.032 if D_ft < 10 else 0.025   # Layer 2's phi 43: 2.9 % low
        assert r.Q_tip / KIP == pytest.approx(Rp_pub, rel=tol), (
            f"D={D_ft} ft: toe {r.Q_tip / KIP:.1f} vs published {Rp_pub}")


def test_v059_the_defaults_move_the_clay_shaft_and_the_deep_toe():
    r = _run(39.99)
    assert r.Q_skin / KIP == pytest.approx(178.5, rel=0.005)  # corrugated
    r = _run(59.99)
    assert r.Q_tip / KIP == pytest.approx(240.3, rel=0.005)   # Step 8
    assert r.toe_details["sigma_p_kPa"] == pytest.approx(3 * KSF, rel=1e-3)


def test_v059_factored_layer_by_layer():
    """Vol 3 p. 111: phi_stat 0.45 on the Nordlund layers and 0.35 on the
    alpha-method layers; the toe in Layer 4 takes Nordlund's 0.45."""
    r = _run(59.99, **_EXAMPLE_CHOICES)
    sand = sum(lb["skin_friction_kN"] for lb in r.layer_breakdown
               if lb["method"] == "nordlund")
    clay = sum(lb["skin_friction_kN"] for lb in r.layer_breakdown
               if lb["method"] == "tomlinson")
    assert clay / KIP == pytest.approx(161.0 - 18.5, rel=0.02)  # Table D-51
    assert r.Q_factored == pytest.approx(0.45 * sand + 0.35 * clay
                                         + 0.45 * r.Q_tip)
