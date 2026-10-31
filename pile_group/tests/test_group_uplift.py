"""GM-289: group uplift as GEC-12 Sec. 7.2.3.2 and UFC 3-220-20 Eqs. 6-40
to 6-43 print it (``module_work/gm289/CHECK.md``).

GEC-12 Vol. III works it twice, each time on the back row in tension:
the North Abutment in cohesionless soil (pp. 52-54, Tables D-17 and D-18:
a 1H:4V block over three layers) and the South Abutment in clay (pp.
204-208, Eq. 7-41). Their printed numbers are pinned here in their own
units (ft, pcf, ksf, kips), converted.
"""

import math

import pytest

from pile_group.group_efficiency import (
    group_uplift_resistance, small_group_reduction)

FT = 0.3048                  # m per ft
PCF = 0.157087464            # kN/m3 per pcf
KSF = 47.880259              # kPa per ksf
KIP = 4.448222               # kN per kip

#: GEC-12 Vol. III Fig. D-26 / Table D-17: gamma' 105 pcf to 10 ft, 42.6
#: pcf to 20 ft, 49.6 pcf below (the table's 'Layer Thickness 20.0' for
#: layer 3 reaches past the 27 ft toe; its Table D-18 volume is for 7 ft).
NORTH_LAYERS = [(10 * FT, 105.0 * PCF), (10 * FT, 42.6 * PCF),
                (20 * FT, 49.6 * PCF)]


def test_north_abutment_block_weight_table_d18():
    """GEC-12 Vol. III p. 54, Table D-18: 6,260.8 + 3,310.8 + 830.1 ft3,
    657.4 + 141.0 + 41.2 = 839.6 kips, for the 41 ft row of 1 ft piles
    with the toes at 27 ft."""
    r = group_uplift_resistance(1 * FT, 41 * FT, 27 * FT, 9,
                                "cohesionless", NORTH_LAYERS)
    vols = [s["volume_m3"] / FT ** 3 for s in r["slices"]]
    assert vols == pytest.approx([6260.8, 3310.8, 830.1], abs=0.05)
    kips = [s["weight_kN"] / KIP for s in r["slices"]]
    assert kips == pytest.approx([657.4, 141.0, 41.2], abs=0.05)
    assert r["R_ug_kN"] / KIP == pytest.approx(839.6, abs=0.05)
    # Table D-17: the block's width and length at the top of layer 1
    assert r["top_width_m"] / FT == pytest.approx(14.5)
    assert r["top_length_m"] / FT == pytest.approx(54.5)


def test_north_abutment_configuration_3_needs_28_ft():
    """GEC-12 Vol. III p. 54: for the 37 ft row 'the minimum pile
    penetration depth to achieve a soil block weight in excess of the
    required nominal tension resistance of 812 kips is 28 feet'."""
    w = [group_uplift_resistance(1 * FT, 37 * FT, d * FT, 9, "cohesionless",
                                 NORTH_LAYERS)["R_ug_kN"] / KIP
         for d in (27, 28)]
    assert w[0] < 812.0 <= w[1]


def test_south_abutment_eq_7_41():
    """GEC-12 Vol. III pp. 206-208: D = 9 ft, B = 1 ft, Z = 37 ft, su =
    0.67 ksf, gamma' = 47.6 pcf (soil 15.85 kips), the cap's 78.83 kips
    and the 13 HP 12x74 piles' 8.66 kips: R_ug = 561.6 kips > 552 = 276 /
    0.5. The module takes the piles as soil, so the piles' steel goes in
    with the cap here, as the example adds it."""
    r = group_uplift_resistance(1 * FT, 37 * FT, 9 * FT, 13, "cohesive",
                                [(9 * FT, 47.6 * PCF)], su=0.67 * KSF,
                                cap_weight=(78.83 + 8.66) * KIP,
                                pile_type="driven")
    assert r["soil_weight_kN"] / KIP == pytest.approx(15.85, abs=0.005)
    assert r["shear_kN"] / KIP == pytest.approx(2 * 9 * 38 * 0.67)
    assert r["R_ug_kN"] / KIP == pytest.approx(561.6, abs=0.05)
    assert r["phi_ug"] == 0.50
    assert r["R_r_kN"] / KIP >= 276.0


def test_one_layer_is_ufc_eq_6_41():
    """UFC 3-220-20 Eq. 6-41: V = B L Z + Z^2 (B + L)/4 + Z^3/12 (made-up
    sizes)."""
    B, L, Z = 2.6, 4.1, 11.0
    r = group_uplift_resistance(B, L, Z, 6, "cohesionless", [(Z, 9.5)])
    v = B * L * Z + Z ** 2 * (B + L) / 4 + Z ** 3 / 12
    assert r["volume_m3"] == pytest.approx(v)
    assert r["R_ug_kN"] == pytest.approx(9.5 * v)


def test_cohesive_block_by_layers_is_ufc_eq_6_42():
    """UFC 3-220-20 Eq. 6-42: W = B L (Z1 gamma_m + Z2 gamma_b) + W_cap,
    and Eq. 6-43 adds 2 Z (B + L) su (made-up values)."""
    r = group_uplift_resistance(3.0, 5.0, 12.0, 9, "cohesive",
                                [(2.0, 18.0), (15.0, 8.0)], su=35.0,
                                cap_weight=250.0)
    w = 3.0 * 5.0 * (2.0 * 18.0 + 10.0 * 8.0) + 250.0
    assert r["R_ug_kN"] == pytest.approx(w + 2 * 12.0 * 8.0 * 35.0)


@pytest.mark.parametrize("pile_type, n, phi", [
    ("driven", 2, 0.40), ("driven", 3, 0.50), ("driven", 4, 0.50),
    ("drilled_shaft", 4, 0.36), ("drilled_shaft", 5, 0.45),
    ("micropile", 6, 0.50), ("cfa", 6, None), ("ddp", 6, None)])
def test_phi_ug_by_pile_type_and_small_group(pile_type, n, phi):
    """GEC-12 Table 7-1: 0.50; UFC 3-220-20 Table 6-28: 0.50, 0.45, 0.50;
    small groups 20 % less, below 3 driven piles (GEC-12 p. 225) or 5 of
    the others (UFC note A); none printed for CFA or DDP."""
    r = group_uplift_resistance(2.0, 3.0, 10.0, n, "cohesionless",
                                [(10.0, 9.0)], pile_type=pile_type)
    if phi is None:
        assert r["phi_ug"] is None and r["R_r_kN"] is None
    else:
        assert r["phi_ug"] == pytest.approx(phi)
        assert r["R_r_kN"] == pytest.approx(phi * r["R_ug_kN"])


def test_ufc_small_group_reading():
    assert small_group_reduction(4, "driven") == 1.0
    assert small_group_reduction(4, "driven", "ufc") == 0.8


def test_the_lesser_governs():
    kw = dict(soil_type="cohesionless", layers=[(10.0, 9.0)],
              pile_type="driven")
    r = group_uplift_resistance(2.0, 3.0, 10.0, 6, **kw,
                                piles_resistance=1.0e6)
    assert r["governs"] == "block" and r["governing_kN"] == r["R_r_kN"]
    r = group_uplift_resistance(2.0, 3.0, 10.0, 6, **kw,
                                piles_resistance=100.0)
    assert r["governs"] == "piles" and r["governing_kN"] == 100.0


@pytest.mark.parametrize("kwargs, match", [
    (dict(n_piles=1), "two or more"),
    (dict(layers=[(4.0, 9.0)]), "toes"),
    (dict(soil_type="cohesive"), "su"),
    (dict(cap_weight=-1.0), "cap_weight"),
])
def test_refused(kwargs, match):
    args = dict(B=2.0, Z=3.0, D=10.0, n_piles=6, soil_type="cohesionless",
                layers=[(10.0, 9.0)])
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        group_uplift_resistance(**args)


def test_layers_below_the_toes_are_cut_off():
    r = group_uplift_resistance(2.0, 3.0, 10.0, 6, "cohesionless",
                                [(6.0, 9.0), (8.0, 10.0), (5.0, 11.0)])
    assert [(s["top_m"], s["bottom_m"]) for s in r["slices"]] == [
        (0.0, 6.0), (6.0, 10.0)]
    assert math.isclose(r["slices"][-1]["unit_weight"], 10.0)
