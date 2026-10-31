"""V-060: pile groups as GEC-12 Vol 3 Appendix D and UFC 3-220-20 Appendix
B work them (GM-286).

Sources, read from ``geotech-references/docs/``:

* GEC-12 (FHWA-NHI-16-009) Vol 3, North Abutment. Table D-11 (p. 47): the
  limit state loads; Table D-12 (p. 47): 2 rows of 9, 11 or 13 HP piles,
  the rows 4.0 ft apart (Sby), the piles in a row 5.0, 4.0 or 3.0 ft apart
  (Sbx); Table D-13 (p. 49): the front and back row reactions of
  Configuration 1; Table D-14 (p. 49): the factored load per pile, each row
  reaction over the piles in the row and Vuy over all the piles. Fig. D-25
  is the cap's free body: the rows share Q and Muy as a rigid cap on
  vertical piles does.
* GEC-12 Vol 3 pp. 56 and 137: the AASHTO p-multipliers interpolated at
  4b (0.90, 0.625, 0.5) and read at 5b (1.0, 0.85, 0.7).
* GEC-12 Vol 3 pp. 50, 136 and 203: the group efficiency in cohesionless
  soil (North Abutment, Pier 2) and in cohesive soil with the cap in firm
  contact (South Abutment) taken as 1, the sum of the individual
  resistances (GEC-12 Vol 1 Secs. 7.2.2.1, 7.2.2.2).
* GEC-12 Vol 3 Table D-23 (p. 64): Meyerhof (1976) group settlement at five
  pile penetrations for the North Abutment's group footprint, B = 5 ft
  (4 ft between rows plus the 1 ft pile), Z = 41 ft (Configurations 1 and
  2) and 37 ft (Configuration 3), Q = 1540 kips (Service I without LL),
  D' = 2/3 DB with DB the embedment in the bearing stratum.
* UFC 3-220-20 (2025) Appendix B-5.4 (pp. 636-638): block failure of a 7x7
  group of 1 ft square piles, 60 ft long, su1 = 1.33 ksf along the sides,
  su2 = 3 ksf below, B = Z = 18 ft, Nc = 10 limited to 9: R = 14,494 kips,
  0.6 R = 8,696 kips. Table B-32 (p. 643): Mokwa's (1999) p-multipliers at
  s/b = 3, rows 1 to 4 and beyond, weighted over 7 rows.

Print discrepancies, kept as printed: Table D-13's Strength I back row is
-406 kips where Q/2 - Muy/(2 x 2 ft) = -403.75; Table D-14's Configuration 1
tension is -46 kips where -403.75/9 = -44.9 (-406/9 = -45.1). UFC
Appendix B-5.4 measures B and Z to the centres of the outer piles (18 ft;
to their outer faces, as GEC-12 Fig. 7-31 and UFC Fig. 6-15 draw, it would
be 19 ft and 15,813 kips); the test gives the printed 18 ft.
"""

import pytest

from pile_group import (
    GroupLoad, analyze_group_6dof, analyze_vertical_group_simple,
    block_failure_resistance, create_rectangular_layout,
    group_efficiency_factor, meyerhof_group_settlement, p_multiplier,
)

FT, KIP, KSF = 0.3048, 4.448222, 47.880259
KIPFT = KIP * FT

# Table D-11: limit state -> (Q kips, Muy kip-ft); Strength I Vuy 846 kips
_LOADS = {"Strength I": (2505.0, 6625.0), "Service I": (1838.0, 4600.0),
          "Service I without LL": (1540.0, 4162.0)}
_VUY = 846.0
# Table D-13, Configuration 1: (Rfront, Rback) kips
_TABLE_D13 = {"Strength I": (2908.0, -406.0), "Service I": (2069.0, -231.0),
              "Service I without LL": (1811.0, -271.0)}
# Table D-12: configuration -> (piles per row, Sbx ft); Sby 4.0 ft
_CONFIGS = {1: (9, 5.0), 2: (11, 4.0), 3: (13, 3.0)}
# Table D-14: configuration -> (Strength I compression, tension, Vuy,
# Service I without LL), kips per pile
_TABLE_D14 = {1: (323, -46, 47, 201), 2: (265, -37, 39, 165),
              3: (224, -31, 33, 139)}


def _layout(config, axial_stiffness=None, lateral_stiffness=None):
    # the two rows along the module's x, 4 ft apart, so the example's Muy
    # (front row in compression) is the module's +My
    per_row, sbx = _CONFIGS[config]
    return create_rectangular_layout(
        per_row, 2, 4.0 * FT, sbx * FT, axial_stiffness=axial_stiffness,
        lateral_stiffness=lateral_stiffness)


def _rows(result, piles):
    front = [f["axial_kN"] for f, p in zip(result.pile_forces, piles)
             if p.x > 0]
    back = [f["axial_kN"] for f, p in zip(result.pile_forces, piles)
            if p.x < 0]
    return sum(front) / KIP, sum(back) / KIP, len(front)


@pytest.mark.parametrize("state", list(_LOADS))
def test_table_d13_row_reactions(state):
    q, m = _LOADS[state]
    piles = _layout(1)
    r = analyze_vertical_group_simple(
        piles, GroupLoad(Vz=q * KIP, My=m * KIPFT))
    front, back, n = _rows(r, piles)
    assert n == 9
    printed_front, printed_back = _TABLE_D13[state]
    # the free body by hand: Q/2 +/- Muy/(2 x 2 ft)
    assert front == pytest.approx(q / 2 + m / 4, abs=0.5)
    assert back == pytest.approx(q / 2 - m / 4, abs=0.5)
    assert front == pytest.approx(printed_front, abs=1.0)
    # Strength I's back row prints -406 against -403.75 by its own loads
    assert back == pytest.approx(printed_back,
                                 abs=2.5 if state == "Strength I" else 0.6)


@pytest.mark.parametrize("config", list(_CONFIGS))
def test_table_d14_load_per_pile(config):
    comp, tens, vuy, service = _TABLE_D14[config]
    piles = _layout(config)
    q, m = _LOADS["Strength I"]
    r = analyze_vertical_group_simple(
        piles, GroupLoad(Vz=q * KIP, My=m * KIPFT))
    per_pile = [f["axial_kN"] / KIP for f in r.pile_forces]
    assert max(per_pile) == pytest.approx(comp, abs=0.7)
    # Configuration 1 prints -46 against -44.9
    assert min(per_pile) == pytest.approx(tens,
                                          abs=1.2 if config == 1 else 0.5)
    q, m = _LOADS["Service I without LL"]
    r = analyze_vertical_group_simple(
        piles, GroupLoad(Vz=q * KIP, My=m * KIPFT))
    assert max(f["axial_kN"] / KIP for f in r.pile_forces) == \
        pytest.approx(service, abs=0.5)
    # Vuy shared equally (Table D-14 rounds 38.5 and 32.5 up)
    assert _VUY / len(piles) == pytest.approx(vuy, abs=0.6)


@pytest.mark.parametrize("config", list(_CONFIGS))
def test_6dof_with_equal_stiffness_gives_the_same_shares(config):
    """The general rigid cap with equal springs: Fig. D-25's free body, and
    Vuy shared equally (Table D-14)."""
    comp, tens, vuy, _ = _TABLE_D14[config]
    piles = _layout(config, axial_stiffness=2.0e5, lateral_stiffness=2.0e4)
    q, m = _LOADS["Strength I"]
    r = analyze_group_6dof(
        piles, GroupLoad(Vz=q * KIP, My=m * KIPFT, Vx=_VUY * KIP))
    simple = analyze_vertical_group_simple(
        _layout(config), GroupLoad(Vz=q * KIP, My=m * KIPFT))
    assert [f["axial_kN"] for f in r.pile_forces] == pytest.approx(
        [f["axial_kN"] for f in simple.pile_forces], abs=0.11)
    shares = [h["Fx_kN"] / KIP for h in r.pile_head_response]
    assert shares == pytest.approx([_VUY / len(piles)] * len(piles))
    assert max(shares) == pytest.approx(vuy, abs=0.6)


def test_p_multipliers_at_4b_and_5b():
    """GEC-12 Vol 3 pp. 56 and 137 (AASHTO, GEC-12 Vol 1 Sec. 7.3.7.6)."""
    assert [p_multiplier(r, 4.0) for r in (1, 2, 3, 4)] == \
        pytest.approx([0.90, 0.625, 0.5, 0.5])
    assert [p_multiplier(r, 5.0) for r in (1, 2, 3, 4)] == \
        pytest.approx([1.0, 0.85, 0.7, 0.7])


def test_group_efficiency_as_the_examples_take_it():
    """Vol 3 pp. 50 and 136: cohesionless, not over weak soil, s >= 3D,
    not jetted or predrilled -> the sum of the individual resistances;
    p. 203: cohesive, cap in firm contact, s >= 3D and 3 ft -> the same."""
    assert group_efficiency_factor("cohesionless", 4.0)["eta_g"] == 1.0
    assert group_efficiency_factor(
        "cohesive", 3.0, cap_in_contact=True)["eta_g"] == 1.0


# Table D-23: (penetration ft, DB ft, N1(60), S Configs 1 & 2, S Config 3)
_TABLE_D23 = [
    (40.0, 20.0, 17, 2.63, 2.92),
    (42.5, 22.5, 38, 1.11, 1.22),
    (45.5, 0.5, 59, 1.13, 1.25),
    (50.0, 5.0, 59, 1.04, 1.16),
    (55.0, 10.0, 59, 0.95, 1.05),
]


@pytest.mark.parametrize("row", _TABLE_D23, ids=lambda r: f"{r[0]}ft")
def test_table_d23_meyerhof_group_settlement(row):
    pen, db, n160, s12, s3 = row
    for z_ft, printed in ((41.0, s12), (37.0, s3)):
        r = meyerhof_group_settlement(
            group_width=5.0 * FT, group_length=z_ft * FT, N160=n160,
            load_kN=1540.0 * KIP, embedment_DB=db * FT)
        # the printed values to two places: within 0.5 %
        assert r["settlement_in"] == pytest.approx(printed, rel=0.005)
        assert r["equation"] == "7-50"
    assert meyerhof_group_settlement(
        group_width=5.0 * FT, group_length=41.0 * FT, N160=59,
        load_kN=1540.0 * KIP, embedment_DB=5.0 * FT)["pf_ksf"] == \
        pytest.approx(7.512, abs=0.001)


def test_ufc_b_5_4_block_failure():
    """UFC 3-220-20 Appendix B-5.4: 2(60)(18 + 18)(1.33) + (18)(18)(3)(9)
    = 14,494 kips; 0.6 x 14,494 = 8,696 kips."""
    r = block_failure_resistance(18.0 * FT, 18.0 * FT, 60.0 * FT,
                                 1.33 * KSF, 3.0 * KSF)
    assert r["Nc_eq_7_37"] == pytest.approx(10.0)
    assert r["Nc"] == 9.0
    assert r["R_ng_kN"] / KIP == pytest.approx(14494.0, abs=1.0)
    assert 0.6 * r["R_ng_kN"] / KIP == pytest.approx(8696.0, abs=1.0)


def test_ufc_table_b_32_mokwa():
    """UFC 3-220-20 Table B-32 at s/b = 3: 0.82, 0.67, 0.58 and 0.52 for
    rows 4 to 7; their average over the 7 rows 0.593."""
    pm = [p_multiplier(r, 3.0, "mokwa") for r in range(1, 8)]
    assert pm == pytest.approx([0.82, 0.67, 0.58, 0.52, 0.52, 0.52, 0.52])
    assert sum(pm) / 7 == pytest.approx(0.593, abs=0.0005)
