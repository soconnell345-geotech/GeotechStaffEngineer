"""GM-286 review: where the print reads two ways, the group_efficiency tool
says which reading it takes and gives the other in ``choices``.

- Block failure: B and Z to the outer pile faces (GEC-12 Fig. 7-31); UFC
  3-220-20 App. B-5.4 (pp. 637-638) takes them centre to centre of the
  outer piles.
- p-multiplier from 5b to 6b: a straight line to 1.00 at 6b, the module's
  reading of UFC p. 465's "extrapolated to a pile spacing of 6·b"; the
  other reading carries AASHTO's 3b-5b line on.
"""

import pytest

from funhouse_agent.adapters.pile_group import METHOD_REGISTRY as R

BLOCK_DIMS = "group width and length B and Z (block failure)"


def _choice(result, what):
    return next(c for c in result["choices"] if c["what"] == what)


def _block(n_rows=3, n_cols=4):
    return R["group_efficiency"]({
        "n_rows": n_rows, "n_cols": n_cols, "spacing": 1.5,
        "pile_diameter": 0.3, "pile_length": 15.0, "cu": 50.0})


def test_block_to_the_outer_faces_centres_the_alternative():
    r = _block()
    c = _choice(r, BLOCK_DIMS)
    assert c["chosen"] == ("to the outer faces of the outer piles: "
                           "B = 3.30 m, Z = 4.80 m")
    alt = c["alternatives"][0]
    assert "centre to centre" in alt and "App. B-5.4" in alt
    # 3.0 x 4.5 m centre to centre: Nc = 5(1 + 15/15)(1 + 3/22.5) = 11.3
    # -> 9; R = 2(15)(3.0 + 4.5)(50) + (3.0)(4.5)(50)(9) = 17,325 kN
    assert alt.endswith("B = 3.00 m, Z = 4.50 m, nominal 17325.0 kN")
    # the result itself stays to the outer faces
    assert r["block_width_m"] == 3.3 and r["block_length_m"] == 4.8


def test_single_row_has_no_centre_to_centre_block():
    alt = _choice(_block(n_rows=1), BLOCK_DIMS)["alternatives"][0]
    assert "App. B-5.4" in alt and "nominal" not in alt


def _pm(row, sd):
    return R["group_efficiency"]({"row_position": row,
                                  "spacing_diameter_ratio": sd})


@pytest.mark.parametrize("row, sd, module, line", [
    (3, 5.5, 0.85, 0.800),     # 0.7 + (0.7 - 0.3)/2 x 0.5
    (3, 6.0, 1.00, 0.900),
    (2, 5.4, 0.91, 0.940),     # 0.85 + (0.85 - 0.4)/2 x 0.4
])
def test_pm_between_5b_and_6b_gives_the_other_reading(row, sd, module,
                                                      line):
    r = _pm(row, sd)
    assert r["p_multiplier"] == pytest.approx(module, abs=1e-3)
    c = _choice(r, "p-multiplier")
    assert "the module's reading of UFC 3-220-20 p. 465" in c["basis"]
    other = [a for a in c["alternatives"] if "3b-5b line" in a]
    assert len(other) == 1
    assert other[0].endswith(f": {line:.3f}")


@pytest.mark.parametrize("row, sd", [(3, 4.0), (1, 5.5), (3, 7.0)])
def test_pm_readings_agree_elsewhere(row, sd):
    c = _choice(_pm(row, sd), "p-multiplier")
    assert "the module's reading" in c["basis"]
    assert not any("3b-5b line" in a for a in c["alternatives"])
