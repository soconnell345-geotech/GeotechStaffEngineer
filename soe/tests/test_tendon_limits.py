"""GM-160: the tendon is sized to both GEC-4 limits (Sec. 5.3.8, p. 77):
design load <= max_load_pct SMTS (0.60) and the maximum test load (1.50
DL) <= 0.80 SMTS, for bars as for strands."""

import pytest

from soe.anchor_design import select_tendon


def test_a_bar_is_held_to_the_test_load_limit():
    # DL 500 kN: a 32 mm bar (848 kN) meets 0.60 SMTS (509 kN) but the
    # 750 kN test load would be 88 % SMTS; the 36 mm bar (1054 kN) is not
    t = select_tendon(500, tendon_type="bar_32mm")
    assert t["tendon_type"] == "bar_36mm"
    assert t["max_test_load_pct_GUTS"] <= 80.0
    assert "test load" in t["governed_by"]


def test_max_load_pct_governs_only_below_the_test_load_limit():
    at = {m: select_tendon(500, max_load_pct=m) for m in (0.60, 0.53, 0.4)}
    assert at[0.60]["n_strands"] == at[0.53]["n_strands"] == 4
    assert "test load" in at[0.60]["governed_by"]
    assert at[0.4]["n_strands"] == 5
    assert "max_load_pct" in at[0.4]["governed_by"]
    for t in at.values():
        assert t["design_load_pct_GUTS"] <= 60.0
        assert t["max_test_load_pct_GUTS"] <= 80.0
