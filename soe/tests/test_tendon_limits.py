"""GM-160: the tendon is sized to both GEC-4 limits (Sec. 5.3.8, p. 77):
design load <= max_load_pct SMTS (0.60) and the maximum test load <= 0.80
SMTS, for bars as for strands. GM-185: the test load is an input,
test_load_factor x DL, default 1.33 (GEC-4 Sec. 7.4.2.2, p. 143)."""

import pytest

from soe.anchor_design import TEST_LOAD_FACTOR, select_tendon


def test_a_bar_is_held_to_the_test_load_limit():
    # DL 500 kN at a 1.50 DL test load: a 32 mm bar (848 kN) meets 0.60
    # SMTS (509 kN) but the 750 kN test load would be 88 % SMTS; the 36 mm
    # bar (1054 kN) is not
    t = select_tendon(500, tendon_type="bar_32mm", test_load_factor=1.5)
    assert t["tendon_type"] == "bar_36mm"
    assert t["max_test_load_pct_GUTS"] <= 80.0
    assert "test load" in t["governed_by"]


def test_max_load_pct_governs_only_below_the_test_load_limit():
    at = {m: select_tendon(500, max_load_pct=m, test_load_factor=1.5)
          for m in (0.60, 0.53, 0.4)}
    assert at[0.60]["n_strands"] == at[0.53]["n_strands"] == 4
    assert "test load" in at[0.60]["governed_by"]
    assert at[0.4]["n_strands"] == 5
    assert "max_load_pct" in at[0.4]["governed_by"]
    for t in at.values():
        assert t["design_load_pct_GUTS"] <= 60.0
        assert t["max_test_load_pct_GUTS"] <= 80.0


def test_the_default_test_load_is_gec4s_133_percent():
    assert TEST_LOAD_FACTOR == 1.33
    t = select_tendon(400)
    assert t["test_load_factor"] == 1.33
    assert t["test_load_kN"] == t["proof_test_kN"] == pytest.approx(532.0)
    assert t["performance_test_kN"] == t["test_load_kN"]
    # GEC-4 p. 77: at 133 % the two limits meet (0.8/1.33 = 0.60 SMTS);
    # 0.80/1.33 = 0.6015 > 0.60, so the design-load limit sizes it
    assert "max_load_pct" in t["governed_by"]
    assert t["design_load_pct_GUTS"] <= 60.0
    assert t["max_test_load_pct_GUTS"] <= 80.0


def test_the_test_load_sizes_the_tendon():
    # DL 300 kN, 15 mm strand (260.7 kN): 1.50 DL needs 562.5 kN SMTS
    # (3 strands); 1.33 DL needs 0.60 SMTS, 500 kN (2 strands)
    assert select_tendon(300, test_load_factor=1.5)["n_strands"] == 3
    assert select_tendon(300)["n_strands"] == 2
    assert select_tendon(300, test_load_factor=1.2)["n_strands"] == 2


def test_a_test_load_below_the_design_load_is_refused():
    with pytest.raises(ValueError, match="test_load_factor"):
        select_tendon(300, test_load_factor=0.9)
