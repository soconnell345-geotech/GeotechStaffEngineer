"""GM-320: Schmertmann layers that do not cover the strain influence
diagram, and consolidation stress lists of the wrong length.

NHI-06-089 Sec. 8.5.1 (pp. 8-45 to 8-51) sums Eq. 8-16 over "the soil
layers within the zone of strain influence". Its App. A.7 (pp. A.7-10 to
A.7-12) works a strip on sand over clay: the diagram unchanged, the sum
stopped at the bottom of the sand, the clay settled by consolidation. So
a depth the layers do not cover is refused unless the caller says what
lies there: not granular soil settled by another method (App. A.7),
incompressible ground (rock), or the deepest layer continued. Made-up
values throughout.
"""

import math

import pytest

from settlement import ConsolidationLayer, SchmertmannLayer, SettlementAnalysis
from settlement.immediate import (
    COVER_TOL_M, schmertmann_settlement, schmertmann_terms,
)

B = 2.0          # square: the diagram ends at 2B = 4 m
KW = dict(q_net=120.0, q0=20.0, B=B, footing_shape="square",
          gamma_soil=18.0)
FULL = [SchmertmannLayer(0.0, 2.0, 12000.0),
        SchmertmannLayer(2.0, 4.0, 20000.0)]
SHORT = [SchmertmannLayer(0.0, 2.0, 12000.0)]    # covers 1B of the 2B
CHOICE = "ground the Schmertmann layers do not cover"


def test_short_profile_refused_with_the_depth_not_covered():
    with pytest.raises(ValueError) as e:
        schmertmann_settlement(layers=SHORT, **KW)
    msg = str(e.value)
    assert "runs from the footing base to 4 m" in msg
    assert "2 to 4 m below the footing base is not covered" in msg
    assert "with below_profile: " in msg
    for value in ("settled_by_another_method", "incompressible",
                  "last_layer_continued"):
        assert f"'{value}'" in msg
    assert "App. A.7" in msg


def test_last_layer_continued_is_the_same_as_giving_that_layer():
    explicit = [SchmertmannLayer(0.0, 2.0, 12000.0),
                SchmertmannLayer(2.0, 4.0, 12000.0)]
    assert schmertmann_settlement(
        layers=SHORT, below_profile="last_layer_continued", **KW
    ) == pytest.approx(schmertmann_settlement(layers=explicit, **KW),
                       rel=1e-12)


@pytest.mark.parametrize("value", ["settled_by_another_method",
                                   "incompressible"])
def test_the_two_values_count_nothing_below_and_say_so(value):
    terms = schmertmann_terms(layers=SHORT, below_profile=value, **KW)
    cont = schmertmann_terms(layers=SHORT,
                             below_profile="last_layer_continued", **KW)
    assert terms["below_profile"] == value
    assert terms["profile_bottom"] == 2.0 and terms["z_max"] == 4.0
    assert terms["uncovered"] == [[2.0, 4.0]]
    # the same diagram, summed to the bottom of the layers only: the part
    # from 2 to 4 m is a triangle from Iz(2 m) to 0
    iz2 = terms["Iz_peak"] * (4.0 - 2.0) / (4.0 - 1.0)
    missing = 0.5 * iz2 * 2.0 / 12000.0
    assert cont["sum_Iz_dz_over_Es"] - terms["sum_Iz_dz_over_Es"] == \
        pytest.approx(missing, rel=1e-12)


def test_full_profile_unchanged_and_below_profile_inert():
    base = schmertmann_terms(layers=FULL, **KW)
    assert base["below_profile"] is None and base["uncovered"] == []
    for option in ("settled_by_another_method", "incompressible",
                   "last_layer_continued"):
        got = schmertmann_terms(layers=FULL, below_profile=option, **KW)
        assert got["Se"] == base["Se"] and got["below_profile"] is None


def test_strip_needs_4B():
    kw = dict(KW, footing_shape="strip")
    with pytest.raises(ValueError, match="runs from the footing base to 8 m"):
        schmertmann_settlement(layers=FULL, **kw)


def test_unknown_below_profile_refused():
    with pytest.raises(ValueError, match="below_profile must be one of"):
        schmertmann_settlement(layers=SHORT, below_profile="rock", **KW)


def test_empty_layers_refused_by_name():
    with pytest.raises(ValueError, match="no Schmertmann layers given"):
        schmertmann_settlement(layers=[], below_profile="last_layer_continued",
                               **KW)


# -- the refusal's own depth passes (review A2) ----------------------------

def test_the_depth_the_refusal_prints_passes_when_followed():
    # B 2 m, L 8 m: L/B = 4, the diagram 1/3 of the way to plane strain,
    # to (2 + 2/3) B = 5.3333 m, which the message rounds up to 5.34 m
    kw = dict(KW, L=8.0, footing_shape="rectangular")
    with pytest.raises(ValueError) as e:
        schmertmann_settlement(layers=SHORT, **kw)
    msg = str(e.value)
    assert "down to 5.34 m with no gaps" in msg
    given = float(msg.split("down to ")[1].split(" m")[0])
    t = schmertmann_terms(
        layers=SHORT + [SchmertmannLayer(2.0, given, 12000.0)], **kw)
    assert t["below_profile"] is None and t["uncovered"] == []
    # within the 10 mm tolerance (5.33 is 3.3 mm short): accepted
    t2 = schmertmann_terms(
        layers=SHORT + [SchmertmannLayer(2.0, 5.33, 12000.0)], **kw)
    assert t2["uncovered"] == []
    assert t2["Se"] == pytest.approx(t["Se"], rel=1e-5)
    # 20 mm short is more than the tolerance: refused
    assert COVER_TOL_M == 0.01
    with pytest.raises(ValueError, match="is not covered"):
        schmertmann_terms(
            layers=SHORT + [SchmertmannLayer(2.0, 5.3133, 12000.0)], **kw)


# -- gaps and a late start (review B1) -------------------------------------

@pytest.mark.parametrize("layers, ranges", [
    ([SchmertmannLayer(0.0, 1.0, 12000.0), SchmertmannLayer(3.0, 4.0, 12000.0)],
     "1 to 3 m below the footing base is not covered"),
    ([SchmertmannLayer(1.5, 4.0, 12000.0)],
     "0 to 1.5 m below the footing base is not covered"),
    ([SchmertmannLayer(5.0, 6.0, 12000.0)],
     "0 to 4 m below the footing base is not covered"),
    ([SchmertmannLayer(0.5, 1.0, 12000.0), SchmertmannLayer(2.0, 3.0, 12000.0)],
     "0 to 0.5 m and 1 to 2 m and 3 to 4 m below the footing base are not "
     "covered"),
])
def test_gaps_and_a_late_start_refused_with_the_ranges(layers, ranges):
    with pytest.raises(ValueError) as e:
        schmertmann_settlement(layers=layers, **KW)
    assert ranges in str(e.value)


def test_a_gap_stated_is_listed_and_counts_nothing():
    gap = [SchmertmannLayer(0.0, 1.0, 12000.0),
           SchmertmannLayer(3.0, 4.0, 12000.0)]
    t = schmertmann_terms(layers=gap, below_profile="incompressible", **KW)
    assert t["uncovered"] == [[1.0, 3.0]]
    # the same as the two layers summed alone: Iz between 1 and 3 m left out
    full = schmertmann_terms(layers=[
        SchmertmannLayer(0.0, 1.0, 12000.0), SchmertmannLayer(1.0, 3.0, 1e12),
        SchmertmannLayer(3.0, 4.0, 12000.0)], **KW)
    assert t["Se"] == pytest.approx(full["Se"], rel=1e-6)


def test_last_layer_continued_does_not_fill_a_gap():
    gap = [SchmertmannLayer(0.0, 1.0, 12000.0),
           SchmertmannLayer(3.0, 4.0, 12000.0)]
    with pytest.raises(ValueError, match="says nothing of 1 to 3 m"):
        schmertmann_terms(layers=gap, below_profile="last_layer_continued",
                          **KW)


def test_a_gap_within_the_tolerance_is_covered():
    t = schmertmann_terms(layers=[SchmertmannLayer(0.0, 1.995, 12000.0),
                                  SchmertmannLayer(2.0, 4.0, 12000.0)], **KW)
    assert t["uncovered"] == [] and t["below_profile"] is None


# -- the combined analysis ---------------------------------------------------

def _analysis(**kw):
    return SettlementAnalysis(
        q_applied=140.0, q_overburden=20.0, B=B, footing_shape="square",
        immediate_method="schmertmann", gamma_soil=18.0, **kw)


def test_combined_analysis_names_its_own_parameter():
    with pytest.raises(ValueError) as e:
        _analysis(schmertmann_layers=SHORT).compute()
    msg = str(e.value)
    assert "with schmertmann_below_profile:" in msg
    assert " below_profile:" not in msg.replace("schmertmann_below_profile", "")
    r = _analysis(schmertmann_layers=SHORT,
                  schmertmann_below_profile="incompressible").compute()
    entry = [c for c in r.choices if c["what"] == CHOICE][0]
    assert entry["how"] == "given" and "incompressible" in entry["chosen"]
    assert "stated (schmertmann_below_profile)" in entry["basis"]
    assert all("schmertmann_below_profile" in a for a in entry["alternatives"]
               if "below_profile" in a)
    assert any("(schmertmann_below_profile)" in w for w in r.warnings)
    full = _analysis(schmertmann_layers=FULL).compute()
    assert CHOICE not in [c["what"] for c in full.choices]
    assert not any("do not cover" in w for w in full.warnings)


_CLAY = dict(e0=0.9, Cc=0.3, Cr=0.05, sigma_v0=60.0)


def test_with_consolidation_below_only_settled_by_another_method():
    """Clay consolidated in the depth the Schmertmann layers leave out:
    the refusal offers the App. A.7 value only, and calling that clay
    incompressible, or counting it with the sand's Es, is refused."""
    clay = [ConsolidationLayer(thickness=2.0, depth_to_center=3.0, **_CLAY)]
    with pytest.raises(ValueError) as e:
        _analysis(schmertmann_layers=SHORT, consolidation_layers=clay).compute()
    msg = str(e.value)
    assert "'settled_by_another_method'" in msg
    assert "'incompressible'" not in msg and "'last_layer_continued'" not in msg
    for value in ("incompressible", "last_layer_continued"):
        with pytest.raises(ValueError, match="consolidates its consolidation"):
            _analysis(schmertmann_layers=SHORT, consolidation_layers=clay,
                      schmertmann_below_profile=value).compute()
    r = _analysis(schmertmann_layers=SHORT, consolidation_layers=clay,
                  schmertmann_below_profile="settled_by_another_method"
                  ).compute()
    entry = [c for c in r.choices if c["what"] == CHOICE][0]
    assert "not granular soil" in entry["chosen"]
    assert "App. A.7" in entry["basis"]
    assert not any("incompressible" in w for w in r.warnings)
    assert r.consolidation > 0


def test_app_a7_arrangement_by_hand():
    """NHI-06-089 App. A.7's arrangement with made-up numbers: a strip
    (B 2 m) on 3 m of sand over clay. The diagram runs to 4B = 8 m; the
    sand covers 0 to 3 m; the clay below is consolidated.

    By hand (NHI-06-089 Eqs. 8-16, 8-17, Fig. 8-21):
      plane strain: Iz = 0.2 at the base, peak at z = B = 2 m, 0 at 8 m;
      sigma'_vp = q0 + gamma z_peak = 30 + 18 x 2 = 66 kPa;
      Izp = 0.5 + 0.1 sqrt(150 / 66) = 0.65076;
      Iz(3 m) = 0.65076 x (8 - 3) / (8 - 2) = 0.54230;
      area of the diagram over the sand, 0 to 2 m: (0.2 + 0.65076) / 2 x 2
      = 0.85076; 2 to 3 m: (0.65076 + 0.54230) / 2 x 1 = 0.59653;
      sum = 1.44728 m; / Es 30 000 kPa = 4.8243e-5 m/kPa;
      C1 = 1 - 0.5 x 30 / 150 = 0.9, C2 = 1;
      Se = 0.9 x 1 x 150 x 4.8243e-5 = 6.513e-3 m = 6.51 mm.
    """
    hand_mm = 6.51
    sand = [SchmertmannLayer(0.0, 3.0, 30000.0)]
    clay = [ConsolidationLayer(thickness=4.0, depth_to_center=5.0,
                               e0=0.9, Cc=0.3, Cr=0.05, sigma_v0=80.0)]
    r = SettlementAnalysis(
        q_applied=180.0, q_overburden=30.0, B=2.0, footing_shape="strip",
        immediate_method="schmertmann", gamma_soil=18.0,
        schmertmann_layers=sand, consolidation_layers=clay,
        schmertmann_below_profile="settled_by_another_method").compute()
    assert r.immediate * 1000 == pytest.approx(hand_mm, abs=0.005)
    assert r.schmertmann_terms["uncovered"] == [[3.0, 8.0]]
    assert r.consolidation > 0
    assert r.total == pytest.approx(r.immediate + r.consolidation
                                    + r.secondary)
    # the arithmetic above, step by step
    izp = 0.5 + 0.1 * math.sqrt(150.0 / 66.0)
    area = (0.2 + izp) + (izp + izp * 5.0 / 6.0) / 2.0
    assert 0.9 * 150.0 * area / 30000.0 * 1000 == pytest.approx(
        hand_mm, abs=0.005)
