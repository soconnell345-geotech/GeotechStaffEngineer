"""GM-320: Schmertmann layers that stop above the bottom of the strain
influence diagram, and consolidation stress lists of the wrong length.

NHI-06-089 Sec. 8.5.1 (pp. 8-45 to 8-51) sums Eq. 8-16 over "the soil
layers within the zone of strain influence" and prints no rule for a
profile that stops above it; nor does UFS 3-220-10 Sec. 5-3. So a short
profile is refused unless the caller says what lies below. Made-up
values throughout.
"""

import pytest

from settlement import SchmertmannLayer, SettlementAnalysis
from settlement.immediate import schmertmann_settlement, schmertmann_terms

B = 2.0          # square: the diagram ends at 2B = 4 m
KW = dict(q_net=120.0, q0=20.0, B=B, footing_shape="square",
          gamma_soil=18.0)
FULL = [SchmertmannLayer(0.0, 2.0, 12000.0),
        SchmertmannLayer(2.0, 4.0, 20000.0)]
SHORT = [SchmertmannLayer(0.0, 2.0, 12000.0)]    # covers 1B of the 2B


def test_short_profile_refused_with_the_depth_not_covered():
    with pytest.raises(ValueError) as e:
        schmertmann_settlement(layers=SHORT, **KW)
    msg = str(e.value)
    assert "stop at 2 m" in msg and "runs to 4 m" in msg
    assert "from 2 to 4 m is not covered" in msg
    assert "last_layer_continued" in msg and "incompressible" in msg


def test_last_layer_continued_is_the_same_as_giving_that_layer():
    explicit = [SchmertmannLayer(0.0, 2.0, 12000.0),
                SchmertmannLayer(2.0, 4.0, 12000.0)]
    assert schmertmann_settlement(
        layers=SHORT, below_profile="last_layer_continued", **KW
    ) == pytest.approx(schmertmann_settlement(layers=explicit, **KW),
                       rel=1e-12)


def test_incompressible_counts_nothing_below_and_says_so():
    terms = schmertmann_terms(layers=SHORT, below_profile="incompressible",
                              **KW)
    cont = schmertmann_terms(layers=SHORT,
                             below_profile="last_layer_continued", **KW)
    assert terms["below_profile"] == "incompressible"
    assert terms["profile_bottom"] == 2.0 and terms["z_max"] == 4.0
    # the same diagram, summed to the bottom of the layers only: the part
    # from 2 to 4 m is a triangle from Iz(2 m) to 0
    iz2 = terms["Iz_peak"] * (4.0 - 2.0) / (4.0 - 1.0)
    missing = 0.5 * iz2 * 2.0 / 12000.0
    assert cont["sum_Iz_dz_over_Es"] - terms["sum_Iz_dz_over_Es"] == \
        pytest.approx(missing, rel=1e-12)
    assert terms["Se"] < cont["Se"]


def test_full_profile_unchanged_and_below_profile_inert():
    base = schmertmann_terms(layers=FULL, **KW)
    assert base["below_profile"] is None
    for option in ("last_layer_continued", "incompressible"):
        got = schmertmann_terms(layers=FULL, below_profile=option, **KW)
        assert got["Se"] == base["Se"] and got["below_profile"] is None


def test_strip_needs_4B():
    kw = dict(KW, footing_shape="strip")
    with pytest.raises(ValueError, match="runs to 8 m"):
        schmertmann_settlement(layers=FULL, **kw)


def test_unknown_below_profile_refused():
    with pytest.raises(ValueError, match="below_profile must be one of"):
        schmertmann_settlement(layers=SHORT, below_profile="rock", **KW)


def _analysis(**kw):
    return SettlementAnalysis(
        q_applied=140.0, q_overburden=20.0, B=B, footing_shape="square",
        immediate_method="schmertmann", gamma_soil=18.0, **kw)


def test_combined_analysis_refuses_and_then_states_the_choice():
    with pytest.raises(ValueError, match="is not covered"):
        _analysis(schmertmann_layers=SHORT).compute()
    r = _analysis(schmertmann_layers=SHORT,
                  schmertmann_below_profile="incompressible").compute()
    what = [c["what"] for c in r.choices]
    assert "soil below the Schmertmann layers" in what
    entry = r.choices[what.index("soil below the Schmertmann layers")]
    assert entry["how"] == "given" and "incompressible" in entry["chosen"]
    assert any("stop at 2 m" in w for w in r.warnings)
    full = _analysis(schmertmann_layers=FULL).compute()
    assert "soil below the Schmertmann layers" not in [
        c["what"] for c in full.choices]
    assert not any("Schmertmann layers stop" in w for w in full.warnings)
