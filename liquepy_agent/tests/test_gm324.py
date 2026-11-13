"""BACKLOG GM-324: the total-stress rule of the B&I (2014) SPT route, and
what the CPT and field-correlation results carry. Made-up profiles."""

import numpy as np
import pytest

from liquepy_agent import has_liquepy
from liquepy_agent.results import FieldCorrelationsResult

needs_liquepy = pytest.mark.skipif(not has_liquepy(),
                                   reason="liquepy not installed")

# loose fill over denser sand, water at 3 m
DEPTH = [2.0, 5.0, 8.0, 12.0]
GAMMA = [16.0, 18.5, 19.5, 20.5]


def _spt(**over):
    from liquepy_agent import analyze_spt_liquefaction
    kw = dict(depth=DEPTH, n1_60=[8, 12, 18, 22], fc=[5, 10, 15, 8],
              gamma=GAMMA, amax_g=0.25, gwt_depth=3.0, m_w=7.0)
    kw.update(over)
    return analyze_spt_liquefaction(**kw)


@needs_liquepy
def test_total_stress_is_the_sum_of_the_layers_above():
    r = _spt()
    # 16 x 2; + 18.5 x 3; + 19.5 x 3; + 20.5 x 4
    assert r.sigma_v == pytest.approx([32.0, 87.5, 146.0, 228.0])
    u = [0.0, 9.81 * 2.0, 9.81 * 5.0, 9.81 * 9.0]
    assert r.sigma_veff == pytest.approx(np.array(r.sigma_v) - u)
    # not the depth's own unit weight over everything above it
    assert r.sigma_v[1] != pytest.approx(18.5 * 5.0)


@needs_liquepy
def test_one_unit_weight_is_gamma_z():
    r = _spt(gamma=[19.0] * 4)
    assert r.sigma_v == pytest.approx(19.0 * np.array(DEPTH))


@needs_liquepy
def test_depths_must_increase_and_be_positive():
    with pytest.raises(ValueError, match="strictly increasing"):
        _spt(depth=[2.0, 8.0, 5.0, 12.0])
    with pytest.raises(ValueError, match="positive"):
        _spt(depth=[0.0, 5.0, 8.0, 12.0])


def _cpt(gwl):
    from liquepy_agent import analyze_cpt_liquefaction
    z = np.arange(0.5, 12.01, 0.5)
    q_c = np.where(z < 6, 4000.0, 9000.0)          # kPa
    f_s = np.where(z < 6, 30.0, 60.0)
    return analyze_cpt_liquefaction(depth=z, q_c=q_c, f_s=f_s, gwl=gwl,
                                    pga=0.3, m_w=7.0)


@needs_liquepy
def test_cpt_counts_its_liquefiable_readings():
    r = _cpt(gwl=2.0)
    sus = (r.depth >= 2.0) & (r.i_c < r.i_c_limit)
    assert r.n_susceptible == int(sus.sum()) > 0
    assert r.n_liquefiable == int((sus & (r.factor_of_safety < 1.0)).sum())
    assert r.n_liquefiable > 0
    assert r.min_fos == pytest.approx(float(r.factor_of_safety[sus].min()))
    d = r.to_dict()
    for key in ("lpi", "lsn", "ldi_m", "min_fos", "min_fos_basis",
                "n_susceptible", "n_liquefiable", "max_settlement_mm"):
        assert key in d
    assert "below the water table" in d["min_fos_basis"]
    assert "warnings" not in d


@needs_liquepy
def test_cpt_with_no_susceptible_reading_says_so():
    r = _cpt(gwl=50.0)                 # the water table below the sounding
    assert r.n_susceptible == 0 and r.n_liquefiable == 0
    d = r.to_dict()
    assert d["min_fos_basis"] == "all readings"
    assert len(d["warnings"]) == 1
    assert "ALL readings" in d["warnings"][0]


def _fc(n):
    z = np.linspace(0.2, 0.2 * n, n)
    return FieldCorrelationsResult(
        n_points=n, gwl_m=1.0, vs_method="mcgann2015", depth=z,
        vs_m_per_s=150.0 + 5.0 * z, relative_density=np.full(n, 0.6),
        su_ratio=np.full(n, 0.25), i_c=np.linspace(1.5, 3.0, n),
        permeability_cm_per_s=np.full(n, 1e-5))


def test_correlation_profile_one_row_per_reading():
    r = _fc(10)
    d = r.to_dict()
    rows = d["profile"]
    assert len(rows) == 10 and "profile_note" not in d
    assert rows[0]["top_m"] == pytest.approx(0.1)
    assert rows[0]["bottom_m"] == rows[1]["top_m"] == pytest.approx(0.3)
    assert rows[3] == {"top_m": 0.7, "bottom_m": 0.9, "Vs_m_per_s": 154.0,
                       "Dr": 0.6, "su_ratio": 0.25,
                       "I_c": round(float(r.i_c[3]), 3)}
    assert d["su_ratio_min"] == d["su_ratio_max"] == 0.25
    assert d["i_c_min"] == 1.5 and d["i_c_max"] == 3.0


def test_correlation_profile_of_a_long_sounding_is_banded():
    r = _fc(400)
    d = r.to_dict()
    rows = d["profile"]
    assert len(rows) == 40
    assert "400 readings summarised in 40 depth bands" in d["profile_note"]
    # the bands tile the sounding and carry band averages
    assert all(a["bottom_m"] == b["top_m"] for a, b in zip(rows, rows[1:]))
    assert rows[0]["Vs_m_per_s"] == pytest.approx(
        float(np.mean(r.vs_m_per_s[:10])), abs=0.05)
    import json
    assert len(json.dumps(d)) < 8000


def test_correlation_profile_keeps_a_missing_value_missing():
    r = _fc(5)
    r.su_ratio[2] = np.nan
    assert r.to_dict()["profile"][2]["su_ratio"] is None


@needs_liquepy
def test_field_correlations_return_the_profiles():
    from liquepy_agent import analyze_field_correlations
    z = np.arange(0.5, 10.01, 0.5)
    r = analyze_field_correlations(
        depth=z, q_c=np.where(z < 5, 3000.0, 12000.0),
        f_s=np.where(z < 5, 40.0, 60.0), gwl=1.5)
    d = r.to_dict()
    assert len(d["profile"]) == len(z)
    for row in d["profile"]:
        assert set(row) == {"top_m", "bottom_m", "Vs_m_per_s", "Dr",
                            "su_ratio", "I_c"}
