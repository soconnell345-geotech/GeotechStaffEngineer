"""
Tests for liquepy_agent SPT-based B&I-2014 liquefaction triggering.

The SPT triggering procedure is assembled from liquepy's tested B&I-2014
building blocks (liquepy ships no packaged SPT triggering object — only field
correlations). These tests validate the assembled procedure against the
published B&I-2014 SPT triggering curve and expected qualitative behavior.

Tier 1: pure functions (fines correction, MSF) — no liquepy needed for the
        closed-form helpers, but analyze_spt_liquefaction needs liquepy.
"""

import math

import numpy as np
import pytest

from liquepy_agent.liquepy_utils import has_liquepy
from liquepy_agent.results import SPTLiquefactionResult
from liquepy_agent.spt_liquefaction import (
    analyze_spt_liquefaction,
    bi2014_spt_fines_correction,
    bi2014_spt_msf,
)

requires_liquepy = pytest.mark.skipif(
    not has_liquepy(), reason="liquepy not installed"
)


# =====================================================================
# Closed-form helpers — no liquepy dependency
# =====================================================================

class TestBI2014SptFinesCorrection:
    def test_clean_sand_no_correction(self):
        # FC <= ~5% -> negligible delta
        assert bi2014_spt_fines_correction(10.0, 2.0) == pytest.approx(10.0, abs=0.05)

    def test_high_fines_plateau(self):
        # B&I delta plateaus ~5.6 at high FC (Eq 2.23)
        delta = bi2014_spt_fines_correction(10.0, 50.0) - 10.0
        assert delta == pytest.approx(5.6, abs=0.2)

    def test_intermediate_fines(self):
        # FC=20% -> delta ~ 4.48
        delta = bi2014_spt_fines_correction(10.0, 20.0) - 10.0
        assert delta == pytest.approx(4.48, abs=0.1)

    def test_monotonic_increasing(self):
        prev = -1.0
        for fc in [0, 5, 10, 20, 35, 50]:
            d = bi2014_spt_fines_correction(0.0, fc)
            assert d >= prev - 1e-9
            prev = d

    def test_fc_clipped(self):
        # FC > 100 clipped, no crash
        assert bi2014_spt_fines_correction(10.0, 150.0) > 10.0


class TestBI2014SptMsf:
    def test_msf_unity_at_m7p5(self):
        assert bi2014_spt_msf(7.5, 15.0) == pytest.approx(1.0, abs=1e-9)

    def test_msf_greater_below_7p5(self):
        assert bi2014_spt_msf(6.5, 15.0) > 1.0

    def test_msf_less_above_7p5(self):
        assert bi2014_spt_msf(8.0, 15.0) < 1.0

    def test_msf_known_value(self):
        # M=6.5, ncs=15 -> ~1.12
        assert bi2014_spt_msf(6.5, 15.0) == pytest.approx(1.12, abs=0.02)


# =====================================================================
# Result dataclass — no liquepy
# =====================================================================

class TestSPTLiquefactionResult:
    def test_default_construction(self):
        r = SPTLiquefactionResult()
        assert r.n_layers == 0
        assert r.m_w == 7.5
        d = r.to_dict()
        assert d["method"] == "bi2014"
        assert d["input_type"] == "SPT"
        assert d["layer_results"] == []

    def test_summary_runs(self):
        r = SPTLiquefactionResult(n_layers=2, amax_g=0.3, m_w=7.0,
                                  min_fos=0.8, n_liquefiable=1)
        assert "B&I 2014" in r.summary()


# =====================================================================
# Full SPT triggering — requires liquepy
# =====================================================================

@requires_liquepy
class TestSptTriggering:
    def test_dense_layer_not_liquefiable(self):
        r = analyze_spt_liquefaction(
            depth=[5], n1_60=[35], fc=[3], gamma=[18],
            amax_g=0.3, gwt_depth=1.0, m_w=7.5,
        )
        assert r.liquefiable[0] == False  # noqa: E712
        assert r.factor_of_safety[0] > 1.0

    def test_loose_layer_liquefiable(self):
        r = analyze_spt_liquefaction(
            depth=[4], n1_60=[5], fc=[5], gamma=[18],
            amax_g=0.4, gwt_depth=1.0, m_w=7.5,
        )
        assert r.liquefiable[0] == True  # noqa: E712
        assert r.factor_of_safety[0] < 1.0

    def test_crr_matches_published_curve(self):
        # At (N1)60cs ~ 25, B&I-2014 CRR_M7.5 ~ 0.29. Build a layer where
        # N1_60cs ~= 25 (clean sand, FC<=5 so no fines bump), shallow, M7.5,
        # K_sigma ~= 1 at sigma' ~ 100 kPa. Then CRR (=CRR_M7.5 here) ~ 0.29.
        r = analyze_spt_liquefaction(
            depth=[6], n1_60=[25], fc=[3], gamma=[18.5],
            amax_g=0.2, gwt_depth=0.0, m_w=7.5,
        )
        # CRR at this clean-sand layer should be near the published 0.29.
        assert r.crr[0] == pytest.approx(0.29, abs=0.04)
        # N1_60cs should equal N1_60 for clean sand.
        assert r.n1_60cs[0] == pytest.approx(25.0, abs=0.1)

    def test_fines_increase_resistance(self):
        # Adding fines raises (N1)60cs -> raises CRR -> raises FoS.
        clean = analyze_spt_liquefaction(
            depth=[5], n1_60=[12], fc=[2], gamma=[18],
            amax_g=0.3, gwt_depth=1.0, m_w=7.5,
        )
        silty = analyze_spt_liquefaction(
            depth=[5], n1_60=[12], fc=[30], gamma=[18],
            amax_g=0.3, gwt_depth=1.0, m_w=7.5,
        )
        assert silty.n1_60cs[0] > clean.n1_60cs[0]
        assert silty.crr[0] > clean.crr[0]
        assert silty.factor_of_safety[0] > clean.factor_of_safety[0]

    def test_msf_smaller_magnitude_raises_fos(self):
        big = analyze_spt_liquefaction(
            depth=[5], n1_60=[12], fc=[5], gamma=[18],
            amax_g=0.3, gwt_depth=1.0, m_w=8.0,
        )
        small = analyze_spt_liquefaction(
            depth=[5], n1_60=[12], fc=[5], gamma=[18],
            amax_g=0.3, gwt_depth=1.0, m_w=6.0,
        )
        assert small.factor_of_safety[0] > big.factor_of_safety[0]

    def test_multi_layer_profile(self):
        r = analyze_spt_liquefaction(
            depth=[3, 6, 9, 12], n1_60=[8, 12, 25, 10],
            fc=[5, 10, 15, 5], gamma=[18, 18, 18, 18],
            amax_g=0.35, gwt_depth=2.0, m_w=7.0,
        )
        assert r.n_layers == 4
        # Dense mid-layer (N=25) should be the safest.
        assert r.factor_of_safety[2] == max(r.factor_of_safety)
        assert r.min_fos == pytest.approx(min(r.factor_of_safety))
        assert r.n_liquefiable == int(np.sum(r.factor_of_safety < 1.0))

    def test_to_dict_layer_results(self):
        r = analyze_spt_liquefaction(
            depth=[3, 6], n1_60=[8, 25], fc=[5, 5], gamma=[18, 18],
            amax_g=0.3, gwt_depth=1.0, m_w=7.5,
        )
        d = r.to_dict()
        assert d["method"] == "bi2014"
        assert d["input_type"] == "SPT"
        assert len(d["layer_results"]) == 2
        for lr in d["layer_results"]:
            assert {"depth_m", "N1_60", "N1_60cs", "CSR", "CRR",
                    "FOS_liq", "liquefiable"} <= set(lr)

    def test_above_gwt_uses_total_as_effective(self):
        r = analyze_spt_liquefaction(
            depth=[2], n1_60=[10], fc=[5], gamma=[18],
            amax_g=0.3, gwt_depth=5.0, m_w=7.5,
        )
        # Above GWT: sigma_v_eff == sigma_v.
        assert r.sigma_veff[0] == pytest.approx(r.sigma_v[0])

    def test_validation_length_mismatch(self):
        with pytest.raises(ValueError):
            analyze_spt_liquefaction(
                depth=[3, 6], n1_60=[8], fc=[5, 5], gamma=[18, 18],
                amax_g=0.3, gwt_depth=1.0,
            )

    def test_validation_bad_amax(self):
        with pytest.raises(ValueError):
            analyze_spt_liquefaction(
                depth=[3], n1_60=[8], fc=[5], gamma=[18],
                amax_g=0.0, gwt_depth=1.0,
            )
