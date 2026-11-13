"""Bracketed duration (BACKLOG GM-325): None with the reason when it cannot
be computed, never 0.0. Until GM-325 the module called an eqsig function
that does not exist and returned 0.0 for every record."""

import numpy as np
import pytest

from seismic_signals_agent.intensity_measures import (
    NO_THRESHOLD, bracketed_duration)
from seismic_signals_agent.results import IntensityMeasuresResult
from seismic_signals_agent.signal_utils import has_eqsig

DT = 0.01


def _record():
    """A made-up record: quiet, a burst from 2.0 s to 5.0 s, quiet."""
    t = np.arange(0, 8.0, DT)
    a = 0.005 * np.sin(2 * np.pi * 1.0 * t)
    burst = (t >= 2.0) & (t <= 5.0)
    a[burst] = 0.2 * np.sin(2 * np.pi * 2.5 * t[burst])
    return a


def test_no_threshold_is_none_with_the_reason():
    dur, note = bracketed_duration(_record(), DT, None)
    assert dur is None
    assert note == NO_THRESHOLD and "bracketed_threshold_g" in note


def test_first_to_last_exceedance():
    a = _record()
    dur, note = bracketed_duration(a, DT, 0.05)
    over = np.nonzero(np.abs(a) > 0.05)[0]
    assert note is None
    assert dur == pytest.approx((over[-1] - over[0]) * DT)
    assert 2.8 < dur < 3.0          # the burst, less its first and last ramps


def test_never_exceeded_is_none_not_zero():
    dur, note = bracketed_duration(_record(), DT, 0.5)
    assert dur is None
    assert "never exceeds" in note and "0.5 g" in note and "0.2 g" in note


def test_a_threshold_must_be_positive():
    with pytest.raises(ValueError, match="bracketed_threshold_g"):
        bracketed_duration(_record(), DT, 0.0)


def test_result_says_none_and_why():
    r = IntensityMeasuresResult(bracketed_duration_note=NO_THRESHOLD)
    d = r.to_dict()
    assert d["bracketed_duration_s"] is None
    assert d["bracketed_threshold_g"] is None
    assert d["bracketed_duration_note"] == NO_THRESHOLD
    assert "not computed" in r.summary()
    r = IntensityMeasuresResult(bracketed_duration_s=3.0,
                                bracketed_threshold_g=0.05)
    d = r.to_dict()
    assert d["bracketed_duration_s"] == 3.0
    assert "bracketed_duration_note" not in d
    assert "above 0.05 g" in r.summary()


@pytest.mark.skipif(not has_eqsig(), reason="eqsig not installed")
def test_through_the_analysis_and_against_eqsig():
    from seismic_signals_agent import analyze_intensity_measures
    a = _record()
    r = analyze_intensity_measures(accel_history=a, dt=DT)
    assert r.bracketed_duration_s is None
    assert r.bracketed_duration_note == NO_THRESHOLD
    r = analyze_intensity_measures(accel_history=a, dt=DT,
                                   bracketed_threshold_g=0.05)
    import eqsig
    import eqsig.im
    from seismic_signals_agent.signal_utils import _G
    theirs = eqsig.im.calc_brac_dur(eqsig.AccSignal(a * _G, DT), 0.05 * _G)
    assert r.bracketed_duration_s == pytest.approx(float(theirs), abs=1e-9)
    assert r.bracketed_threshold_g == 0.05
    r = analyze_intensity_measures(accel_history=a, dt=DT,
                                   bracketed_threshold_g=0.9)
    assert r.bracketed_duration_s is None
    assert "never exceeds" in r.bracketed_duration_note
