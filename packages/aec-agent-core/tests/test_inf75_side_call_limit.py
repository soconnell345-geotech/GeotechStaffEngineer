"""The side-call time limit is set from the measured tail (BACKLOG VIS-31
part 2, done with INF-75 so its effect can be measured).

The limit was 180 s. Foundry brief 7 (2026-11-12): 15,545 tile calls took
5 s at the median and under 22 s for 99 %; each of 34 stalled tiles held
its batch for the whole 180 s before a second ask that answered in a
second or two. Brief 5 (another deployment, 2026-10-08) had a 99th
percentile of 74 s. The default is now 90 s: above both measured tails,
half the old wait. ``GEOTECH_VISION_CALL_TIMEOUT_S`` still overrides it.
"""

from __future__ import annotations

from aec_agent_core.deep import busy_retry, vision_engine

#: The 99th percentile of a side call on the two deployments measured.
BRIEF_7_P99_S = 22.0
BRIEF_5_P99_S = 74.0


def test_the_default_limit_is_90_seconds_and_the_setting_overrides_it(
        monkeypatch):
    monkeypatch.delenv(vision_engine.TIMEOUT_ENV, raising=False)
    assert vision_engine.DEFAULT_TIMEOUT_S == 90.0
    assert vision_engine.call_timeout_s() == 90.0
    assert vision_engine.TIMEOUT_RETRIES == 1          # still asked once more
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "180")
    assert vision_engine.call_timeout_s() == 180.0
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "45.5")
    assert vision_engine.call_timeout_s() == 45.5
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    assert vision_engine.call_timeout_s() is None       # as long as the engine


def test_the_limit_is_above_both_measured_tails_and_half_the_old_one():
    """Under a deployment's own tail, calls that would have answered are
    given up as a matter of course; at the old 180 s a stalled tile cost
    its batch three minutes."""
    assert vision_engine.DEFAULT_TIMEOUT_S > BRIEF_5_P99_S > BRIEF_7_P99_S
    assert vision_engine.DEFAULT_TIMEOUT_S <= 180.0 / 2


def test_the_limit_in_force_is_on_the_turns_record(monkeypatch):
    """``turn_start``'s engine record says which limit a run ran under, so
    two runs' timeouts are compared knowing it."""
    monkeypatch.delenv(vision_engine.TIMEOUT_ENV, raising=False)
    assert busy_retry.retry_policy()["vision_call_timeout_s"] == 90.0
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "60")
    assert busy_retry.retry_policy()["vision_call_timeout_s"] == 60.0
