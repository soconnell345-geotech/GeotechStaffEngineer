"""Offline tests for the self-check summary formatter (no API / no network)."""

from funhouse_agent.deep.selfcheck import _format_summary, run_selfcheck, CHECK_NAMES


def test_summary_all_pass():
    out = _format_summary({
        "agent_tool_call": {"ok": True},
        "cross_session_memory": {"ok": True},
    })
    assert "2/2 checks passed" in out
    assert "ALL GOOD" in out
    assert "[PASS] agent_tool_call" in out
    assert "[PASS] cross_session_memory" in out


def test_summary_mixed_with_error():
    out = _format_summary({
        "agent_tool_call": {"ok": True},
        "cross_session_memory": {"ok": False, "error": "RuntimeError: boom"},
    })
    assert "1/2 checks passed" in out
    assert "SEE FAILURES ABOVE" in out
    assert "[FAIL] cross_session_memory" in out
    assert "RuntimeError: boom" in out


def test_summary_empty():
    out = _format_summary({})
    assert "0/0 checks passed" in out


def test_exports_present():
    # run_selfcheck is importable and CHECK_NAMES documents the checks.
    assert callable(run_selfcheck)
    assert "agent_tool_call" in CHECK_NAMES
    assert "cross_session_memory" in CHECK_NAMES
