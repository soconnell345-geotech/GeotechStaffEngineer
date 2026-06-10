"""Offline tests for module_work/triage_feedback.py against the v2 results format.

v5.1 re-pointed the triage tool at the deepagents eval harness's output
(``funhouse_agent.deep.eval_harness``): ``run_suite`` dicts
(``{"results": [QAResult dicts]}``) and ``run_ab`` dicts (``{"v1": {...},
"v2": {...}}``), while keeping the legacy v1 flat-list format working. These
tests drive it with SYNTHETIC results JSON only — no agent, no API calls.

The test lives in the deep suite (not module_work) because the contract under
test is "triage understands the deep eval harness's QAResult dict shape";
the synthetic records below mirror ``QAResult.to_dict()`` exactly.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_triage_feedback_offline.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# module_work/ is not a package; load the script by file path (repo root is
# three levels above this test file: tests/ -> deep/ -> funhouse_agent/ -> root).
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "module_work" / "triage_feedback.py"
_spec = importlib.util.spec_from_file_location("triage_feedback", _SCRIPT)
triage_feedback = importlib.util.module_from_spec(_spec)
sys.modules["triage_feedback"] = triage_feedback
_spec.loader.exec_module(triage_feedback)

extract_records = triage_feedback.extract_records
normalize_record = triage_feedback.normalize_record
triage = triage_feedback.triage


# ---------------------------------------------------------------------------
# Synthetic results (mirror QAResult.to_dict() / the legacy flat list)
# ---------------------------------------------------------------------------

def _v2_record(qid="BC-1", module="bearing_capacity", errored=False,
               exception=None):
    """A synthetic v2 QAResult dict, shaped like ``QAResult.to_dict()``."""
    trace = [
        {"name": "list_methods", "args": {"agent_name": module},
         "errored": False, "note": ""},
        {"name": "call_agent",
         "args": {"agent_name": module, "method": "bearing_capacity_analysis",
                  "parameters": {"width": 2.0}},
         "errored": errored,
         "note": "KeyError: 'friction_angle'" if errored else ""},
    ]
    errors = []
    if errored:
        errors.append({"tool": "call_agent", "type": "dispatch",
                       "message": "KeyError: 'friction_angle'"})
    return {
        "qid": qid, "module": module, "agent": "v2",
        "answer": "q_ult = 1234 kPa", "trace": trace, "rounds": 2,
        "errors": errors, "latency_s": 3.2,
        "usage": {"input_tokens": 1000, "output_tokens": 50,
                  "total_tokens": 1050},
        "exception": exception,
        "n_tool_errors": 1 if errored else 0,
        "has_tool_error": bool(errored),
        "total_tokens": 1050, "input_tokens": 1000, "output_tokens": 50,
    }


def _v1_record(rid="SS-1", module="slope_stability", errored=False):
    """A synthetic legacy record (docs/geotech_test_suite_results.json shape)."""
    return {
        "id": rid, "module": module, "rounds": 3,
        "tool_sequence": [f"{module}.describe", f"{module}.run"],
        "errors": ([{"message": "Unknown method 'run'"}] if errored else []),
        "exception": None,
    }


def _run_suite_payload(records):
    return {"model": "claude-x", "n": len(records), "results": records,
            "metrics": {"p1": 0}}


def _run_ab_payload(v1_records, v2_records):
    return {
        "questions": [r.get("qid") or r.get("id") for r in v2_records],
        "v1": {"results": v1_records, "metrics": {}},
        "v2": {"results": v2_records, "metrics": {}},
        "model": None,
    }


# ---------------------------------------------------------------------------
# Format autodetection
# ---------------------------------------------------------------------------

def test_extract_records_v2_run_suite():
    recs = extract_records(_run_suite_payload([_v2_record(), _v2_record("BC-2")]))
    assert [r["id"] for r in recs] == ["BC-1", "BC-2"]
    assert all(r["module"] == "bearing_capacity" for r in recs)


def test_extract_records_v2_run_ab_picks_v2_side_by_default():
    payload = _run_ab_payload(
        v1_records=[_v1_record("Q-1")],  # the v1 side uses the legacy shape
        v2_records=[_v2_record("Q-1"), _v2_record("Q-2")],
    )
    recs = extract_records(payload)
    assert len(recs) == 2  # v2 side, not v1
    assert recs[0]["id"] == "Q-1" and recs[1]["id"] == "Q-2"
    # agent= selects the other side when asked.
    v1_recs = extract_records(payload, agent="v1")
    assert len(v1_recs) == 1


def test_extract_records_v1_legacy_list():
    recs = extract_records([_v1_record(), _v1_record("SS-2", errored=True)])
    assert [r["id"] for r in recs] == ["SS-1", "SS-2"]
    assert recs[0]["tool_sequence"] == ["slope_stability.describe",
                                        "slope_stability.run"]


def test_extract_records_rejects_unknown_shape():
    with pytest.raises(ValueError):
        extract_records({"foo": "bar"})


# ---------------------------------------------------------------------------
# v2 record normalization
# ---------------------------------------------------------------------------

def test_normalize_v2_record_builds_tool_sequence_from_trace():
    rec = normalize_record(_v2_record(errored=True))
    assert rec["id"] == "BC-1"
    assert rec["module"] == "bearing_capacity"
    assert rec["rounds"] == 2
    # call_agent renders as agent.method; the errored call is flagged.
    assert rec["tool_sequence"] == [
        "bearing_capacity",  # list_methods has agent but no method arg
        "bearing_capacity.bearing_capacity_analysis [errored]",
    ]
    assert rec["errors"][0]["message"] == "KeyError: 'friction_angle'"


def test_normalize_v2_record_clean():
    rec = normalize_record(_v2_record(errored=False))
    assert rec["errors"] == []
    assert not any("[errored]" in t for t in rec["tool_sequence"])


# ---------------------------------------------------------------------------
# Triage over a synthetic v2 suite
# ---------------------------------------------------------------------------

def test_triage_classifies_and_groups_v2_records():
    records = extract_records(_run_suite_payload([
        _v2_record("BC-1", errored=False),
        _v2_record("BC-2", errored=True),
        _v2_record("SS-9", module="slope_stability",
                   exception="RuntimeError: boom\nline2"),
    ]))
    feedback, stats = triage(records)

    assert stats == {"n": 3, "clean": 1, "soft": 1, "hard": 1}
    # Modules grouped under their domain, with per-module rollups.
    all_mods = {m: v for dom in feedback.values() for m, v in dom.items()}
    bc = all_mods["bearing_capacity"]
    assert bc["n_questions"] == 2 and bc["n_with_errors"] == 1
    assert bc["classes"].get("ergonomics:param-name", 0) == 1
    assert any("friction_angle" in e for e in bc["errors"])
    # The exception record keeps only the first line of the traceback.
    ss = all_mods["slope_stability"]
    q = ss["questions"][0]
    assert q["exception"] == "RuntimeError: boom"
    assert q["tools"] == ["slope_stability",
                          "slope_stability.bearing_capacity_analysis"]


def test_triage_v1_records_still_work():
    """Legacy compat: the flat v1 list triages exactly as before."""
    records = extract_records([
        _v1_record("SS-1"), _v1_record("SS-2", errored=True),
    ])
    feedback, stats = triage(records)
    assert stats["n"] == 2 and stats["clean"] == 1 and stats["soft"] == 1
    all_mods = {m: v for dom in feedback.values() for m, v in dom.items()}
    ss = all_mods["slope_stability"]
    assert ss["classes"].get("ergonomics:method-name", 0) == 1


# ---------------------------------------------------------------------------
# End-to-end main(): file in -> summary out -> module_feedback.json
# ---------------------------------------------------------------------------

def test_main_end_to_end_with_v2_suite_file(tmp_path, capsys):
    results = tmp_path / "results.json"
    out = tmp_path / "module_feedback.json"
    results.write_text(json.dumps(_run_suite_payload([
        _v2_record("BC-1"), _v2_record("BC-2", errored=True),
    ])), encoding="utf-8")

    rc = triage_feedback.main([str(results), str(out)])
    assert rc == 0

    printed = capsys.readouterr().out
    assert "records: 2" in printed
    assert "clean 1" in printed

    feedback = json.loads(out.read_text(encoding="utf-8"))
    all_mods = {m: v for dom in feedback.values() for m, v in dom.items()}
    assert all_mods["bearing_capacity"]["n_with_errors"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
