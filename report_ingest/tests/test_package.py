"""The package's shipping contract: lazy imports, no hard SDK dependency."""

from __future__ import annotations

import subprocess
import sys

CODE = """
import sys
import report_ingest
assert "anthropic" not in sys.modules, (
    "importing report_ingest pulled in anthropic; it must stay optional")
assert "report_ingest.engine" not in sys.modules, (
    "importing report_ingest pulled in its engine module")
assert "planlens" not in sys.modules, (
    "importing report_ingest pulled in planlens")
assert "langgraph" not in sys.modules, (
    "importing report_ingest pulled in langgraph; the sub-agent graph is "
    "built on demand and the package must not cost the app that at startup")
assert "pydantic" not in sys.modules, (
    "importing report_ingest pulled in pydantic; the record model is lazy too")
for name in ("triage", "review_labels", "classify_pages_by_vision",
             "read_log", "read_lab_sheet",
             "read_narrative", "reconcile", "write_outputs", "ingest_report",
             "run_folder", "build_report_ingest_subagent", "write_diggs",
             "diggs_schema_gate", "diggs_roundtrip_gate", "score_on_cluster",
             "PrompterEngine", "ClaudeEngine", "CostMeter"):
    assert callable(getattr(report_ingest, name)), name
# The record's classes arrive through a module __getattr__, so asking for one
# is what imports the model.
assert report_ingest.ReportRecord().counts()["investigations"] == 0
assert report_ingest.Quantity(value=1.0, unit="ft").to_si().unit == "m"
try:
    report_ingest.NoSuchThing
except AttributeError:
    pass
else:
    raise AssertionError("an unknown attribute must still raise")
print("ok")
"""


def test_importing_the_package_imports_neither_anthropic_nor_planlens():
    # In a fresh interpreter, because this suite has already imported both.
    done = subprocess.run([sys.executable, "-c", CODE], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_the_entry_points_reach_the_real_functions():
    import report_ingest
    from report_ingest.label_review import review_labels
    from report_ingest.triage import triage

    # Same function, reached lazily: the wrapper must not shadow or copy it.
    assert report_ingest.triage.__doc__ and triage.__doc__
    assert review_labels.__name__ == "review_labels"


def test_the_record_classes_are_reachable_from_the_package():
    import report_ingest
    from report_ingest.model import Investigation, ReportRecord

    assert report_ingest.ReportRecord is ReportRecord
    assert report_ingest.Investigation is Investigation
    assert report_ingest.SCHEMA_VERSION


def test_the_cost_meter_is_reachable_without_a_credential():
    import report_ingest

    meter = report_ingest.CostMeter()
    assert meter.calls == 0
    assert meter.to_dict()["dollars"] == 0.0
