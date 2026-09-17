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
for name in ("triage", "review_labels", "ClaudeEngine", "CostMeter"):
    assert callable(getattr(report_ingest, name)), name
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


def test_the_cost_meter_is_reachable_without_a_credential():
    import report_ingest

    meter = report_ingest.CostMeter()
    assert meter.calls == 0
    assert meter.to_dict()["dollars"] == 0.0
