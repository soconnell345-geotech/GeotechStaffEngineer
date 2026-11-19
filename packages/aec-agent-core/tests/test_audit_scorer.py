"""The scorer audit of 2026-11-18 (module_work/audit_2026_11_18/SCORER.md):
one test per fix made in the runner. Made-up values only."""

import json
import os

from aec_agent_core.review_eval import runner
from aec_agent_core.review_eval.tasks import Task


def test_rescore_reads_files_saved_with_the_other_path_separator(tmp_path):
    """A run saved on Windows names its files "files\\\\log.md". Re-scored
    on another system the path did not resolve, the delivered file was
    skipped without a word, and the check that reads it failed."""
    run_dir = tmp_path / "runs" / "baseline" / "made-up"
    (run_dir / "files").mkdir(parents=True)
    (run_dir / "files" / "log.md").write_text(
        "Door 106 is on the plan and not in the schedule.", encoding="utf-8")
    task = Task(id="made-up", question="?", documents=[], category="check",
                doc_type="made_up", auto_checks=False,
                checks=[{"type": "contains_all", "with_files": True,
                         "terms": ["106", "not in the schedule"]}])
    here = {"answer": "See the log.", "files": [os.path.join("files", "log.md")]}
    there = {"answer": "See the log.", "files": ["files\\log.md"]}
    want = runner.rescore_saved(here, task, str(run_dir))["score"]
    got = runner.rescore_saved(there, task, str(run_dir))["score"]
    assert want["passed"], want
    assert got == want, json.dumps(got)
