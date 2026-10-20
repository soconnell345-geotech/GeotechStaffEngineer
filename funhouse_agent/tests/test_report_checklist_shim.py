"""``funhouse_agent.report_checklist`` is a deprecated re-export of
:mod:`report_ingest.checklist` (BACKLOG RI-34), kept until the two apps are
split (INF-26)."""

import importlib
import sys
import warnings


def test_the_old_import_still_works_and_says_where_it_went():
    from report_ingest import checklist as CL
    sys.modules.pop("funhouse_agent.report_checklist", None)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        old = importlib.import_module("funhouse_agent.report_checklist")
    assert any(issubclass(w.category, DeprecationWarning)
               and "report_ingest.checklist" in str(w.message) for w in seen)
    for name in old.__all__:
        assert getattr(old, name) is getattr(CL, name), name
    assert old.load_checklist() == CL.load_checklist()


def test_an_old_ledger_call_says_what_it_reads_now(tmp_path):
    """Before RI-34 a caller passed (coverage ledger, document key); the
    checks now read report ingest's record, so that call is refused by
    name rather than read as a path."""
    import pytest
    from funhouse_agent.coverage import CoverageLedger
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("funhouse_agent.report_checklist")
    ledger = CoverageLedger(path=str(tmp_path / "coverage.json"))
    with pytest.raises(TypeError, match="CoverageLedger"):
        old.run_checklist(ledger, "doc")
    with pytest.raises(TypeError, match="record"):
        old.failed_lines(ledger, "doc")
