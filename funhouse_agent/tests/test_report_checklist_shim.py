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
