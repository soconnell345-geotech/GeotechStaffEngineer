"""GeotechStaffEngineer's part of the shared core, handed in through hooks
(BACKLOG INF-32, the app split's step 11 (c); ``aec_agent_core.hooks``).

The core imports nothing of GSE (plan ``module_work/APP_SPLIT_PLAN.md``
section 2.7 B). Where a core module needs to know something only GSE knows,
it asks for a hook, and ``funhouse_agent/__init__.py`` fills it with a
function here (as ``"module:attribute"`` strings, so nothing loads until a
hook is first asked for):

- ``held_records.record_fields`` -> :func:`record_fields`: the shapes of an
  exploration and a laboratory-test record (``report_ingest.model``), which
  ``aec_agent_core.deep.held_records`` finds in a stopped helper's words;
- ``cited_documents.reference_modules`` -> :func:`reference_modules`: the
  dispatch's reference modules, whose ``call_agent`` answers are reference
  sources for ``aec_agent_core.cited_documents``.

The review suite's hooks (its tasks, documents and the page's turn) are
filled by ``funhouse_agent/review_eval/__init__.py``.
"""

from __future__ import annotations

from typing import Tuple


def record_fields() -> Tuple[frozenset, frozenset, frozenset, frozenset]:
    """(exploration fields, lab-test fields, exploration kinds, lab kinds),
    read off ``report_ingest.model`` (what ``held_records._fields`` read
    before step 11 (c)). Raises ImportError where report_ingest is missing;
    the core then holds no records, as before."""
    from typing import get_args

    from report_ingest.model import (Investigation, InvestigationKind,
                                     LabKind, LabTest)
    return (frozenset(Investigation.model_fields),
            frozenset(LabTest.model_fields),
            frozenset(get_args(InvestigationKind)),
            frozenset(get_args(LabKind)))


def reference_modules() -> frozenset:
    """The dispatch's reference modules (``funhouse_agent.dispatch.
    REFERENCE_MODULES``)."""
    from funhouse_agent.dispatch import REFERENCE_MODULES
    return frozenset(REFERENCE_MODULES)
