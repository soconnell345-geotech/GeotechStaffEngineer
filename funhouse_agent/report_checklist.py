"""Deprecated: the report-review checklist lives in :mod:`report_ingest.checklist`.

BACKLOG RI-34 (owner, 2026-10-20): every question a geotechnical report is
asked lives in ``report_ingest``. The checklist moved there with its data
(``report_ingest/report_review_checklist.json``, still DRAFT for OWN-31),
and its code checks now read the record report ingest writes
(``ReportRecord``) instead of this app's coverage ledger, the PDF's text
layer and ``write_diggs`` cross-checks. The tool is offered on the
GeotechStaffEngineer page only, behind ``GEOTECH_REVIEW_CHECKLIST``.

This module re-exports it for code that imports the old name, and goes when
the two apps are split (BACKLOG INF-26). What changed for a caller:

* :func:`run_checklist` and :func:`failed_lines` take the record (a
  ``ReportRecord``, its dict, or the path of ``report.record.json`` or of
  the folder report ingest wrote) instead of ``(ledger, key)``;
* ``exploration_ids`` (exploration ids read out of a page's text) is gone:
  the record carries the explorations the narrative names and the logs read.
"""

from __future__ import annotations

import warnings

from report_ingest.checklist import (  # noqa: F401 - re-exported
    CHECKS, DATA_FILE, STATUSES, CheckContext, failed_lines, load_checklist,
    make_checklist_tool, register_check, run_checklist,
)

warnings.warn(
    "funhouse_agent.report_checklist is deprecated: import "
    "report_ingest.checklist (BACKLOG RI-34)", DeprecationWarning,
    stacklevel=2)

__all__ = ["load_checklist", "run_checklist", "failed_lines",
           "register_check", "make_checklist_tool", "CHECKS", "CheckContext",
           "DATA_FILE", "STATUSES"]
