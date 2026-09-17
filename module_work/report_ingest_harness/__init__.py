"""Dev-only harness over the private report-ingest corpus (WP0).

Not part of the shipped wheel: ``pyproject.toml``'s
``[tool.setuptools.packages.find]`` names every package it includes and
``module_work`` is not one of them, so this folder is source-tree tooling
like ``module_work/drawing_ground_truth/doc_claims_check.py``.

Run the measurement scripts from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp0

Everything the harness reads lives under
``module_work/field_feedback/2026-09-16_report-ingest/raw/``, which is
gitignored. Reports are named by ID (R01-R38) in everything this package
prints; the manifest's file stems are loaded but never displayed.
"""

from module_work.report_ingest_harness.corpus import (  # noqa: F401
    RAW_DIR,
    ReportInfo,
    has_di,
    list_reports,
    load_di,
    open_report,
    pdf_path,
    raw_available,
    report,
)
from module_work.report_ingest_harness.labels import (  # noqa: F401
    LABELS,
    RAW_TO_LABEL,
    PageLabel,
    label_counts,
    labels_for,
    sheet_map,
)
