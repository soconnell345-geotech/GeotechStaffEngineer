"""Whole-report ingest: the model passes that sit on top of planlens.

planlens turns a PDF into pages with kinds, located text, an outline and a
first draft of what each page IS. This package adds the geotechnical
semantics: what kind of document this is, which pages are really what, and
(later work packages) the record a reader fills in.

Two passes ship here today, both scored in
``module_work/report_ingest_harness/measure_wp1b.py``:

``report_ingest.triage``
    One structured call over the whole-document ledger, the outline, the
    front matter and the first contact sheet. Answers what the document is
    and which workflow it needs.
``report_ingest.label_review``
    An agent loop with four tools that checks the rule labels against the
    report's own account of itself and against the pages themselves.
``report_ingest.log_reader`` and ``report_ingest.lab_reader``
    One exploration log, and one laboratory sheet, read into the record.
    Geometry says where and the model says what; Python refuses what the
    page cannot support.
``report_ingest.diggs_writer``
    The record as real DIGGS 2.6, with the two gates on the file: the
    bundled XSD, and a round trip through the app's own readers.
``report_ingest.cluster_scoring``
    The run that produces the real numbers: the whole corpus through
    Funhouse's Prompter, on the cluster, scored against the hand labels.
    The app runs against OpenAI models there, so a score measured on any
    other model measures a model that will never do the work.

IMPORTS ARE LAZY ON PURPOSE. Importing this package pulls in nothing but the
standard library: the passes are reached through :func:`triage` and
:func:`review_labels` below, which import their modules on the first call,
and ``anthropic`` is imported only inside
:class:`report_ingest.engine.ClaudeEngine`. So the app never pays for this
package at startup, and ``anthropic`` stays an optional dependency the
cluster does not need -- on the cluster the same passes run against the
Prompter through the same one-method engine protocol.
"""

from __future__ import annotations

from typing import Any

__all__ = ["triage", "review_labels", "read_log", "read_lab_sheet",
           "write_diggs", "diggs_schema_gate", "diggs_roundtrip_gate",
           "ReportRecord", "Investigation", "LabTest", "score_on_cluster",
           "PrompterEngine", "ClaudeEngine", "CostMeter"]


def __getattr__(name: str) -> Any:
    """The record's own classes, imported on first use.

    ``model.py`` costs only pydantic, which the app already has, but importing
    it from here would still pull pydantic in at app startup for a package
    nothing has called yet. A module-level ``__getattr__`` keeps
    ``report_ingest.ReportRecord`` spelled the obvious way and still pays
    nothing until somebody asks for it.
    """
    if name in ("ReportRecord", "Investigation", "Quantity", "Provenance",
                "Layer", "Sample", "SPT", "WaterLevel", "LabTest", "Project",
                "QAEntry", "SCHEMA_VERSION", "record_json_schema",
                "LabKind", "LabResult", "RESULT_CLASS", "SievePoint",
                "AtterbergResult", "GradationResult", "ConsolidationPoint",
                "ConsolidationResult", "ShearPoint", "StrengthSpecimen",
                "StrengthResult", "CompactionPoint", "CompactionResult",
                "CBRResult", "MoistureDensityResult", "ChemicalResult",
                "SummaryRow", "SummaryTableResult", "OtherResult",
                "si_numbers"):
        from report_ingest import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def read_log(*args: Any, **kwargs: Any):
    """:func:`report_ingest.log_reader.read_log`, imported on first use.

    One exploration log -- continuation sheets included -- read into one
    :class:`~report_ingest.model.Investigation`.
    """
    from report_ingest.log_reader import read_log as _read_log
    return _read_log(*args, **kwargs)


def read_lab_sheet(*args: Any, **kwargs: Any):
    """:func:`report_ingest.lab_reader.read_lab_sheet`, on first use.

    One laboratory sheet -- a multi-page test included -- read into typed
    :class:`~report_ingest.model.LabTest` records.
    """
    from report_ingest.lab_reader import read_lab_sheet as _read
    return _read(*args, **kwargs)


def write_diggs(*args: Any, **kwargs: Any):
    """:func:`report_ingest.diggs_writer.write_diggs`, on first use."""
    from report_ingest.diggs_writer import write_diggs as _write
    return _write(*args, **kwargs)


def diggs_schema_gate(*args: Any, **kwargs: Any):
    """:func:`report_ingest.diggs_writer.diggs_schema_gate`, on first use."""
    from report_ingest.diggs_writer import diggs_schema_gate as _gate
    return _gate(*args, **kwargs)


def diggs_roundtrip_gate(*args: Any, **kwargs: Any):
    """:func:`report_ingest.diggs_writer.diggs_roundtrip_gate`, on first use."""
    from report_ingest.diggs_writer import diggs_roundtrip_gate as _gate
    return _gate(*args, **kwargs)


def score_on_cluster(*args: Any, **kwargs: Any):
    """:func:`report_ingest.cluster_scoring.score_on_cluster`, on first use.

    The run that produces the real numbers: the whole corpus through
    Prompter, scored, on the cluster.
    """
    from report_ingest.cluster_scoring import score_on_cluster as _score
    return _score(*args, **kwargs)


def PrompterEngine(*args: Any, **kwargs: Any):  # noqa: N802 - a class in effect
    """:class:`report_ingest.engine.PrompterEngine`, imported on first use."""
    from report_ingest.engine import PrompterEngine as _PrompterEngine
    return _PrompterEngine(*args, **kwargs)


def triage(*args: Any, **kwargs: Any):
    """:func:`report_ingest.triage.triage`, imported on first use."""
    from report_ingest.triage import triage as _triage
    return _triage(*args, **kwargs)


def review_labels(*args: Any, **kwargs: Any):
    """:func:`report_ingest.label_review.review_labels`, imported on use."""
    from report_ingest.label_review import review_labels as _review
    return _review(*args, **kwargs)


def ClaudeEngine(*args: Any, **kwargs: Any):  # noqa: N802 - a class in effect
    """:class:`report_ingest.engine.ClaudeEngine`, imported on first use.

    A function rather than a re-export so that importing this package never
    imports ``anthropic``.
    """
    from report_ingest.engine import ClaudeEngine as _ClaudeEngine
    return _ClaudeEngine(*args, **kwargs)


def CostMeter(*args: Any, **kwargs: Any):     # noqa: N802 - a class in effect
    """:class:`report_ingest.engine.CostMeter`, imported on first use."""
    from report_ingest.engine import CostMeter as _CostMeter
    return _CostMeter(*args, **kwargs)
