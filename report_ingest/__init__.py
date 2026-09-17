"""Whole-report ingest: a geotechnical report as organised, cited data.

planlens turns a PDF into pages with kinds, located text, an outline and a
first draft of what each page IS. This package adds the geotechnical
semantics: what kind of document this is, which pages are really what, what
each of them says, and the record that holds it.

``report_ingest.graph.ingest_report`` is the whole of it -- one PDF in, one
record and its exports out -- and the pieces it drives are:

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
``report_ingest.narrative_reader``
    The narrative read into the owner's two standing query schemas, with a
    citation on every answer and null wherever the report does not say.
``report_ingest.reconciler``
    The only pass that sees every reading at once: it links the laboratory
    tests to the ground, counts what the narrative claimed against what the
    appendix held, and RECORDS every disagreement rather than settling one.
``report_ingest.writers``
    The record, a summary page, a library page in the owner's WikiLLM format,
    a DIGGS file and a row in a SQLite index of many reports.
``report_ingest.graph`` and ``report_ingest.run_folder``
    The deterministic loop over one report, and the same loop headless over a
    folder of them into one library.
``report_ingest.subagent``
    The ingest as one delegation of the app's agent, returning a paragraph
    and five paths rather than the record.
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
           "read_narrative", "reconcile", "write_outputs", "ingest_report",
           "run_folder", "build_report_ingest_subagent",
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


def read_narrative(*args: Any, **kwargs: Any):
    """:func:`report_ingest.narrative_reader.read_narrative`, on first use.

    The narrative read into the owner's two query schemas, with a citation on
    every answer.
    """
    from report_ingest.narrative_reader import read_narrative as _read
    return _read(*args, **kwargs)


def reconcile(*args: Any, **kwargs: Any):
    """:func:`report_ingest.reconciler.reconcile`, imported on first use."""
    from report_ingest.reconciler import reconcile as _reconcile
    return _reconcile(*args, **kwargs)


def write_outputs(*args: Any, **kwargs: Any):
    """:func:`report_ingest.writers.write_outputs`, on first use.

    The record, the summary page, the library page, the DIGGS file and the
    library row.
    """
    from report_ingest.writers import write_outputs as _write
    return _write(*args, **kwargs)


def ingest_report(*args: Any, **kwargs: Any):
    """:func:`report_ingest.graph.ingest_report`, imported on first use.

    One PDF in, one record and its exports out.
    """
    from report_ingest.graph import ingest_report as _ingest
    return _ingest(*args, **kwargs)


def run_folder(*args: Any, **kwargs: Any):
    """:func:`report_ingest.run_folder.run_folder`, on first use.

    A folder of reports into one library.
    """
    from report_ingest.run_folder import run_folder as _run
    return _run(*args, **kwargs)


def build_report_ingest_subagent(*args: Any, **kwargs: Any):
    """:func:`report_ingest.subagent.build_report_ingest_subagent`, on use.

    A function rather than a re-export so that importing this package never
    imports LangGraph.
    """
    from report_ingest.subagent import (
        build_report_ingest_subagent as _build,
    )
    return _build(*args, **kwargs)


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
