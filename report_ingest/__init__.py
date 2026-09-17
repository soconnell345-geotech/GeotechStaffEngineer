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

__all__ = ["triage", "review_labels", "ClaudeEngine", "CostMeter"]


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
