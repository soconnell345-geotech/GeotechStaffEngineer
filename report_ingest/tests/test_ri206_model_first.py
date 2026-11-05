"""RI-206 under the default (model first): a resumed record counts each call
once, cache writes included, and the descriptions are not counted twice.
Scripted engines on the synthetic report; every number is made up."""

from __future__ import annotations

import json
import shutil

import pytest

pytest.importorskip("planlens.document.roles")
pytest.importorskip("report_ingest.locate")

from report_ingest.engine import Usage  # noqa: E402
from report_ingest.tests import test_graph_model_first as mf  # noqa: E402
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report)

CACHED = Usage(input_tokens=200, cache_write_tokens=3000,
               cache_read_tokens=700, output_tokens=100)


def _cached_engines():
    main, describer, labeller, stages = mf._engines()
    for engine in (main, describer, labeller, *stages.values()):
        engine.usage = CACHED
    return main, describer, labeller, stages


def test_a_resumed_model_first_run_counts_each_call_once(tmp_path):
    pdf = tmp_path / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    first = mf._run_on(pdf, tmp_path / "first", _cached_engines())
    assert first.document.input_tokens == first.document.model_calls * 3900

    shutil.copytree(tmp_path / "first", tmp_path / "again")
    main, describer, labeller, stages = _cached_engines()
    for engine in (main, describer, labeller, *stages.values()):
        engine.turns = []
    # The laboratory window was not saved: it is asked again, twice, as the
    # first run asked it, so the whole report's count is the first run's.
    stages["lab"].turns = [{"text": "Still no answer."}] * 2
    again = mf._run_on(pdf, tmp_path / "again",
                       (main, describer, labeller, stages))
    took = again.document.resumed
    assert "descriptions" in took.steps and "ranges" in took.steps
    assert took.report_calls == first.document.model_calls
    assert took.report_input_tokens == first.document.input_tokens
    ranges = json.loads((tmp_path / "again" / "labels.json")
                        .read_text("utf-8"))["cost"]["ranges"]
    assert ranges["calls"] == 1                    # the labeller's one call


def test_a_page_description_keeps_its_cache_writes(tmp_path):
    pdf = tmp_path / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    mf._run_on(pdf, tmp_path, _cached_engines())
    pages = sorted((tmp_path / "descriptions").glob("*.json"))
    assert len(pages) == 22
    for path in pages:
        cost = json.loads(path.read_text("utf-8"))["cost"]
        assert cost["cache_write_tokens"] == cost["calls"] * 3000, path.name
