"""RI-206: a resumed record states what the report's tokens in were.

A reader's own cost leaves out its cache writes; the run topped its count up
from the calls it saw, but the item file it saved kept the short count, so a
resumed run's whole-report figure came out smaller than what was paid. The
saved file now carries every part of what the step paid. Scripted engines on
the synthetic report; every number is made up.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("planlens.document.roles")

from report_ingest.engine import Usage  # noqa: E402
from report_ingest.graph import ingest_report  # noqa: E402
from report_ingest.tests import test_graph as tg  # noqa: E402
from report_ingest.tests.fake_engine import FakeEngine  # noqa: E402
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report)
# Today's readers are scripted: model first is set off (RI-186).
from report_ingest.tests.test_graph import (  # noqa: E402,F401
    todays_readers)

CACHED = Usage(input_tokens=200, cache_write_tokens=3000,
               cache_read_tokens=700, output_tokens=100)


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("ri206") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


def test_a_resumed_report_counts_the_cache_writes_it_paid(pdf, tmp_path):
    first = ingest_report(pdf, FakeEngine(tg.full_script(), usage=CACHED),
                          out_dir=tmp_path, report_id="SYN")
    paid = first.document.input_tokens
    # Every call carried its cache writes and reads: all of them are paid.
    assert paid == first.document.model_calls * 3900

    again = ingest_report(pdf, FakeEngine([]), out_dir=tmp_path,
                          report_id="SYN")
    took = again.document.resumed
    assert took.report_calls == first.document.model_calls
    assert took.report_input_tokens == paid
    assert took.saved_input_tokens == paid

    written = json.loads((tmp_path / "report.record.json")
                         .read_text(encoding="utf-8"))["document"]
    assert written["resumed"]["report_input_tokens"] == paid
    summary = (tmp_path / "report.summary.md").read_text(encoding="utf-8")
    assert f"{paid:,} input and" in summary


def test_an_item_file_keeps_every_part_of_what_it_paid(pdf, tmp_path):
    ingest_report(pdf, FakeEngine(tg.full_script(), usage=CACHED),
                  out_dir=tmp_path, report_id="SYN")
    for path in sorted((tmp_path / "items").glob("*.json")):
        cost = json.loads(path.read_text(encoding="utf-8"))["cost"]
        calls = int(cost["calls"])
        assert cost["cache_write_tokens"] == calls * 3000, path.name
        assert cost["cache_read_tokens"] == calls * 700, path.name
        assert cost["input_tokens"] == calls * 200, path.name
