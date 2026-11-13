"""The report tables (BACKLOG RI-209): one ingest's results as flat, keyed
rows a pipeline reads one step per table, built from the saved outputs.

Two kinds of report go through it here. The synthetic report goes through
the whole ingest with a scripted engine, once for the module, so the export
is the one a real run writes (a report with another bound inside it). A
made-up record written by the writers, with page votes, descriptions and
answers set by hand, shows each column filled from what the folder holds.
Every value in this file is invented.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from report_ingest import tables
from report_ingest.model import (
    GENERAL_FIELDS, NATURAL_HAZARD_FIELDS, Citation, LabelVote, Mention,
    PageLabel, QAEntry, ReportRecord,
)
from report_ingest.tests.record_fixtures import build_record
from report_ingest.writers import record_key, write_outputs

pytest.importorskip("planlens.document.roles")

from report_ingest.tests.test_graph import (  # noqa: E402
    full_script, triage_turn,
)

_ENVS = (tables.ENV_ROOT, tables.ENV_DATASET, tables.LIBRARY_ENV,
         "GSE_REPORT_QUESTIONS", "GSE_REPORT_POSTS", "GEOTECH_DEPLOYMENT")


def _plain(mp: pytest.MonkeyPatch) -> None:
    from report_ingest.graph import MODEL_FIRST_ENV

    mp.setenv(MODEL_FIRST_ENV, "off")          # today's readers, scripted
    for name in _ENVS:
        mp.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def plain_settings(monkeypatch):
    _plain(monkeypatch)


def _pdf_bytes() -> bytes:
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    return build_narrative_report().pdf


@dataclass
class Ingested:
    base: Path
    pdf: Path
    out: Path
    record: ReportRecord

    @property
    def v1(self) -> Path:
        return self.out / "report_tables" / "v1"


@pytest.fixture(scope="module")
def ingested(tmp_path_factory) -> Ingested:
    """One standard run of the synthetic report (a report bound in at pages
    15-18), with the export on by default."""
    from report_ingest.graph import ingest_report
    from report_ingest.tests.fake_engine import FakeEngine

    base = tmp_path_factory.mktemp("ingested")
    pdf = base / "made_up_report.pdf"
    pdf.write_bytes(_pdf_bytes())
    out = base / "out"
    with pytest.MonkeyPatch.context() as mp:
        _plain(mp)
        record = ingest_report(pdf, FakeEngine(full_script()), out_dir=out,
                               report_id="SYN")
    return Ingested(base=base, pdf=pdf, out=out, record=record)


def _rows(v1: Path, table: str, name: str) -> List[Dict[str, Any]]:
    path = v1 / table / (name if name.endswith(".jsonl") else
                         f"{name}.jsonl")
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]


def _keys(ingested: Ingested):
    run = json.loads((ingested.out / tables.RUN_NAME).read_text("utf-8"))
    parent = run["report_key"]
    child = run["export"]["keys"][1]
    return parent, child


def _needs_person(pdf: Any, out: Path, **kwargs) -> ReportRecord:
    """The cheapest whole ingest: triage sends the report to a person."""
    from report_ingest.graph import ingest_report
    from report_ingest.tests.fake_engine import FakeEngine

    return ingest_report(pdf, FakeEngine([triage_turn("needs_person")]),
                         out_dir=out, report_id="SYN", **kwargs)


def _made_up_pdf(tmp_path: Path, name: str = "made_up.pdf") -> Path:
    pdf = tmp_path / name
    if not pdf.exists():
        pdf.write_bytes(b"%PDF-1.7\n" + b"made up " * 900)
    return pdf


def _voted_record() -> ReportRecord:
    """The made-up record with page votes, a printed page number, a
    yes/no answer with its sentence and a QA entry on one answer."""
    record = build_record()
    record.page_labels = [
        PageLabel(page=0, label="cover", confidence=0.9, agreed=True,
                  policy="structural", voters=[
                      LabelVote(voter="rules", label="cover",
                                confidence=0.9),
                      LabelVote(voter="vision", label="cover",
                                confidence=0.8)]),
        PageLabel(page=1, label="narrative", confidence=0.0, agreed=True,
                  policy="structural", voters=[
                      LabelVote(voter="ranges", label="narrative")]),
        PageLabel(page=2, label="boring_log", confidence=0.7, agreed=False,
                  policy="structural", settled_by="review", voters=[
                      LabelVote(voter="rules", label="figure",
                                confidence=0.4),
                      LabelVote(voter="template",
                                family="made-up form A", confidence=0.7),
                      LabelVote(voter="ranges", label="boring_log")]),
    ]
    record.document.printed_pages = {1: "i", 2: "3"}
    record.natural_hazards.soilCorrosion = "The soils are mildly corrosive."
    record.natural_hazards.soilCorrosionAnswer = Mention(
        answer="yes", citation=[Citation(page=2, quote="mildly corrosive")])
    record.general.outsideProject = "no"
    record.qa.append(QAEntry(kind="count_mismatch",
                             where="general.boringCount",
                             detail="a made-up count disagreement"))
    return record


def _write_folder(tmp_path: Path, record: ReportRecord = None,
                  name: str = "R1"):
    """A made-up record written by the writers as an ingest writes it, with
    a labels.json, one page description and one model-first item file."""
    pdf = _made_up_pdf(tmp_path)
    folder = tmp_path / "run" / name
    written = write_outputs(record or _voted_record(), folder, source=pdf,
                            db_path=tmp_path / "run" / "reports.db")
    (folder / "labels.json").write_text(json.dumps({"pages": [
        {"page": 0, "label": "cover", "why": "structural: rules, vision"},
        {"page": 2, "label": "boring_log", "why": "review: boring_log"}]}),
        encoding="utf-8")
    (folder / "descriptions").mkdir()
    (folder / "descriptions" / "p0002.json").write_text(json.dumps({
        "page": 2, "pdf_page": 3, "model": "made-up-describer",
        "prompt": "made-up-describe-1", "description": {
            "title": "Log of Boring B-1", "kind": "boring log",
            "names": [], "identifiers": ["B-1"], "exploration": "B-1",
            "continues": False, "header_footer": "", "issuer_date": "",
            "language": "English",
            "summary": "A made-up boring log to 30 ft."}}), encoding="utf-8")
    (folder / "items").mkdir(exist_ok=True)
    (folder / "items" / "mf-narrative.json").write_text(json.dumps({
        "model": "made-up-reader", "prompt_version": "mf-10"}),
        encoding="utf-8")
    return folder, pdf, written


def _answers(rows: List[Dict[str, Any]], question_set: str):
    return {r["question_key"]: r for r in rows
            if r["question_set"] == question_set}


# ---------------------------------------------------------------------------
# the definitions
# ---------------------------------------------------------------------------

class TestTheDefinitions:

    def test_every_column_has_a_known_type_and_says_what_it_holds(self):
        for table in tables.TABLES:
            names = [c.name for c in table.columns]
            assert len(names) == len(set(names)), table.name
            assert names[:2] == ["export_version", "report_key"]
            for column in table.columns:
                assert column.type in tables.TYPES, (table.name, column)
                assert column.description.strip(), (table.name, column)

    def test_the_keys_and_links_name_real_columns(self):
        for table in tables.TABLES:
            assert set(table.primary_key) <= set(table.column_names)
            for column in table.primary_key:
                spec = next(c for c in table.columns if c.name == column)
                assert not spec.nullable, (table.name, column)
            for link in table.links:
                target = tables.TABLE[link.table]
                assert set(link.columns) <= set(table.column_names)
                assert set(link.references) <= set(target.column_names)

    def test_schema_json_is_the_definitions(self, ingested):
        written = json.loads((ingested.v1 / "schema.json")
                             .read_text("utf-8"))

        assert written == tables.schema()
        for table, spec in zip(tables.TABLES, written["tables"]):
            assert spec["name"] == table.name
            assert [(c["name"], c["type"]) for c in spec["columns"]] == \
                [(c.name, c.type) for c in table.columns]
            assert spec["primary_key"] == list(table.primary_key)

    def test_a_row_is_every_column_in_order_with_its_nulls(self):
        row = tables.make_row(tables.TABLE["report_page"], {
            "report_key": "0123456789abcdef", "pdf_page": 3,
            "file_report_key": "0123456789abcdef", "label": "figure",
            "agreed": True, "settled_by": "vote"})

        assert tuple(row) == tables.TABLE["report_page"].column_names
        assert row["export_version"] == "v1"
        assert row["label_confidence"] is None and row["title"] is None

    @pytest.mark.parametrize("values, said", [
        ({"made_up_column": 1}, "has no column"),
        ({"pdf_page": "3"}, "is not a integer"),
        ({"label": None}, "may not be null"),
        ({"label_confidence": "high"}, "is not a double"),
    ])
    def test_a_row_refuses_what_its_table_does_not_hold(self, values, said):
        given = {"report_key": "0123456789abcdef", "pdf_page": 3,
                 "file_report_key": "0123456789abcdef", "label": "figure",
                 "agreed": True, "settled_by": "vote", **values}
        with pytest.raises(ValueError, match=said):
            tables.make_row(tables.TABLE["report_page"], given)

    def test_dates_and_times_are_checked_and_a_whole_double_is_a_float(self):
        table = tables.TABLE["ingest_run"]
        given = dict(report_key="0123456789abcdef", run_id="r",
                     ingested_at="2026-01-02T03:04:05Z", n_pages=1,
                     n_investigations=0, n_lab_tests=0, n_calculations=0,
                     n_answers=0, n_answers_answered=0, n_qa_open=0,
                     n_qa_changed=0, n_qa_note=0, diggs_written=False,
                     record_schema_version="4.0", prompt_versions=[],
                     models=[], model_calls=0, input_tokens=0,
                     output_tokens=0, dollars=0, seconds=2, resumed=False)

        row = tables.make_row(table, given)
        assert row["dollars"] == 0.0 and isinstance(row["dollars"], float)
        with pytest.raises(ValueError, match="timestamp"):
            tables.make_row(table, {**given,
                                    "ingested_at": "2026-01-02 03:04:05"})
        with pytest.raises(ValueError, match="date"):
            tables.make_row(tables.TABLE["report_answer"], {
                "report_key": "k", "question_set": "general",
                "question_key": "reportDate", "position": 1,
                "pdf_pages": [], "quotes": [], "qa_kinds": [],
                "value_date": "2026-02-30"})

    def test_the_document_lists_the_columns_the_code_defines(self):
        doc = (Path(__file__).resolve().parents[2] / "docs" /
               "REPORT_TABLES.md").read_text("utf-8").replace("\r\n", "\n")
        start = doc.index("<!-- columns:")
        section = doc[doc.index("\n", start) + 1:
                      doc.index("<!-- end columns -->")]

        assert section.strip() == tables.columns_markdown().strip(), \
            "run python -m report_ingest.tables --columns and paste it in"

    def test_every_exported_row_fits_its_table(self, ingested):
        seen = set()
        for table in tables.TABLES:
            files = sorted((ingested.v1 / table.name).glob("*.jsonl"))
            assert files, table.name
            for path in files:
                for line in path.read_text("utf-8").splitlines():
                    tables.check_row(table, json.loads(line))
                    seen.add(table.name)
        assert seen == {t.name for t in tables.TABLES}


# ---------------------------------------------------------------------------
# a whole ingest
# ---------------------------------------------------------------------------

class TestAnIngest:

    def test_the_export_goes_beside_the_lone_reports_library(self, ingested):
        outcome = tables.export_outcome(ingested.out)

        assert outcome["status"] == "written"
        assert (ingested.v1 / "schema.json").is_file()
        assert outcome["rows"]["report"] == 2           # it and its child
        assert outcome["delivery"] == ""                # no dataset named

    def test_the_key_is_the_librarys_own(self, ingested):
        parent, child = _keys(ingested)
        page = (ingested.out / "report.page.md").read_text("utf-8")
        connection = sqlite3.connect(ingested.out / "reports.db")
        ids = {r[0]: (r[1], r[2]) for r in connection.execute(
            "SELECT report_id, id, parent FROM reports")}
        connection.close()

        assert f'id: "{parent}"' in page or f"id: {parent}" in page
        assert ids["SYN"][0] == parent
        assert ids["SYN.bound1"] == (child, parent)

    def test_the_source_file_is_named_sized_and_hashed(self, ingested):
        parent, child = _keys(ingested)
        whole = hashlib.sha256(ingested.pdf.read_bytes()).hexdigest()

        for key in (parent, child):
            (row,) = _rows(ingested.v1, "report", key)
            assert row["source_name"] == "made_up_report.pdf"
            assert row["source_bytes"] == ingested.pdf.stat().st_size
            assert row["source_sha256"] == whole

    def test_the_pages_count_from_one(self, ingested):
        parent, _child = _keys(ingested)
        rows = _rows(ingested.v1, "report_page", parent)
        labels = {p.page: p for p in ingested.record.page_labels}

        assert [r["pdf_page"] for r in rows] == \
            list(range(1, ingested.record.document.n_pages + 1))
        for row in rows:
            label = labels[row["pdf_page"] - 1]
            assert row["label"] == label.label
            assert row["file_report_key"] == parent
            assert row["label_rules"] and row["label_vision"]
            assert row["reason"]                        # from labels.json

    def test_the_bound_report_is_its_own_report_in_its_parents_file(
            self, ingested):
        parent, child = _keys(ingested)
        (top,) = _rows(ingested.v1, "report", parent)
        (bound,) = _rows(ingested.v1, "report", child)
        span = ingested.record.bound_documents[0]
        pages = list(range(span.first_page + 1, span.last_page + 2))

        assert top["bound_report_keys"] == [child]
        assert top["parent_report_key"] is None
        assert bound["parent_report_key"] == parent
        assert bound["bound_id"] == "bound1"
        assert [r["pdf_page"] for r in
                _rows(ingested.v1, "report_page", child)] == pages
        assert {r["file_report_key"] for r in
                _rows(ingested.v1, "report_page", child)} == {parent}
        marked = {r["pdf_page"]: r["bound_report_key"]
                  for r in _rows(ingested.v1, "report_page", parent)}
        assert {p for p, k in marked.items() if k == child} == set(pages)
        assert {k for p, k in marked.items() if p not in pages} == {None}

    def test_every_built_in_question_is_a_row(self, ingested):
        parent, _child = _keys(ingested)
        rows = _rows(ingested.v1, "report_answer", parent)
        record = ingested.record

        for question_set, names in (("general", GENERAL_FIELDS),
                                    ("natural_hazards",
                                     NATURAL_HAZARD_FIELDS)):
            got = [r for r in rows if r["question_set"] == question_set]
            assert [r["question_key"] for r in got] == list(names)
            assert [r["position"] for r in got] == \
                list(range(1, len(names) + 1))
            facts = getattr(record, question_set)
            for row in got:
                said = getattr(facts, row["question_key"]) is not None
                assert row["status"] == ("answered" if said
                                         else "not said"), row
                if not said:
                    assert row["value_text"] is None
                    assert row["value_json"] is None
        (report,) = _rows(ingested.v1, "report", parent)
        assert report["n_answers"] == 38
        assert report["n_answers_answered"] == \
            sum(1 for r in rows if r["status"] == "answered")

    def test_citations_become_pages_from_one(self, ingested):
        parent, _child = _keys(ingested)
        row = _answers(_rows(ingested.v1, "report_answer", parent),
                       "general")["boringCount"]
        cited = ingested.record.general.citations["boringCount"]

        assert row["pdf_pages"] == sorted({c.page + 1 for c in cited})
        assert row["quotes"] == [c.quote for c in cited]
        assert row["value_number"] == 4.0 and row["value_json"] == "4"

    def test_the_diggs_verdicts_and_the_files_delivered(self, ingested):
        parent, _child = _keys(ingested)
        (row,) = _rows(ingested.v1, "report", parent)
        files = _rows(ingested.v1, "report_file", parent)

        assert row["diggs_written"] is True
        assert row["diggs_schema_valid"] is True
        assert row["diggs_roundtrip_ok"] is True
        assert {f["kind"] for f in files} == {"diggs", "record", "summary"}
        for entry in files:
            local = ingested.v1 / entry["path"]
            assert hashlib.sha256(local.read_bytes()).hexdigest() == \
                entry["sha256"]
            assert local.stat().st_size == entry["bytes"]

    def test_the_run_is_a_row_of_its_own(self, ingested):
        parent, _child = _keys(ingested)
        (report,) = _rows(ingested.v1, "report", parent)
        run_files = sorted((ingested.v1 / "ingest_run").glob(
            f"{parent}.*.jsonl"))

        assert [p.name for p in run_files] == \
            [f"{parent}.{report['latest_run_id']}.jsonl"]
        (run,) = _rows(ingested.v1, "ingest_run", run_files[0].name)
        assert run["model_calls"] == ingested.record.document.model_calls
        assert run["models"]                           # the scripted one
        assert run["resumed"] is False
        assert run["record_schema_version"] == \
            ingested.record.schema_version
        assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$", run["run_id"])

    def test_no_path_of_this_machine_is_in_any_row(self, ingested):
        # The run's own folders and this machine's user folder.
        names = {ingested.base.name, ingested.base.parent.name,
                 Path.home().name}
        for path in ingested.v1.rglob("*.jsonl"):
            text = path.read_text("utf-8")
            assert str(ingested.base) not in text
            assert str(ingested.base).replace(os.sep, "/") not in text
            for row in map(json.loads, text.splitlines()):
                for value in row.values():
                    for word in (value if isinstance(value, list)
                                 else [value]):
                        if not isinstance(word, str):
                            continue
                        assert not re.search(r"[A-Za-z]:[\\/]", word), word
                        assert not any(n in word for n in names
                                       if len(n) > 3), word


# ---------------------------------------------------------------------------
# the export adds files and changes none of the ingest's own
# ---------------------------------------------------------------------------

class _StandingClock:
    """``time`` for the graph, standing still: a run's seconds are 0."""

    @staticmethod
    def time() -> float:
        return 1000.0


_TIME = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")
_CREATION_DATE = re.compile(rb"<creationDate>\d{4}-\d{2}-\d{2}</creationDate>")


def _folder_files(out: Path) -> Dict[str, bytes]:
    """Every file under ``out`` by its path there, with what two runs of
    one graph cannot share taken out: the run's own folder, the library's
    clock and the DIGGS file's creation date (and a library database as
    its SQL dump, not its pages)."""
    files: Dict[str, bytes] = {}
    names = {str(out), out.as_posix()}
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".db":
            connection = sqlite3.connect(path)
            blob = "\n".join(connection.iterdump()).encode("utf-8")
            connection.close()
        else:
            blob = path.read_bytes()
        for text in sorted(names | {json.dumps(n)[1:-1] for n in names},
                           key=len, reverse=True):
            blob = blob.replace(text.encode("utf-8"), b"<RUN>")
        blob = _CREATION_DATE.sub(b"<DATE>", blob.replace(b"\r\n", b"\n"))
        files[path.relative_to(out).as_posix()] = _TIME.sub(b"<TIME>", blob)
    return files


def _of_the_export(rel: str) -> bool:
    return rel.startswith("report_tables/") or \
        rel.rsplit("/", 1)[-1] == tables.RUN_NAME


@pytest.fixture(scope="module")
def three_runs(tmp_path_factory) -> Dict[str, Dict[str, bytes]]:
    """The synthetic report (a report bound inside it) ingested three times
    off ONE file, on a standing clock: twice with the export on, once with
    ``GSE_REPORT_TABLES=0``."""
    from report_ingest import graph
    from report_ingest.tests.fake_engine import FakeEngine

    base = tmp_path_factory.mktemp("three_runs")
    pdf = base / "made_up_report.pdf"
    pdf.write_bytes(_pdf_bytes())       # once: the bytes carry their time
    runs: Dict[str, Dict[str, bytes]] = {}
    for name, setting in (("on_a", None), ("on_b", None), ("off", "0")):
        with pytest.MonkeyPatch.context() as mp:
            _plain(mp)
            if setting is not None:
                mp.setenv(tables.ENV_ROOT, setting)
            mp.setattr(graph, "time", _StandingClock)
            graph.ingest_report(pdf, FakeEngine(full_script()),
                                out_dir=base / name, report_id="SYN")
        runs[name] = _folder_files(base / name)
    return runs


class TestTheIngestsOwnFiles:

    def test_with_the_export_off_nothing_of_it_is_written(self, three_runs):
        off = three_runs["off"]

        assert not [rel for rel in off if _of_the_export(rel)]
        assert "report.record.json" in off
        assert "bound/bound1/report.record.json" in off

    def test_two_runs_with_the_export_on_leave_the_same_files_as_off(
            self, three_runs):
        on_a, on_b, off = (three_runs[n] for n in ("on_a", "on_b", "off"))
        own_a = {rel: blob for rel, blob in on_a.items()
                 if not _of_the_export(rel)}
        own_b = {rel: blob for rel, blob in on_b.items()
                 if not _of_the_export(rel)}

        assert sorted(own_a) == sorted(off)
        assert own_a == off
        assert own_b == off
        # What the export adds, and nothing else: its folder and a run
        # file beside each record; the run files differ between runs.
        added = set(on_a) - set(off)
        assert added == {rel for rel in on_a if _of_the_export(rel)}
        assert {tables.RUN_NAME, f"bound/bound1/{tables.RUN_NAME}",
                "report_tables/v1/schema.json"} <= added
        assert on_a[tables.RUN_NAME] != on_b[tables.RUN_NAME]


# ---------------------------------------------------------------------------
# a made-up record: each column from what the folder holds
# ---------------------------------------------------------------------------

class TestEachColumn:

    @pytest.fixture
    def exported(self, tmp_path):
        folder, pdf, written = _write_folder(tmp_path)
        outcome = tables.export_report(folder, tmp_path / "root")
        assert outcome.status == "written", outcome.detail
        return outcome, Path(outcome.folder), written

    def test_the_voters_and_the_description_of_each_page(self, exported):
        _outcome, v1, written = exported
        rows = {r["pdf_page"]: r for r in _rows(v1, "report_page",
                                                written.key)}

        assert rows[1]["label_confidence"] == 0.9
        assert (rows[1]["label_rules"], rows[1]["label_vision"]) == \
            ("cover", "cover")
        assert rows[1]["reason"] == "structural: rules, vision"
        # The section labeller states no confidence: null, not 0.
        assert rows[2]["label_confidence"] is None
        assert rows[2]["label_ranges"] == "narrative"
        assert rows[2]["printed_page"] == "i"
        assert rows[2]["reason"] is None and rows[2]["title"] is None
        assert rows[3]["template_family"] == "made-up form A"
        assert (rows[3]["label_rules"], rows[3]["label_ranges"]) == \
            ("figure", "boring_log")
        assert rows[3]["settled_by"] == "review" and not rows[3]["agreed"]
        assert (rows[3]["title"], rows[3]["kind"], rows[3]["exploration"]) \
            == ("Log of Boring B-1", "boring log", "B-1")
        assert rows[3]["summary"] == "A made-up boring log to 30 ft."

    def test_the_typed_twins_give_the_typed_columns(self, exported):
        _outcome, v1, written = exported
        rows = _rows(v1, "report_answer", written.key)
        general = _answers(rows, "general")
        hazards = _answers(rows, "natural_hazards")

        bearing = general["bearingCapacity"]
        assert bearing["value_list"] == \
            ["3,000 psf allowable for spread footings"]
        (typed,) = json.loads(bearing["value_json"])
        assert typed["value"]["value"] == 3000.0
        assert "citation" not in typed and "prov" not in typed["value"]
        assert bearing["pdf_pages"] == [5]               # page 4, from 0
        assert hazards["siteClass"]["value_json"] == '"D"'
        assert hazards["siteClass"]["pdf_pages"] == [5]
        assert hazards["reportDate"]["value_date"] == "2026-03-14"
        assert hazards["reportDate"]["value_text"] == "14 March 2026"
        corrosion = hazards["soilCorrosion"]
        assert corrosion["value_boolean"] is True
        assert corrosion["value_json"] == '"yes"'
        assert (corrosion["pdf_pages"], corrosion["quotes"]) == \
            ([3], ["mildly corrosive"])
        assert general["outsideProject"]["value_boolean"] is False

    def test_the_kinds_and_the_sources_of_the_answers(self, exported):
        _outcome, v1, written = exported
        general = _answers(_rows(v1, "report_answer", written.key),
                           "general")

        assert general["boringCount"]["qa_kinds"] == ["count_mismatch"]
        assert general["boringCount"]["data_type"] == "integer"
        assert general["recommendedFoundations"]["data_type"] == \
            "list of strings"
        assert general["boringCount"]["prompt_version"] == "mf-10"
        silent = general["primeContractor"]
        assert silent["status"] == "not said"
        assert [silent[c] for c in ("value_text", "value_number",
                                    "value_json", "value_list")] == \
            [None] * 4
        assert silent["pdf_pages"] == [] and silent["question_set"] == \
            "general"

    def test_the_run_names_the_models_and_versions_it_finds(self, exported):
        _outcome, v1, written = exported
        (run_file,) = (v1 / "ingest_run").glob(f"{written.key}.*.jsonl")
        (run,) = _rows(v1, "ingest_run", run_file.name)

        assert {"made-up-describer", "made-up-reader"} <= set(run["models"])
        assert {"mf-10", "made-up-describe-1"} <= \
            set(run["prompt_versions"])
        assert run["planlens_version"] == "test"       # off the record


# ---------------------------------------------------------------------------
# the owner's question list
# ---------------------------------------------------------------------------

def _question_list():
    from report_ingest import question_list as ql

    def q(row, key, data_type, category="Site"):
        return ql.Question(row=row, output=key.replace("_", " ").title(),
                           category=category, data_type=data_type,
                           allow_null=True, prompt=f"Made-up {key}?",
                           key=key, answered_by="Model")

    qlist = ql.QuestionList(questions=[
        q(2, "site_name", "string"),
        q(3, "has_fill", "boolean"),
        q(4, "hole_count", "integer"),
        q(5, "field_date", "date"),
        q(6, "soil_kinds", "list of strings"),
        q(7, "wells", "list of records"),
        q(8, "depths", "dictionary"),
        q(9, "rock_cored", "boolean", category="Screening"),
        q(10, "pile_type", "string"),
        q(11, "lost_row", "string"),
    ], name="made_up_list.xlsx",
        modified=datetime(2026, 1, 2, 3, 4), map_name="made_up_map.json")
    answers = {
        "site_name": ql.Answer(value="Made-up Yard", status="answered",
                               pdf_pages=[3], source="report text"),
        "has_fill": ql.Answer(value=True, status="answered", pdf_pages=[4],
                              note="fill to 2 ft", source="code",
                              code_value=True),
        "hole_count": ql.Answer(value=3, status="answered", pdf_pages=[5]),
        "field_date": ql.Answer(value="2026-01-05", status="answered"),
        "soil_kinds": ql.Answer(value=["clay", "sand"], status="answered"),
        "wells": ql.Answer(value=[{"well": "MW-1", "depth_ft": 20}],
                           status="answered"),
        "depths": ql.Answer(value=[{"hole": "B-1", "depth": 30}],
                            status="answered"),
        "rock_cored": ql.Answer(status="not said"),
        "pile_type": ql.Answer(status="does not apply",
                               note="no piles recommended"),
        # lost_row: no answer came back (a failed call)
    }
    calls = [{"call": 1, "model": "made-up-list-model", "questions": 9,
              "answered": 7, "input_tokens": 100, "output_tokens": 20}]
    return qlist, answers, calls


class TestTheQuestionList:

    def test_the_answers_are_saved_as_the_sheet_shows_them(self, tmp_path):
        from report_ingest import question_list as ql

        qlist, answers, calls = _question_list()
        path = ql.answer_json(qlist, answers, tmp_path / ql.ANSWERS_NAME,
                              report_id="SYN", calls=calls)
        blob = json.loads(path.read_text("utf-8"))
        rows = {r["key"]: r for r in blob["answers"]}

        assert [r["key"] for r in blob["answers"]] == \
            [q.key for q in qlist]
        assert rows["has_fill"]["text"] == "yes"
        assert rows["has_fill"]["value"] is True
        assert rows["pile_type"]["text"] == "N/A"
        assert rows["rock_cored"]["text"] == ""
        assert (rows["lost_row"]["status"], rows["lost_row"]["note"]) == \
            ("", "not run")
        assert rows["wells"]["value"] == [{"well": "MW-1", "depth_ft": 20}]
        assert rows["has_fill"]["code_value_text"] == "yes"
        assert blob["run"]["question_list"] == "made_up_list.xlsx"
        assert blob["run"]["list_modified"] == "2026-01-02 03:04"
        assert blob["run"]["prompt_version"] == ql.PROMPT_VERSION
        assert blob["run"]["calls"][0]["model"] == "made-up-list-model"

    @pytest.fixture
    def listed(self, tmp_path):
        from report_ingest import question_list as ql

        folder, _pdf, written = _write_folder(tmp_path)
        qlist, answers, calls = _question_list()
        ql.answer_json(qlist, answers, folder / ql.ANSWERS_NAME,
                       report_id="SYN", calls=calls)
        ql.answer_sheet(qlist, answers, folder / ql.SHEET_NAME,
                        report_id="SYN", calls=calls)
        return folder, written

    def test_each_data_type_lands_in_its_column(self, listed, tmp_path):
        folder, written = listed
        outcome = tables.export_report(folder, tmp_path / "root")
        v1 = Path(outcome.folder)
        rows = _answers(_rows(v1, "report_answer", written.key),
                        "question_list")

        assert list(rows) == [q.key for q in _question_list()[0]]
        assert [r["position"] for r in rows.values()] == list(range(1, 11))
        site = rows["site_name"]
        assert (site["value_text"], site["question_label"],
                site["category"], site["data_type"]) == \
            ("Made-up Yard", "Site Name", "Site", "string")
        assert site["pdf_pages"] == [3] and site["source"] == "report text"
        assert rows["has_fill"]["value_boolean"] is True
        assert rows["has_fill"]["code_value_text"] == "yes"
        assert rows["has_fill"]["note"] == "fill to 2 ft"
        assert rows["hole_count"]["value_number"] == 3.0
        assert rows["field_date"]["value_date"] == "2026-01-05"
        assert rows["soil_kinds"]["value_list"] == ["clay", "sand"]
        for key in ("wells", "depths"):
            assert rows[key]["value_list"] is None
            assert json.loads(rows[key]["value_json"]) == \
                _question_list()[1][key].value
        assert rows["site_name"]["question_list"] == \
            "made_up_list.xlsx (modified 2026-01-02 03:04)"
        assert rows["site_name"]["prompt_version"] == "ql-1"

    def test_the_three_status_words_and_a_failed_call(self, listed,
                                                      tmp_path):
        folder, written = listed
        v1 = Path(tables.export_report(folder, tmp_path / "root").folder)
        rows = _answers(_rows(v1, "report_answer", written.key),
                        "question_list")
        (report,) = _rows(v1, "report", written.key)

        assert rows["site_name"]["status"] == "answered"
        assert rows["rock_cored"]["status"] == "not said"
        assert rows["pile_type"]["status"] == "does not apply"
        assert rows["lost_row"]["status"] is None
        assert rows["lost_row"]["note"] == "not run"
        assert report["n_answers"] == 38 + 10

    def test_the_sheet_and_the_models_go_out_with_the_report(self, listed,
                                                            tmp_path):
        folder, written = listed
        v1 = Path(tables.export_report(folder, tmp_path / "root").folder)
        kinds = {f["kind"] for f in _rows(v1, "report_file", written.key)}
        (run_file,) = (v1 / "ingest_run").glob(f"{written.key}.*.jsonl")
        (run,) = _rows(v1, "ingest_run", run_file.name)

        assert "questions" in kinds
        assert (v1 / "questions" / f"{written.key}.questions.xlsx").is_file()
        assert "made-up-list-model" in run["models"]
        assert "ql-1" in run["prompt_versions"]

    def test_a_run_without_the_list_hides_an_earlier_runs_answers(
            self, listed, tmp_path):
        folder, written = listed
        tables._save_run(str(folder), {"report_key": written.key,
                                       "question_list": False})
        v1 = Path(tables.export_report(folder, tmp_path / "root").folder)

        assert not _answers(_rows(v1, "report_answer", written.key),
                            "question_list")
        assert "questions" not in {
            f["kind"] for f in _rows(v1, "report_file", written.key)}


# ---------------------------------------------------------------------------
# a second ingest
# ---------------------------------------------------------------------------

class TestAReingest:

    def test_the_rows_are_replaced_and_the_history_gains_a_run(self,
                                                               tmp_path):
        folder, pdf, written = _write_folder(tmp_path)
        record = ReportRecord.model_validate_json(
            (folder / "report.record.json").read_text("utf-8"))
        root = tmp_path / "root"
        first = tables.after_ingest(
            folder, record=record, source=pdf, key_source=pdf,
            setting=str(root), now=datetime(2026, 1, 2, 3, 4, 5,
                                            tzinfo=timezone.utc))
        second = tables.after_ingest(
            folder, record=record, source=pdf, key_source=pdf,
            setting=str(root), now=datetime(2026, 1, 3, 3, 4, 5,
                                            tzinfo=timezone.utc))
        v1 = Path(second.folder)

        assert first.status == second.status == "written"
        runs = sorted((v1 / "ingest_run").glob(f"{written.key}.*.jsonl"))
        assert [p.name[17:33] for p in runs] == \
            ["20260102T030405Z", "20260103T030405Z"]
        (report,) = _rows(v1, "report", written.key)
        assert report["latest_run_id"].startswith("20260103T030405Z-")
        assert report["ingested_at"] == "2026-01-03T03:04:05Z"
        assert len(_rows(v1, "report_answer", written.key)) == 38

    def test_a_run_file_older_than_its_record_is_not_used(self, tmp_path):
        """RI-209 review B3: a re-ingest with the export off leaves the
        earlier run's run file behind. A later export reads the folder as
        one ingested before the export: a new run id and the record's own
        time; the earlier run's history row is left as it was and its
        question-list answers are not shown as this ingest's."""
        folder, pdf, written = _write_folder(tmp_path)
        record = ReportRecord.model_validate_json(
            (folder / "report.record.json").read_text("utf-8"))
        answers = folder / "report.questions.json"
        answers.write_text(json.dumps({"answers": [{
            "key": "made_up_key", "status": "answered",
            "text": "an earlier answer", "value": "an earlier answer",
            "pdf_pages": [1]}], "run": {}}), encoding="utf-8")
        root = tmp_path / "root"
        first = tables.after_ingest(
            folder, record=record, source=pdf, key_source=pdf,
            setting=str(root), question_list=True,
            now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        v1 = Path(first.folder)
        (old_run,) = (v1 / "ingest_run").glob(f"{written.key}.*.jsonl")
        old_bytes = old_run.read_bytes()
        assert _answers(_rows(v1, "report_answer", written.key),
                        "question_list")

        # the re-ingest, export off: the writers run, no run file is written
        record.general.projectName = "A second made-up reading"
        write_outputs(record, folder, source=pdf,
                      db_path=tmp_path / "run" / "reports.db")
        stamp = (folder / "report.record.json").stat().st_mtime
        os.utime(folder / tables.RUN_NAME, (stamp - 60, stamp - 60))
        os.utime(answers, (stamp - 61, stamp - 61))
        assert tables.export_outcome(folder) is None
        (outcome,) = tables.export_folder(tmp_path / "run", root)

        assert outcome.status == "written"
        (report,) = _rows(v1, "report", written.key)
        assert report["project_name"] == "A second made-up reading"
        assert report["ingested_at"] == datetime.fromtimestamp(
            stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not report["latest_run_id"].startswith("20260102T030405Z")
        assert report["source_sha256"] is None
        assert not _answers(_rows(v1, "report_answer", written.key),
                            "question_list")
        runs = sorted((v1 / "ingest_run").glob(f"{written.key}.*.jsonl"))
        assert len(runs) == 2 and old_run in runs
        assert old_run.read_bytes() == old_bytes

    def test_an_ingest_run_file_already_there_is_never_overwritten(
            self, tmp_path):
        """A run's history row stands as first written, whatever a later
        export of the same run makes of it (RI-209 review B3)."""
        folder, _pdf, written = _write_folder(tmp_path)
        root = tmp_path / "root"
        v1 = Path(tables.export_report(folder, root).folder)
        (history,) = (v1 / "ingest_run").glob(f"{written.key}.*.jsonl")
        history.write_bytes(b'{"made_up": "as first written"}\n')

        again = tables.export_report(folder, root)

        assert again.status == "written"
        assert history.read_bytes() == b'{"made_up": "as first written"}\n'
        assert not any(f.startswith("ingest_run/") for f in again.files)

    def test_a_diggs_file_from_an_earlier_run_is_removed(self, tmp_path):
        folder, pdf, written = _write_folder(tmp_path)
        root = tmp_path / "root"
        tables.export_report(folder, root)
        v1 = Path(tables.export_dir(root))
        assert (v1 / "diggs" / f"{written.key}.diggs.xml").is_file()

        # The next reading finds no exploration and no test: the writer
        # writes no DIGGS file (and leaves the old one in the folder).
        bare = build_record(investigations=[], lab_tests=[])
        again = write_outputs(bare, folder, source=pdf,
                              db_path=tmp_path / "run" / "reports.db")
        assert again.key == written.key
        assert (folder / "report.diggs.xml").is_file()
        tables.export_report(folder, root)

        assert not (v1 / "diggs" / f"{written.key}.diggs.xml").exists()
        (report,) = _rows(v1, "report", written.key)
        assert report["diggs_written"] is False
        assert report["diggs_schema_valid"] is None
        assert "diggs" not in {
            f["kind"] for f in _rows(v1, "report_file", written.key)}

    def test_a_bound_report_no_longer_found_is_dropped(self, ingested,
                                                       tmp_path):
        out = tmp_path / "out"
        shutil.copytree(ingested.out, out)
        parent, child = _keys(ingested)
        blob = json.loads((out / "report.record.json").read_text("utf-8"))
        blob["bound_documents"] = []
        (out / "report.record.json").write_text(json.dumps(blob),
                                                encoding="utf-8")
        outcome = tables.export_report(out, out)
        v1 = Path(outcome.folder)

        assert outcome.keys == [parent]
        # RI-209 review B5: an EMPTY file in each keyed table, so the copy a
        # dataset keeps is replaced by no rows; its documents go locally
        assert outcome.dropped == [child]
        for table in ("report", "report_page", "report_answer",
                      "report_file"):
            assert (v1 / table / f"{child}.jsonl").read_bytes() == b"", table
            assert f"{table}/{child}.jsonl" in outcome.files
        assert not (v1 / "record" / f"{child}.record.json").exists()
        assert list((v1 / "ingest_run").glob(f"{child}.*.jsonl"))
        (top,) = _rows(v1, "report", parent)
        assert top["bound_report_keys"] == []

    def test_a_report_whose_key_changed_empties_its_old_key(self, tmp_path):
        """RI-209 review B5: the report's own key moves (here: first keyed
        without its file, then with it, as a cluster library made before
        RI-209 does once). The old key's table files are emptied and sent in
        the same save; its history stays."""
        folder, pdf, _written = _write_folder(tmp_path)
        record = ReportRecord.model_validate_json(
            (folder / "report.record.json").read_text("utf-8"))
        root = tmp_path / "root"
        first = tables.after_ingest(folder, record=record, key_source=None,
                                    setting=str(root))
        lib = _fake().FakeLibrary(tmp_path / "fake")
        with pytest.MonkeyPatch.context() as mp:
            import aec_agent_core.foundry_dataset_store as fds

            lib.install(mp)
            mp.setenv("GEOTECH_DEPLOYMENT", "foundry")
            mp.setenv(tables.ENV_DATASET, _fake().ALIAS)
            mp.delenv(fds.ENV_ROOT, raising=False)
            second = tables.after_ingest(folder, record=record, source=pdf,
                                         key_source=pdf, setting=str(root))
        v1 = Path(second.folder)
        old, new = first.report_key, second.report_key

        assert old != new and second.dropped == [old]
        for table in ("report", "report_page", "report_answer",
                      "report_file"):
            assert (v1 / table / f"{old}.jsonl").read_bytes() == b""
            assert _rows(v1, table, new)
        assert not (v1 / "record" / f"{old}.record.json").exists()
        assert list((v1 / "ingest_run").glob(f"{old}.*.jsonl"))
        assert second.delivery.startswith("sent ")
        (save,) = lib.ds.uploads()
        top = "GeotechStaffEngineer/report_tables/v1/"
        assert top + f"report/{old}.jsonl" in save[1]
        assert top + f"report_page/{old}.jsonl" in save[1]


# ---------------------------------------------------------------------------
# a report sent to a person, the key, older folders
# ---------------------------------------------------------------------------

class TestAReportSentToAPerson:

    def test_its_report_row_goes_out_with_no_page_rows(self, tmp_path):
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        _needs_person(pdf, tmp_path / "out")
        run = json.loads((tmp_path / "out" / tables.RUN_NAME)
                         .read_text("utf-8"))
        v1 = tmp_path / "out" / "report_tables" / "v1"

        (row,) = _rows(v1, "report", run["report_key"])
        assert row["workflow"] == "needs_person"
        assert _rows(v1, "report_page", run["report_key"]) == []
        assert len(_rows(v1, "report_answer", run["report_key"])) == 38


class TestTheKey:

    def test_the_same_file_as_a_path_or_as_bytes_keys_alike(self, tmp_path):
        data = _pdf_bytes()
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(data)
        _needs_person(pdf, tmp_path / "by_path")
        _needs_person(data, tmp_path / "by_bytes")

        def report(folder):
            run = json.loads((tmp_path / folder / tables.RUN_NAME)
                             .read_text("utf-8"))
            (row,) = _rows(tmp_path / folder / "report_tables" / "v1",
                           "report", run["report_key"])
            return row

        by_path, by_bytes = report("by_path"), report("by_bytes")
        assert by_path["report_key"] == by_bytes["report_key"]
        assert by_path["source_sha256"] == by_bytes["source_sha256"] == \
            hashlib.sha256(data).hexdigest()
        assert by_bytes["source_bytes"] == len(data)
        assert (by_path["source_name"], by_bytes["source_name"]) == \
            ("made_up_report.pdf", None)
        page = (tmp_path / "by_bytes" / "report.page.md").read_text("utf-8")
        assert "%PDF" not in page and "b'" not in page

    def test_an_open_document_keys_as_its_file_does(self, tmp_path):
        """RI-209 review A1: the cluster stage hands the graph an open
        planlens document. Its file, named by ``source_path`` (or, failing
        that, the one planlens opened), keys the report and gives the source
        facts, so a path, bytes and a document of one PDF give one key and
        one SHA-256, whatever the report_id."""
        from planlens.document import open_document

        data = _pdf_bytes()
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(data)
        _needs_person(pdf, tmp_path / "by_path")
        _needs_person(data, tmp_path / "by_bytes")
        with open_document(str(pdf), name="SYN") as doc:
            _needs_person(doc, tmp_path / "by_document")
        with open_document(data, name="SYN") as doc:
            # opened from bytes, so only the caller can name the file
            _needs_person(doc, tmp_path / "by_document_and_path",
                          source_path=pdf)

        def run(folder):
            return json.loads((tmp_path / folder / tables.RUN_NAME)
                              .read_text("utf-8"))

        runs = [run(name) for name in ("by_path", "by_bytes", "by_document",
                                       "by_document_and_path")]
        assert len({r["report_key"] for r in runs}) == 1
        assert {r["source_sha256"] for r in runs} == \
            {hashlib.sha256(data).hexdigest()}
        assert [r["source_name"] for r in runs] == \
            ["made_up_report.pdf", None, "made_up_report.pdf",
             "made_up_report.pdf"]

    def test_a_document_with_no_file_keys_as_before(self, tmp_path):
        """With no file to read, a document still keys off what the record
        says (its report_id), and the source facts stay null."""
        from planlens.document import open_document

        with open_document(_pdf_bytes(), name="SYN") as doc:
            _needs_person(doc, tmp_path / "no_file")
        run = json.loads((tmp_path / "no_file" / tables.RUN_NAME)
                         .read_text("utf-8"))
        assert run["report_key"]
        assert (run["source_name"], run["source_sha256"]) == (None, None)

    def test_bytes_key_a_record_as_the_file_does(self, tmp_path):
        pdf = _made_up_pdf(tmp_path)
        record = build_record()

        assert record_key(record, pdf.read_bytes()) == record_key(record, pdf)


class TestFoldersIngestedBefore:

    def test_an_old_folder_is_keyed_off_its_library_page(self, tmp_path):
        folder, _pdf, written = _write_folder(tmp_path)
        moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        os.utime(folder / "report.record.json",
                 (moment.timestamp(), moment.timestamp()))
        (outcome,) = tables.export_folder(tmp_path / "run")
        v1 = Path(outcome.folder)
        (report,) = _rows(v1, "report", written.key)
        (run_file,) = (v1 / "ingest_run").glob(f"{written.key}.*.jsonl")
        (run,) = _rows(v1, "ingest_run", run_file.name)

        assert outcome.status == "written" and outcome.keys == [written.key]
        assert v1 == tmp_path / "run" / "report_tables" / "v1"
        assert (report["source_name"], report["source_bytes"],
                report["source_sha256"]) == (None, None, None)
        assert report["ingested_at"] == "2026-01-02T03:04:05Z"
        assert report["latest_run_id"].startswith("20260102T030405Z-")
        assert (run["app_version"], run["core_version"]) == (None, None)
        assert not (folder / tables.RUN_NAME).exists()  # nothing added

    def test_an_answer_sheet_alone_is_read_back(self, tmp_path):
        from report_ingest import question_list as ql

        folder, _pdf, written = _write_folder(tmp_path)
        qlist, answers, calls = _question_list()
        ql.answer_sheet(qlist, answers, folder / ql.SHEET_NAME,
                        report_id="SYN", calls=calls)
        (outcome,) = tables.export_folder(tmp_path / "run")
        rows = _answers(_rows(Path(outcome.folder), "report_answer",
                              written.key), "question_list")

        assert rows["site_name"]["value_text"] == "Made-up Yard"
        assert rows["site_name"]["pdf_pages"] == [3]
        assert rows["site_name"]["source"] == "report text"
        assert rows["has_fill"]["value_text"] == "yes"
        assert rows["has_fill"]["note"] == "fill to 2 ft"
        assert rows["pile_type"]["status"] == "does not apply"
        assert rows["rock_cored"]["status"] == "not said"
        assert rows["site_name"]["data_type"] is None   # the sheet has none
        assert rows["site_name"]["question_list"].startswith(
            "made_up_list.xlsx")

    def test_a_folder_of_reports_exports_each_with_its_bound_reports(
            self, ingested, tmp_path):
        run_root = tmp_path / "run_root"
        shutil.copytree(ingested.out, run_root / "SYN")
        shutil.rmtree(run_root / "SYN" / "report_tables")
        parent, child = _keys(ingested)

        (outcome,) = tables.export_folder(run_root)

        assert outcome.keys == [parent, child]
        assert (run_root / "report_tables" / "v1" / "report" /
                f"{child}.jsonl").is_file()

    def test_the_command_line(self, tmp_path, capsys):
        _folder, _pdf, written = _write_folder(tmp_path)

        assert tables.main([str(tmp_path / "run")]) == 0
        said = capsys.readouterr().out
        assert f"{written.key}: written" in said
        assert "1 report(s)" in said


# ---------------------------------------------------------------------------
# the settings
# ---------------------------------------------------------------------------

class TestTheSwitch:

    @pytest.mark.parametrize("setting, env, expect", [
        (None, None, "out"),
        (None, "on", "out"),
        (None, "0", None),
        (None, "off", None),
        (None, "False", None),
        (False, None, None),
        (True, "0", "out"),
        ("elsewhere", "0", "elsewhere"),
        (None, "elsewhere", "elsewhere"),
    ])
    def test_where_the_export_goes(self, monkeypatch, setting, env, expect):
        if env is not None:
            monkeypatch.setenv(tables.ENV_ROOT, env)

        assert tables.export_root(setting, "out") == expect

    def test_the_default_is_beside_the_library(self, tmp_path):
        db = tmp_path / "shared" / "reports.db"

        assert tables.export_root(None, tmp_path / "one", db) == \
            str(tmp_path / "shared")

    def test_in_the_apps_shared_library_the_root_is_the_library(
            self, monkeypatch, tmp_path):
        library = tmp_path / "library"
        own = tmp_path / "conversation" / "report_ingest" / "Report"
        monkeypatch.setenv(tables.LIBRARY_ENV, str(library))

        assert tables.export_root(None, library / "Report-1a2b3c4d") == \
            str(library)
        assert tables.export_root(None, own) == str(own)   # not in it
        assert tables.export_root(None, library / "R",
                                  tmp_path / "run" / "reports.db") == \
            str(tmp_path / "run")                          # a named library
        assert tables.export_root(str(tmp_path / "elsewhere"),
                                  library / "R") == str(tmp_path / "elsewhere")
        monkeypatch.delenv(tables.LIBRARY_ENV)
        assert tables.export_root(None, library / "R") == str(library / "R")

    def test_every_report_read_into_the_library_shares_one_export(
            self, monkeypatch, tmp_path):
        library = tmp_path / "library"
        monkeypatch.setenv(tables.LIBRARY_ENV, str(library))
        keys = []
        for name in ("Report-one", "Report-two"):
            pdf = tmp_path / f"{name}.pdf"
            pdf.write_bytes(b"%PDF-1.7\n" + name.encode("ascii") * 900)
            folder = library / name             # as the app names it
            written = write_outputs(build_record(), folder, source=pdf)
            record = ReportRecord.model_validate_json(
                (folder / "report.record.json").read_text("utf-8"))
            outcome = tables.after_ingest(folder, record=record, source=pdf,
                                          key_source=pdf)
            assert outcome.folder == str(library / "report_tables" / "v1")
            keys.append(written.key)

        v1 = library / "report_tables" / "v1"
        assert sorted(p.stem for p in (v1 / "report").glob("*.jsonl")) == \
            sorted(keys)
        assert not list(library.glob("Report-*/report_tables"))

    def test_the_library_setting_is_the_apps_own(self):
        tools = (Path(__file__).resolve().parents[2] / "funhouse_agent" /
                 "deep" / "tools.py").read_text("utf-8")

        assert f'REPORT_LIBRARY_ENV = "{tables.LIBRARY_ENV}"' in tools

    def test_off_writes_nothing_at_all(self, monkeypatch, tmp_path):
        monkeypatch.setenv(tables.ENV_ROOT, "0")
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        _needs_person(pdf, tmp_path / "out")

        assert not (tmp_path / "out" / tables.RUN_NAME).exists()
        assert not list(tmp_path.rglob("report_tables"))
        assert tables.export_outcome(tmp_path / "out") is None

    def test_the_argument_wins_over_the_setting(self, monkeypatch,
                                                tmp_path):
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        monkeypatch.setenv(tables.ENV_ROOT, "0")
        _needs_person(pdf, tmp_path / "on", report_tables=True)
        monkeypatch.delenv(tables.ENV_ROOT)
        _needs_person(pdf, tmp_path / "off", report_tables=False)
        _needs_person(pdf, tmp_path / "there",
                      report_tables=str(tmp_path / "elsewhere"))

        assert (tmp_path / "on" / "report_tables" / "v1").is_dir()
        assert not (tmp_path / "off" / "report_tables").exists()
        assert not (tmp_path / "off" / tables.RUN_NAME).exists()
        assert (tmp_path / "elsewhere" / "report_tables" / "v1" /
                "schema.json").is_file()
        assert not (tmp_path / "there" / "report_tables").exists()

    def test_a_failure_is_an_outcome_never_an_exception(self, tmp_path):
        folder = tmp_path / "not_a_report"
        folder.mkdir()

        outcome = tables.export_report(folder, tmp_path / "root")

        assert outcome.status == "error"
        assert "report.record.json" in outcome.detail


# ---------------------------------------------------------------------------
# delivery to a Foundry dataset (the core's fake library)
# ---------------------------------------------------------------------------

class TestWhatTheAppIsTold:
    """RI-209 review B4: the export's outcome reaches the app's tool result
    and a folder run's index as one plain line."""

    @pytest.mark.parametrize("outcome, said", [
        (None, "not written (the export is off"),
        ({"status": "off"}, "not written (the export is off"),
        ({"status": "error", "detail": "OSError: made up"},
         "NOT written (OSError: made up); the ingest itself is complete"),
        ({"status": "written", "folder": "R/report_tables/v1",
          "delivery": "sent 9 file(s)"},
         "written under R/report_tables/v1, and sent to the Foundry dataset"),
        ({"status": "written", "folder": "R/report_tables/v1",
          "delivery": ""}, "not sent to a dataset "
         "(GSE_REPORT_TABLES_DATASET is not set)"),
        ({"status": "written", "folder": "R/report_tables/v1",
          "delivery": "not a Foundry deployment"},
         "not sent to a dataset (not a Foundry deployment)"),
        ({"status": "written", "folder": "R/report_tables/v1",
          "delivery": "busy: StorageBusy: made up"},
         "NOT sent to the Foundry dataset, which was busy"),
        ({"status": "written", "folder": "R/report_tables/v1",
          "delivery": "failed: AliasNotDefined: made up"},
         "NOT sent to the Foundry dataset (AliasNotDefined: made up); check "
         "that GSE_REPORT_TABLES_DATASET names"),
    ])
    def test_one_line_says_what_happened_and_why_not(self, outcome, said):
        line = tables.outcome_line(outcome)
        assert line.startswith("Report tables: ") and "\n" not in line
        assert said in line

    def test_a_failed_send_says_how_to_send_again(self):
        line = tables.outcome_line(
            {"status": "written", "folder": os.path.join("lib", "report_tables",
                                                         "v1"),
             "delivery": "failed: AliasNotDefined: made up"},
            report_dir=os.path.join("lib", "R1"))
        assert line.endswith(
            f"python -m report_ingest.tables \"{os.path.join('lib', 'R1')}\" "
            "--root \"lib\" --deliver")

    def test_the_tool_result_carries_the_line(self, tmp_path):
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.fake_engine import FakeEngine

        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        answer = run_ingest(str(pdf), [], out_dir=str(tmp_path / "out"),
                            engine=FakeEngine([triage_turn("needs_person")]))

        assert answer.report_tables == (
            "Report tables: written under "
            f"{tmp_path / 'out' / 'report_tables' / 'v1'}; not sent to a "
            "dataset (GSE_REPORT_TABLES_DATASET is not set).")

    def test_the_tool_result_says_when_the_export_is_off(self, monkeypatch,
                                                         tmp_path):
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.fake_engine import FakeEngine

        monkeypatch.setenv(tables.ENV_ROOT, "0")
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        answer = run_ingest(str(pdf), [], out_dir=str(tmp_path / "out"),
                            engine=FakeEngine([triage_turn("needs_person")]))

        assert answer.report_tables == \
            "Report tables: not written (the export is off, GSE_REPORT_TABLES)."

    def test_a_folder_runs_index_carries_each_reports_outcome(self, tmp_path):
        from report_ingest.run_folder import run_folder
        from report_ingest.tests.fake_engine import FakeEngine

        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "R1.pdf").write_bytes(_pdf_bytes())
        out = tmp_path / "out"
        run_folder(reports, lambda: FakeEngine([triage_turn("needs_person")]),
                   out_dir=out, log=lambda *_: None)

        index = json.loads((out / "index.json").read_text("utf-8"))
        (row,) = index["reports"]
        assert row["report_tables"]["status"] == "written"
        assert row["report_tables"]["delivery"] == ""
        assert row["report_tables_line"].startswith(
            "Report tables: written under ")
        text = (out / "INDEX.md").read_text("utf-8")
        assert "## The report tables" in text
        assert row["report_tables_line"] in text


def _fake():
    if "foundry_fake" in sys.modules:
        return sys.modules["foundry_fake"]
    path = (Path(__file__).resolve().parents[2] / "packages" /
            "aec-agent-core" / "tests" / "foundry_fake.py")
    spec = importlib.util.spec_from_file_location("foundry_fake", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["foundry_fake"] = module
    spec.loader.exec_module(module)
    return module


class TestDelivery:

    @pytest.fixture
    def foundry(self, monkeypatch, tmp_path):
        import aec_agent_core.foundry_dataset_store as fds

        fake = _fake()
        library = fake.FakeLibrary(tmp_path / "fake").install(monkeypatch)
        monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
        monkeypatch.setenv(tables.ENV_DATASET, fake.ALIAS)
        monkeypatch.delenv(fds.ENV_ROOT, raising=False)
        return library

    def test_one_save_per_report_under_the_apps_folder(self, foundry,
                                                      ingested, tmp_path):
        import aec_agent_core.foundry_dataset_store as fds

        out = tmp_path / "out"
        shutil.copytree(ingested.out, out)
        parent, child = _keys(ingested)
        outcome = tables.export_report(out, out, deliver=True)
        top = "GeotechStaffEngineer/report_tables/v1/"
        assert fds._app_top() == "GeotechStaffEngineer"

        assert outcome.status == "written"
        assert outcome.delivery.startswith("sent ")
        (save,) = foundry.ds.uploads()
        assert save[0] == "upload_directory"
        assert foundry.ds.count("upload_file") == 0
        sent = save[1]
        assert all(path.startswith(top) for path in sent)
        assert top + "schema.json" in sent
        for key in (parent, child):
            assert top + f"report/{key}.jsonl" in sent
            assert any(p.startswith(top + f"ingest_run/{key}.")
                       for p in sent)
        assert foundry.leftovers() == []

    def test_a_process_with_only_report_ingest_sends_under_the_apps_name(
            self, tmp_path):
        """RI-209 review A2: the command line's --deliver, a notebook or the
        cluster loads report_ingest alone; its files must land under the
        same top folder as the app's (GeotechStaffEngineer), never under the
        core's default name. In a fresh process, since this one has loaded
        the app's packages."""
        import subprocess

        root = Path(__file__).resolve().parents[2]
        code = (
            "import importlib.util, os, sys\n"
            "from pathlib import Path\n"
            f"tmp = Path({str(tmp_path)!r})\n"
            "spec = importlib.util.spec_from_file_location('foundry_fake', "
            f"{str(root / 'packages' / 'aec-agent-core' / 'tests' / 'foundry_fake.py')!r})\n"
            "fake = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(fake)\n"
            "lib = fake.FakeLibrary(tmp / 'fake')\n"
            "sys.modules.update(lib.modules)\n"
            "import aec_agent_core.foundry_dataset_store as fds\n"
            "fds.OUTPUTS_DIR = lib.outputs\n"
            "from report_ingest import tables\n"
            "from report_ingest.model import ReportRecord\n"
            "from report_ingest.writers import write_outputs\n"
            "src = tmp / 'in.pdf'\n"
            "src.write_bytes(b'%PDF-1.4' + b' made up' * 200)\n"
            "r = ReportRecord()\n"
            "r.document.report_id = 'SYN'\n"
            "r.document.n_pages = 1\n"
            "write_outputs(r, tmp / 'run' / 'R1', source=str(src))\n"
            "code = tables.main([str(tmp / 'run'), '--deliver'])\n"
            "loaded = [m for m in ('funhouse_agent', 'webapp', "
            "'calc_package') if m in sys.modules]\n"
            "assert not loaded, loaded\n"
            "sent = sorted(lib.ds.files_)\n"
            "assert code == 0 and sent, (code, sent)\n"
            "bad = [p for p in sent if not "
            "p.startswith('GeotechStaffEngineer/report_tables/v1/')]\n"
            "assert not bad, bad\n"
            "print('ok')\n")
        env = dict(os.environ, GEOTECH_DEPLOYMENT="foundry",
                   GSE_REPORT_TABLES_DATASET=_fake().ALIAS)
        env.pop("GEOTECH_SHAREPOINT_ROOT", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root / "packages" / "aec-agent-core"), str(root),
             env.get("PYTHONPATH", "")])
        done = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=300,
                              cwd=str(tmp_path))
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip().endswith("ok")

    def test_an_ingest_sends_its_export(self, foundry, tmp_path):
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        _needs_person(pdf, tmp_path / "out")

        outcome = tables.export_outcome(tmp_path / "out")
        assert outcome["status"] == "written"
        assert outcome["delivery"].startswith("sent ")
        assert len(foundry.ds.uploads()) == 1

    def test_a_busy_store_does_not_fail_the_export(self, foundry, ingested,
                                                   tmp_path):
        import aec_agent_core.foundry_dataset_store as fds

        out = tmp_path / "out"
        shutil.copytree(ingested.out, out)
        foundry.ds.busy = 10
        manager = fds.DatasetFileManager(alias=_fake().ALIAS,
                                         busy_pauses=(),
                                         sleep=lambda _s: None)
        outcome = tables.export_report(out, out, deliver=True,
                                       file_manager=manager)

        assert outcome.status == "written"
        assert outcome.delivery.startswith("busy: StorageBusy")
        assert (Path(outcome.folder) / "schema.json").is_file()

    def test_a_failing_store_does_not_fail_the_ingest(self, foundry,
                                                      tmp_path):
        foundry.ds.get_error = "500 made-up failure"
        pdf = tmp_path / "made_up_report.pdf"
        pdf.write_bytes(_pdf_bytes())
        record = _needs_person(pdf, tmp_path / "out")

        outcome = tables.export_outcome(tmp_path / "out")
        assert record.document.workflow == "needs_person"
        assert outcome["status"] == "written"
        assert outcome["delivery"].startswith("failed: ")

    def test_nothing_is_sent_or_imported_without_a_dataset(self,
                                                           monkeypatch,
                                                           tmp_path):
        import aec_agent_core

        for name in ("aec_agent_core.foundry_dataset_store", "foundry",
                     "foundry.transforms"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delattr(aec_agent_core, "foundry_dataset_store",
                            raising=False)
        folder, pdf, _written = _write_folder(tmp_path)
        record = ReportRecord.model_validate_json(
            (folder / "report.record.json").read_text("utf-8"))

        unset = tables.after_ingest(folder, record=record, source=pdf,
                                    key_source=pdf)
        monkeypatch.setenv(tables.ENV_DATASET, "made_up_alias")
        elsewhere = tables.after_ingest(folder, record=record, source=pdf,
                                        key_source=pdf)

        assert (unset.status, unset.delivery) == ("written", "")
        assert elsewhere.delivery == "not a Foundry deployment"
        for name in ("aec_agent_core.foundry_dataset_store", "foundry",
                     "foundry.transforms"):
            assert name not in sys.modules, name
