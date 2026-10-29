"""RI-125: a resumed record says what its cost covers, and the run file names
the steps it took from the folder.

``record.document``'s cost is THIS run's calls; ``document.resumed`` (set
only when a step came off disk) names those steps, counts the item readings
that came off disk, and gives the whole report's cost. A record read from
scratch is written as it always was. In ``run.json``, ``triage_reused`` and
``review_reused`` mean "taken from the folder", ``resumed`` names the steps,
and ``label_run_engine`` is named only off a label run file on disk. RESULTS
keeps a label run's seeded calls out of ``unattributed``, and a folder run's
index gives a resumed report's whole cost. Scripted engines on the synthetic
report; every number is made up.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("planlens.document.roles")

from pydantic import ValidationError  # noqa: E402

from report_ingest.graph import ingest_report  # noqa: E402
from report_ingest.model import DocumentFacts, ResumedSteps  # noqa: E402
from report_ingest.tests import test_graph as tg  # noqa: E402
from report_ingest.tests.fake_engine import FakeEngine  # noqa: E402
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report)
from report_ingest.tests.test_cluster_scoring import (  # noqa: E402,F401
    _ingest_run, _metered, _run_blob, chat_cluster, ingest_cluster)
# Today's readers are scripted: model first is set off (RI-186).
from report_ingest.tests.test_graph import (  # noqa: E402,F401
    todays_readers)


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("ri125") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _item_files(folder):
    """Every reader item file under ``folder``, its bound documents' too."""
    return (sorted((folder / "items").glob("*.json"))
            + sorted(folder.glob("bound/*/items/*.json")))


def _review_script():
    """A run whose vision voter splits on page 10, so the review runs and
    moves it to a laboratory sheet, read as one more item."""
    return ([tg.triage_turn()] + tg.vision_turns({10: "figure"})
            + tg.review_turns([tg.LabelChange(
                page=10, from_label="figure", to_label="lab_test",
                reason="a results sheet", evidence="render_page")])
            + [tg.identity_turn()]
            + [tg.narrative_turn(), tg.log_turn("B-1"),
               tg.log_turn("TP-1", "test_pit"),
               tg.lab_turn("moisture_content", 10),
               tg.lab_turn("atterberg", 12), tg.lab_turn("gradation", 13),
               tg.calc_turn()]
            + tg.bound_reader_turns())


# ---------------------------------------------------------------------------
# the field itself
# ---------------------------------------------------------------------------

class TestTheField:

    def test_unset_it_is_left_out_of_the_written_record(self):
        facts = DocumentFacts(report_id="R00", model_calls=3)
        assert "resumed" not in facts.model_dump()
        assert "resumed" not in facts.model_dump(mode="json")
        assert '"resumed"' not in facts.model_dump_json()

    def test_set_it_is_written_and_reads_back(self):
        facts = DocumentFacts(
            report_id="R00", model_calls=2, input_tokens=300,
            resumed=ResumedSteps(steps=["triage", "items"], items=3,
                                 items_of=4, saved_calls=7,
                                 saved_input_tokens=900, report_calls=9,
                                 report_input_tokens=1200))
        dumped = facts.model_dump(mode="json")
        assert dumped["resumed"]["steps"] == ["triage", "items"]
        assert dumped["resumed"]["report_calls"] == 9
        assert DocumentFacts.model_validate(dumped) == facts
        assert DocumentFacts.model_validate_json(
            facts.model_dump_json()) == facts

    def test_a_step_outside_the_list_is_refused(self):
        with pytest.raises(ValidationError):
            ResumedSteps(steps=["labels"])


# ---------------------------------------------------------------------------
# the graph: a fresh run, and resumed runs
# ---------------------------------------------------------------------------

class TestAFreshRun:

    def test_nothing_is_resumed_and_the_cost_is_the_whole_reports(
            self, pdf, tmp_path):
        record = ingest_report(pdf, FakeEngine(tg.full_script()),
                               out_dir=tmp_path, report_id="SYN")
        assert record.document.resumed is None
        assert record.document.model_calls == len(tg.full_script())
        assert "resumed" not in _json(
            tmp_path / "report.record.json")["document"]
        assert "resumed" not in _json(
            tmp_path / "bound" / "bound1" / "report.record.json")["document"]
        summary = (tmp_path / "report.summary.md").read_text(encoding="utf-8")
        assert "output tokens. Record schema" in summary
        assert "taken from the folder" not in summary


class TestAResumedRun:

    def test_all_off_disk_it_costs_nothing_and_says_what_the_report_cost(
            self, pdf, tmp_path):
        first = ingest_report(pdf, FakeEngine(tg.full_script()),
                              out_dir=tmp_path, report_id="SYN")
        first_child = _json(tmp_path / "bound" / "bound1"
                            / "report.record.json")["document"]
        empty = FakeEngine([])
        again = ingest_report(pdf, empty, out_dir=tmp_path, report_id="SYN")

        assert empty.n_calls == 0
        doc = again.document
        assert (doc.model_calls, doc.input_tokens, doc.output_tokens) == (
            0, 0, 0)
        took = doc.resumed
        # The voters agreed everywhere, so no review was ever made.
        assert took.steps == ["triage", "vision", "identity", "items"]
        n_items = len(_item_files(tmp_path))
        assert (took.items, took.items_of) == (n_items, n_items)
        assert took.saved_calls == first.document.model_calls
        assert took.report_calls == first.document.model_calls
        assert took.report_input_tokens == first.document.input_tokens
        assert took.report_output_tokens == first.document.output_tokens
        assert took.report_dollars == pytest.approx(first.document.dollars)

        written = _json(tmp_path / "report.record.json")["document"]
        assert written["model_calls"] == 0
        assert written["resumed"]["steps"] == took.steps
        # The bound document's own record says the same of its own steps.
        child = _json(tmp_path / "bound" / "bound1"
                      / "report.record.json")["document"]
        assert child["model_calls"] == 0
        assert child["resumed"]["steps"] == ["items"]
        assert child["resumed"]["items"] == child["resumed"]["items_of"] == 2
        assert child["resumed"]["report_calls"] == first_child["model_calls"]

        summary = (tmp_path / "report.summary.md").read_text(encoding="utf-8")
        assert (f"0 model call(s), 0 input and 0 output tokens this run "
                f"(taken from the folder: triage, vision, identity, items "
                f"{n_items} of {n_items}); the whole report came to "
                f"{first.document.model_calls} model call(s)") in summary

    def test_a_saved_review_and_part_of_the_items_off_disk(self, pdf,
                                                           tmp_path):
        first = ingest_report(pdf, FakeEngine(_review_script()),
                              out_dir=tmp_path, report_id="SYN")
        assert (tmp_path / "review.json").is_file()
        (calc,) = [path for path in (tmp_path / "items").glob("*.json")
                   if "calculation" in _json(path)]
        calc.unlink()                    # the printout is read again
        engine = FakeEngine([tg.calc_turn()])
        again = ingest_report(pdf, engine, out_dir=tmp_path, report_id="SYN")

        assert engine.n_calls == 1
        doc = again.document
        assert doc.model_calls == 1
        took = doc.resumed
        assert took.steps == ["triage", "vision", "review", "identity",
                              "items"]
        n_items = len(_item_files(tmp_path))
        assert (took.items, took.items_of) == (n_items - 1, n_items)
        assert took.saved_calls == first.document.model_calls - 1
        # The printout cost one call either time: the report's total stands.
        assert took.report_calls == first.document.model_calls
        assert took.report_input_tokens == first.document.input_tokens
        assert [c.kind for c in again.calculations] == ["lateral_pile"]


# ---------------------------------------------------------------------------
# the cluster stage's run file
# ---------------------------------------------------------------------------

def _files(out):
    folder = out / "ingest" / "R36"
    return folder, folder / "run.json", folder / "report.record.json"


class TestTheRunFile:

    def test_a_first_attempt_writes_the_flags_false_and_no_resumed(
            self, ingest_cluster, monkeypatch):
        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster)
        _folder, run_file, record_file = _files(out)
        run = _json(run_file)
        assert run["triage_reused"] is False and run["review_reused"] is False
        assert "resumed" not in run
        document = _json(record_file)["document"]
        assert "resumed" not in document
        # The record's cost and the run file's are this attempt's calls.
        assert document["model_calls"] == run["cost"]["calls"]
        assert document["model_calls"] == (len(tg.main_script())
                                           + len(tg.vision_turns()))

    def test_a_resume_after_a_give_up_names_its_steps_and_agrees(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest import graph

        _reports, out, scripts = ingest_cluster
        full = tg.main_script()
        scripts["turns"] = full[:5] + full[7:]      # the two lab turns out
        _metered(monkeypatch, scripts)
        real = graph._read_lab

        def give_up(*args, **kwargs):
            raise RuntimeError("made-up rate limit: retries exhausted")

        monkeypatch.setattr(graph, "_read_lab", give_up)
        _ingest_run(ingest_cluster)
        _folder, run_file, record_file = _files(out)
        first = _json(run_file)
        assert first["item_errors"]
        assert first["triage_reused"] is False and "resumed" not in first

        monkeypatch.setattr(graph, "_read_lab", real)
        scripts["turns"] = [full[5], full[6]]
        capsys.readouterr()
        _ingest_run(ingest_cluster)
        again = _json(run_file)
        # No triage call this attempt: its file was taken from the folder.
        assert again["triage_reused"] is True
        assert again["review_reused"] is False      # never made: no split
        n_items = len(_item_files(out / "ingest" / "R36"))
        assert again["resumed"] == {
            "steps": ["triage", "vision", "identity", "items"],
            "items": n_items - 2, "items_of": n_items}
        assert (f"R36: taken from the folder: triage, vision, identity, "
                f"items {n_items - 2} of {n_items}") in capsys.readouterr().out

        document = _json(record_file)["document"]
        # This attempt: the two lab readings, in the record and the run file.
        assert document["model_calls"] == again["cost"]["calls"] == 2
        # Every call of the first attempt left a file and nothing came from
        # a label run, so the two whole-report figures are the same number.
        took = document["resumed"]
        assert took["saved_calls"] == first["cost"]["calls"]
        assert took["report_calls"] == again["all_attempts"]["calls"]
        assert took["report_input_tokens"] == (
            again["all_attempts"]["input_tokens"])

    def test_a_label_run_seed_is_named_and_its_cost_counted(
            self, ingest_cluster, monkeypatch):
        from planlens.document import open_document
        from planlens.document.roles import page_roles

        reports_dir, out, scripts = ingest_cluster
        with open_document(str(reports_dir / "R36.pdf")) as doc:
            rules = {r.page: r.role for r in page_roles(doc)}
        seed = _run_blob("R36", 22, rules, rules)
        seed["profile"]["cost"] = {"calls": 1, "input_tokens": 2100,
                                   "output_tokens": 110, "dollars": 0.0}
        seed["review"]["cost"] = {"calls": 3, "input_tokens": 9300,
                                  "output_tokens": 420, "dollars": 0.0}
        (out / "runs").mkdir(parents=True)
        (out / "runs" / "R36.json").write_text(json.dumps(seed),
                                               encoding="utf-8")
        main = tg.main_script()
        scripts["turns"] = main[1:]                 # no triage turn
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster)
        folder, run_file, record_file = _files(out)

        run = _json(run_file)
        assert run["triage_reused"] is True and run["review_reused"] is True
        assert run["resumed"]["steps"] == ["triage", "review"]
        assert run["resumed"]["from_label_run"] == ["triage", "review"]
        assert run["resumed"]["items"] == 0
        assert _json(folder / "review.json")["cost"]["calls"] == 3
        assert "label_run_engine" not in run        # the same engine
        document = _json(record_file)["document"]
        assert document["model_calls"] == run["cost"]["calls"]
        took = document["resumed"]
        assert took["saved_calls"] == 1 + 3
        assert took["saved_input_tokens"] == 2100 + 9300
        assert took["report_calls"] == run["cost"]["calls"] + 4
        # RESULTS: the seeded calls stay in their cells, out of
        # unattributed, and are named on a line of their own.
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ("R36: triage, review seeded from a label run (4 calls), "
                "not this ingest's") in text
        cells = _report_row(text)
        assert cells[2] == "1" and cells[4] == "3"     # triage, review
        assert cells[-1] == "+0"                        # every call has a file

        # The run file moved aside: the next call reads the folder again,
        # pays nothing, and still says which steps a label run seeded.
        run_file.unlink()
        scripts["turns"] = []
        scripts["vision"] = []
        _ingest_run(ingest_cluster)
        again = _json(run_file)
        assert again["resumed"]["steps"] == ["triage", "vision", "review",
                                             "identity", "items"]
        assert again["resumed"]["from_label_run"] == ["triage", "review"]
        assert again["resumed"]["items"] == again["resumed"]["items_of"]
        assert again["cost"]["calls"] == 0
        later = _json(record_file)["document"]
        assert later["model_calls"] == 0
        assert later["resumed"]["report_calls"] == took["report_calls"]
        # Every file is now an earlier attempt's, the seeded ones aside.
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert _report_row(text)[-1] == f"{-run['cost']['calls']:+d}"

    def test_a_first_attempt_with_no_seed_has_no_seeded_line(
            self, ingest_cluster, monkeypatch):
        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "seeded from a label run" not in text
        assert _report_row(text)[-1] == "+0"

    def test_a_copied_folder_names_no_engine_it_cannot_read(
            self, chat_cluster, monkeypatch):
        pytest.importorskip("langchain_core")
        from report_ingest import cluster_scoring as cs
        from report_ingest.tests.chat_fixtures import (ScriptedChatModel,
                                                       from_turns)

        def chat(name, turns):
            return ScriptedChatModel(replies=from_turns(list(turns)),
                                     model_name=name)

        def run():
            return _ingest_run(
                chat_cluster, prompter=None,
                chat_model=chat("made-up-reader", tg.main_script()),
                vision_chat_model=chat("made-up-looker", tg.vision_turns()))

        # The scripted chat answers do not all fit the readers' schemas;
        # unread items are not what this test is about.
        monkeypatch.setattr(cs, "_item_error_fields",
                            lambda record, bound=(): {})
        _reports, out, _scripts = chat_cluster
        runs = out / "runs"
        runs.mkdir(parents=True)
        seed = _run_blob("R36", 22, {3: "plan"}, {3: "plan"})
        seed["engine"] = "chat_model"               # this run's engine
        (runs / "R36.json").write_text(json.dumps(seed), encoding="utf-8")
        run()
        _folder, run_file, _record = _files(out)
        first = _json(run_file)
        assert first["resumed"]["from_label_run"]
        assert "label_run_engine" not in first

        # The folder as it stands after a copy: no run file and no label
        # run beside it. The stamps still say a label run seeded it, but no
        # file on disk says on what engine, so none is named.
        run_file.unlink()
        (runs / "R36.json").unlink()
        run()
        again = _json(run_file)
        assert again["resumed"]["from_label_run"] == (
            first["resumed"]["from_label_run"])
        assert "label_run_engine" not in again
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "other engine" not in text


class TestTheLabelRunEngine:

    def _folder(self, tmp_path, stamp):
        folder = tmp_path / "ingest" / "R00"
        folder.mkdir(parents=True)
        (folder / "triage.json").write_text(json.dumps(
            {"workflow": "standard", "reused_from": str(stamp)}),
            encoding="utf-8")
        return folder

    def test_read_off_the_file_the_stamp_names(self, tmp_path):
        from report_ingest.cluster_scoring import _label_run_engine

        elsewhere = tmp_path / "another_out" / "runs"
        elsewhere.mkdir(parents=True)
        (elsewhere / "R00.json").write_text(json.dumps(
            {"id": "R00", "engine": "chat_model"}), encoding="utf-8")
        folder = self._folder(tmp_path, elsewhere / "R00.json")
        assert _label_run_engine(folder, ["triage"]) == "chat_model"
        # A step the record did not take from the folder is not asked.
        assert _label_run_engine(folder, ["review"]) == ""

    def test_a_file_from_before_the_engine_was_recorded_is_the_prompters(
            self, tmp_path):
        from report_ingest.cluster_scoring import _label_run_engine

        (tmp_path / "R00.json").write_text(json.dumps({"id": "R00"}),
                                           encoding="utf-8")
        folder = self._folder(tmp_path, tmp_path / "R00.json")
        assert _label_run_engine(folder, ["triage"]) == "prompter"

    def test_no_file_on_disk_names_no_engine(self, tmp_path):
        from report_ingest.cluster_scoring import _label_run_engine

        folder = self._folder(tmp_path, tmp_path / "gone" / "R00.json")
        assert _label_run_engine(folder, ["triage"]) == ""


def _report_row(text, rid="R36"):
    """The report's own row of RESULTS' calls-by-step table, as cells."""
    block = text.split("## Calls by step", 1)[1].split("```")[1]
    (row,) = [line.split() for line in block.splitlines()
              if line.split()[:2] == [rid, "report"]]
    return row


# ---------------------------------------------------------------------------
# a folder run's index
# ---------------------------------------------------------------------------

class TestTheFolderIndex:

    @pytest.fixture()
    def folder(self, pdf, tmp_path):
        reports = tmp_path / "reports"
        reports.mkdir()
        (reports / "R01.pdf").write_bytes(pdf.read_bytes())
        return reports

    def _run(self, folder, out, turns, lines=None):
        from report_ingest.run_folder import run_folder

        return run_folder(folder, lambda: FakeEngine(turns), out_dir=out,
                          log=(lines.append if lines is not None
                               else (lambda *_a: None)))

    def test_a_fresh_run_writes_what_it_always_wrote(self, folder, tmp_path):
        out = tmp_path / "out"
        run = self._run(folder, out, tg.full_script())
        (row,) = run.reports
        assert row.model_calls == len(tg.full_script())
        assert row.calls_this_run is None
        text = (out / "INDEX.md").read_text(encoding="utf-8")
        assert text.splitlines()[6].endswith(f"| {row.model_calls} |")
        assert "this run" not in text
        (saved,) = _json(out / "index.json")["reports"]
        assert "calls_this_run" not in saved
        assert "dollars_this_run" not in saved
        assert saved["model_calls"] == row.model_calls

    def test_a_resumed_report_shows_the_whole_reports_calls(self, folder,
                                                           tmp_path):
        out = tmp_path / "out"
        (first,) = self._run(folder, out, tg.full_script()).reports
        # The record went missing; its step and item files did not.
        (out / "R01" / "report.record.json").unlink()
        lines: list = []
        (row,) = self._run(folder, out, [], lines).reports

        assert not row.skipped
        assert row.calls_this_run == 0 and row.dollars_this_run == 0.0
        assert row.model_calls == first.model_calls
        assert row.dollars == pytest.approx(first.dollars)
        text = (out / "INDEX.md").read_text(encoding="utf-8")
        assert f"| {first.model_calls} (0 this run) |" in text
        assert "`Model calls` is what each report cost over every run" in text
        (saved,) = _json(out / "index.json")["reports"]
        assert saved["model_calls"] == first.model_calls
        assert saved["calls_this_run"] == 0
        assert any(f"0 model call(s) this run (triage, vision, identity, "
                   f"items taken from its folder; {first.model_calls} over "
                   f"the report)" in line for line in lines)

        # Asked again, the report is skipped as done, and its row is read
        # off the record: the whole report's calls, no "this run".
        (skipped,) = self._run(folder, out, []).reports
        assert skipped.skipped
        assert skipped.model_calls == first.model_calls
        assert skipped.calls_this_run is None
