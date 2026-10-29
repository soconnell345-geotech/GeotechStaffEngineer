"""The ingest sub-agent's own ceiling (BACKLOG RI-02), offline.

deepagents runs a compiled sub-agent as provided, so a model-call budget put
on its spec ran nowhere. The ceiling is :class:`CallCeiling` now, counted
inside the graph; the wiring through the app's build is tested in
``funhouse_agent/deep/tests/test_report_ingest_offline.py``.
"""

from __future__ import annotations

import json

import pytest

from report_ingest.subagent import (
    CallCeiling, CallCeilingReached, build_report_ingest_subagent,
)
from report_ingest.tests.fake_engine import FakeEngine


@pytest.fixture(autouse=True)
def todays_readers(monkeypatch):
    """These runs script today's readers: model first, on for every stage
    when the setting is unset since BACKLOG RI-186, is set off for them."""
    from report_ingest.graph import MODEL_FIRST_ENV

    monkeypatch.setenv(MODEL_FIRST_ENV, "off")


def _ask(engine):
    return engine.complete([{"role": "user", "content": "hello"}])


class TestTheCeiling:

    def test_two_engines_share_one_ceiling(self):
        reader = FakeEngine([{"text": "a"}, {"text": "b"}], name="reader")
        vision = FakeEngine([{"text": "c"}], name="vision")
        ceiling = CallCeiling(2)
        counted_reader, counted_vision = (ceiling.wrap(reader),
                                          ceiling.wrap(vision))

        assert _ask(counted_reader).text == "a"
        assert _ask(counted_vision).text == "c"
        with pytest.raises(CallCeilingReached, match="ceiling of 2"):
            _ask(counted_reader)
        # Refused before anything was sent.
        assert reader.n_calls == 1 and vision.n_calls == 1
        assert ceiling.used == 2 and ceiling.reached

    def test_the_wrapped_engine_is_the_engine_otherwise(self):
        engine = FakeEngine([{"text": "a"}], name="fake-model")
        counted = CallCeiling(5).wrap(engine)

        assert counted.name == "fake-model"
        assert counted.calls is engine.calls

    def test_no_engine_stays_no_engine(self):
        assert CallCeiling(1).wrap(None) is None

    def test_a_ceiling_below_one_is_refused(self):
        with pytest.raises(ValueError):
            CallCeiling(0)


class TestTheSpec:

    def test_it_carries_no_middleware(self):
        pytest.importorskip("langgraph")
        spec = build_report_ingest_subagent(lambda: None,
                                            max_model_calls=10)

        assert set(spec) == {"name", "description", "runnable"}

    def test_middleware_is_refused_not_ignored(self):
        with pytest.raises(ValueError, match="runs no middleware"):
            build_report_ingest_subagent(lambda: None,
                                         middleware=[object()])


# ---------------------------------------------------------------------------
# What the ceiling cut short (the review of RI-02): counted, and not kept
# ---------------------------------------------------------------------------

class TestAStepCutShort:
    """Each step hands its cost in when it finishes and is saved so a
    resumed run does not pay again. One the ceiling cut short is counted all
    the same, off the run's tally, and is not saved as finished."""

    def test_what_a_step_spent_before_it_failed_is_counted(self):
        from report_ingest.graph import _Spend, _Tally

        tally = _Tally()
        engine = tally.wrap(CallCeiling(1).wrap(
            FakeEngine([{"text": "a"}])))
        spend = _Spend(tally)
        mark = spend.mark()
        _ask(engine)
        with pytest.raises(CallCeilingReached):
            _ask(engine)
        spend.settle(mark)

        assert (spend.calls, spend.input_tokens, spend.output_tokens) == \
            (1, 1000, 100)
        assert tally.refused == 1 and spend.cut_short(mark)

    def test_a_step_that_handed_its_cost_in_is_not_counted_twice(self):
        from report_ingest.graph import _Spend, _Tally

        tally = _Tally()
        engine = tally.wrap(FakeEngine([{"text": "a"}]))
        spend = _Spend(tally)
        mark = spend.mark()
        _ask(engine)
        spend.add({"calls": 1, "input_tokens": 1000, "output_tokens": 100})
        spend.settle(mark)

        assert spend.calls == 1 and not spend.cut_short(mark)

    def test_a_refusal_a_voter_swallowed_still_stops_the_save(self, tmp_path):
        # A log's layer votes take a failed call as their own and the
        # reader finishes; what it read is used, but not kept as finished.
        from report_ingest.graph import _keep, _Spend, _Tally

        tally = _Tally()
        engine = tally.wrap(CallCeiling(1).wrap(
            FakeEngine([{"text": "a"}])))
        spend = _Spend(tally)
        mark = spend.mark()
        _ask(engine)
        try:
            _ask(engine)
        except CallCeilingReached:
            pass                              # the voter's own failure
        qa = []
        path = tmp_path / "L1.json"

        assert _keep(str(path), {"read": 1}, spend, mark, qa,
                     "items.log.L1", [3]) is False
        assert not path.exists()
        (entry,) = qa
        assert entry.kind == "partial" and entry.pages == [3]
        assert "reads it again" in entry.detail
        # A later step the ceiling did not touch is kept.
        assert _keep(str(path), {"read": 1}, spend, spend.mark(), qa,
                     "items.log.L2") is True
        assert path.is_file()


class TestTheRecordCountsEveryInputToken:
    """BACKLOG RI-171: the record's input_tokens is plain input, cache
    writes and cache reads, whichever of them a step's own cost left out.
    Made-up counts."""

    @staticmethod
    def _cached():
        from report_ingest.engine import Usage

        return FakeEngine([{"text": "a"}], usage=Usage(
            input_tokens=4, output_tokens=50, cache_write_tokens=1700,
            cache_read_tokens=300))

    def test_a_step_that_handed_in_no_cost_is_counted_whole(self):
        from report_ingest.graph import _Spend, _Tally

        tally = _Tally()
        engine = tally.wrap(self._cached())
        spend = _Spend(tally)
        mark = spend.mark()
        _ask(engine)
        spend.settle(mark)
        assert (spend.input_tokens, spend.cache_write_tokens,
                spend.cache_read_tokens) == (4, 1700, 300)
        assert spend.tokens_in == 2004

    def test_a_cache_write_the_step_s_cost_left_out_is_topped_up(self):
        """A reader's own cost carries plain input and cache reads only."""
        from report_ingest.graph import _Spend, _Tally

        tally = _Tally()
        engine = tally.wrap(self._cached())
        spend = _Spend(tally)
        mark = spend.mark()
        _ask(engine)
        spend.add({"calls": 1, "input_tokens": 4, "cache_read_tokens": 300,
                   "output_tokens": 50})
        spend.settle(mark)
        assert spend.calls == 1 and spend.tokens_in == 2004
        # Handed in whole, nothing is counted twice.
        again = _Spend(tally)
        mark = again.mark()
        _ask(tally.wrap(self._cached()))
        again.add({"calls": 1, "input_tokens": 4, "cache_write_tokens": 1700,
                   "cache_read_tokens": 300, "output_tokens": 50})
        again.settle(mark)
        assert again.tokens_in == 2004

    def test_a_bound_run_and_a_saved_step_count_their_tokens_in(self):
        from report_ingest.graph import _Spend

        inner = _Spend()
        inner.add({"calls": 1, "input_tokens": 5, "cache_write_tokens": 900,
                   "cache_read_tokens": 100})
        outer = _Spend()
        outer.absorb(inner)
        assert outer.tokens_in == 1005
        outer.reuse("triage", {"calls": 2, "input_tokens": 8,
                               "cache_write_tokens": 600,
                               "cache_read_tokens": 40})
        resumed = outer.resumed()
        assert resumed.saved_input_tokens == 648
        assert resumed.report_input_tokens == 1005 + 648

    def test_the_record_says_the_run_s_tokens_in(self, pdf, tmp_path):
        from report_ingest.engine import Usage
        from report_ingest.graph import ingest_report
        from report_ingest.tests.test_graph import full_script

        engine = FakeEngine(full_script(), usage=Usage(
            input_tokens=4, output_tokens=50, cache_write_tokens=1700,
            cache_read_tokens=300))
        ingest_report(str(pdf), engine, out_dir=str(tmp_path),
                      report_id="SYN")
        record = json.loads((tmp_path / "report.record.json")
                            .read_text(encoding="utf-8"))
        document = record["document"]
        assert document["model_calls"] == engine.n_calls
        assert document["input_tokens"] == 2004 * engine.n_calls


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    pytest.importorskip("planlens.document.roles")
    pytest.importorskip("pymupdf")
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    path = tmp_path_factory.mktemp("corpus") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


class TestAskingAgainAfterTheCeiling:

    def test_every_call_is_counted_and_what_was_cut_is_read_again(
            self, pdf, tmp_path):
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.test_graph import full_script

        # Triage and two of the vision voter's four calls; its third is
        # refused, and so is everything after it.
        first = FakeEngine(full_script()[:3])
        answer = run_ingest(str(pdf), [], engine=first, out_dir=str(tmp_path),
                            report_id="SYN", max_model_calls=3)

        assert answer.ceiling_reached
        assert answer.ceiling_used == 3 == first.n_calls
        assert answer.model_call_ceiling == 3
        # The voter the ceiling stopped is counted with the rest.
        assert answer.model_calls == 3
        record = json.loads((tmp_path / "report.record.json")
                            .read_text(encoding="utf-8"))
        assert record["document"]["input_tokens"] == 3000
        # Nothing it cut short was saved as finished.
        assert (tmp_path / "triage.json").is_file()
        assert not (tmp_path / "vision.json").exists()
        assert not (tmp_path / "bound" / "bound1" / "identity.json").exists()
        assert not list((tmp_path / "items").glob("*.json"))

        # Asked again: triage is reused, the rest is read now.
        rest = full_script()[1:]
        again = FakeEngine(rest)
        answer = run_ingest(str(pdf), [], engine=again, out_dir=str(tmp_path),
                            report_id="SYN", max_model_calls=1000)

        assert again.n_calls == len(rest)
        assert not answer.ceiling_reached and answer.ceiling_used == len(rest)
        assert answer.investigations == 2 and answer.error == ""
        assert (tmp_path / "vision.json").is_file()
        assert (tmp_path / "bound" / "bound1" / "identity.json").is_file()

    def test_a_call_sent_that_failed_is_counted_too(self, pdf, tmp_path):
        # The bound report's identity call goes out and fails: no reply, no
        # cost of its own, but a call the ceiling counted (the review of
        # RI-03). The record's count is the ceiling's.
        from report_ingest.subagent import run_ingest
        from report_ingest.tests.test_graph import full_script

        script = full_script()
        script[5] = {"raise": RuntimeError("the gateway failed")}
        engine = FakeEngine(script)
        answer = run_ingest(str(pdf), [], engine=engine, out_dir=str(tmp_path),
                            report_id="SYN", max_model_calls=1000)

        assert engine.n_calls == len(script)
        assert answer.model_calls == answer.ceiling_used == len(script)
        assert not (tmp_path / "bound" / "bound1" / "identity.json").exists()

    def test_a_label_review_the_ceiling_stopped_is_asked_again(self, pdf,
                                                              tmp_path):
        from report_ingest.graph import ingest_report
        from report_ingest.tests.test_graph import (
            full_script, review_turns, triage_turn, vision_turns,
        )

        # The review's first call goes out; its findings are refused.
        script = ([triage_turn()] + vision_turns({10: "figure"})
                  + review_turns()[:1])
        record = ingest_report(pdf, CallCeiling(len(script)).wrap(
            FakeEngine(script)), out_dir=tmp_path, report_id="SYN")

        (note,) = [e for e in record.qa if e.where == "labels.review"]
        assert note.kind == "partial" and "CallCeilingReached" in note.detail
        assert not (tmp_path / "review.json").exists()
        assert record.document.model_calls == len(script)

        # Asked again: the review, then the identity call and the readers
        # the ceiling refused (the run's script from the identity call on).
        again = FakeEngine(review_turns() + full_script()[5:])
        record = ingest_report(pdf, again, out_dir=tmp_path, report_id="SYN")
        assert again.n_calls == len(again.turns)
        assert (tmp_path / "review.json").is_file()
        assert not [e for e in record.qa if e.kind == "skipped"]
        assert len(record.investigations) == 2


class TestTheChecklistOnEveryRecord:

    def test_bound_reports_get_their_own_and_failures_read_as_printed(
            self, pdf, tmp_path):
        from report_ingest.checklist import load_checklist
        from report_ingest.subagent import CHECKLIST_NAME, run_ingest
        from report_ingest.tests.test_graph import full_script

        answer = run_ingest(str(pdf), [], engine=FakeEngine(full_script()),
                            out_dir=str(tmp_path), report_id="SYN",
                            checklist=True)

        (bound,) = answer.bound_documents
        beside = tmp_path / "bound" / "bound1" / CHECKLIST_NAME
        assert beside.is_file()
        assert bound["checklist"] == str(beside)
        assert bound["checklist_failed"].isdigit()
        assert json.loads(beside.read_text(encoding="utf-8"))["status"]
        assert answer.paths["checklist"] == str(tmp_path / CHECKLIST_NAME)
        # Failed checks in the checklist's own words, not their ids.
        printed = {item["text"] for item in load_checklist()["items"]}
        assert answer.checklist["failed"]
        for line in answer.checklist["failed"]:
            assert line.split(" -> ")[0] in printed, line
