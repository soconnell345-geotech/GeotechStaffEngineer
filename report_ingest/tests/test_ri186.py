"""Model first, stage by stage, and on by default (BACKLOG RI-186).

``model_first`` sets the page labels and the four reader stages each on or
off; unset, every stage is on (the owner, 2026-10-29). A reader stage off is
today's reader on today's work items; the labels off, today's vote and
review label the pages, no page is described, and the model-first stages'
groups are today's work items. No model is called: the made-up report and
its scripts are the model-first tests' own.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from report_ingest import cluster_scoring as cs
from report_ingest.graph import (MODEL_FIRST_ENV, model_first_setting,
                                 model_first_stages)

pytest.importorskip("planlens.document.roles")

ALL_ON = {"labels": "on", "logs": "on", "lab": "on", "calc": "on",
          "narrative": "on"}


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------

class TestTheSetting:

    def test_unset_is_on_for_every_stage(self, monkeypatch):
        monkeypatch.delenv(MODEL_FIRST_ENV, raising=False)
        assert model_first_stages() == ALL_ON
        assert model_first_setting() == "on"

    @pytest.mark.parametrize("value, off", [
        ("logs=off,lab=off", {"logs", "lab"}),
        ("off,narrative=on", {"labels", "logs", "lab", "calc"}),
        ("labels = no; calc=0", {"labels", "calc"}),
        ({"labels": False, "Calc": "off"}, {"labels", "calc"}),
        ({"narrative": True}, set()),
        ("on,on", set()),
        (False, set(ALL_ON))])
    def test_a_setting_names_the_stages_it_sets(self, value, off):
        stages = model_first_stages(value)
        assert {stage for stage, word in stages.items()
                if word == "off"} == off
        assert model_first_stages(model_first_setting(value)) == stages

    def test_one_word_for_the_run_files(self):
        assert model_first_setting("logs=off,lab=off") == (
            "labels=on,logs=off,lab=off,calc=on,narrative=on")
        assert model_first_setting({"labels": "off", "logs": "off",
                                    "lab": "off", "calc": "off",
                                    "narrative": "off"}) == "off"

    def test_the_variable_is_read_stage_by_stage(self, monkeypatch):
        monkeypatch.setenv(MODEL_FIRST_ENV, "calc=off")
        assert model_first_stages()["calc"] == "off"
        assert model_first_stages("on")["calc"] == "on"

    @pytest.mark.parametrize("value, why", [
        ("soundings=on", "no stage 'soundings'"),
        ("logs=maybe", "logs='maybe': use off or on"),
        ("model first", "use off or on")])
    def test_a_typo_is_refused(self, value, why):
        with pytest.raises(ValueError, match=why):
            model_first_stages(value)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_pdf(tmp_path_factory):
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    path = tmp_path_factory.mktemp("ri186") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


def _stage(turns, name):
    from report_ingest.tests.fake_engine import FakeEngine

    made = FakeEngine(turns, name=name)
    made.accepts_pdf = True
    return made


def _run(pdf, out, main, model_first, stages, **kwargs):
    from report_ingest.graph import ingest_report
    from report_ingest.tests.fake_engine import FakeEngine

    describer = FakeEngine([], name="made-up-describer")
    labeller = FakeEngine([], name="made-up-labeller")
    record = ingest_report(pdf, main, out_dir=out, report_id="SYN",
                           model_first=model_first,
                           describe_engine=kwargs.pop("describer", describer),
                           range_engine=kwargs.pop("labeller", labeller),
                           stage_engines=stages, **kwargs)
    return record, describer, labeller


def _items(out):
    return sorted(path.name for path in (out / "items").iterdir())


def test_the_narrative_alone_reads_the_pages_today_s_labels_give_it(
        synthetic_pdf, tmp_path):
    """The labels and the other readers off: the vote labels the pages, no
    page is described, today's readers read the logs, the sheets and the
    printout, and the narrative is read model first off today's labels."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine

    main = FakeEngine([tg.triage_turn()] + tg.vision_turns()
                      + [tg.identity_turn()] + tg.reader_turns()[1:]
                      + tg.bound_reader_turns()[1:], name="made-up-main")
    narrative = _stage([tg.narrative_turn(), tg.bound_narrative_turn()],
                       "made-up-narrative")
    out = tmp_path / "out"
    record, describer, labeller = _run(
        synthetic_pdf, out, main, "off,narrative=on",
        {"narrative": narrative})
    assert describer.n_calls == 0 and labeller.n_calls == 0
    assert not (out / "descriptions").exists()
    assert main.n_calls == len(main.turns) and narrative.n_calls == 2
    assert [name for name in _items(out) if name.startswith("mf-")] == [
        "mf-narrative.json"]
    assert record.general.projectName == "Rosewood Terrace Development"
    assert sorted(i.investigation_id for i in record.investigations) == [
        "B-1", "TP-1"]
    labels = json.loads((out / "labels.json").read_text("utf-8"))
    assert labels["model_first"] == (
        "labels=off,logs=off,lab=off,calc=off,narrative=on")
    assert labels["vision_mode"] == "sheet"
    (note,) = [q for q in record.qa if q.where == "model_first"]
    assert "(narrative; logs, lab, calc by today's readers)" in note.detail
    assert "labelled as today, by the vote and the label review" \
        in note.detail
    assert note.values == ["narrative: made-up-narrative, pages as pdf"]


def test_the_labels_off_group_today_s_work_items(synthetic_pdf, tmp_path):
    """Every reader model first, the labels as today: each of today's work
    items is one group, a log's pages one log, and nothing is described."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests import test_graph_model_first as tgm
    from report_ingest.tests.fake_engine import FakeEngine

    main = FakeEngine([tg.triage_turn()] + tg.vision_turns()
                      + [tg.identity_turn()], name="made-up-main")
    stages = {
        "narrative": _stage([tg.narrative_turn(), tg.bound_narrative_turn()],
                            "made-up-narrative"),
        "logs": _stage([
            tgm._log("B-1", "boring", 7, [(0.0, 3.0, "Made-up CLAY")]),
            tgm._log("TP-1", "test_pit", 9, [(0.0, 2.0, "Made-up FILL")]),
            tgm._log("BH-1", "boring", 18, [(0.0, 9.0, "Made-up SILT")])],
            "made-up-logs"),
        "lab": _stage([{"text": "Made-up words and no answer."}] * 2,
                      "made-up-lab"),
        "calc": _stage([tgm._calc()], "made-up-calc"),
    }
    out = tmp_path / "out"
    record, describer, labeller = _run(synthetic_pdf, out, main,
                                       "labels=off", stages)
    assert describer.n_calls == 0 and labeller.n_calls == 0
    assert main.n_calls == len(main.turns)
    assert {s: e.n_calls for s, e in stages.items()} == {
        "narrative": 2, "logs": 3, "lab": 2, "calc": 1}
    names = _items(out)
    assert "mf-log-7.json" in names and "mf-log-9.json" in names
    assert "mf-calc-20.json" in names
    assert sorted(i.investigation_id for i in record.investigations) == [
        "B-1", "TP-1"]
    labels = json.loads((out / "labels.json").read_text("utf-8"))
    assert labels["model_first"] == (
        "labels=off,logs=on,lab=on,calc=on,narrative=on")


def test_the_labels_on_with_today_s_readers(synthetic_pdf, tmp_path):
    """The labels model first and the logs, sheets and printout off: the
    range labels make today's work items, and today's readers read them on
    the run's engine; the narrative is read model first."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests import test_graph_model_first as tgm

    main, describer, labeller, stages = tgm._engines()
    main.turns = ([tg.triage_turn(), tg.identity_turn()]
                  + tg.reader_turns()[1:] + tg.bound_reader_turns()[1:])
    out = tmp_path / "out"
    record, _d, _l = _run(synthetic_pdf, out, main,
                          "logs=off,lab=off,calc=off",
                          {"narrative": stages["narrative"]},
                          describer=describer, labeller=labeller)
    assert describer.n_calls == 22 and labeller.n_calls == 2
    assert main.n_calls == len(main.turns)
    assert [name for name in _items(out) if name.startswith("mf-")] == [
        "mf-narrative.json"]
    assert sorted(i.investigation_id for i in record.investigations) == [
        "B-1", "TP-1"]
    assert not (out / "vision.json").exists()


def test_the_rules_alone_turn_the_default_labels_off(monkeypatch, tmp_path):
    """``label_policy='rules'`` with the setting left to its default takes
    the rules alone: the labels are not asked model first."""
    from report_ingest import graph
    from report_ingest.tests.fake_engine import FakeEngine

    monkeypatch.delenv(MODEL_FIRST_ENV, raising=False)
    seen = {}

    def first_run(*args, stages=None, **kwargs):
        seen.update(stages)
        raise RuntimeError("made up: stop here")

    monkeypatch.setattr(graph, "_model_first_run", first_run)
    with pytest.raises(RuntimeError, match="stop here"):
        graph.ingest_report(b"%PDF", FakeEngine([]), out_dir=tmp_path,
                            label_policy="rules")
    assert seen == {**ALL_ON, "labels": "off"}


# ---------------------------------------------------------------------------
# the cluster's arm
# ---------------------------------------------------------------------------

class TestTheClusterArm:

    def test_a_stage_s_own_arm_and_the_run_s(self):
        first = cs._FirstArm("logs=off,lab=off")
        assert first.on and first.reading == ("calc", "narrative")
        assert (first.arm("logs"), first.arm("calc")) == ("off", "on")
        assert first.arm() == "labels=on,logs=off,lab=off,calc=on,narrative=on"
        assert first.stage_on("labels")
        assert not cs._FirstArm("off").on

    def test_an_item_is_held_to_its_stage_s_setting(self, tmp_path):
        first = cs._FirstArm("logs=off")
        run_file = tmp_path / "logs" / "R00_p1.json"
        run_file.parent.mkdir()
        run_file.write_text(json.dumps({"model_first": "off"}), "utf-8")
        assert cs._arm_clash(run_file, first, "logs") is None
        assert "model first off" in cs._arm_clash(run_file, first, "calc")
        ingest = tmp_path / "ingest" / "R00" / "run.json"
        ingest.parent.mkdir(parents=True)
        ingest.write_text(json.dumps({"model_first": "on"}), "utf-8")
        assert "this run is labels=on,logs=off" in cs._arm_clash(ingest,
                                                                 first)

    def test_the_ingest_arm_builds_only_what_the_setting_reads(self):
        built = []

        class Engines:
            def build(self, name, meter, vision=False):
                built.append(name)
                return SimpleNamespace(name=name, accepts_pdf=True)

        kwargs, names = cs._ingest_arm(
            cs._FirstArm("labels=off,logs=off", {"logs": "made-up-log"}),
            Engines(), None, "made-up-main", "made-up-describer",
            "made-up-labeller")
        assert kwargs["model_first"] == (
            "labels=off,logs=off,lab=on,calc=on,narrative=on")
        assert "describe_engine" not in kwargs
        assert names == {"logs": "made-up-log", "lab": "made-up-main",
                         "calc": "made-up-main", "narrative": "made-up-main"}
        assert sorted(kwargs["stage_engines"]) == ["calc", "lab", "logs",
                                                   "narrative"]


# ---------------------------------------------------------------------------
# the default, end to end
# ---------------------------------------------------------------------------

def test_left_unset_every_stage_is_read_model_first(synthetic_pdf, tmp_path,
                                                     monkeypatch):
    """The default itself (RI-186): ``model_first`` not passed and the
    variable unset, the made-up report is read model first for every stage,
    as the model-first tests' own run with ``model_first="on"`` is."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.graph import ingest_report
    from report_ingest.tests import test_graph_model_first as tgm

    monkeypatch.delenv(MODEL_FIRST_ENV, raising=False)
    main, describer, labeller, stages = tgm._engines()
    out = tmp_path / "out"
    record = ingest_report(synthetic_pdf, main, out_dir=out, report_id="SYN",
                           describe_engine=describer, range_engine=labeller,
                           stage_engines=stages)
    assert main.n_calls == 2
    assert describer.n_calls == 22 and labeller.n_calls == 2
    assert {stage: e.n_calls for stage, e in stages.items()} == {
        "narrative": 2, "logs": 3, "lab": 1, "calc": 1}
    assert _items(out) == ["mf-calc-20.json", "mf-log-7.json",
                           "mf-log-9.json", "mf-narrative.json"]
    assert not (out / "vision.json").exists()
    labels = json.loads((out / "labels.json").read_text("utf-8"))
    assert labels["model_first"] == "on"
    (note,) = [q for q in record.qa if q.where == "model_first"]
    assert f"{MODEL_FIRST_ENV}=on" in note.detail


def _stopped_at_the_model_first_run(monkeypatch):
    """What each run's stages were when it reached the model-first run,
    which stops it there (no model is called)."""
    from report_ingest import graph

    seen = []

    def first_run(*args, stages=None, **kwargs):
        seen.append(dict(stages))
        raise RuntimeError("made up: stop here")

    monkeypatch.delenv(MODEL_FIRST_ENV, raising=False)
    monkeypatch.setattr(graph, "_model_first_run", first_run)
    return seen


def test_the_folder_run_and_the_tool_take_the_default(monkeypatch, tmp_path):
    """``run_folder`` and the sub-agent tool's ``run_ingest`` (the app's
    report-ingest page reads through it) name no setting, so the default
    reaches the graph: every stage on."""
    from report_ingest.run_folder import run_folder
    from report_ingest.subagent import run_ingest
    from report_ingest.tests.fake_engine import FakeEngine

    seen = _stopped_at_the_model_first_run(monkeypatch)
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "SYN.pdf").write_bytes(b"%PDF")
    run = run_folder(folder, lambda: FakeEngine([]), out_dir=tmp_path / "out")
    assert [row.ok for row in run.reports] == [False]
    with pytest.raises(RuntimeError, match="stop here"):
        run_ingest(str(folder / "SYN.pdf"), [], engine=FakeEngine([]),
                   out_dir=str(tmp_path / "tool"))
    assert seen == [ALL_ON, ALL_ON]
