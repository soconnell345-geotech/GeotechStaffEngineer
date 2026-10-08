"""The suite's measuring tasks (W3) - offline, no model, no network.

Pinned: the two synthetic pages are what their tasks say (the truth is the
fixture's own; the log's depth labels are not centred on their depths, so
reading the centres is biased past the tolerance; the plan's scale note is
wrong and its bar right); ``value_within`` takes right answers and refuses
wrong ones; ``tools_called`` reads the run's own activity and is recorded,
not scored; and on a scripted model each task passes when the model
measures and answers with what ``measure`` returned, and fails when it
answers by eye.
"""

import json
import math
import re

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")
pytest.importorskip("planlens.testing.visual_scale_fixtures")

from funhouse_agent import document_tools, review_flags, scale_labels  # noqa: E402
from funhouse_agent.review_eval import checks as C  # noqa: E402
from funhouse_agent.review_eval import documents as D  # noqa: E402
from funhouse_agent.review_eval.tasks import (  # noqa: E402
    OPEN_TASKS, load_tasks, select)

if not document_tools.has_tool("measure"):            # pragma: no cover
    pytest.skip("installed planlens predates measure", allow_module_level=True)

from planlens.testing.visual_scale_fixtures import (  # noqa: E402
    PLAN_BORINGS, _LOG_DESCRIPTIONS, _overlap)

IDS = {"scale-log-depth", "scale-plan-distance"}


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def log_fx():
    return D.scale_fixture("log")


@pytest.fixture(scope="module")
def plan_fx():
    return D.scale_fixture("plan")


def _task(task_id):
    return next(t for t in OPEN_TASKS if t.id == task_id)


# ---------------------------------------------------------------------------
# The two pages are what their tasks say
# ---------------------------------------------------------------------------

def test_the_log_is_what_its_task_says(log_fx):
    task = _task("scale-log-depth")
    check = task.checks[0]
    rd = log_fx.readings[D.SCALE_LOG_CONTACT]
    assert rd.kind == "line" and rd.value == pytest.approx(check["value"])
    # the layer that starts there is the one the question names
    assert "silty sandy GRAVEL" in _LOG_DESCRIPTIONS[D.SCALE_LOG_CONTACT + 1]
    assert "silty sandy GRAVEL" in task.question
    # one image-only page with no text layer, stored as the real scans are
    doc = fitz.open(stream=log_fx.pdf, filetype="pdf")
    try:
        assert doc.page_count == 1
        assert not doc[0].get_text().strip() and doc[0].get_images()
        assert doc[0].rotation == 270
    finally:
        doc.close()
    # The labels are NOT centred on their depths: a line through the label
    # centres, read at the contact, puts it about 0.10 m too deep - outside
    # the tolerance, so the naive reading fails the task.
    ys = [(lb.box[1] + lb.box[3]) / 2.0 for lb in log_fx.labels]
    vs = [lb.value for lb in log_fx.labels]
    my, mv = sum(ys) / len(ys), sum(vs) / len(vs)
    slope = (sum((y - my) * (v - mv) for y, v in zip(ys, vs))
             / sum((y - my) ** 2 for y in ys))
    naive = mv + slope * (rd.at_pt[1] - my)
    assert check["tol"] < naive - rd.value < 0.15, naive


def test_the_plan_is_what_its_task_says(plan_fx):
    task = _task("scale-plan-distance")
    pts = {name: (e, n) for name, e, n in PLAN_BORINGS}
    a, b = D.SCALE_PLAN_PAIR
    truth = math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
    assert truth == pytest.approx(task.checks[0]["value"], abs=0.01)
    assert a in task.question and b in task.question
    # re-plotted at half size: the note is wrong by 2x, the bar is right
    sc = plan_fx.scales["distance"]
    assert sc["winner"] == "bar"
    assert sc["per_point"] == pytest.approx(2.0 * sc["stated_per_point"])
    doc = fitz.open(stream=plan_fx.pdf, filetype="pdf")
    try:
        text = doc[0].get_text()
    finally:
        doc.close()
    assert "SCALE: 1\" = 20'" in text and "FEET" in text
    assert a in text and b in text


def test_both_are_suite_documents_and_run_by_default(tmp_path):
    for doc_id in ("fixture_scale_log", "fixture_scale_plan"):
        name, data = D.resolve(doc_id)
        assert data[:4] == b"%PDF"
        # the upload's name gives nothing away
        assert not re.search(r"replot|half|scale|gravel|bar", name)
    assert IDS <= {t.id for t in load_tasks()}
    assert {t.id for t in select(OPEN_TASKS, ids=["scale-"])} == IDS
    assert {t.id for t in select(OPEN_TASKS, categories=["measure"])} == IDS
    assert _task("scale-log-depth").page == "geotech"
    assert _task("scale-plan-distance").page == "review"
    from funhouse_agent.review_eval import score_review_suite
    res = score_review_suite(None, ids=["scale-"], out_dir=tmp_path,
                             dry_run=True, verbose=False)
    assert "fixture_scale_log -> boring_log.pdf" in res["results_md"]
    assert "fixture_scale_plan -> site_plan.pdf" in res["results_md"]


def test_an_older_planlens_skips_them_rather_than_failing(monkeypatch):
    def older(which):
        raise ImportError("needs a planlens with the visual-scale fixtures")
    monkeypatch.setattr(D, "scale_fixture", older)
    for doc_id in ("fixture_scale_log", "fixture_scale_plan"):
        with pytest.raises(D.MissingDocument):
            D.resolve(doc_id)


# ---------------------------------------------------------------------------
# The checks: right answers pass, wrong ones fail
# ---------------------------------------------------------------------------

def _passes(task_id, answer, activity=()):
    """Scored the way the runner scores: every check, info checks aside."""
    task = _task(task_id)
    return C.score([C.run_check(c, answer, activity=activity)
                    for c in task.all_checks()])["passed"]


_LOG_RIGHT = [
    "The silty sandy GRAVEL starts at 3.60 m (measured 3.62 m +/- 0.02 m "
    "from the stratum line drawn on the scan).",
    "**Top of the GRAVEL: 3.6 m** below ground.",
    "At 3.65 m the log changes to brown, medium dense, silty sandy GRAVEL.",
    "**Silty sandy GRAVEL**\n- Top: 3.62 m\n- Bottom: 4.87 m",
    "| Layer | Top (m) | Bottom (m) |\n|---|---|---|\n| Grey clean SAND with "
    "fine gravel | 1.64 | 3.62 |\n| Brown silty sandy GRAVEL | 3.62 | 4.87 |",
    "By eye it looked like 3.7 m, but measured from the drawn line the "
    "silty sandy GRAVEL starts at 3.62 m.",
    "It starts at 3.6 m.",                  # the thing not named: first value
]
_LOG_WRONG = [
    # the label centres taken as the depths (the bias the page is built for)
    "The silty sandy GRAVEL starts at about 3.72 m.",
    "The gravel layer starts at about 3.7 m, just above the 4.0 label.",
    "The GRAVEL starts at 3.75 m.",
    # the wrong layer
    "Brown silty sandy GRAVEL from 4.87 m to 6.93 m.",
    "The grey SAND with fine gravel starts at 1.64 m; the GRAVEL starts at "
    "3.8 m.",
    "Layer boundaries at 1.64, 3.62, 4.87 and 6.93 m; the GRAVEL is the "
    "layer from 4.87 m.",
    # the right number in the wrong unit
    "The silty sandy GRAVEL starts at 3.62 ft.",
    "Lines at 1.64 m, 3.62 m and 4.87 m.",  # not named: held to its first
    "I could not tell where the gravel starts.",
]
_PLAN_RIGHT = [
    "B-1 to B-4: 183.8 ft, measured with the graphic scale bar; the note's "
    "1\" = 20' is wrong for this re-plotted sheet.",
    "The distance between borings B-1 and B-4 is about 184 feet. Note: the "
    "sheet's printed scale (1\"=20') does not match its scale bar, which "
    "shows 1\" = 40'; I used the bar.",
    "Borings B-1 and B-4 are 184 ft apart. At the stated scale it would be "
    "92 ft, but the stated scale disagrees with the bar by 50 % (the sheet "
    "was plotted at half size), so the bar governs.",
    "**B1 - B4: 183.85 ft** (+/- 0.08 ft). The title block scale is wrong: "
    "the drawing is at half size.",
]
_PLAN_WRONG = [
    # the stated scale used as printed
    "B-1 and B-4 are about 92 ft apart at the sheet's stated scale of "
    "1\" = 20'.",
    # by eye against the bar
    "B-1 to B-4 is roughly 190 ft by eye against the scale bar; the note "
    "and the bar disagree.",
    # the right distance, but the wrong scale never flagged
    "B-1 to B-4 is 183.8 ft.",
    "There is no discrepancy between the scale note and the bar; B-1 to "
    "B-4 is 183.8 ft.",
    # the wrong pair
    "B-1 to B-2 is 96.2 ft; the scale note and the bar disagree.",
]


@pytest.mark.parametrize("task_id,right,wrong", [
    ("scale-log-depth", _LOG_RIGHT, _LOG_WRONG),
    ("scale-plan-distance", _PLAN_RIGHT, _PLAN_WRONG),
])
def test_the_checks_take_right_answers_and_refuse_wrong_ones(task_id, right,
                                                             wrong):
    assert _passes(task_id, _task(task_id).truth)
    for answer in right:
        assert _passes(task_id, answer), answer
    for answer in wrong:
        assert not _passes(task_id, answer), answer


def test_value_within_reads_only_values_said_with_the_thing():
    near = [{"re": r"\bgravel\b"}]
    kw = {"value": 3.62, "tol": 0.05, "unit": "m", "near": near}
    ok, detail = C.check_value_within("Gravel from 3.62 m.", **kw)
    assert ok and "closest 3.62 m is 0 m" in detail
    # a +/- is a spread, not a value; a range's second end is not its start
    assert not C.check_value_within("Gravel at 3.9 m +/- 3.62 m", **kw)[0]
    assert not C.check_value_within("Gravel: 3.40-3.62 m", **kw)[0]
    # a value said of another thing, far from the name, is not counted
    far = "Gravel is the third layer. " + "x" * 200 + ". The SAND ends at 3.62 m."
    ok, detail = C.check_value_within(far, **kw)
    assert not ok and "no value m stated" in detail
    # feet asked: metres and inches are not feet; a foot mark is
    ft = {"value": 183.85, "tol": 1.0, "unit": "ft",
          "near": [{"re": r"\bb-1\b"}]}
    assert not C.check_value_within("B-1: 183.8 m", **ft)[0]
    assert C.check_value_within("B-1 to B-4 = 183.8'", **ft)[0]
    assert not C.check_value_within("", **ft)[0]


# ---------------------------------------------------------------------------
# The tool-use record: read from the run's activity, not scored
# ---------------------------------------------------------------------------

def _records(calls):
    recs = []
    for i, (name, result, how) in enumerate(calls):
        recs.append({"event": "tool_start", "run_id": str(i), "name": name,
                     "args": {}, "agent": "primary"})
        if how == "end":
            recs.append({"event": "tool_end", "run_id": str(i), "name": name,
                         "result": result, "agent": "primary"})
        elif how == "error":
            recs.append({"event": "tool_error", "run_id": str(i),
                         "name": name, "error": "ValueError: x",
                         "agent": "primary"})
    return recs


def test_tools_called_records_what_each_call_gave_back():
    value = json.dumps({"value": {"depth": 3.619, "plus_minus": 0.016,
                                  "display": "3.6 +/- 0.016"}})
    listed = json.dumps({"value": None, "ambiguous": True,
                         "alternatives": [{"bbox": [1, 2, 3, 4]}]})
    scales = json.dumps({"page": 0, "scales": {"frames": []}})
    grid = json.dumps({"layers": [{"top": 0.0}, {"top": 1.64}]})
    act = _records([
        ("open_document", "{}", "end"), ("measure", value, "end"),
        ("measure", listed, "end"), ("measure", scales, "end"),
        ("measure", json.dumps({"error": "no page 3"}), "end"),
        ("measure", None, "error"), ("log_grid", grid, "end")])
    ok, detail = C.check_tools_called("", activity=act,
                                      tools=["measure", "log_grid"])
    assert ok, detail
    assert detail.startswith("measure: 5 call(s) - value 1, candidates "
                             "listed, none chosen 1, scales listed 1, "
                             "error 2; gave 3.6 +/- 0.016")
    assert detail.endswith("log_grid: 1 call(s) - layers 1; gave 2 layer(s)")
    none = _records([("analyze_pdf_page", "{}", "end")])
    ok, detail = C.check_tools_called("", activity=none,
                                      tools=["measure", "log_grid"])
    assert not ok and detail == "measure: not called; log_grid: not called"
    assert "no activity" in C.check_tools_called("", tools=["measure"])[1]
    # recorded, not scored: a right answer reached another way still passes
    assert "tools_called" in C.PROCESS_CHECKS
    task = _task("scale-log-depth")
    info = next(c for c in task.checks if c["type"] == "tools_called")
    assert info["info"]
    r = C.run_check(info, "", activity=none)
    assert r["info"] and not r["passed"]
    assert _passes("scale-log-depth", _LOG_RIGHT[0], activity=none)


# ---------------------------------------------------------------------------
# End to end on a scripted model: measured passes, by eye fails
# ---------------------------------------------------------------------------

@pytest.fixture
def label_sheets(monkeypatch):
    """Every label sheet drawn is real; the boxes it was drawn from are
    kept so the scripted model can read them as a model would."""
    seen = []
    real = scale_labels.label_sheets

    def spy(doc, items, *a, **kw):
        seen.append(list(items))
        return real(doc, items, *a, **kw)

    monkeypatch.setattr(scale_labels, "label_sheets", spy)
    scale_labels.clear_cache()
    yield seen
    scale_labels.clear_cache()


def _scripted(fx, sheets, *, measure=None, answer=None, eyeball=""):
    """A scripted chat model on the page's agent. With ``measure``: calls
    the real ``measure`` tool on the uploaded page with those arguments,
    then answers ``answer(result)``. Without: answers ``eyeball`` at once
    and calls no tool. It also stands in for the label-reading vision call:
    one ``#N | text`` line per numbered crop, the text of the printed label
    the crop shows (the fixture's)."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    def read_labels():
        lines = []
        for i, (_page, box) in enumerate(sheets[-1]):
            best = max(fx.labels, key=lambda lb: _overlap(box, lb.box))
            text = best.text if _overlap(box, best.box) > 0.2 else "-"
            lines.append(f"#{i + 1} | {text}")
        return "\n".join(lines)

    class Model(FakeMessagesListChatModel):
        turns: list = []
        label_reads: list = []

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            first = messages[0]
            if len(messages) == 1 and isinstance(first.content, list):
                said = " ".join(str(b.get("text", "")) for b in first.content
                                if isinstance(b, dict))
                if "Each numbered cell" in said:
                    self.label_reads.append(1)
                    return self._r(read_labels())
                return self._r("A page.")
            self.turns.append(1)
            if measure is None:
                return self._r(eyeball)
            results = [m for m in messages if m.type == "tool"]
            if not results:
                human = next(m for m in messages if m.type == "human")
                key = re.search(r"'([^']+\.pdf)'",
                                str(human.content)).group(1)
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="", tool_calls=[{
                        "name": "measure", "id": "m1",
                        "args": dict(measure, source=key, page=0)}]))])
            return self._r(answer(json.loads(results[-1].content)))

        def _r(self, text):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=text))])

    return Model(responses=[AIMessage(content="x")], turns=[],
                 label_reads=[])


def _record(run):
    return next(c for c in run["checks"] if c["type"] == "tools_called")


def test_the_log_task_passes_measured_and_fails_by_eye(tmp_path, log_fx,
                                                       label_sheets):
    from funhouse_agent.review_eval.runner import run_task
    task = _task("scale-log-depth")
    rd = log_fx.readings[D.SCALE_LOG_CONTACT]

    def said(out):
        v = out["value"]
        return (f"The silty sandy GRAVEL starts at {v['depth']:.2f} m "
                f"(measured from the stratum line on the scan, +/- "
                f"{v['plus_minus']:.2f} m).")

    model = _scripted(log_fx, label_sheets,
                      measure={"kind": "line", "bbox": list(rd.box_pt)},
                      answer=said)
    run = run_task(task, model, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "measured"))
    assert run["error"] is None, run.get("traceback")
    assert run["page"] == "geotech"
    assert run["score"]["passed"], run["checks"]
    assert len(model.label_reads) == 1          # one label-reading call
    rec = _record(run)
    assert rec["info"] and rec["passed"]
    assert rec["detail"].startswith("measure: 1 call(s) - value 1; gave 3.6")
    assert rec["detail"].endswith("log_grid: not called")

    eye = _scripted(log_fx, label_sheets, eyeball=(
        "The silty sandy GRAVEL starts at about 3.70 m, just above the "
        "4.0 label."))
    run = run_task(task, eye, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "by_eye"))
    assert run["error"] is None, run.get("traceback")
    assert not run["score"]["passed"]
    failed = [c["label"] for c in run["checks"]
              if not c["passed"] and not c["info"]]
    assert failed == ["the top of the GRAVEL within 0.05 m of 3.62 m"]
    rec = _record(run)
    assert not rec["passed"]
    assert rec["detail"] == "measure: not called; log_grid: not called"


def test_the_plan_task_passes_measured_and_fails_by_eye(tmp_path, plan_fx,
                                                        label_sheets):
    from funhouse_agent.review_eval import score_review_suite
    from funhouse_agent.review_eval.runner import run_task
    task = _task("scale-plan-distance")
    a, b = D.SCALE_PLAN_PAIR
    box = {r.tag.split("-B")[0]: list(r.box_pt) for r in plan_fx.readings}
    assert {a, b} <= set(box)

    def said(out):
        return (f"B-1 and B-4 are {out['value']['distance']:.1f} ft apart, "
                f"measured with the sheet's graphic scale bar. Note: "
                f"{out['reconciled'][0]}.")

    model = _scripted(plan_fx, label_sheets,
                      measure={"kind": "distance", "bbox": box[a],
                               "to": box[b]},
                      answer=said)
    res = score_review_suite(model, ids=["scale-plan-distance"],
                             out_dir=tmp_path / "suite", verbose=False)
    run = res["results"]["baseline"]["scale-plan-distance"]
    assert run["error"] is None, run.get("traceback")
    assert run["page"] == "review"
    assert run["score"]["passed"], run["checks"]
    assert not model.label_reads                # the bar's labels are text
    rec = _record(run)
    assert rec["passed"]
    assert rec["detail"].startswith("measure: 1 call(s) - value 1; gave 183.8")
    assert rec["detail"].endswith(" ft")
    md = res["results_md"]
    assert "## Recorded, not scored" in md
    assert ("`baseline` scale-plan-distance: called measure (recorded, not "
            "scored) - yes; measure: 1 call(s)") in md

    eye = _scripted(plan_fx, label_sheets, eyeball=(
        "B-1 and B-4 are about 92 ft apart at the sheet's stated scale of "
        "1\" = 20'."))
    run = run_task(task, eye, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "by_eye"))
    assert run["error"] is None, run.get("traceback")
    assert not run["score"]["passed"]
    failed = {c["label"] for c in run["checks"]
              if not c["passed"] and not c["info"]}
    assert failed == {"B-1 to B-4 within 1 ft of 183.85 ft",
                      "notices that the stated scale disagrees with the bar"}
    assert _record(run)["detail"] == "measure: not called"
