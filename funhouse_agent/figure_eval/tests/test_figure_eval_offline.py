"""The figure test set (BACKLOG GM-314), offline: fake models, no network
and no model spend."""

from __future__ import annotations

import json
import os

import pytest

from funhouse_agent.figure_eval import checklist, report, runner
from funhouse_agent.figure_eval import request_set as freq
from funhouse_agent.figure_eval.compare import compare, item_outcome

ITEM_IDS = checklist.ITEM_IDS


# ---------------------------------------------------------------------------
# the requests
# ---------------------------------------------------------------------------

def test_the_set_is_about_twenty_general_requests_on_the_sample_inputs():
    defs = freq.definitions()
    rows = freq.sample_rows()
    probes = freq.probes()
    assert 18 <= len(defs) <= 24
    ids = [d["id"] for d in defs]
    assert len({i.split("-")[0] for i in ids}) == len(ids)
    for d in defs:
        assert d["row"] in rows, d["id"]
        assert d["kind"] in freq.KINDS, d["id"]
        assert set(d["tests"]) <= set(ITEM_IDS), d["id"]
        assert d["extra"] and all(x.strip() for x in d["extra"]), d["id"]
        assert "made-up" in d["ask"], d["id"]
        probe = (d.get("replace") or {}).get("probe")
        assert probe is None or probe in probes, d["id"]
    # spread: every kind of figure, most modules, figures nobody names,
    # and every checklist item tested by some request
    assert {d["kind"] for d in defs} == set(freq.KINDS)
    assert len({rows[d["row"]]["module"] for d in defs}) >= 15
    assert sum(1 for d in defs if not d["named"]) >= 8
    assert set().union(*(d["tests"] for d in defs)) == set(ITEM_IDS)


def test_requests_pass_the_sample_rows_own_numbers():
    rows = freq.sample_rows()
    probes = freq.probes()
    defs = {d["id"]: d for d in freq.definitions()}
    for r in freq.resolve():
        assert json.dumps(r["inputs"]) in r["prompt"]
        assert "project_name" not in r["inputs"]
        rep = defs[r["id"]].get("replace")
        if not rep:
            want = {k: v for k, v in rows[r["method"]]["params"].items()
                    if k not in freq.HEADER}
            assert r["inputs"] == want, r["id"]
            assert r["sample_row"]["params"] == rows[r["method"]]["params"]
        else:
            key = rep.get("probe")
            probe = probes[key]["params"] if key else {}
            extra = {k: v for k, v in rep.items() if k != "probe"}
            assert r["inputs"] == {**probe, **extra}, r["id"]
            assert r["sample_row"]["origin"] == (
                "method_probe" if key else "made_up"), r["id"]
            assert "note" not in r["sample_row"], r["id"]
    assert [r["id"] for r in freq.select(freq.resolve(), ["R01"])] == [
        "R01-bearing-width"]
    with pytest.raises(KeyError):
        freq.select(freq.resolve(), ["R99"])


# ---------------------------------------------------------------------------
# the marks
# ---------------------------------------------------------------------------

REQ = {"id": "R1-x", "ask": "Draw it for a made-up job.", "extra": ["must"]}


def test_parse_marks_reads_fenced_json_and_lists_what_is_missing():
    text = "Here:\n```json\n" + json.dumps({"marks": [
        {"id": "axes_units", "mark": "YES", "reason": "both axes"},
        {"id": "kpa_kn", "mark": "n/a", "reason": "no stress"},
        {"id": "x1", "mark": "no", "reason": "missing"},
        {"id": "data_ink", "mark": "maybe", "reason": "?"}],
        "note": "fine"}) + "\n```"
    out = checklist.parse_marks(text, REQ)
    m = out["marks"]
    assert len(m) == len(ITEM_IDS) + 1
    assert (m["axes_units"]["mark"], m["kpa_kn"]["mark"],
            m["x1"]["mark"]) == ("yes", "na", "no")
    assert m["data_ink"]["mark"] == checklist.UNMARKED
    assert "data_ink" in out["unmarked"] and "quantity_axes" in out[
        "unmarked"] and "axes_units" not in out["unmarked"]
    bad = checklist.parse_marks("I cannot see a figure.", REQ)
    assert bad["unmarked"] == [i["id"] for i in checklist.items_for(REQ)]
    assert bad["not_scored"] and out["not_scored"]     # 3 items marked
    full = checklist.parse_marks(json.dumps({"marks": [
        {"id": i["id"], "mark": "yes", "reason": "r"}
        for i in checklist.items_for(REQ)][1:]}), REQ)
    assert not full.get("not_scored") and full["unmarked"] == [ITEM_IDS[0]]


def _check(marks, req=REQ, no_figure=False):
    ids = [i["id"] for i in checklist.items_for(req)]
    return {"marks": {i: {"mark": marks.get(i, "yes"), "reason": "r"}
                      for i in ids}, "no_figure": no_figure,
            "unmarked": [], "note": ""}


def test_the_package_names_what_the_notebook_uses():
    import funhouse_agent.figure_eval as fe
    assert fe.score_figure_set is runner.score_figure_set
    assert fe.ARMS == ("off", "on") and fe.ARM_ENV["off"] == {
        "GEOTECH_FIGURE_HELPER": "1", "GEOTECH_PLOT_CODE": "0"}
    assert fe.prepare is freq.prepare and fe.ITEMS is checklist.ITEMS


def test_item_outcomes():
    assert item_outcome("yes", "yes") == item_outcome("na", "na") == "same"
    assert item_outcome("no", "yes") == "better"
    assert item_outcome("yes", "no") == "worse"
    assert item_outcome("na", "yes") == item_outcome("no", "na") == "one_na"


def test_equal_marks_compare_same_on_every_item_and_overall():
    reqs = [{"id": "R1-a", "extra": ["one"]}, {"id": "R2-b", "extra": []}]
    marks = {iid: ("yes", "no", "na")[k % 3]
             for k, iid in enumerate(ITEM_IDS)}
    checks = {r["id"]: {a: _check(marks, r) for a in ("off", "on")}
              for r in reqs}
    res = compare(reqs, checks)
    assert res["requests"] == {"better": [], "same": ["R1-a", "R2-b"],
                               "worse": []}
    assert res["items"] == {"better": 0, "same": 2 * len(ITEM_IDS) + 1,
                            "worse": 0, "one_na": 0, "unmarked": 0}
    assert not any(res["apart"].values()) and not res["drew_none"]
    assert all(not row["better"] and not row["worse"] and not row["one_na"]
               for row in res["per_item"].values())
    assert res["marks"]["off"] == res["marks"]["on"]
    md = report.render_markdown(_results(reqs, checks, res))
    assert "| same | 2 | R1, R2 |" in md
    assert "| better | 0 | - |" in md and "| worse | 0 | - |" in md


def test_different_marks_give_the_right_better_and_worse_counts():
    reqs = [{"id": f"R{n}-x", "extra": []} for n in (1, 2, 3)]
    no = {iid: "no" for iid in ITEM_IDS}
    checks = {
        "R1-x": {"off": _check({**no, "data_ink": "yes"}, reqs[0]),
                 "on": _check({**no, "axes_units": "yes", "kpa_kn": "yes"},
                              reqs[0])},
        "R2-x": {"off": _check({"plan_drawing": "na"}, reqs[1]),
                 "on": _check({"axes_units": "no"}, reqs[1])},
        "R3-x": {"off": _check({}, reqs[2]), "on": _check({}, reqs[2])}}
    res = compare(reqs, checks)
    assert res["requests"] == {"better": ["R1-x"], "same": ["R3-x"],
                               "worse": ["R2-x"]}
    n = len(ITEM_IDS)
    assert res["items"] == {"better": 2, "worse": 2, "one_na": 1,
                            "same": 3 * n - 5, "unmarked": 0}
    pi = res["per_item"]
    assert pi["axes_units"]["better"] == ["R1-x"]
    assert pi["axes_units"]["worse"] == ["R2-x"]
    assert pi["kpa_kn"]["better"] == ["R1-x"]
    assert pi["data_ink"]["worse"] == ["R1-x"]
    assert pi["plan_drawing"]["one_na"] == ["R2-x"]
    assert res["per_request"]["R1-x"]["counts"] == {
        "better": 2, "same": n - 3, "worse": 1, "one_na": 0, "unmarked": 0}


def test_a_figure_that_did_not_appear_fails_every_item():
    class Eyes:
        def invoke(self, messages):
            raise AssertionError("nothing to look at")
    none = checklist.check_run(Eyes(), REQ, {"answer": "x"}, [])
    assert none["no_figure"] and none["note"] == checklist.NO_FIGURE
    assert {m["mark"] for m in none["marks"].values()} == {"no"}
    failed = checklist.check_run(Eyes(), REQ, {"error": "Timeout"}, [])
    assert failed["marks"]["x1"]["reason"].startswith("the run failed")
    assert failed["run_failed"] and not none.get("run_failed")
    res = compare([REQ], {REQ["id"]: {"off": none,
                                      "on": _check({"to_scale": "na"})}})
    assert res["requests"]["better"] == [REQ["id"]]
    assert res["items"]["better"] == len(ITEM_IDS)   # the items + x1 - 1 n/a
    assert res["items"]["one_na"] == 1
    assert res["drew_none"] == {REQ["id"]: "off"}


def _png(path, label="x"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1], label=label)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


class _Eyes:
    def __init__(self, name, marks=None):
        self.model_name = name
        self.sent = []
        self.marks = marks or {}

    def invoke(self, messages):
        self.sent.append(messages[0].content)
        text = messages[0].content[0]["text"]
        ids = [ln[2:].split(":", 1)[0] for ln in text.splitlines()
               if ln.startswith("- ")]

        class R:
            content = json.dumps({"marks": [
                {"id": i, "mark": self.marks.get(i, "yes"), "reason": "seen"}
                for i in ids], "note": "ok"})
            usage_metadata = {"input_tokens": 900, "output_tokens": 90}
        return R()


def test_check_run_sends_the_figures_at_auto_detail_and_reads_marks(
        tmp_path):
    paths = [_png(tmp_path / f"f{k}.png") for k in range(5)]
    req = dict(REQ, reference={"numbers": {"q_ult_kPa": 1234.5}})
    eyes = _Eyes("gpt-5.4")
    out = checklist.check_run(eyes, req, {"answer": "drawn"}, paths)
    (content,) = eyes.sent
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == checklist.MAX_FIGURES and out["dropped"] == 1
    assert {c["image_url"]["detail"] for c in images} == {"auto"}
    assert "1234.5" in content[0]["text"] and "x1: must" in content[0][
        "text"]
    assert out["marks"]["x1"]["mark"] == "yes" and not out["unmarked"]
    assert out["usage"] == {"input_tokens": 900, "output_tokens": 90}
    assert out["picked_by"] == "newest"
    claude = _Eyes("claude-opus-5-5")
    checklist.check_run(claude, req, {"answer": "drawn"}, paths[:1])
    assert "detail" not in claude.sent[0][1]["image_url"]


# ---------------------------------------------------------------------------
# the runner and the report
# ---------------------------------------------------------------------------

def _results(reqs, checks, cmp_):
    return {"model": "m", "vision_model": "v", "detail": "auto",
            "arms": ["off", "on"], "arm_env": runner.ARM_ENV,
            "requests": [dict(r, kind="curve", ask="a") for r in reqs],
            "runs": {}, "checks": checks, "comparison": cmp_,
            "totals": {a: {"runs": 0, "tokens_in": 0, "tokens_out": 0,
                           "vision_in": 0, "vision_out": 0, "seconds": 0}
                       for a in ("off", "on")}, "written": "now"}


def test_the_runner_writes_results_and_the_page_and_resumes(tmp_path):
    data = tmp_path / "data"
    freq.prepare(data, ["R01", "R17"], references=False, say=lambda s: None)
    calls = []
    failing = {("R17-pile-group-plan", "on")}

    def run_fn(model, req, arm, agent_dir, recursion_limit=None):
        calls.append((req["id"], arm))
        if (req["id"], arm) in failing:
            failing.clear()
            return {"started": "t", "error": "RuntimeError: boom",
                    "files": []}
        os.makedirs(os.path.join(agent_dir, "files"), exist_ok=True)
        _png(os.path.join(agent_dir, "files", f"{arm}.png"))
        return {"started": "t", "answer": f"drew {arm}",
                "files": [f"files/{arm}.png"], "tokens_in": 100,
                "tokens_out": 10, "seconds": 1.5,
                "tool_calls": [{"name": "call_agent", "args": "{}"}]}

    out = tmp_path / "out"
    eyes = _Eyes("gpt-5.4", {"data_ink": "no"})
    res = runner.score_figure_set(object(), vision_model=eyes,
                                  data_dir=data, out_dir=out,
                                  run_fn=run_fn, say=lambda s: None)
    assert len(calls) == 4 and len(eyes.sent) == 3   # the failed run: none
    for name in ("results.json", "RESULTS.md", "side_by_side.html"):
        assert (out / name).is_file()
    cmp_ = res["comparison"]
    assert cmp_["requests"] == {"better": [], "same": ["R01-bearing-width"],
                                "worse": []}
    assert cmp_["apart"]["run_failed"] == {"R17-pile-group-plan": ["on"]}
    run = json.loads((out / "runs" / "R01-bearing-width" / "on" /
                      "run.json").read_text("utf-8"))
    assert run["figures"] == [os.path.join("runs", "R01-bearing-width", "on",
                                           "agent", "files", "on.png")]
    assert run["env"] == {"GEOTECH_FIGURE_HELPER": "1",
                          "GEOTECH_PLOT_CODE": "1"}
    page = (out / "side_by_side.html").read_text("utf-8")
    assert page.count("<img ") == 3 and page.count("<textarea") == 2
    assert "The run failed: RuntimeError: boom" in page
    assert page.index("Plot code OFF") < page.index("Plot code ON")
    md = (out / "RESULTS.md").read_text("utf-8")
    assert "R17-pile-group-plan on: the run failed" in md
    assert "| run failed, not compared | 1 | R17 (ON) |" in md
    assert res["totals"]["off"]["tokens_in"] == 200
    # resumed: finished runs and their checks are kept; the failed run is
    # asked again
    res2 = runner.score_figure_set(object(), vision_model=eyes,
                                   data_dir=data, out_dir=out,
                                   run_fn=run_fn, say=lambda s: None)
    assert calls[4:] == [("R17-pile-group-plan", "on")]
    assert len(eyes.sent) == 4
    assert res2["comparison"]["requests"]["same"] == [
        "R01-bearing-width", "R17-pile-group-plan"]


def test_the_dry_run_goes_through_the_real_page_agent(tmp_path):
    pytest.importorskip("deepagents")
    res = runner.dry_run(tmp_path, ["R01"], references=True,
                         say=lambda s: None)
    ref = json.loads((tmp_path / "data" / "references" / "R01-bearing-width"
                      / "reference.json").read_text("utf-8"))
    assert ref["status"] == "success" and ref["numbers"]
    runs = res["runs"]["R01-bearing-width"]
    assert [os.path.basename(p) for p in runs["off"]["figures"]] == [
        "dry_off.png"]
    assert [os.path.basename(p) for p in runs["on"]["figures"]] == [
        "dry_on.png"]
    assert runs["on"]["env"]["GEOTECH_PLOT_CODE"] == "1"
    pr = res["comparison"]["per_request"]["R01-bearing-width"]
    assert pr["verdict"] == "worse" and pr["counts"]["worse"] == 1
    assert pr["counts"]["one_na"] == 1
    assert (tmp_path / "results" / "side_by_side.html").is_file()


def test_the_figures_marked_are_the_reply_s_final_figures(tmp_path):
    import time
    os.makedirs(tmp_path / ".scratch")
    draft = _png(tmp_path / "bearing_draft.png")
    time.sleep(0.05)
    final = _png(tmp_path / "bearing_vs_width.png")
    time.sleep(0.05)
    module = _png(tmp_path / "package_fig.png")
    time.sleep(0.05)
    look = _png(tmp_path / ".scratch" / "bearing_vs_width_look.png")
    paths = [draft, final, module, look]
    assert checklist.is_look_copy(look) and not checklist.is_look_copy(final)
    shown, how = checklist.choose_figures(
        paths, "The figure is bearing_vs_width.png: q_ult against B.")
    assert (shown, how) == ([final], "reply")
    shown, how = checklist.choose_figures(paths, "Drawn.")
    assert (shown, how) == ([draft, final, module], "newest")
    eyes = _Eyes("gpt-5.4")
    out = checklist.check_run(eyes, REQ, {"answer": "See "
                                          "BEARING_VS_WIDTH.PNG."}, paths)
    images = [c for c in eyes.sent[0] if c["type"] == "image_url"]
    assert len(images) == 1 and out["figures"] == ["bearing_vs_width.png"]
    assert out["picked_by"] == "reply" and out["dropped"] == 2
    rec = runner._record({"id": "R1-x", "prompt": "p"}, "on",
                         {"files": ["files/a.png",
                                    "files/.scratch/a_look.png"]},
                         str(tmp_path / "agent"), str(tmp_path))
    assert rec["figures"] == [os.path.join("agent", "files", "a.png")]


def test_a_run_that_leaves_a_draft_and_a_final_sends_only_the_final(
        tmp_path):
    data = tmp_path / "data"
    freq.prepare(data, ["R01"], references=False, say=lambda s: None)

    def run_fn(model, req, arm, agent_dir, recursion_limit=None):
        os.makedirs(os.path.join(agent_dir, "files"), exist_ok=True)
        for name in ("draft_width.png", "bearing_width.png"):
            _png(os.path.join(agent_dir, "files", name))
        return {"started": "t", "answer": "The figure bearing_width.png "
                "shows q_ult against B.",
                "files": ["files/bearing_width.png", "files/draft_width.png"]}

    eyes = _Eyes("gpt-5.4")
    out = tmp_path / "out"
    res = runner.score_figure_set(object(), vision_model=eyes, data_dir=data,
                                  out_dir=out, run_fn=run_fn,
                                  say=lambda s: None)
    run = res["runs"]["R01-bearing-width"]["on"]
    final = os.path.join("runs", "R01-bearing-width", "on", "agent", "files",
                         "bearing_width.png")
    assert run["figures"] == [final] and run["figures_by"] == "reply"
    assert len(run["files"]) == 2                      # the draft is kept
    for sent in eyes.sent:
        assert len([c for c in sent if c["type"] == "image_url"]) == 1
    chk = res["checks"]["R01-bearing-width"]["on"]
    assert chk["figures"] == ["bearing_width.png"] and not chk["dropped"]
    page = (out / "side_by_side.html").read_text("utf-8")
    assert "draft_width" not in page


def test_an_unreadable_reply_is_asked_once_more_then_kept_apart(tmp_path):
    data = tmp_path / "data"
    freq.prepare(data, ["R01", "R17"], references=False, say=lambda s: None)

    def run_fn(model, req, arm, agent_dir, recursion_limit=None):
        os.makedirs(os.path.join(agent_dir, "files"), exist_ok=True)
        _png(os.path.join(agent_dir, "files", f"{arm}.png"))
        return {"started": "t", "answer": f"drew {req['id']} {arm}",
                "files": [f"files/{arm}.png"]}

    bad = {"R01-bearing-width on": 1, "R17-pile-group-plan off": 9}

    class Eyes(_Eyes):
        def invoke(self, messages):
            text = messages[0].content[0]["text"]
            key = next((k for k in bad if f"drew {k}" in text), None)
            if key and bad[key]:
                bad[key] -= 1
                self.sent.append(messages[0].content)

                class R:
                    content = "I'm sorry, the image is too small to judge."
                    usage_metadata = {}
                return R()
            return super().invoke(messages)

    eyes = Eyes("gpt-5.4")
    out = tmp_path / "out"
    res = runner.score_figure_set(object(), vision_model=eyes,
                                  data_dir=data, out_dir=out,
                                  run_fn=run_fn, say=lambda s: None)
    assert len(eyes.sent) == 6        # R01 on and R17 off asked twice
    cmp_ = res["comparison"]
    assert cmp_["requests"] == {"better": [], "same": ["R01-bearing-width"],
                                "worse": []}
    assert cmp_["apart"]["not_scored"] == {"R17-pile-group-plan": ["off"]}
    chk = json.loads((out / "runs" / "R17-pile-group-plan" / "off" /
                      "check.json").read_text("utf-8"))
    assert chk["not_scored"] and chk["tries"] == 2
    md = (out / "RESULTS.md").read_text("utf-8")
    assert "| vision reply unreadable, not scored | 1 | R17 (OFF) |" in md
    assert "| same | 1 | R01 |" in md
    runner.score_figure_set(object(), vision_model=eyes, data_dir=data,
                            out_dir=out, run_fn=run_fn, say=lambda s: None)
    assert len(eyes.sent) == 6        # asked twice already: kept
    # an item unmarked in both arms is counted apart, never same
    one = _check({"x1": checklist.UNMARKED})
    res = compare([REQ], {REQ["id"]: {"off": one, "on": one}})
    assert res["items"]["unmarked"] == 1
    assert res["items"]["same"] == len(ITEM_IDS)
    assert res["requests"]["same"] == [REQ["id"]]


def test_failed_runs_and_figures_drawn_by_neither_arm_are_kept_apart():
    class Eyes:
        def invoke(self, messages):
            raise AssertionError("nothing to look at")
    reqs = [{"id": f"R{n}-x", "extra": []} for n in (1, 2, 3)]
    failed = checklist.check_run(Eyes(), reqs[0],
                                 {"error": "RateLimitError: 429"}, [])
    assert failed["run_failed"]
    drew_none = [checklist.check_run(Eyes(), r, {"answer": "x"}, [])
                 for r in reqs]
    checks = {"R1-x": {"off": failed, "on": _check({}, reqs[0])},
              "R2-x": {"off": drew_none[1], "on": drew_none[1]},
              "R3-x": {"off": drew_none[2], "on": _check({}, reqs[2])}}
    res = compare(reqs, checks)
    assert res["requests"] == {"better": ["R3-x"], "same": [], "worse": []}
    assert res["apart"] == {"run_failed": {"R1-x": ["off"]},
                            "not_scored": {},
                            "neither_drew": {"R2-x": ["off", "on"]}}
    assert res["drew_none"] == {"R3-x": "off"}
    md = report.render_markdown(_results(reqs, checks, res))
    assert "| run failed, not compared | 1 | R1 (OFF) |" in md
    assert ("| neither arm drew a figure, not compared | 1 | R2 (OFF, ON) |"
            in md)
    assert "| same | 0 | - |" in md and "R3 (OFF drew none)" in md
    assert "Name each run that still fails" in md
    page = report.render_html(_results(reqs, checks, res))
    assert "vision: not compared (run failed)" in page


def test_the_on_arm_builds_the_plot_code_tool_and_the_off_arm_does_not(
        monkeypatch, tmp_path):
    pytest.importorskip("deepagents")
    monkeypatch.setenv("GEOTECH_NO_DISCLAIMER", "1")
    import funhouse_agent.review_eval  # noqa: F401 - hands in the pages
    from aec_agent_core import plot_box, review_flags
    from aec_agent_core.review_eval import runner as page_runner
    made = []
    real = plot_box.make_plot_code_tool

    def counting():
        made.append(1)
        return real()
    monkeypatch.setattr(plot_box, "make_plot_code_tool", counting)
    built = []
    for k, arm in enumerate(("off", "on", "off", "on")):
        made.clear()
        run_dir = tmp_path / str(k)
        run_dir.mkdir()
        with review_flags.switches(runner.ARM_ENV[arm]):
            page_runner.build_page_agent(runner._dry_agent(), {},
                                         str(run_dir), [], page="geotech")
        built.append((arm, len(made)))
    assert [n for a, n in built if a == "off"] == [0, 0], built
    assert all(n > 0 for a, n in built if a == "on"), built
