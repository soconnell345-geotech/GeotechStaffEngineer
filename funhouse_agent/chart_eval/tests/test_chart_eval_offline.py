"""The chart suite, offline: scripted chat models, synthetic charts, no
network and no model spend."""

from __future__ import annotations

import base64
import io
import json
import os

import pytest

from funhouse_agent.chart_eval import imaging
from funhouse_agent.chart_eval.arms import run_direct
from funhouse_agent.chart_eval.data import (ChartDataError,
                                            judge_prompt_template,
                                            load_chart_tasks)
from funhouse_agent.chart_eval.judge import (OWN_TEMPLATE, build_prompt,
                                             judge_answer, parse_verdict)
from funhouse_agent.chart_eval.prepare import judge_template_from_source
from funhouse_agent.chart_eval.runner import resolve_arm, score_chart_suite


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def scripted(fn, name="claude-opus-5-5"):
    """A chat model whose reply to the n-th call is ``fn(messages, n)``."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Model(FakeMessagesListChatModel):
        calls: list = []
        bound: list = []
        model: str = name

        def bind_tools(self, tools, **kw):
            self.bound.append([getattr(t, "name", None) for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(list(messages))
            msg = fn(messages, len(self.calls))
            if isinstance(msg, Exception):
                raise msg
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return Model(responses=[AIMessage(content="x")], calls=[], bound=[],
                 model=name)


def ai(text="", tool_calls=None, tokens=(10, 5)):
    from langchain_core.messages import AIMessage
    return AIMessage(content=text, tool_calls=tool_calls or [],
                     usage_metadata={"input_tokens": tokens[0],
                                     "output_tokens": tokens[1],
                                     "total_tokens": sum(tokens)})


def chart_png(path, size=(2000, 1000), square=(1500, 500, 1600, 600)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle(square, fill=(255, 0, 0))
    img.save(path)
    return path


def make_data(tmp_path, n=2, judge_prompt=None):
    root = tmp_path / "data"
    (root / "charts").mkdir(parents=True)
    with open(root / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n):
            chart_png(root / "charts" / f"c{i}.png")
            fh.write(json.dumps({
                "task_id": f"t{i}", "prompt": f"SECRET QUESTION {i}?",
                "golden_answer": f"GOLDEN {i}", "chart_path":
                f"charts/c{i}.png", "domain_combined": "STEM"}) + "\n")
    if judge_prompt:
        (root / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
    return root


def judge_fake(passing_word="42"):
    """A judge (on the package's own prompt) that passes a response
    containing ``passing_word``."""
    def fn(messages, n):
        text = messages[-1].content
        response = text.split("Answer to check:", 1)[1].split(
            "Judge the result")[0].strip()
        return ai("```json\n" + json.dumps({
            "final_answer": response[:20],
            "correct": passing_word in response, "reason": "r"}) + "\n```")
    return scripted(fn, name="judge")


def images_in(message):
    return [b for b in (message.content if isinstance(message.content, list)
                        else []) if isinstance(b, dict)
            and b.get("type") == "image_url"]


def decode(block):
    from PIL import Image
    data = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# imaging
# ---------------------------------------------------------------------------

def test_resized_size_is_the_largest_size_within_the_limits():
    lim = imaging.CLAUDE_HIRES
    for w, h in [(4000, 3000), (3000, 4000), (5000, 500), (800, 600)]:
        rw, rh = imaging.resized_size(w, h, lim)
        assert rw <= w and rh <= h
        assert imaging.count_image_tokens(rw, rh, 28) <= lim.max_tokens
        assert -(-rw // 28) * 28 <= lim.max_edge
        if (w, h) != (rw, rh):
            # one pixel more on the long edge no longer fits
            big = imaging.resized_size(w, h, imaging.ImageLimits(
                "x", lim.max_edge, lim.max_tokens, 28))
            assert big == (rw, rh)
    assert imaging.resized_size(800, 600) == (800, 600)


def test_zoom_crops_the_original_at_the_place_seen_in_the_view(tmp_path):
    lim = imaging.ImageLimits("t", 1000, 100000, 25)
    original = imaging.open_image(chart_png(tmp_path / "c.png"))
    view = imaging.prepare_image(original, lim)
    assert (view.width, view.height) == (1000, 500)
    got = imaging.zoom(original, (1000, 500), 740, 240, 810, 310, lim)
    assert got.ok and got.crop_px == (1480, 480, 1620, 620)
    from PIL import Image
    zoomed = Image.open(io.BytesIO(got.image)).convert("RGB")
    assert got.image[:3] == b"\xff\xd8\xff"            # JPEG, as the recipe
    assert max(zoomed.size) == 1000                      # magnified to fill
    cx, cy = zoomed.width // 2, zoomed.height // 2
    r, g, b = zoomed.getpixel((cx, cy))
    assert r > 200 and g < 60 and b < 60                 # the red square
    bad = imaging.zoom(original, (1000, 500), 50, 50, 40, 60, lim)
    assert not bad.ok and "invalid region" in bad.text
    assert not imaging.zoom(original, (1000, 500), "a", 0, 1, 1, lim).ok


def test_limits_follow_the_model_family(monkeypatch):
    claude = scripted(lambda m, n: ai("x"), name="claude-sonnet-5-5")
    gpt = scripted(lambda m, n: ai("x"), name="GPT_5_4")
    assert imaging.family_of(claude) == "anthropic"
    assert imaging.family_of(gpt) == "openai"
    assert imaging.limits_for(claude) is imaging.CLAUDE_HIRES
    assert imaging.limits_for(gpt).max_edge == 2048      # the host cap
    monkeypatch.setenv("GEOTECH_VISION_MAX_PX", "4096")
    assert imaging.limits_for(gpt).max_edge == 4096
    assert imaging.limits_for(gpt, "claude").name == "claude"
    assert imaging.limits_for(gpt, (900, 500, 32)).max_edge == 900
    with pytest.raises(ValueError):
        imaging.limits_for(gpt, "nope")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_tasks_load_from_jsonl_and_results_name_no_question(tmp_path):
    root = make_data(tmp_path, n=3, judge_prompt="T {{ rubric.response_to_grade }}")
    tasks = load_chart_tasks(root)
    assert [t.id for t in tasks] == ["t0", "t1", "t2"]
    assert os.path.isfile(tasks[0].image_path)
    assert [t.id for t in load_chart_tasks(root, ids=["t2"])] == ["t2"]
    assert len(load_chart_tasks(root, limit=2)) == 2
    rec = tasks[0].public_record()
    assert "SECRET" not in json.dumps(rec) and "GOLDEN" not in json.dumps(rec)
    assert judge_prompt_template(root).startswith("T ")
    with pytest.raises(ChartDataError):
        load_chart_tasks(root, ids=["nope"])


def test_a_missing_chart_or_duplicate_id_stops_the_load(tmp_path):
    root = make_data(tmp_path, n=2)
    os.remove(root / "charts" / "c1.png")
    with pytest.raises(ChartDataError, match="not found"):
        load_chart_tasks(root)
    root2 = tmp_path / "d2"
    root2.mkdir()
    chart_png(root2 / "a.png")
    row = {"task_id": "x", "prompt": "p", "golden_answer": "g",
           "chart_path": "a.png"}
    (root2 / "tasks.jsonl").write_text(json.dumps(row) + "\n"
                                       + json.dumps(row) + "\n")
    with pytest.raises(ChartDataError, match="twice"):
        load_chart_tasks(root2)


def test_surge_task_pack_layout_loads(tmp_path):
    root = tmp_path / "pack"
    (root / "tasks").mkdir(parents=True)
    (root / "charts").mkdir()
    chart_png(root / "charts" / "s.png")
    (root / "manifest.yaml").write_text("tasks:\n- tasks/one.yaml\n")
    (root / "tasks" / "one.yaml").write_text(
        "task_id: s1\nprompt: what?\ngolden_answer: '5'\n"
        "chart_path: charts/s.png\n")
    assert [t.id for t in load_chart_tasks(root)] == ["s1"]


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

def test_verdicts_parse_strictly():
    # this package's form, in a code fence
    ok = parse_verdict('```json\n{"final_answer": "17.5", "correct": true, '
                       '"reason": "the same value"}\n```')
    assert ok == [{"part": "1", "answer": "17.5", "correct": True,
                   "reason": "the same value"}]
    # parts listed under "scores", with a sentence round the object
    parts = parse_verdict('Verdict: {"scores": [{"score": 1}, '
                          '{"score": "0"}, {"score": false}]} (done)')
    assert [p["correct"] for p in parts] == [True, False, False]
    assert [p["part"] for p in parts] == ["1", "2", "3"]
    for bad in ('{"scores": []}', '{"scores": [{"score": 0.5}]}',
                '{"correct": "partly"}', '{"correct": null}',
                '{"scores": [3]}', '{"reason": "no decision"}',
                '{"correct": tru', 'no json',
                # a verdict cut off part way never reads as its first part
                '{"scores": [{"score": 1}, {"score": 0}'):
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_judge_grades_all_parts_and_reports_unscored():
    def two_parts(messages, n):
        return ai(json.dumps({"scores": [{"score": 1}, {"score": 0}]}))
    assert judge_answer(scripted(two_parts), "q", "g", "r")["passed"] is False
    good = judge_answer(judge_fake(), "q", "g", "it is 42")
    assert good["passed"] is True and good["prompt"] == "own"
    garbage = scripted(lambda m, n: ai("I think it's fine"))
    res = judge_answer(garbage, "q", "g", "r", retries=2)
    assert res["passed"] is None and res["attempts"] == 3
    empty = scripted(lambda m, n: ai("never asked"))
    assert judge_answer(empty, "q", "g", "  ")["passed"] is False
    assert empty.calls == []


def test_the_data_folders_judge_prompt_is_used_word_for_word():
    tpl = ("Prompt: {{ rubric.response_generation_prompt }}\nResponse: "
           "{{ rubric.response_to_grade }}\nGolden answer: "
           "{{ rubric.all_criteria }}")
    text = build_prompt("Q?", "G!", "R.", tpl)
    assert text.startswith("Prompt: Q?\nResponse: R.\nGolden answer: G!")
    assert "JSON object only" in text
    own = build_prompt("Q?", "G!", "R.")
    assert "Q?" in own and "{{" not in own and "<|" not in own
    assert own.startswith(OWN_TEMPLATE.split("\n")[0])
    assert own.index("Q?") < own.index("G!") < own.index("R.")
    # the package's own prompt asks for the reply its parser reads
    asked = own[own.index('{"final_answer"'):].split("\n")[0]
    assert parse_verdict(asked)[0]["correct"] is True


def test_the_judge_prompt_is_read_from_surges_source_without_running_it():
    src = ('import os\nos.system("echo never")\n'
           'JUDGE_PROMPT_TEMPLATE = """Read {{ rubric.all_criteria }}"""\n')
    assert judge_template_from_source(src) == "Read {{ rubric.all_criteria }}"
    assert judge_template_from_source("x = 1") is None


# ---------------------------------------------------------------------------
# direct arms
# ---------------------------------------------------------------------------

def test_none_arm_sends_the_question_then_the_chart(tmp_path):
    chart = chart_png(tmp_path / "c.png", size=(4000, 2000))
    for name, detail in (("claude-opus-5-5", None), ("GPT_5_6", "original")):
        model = scripted(lambda m, n: ai("The answer is 42"), name=name)
        run = tmp_path / name
        res = run_direct("none", model, "How tall?", str(chart), str(run))
        assert res["answer"] == "The answer is 42" and res["model_calls"] == 1
        sent = model.calls[0][0]
        assert sent.content[0] == {"type": "text", "text": "How tall?"}
        block = images_in(sent)[0]
        assert block["image_url"].get("detail") == detail
        assert decode(block).size == tuple(res["chart"]["view_px"])
        assert res["usage"] == {"input_tokens": 10, "output_tokens": 5}
        lines = [json.loads(x) for x in
                 (run / "transcript.jsonl").read_text().splitlines()]
        assert [x["event"] for x in lines] == ["image_shown", "model_call"]
        assert os.path.isfile(run / lines[0]["file"])


def test_zoom_arm_runs_the_cookbook_loop(tmp_path):
    chart = chart_png(tmp_path / "c.png")

    def fn(messages, n):
        if n == 1:
            return ai("Let me look closer.", [{"name": "zoom", "id": "z1",
                                              "args": {"x1": 1400, "y1": 400,
                                                       "x2": 1700,
                                                       "y2": 700}}])
        return ai("Final: 42")

    model = scripted(fn, name="claude-opus-5-5")
    res = run_direct("zoom", model, "Q?", str(chart), str(tmp_path / "r"))
    assert res["answer"] == "Final: 42" and res["stop"] == "answered"
    assert res["tool_counts"] == {"zoom": 1} and res["model_calls"] == 2
    assert model.bound == [["zoom"]]
    first = model.calls[0][0]
    texts = [b["text"] for b in first.content if b.get("type") == "text"]
    assert "The question is: Q?" in texts[0] and "2000x1000" in texts[1]
    second = model.calls[1]
    tool_msg, image_msg = second[-2], second[-1]
    assert tool_msg.type == "tool" and tool_msg.tool_call_id == "z1"
    assert "Zoomed into (1400,400)-(1700,700)" in tool_msg.content
    zoomed = decode(images_in(image_msg)[0])
    r, g, b = zoomed.getpixel((zoomed.width // 2, zoomed.height // 2))
    assert r > 200 and g < 60                            # the red square
    assert len(os.listdir(tmp_path / "r" / "images")) == 2


def test_zoom_arm_stops_at_the_step_cap_and_asks_for_an_answer(tmp_path):
    chart = chart_png(tmp_path / "c.png", size=(400, 300))
    call = [{"name": "zoom", "id": "z", "args": {"x1": 0, "y1": 0, "x2": 10,
                                                  "y2": 10}}]
    model = scripted(lambda m, n: ai(f"still looking {n}", call))
    res = run_direct("zoom", model, "Q?", str(chart), str(tmp_path / "r"),
                     max_steps=3)
    assert res["stop"] == "step_cap" and res["model_calls"] == 4
    assert res["answer"] == "still looking 4"
    assert "last tool call" in model.calls[3][-1].content


def test_an_empty_turn_gets_one_nudge(tmp_path):
    chart = chart_png(tmp_path / "c.png", size=(400, 300))
    model = scripted(lambda m, n: ai("" if n == 1 else "7"))
    res = run_direct("zoom", model, "Q?", str(chart), str(tmp_path / "r"))
    assert res["answer"] == "7" and res["model_calls"] == 2


def test_bad_tool_input_comes_back_as_an_error_not_a_crash(tmp_path):
    chart = chart_png(tmp_path / "c.png", size=(400, 300))

    def fn(messages, n):
        if n == 1:
            return ai("", [{"name": "zoom", "id": "a", "args": {
                "x1": 50, "y1": 50, "x2": 10, "y2": 10}},
                {"name": "crop", "id": "b", "args": {}}])
        return ai("done")

    model = scripted(fn)
    res = run_direct("zoom", model, "Q?", str(chart), str(tmp_path / "r"))
    tools = [m for m in model.calls[1] if m.type == "tool"]
    assert [t.status for t in tools] == ["error", "error"]
    assert "Unknown tool: crop" in tools[1].content
    assert res["answer"] == "done"


def test_a_busy_model_is_waited_for_and_a_refused_detail_dropped(tmp_path):
    chart = chart_png(tmp_path / "c.png", size=(400, 300))
    waits = []

    class RateLimitError(Exception):        # read by type and status, as the
        status_code = 429                    # app's busy check reads it

    def fn(messages, n):
        if n == 1:
            return RateLimitError("Error code: 429 rate limit exceeded")
        if n == 2:
            return ValueError("Invalid value for 'detail': 'original' is not "
                              "supported with this model.")
        return ai("ok")

    model = scripted(fn, name="GPT_5_2")
    res = run_direct("none", model, "Q?", str(chart), str(tmp_path / "r"),
                     sleep=waits.append)
    assert res["answer"] == "ok" and waits == [15]
    assert res["detail_dropped"] is True
    assert "detail" not in images_in(model.calls[-1][0])[0]["image_url"]


def test_zoom_code_runs_python_on_the_full_resolution_chart(tmp_path):
    from funhouse_agent import plot_box
    if not plot_box.available():
        pytest.skip("the plotting box needs matplotlib")
    chart = chart_png(tmp_path / "c.png", size=(640, 480),
                      square=(100, 100, 199, 149))
    code = ("import matplotlib.pyplot as plt\nimport numpy as np\n"
            "img = plt.imread('chart.png')\n"
            "red = (img[:, :, 0] > 0.9) & (img[:, :, 1] < 0.1)\n"
            "ys, xs = np.nonzero(red)\n"
            "print(img.shape[1], img.shape[0], xs.min(), xs.max(), "
            "ys.min(), ys.max())\n")

    def fn(messages, n):
        if n == 1:
            return ai("", [{"name": "run_python", "id": "p1",
                            "args": {"code": code}}])
        return ai("measured")

    model = scripted(fn)
    res = run_direct("zoom_code", model, "Q?", str(chart),
                     str(tmp_path / "r"))
    assert model.bound == [["zoom", "run_python"]]
    tool = [m for m in model.calls[1] if m.type == "tool"][0]
    assert tool.status == "success", tool.content
    assert "640 480 100 199 100 149" in tool.content
    assert res["tool_counts"] == {"run_python": 1}


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def test_arm_names_resolve():
    assert resolve_arm("zoom") == ("direct", "zoom", {})
    assert resolve_arm("page")[1] == "page:baseline"
    assert resolve_arm("page:minimal")[2] == {"GEOTECH_REVIEW_AGENT":
                                              "minimal"}
    for bad in ("zoomz", "page:nope"):
        with pytest.raises(ValueError):
            resolve_arm(bad)


def test_suite_runs_grades_writes_results_and_resumes(tmp_path):
    root = make_data(tmp_path, n=2)
    model = scripted(lambda m, n: ai("It is 42" if "QUESTION 0" in str(
        m[0].content) else "It is 7"))
    judge = judge_fake("42")
    out = tmp_path / "out"
    res = score_chart_suite(model, judge_model=judge, data_dir=root,
                            out_dir=out, arms=("none", "zoom"), workers=2,
                            verbose=False)
    for arm in ("none", "zoom"):
        assert res["results"][arm]["t0"]["correct"] is True
        assert res["results"][arm]["t1"]["correct"] is False
        assert (out / "runs" / arm / "t0" / "run.json").is_file()
        assert (out / "runs" / arm / "t0" / "transcript.jsonl").is_file()
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "| none | 1 | 2 | 50.0 % |" in md and "| zoom | 1 | 2 |" in md
    assert "| t0 | STEM | ✓ | ✓ |" in md
    assert "SECRET" not in md and "GOLDEN" not in md     # ids only
    n_model, n_judge = len(model.calls), len(judge.calls)
    score_chart_suite(model, judge_model=judge, data_dir=root, out_dir=out,
                      arms=("none", "zoom"), verbose=False)
    assert (len(model.calls), len(judge.calls)) == (n_model, n_judge)
    score_chart_suite(model, judge_model=judge, data_dir=root, out_dir=out,
                      arms=("none", "zoom"), rejudge=True, verbose=False)
    assert len(model.calls) == n_model and len(judge.calls) == n_judge + 4


def test_a_failed_run_is_retried_and_never_graded(tmp_path):
    root = make_data(tmp_path, n=1)
    broken = scripted(lambda m, n: ValueError("proxy said no"))
    judge = judge_fake()
    out = tmp_path / "out"
    res = score_chart_suite(broken, judge_model=judge, data_dir=root,
                            out_dir=out, arms=("none",), verbose=False)
    run = res["results"]["none"]["t0"]
    assert "proxy said no" in run["error"] and run["correct"] is None
    assert judge.calls == []
    assert "| t0 | STEM | E |" in (out / "RESULTS.md").read_text("utf-8")
    good = scripted(lambda m, n: ai("42"))
    res = score_chart_suite(good, judge_model=judge, data_dir=root,
                            out_dir=out, arms=("none",), verbose=False)
    assert res["results"]["none"]["t0"]["correct"] is True


def test_repeats_and_dry_run(tmp_path):
    root = make_data(tmp_path, n=1)
    out = tmp_path / "out"
    dry = score_chart_suite(None, data_dir=root, out_dir=out,
                            arms=("none", "page:baseline"), repeats=2,
                            dry_run=True, verbose=False)
    assert "4 runs still to do" in dry["results_md"]
    model = scripted(lambda m, n: ai("42"))
    res = score_chart_suite(model, judge_model=judge_fake(), data_dir=root,
                            out_dir=out, arms=("none",), repeats=2,
                            verbose=False)
    assert set(res["results"]["none"]) == {"t0", "t0__r2"}
    assert "| t0 | STEM | 2/2 |" in (out / "RESULTS.md").read_text("utf-8")


def test_page_arm_asks_the_document_review_page(tmp_path):
    root = make_data(tmp_path, n=1)
    model = scripted(lambda m, n: ai("The value is 42."))
    out = tmp_path / "out"
    res = score_chart_suite(model, judge_model=judge_fake(), data_dir=root,
                            out_dir=out, arms=("page:baseline",),
                            verbose=False)
    run = res["results"]["page:baseline"]["t0"]
    assert run["error"] is None, run.get("traceback")
    assert run["answer"] == "The value is 42." and run["correct"] is True
    page = out / "runs" / "page_baseline" / "t0" / "page"
    assert (page / "activity.jsonl").is_file()
    # the chart went in as the user's upload: the page's attachment note
    first = model.calls[0]
    assert any("c0.png" in str(m.content) for m in first)


@pytest.mark.parametrize("arm", ["page:baseline", "page:minimal"])
def test_page_agents_look_at_the_uploaded_chart(tmp_path, arm):
    """The page's agent looks at the PNG upload with analyze_image; the
    vision call gets the chart's pixels."""
    root = make_data(tmp_path, n=1)
    seen = []

    def fn(messages, n):
        imgs = [b for m in messages for b in images_in(m)]
        if imgs:                                # the vision side call
            seen.append(decode(imgs[0]).size)
            return ai("A red square near the right edge; value 42.")
        if not any(m.type == "tool" for m in messages):
            return ai("", [{"name": "analyze_image", "id": "a1", "args": {
                "attachment_key": "c0.png", "prompt": "Read the chart."}}])
        return ai("The value is 42.")

    model = scripted(fn)
    out = tmp_path / "out"
    res = score_chart_suite(model, judge_model=judge_fake(), data_dir=root,
                            out_dir=out, arms=(arm,), verbose=False)
    run = res["results"][arm]["t0"]
    assert run["error"] is None, run.get("traceback")
    assert run["tool_counts"].get("analyze_image") == 1, run["tool_counts"]
    assert seen and max(seen[0]) >= 1000     # the chart, not a placeholder
    assert run["correct"] is True


@pytest.mark.skipif(os.name != "posix", reason="forked page runs are POSIX")
def test_page_runs_in_forked_children(tmp_path):
    root = make_data(tmp_path, n=2)
    model = scripted(lambda m, n: ai("It is 42"))
    out = tmp_path / "out"
    res = score_chart_suite(model, judge_model=judge_fake(), data_dir=root,
                            out_dir=out, arms=("page:baseline",),
                            page_workers=2, verbose=False)
    runs = res["results"]["page:baseline"]
    assert set(runs) == {"t0", "t1"}
    assert all(r["error"] is None and r["correct"] is True
               for r in runs.values()), runs


def client_model():
    """A chat model that builds its HTTP client on first use and keeps it,
    as ChatAnthropic does; its reply names the process that built it."""
    from functools import cached_property

    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ClientModel(FakeMessagesListChatModel):
        model: str = "claude-made-up-5"

        @cached_property
        def _client(self):
            return {"pid": os.getpid()}

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            text = f"It is 42; client built in process {self._client['pid']}"
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content=text))])

    return ClientModel(responses=[AIMessage(content="x")])


def test_a_child_drops_the_clients_it_inherits():
    from funhouse_agent.chart_eval import runner
    lca = pytest.importorskip("langchain_anthropic._client_utils")
    url = "https://proxy.example.invalid"
    shared = lca._get_default_httpx_client(base_url=url, timeout=30)
    model = client_model()
    inherited = model._client                 # made before the fork
    same, clients = runner.model_for_child(model)
    assert same is model and clients["rebuilt"] is False
    assert lca._get_default_httpx_client.cache_info().currsize == 0
    assert lca._get_default_httpx_client(base_url=url, timeout=30) \
        is not shared
    assert model._client is not inherited     # built again on first use
    assert "ClientModel._client" in clients["dropped"]
    assert ("langchain_anthropic._client_utils._get_default_httpx_client"
            in clients["dropped"])


def test_a_model_foundry_model_built_is_built_again_in_the_child(
        monkeypatch):
    import webapp.engine_config as engine_config

    import funhouse_agent.chart_eval as ce
    from funhouse_agent.chart_eval import runner
    built = []

    class Resolution:
        message = "made up"

        def __init__(self):
            self.model = client_model()

    def resolve(mid):
        built.append(mid)
        return Resolution()

    monkeypatch.setattr(engine_config, "resolve_engine", resolve)
    model = ce.foundry_model("Made-up label=anthropic:made-up-id")
    fresh, clients = runner.model_for_child(model)
    assert fresh is not model and clients["rebuilt"] is True
    assert built == ["anthropic:made-up-id", "anthropic:made-up-id"]
    assert ce.rebuild_model(client_model()) is None


@pytest.mark.skipif(os.name != "posix", reason="forked page runs are POSIX")
def test_forked_page_runs_call_the_model_through_clients_of_their_own(
        tmp_path, monkeypatch):
    from funhouse_agent.chart_eval import runner
    root = make_data(tmp_path, n=2)
    model = client_model()
    parent_client = model._client             # as after the direct runs

    def page(model, task, run_dir, env, arm, **kw):
        reply = model.invoke("Read the chart.")
        return {"answer": reply.content, "stop": "answered",
                "model_calls": 1, "usage": {}, "tool_counts": {}}

    monkeypatch.setattr(runner, "run_page", page)
    res = score_chart_suite(model, judge_model=judge_fake(), data_dir=root,
                            out_dir=tmp_path / "out", arms=("page:baseline",),
                            page_workers=2, verbose=False)
    runs = res["results"]["page:baseline"]
    assert set(runs) == {"t0", "t1"}
    for run in runs.values():
        child = run["child_clients"]["pid"]
        assert child != os.getpid()
        assert run["answer"].endswith(f"process {child}")
        assert "ClientModel._client" in run["child_clients"]["dropped"]
    assert model._client is parent_client     # the parent's own, untouched


# ---------------------------------------------------------------------------
# CharXiv
# ---------------------------------------------------------------------------

CHARXIV_CONSTANTS = '''
import os
os.system("never run")
REASONING_GRADING_PREFIX = """PREFIX """
REASONING_GRADING_INST = {
    1: """T1 Q:<|question|> G:<|ground_truth|> R:<|response|>""",
    2: """T2 Q:<|question|> G:<|ground_truth|> R:<|response|>""",
    3: """T3 Q:<|question|> G:<|ground_truth|> R:<|response|>""",
    4: """T4 Q:<|question|> G:<|ground_truth|> R:<|response|>""",
}
REASONING_RESP_INST = {
    1: """{}
    * text rule""",
    2: """{}
    * phrase rule""",
    3: """{}
    * number rule""",
    4: """{}
    {}""",
}
'''


def charxiv_fetcher(tmp_path):
    from funhouse_agent.chart_eval import charxiv
    answers = [("Joint-CNN", 1), ("up then down", 2), ("12", 3), ("0.25", 4),
               ("7", 4), ("B", 1)]
    questions = {str(i): {"figure_id": i, "query": f"Q{i}?", "answer": a,
                          "inst_category": c, "qa_source": 1}
                 for i, (a, c) in enumerate(answers)}
    meta = {str(i): {"category": "cs" if i % 2 else "physics",
                     "paper_id": f"2004.{i:05d}"} for i in range(6)}
    buf = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(buf, "w") as z:
        for i in list(range(6)) + [1500]:
            img = io.BytesIO()
            from PIL import Image
            Image.new("RGB", (640, 480), "white").save(img, format="JPEG")
            z.writestr(f"{i}.jpg", img.getvalue())
    files = {
        charxiv.GITHUB_RAW + "src/constants.py": CHARXIV_CONSTANTS.encode(),
        charxiv.GITHUB_RAW + "data/reasoning_val.json":
            json.dumps(questions).encode(),
        charxiv.GITHUB_RAW + "data/image_metadata_val.json":
            json.dumps(meta).encode(),
        charxiv.IMAGES_URL: buf.getvalue(),
    }
    return files.__getitem__


def test_charxiv_folder_holds_the_questions_as_charxiv_asks_them(tmp_path):
    from funhouse_agent.chart_eval.charxiv import prepare_charxiv
    out = tmp_path / "cx"
    prov = prepare_charxiv(out, sample=3, fetcher=charxiv_fetcher(tmp_path))
    assert prov["tasks"] == 6 and prov["by_answer_type"] == {
        "1": 2, "2": 1, "3": 1, "4": 2}
    rows = [json.loads(x) for x in
            (out / "tasks.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["prompt"].startswith("Q0?\n") and "text rule" in \
        rows[0]["prompt"]
    assert "2 decimal places" in rows[3]["prompt"]
    assert "exact integer" in rows[4]["prompt"]
    assert rows[3]["judge_question"] == "Q3?" and rows[3]["judge_key"] == "4"
    assert rows[1]["source_url"] == "https://arxiv.org/abs/2004.00001"
    assert sorted(os.listdir(out / "charts")) == [f"{i}.jpg"
                                                  for i in range(6)]
    prompts = json.loads((out / "judge_prompts.json").read_text("utf-8"))
    assert set(prompts) == {"1", "2", "3", "4"}
    assert prompts["3"].startswith("PREFIX T3")
    ids = (out / "sample_3.txt").read_text().split()
    assert len(ids) == 3
    assert (out / "ATTRIBUTION.md").is_file()
    tasks = load_chart_tasks(out, subset="sample_3.txt")
    assert [t.id for t in tasks] == ids


def test_charxiv_sample_keeps_the_answer_type_mix():
    from funhouse_agent.chart_eval.charxiv import stratified_sample
    rows = [{"task_id": str(i), "judge_key": str(1 + (i % 10 >= 4) +
                                                 (i % 10 >= 6) +
                                                 (i % 10 >= 8))}
            for i in range(1000)]
    ids = stratified_sample(rows, 100, seed=0)
    assert len(ids) == 100 and ids == stratified_sample(rows, 100, seed=0)
    kinds = [rows[int(i)]["judge_key"] for i in ids]
    assert {k: kinds.count(k) for k in "1234"} == {"1": 40, "2": 20,
                                                   "3": 20, "4": 20}


def test_charxiv_verdicts_and_slots():
    got = parse_verdict('{"extracted_answer": "12", "score": 1}')
    assert got[0]["correct"] is True and got[0]["answer"] == "12"
    tpl = "Q:<|question|> G:<|ground_truth|> R:<|response|>"
    text = build_prompt("q?", "7", "it says <|ground_truth|>", tpl)
    assert text.startswith("Q:q? G:7 R:it says <|ground_truth|>")


def test_suite_grades_each_charxiv_task_with_its_own_prompt(tmp_path):
    from funhouse_agent.chart_eval.charxiv import prepare_charxiv
    data = tmp_path / "cx"
    prepare_charxiv(data, sample=3, fetcher=charxiv_fetcher(tmp_path))
    model = scripted(lambda m, n: ai("My answer: 12"))
    seen = []

    def fn(messages, n):
        text = messages[-1].content
        seen.append(text)
        golden = text.split(" G:", 1)[1].split(" R:", 1)[0]
        return ai(json.dumps({"extracted_answer": "12",
                              "score": int(golden == "12")}))

    judge = scripted(fn, name="judge")
    out = tmp_path / "out"
    res = score_chart_suite(model, judge_model=judge, data_dir=data,
                            out_dir=out, arms=("none",), verbose=False)
    runs = res["results"]["none"]
    assert len(runs) == 6
    assert runs["2"]["correct"] is True and runs["0"]["correct"] is False
    assert runs["2"]["judge"]["prompt"] == "data folder (3)"
    assert any(t.startswith("PREFIX T3 Q:Q2? G:12 R:My answer: 12")
               for t in seen)                      # the bare question
    md = (out / "RESULTS.md").read_text("utf-8")
    assert md.startswith("# CharXiv reasoning, validation split:")
    assert "per task (judge_prompts.json)" in md
    small = score_chart_suite(model, judge_model=judge, data_dir=data,
                              out_dir=tmp_path / "out2", arms=("none",),
                              subset="sample_3.txt", verbose=False)
    assert len(small["results"]["none"]) == 3
