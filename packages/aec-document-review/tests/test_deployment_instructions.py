"""A deployment's instructions on Document Review's page (REF-64 (B), in the
core's shared assembly; after GeotechStaffEngineer's
``webapp/tests/test_deployment_instructions.py``).

Built through ``tests.page_builds`` (``aec_document_review.core.build_agent``
with the page's profile, a fake model, every ``GEOTECH_*`` unset) with
``GEOTECH_INSTRUCTIONS`` pointing at a folder of made-up text. Unset, the
builds are the recorded fingerprints (``test_review_fingerprints.py``)."""

import time

import pytest

from aec_agent_core.deep import instructions as ins
from tests import page_builds

TAIL = "\n\n" + ins.PROTECTED_RULES + "\n\n" + ins.HEADING + "\n\n"


@pytest.fixture()
def folder(tmp_path):
    (tmp_path / "all.md").write_text("Quote lengths in metres.",
                                     encoding="utf-8")
    (tmp_path / "main.md").write_text("Main agent rule.", encoding="utf-8")
    (tmp_path / "general-purpose.md").write_text("Reading helper rule.",
                                                 encoding="utf-8")
    return str(tmp_path)


def _main_call(calls):
    return next(c for c in calls if c["kind"] == "create_deep_agent")


def test_unset_the_build_appends_nothing():
    agent, calls, _ = page_builds.build("review_default", capture=True)
    assert "appended" not in agent.geotech_build
    kw = _main_call(calls)["kwargs"]
    assert ins.HEADING not in kw["system_prompt"]
    for spec in kw["subagents"]:
        assert ins.HEADING not in (spec.get("system_prompt") or "")


def test_the_default_build_gets_the_block_once_last(folder):
    agent, calls, _ = page_builds.build(
        "review_default", capture=True, settings={ins.ENV: folder})
    kw = _main_call(calls)["kwargs"]
    prompt = kw["system_prompt"]
    assert prompt.endswith(TAIL + "Quote lengths in metres.\n\n"
                           "Main agent rule.")
    assert prompt.count(ins.HEADING) == 1
    assert prompt.count(ins.PROTECTED_RULES) == 1
    roles = agent.geotech_build["appended"]["roles"]
    assert roles["main"]["files"] == ["all.md", "main.md"]
    # the recorded size is the prompt as compiled, the block in it
    assert agent.geotech_build["prompt_chars"] == len(prompt)
    helpers = [s for s in kw["subagents"] if "runnable" not in s]
    assert {s["name"] for s in helpers} >= {"general-purpose"}
    for spec in helpers:
        expected = "Quote lengths in metres."
        if spec["name"] == "general-purpose":
            expected += "\n\nReading helper rule."
        assert spec["system_prompt"].endswith(TAIL + expected), spec["name"]
        assert spec["system_prompt"].count(ins.HEADING) == 1, spec["name"]
        assert spec["name"] in roles


def test_the_minimal_build_appends_nothing(folder):
    """The measuring-stick build calls create_agent itself, not the core's
    compile_agent, so it gets no block (as in GeotechStaffEngineer)."""
    agent, calls, _ = page_builds.build(
        "review_minimal", capture=True, settings={ins.ENV: folder})
    assert calls
    for call in calls:
        assert ins.HEADING not in str(call["kwargs"].get("system_prompt"))
    assert "appended" not in (getattr(agent, "geotech_build", None) or {})


def test_the_turn_record_carries_it(folder, tmp_path, monkeypatch):
    """The page's turn job writes ``turn_start`` with ``engine``
    (``core.engine_record``); its ``build`` carries the appended record."""
    import aec_document_review.core as core
    import aec_document_review.turn_jobs as tj
    from aec_agent_core import activity_log as al
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    agent, _, _ = page_builds.build("review_default",
                                    settings={ins.ENV: folder})
    plain, _, _ = page_builds.build("review_default")
    assert "appended" not in core.engine_record(plain).get("build", {})
    tid = "instructed"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    ctx = {"prompt": "hello", "temp_dir": files,
           "before": core.snapshot_dir(files),
           "before_mtimes": core.snapshot_mtimes(files),
           "staged_inputs": set(), "working_dir": files, "before_wd": None,
           "artifacts": [], "artifacts_before_len": 0,
           "transcript": [{"role": "user", "text": "hello"}],
           "trace_on": False, "model": "m",
           "behavior": core.default_behavior()}
    tj._JOBS.clear()
    job = tj.start_turn_job(agent, [{"role": "user", "content": "hello"}],
                            tid, 40, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 60:
        time.sleep(0.05)
    assert job.done
    tj._JOBS.clear()
    (start,) = [r for r in al.load(core.conversation_dir(tid))
                if r["event"] == "turn_start"]
    rec = start["engine"]["build"]["appended"]
    assert rec["roles"]["main"]["sha"] and rec["roles"]["main"]["chars"]
    assert "general-purpose" in rec["roles"]


def test_the_diagnostics_line(folder, monkeypatch):
    """The connection diagnostics name what each role gets, or none."""
    from aec_document_review import diagnostics as diag
    monkeypatch.delenv(ins.ENV, raising=False)
    line = diag._instructions_check()
    assert line["name"] == "deployment instructions"
    assert line["status"] == diag.PASS and line["detail"].startswith("none")
    monkeypatch.setenv(ins.ENV, folder)
    line = diag._instructions_check()
    assert line["status"] == diag.PASS
    assert "main: all.md + main.md, " in line["detail"]
    assert "general-purpose: all.md + general-purpose.md, " in line["detail"]
    monkeypatch.setenv(ins.ENV, folder + "_missing")
    assert diag._instructions_check()["status"] == diag.WARN
