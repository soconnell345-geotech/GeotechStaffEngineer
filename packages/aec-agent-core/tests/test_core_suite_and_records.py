"""Step 11 (c) in the core (BACKLOG INF-32; plan
``module_work/APP_SPLIT_PLAN.md`` sections 2.3 and 2.7 B): the review
suite's framework runs a task on a page an app hands in, and the records
and checks that asked GSE for something ask a hook instead.

These run with only the core and planlens installed: the page host here is
a made-up one (no model, no app), and the hooks are filled and emptied by
the tests themselves. Made-up values only.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from aec_agent_core import hooks


@pytest.fixture
def clean_hooks(monkeypatch):
    """No app's hooks (restored afterwards)."""
    monkeypatch.setattr(hooks, "_HOOKS", {})
    monkeypatch.setattr(hooks, "_RESOLVED", {})


def _fake_host(answer: str):
    """A page host whose 'agent' answers ``answer`` in one turn and writes
    a file, as a page's turn would."""
    host = types.SimpleNamespace(
        APP_DIST="made-up-app", APP_PACKAGE="made_up_app",
        APP_COMMIT_ENV="MADE_UP_APP_COMMIT", calls=[])

    def build_agent(model, attachments, files_dir, artifacts, *, page,
                    markup_author):
        host.calls.append(("build", page, markup_author))
        return {"files_dir": files_dir}

    def stage_upload(attachments, files_dir, name, data):
        import os
        path = os.path.join(files_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        attachments[name] = data
        return types.SimpleNamespace(path=path, name=name)

    def stream_turn(agent, history, thread, *, recursion_limit, callbacks):
        import os
        host.calls.append(("turn", history[-1]["content"], recursion_limit))
        with open(os.path.join(agent["files_dir"], "memo.txt"), "w") as fh:
            fh.write("made-up memo")
        yield {"kind": "turn_done", "answer": answer, "turn_tokens": 42}

    host.build_agent = build_agent
    host.stage_upload = stage_upload
    host.attachment_note = lambda atts: f"[{len(atts)} attached]"
    host.assemble_user_message = lambda notes, text: " ".join(
        list(notes) + [text])
    host.orientation_request = lambda atts, page: None
    host.stream_turn = stream_turn
    host.default_recursion_limit = lambda: 7
    host.engine_record = lambda agent: {"model": "made-up"}
    host.tools_fingerprint = lambda agent: {"names": []}
    host.output_collector = lambda: None
    host.import_reported_outputs = lambda *a, **k: {}
    return host


def test_without_a_page_host_the_suite_says_so(clean_hooks):
    from aec_agent_core.review_eval import runner
    with pytest.raises(runner.NoPageHost):
        runner.page_host()
    assert runner._app() == ("aec-agent-core", "aec_agent_core",
                             "AEC_CORE_COMMIT")


def test_a_task_runs_on_the_page_an_app_hands_in(tmp_path, clean_hooks):
    from aec_agent_core.review_eval import runner
    from aec_agent_core.review_eval.tasks import Task
    host = _fake_host("The wall is 3.5 m high.")
    hooks.set_hook(runner.PAGE_HOST_HOOK, lambda: host)
    for attr in runner.PAGE_HOST_ATTRIBUTES:
        assert hasattr(host, attr), attr
    task = Task(id="made-up", question="How high is the wall?",
                documents=["fixture_review_document"], category="locate",
                doc_type="markup_set",
                checks=[{"type": "contains_all", "terms": ["3.5"],
                         "label": "says the height"}])
    res = runner.run_task(task, model=None, arm="baseline", arm_env={},
                          docs_dir=None, run_dir=str(tmp_path / "run"))
    assert res["error"] is None, res.get("traceback") or res["error"]
    assert res["answer"] == "The wall is 3.5 m high."
    assert res["score"]["passed"], res["checks"]
    # the runner records os.path.relpath (backslashes on Windows)
    assert [f.replace(os.sep, "/") for f in res["files"]] == ["files/memo.txt"]
    assert host.calls[0] == ("build", "review", runner.SUITE_AUTHOR)
    assert host.calls[1][2] == 7                    # the page's own limit
    assert res["versions"]["made-up-app"]
    assert set(res["commits"]) == {"app", "planlens"}


def test_the_open_tasks_come_from_the_app(clean_hooks):
    from aec_agent_core.review_eval import tasks
    assert tasks.load_tasks() == []
    mine = [tasks.Task(id="t1", question="q", documents=[],
                       category="orient", doc_type="long_text", checks=[])]
    hooks.set_hook(tasks.OPEN_TASKS_HOOK, lambda: mine)
    assert [t.id for t in tasks.load_tasks()] == ["t1"]
    extra = {"id": "t2", "question": "q2", "documents": []}
    assert [t.id for t in tasks.load_tasks([extra])] == ["t1", "t2"]
    assert [t.id for t in tasks.select(tasks.load_tasks([extra]),
                                       ids=["t2"])] == ["t2"]


def test_documents_ask_the_app_for_its_own(clean_hooks, tmp_path):
    from aec_agent_core.review_eval import documents
    assert documents._dev_dirs() == []
    with pytest.raises(KeyError):                # the app's, not handed in
        documents.resolve("fixture_report")
    hooks.set_hook(documents.FIXTURE_HOOK,
                   lambda name: b"%PDF-made-up" if name == "report" else None)
    assert documents.resolve("fixture_report")[1] == b"%PDF-made-up"
    (tmp_path / "made_up.pdf").write_bytes(b"%PDF-1")
    hooks.set_hook(documents.DIRS_HOOK, lambda: [tmp_path])
    assert documents.find_file("made_up.pdf") == str(tmp_path / "made_up.pdf")
    hooks.set_hook(documents.FETCH_HOOK, lambda name: None)
    assert documents.find_file("nowhere.pdf") is None


def test_held_records_need_the_apps_record_shapes(clean_hooks):
    from aec_agent_core.deep import held_records
    record = {"investigation_id": "B-1", "layers": [{"top": 0.0}]}
    assert not held_records.is_investigation(record)
    hooks.set_hook(held_records.RECORD_FIELDS_HOOK, lambda: (
        {"investigation_id", "layers", "kind"}, {"kind", "result"},
        {"boring"}, {"moisture"}))
    assert held_records.is_investigation(record)
    hooks.set_hook(held_records.RECORD_FIELDS_HOOK, "no_such_module:fields")
    assert held_records._fields() == (frozenset(),) * 4


def test_cited_documents_ask_the_app_for_its_reference_modules(clean_hooks):
    from aec_agent_core import cited_documents as cd
    assert cd._reference_modules() == frozenset(cd.MODULE_DOCUMENTS)
    hooks.set_hook(cd.REFERENCE_MODULES_HOOK, lambda: {"made_up_refs"})
    assert cd.source_kind("call_agent",
                          {"agent_name": "made_up_refs"}) == cd.REFERENCES


def test_the_core_imports_without_either_app():
    """Every step 11 (c) module imports, and none loads an app module (in
    the core-only venv neither app is installed; here the test checks what
    importing them loaded)."""
    import importlib
    before = set(sys.modules)
    for name in ("activity_log", "cited_documents", "cited_pages",
                 "coverage", "document_tools", "docx_renderer", "dxf_notes",
                 "measure_tool", "office_text", "review_flags", "run_mirror",
                 "runtime_check", "safe_expr", "scale_labels", "xlsx_writer",
                 "deep.assembly", "deep.busy_retry", "deep.calculate_tool",
                 "deep.delivered_log", "deep.held_records", "deep.keep_work",
                 "deep.limits", "deep.result_map", "deep.scratch_guard",
                 "deep.strict_args", "deep.tool_guards", "deep.vision_kit",
                 "deep.warm_imports", "review_eval.runner",
                 "review_eval.checks", "review_eval.documents",
                 "review_eval.tasks"):
        importlib.import_module(f"aec_agent_core.{name}")
    new = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not new & {"funhouse_agent", "webapp", "report_ingest",
                      "geotech_references", "geotech_common"}, new


def test_the_vision_tools_assemble_in_the_core(tmp_path):
    """The tools both pages' agents carry (the vision kit over the document
    tools' binding) build from the core alone, with an engine stand-in."""
    from aec_agent_core.deep.vision_kit import make_vision_tools
    tools = make_vision_tools(engine=None)
    names = {t.name for t in tools}
    for must in ("open_document", "read_document", "analyze_pdf_page",
                 "plot_data", "write_docx"):
        assert must in names, sorted(names)
    json.dumps(sorted(names))
