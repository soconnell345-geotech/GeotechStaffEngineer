"""The app side of step 11 (c) of the split (BACKLOG INF-32;
``module_work/inf32/CHECK_STEP11C.md``): guards and records, Word and Excel,
the checks, the review suite's framework, the document tools' binding and
the shared assembly are the shared core's. Every old import path IS the
core's module, no module is loaded twice whatever the import order, and GSE
fills the core's new hooks with the code they replaced.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

import funhouse_agent
import webapp
from aec_agent_core import hooks

TOP = ("cited_documents", "cited_pages", "coverage", "document_tools",
       "docx_renderer", "dxf_notes", "measure_tool", "office_text",
       "review_flags", "run_mirror", "runtime_check", "safe_expr",
       "scale_labels", "xlsx_writer")
DEEP = ("assembly", "busy_retry", "calculate_tool", "delivered_log",
        "held_records", "keep_work", "limits", "result_map", "scratch_guard",
        "strict_args", "tool_guards", "vision_kit", "warm_imports")
SUITE = ("checks", "documents", "runner")

STEP_11C = {**{f"funhouse_agent.{m}": f"aec_agent_core.{m}" for m in TOP},
            **{f"funhouse_agent.deep.{m}": f"aec_agent_core.deep.{m}"
               for m in DEEP},
            **{f"funhouse_agent.review_eval.{m}":
               f"aec_agent_core.review_eval.{m}" for m in SUITE},
            "webapp.activity_log": "aec_agent_core.activity_log"}


def test_the_tables_hold_step_11c():
    tables = {**funhouse_agent.CORE_ALIASES, **webapp.CORE_ALIASES}
    for old, new in STEP_11C.items():
        assert tables[old] == new, old
    assert len(STEP_11C) == 31


@pytest.mark.parametrize("old,new", sorted(STEP_11C.items()))
def test_every_old_name_is_the_core_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)
    parent, _, leaf = old.rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is \
        importlib.import_module(new)


def test_the_old_from_imports_give_the_core_objects():
    from funhouse_agent.deep.assembly import compile_agent
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
    from funhouse_agent.review_eval import Task, score_review_suite
    from funhouse_agent.review_eval.tasks import Task as GseTask
    from webapp.activity_log import ActivityLogger
    import aec_agent_core.activity_log as core_log
    import aec_agent_core.deep.assembly as core_assembly
    import aec_agent_core.deep.limits as core_limits
    import aec_agent_core.review_eval.runner as core_runner
    import aec_agent_core.review_eval.tasks as core_tasks
    assert compile_agent is core_assembly.compile_agent
    assert ModelCallBudgetMiddleware is core_limits.ModelCallBudgetMiddleware
    assert score_review_suite is core_runner.score_review_suite
    assert Task is GseTask is core_tasks.Task
    assert ActivityLogger is core_log.ActivityLogger


def test_a_patch_by_the_old_name_reaches_the_core(monkeypatch):
    monkeypatch.setattr("funhouse_agent.deep.limits.DEFAULT_MAX_RESULT_CHARS",
                        123)
    from aec_agent_core.deep import limits
    assert limits.DEFAULT_MAX_RESULT_CHARS == 123


#: Orders a caller may import in; each must leave exactly one copy.
_ORDERS = [
    ["aec_agent_core.deep.assembly", "funhouse_agent.deep.assembly"],
    ["funhouse_agent.deep.assembly", "aec_agent_core.deep.assembly"],
    ["funhouse_agent.review_eval", "aec_agent_core.review_eval.runner"],
    ["aec_agent_core.review_eval.runner", "funhouse_agent.review_eval"],
    ["webapp.activity_log", "funhouse_agent.deep.agent",
     "aec_agent_core.activity_log"],
    ["aec_agent_core.document_tools", "funhouse_agent.markup_check",
     "funhouse_agent.document_tools"],
    ["funhouse_agent.review_build", "webapp.core"],
    ["calc_package.docx_renderer", "funhouse_agent.docx_renderer",
     "aec_agent_core.docx_renderer"],
]

_PROBE = r"""
import importlib, json, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
seen = {}
twice = []
for key, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if not f or "aec_agent_core" not in f:
        continue
    if f in seen and seen[f] is not mod:
        twice.append(f)
    seen.setdefault(f, mod)
# every core module, under each name it is known by, is one object
names = {}
for key, mod in list(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if f and "aec_agent_core" in f:
        names.setdefault(f, set()).add(id(mod))
print(json.dumps({"twice": twice,
                  "split": [f for f, ids in names.items() if len(ids) > 1],
                  "core_modules": len(names)}))
"""


@pytest.mark.parametrize("order", _ORDERS, ids=lambda o: " then ".join(o))
def test_no_module_is_loaded_twice_in_any_import_order(order):
    out = subprocess.run([sys.executable, "-c", _PROBE, *order],
                         capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ,
                              "GEOTECH_NO_DISCLAIMER": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["twice"] == [] and got["split"] == [], got
    assert got["core_modules"] > 0


def test_the_package_doors_fill_the_new_hooks():
    import funhouse_agent.review_eval  # noqa: F401 - fills the suite's hooks
    filled = hooks.hooks()
    assert filled["held_records.record_fields"] == \
        "funhouse_agent.core_hooks:record_fields"
    assert filled["cited_documents.reference_modules"] == \
        "funhouse_agent.core_hooks:reference_modules"
    assert filled["review_eval.open_tasks"] == \
        "funhouse_agent.review_eval.tasks:open_tasks"
    assert filled["review_eval.page_host"] == "webapp.suite_page:host"
    for name in ("document_dirs", "fetch_document", "fixture"):
        assert filled[f"review_eval.{name}"] == \
            f"funhouse_agent.review_eval.host:{name}"


def test_calc_packages_old_name_for_the_word_renderer_is_the_core_module():
    """calc_package.docx_renderer (INF-30 step 5's alias) now points at the
    core's module directly, and importing it loads nothing of
    funhouse_agent (a fresh interpreter)."""
    code = ("import sys, calc_package.docx_renderer as a\n"
            "import aec_agent_core.docx_renderer as b\n"
            "print(a is b, 'funhouse_agent' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.stdout.split() == ["True", "False"], out.stderr[-2000:]


def test_held_records_reads_report_ingests_shapes_through_the_hook():
    pytest.importorskip("report_ingest.model")
    from report_ingest.model import Investigation, LabTest
    from aec_agent_core.deep import held_records
    inv, lab, inv_kinds, lab_kinds = held_records._fields()
    assert inv == frozenset(Investigation.model_fields)
    assert lab == frozenset(LabTest.model_fields)
    assert "boring" in inv_kinds
    record = {"investigation_id": "B-1",
              "layers": [{"top": 0.0, "bottom": 1.5}]}
    assert held_records.is_investigation(record)


def test_held_records_holds_nothing_without_the_hook(monkeypatch):
    from aec_agent_core.deep import held_records
    monkeypatch.setattr(hooks, "_HOOKS", {})
    monkeypatch.setattr(hooks, "_RESOLVED", {})
    assert held_records._fields() == (frozenset(),) * 4
    assert not held_records.is_investigation(
        {"investigation_id": "B-1", "layers": [{"top": 0.0}]})


def test_cited_documents_reads_the_dispatchs_reference_modules():
    from aec_agent_core import cited_documents as cd
    from funhouse_agent.dispatch import REFERENCE_MODULES
    assert cd._reference_modules() == frozenset(REFERENCE_MODULES)
    some = sorted(REFERENCE_MODULES)[0]
    assert cd.source_kind("call_agent", {"agent_name": some}) == cd.REFERENCES


def test_cited_documents_falls_back_without_the_hook(monkeypatch):
    from aec_agent_core import cited_documents as cd
    monkeypatch.setattr(hooks, "_HOOKS", {})
    monkeypatch.setattr(hooks, "_RESOLVED", {})
    assert cd._reference_modules() == frozenset(cd.MODULE_DOCUMENTS)


def test_gse_hands_the_suite_its_tasks_and_its_page():
    from aec_agent_core.review_eval import runner, tasks as core_tasks
    from funhouse_agent.review_eval.tasks import OPEN_TASKS, load_tasks
    from webapp import suite_page
    assert core_tasks.open_tasks() == list(OPEN_TASKS)
    assert [t.id for t in load_tasks()] == [t.id for t in OPEN_TASKS]
    assert [t.id for t in core_tasks.load_tasks()] == \
        [t.id for t in OPEN_TASKS]
    host = runner.page_host()
    assert host is suite_page
    for attr in runner.PAGE_HOST_ATTRIBUTES:
        assert hasattr(host, attr), attr
    assert runner._app() == ("geotech-staff-engineer", "funhouse_agent",
                             "GEOTECH_APP_COMMIT")


def test_gse_hands_the_suite_its_report_fixture_and_folders():
    from aec_agent_core.review_eval import documents
    name, data = documents.resolve("fixture_report")
    assert name.endswith(".pdf") and data[:4] == b"%PDF"
    dirs = documents._dev_dirs()
    assert any(d.name == "sample_pdfs" for d in dirs)
