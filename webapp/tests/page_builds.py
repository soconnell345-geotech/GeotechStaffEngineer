"""Offline builds of both pages' agents, for the app split's guards.

BACKLOG INF-30 (``module_work/APP_SPLIT_PLAN.md`` §4, step 1). Two tests
use this module, each through a child process so the test order and what
the parent process has already imported cannot change the answer:

* ``test_page_fingerprints.py`` pins what each build hands the agent
  framework (tools, prompts, helpers, middleware, budgets), so a step of
  the split that should change nothing is shown to change nothing;
* ``test_review_boundary.py`` builds the Document Review agent, runs one
  turn on a fake model, and records every import that crosses from the
  shared and review code into GeotechStaffEngineer's own modules (and the
  reverse); or, blocking those modules, sees whether the page still works.

Each build goes through ``webapp.core.build_agent`` with the page's own
keyword arguments, as ``webapp/app.py`` puts them together, on a fake chat
model, with every ``GEOTECH_*`` setting unset (the page as shipped) except
the one that picks the review page's build. Nothing calls a model or the
network.

Re-record the fingerprints after a DELIBERATE change, in its own commit
with the reason (plan §4): ``python -m webapp.tests.page_builds record``
(from the repository root). When to, and when not to, is in
``test_page_fingerprints.py``.

This module imports only the standard library at load, so the boundary
child can install its import hooks before anything else is imported.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import tempfile
from typing import Any, Dict, Iterator, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GOLDEN = os.path.join(HERE, "page_fingerprints.json")

#: The builds: (page, the review page's build switch).
BUILDS = {
    "review_default": ("review", None),
    "review_lean": ("review", "lean"),
    "review_minimal": ("review", "minimal"),
    "geotech_default": ("geotech", None),
}
REVIEW_BUILDS = ("review_default", "review_lean", "review_minimal")

#: Settings a build can read besides ``GEOTECH_*``; unset for every build,
#: so a developer's shell cannot change the result.
_OTHER_ENVS = ("SHAREPOINT_SITE_URL", "SHAREPOINT_ROOT", "GRAPH_TENANT_ID",
               "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET", "KV_NAME",
               "DEV_IDENTITY")


@contextlib.contextmanager
def page_environment(switch=None, **settings: str) -> Iterator[None]:
    """Every ``GEOTECH_*`` setting unset, the vision probe off,
    ``GEOTECH_REVIEW_AGENT`` set to ``switch`` (None: the default build) and
    ``settings`` set. Restored afterwards."""
    names = {n for n in os.environ if n.startswith("GEOTECH_")}
    names |= set(_OTHER_ENVS) | set(settings) | {"GEOTECH_VISION_PROBE",
                                                 "GEOTECH_REVIEW_AGENT"}
    saved = {n: os.environ.get(n) for n in names}
    try:
        for n in names:
            os.environ.pop(n, None)
        os.environ["GEOTECH_VISION_PROBE"] = "0"
        if switch:
            os.environ["GEOTECH_REVIEW_AGENT"] = switch
        os.environ.update(settings)
        yield
    finally:
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v


def fake_model(first_step=None):
    """A chat model that answers its first call with ``first_step(names)``
    (``names``: the tools bound to the agent making the call; None: "done")
    and every later call with "done"."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _Fake(FakeMessagesListChatModel):
        steps: list = []
        bound: list = []

        def bind_tools(self, tools, **kw):
            names = [t.get("name") if isinstance(t, dict)
                     else getattr(t, "name", None) for t in tools]
            # A shallow copy: the copies share one ``steps`` list.
            return self.model_copy(update={"bound": names})

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            step = self.steps.pop(0) if self.steps else None
            msg = step(list(self.bound)) if step else None
            return ChatResult(generations=[ChatGeneration(
                message=msg or AIMessage(content="done"))])

    return _Fake(responses=[AIMessage(content="x")],
                 steps=[first_step] if first_step else [])


def page_kwargs(page: str) -> dict:
    """The page's build arguments, as ``webapp/app.py`` puts them together
    for a new conversation (default Behavior, then the page's own)."""
    from webapp import core, profiles
    profile = profiles.DOCUMENT_REVIEW if page == "review" \
        else profiles.GEOTECH
    kw = core.behavior_build_kwargs(None)
    kw.update(profile.build_kwargs())
    kw.setdefault("markup_author", "Fingerprint via GeotechStaffEngineer")
    return kw


def _new_files_dir() -> str:
    conv = tempfile.mkdtemp(prefix="pagebuild_")
    files_dir = os.path.join(conv, "files")
    os.makedirs(files_dir, exist_ok=True)
    return files_dir


def build(name: str, model=None, files_dir=None, capture: bool = False,
          settings: Optional[Dict[str, str]] = None):
    """Build ``BUILDS[name]`` (``settings``: environment settings to build
    under); return ``(agent, calls, files_dir)``.

    With ``capture``, ``calls`` lists what our code handed the agent
    framework: one entry per ``create_deep_agent`` / ``create_agent`` call
    of ours (a call made inside another is the framework's own, and
    deepagents' rebuild that only adds its todo middleware is left out).
    The lean and minimal builds make their helper on first use, so their
    ``task`` tool is called once, on the fake model, to record it.
    """
    page, switch = BUILDS[name]
    files_dir = files_dir or _new_files_dir()
    calls: List[dict] = []
    patches: list = []
    helper = [None]
    if capture:
        import langchain.agents as lc_agents
        from funhouse_agent.deep import agent as deep_agent
        depth = [0]

        def _wrap(kind, real):
            def _call(*args, **kwargs):
                first_deep = kind != "create_deep_agent" or not any(
                    c["kind"] == kind for c in calls)
                if depth[0] == 0 and first_deep:
                    calls.append({"kind": kind, "helper": helper[0],
                                  "args": args, "kwargs": kwargs})
                depth[0] += 1
                try:
                    return real(*args, **kwargs)
                finally:
                    depth[0] -= 1
            return _call

        patches = [
            (deep_agent, "create_deep_agent",
             _wrap("create_deep_agent", deep_agent.create_deep_agent)),
            (lc_agents, "create_agent",
             _wrap("create_agent", lc_agents.create_agent)),
        ]
    saved = [(obj, attr, getattr(obj, attr)) for obj, attr, _ in patches]
    with page_environment(switch, **(settings or {})):
        for obj, attr, fn in patches:
            setattr(obj, attr, fn)
        try:
            from webapp import core
            agent = core.build_agent(model or fake_model(), {}, files_dir,
                                     [], **page_kwargs(page))
            if capture:
                for call in [c for c in calls if c["kind"] == "create_agent"]:
                    for tool in _call_tools(call):
                        if getattr(tool, "name", None) == "task":
                            helper[0] = "task"
                            tool.invoke({"description": "Say done."})
                            helper[0] = None
        finally:
            for obj, attr, real in saved:
                setattr(obj, attr, real)
    return agent, calls, files_dir


def _call_tools(call: dict) -> list:
    tools = call["kwargs"].get("tools")
    if tools is None and len(call["args"]) > 1:
        tools = call["args"][1]
    return list(tools or [])


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------

def _h(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


class _Normaliser:
    """Text with this build's own folder taken out, so two runs (or two
    machines) of the same code give the same hashes."""

    def __init__(self, files_dir: str):
        conv = os.path.dirname(os.path.abspath(files_dir))
        roots = {files_dir, os.path.abspath(files_dir), conv}
        self._roots = sorted(
            {r for p in roots for r in (p, p.replace("\\", "/"),
                                        p.replace("\\", "\\\\"))},
            key=len, reverse=True)

    def __call__(self, text: Any) -> str:
        s = "" if text is None else str(text)
        for root in self._roots:
            s = s.replace(root, "<DIR>")
        return s


def _schema_shape(schema: Any) -> Any:
    """A tool's argument schema without what pydantic and langchain versions
    word differently (titles, ``additionalProperties``, defaults shown
    twice), and with descriptions cleaned as docstrings are (Python 3.13
    strips their indentation, 3.12 does not)."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key in ("title", "additionalProperties", "$defs",
                       "definitions", "default", "examples"):
                continue
            if key == "description" and isinstance(value, str):
                out[key] = inspect.cleandoc(value)
            else:
                out[key] = _schema_shape(value)
        return out
    if isinstance(schema, list):
        return [_schema_shape(v) for v in schema]
    return schema


def _tool_args(tool) -> Any:
    try:
        schema = tool.tool_call_schema
        if isinstance(schema, dict):
            return schema
        return schema.model_json_schema()
    except Exception:                                    # noqa: BLE001
        return getattr(tool, "args", None)


def _tool_name(tool) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name"))
    return str(getattr(tool, "name", type(tool).__name__))


#: Settings on our middleware that change what the model is sent or how
#: many calls it gets.
_MW_NUMBERS = ("run_limit", "thread_limit", "warn_remaining")
_MW_TEXTS = ("final_turn_nudge", "exhausted_message", "low_budget_note")


def _middleware(items, norm) -> List[dict]:
    """Each middleware's class, in order; for ours, its budget, the hashes
    of its texts and the tools it hides."""
    out = []
    for m in items or []:
        cls = type(m)
        entry: Dict[str, Any] = {"class": cls.__name__}
        if cls.__module__.startswith("funhouse_agent"):
            for a in _MW_NUMBERS:
                v = getattr(m, a, None)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    entry[a] = v
            for a in _MW_TEXTS:
                v = getattr(m, a, None)
                if isinstance(v, str):
                    entry[a] = _h(norm(v))
            hidden = getattr(m, "hidden", None)
            if isinstance(hidden, (set, frozenset, list, tuple)):
                entry["hidden"] = sorted(str(n) for n in hidden)
        out.append(entry)
    return out


def _tools(tools, norm, planlens: frozenset):
    """(app tools, planlens tools): name -> hashes of its description and
    argument schema."""
    app, served = {}, {}
    for t in tools or []:
        name = _tool_name(t)
        if isinstance(t, dict):
            desc = t.get("description")
            args = t.get("parameters") or t.get("input_schema")
        else:
            desc = getattr(t, "description", "")
            args = _tool_args(t)
        entry = {"description": _h(norm(inspect.cleandoc(str(desc or "")))),
                 "args": _h(norm(json.dumps(_schema_shape(args),
                                            sort_keys=True, default=str)))}
        (served if name in planlens else app)[name] = entry
    return app, served


def _call_record(call: dict, norm, planlens: frozenset) -> dict:
    kw = dict(call["kwargs"])
    tools = _call_tools(call)
    kw.pop("tools", None)
    kw.pop("model", None)
    prompt = kw.pop("system_prompt", None)
    app, served = _tools(tools, norm, planlens)
    rec: Dict[str, Any] = {
        "kind": call["kind"],
        "system_prompt": _h(norm(prompt)),
        "system_prompt_chars": len(norm(prompt)),
        "tools": app,
        "planlens_tools": served,
        "middleware": _middleware(kw.pop("middleware", None), norm),
    }
    if call.get("helper"):
        rec["helper_made_by"] = call["helper"]
    subagents = kw.pop("subagents", None)
    if subagents is not None:
        subs = []
        for s in subagents:
            if not isinstance(s, dict):
                subs.append({"name": type(s).__name__})
                continue
            if "runnable" in s:
                subs.append({"name": s.get("name"), "compiled": True,
                             "description": _h(norm(s.get("description")))})
                continue
            s_app, s_served = _tools(s.get("tools"), norm, planlens)
            subs.append({
                "name": s.get("name"),
                "description": _h(norm(s.get("description"))),
                "system_prompt": _h(norm(s.get("system_prompt"))),
                "tools": s_app,
                "planlens_tools": s_served,
                "middleware": _middleware(s.get("middleware"), norm),
            })
        rec["subagents"] = subs
    rec["other_arguments"] = sorted(k for k in kw
                                    if k not in ("checkpointer", "store"))
    return rec


#: Attributes the builds set on the compiled agent that bear on a turn.
_AGENT_ATTRS = ("geotech_min_recursion_limit",
                "geotech_extraction_recursion_limit",
                "geotech_review_agent", "geotech_judgment_consult")


def planlens_tool_names() -> frozenset:
    """The document tools planlens serves (``funhouse_agent.document_tools``).
    Which of them exist, and their descriptions and schemas, follow the
    installed planlens (CI installs it from PyPI; a developer often has
    planlens main), so they are recorded apart from the app's own tools."""
    from funhouse_agent import document_tools
    return frozenset(document_tools.DOCUMENT_TOOL_NAMES
                     + document_tools.OPTIONAL_DOCUMENT_TOOL_NAMES)


def planlens_version() -> str:
    """The installed planlens' version, to label a recorded ``planlens``
    part (a main checkout carries the last release's number; an editable
    install's metadata can be older still, so the package's own first)."""
    try:
        import planlens
        where = os.path.abspath(planlens.__file__).replace("\\", "/")
        checkout = not any(p in where for p in ("/site-packages/",
                                                "/dist-packages/"))
        return str(planlens.__version__) + (" (source checkout)"
                                            if checkout else "")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        from importlib.metadata import version
        return version("planlens")
    except Exception:                                    # noqa: BLE001
        return "unknown"


def planlens_surface() -> str:
    """A hash of the planlens tool specs this app reads, so a recorded
    ``planlens`` part is compared only against the same planlens."""
    from funhouse_agent import document_tools
    specs = []
    for name in sorted(planlens_tool_names()):
        try:
            spec = document_tools._spec(name)
        except Exception:                                # noqa: BLE001
            spec = None
        specs.append([name, spec])
    return _h(json.dumps(specs, sort_keys=True, default=str))


def fingerprint(name: str) -> dict:
    """What build ``name`` hands the agent framework, as names, numbers and
    hashes. ``app``: the app's own tools, prompts, helpers, middleware and
    budgets, the same on every machine (checked on Windows with Python
    3.14, deepagents 0.6 and planlens main, and on Linux with Pythons 3.12
    and 3.13, deepagents 0.7 and planlens 0.13 from PyPI), given the
    reference library CI installs (the submodule: ``plot_data`` offers
    ``reference_chart`` only where it has ``chart_lines``). ``planlens``:
    the planlens-served document tools. What deepagents itself adds (its
    own tools and prompt text) is not ours and is left out."""
    agent, calls, files_dir = build(name, capture=True)
    norm = _Normaliser(files_dir)
    planlens = planlens_tool_names()
    records = [_call_record(c, norm, planlens) for c in calls]
    served = []
    for i, r in enumerate(records):
        served.append({"call": i, "tools": r.pop("planlens_tools"),
                       "subagents": {s["name"]: s.pop("planlens_tools")
                                     for s in r.get("subagents", [])
                                     if "planlens_tools" in s}})
    attrs = {}
    for a in _AGENT_ATTRS:
        v = getattr(agent, a, None)
        if isinstance(v, (int, float, str, bool)):
            attrs[a] = v
    return {"app": {"calls": records, "agent": attrs}, "planlens": served}


def all_fingerprints() -> dict:
    return {"planlens_surface": planlens_surface(),
            "planlens_version": planlens_version(),
            "builds": {name: fingerprint(name) for name in BUILDS}}


def golden_from(current: dict, old: Optional[dict] = None) -> dict:
    """The recorded file for this tree's fingerprints: the app part as
    built here, and the planlens part filed under the planlens it was built
    with, beside those ``old`` recorded under other planlens versions (CI
    installs planlens from PyPI; a developer often has planlens main)."""
    planlens = dict((old or {}).get("planlens") or {})
    planlens[current["planlens_surface"]] = {
        "planlens": current.get("planlens_version", "unknown"),
        "builds": {n: b["planlens"] for n, b in current["builds"].items()},
    }
    return {
        "_about": ("What each page's agent is handed with default switches "
                   "(BACKLOG INF-30). Re-record only for a deliberate "
                   "change, in its own commit with the reason; a split "
                   "step never re-records (plan step 12 is the one "
                   "exception): python -m webapp.tests.page_builds record "
                   "(see test_page_fingerprints.py)"),
        "builds": {n: {"app": b["app"]}
                   for n, b in current["builds"].items()},
        "planlens": planlens,
    }


def differences(old: Any, new: Any, path: str = "") -> List[str]:
    """Where two fingerprints differ, as readable paths."""
    out: List[str] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            p = f"{path}/{k}"
            if k not in old:
                out.append(f"{p}: added")
            elif k not in new:
                out.append(f"{p}: gone")
            else:
                out += differences(old[k], new[k], p)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            def _names(xs):
                return [x.get("class") or x.get("name") or x.get("kind")
                        if isinstance(x, dict) else x for x in xs]
            out.append(f"{path}: {_names(old)} -> {_names(new)}")
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                out += differences(a, b, f"{path}[{i}]")
    elif old != new:
        out.append(f"{path}: {old!r} -> {new!r}")
    return out


# ---------------------------------------------------------------------------
# The boundary: what the review page imports from GeotechStaffEngineer
# ---------------------------------------------------------------------------

#: The geotech packages, the reference library, and GSE's own modules of
#: ``funhouse_agent`` and ``webapp`` (plan §2.4): what Document Review must
#: not import once the split is done. Named by where each module is TODAY;
#: a piece that moves to a neutral package (the chart stack, step 6) stops
#: matching by itself.
GEOTECH_TOP_LEVEL = frozenset({
    "bearing_capacity", "settlement", "axial_pile", "sheet_pile", "soe",
    "lateral_pile", "pile_group", "wave_equation", "drilled_shaft",
    "seismic_geotech", "retaining_walls", "pavement_design",
    "ground_improvement", "slope_stability", "reliability", "downdrag",
    "geotech_common", "calc_package", "profile_figure", "opensees_agent",
    "pystrata_agent", "seismic_signals_agent", "liquepy_agent",
    "gstools_agent", "salib_agent", "pystra_agent", "section_props_agent",
    "concrete_props_agent", "pynite_agent", "subsurface_characterization",
    "dxf_import", "dxf_export", "fem2d", "geo_project", "report_ingest",
    "geotech_references", "validation_examples", "foundry",
    "foundry_test_harness",
})

GSE_MODULES = frozenset({
    "funhouse_agent._disclaimer", "funhouse_agent.agent",
    "funhouse_agent.engine", "funhouse_agent.native_tools",
    "funhouse_agent.react_support", "funhouse_agent.notebook",
    "funhouse_agent.run_local", "funhouse_agent.run_scenarios",
    "funhouse_agent.smoke_test_native", "funhouse_agent.dispatch",
    "funhouse_agent.system_prompt", "funhouse_agent.reviewer",
    "funhouse_agent.reviewers", "funhouse_agent.review_checklists",
    "funhouse_agent.worked_examples", "funhouse_agent.reference_docs",
    "funhouse_agent.chart_reading", "funhouse_agent.judgment_check",
    "funhouse_agent.report_checklist", "funhouse_agent.adapters",
    "funhouse_agent.deep.agent", "funhouse_agent.deep.prompt",
    "funhouse_agent.deep.tools", "funhouse_agent.deep.fit_results",
    "funhouse_agent.deep.setup_agent", "funhouse_agent.deep.setup_tools",
    "funhouse_agent.deep.unknown_tool", "funhouse_agent.deep.eval_harness",
    "funhouse_agent.deep.selfcheck", "funhouse_agent.deep.rc_wheel_check",
    "funhouse_agent.deep.run_live", "funhouse_agent.deep.notebook",
    "webapp.foundry_entry", "webapp.reference_fetch",
})

#: The review page's own files (plan §2.2), which GSE's code must not
#: import once the split is done.
REVIEW_MODULES = frozenset({
    "funhouse_agent.deep.review_agent", "funhouse_agent.deep.minimal_agent",
    "funhouse_agent.deep.sweep", "funhouse_agent.deep.findings_tools",
    "funhouse_agent.review_findings", "funhouse_agent.deep.geometry_tools",
    "funhouse_agent.deep.digest_tools", "funhouse_agent.review_digest",
    "webapp.tinyapps_entry",
})


def _in(module: str, names: frozenset) -> bool:
    return any(module == m or module.startswith(m + ".") for m in names)


def is_geotech(module: str) -> bool:
    """Whether ``module`` stays with GeotechStaffEngineer after the split."""
    return (module.split(".")[0] in GEOTECH_TOP_LEVEL
            or _in(module, GSE_MODULES))


def is_review(module: str) -> bool:
    """Whether ``module`` goes to the Document Review package."""
    return _in(module, REVIEW_MODULES)


def _is_harness(module: Optional[str]) -> bool:
    return not module or module == "__main__" or \
        module.startswith("webapp.tests")


def _resolve(name: str, globals_: Optional[dict], level: int) -> str:
    if level == 0:
        return name
    g = globals_ or {}
    package = g.get("__package__")
    if not package:
        package = g.get("__name__", "")
        if "__path__" not in g:
            package = package.rpartition(".")[0]
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{name}" if name else base


def install_import_recorder(edges: set) -> None:
    """Record ``(importer, imported)`` for every import statement and every
    ``importlib.import_module`` call from here on, including ones of modules
    already loaded (a statement that finds its module loaded is still an
    import of it)."""
    import builtins
    import importlib

    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        try:
            importer = (globals or {}).get("__name__")
            target = _resolve(name, globals, level)
            edges.add((importer, target))
            for item in fromlist or ():
                sub = f"{target}.{item}"
                if item != "*" and sub in sys.modules:
                    edges.add((importer, sub))
        except Exception:                                # noqa: BLE001
            pass
        return module

    def _importer(frame):
        # A package's lazy ``__getattr__`` imports for whoever asked for
        # the name: the first frame above it that is not the import
        # machinery or this recorder.
        if frame.f_code.co_name != "__getattr__":
            return frame.f_globals.get("__name__")
        f = frame.f_back
        while f is not None and (
                f.f_globals is globals()
                or str(f.f_globals.get("__name__", "")).startswith(
                    ("importlib", "_frozen_importlib"))):
            f = f.f_back
        return f.f_globals.get("__name__") if f is not None else None

    def _import_module(name, package=None):
        module = real_import_module(name, package)
        try:
            edges.add((_importer(sys._getframe(1)), module.__name__))
        except Exception:                                # noqa: BLE001
            pass
        return module

    builtins.__import__ = _import
    importlib.import_module = _import_module


class _Blocker:
    """A finder that refuses every GeotechStaffEngineer module, as an
    install without the geotech packages would."""

    def __init__(self):
        self.refused: List[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if is_geotech(fullname):
            if fullname not in self.refused:
                self.refused.append(fullname)
            raise ModuleNotFoundError(
                f"blocked for the boundary test: {fullname}", name=fullname)
        return None

    def install(self) -> None:
        """First on ``sys.meta_path``; and ``importlib.util.find_spec``
        answers None for a refused module whose parent is not refused, as
        it does for a module that is not installed (the finder alone would
        make it raise)."""
        import importlib.util
        real_find_spec = importlib.util.find_spec

        def _find_spec(name, package=None):
            full = importlib.util.resolve_name(name, package) \
                if name.startswith(".") else name
            if is_geotech(full) and not is_geotech(full.rpartition(".")[0]):
                if full not in self.refused:
                    self.refused.append(full)
                return None
            return real_find_spec(name, package)

        sys.meta_path.insert(0, self)
        importlib.util.find_spec = _find_spec


def _turn_calls(build_name: str):
    """The turn's first model answer: one call of each tool on the page
    whose code reaches into geotech modules when it runs (plan §2.7), and
    one helper."""
    helper = {"review_default": "general-purpose",
              "review_lean": "page_reader",
              "review_minimal": "page_looker"}.get(build_name)
    calls = {
        "plot_data": {"series": [{"x": [0, 1, 2], "y": [0, 1, 4],
                                  "label": "a"}],
                      "title": "Boundary check"},
        "calculate": {"expression": "2*3"},
        "write_docx": {"path": "boundary_note.docx",
                       "markdown": "# Note\n\nOne line."},
    }
    if helper:
        calls["task"] = {"description": "Say done.",
                         "subagent_type": helper}

    def step(bound):
        from langchain_core.messages import AIMessage
        wanted = [{"name": n, "args": a, "id": f"call_{n}"}
                  for n, a in calls.items() if n in bound]
        return AIMessage(content="", tool_calls=wanted) if wanted else None
    return step


def run_page(build_name: str) -> dict:
    """Build ``build_name`` and run one turn: the model calls the tools of
    :func:`_turn_calls`, then answers. Files land in a temporary
    conversation folder, as the app's working folder setting does."""
    model = fake_model(_turn_calls(build_name))
    files_dir = _new_files_dir()
    settings = {"GEOTECH_DEFAULT_OUTPUT_DIR": files_dir}
    with page_environment(BUILDS[build_name][1], **settings):
        agent, _, _ = build(build_name, model=model, files_dir=files_dir,
                            settings=settings)
        limit = max(100, int(getattr(agent, "geotech_min_recursion_limit",
                                     0) or 0))
        out = agent.invoke(
            {"messages": [{"role": "user",
                           "content": "Plot, work out, write."}]},
            config={"recursion_limit": limit,
                    "configurable": {"thread_id": "boundary"}})
    results = {}
    for m in out.get("messages", []):
        if type(m).__name__ == "ToolMessage":
            results[m.name] = str(m.content)[:200]
    return {"tool_results": results}


def boundary(mode: str, builds: List[str]) -> dict:
    """``record``: every import edge seen while building and running
    ``builds``; ``block``: whether they build and run with every geotech
    module refused. Call in a fresh process, before anything else is
    imported."""
    edges: set = set()
    blocker = None
    if mode == "record":
        install_import_recorder(edges)
    else:
        blocker = _Blocker()
        blocker.install()
    report: Dict[str, Any] = {"mode": mode, "builds": {}}
    for name in builds:
        try:
            report["builds"][name] = {"ok": True, **run_page(name)}
        except BaseException as exc:                     # noqa: BLE001
            report["builds"][name] = {
                "ok": False, "error": f"{type(exc).__name__}: {exc}"[:600]}
    if blocker is not None:
        report["refused"] = blocker.refused
    report["into_gse"] = sorted(
        f"{a} -> {b}" for a, b in edges
        if a and b and is_geotech(b) and not is_geotech(a)
        and not _is_harness(a))
    report["gse_into_review"] = sorted(
        f"{a} -> {b}" for a, b in edges
        if a and b and is_review(b) and is_geotech(a))
    return report


def run_child(args: List[str], timeout: int = 900) -> dict:
    """Run this module in a fresh interpreter (this repo first on the path)
    and return the JSON it writes."""
    import subprocess
    fd, out = tempfile.mkstemp(suffix=".json", prefix="pagebuild_")
    os.close(fd)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO, env.get("PYTHONPATH")) if p)
    env.setdefault("MPLBACKEND", "Agg")
    env["GEOTECH_NO_DISCLAIMER"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "webapp.tests.page_builds", *args, out],
            cwd=REPO, env=env, capture_output=True, text=True,
            timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"page_builds {' '.join(args)} exited {proc.returncode}:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        with contextlib.suppress(OSError):
            os.remove(out)


def _write(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")


def main(argv: List[str]) -> int:
    cmd = argv[0] if argv else ""
    # As run_child runs the test's builds, so a record from a shell matches.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["GEOTECH_NO_DISCLAIMER"] = "1"
    if cmd == "record":
        old = None
        if os.path.exists(GOLDEN):
            with open(GOLDEN, encoding="utf-8") as fh:
                old = json.load(fh)
        current = all_fingerprints()
        data = golden_from(current, old)
        _write(GOLDEN, data)
        print(f"wrote {GOLDEN} (planlens {current['planlens_version']}, "
              f"surface {current['planlens_surface']})")
        others = sorted(f"{v['planlens']} ({s})"
                        for s, v in data["planlens"].items()
                        if s != current["planlens_surface"])
        if others:
            print("kept the planlens tools recorded under "
                  + ", ".join(others) + ". If this change touched the "
                  "document tools, record under those too (a venv with "
                  "that planlens), or delete a stale entry.")
        return 0
    if cmd == "fingerprints":
        _write(argv[1], all_fingerprints())
        return 0
    if cmd == "boundary":
        mode, builds, out = argv[1], argv[2].split(","), argv[3]
        _write(out, boundary(mode, builds))
        return 0
    print("usage: python -m webapp.tests.page_builds record")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
