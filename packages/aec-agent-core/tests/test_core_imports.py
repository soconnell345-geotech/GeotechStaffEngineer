"""The core imports neither app (BACKLOG INF-32, the app split's step 11;
plan ``module_work/APP_SPLIT_PLAN.md`` section 2.7 B), read off the source.

Every import in every file of ``aec_agent_core``, at any depth (inside
functions too), is checked:

- nothing of ``webapp`` (the chat pages and their host files) at all;
- nothing of GeotechStaffEngineer's own modules: the analysis packages,
  the reference library, ``report_ingest``, ``calc_package``, GSE's
  dispatch, adapters, agent, prompts, reference and worked-example readers,
  and the review page's own files;
- of ``funhouse_agent``, only the modules on :data:`NOT_MOVED_YET`: files
  plan section 2.3 puts in the core that merges (b) and (c) have not moved
  yet, each imported INSIDE a function (so every core module imports with
  only the core and planlens installed). The list only shrinks: an entry
  nothing imports any more fails, like the boundary test's lists.

What the core may import at module level is the standard library, itself,
planlens and third-party packages; this test reads the source, so it runs
anywhere, the fresh venv with only the core and planlens included.
"""

from __future__ import annotations

import ast
import pathlib
import sys

CORE = pathlib.Path(__file__).resolve().parents[1] / "aec_agent_core"

#: funhouse_agent modules the core may still reach, inside a function, until
#: the merge named (plan 2.3: each is a core file). When the merge moves it,
#: the import becomes the core's own and the entry goes.
NOT_MOVED_YET = {
    "funhouse_agent.cited_documents": "11 (c), the run record's checks",
    "funhouse_agent.document_tools": "11 (c), the document tools' binding",
    "funhouse_agent.docx_renderer": "11 (c), Word and Excel",
    "funhouse_agent.office_text": "11 (c), Word and Excel",
    "funhouse_agent.dxf_notes": "11 (c) or planlens (plan 2.6, proposed)",
    "funhouse_agent.plot_box.figures": "11 (b), the plotting box",
}

#: GeotechStaffEngineer's packages (plan 2.4), never imported by the core.
GSE_TOP_LEVEL = frozenset({
    "webapp", "bearing_capacity", "settlement", "axial_pile", "sheet_pile",
    "soe", "lateral_pile", "pile_group", "wave_equation", "drilled_shaft",
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


def _imports(path: pathlib.Path):
    """``(module, lineno, at_module_level)`` for every import in ``path``,
    relative imports resolved; for ``from X import y`` both X and X.y."""
    rel = path.relative_to(CORE.parent).with_suffix("")
    module = ".".join(rel.parts)
    package = module if path.name == "__init__.py" else \
        module.rpartition(".")[0]
    if module.endswith(".__init__"):
        package = module[:-len(".__init__")]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {id(n) for n in tree.body}
    for stmt in tree.body:                 # module level, incl. try/if
        if isinstance(stmt, (ast.Try, ast.If)):
            top |= {id(n) for n in ast.walk(stmt)}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno, id(node) in top))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                up = parts[:len(parts) - node.level + 1]
                base = ".".join(up + ([base] if base else []))
            out.append((base, node.lineno, id(node) in top))
            out += [(f"{base}.{a.name}", node.lineno, id(node) in top)
                    for a in node.names if a.name != "*"]
    return out


def _core_files():
    files = sorted(CORE.rglob("*.py"))
    assert files, CORE
    return files


def _first_party(name: str) -> bool:
    return name.split(".")[0] in GSE_TOP_LEVEL | {"funhouse_agent"}


def test_the_core_imports_nothing_of_either_app():
    bad = []
    for path in _core_files():
        rel = path.relative_to(CORE.parent).as_posix()
        for name, line, _ in _imports(path):
            top = name.split(".")[0]
            if top in GSE_TOP_LEVEL:
                bad.append(f"{rel}:{line} -> {name}")
            elif top == "funhouse_agent" and name != "funhouse_agent" and \
                    not any(name == o or name.startswith(o + ".")
                            for o in NOT_MOVED_YET):
                # (the bare package is what ``from funhouse_agent import x``
                # lists beside ``funhouse_agent.x``)
                bad.append(f"{rel}:{line} -> {name}")
    assert not bad, ("The core imports GeotechStaffEngineer or the chat "
                     "pages (plan 2.7 B):\n  " + "\n  ".join(bad))


def test_what_has_not_moved_yet_is_reached_only_inside_functions():
    bad = []
    for path in _core_files():
        rel = path.relative_to(CORE.parent).as_posix()
        for name, line, top in _imports(path):
            if top and _first_party(name):
                bad.append(f"{rel}:{line} -> {name}")
    assert not bad, ("A core module imports app code at module level, so it "
                     "would not import with only the core and planlens "
                     "installed:\n  " + "\n  ".join(bad))


def test_the_not_moved_yet_list_only_shrinks():
    seen = {name for path in _core_files()
            for name, _, _ in _imports(path)}
    gone = sorted(m for m in NOT_MOVED_YET if m not in seen)
    assert not gone, ("Nothing in the core imports these any more; take "
                      "them out of NOT_MOVED_YET:\n  " + "\n  ".join(gone))


def test_the_list_names_no_geotech_module():
    """Only plan 2.3's core files may wait on the list, never a GSE one."""
    gse_files = ("dispatch", "adapters", "agent", "engine", "system_prompt",
                 "reference_docs", "worked_examples", "chart_reading",
                 "reference_vision", "review_agent", "minimal_agent",
                 "review_prompt", "review_build", "deep.agent",
                 "deep.prompt", "deep.tools")
    for m in NOT_MOVED_YET:
        rest = m[len("funhouse_agent."):]
        assert not any(rest == g or rest.startswith(g + ".")
                       for g in gse_files), m


def test_the_core_is_a_package_of_its_own():
    """A sanity check on the walk itself: it sees the moved modules."""
    names = {p.relative_to(CORE).as_posix() for p in _core_files()}
    for must in ("__init__.py", "hooks.py", "vision_tools.py",
                 "engine_config.py", "tinyapps_engine.py",
                 "deep/vision_engine.py", "deep/databricks_bridge.py"):
        assert must in names, must
    # the standard library check the module-level rule leans on
    assert "json" in getattr(sys, "stdlib_module_names", {"json"})
