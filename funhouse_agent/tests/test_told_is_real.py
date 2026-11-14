"""What the agent and the builders are told about a calculation is real
(BACKLOG INF-77, the module audit of 2026-11-12 §6): the standing checks
that need no run. The ones that run a probe live where the probe already
runs, so no probe runs twice:

- T1 (a) every ``returns`` key is in the probe's result, and T2 every
  module warning a call raises is in its result: in
  ``test_defaults_used.py``'s ``test_the_defaults_a_run_lists_are_the_
  values_it_used``.
- T1 (b) every parameter changes the result or is refused: in
  ``test_defaults_used.py``'s ``test_every_default_a_run_lists_changes_
  its_result`` (the defaults a run lists and the parameters a probe gives).
- T1 (c) every ``allowed_values`` violation is refused:
  ``test_allowed_values.py``'s ``test_every_allowed_values_parameter_
  refuses_a_made_up_value``.

Here: T1 (d) every calc_package module has a ``*_package`` method, a count
in a brief is the registry's, T3 the names each DESIGN.md "Public API"
section lists exist (read, not imported), and T4 no new swallowed failure
in the analysis packages.
"""

import ast
import re
from pathlib import Path

from funhouse_agent import _WARNING_PACKAGES
from funhouse_agent import dispatch as D
from funhouse_agent.adapters import MODULE_REGISTRY

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T1 (d): a calc package for every module calc_package can write up

#: ``_IMPORT_MAP`` modules whose ``*_package`` method has another name.
_PACKAGE_METHOD = {"retaining_walls": "retaining_wall_package",
                   "seismic_geotech": "seismic_package"}

#: ``_IMPORT_MAP`` modules with no ``*_package`` method, and why (none
#: today: fem2d_package came with GM-328).
NO_PACKAGE_METHOD = {}


def test_every_calc_package_module_has_a_package_method():
    from calc_package import _IMPORT_MAP
    methods = set(D._load_adapter("calc_package").METHOD_REGISTRY)
    missing = [m for m in _IMPORT_MAP
               if _PACKAGE_METHOD.get(m, f"{m}_package") not in methods
               and m not in NO_PACKAGE_METHOD]
    assert missing == [], (
        f"calc_package can write up {missing}, but the agent has no "
        f"*_package method for them: add one, or say why in "
        f"NO_PACKAGE_METHOD")
    assert set(NO_PACKAGE_METHOD) <= set(_IMPORT_MAP)
    assert all(NO_PACKAGE_METHOD.values())


# ---------------------------------------------------------------------------
# T1: a count in a brief is the registry's

_WORDS = ("two three four five six seven eight nine ten eleven twelve "
          "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
          "twenty").split()
_NUMBER = {w: i for i, w in enumerate(_WORDS, start=2)}
_COUNT = re.compile(
    r"\b(\d+|" + "|".join(_WORDS) + r")\s+(?:analysis |calculation |"
    r"canned |calc[- ]package |calc )?(modules|tools|packages|generators)\b",
    re.I)


def _registry_counts() -> set:
    """Every count of the registry a brief could mean: all modules, the
    analysis modules, and the calc packages."""
    packages = [m for m in D._load_adapter("calc_package").METHOD_REGISTRY
                if m.endswith("_package")]
    analysis = [m for m in MODULE_REGISTRY if m not in D.REFERENCE_MODULES]
    return {len(MODULE_REGISTRY), len(analysis), len(packages)}


def _briefs():
    """(where, text) of every brief the agent reads: the module catalog
    and each method's brief, and the agent package's own docstring."""
    import funhouse_agent
    yield "funhouse_agent docstring", funhouse_agent.__doc__ or ""
    for agent, brief in D.list_agents().items():
        yield f"catalog: {agent}", str(brief)
    for agent in sorted(MODULE_REGISTRY):
        for method, info in D._load_adapter(agent).METHOD_INFO.items():
            yield f"{agent}.{method}", str(info.get("brief") or "")


def test_a_count_in_a_brief_is_the_registrys():
    counts = _registry_counts()
    wrong = []
    for where, text in _briefs():
        for m in _COUNT.finditer(text):
            n = m.group(1).lower()
            n = int(n) if n.isdigit() else _NUMBER[n]
            if n not in counts:
                wrong.append(f"{where}: {m.group(0)!r}")
    assert wrong == [], (
        f"these counts are not the registry's ({sorted(counts)}): {wrong}")


def test_the_count_check_reads_words_and_figures():
    assert _COUNT.search("any of the seventeen calc packages")
    assert _COUNT.search("routes to 50 modules")
    assert not _COUNT.search("both methods give 2 values")


# ---------------------------------------------------------------------------
# T3: the names a DESIGN.md "Public API" section lists exist

def _api_names(text: str):
    """The names the code blocks of a DESIGN.md's "Public API" section
    list (``Name``, ``Name(...)``, ``x = Name(...)``, ``Name.method``),
    or None where it has no such section. Text outside code blocks is
    prose ("there is no `analyze_x`") and is not read."""
    m = re.search(r"^## Public API[^\n]*\n(.*?)(?=^## |\Z)", text,
                  re.M | re.S)
    if not m:
        return None
    names = set()
    for block in re.findall(r"```[a-z]*\n(.*?)```", m.group(1), re.S):
        code = re.sub(r"#[^\n]*", "", block)
        prev = None
        while prev != code:
            prev, code = code, re.sub(r"\([^()]*\)|\[[^\[\]]*\]", "", code)
        code = re.sub(r"->\s*[\w.]+", "", code)
        code = re.sub(r"^\s*from\s+[\w.]+\s+import\s+", "", code,
                      flags=re.M)
        for tok in re.split(r"[,\n]", code):
            tok = re.sub(r"^\w+\s*=\s*", "", tok.strip())
            head = re.fullmatch(r"([A-Z]\w*)\.\w+", tok)
            tok = head.group(1) if head else tok
            if re.fullmatch(r"[A-Za-z_]\w*", tok):
                names.add(tok)
    return names


def _bound_names(init: Path) -> set:
    """The names a package ``__init__.py`` binds at its top level (defs,
    classes, assignments, imports, ``__all__``), read by AST."""
    out = set()

    def top(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name).split(".")[0]
                           for a in node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = getattr(node, "targets", None) or [node.target]
                for t in targets:
                    out.update(n.id for n in ast.walk(t)
                               if isinstance(n, ast.Name))
                if any(isinstance(t, ast.Name) and t.id == "__all__"
                       for t in targets) and isinstance(
                           node.value, (ast.List, ast.Tuple)):
                    out.update(e.value for e in node.value.elts
                               if isinstance(e, ast.Constant))
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                for part in ("body", "orelse", "finalbody"):
                    top(getattr(node, part, None) or [])
                for h in getattr(node, "handlers", None) or []:
                    top(h.body)

    top(ast.parse(init.read_text(encoding="utf-8")).body)
    return out


def test_every_design_public_api_name_exists():
    wrong, read = [], 0
    for design in sorted(REPO.glob("*/DESIGN.md")):
        names = _api_names(design.read_text(encoding="utf-8"))
        init = design.parent / "__init__.py"
        if names is None or not init.exists():
            continue
        read += 1
        missing = sorted(names - _bound_names(init))
        if missing:
            wrong.append(f"{design.parent.name}/DESIGN.md: {missing}")
    assert read >= 10, read
    assert wrong == [], (
        f"DESIGN.md Public API sections name what the package does not "
        f"export: {wrong}")


def test_the_api_reader_takes_calls_assignments_and_methods():
    text = ("## Public API\n```python\nFoo, bar(x, y=1) -> Baz\n"
            "r = Qux.compute(a)  # note\nbaz.y\n```\nThere is no `nope()`.\n"
            "## Next\n```python\nother\n```\n")
    assert _api_names(text) == {"Foo", "bar", "Qux"}


# ---------------------------------------------------------------------------
# T4: no swallowed failure in the analysis packages

#: The calculation packages (``funhouse_agent._WARNING_PACKAGES``) but the
#: adapters and the reference library, which have their own checks.
_ANALYSIS = [p for p in _WARNING_PACKAGES
             if p not in ("funhouse_agent.adapters", "geotech_references")]

_FIGURE_LEFT_OUT = ("a figure that fails to draw is left out of the "
                    "package; the numbers and the steps are unchanged")
_CLEANUP = "best-effort removal of a temporary file after validation"

#: Today's ``except Exception:`` (or bare ``except:``) handlers whose whole
#: body is ``pass``, ``continue`` or an empty return: (file, function) ->
#: (how many, why). A new one, or one more in a listed function, fails.
SWALLOWED = {
    ("lateral_pile/calc_steps.py", "get_tables"): (
        1, "a layer whose p-y curve cannot be drawn at its mid-depth is "
           "left out of the p-y table; the solve used the same models"),
    ("wave_equation/calc_steps.py", "_bearing_graph_figures"): (
        4, _FIGURE_LEFT_OUT),
    ("wave_equation/calc_steps.py", "_drivability_figures"): (
        4, _FIGURE_LEFT_OUT),
    ("calc_package/renderer.py", "figure_to_base64"): (
        1, "the figure is saved as drawn when the print-width pass fails; "
           "its record says so"),
    ("gstools_agent/plan_map.py", "_draw_plotly"): (
        1, "plotly is optional: without it there is no interactive chart "
           "and the static figure stands"),
    ("subsurface_characterization/diggs_parser.py", "_findall"): (
        1, "a path the XML element cannot take finds no element"),
    ("subsurface_characterization/diggs_parser.py", "_find"): (
        1, "a path the XML element cannot take finds no element"),
    ("subsurface_characterization/formats/diggs_validation.py",
     "validate_diggs_schema"): (1, _CLEANUP),
    ("subsurface_characterization/formats/diggs_validation.py",
     "validate_diggs_dictionary"): (1, _CLEANUP),
    ("dxf_import/discovery.py", "_collect_coords"): (
        1, "an entity with no geometry data is left out of the drawing's "
           "discovery summary"),
}


def _empty(node: ast.Return) -> bool:
    v = node.value
    if v is None or (isinstance(v, ast.Constant)
                     and v.value in (None, 0, 0.0, "", False)):
        return True
    return isinstance(v, (ast.Dict, ast.List, ast.Tuple)) and not (
        getattr(v, "keys", None) or getattr(v, "elts", None))


def _swallowed(tree) -> list:
    """The enclosing function of each broad handler in ``tree`` whose body
    is one ``pass``, ``continue`` or empty return."""
    out = []

    def visit(node, where):
        for child in ast.iter_child_nodes(node):
            inner = where
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = child.name
            if isinstance(child, ast.ExceptHandler) and (
                    child.type is None or (
                        isinstance(child.type, ast.Name)
                        and child.type.id in ("Exception",
                                              "BaseException"))) \
                    and len(child.body) == 1:
                s = child.body[0]
                if isinstance(s, (ast.Pass, ast.Continue)) or (
                        isinstance(s, ast.Return) and _empty(s)):
                    out.append(where)
            visit(child, inner)

    visit(tree, "<module>")
    return out


def test_no_new_swallowed_failure_in_the_analysis_packages():
    found = {}
    for pkg in _ANALYSIS:
        root = REPO / pkg.replace(".", "/")
        for f in sorted(root.rglob("*.py")):
            if "tests" in f.relative_to(root).parts:
                continue
            rel = f.relative_to(REPO).as_posix()
            for where in _swallowed(ast.parse(f.read_text("utf-8"))):
                found[(rel, where)] = found.get((rel, where), 0) + 1
    listed = {k: n for k, (n, why) in SWALLOWED.items()}
    assert found == listed, (
        "a broad except whose body is pass, continue or an empty return "
        "hides a failure from the engineer: let it raise, or put it in the "
        "result, or (for a figure or a clean-up) list it in SWALLOWED with "
        f"why. Found {sorted(set(found.items()) - set(listed.items()))}, "
        f"listed but gone {sorted(set(listed.items()) - set(found.items()))}")
    assert all(why for _, why in SWALLOWED.values())


def test_the_scan_sees_each_form():
    tree = ast.parse(
        "def a():\n    try:\n        x()\n    except Exception:\n"
        "        pass\n"
        "def b():\n    for i in y:\n        try:\n            x()\n"
        "        except:\n            continue\n"
        "def c():\n    try:\n        x()\n    except Exception:\n"
        "        return {}\n"
        "def d():\n    try:\n        x()\n    except ValueError:\n"
        "        pass\n"
        "def e():\n    try:\n        x()\n    except Exception:\n"
        "        return 1.5\n")
    assert _swallowed(tree) == ["a", "b", "c"]
