"""The references helper reaches everything it is told it has.

The owner's acceptance test for the reference release (BACKLOG REF-92):
every reference, method, figure and data set the helper is told about is
reachable, and nothing it is told about is missing. The 2026-11-10 reach
audit checked 22,467 such statements once; these tests keep the checks
standing, against whatever reference library is on the path:

* every lookup each reference module lists, and every alias of one, is
  called through dispatch with arguments sent as JSON (as a model sends
  them) and must answer; no listed method asks for what JSON cannot carry;
* every figure catalog row resolves to its page (and, where its PDF is on
  this machine, to a page inside that PDF);
* the helper's description names the references with no chapter text as
  the library indexes them;
* on a host with no reference PDFs and no fetcher, nothing the helpers read
  promises a chart read; with one PDF, only that reference's charts;
* every "Table / Fig. / Eq. N" a reference module's brief or user-manual
  entry names resolves to a lookup, a catalog figure or the method text;
* the helper's prompt names only modules in its scope, and each specialist
  reviewer's preamble only modules in its own.

The arguments are in ``reference_lookup_examples.json`` (``{module:
{method: parameters}}``), taken from the reference library's own tests
(the manuals' values) or made up; a method with no entry is tried with
arguments made from its description, and a failure says to add one.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("geotech_references")

from funhouse_agent import dispatch, reference_docs  # noqa: E402
from funhouse_agent.adapters import MODULE_REGISTRY  # noqa: E402
from funhouse_agent.adapters._reference_common import (  # noqa: E402
    build_lookup_registry, json_cannot_carry, takes)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = json.loads((HERE / "reference_lookup_examples.json").read_text(
    encoding="utf-8"))
SEARCH = {"reference_db", "figure_db"}
LOOKUP_MODULES = sorted(dispatch.REFERENCE_MODULES - SEARCH)
TEXT_RETRIEVAL = "Text Retrieval"


def _info(module: str) -> dict:
    return dispatch._load_adapter(module).METHOD_INFO


def _lookups(module: str) -> dict:
    """The lookups ``list_methods`` lists for ``module`` (no aliases, no
    text retrieval: those are the search tools, tested elsewhere)."""
    return {n: i for n, i in _info(module).items()
            if not i.get("alias_of") and i.get("category") != TEXT_RETRIEVAL}


def _as_sent(params: dict) -> dict:
    """``params`` as a model's tool call carries them."""
    return json.loads(json.dumps(params, allow_nan=False))


def _answered(result) -> bool:
    """Whether a lookup came back with something (not an error, not an
    empty result)."""
    if not isinstance(result, dict):
        return result not in (None, "", [], {})
    if "error" in result:
        return False
    body = {k: v for k, v in result.items()
            if k not in ("defaults_used", "note", "choices")}
    return bool(body) and body.get("result", body) not in (None, "", [], {})


_RANGE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(-?\d+(?:\.\d+)?)")
_QUOTED = re.compile(r"[`'\"]([A-Za-z0-9_\-+./ ]{1,40})[`'\"]")


def _made_up(spec: dict) -> list:
    """Values to try for a parameter with no example, from its type, its
    default and the numbers and quoted words in its description."""
    kind = str(spec.get("type", "")).lower()
    desc = str(spec.get("description", ""))
    values = [spec["default"]] if spec.get("default") is not None else []
    if any(t in kind for t in ("float", "int", "number", "any")):
        for a, b in _RANGE.findall(desc)[:2]:
            if float(a) < float(b):
                values.append(round((float(a) + float(b)) / 2, 3))
        values += [30, 1.0, 10, 0.5, 100, 2]
    if "str" in kind or kind == "any":
        values += [q.strip() for q in _QUOTED.findall(desc)][:6]
        values += ["sand", "clay"]
    if "bool" in kind:
        values += [True, False]
    if "list" in kind or "array" in kind:
        values.append([1.0, 2.0, 3.0])
    unique = []
    for v in values:
        if v not in unique:
            unique.append(v)
    return unique[:8]


def _calls_to_try(module: str, method: str, info: dict) -> list:
    if method in EXAMPLES.get(module, {}):
        return [EXAMPLES[module][method]]
    required = [n for n, s in (info.get("parameters") or {}).items()
                if isinstance(s, dict) and s.get("required")]
    if not required:
        return [{}]
    pools = [_made_up(info["parameters"][n]) for n in required]
    if not all(pools):
        return []
    return [dict(zip(required, combo))
            for combo in itertools.islice(itertools.product(*pools), 24)]


def _call(module: str, method: str, params: dict):
    return dispatch.call_agent(module, method, _as_sent(params),
                               allowed_agents=dispatch.REFERENCE_MODULES)


# ---------------------------------------------------------------------------
# Every listed lookup and alias answers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", LOOKUP_MODULES)
def test_every_listed_lookup_answers(module):
    failed = []
    for method, info in sorted(_lookups(module).items()):
        tries = _calls_to_try(module, method, info)
        last = "no arguments could be made from its description"
        for params in tries:
            result = _call(module, method, params)
            if _answered(result):
                break
            last = str(result.get("error") if isinstance(result, dict)
                       else result)[:200]
        else:
            where = ("its example" if method in EXAMPLES.get(module, {})
                     else "made-up arguments (add an example to "
                          "reference_lookup_examples.json)")
            failed.append(f"{module}.{method} with {where}: {last}")
    assert not failed, "\n".join(failed)


@pytest.mark.parametrize("module", LOOKUP_MODULES)
def test_every_alias_answers(module):
    info = _info(module)
    failed = []
    for alias, ai in sorted(info.items()):
        canonical = ai.get("alias_of")
        if not canonical or ai.get("category") == TEXT_RETRIEVAL:
            continue
        params = EXAMPLES.get(module, {}).get(canonical)
        if params is None:
            continue                      # the canonical's own test says so
        if not _answered(_call(module, alias, params)):
            failed.append(f"{module}.{alias} (alias of {canonical})")
    assert not failed, "\n".join(failed)


def test_no_listed_method_asks_for_what_json_cannot_carry():
    found = []
    for module in sorted(dispatch.REFERENCE_MODULES):
        adapter = dispatch._load_adapter(module)
        for method, info in adapter.METHOD_INFO.items():
            func = getattr(adapter.METHOD_REGISTRY.get(method),
                           "__wrapped__", None)
            for pname, pinfo in (info.get("parameters") or {}).items():
                why = json_cannot_carry(func, pinfo, pname)
                if why:
                    found.append(f"{module}.{method}({pname}): {why}")
    assert not found, "\n".join(found)


def test_a_method_json_cannot_carry_is_left_out():
    import types

    mod = types.ModuleType("made_up_lookups")

    def takes_a_function(curve_fn, depth_m=1.0):
        """A made-up lookup.

        Parameters
        ----------
        curve_fn : callable
            A function of depth.
        depth_m : float, optional
            Depth, m.
        """
        return {"value": depth_m}

    def takes_number_keys(pct_by_size):
        """A made-up lookup.

        Parameters
        ----------
        pct_by_size : dict
            Mapping of size (mm, the table's keys: 4.75, 2.0) to percent.
        """
        return {"value": 1.0}

    def takes_an_optional_function(depth_m, override_fn=None):
        """A made-up lookup.

        Parameters
        ----------
        depth_m : float
            Depth, m.
        override_fn : callable, optional
            Replaces the table.
        """
        return {"value": depth_m}

    def takes_word_keys(weights):
        """A made-up lookup.

        Parameters
        ----------
        weights : dict
            Mapping of soil name to weight.
        """
        return {"value": 1.0}

    for fn in (takes_a_function, takes_number_keys,
               takes_an_optional_function, takes_word_keys):
        fn.__module__ = mod.__name__
        setattr(mod, fn.__name__, fn)
    registry, info = build_lookup_registry([(mod, "Made up", "Made-up 1")])
    assert "takes_a_function" not in registry
    assert "takes_number_keys" not in registry
    assert set(info["takes_an_optional_function"]["parameters"]) == {
        "depth_m"}
    assert "takes_word_keys" in registry


# ---------------------------------------------------------------------------
# Every figure catalog row resolves
# ---------------------------------------------------------------------------

def test_every_catalog_figure_resolves_to_its_page(monkeypatch):
    from geotech_references import _figures_db

    # the database is checked against its sources, and its list of
    # catalogs read, once rather than on each of the ~5,000 fetches (each
    # reads the whole table: about 10 ms a fetch)
    db = _figures_db._ensure_db()
    monkeypatch.setattr(_figures_db, "_ensure_db", lambda: db)
    figure_get = dispatch._load_adapter("figure_db").METHOD_REGISTRY[
        "figure_get"]
    conn = _figures_db._ro_connect()
    try:
        rows = [tuple(r) for r in conn.execute(
            "SELECT reference, figure_number, pdf_page_index FROM figures")]
        if hasattr(_figures_db, "_catalog_refs"):
            catalogs = _figures_db._catalog_refs(conn)
            monkeypatch.setattr(_figures_db, "_catalog_refs",
                                lambda _conn: catalogs)
    finally:
        conn.close()
    assert rows
    try:
        import pymupdf
    except ImportError:                                # pragma: no cover
        pymupdf = None
    if not takes(_figures_db.resolve_pdf, "pdf_page_index"):
        pytest.skip("this reference library fetches a figure by its number "
                    "only, so a number held on several pages cannot be "
                    "fetched by page")
    pages = {}
    failed = []
    for reference, number, page in rows:
        row = figure_get({"reference": reference, "figure_number": number,
                          "pdf_page_index": page})
        if not isinstance(row, dict) or "error" in row or int(
                row.get("pdf_page_index", -1)) != int(page):
            failed.append(f"{reference} {number} p{page}: {str(row)[:160]}")
            continue
        if pymupdf is None:
            continue
        try:
            path, index = _figures_db.resolve_pdf(reference, number,
                                                  pdf_page_index=page)
        except FileNotFoundError:
            continue                      # the PDF is not on this machine
        if path not in pages:
            with pymupdf.open(path) as doc:
                pages[path] = doc.page_count
        if not 0 <= index < pages[path]:
            failed.append(f"{reference} {number}: page {index} of a "
                          f"{pages[path]}-page PDF")
    assert not failed, "\n".join(failed[:40])


# ---------------------------------------------------------------------------
# What the helper is told the references hold
# ---------------------------------------------------------------------------

def test_the_description_names_the_references_with_no_chapter_text():
    from funhouse_agent.deep.agent import references_description

    coverage = dispatch.reference_coverage()
    assert coverage, "the reference library on the path says nothing"
    titles = {name: MODULE_REGISTRY[name]["title"] for name in coverage}
    no_text = [titles[n] for n in coverage if not coverage[n]["text"]]
    desc = references_description()
    clause = desc.split("It searches", 1)[1].split(", tables")[0] \
        if "every one but" in desc else ""
    for name, title in titles.items():
        if coverage[name]["text"]:
            assert title not in clause, f"{title} has chapter text"
        else:
            assert title in clause, f"{title} has no chapter text"
    assert bool(no_text) == bool(clause)
    manual = _manual().REFERENCE_INFO["reference_db"][2]
    for title in no_text:
        assert title.split(" (")[0] in manual, title
    assert "every reference" not in manual


def _manual():
    spec = importlib.util.spec_from_file_location(
        "manual_content", REPO / "docs" / "user_manual" / "manual_content.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Chart reads promised only where the PDFs reach the host
# ---------------------------------------------------------------------------

@pytest.fixture
def host(tmp_path, monkeypatch):
    """A host with an empty docs folder, an empty cache, no packs and no
    fetcher; chart reads looked for, not pinned."""
    docs = tmp_path / "docs"
    cache = tmp_path / "cache"
    docs.mkdir()
    cache.mkdir()
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(docs))
    monkeypatch.setenv(reference_docs.CACHE_ENV, str(cache))
    monkeypatch.delenv("GEOTECH_REFERENCES_PACKS", raising=False)
    saved = reference_docs._FETCHER
    reference_docs.register_fetcher(None)
    reference_docs.pin_chart_reads(None)
    yield {"docs": docs, "cache": cache}
    reference_docs.register_fetcher(saved)


def _search(query: str) -> list:
    out = dispatch.call_agent("figure_db", "figure_search",
                              {"query": query, "limit": 5},
                              allowed_agents=dispatch.REFERENCE_MODULES)
    return [h for h in out.get("result", []) if isinstance(h, dict)
            and h.get("figure_number")]


def test_a_host_without_the_pdfs_is_promised_no_chart_read(host):
    from funhouse_agent.agent import _build_agent_system_prompt
    from funhouse_agent.deep.agent import (build_primary_tools,
                                           build_references_subagent)
    from funhouse_agent.system_prompt import CHART_READS_PARAGRAPH

    assert reference_docs.chart_reads_available() is False
    spec = build_references_subagent()
    names = {getattr(t, "name", None) for t in spec["tools"]}
    assert "read_reference_figure" not in names
    assert "read_reference_figure" not in spec["system_prompt"]
    assert "No value can be read off a reference chart here" in \
        spec["system_prompt"]
    assert "reads chart values" not in spec["description"]
    assert "value read off a design chart" not in spec["description"]
    assert "cannot read a value off a chart here" in spec["description"]
    primary = {getattr(t, "name", None) for t in build_primary_tools()}
    assert "read_reference_figure" not in primary
    assert "view_worked_example_source" in primary
    v1 = _build_agent_system_prompt(dispatch.REFERENCE_MODULES)
    assert CHART_READS_PARAGRAPH not in v1
    assert "### 10. read_reference_figure" not in v1
    hits = _search("log spiral passive earth pressure")
    assert hits
    for hit in hits:
        assert hit["read_value"] == reference_docs.NO_CHART_READ


def test_a_host_with_one_pdf_reads_only_its_charts(host):
    rows = reference_docs._catalog_pdfs()
    assert len({r[3] for r in rows}) > 1
    here, elsewhere = rows[0], next(r for r in rows if r[3] != rows[0][3])
    (host["docs"] / os.path.basename(here[3])).write_bytes(b"%PDF-1.4\n")
    assert reference_docs.chart_reads_available() is True
    assert reference_docs.figure_readable(*here[:3]) is True
    assert reference_docs.figure_readable(*elsewhere[:3]) is False
    # a copy fetched earlier is in the cache
    (host["cache"] / os.path.basename(elsewhere[3])).write_bytes(b"%PDF")
    assert reference_docs.figure_readable(*elsewhere[:3]) is True


def test_a_fetcher_makes_every_chart_readable(host):
    rows = reference_docs._catalog_pdfs()
    assert reference_docs.chart_reads_available() is False
    reference_docs.register_fetcher(lambda name, dest: None)
    assert reference_docs.chart_reads_available() is True
    assert reference_docs.figure_readable(*rows[0][:3]) is True
    from funhouse_agent.deep.agent import build_references_subagent
    spec = build_references_subagent()
    assert "read_reference_figure" in {getattr(t, "name", None)
                                       for t in spec["tools"]}
    assert "reads chart values off the actual figures" in spec["description"]


# ---------------------------------------------------------------------------
# Every table, figure and equation a brief or manual entry names resolves
# ---------------------------------------------------------------------------

_NUMBER = r"(?:[A-Z]{1,2}-)?\d+(?:[.-]\d+)*[a-z]?"
_DESIGNATION = re.compile(
    r"\b(Tables?|Figs?\.?|Figures?|Eqs?\.?|Equations?)\s+"
    rf"({_NUMBER}(?:\s*(?:,|and|to)\s*{_NUMBER})*)")
_PREFIXES = {"table": ("table",), "figure": ("figure", "fig"),
             "eq": ("eq", "equation")}


def _kind(word: str) -> str:
    word = word.lower()
    return ("table" if word.startswith("table")
            else "figure" if word.startswith("fig") else "eq")


def _numbers(text: str) -> list:
    out = []
    for part in re.split(r"\s*(?:,|\band\b)\s*", text):
        ends = [p.strip() for p in part.split(" to ")]
        if len(ends) == 2 and all(e.isdigit() for e in ends):
            out += [str(n) for n in range(int(ends[0]), int(ends[1]) + 1)]
        else:
            out += [e for e in ends if e]
    return out


def _designations(text: str) -> list:
    return [(_kind(m.group(1)), n) for m in _DESIGNATION.finditer(text)
            for n in _numbers(m.group(2))]


def _resolves(module: str, kind: str, number: str) -> bool:
    plain = number.lower().replace("-", "_").replace(".", "_")
    forms = {plain, re.sub(r"^([a-z])_", r"\1", plain)}
    info = _lookups(module)
    for name in info:
        for prefix in _PREFIXES[kind]:
            if any(re.search(rf"(^|_){prefix}_{f}(_|$)", name)
                   for f in forms):
                return True
    for entry in info.values():
        text = " ".join(str(entry.get(k) or "") for k in
                        ("brief", "returns", "notes", "references"))
        if (kind, number) in _designations(text):
            return True
    if kind == "figure":
        from geotech_references import _figures_db
        for row in _figures_db.list_indexed_figures():
            if dispatch.reference_module(row["reference"]) != module:
                continue
            try:
                _figures_db.figure_get(row["reference"], number)
                return True
            except KeyError as exc:
                if "pdf_page_index" in str(exc):
                    return True       # held on more than one page
            except Exception:                          # noqa: BLE001
                pass
    return False


def test_every_designation_in_a_brief_or_manual_entry_resolves():
    manual = _manual().REFERENCE_INFO
    failed = []
    checked = 0
    for module in LOOKUP_MODULES:
        texts = [("brief", MODULE_REGISTRY[module].get("brief", ""))]
        if module in manual:
            texts.append(("user manual", " ".join(manual[module][1:])))
        for where, text in texts:
            for kind, number in _designations(text):
                checked += 1
                if not _resolves(module, kind, number):
                    failed.append(f"{module} {where}: {kind} {number}")
    assert checked
    assert not failed, "\n".join(failed)


def test_designations_are_read_as_engineers_write_them():
    assert _designations("Tables 6 to 8 and Table 20") == [
        ("table", "6"), ("table", "7"), ("table", "8"), ("table", "20")]
    assert _designations("Eqs. 5-2 and 5-8, Fig. E-1") == [
        ("eq", "5-2"), ("eq", "5-8"), ("figure", "E-1")]


# ---------------------------------------------------------------------------
# Prompts name only the modules in scope
# ---------------------------------------------------------------------------

def _named_modules(text: str) -> set:
    """Registry module names a prompt names: in backticks, or any with an
    underscore as a whole word (plain words such as "settlement" are also
    module names and are left out)."""
    names = set(re.findall(r"`([a-z0-9_]+)`", text))
    names |= {n for n in MODULE_REGISTRY if "_" in n
              and re.search(rf"(?<![\w.]){re.escape(n)}(?![\w(])", text)}
    return names & set(MODULE_REGISTRY)


def test_the_references_helper_names_only_modules_it_holds():
    from funhouse_agent.deep.agent import build_references_subagent
    spec = build_references_subagent()
    named = _named_modules(spec["system_prompt"] + " " + spec["description"])
    assert named <= set(dispatch.REFERENCE_MODULES), (
        named - set(dispatch.REFERENCE_MODULES))


def test_each_specialist_names_only_modules_in_its_scope():
    from funhouse_agent import review_checklists as rc
    from funhouse_agent import reviewers as rv
    pairs = {
        "seismic": (rc.SEISMIC_REVIEWER_PREAMBLE, rv.SEISMIC_REVIEWER_SCOPE),
        "foundations": (rc.FOUNDATIONS_REVIEWER_PREAMBLE,
                        rv.FOUNDATIONS_REVIEWER_SCOPE),
        "earth retention": (rc.EARTH_RETENTION_REVIEWER_PREAMBLE,
                            rv.EARTH_RETENTION_REVIEWER_SCOPE),
        "slope/FEM": (rc.SLOPE_FEM_REVIEWER_PREAMBLE,
                      rv.SLOPE_FEM_REVIEWER_SCOPE),
        "pavement": (rc.PAVEMENT_SPECIALIST_PREAMBLE,
                     rv.PAVEMENT_SPECIALIST_SCOPE),
        "structural": (rc.STRUCTURAL_SPECIALIST_PREAMBLE,
                       rv.STRUCTURAL_SPECIALIST_SCOPE),
    }
    outside = {name: sorted(_named_modules(text) - set(scope))
               for name, (text, scope) in pairs.items()}
    assert not any(outside.values()), outside
