"""``render_tables``: a calc package's calculation in a report of the
agent's own (BACKLOG GM-67).

Live (F18): a bearing-capacity PDF the agent composed with ``html_to_pdf``
had no calculation tables; they were only in the second, standard package
PDF. ``render_tables`` writes the package's calculation (given and assumed
inputs, values used, method choices, steps with their equations and
references, checks, calculation tables, references) as an HTML file, by the
package's own template, and ``html_to_pdf`` puts it where the report's
``<div data-include="...">`` stands.

These tests hold:

* the file prints the same calculation as the package, section for section,
  for a bearing capacity, a retaining wall and an SOE wall;
* a report built with it prints every step, result, reference and table
  cell of that calculation;
* an include that cannot be filled stops the PDF, never a silent gap;
* the tool text names nothing a page or helper with calc_package lacks.
"""

import html as _html
import json
import os
import re
import unicodedata

import pytest
import matplotlib

matplotlib.use("Agg")  # avoid Tk backend errors

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import calc_package as CP

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "method_probes.json"), encoding="utf-8") as _fh:
    _PROBES = json.load(_fh)


def _probe(package: str) -> dict:
    return dict(_PROBES["calc_package|" + package]["params"])


#: The live failure's analysis, a wall, and the SOE package GM-67 names.
CASES = {
    "bearing_capacity_package": {
        **_probe("bearing_capacity_package"), "depth": 1.5,
        "shape": "square", "given": ["width", "friction_angle"],
        "assumptions": {"unit_weight": "not given; typical of the sand "
                                       "described"}},
    "retaining_wall_package": _probe("retaining_wall_package"),
    "soe_package": _probe("soe_package"),
}

_TAG = re.compile(r"<[^>]+>")


def _printed(words: str) -> str:
    """Text compared as a PDF prints it: ligatures and compatibility forms
    folded (NFKC) and no whitespace (a PDF breaks lines anywhere)."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", words))


def _text(fragment: str) -> str:
    """The words of an HTML fragment (no tags or entities), as printed."""
    return _printed(_html.unescape(_TAG.sub(" ", fragment)))


def _package_and_tables(package: str, params: dict, tmp_path):
    """The package's HTML and render_tables' file for the same inputs,
    each written as ``calc.html`` in a folder of its own (so their CSVs
    have the same names)."""
    pkg_dir, tab_dir = tmp_path / "pkg", tmp_path / "tab"
    pkg_dir.mkdir()
    tab_dir.mkdir()
    pkg = CP.METHOD_REGISTRY[package]({
        **params, "output_path": str(pkg_dir / "calc.html")})
    assert pkg["status"] == "success", pkg.get("error")
    tab = CP._render_tables({
        "package": package, **params,
        "output_path": str(tab_dir / "calc.html")})
    assert tab["status"] == "success", tab.get("error")
    with open(pkg["output_path"], encoding="utf-8") as fh:
        pkg_html = fh.read()
    with open(tab["output_path"], encoding="utf-8") as fh:
        tab_html = fh.read()
    return pkg, pkg_html, tab, tab_html


def _package_calculation(pkg_html: str) -> str:
    """The package's sections and references, its section numbers and its
    Figures section taken out (render_figures gives those)."""
    body = pkg_html[pkg_html.index("<!-- Sections -->"):
                    pkg_html.index("<!-- Footer -->")]
    body = re.sub(r'<div class="section">\s*<h2>\d+\. Figures</h2>.*?'
                  r'(?=<div class="section)', "", body, flags=re.DOTALL)
    return re.sub(r"<h2>\d+\. ", "<h2>", body)


def _file_calculation(tab_html: str) -> str:
    return tab_html[tab_html.index("<!-- Sections -->"):
                    tab_html.index("</body>")]


# ---------------------------------------------------------------------------
# The file prints the package's calculation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("package", sorted(CASES))
def test_file_prints_the_packages_calculation(package, tmp_path):
    pkg, pkg_html, tab, tab_html = _package_and_tables(
        package, CASES[package], tmp_path)
    assert _text(_file_calculation(tab_html)) == \
        _text(_package_calculation(pkg_html))
    # the sections it reports are the package's, less Figures, unnumbered
    titles = [_html.unescape(t) for t in
              re.findall(r"<h2>\d+\. (.*?)</h2>", pkg_html)]
    titles = [t for t in titles if t != "Figures"]
    assert tab["sections"] == titles + ["References"]
    # no title block, no figure, no footer; the calculation steps are there
    assert "header-block" not in tab_html and "<img" not in tab_html
    assert "footer" not in tab_html
    assert 'class="calc-step"' in tab_html
    # the CSVs are the package's, by the same names, with the same rows
    assert [os.path.basename(t["output_path"]) for t in tab["tables"]] == \
        [os.path.basename(t["output_path"]) for t in pkg["tables"]]
    for a, b in zip(tab["tables"], pkg["tables"]):
        with open(a["output_path"], encoding="utf-8-sig") as fa, \
                open(b["output_path"], encoding="utf-8-sig") as fb:
            assert fa.read() == fb.read()
    # the key results are echoed as the package echoes them
    for key, value in pkg.items():
        if key not in tab and key not in (
                "format", "html_length", "tables_note"):
            pytest.fail(f"{key} is in the package's result only")


def test_given_and_assumed_inputs_print_apart(tmp_path):
    _, _, _, tab_html = _package_and_tables(
        "bearing_capacity_package", CASES["bearing_capacity_package"],
        tmp_path)
    given = tab_html[tab_html.index("<h3>Given Inputs</h3>"):
                     tab_html.index("<h3>Assumed Inputs</h3>")]
    assumed = tab_html[tab_html.index("<h3>Assumed Inputs</h3>"):
                       tab_html.index("<h3>Values Used</h3>")]
    assert "Footing width" in given and "unit weight" not in given.lower()
    assert "typical of the sand described" in assumed


# ---------------------------------------------------------------------------
# A report built with it
# ---------------------------------------------------------------------------

def test_report_prints_every_step_result_reference_and_cell(tmp_path):
    """Two analyses in one report: every step's title, equation, values
    put in, result and reference, and every table cell, is in the PDF;
    the calculation's styles go into the report's head once."""
    pytest.importorskip("fitz")
    import fitz

    files, tags = [], []
    for package in ("bearing_capacity_package", "retaining_wall_package"):
        res = CP._render_tables({
            "package": package, **CASES[package],
            "output_path": str(tmp_path / f"{package}.html")})
        assert res["status"] == "success", res.get("error")
        assert res["output_path"] in res["html_include_tag"]
        files.append(res["output_path"])
        tags.append(res["html_include_tag"])
    report = ("<html><head><style>h1{color:#234;}</style></head><body>"
              "<h1>Wall and footing report</h1><p>Our narrative.</p>"
              "<h2>Appendix A. Footing</h2>" + tags[0] +
              "<h2>Appendix B. Wall</h2>" + tags[1] +
              "<p>End of report.</p></body></html>")

    kept, included, problems = CP._include_files(report)
    assert problems == [] and included == files
    assert kept.count(".calc-tables .calc-step {") == 1   # one style sheet
    assert kept.index(".calc-tables") < kept.index("</head>")
    assert "data-include" not in kept

    pdf = CP._generate_html_to_pdf({"html": report,
                                    "output_path": str(tmp_path / "r.pdf")})
    assert pdf["status"] == "success", pdf.get("error")
    assert pdf["included"] == files
    assert "fills_not_drawn" not in pdf      # the calculation sets no fill
    doc = fitz.open(pdf["output_path"])
    printed = _printed(" ".join(page.get_text() for page in doc))
    doc.close()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            calc = fh.read()
        parts = [m.group(1) for cls in ("step-title", "equation",
                                        "substitution", "result",
                                        "reference")
                 for m in re.finditer(rf'<div class="{cls}">(.*?)</div>',
                                      calc, re.DOTALL)]
        parts += re.findall(r"<td>(.*?)</td>", calc, re.DOTALL)
        parts += re.findall(r"<li>(.*?)</li>", calc, re.DOTALL)
        assert len(parts) > 40
        missing = [_text(p) for p in parts if _text(p) not in printed]
        assert missing == [], (os.path.basename(path), missing[:5])
    assert printed.index(_text("Appendix A. Footing")) < \
        printed.index(_text("Appendix B. Wall")) < \
        printed.index(_text("End of report."))


def test_an_include_that_cannot_be_filled_stops_the_pdf(tmp_path):
    pytest.importorskip("fitz")
    out = str(tmp_path / "r.pdf")
    for bad, why in (
            ('<div data-include="nowhere.html"></div>', "no such file"),
            ('<div data-include=""></div>', "names no file"),
            ('<div data-include="calc.html"><p>x</p></div>', "holds content"),
    ):
        (tmp_path / "calc.html").write_text("<body><p>calc</p></body>",
                                            encoding="utf-8")
        res = CP._generate_html_to_pdf({
            "html": f"<html><body><h1>R</h1>{bad}</body></html>",
            "output_path": out})
        assert res["status"] == "error", bad
        assert any(why in p for p in res["include_problems"]), (
            bad, res["include_problems"])
        assert "render_tables" in res["error"]
        assert not os.path.exists(out)
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    res = CP._generate_html_to_pdf({
        "html": f'<html><body><div data-include="{png}"></div></body></html>',
        "output_path": out})
    assert any("not an HTML file" in p for p in res["include_problems"])


def test_a_relative_include_resolves_beside_html_path(tmp_path):
    pytest.importorskip("fitz")
    (tmp_path / "calc.html").write_text(
        "<html><head><style>.calc-tables p{color:#123;}</style></head>"
        '<body><div class="calc-tables"><p>Step one.</p></div></body></html>',
        encoding="utf-8")
    (tmp_path / "report.html").write_text(
        "<html><body><h1>R</h1><div data-include='calc.html'/></body></html>",
        encoding="utf-8")
    res = CP._generate_html_to_pdf({"html_path": str(tmp_path / "report.html"),
                                    "output_path": str(tmp_path / "r.pdf")})
    assert res["status"] == "success", res.get("error")
    assert res["included"] == [str(tmp_path / "calc.html")]


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------

def test_registered_and_routed():
    assert "render_tables" in CP.METHOD_REGISTRY
    info = CP.METHOD_INFO["render_tables"]
    assert info["parameters"]["package"]["required"]
    assert info["parameters_of"] == "package"
    listed = D.list_methods("calc_package")
    assert any("render_tables" in v for v in listed.values())


def test_names_the_package_or_refuses():
    for bad in ({}, {"package": "html_to_pdf"}, {"package": "render_tables"},
                {"package": "render_figures"}, {"package": "nope"}):
        res = CP._render_tables(bad)
        assert res["status"] == "error"
        assert "bearing_capacity_package" in res["error"]


def test_dispatch_checks_the_named_packages_inputs():
    mod = D._load_adapter("calc_package")
    D._check_parameter_names(mod, "render_tables", {
        "package": "bearing_capacity_package", "width": 2,
        "output_path": "x.html"})
    with pytest.raises(ValueError, match="unknown parameter"):
        D._check_parameter_names(mod, "render_tables", {
            "package": "bearing_capacity_package", "widht": 2})
    # figures' own options are not this method's
    with pytest.raises(ValueError, match="output_dir"):
        D._check_parameter_names(mod, "render_tables", {
            "package": "bearing_capacity_package", "output_dir": "x"})


def test_the_mode_lasts_one_call(tmp_path):
    """A package built after a render_tables call is a whole package."""
    CP._render_tables({"package": "bearing_capacity_package",
                       **CASES["bearing_capacity_package"],
                       "output_path": str(tmp_path / "t.html")})
    pkg = CP._generate_bearing_capacity_package({
        **CASES["bearing_capacity_package"],
        "output_path": str(tmp_path / "p.html")})
    assert pkg["status"] == "success"
    with open(pkg["output_path"], encoding="utf-8") as fh:
        assert "header-block" in fh.read()
    assert CP._RENDER_MODE.get() is None


def test_default_file_names_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    paths = set()
    for _ in range(2):
        res = CP._render_tables({"package": "bearing_capacity_package",
                                 **CASES["bearing_capacity_package"]})
        assert res["status"] == "success", res.get("error")
        assert os.path.dirname(res["output_path"]) == str(tmp_path)
        assert os.path.basename(res["output_path"]).startswith(
            "bearing_capacity_calculation_")
        paths.add(res["output_path"])
    assert len(paths) == 2


def test_an_output_path_becomes_an_html_file(tmp_path):
    res = CP._render_tables({"package": "bearing_capacity_package",
                             **CASES["bearing_capacity_package"],
                             "output_path": str(tmp_path / "calc.pdf")})
    assert res["output_path"] == str(tmp_path / "calc.html")


# ---------------------------------------------------------------------------
# The tool text names nothing a page with calc_package lacks
# ---------------------------------------------------------------------------

#: The tools every agent reaching calc_package has: the module tools.
_MODULE_TOOLS = {"list_agents", "list_methods", "describe_method",
                 "call_agent"}


def _recorded_agents() -> dict:
    """``{(build, agent): [tool names]}`` from the recorded page builds."""
    path = os.path.join(_HERE, os.pardir, os.pardir, "webapp", "tests",
                        "page_fingerprints.json")
    with open(path, encoding="utf-8") as fh:
        golden = json.load(fh)
    out = {}
    for build, entry in golden["builds"].items():
        for call in entry["app"]["calls"]:
            if not call:
                continue
            out[(build, "primary")] = call.get("tool_order") or []
            for helper in call.get("subagents") or []:
                out[(build, helper["name"])] = helper.get("tool_order") or []
    return out


def _named(text: str) -> set:
    return set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b",
                          text))


@pytest.mark.parametrize("method", ["render_tables", "html_to_pdf"])
def test_tool_text_names_no_tool_the_page_lacks(method):
    agents = _recorded_agents()
    # the agents that reach calc_package: the geotech page's primary and
    # its calc helper (each specialist has the module tools and
    # calc_package, test_reviewers)
    reach = {k: set(v) for k, v in agents.items()
             if k in {("geotech_default", "primary"),
                      ("geotech_default", "calc")}}
    assert len(reach) == 2 and all(_MODULE_TOOLS <= v
                                   for v in reach.values())
    every_tool = set().union(*map(set, agents.values()))
    info = CP.METHOD_INFO[method]
    named = _named(json.dumps(info))
    tools = named & every_tool
    for who, has in reach.items():
        assert tools <= has, (who, sorted(tools - has))
    if method == "render_tables":
        # its own text names only the module tools
        assert tools <= _MODULE_TOOLS, sorted(tools)
    # modules: only those every specialist has beside its own
    assert named & set(D.MODULE_REGISTRY) <= D.DELIVERABLE_MODULES
    # packages and calc_package methods: real ones
    calc = {n for n in named if n.endswith("_package")
            or n.startswith("render_") or n == "html_to_pdf"} - {
                "calc_package"}
    assert calc <= set(CP.METHOD_REGISTRY), sorted(calc - set(
        CP.METHOD_REGISTRY))
