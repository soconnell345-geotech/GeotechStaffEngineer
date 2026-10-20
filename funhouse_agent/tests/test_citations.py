"""Every manual citation the shipped calc text prints exists in that manual (GM-178).

A delivered calc PDF printed "GEC-6 Eq. 6-1" for a bearing capacity
equation; in GEC-6 that number is a chapter 6 equation, and the bearing
capacity equations are in chapter 5. This guard reads every string in the
calc steps (``*/calc_steps.py``), the calc package builders
(``calc_package/``, ``funhouse_agent/adapters/calc_package.py``), the files
that build a result's ``choices`` (every call to ``choice(``) and the
adapters (``funhouse_agent/adapters/``), finds each citation of the form
"<manual> Eq./Table/Figure x-y" and checks that the manual's chapter files
in ``geotech-references`` (``geotech_references/<manual>/text/*.json``: the
equations, tables and figures each section lists and the numbers its text
names) or its figure catalog carry that number.

Manuals checked: the GECs, DM 7.1 (UFS/UFC 3-220-10) and DM 7.2
(UFC 3-220-20), and the UFCs the reference library holds. A table, figure
or equation named after a manual in the same string belongs to it until
another source is named.

``REFERENCE_GAPS`` lists citations checked against the printed page and
found right, whose number the reference library's chapter text leaves out:
they are for the reference library to fill, not passes. An entry that the
library comes to carry, or that the code stops citing, fails the test, so
the list stays exact.

``slope_stability/`` (frozen) and ``foundry/`` (retired) are not scanned.

Run ``python -m funhouse_agent.tests.test_citations`` for the full listing,
including what the matcher could not parse.
"""

import ast
import glob
import json
import os
import re
from functools import lru_cache

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: Directories never scanned: frozen, retired, or not shipped.
SKIP_DIRS = {"slope_stability", "foundry", "foundry_test_harness",
             "geotech-references", "module_work", "build", "docs",
             "validation_examples", "tests", "reference_work", ".git",
             ".venv", "webapp", "tinyapps"}

# --- manuals -----------------------------------------------------------

#: Citation name -> reference-library package. Order matters: the longer
#: names first so "DM 7.2" is not read as "DM 7".
MANUALS = [
    (r"GEC[- ]?(\d{1,2})(?!\d)", None),  # GEC-n -> gec_n (n from the match)
    (r"(?:NAVFAC\s+)?DM[- ]?7\.0?1(?!\d)", "dm7_1"),
    (r"(?:NAVFAC\s+)?DM[- ]?7\.0?2(?!\d)", "dm7_2"),
    (r"DM7-1(?!\d)", "dm7_1"),
    (r"DM7-2(?!\d)", "dm7_2"),
    (r"UF[CS] 3-220-10N?(?!\d)", "dm7_1"),
    (r"UFC 3-220-20(?!\d)", "dm7_2"),
    (r"UFC 3-250-01(?!\d)", "ufc_pavement"),
    (r"UFC 3-250-03(?!\d)", "ufc_flexible_practice"),
    (r"UFC 3-250-04(?!\d)", "ufc_concrete_practice"),
    (r"UFC 3-250-11(?!\d)", "ufc_stabilization"),
    (r"UFC 3-301-01(?!\d)", "ufc_structural"),
    (r"UFC 4-023-03(?!\d)", "ufc_collapse"),
]

#: Other sources: a citation after one of these is not the previous
#: manual's. Matched case-sensitively as whole words.
OTHER_SOURCES = (
    r"UFC \d-\d{3}-\d{2}[A-Z]*|EM 1110-[\d-]+|AASHTO|EN 1997|Eurocode|"
    r"Caltrans|T&S|NAVFAC|NHI[- ]\d|FEMA|ASCE|ACI|AISC|API|ASTM|IBC|"
    r"COM624P|CGPR|Das\b|Coduto|Bowles|Terzaghi|Duncan|Brinch Hansen|"
    r"Hansen|Vesic|Meyerhof|Reese|Matlock|Fellenius|Poulos|Kulhawy|"
    r"Mayne|Seed|Idriss|Boulanger|Youd|Priebe|Schmertmann|Hough|Barron|"
    r"Hansbo|Tomlinson|Nordlund|Smith|Rausche|Caquot|Kerisel|Rankine|"
    r"Coulomb|Mononobe|Okabe|Seed|Whitman|Bray|Newmark|Jibson|Westergaard|"
    r"Huang|Peck|Gifford|Burland|D'Appolonia|Kumbhojkar|Prandtl|Reissner|"
    r"Wood Handbook|GSA|PDCA|DFI|PTI|Ensoft|LPILE|GRLWEAP|Plaxis|Slide2|"
    r"Ito|Matsui|Converse|Labarre|Cheney|Chassie|Sabatini|Elias|Christopher|"
    r"Lazarte|Samtani|Nowatzki|Brown|O'Neill|Reese & O'Neill|Zhang|"
    r"Kimmerling|Munfakh|Chen|McCarron|Hannigan|Elias|Hoek|Brown"
)

KIND_WORDS = {
    "eq": r"Eqs?\.?|Equations?|Eqns?\.?",
    "table": r"Tables?|Tbl\.?",
    "figure": r"Figs?\.?|Figures?",
}

#: One printed number: 5-14, 7-15a, E-1, A.3, 3.2, D-2. Not a bare "5".
NUM = r"(?:[A-Z]{1,2}[-.])?\d{1,2}[-.]\d{1,3}[a-z]?|[A-Z]{1,2}[-.]\d{1,3}[a-z]?"
NUM_LIST = (rf"(?:{NUM})(?:\s*(?:,|and|&|to|through|thru|–|—)\s*(?:{NUM}))*")


def _manual_regex():
    parts = [f"(?P<m{i}>{pat})" for i, (pat, _) in enumerate(MANUALS)]
    return re.compile("|".join(parts))


_MANUAL_RE = _manual_regex()
_OTHER_RE = re.compile(rf"(?<![\w-])(?:{OTHER_SOURCES})")
#: Where a manual's scope ends: a blank line, or a sentence's end (not a
#: reference-list entry's "(2025). Title. Ch. 5, Tables 5-2 to 5-5").
_BREAK_RE = re.compile(
    r"\n\s*\n|(?<![A-Z][a-z])(?<!\b[A-Za-z])(?<!\))\.\s+(?=[A-Z])"
    r"(?!(?:Ch|Chapter|Sec|Section|Vol|Part|App|Appendix|Annex)\b)")
_ITEM_RE = re.compile(
    "|".join(rf"(?P<{k}>(?<![A-Za-z])(?:{w})\s*(?:{NUM_LIST}))"
             for k, w in KIND_WORDS.items()))
_NUM_RE = re.compile(NUM)


def _norm(num):
    """One spelling for a number: dots as dashes, upper case, a figure's
    part letter dropped ("7-16a" is Figure 7-16)."""
    num = num.strip().replace(".", "-").replace(" ", "")
    if re.search(r"\d[a-h]$", num):
        num = num[:-1]
    return num.upper()


def _expand(text):
    """The numbers in "5-2 to 5-8" or "5-7, 5-8": a range within one
    chapter is filled in."""
    nums = _NUM_RE.findall(text)
    out = []
    for i, n in enumerate(nums):
        out.append(n)
        if i + 1 < len(nums):
            between = text[text.find(n) + len(n):]
            between = between[:between.find(nums[i + 1])]
            if re.search(r"to|through|thru|–|—", between):
                a = re.match(r"(.*[-.])(\d+)$", n)
                b = re.match(r"(.*[-.])(\d+)$", nums[i + 1])
                if a and b and a.group(1) == b.group(1):
                    for k in range(int(a.group(2)) + 1, int(b.group(2))):
                        out.append(f"{a.group(1)}{k}")
    return out


def _package_of(match):
    for i, (_, pkg) in enumerate(MANUALS):
        if match.group(f"m{i}") is not None:
            if pkg is None:
                # No backslash inside the f-string: Python 3.10/3.11 refuse it.
                n = int(re.search(r"\d+", match.group(0)).group())
                return f"gec_{n}"
            return pkg
    return None


def citations_in(text):
    """``(manual_label, package, kind, number)`` for each citation in one
    string, plus the item mentions that follow no manual (``package``
    None): those are for :func:`unparsed`."""
    events = []
    for m in _MANUAL_RE.finditer(text):
        events.append((m.start(), m.end(), "manual", m))
    for m in _OTHER_RE.finditer(text):
        if not any(s <= m.start() < e for s, e, k, _ in events
                   if k == "manual"):
            events.append((m.start(), m.end(), "other", m))
    for m in _ITEM_RE.finditer(text):
        events.append((m.start(), m.end(), "item", m))
    for m in _BREAK_RE.finditer(text):
        events.append((m.start(), m.end(), "other", m))
    events.sort(key=lambda e: e[0])
    current = None
    label = None
    out = []
    for start, end, kind, m in events:
        if kind == "manual":
            current, label = _package_of(m), m.group(0)
        elif kind == "other":
            current, label = None, None
        else:
            k = m.lastgroup
            for n in _expand(m.group(0)):
                out.append((label, current, k, _norm(n)))
    return out


# --- the reference library --------------------------------------------

def _ref_root():
    """The reference library's package folder: the submodule's (the pinned
    version CI checks out), else the installed package's (a worktree with
    the submodule not checked out), else None."""
    sub = os.path.join(ROOT, "geotech-references", "geotech_references")
    if glob.glob(os.path.join(sub, "gec_6", "text", "*.json")):
        return sub
    try:
        import geotech_references
    except ImportError:
        return None
    pkg = os.path.dirname(os.path.abspath(geotech_references.__file__))
    if glob.glob(os.path.join(pkg, "gec_6", "text", "*.json")):
        return pkg
    return None


REF_ROOT = _ref_root()
if REF_ROOT is None:
    pytest.skip("the reference library's chapter text is not here (no "
                "geotech-references submodule or package)",
                allow_module_level=True)

_LISTED = re.compile(
    rf"^\s*(?P<k>Eq(?:uation)?s?\.?|Tables?|Fig(?:ure)?s?\.?)\s*(?P<n>{NUM})")
_IN_BODY = {
    "eq": re.compile(rf"(?:Eq(?:uation)?s?\.?|Eqns?\.?)\s*\(?((?:{NUM})(?:\s*(?:,|and|to|through)\s*(?:{NUM}))*)"),
    "table": re.compile(rf"(?:TABLE|Tables?)\s*((?:{NUM})(?:\s*(?:,|and|to|through)\s*(?:{NUM}))*)"),
    "figure": re.compile(rf"(?:FIGURE|Fig(?:ure)?s?\.?)\s*((?:{NUM})(?:\s*(?:,|and|to|through)\s*(?:{NUM}))*)"),
}
#: An equation number printed at the right of its equation: "(5-8)".
_EQ_TAG = re.compile(rf"\(({NUM})\)")
#: A manual numbered in one sequence (GEC-4, 1999: "Equation 15",
#: "figure 31").
_BARE = {
    "eq": re.compile(r"\(?Equation\s+(\d{1,3})\)?(?![-.\d])"),
    "table": re.compile(r"(?:TABLE|Table)\s+(\d{1,3})(?![-.\d])"),
    "figure": re.compile(r"(?:FIGURE|[Ff]igure)\s+(\d{1,3})(?![-.\d])"),
}


def _kind_of(word):
    w = word.lower()
    return "eq" if w.startswith("eq") else "table" if w.startswith(
        "tab") else "figure"


@lru_cache(maxsize=None)
def library_index(package):
    """``{kind: set(number)}`` for one reference package, or None when the
    library has no such package."""
    base = os.path.join(REF_ROOT, package)
    if not os.path.isdir(base):
        return None
    idx = {"eq": set(), "table": set(), "figure": set()}
    for path in sorted(glob.glob(os.path.join(base, "text", "*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for sec in data.get("sections", []):
            for key, kind in (("equations", "eq"), ("tables", "table"),
                              ("figures", "figure")):
                for entry in sec.get(key) or []:
                    name = entry.get("id", "") if isinstance(entry, dict) \
                        else str(entry)
                    m = _LISTED.match(name)
                    if m:
                        idx[_kind_of(m.group("k"))].add(_norm(m.group("n")))
                    elif re.match(rf"^\s*({NUM})\s*$", name):
                        idx[kind].add(_norm(name))
            body = " ".join(str(sec.get(f) or "") for f in
                            ("title", "summary", "body", "key_points",
                             "equations", "tables", "figures"))
            for kind, rx in _IN_BODY.items():
                for m in rx.finditer(body):
                    for n in _expand(m.group(1)):
                        idx[kind].add(_norm(n))
            for m in _EQ_TAG.finditer(body):
                idx["eq"].add(_norm(m.group(1)))
            for kind, rx in _BARE.items():
                for m in rx.finditer(body):
                    idx[kind].add(m.group(1))
    cat = os.path.join(base, "figures_catalog.json")
    if os.path.exists(cat):
        with open(cat, encoding="utf-8") as fh:
            data = json.load(fh)
        for fig in data.get("figures", []):
            num = str(fig.get("figure_number") or "")
            if num.isdigit():
                idx["figure"].add(num)
            elif num:
                kind = "table" if num.lower().startswith("table") else "figure"
                num = re.sub(r"^(?:table|figure)\s*", "", num, flags=re.I)
                idx[kind].add(_norm(num))
    return idx


# --- the code ----------------------------------------------------------

def scanned_files():
    """The shipped files whose strings print as calc text or as a
    result's ``choices``."""
    files = set()
    for path in glob.glob(os.path.join(ROOT, "*", "calc_steps.py")):
        files.add(path)
    files.update(glob.glob(os.path.join(ROOT, "calc_package", "*.py")))
    files.update(glob.glob(os.path.join(ROOT, "funhouse_agent", "adapters",
                                        "*.py")))
    for path in glob.glob(os.path.join(ROOT, "*", "**", "*.py"),
                          recursive=True):
        rel = os.path.relpath(path, ROOT).split(os.sep)
        if any(part in SKIP_DIRS for part in rel[:-1]):
            continue
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if re.search(r"(?<![\w.])choice\(|\bchoices\.choice\(", src):
            files.add(path)
    return sorted(p for p in files if not any(
        part in SKIP_DIRS for part in
        os.path.relpath(p, ROOT).split(os.sep)[:-1]))


def _strings(tree):
    """``(lineno, text)`` for every string literal; an f-string's or a
    ``+`` concatenation's parts are joined, values read as ``{}``."""
    def text(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(text(v) if isinstance(v, ast.Constant) else "{}"
                           for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = text(node.left), text(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            t = text(node)
            if t is not None:
                for sub in ast.walk(node):
                    seen.add(id(sub))
                yield node.lineno, t
    for node in ast.walk(tree):
        if id(node) in seen:
            continue
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            t = text(node)
            if t is not None:
                if isinstance(node, ast.JoinedStr):
                    for sub in ast.walk(node):
                        seen.add(id(sub))
                yield node.lineno, t


def in_library(idx, kind, num):
    """Whether ``num`` is in the index; "15-16" in a manual numbered in
    one sequence (no chapter 15) is the numbers 15 to 16."""
    if num in idx[kind]:
        return True
    m = re.fullmatch(r"(\d+)-(\d+)", num)
    if m and not any(n.startswith(m.group(1) + "-") for n in idx[kind]):
        a, b = int(m.group(1)), int(m.group(2))
        return a < b and all(str(k) in idx[kind] for k in range(a, b + 1))
    return False


@lru_cache(maxsize=None)
def scan():
    """``(found, unknown, unparsed)``.

    ``found``: ``(file, line, manual, package, kind, number, ok)`` for every
    citation of a manual the library holds; ``unknown``: citations of a
    manual named in :data:`MANUALS` that the library does not hold;
    ``unparsed``: ``(file, line, snippet)`` for each Eq./Table/Figure
    number that follows no manual in its string (a bare "Table 5-4", or
    one after an author-year citation), which the guard cannot place.
    """
    found, unknown, unparsed = [], [], []
    for path in scanned_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for line, text in _strings(tree):
            for label, pkg, kind, num in citations_in(text):
                if pkg is None:
                    unparsed.append((rel, line, kind, num))
                    continue
                idx = library_index(pkg)
                if idx is None:
                    unknown.append((rel, line, label, kind, num))
                    continue
                found.append((rel, line, label, pkg, kind, num,
                              in_library(idx, kind, num)))
    return found, unknown, unparsed


#: Citations checked against the printed page and right, whose number the
#: library's chapter text and figure catalog leave out. For the reference
#: library to fill. Keyed ``(package, kind, number)``; the value says where
#: it is printed.
REFERENCE_GAPS = {
    ("gec_4", "eq", "15-16"): "Equations 15 and 16, p. 62 (Sec. 5.2.9)",
    ("gec_9", "eq", "A-9"): "Appendix A, p. 222",
    ("gec_9", "eq", "A-20"): "Appendix A, p. 226",
    ("gec_9", "eq", "A-21"): "Appendix A, p. 226",
    ("gec_9", "table", "A-1"): "Appendix A, p. 222",
    ("gec_9", "table", "A-2"): "Appendix A, p. 222",
    ("gec_10", "eq", "9-13"): "p. 9-43",
    ("gec_10", "eq", "10-2"): "p. 10-9",
    ("gec_10", "eq", "10-5"): "p. 10-10 (side resistance in cohesionless "
                              "soil, GM-241)",
    ("gec_10", "eq", "10-6"): "p. 10-10",
    ("gec_10", "eq", "10-7"): "p. 10-10",
    ("gec_10", "eq", "10-8"): "p. 10-11",
    ("gec_10", "eq", "10-9"): "p. 10-11",
    ("gec_10", "eq", "10-10"): "p. 10-11",
    ("gec_10", "eq", "10-11"): "p. 10-11",
    ("gec_10", "eq", "10-12"): "p. 10-11",
    ("gec_10", "eq", "10-13"): "p. 10-11",
    ("gec_10", "eq", "10-14"): "p. 10-14 (base resistance in cohesionless "
                               "soil)",
    ("gec_10", "eq", "10-15"): "p. 10-15 (side resistance in cohesive "
                               "soil, GM-244)",
    ("gec_10", "eq", "10-16"): "p. 10-16 (UC to CIUC strength)",
    ("gec_10", "eq", "10-17"): "p. 10-16 (UU to CIUC strength)",
    ("gec_10", "eq", "10-18"): "p. 10-16 (alpha by su(CIUC)/pa)",
    ("gec_10", "eq", "10-19"): "p. 10-19 (base resistance in cohesive "
                               "soil, GM-244)",
    ("gec_10", "eq", "10-20"): "p. 10-19 (the same, base depth under 3B)",
    ("gec_10", "table", "10-2"): "p. 10-19 (N*c by su)",
    ("gec_10", "eq", "10-21"): "p. 10-20 (rock side resistance)",
    ("gec_10", "eq", "10-22"): "p. 10-21 (rock side resistance where "
                               "caving is anticipated)",
    ("gec_10", "eq", "10-23"): "p. 10-22 (base resistance in rock)",
    ("gec_10", "eq", "10-24"): "p. 10-23 (Hoek-Brown base in fractured "
                               "rock, named as not built, GM-244)",
    ("gec_10", "eq", "10-25"): "p. 10-24",
    ("gec_10", "eq", "10-26"): "p. 10-24",
    ("gec_10", "eq", "10-27"): "p. 10-24",
    ("gec_10", "eq", "10-28"): "p. 10-24",
    ("gec_10", "eq", "10-29"): "p. 10-24",
    ("gec_10", "eq", "A-2"): "Appendix A, p. A-8 (phi' from (N1)60, "
                             "GM-241)",
    ("gec_10", "table", "8-4"): "p. 8-14 (resistance factors)",
    ("gec_10", "table", "10-3"): "p. 10-21 (alpha_E for caving rock)",
    ("gec_11", "eq", "4-21"): "Vol I p. 4-28",
    ("gec_11", "eq", "4-22"): "Vol I p. 4-28",
    ("gec_11", "eq", "4-23"): "Vol I p. 4-30",
    ("gec_11", "table", "4-6"): "Vol I p. 4-29",
    ("gec_12", "eq", "7-8"): "Vol I p. 244",
    ("gec_12", "eq", "7-9"): "Vol I p. 244",
    ("gec_12", "eq", "7-11"): "Vol I p. 248",
    ("gec_12", "eq", "7-19"): "Vol I p. 255",
    ("gec_12", "eq", "7-20"): "Vol I p. 255",
    ("gec_12", "eq", "7-38"): "Vol I p. 289",
    ("gec_12", "eq", "7-70"): "Vol I p. 346",
    ("gec_12", "eq", "15-1"): "Vol II p. 358",
    ("gec_12", "table", "7-6"): "Vol I p. 240",
    ("gec_12", "table", "7-9"): "Vol I p. 256",
    ("gec_12", "table", "D-6"): "Vol III p. 28",
    ("gec_12", "table", "D-51"): "Vol III p. 100",
    ("gec_12", "table", "D-100"): "Vol III p. 179",
}


def _bad(found):
    return sorted({(pkg, kind, num) for _, _, _, pkg, kind, num, ok in found
                   if not ok})


def test_the_matcher_reads_the_codes_styles():
    got = citations_in(
        "Brinch Hansen (1970); UFC 3-220-20 Tables 5-7, 5-8 (d_c by its "
        "note C); FHWA GEC-6 Eqs. 5-2 to 5-4, Table 5-4; Vesic (1975), "
        "Table 9-9; GEC 9 Fig. 4-2")
    assert ("dm7_2", "table", "5-7") in [g[1:] for g in got]
    assert ("dm7_2", "table", "5-8") in [g[1:] for g in got]
    assert ("gec_6", "eq", "5-3") in [g[1:] for g in got]
    assert ("gec_6", "table", "5-4") in [g[1:] for g in got]
    assert (None, "table", "9-9") in [g[1:] for g in got]
    assert ("gec_9", "figure", "4-2") in [g[1:] for g in got]
    # a sentence or a paragraph ends a manual's scope
    got = citations_in("UFC 3-220-20 Table 5-2. Table 2.6 of AASHTO.")
    assert (None, "table", "2-6") in [g[1:] for g in got]
    got = citations_in("UFC 3-220-20 (2025). Foundations. Ch. 5, Tables 5-2 "
                       "to 5-5 (Terzaghi factors).")
    assert ("dm7_2", "table", "5-4") in [g[1:] for g in got]
    # GEC-4 numbers in one sequence
    assert in_library(library_index("gec_4"), "figure", "31-32")
    assert in_library({"eq": {"15", "16"}}, "eq", "15-16")
    assert not in_library({"eq": {"5", "14", "5-1"}}, "eq", "5-14")


def test_the_library_index_reads_gec6_chapter_5():
    idx = library_index("gec_6")
    for num in ("5-2", "5-14"):
        assert num in idx["eq"]
    for num in ("5-1", "5-2", "5-3", "5-4", "5-5"):
        assert num in idx["table"]


def test_every_cited_number_is_in_its_manual():
    found, _, _ = scan()
    assert found, "the scan found no citations at all"
    bad = set(_bad(found)) - set(REFERENCE_GAPS)
    where = [f"{f}:{ln} {lab} {k} {n}" for f, ln, lab, pkg, k, n, ok in found
             if (pkg, k, n) in bad]
    assert not where, ("cited numbers not in the manual's chapter files "
                       "(check each against the printed page):\n"
                       + "\n".join(where))


def test_the_reference_gap_list_is_exact():
    found, _, _ = scan()
    bad = set(_bad(found))
    stale = sorted(set(REFERENCE_GAPS) - bad)
    assert not stale, ("REFERENCE_GAPS entries no longer needed (the "
                       "library carries them, or the code no longer cites "
                       f"them): {stale}")


def _report():
    found, unknown, unparsed = scan()
    print(f"{len(found)} citations of library manuals, "
          f"{sum(1 for x in found if not x[-1])} not found")
    for f, ln, lab, pkg, k, n, ok in found:
        if not ok:
            gap = " (reference gap)" if (pkg, k, n) in REFERENCE_GAPS else ""
            print(f"  MISSING {f}:{ln} {lab} {k} {n}{gap}")
    print(f"{len(unknown)} citations of manuals the library does not hold")
    for row in unknown:
        print("  ", *row)
    print(f"{len(unparsed)} numbers that follow no manual (not checked)")
    for row in unparsed:
        print("  ", *row)


def test_the_bearing_package_cites_gec6_chapter_5():
    """GEC-6's bearing capacity equations, factors and tables are in
    chapter 5 (Sec. 5.2, pp. 51-63); chapter 6 has numbers of its own
    (Eq. 6-1 is a lateral earth pressure equation), so an existence check
    alone would pass the old "Eq. 6-1" (GM-178)."""
    found, _, _ = scan()
    gec6 = [(f, ln, k, n) for f, ln, _, pkg, k, n, _ in found
            if f.startswith("bearing_capacity/") and pkg == "gec_6"]
    assert gec6
    assert all(n.startswith("5-") for _, _, _, n in gec6), gec6


if __name__ == "__main__":
    _report()
