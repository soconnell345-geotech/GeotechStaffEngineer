"""FHWA NHI-06-088/089 (Soils and Foundations) through the reference tools.

REF-29: the app's tools cite this manual (the slope search's Figs. 6-14 to
6-16 and 6-19 and p. 6-38, the sheet-pile Fig. 10-10), and a references
consult could not find it. These check each lookup a consult makes: a text
search, a section by number, a figure search and a figure by number, all
through ``dispatch.call_agent`` as the consult sub-agent calls them.
"""

import json

import pytest

pytest.importorskip("geotech_references.fhwa_soils_foundations")

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.cited_documents import module_documents  # noqa: E402

REF = "fhwa_soils_foundations"


def _call(agent, method, params):
    out = dispatch.call_agent(agent, method, params)
    json.dumps(out)
    assert "error" not in out, out
    return out


def test_registered_as_a_reference_module():
    assert REF in dispatch.REFERENCE_MODULES
    assert REF not in dispatch.ANALYSIS_MODULES
    assert module_documents(REF) == {"fhwa-nhi-06-088", "fhwa-nhi-06-089"}


def test_search_finds_janbu_charts():
    out = _call("reference_db", "reference_search",
                {"query": "Janbu stability charts", "reference": REF})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert "6.6.3" in [h["section_id"] for h in hits[:5]]
    out = _call(REF, "search_sections", {"query": "Nordlund"})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert "9.5.1.1" in [h["section_id"] for h in hits[:5]]


def test_section_lookup_cites_the_printed_page():
    out = _call(REF, "retrieve_section", {"section_id": "6.6.3"})
    sec = out["result"] if "result" in out else out
    assert sec["title"] == "Janbu’s Stability Charts"
    # the slope search cites NHI-06-088 p. 6-38 for circles above the toe
    assert "[p. 6-38]" in sec["body"]
    assert "a circle passing above the toe of the slope may be more critical" \
        in sec["body"]
    out = _call("reference_db", "reference_get",
                {"reference": REF, "section_id": "10.5.5"})
    sec = out["result"] if "result" in out else out
    assert sec["title"].startswith("Step 7")
    assert "[p. 10-43]" in sec["body"]


def test_figure_lookup_cited_figures():
    for num, page in (("6-14", "6-27"), ("6-15", "6-28"), ("6-16", "6-34"),
                      ("6-19", "6-39"), ("10-10", "10-22")):
        out = _call("figure_db", "figure_get",
                    {"reference": REF, "figure_number": num})
        fig = out["result"] if "result" in out else out
        assert fig["printed_page"] == page, num
    out = _call("figure_db", "figure_search",
                {"query": "Janbu stability chart", "reference": REF})
    hits = out["result"]
    hits = hits["hits"] if isinstance(hits, dict) else hits
    assert {"6-16", "6-19"} & {h["figure_number"] for h in hits[:5]}


# --- REF-38: every equation the app cites carries its printed formula ---

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
_SKIP_DIRS = {".git", "module_work", "geotech-references", "build", ".venv",
              "node_modules", "__pycache__", "packages"}
_CITE = __import__("re").compile(r"NHI[- ]06[- ]08[89]\b(?P<rest>.*)")
#: Where another reference's name starts, the citation of this one ends.
_OTHER = __import__("re").compile(
    r"\b(?:UFS|UFC|GEC|AASHTO|EN \d|O'Neill|Das|DM ?7|EM \d|NAVFAC|Caltrans"
    r"|FHWA-NHI-1|NHI-0[^6]|NHI-06-08[89])")
_EQS = __import__("re").compile(
    r"\bEqs?\.?\s*(\d{1,2}-\d{1,3}[a-z]?(?:/[a-z])?"
    r"(?:\s*(?:,|and|to)\s*\d{1,2}-\d{1,3}[a-z]?)*)")


def _cited_equations():
    """Each NHI-06-088/089 equation number the app's code and data cite on
    the line that names the manual ("NHI-06-089 Eqs. 9-41 to 9-43",
    "Eqs. 9-37a/b"), with a file that cites it."""
    import os
    import re
    out = {}
    for dp, dn, fn in os.walk(_ROOT):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS]
        for f in fn:
            if not f.endswith((".py", ".json")) or f == os.path.basename(__file__):
                continue
            path = os.path.join(dp, f)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    for m in _CITE.finditer(line):
                        rest = m.group("rest")
                        stop = _OTHER.search(rest)
                        rest = rest[:stop.start()] if stop else rest
                        for e in _EQS.finditer(rest):
                            prev, span = None, False
                            for ch, n, let, let2, to in re.findall(
                                    r"(\d{1,2})-(\d{1,3})([a-z]?)(?:/([a-z]))?|(to)",
                                    e.group(1)):
                                if to:
                                    span = True
                                    continue
                                nums = [f"{ch}-{n}{let}"] + ([f"{ch}-{n}{let2}"] if let2 else [])
                                if span and prev:
                                    nums += [f"{ch}-{k}" for k in range(int(prev) + 1, int(n))]
                                for num in nums:
                                    out.setdefault(num, os.path.relpath(path, _ROOT))
                                prev, span = n, False
    return out


def _library_equations():
    from geotech_references import _retrieval
    out = {}
    for ch in range(1, 13):
        for sec in _retrieval.load_chapter(REF, ch)["sections"]:
            for e in sec["equations"]:
                out[e["id"][4:]] = e
    return out


def test_every_cited_equation_has_its_printed_formula():
    cited = _cited_equations()
    # drilled shafts, settlement, earth pressure, SPT, unit weights at least
    assert {"9-42", "9-37a", "8-17", "10-10", "3-3", "2-7"} <= set(cited), cited
    eqs = _library_equations()
    missing = []
    for num, where in sorted(cited.items()):
        # "Eq. 9-37" for the pair 9-37a and 9-37b
        group = [num] if num in eqs else [k for k in eqs if k[:-1] == num and k[-1].isalpha()]
        if not group or not all(eqs[k].get("formula") for k in group):
            missing.append(f"{num} ({where})")
    assert not missing, ("NHI-06-088/089 equations cited without a formula in the "
                         "library (add them to the refs manifest's printed_equations): "
                         + ", ".join(missing))


def test_section_fetch_shows_the_formula():
    out = _call("reference_db", "reference_get",
                {"reference": REF, "section_id": "9.12.3"})
    sec = out["result"] if "result" in out else out
    by_id = {e["id"]: e for e in sec["equations"]}
    assert by_id["Eq. 9-42"]["formula"] == "βi = 1.5 – 0.135 √zi   with 1.2 > βi > 0.25"
    assert by_id["Eq. 9-42"]["source"] == "Typed in from p. 9-132."
