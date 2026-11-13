"""The references helper's description names what the helper can reach
(2026-11-10 review, item 3).

The hand-written list named a NOAA reference that does not exist and left
out Eurocode 7, AASHTO 1993, the USACE EMs and Caltrans. It is now built
from the registry (``dispatch.held_references``); these tests fail when a
reference module or a library reference package is added without
appearing there.
"""

import pytest

from funhouse_agent import dispatch
from funhouse_agent.adapters import MODULE_REGISTRY

#: The reference modules that search the others and hold no reference of
#: their own.
SEARCH_MODULES = {"reference_db", "figure_db"}


def _module_for(reference_id: str):
    """The reference module that serves a library reference id
    ('gec_12' -> 'gec12', 'dm7_1' -> 'dm7'), or None."""
    for cand in (reference_id, reference_id.replace("_", ""),
                 reference_id.rsplit("_", 1)[0]):
        if cand in dispatch.REFERENCE_MODULES:
            return cand
    return None


def test_every_reference_module_has_a_title():
    missing = sorted(n for n in dispatch.REFERENCE_MODULES - SEARCH_MODULES
                     if not MODULE_REGISTRY.get(n, {}).get("title"))
    assert not missing, (
        f"reference modules with no 'title' in MODULE_REGISTRY: {missing}; "
        "give each the name its manual is known by, so the references "
        "helper's description names it")


def test_the_description_names_every_held_reference():
    from funhouse_agent.deep.agent import (build_references_subagent,
                                           references_description)
    titles = dispatch.held_references()
    assert len(titles) == len(dispatch.REFERENCE_MODULES - SEARCH_MODULES)
    desc = build_references_subagent()["description"]
    for title in titles:
        assert title in references_description(False)
        assert title in desc
    assert "NOAA" not in desc
    for name in ("Eurocode 7", "AASHTO 1993", "EM 1110-2-2104",
                 "EM 1110-2-2107", "Caltrans"):
        assert name in desc


def test_every_library_reference_is_named(monkeypatch):
    """Each reference the installed library indexes (text or figures) is
    served by a reference module with a title: a package added to the
    library without a module and a title here fails."""
    monkeypatch.delenv("GEOTECH_REFERENCES_PACKS", raising=False)
    from geotech_references import _figures_db, _retrieval_db
    ids = {r["reference"] for r in _retrieval_db.list_indexed_references()}
    ids |= {r["reference"] for r in _figures_db.list_indexed_figures()}
    assert ids
    unnamed = sorted(i for i in ids if not MODULE_REGISTRY.get(
        _module_for(i) or "", {}).get("title"))
    assert not unnamed, (
        f"library references with no titled reference module: {unnamed}")


def test_a_new_titled_module_appears(monkeypatch):
    monkeypatch.setitem(MODULE_REGISTRY, "made_up_manual", {
        "adapter": "x", "brief": "x", "title": "Made-Up Manual 9"})
    monkeypatch.setattr(dispatch, "REFERENCE_MODULES",
                        dispatch.REFERENCE_MODULES | {"made_up_manual"})
    assert "Made-Up Manual 9" in dispatch.held_references()


@pytest.mark.parametrize("where", ["text", "native"])
def test_the_v1_consult_descriptions_name_them_too(where):
    if where == "text":
        from funhouse_agent.agent import CONSULT_TEXT_DESCRIPTION as desc
    else:
        from funhouse_agent.native_tools import CONSULT_REFERENCES_TOOL
        desc = CONSULT_REFERENCES_TOOL["function"]["description"]
    assert "NOAA" not in desc
    assert "Eurocode 7 Part 1 (EN 1997-1)" in desc
    # neither promises a chart read (the native consultant has no reader;
    # the text one has it only where the host reaches the PDFs)
    assert "read off a design chart" not in desc
    assert "figure charts" not in desc


@pytest.mark.parametrize("native", [True, False])
def test_the_v1_consultant_is_told_what_it_can_read(monkeypatch, native):
    from funhouse_agent import reference_docs, reviewer

    asked = {}

    class _Agent:
        def __init__(self, **kwargs):
            pass

        def ask(self, text):
            asked["text"] = text
            return type("R", (), {"answer": "ok"})()

    monkeypatch.setattr("funhouse_agent.agent.GeotechAgent", _Agent)
    reference_docs.pin_chart_reads(True)          # a host with the PDFs
    engine = type("E", (), {"native_tool_calling": native})()
    assert reviewer.consult_references(engine, "Q?") == "ok"
    if native:
        assert "No value can be read off a reference chart here" in \
            asked["text"]
        assert "read_reference_figure" not in asked["text"]
    else:
        assert "read it off with read_reference_figure" in asked["text"]
