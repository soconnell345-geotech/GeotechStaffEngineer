"""The digest tools on the review page (``GEOTECH_REVIEW_DIGEST``) - offline.

Pins both sides of the switch (off: the page is what it was, legacy or lean;
on: the lean agent and its reading helper carry four read-only digest tools),
and what the tools return: valid JSON under the budget, strongest first, a
``next`` wherever a list continues, ``pdf_page`` on every page, and the
digest written under the conversation's working folder.
"""

import json
import os

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep import digest_tools as DT  # noqa: E402
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.review_digest import clear_cache  # noqa: E402
from planlens.testing import (  # noqa: E402
    build_synthetic_report, build_synthetic_review_document,
    build_synthetic_submittal)

DIGEST = set(DT.DIGEST_TOOLS)
GEOTECH_WORDS = ("drawing_ir", "digitize_drawing", "query_drawing",
                 "snip_region", "call_agent", "figure_db", "geotech")
BUDGET = 12000


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)
    monkeypatch.delenv("GEOTECH_REVIEW_SHAPE1_PAGES", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def uploads():
    return {"submittal.pdf": build_synthetic_submittal().pdf,
            "review_set.pdf": build_synthetic_review_document().pdf,
            "report.pdf": build_synthetic_report().pdf}


def long_pdf(n_pages=150):
    """A long manual: running header, printed page numbers, captions,
    references on every page, a repeated phrase to search for."""
    doc = fitz.open()
    for i in range(n_pages):
        p = doc.new_page(width=612, height=792)
        p.insert_text((460, 50), "MANUAL 1-2-3", fontsize=9)
        ch = i // 10 + 1
        p.insert_text((72, 110), f"Table {ch}-{i % 10 + 1}", fontsize=10)
        p.insert_text((72, 124), f"Allowable bearing values for zone {i}",
                      fontsize=10)
        body = (f"The allowable bearing pressure for zone {i} is given in "
                f"Table {ch}-{i % 10 + 1}; see also Figure {ch}-2, SHEET "
                f"S-{100 + i} and DETAIL 4/S-{300 + i}. ") * 4
        p.insert_textbox(fitz.Rect(72, 150, 540, 700), body, fontsize=10)
        p.insert_text((300, 740), f"{ch}-{i % 10 + 1}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def _review_kwargs():
    from webapp.profiles import DOCUMENT_REVIEW
    return DOCUMENT_REVIEW.build_kwargs()


class Scripted(FakeMessagesListChatModel):
    script: list = []
    seen: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="done")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script):
    return Scripted(responses=[AIMessage(content="x")], script=list(script),
                    seen=[])


def _tools_of(agent):
    return agent.nodes["tools"].bound.tools_by_name


def _tools(attachments):
    return {t.name: t for t in DT.make_digest_tools(attachments)}


def _call(tool, args=None):
    out = tool.invoke(args or {})
    assert len(out) <= BUDGET, (tool.name, len(out))
    return json.loads(out)


def _pages_carry_pdf_page(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("page"), int):
            assert obj.get("pdf_page") == obj["page"] + 1, obj
        for v in obj.values():
            _pages_carry_pdf_page(v)
    elif isinstance(obj, list):
        for v in obj:
            _pages_carry_pdf_page(v)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_digest_tools_only_with_lean_and_the_switch(monkeypatch):
    def names():
        return set(_tools_of(build_deep_agent(_model([]),
                                              **_review_kwargs())))
    legacy_off = names()
    assert not legacy_off & DIGEST                        # everything off
    monkeypatch.setenv(review_flags.DIGEST_ENV, "1")
    assert names() == legacy_off                          # legacy: unchanged
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.delenv(review_flags.DIGEST_ENV)
    lean_off = names()
    assert not lean_off & DIGEST                          # lean alone: no
    monkeypatch.setenv(review_flags.DIGEST_ENV, "1")
    tools = _tools_of(build_deep_agent(_model([]), **_review_kwargs()))
    assert set(tools) == lean_off | DIGEST                # exactly the four
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for name in DIGEST:
        desc = tools[name].description or ""
        for word in GEOTECH_WORDS:
            assert word not in desc.lower(), (name, word)
        assert "free layer" in desc.lower() and "needs_look" in desc
        schema = json.dumps(convert_to_openai_tool(tools[name]))
        assert "$ref" not in schema and "$defs" not in schema
    props = convert_to_openai_tool(tools["digest_search"])["function"][
        "parameters"]["properties"]
    assert {"query", "documents", "pages", "kinds", "offset"} <= set(props)


def test_the_reading_helper_gets_them_only_with_the_switch(monkeypatch):
    from funhouse_agent.deep import review_agent
    seen = []
    real = review_agent.make_reader_tool

    def spy(model, reader_tools, *a, **kw):
        seen.append({t.name for t in reader_tools})
        return real(model, reader_tools, *a, **kw)

    monkeypatch.setattr(review_agent, "make_reader_tool", spy)
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    build_deep_agent(_model([]), **_review_kwargs())
    monkeypatch.setenv(review_flags.DIGEST_ENV, "1")
    build_deep_agent(_model([]), **_review_kwargs())
    off, on = seen
    assert not off & DIGEST
    assert on == off | DIGEST
    assert "annotate_document" not in on                 # still read-only


def test_the_arm_and_the_switch_are_registered(monkeypatch):
    A = review_flags
    assert A.DIGEST_ENV == "GEOTECH_REVIEW_DIGEST"
    assert A.DIGEST_ENV in A.ALL_ENVS
    assert A.ARMS["digest"] == {
        A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
        A.VISION_STRUCTURED_ENV: "1", A.DIGEST_ENV: "1"}
    monkeypatch.setenv(A.DIGEST_ENV, "1")
    with A.switches(A.ARMS["lean"]):
        assert not A.digest()                             # an arm clears it
    assert A.digest()
    with A.switches(A.ARMS["digest"]):
        assert A.digest() and A.lean_agent() and A.vision_structured()
        assert not A.geometry()
    assert "GEOTECH_REVIEW_DIGEST=1" in A.describe()


# ---------------------------------------------------------------------------
# document_inventory
# ---------------------------------------------------------------------------

def test_inventory_of_every_upload(uploads, tmp_path):
    out = _call(_tools(uploads)["document_inventory"])
    assert [d["document"] for d in out["documents"]] == list(uploads)
    assert out["totals"] == {"documents": 3, "pages": 38, "needs_look": 8,
                             "markups": 5}
    assert out["shape_hint"] == "large" and "20 pages" in out["shape_note"]
    review = next(d for d in out["documents"]
                  if d["document"] == "review_set.pdf")
    assert review["markups"]["authors"] == {"Contractor B": 3,
                                            "Reviewer A": 2}
    assert review["needs_look"]["pdf_pages"] == "2,5"
    _pages_carry_pdf_page(out)
    # built once, into the working folder, and reused
    root = tmp_path / "work" / "digest"
    assert len(os.listdir(root)) == 3
    for folder in os.listdir(root):
        assert sorted(os.listdir(root / folder)) == [
            "index.sqlite", "inventory.json", "pages.jsonl",
            "references.json"]
    again = _call(_tools(uploads)["document_inventory"])
    assert again["totals"] == out["totals"]
    # one upload, and a name that is not one
    one = _call(_tools(uploads)["document_inventory"],
                {"documents": ["submittal.pdf", "nope.pdf"]})
    assert [d["document"] for d in one["documents"]] == ["submittal.pdf"]
    assert one["shape_hint"] == "small"
    assert one["errors"][0]["document"] == "nope.pdf"


def test_inventory_says_what_is_cited_but_not_uploaded(uploads):
    from funhouse_agent.review_digest.tests.test_review_digest_offline \
        import one_sheet_pdf, typed_set_pdf
    out = _call(_tools({"set.pdf": typed_set_pdf(),
                        "C-102.pdf": one_sheet_pdf()})["document_inventory"])
    missing = {m["id"] for m in out["cited_but_not_uploaded"]}
    assert missing == {"S-501", "10.35B", "11.51", "840.54", "30.19"}
    assert {f["id"] for f in out["cited_and_found"]} == {"10.17", "C-102"}


def test_nothing_uploaded_and_a_bad_upload(uploads):
    assert "error" in _call(_tools({})["document_inventory"])
    out = _call(_tools({"notes.docx": b"PK\x03\x04 not a pdf",
                        "report.pdf": uploads["report.pdf"]})[
        "document_inventory"])
    assert [d["document"] for d in out["documents"]] == ["report.pdf"]
    assert out["errors"][0]["document"] == "notes.docx"


def test_a_big_upload_set_stays_within_budget(uploads):
    many = dict(uploads)
    for i in range(12):
        many[f"manual_{i}.pdf"] = long_pdf(40 + i)
    out = _call(_tools(many)["document_inventory"])
    names = [d["document"] for d in out["documents"]]
    if "next" in out:
        assert out["left_out"] == len(many) - len(names)
        rest = _call(_tools(many)["document_inventory"],
                     {"documents": out["next"]["documents"]})
        names += [d["document"] for d in rest["documents"]]
    assert sorted(names) == sorted(many)


# ---------------------------------------------------------------------------
# digest_search
# ---------------------------------------------------------------------------

def test_search_across_uploads_and_narrowed(uploads):
    tools = _tools(uploads)
    out = _call(tools["digest_search"], {"query": "pile embedment"})
    hit = out["hits"][0]
    assert (hit["document"], hit["page"], hit["pdf_page"]) == \
        ("review_set.pdf", 1, 2)
    assert hit["source"] == "markup" and len(hit["bbox"]) == 4
    assert hit["needs_look"] is True
    # documents, kinds and pages narrow it
    out = _call(tools["digest_search"], {"query": "borings",
                                         "documents": ["submittal.pdf"]})
    assert out["document"] == "submittal.pdf" and out["hits"]
    assert all("document" not in h for h in out["hits"])
    assert all(h["page"] in (1, 2, 3) for h in _call(
        tools["digest_search"], {"query": "borings", "pages": "1-3",
                                 "documents": ["submittal.pdf"],
                                 "limit": 50})["hits"])
    assert not _call(tools["digest_search"], {
        "query": "borings", "kinds": ["drawing_sheet"],
        "documents": ["submittal.pdf"]})["hits"]
    # nothing found says what the free layer cannot see
    none = _call(tools["digest_search"], {"query": "zzqx nonexistent"})
    assert none["hits"] == [] and "needs_look" in none["note"]
    bad = _call(tools["digest_search"], {"query": "x", "pages": "400",
                                         "documents": ["report.pdf"]})
    assert "out of range" in bad["errors"][0]["error"]


@pytest.mark.parametrize("query", ['"', "AND OR NOT", "NEAR(a b)", "*", "%",
                                   "'; DROP TABLE lines; --", "((("])
def test_search_takes_fts_hostile_text(uploads, query):
    out = _call(_tools(uploads)["digest_search"], {"query": query})
    assert isinstance(out["hits"], list)


def test_search_pages_on_with_next(uploads):
    tools = _tools({"manual.pdf": long_pdf(150)})
    first = _call(tools["digest_search"], {"query": "allowable bearing",
                                           "limit": 60})
    assert len(first["hits"]) >= 1 and first["next"]["offset"] >= 1
    assert all(h["match"] == "exact" for h in first["hits"])
    _pages_carry_pdf_page(first)
    second = _call(tools["digest_search"], {
        "query": "allowable bearing", "limit": 60,
        "offset": first["next"]["offset"]})
    a = {(h["page"], h["line_id"]) for h in first["hits"]}
    b = {(h["page"], h["line_id"]) for h in second["hits"]}
    assert b and a.isdisjoint(b)


def _page_through(tool, args, max_calls=200):
    """Follow ``next`` to the end; every page must hold hits, and the walk
    must end."""
    seen, offset, calls = [], 0, 0
    while True:
        calls += 1
        assert calls <= max_calls, "digest_search never stopped paging"
        out = _call(tool, dict(args, offset=offset))
        assert out["hits"], (offset, out)       # next never points at nothing
        seen += [(h.get("document"), h["page"], h["line_id"])
                 for h in out["hits"]]
        if "next" not in out:
            return seen, out
        assert out["next"]["offset"] > offset
        offset = out["next"]["offset"]


def test_search_pages_past_200_hits_and_stops():
    """Review fix 1: the tool asked the digest for offset + limit hits and
    the digest capped that at 200, so page 5 came back empty with a ``next``
    that never ended."""
    tools = _tools({"manual.pdf": long_pdf(150)})
    seen, last = _page_through(tools["digest_search"],
                               {"query": "allowable bearing", "limit": 60})
    assert len(seen) > 200 and len(set(seen)) == len(seen)
    assert len(seen) == last["total"]
    # past the end: no hits and no next
    out = _call(tools["digest_search"], {"query": "allowable bearing",
                                         "offset": last["total"] + 5})
    assert out["hits"] == [] and "next" not in out


def test_search_pages_past_200_across_documents():
    manual = long_pdf(150)
    tools = _tools({"a.pdf": manual, "b.pdf": manual})
    seen, last = _page_through(tools["digest_search"],
                               {"query": "allowable bearing", "limit": 60})
    assert len(seen) > 400 and len(set(seen)) == len(seen)
    assert {d for d, _p, _l in seen} == {"a.pdf", "b.pdf"}
    assert len(seen) == last["total"]


def test_search_pages_through_a_538_page_manual():
    ufc = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "geotech-references", "docs", "ufc_3_260_02_2001.pdf")
    if not os.path.isfile(ufc):
        pytest.skip("UFC 3-260-02 is not in this checkout")
    tools = _tools({"ufc.pdf": os.path.abspath(ufc)})
    seen, last = _page_through(tools["digest_search"],
                               {"query": "pavement", "limit": 60})
    assert len(seen) > 200 and len(set(seen)) == len(seen)
    assert len(seen) == last["total"]
    assert max(p for _d, p, _l in seen) > 400      # into the appendices


def test_digests_go_to_the_working_folder_the_tools_were_built_with(
        uploads, tmp_path, monkeypatch):
    """Review fix 6: bound at build time, not read at call time."""
    mine = tmp_path / "conversation_a"
    tools = {t.name: t for t in DT.make_digest_tools(
        {"report.pdf": uploads["report.pdf"]}, working_dir=str(mine))}
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "other"))
    assert _call(tools["document_inventory"])["totals"]["documents"] == 1
    _call(tools["digest_search"], {"query": "boring"})
    assert os.listdir(mine / "digest")
    assert not (tmp_path / "other").exists()


# ---------------------------------------------------------------------------
# digest_pages and digest_references
# ---------------------------------------------------------------------------

def test_pages_rows_and_their_cursor():
    tools = _tools({"manual.pdf": long_pdf(150)})
    out = _call(tools["digest_pages"], {"source": "manual.pdf"})
    assert out["pages_in_document"] == 150
    rows = out["rows"]
    assert rows[0]["page"] == 0 and rows[0]["pdf_page"] == 1
    assert rows[0]["printed"] == "1-1" and rows[0]["heading"] == "Table 1-1"
    assert "MANUAL 1-2-3" not in rows[0]["excerpt"]      # running header
    assert out["left_out"] == 150 - len(rows)
    nxt = _call(tools["digest_pages"], {"source": "manual.pdf",
                                        "pages": out["next"]["pages"]})
    assert nxt["rows"][0]["page"] == len(rows)
    one = _call(tools["digest_pages"], {"source": "manual.pdf",
                                        "pages": "20-21"})
    assert [r["page"] for r in one["rows"]] == [20, 21]
    assert "next" not in one
    bad = _call(tools["digest_pages"], {"source": "manual.pdf",
                                        "pages": "150"})
    assert "out of range" in bad["error"] and "0 to 149" in bad["hint"]
    missing = _call(tools["digest_pages"], {"source": "nope.pdf"})
    assert "error" in missing and missing.get("hint")


def test_references_by_target_and_overall(uploads):
    from funhouse_agent.review_digest.tests.test_review_digest_offline \
        import typed_set_pdf
    tools = _tools({"set.pdf": typed_set_pdf(),
                    "report.pdf": uploads["report.pdf"]})
    out = _call(tools["digest_references"], {"target": "10.17"})
    assert [(o["document"], o["page"], o["pdf_page"]) for o in
            out["occurrences"]] == [("set.pdf", 1, 2)]
    apx = _call(tools["digest_references"], {"target": "Appendix B",
                                             "source": "report.pdf"})
    first = apx["occurrences"][0]
    assert first["role"] == "caption" and first["pdf_page"] == 12
    overall = _call(tools["digest_references"])
    groups = overall["references"]
    flagged = [g for g in groups if g.get("not_uploaded")]
    # C-102 too: this time its one-sheet upload is not among the documents
    assert {g["id"] for g in flagged} == {"3/S-501", "10.35B", "11.51",
                                          "840.54", "30.19", "C-102"}
    assert groups[:len(flagged)] == flagged              # missing first
    fig = next(g for g in groups if g["kind"] == "figure" and g["id"] == "1")
    assert fig["caption"]["pdf_page"] == 6
    nothing = _call(tools["digest_references"], {"target": "Z-999"})
    assert nothing["occurrences"] == [] and "note" in nothing


def test_references_page_on_with_next():
    tools = _tools({"manual.pdf": long_pdf(150)})
    out = _call(tools["digest_references"])
    assert out["left_out"] > 0
    more = _call(tools["digest_references"], {"offset": out["next"]["offset"]})
    a = {(g["kind"], g["id"]) for g in out["references"]}
    b = {(g["kind"], g["id"]) for g in more["references"]}
    assert b and a.isdisjoint(b)
    tables = _call(tools["digest_references"], {"target": "Table 3-"})
    caps = [o for o in tables["occurrences"] if o["role"] == "caption"]
    assert [c["id"] for c in caps] == [f"3-{n}" for n in range(1, 11)]
    assert caps[0]["title"] == "Allowable bearing values for zone 20"


# ---------------------------------------------------------------------------
# The lean agent uses it
# ---------------------------------------------------------------------------

def test_the_lean_agent_takes_stock_end_to_end(uploads, monkeypatch,
                                                tmp_path):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.DIGEST_ENV, "1")
    model = _model([
        AIMessage(content="", tool_calls=[{"name": "document_inventory",
                                           "args": {}, "id": "c1"}]),
        AIMessage(content="Three documents, 38 pages."),
    ])
    agent = build_deep_agent(model, attachments=dict(uploads),
                             **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "take stock"}]})
    result = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    data = json.loads(result.content)
    assert data["totals"]["pages"] == 38
    assert os.path.isdir(tmp_path / "work" / "digest")
