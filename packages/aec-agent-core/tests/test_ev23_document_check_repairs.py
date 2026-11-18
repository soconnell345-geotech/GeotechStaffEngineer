"""BACKLOG EV-23: repairs of the cited-documents check.

The check lists a document an answer cites that nothing in the conversation
looked up. It fired where it should not in these ways, each repaired here
for every model and every text, never for one example:

* J: a tool result was searched as raw JSON text, so an escaped newline
  stood against the next word and a designation after it was not found
  (in a document's text and in a references result alike). The result is
  now read as the text it holds. A standard published in parts is the
  family of its parts, as EN 1997 already was.
* P: a pointer ("verify it with X"), a denial ("I have not relied on X")
  or a condition ("if the specification calls for X, ...") was read as a
  citation. It is now kept in the record as a mention, never flagged,
  unless its clause states a value.
* K: a surname with a one-letter particle ("O'Hara") was read without it
  in the answer and whole in the sources.
* L: brackets holding a pair of plain numbers after a capitalised word
  were read as an author and a year.
* M: a calculation package's own method basis, read back from the file the
  package wrote, was no source.

Beside each case that must stop firing stands one that must still fire.
With ``GEOTECH_CITED_DOCUMENTS`` unset (the default) what the model and the
user are handed does not change; only the turn's recorded check does. All
documents, names, values and answers here are made up; no model is called.
"""

from __future__ import annotations

import json

import pytest

from aec_agent_core import cited_documents as cd
from aec_agent_core import hooks, review_flags


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(review_flags.CITED_DOCUMENTS_ENV, raising=False)
    with hooks._LOCK:
        saved = (dict(hooks._HOOKS), dict(hooks._RESOLVED))
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
    cd.clear_cache()
    yield
    cd.clear_cache()
    with hooks._LOCK:
        hooks._HOOKS.clear()
        hooks._HOOKS.update(saved[0])
        hooks._RESOLVED.clear()
        hooks._RESOLVED.update(saved[1])


PROMPT = "Review the drainage layout of the made-up yard and size the drains."


def _conv(tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    return conv


def _turn(conv, calls, prompt=PROMPT):
    """One turn as the activity log records it: a result is kept as the
    JSON text of what the tool returned, as the app's logger keeps it."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"turn": 1, "event": "turn_start",
                             "agent": "primary", "prompt": prompt}) + "\n")
        for i, call in enumerate(calls):
            rid = f"r1_{i}_{call['name']}"
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "agent": call.get("agent", "primary"),
                                 "name": call["name"], "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            result = call["result"]
            fh.write(json.dumps({
                "turn": 1, "event": "tool_end",
                "agent": call.get("agent", "primary"), "name": call["name"],
                "run_id": rid,
                "result": result if isinstance(result, str)
                else json.dumps(result)}) + "\n")
    return conv


def _check(conv, answer):
    return cd.check_text(answer, record_folder=str(conv), user_words=[PROMPT])


def _flagged(res):
    return [r["cited"] for r in res.get("not_looked_up") or []]


def _mentioned(res):
    return [(r["mentioned"], r["as"]) for r in res.get(cd.MENTIONED) or []]


# ---------------------------------------------------------------------------
# J: a result is read as the text it holds
# ---------------------------------------------------------------------------

LAB_SHEET = {"name": "read_document", "args": {"source": "yard_report.pdf"},
             "result": {"pages": [{"page": 7, "text":
                                   "LIQUID LIMIT\nASTM D9999\nSample S-1\n"
                                   "LL = 41"}]}}


def test_a_designation_after_a_newline_in_a_document_read_is_looked_up(
        tmp_path):
    assert "\\nASTM D9999" in json.dumps(LAB_SHEET["result"])  # as logged
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, "The lab sheet gives LL = 41 by ASTM D9999.")
    assert _flagged(res) == []
    assert res["looked_up"] == [{"cited": "ASTM D9999",
                                 "document": "astm-d9999",
                                 "by": "a document opened"}]


def test_a_standard_no_result_names_still_fires(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, "The plastic limit was run to ASTM D9998: PL = 22.")
    assert _flagged(res) == ["ASTM D9998"]


def test_a_references_result_hid_its_designations_the_same_way(tmp_path):
    conv = _turn(_conv(tmp_path), [
        {"name": "consult_references",
         "args": {"question": "drain spacing in a made-up yard"},
         "result": {"answer": "Spacing limits:\nUFC 3-999-01 Table 2-1 "
                              "gives 3.0 m at most."}}])
    res = _check(conv, "UFC 3-999-01 Table 2-1 gives 3.0 m at most.")
    assert _flagged(res) == []
    assert res["looked_up"][0]["by"] == "a references result"
    res = _check(conv, "GEC-15 gives 2.5 m for the same drains.")
    assert _flagged(res) == ["GEC-15"]


def test_an_escaped_letter_is_read_as_the_letter(tmp_path):
    conv = _turn(_conv(tmp_path), [
        {"name": "consult_references", "args": {"question": "drain filter"},
         "result": {"answer": "Filter rule after Kovač (1988): D15 below "
                              "five times d85."}}])
    res = _check(conv, "Kovač (1988) gives the filter rule used here.")
    assert _flagged(res) == []
    assert res["looked_up"] == [{"cited": "Kovač (1988)",
                                 "by": "a references result"}]
    res = _check(conv, "Lind (2004) gives a looser rule.")
    assert _flagged(res) == ["Lind (2004)"]


def test_one_records_fields_still_make_an_author_and_year(tmp_path):
    conv = _turn(_conv(tmp_path), [
        {"name": "consult_references", "args": {"question": "beta"},
         "result": {"sources": [{"author": "Lind", "year": 2004,
                                 "title": "Made-up shaft friction"}]}}])
    res = _check(conv, "Beta follows Lind (2004).")
    assert res.get("looked_up") == [{"cited": "Lind (2004)",
                                     "by": "a references result"}]


def test_a_standard_in_parts_is_looked_up_by_a_part():
    assert cd.covered("iso-77777", ["iso-77777-1"])
    assert cd.covered("iso-77777-2", ["iso-77777"])
    assert cd.covered("bs-4444", ["bs-4444-3"])
    assert not cd.covered("iso-77778", ["iso-77777-1"])
    assert not cd.covered("iso-77777-2", ["iso-77777-1"])
    assert not cd.covered("ufc-3-999-01", ["ufc-3-999"])


def test_a_part_read_in_a_document_covers_the_whole_standard(tmp_path):
    conv = _turn(_conv(tmp_path), [
        {"name": "read_document", "args": {"source": "yard_report.pdf"},
         "result": {"text": "GRADING\n:\nISO 77777-1:2019, made-up "
                            "classification by grading"}}])
    assert _flagged(_check(conv, "The lab classified by ISO 77777.")) == []
    assert _flagged(_check(conv, "The lab classified by ISO 77778.")) == \
        ["ISO 77778"]


# ---------------------------------------------------------------------------
# P: a pointer, a denial or a condition is a mention, not a citation
# ---------------------------------------------------------------------------

MENTIONS = [
    ("Verify the drain spacing with UFC 3-999-01 before the design is "
     "issued.", "UFC 3-999-01", cd.POINTER),
    ("The drain factor should be compared against GEC-15 for this layer.",
     "GEC-15", cd.POINTER),
    ("Before issue, check the wall drainage and, ideally, consult GEC-15.",
     "GEC-15", cd.POINTER),
    ("The outlet was sized by the tool, so confirm the grade against "
     "UFC 3-999-01 or a comparable manual.", "UFC 3-999-01", cd.POINTER),
    ("I recommend confirming the outlet grade against FHWA-NHI-99-001.",
     "FHWA-NHI-99-001", cd.POINTER),
    ("Use a current manual such as GEC-15 for the final design.", "GEC-15",
     cd.POINTER),
    ("For the final design, cross-check with a second method (e.g. the "
     "GEC-15 approach).", "GEC-15", cd.POINTER),
    ("I have not relied on ASTM D9997 here.", "ASTM D9997", cd.DENIAL),
    ("We did not find any GEC-15 table for this case.", "GEC-15",
     cd.DENIAL),
    ("GEC-15 was not used for the outlet.", "GEC-15", cd.DENIAL),
    ("If the specification calls for ASTM D9998, I can add its limits.",
     "ASTM D9998", cd.CONDITION),
]


@pytest.mark.parametrize("answer,shown,kind", MENTIONS)
def test_a_mention_is_kept_on_the_record_and_not_flagged(tmp_path, answer,
                                                         shown, kind):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, answer)
    assert _flagged(res) == []
    assert res["n_cited"] == 0
    assert _mentioned(res) == [(shown, kind)]
    assert res[cd.MENTIONED][0]["looked_up"] is False


CITATIONS = [
    ("Per UFC 3-999-01, the spacing is 3.0 m.", "UFC 3-999-01"),
    ("UFC 3-999-01 gives a spacing of 3.0 m.", "UFC 3-999-01"),
    ("Use 0.29 from GEC-15 for the clay.", "GEC-15"),
    ("Use the GEC-15 drain factor of 0.29.", "GEC-15"),
    ("Check the spacing: 3 m per UFC 3-999-01.", "UFC 3-999-01"),
    ("The values were checked against GEC-15 Table 7-1 and agree.",
     "GEC-15"),
    ("We confirm the spacing against UFC 3-999-01.", "UFC 3-999-01"),
    ("Check the result, as GEC-15 recommends.", "GEC-15"),
    ("Confirm the cover against UFC 3-999-01, which asks for more near the "
     "sea.", "UFC 3-999-01"),
    ("The drains follow the GEC-15 layout (see GEC-15 Fig. 4-2).", "GEC-15"),
    ("Hale & Ward (1971) report efficiencies near 1.0 for such groups.",
     "Hale & Ward (1971)"),
    ("The increment follows the Lind (2004) convention.", "Lind (2004)"),
    ("If the yard drains to the east, GEC-15 sets the outlet at 0.5 m.",
     "GEC-15"),
    ("I did not look up the grades. The spacing follows GEC-15.", "GEC-15"),
]


@pytest.mark.parametrize("answer,shown", CITATIONS)
def test_a_citation_nothing_looked_up_still_fires(tmp_path, answer, shown):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, answer)
    assert _flagged(res) == [shown]
    assert not res.get(cd.MENTIONED)


def test_a_document_cited_after_a_pointer_to_it_fires(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    for answer in ("Verify the spacing with UFC 3-999-01. UFC 3-999-01 "
                   "gives 3.0 m.",
                   "UFC 3-999-01 gives 3.0 m. Verify it with UFC 3-999-01 "
                   "on site."):
        res = _check(conv, answer)
        assert _flagged(res) == ["UFC 3-999-01"]
        assert not res.get(cd.MENTIONED)


def test_a_pointer_to_a_document_looked_up_says_so(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, "Confirm the limit against ASTM D9999 on the sheet.")
    assert res[cd.MENTIONED] == [{"mentioned": "ASTM D9999",
                                  "as": cd.POINTER,
                                  "document": "astm-d9999",
                                  "looked_up": True}]


def _turn_check(conv, mode):
    logged = []
    check = cd.TurnCitationCheck(record_folder=str(conv),
                                 user_words=[PROMPT], log=logged.append,
                                 mode=mode)
    return check, logged


POINTER_ONLY = "Verify the drain spacing with UFC 3-999-01 before issue."
CITING = "UFC 3-999-01 gives a drain spacing of 3.0 m."


def test_with_the_switch_unset_nothing_handed_out_changes(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    check, logged = _turn_check(conv, None)
    assert check.mode == "off"
    for answer in (POINTER_ONLY, CITING):
        assert check.feedback(answer) is None
        assert check.finish(answer) == answer
    # the record keeps the mention under its own key, and the citation
    assert logged[0]["n_cited"] == 0 and logged[0]["action"] == "none"
    assert _mentioned(logged[0]) == [("UFC 3-999-01", cd.POINTER)]
    assert _flagged(logged[1]) == ["UFC 3-999-01"]


@pytest.mark.parametrize("mode", ["note", "retry"])
def test_with_the_switch_on_a_mention_gets_no_note(tmp_path, mode):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    check, logged = _turn_check(conv, mode)
    assert check.feedback(POINTER_ONLY) is None
    assert check.finish(POINTER_ONLY) == POINTER_ONLY
    assert logged[-1]["action"] == "none"
    check, _ = _turn_check(conv, mode)
    if mode == "retry":
        assert "- UFC 3-999-01" in check.feedback(CITING)
    assert "UFC 3-999-01" in check.finish(CITING).split("---")[-1]


def test_a_writers_mention_is_recorded_but_adds_nothing(tmp_path,
                                                        monkeypatch):
    from aec_agent_core.cited_pages import record_bound
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    seen = []
    monkeypatch.setattr(cd, "_event", seen.append)
    with record_bound(str(conv)):
        add = cd.check_deliverable([POINTER_ONLY], "write_docx", mode="note")
    assert add == {}
    assert _mentioned(seen[-1]) == [("UFC 3-999-01", cd.POINTER)]


# ---------------------------------------------------------------------------
# K: a surname with a particle; L: a pair of numbers
# ---------------------------------------------------------------------------

def test_a_particle_surname_is_read_whole_by_the_check(tmp_path):
    conv = _turn(_conv(tmp_path), [
        {"name": "describe_method",
         "args": {"agent_name": "made_up_shaft", "method": "side"},
         "result": {"brief": "beta by O'Hara and Lind (2004), 0.25 to 1.2"}}])
    for answer in ("O'Hara & Lind (2004) give beta = 0.8 near the top.",
                   "O’Hara & Lind (2004) give beta = 0.8 near the top.",
                   "Beta is 0.8 near the top (O'Hara and Lind 2004)."):
        res = _check(conv, answer)
        assert _flagged(res) == [], answer
        assert res["looked_up"][0]["by"] == "a calculation's own result"


def test_a_particle_surname_nothing_names_still_fires(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, "D'Amato (1997) gives beta = 0.9 near the top.")
    assert _flagged(res) == ["D'Amato (1997)"]


def test_a_pair_of_numbers_is_no_citation(tmp_path):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    res = _check(conv, "The cloud spans Corner (1710, 1342) to (2210, 1490).")
    assert res == {"checked": True, "n_cited": 0}


@pytest.mark.parametrize("answer,shown", [
    ("Hale (1979, 1985) gives the drain rule.", "Hale (1979)"),
    ("Lind (2004, p. 12) gives the drain rule.", "Lind (2004)"),
    ("Lind (2004) gives 0.6.", "Lind (2004)"),
])
def test_a_work_with_a_page_or_a_second_year_still_fires(tmp_path, answer,
                                                         shown):
    conv = _turn(_conv(tmp_path), [LAB_SHEET])
    assert _flagged(_check(conv, answer)) == [shown]


def test_the_plain_reading_is_unchanged_for_the_judgment_check():
    """``find_citations`` without ``refined`` reads as before EV-23: the
    judgment-call check asks it whether a work is cited, and its note to
    the model must not move."""
    def works(text, **kw):
        return [(c.shown, c.author) for c in cd.find_citations(text, **kw)
                if c.kind == "work"]
    assert works("O'Hara & Lind (2004) give beta.") == \
        [("Hara & Lind (2004)", "hara")]
    assert works("Beta (O'Hara and Lind 2004).") == []
    assert works("Corner (1710, 1342) to (2210, 1490).") == \
        [("Corner (1710)", "corner")]
    assert works("O'Hara & Lind (2004) give beta.", refined=True) == \
        [("O'Hara & Lind (2004)", "o'hara")]
    assert works("Beta (O'Hara and Lind 2004).", refined=True) == \
        [("O'Hara and Lind (2004)", "o'hara")]
    assert works("Corner (1710, 1342) to (2210, 1490).", refined=True) == []


# ---------------------------------------------------------------------------
# M: what a calculation package wrote from the app's own template
# ---------------------------------------------------------------------------

def _package(params=None, method="footing_package"):
    return {"name": "call_agent",
            "args": {"agent_name": "calc_package", "method": method,
                     "parameters": {"width": 2.0, "output_path":
                                    "footing_calc.pdf", **(params or {})}},
            "result": {"status": "success", "output_path": "footing_calc.pdf",
                       "file_exists": True}}


def _read_back(text, name="footing_calc.pdf"):
    return {"name": "read_pdf_text", "args": {"source": name},
            "result": {"pages": [{"page": 0, "text": text}]}}


PACKAGE_TEXT = ("Bearing factor\nN = 23.2\nMade-up rule; FHWA-NHI-99-001 "
                "Eq. 4-2")


def test_a_packages_own_basis_read_back_is_a_calculation(tmp_path):
    conv = _turn(_conv(tmp_path), [_package(), _read_back(PACKAGE_TEXT)])
    res = _check(conv, "The factor follows the FHWA-NHI-99-001 equations.")
    assert _flagged(res) == []
    assert res["looked_up"][0]["by"] == "a calculation's own result"


def test_the_models_own_pdf_read_back_still_fires(tmp_path):
    conv = _turn(_conv(tmp_path), [
        _package({"html": "<p>Made-up rule; FHWA-NHI-99-001 Eq. 4-2</p>"},
                 method="html_to_pdf"),
        _read_back(PACKAGE_TEXT)])
    res = _check(conv, "The factor follows the FHWA-NHI-99-001 equations.")
    assert _flagged(res) == ["FHWA-NHI-99-001"]


@pytest.mark.parametrize("params,printed", [
    ({"notes": "Basis GEC-15 as instructed by the brief"},
     "Basis GEC-15\nas instructed by the\nbrief"),
    ({"basis": "GEC-15"}, "Basis: GEC-15"),
])
def test_what_the_model_gave_a_package_is_not_its_basis(tmp_path, params,
                                                        printed):
    conv = _turn(_conv(tmp_path), [_package(params),
                                   _read_back(PACKAGE_TEXT + "\n" + printed)])
    res = _check(conv, "The basis is GEC-15 and FHWA-NHI-99-001.")
    assert _flagged(res) == ["GEC-15"]


def test_a_file_the_model_wrote_over_the_package_is_no_source(tmp_path):
    conv = _turn(_conv(tmp_path), [
        _package(),
        {"name": "save_file", "args": {"path": "footing_calc.pdf",
                                       "content": "..."},
         "result": {"saved_path": "footing_calc.pdf"}},
        _read_back(PACKAGE_TEXT)])
    res = _check(conv, "The factor follows the FHWA-NHI-99-001 equations.")
    assert _flagged(res) == ["FHWA-NHI-99-001"]


def test_a_packages_own_result_is_a_calculation(tmp_path):
    assert cd.source_kind("call_agent", {"agent_name": "calc_package",
                                         "method": "html_to_pdf"}) is None
    assert cd.source_kind("call_agent", {"agent_name": "calc_package",
                                         "method": "footing_package"}) == \
        cd.CALCULATION
    call = _package()
    call["result"] = {**call["result"],
                      "basis": "FHWA-NHI-99-002 Eq. 3-1"}
    conv = _turn(_conv(tmp_path), [call])
    res = _check(conv, "The settlement follows FHWA-NHI-99-002.")
    assert _flagged(res) == []
