"""Cited documents checked against what a conversation looked up (EV-21).

Live wave w8b (``module_work/live_smoke/runs/w8b-plot-code/TRACE_REVIEW.md``
§4.2): the main agent told the calc helper to "cite Rankine (1857) and
GEC-11 / AASHTO LRFD 3.11.5"; the user named neither, no references tool ran,
the PDF printed the citation "as given by the project" and the answer said
"as you stated them". These tests drive the general check with synthetic
records: the citation finder and its names, the record of what was looked
up, the w8b case rebuilt as a fixture, the writers' results, the answer
check and the turn object.
"""

import json
import os

import pytest

from funhouse_agent import cited_documents as cd
from funhouse_agent import review_flags


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    cd.clear_cache()
    yield
    cd.clear_cache()


def _found(text):
    return [(c.shown, c.key) for c in cd.find_citations(text)]


def _keys(text):
    return [c.key for c in cd.find_citations(text)]


# ---------------------------------------------------------------------------
# Finding citations, and naming them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "GEC-11", "GEC 11", "GEC11", "FHWA GEC-11", "GEC No. 11",
    "gec 11", "Geotechnical Engineering Circular No. 11",
    "FHWA-NHI-10-024", "FHWA NHI-10-025", "NHI-10-024",
])
def test_one_document_by_any_of_its_names(text):
    assert _keys(f"see {text} for MSE walls") == ["gec-11"]


@pytest.mark.parametrize("text,shown,key", [
    ("AASHTO LRFD 3.11.5", "AASHTO LRFD 3.11.5", "aashto-lrfd"),
    ("AASHTO LRFD", "AASHTO LRFD", "aashto-lrfd"),
    ("AASHTO LRFD Bridge Design Specifications, Article 10.6.3.1",
     "AASHTO LRFD Art. 10.6.3.1", "aashto-lrfd"),
    ("AASHTO (2020) Article 3.11.5.1", "AASHTO LRFD Art. 3.11.5.1",
     "aashto-lrfd"),
    ("AASHTO 3.11.5", "AASHTO LRFD 3.11.5", "aashto-lrfd"),
    ("AASHTO 1993", "AASHTO 1993", "aashto-1993"),
    ("the AASHTO Guide for Design of Pavement Structures", "AASHTO 1993",
     "aashto-1993"),
    ("AASHTO T 99", "AASHTO T 99", "aashto-t99"),
    ("UFC 3-220-10N", "UFC 3-220-10N", "ufc-3-220-10"),
    ("UFS 3-220-10", "UFS 3-220-10", "ufc-3-220-10"),
    ("NAVFAC DM 7.01", "NAVFAC DM 7.01", "ufc-3-220-10"),
    ("DM7.2", "NAVFAC DM 7.02", "ufc-3-220-20"),
    ("DM7 §6-7.4", "NAVFAC DM 7 §6-7.4", "dm7"),
    ("EM 1110-2-1902", "EM 1110-2-1902", "em-1110-2-1902"),
    ("TM 5-818-1", "TM 5-818-1", "tm-5-818-1"),
    ("ASTM D1586-11", "ASTM D1586", "astm-d1586"),
    ("ASTM D 4318", "ASTM D4318", "astm-d4318"),
    ("ACI 318-19", "ACI 318", "aci-318"),
    ("ASCE 7-22.", "ASCE 7", "asce-7"),
    ("the 2021 IBC", "IBC", "ibc"),
    ("AISC 360", "AISC 360", "aisc-360"),
    ("API RP 2GEO", "API RP 2GEO", "api-2geo"),
    ("Caltrans Trenching and Shoring Manual", "Caltrans Trenching and "
     "Shoring Manual", "caltrans-trenching-shoring"),
    ("Caltrans Example 8-1", "Caltrans", "caltrans"),
    ("NCHRP Report 611", "NCHRP Report 611", "nchrp-611"),
    ("CGPR #56", "CGPR 56", "cgpr-56"),
    ("EN 1997-1:2004", "EN 1997-1", "en-1997-1"),
    ("Eurocode 7", "EN 1997", "en-1997"),
    ("EC7", "EN 1997", "en-1997"),
    ("FEMA P-2082", "FEMA P-2082", "fema-p-2082"),
    ("ISO 22476-3", "ISO 22476-3", "iso-22476-3"),
    ("BS 5930", "BS 5930", "bs-5930"),
    ("PTI DC35.1", "PTI DC35.1", "pti-dc35.1"),
    ("GSA Alternate Path guidelines", "GSA Alternate Path",
     "gsa-alternate-path"),
    ("the USDA Wood Handbook", "Wood Handbook", "wood-handbook"),
])
def test_designations(text, shown, key):
    assert _found(text) == [(shown, key)]


@pytest.mark.parametrize("text,shown,key", [
    ("Rankine (1857)", "Rankine (1857)", "work:rankine:1857"),
    ("Duncan & Wright (2014)", "Duncan & Wright (2014)", "work:duncan:2014"),
    ("Duncan and Wright (2014, p. 12)", "Duncan and Wright (2014)",
     "work:duncan:2014"),
    ("Seed et al. (1985)", "Seed et al. (1985)", "work:seed:1985"),
    ("Terzaghi's (1943) theory", "Terzaghi (1943)", "work:terzaghi:1943"),
    ("(Bowles, 1996, p. 512)", "Bowles (1996)", "work:bowles:1996"),
    ("(e.g., Rankine 1857; see below)", "Rankine (1857)",
     "work:rankine:1857"),
    ("FHWA (2010)", "FHWA (2010)", "work:fhwa:2010"),
    ("Vesić (1975)", "Vesić (1975)", "work:vesic:1975"),
])
def test_author_and_year(text, shown, key):
    assert _found(text) == [(shown, key)]


def test_the_w8b_pdf_line_and_answer():
    pdf = ("Ka = tan2(45 - phi'/2) = 0.3073 (Rankine 1857; see GEC-11 / "
           "AASHTO LRFD 3.11.5 - section numbers as given by the project, "
           "not independently verified).")
    assert _found(pdf) == [("Rankine (1857)", "work:rankine:1857"),
                           ("GEC-11", "gec-11"),
                           ("AASHTO LRFD 3.11.5", "aashto-lrfd")]
    answer = ("The Rankine, GEC-11 and AASHTO LRFD 3.11.5 citations in the "
              "PDF are as you stated them.")
    assert _keys(answer) == ["gec-11", "aashto-lrfd"]


def test_html_is_read_as_its_text():
    html = ("<html><style>p{font:9pt}</style><p>Ka per <b>AASHTO&nbsp;LRFD"
            "</b> &sect;3.11.5.1 and GEC&#8209;11</p></html>")
    assert _found(html) == [("AASHTO LRFD §3.11.5.1", "aashto-lrfd"),
                            ("GEC-11", "gec-11")]


def test_one_entry_per_document():
    assert _keys("GEC-11 (FHWA-NHI-10-024) ... later GEC 11 again") == \
        ["gec-11"]


@pytest.mark.parametrize("text", [
    "the boring was drilled in March (2021)",
    "Boring B-1 (2019) and Table 4 (2014)",
    "a 40-page report; Load 2000 kN; DM 7.5 kg of sample",
    "FEMA 2020 flood maps",
    "EC 2.5 dS/m in the pore water",
    "the AASHTO soil classification",
    "a Caltrans project",
    "In 2014 the wall was built; Rankine 1857 without brackets",
    "Smith Engineering (2019)",
    "J. SMFD ASCE 99(SM1); Journal of the Geotechnical Division, ASCE 103",
])
def test_words_that_are_not_citations(text):
    assert _found(text) == []


def test_every_reference_module_names_its_own_documents():
    """The names come from the app's own: each reference module's brief
    names a document the module serves (``MODULE_DOCUMENTS``)."""
    from funhouse_agent.adapters import MODULE_REGISTRY
    from funhouse_agent.dispatch import REFERENCE_MODULES
    for module in sorted(REFERENCE_MODULES - {"reference_db", "figure_db"}):
        served = cd.module_documents(module)
        assert served, module
        brief = MODULE_REGISTRY[module]["brief"]
        named = set(_keys(brief))
        assert any(cd.covered(k, served) for k in named), (module, brief)


def test_a_chapter_file_reference_id_is_its_gec():
    pkg = pytest.importorskip("geotech_references.gec_11")
    path = os.path.join(os.path.dirname(pkg.__file__), "text",
                        "chapter08.json")
    if not os.path.isfile(path):
        pytest.skip("no GEC-11 volume II chapter here")
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)["reference_id"]
    assert _keys(ref) == ["gec-11"]


def test_families():
    assert cd.covered("dm7", ["ufc-3-220-20"])
    assert cd.covered("ufc-3-220-10", ["dm7"])
    assert cd.covered("en-1997", ["en-1997-1"])
    assert not cd.covered("gec-11", ["gec-12", "aashto-lrfd"])


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def _conv(tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    return conv


def _write_events(conv, calls, turn=1, prompt=None):
    """A turn as the activity log records it."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        if prompt is not None:
            fh.write(json.dumps({"turn": turn, "event": "turn_start",
                                 "agent": "primary",
                                 "prompt": prompt[:500]}) + "\n")
        for i, call in enumerate(calls):
            rid = call.get("run_id") or f"r{turn}_{i}_{call['name']}"
            fh.write(json.dumps({"turn": turn, "event": "tool_start",
                                 "agent": call.get("agent", "primary"),
                                 "name": call["name"], "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            result = call.get("result", {})
            fh.write(json.dumps({
                "turn": turn, "event": "tool_end",
                "agent": call.get("agent", "primary"), "name": call["name"],
                "run_id": rid,
                "result": result if isinstance(result, str)
                else json.dumps(result)}) + "\n")


USER = ("Make a short calc PDF for the report on our cantilever retaining "
        "wall: retained height 4.0 m, stem 0.35 m thick, base 2.8 m wide... "
        "Use Rankine active pressure. Include a figure of the wall drawn to "
        "scale with the active earth pressure and the surcharge pressure "
        "drawn beside it.")

BRIEF = ("Produce a SHORT calc PDF ... Cite Rankine (1857) and GEC-11 / "
         "AASHTO LRFD 3.11.5.")

HTML = ("<html><body><h1>Cantilever Retaining Wall - Rankine Active Earth "
        "Pressure</h1><h2>2. Method</h2><p>Ka = tan<sup>2</sup>(45 - phi'/2)"
        " = 0.3073 (Rankine 1857; see GEC-11 / AASHTO LRFD 3.11.5 - section "
        "numbers as given by the project, not independently verified).</p>"
        "</body></html>")

#: The w8b turn rebuilt (F55, events [1]-[112]): the brief that made the
#: citation up, the catalog naming GEC-11 for MSE walls, the calculation
#: (Rankine), the figure looked at, the hand-back, the PDF and its read-back.
W8B = [
    {"name": "task", "args": {"subagent_type": "calc", "description": BRIEF},
     "result": "The citations Rankine (1857), GEC-11 and AASHTO LRFD 3.11.5 "
               "come from your request."},
    {"name": "list_methods", "agent": "calc",
     "args": {"agent_name": "retaining_walls"},
     "result": {"Earth Pressure": {"earth_pressure_coefficient":
                                   "Lateral earth pressure coefficient "
                                   "Ka/Kp/K0 (Rankine, Coulomb, or "
                                   "Caquot-Kerisel log-spiral passive Kp)."},
                "MSE Wall": {"mse_wall": "MSE wall external + internal "
                                         "stability analysis (GEC-11).",
                             "mse_lrfd_external_stability":
                                 "MSE wall external stability by AASHTO/"
                                 "GEC-11 LRFD."}}},
    {"name": "describe_method", "agent": "calc",
     "args": {"agent_name": "retaining_walls",
              "method": "earth_pressure_coefficient"},
     "result": {"brief": "Lateral earth pressure coefficient Ka/Kp/K0 "
                         "(Rankine, Coulomb, or Caquot-Kerisel (1948) "
                         "log-spiral passive Kp)."}},
    {"name": "call_agent", "agent": "calc",
     "args": {"agent_name": "retaining_walls",
              "method": "earth_pressure_coefficient",
              "parameters": {"phi_deg": 32, "theory": "rankine",
                             "state": "active"}},
     "result": {"K": 0.3073, "coefficient": "Ka", "theory": "rankine",
                "state": "active", "phi_deg": 32}},
    {"name": "plot_code", "agent": "calc", "args": {"code": "..."},
     "result": {"images": [{"file": "cantilever_rankine_pressures.png",
                            "output_path": "cantilever_rankine_pressures.png"
                            }]}},
    {"name": "analyze_image", "agent": "calc",
     "args": {"attachment_key": "cantilever_rankine_pressures.png"},
     "result": {"analysis": "The title reads 'Rankine active pressure per "
                            "GEC-11 / AASHTO LRFD 3.11.5'."}},
    {"name": "call_agent", "agent": "calc",
     "args": {"agent_name": "calc_package", "method": "html_to_pdf",
              "parameters": {"output_path": "cantilever_rankine_calc.pdf",
                             "html": HTML}},
     "result": {"status": "success",
                "output_path": "cantilever_rankine_calc.pdf",
                "file_exists": True}},
    {"name": "read_pdf_text", "agent": "calc",
     "args": {"source": "cantilever_rankine_calc.pdf"},
     "result": {"pages": [{"page": 0, "text": "(Rankine 1857; see GEC-11 / "
                                              "AASHTO LRFD 3.11.5)"}]}},
]

ANSWER = ("Ka = 0.3073 (Rankine). The Rankine, GEC-11 and AASHTO LRFD "
          "3.11.5 citations in the PDF are as you stated them.")


def _w8b(tmp_path, extra=(), prompt=USER):
    conv = _conv(tmp_path)
    _write_events(conv, W8B, prompt=prompt)
    if extra:
        _write_events(conv, list(extra), turn=1)
    return conv


def _not_looked_up(res):
    return [r["cited"] for r in res.get("not_looked_up") or []]


def test_w8b_answer_flags_gec11_and_aashto(tmp_path):
    conv = _w8b(tmp_path)
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert res["checked"] is True
    assert _not_looked_up(res) == ["GEC-11", "AASHTO LRFD 3.11.5"]


def test_w8b_pdf_flags_gec11_and_aashto_not_rankine(tmp_path):
    conv = _w8b(tmp_path)
    res = cd.check_text(HTML, record_folder=str(conv), user_words=[USER])
    assert _not_looked_up(res) == ["GEC-11", "AASHTO LRFD 3.11.5"]
    # The user asked for Rankine pressure and the module ran it: the work's
    # author is in the record, only the year is the model's.
    assert res["year_not_checked"] == [{"cited": "Rankine (1857)",
                                        "named_by": "your messages"}]


def test_the_files_the_conversation_wrote_are_not_sources(tmp_path):
    conv = _w8b(tmp_path)
    st_path = os.path.join(str(conv), "activity.jsonl")
    cd.looked_up(str(conv))
    st = cd._state_for(st_path)
    assert {"cantilever_rankine_calc.pdf",
            "cantilever_rankine_pressures.png"} <= st.outputs
    # the catalog naming GEC-11 for MSE walls, the brief and the hand-back
    # do not make it citable either
    assert st.marks.document_kind("gec-11") is None


def test_a_references_result_naming_gec11_is_looked_up(tmp_path):
    conv = _w8b(tmp_path, extra=[
        {"name": "task", "args": {"subagent_type": "references",
                                  "description": "Rankine Ka for a "
                                                 "cantilever wall"},
         "result": "GEC-11 Section 4.3 treats lateral earth pressure on MSE "
                   "walls; for cantilever walls see the earth-pressure "
                   "chapter of UFC 3-220-20."}])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert _not_looked_up(res) == ["AASHTO LRFD 3.11.5"]
    assert {"cited": "GEC-11", "document": "gec-11",
            "by": "a references result"} in res["looked_up"]


def test_a_reference_module_call_looks_up_its_document(tmp_path):
    """``call_agent`` on ``gec11`` looks GEC-11 up even when the result
    prints only numbers."""
    conv = _w8b(tmp_path, extra=[
        {"name": "call_agent", "agent": "references",
         "args": {"agent_name": "gec11",
                  "method": "table_3_6_pullout_parameters",
                  "parameters": {}},
         "result": {"Ci": 0.67, "F_star": 1.2}}])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert "GEC-11" not in _not_looked_up(res)


def test_the_users_own_words_are_never_flagged(tmp_path):
    conv = _w8b(tmp_path)
    said = USER + " Cite GEC-11 and AASHTO LRFD as the basis."
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[said])
    assert _not_looked_up(res) == []
    assert [r["cited"] for r in res["user_named"]] == [
        "GEC-11", "AASHTO LRFD 3.11.5"]


def test_the_users_words_from_an_earlier_turn_count(tmp_path):
    conv = _w8b(tmp_path)
    with open(conv / "transcript.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user",
                             "text": "We design to AASHTO LRFD."}) + "\n")
        fh.write(json.dumps({"role": "assistant",
                             "text": "GEC-11 says so."}) + "\n")
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert _not_looked_up(res) == ["GEC-11"]       # the answer's, not theirs


def test_a_document_the_conversation_opened_makes_its_citations_citable(
        tmp_path):
    conv = _w8b(tmp_path, extra=[
        {"name": "open_document", "args": {"source": "geotech_report.pdf"},
         "result": {"handle": "doc_r", "name": "geotech_report.pdf"}},
        {"name": "read_document", "args": {"handle": "doc_r"},
         "result": {"pages_returned": "0-3",
                    "text": "Walls were designed per AASHTO LRFD Bridge "
                            "Design Specifications (2020)."}}])
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert _not_looked_up(res) == ["GEC-11"]


def test_a_calculations_result_names_its_basis_but_not_its_echo(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, [
        {"name": "call_agent", "args": {
            "agent_name": "downdrag", "method": "analyze",
            "parameters": {"project_name": "Pier 3 per FHWA GEC-12 here"}},
         "result": {"method": "UFC 3-220-20 neutral plane",
                    "project_name": "Pier 3 per FHWA GEC-12 here"}}],
        prompt="Downdrag on pier 3")
    res = cd.check_text("Basis: UFC 3-220-20 and GEC-12.",
                        record_folder=str(conv), user_words=[])
    assert _not_looked_up(res) == ["GEC-12"]


def test_a_worked_example_names_its_source(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, [
        {"name": "call_agent", "args": {"agent_name": "worked_examples",
                                        "method": "get_worked_example",
                                        "parameters": {"example_id": "x"}},
         "result": {"source": "FHWA GEC-11 Vol 2 (FHWA-NHI-10-025), "
                              "Appendix E, Example E4"}}], prompt="MSE wall")
    res = cd.check_text("Per GEC-11 Example E4.", record_folder=str(conv),
                        user_words=[])
    assert _not_looked_up(res) == []
    assert res["looked_up"][0]["by"] == "a worked example"


def test_works_by_author_and_year(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, [
        {"name": "task", "args": {"subagent_type": "references"},
         "result": "Reference list: Duncan, J. M., and Wright, S. G. (2005). "
                   "Soil Strength and Slope Stability. Wiley."},
        {"name": "list_methods", "args": {"agent_name": "settlement"},
         "result": {"schmertmann_settlement":
                    "Schmertmann (1978) strain influence factor method."}}],
        prompt="Check the slope")
    res = cd.check_text(
        "Per Duncan & Wright (2005); see Schmertmann (1978) and Kulhawy & "
        "Mayne (1990).", record_folder=str(conv), user_words=[])
    assert _not_looked_up(res) == ["Kulhawy & Mayne (1990)"]
    assert res["looked_up"] == [{"cited": "Duncan & Wright (2005)",
                                 "by": "a references result"}]
    assert res["year_not_checked"] == [{"cited": "Schmertmann (1978)",
                                        "named_by": "the module catalog"}]


def test_no_record_no_check(tmp_path):
    res = cd.check_text(ANSWER, record_folder=str(tmp_path), user_words=[])
    assert res["checked"] is False and "no activity log" in res["why"]


def test_the_record_is_read_incrementally(tmp_path):
    conv = _w8b(tmp_path)
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert "GEC-11" in _not_looked_up(res)
    _write_events(conv, [{"name": "task",
                          "args": {"subagent_type": "references"},
                          "result": "GEC-11 Chapter 4."}], turn=2)
    res = cd.check_text(ANSWER, record_folder=str(conv), user_words=[USER])
    assert "GEC-11" not in _not_looked_up(res)


# ---------------------------------------------------------------------------
# The writers
# ---------------------------------------------------------------------------

def _html_to_pdf(conv, html=HTML):
    pytest.importorskip("fitz")
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.dispatch import call_agent
    with working_dir_bound(str(conv / "files")), \
            cd.user_words_bound([USER]):
        return call_agent("calc_package", "html_to_pdf",
                          {"output_path": "rebuilt_calc.pdf", "html": html})


def test_html_to_pdf_with_the_switch_on_says_what_nothing_looked_up(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _w8b(tmp_path)
    res = _html_to_pdf(conv)
    assert res.get("file_exists") is True, res      # never refused
    assert res[cd.RESULT_LIST] == ["GEC-11", "AASHTO LRFD 3.11.5"]
    note = res[cd.RESULT_NOTE]
    assert note.startswith("cited but never looked up in this "
                           "conversation: GEC-11, AASHTO LRFD 3.11.5")
    assert "Rankine" not in note


def test_html_to_pdf_with_the_switch_off_is_unchanged(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(cd, "_event", seen.append)
    conv = _w8b(tmp_path)
    res = _html_to_pdf(conv)
    assert res.get("file_exists") is True
    assert cd.RESULT_LIST not in res and cd.RESULT_NOTE not in res
    # ... and recorded (inside a turn the activity log keeps it)
    assert seen and seen[-1]["mode"] == "off" and seen[-1]["action"] == "none"
    assert [r["cited"] for r in seen[-1]["not_looked_up"]] == [
        "GEC-11", "AASHTO LRFD 3.11.5"]
    assert seen[-1]["where"] == "calc_package.html_to_pdf"


def test_html_to_pdf_after_the_references_says_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    conv = _w8b(tmp_path, extra=[
        {"name": "task", "args": {"subagent_type": "references"},
         "result": "GEC-11 Ch. 4 and AASHTO LRFD Article 3.11.5 cover it."}])
    res = _html_to_pdf(conv)
    assert res.get("file_exists") is True
    assert cd.RESULT_LIST not in res


def test_a_writer_on_a_host_with_no_record_is_unchanged(tmp_path,
                                                         monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    bare = tmp_path / "bare"
    bare.mkdir()
    res = _html_to_pdf_in(bare)
    assert res.get("file_exists") is True and cd.RESULT_LIST not in res


def _html_to_pdf_in(folder):
    pytest.importorskip("fitz")
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.dispatch import call_agent
    with working_dir_bound(str(folder)):
        return call_agent("calc_package", "html_to_pdf",
                          {"output_path": "x.pdf", "html": HTML})


def _extended(conv, tool, arguments):
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.vision_tools import dispatch_extended_tool
    with working_dir_bound(str(conv / "files")), \
            cd.user_words_bound([USER]):
        return json.loads(dispatch_extended_tool(tool, arguments, None, {}))


def test_write_docx_gets_the_line(tmp_path, monkeypatch):
    pytest.importorskip("docx")
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _w8b(tmp_path)
    res = _extended(conv, "write_docx", {
        "path": "memo.docx", "title": "Wall memo",
        "markdown": "# Basis\n\nKa per GEC-11 and ASTM D1586."})
    assert "error" not in res, res
    assert res[cd.RESULT_LIST] == ["GEC-11", "ASTM D1586"]


def test_save_file_checks_reports_not_data(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _w8b(tmp_path)
    md = _extended(conv, "save_file", {"path": "notes.md",
                                       "content": "Per GEC-11, Ka = 0.31."})
    assert md[cd.RESULT_LIST] == ["GEC-11"]
    csv = _extended(conv, "save_file", {"path": "data.csv",
                                        "content": "source,GEC-11\n"})
    assert cd.RESULT_LIST not in csv


def test_annotate_document_notes_are_checked(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    from funhouse_agent import document_tools
    conv = _w8b(tmp_path)
    monkeypatch.setattr(document_tools, "_write_marked_copy",
                        lambda *a, **k: {"output_path": "plan_marked.pdf",
                                         "n_written": 1})
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(conv / "files")), \
            cd.user_words_bound([USER]):
        res = document_tools.write_marked_copy(
            "doc_x", [{"page": 0, "text": "Embedment short of GEC-11 "
                                          "Table 2-1 minimum.",
                       "quote": "AASHTO LRFD 11.10"}],
            "", True, "tester", lambda args: "{}")
    assert res[cd.RESULT_LIST] == ["GEC-11"]          # the quote is not read


def test_deliverable_texts_leave_out_paths_and_anchors():
    got = cd.deliverable_texts({"output_path": "GEC-11.pdf",
                                "html": "<p>x</p>",
                                "markups": [{"quote": "q", "text": "t"}]})
    assert got == ["<p>x</p>", "t"]


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------

def _turn(tmp_path, mode):
    conv = _w8b(tmp_path)
    logged = []
    check = cd.TurnCitationCheck(record_folder=str(conv), user_words=[USER],
                                 log=logged.append, mode=mode)
    return check, logged


GOOD = "Ka = 0.3073 by Rankine theory, from the module."


def test_off_records_and_leaves_the_answer(tmp_path):
    check, logged = _turn(tmp_path, "off")
    assert check.feedback(ANSWER) is None
    assert check.finish(ANSWER) == ANSWER
    assert logged[-1]["action"] == "none" and logged[-1]["mode"] == "off"
    assert logged[-1]["where"] == "answer"
    assert _not_looked_up(logged[-1]) == ["GEC-11", "AASHTO LRFD 3.11.5"]


def test_note_appends_the_note(tmp_path):
    check, logged = _turn(tmp_path, "note")
    out = check.finish(ANSWER)
    assert out.startswith(ANSWER) and "Citation check" in out
    assert "GEC-11, AASHTO LRFD 3.11.5" in out
    assert logged[-1]["action"] == "noted"
    assert check.finish(GOOD) == GOOD


def test_retry_tells_the_model_once(tmp_path):
    check, logged = _turn(tmp_path, "retry")
    note = check.feedback(ANSWER)
    assert note and note.startswith(cd.NOTE_PREFIX)
    assert "- GEC-11" in note and "the user's own words" in note
    assert logged[-1]["action"] == "fed_back"
    assert check.feedback(ANSWER) is None            # once a turn
    assert "Citation check" in check.finish(ANSWER)  # still cites them
    assert check.finish(GOOD) == GOOD


def test_answers_citing_nothing_are_not_recorded(tmp_path):
    check, logged = _turn(tmp_path, "note")
    assert check.finish("No document is cited.") == "No document is cited."
    assert logged == []


def test_a_turn_with_no_record_folder_claims_nothing():
    check = cd.TurnCitationCheck(record_folder=None, user_words=[],
                                 mode="note")
    assert check.finish(ANSWER) == ANSWER


def test_user_texts_of_messages():
    from types import SimpleNamespace
    msgs = [{"role": "user", "content": "one"},
            {"role": "assistant", "content": "GEC-11"},
            {"role": "user", "content": [{"type": "text", "text": "two"},
                                         {"type": "image_url"}]},
            SimpleNamespace(type="human", content="three"),
            SimpleNamespace(type="ai", content="four")]
    assert cd.user_texts(msgs) == ["one", "two", "three"]


def test_the_switch():
    assert review_flags.cited_documents() == "off"
    for value, mode in (("note", "note"), ("1", "note"), ("on", "note"),
                        ("retry", "retry"), ("RETRY", "retry"),
                        ("0", "off"), ("nonsense", "off")):
        with review_flags.switches({review_flags.CITED_DOCUMENTS_ENV: value}):
            assert review_flags.cited_documents() == mode
    assert review_flags.ARMS["cited_documents_note"] == {
        review_flags.CITED_DOCUMENTS_ENV: "note"}
    assert review_flags.ARMS["cited_documents_retry"] == {
        review_flags.CITED_DOCUMENTS_ENV: "retry"}
    assert review_flags.CITED_DOCUMENTS_ENV in review_flags.ALL_ENVS
