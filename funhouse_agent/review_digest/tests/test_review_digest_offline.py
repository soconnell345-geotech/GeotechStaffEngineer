"""The document digest's free layer - offline, no model, no network.

Built on planlens' synthetic fixtures (a stapled submittal with segments,
printed page numbers and a duplicate; a review set with markups, a /Rotate 90
sheet, hidden CAD text and a scan; a report with captions and appendix tabs),
a small synthetic drawing set typed with the references a set cites, and -
when the public documents are in this checkout - the Mecklenburg set and the
538-page UFC 3-260-02.
"""

import importlib
import json
import os
import sqlite3
import time

import pytest

pytest.importorskip("planlens.document")
fitz = pytest.importorskip("fitz")

from funhouse_agent.review_digest import (  # noqa: E402
    FORMAT_VERSION, DigestError, build, clear_cache, default_root,
    find_references, inventory_of, label_matches, parse_pages)
from funhouse_agent.review_digest import digest as DG  # noqa: E402
# The module, not the package's build() function of the same name.
B = importlib.import_module("funhouse_agent.review_digest.build")
from planlens.testing import (  # noqa: E402
    build_synthetic_report, build_synthetic_review_document,
    build_synthetic_submittal)

UFC_260 = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                       "geotech-references", "docs", "ufc_3_260_02_2001.pdf")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(B.SHAPE1_PAGES_ENV, raising=False)
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def submittal():
    return build_synthetic_submittal()


@pytest.fixture(scope="module")
def review():
    return build_synthetic_review_document()


@pytest.fixture(scope="module")
def report():
    return build_synthetic_report()


# ---------------------------------------------------------------------------
# A synthetic drawing set that cites sheets and standards
# ---------------------------------------------------------------------------

def _sheet(doc, label, notes, label_word="STD. NO."):
    """A landscape sheet with a title block (label word, number under it)
    and typed notes."""
    p = doc.new_page(width=792, height=612)
    p.draw_rect(fitz.Rect(20, 20, 772, 592))
    p.draw_rect(fitz.Rect(640, 520, 772, 592))
    p.insert_text((660, 540), label_word, fontsize=7)
    p.insert_text((660, 560), label, fontsize=12)
    y = 80
    for note in notes:
        p.insert_text((60, y), note, fontsize=8)
        y += 22
    return p


def typed_set_pdf():
    doc = fitz.open()
    _sheet(doc, "10.17A", ["1'-6\" STANDARD CURB AND GUTTER",
                           "SEE DETAIL 3/S-501 FOR THE JOINT."])
    _sheet(doc, "11.01", [
        "1. SEE STD. NO. 10.17 FOR CURB.",
        "2. DETECTABLE WARNING MAT PER MCLDS #10.35B.",
        "3. PAVEMENT TRANSITION PER DETAIL 11.51.",
        "4. INLET PER NCDOT STD. 840.54.",
        "5. SILT FENCE PER STD #30.19.",
        "6. USE #4 BARS @ 12\" O.C. WITH 3/4\" CHAMFER.",
        "SHEET 2 OF 3"])
    _sheet(doc, "C-101", ["REFER TO SHEET C-102 FOR GRADING.",
                          "SEE SHEET C-101 FOR NOTES.",
                          "CONCRETE PER SECTION 03 30 00."],
           label_word="SHEET NO.")
    data = doc.tobytes()
    doc.close()
    return data


def one_sheet_pdf(text="GRADING PLAN"):
    doc = fitz.open()
    p = doc.new_page(width=792, height=612)
    p.insert_text((60, 80), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# Layout and rows
# ---------------------------------------------------------------------------

def test_the_digest_folder_and_its_four_files(submittal, tmp_path):
    d = build(submittal.pdf, name="submittal.pdf", root=str(tmp_path))
    sha = B.sha256_of(submittal.pdf)
    assert d.folder == os.path.join(str(tmp_path), sha[:16])
    assert sorted(os.listdir(d.folder)) == ["index.sqlite", "inventory.json",
                                            "pages.jsonl", "references.json"]
    inv = json.load(open(os.path.join(d.folder, "inventory.json"),
                         encoding="utf-8"))
    assert inv["format"] == FORMAT_VERSION and inv["sha256"] == sha
    assert inv["name"] == "submittal.pdf" and inv["pages"] == 11
    assert inv["kinds"] == {"text": 4, "mixed": 3, "form": 2,
                            "drawing_sheet": 2}
    # the forms and the drawing sheets are the pages to look at
    assert inv["needs_look"] == {"pages": "5-6,8-9", "pdf_pages": "6-7,9-10",
                                 "n": 4}
    rows = [json.loads(line) for line in open(
        os.path.join(d.folder, "pages.jsonl"), encoding="utf-8")]
    assert [r["page"] for r in rows] == list(range(11))
    assert all(r["pdf_page"] == r["page"] + 1 for r in rows)
    with sqlite3.connect(os.path.join(d.folder, "index.sqlite")) as con:
        n = con.execute("SELECT count(*) FROM lines").fetchone()[0]
        assert n == inv["indexed_lines"] > 0
        assert con.execute("SELECT value FROM meta WHERE key='fts'"
                           ).fetchone()[0] in ("0", "1")


def test_segments_printed_pages_duplicate_and_rows(submittal, tmp_path):
    d = build(submittal.pdf, name="submittal.pdf", root=str(tmp_path))
    inv = d.inventory()
    got = {s["pages"]: s["title"] for s in inv["segments"]}
    assert list(got) == [f"{a}-{b}" if b > a else str(a)
                         for a, b, _t in submittal.expected_segments]
    for first, last, fragment in submittal.expected_segments:
        if first in (0, 1, 4, 10):         # titles the pages print
            key = f"{first}-{last}" if last > first else str(first)
            assert fragment.lower() in got[key].lower(), (key, got[key])
    report_seg = next(s for s in inv["segments"] if s["pages"] == "1-3")
    assert report_seg["printed"] == "1 to 3"
    assert report_seg["pdf_pages"] == "2-4"
    rows = d.pages()
    dup = rows[submittal.duplicate_page]
    assert dup["duplicate_of"] == submittal.duplicate_of
    assert rows[1]["printed"] == "1" and rows[2]["printed"] == "2"
    # a running header is not a heading; the page's own heading is
    assert rows[1]["heading"] == "1. Subsurface Conditions"
    assert "Acme Geotechnical" not in rows[1].get("excerpt", "")
    assert len(rows[1]["excerpt"]) <= B.EXCERPT_CHARS
    assert rows[5]["needs_look"] and rows[5]["kind"] == "form"
    assert rows[8]["needs_look"] and rows[8]["kind"] == "drawing_sheet"
    assert not rows[1]["needs_look"] and rows[1]["text_reliable"]
    # "SHEET 1 OF 2" is a count, not a sheet label
    assert inv["sheet_labels"] == []
    divider = rows[submittal.appendix_divider_page]
    assert "appendix A (caption)" in divider["references"]


def test_markups_rotation_hidden_text_and_scan(review, tmp_path):
    d = build(review.pdf, name="review_set.pdf", root=str(tmp_path))
    inv = d.inventory()
    assert inv["markups"] == {"n": 5, "authors": {"Contractor B": 3,
                                                  "Reviewer A": 2}}
    assert inv["no_text_layer"]["pages"] == str(review.scanned_page)
    assert inv["needs_look"]["pages"] == f"{review.sheet_page}," \
                                         f"{review.scanned_page}"
    sheet = d.page(review.sheet_page)
    assert sheet["sheet"] == review.sheet_label == "S-1"
    assert sheet["markups"] == 4 and sheet["hidden_cad_text"] == 1
    scan = d.page(review.scanned_page)
    assert scan["kind"] == "scanned" and scan["no_text_layer"]
    assert [e["title"] for e in inv["outline"]] == [t for _l, t, _p in
                                                    review.toc]
    # text on the /Rotate 90 sheet is indexed in the DISPLAYED frame:
    # upright text planted at baseline (300, 400)
    hit = d.search(review.sheet_text_upright)["hits"][0]
    assert hit["page"] == review.sheet_page and hit["pdf_page"] == 2
    x0, y0, x1, y1 = hit["bbox"]
    assert abs(x0 - 300) < 3 and y0 < 400 < y1 + 3
    # hidden CAD text and markup text are searchable, and say so
    cad = d.search(review.hidden_cad_text)["hits"][0]
    assert cad["source"] == "cad_hidden_text"
    mk = d.search("pile embedment")["hits"][0]
    assert mk["source"] == "markup" and mk["author"] == review.reviewer
    assert mk["page"] == review.sheet_page


def test_report_captions_and_appendix_tabs(report, tmp_path):
    d = build(report.pdf, name="report.pdf", root=str(tmp_path))
    caps = [(r["kind"], r["id"], r["page"]) for r in d.references()
            if r["role"] == "caption"]
    assert ("figure", "1", report.caption_page) in caps
    for kind, number, title, _printed, page in report.outline_entries:
        if kind == "appendix":
            assert ("appendix", number, page) in caps, number
    # the contents page LISTS them; it does not caption them
    listed = [r for r in d.references() if r["role"] == "listed"]
    assert {r["page"] for r in listed} == {1}
    got = d.references(target="Appendix B")
    assert got[0]["role"] == "caption" and got[0]["page"] == 11
    assert got[0]["title"].upper().startswith("LABORATORY TEST DATA")
    roles = d.inventory()["roles"]
    assert roles.get("divider") and roles.get("narrative")


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------

def test_a_digest_is_reused_by_content_and_version(submittal, tmp_path,
                                                   monkeypatch):
    root = str(tmp_path)
    first = build(submittal.pdf, name="a.pdf", root=root)
    calls = []
    real = B._read

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(B, "_read", counting)
    # same bytes under another name, and from a path: no second read
    again = build(submittal.pdf, name="renamed.pdf", root=root)
    path = tmp_path / "copy.pdf"
    path.write_bytes(submittal.pdf)
    clear_cache()
    from_path = build(str(path), root=root)
    assert calls == [] and again.folder == first.folder == from_path.folder
    # an older format is rebuilt
    inv_path = os.path.join(first.folder, "inventory.json")
    inv = json.load(open(inv_path, encoding="utf-8"))
    inv["format"] = FORMAT_VERSION - 1
    json.dump(inv, open(inv_path, "w", encoding="utf-8"))
    clear_cache()
    rebuilt = build(submittal.pdf, root=root)
    assert calls == [1] and rebuilt.inventory()["format"] == FORMAT_VERSION
    # a half-written digest (no index) is rebuilt too
    os.remove(os.path.join(first.folder, "index.sqlite"))
    clear_cache()
    build(submittal.pdf, root=root)
    assert calls == [1, 1]
    build(submittal.pdf, root=root, force=True)
    assert calls == [1, 1, 1]
    assert not [n for n in os.listdir(root) if n.endswith(".tmp")]


def test_the_default_root_is_the_working_folder(submittal, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "conv"))
    assert default_root() == os.path.abspath(str(tmp_path / "conv" /
                                                 "digest"))
    d = build(submittal.pdf)
    assert d.folder.startswith(str(tmp_path / "conv" / "digest"))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

HOSTILE = ['"', '""', "AND", "OR NOT", "NEAR(pile embedment, 2)", "*",
           "pile*", "text:pile", "(((", "'; DROP TABLE lines; --", "%", "_",
           "%%%", "\\", "-borings", "^", "{a b}", "éè", "   "]


@pytest.mark.parametrize("query", HOSTILE)
def test_search_never_raises_on_fts_syntax(review, tmp_path, query):
    d = build(review.pdf, name="r.pdf", root=str(tmp_path))
    out = d.search(query)
    assert isinstance(out["hits"], list)
    json.dumps(out)
    # the index is intact afterwards
    assert d.search("pile embedment")["hits"]


def test_search_ranks_exact_wording_first_and_pages_on(submittal, tmp_path):
    d = build(submittal.pdf, name="s.pdf", root=str(tmp_path))
    out = d.search("standard penetration tests", limit=2)
    assert out["method"] == "fts5"
    assert [h["match"] for h in out["hits"]] == ["exact", "exact"]
    seen = {(h["page"], h["line_id"]) for h in out["hits"]}
    nxt = d.search("standard penetration tests", limit=2,
                   offset=out["next"]["offset"])
    assert seen.isdisjoint({(h["page"], h["line_id"]) for h in nxt["hits"]})
    # all the words, not the phrase: ranked after the exact wording
    words = d.search("borings groundwater", limit=5)
    assert words["hits"] and all(h["match"] == "all_words"
                                 for h in words["hits"])
    # pages and kinds narrow it
    assert {h["page"] for h in d.search("penetration", pages="2-3",
                                        limit=50)["hits"]} <= {2, 3}
    assert d.search("Depth", kinds=["form"], limit=50)["hits"]
    assert not d.search("Depth", kinds=["drawing_sheet"])["hits"]
    with pytest.raises(IndexError):
        d.search("x", pages="40")


def test_search_pages_are_one_ranking_with_a_total(submittal, tmp_path):
    """Review fix 1: offset pages through ONE ranking (no page overlaps
    another), ``total`` says how many were ranked, and ``next`` is given
    only while a later page holds hits."""
    d = build(submittal.pdf, name="s.pdf", root=str(tmp_path))
    everything = d.search("penetration", limit=500)
    total = everything["total"]
    assert total == len(everything["hits"]) >= 3 and "next" not in everything
    walked, offset = [], 0
    while True:
        out = d.search("penetration", limit=2, offset=offset)
        assert out["hits"] and out["total"] == total
        walked += [(h["page"], h["line_id"]) for h in out["hits"]]
        if "next" not in out:
            break
        offset = out["next"]["offset"]
    assert walked == [(h["page"], h["line_id"]) for h in everything["hits"]]
    past = d.search("penetration", limit=2, offset=total)
    assert past["hits"] == [] and "next" not in past


def test_search_pages_past_200_on_a_long_manual(tmp_path):
    if not os.path.isfile(UFC_260):
        pytest.skip("UFC 3-260-02 is not in this checkout")
    d = build(UFC_260, root=str(tmp_path))
    first = d.search("pavement", limit=60)
    assert first["total"] > 260 and first["next"] == {"offset": 60}
    deep = d.search("pavement", limit=60, offset=200)
    assert len(deep["hits"]) == 60
    seen = set()
    for off in range(0, first["total"], 60):
        out = d.search("pavement", limit=60, offset=off)
        keys = {(h["page"], h["line_id"]) for h in out["hits"]}
        assert keys and seen.isdisjoint(keys)
        seen |= keys
        assert ("next" in out) == (off + 60 < first["total"])
    assert len(seen) == first["total"]


def test_the_open_digests_and_build_locks_do_not_grow_without_bound(
        tmp_path, monkeypatch):
    """Review fix 14: a long-lived host sees many conversations' uploads."""
    import gc
    for i in range(DG.CACHE_SIZE + 5):
        build(one_sheet_pdf(f"SHEET {i}"), name=f"s{i}.pdf",
              root=str(tmp_path))
    assert DG.cache_size() == DG.CACHE_SIZE
    gc.collect()
    assert len(B._BUILD_LOCKS) == 0          # none held, none kept
    monkeypatch.setattr(B, "PATH_SHA_SIZE", 3)
    for i in range(6):
        p = tmp_path / f"f{i}.pdf"
        p.write_bytes(one_sheet_pdf(f"FILE {i}"))
        B.sha256_of(str(p))
    assert len(B._PATH_SHA) == 3
    # the newest are the ones kept, and a kept one is still right
    p = tmp_path / "f5.pdf"
    assert B.sha256_of(str(p)) == B.sha256_of(p.read_bytes())


def test_search_falls_back_to_like(review, tmp_path, monkeypatch):
    d = build(review.pdf, name="r.pdf", root=str(tmp_path))

    def refuse(*a, **kw):
        raise sqlite3.OperationalError("fts5: syntax error")

    monkeypatch.setattr(DG.Digest, "_fts", staticmethod(refuse))
    out = d.search("pile embedment")
    assert out["method"] == "like" and out["hits"][0]["source"] == "markup"


def test_a_digest_without_fts5_searches_with_like(review, tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(B, "_FTS", "CREATE VIRTUAL TABLE lines_fts USING "
                                   "no_such_module(text)")
    d = build(review.pdf, name="r.pdf", root=str(tmp_path))
    assert d.inventory()["search"] == "like"
    assert d.search("GENERAL NOTE")["hits"][0]["page"] == review.sheet_page


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

POSITIVE = [
    ("SEE DETAIL 3/S-501 FOR REBAR", ("detail", "3/S-501", "S-501")),
    ("3/S-501", ("detail", "3/S-501", "S-501")),
    ("SEE DETAIL A/C-3.1", ("detail", "A/C-3.1", "C-3.1")),
    ("REFER TO SHEET S-501", ("sheet", "S-501", "S-501")),
    ("DWG. NO. C-101", ("sheet", "C-101", "C-101")),
    ("SEE STD. NO. 10.17", ("standard", "10.17", "10.17")),
    ("MAT PER MCLDS #10.35B", ("standard", "10.35B", "10.35B")),
    ("INLET PER NCDOT STD. 840.54", ("standard", "840.54", "840.54")),
    ("RIPRAP PER NCESCPDM #6.60", ("standard", "6.60", "6.60")),
    ("SEE DETAIL 11.51", ("detail", "11.51", "11.51")),
    ("TRANSITION PER DETAIL 11.51.", ("detail", "11.51", "11.51")),
    ("See Appendix 3.", ("appendix", "3", None)),
    ("SECTION 03 30 00 CAST-IN-PLACE CONCRETE",
     ("spec_section", "03 30 00", None)),
    ("Concrete shall conform to 03 30 00.", ("spec_section", "03 30 00",
                                             None)),
    ("per 31 23 16 EXCAVATION", ("spec_section", "31 23 16", None)),
    ("See Section 02300", ("spec_section", "02300", None)),
    ("Section 12.2 - STRUCTURAL SYSTEM SELECTION", ("section", "12.2", None)),
    ("as shown in Table 5-1", ("table", "5-1", None)),
    ("see Fig. 3", ("figure", "3", None)),
    ("Figure 1-1 illustrates", ("figure", "1-1", None)),
    ("see Appendix B", ("appendix", "B", None)),
    ("Attachment C - Photographs", ("attachment", "C", None)),
    ("Table 4-1(a) Structural Performance Objectives",
     ("table", "4-1(a)", None)),
]

NEGATIVE = [
    "SHEET 1 OF 12", "Table of Contents", "TABLE OF CONTENTS",
    "USE #4 BARS @ 12\" O.C.", "(2) #5 BARS TOP AND BOTTOM",
    "the standard 2 inch pipe", "standard penetration test N = 15", "5-1",
    "see 5-1 and 3-2", "Page 5-2", "3/4\" CHAMFER", "1/2 IN. GAP", "N/A",
    "and/or", "10/20/2024", "F/A-18 aircraft", "APPENDIX TO THE REPORT",
    "DETAIL AS SHOWN", "12 18 24 30", "10 20 30 FEET", "sheet 5 mm steel",
    "standard 1.5 in thick", "in more detail than", "Std. 0.12 of the set",
    "tablespoon", "SHEET PILE WALL", "a 5-1 slope",
]


@pytest.mark.parametrize("text,want", POSITIVE, ids=[p[0] for p in POSITIVE])
def test_reference_patterns_find_what_is_printed(text, want):
    kind, ident, target = want
    refs = find_references(text)
    assert [(r["kind"], r["id"], r.get("target")) for r in refs][:1] == \
        [(kind, ident, target)], refs


@pytest.mark.parametrize("text", NEGATIVE)
def test_reference_patterns_leave_look_alikes_alone(text):
    assert find_references(text) == []


def test_lists_and_the_agency_standard_forms():
    ids = [r["id"] for r in find_references("from Tables 8-3, 8-4, or 8-5.")]
    assert ids == ["8-3", "8-4", "8-5"]
    ids = [r["id"] for r in find_references("Appendices A through N")]
    assert ids == ["A", "N"]
    ids = [r["id"] for r in find_references("REFER TO SHEETS S-502, S-503")]
    assert ids == ["S-502", "S-503"]
    # "Table 5-1 and 50 kN" does not make 50 a table
    assert [r["id"] for r in find_references("Table 5-1 and 50 kN")] == \
        ["5-1"]
    # a whole standard document is a reference but not a sheet of a set
    (mil,) = find_references("prescribed by MIL-STD 3007")
    assert mil["kind"] == "standard" and "target" not in mil


def test_captions_versus_references():
    (cap,) = find_references("Table 12-1")
    assert cap["role"] == "caption"
    (cap,) = find_references("Figure 1-1. Examples of cracks in a wall")
    assert cap["role"] == "caption" and cap["title"].startswith("Examples")
    assert find_references("Table 4-1 presents the gear loads")[0]["role"] \
        == "ref"
    assert find_references("Table 3-1, Replacement for ASCE 7-22 Table "
                           "12.2-1, must be used in lieu of")[0]["role"] \
        == "ref"
    # a sentence that wrapped onto a line starting "Table 12-3." is a ref
    text = "coverages listed in Table 12-3.  Divide the passes"
    (ref,) = find_references(text, [0, text.index("Table")])
    assert ref["role"] == "ref"
    listed = find_references("Table 5-1 Recommended compaction ...... 5-3")
    assert listed[0]["role"] == "listed"
    assert listed[0]["title"] == "Recommended compaction"


def test_the_letter_suffix_rule():
    assert label_matches("10.17", ["10.17A", "10.31A"]) == "10.17A"
    assert label_matches("S-501", ["S501"]) == "S501"
    assert label_matches("s 501", ["S-501"]) == "S-501"
    # a sheet B is not satisfied by the standard's other number
    assert label_matches("10.35B", ["10.35"]) is None
    assert label_matches("10.1", ["10.17"]) is None
    assert label_matches("S-50", ["S-501"]) is None
    assert label_matches("10.17", ["10.17AB"]) is None


def test_missing_references_across_uploads(tmp_path):
    sources = {"set.pdf": typed_set_pdf(), "C-102.pdf": one_sheet_pdf()}
    inv = inventory_of(sources, root=str(tmp_path))
    set_doc = next(d for d in inv["documents"] if d["document"] == "set.pdf")
    labels = [s["sheet"] for s in set_doc["sheet_labels"]]
    assert labels == ["10.17A", "11.01", "C-101"]
    x = inv["cross_references"]
    missing = {m["id"] for m in x["missing"]}
    assert missing == {"S-501", "10.35B", "11.51", "840.54", "30.19"}
    found = {f["id"]: f for f in x["found"]}
    # 10.17 is in the set as its lettered sheet 10.17A
    assert found["10.17"]["sheet"] == "10.17A" and found["10.17"]["by_suffix"]
    # C-102 is its own one-sheet upload, named by its sheet number
    assert found["C-102"]["document"] == "C-102.pdf"
    assert not found["C-102"]["by_suffix"]
    # the sheet's own number in its title block is not a reference to it
    assert "C-101" not in found and "C-101" not in missing
    cited = next(m for m in x["missing"] if m["id"] == "10.35B")["cited"][0]
    assert (cited["page"], cited["pdf_page"]) == (1, 2)
    assert inv["shape_hint"] == "small" and inv["totals"]["pages"] == 4


def test_the_mecklenburg_set(tmp_path):
    """No text layer on these sheets: the one reference that can be read is
    on 10.31A's hidden CAD text (MCLDS #10.35B, not in the set)."""
    from funhouse_agent.review_eval import documents as RD
    try:
        name, data = RD.resolve("meck_set")
    except RD.MissingDocument:
        pytest.skip("the Mecklenburg sheets are not in this checkout")
    inv = inventory_of({name: data}, root=str(tmp_path))
    (doc,) = inv["documents"]
    assert doc["pages"] == 10 and doc["no_text_layer"]["n"] == 10
    assert doc["needs_look"]["n"] == 10
    assert doc["sheet_labels"] == [{"sheet": "10.31A", "page": 2,
                                    "pdf_page": 3}]
    x = inv["cross_references"]
    assert [m["id"] for m in x["missing"]] == ["10.35B"]
    assert x["missing"][0]["cited"][0]["pdf_page"] == 3


# ---------------------------------------------------------------------------
# Robustness, shape, pages
# ---------------------------------------------------------------------------

def test_an_odd_page_is_recorded_not_raised(submittal, tmp_path,
                                            monkeypatch):
    from planlens.document import Document
    real = Document.summary

    def broken(self, index):
        if index == 3:
            raise RuntimeError("damaged content stream")
        return real(self, index)

    monkeypatch.setattr(Document, "summary", broken)
    d = build(submittal.pdf, name="s.pdf", root=str(tmp_path))
    row = d.page(3)
    assert row["kind"] == "unread" and row["needs_look"]
    assert "damaged content stream" in row["unread"]
    inv = d.inventory()
    assert inv["unread_pages"]["pages"] == "3"
    assert inv["unread_pages"]["pdf_pages"] == "4"
    assert d.page(2)["kind"] == "text"


def test_not_a_document_is_an_error_and_does_not_stop_the_rest(
        submittal, tmp_path):
    with pytest.raises(DigestError):
        build(b"just some text", name="notes.txt", root=str(tmp_path))
    inv = inventory_of({"notes.txt": b"just some text",
                        "s.pdf": submittal.pdf}, root=str(tmp_path))
    assert [d["document"] for d in inv["documents"]] == ["s.pdf"]
    assert inv["errors"][0]["document"] == "notes.txt"


def test_shape_hint_follows_the_setting(submittal, tmp_path, monkeypatch):
    assert inventory_of({"s.pdf": submittal.pdf},
                        root=str(tmp_path))["shape_hint"] == "small"
    monkeypatch.setenv(B.SHAPE1_PAGES_ENV, "10")
    out = inventory_of({"s.pdf": submittal.pdf}, root=str(tmp_path))
    assert out["shape_hint"] == "large" and out["shape1_pages"] == 10
    monkeypatch.setenv(B.SHAPE1_PAGES_ENV, "lots")
    assert B.shape1_pages() == B.DEFAULT_SHAPE1_PAGES


def test_page_specs(submittal, tmp_path):
    d = build(submittal.pdf, name="s.pdf", root=str(tmp_path))
    assert [r["page"] for r in d.pages("0-2,5")] == [0, 1, 2, 5]
    assert [r["page"] for r in d.pages(7)] == [7]
    assert len(d.pages()) == 11
    with pytest.raises(IndexError, match="0-10"):
        d.pages("9-12")
    assert parse_pages("3-1", 5) == [1, 2, 3]


# ---------------------------------------------------------------------------
# A long public manual (538 pages), when it is in this checkout
# ---------------------------------------------------------------------------

def test_a_538_page_manual_in_reasonable_time(tmp_path):
    if not os.path.isfile(UFC_260):
        pytest.skip("UFC 3-260-02 is not in this checkout")
    t0 = time.time()
    d = build(UFC_260, root=str(tmp_path))
    seconds = time.time() - t0
    assert d.n_pages == 538
    assert seconds < 120, seconds          # measured 3.5 s on a laptop
    inv = d.inventory()
    assert inv["kinds"]["figure"] == inv["needs_look"]["n"] == 99
    level1 = [e["title"] for e in inv["outline"] if e["level"] == 1]
    assert sum(t.startswith("APPENDIX") for t in level1) == 14
    # every appendix's own page captions it, with its printed number
    caps = {r["id"]: r for r in d.references(kinds=["appendix"])
            if r["role"] == "caption" and r["page"] >= 400}
    assert sorted(caps) == list("ABCDEFGHIJKLMN")
    assert caps["N"]["printed"] == "N-1" and caps["N"]["pdf_page"] == 528
    tables = [r for r in d.references(target="Table 12-")
              if r["role"] == "caption" and r.get("title") != "(Continued)"]
    assert [t["id"] for t in tables] == [f"12-{n}" for n in range(1, 9)]
    assert d.search("resilient modulus")["hits"][0]["match"] == "exact"
