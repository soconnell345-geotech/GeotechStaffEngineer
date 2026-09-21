"""The log-template recogniser, on invented forms only.

NOTHING HERE NAMES A REAL FIRM. The fingerprints are made up for the test the
same way the shipped ``templates.json.EXAMPLE`` is, and the pages they are
matched against are planlens' synthetic report. The real fingerprints live in
a private file that is never committed, and one test asserts that the shipped
example holds no real firm either.
"""

from __future__ import annotations

import json

import pytest

from report_ingest.log_templates import (
    EXAMPLE_PATH, MATCH_THRESHOLD, Fingerprint, TemplateMatch, active_templates,
    annotate_ledger, column_names, ledger_note, load_templates, recognise,
    recognise_pages, templates_beside, use_templates,
)

pytest.importorskip("planlens.document.loggrid")

#: The pages of the synthetic report's boring log.
LOG_PAGES = (7, 8)


@pytest.fixture(scope="module")
def pdf() -> bytes:
    from report_ingest.tests.narrative_fixtures import build_narrative_report
    return build_narrative_report().pdf


@pytest.fixture()
def doc(pdf):
    from planlens.document import open_document
    document = open_document(pdf, name="SYN")
    try:
        yield document
    finally:
        document.close()


@pytest.fixture()
def grid(doc):
    from planlens.document.loggrid import log_grid
    return log_grid(doc, list(LOG_PAGES))


def fingerprint(**overrides) -> Fingerprint:
    """A form that the synthetic log page really does print."""
    data = dict(
        name="Quillfeather Geotechnics log",
        family="Quillfeather Geotechnics",
        years="2019-",
        title_phrases=["LOG OF BORING", "BORING NO.", "Sheet 1 of 2"],
        column_headers=["DEPTH (m)", "ELEV (m)", "SAMPLE", "BLOWS N",
                        "RECOVERY", "USCS"],
    )
    data.update(overrides)
    return Fingerprint(**{k: (tuple(v) if isinstance(v, list) else v)
                          for k, v in data.items()})


@pytest.fixture(autouse=True)
def no_ambient_templates():
    """No process-wide fingerprints leak between tests or out of them."""
    use_templates(None)
    yield
    use_templates(None)


class TestRecognising:
    """A page prints a form's own words, or it does not."""

    def test_a_page_that_prints_the_form_is_recognised_with_its_evidence(
            self, doc, grid):
        match = recognise(doc, 7, grid=grid, templates=[fingerprint()])

        assert isinstance(match, TemplateMatch)
        assert match.family == "Quillfeather Geotechnics"
        assert match.confidence >= MATCH_THRESHOLD
        assert match.page == 7
        # The evidence names what matched and where it came from, so a
        # person can disagree with the match rather than only distrust it.
        assert match.evidence
        joined = " ".join(match.evidence)
        assert "LOG OF BORING" in joined
        assert any(part.startswith("title ") for part in match.evidence)
        assert any(part.startswith("columns ") for part in match.evidence)

    def test_a_near_miss_below_the_threshold_is_no_match(self, doc, grid):
        near = fingerprint(
            name="Marchbank Soils log", family="Marchbank Soils",
            title_phrases=["RECORD OF SUBSURFACE EXPLORATION",
                           "Marchbank Representative:",
                           "Groundwater Observations"],
            column_headers=["SAMPLING DATA", "TESTS", "STRATUM"])

        assert recognise(doc, 7, grid=grid, templates=[near]) is None

    def test_a_page_that_is_not_a_log_is_not_claimed(self, doc, grid):
        # Page 2 is narrative prose; it prints none of the form's words.
        assert recognise(doc, 2, templates=[fingerprint()]) is None

    def test_no_fingerprints_means_no_match_and_no_work(self, doc, grid):
        assert recognise(doc, 7, grid=grid, templates=[]) is None
        assert recognise(doc, 7, grid=grid) is None      # none in force
        assert recognise_pages(doc, LOG_PAGES) == {}

    def test_a_missing_templates_file_is_a_no_op(self, tmp_path, doc):
        missing = tmp_path / "nothing" / "templates.json"

        assert load_templates(missing) == []
        assert templates_beside(tmp_path / "truth" / "logs") == []
        assert recognise(doc, 7, templates_path=missing) is None

    def test_recognise_pages_answers_only_for_the_pages_that_matched(
            self, doc):
        got = recognise_pages(doc, [2, 7, 8], templates=[fingerprint()])

        assert set(got) <= {7, 8}
        assert 2 not in got
        assert all(m.page == page for page, m in got.items())


class TestAFamilyThatDrifted:
    """Several fingerprints, one family -- the template over the years."""

    def test_two_drifted_forms_report_the_one_family(self, doc, grid):
        old = fingerprint(name="Quillfeather log, 2014 form",
                          years="2012-2018",
                          title_phrases=["LOG OF BORING", "BORING NO.",
                                         "Drilling Foreman:"])
        new = fingerprint(name="Quillfeather log, 2019 form")

        for order in ([old, new], [new, old]):
            match = recognise(doc, 7, grid=grid, templates=order)
            assert match is not None
            assert match.family == "Quillfeather Geotechnics"
        # The one that prints more of itself on this page wins the NAME.
        best = recognise(doc, 7, grid=grid, templates=[old, new])
        assert best.name == "Quillfeather log, 2019 form"

    def test_the_margin_is_measured_against_another_family_not_a_sibling(
            self, doc, grid):
        sibling = fingerprint(name="Quillfeather log, 2014 form",
                              years="2012-2018")
        other = fingerprint(name="Marchbank Soils log",
                            family="Marchbank Soils",
                            title_phrases=["RECORD OF SUBSURFACE EXPLORATION"],
                            column_headers=["SAMPLING DATA"])

        match = recognise(doc, 7, grid=grid,
                          templates=[fingerprint(), sibling, other])

        # A sibling scoring just as well is the form drifting, not a doubt
        # about whose form it is, so the margin ignores it.
        assert match.runner_up == "Marchbank Soils log"
        assert match.margin > 0.5


class TestWhatAMatchIsWorthToTheGrid:
    """The column map, and the floor that reads the columns it names."""

    def test_the_column_map_names_a_column_the_grid_could_not(self, grid):
        match = recognise_column_map(grid)
        named = column_names(grid, match)

        # The USCS column is claimed for the sample id by this (invented)
        # form; the grid's own name for it survives beside the new one.
        by_id = {c.id: c for c in grid.columns}
        claimed = [cid for cid, names in named.items()
                   if "sample_id" in names and by_id[cid].header == "USCS"]
        assert claimed
        for cid in claimed:
            assert named[cid][0] == "sample_id"
            assert "uscs" in named[cid]

    def test_no_column_map_changes_nothing(self, grid):
        plain = TemplateMatch(name="n", family="f", confidence=0.9)

        assert column_names(grid, plain) == {}
        assert column_names(grid, None) == {}

    def test_the_seed_reads_a_stacked_column_only_with_the_map(self):
        from report_ingest.log_floor import seed_from_grid

        grid = stacked_grid()
        match = TemplateMatch(
            name="Marchbank Soils 2008", family="Marchbank Soils",
            confidence=0.97,
            column_map={"sample_id": "SAMPLING DATA",
                        "sample_type": "SAMPLING DATA",
                        "blows": "SAMPLING DATA",
                        "recovery": "SAMPLING DATA",
                        "index": "TESTS"})

        # WITHOUT the fingerprint the two columns are headed DATA and TESTS,
        # which the general header vocabulary cannot name, so nothing is
        # seeded from them at all.
        bare = seed_from_grid(grid, [0])
        assert bare.samples == []
        assert bare.spt == []

        # WITH it the form says what those columns carry, and the floor
        # reads every value they print.
        seeded = seed_from_grid(grid, [0], template=match)
        assert len(seeded.samples) == 1
        sample = seeded.samples[0]
        assert sample.sample_id == "S-1"
        assert sample.kind == "spt"
        assert sample.recovery_percent == pytest.approx(64.0)
        assert sample.recovery.value == pytest.approx(29.0)
        assert sample.recovery.unit == "cm"
        assert sample.water_content == pytest.approx(12.0)
        assert sample.liquid_limit == pytest.approx(38.0)
        assert [d.blows for d in seeded.spt] == [[2, 1, 2]]

    def test_the_seed_records_the_template_without_claiming_a_new_method(
            self):
        from report_ingest.log_floor import seed_from_grid

        grid = stacked_grid()
        match = TemplateMatch(name="Marchbank Soils 2008",
                              family="Marchbank Soils", confidence=0.97,
                              column_map={"blows": "SAMPLING DATA"})
        seeded = seed_from_grid(grid, [0], template=match)

        notes = [p.note for p in seeded.prov]
        assert any("Marchbank Soils" in note for note in notes)
        # The value was still PLACED by geometry; the form only said what
        # the column carries, so the method stays the grid's.
        assert all(p.method == "grid" for p in seeded.prov)
        assert all(d.prov.method == "grid" for d in seeded.spt)


class TestTheLedgerLine:
    """What a voter is told: the family and a number."""

    def test_the_note_is_the_family_and_the_confidence(self):
        match = TemplateMatch(name="a long internal name",
                              family="Marchbank Soils", confidence=0.9231)

        assert ledger_note(match) == "template Marchbank Soils (0.92)"
        assert ledger_note(None) == ""

    def test_the_ledger_gains_the_note_on_the_pages_that_matched(self):
        lines = ["p006 text narrative 0.90", "p007 form boring_log 0.90",
                 "p008 form boring_log 0.90"]
        match = TemplateMatch(name="n", family="Marchbank Soils",
                              confidence=0.97)

        got = annotate_ledger(lines, {7: match})

        assert got[0] == lines[0]
        assert got[1].startswith(lines[1])
        assert got[1].endswith("template Marchbank Soils (0.97)")
        assert got[2] == lines[2]
        assert annotate_ledger(lines, {}) == lines


class TestTheProcessWideRegister:
    """Fingerprints are set once by whoever knows where the file is."""

    def test_use_templates_takes_a_path_a_list_or_none(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text(json.dumps({"templates": [
            {"name": "n", "family": "Marchbank Soils",
             "footer_phrases": ["MARCHBANK STANDARD LOG.GDT"]}]}),
            encoding="utf-8")

        assert len(use_templates(path)) == 1
        assert active_templates()[0].family == "Marchbank Soils"
        assert use_templates([fingerprint()])[0].family.startswith("Quill")
        assert use_templates(None) == []
        assert active_templates() == []

    def test_a_fingerprint_needs_a_name_and_a_family(self):
        with pytest.raises(ValueError):
            Fingerprint.from_dict({"family": "Marchbank Soils"})
        with pytest.raises(ValueError):
            Fingerprint.from_dict({"name": "n"})


class TestTheShippedExample:
    """The committed example documents the shape and names nobody real."""

    def test_the_example_parses_and_carries_two_families(self):
        rows = load_templates(EXAMPLE_PATH)

        assert len(rows) >= 3
        families = {row.family for row in rows}
        assert len(families) == 2
        # One of the two is described by more than one fingerprint, which is
        # the drifted-template case the file exists to document.
        counts = {f: sum(1 for r in rows if r.family == f) for f in families}
        assert max(counts.values()) >= 2
        for row in rows:
            assert row.footer_phrases or len(row.title_phrases) >= 3

    def test_the_example_names_no_real_firm_or_site(self):
        offences = privacy_offences(EXAMPLE_PATH.read_text(encoding="utf-8"))

        assert not offences, (
            f"the shipped example carries {len(offences)} privacy-listed "
            f"term(s); look the hashes up in the owner's own list: "
            f"{offences}")

    def test_the_shipped_modules_name_no_real_firm_or_site(self):
        from report_ingest import log_templates, narrative_glossary

        for module in (log_templates, narrative_glossary):
            text = open(module.__file__, encoding="utf-8").read()
            offences = privacy_offences(text)
            assert not offences, (
                f"{module.__name__} carries privacy-listed term(s): "
                f"{offences}")

    def test_the_guard_itself_catches_something(self, monkeypatch):
        # A guard nobody has seen fire is a guard nobody can trust -- and it
        # cannot be demonstrated on a real listed term, because writing one
        # into this file is the thing it exists to forbid. So it fires on an
        # INVENTED term whose hash is put in the set for the length of the
        # test, one word and two.
        import hashlib

        from report_ingest.tests import test_log_templates as here

        one = "quillfeather"
        two = "marchbank soils"
        digests = {t: hashlib.sha256(t.encode()).hexdigest()[:16]
                   for t in (one, two)}
        monkeypatch.setattr(here, "PRIVACY_HASHES",
                            frozenset(digests.values()))

        assert privacy_offences("a perfectly ordinary sentence") == []
        assert privacy_offences(f"a log by {one} of 2019") == [digests[one]]
        assert privacy_offences(f"printed by {two} in 2008") == [digests[two]]


#: The owner's privacy word list, AS HASHES. This repo is public and none of
#: those words may appear in a committed file -- which includes this one, so
#: the guard cannot spell them. Each entry is the first 16 hex characters of
#: the sha256 of one lower-case term; :func:`privacy_offences` hashes every
#: word and adjacent word pair of a file and looks for a hit, so a two-word
#: place name is caught as well as a one-word firm.
PRIVACY_HASHES = frozenset((
    "1097bf50dd7ffb4c", "19e2dac39c1231c4", "1dee6c1f0f6e4200",
    "204737f850cf4576", "21bd3229a131a2cc", "22ff277b10da3a6f",
    "3553be1e6e882992", "4007848811fc5701", "431debec82077bf2",
    "476dc40524cccea0", "4a775ac07fd8dcc7", "51e2a46721d104d9",
    "5f211d4244df430c", "795325e39e586917", "7ba9b19fc81560f4",
    "7c59f71bb7ccfdb0", "86846e84f9bfb205", "92cd70149866adaa",
    "9cbde7fbf7daa2b9", "9d733da86c097c51", "a5b7d83d97c95936",
    "a5c49775b0a68d9f", "aca04351dc9bd7bd", "bb08f449f6fbc962",
    "e18ad88cf9a0960f", "e5fb7fff750bc9bb", "f1727e3300b0a6a6",
    "f3021978350cdd12",
))


def privacy_offences(text: str) -> list:
    """The hashes of any privacy-listed term this text carries.

    Empty is the only acceptable answer for a committed file. The hash of
    the offending term is what comes back, which is enough to look it up in
    the owner's own list and nowhere near enough to publish it.
    """
    import hashlib
    import re as _re

    words = _re.findall(r"[a-z]+", str(text).lower())
    pairs = [f"{a} {b}" for a, b in zip(words, words[1:])]
    out = []
    for token in words + pairs:
        digest = hashlib.sha256(token.encode()).hexdigest()[:16]
        if digest in PRIVACY_HASHES and digest not in out:
            out.append(digest)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def recognise_column_map(grid) -> TemplateMatch:
    """A match whose column map claims the synthetic log's USCS column."""
    return TemplateMatch(name="Quillfeather Geotechnics log",
                         family="Quillfeather Geotechnics", confidence=0.9,
                         column_map={"sample_id": "USCS"})


def stacked_grid():
    """A grid of the shape a form that STACKS its sampling data produces.

    One column headed ``DATA`` carrying the sample id, the sampler code, the
    blow record and the recovery one under another, and one headed ``TESTS``
    carrying each index result with its own label. Neither header says
    anything the general vocabulary can classify, so both come back unnamed
    -- which is the whole case for a fingerprint.
    """
    from planlens.document.loggrid import Cell, Column, LogGrid, Ruler

    columns = [
        Column(id="c0", page=0, x0=100.0, x1=180.0, name="other", names=(),
               header="DATA"),
        Column(id="c1", page=0, x0=190.0, x1=280.0, name="other", names=(),
               header="TESTS"),
    ]
    rows = [
        Cell(page=0, column="other", column_id="c0", text="S-1, SS",
             bbox=(100.0, 200.0, 180.0, 210.0), depth=1.5, numbers=(1.0,)),
        Cell(page=0, column="other", column_id="c0", text="2+1+2",
             bbox=(100.0, 212.0, 180.0, 222.0), depth=1.6,
             numbers=(2.0, 1.0, 2.0)),
        Cell(page=0, column="other", column_id="c0", text="REC=29cm, 64%",
             bbox=(100.0, 224.0, 180.0, 234.0), depth=1.7,
             numbers=(29.0, 64.0)),
        Cell(page=0, column="other", column_id="c1", text="MC = 12.0%",
             bbox=(190.0, 200.0, 280.0, 210.0), depth=1.5, numbers=(12.0,)),
        Cell(page=0, column="other", column_id="c1", text="LL = 38",
             bbox=(190.0, 212.0, 280.0, 222.0), depth=1.6, numbers=(38.0,)),
    ]
    ruler = Ruler(page=0, kind="depth", column_id=None, slope=0.1,
                  intercept=0.0, residual=0.0, step=1.0, unit="m")
    return LogGrid(pages=[0], unit="m", columns=columns,
                   rulers={0: ruler}, rows=rows)
