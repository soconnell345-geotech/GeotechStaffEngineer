"""Logs whose depth ruler cannot be fitted (BACKLOG RI-18), offline.

A page with no fitted ruler used to lose every depth: the floor seeded
nothing and the reader's gate refused every depth the model returned. These
tests pin what is recovered now -- the depths the page PRINTS, each marked
with how it was read -- and what is still refused: any depth nothing on the
page prints. They also pin that a log whose ruler fits reads as before.

The pages are synthetic (``report_ingest.tests.no_ruler_fixtures`` and
``planlens.testing.loggrid_fixtures``); nothing is copied from a real log.
"""

from __future__ import annotations

import pytest

from report_ingest.log_floor import seed_from_grid
from report_ingest.log_reader import (
    LOG_READER_SYSTEM, LOG_READER_SYSTEM_PRINTED, LogReading, ReadLayer,
    ReadProv, ReadSPT, ReadSample, read_log,
)
from report_ingest.printed_depths import (
    TRUST, _depth_cell, _downward, _lead_cell, _sample_cell, PrintedDepth,
    PrintedRow, printed_depths,
)
from report_ingest.tests import no_ruler_fixtures as F
from report_ingest.tests.fake_engine import FakeEngine

pymupdf = pytest.importorskip("pymupdf", reason="the fixtures are drawn with PyMuPDF")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _open(pdf: bytes, tmp_path, name: str = "log.pdf"):
    from planlens.document import open_document
    path = tmp_path / name
    path.write_bytes(pdf)
    return open_document(str(path))


def _grid(doc, pages):
    from planlens.document.loggrid import log_grid
    return log_grid(doc, list(pages))


@pytest.fixture
def tabular(tmp_path):
    gt = F.build_tabular_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def sample_depths(tmp_path):
    gt = F.build_sample_depths_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def nothing(tmp_path):
    gt = F.build_sample_depths_log(depths=False)
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def imperial(tmp_path):
    from planlens.testing.loggrid_fixtures import build_imperial_log
    gt = build_imperial_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def metric(tmp_path):
    from planlens.testing.loggrid_fixtures import build_metric_log
    gt = build_metric_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def two_sheets(tmp_path):
    """Sheet 1 has a ruler (0 to about 30 ft); sheet 2, the continuation,
    has none and prints its samples' intervals from 60 ft down."""
    from planlens.testing.loggrid_fixtures import build_imperial_log
    second = F.build_sample_depths_log(first=60.0, hole="B-12")
    doc = _open(F.two_page_log(build_imperial_log().pdf, second.pdf),
                tmp_path, name="two_sheets.pdf")
    yield doc, second
    doc.close()


def _prov(page=0, bbox=(100.0, 200.0, 180.0, 212.0)):
    return ReadProv(page=page, bbox=bbox)


def _reading(unit, *, layers=(), samples=(), spt=()) -> LogReading:
    return LogReading(investigation_id="BH-3", kind="boring",
                      depth_unit=unit, units_known=True, layers=list(layers),
                      samples=list(samples), spt=list(spt), pages_read=[0])


def _brief(engine) -> str:
    return engine.calls[0]["messages"][0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# the premise: no ruler fits on these pages
# ---------------------------------------------------------------------------

class TestThePremise:
    @pytest.mark.parametrize("which", ["tabular", "sample_depths", "nothing"])
    def test_log_grid_fits_no_ruler(self, which, request):
        doc, _gt = request.getfixturevalue(which)
        grid = _grid(doc, [0])
        assert grid.rulers == {}
        assert grid.unit is None
        assert all(c.depth is None for c in grid.rows)
        assert any("no depth ruler" in w for w in grid.warnings)

    def test_the_two_sheet_log_has_a_ruler_on_its_first_sheet_only(
            self, two_sheets):
        doc, _gt = two_sheets
        grid = _grid(doc, [0, 1])
        assert set(grid.rulers) == {0}


# ---------------------------------------------------------------------------
# what is read off the page
# ---------------------------------------------------------------------------

class TestWhatIsRead:
    def test_a_depth_column_is_read_as_printed(self, tabular):
        doc, gt = tabular
        found = printed_depths(_grid(doc, [0]), [0])
        assert found.pages == [0]
        assert {r.source for r in found.readings} == {"depth_column"}
        printed = {v for r in found.readings for v in (r.top, r.bottom)
                   if v is not None}
        assert printed == set(gt.printed)
        assert found.unit == "m" and "heading" in found.unit_from

    def test_a_samples_own_interval_is_read_with_its_label(
            self, sample_depths):
        doc, gt = sample_depths
        found = printed_depths(_grid(doc, [0]), [0])
        assert [(r.source, r.label, r.top, r.bottom)
                for r in found.readings] == [
            ("sample_column", label, top, bottom)
            for label, top, bottom, _blows, _n in gt.samples]
        assert found.unit == "ft"

    def test_a_log_that_prints_no_depth_yields_none(self, nothing):
        doc, _gt = nothing
        found = printed_depths(_grid(doc, [0]), [0])
        assert found.pages == [0] and found.readings == []
        assert not found

    @pytest.mark.parametrize("which", ["imperial", "metric"])
    def test_nothing_is_read_on_a_page_with_a_ruler(self, which, request):
        doc, _gt = request.getfixturevalue(which)
        found = printed_depths(_grid(doc, [0]), [0])
        assert found.pages == [] and found.readings == []
        assert found.to_dict()["readings"] == []

    def test_only_the_sheet_with_no_ruler_is_read(self, two_sheets):
        doc, gt = two_sheets
        found = printed_depths(_grid(doc, [0, 1]), [0, 1])
        assert found.pages == [1]
        assert {r.page for r in found.readings} == {1}
        assert [r.top for r in found.readings] == [s[1] for s in gt.samples]


class TestWhatACellIsReadAs:
    @pytest.mark.parametrize("text, want", [
        ("1.50", (1.5, None, "")),
        ("1.50 - 3.00", (1.5, 3.0, "")),
        ("2.5 ft", (2.5, None, "ft")),
        ("0.0 to 1.2 m", (0.0, 1.2, "m")),
        ("3.0 - 1.5", None),                 # an interval goes down the hole
        ("SPT", None),
        ("12 blows", None),
    ])
    def test_a_depth_column_cell(self, text, want):
        assert _depth_cell(text) == want

    @pytest.mark.parametrize("text, want", [
        ("S-1  2.5-4.0", (2.5, 4.0, "", "S-1")),
        ("SS-3 (10-11.5)", (10.0, 11.5, "", "SS-3")),
        ("U-2 @ 6.0 m", (6.0, None, "m", "U-2")),
        ("5-7", None),            # two whole numbers alone: a label as likely
        ("S-1", None),
        ("2-3-4", None),
    ])
    def test_a_sample_column_cell(self, text, want):
        assert _sample_cell(text) == want

    @pytest.mark.parametrize("text, want", [
        ("0.0 - 1.2 m: FILL, brown", (0.0, 1.2, "m", "FILL, brown")),
        ("1.2-4.5 SILTY CLAY", (1.2, 4.5, "", "SILTY CLAY")),
        ("2-3 inch gravel", None),           # a size, not a stratum
        ("1-2 percent organics", None),      # an amount
        ("SAND with 10-20% gravel", None),   # not at the head of the line
    ])
    def test_a_description_line(self, text, want):
        assert _lead_cell(text) == want

    def test_a_number_that_breaks_the_downward_order_is_not_a_depth(self):
        def at(y, top):
            return PrintedDepth(page=0, top=top, bottom=None,
                                source="depth_column", text=f"{top:g}",
                                bbox=(0.0, y, 10.0, y + 8.0))
        kept, dropped = _downward([at(100, 1.0), at(140, 2.5), at(180, 105.3),
                                   at(220, 4.0), at(260, 6.0)])
        assert [r.top for r in kept] == [1.0, 2.5, 4.0, 6.0]
        assert [r.top for r in dropped] == [105.3]


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

class TestTheFloor:
    def test_a_printed_depth_column_seeds_layers_and_samples(self, tabular):
        doc, gt = tabular
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        assert seed.depth_unit == "m"
        assert [(ly.top.value, ly.bottom.value) for ly in seed.layers] == [
            (top, bottom) for top, bottom, _words in gt.layers]
        for layer, (_t, _b, words) in zip(seed.layers, gt.layers):
            assert layer.description.startswith(words)
            assert "printed depths" in layer.prov.note
        # "7.00 End of borehole" is one depth beside a description: it names
        # an end of a stratum, not which end, so it seeds no layer.
        assert len(seed.layers) == 3
        # The clay's fourth line stands on S-2's line, where it could as
        # well open a new stratum beside the sample: the layer keeps the
        # three lines that are surely its own, and no layer takes the fourth.
        assert seed.layers[1].description.endswith("occasional rootlets,")
        assert not any("mottled" in ly.description for ly in seed.layers)
        assert [(s.sample_id, s.top.value) for s in seed.samples] == [
            (label, top) for label, top, _b, _blows, _n in gt.samples]
        assert [(d.blows, d.n) for d in seed.spt] == [
            ([int(b) for b in blows.split("-")], n)
            for _l, _t, _b, blows, n in gt.samples]
        for sample in seed.samples:
            assert "no depth ruler on page 0" in sample.prov.note
            assert "printed in the depth column" in sample.prov.note
            assert sample.prov.confidence == TRUST["depth_column"]

    def test_a_value_on_no_printed_depths_line_gets_no_depth(self, tabular):
        # S-2's moisture content is printed a line below its row, where no
        # depth stands: it is not moved onto the nearest printed depth.
        doc, _gt = tabular
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        by_id = {s.sample_id: s for s in seed.samples}
        assert by_id["S-1"].water_content == 25.0
        assert by_id["S-2"].water_content is None
        assert 14.0 not in [s.water_content for s in seed.samples]

    def test_a_samples_base_is_printed_or_unknown(self):
        def at(source, top, bottom, text):
            return PrintedDepth(page=0, top=top, bottom=bottom, source=source,
                                text=text, bbox=(0.0, 100.0, 10.0, 108.0))
        depth = at("depth_column", 1.5, 3.0, "1.50 - 3.00")
        own = at("sample_column", 1.5, 1.95, "S-2 1.5-1.95")
        assert PrintedRow([depth]).sample_bottom(beside_words=False) == 3.0
        # Beside a description, the depth column's interval may be the
        # stratum's: the sample's base is then not known.
        assert PrintedRow([depth]).sample_bottom(beside_words=True) is None
        assert PrintedRow([depth, own]).sample_bottom(
            beside_words=True) == 1.95

    def test_a_samples_own_interval_seeds_the_sample(self, sample_depths):
        doc, gt = sample_depths
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        assert seed.depth_unit == "ft"
        assert [(s.sample_id, s.top.value, s.bottom.value)
                for s in seed.samples] == [
            (label, top, bottom) for label, top, bottom, _b, _n in gt.samples]
        assert [d.n for d in seed.spt] == [n for *_rest, n in gt.samples]
        assert all("as the sample's own interval" in s.prov.note
                   for s in seed.samples)
        # The strata print no depth, so none is seeded as a layer.
        assert seed.layers == []

    def test_a_log_that_prints_no_depth_seeds_none(self, nothing):
        doc, _gt = nothing
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        assert seed.samples == [] and seed.spt == [] and seed.layers == []
        assert seed.depth_unit == "" and seed.total_depth is None

    def test_a_ruled_sheet_seeds_as_before_and_the_next_sheet_adds_to_it(
            self, two_sheets, imperial):
        doc, gt = two_sheets
        seed = seed_from_grid(_grid(doc, [0, 1]), [0, 1], "RX")
        one_doc, _ = imperial
        alone = seed_from_grid(_grid(one_doc, [0]), [0], "RX")
        first = [s for s in seed.samples if s.prov.page == 0]
        assert [s.model_dump() for s in first] == \
            [s.model_dump() for s in alone.samples]
        second = [s for s in seed.samples if s.prov.page == 1]
        assert [(s.sample_id, s.top.value, s.bottom.value) for s in second] \
            == [(label, top, bottom) for label, top, bottom, _b, _n
                in gt.samples]


# ---------------------------------------------------------------------------
# the reader: what it is told, what it keeps, what it refuses
# ---------------------------------------------------------------------------

class TestTheReader:
    def _tabular_reading(self, extra_layers=()):
        layers = [
            ReadLayer(top=0.0, bottom=1.2, description="FILL", prov=_prov()),
            ReadLayer(top=1.2, bottom=4.5, description="SILTY CLAY (CL)",
                      prov=_prov()),
            ReadLayer(top=4.5, bottom=7.0, description="SAND (SP)",
                      prov=_prov()),
            *extra_layers,
        ]
        samples = [ReadSample(sample_id=f"S-{k + 1}", top=top, kind="spt",
                              prov=_prov())
                   for k, top in enumerate((0.5, 1.5, 3.0, 6.0))]
        spt = [ReadSPT(depth_top=0.5, blows=["2", "3", "4"], n=12,
                       sample_id="S-1", prov=_prov())]
        return _reading("m", layers=layers, samples=samples, spt=spt)

    def test_the_reader_is_told_the_depths_the_page_prints(self, tabular):
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading()}])
        read_log(doc, [0], engine)
        brief = _brief(engine)
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM_PRINTED
        assert "THE DEPTHS THESE PAGES PRINT" in brief
        assert "in the depth column | 0 to 1.2" in brief
        assert "its depths are the ones it PRINTS" in brief
        assert "NO DEPTH SCALE ON THIS PAGE" in brief

    def test_the_printed_rule_replaces_only_the_no_scale_rule(self):
        assert LOG_READER_SYSTEM_PRINTED != LOG_READER_SYSTEM
        assert "report NO depths" in LOG_READER_SYSTEM
        assert "report NO depths" not in LOG_READER_SYSTEM_PRINTED
        assert "PRINTS" in LOG_READER_SYSTEM_PRINTED
        head = LOG_READER_SYSTEM.split("NEVER INVENT A DEPTH")[0]
        assert LOG_READER_SYSTEM_PRINTED.startswith(head)

    def test_printed_depths_are_kept_and_say_how_they_were_read(
            self, tabular):
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading()}])
        result = read_log(doc, [0], engine)
        model = result.model_investigation
        assert [ly.top.value for ly in model.layers] == [0.0, 1.2, 4.5]
        assert [s.top.value for s in model.samples] == [0.5, 1.5, 3.0, 6.0]
        assert len(result.changes) == 3 * 2 + 4 + 1   # layers, samples, spt
        assert all("the page prints in the depth column" in c["why"]
                   for c in result.changes)
        assert not [u for u in result.unresolved
                    if u["refused_by"] == "python"]

    def test_a_depth_nothing_on_the_page_prints_is_refused(self, tabular):
        doc, _gt = tabular
        stray = ReadLayer(top=2.0, bottom=3.0, description="a lens of sand",
                          prov=_prov())
        engine = FakeEngine([{"final": self._tabular_reading([stray])}])
        result = read_log(doc, [0], engine)
        refused = [u for u in result.unresolved if u["refused_by"] == "python"]
        assert [u["value"] for u in refused] == [2.0]
        assert "nothing printed on it reads 2" in refused[0]["why"]
        assert 2.0 not in [ly.top.value
                           for ly in result.investigation.layers]

    def test_the_qa_says_the_ruler_was_not_fitted_and_what_was_read(
            self, tabular):
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading()}])
        result = read_log(doc, [0], engine)
        scale = [u for u in result.unresolved if u["refused_by"] == "no_ruler"]
        assert len(scale) == 1 and scale[0]["page"] == 0
        why = scale[0]["why"]
        assert "could not be fitted" in why
        assert "8 depth(s) were read from what the page prints" in why
        assert "1 of the 16 sample and test value(s)" in why
        record = result.to_dict()["printed_depths"]
        assert record["pages"] == [0] and len(record["readings"]) == 8

    def test_a_samples_own_interval_is_kept_top_and_base(self, sample_depths):
        doc, gt = sample_depths
        samples = [ReadSample(sample_id=label, top=top, bottom=bottom,
                              kind="spt", prov=_prov())
                   for label, top, bottom, _b, _n in gt.samples]
        engine = FakeEngine([{"final": _reading("ft", samples=samples)}])
        result = read_log(doc, [0], engine)
        assert [(s.top.value, s.bottom.value)
                for s in result.model_investigation.samples] == [
            (top, bottom) for _l, top, bottom, _b, _n in gt.samples]
        assert all("the page prints as the sample's own interval" in c["why"]
                   for c in result.changes)

    def test_a_log_that_prints_nothing_still_refuses_every_depth(
            self, nothing):
        doc, _gt = nothing
        samples = [ReadSample(sample_id="S-1", top=2.5, bottom=4.0,
                              prov=_prov())]
        layers = [ReadLayer(top=0.0, bottom=6.0, description="LEAN CLAY",
                            prov=_prov())]
        engine = FakeEngine([{"final": _reading("ft", layers=layers,
                                                samples=samples)}])
        result = read_log(doc, [0], engine)
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM
        assert "THE DEPTHS THESE PAGES PRINT" not in _brief(engine)
        inv = result.investigation
        assert inv.samples == [] and inv.layers == []
        refused = [u for u in result.unresolved if u["refused_by"] == "python"]
        assert refused and all("no depth scale was found" in u["why"]
                               for u in refused)
        scale = [u for u in result.unresolved if u["refused_by"] == "no_ruler"]
        assert len(scale) == 1
        assert "could not be fitted" in scale[0]["why"]
        assert "prints no depth" in scale[0]["why"]
        assert result.changes == []

    def test_a_continuation_sheet_with_no_ruler_keeps_its_printed_depths(
            self, two_sheets):
        doc, gt = two_sheets
        reading = _reading("ft", samples=[
            ReadSample(sample_id="S-1", top=60.0, bottom=61.5,
                       prov=_prov(page=1)),
            ReadSample(sample_id="S-x", top=65.0, prov=_prov(page=1)),
        ])
        reading.pages_read = [0, 1]
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0, 1], engine)
        # Sheet 1 has a ruler, so the log reads under the usual rule; the
        # brief lists what sheet 2 prints.
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM
        assert "THE DEPTHS THESE PAGES PRINT" in _brief(engine)
        kept = result.model_investigation.samples
        assert [(s.top.value, s.bottom.value) for s in kept] == [(60.0, 61.5)]
        refused = [u for u in result.unresolved if u["refused_by"] == "python"]
        assert [u["value"] for u in refused] == [65.0]
        assert "outside what the scale on this log reads" in refused[0]["why"]
        scale = [u for u in result.unresolved if u["refused_by"] == "no_ruler"]
        assert [u["page"] for u in scale] == [1]


# ---------------------------------------------------------------------------
# a log whose ruler fits reads as before
# ---------------------------------------------------------------------------

class TestALogThatFitsIsUnchanged:
    @pytest.mark.parametrize("which", ["imperial", "metric"])
    def test_nothing_new_reaches_the_model_or_the_record(self, which,
                                                         request):
        from report_ingest.tests.test_log_reader import _good_imperial_reading
        doc, _gt = request.getfixturevalue(which)
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM
        brief = _brief(engine)
        assert "THE DEPTHS THESE PAGES PRINT" not in brief
        assert "NO DEPTH SCALE" not in brief
        assert "printed_depths" not in result.to_dict()
        assert all(u["refused_by"] != "no_ruler" for u in result.unresolved)
        assert not any("no depth ruler on page" in c["why"]
                       for c in result.changes)
        assert not any("no depth ruler on page" in s.prov.note
                       for s in result.floor.samples)

    @pytest.mark.parametrize("which", ["imperial", "metric"])
    def test_the_floor_is_the_same_with_the_printed_pass_forced_empty(
            self, which, request):
        from report_ingest.printed_depths import PrintedDepths
        doc, _gt = request.getfixturevalue(which)
        grid = _grid(doc, [0])
        found = seed_from_grid(grid, [0], "RX")
        forced = seed_from_grid(grid, [0], "RX", printed=PrintedDepths())
        assert found.model_dump() == forced.model_dump()
