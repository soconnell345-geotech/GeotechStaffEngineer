"""Logs whose depth ruler cannot be fitted (BACKLOG RI-18), offline.

A page with no fitted ruler used to lose every depth: the floor seeded
nothing and the reader's gate refused every depth the model returned. These
tests pin what is recovered now -- the depths the page PRINTS, each marked
with how it was read -- and what is still refused: any depth nothing on the
page prints, in the reader's unit, for that kind of value. They also pin
that a log whose ruler fits reads as before.

The pages are synthetic (``report_ingest.tests.no_ruler_fixtures``,
``planlens.testing.loggrid_fixtures`` and the small stand-in grids below);
nothing is copied from a real log.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_ingest.log_floor import seed_from_grid
from report_ingest.log_reader import (
    LOG_READER_SYSTEM, LOG_READER_SYSTEM_PRINTED, LogReading, ReadLayer,
    ReadProv, ReadSPT, ReadSample, ReadWater, read_log,
)
from report_ingest.printed_depths import (
    TRUST, PrintedDepth, PrintedDepths, PrintedRow, _depth_cell,
    _header_depth, _lead_cell, _sample_cell, printed_depths,
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
def no_ruler(tmp_path):
    from planlens.testing.loggrid_fixtures import build_log_without_ruler
    gt = build_log_without_ruler()
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


def _with_header(tmp_path, *lines):
    """The sample-depths log with these header lines printed on it."""
    gt = F.build_sample_depths_log(header=lines)
    return _open(gt.pdf, tmp_path, name="header.pdf"), gt


def _prov(page=0, bbox=(100.0, 200.0, 180.0, 212.0)):
    return ReadProv(page=page, bbox=bbox)


def _reading(unit, *, layers=(), samples=(), spt=(), water=()
             ) -> LogReading:
    return LogReading(investigation_id="BH-3", kind="boring",
                      depth_unit=unit, units_known=True, layers=list(layers),
                      samples=list(samples), spt=list(spt),
                      water=list(water), pages_read=[0])


def _brief(engine) -> str:
    return engine.calls[0]["messages"][0]["content"][0]["text"]


def _flagged(result):
    return [u for u in result.unresolved if u.get("flag") == "no_ruler"]


def _refused(result):
    return [u for u in result.unresolved if u.get("refused_by") == "python"]


# A stand-in for log_grid's result: one page, no ruler, the cells given.
class _Grid:
    def __init__(self, cells=(), columns=(), fields=None, unit=None):
        self.rows = list(cells)
        self.columns = list(columns)
        self.fields = dict(fields or {})
        self.field_boxes = {}
        self.rulers = {}
        self.unit = unit
        self.warnings = ["page 0: no depth ruler was found"]
        self.layers, self.samples, self.elevation_rulers = [], [], {}

    def column(self, column_id):
        return next((c for c in self.columns if c.id == column_id), None)


def _column(column_id, *names, unit=None):
    return SimpleNamespace(id=column_id, page=0, names=tuple(names),
                           unit=unit)


def _cell(column_id, text, y, column=None):
    return SimpleNamespace(page=0, column_id=column_id,
                           column=column or column_id, text=text,
                           bbox=(10.0, y, 40.0, y + 9.0), depth=None,
                           numbers=[], confidence=0.3)


def _depth_column_grid(*texts, margin=False):
    """A page whose depth column (or, with ``margin``, the description's
    margin) prints these numbers, one per line, top to bottom."""
    if margin:
        columns = [_column("desc", "description")]
        cells = [_cell("desc", t, 100.0 + 30.0 * k, column="depth")
                 for k, t in enumerate(texts)]
    else:
        columns = [_column("dep", "depth")]
        cells = [_cell("dep", t, 100.0 + 30.0 * k)
                 for k, t in enumerate(texts)]
    return _Grid(cells, columns)


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
        assert {r.source for r in found.readings} == {"depth_column",
                                                      "description"}
        printed = {v for r in found.readings for v in (r.top, r.bottom)
                   if v is not None}
        assert printed == set(gt.printed)
        assert found.unit == "m"
        assert all(r.seeds for r in found.readings)

    def test_a_samples_own_interval_is_read_with_its_label(
            self, sample_depths):
        doc, gt = sample_depths
        found = printed_depths(_grid(doc, [0]), [0])
        assert [(r.source, r.label, r.top, r.bottom, r.unit)
                for r in found.readings] == [
            ("sample_column", label, top, bottom, "ft")
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
        ("EL. 98.5", None),
    ])
    def test_a_depth_column_cell(self, text, want):
        assert _depth_cell(text) == want

    @pytest.mark.parametrize("text, want", [
        ("S-1  2.5-4.0 ft", (2.5, 4.0, "ft", "S-1")),
        ("SS-3 (10'-11.5')", (10.0, 11.5, "ft", "SS-3")),
        ("U-2 @ 6.0 m", (6.0, None, "m", "U-2")),
        # A blow record looks like an interval: no unit, no depth column,
        # no depth.
        ("S-1 6-8", None),
        ("S-1  2.5-4.0", None),
        ("S-3 @ 10.0", None),
        ("S-1 EL 98.5-97.0 ft", None),       # an elevation, not a depth
        ("5-7", None),
        ("S-1", None),
        ("2-3-4", None),
    ])
    def test_a_sample_column_cell(self, text, want):
        assert _sample_cell(text) == want

    def test_a_sample_interval_with_no_unit_is_read_in_a_depth_column(self):
        # "SAMPLE NO. & DEPTH": the column itself says the numbers are depths.
        assert _sample_cell("S-1  2.5-4.0", in_depth_column=True) == \
            (2.5, 4.0, "", "S-1")

    @pytest.mark.parametrize("text, want", [
        ("0.0 - 1.2 m: FILL, brown", (0.0, 1.2, "m", "FILL, brown")),
        ("1.2-4.5 SILTY CLAY", (1.2, 4.5, "", "SILTY CLAY")),
        ("2-3 inch gravel", None),           # a size, not a stratum
        ("2 to 4 ft thick sand lenses", None),   # a thickness
        ("1-2 percent organics", None),      # an amount
        ("95.0-90.0 ft msl: SAND", None),    # an elevation
        ("SAND with 10-20% gravel", None),   # not at the head of the line
        ("10-12 N", None),                   # a blow count
        ("2-3 cobbles", None),               # a count
        ("3-4 feet of fill over clay", None),    # a thickness
        ("10-12 blows", None),
        ("10-12 bpf", None),
        ("10-12% fines", None),
        ("2-3 in. gravel", None),
        ("2-3 inches of topsoil", None),
        ("2-3 ft boulders", None),           # a size
        # After a colon the same word is the stratum's own.
        ("0.0 - 1.5 m: COBBLES and BOULDERS",
         (0.0, 1.5, "m", "COBBLES and BOULDERS")),
    ])
    def test_a_description_line(self, text, want):
        assert _lead_cell(text) == want

    @pytest.mark.parametrize("raw, want", [
        ("Groundwater encountered at 10 ft", (10.0, "ft")),
        ("After 24 hrs 12.5 ft", (12.5, "ft")),
        ("3/4/2019 at 8 ft", (8.0, "ft")),
        ("Drilled 2019, 25 ft", (25.0, "ft")),
        ("Groundwater at 10 ft (EL. 94.5 ft)", (10.0, "ft")),
        ("EL. 98.5 ft", None),
        ("EL 1520 ft", None),
        ("Elev. 98.5 ft", None),
        ("98.5 ft NAVD88", None),
        ("1520 ft msl", None),
        ("at 8:30 am", None),
        ("Not encountered", None),
    ])
    def test_a_header_depth_is_the_number_its_unit_is_bound_to(self, raw,
                                                              want):
        assert _header_depth(raw) == want


class TestWhatOnlyChecks:
    """A column whose numbers do not rise all the way down, and the
    description's margin, only check the reader's depths."""

    @pytest.mark.parametrize("texts", [
        ("4.5", "95.5", "10.0", "90.0"),     # depths interleaved with elevations
        ("5", "12", "10", "15", "20"),       # a blow count among the depths
    ])
    def test_a_column_out_of_order_places_nothing(self, texts):
        found = printed_depths(_depth_column_grid(*texts), [0])
        assert len(found.readings) == len(texts)
        assert not any(r.seeds for r in found.readings)
        assert found.rows(0) == []
        # ... but each number still backs a depth the reader returns.
        assert found.supporting(10.0, 0, "ft", "sample") is not None

    @pytest.mark.parametrize("texts", [
        ("4.5", "95.5", "10.0", "90.0"),
        ("4.5", "95.5"),                     # one pair rises, and still
    ])
    def test_the_margin_places_nothing(self, texts):
        found = printed_depths(_depth_column_grid(*texts, margin=True), [0])
        assert found.readings and not any(r.seeds for r in found.readings)
        assert found.rows(0) == []

    def test_a_year_is_not_a_depth(self):
        found = printed_depths(_depth_column_grid("5", "10", "15", "2019"),
                               [0])
        assert [r.top for r in found.readings] == [5.0, 10.0, 15.0]
        assert all(r.seeds for r in found.readings)
        assert [d["text"] for d in found.dropped] == ["2019"]
        assert "year" in found.dropped[0]["why"]

    def test_a_column_that_rises_places_its_rows(self):
        found = printed_depths(_depth_column_grid("5", "10", "15"), [0])
        assert all(r.seeds for r in found.readings)
        assert [row.top for row in found.rows(0)] == [5.0, 10.0, 15.0]

    def test_an_elevation_in_the_header_is_not_a_depth(self):
        grid = _Grid(fields={"groundwater": "EL. 98.5 ft",
                             "total_depth": "EL 1520 ft"})
        found = printed_depths(grid, [0])
        assert found.readings == []


class TestUnitsAndKinds:
    def _page(self, *readings, units=()):
        found = PrintedDepths(pages=[0], readings=list(readings))
        found.page_units[0] = sorted(units)
        return found

    @staticmethod
    def _at(top, bottom=None, unit="", source="depth_column", text="",
            column_id="dep"):
        return PrintedDepth(page=0, top=top, bottom=bottom, source=source,
                            text=text or f"{top:g}", unit=unit,
                            column_id=column_id,
                            bbox=(0.0, 100.0, 10.0, 108.0))

    def test_a_depth_printed_in_metres_does_not_back_one_in_feet(self):
        found = self._page(self._at(1.5, 3.0, "m", text="1.5 - 3.0 m"),
                           units=["m"])
        assert found.supporting(1.5, 0, "m", "sample") is not None
        assert found.supporting(1.5, 0, "ft", "sample") is None

    def test_where_the_page_prints_both_units_the_bound_one_rules(self):
        metric = self._at(1.5, 3.0, "m", text="1.5 - 3.0 m")
        feet = self._at(5.0, None, "ft", text="5.0 ft")
        bare = self._at(2.0)
        found = self._page(metric, feet, bare, units=["ft", "m"])
        assert found.supporting(1.5, 0, "m", "sample") is metric
        assert found.supporting(1.5, 0, "ft", "sample") is None
        assert found.supporting(5.0, 0, "ft", "sample") is feet
        # A number with no unit of its own on a page printing two: unclear.
        assert found.supporting(2.0, 0, "m", "sample") is None
        assert found.supporting(2.0, 0, "ft", "sample") is None

    def test_where_the_page_prints_no_unit_any_unit_is_taken(self):
        found = self._page(self._at(2.0))
        assert found.supporting(2.0, 0, "ft", "sample") is not None

    def test_a_depth_backs_only_its_own_kind_of_value(self):
        water = self._at(10.0, unit="ft", source="header",
                         text="Groundwater encountered at 10 ft",
                         column_id="groundwater")
        total = self._at(25.0, unit="ft", source="header", text="25 ft",
                         column_id="total_depth")
        sample = self._at(2.5, 4.0, "ft", source="sample_column",
                          text="S-1 2.5-4.0 ft")
        found = self._page(water, total, sample, units=["ft"])
        assert found.supporting(10.0, 0, "ft", "water") is water
        assert found.supporting(10.0, 0, "ft", "sample") is None
        assert found.supporting(10.0, 0, "ft", "layer") is None
        assert found.supporting(25.0, 0, "ft", "layer") is None
        assert found.supporting(2.5, 0, "ft", "sample") is sample
        assert found.supporting(2.5, 0, "ft", "layer") is None
        assert found.supporting(2.5, 0, "ft", "water") is None


class TestAnIntervalInProse:
    """An interval at the head of a description backs a stratum's depth
    only where it prints a unit of its own or chains with the page's other
    intervals; never where it is a thickness, a count or a blow count."""

    @staticmethod
    def _description(*lines, page_unit=None):
        columns = [_column("desc", "description")]
        if page_unit:
            columns.append(_column("dep", "depth", unit=page_unit))
        cells = [SimpleNamespace(page=0, column_id="desc", column="desc",
                                 text=t, bbox=(10.0, 100.0 + 12.0 * k,
                                               200.0, 109.0 + 12.0 * k),
                                 depth=None, numbers=[], confidence=0.3)
                 for k, t in enumerate(lines)]
        return _Grid(cells, columns)

    @pytest.mark.parametrize("line, values", [
        ("10-12 N", (10.0, 12.0)),
        ("2-3 cobbles", (2.0, 3.0)),
        ("3-4 feet of fill over clay", (3.0, 4.0)),
    ])
    def test_a_thickness_or_a_count_backs_no_stratum(self, line, values):
        grid = self._description("0.5 - 1.0 m: TOPSOIL", line,
                                 page_unit="m")
        found = printed_depths(grid, [0])
        assert [r.text for r in found.readings] == ["0.5 - 1.0 m: TOPSOIL"]
        for value in values:
            for unit in ("m", "ft"):
                assert found.supporting(value, 0, unit, "layer") is None

    def test_an_interval_with_no_unit_that_chains_with_none_backs_nothing(
            self):
        found = printed_depths(self._description("1.2 - 4.5 SILTY CLAY",
                                                 page_unit="m"), [0])
        assert found.readings == []
        assert found.supporting(1.2, 0, "m", "layer") is None
        why = found.why_not(1.2, 0, "m", "layer")
        assert "'1.2 - 4.5 SILTY CLAY'" in why and "chains with no" in why

    def test_intervals_that_chain_back_strata_and_a_seam_with_a_unit_too(
            self):
        grid = self._description(
            "0.0 - 1.2 FILL, brown", "1.2 - 4.5 SILTY CLAY",
            "4.5 - 7.0 SAND, dense", "5.0 - 5.5 m: clay seam",
            "7.0 - 9.0 GRAVEL", "2-3 cobbles")
        found = printed_depths(grid, [0])
        assert found.dropped == []
        for value in (0.0, 1.2, 4.5, 7.0, 9.0):
            assert found.supporting(value, 0, "m", "layer") is not None
            assert found.supporting(value, 0, "ft", "layer") is None
        # The seam chains with nothing, but prints its own unit.
        seam = found.supporting(5.0, 0, "m", "layer")
        assert seam is not None and seam.unit == "m" and not seam.chains
        # The floor opens the chained strata; the seam between the sand and
        # the gravel does not break their chain, and opens no layer itself.
        seed = seed_from_grid(grid, [0], "RX", printed=found)
        assert seed.depth_unit == "m"
        assert [(ly.top.value, ly.bottom.value) for ly in seed.layers] == [
            (0.0, 1.2), (1.2, 4.5), (4.5, 7.0), (7.0, 9.0)]
        assert seed.layers[3].description == "GRAVEL 2-3 cobbles"

    def test_loose_page_text_is_read_by_the_same_rule(self):
        grid = _Grid()
        lines = {0: [SimpleNamespace(text=t, bbox=(10.0, y, 200.0, y + 9.0))
                     for t, y in (("1.2 - 4.5 SILTY CLAY", 100.0),
                                  ("10-12 N", 112.0))]}
        assert printed_depths(grid, [0], lines=lines).readings == []
        lines[0].append(SimpleNamespace(text="4.5 - 7.0 SAND",
                                        bbox=(10.0, 124.0, 200.0, 133.0)))
        found = printed_depths(grid, [0], lines=lines)
        assert [(r.top, r.bottom) for r in found.readings] == [
            (1.2, 4.5), (4.5, 7.0)]


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

    def test_an_interval_that_does_not_chain_opens_no_layer(self, tabular):
        # "5.0 - 5.5 m: clay seam" heads a line of the sand's description,
        # but 5.0 is no stratum's base: it is a seam, and opens no layer.
        doc, _gt = tabular
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        assert 5.0 not in [ly.top.value for ly in seed.layers]
        assert seed.layers[2].description == "SAND (SP), dense, brown, wet,"

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
        own = at("sample_column", 1.5, 1.95, "S-2 1.5-1.95 m")
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


class TestTheHeader:
    """On a log with no ruler, a header depth is the number its unit is
    bound to: never a date, a time, a year or an elevation."""

    def test_a_time_before_the_depth_is_not_the_depth(self, tmp_path):
        doc, _gt = _with_header(tmp_path,
                                "Groundwater:  After 24 hrs 12.5 ft",
                                "Total Depth:  Drilled 2019, 25 ft")
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        doc.close()
        assert [w.depth.value for w in seed.water if w.depth] == [12.5]
        assert seed.total_depth is not None
        assert seed.total_depth.value == 25.0

    def test_a_date_before_the_depth_is_not_the_depth(self, tmp_path):
        doc, _gt = _with_header(tmp_path, "Groundwater:  3/4/2019 at 8 ft")
        seed = seed_from_grid(_grid(doc, [0]), [0], "RX")
        doc.close()
        assert [w.depth.value for w in seed.water if w.depth] == [8.0]

    def test_an_elevation_is_not_a_depth(self, tmp_path):
        doc, _gt = _with_header(tmp_path, "Groundwater:  EL. 98.5 ft",
                                "Total Depth:  EL 1520 ft")
        grid = _grid(doc, [0])
        seed = seed_from_grid(grid, [0], "RX")
        assert [w.depth for w in seed.water if w.depth] == []
        assert seed.total_depth is None
        found = printed_depths(grid, [0])
        assert not [r for r in found.readings if r.source == "header"]
        reading = _reading("ft", water=[ReadWater(depth=98.5, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        doc.close()
        assert [u["value"] for u in _refused(result)] == [98.5]
        assert "98.5 ft" not in _brief(engine).split("THE ROWS")[0].split(
            "THE DEPTHS THESE PAGES PRINT")[-1]


# ---------------------------------------------------------------------------
# the reader: what it is told, what it keeps, what it refuses
# ---------------------------------------------------------------------------

class TestTheReader:
    def _tabular_reading(self, unit="m", extra_layers=()):
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
        return _reading(unit, layers=layers, samples=samples, spt=spt)

    def test_the_reader_is_told_the_depths_the_page_prints(self, tabular):
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading()}])
        read_log(doc, [0], engine)
        brief = _brief(engine)
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM_PRINTED
        assert "THE DEPTHS THESE PAGES PRINT" in brief
        assert "in the depth column | 0 to 1.2 m |" in brief
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
        assert _refused(result) == []

    def test_a_depth_in_the_wrong_unit_is_refused(self, tabular):
        # The page prints its depths in metres; a reading in feet is not
        # backed by them, however the numbers match.
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading(unit="ft")}])
        result = read_log(doc, [0], engine)
        assert result.model_investigation.layers == []
        assert result.model_investigation.samples == []
        refused = _refused(result)
        assert refused and all("in m" in u["why"] and "gave ft" in u["why"]
                               for u in refused)
        assert result.changes == []

    def test_a_depth_nothing_on_the_page_prints_is_refused(self, tabular):
        doc, _gt = tabular
        stray = ReadLayer(top=2.0, bottom=3.0, description="a lens of sand",
                          prov=_prov())
        engine = FakeEngine([{"final": self._tabular_reading(
            extra_layers=[stray])}])
        result = read_log(doc, [0], engine)
        refused = _refused(result)
        assert [u["value"] for u in refused] == [2.0]
        assert "nothing printed on it reads 2" in refused[0]["why"]
        assert 2.0 not in [ly.top.value
                           for ly in result.investigation.layers]

    def test_a_water_depth_does_not_back_a_sample(self, no_ruler):
        # The header prints "Groundwater encountered at 10 ft": that backs a
        # water level at 10 ft, and nothing else at 10 ft.
        doc, _gt = no_ruler
        reading = _reading(
            "ft", samples=[ReadSample(sample_id="1", top=10.0,
                                      prov=_prov())],
            water=[ReadWater(depth=10.0, prov=_prov())])
        result = read_log(doc, [0], FakeEngine([{"final": reading}]))
        model = result.model_investigation
        assert model.samples == []
        assert [w.depth.value for w in model.water] == [10.0]
        refused = _refused(result)
        assert [u["value"] for u in refused] == [10.0]
        assert "not a sample's depth" in refused[0]["why"]

    def test_the_qa_tells_the_engineer_what_to_check(self, tabular):
        doc, _gt = tabular
        engine = FakeEngine([{"final": self._tabular_reading()}])
        result = read_log(doc, [0], engine)
        scale = _flagged(result)
        assert len(scale) == 1 and scale[0]["page"] == 0
        # A gap in the reading, not a refusal.
        assert "refused_by" not in scale[0]
        why = scale[0]["why"]
        assert why.startswith("No depth ruler could be fitted on page 0")
        assert "Check these 9 depth(s) against the printed log" in why
        assert "1 of the 16 sample and test value(s)" in why
        record = result.to_dict()["printed_depths"]
        assert record["pages"] == [0] and len(record["readings"]) == 9

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
        refused = _refused(result)
        assert refused and all("no depth scale was found" in u["why"]
                               for u in refused)
        scale = _flagged(result)
        assert len(scale) == 1
        assert "No depth ruler could be fitted" in scale[0]["why"]
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
        refused = _refused(result)
        assert [u["value"] for u in refused] == [65.0]
        assert "outside what the scale on this log reads" in refused[0]["why"]
        assert [u["page"] for u in _flagged(result)] == [1]


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
        assert _flagged(result) == []
        assert not any("no depth ruler on page" in c["why"]
                       for c in result.changes)
        assert not any("no depth ruler on page" in s.prov.note
                       for s in result.floor.samples)

    @pytest.mark.parametrize("which", ["imperial", "metric"])
    def test_the_floor_is_the_same_with_the_printed_pass_forced_empty(
            self, which, request):
        doc, _gt = request.getfixturevalue(which)
        grid = _grid(doc, [0])
        found = seed_from_grid(grid, [0], "RX")
        forced = seed_from_grid(grid, [0], "RX", printed=PrintedDepths())
        assert found.model_dump() == forced.model_dump()
