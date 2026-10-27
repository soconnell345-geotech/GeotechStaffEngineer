"""RI-130 and RI-131, offline: blow records of more counts than drives, the
model echoing the floor's own depths, and a sample's top read off its
drawn box.

Every grid and record here is a small synthetic stand-in built in this
file, with made-up values; nothing is copied from a real log.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from report_ingest.log_floor import (
    SPT_MAX_DRIVES, echoes_floor, merge_investigations, seed_from_grid,
    too_many_counts,
)
from report_ingest.model import (
    SPT, Investigation, Provenance, Quantity, Sample,
)

_X = {"id": 60.0, "type": 100.0, "blows": 140.0}


class _Grid:
    """What ``seed_from_grid`` reads off planlens' ``log_grid``. ``boxes``
    None is a grid that read no drawn sample boxes at all (an older
    planlens); a list is the boxes it read."""

    def __init__(self, cells, columns, unit="m", boxes=None):
        self.rows = list(cells)
        self.columns = list(columns)
        self.unit = unit
        self.rulers = {p: SimpleNamespace(step=1.0) for p in (0, 1)}
        self.fields, self.field_boxes = {}, {}
        self.layers, self.elevation_rulers = [], {}
        self.warnings = []
        if boxes is not None:
            self.samples = list(boxes)

    def column(self, column_id):
        return next((c for c in self.columns if c.id == column_id), None)


def _column(column_id, header, *names):
    return SimpleNamespace(id=column_id, page=0, header=header,
                           names=tuple(names), unit="")


ID = _column("id", "SAMPLE NO.", "sample_id")
TYPE = _column("type", "TYPE", "sample_type")
BLOWS = _column("blows", "BLOWS/6 IN", "blows")


def _cell(column_id, text, depth, page=0, pitch=0.15, line_pt=9.0):
    """A cell on a sheet printed one line every ``pitch`` of depth."""
    x = _X[column_id]
    y = 100.0 + (line_pt / pitch) * depth
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    return SimpleNamespace(page=page, column_id=column_id, text=text,
                           bbox=(x, y - 3.5, x + 30.0, y + 3.5),
                           depth=depth, depth_top=depth - pitch / 2,
                           depth_bottom=depth + pitch / 2, numbers=numbers,
                           confidence=0.8)


def _box(top, bottom, page=0):
    return SimpleNamespace(top=top, bottom=bottom, page=page, pages=(page,),
                           plus_minus=0.02, label="")


def _seed(cells, columns, pages=(0,), boxes=None):
    qa = []
    inv = seed_from_grid(_Grid(cells, columns, boxes=boxes), list(pages),
                         "SYNTH", qa=qa)
    return inv, qa


def _flags(qa):
    return [row["flag"] for row in qa]


# ---------------------------------------------------------------------------
# RI-130: a blow record of more counts than a driven record has drives
# ---------------------------------------------------------------------------

class TestTooManyCounts:
    def test_the_limit_is_three_increments_and_the_fourth(self):
        assert SPT_MAX_DRIVES == 4
        record = SPT(depth_top=Quantity(value=1.0, unit="m"),
                     blows=[2, 3, 4, 5, 6])
        assert too_many_counts(record)
        record.blows = [2, 3, 4, 5]
        assert not too_many_counts(record)

    def test_the_records_own_lengths_do_not_move_the_limit(self):
        six = Quantity(value=6, unit="in")
        record = SPT(depth_top=Quantity(value=1.0, unit="m"),
                     blows=[2, 3, 4], increments=[six, six, six])
        assert not too_many_counts(record)
        record = SPT(depth_top=Quantity(value=1.0, unit="m"),
                     blows=[2, 3, 4, 5, 6], increments=[six] * 5)
        assert too_many_counts(record)

    def test_one_cell_of_many_counts_is_refused_as_a_blow_record(self):
        cells = [_cell("id", "S-1", 2.0), _cell("blows", "3-4-5-6-7-8-9", 2.0)]
        inv, qa = _seed(cells, [ID, BLOWS])
        assert [r.blows for r in inv.spt] == []
        sample = inv.samples[0]
        assert sample.sample_id == "S-1"
        assert "7 counts read as one blow record" in sample.note
        assert "refused as a blow record" in sample.note
        assert "floor_blows_count" in _flags(qa)

    def test_counts_one_to_a_line_past_four_are_refused(self):
        cells = [_cell("id", "S-2", 3.0)] + [
            _cell("blows", str(v), round(3.0 + 0.075 * k, 4))
            for k, v in enumerate((2, 3, 5, 8, 13))]
        inv, qa = _seed(cells, [ID, BLOWS])
        assert inv.spt == []
        assert "5 counts read as one blow record" in inv.samples[0].note
        assert "floor_blows_count" in _flags(qa)

    @pytest.mark.parametrize("text, blows", [
        ("4-6-7", [4, 6, 7]), ("4-6-7-9", [4, 6, 7, 9])])
    def test_three_drives_and_the_fourth_stay_a_record(self, text, blows):
        cells = [_cell("id", "S-3", 4.0), _cell("blows", text, 4.0)]
        inv, qa = _seed(cells, [ID, BLOWS])
        assert [r.blows for r in inv.spt] == [blows]
        assert "floor_blows_count" not in _flags(qa)

    def test_counts_on_two_sheets_at_one_depth_are_the_first_sheets(self):
        # Two sheets read as one log print counts one to a line at the same
        # depths: the rows join by depth (a continuation sheet restates the
        # sample it broke on), and together they are more than a record's
        # drives. A driven record is printed on one sheet.
        cells = [_cell("id", "S-1", 2.0, page=0)]
        cells += [_cell("blows", str(v), round(2.0 + 0.15 * k, 4), page=0)
                  for k, v in enumerate((3, 5, 6))]
        cells += [_cell("blows", str(v), round(2.0 + 0.15 * k, 4), page=1)
                  for k, v in enumerate((11, 12, 14))]
        inv, qa = _seed(cells, [ID, BLOWS], pages=(0, 1))
        assert [r.blows for r in inv.spt] == [[3, 5, 6]]
        assert "another sheet" in inv.samples[0].note
        assert "page 1 11-12-14" in inv.samples[0].note
        assert "floor_other_sheet" in _flags(qa)

    def _row(self, column, texts, depth=3.0):
        """One printed line of numbers side by side in one column band,
        each its own cell, ten points apart."""
        cells = [_cell("id", "S-6", depth)]
        for k, text in enumerate(texts):
            c = _cell("blows", text, depth)
            c.bbox = (140.0 + 10.0 * k, c.bbox[1], 146.0 + 10.0 * k,
                      c.bbox[3])
            cells.append(c)
        return _seed(cells, [ID, column])

    def test_a_row_under_several_quantities_is_not_drives(self):
        # A sheet with no ruled column edges: one band headed for the
        # drives, the N and the recovery prints them side by side.
        band = _column("blows", "BLOWS / N / REC (cm)", "blows", "n_value",
                       "recovery")
        inv, qa = self._row(band, ["3", "4", "6", "10", "38"])
        assert inv.spt == []
        sample = inv.samples[0]
        assert "not assigned to a quantity" in sample.note
        assert "floor_unassigned" in _flags(qa)
        assert "floor_blows_count" not in _flags(qa)

    def test_a_row_under_the_blows_alone_is_its_drives(self):
        inv, _qa = self._row(BLOWS, ["3", "4", "6"])
        assert [r.blows for r in inv.spt] == [[3, 4, 6]]

    @staticmethod
    def _at(text, depth, x):
        c = _cell("blows", text, depth)
        c.bbox = (x, c.bbox[1], x + 6.0, c.bbox[3])
        return c

    def test_an_n_beside_the_last_drive_leaves_no_part_record(self):
        # Drives one to a line and the N printed on the last drive's line:
        # the band is two printed columns, so none of its numbers is read
        # as a drive -- not the two lines above the pair either.
        band = _column("blows", "BLOWS / N", "blows", "n_value")
        cells = [_cell("id", "S-6", 3.0), self._at("5", 3.0, 140.0),
                 self._at("7", 3.15, 140.0), self._at("9", 3.3, 140.0),
                 self._at("16", 3.3, 160.0)]
        inv, qa = _seed(cells, [ID, band])
        assert [r.blows for r in inv.spt if r.blows] == []
        assert "not assigned to a quantity" in inv.samples[0].note
        assert "floor_unassigned" in _flags(qa)

    def test_lines_after_a_row_of_several_quantities_are_not_drives(self):
        band = _column("blows", "REC (cm) N N1 N2", "recovery", "n_value",
                       "blows")
        cells = [_cell("id", "S-8", 5.0), self._at("4", 5.0, 140.0),
                 self._at("5", 5.0, 150.0), self._at("9", 5.0, 160.0),
                 self._at("33", 5.15, 140.0), self._at("2", 5.3, 140.0)]
        inv, qa = _seed(cells, [ID, band])
        assert [r.blows for r in inv.spt if r.blows] == []
        assert "floor_unassigned" in _flags(qa)

    def test_drives_a_page_break_splits_stay_whole(self):
        # Two drives at the foot of one sheet and the third at the head of
        # the next: three counts in all, one record.
        cells = [_cell("id", "S-5", 5.0, page=0),
                 _cell("blows", "4", 5.0, page=0),
                 _cell("blows", "6", 5.15, page=0),
                 _cell("blows", "9", 5.3, page=1)]
        inv, qa = _seed(cells, [ID, BLOWS], pages=(0, 1))
        assert [r.blows for r in inv.spt] == [[4, 6, 9]]
        assert "floor_other_sheet" not in _flags(qa)


def _prov(method, confidence, note="", bbox=None):
    return Provenance(page=0, method=method, confidence=confidence,
                      note=note, bbox=bbox)


def _inv(spt=(), samples=(), method="grid", confidence=0.6):
    return Investigation(investigation_id="B-9", depth_unit="m",
                         samples=list(samples), spt=list(spt),
                         prov=[_prov(method, confidence)])


class TestTheMergeRefusesThem:
    def test_a_floor_record_of_many_counts_is_not_kept_beside_the_models(
            self):
        floor = _inv(spt=[SPT(depth_top=Quantity(value=1.2, unit="m"),
                              blows=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                              prov=_prov("grid", 0.6))])
        model = _inv(spt=[SPT(depth_top=Quantity(value=1.2, unit="m"),
                              blows=[2, 4, 5], n=9,
                              prov=_prov("model", 0.9))], method="model",
                     confidence=0.9)
        merged, log = merge_investigations(floor, model)
        record = merged.spt[0]
        assert record.blows == [2, 4, 5]
        assert not [a for a in record.prov.alternatives
                    if a.field == "blows"]
        assert not [d for d in log.disagreements
                    if d["what"].endswith(": blows")]
        assert [(r["voter"], r["what"]) for r in log.refused] == [
            ("floor", "driven record at 1.2 m: blows")]
        assert "refused" in log.to_dict()

    def test_a_model_record_of_many_counts_is_refused_too(self):
        floor = _inv(spt=[SPT(depth_top=Quantity(value=2.4, unit="m"),
                              blows=[3, 3, 4], prov=_prov("grid", 0.6))])
        model = _inv(spt=[SPT(depth_top=Quantity(value=2.4, unit="m"),
                              blows=[3, 3, 4, 6, 7, 8],
                              prov=_prov("model", 0.9))], method="model",
                     confidence=0.9)
        merged, log = merge_investigations(floor, model)
        record = merged.spt[0]
        assert record.blows == [3, 3, 4]
        assert not [a for a in record.prov.alternatives
                    if a.field == "blows"]
        assert [r["voter"] for r in log.refused] == ["model"]

    def test_a_record_one_voter_alone_gives_loses_only_its_blows(self):
        floor = _inv()
        model = _inv(spt=[SPT(depth_top=Quantity(value=3.6, unit="m"),
                              blows=[5, 6, 7, 8, 9], n=13,
                              prov=_prov("model", 0.9))], method="model",
                     confidence=0.9)
        merged, log = merge_investigations(floor, model)
        record = merged.spt[0]
        assert record.blows == [] and record.n == 13
        assert "was refused" in record.prov.note
        assert [r["voter"] for r in log.refused] == ["model"]

    def test_nothing_refused_writes_no_new_keys(self):
        floor = _inv(spt=[SPT(depth_top=Quantity(value=1.2, unit="m"),
                              blows=[2, 4, 5], prov=_prov("grid", 0.6))])
        model = _inv(spt=[SPT(depth_top=Quantity(value=1.2, unit="m"),
                              blows=[2, 4, 5], prov=_prov("model", 0.9))])
        _merged, log = merge_investigations(floor, model)
        assert set(log.to_dict()) == {"disagreements", "kept", "added",
                                      "reconciled"}


# ---------------------------------------------------------------------------
# RI-131: the model echoes the floor's own depths
# ---------------------------------------------------------------------------

def _q(value, unit="m"):
    return Quantity(value=value, unit=unit)


class TestEchoes:
    @pytest.mark.parametrize("floor, model, echoed", [
        (2.3456789, 2.3457, True),       # the brief's six figures
        (2.3456789, 2.346, True),        # fewer, still past a log's two
        (2.3456789, 2.3456789, True),    # copied whole
        (2.3456789, 2.35, False),        # as a log prints a depth
        (2.3456789, 2.3458, False),      # a different number
        (2.5, 2.5, False),               # a printed depth agreeing
        (2.4987, 2.5, False),            # a reading of the printed 2.5
    ])
    def test_what_counts_as_the_floors_own_value(self, floor, model,
                                                 echoed):
        assert echoes_floor(_q(floor), _q(model)) is echoed

    def test_another_unit_is_not_an_echo(self):
        assert not echoes_floor(_q(2.3456789, "ft"), _q(2.3457, "m"))

    def _pair(self, floor_top, model_top):
        floor = _inv(samples=[Sample(sample_id="S-7", top=_q(floor_top),
                                     prov=_prov("grid", 0.6))])
        model = _inv(samples=[Sample(sample_id="S-7", top=_q(model_top),
                                     prov=_prov("model", 0.9))],
                     method="model", confidence=0.9)
        return merge_investigations(floor, model)

    def test_an_echoed_top_is_one_reading_the_floors(self):
        merged, log = self._pair(2.3456789, 2.3457)
        sample = merged.samples[0]
        assert sample.top.value == 2.3456789
        assert sample.top.prov.method == "grid"
        assert sample.top.prov.confidence == pytest.approx(0.6)
        # The object's own method and confidence are its top's.
        assert sample.prov.method == "grid"
        assert sample.prov.confidence == pytest.approx(0.6)
        assert "the floor's own top" in sample.prov.note
        assert [e["what"] for e in log.echoed] == ["sample at 2.34568 m: top"]
        assert log.disagreements == []
        # Every other slot both gave is still reconciled; the top is not.
        _printed, agreed = self._pair(2.4987, 2.5)
        assert log.reconciled == agreed.reconciled - 1
        assert "echoed" in log.to_dict()

    def test_a_top_read_as_a_log_prints_it_still_agrees(self):
        merged, log = self._pair(2.4987, 2.5)
        sample = merged.samples[0]
        assert sample.top.value == 2.5
        assert sample.prov.method == "reconciled"
        assert sample.prov.confidence == pytest.approx(0.9)
        assert log.echoed == []
        assert "echoed" not in log.to_dict()


# ---------------------------------------------------------------------------
# RI-131: a sample's top off its drawn edge
# ---------------------------------------------------------------------------

class TestTheDrawnTop:
    def _short_box(self, boxes):
        # A box drawn over the drive, the sample's number printed a little
        # under it: the box does not take every row of the sample.
        cells = [_cell("blows", '50/4"', 6.2), _cell("id", "S-4", 6.6)]
        return _seed(cells, [ID, BLOWS], boxes=boxes)

    def test_a_short_box_draws_its_samples_top(self):
        inv, _qa = self._short_box([_box(6.0, 6.3)])
        sample = inv.samples[0]
        assert sample.top.value == pytest.approx(6.0)
        assert sample.bottom is None
        assert "top from the top edge of the sample box" in sample.prov.note
        assert inv.spt[0].depth_top.value == pytest.approx(6.0)

    def test_a_box_round_every_row_gives_the_interval_as_before(self):
        inv, _qa = self._short_box([_box(6.0, 6.7)])
        sample = inv.samples[0]
        assert (sample.top.value, sample.bottom.value) == (
            pytest.approx(6.0), pytest.approx(6.7))
        assert "interval from the sample box drawn" in sample.prov.note

    def test_no_box_for_the_sample_keeps_its_row_and_says_so(self):
        inv, _qa = self._short_box([_box(9.0, 9.4)])
        sample = inv.samples[0]
        assert sample.top.value == pytest.approx(6.2)
        assert "no sample box drawn for it was found" in sample.prov.note

    def test_a_grid_that_reads_no_boxes_is_read_as_today(self):
        inv, _qa = self._short_box(None)
        sample = inv.samples[0]
        assert sample.top.value == pytest.approx(6.2)
        assert "sample box" not in sample.prov.note

    @staticmethod
    def _first_row_in(box, row, stray, unit="ft", step=None):
        # The sample's first row, and a cell of the same sample printed just
        # under the box, so that no box takes the group whole. A log in feet
        # printed a line every 0.5 ft. ``step`` is the rulers' step, for a
        # log of no known unit.
        pitch = 0.5 if unit == "ft" or (step or 0) >= 2 else 0.15
        cells = [_cell("blows", "6-8-11", row, pitch=pitch),
                 _cell("id", "S-9", stray, pitch=pitch)]
        grid = _Grid(cells, [ID, BLOWS], unit=unit, boxes=[box])
        if step is not None:
            grid.rulers = {p: SimpleNamespace(step=step) for p in (0, 1)}
        return seed_from_grid(grid, [0], "SYNTH", qa=[])

    @pytest.mark.parametrize("step, box, row, stray, top", [
        (5.0, (20.0, 24.0), 23.6, 24.3, 23.6),      # 3.6 units: refused
        (5.0, (30.0, 32.0), 30.6, 32.05, 30.0),     # 0.6 units: taken
        (1.0, (5.0, 5.9), 5.7, 6.0, 5.7),           # 0.7 units: refused
        (1.0, (5.0, 5.7), 5.55, 5.8, 5.0),          # 0.55 units: taken
        # Rulers stepping by 2 do not make it feet: were this log metric, a
        # 24 in reach read as 2 ft would let the top move 1.2 m up.
        (2.0, (10.0, 11.5), 11.2, 11.62, 11.2)])
    def test_a_log_of_no_known_unit_reads_the_reach_in_its_own_unit(
            self, step, box, row, stray, top):
        inv = self._first_row_in(_box(*box), row=row, stray=stray, unit="",
                                 step=step)
        assert inv.depth_unit in ("", None)
        assert inv.samples[0].top.value == pytest.approx(top)

    def test_a_gap_drawn_with_the_next_box_does_not_move_the_top(self):
        # A box 4 ft tall that is the gap above a sample drawn together with
        # the sample's own box (a line planlens did not find): the sample's
        # first row lies near its bottom, its top edge more than 24 in above.
        inv = self._first_row_in(_box(20.0, 24.0), row=23.6, stray=24.3)
        sample = inv.samples[0]
        assert sample.top.value == pytest.approx(23.6)
        assert "no sample box drawn for it was found" in sample.prov.note

    def test_a_drive_box_with_the_row_a_line_below_its_top_is_taken(self):
        inv = self._first_row_in(_box(30.0, 32.0), row=30.6, stray=32.05)
        sample = inv.samples[0]
        assert sample.top.value == pytest.approx(30.0)
        assert "top from the top edge" in sample.prov.note

    def test_a_short_box_with_the_row_at_its_bottom_is_taken(self):
        inv = self._first_row_in(_box(40.0, 40.4), row=40.4, stray=40.9)
        assert inv.samples[0].top.value == pytest.approx(40.0)

    def test_the_reach_is_24_in_in_metres_too(self):
        inv = self._first_row_in(_box(5.0, 5.7), row=5.55, stray=5.8,
                                 unit="m")
        assert inv.samples[0].top.value == pytest.approx(5.0)
        inv = self._first_row_in(_box(5.0, 5.9), row=5.7, stray=6.0,
                                 unit="m")
        assert inv.samples[0].top.value == pytest.approx(5.7)

    def test_boxes_read_only_on_another_sheet_say_nothing_here(self):
        # The grid read boxes on page 1 alone; the sample is on page 0.
        inv, _qa = self._short_box([_box(9.0, 9.4, page=1)])
        assert "sample box" not in inv.samples[0].prov.note
        inv, _qa = self._short_box([])
        assert "sample box" not in inv.samples[0].prov.note

    def test_a_box_another_sample_takes_whole_is_not_borrowed(self):
        # The first sample's box takes its every row; the second sample
        # has no box of its own and keeps its row.
        cells = [_cell("id", "S-1", 1.0), _cell("blows", "3-4-5", 1.0),
                 _cell("id", "S-2", 2.5), _cell("blows", "6-7-8", 2.5)]
        inv, _qa = _seed(cells, [ID, BLOWS], boxes=[_box(0.9, 1.4)])
        tops = sorted(s.top.value for s in inv.samples)
        assert tops == [pytest.approx(0.9), pytest.approx(2.5)]


# ---------------------------------------------------------------------------
# the scorecard: echoes and refusals beside the reconciled count
# ---------------------------------------------------------------------------

def test_the_scorecard_counts_echoes_and_refusals_beside_reconciled(
        monkeypatch):
    import report_ingest.log_reader as reader
    import report_ingest.log_scoring as scoring
    import report_ingest.visual_scales as visual

    truth = {"id": "SYNTH_p1", "pages": [1]}
    monkeypatch.setattr(visual, "log_grid_for",
                        lambda *a, **k: (object(), {}))
    monkeypatch.setattr(scoring, "score_grid", lambda truth, grid:
                        scoring.LogScore(log_id=truth["id"], report="SYNTH",
                                         stage="grid"))
    monkeypatch.setattr(reader, "read_log", lambda *a, **k: SimpleNamespace(
        investigation=_inv(), cost={}, model_calls=1, unresolved=[],
        changes=[], warnings=[], reconciled=5,
        echoed=[{"what": "a sample: top"}] * 3,
        refused=[{"what": "a driven record: blows"}]))
    _before, after = scoring.score_one_log(truth, object(), None,
                                           report_id="SYNTH")
    blob = after.to_dict()
    assert (blob["reconciled"], blob["echoed"], blob["refused"]) == (5, 3, 1)
