"""The log floor's reading of the grid (BACKLOG RI-04), offline.

What the floor reads off ``log_grid`` before any model call: which values
belong to which sample, which cells are not what their column says, an N
from a standard SPT's printed drives, and a model's N checked against them.
Every grid here is a small synthetic stand-in built in this file; nothing is
copied from a real log.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from report_ingest.floor import MergeLog
from report_ingest.log_floor import (
    GROUP_SPAN_M, N_RULE_SOURCE, header_increment, merge_investigations,
    n_from_drives, seed_from_grid, serialise_seed,
)
from report_ingest.model import (
    SPT, Investigation, Provenance, Quantity, Sample,
)


# ---------------------------------------------------------------------------
# a ruled sheet in metres, one row per cell
# ---------------------------------------------------------------------------

#: Where each column sits across the sheet, so a cell's box is where its
#: column is. One metre of depth is 40 points down the page.
_X = {"id": 60.0, "type": 100.0, "blows": 140.0, "n": 180.0, "rec": 220.0,
      "mc": 260.0, "lim": 300.0, "graph": 340.0, "tests": 380.0,
      "remarks": 420.0, "fines": 460.0, "pp": 500.0}


class _Grid:
    """What ``seed_from_grid`` reads off planlens' ``log_grid``."""

    def __init__(self, cells, columns, unit="m"):
        self.rows = list(cells)
        self.columns = list(columns)
        self.unit = unit
        self.rulers = {p: SimpleNamespace(step=1.0) for p in (0, 1)}
        self.fields, self.field_boxes = {}, {}
        self.layers, self.samples, self.elevation_rulers = [], [], {}
        self.warnings = []

    def column(self, column_id):
        return next((c for c in self.columns if c.id == column_id), None)


def _column(column_id, header, *names, unit=""):
    return SimpleNamespace(id=column_id, page=0, header=header,
                           names=tuple(names), unit=unit)


def _cell(column_id, text, depth, page=0):
    x = _X[column_id]
    y = 100.0 + 40.0 * depth
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    return SimpleNamespace(page=page, column_id=column_id, text=text,
                           bbox=(x, y - 4.0, x + 30.0, y + 4.0),
                           depth=depth, depth_top=depth - 0.1,
                           depth_bottom=depth + 0.1, numbers=numbers,
                           confidence=0.9)


ID = _column("id", "SAMPLE NO.", "sample_id")
TYPE = _column("type", "TYPE", "sample_type")
BLOWS = _column("blows", "BLOWS/6 IN", "blows")
MC = _column("mc", "MOISTURE CONTENT (%)", "water_content")
TESTS = _column("tests", "TESTS", "tests")
REMARKS = _column("remarks", "REMARKS", "remarks")


def _seed(cells, columns, unit="m", pages=(0,)):
    qa = []
    inv = seed_from_grid(_Grid(cells, columns, unit), list(pages), "SYNTH",
                         qa=qa)
    return inv, qa


def _at(inv, depth):
    """The sample whose top is at this depth."""
    found = [s for s in inv.samples
             if s.top is not None and abs(s.top.value - depth) < 1e-6]
    assert len(found) == 1, [s.top.value for s in inv.samples]
    return found[0]


def _flags(qa):
    return [row["flag"] for row in qa]


def _three(columns=(ID, TYPE, BLOWS, MC)):
    """Samples S-1..S-3 at 1.5, 3.0 and 4.5 m, each with its id, its
    sampler code and its drives on one line."""
    cells = []
    for k, depth in enumerate((1.5, 3.0, 4.5), start=1):
        cells += [_cell("id", f"S-{k}", depth), _cell("type", "SS", depth),
                  _cell("blows", f"{k + 2}-{k + 3}-{k + 4}", depth)]
    return cells, list(columns)


# ---------------------------------------------------------------------------
# part 1: a value goes to the sample the page's geometry points at
# ---------------------------------------------------------------------------

class TestBinding:
    def test_a_value_printed_above_its_sample_is_that_samples(self):
        cells, columns = _three()
        # Each moisture content printed half a metre above its sample's line:
        # further than one sample's span, so not on the sample's rows.
        cells += [_cell("mc", "12.5", 1.0), _cell("mc", "18.0", 2.5),
                  _cell("mc", "21.0", 4.0)]
        inv, qa = _seed(cells, columns)
        assert [s.top.value for s in inv.samples] == [1.5, 3.0, 4.5]
        assert [s.water_content for s in inv.samples] == [12.5, 18.0, 21.0]
        assert [s.sample_id for s in inv.samples] == ["S-1", "S-2", "S-3"]
        assert qa == []

    def test_the_value_does_not_move_its_sample(self):
        cells, columns = _three()
        cells.append(_cell("mc", "18.0", 2.5))
        inv, _ = _seed(cells, columns)
        sample = _at(inv, 3.0)
        assert sample.water_content == 18.0
        # Read off the sample's own lines, it does not join the sample's
        # cells: the confidence and the id are the sample's.
        assert sample.prov.confidence == pytest.approx(0.9)

    def test_a_value_on_the_sample_rows_reads_as_before(self):
        cells, columns = _three()
        cells.append(_cell("mc", "18.0", 3.2))
        inv, qa = _seed(cells, columns)
        assert _at(inv, 3.0).water_content == 18.0
        assert qa == []

    def test_a_value_midway_between_two_samples_is_not_bound(self):
        cells, columns = _three()
        cells.append(_cell("mc", "18.0", 2.25))
        inv, qa = _seed(cells, columns)
        assert _at(inv, 1.5).water_content is None
        assert _at(inv, 3.0).water_content is None
        # It stays where it is printed, doubted, and QA names it.
        loose = _at(inv, 2.25)
        assert loose.water_content == 18.0
        assert loose.prov.confidence == pytest.approx(0.3)
        assert "not bound to one sample" in loose.note
        assert _flags(qa) == ["floor_unbound"]
        assert "18.0" in qa[0]["value"]

    def test_a_value_far_from_every_sample_is_not_bound(self):
        cells, columns = _three()
        cells.append(_cell("mc", "18.0", 8.0))
        inv, qa = _seed(cells, columns)
        assert _at(inv, 8.0).water_content == 18.0
        assert _flags(qa) == ["floor_unbound"]

    def test_a_value_on_the_wrong_side_for_its_column_is_not_bound(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, MC))
        cells += [_cell("id", "S-4", 6.0), _cell("type", "SS", 6.0),
                  _cell("blows", "6-7-8", 6.0)]
        # This column prints its values above their samples; the third one
        # is nearest the sample ABOVE it, which the column never does.
        cells += [_cell("mc", "11.0", 0.95), _cell("mc", "12.0", 2.45),
                  _cell("mc", "13.0", 5.05)]
        inv, qa = _seed(cells, columns)
        assert _at(inv, 1.5).water_content == 11.0
        assert _at(inv, 3.0).water_content == 12.0
        assert _at(inv, 4.5).water_content is None
        assert _at(inv, 6.0).water_content is None
        assert _at(inv, 5.05).water_content == 13.0
        assert _flags(qa) == ["floor_unbound"]

    def test_a_labelled_result_under_tests_binds_to_its_sample(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, MC, TESTS))
        cells.append(_cell("tests", "qu = 95 kPa", 2.5))
        inv, qa = _seed(cells, columns)
        assert _at(inv, 3.0).qu.value == 95.0
        assert _at(inv, 3.0).qu.unit.lower() == "kpa"
        assert qa == []

    def test_a_result_under_tests_is_never_a_sample_on_its_own(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, MC, TESTS))
        cells.append(_cell("tests", "LL = 41", 9.0))
        inv, qa = _seed(cells, columns)
        assert len(inv.samples) == 3
        assert all(s.liquid_limit is None for s in inv.samples)
        assert _flags(qa) == ["floor_unbound"]
        assert "LL = 41" in qa[0]["value"]

    def test_an_unlabelled_remark_is_not_read(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, MC, REMARKS))
        cells.append(_cell("remarks", "Driller noted 40 mm gravel", 3.0))
        inv, qa = _seed(cells, columns)
        assert len(inv.samples) == 3
        assert qa == []

    def test_a_sheet_with_only_values_reads_as_before(self):
        cells = [_cell("mc", "12.5", 1.0), _cell("mc", "18.0", 2.5)]
        inv, qa = _seed(cells, [MC])
        assert [s.water_content for s in inv.samples] == [12.5, 18.0]
        assert all(s.prov.confidence == pytest.approx(0.9)
                   for s in inv.samples)
        assert qa == []

    def test_two_ids_close_together_are_two_samples(self):
        cells = [_cell("id", "S-1", 1.5), _cell("blows", "3-4-5", 1.5),
                 _cell("id", "S-2", 1.8), _cell("blows", "6-7-8", 1.8)]
        inv, _ = _seed(cells, [ID, BLOWS])
        assert [s.sample_id for s in inv.samples] == ["S-1", "S-2"]
        assert [r.blows for r in inv.spt] == [[3, 4, 5], [6, 7, 8]]

    def test_the_nearer_of_two_samples_takes_a_cell(self):
        # Three 6 in drives, 18 in (FHWA NHI-06-088 Vol. I, p. 3-51).
        assert GROUP_SPAN_M == pytest.approx(3 * 6 * 0.0254)
        cells = [_cell("id", "S-1", 1.5), _cell("id", "S-2", 1.9),
                 _cell("mc", "22.0", 1.85)]
        inv, _ = _seed(cells, [ID, MC])
        assert _at(inv, 1.5).water_content is None
        assert _at(inv, 1.9).water_content == 22.0


# ---------------------------------------------------------------------------
# part 2: the grid's mistakes, caught
# ---------------------------------------------------------------------------

class TestWhatACellIs:
    def test_words_in_a_sample_column_are_not_a_sample_or_its_id(self):
        cells = [_cell("id", "grey silty SAND, moist", 1.4),
                 _cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("id", "Bottom of hole at 6.2 m", 6.2)]
        inv, qa = _seed(cells, [ID, TYPE])
        assert [s.sample_id for s in inv.samples] == ["S-1"]
        assert qa == []

    def test_a_sampler_named_in_words_is_still_the_sampler(self):
        cells = [_cell("type", "Split Spoon", 1.5), _cell("id", "S-1", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE])
        assert [s.sample_id for s in inv.samples] == ["S-1"]

    def test_a_cell_with_words_is_not_its_columns_number(self):
        data = _column("rec", "DATA", "recovery", "rqd")
        cells = [_cell("rec", "S-1, SS", 1.5), _cell("rec", "Run = 3.5 ft",
                                                      3.0)]
        inv, _ = _seed(cells, [data])
        first = _at(inv, 1.5)
        assert first.sample_id == "S-1"
        assert first.kind == "spt"
        assert first.recovery_percent is None and first.rqd_percent is None
        assert _at(inv, 3.0).rqd_percent is None

    def test_a_label_and_its_value_on_two_lines_are_read_together(self):
        fines = _column("fines", "FINES", "fines")
        cells, columns = _three(columns=(ID, TYPE, BLOWS, fines))
        cells += [_cell("fines", "% Passing #200", 3.0),
                  _cell("fines", "= 41.5", 3.15)]
        inv, _ = _seed(cells, columns)
        assert _at(inv, 3.0).fines_percent == 41.5

    def test_a_bare_label_is_not_a_number(self):
        limits = _column("lim", "LL", "liquid_limit")
        cells, columns = _three(columns=(ID, TYPE, BLOWS, limits))
        cells.append(_cell("lim", "% Passing #200", 3.0))
        inv, _ = _seed(cells, columns)
        assert all(s.liquid_limit is None for s in inv.samples)

    def test_a_header_printing_two_quantities_as_a_and_b_in_brackets(self):
        rec = _column("rec", "REC % (RQD %)", "recovery", "rqd")
        cells = [_cell("id", "C-1", 2.0), _cell("type", "NQ", 2.0),
                 _cell("rec", "67", 2.0), _cell("rec", "(45)", 2.1)]
        inv, _ = _seed(cells, [ID, TYPE, rec])
        core = _at(inv, 2.0)
        assert core.recovery_percent == 67.0
        assert core.rqd_percent == 45.0

    def test_an_n_in_brackets_under_blows_per_6_in_n(self):
        blows = _column("blows", "BLOWS PER 6 IN (N)", "blows", "n_value")
        cells = [_cell("id", "S-1", 1.5), _cell("blows", "3-5-9", 1.5),
                 _cell("blows", "(14)", 1.6)]
        inv, _ = _seed(cells, [ID, blows])
        assert inv.spt[0].blows == [3, 5, 9]
        assert inv.spt[0].n == 14
        assert inv.spt[0].n_basis == ""

    def test_a_lone_number_under_several_quantities_is_kept_unassigned(self):
        graph = _column("graph", "W% 10 20 30 / LL PL / N", "plastic_limit",
                        "water_content", "liquid_limit", "n_value", "blows")
        cells = [_cell("id", "S-1", 1.5), _cell("blows", "3-4-5", 1.5),
                 _cell("graph", "31.4", 1.5)]
        inv, qa = _seed(cells, [ID, BLOWS, graph])
        sample = _at(inv, 1.5)
        assert sample.water_content is None and sample.liquid_limit is None
        assert sample.plastic_limit is None
        assert "31.4" in sample.note
        assert _flags(qa) == ["floor_unassigned"]
        assert "31.4" in serialise_seed(inv)

    def test_a_decimal_comma_is_a_decimal_point(self):
        cells, columns = _three()
        cells.append(_cell("mc", "31,4", 3.0))
        inv, _ = _seed(cells, columns)
        assert _at(inv, 3.0).water_content == pytest.approx(31.4)

    def test_a_percentage_past_100_is_doubted_not_clamped(self):
        rec = _column("rec", "RECOVERY (%)", "recovery")
        cells = [_cell("id", "S-1", 1.5), _cell("rec", "120", 1.5)]
        inv, qa = _seed(cells, [ID, rec])
        sample = _at(inv, 1.5)
        assert sample.recovery_percent is None
        assert "120" in sample.note
        assert sample.prov.confidence == pytest.approx(0.3)
        assert _flags(qa) == ["floor_doubtful"]

    @pytest.mark.parametrize("text, why", [
        ("30-42-12", "plastic limit 42 above liquid limit 30"),
        ("42-21-30", "PI printed 30"),
    ])
    def test_limits_that_contradict_each_other_are_doubted(self, text, why):
        limits = _column("lim", "LL-PL-PI", "liquid_limit", "plastic_limit",
                         "plasticity_index")
        cells = [_cell("id", "S-1", 1.5), _cell("lim", text, 1.5)]
        inv, qa = _seed(cells, [ID, limits])
        sample = _at(inv, 1.5)
        # The values stay where they were read, at a lower confidence.
        assert sample.liquid_limit is not None
        assert why in sample.note
        assert sample.prov.confidence == pytest.approx(0.3)
        assert _flags(qa) == ["floor_doubtful"]

    def test_consistent_limits_are_not_doubted(self):
        limits = _column("lim", "LL-PL-PI", "liquid_limit", "plastic_limit",
                         "plasticity_index")
        cells = [_cell("id", "S-1", 1.5), _cell("lim", "42-21-21", 1.5)]
        inv, qa = _seed(cells, [ID, limits])
        sample = _at(inv, 1.5)
        assert (sample.liquid_limit, sample.plastic_limit,
                sample.plasticity_index) == (42.0, 21.0, 21.0)
        assert sample.note == "" and qa == []

    def test_a_sample_out_of_order_down_the_hole_is_doubted(self):
        cells = [_cell("id", "S-1", 1.5), _cell("id", "S-7", 3.0),
                 _cell("id", "S-3", 4.5), _cell("id", "S-4", 6.0)]
        inv, qa = _seed(cells, [ID])
        assert len(inv.samples) == 4
        odd = _at(inv, 3.0)
        assert "S-7" in odd.note and odd.prov.confidence == pytest.approx(0.3)
        assert _at(inv, 4.5).note == ""
        assert _flags(qa) == ["floor_order"]

    def test_drives_alone_do_not_make_a_sample_an_spt(self):
        cells = [_cell("id", "S-1", 1.5), _cell("blows", "3-4-5", 1.5)]
        inv, _ = _seed(cells, [ID, BLOWS])
        assert inv.samples[0].kind == "other"
        assert inv.spt[0].blows == [3, 4, 5]
        assert inv.spt[0].n is None

    def test_a_level_in_a_remark_is_not_a_water_content(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, REMARKS))
        cells.append(_cell("remarks", "W = 2.5 m", 3.0))
        inv, qa = _seed(cells, columns)
        assert all(s.water_content is None for s in inv.samples)

    def test_a_pocket_penetrometer_qu_is_a_pocket_reading(self):
        cells, columns = _three(columns=(ID, TYPE, BLOWS, REMARKS))
        cells.append(_cell("remarks", "qu = 75 kPa (PP)", 3.0))
        inv, _ = _seed(cells, columns)
        sample = _at(inv, 3.0)
        assert sample.qu is None
        assert sample.pocket_pen.value == 75.0


# ---------------------------------------------------------------------------
# part 3: N from a standard SPT's printed drives
# ---------------------------------------------------------------------------

class TestNFromDrives:
    def test_the_rule_is_cited_from_the_printed_manual(self):
        assert "NHI-06-088" in N_RULE_SOURCE and "3-51" in N_RULE_SOURCE

    def test_three_six_inch_drives_give_the_second_plus_the_third(self):
        n, basis = n_from_drives([4, 6, 7], sampler="spt",
                                 increment_m=0.1524)
        assert n == 13
        assert "6 + 7 = 13" in basis and N_RULE_SOURCE in basis

    @pytest.mark.parametrize("blows, sampler, increment, interval", [
        ([4, 6, 7], "ring", 0.1524, None),          # not a standard sampler
        ([4, 6, 7], "other", 0.1524, None),         # the sampler not printed
        ([4, 6, 7, 9], "spt", 0.1524, None),        # four drives
        ([4, 6], "spt", 0.1524, None),              # two
        ([4, 6, '50/3"'], "spt", 0.1524, None),     # a refusal
        ([4, 6, 7], "spt", 0.0762, None),           # 3 in drives
        ([4, 6, 7], "spt", 0.3048, None),           # a foot
        ([4, 6, 7], "spt", None, None),             # no length printed
        ([4, 6, 7], "spt", None, 0.61),             # 24 in over three
    ])
    def test_never_outside_the_printed_case(self, blows, sampler, increment,
                                            interval):
        assert n_from_drives(blows, sampler=sampler, increment_m=increment,
                             interval_m=interval) is None

    def test_an_interval_of_three_drives_says_the_length(self):
        n, basis = n_from_drives([2, 3, 4], sampler="spt", interval_m=0.457)
        assert n == 7 and "interval" in basis

    @pytest.mark.parametrize("header, metres", [
        ("BLOWS/6 IN", 0.1524), ('Blows per 6"', 0.1524),
        ("blows/15 cm", 0.15), ("blows/150 mm", 0.15), ("BLOWS/FT", 0.3048),
        ("BLOWS", None), ("N", None),
        # RI-44: an increment the header states wins over the N's unit.
        ("Blows per 6-inch", 0.1524),
        ('BLOWS / FOOT (6" INCREMENTS)', 0.1524),
        ("SPT N-VALUE (blows/ft) / Blows per 6 in", 0.1524),
        ("N (blows/ft)", 0.3048), ("SAMPLE NO. / BLOWS / REC", None),
        # The hammer's weight and fall are not a drive.
        ("Blows/ft (140 lb / 30 in drop)", 0.3048),
        ("140 lb / 30 in. drop  BLOWS/6 IN", 0.1524),
    ])
    def test_the_increment_a_header_prints(self, header, metres):
        got = header_increment(header)
        assert (got is None) if metres is None else got == pytest.approx(
            metres)

    def test_the_floor_computes_it_for_an_spt_with_a_printed_increment(self):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        record = inv.spt[0]
        assert record.n == 13
        assert "BLOWS/6 IN" in record.n_basis
        assert "N 13 (not printed; the 2nd + 3rd" in serialise_seed(inv)

    def test_a_metric_log_of_15_cm_drives(self):
        blows = _column("blows", "BLOWS/15 cm", "blows")
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SPT", 1.5),
                 _cell("blows", "5-8-10", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, blows])
        assert inv.spt[0].n == 18

    @pytest.mark.parametrize("header, sampler, drives", [
        ("BLOWS/6 IN", "R", "4-6-7"),        # a ring sampler, driven too
        ("BLOWS/FT", "SS", "4-6-7"),         # the drives' length not 6 in
        ("BLOWS/6 IN", "SS", "4-6-7-9"),     # four drives
        ("BLOWS/6 IN", "SS", '4-6-50/3"'),   # a refusal
    ])
    def test_the_floor_does_not_compute_it_otherwise(self, header, sampler,
                                                     drives):
        blows = _column("blows", header, "blows")
        cells = [_cell("id", "S-1", 1.5), _cell("type", sampler, 1.5),
                 _cell("blows", drives, 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, blows])
        assert inv.spt[0].n is None and inv.spt[0].n_basis == ""

    def test_a_printed_n_is_kept_as_printed(self):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5), _cell("blows", "N=15", 1.6)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[0].n == 15 and inv.spt[0].n_basis == ""

    def test_a_diggs_file_carries_the_drives_not_a_computed_n(self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, write_diggs,
        )
        from report_ingest.model import Project

        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        inv.investigation_id = "B-1"
        assert inv.spt[0].n == 13
        xml = write_diggs(inv, Project(name="R"))
        assert "Blow count 2" in xml and "N-Value" not in xml
        ok, diffs = diggs_roundtrip_gate(xml, [inv], project=Project(name="R"))
        assert ok, diffs


# ---------------------------------------------------------------------------
# the review of 06a5c2c: each case it found, kept found
# ---------------------------------------------------------------------------

DATA = _column("rec", "DATA", "sample_id", "blows", "recovery")
NUMBERED = _column("id", "SAMPLE NO.", "sample_id")


class TestTheReviewsCases:
    def test_a_data_column_stacking_an_id_and_its_drives_is_one_sample(self):
        cells = [_cell("rec", "S-3", 3.0), _cell("rec", "4", 3.15),
                 _cell("rec", "6", 3.3), _cell("rec", "7", 3.45)]
        inv, qa = _seed(cells, [DATA])
        assert [s.sample_id for s in inv.samples] == ["S-3"]
        assert [r.blows for r in inv.spt] == [[4, 6, 7]]
        # The id is the sample's name, not a recovery of 3 %.
        assert inv.samples[0].recovery_percent is None
        assert qa == []

    @pytest.mark.parametrize("word", ["NR", "WOH", "WOR", "REC", "N/A"])
    def test_a_word_under_sample_no_is_never_an_id(self, word):
        cells = [_cell("id", "S-3", 3.0), _cell("id", word, 3.15)]
        inv, _ = _seed(cells, [NUMBERED])
        assert [s.sample_id for s in inv.samples] == ["S-3"]

    def test_a_bare_number_under_a_prefixed_id_is_not_a_second_id(self):
        cells = [_cell("id", "S-3", 3.0), _cell("id", "4", 3.2)]
        inv, _ = _seed(cells, [NUMBERED])
        assert [s.sample_id for s in inv.samples] == ["S-3"]

    def test_numbers_are_ids_where_the_header_numbers_the_samples(self):
        cells = [_cell("id", "1", 1.5), _cell("id", "2", 1.8)]
        inv, _ = _seed(cells, [NUMBERED])
        assert [s.sample_id for s in inv.samples] == ["1", "2"]

    def test_a_value_midway_inside_the_span_is_not_bound(self):
        cells = [_cell("id", "S-1", 1.5), _cell("id", "S-2", 2.1),
                 _cell("mc", "18.0", 1.8)]
        inv, qa = _seed(cells, [ID, MC])
        assert _at(inv, 1.5).water_content is None
        assert _at(inv, 2.1).water_content is None
        assert _at(inv, 1.8).prov.confidence == pytest.approx(0.3)
        assert _flags(qa) == ["floor_unbound"]

    def test_a_sample_on_the_sheet_before_still_competes(self):
        cells = [_cell("id", "S-1", 9.0, page=0),
                 _cell("id", "S-2", 10.6, page=1),
                 _cell("mc", "18.0", 9.9, page=1)]
        inv, qa = _seed(cells, [ID, MC], pages=(0, 1))
        assert _at(inv, 9.0).water_content is None
        assert _at(inv, 10.6).water_content is None
        assert _at(inv, 9.9).water_content == 18.0
        assert _flags(qa) == ["floor_unbound"]

    def test_a_drive_printed_as_woh_stays_and_stops_the_n(self):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "WOH", 1.5), _cell("blows", "1", 1.62),
                 _cell("blows", "2", 1.74), _cell("blows", "3", 1.86)]
        inv, qa = _seed(cells, [ID, TYPE, BLOWS])
        record = inv.spt[0]
        assert record.blows == ["WOH", 1, 2, 3]
        assert record.n is None and record.n_basis == ""
        assert "WOH" in inv.samples[0].note
        assert _flags(qa) == ["floor_drives"]

    def test_a_drive_past_the_printed_halt_is_a_refusal_with_no_n(self):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "10-25-60", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        record = inv.spt[0]
        assert record.blows == [10, 25, 60]
        assert record.n is None and record.refusal
        assert "exceeds 50" in inv.samples[0].note
        assert "p. 3-51" in inv.samples[0].note

    def test_fifty_in_a_drive_is_not_past_the_halt(self):
        n, _ = n_from_drives([10, 25, 50], sampler="spt", increment_m=0.1524)
        assert n == 75

    def test_a_low_dry_unit_weight_is_not_doubted(self):
        dry = _column("mc", "DRY DENSITY (pcf)", "dry_unit_weight")
        cells = [_cell("id", "S-1", 1.5), _cell("mc", "30", 1.5)]
        inv, qa = _seed(cells, [ID, dry])
        sample = _at(inv, 1.5)
        assert sample.dry_unit_weight.value == 30.0
        assert sample.note == "" and qa == []

    @pytest.mark.parametrize("drives, n", [
        ("50/4\"", None), ('12-50/3"', None), ("4-6", None),
        ("4-6-7-9", None), ("4-6-7", 13),
    ])
    def test_the_cases_the_review_confirmed(self, drives, n):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", drives, 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[0].n == n

    @pytest.mark.parametrize("code", ["R", "MC", "CA"])
    def test_ring_and_california_samplers_get_no_n(self, code):
        cells = [_cell("id", "S-1", 1.5), _cell("type", code, 1.5),
                 _cell("blows", "4-6-7", 1.5)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.samples[0].kind == "ring"
        assert inv.spt[0].n is None


# ---------------------------------------------------------------------------
# RI-44: a column headed as both the sample numbers and the blows, printed
# at the scales logs are printed at
# ---------------------------------------------------------------------------

BOTH = _column("blows", "SAMPLE NO. / BLOWS / REC", "sample_id", "blows",
               "recovery")
BOTH6 = _column("blows", "SAMPLE NO. / BLOWS/6 IN / REC", "sample_id",
                "blows", "recovery")
NO_REC6 = _column("blows", "SAMPLE NO. / BLOWS/6 IN", "sample_id", "blows")


def _sheet(pitch, line_pt=9.0, text_pt=7.0):
    """A cell maker for a sheet printed at a real scale: one line of text,
    ``line_pt`` points apart, is ``pitch`` of depth. 0.15 m a line is about
    1:50 with 9 pt lines; 1 in = 5 ft with 9 pt lines is 0.625 ft a line.
    (The fixture's ``_cell`` puts lines 0.05 m apart: no log prints so.)"""
    per_unit = line_pt / pitch

    def cell(column_id, text, depth, page=0):
        x = _X[column_id]
        y = 100.0 + per_unit * depth
        numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
        return SimpleNamespace(page=page, column_id=column_id, text=text,
                               bbox=(x, y - text_pt / 2, x + 30.0,
                                     y + text_pt / 2),
                               depth=depth, depth_top=depth - pitch / 2,
                               depth_bottom=depth + pitch / 2,
                               numbers=numbers, confidence=0.9)
    return cell


def _stacked(rows, pitch, column=BOTH6, unit="m", typed=True, at_line=0,
             boxes=()):
    """Seed a sheet whose ``column`` prints each sample as a stack of lines,
    one line every ``pitch``; ``rows`` is ``[(top, [lines])]``. With
    ``typed`` a TYPE column prints SS on the stack's line ``at_line``."""
    cell = _sheet(pitch)
    cells = []
    for top, lines in rows:
        cells += [cell("blows", text, round(top + j * pitch, 4))
                  for j, text in enumerate(lines)]
        if typed:
            cells.append(cell("type", "SS", round(top + at_line * pitch, 4)))
    grid = _Grid(cells, [column, TYPE] if typed else [column], unit)
    grid.samples = list(boxes)
    qa = []
    return seed_from_grid(grid, [0], "SYNTH", qa=qa), qa


#: Three samples as the review printed them: number, three drives, recovery.
_THREE = [("1", [4, 6, 7], "12"), ("2", [5, 8, 9], "14"),
          ("3", [10, 12, 15], "16")]


def _rows(samples=_THREE, pitch=0.15, rec=True, blank=0, start=1.0,
          spacing=None):
    rows, top = [], start
    for sid, drives, recovery in samples:
        lines = [sid] + [str(d) for d in drives] + ([recovery] if rec else [])
        rows.append((round(top, 4), lines))
        top += max(spacing or 0.0, (len(lines) + blank) * pitch)
    return rows


def _ids(inv):
    return [s.sample_id for s in inv.samples]


class TestTheStackedColumn:
    def test_the_reviews_back_to_back_stacks_read_whole(self):
        # Truth: S1 [4,6,7] N 13, S2 [5,8,9] N 17, S3 [10,12,15] N 27.
        inv, qa = _stacked(_rows(pitch=0.15), 0.15)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [[4, 6, 7], [5, 8, 9],
                                              [10, 12, 15]]
        assert [r.n for r in inv.spt] == [13, 17, 27]
        assert [s.recovery_percent for s in inv.samples] == [12, 14, 16]
        assert qa == []

    def test_four_lines_under_rec_keep_the_drives_but_give_no_n(self):
        # The fourth line may be a third drive or the recovery.
        inv, qa = _stacked(_rows(pitch=0.17, rec=False), 0.17)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [[4, 6, 7], [5, 8, 9],
                                              [10, 12, 15]]
        assert [r.n for r in inv.spt] == [None, None, None]
        assert _flags(qa) == ["floor_stack_tail"]
        assert "recovery" in inv.samples[0].note

    @pytest.mark.parametrize("pitch", [0.12, 0.15, 0.25, 0.35])
    @pytest.mark.parametrize("rec", [True, False])
    def test_at_real_line_pitches(self, pitch, rec):
        column = BOTH6 if rec else NO_REC6
        rows = _rows(pitch=pitch, rec=rec, spacing=1.5)
        inv, qa = _stacked(rows, pitch, column)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [d for _s, d, _r in _THREE]
        assert [r.n for r in inv.spt] == [13, 17, 27]
        assert qa == []

    @pytest.mark.parametrize("pitch", [0.12, 0.25, 0.35])
    def test_with_no_sampler_code_a_blank_line_parts_the_stacks(self, pitch):
        inv, qa = _stacked(_rows(pitch=pitch, blank=1), pitch, typed=False)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [d for _s, d, _r in _THREE]
        assert qa == []

    @pytest.mark.parametrize("pitch", [0.12, 0.25])
    def test_with_nothing_to_part_them_the_column_reads_as_before(self,
                                                                  pitch):
        inv, qa = _stacked(_rows(pitch=pitch), pitch, typed=False)
        printed = {"4", "5", "6", "7", "8", "9", "10", "12", "14", "15", "16"}
        assert not set(_ids(inv)) & printed
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    def test_a_sampler_code_a_line_below_the_number_starts_nothing(self):
        # SS printed on each stack's second line would start every sample
        # at its first drive; the line above the first one is left over, so
        # the column is read as before.
        inv, qa = _stacked(_rows(pitch=0.15), 0.15, at_line=1)
        assert not set(_ids(inv)) & {"4", "5", "10"}
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    def test_a_number_set_apart_from_its_drives_starts_nothing(self):
        # A blank line under each number: the runs are a number alone and
        # drives alone, and none of them reads as a sample.
        cell = _sheet(0.15)
        cells = []
        for k, (sid, drives, rec) in enumerate(_THREE):
            top = 1.0 + 1.5 * k
            cells.append(cell("blows", sid, top))
            cells += [cell("blows", str(d), round(top + 0.3 + 0.15 * j, 4))
                      for j, d in enumerate(drives + [int(rec)])]
        inv, qa = _seed(cells, [BOTH6])
        assert not set(_ids(inv)) & {"4", "5", "10"}
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    @pytest.mark.parametrize("spacing", [2.5, 5.0])
    def test_a_ft_log_at_one_inch_to_five_feet(self, spacing):
        pitch = 5.0 * 9.0 / 72.0              # 9 pt lines: 0.625 ft a line
        rows = _rows(pitch=pitch, rec=False, start=2.5, spacing=spacing)
        inv, qa = _stacked(rows, pitch, NO_REC6, unit="ft")
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.n for r in inv.spt] == [13, 17, 27]
        assert qa == []

    @pytest.mark.parametrize("typed", [True, False])
    def test_continuous_sampling_every_one_and_a_half_feet(self, typed):
        pitch = 2.0 * 9.0 / 72.0              # 1 in = 2 ft: 0.25 ft a line
        rows = _rows(pitch=pitch, rec=False, start=2.0, spacing=1.5)
        inv, qa = _stacked(rows, pitch, NO_REC6, unit="ft", typed=typed)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [d for _s, d, _r in _THREE]
        assert [r.n for r in inv.spt] == ([13, 17, 27] if typed
                                          else [None] * 3)
        assert qa == []

    def test_a_drawn_box_starts_a_sample(self):
        boxes = [SimpleNamespace(top=t, bottom=t + 0.4572, page=0,
                                 pages=(0,), plus_minus=None, label="")
                 for t in (1.0, 1.48, 1.96)]
        rows = _rows(pitch=0.12, rec=False, spacing=0.48)
        inv, qa = _stacked(rows, 0.12, NO_REC6, typed=False, boxes=boxes)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [d for _s, d, _r in _THREE]
        assert qa == []

    def test_a_stack_that_starts_with_woh_has_no_number(self):
        inv, qa = _stacked([(3.0, ["WOH", "2", "3"])], 0.15, BOTH)
        assert not set(_ids(inv)) & {"2", "3"}
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    def test_a_refusal_ends_the_drives_and_the_next_line_is_the_recovery(
            self):
        inv, qa = _stacked([(3.0, ["3", "25", '50/3"', "9"])], 0.15)
        assert _ids(inv) == ["3"]
        assert inv.spt[0].blows == [25, '50/3"'] and inv.spt[0].refusal
        assert inv.spt[0].n is None
        assert inv.samples[0].recovery_percent == 9

    @pytest.mark.parametrize("header, layout", [
        ("SAMPLE NO. / BLOWS/6 IN / REC", ["rec"]),
        ("SAMPLE NO. / BLOWS/6 IN", []),
        ("SAMPLE NO. / BLOWS/6 IN / N / REC", ["n", "rec"]),
        ("SAMPLE NO. / BLOW RECORD (6 IN)", []),        # RECORD is not REC
        ("SAMPLE NO. / BLOWS / % RECOVERED", ["rec"]),
        ("No. / Blows per 6 in. / Recov.", ["rec"]),
        ("SAMPLE NO. / REC / BLOWS/6 IN", None),        # REC before BLOWS
        ("BLOWS / SAMPLE NO.", None),
    ])
    def test_the_header_says_what_follows_the_drives(self, header, layout):
        from report_ingest.log_floor import _stack_layout

        assert _stack_layout(header) == layout

    def test_a_lost_drive_under_a_recovered_header_gives_no_n(self):
        # The first drive (46) lost: 28, 31 and the recovery 9 are left
        # under the number. Truth N 59; 31 + 9 = 40 is wrong.
        column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / % RECOVERED",
                         "sample_id", "blows", "recovery")
        inv, qa = _stacked([(1.0, ["2", "28", "31", "9"])], 0.15, column)
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack_tail" in _flags(qa)

    def test_a_quantity_the_header_words_do_not_place_is_read_as_before(
            self):
        # The column is named for a recovery the layout cannot place.
        column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / % RCVY",
                         "sample_id", "blows", "recovery")
        inv, qa = _stacked([(1.0, ["2", "28", "31", "9"]),
                            (2.5, ["3", "4", "6", "7", "12"])], 0.15, column)
        assert not set(_ids(inv)) & {"28", "31", "9", "4", "6", "7", "12"}
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    def test_a_number_under_50_after_a_refusal_is_never_its_n(self):
        # Sample 3's SS printed a line low, on its first drive "3": the
        # stacks cut there and sample 2's refusal takes the number "3" as
        # the N the header names. A refusal has no N (p. 3-51).
        column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / N",
                         "sample_id", "blows", "n_value")
        cell = _sheet(0.15)
        rows = [(1.0, ["1", "4", "6", "7", "13"]),
                (1.75, ["2", "25", '50/3"']),
                (2.2, ["3", "3", "8", "9", "17"])]
        cells = [cell("blows", t, round(top + j * 0.15, 4))
                 for top, lines in rows for j, t in enumerate(lines)]
        cells += [cell("type", "SS", d) for d in (1.0, 1.75, 2.35)]
        inv, qa = _seed(cells, [column, TYPE])
        refused = [r for r in inv.spt if r.refusal]
        assert [(r.blows, r.n) for r in refused] == [([25, '50/3"'], None)]
        assert all(r.n in (None, 13, 17) for r in inv.spt)
        assert "floor_refusal_n" in _flags(qa)
        assert "3 printed as the N" in _at(inv, 1.75).note

    @pytest.mark.parametrize("typed", [True, False])
    def test_a_recovery_named_before_the_blows_is_read_as_before(self,
                                                                  typed):
        # Number, recovery, drives: whether a recovery is printed cannot be
        # told from the first drive (truth N 13 and 17, REC 12 and 14).
        column = _column("blows", "SAMPLE NO. / REC / BLOWS/6 IN",
                         "sample_id", "blows", "recovery")
        rows = [(1.0, ["1", "12", "4", "6", "7"]),
                (2.5, ["2", "14", "5", "8", "9"])]
        inv, qa = _stacked(rows, 0.15, column, typed=typed)
        assert not set(_ids(inv)) & {"4", "5", "7", "9", "12", "14"}
        assert all(r.n is None for r in inv.spt)
        assert all(s.recovery_percent is None for s in inv.samples)
        assert "floor_stack" in _flags(qa)

    def test_an_n_line_where_the_header_names_n_and_rec(self):
        # One line after the drives where the header names two: kept in the
        # note, never the recovery, and no N (the 7 may be the N).
        column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / N / REC",
                         "sample_id", "blows", "n_value", "recovery")
        rows = [(1.0, ["1", "4", "6", "7", "13"]),
                (2.5, ["2", "5", "8", "9", "17"])]
        inv, qa = _stacked(rows, 0.15, column)
        assert _ids(inv) == ["1", "2"]
        assert all(s.recovery_percent is None for s in inv.samples)
        assert all(r.n is None for r in inv.spt)
        assert "13" in inv.samples[0].note
        assert {"floor_stack_tail", "floor_unassigned"} <= set(_flags(qa))

    def test_the_n_line_the_header_names_is_the_printed_n(self):
        column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / N",
                         "sample_id", "blows", "n_value")
        rows = [(1.0, ["1", "4", "6", "7", "13"]),
                (2.5, ["2", "5", "8", "9", "17"])]
        inv, qa = _stacked(rows, 0.15, column)
        assert _ids(inv) == ["1", "2"]
        assert [r.blows for r in inv.spt] == [[4, 6, 7], [5, 8, 9]]
        assert [(r.n, r.n_basis) for r in inv.spt] == [(13, ""), (17, "")]
        assert qa == []

    def test_a_blow_record_header_has_no_recovery(self):
        column = _column("blows", "SAMPLE NO. / BLOW RECORD (6 IN)",
                         "sample_id", "blows")
        inv, qa = _stacked(_rows(pitch=0.15, rec=False, spacing=1.5), 0.15,
                           column)
        assert _ids(inv) == ["1", "2", "3"]
        assert [r.blows for r in inv.spt] == [d for _s, d, _r in _THREE]
        assert "floor_stack_tail" not in _flags(qa)

    @pytest.mark.parametrize("typed", [True, False])
    def test_a_number_over_a_refusal_in_a_column_read_as_before(self, typed):
        # Back to back with sample 2 uncoded: the column reads as before,
        # and the number "2" over '50/3"' is never taken as its drive.
        rows = [(1.0, ["1", "4", "6", "7"]), (1.6, ["2", '50/3"']),
                (1.9, ["3", "5", "8", "9"])]
        cell = _sheet(0.15)
        cells = [cell("blows", t, round(top + j * 0.15, 4))
                 for top, lines in rows for j, t in enumerate(lines)]
        if typed:
            cells += [cell("type", "SS", 1.0), cell("type", "SS", 1.9)]
        inv, qa = _seed(cells, [NO_REC6, TYPE] if typed else [NO_REC6])
        refused = [r for r in inv.spt if r.refusal]
        assert [r.blows for r in refused] == [['50/3"']]
        assert "floor_stack" in _flags(qa)

    def test_a_box_and_a_sampler_code_a_line_apart_start_nothing(self):
        # The box starts sample 2 at "2"; its SS is printed a line lower, on
        # its first drive, which is 3. Neither "2" alone nor "3" is read.
        boxes = [SimpleNamespace(top=t, bottom=t + 0.4572, page=0,
                                 pages=(0,), plus_minus=None, label="")
                 for t in (1.0, 2.0)]
        cell = _sheet(0.15)
        rows = [(1.0, ["1", "4", "6", "7"]), (2.0, ["2", "3", "8", "9"])]
        cells = [cell("blows", t, round(top + j * 0.15, 4))
                 for top, lines in rows for j, t in enumerate(lines)]
        cells += [cell("type", "SS", 1.0), cell("type", "SS", 2.15)]
        grid = _Grid(cells, [NO_REC6, TYPE])
        grid.samples = boxes
        qa = []
        inv = seed_from_grid(grid, [0], "SYNTH", qa=qa)
        assert "3" not in _ids(inv)
        assert all(r.n is None for r in inv.spt)
        assert "floor_stack" in _flags(qa)

    def test_a_prefixed_id_in_that_column_reads_as_before(self):
        cells = [_cell("blows", "S-3", 3.0), _cell("blows", "4", 3.05),
                 _cell("blows", "6", 3.10), _cell("blows", "7", 3.15)]
        inv, _ = _seed(cells, [BOTH])
        assert _ids(inv) == ["S-3"]
        assert [r.blows for r in inv.spt] == [[4, 6, 7]]

    def test_random_layouts_never_make_up_an_n_or_a_number(self):
        # The review's fuzz at real scales: 0.12-0.35 m a line, samples
        # 0.75-2.5 m apart, back to back or not, with and without a sampler
        # code; drives one to a line, as one record, or with a WOH; a
        # recovery line or not. A sample may go without a number or an N;
        # it may never get a wrong one.
        import random

        rng = random.Random(44)
        read_all = 0
        for _trial in range(400):
            unit = rng.choice(["m", "ft"])
            scale = 1.0 if unit == "m" else 3.2808
            rec = rng.random() < 0.5
            column = BOTH6 if rec else NO_REC6
            pitch = rng.uniform(0.12, 0.35) * scale
            spacing = rng.uniform(0.75, 2.5) * scale
            typed = rng.random() < 0.7
            cell = _sheet(pitch)
            cells, truth, top = [], {}, 1.0 * scale
            for k in range(1, rng.randint(2, 6)):
                drives = [rng.randint(1, 60) for _ in range(3)]
                style = rng.random()
                if style < 0.15:
                    drives[0] = "WOH"
                    lines = [str(k)] + [str(d) for d in drives]
                elif style < 0.3:
                    lines = [str(k), "-".join(str(d) for d in drives)]
                else:
                    lines = [str(k)] + [str(d) for d in drives]
                if rec:
                    lines.append(str(rng.randint(0, 18)))
                cells += [cell("blows", text, round(top + j * pitch, 4))
                          for j, text in enumerate(lines)]
                if typed:
                    cells.append(cell("type", "SS", round(top, 4)))
                truth[round(top, 4)] = (str(k), drives)
                top += max(spacing, len(lines) * pitch)
            grid = _Grid(cells, [column, TYPE] if typed else [column], unit)
            inv = seed_from_grid(grid, [0], "SYNTH", qa=[])
            for sample in inv.samples:
                if sample.sample_id:
                    assert truth.get(round(sample.top.value, 4),
                                     ("",))[0] == sample.sample_id
            for record in inv.spt:
                if record.n is not None:
                    sid, drives = truth[round(record.depth_top.value, 4)]
                    assert record.blows == drives
                    assert record.n == drives[1] + drives[2]
            read_all += _ids(inv) == [sid for sid, _d in truth.values()]
        # Not a test that passes by reading nothing.
        assert read_all >= 200


class TestRI44:
    def test_bare_number_ids_beside_a_blows_column_are_still_samples(self):
        cells = []
        for k, depth in enumerate((1.5, 3.0, 4.5), start=1):
            cells += [_cell("id", str(k), depth), _cell("type", "SS", depth),
                      _cell("blows", f"{k + 2}-{k + 3}-{k + 4}", depth)]
        inv, _ = _seed(cells, [NUMBERED, TYPE, BLOWS])
        assert [s.sample_id for s in inv.samples] == ["1", "2", "3"]
        assert [r.n for r in inv.spt] == [9, 11, 13]

    @pytest.mark.parametrize("code", [None, "SS"])
    def test_a_woh_line_over_a_one_cell_record_stays_in_it(self, code):
        cells = [_cell("rec", "S-3", 3.0), _cell("rec", "WOH", 3.05),
                 _cell("rec", "2-3", 3.10)]
        columns = [DATA]
        if code:
            cells.append(_cell("type", code, 3.0))
            columns.append(TYPE)
        inv, qa = _seed(cells, columns)
        record = inv.spt[0]
        assert record.blows == ["WOH", 2, 3]
        assert record.n is None and not record.refusal
        assert "WOH" in inv.samples[0].note
        assert _flags(qa) == ["floor_drives"]

    def test_a_lone_number_beside_a_one_cell_record_stays_out(self):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5), _cell("blows", "13", 1.6)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[0].blows == [4, 6, 7]
        assert inv.spt[0].n == 13

    @pytest.mark.parametrize("lines, blows", [
        ([("1-2-3", 3.0), ("WOH", 3.3)], [1, 2, 3]),
        ([("WOH", 2.95), ("WOH-1-2", 3.0)], ["WOH", 1, 2]),
    ])
    def test_a_woh_line_never_takes_a_record_past_three_drives(self, lines,
                                                               blows):
        cells = [_cell("id", "S-1", 3.0), _cell("type", "SS", 3.0)]
        cells += [_cell("blows", text, depth) for text, depth in lines]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[0].blows == blows
        assert inv.spt[0].n is None

    def test_a_count_printed_above_a_refusal_is_its_drive(self):
        cells = [_cell("id", "S-1", 3.0), _cell("type", "SS", 3.0),
                 _cell("blows", "25", 3.0), _cell("blows", '50/3"', 3.1)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        record = inv.spt[0]
        assert record.blows == [25, '50/3"']
        assert record.refusal and record.n is None

    def test_never_a_count_in_a_sample_number_column_a_stack_did_not_read(
            self):
        # In a DATA column a bare number may be anything the column holds;
        # read as before.
        cells = [_cell("rec", "S-3", 3.0), _cell("rec", "25", 3.05),
                 _cell("rec", '50/3"', 3.1), _cell("type", "SS", 3.0)]
        inv, _ = _seed(cells, [DATA, TYPE])
        assert inv.spt[0].blows == ['50/3"']

    @pytest.mark.parametrize("pitch", [0.2, 0.25])
    def test_never_the_sample_befores_drive_above_the_id_line(self, pitch):
        # S-1's third drive, printed a line above S-2, is past S-1's span
        # and falls in S-2's rows; it is S-1's, never S-2's.
        cell = _sheet(pitch)
        cells = [cell("id", "S-1", 3.0), cell("type", "SS", 3.0),
                 cell("blows", "4", 3.0), cell("blows", "6", 3.0 + pitch),
                 cell("blows", "7", 3.0 + 2 * pitch),
                 cell("id", "S-2", 3.0 + 3 * pitch),
                 cell("type", "SS", 3.0 + 3 * pitch),
                 cell("blows", '50/3"', 3.0 + 3 * pitch)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[-1].blows == ['50/3"']

    def test_a_count_printed_below_a_refusal_stays_out(self):
        cells = [_cell("id", "S-1", 3.0), _cell("type", "SS", 3.0),
                 _cell("blows", '4-50/3"', 3.0), _cell("blows", "54", 3.1)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        assert inv.spt[0].blows == [4, '50/3"']

    @staticmethod
    def _printed_n(n_text, stacked):
        """S-1's drives 4-6-7 with an N printed: on the stack's own line
        under a header naming N, or in an N column beside the record."""
        if stacked:
            column = _column("blows", "SAMPLE NO. / BLOWS/6 IN / N",
                             "sample_id", "blows", "n_value")
            return _stacked([(1.0, ["1", "4", "6", "7", n_text])], 0.15,
                            column)
        cells = [_cell("id", "S-1", 1.0), _cell("type", "SS", 1.0),
                 _cell("blows", "4-6-7", 1.0), _cell("n", n_text, 1.0)]
        return _seed(cells, [ID, TYPE, BLOWS, _column("n", "N", "n_value")])

    @pytest.mark.parametrize("stacked", [True, False])
    def test_a_printed_n_the_drives_contradict_stands_with_theirs_beside(
            self, stacked):
        inv, qa = self._printed_n("12", stacked)
        record = inv.spt[0]
        assert (record.blows, record.n, record.n_basis) == ([4, 6, 7], 12, "")
        assert [(a.field, a.value, a.method)
                for a in record.prov.alternatives] == [("n", "13", "derived")]
        assert "6 + 7 = 13" in record.prov.alternatives[0].note
        assert "floor_n_drives" in _flags(qa)
        row = next(r for r in qa if r["flag"] == "floor_n_drives")
        assert "printed 12, drives 13" in str(row)

    @pytest.mark.parametrize("stacked", [True, False])
    def test_a_printed_n_the_drives_give_is_noted_as_confirmed(self,
                                                               stacked):
        inv, qa = self._printed_n("13", stacked)
        record = inv.spt[0]
        assert (record.n, record.n_basis) == (13, "")
        assert record.prov.alternatives == []
        assert "is the 2nd + 3rd of the printed drives" in inv.samples[0].note
        assert "floor_n_drives" not in _flags(qa)

    @pytest.mark.parametrize("n_text, n, flagged", [
        ("3", None, True), ("49", None, True), ("50", 50, False),
        ("100", 100, False)])
    def test_an_n_column_under_50_beside_a_refusal_is_not_its_n(
            self, n_text, n, flagged):
        cells = [_cell("id", "S-1", 1.0), _cell("type", "SS", 1.0),
                 _cell("blows", '25-50/3"', 1.0), _cell("n", n_text, 1.0)]
        inv, qa = _seed(cells, [ID, TYPE, BLOWS,
                                _column("n", "N", "n_value")])
        assert inv.spt[0].refusal and inv.spt[0].n == n
        assert ("floor_refusal_n" in _flags(qa)) is flagged

    @pytest.mark.parametrize("blows, halted", [
        ([25, '50/3"'], True),                  # short of the increment
        ([3, 4, '10/0"'], True),                # failed to advance, under 50
        ([12, 18, '35/2"'], True),
        ([12, 18, "35/2"], True),               # no unit, under 6
        ([25, '50/6"'], True),                  # ended before the third
        ([20, 30, 51], True),                   # more than 50 in the last
        ([50, 50], True),                       # 100 before the third
        ([40, 30, 30], False),                  # complete at 100: N 60
        ([52, 4, 28], False),                   # complete: N 32
        (['4/6"', '6/6"', '7/6"'], False),       # written per increment
        (["4/6", "6/6", "7/6"], False),
        ([10, 20, "15/150 mm"], False),
        (["10/15", "20/15", "30/15 cm"], False),
        ([4, 6, 7], False),
        (["WOH", 2, 3], False),
    ])
    def test_a_halt_is_one_the_manual_prints(self, blows, halted):
        from report_ingest.log_floor import N_HALT_RULE, _halted

        assert _halted(blows) is halted
        assert "a total of 100 blows" in N_HALT_RULE
        assert "fails to advance during 10 consecutive blows" in N_HALT_RULE

    @pytest.mark.parametrize("drives", ['4/6"-6/6"-7/6"', "4/6-6/6-7/6"])
    def test_a_record_written_per_increment_keeps_its_printed_n(self,
                                                                drives):
        cells = [_cell("id", "S-1", 1.0), _cell("type", "SS", 1.0),
                 _cell("blows", drives, 1.0), _cell("n", "13", 1.0)]
        inv, qa = _seed(cells, [ID, TYPE, BLOWS,
                                _column("n", "N", "n_value")])
        record = inv.spt[0]
        assert (record.n, record.refusal) == (13, False)
        assert "floor_refusal_n" not in _flags(qa)

    def test_a_complete_record_after_a_hard_seating_drive_keeps_its_n(
            self):
        cells = [_cell("id", "S-1", 1.0), _cell("type", "SS", 1.0),
                 _cell("blows", "52-4-28", 1.0), _cell("n", "32", 1.0)]
        inv, qa = _seed(cells, [ID, TYPE, BLOWS,
                                _column("n", "N", "n_value")])
        assert inv.spt[0].n == 32
        assert "floor_refusal_n" not in _flags(qa)

    def test_a_no_advance_halt_under_50_has_no_n(self):
        cells = [_cell("id", "S-1", 1.0), _cell("type", "SS", 1.0),
                 _cell("blows", '3-4-10/0"', 1.0), _cell("n", "14", 1.0)]
        inv, qa = _seed(cells, [ID, TYPE, BLOWS,
                                _column("n", "N", "n_value")])
        record = inv.spt[0]
        assert (record.n, record.refusal) == (None, True)
        assert "floor_refusal_n" in _flags(qa)
        assert "14 printed as the N" in inv.samples[0].note

    @pytest.mark.parametrize("text", ["WOH-2-3", "WOH/2/3"])
    @pytest.mark.parametrize("code", [None, "SS"])
    def test_woh_in_one_cell_is_a_drive_of_the_record(self, text, code):
        cells = [_cell("id", "S-1", 3.0), _cell("blows", text, 3.0)]
        columns = [ID, BLOWS]
        if code:
            cells.append(_cell("type", code, 3.0))
            columns.append(TYPE)
        inv, qa = _seed(cells, columns)
        record = inv.spt[0]
        assert record.blows == ["WOH", 2, 3]
        assert record.n is None and record.n_basis == ""
        assert not record.refusal
        assert _flags(qa) == ["floor_drives"]

    def test_a_woh_record_writes_to_diggs(self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, write_diggs,
        )
        from report_ingest.model import Project

        cells = [_cell("id", "S-1", 3.0), _cell("type", "SS", 3.0),
                 _cell("blows", "WOH-2-3", 3.0)]
        inv, _ = _seed(cells, [ID, TYPE, BLOWS])
        inv.investigation_id = "B-1"
        xml = write_diggs(inv, Project(name="R"))
        ok, diffs = diggs_roundtrip_gate(xml, [inv], project=Project(name="R"))
        assert ok, diffs

    def test_a_header_naming_the_6_inch_increment_gives_the_n(self):
        blows = _column("blows", 'BLOWS / FOOT (6" INCREMENTS)', "blows")
        cells = [_cell("id", "S-1", 3.0), _cell("type", "SS", 3.0),
                 _cell("blows", "3-4-5", 3.0)]
        inv, _ = _seed(cells, [ID, TYPE, blows])
        assert inv.spt[0].n == 9
        assert "INCREMENTS" in inv.spt[0].n_basis

    def test_a_sheet_of_values_only_binds_by_the_same_rule(self):
        # Sheet 1 prints values and no sample. The value at its head is
        # clearly nearest the sample at the foot of sheet 0 and binds to it;
        # the one far below binds to nothing and stays where printed,
        # doubted and listed (left as RI-04 made it).
        cells = [_cell("id", "S-1", 9.6, page=0),
                 _cell("blows", "3-4-5", 9.6, page=0),
                 _cell("mc", "22.0", 9.9, page=1),
                 _cell("mc", "20.0", 12.0, page=1)]
        inv, qa = _seed(cells, [ID, BLOWS, MC], pages=(0, 1))
        assert _at(inv, 9.6).water_content == 22.0
        loose = _at(inv, 12.0)
        assert loose.water_content == 20.0
        assert loose.prov.confidence == pytest.approx(0.3)
        assert _flags(qa) == ["floor_unbound"]
        assert "20.0" in qa[0]["value"] and "22.0" not in qa[0]["value"]


# ---------------------------------------------------------------------------
# part 4: a model's N, checked against the printed drives
# ---------------------------------------------------------------------------

def _q(value):
    return Quantity(value=value, unit="m")


def _model(n, *, kind="spt", evidence=False, blows=(4, 6, 7),
           increment=0.1524):
    prov = Provenance(page=0, method="model", confidence=0.9,
                      bbox=(140.0, 156.0, 170.0, 164.0) if evidence else None,
                      note="N printed beside the drives" if evidence else "")
    return Investigation(
        investigation_id="B-1", depth_unit="m",
        samples=[Sample(sample_id="S-1", top=_q(1.5), kind=kind, prov=prov)],
        spt=[SPT(depth_top=_q(1.5), blows=list(blows), n=n,
                 sample_id="S-1",
                 increment=(Quantity(value=increment, unit="m")
                            if increment else None),
                 prov=prov)])


class TestTheModelsN:
    def _floor(self, code=None, drives="4-6-7"):
        cells = [_cell("id", "S-1", 1.5), _cell("blows", drives, 1.5)]
        columns = [ID, BLOWS]
        if code:
            cells.append(_cell("type", code, 1.5))
            columns.append(TYPE)
        floor, _ = _seed(cells, columns)
        floor.investigation_id = "B-1"
        return floor

    def _n_rows(self, log):
        return [row for row in log.disagreements if row["what"].endswith(
            ": n")]

    def test_a_model_n_that_disagrees_without_evidence_gives_way(self):
        merged, log = merge_investigations(self._floor(), _model(15))
        record = merged.spt[0]
        assert record.n == 13 and "2nd + 3rd" in record.n_basis
        alternatives = [a for a in record.prov.alternatives if a.field == "n"]
        assert [a.value for a in alternatives] == ["15"]
        rows = self._n_rows(log)
        assert len(rows) == 1 and rows[0]["kept"] == "floor"
        assert rows[0]["floor_method"] == "derived"

    def test_a_model_n_with_evidence_stands_and_the_drives_n_is_kept(self):
        merged, log = merge_investigations(self._floor(),
                                           _model(15, evidence=True))
        record = merged.spt[0]
        assert record.n == 15 and record.n_basis == ""
        alternatives = [a for a in record.prov.alternatives if a.field == "n"]
        assert [(a.value, a.method) for a in alternatives] == [
            ("13", "derived")]
        assert self._n_rows(log)[0]["kept"] == "model"

    @pytest.mark.parametrize("model_n, evidence, n, alternatives", [
        (12, False, 12, [("13", "derived")]),
        (None, False, 12, [("13", "derived")]),
        (15, False, 12, [("15", "model"), ("13", "derived")]),
        (13, True, 13, [("12", "grid")]),       # the drives' N is the slot
    ])
    def test_the_drives_n_beside_a_printed_n_survives_the_merge(
            self, model_n, evidence, n, alternatives):
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5), _cell("n", "12", 1.5)]
        floor, _ = _seed(cells, [ID, TYPE, BLOWS,
                                 _column("n", "N", "n_value")])
        floor.investigation_id = "B-1"
        merged, _ = merge_investigations(floor,
                                         _model(model_n, evidence=evidence))
        record = merged.spt[0]
        assert record.n == n
        assert [(a.value, a.method) for a in record.prov.alternatives
                if a.field == "n"] == alternatives

    def test_the_drives_n_of_drives_that_lost_is_not_carried(self):
        # The model's 4-6-8 with N 14, box and note, wins: the floor's
        # derived 13 was the N of 4-6-7.
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", "4-6-7", 1.5), _cell("n", "12", 1.5)]
        floor, _ = _seed(cells, [ID, TYPE, BLOWS,
                                 _column("n", "N", "n_value")])
        floor.investigation_id = "B-1"
        merged, _ = merge_investigations(
            floor, _model(14, evidence=True, blows=(4, 6, 8)))
        record = merged.spt[0]
        assert (record.blows, record.n) == ([4, 6, 8], 14)
        assert ("13", "derived") not in [
            (a.value, a.method) for a in record.prov.alternatives]

    @pytest.mark.parametrize("evidence", [False, True])
    def test_a_model_n_under_50_on_a_halt_stays_beside_the_slot(
            self, evidence):
        # The floor dropped the printed 13 on 25-50/3"; the model reads the
        # same 13. A halted record has no N (p. 3-51).
        cells = [_cell("id", "S-1", 1.5), _cell("type", "SS", 1.5),
                 _cell("blows", '25-50/3"', 1.5), _cell("n", "13", 1.5)]
        floor, _ = _seed(cells, [ID, TYPE, BLOWS,
                                 _column("n", "N", "n_value")])
        floor.investigation_id = "B-1"
        merged, log = merge_investigations(
            floor, _model(13, evidence=evidence, blows=(25, '50/3"')))
        record = merged.spt[0]
        assert record.n is None and record.n_basis == ""
        assert [(a.value, a.method) for a in record.prov.alternatives
                if a.field == "n"] == [("13", "model")]
        rows = self._n_rows(log)
        assert len(rows) == 1 and rows[0]["model"] == "13"
        assert "halt" in rows[0]["why"]

    def test_a_model_only_record_under_50_on_a_halt_has_no_n(self):
        floor = Investigation(investigation_id="B-1", depth_unit="m")
        merged, log = merge_investigations(
            floor, _model(13, blows=(3, 4, '10/0"')))
        record = merged.spt[0]
        assert record.n is None
        assert [a.value for a in record.prov.alternatives
                if a.field == "n"] == ["13"]
        assert len(self._n_rows(log)) == 1

    def test_a_model_n_of_50_or_more_on_a_halt_stands(self):
        floor = Investigation(investigation_id="B-1", depth_unit="m")
        merged, log = merge_investigations(
            floor, _model(75, blows=(25, '50/3"')))
        assert merged.spt[0].n == 75
        assert self._n_rows(log) == []

    def test_a_model_n_that_agrees_with_the_drives_is_not_a_disagreement(
            self):
        merged, log = merge_investigations(self._floor(), _model(13))
        assert merged.spt[0].n == 13
        assert self._n_rows(log) == []

    @pytest.mark.parametrize("drives, model", [
        ("4-6-7", _model(15, kind="ring")),            # not an SPT
        ("4-6-7", _model(15, increment=None)),         # no length printed
        ("4-6-7-9", _model(15, blows=(4, 6, 7, 9))),   # four drives
    ])
    def test_no_check_where_the_rule_does_not_apply(self, drives, model):
        merged, log = merge_investigations(self._floor(drives=drives), model)
        assert merged.spt[0].n == 15
        assert self._n_rows(log) == []

    def test_the_drives_checked_are_the_ones_the_merge_kept(self):
        # The model says four drives and the floor three, with no evidence:
        # the floor's printed drives stand, and the N is checked on them.
        merged, log = merge_investigations(self._floor(),
                                           _model(15, blows=(4, 6, 7, 9)))
        assert merged.spt[0].blows == [4, 6, 7]
        assert merged.spt[0].n == 13
        assert len(self._n_rows(log)) == 1

    def test_a_record_only_the_model_has_is_checked_too(self):
        floor = self._floor()
        floor.spt, floor.samples = [], []
        merged, log = merge_investigations(floor, _model(15))
        assert merged.spt[0].n == 13
        assert len(self._n_rows(log)) == 1

    def test_the_floors_computed_n_keeps_its_basis_when_it_stands(self):
        merged, log = merge_investigations(self._floor(code="SS"),
                                           _model(13))
        assert merged.spt[0].n == 13
        assert "2nd + 3rd" in merged.spt[0].n_basis

    def test_a_printed_n_the_model_read_replaces_the_computed_one(self):
        merged, log = merge_investigations(self._floor(code="SS"),
                                           _model(15, evidence=True))
        record = merged.spt[0]
        assert record.n == 15 and record.n_basis == ""
        alternatives = [a for a in record.prov.alternatives if a.field == "n"]
        assert [(a.value, a.method) for a in alternatives] == [
            ("13", "derived")]
        assert "2nd + 3rd" in alternatives[0].note

    def test_the_merge_log_is_the_usual_one(self):
        merged, log = merge_investigations(self._floor(), _model(15),
                                           MergeLog())
        assert isinstance(log, MergeLog)
