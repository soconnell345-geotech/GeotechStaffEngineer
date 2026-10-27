"""No depth below the ground surface is negative (BACKLOG RI-123), offline.

A depth read a few centimetres above the surface is the surface, as a ruler
fitted to the printed ticks reads it at the top of a log, and is recorded as
0; one further above is refused under ``checks="refuse"``, kept out and
listed as read under ``checks="flag"``, and never written to a DIGGS file.
An elevation is on a datum, not below the ground, and is never refused.

Every log, ruler and value here is made up in this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_ingest.checks import (
    ABOVE, SNAPPED, SURFACE_TOL_M, ground_investigation, surface,
)
from report_ingest.floor import DEPTH_TOL_M
from report_ingest.model import (
    AtterbergResult, CPTData, CPTPoint, Investigation, LabTest, Layer,
    Project, Provenance, QAEntry, Quantity, Sample, SummaryRow,
    SummaryTableResult, WaterLevel,
)


def _m(value: float) -> Quantity:
    return Quantity(value=value, unit="m")


def _prov(page: int = 3) -> Provenance:
    return Provenance(page=page, method="grid", confidence=0.8)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

class TestTheRule:
    def test_the_tolerance_is_inside_the_scorers(self):
        """Set to 0, a depth this close to it scores as a printed 0 does."""
        assert 0.0 < SURFACE_TOL_M < DEPTH_TOL_M

    def test_a_depth_at_or_below_the_surface_is_as_read(self):
        assert surface(0.0, "m") == (0.0, "")
        assert surface(2.5, "ft") == (2.5, "")

    def test_a_few_centimetres_above_is_the_surface(self):
        assert surface(-0.04, "m") == (0.0, SNAPPED)
        assert surface(-0.0001, "m") == (0.0, SNAPPED)

    def test_further_above_is_above(self):
        assert surface(-0.3, "m") == (-0.3, ABOVE)
        assert surface(-2.0, "m") == (-2.0, ABOVE)

    def test_the_tolerance_is_in_metres_whatever_the_unit(self):
        # 0.3 ft is about 0.09 m, 0.5 ft about 0.15 m.
        assert surface(-0.3, "ft")[1] == SNAPPED
        assert surface(-0.5, "ft")[1] == ABOVE

    def test_a_number_with_no_unit_is_read_as_metres(self):
        """The larger of its two likely readings, so never looser."""
        assert surface(-0.08, "")[1] == SNAPPED
        assert surface(-0.2, "")[1] == ABOVE


# ---------------------------------------------------------------------------
# one investigation, every slot
# ---------------------------------------------------------------------------

def _boring(**over) -> Investigation:
    base = dict(
        investigation_id="QX-1", kind="boring", depth_unit="m",
        units_known=True, elevation=_m(-5.0), total_depth=_m(9.0),
        layers=[
            Layer(top=_m(-0.04), bottom=_m(2.0), description="made-up SAND",
                  prov=_prov()),
            Layer(top=_m(-1.2), bottom=_m(-0.5),
                  description="made-up FILL", prov=_prov()),
            Layer(top=_m(2.0), bottom=_m(9.0), description="made-up CLAY",
                  prov=_prov()),
        ],
        samples=[
            Sample(sample_id="S1", top=_m(1.0), bottom=_m(-0.3),
                   prov=_prov()),
            Sample(sample_id="S2", top=_m(-0.6), bottom=_m(0.4),
                   prov=_prov()),
        ],
        water=[WaterLevel(depth=_m(-0.6), elevation=_m(-4.4),
                          when="while_drilling", prov=_prov())],
        pages=[3])
    base.update(over)
    return Investigation(**base)


class TestOneInvestigation:
    def test_each_depth_against_the_surface(self):
        inv = _boring()
        rows, snapped = ground_investigation(inv)
        assert snapped == 1
        assert inv.layers[0].top.value == 0.0
        assert "recorded as 0" in inv.layers[0].prov.note
        # The fill's top is above the surface: the layer goes whole, as read.
        assert [ly.description for ly in inv.layers] == [
            "made-up SAND", "made-up CLAY"]
        # A base above the surface is emptied, its sample kept.
        assert [s.sample_id for s in inv.samples] == ["S1"]
        assert inv.samples[0].bottom is None
        assert inv.water[0].depth is None
        whats = [row["what"] for row in rows]
        assert whats == ["QX-1: layer 2 top", "QX-1: sample S1 base",
                         "QX-1: sample S2 top", "QX-1: water observation 1"]
        assert all(row["refused_by"] == "python" for row in rows)
        assert all("is negative" in row["why"] for row in rows)
        whole = next(r for r in rows if r["what"] == "QX-1: layer 2 top")
        assert "goes whole" in whole["why"] and "made-up FILL" in whole["read"]
        water = next(r for r in rows if "water" in r["what"])
        assert "artesian" in water["why"]

    def test_elevations_are_never_looked_at(self):
        inv = _boring()
        ground_investigation(inv)
        assert inv.elevation.value == -5.0
        assert inv.water[0].elevation.value == -4.4

    def test_a_negative_total_depth_is_emptied(self):
        inv = _boring(total_depth=_m(-0.05))
        rows, _ = ground_investigation(inv)
        assert inv.total_depth is None
        assert any(r["what"] == "QX-1: total depth" for r in rows)

    def test_a_cone_on_an_elevation_axis_is_not_looked_at(self):
        cpt = CPTData(vertical_axis="elevation", datum="made-up datum",
                      points=[CPTPoint(depth=_m(-1.0)),
                              CPTPoint(depth=_m(-2.0))])
        inv = Investigation(investigation_id="QX-C", kind="cpt",
                            depth_unit="m", units_known=True, cpt=cpt)
        rows, snapped = ground_investigation(inv)
        assert (rows, snapped) == ([], 0)
        assert len(inv.cpt.points) == 2

    def test_a_cone_point_above_the_surface_leaves_the_series(self):
        cpt = CPTData(vertical_axis="depth",
                      points=[CPTPoint(depth=_m(-0.5)),
                              CPTPoint(depth=_m(-0.02)),
                              CPTPoint(depth=_m(0.5))])
        inv = Investigation(investigation_id="QX-C", kind="cpt",
                            depth_unit="m", units_known=True, cpt=cpt)
        rows, snapped = ground_investigation(inv)
        assert [p.depth.value for p in inv.cpt.points] == [0.0, 0.5]
        assert snapped == 1 and len(rows) == 1

    def test_a_record_with_none_is_untouched(self):
        inv = _boring(layers=[], samples=[], water=[], elevation=None)
        before = inv.model_dump()
        assert ground_investigation(inv) == ([], 0)
        assert inv.model_dump() == before


# ---------------------------------------------------------------------------
# the log floor, and its ruler
# ---------------------------------------------------------------------------

class _Grid:
    """What ``seed_from_grid`` reads off planlens' ``log_grid``."""

    def __init__(self, cells=(), layers=(), ticks=((100.0, 0.0),
                                                     (140.0, 1.0))):
        self.rows = list(cells)
        self.columns = [SimpleNamespace(id="id", page=0, header="SAMPLE",
                                        names=("sample_id",), unit="")]
        self.unit = "m"
        self.rulers = {0: SimpleNamespace(step=1.0, ticks=tuple(ticks),
                                          unit="m")}
        self.fields, self.field_boxes = {}, {}
        self.layers = list(layers)
        self.samples, self.elevation_rulers = [], {}
        self.warnings = []

    def column(self, column_id):
        return next((c for c in self.columns if c.id == column_id), None)


def _cell(text, depth):
    y = 100.0 + 40.0 * depth
    return SimpleNamespace(page=0, column_id="id", text=text,
                           bbox=(60.0, y - 4.0, 90.0, y + 4.0), depth=depth,
                           depth_top=depth, depth_bottom=depth + 0.3,
                           numbers=[], confidence=0.9)


def _layer(top, bottom, words, source="page_top"):
    return SimpleNamespace(top=top, bottom=bottom, description=words,
                           pages=(0,), source=source, bbox=None,
                           confidence=0.8)


def _seed(grid):
    from report_ingest.log_floor import seed_from_grid
    qa = []
    return seed_from_grid(grid, [0], "MADE-UP", qa=qa), qa


class TestLogFloor:
    def test_a_sample_a_hair_above_the_surface_is_at_the_surface(self):
        inv, qa = _seed(_Grid(cells=[_cell("S-1", -0.03),
                                     _cell("S-2", 1.5)]))
        assert [s.top.value for s in inv.samples] == [0.0, 1.5]
        assert any(row.get("flag") == "surface_snap" for row in qa)

    def test_a_layer_above_the_surface_is_refused_and_the_ruler_doubted(self):
        grid = _Grid(layers=[_layer(-0.7, 1.2, "made-up brown SAND (SP)"),
                             _layer(1.2, 4.0, "made-up grey CLAY (CL)",
                                    source="rule")],
                     ticks=((100.0, 0.0), (140.0, 1.0)))
        inv, qa = _seed(grid)
        assert [ly.top.value for ly in inv.layers] == [1.2]
        refused = [r for r in qa if r.get("refused_by") == "python"]
        assert len(refused) == 1 and "is negative" in refused[0]["why"]
        doubt = [r for r in qa if r.get("flag") == "ruler_above_ground"]
        assert len(doubt) == 1
        assert "top of the log's body" in doubt[0]["why"]
        assert doubt[0]["page"] == 0

    def test_a_ruler_printing_a_tick_above_the_surface_is_doubted(self):
        _inv, qa = _seed(_Grid(ticks=((100.0, -0.5), (140.0, 0.5))))
        doubt = [r for r in qa if r.get("flag") == "ruler_above_ground"]
        assert len(doubt) == 1 and "prints a tick of -0.5" in doubt[0]["why"]

    def test_a_ruler_from_the_surface_is_not_doubted(self):
        _inv, qa = _seed(_Grid(ticks=((100.0, 0.0), (140.0, 1.0)),
                               layers=[_layer(-0.02, 2.0,
                                              "made-up SILT (ML)")]))
        assert not [r for r in qa if r.get("flag") == "ruler_above_ground"]

    def test_the_doubt_is_ranked_open(self):
        from report_ingest.reconciler import _carry_unresolved
        _inv, qa = _seed(_Grid(ticks=((100.0, -0.5), (140.0, 0.5))))
        entries = []
        _carry_unresolved(entries, [r for r in qa
                                    if r.get("flag") == "ruler_above_ground"])
        assert [e.rank for e in entries] == ["open"]

    def test_the_snap_note_is_ranked_changed(self):
        entry = QAEntry(kind="partial", where="depths",
                        detail="2 depth(s) the grid read above the ground "
                               "surface by no more than 0.1 m were "
                               "recorded as 0")
        assert entry.rank == "changed"


# ---------------------------------------------------------------------------
# the log reader's builder, both modes
# ---------------------------------------------------------------------------

def _reading(sample_top: float, layer_top: float):
    from report_ingest.log_reader import LogReading
    return LogReading.model_validate({
        "investigation_id": "QX-2", "kind": "boring", "depth_unit": "m",
        "units_known": True, "total_depth": 6.0,
        "layers": [
            {"top": layer_top, "bottom": 1.0, "description": "made-up SAND",
             "prov": {"page": 0, "bbox": [10, 20, 90, 30]}},
            {"top": 1.0, "bottom": 6.0, "description": "made-up CLAY",
             "prov": {"page": 0, "bbox": [10, 40, 90, 50]}}],
        "samples": [
            {"sample_id": "A", "top": sample_top, "bottom": 0.45,
             "kind": "spt", "prov": {"page": 0, "bbox": [10, 20, 90, 30]}}],
        "pages_read": [0]})


def _build(reading, checks):
    from report_ingest.log_reader import _Builder
    builder = _Builder(reading, (-0.5, 6.5), [0], "MADE-UP", checks=checks)
    return builder, builder.build()


@pytest.mark.parametrize("checks", ["refuse", "flag"])
class TestBuilder:
    def test_a_hair_above_is_recorded_as_0_with_a_change(self, checks):
        builder, inv = _build(_reading(-0.05, 0.0), checks)
        assert inv.samples[0].top.value == 0.0
        assert any("recorded as 0" in c["why"] for c in builder.changes)
        assert any(r.get("flag") == "surface_snap"
                   for r in builder.unresolved)

    def test_inside_the_window_but_above_the_surface_is_out(self, checks):
        """The window starts at the paper's top edge, above the surface."""
        builder, inv = _build(_reading(0.0, -0.3), checks)
        assert [ly.top.value for ly in inv.layers] == [1.0]
        rows = [r for r in builder.unresolved
                if r.get("refused_by") == "python"]
        assert rows and all("is negative" in r["why"] for r in rows)
        if checks == "flag":
            assert any("listed here as read" in r["why"] for r in rows)
        else:
            assert any("refused rather than accepted" in r["why"]
                       for r in rows)


# ---------------------------------------------------------------------------
# a lab summary table's elevation column
# ---------------------------------------------------------------------------

class TestLabSummaryRows:
    def _rows(self, header, row):
        from report_ingest.lab_floor import _summary_rows
        columns = {0: "investigation_id", 1: "depth_top", 2: "ll"}
        return _summary_rows([row], header, columns, "ft", 0.8,
                             Provenance(page=4, method="tables"))

    def test_an_elevation_column_is_the_rows_elevation(self):
        rows = self._rows(["BORING", "ELEVATION (FT)", "LL"],
                          ["QX-3", "-4.5", "31"])
        assert rows[0].depth_top is None
        assert rows[0].elevation_top.value == -4.5

    def test_a_depth_column_stays_the_rows_depth(self):
        rows = self._rows(["BORING", "DEPTH (FT)", "LL"],
                          ["QX-3", "4.5", "31"])
        assert rows[0].depth_top.value == 4.5
        assert rows[0].elevation_top is None

    def test_a_negative_depth_is_kept_as_printed_not_as_the_depth(self):
        rows = self._rows(["BORING", "DEPTH (FT)", "LL"],
                          ["QX-3", "-2", "31"])
        assert rows[0].depth_top is None
        assert ("DEPTH (FT)", "-2") in [tuple(o) for o in rows[0].other]


# ---------------------------------------------------------------------------
# the DIGGS writer's last gate
# ---------------------------------------------------------------------------

class TestDiggsLastGate:
    def _hole(self):
        return Investigation(
            investigation_id="QX-4", kind="boring", depth_unit="m",
            units_known=True, total_depth=_m(5.0),
            layers=[Layer(top=_m(-0.04), bottom=_m(2.0),
                          description="made-up SAND"),
                    Layer(top=_m(2.0), bottom=_m(5.0),
                          description="made-up CLAY")],
            samples=[Sample(sample_id="1", top=_m(-1.0), bottom=_m(-0.5)),
                     Sample(sample_id="2", top=_m(1.0), bottom=_m(1.5))],
            water=[WaterLevel(depth=_m(-0.5), when="while_drilling")],
            pages=[6])

    def _lab(self):
        return [
            LabTest(kind="atterberg", investigation_id="QX-4",
                    depth_top=_m(-2.0),
                    result=AtterbergResult(ll=40, pl=20, pi=20)),
            LabTest(kind="summary_table", result=SummaryTableResult(rows=[
                SummaryRow(investigation_id="QX-4", depth_top=_m(-3.0),
                           ll=33.0),
                SummaryRow(investigation_id="QX-4", depth_top=_m(2.0),
                           ll=35.0)])),
        ]

    def test_nothing_above_the_surface_is_written_and_the_rest_reads_back(
            self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, write_diggs,
        )
        hole, lab = self._hole(), self._lab()
        before = (hole.model_dump(), [t.model_dump() for t in lab])
        notes = []
        xml = write_diggs([hole] + lab, Project(name="made-up"), notes=notes)
        note = notes[0]
        whats = [row["what"] for row in note.above_ground]
        assert whats == ["QX-4: sample 1 top", "QX-4: water observation 1",
                         "lab test 1 (atterberg, QX-4)",
                         "lab test 2 (summary_table) row 1"]
        assert note.samples == 1 and note.layers == 2 and note.water == 0
        assert any("written as 0" in line for line in note.normalised)
        assert all("left out of the DIGGS file" in row["why"]
                   for row in note.above_ground)
        # Said once: in above_ground, not again among the skipped lines.
        assert not [line for line in note.skipped if "is negative" in line]
        # The inputs are never changed; the file holds no negative depth.
        assert (hole.model_dump(), [t.model_dump() for t in lab]) == before
        ok, diffs = diggs_roundtrip_gate(xml, [hole], lab_tests=lab,
                                         notes=note)
        assert ok, diffs
        ok, diffs = diggs_roundtrip_gate(xml, [hole], lab_tests=lab)
        assert ok, diffs

    def test_the_record_says_what_the_file_left_out(self):
        from report_ingest.diggs_writer import write_diggs
        from report_ingest.writers import _diggs_notes_to_qa
        notes = []
        write_diggs([self._hole()] + self._lab(), Project(name="made-up"),
                    notes=notes)
        record = SimpleNamespace(qa=[])
        _diggs_notes_to_qa(record, notes[0])
        above = [e for e in record.qa if e.where == "diggs.above_ground"]
        assert len(above) == 1
        assert above[0].kind == "out_of_range" and above[0].rank == "changed"
        assert "4 depth(s)" in above[0].detail
        # Not counted a second time in the roll-up of the writer's notes.
        for entry in record.qa:
            if entry.where == "diggs.notes":
                assert not [v for v in entry.values if "is negative" in v]

    @staticmethod
    def _wet(depth: float) -> Investigation:
        """A boring whose log says no water while drilling and gives a
        reading the next day."""
        return Investigation(
            investigation_id="ZZ-W", kind="boring", depth_unit="m",
            units_known=True, total_depth=_m(6.0),
            layers=[Layer(top=_m(0.0), bottom=_m(6.0),
                          description="made-up SILT")],
            water=[WaterLevel(when="not_encountered"),
                   WaterLevel(depth=_m(depth), when="after_hours",
                              hours=24.0)],
            pages=[2])

    def test_a_reading_above_ground_does_not_become_none_encountered(self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, write_diggs,
        )
        hole = self._wet(-0.4)
        notes = []
        xml = write_diggs([hole], Project(name="made-up"), notes=notes)
        assert "notEncountered" not in xml
        assert "WaterStrike" not in xml
        assert notes[0].water == 0
        assert [r["what"] for r in notes[0].above_ground] == [
            "ZZ-W: water observation 2"]
        assert any("a water reading above the ground surface was left out"
                   in line and "not encountered" in line
                   for line in notes[0].skipped)
        assert [w.when for w in hole.water] == ["not_encountered",
                                                 "after_hours"]
        ok, diffs = diggs_roundtrip_gate(xml, [hole], notes=notes[0])
        assert ok, diffs

    def test_the_same_reading_below_ground_is_written(self):
        from report_ingest.diggs_writer import write_diggs
        notes = []
        xml = write_diggs([self._wet(0.4)], Project(name="made-up"),
                          notes=notes)
        assert "notEncountered" not in xml
        assert "WaterStrikeReading" in xml
        assert notes[0].water == 1
        assert notes[0].above_ground == []

    def test_a_record_with_none_writes_as_before(self):
        from report_ingest.diggs_writer import write_diggs
        hole = self._hole()
        hole.layers[0].top = _m(0.0)
        hole.samples = hole.samples[1:]
        hole.water = []
        notes = []
        write_diggs([hole], Project(name="made-up"), notes=notes)
        assert notes[0].above_ground == []
        assert notes[0].normalised == []
