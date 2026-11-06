"""RI-208: a calculation's printed table keeps EVERY row.

Before, the calc floor took a grid's last filled row only, labelled
"..., last row of the table", and every other row's values were lost; the
merge and the model-first fold also took two rows of one table that printed
the same number under labels the partial ratio calls one label as one value.
Now each row's values are their own, labelled by the column heading and the
row's qualifier, and the floor, the merge, the model-first fold and the
scorer keep them apart. Every value here is made up.
"""

from __future__ import annotations

import pytest

from report_ingest import model_first as mf
from report_ingest.calc_reader import (
    CalcReading, ReadProv, ReadValue, _fold_readings, _row_names,
    floor_from_pages, merge_calculation,
)
from report_ingest.calc_scoring import other_rows, same_name, score_record
from report_ingest.model import Calculation, NamedQuantity, Provenance, Quantity

pymupdf = pytest.importorskip("pymupdf", reason="the fixture is drawn with PyMuPDF")

LAYERS = {"1": (4.0, 115.0, 30.0), "2": (6.5, 120.0, 32.0),
          "3": (8.0, 125.0, 34.0), "4": (10.0, 130.0, 32.0)}


@pytest.fixture
def table(tmp_path):
    from planlens.document import open_document
    from report_ingest.tests.calc_fixtures import build_layer_table
    gt = build_layer_table()
    path = tmp_path / "layers.pdf"
    path.write_bytes(gt.pdf)
    doc = open_document(str(path))
    yield doc, gt
    doc.close()


def _model_rows(bbox=None):
    """The model-first reading of the table: every row, no box."""
    rows = []
    for layer, (thick, gamma, phi) in LAYERS.items():
        rows += [(f"Thickness (ft), Layer {layer}", thick, "ft"),
                 (f"Unit weight (pcf), Layer {layer}", gamma, "pcf"),
                 (f"Phi (deg), Layer {layer}", phi, "deg")]
    return rows


def _calc(rows) -> Calculation:
    return Calculation(
        kind="shallow_foundation_bearing",
        inputs=[NamedQuantity(name=n, value=Quantity(value=v, unit=u),
                              prov=Provenance(page=0, method="model",
                                              confidence=0.9))
                for n, v, u in rows],
        pages=[0])


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

def test_the_floor_keeps_every_row_of_a_four_row_table(table):
    doc, gt = table
    floor = floor_from_pages(doc, [0], "SYN")

    got = {v.name: (v.value, v.unit) for v in floor.values
           if v.method == "tables"}
    assert got == {name: value for name, value in gt.inputs.items()}
    assert len(got) == 12
    assert not any("last row" in v.name for v in floor.values)


def test_a_one_row_table_needs_no_row_qualifier():
    assert _row_names(["A (ft)", "B (ft)"], [["1", "2"]]) == [None]


def test_row_qualifiers_come_from_the_first_cell_and_stay_distinct():
    header = ["Layer", "Phi (deg)"]
    assert _row_names(header, [["1", "30"], ["2", "32"]]) == \
        ["Layer 1", "Layer 2"]
    assert _row_names(header, [["Silty sand", "30"], ["Clay", "0"]]) == \
        ["Silty sand", "Clay"]
    assert _row_names(header, [["", "30"], ["", "32"]]) == ["row 1", "row 2"]
    assert _row_names(header, [["Sand", "30"], ["Sand", "32"]]) == \
        ["Sand, row 1", "Sand, row 2"]


# ---------------------------------------------------------------------------
# the merge (today's reader: the floor and the model)
# ---------------------------------------------------------------------------

def test_the_merge_keeps_all_four_rows_and_reconciles_each(table):
    """A canned model-first answer, no boxes, against the floor of the same
    page: every row reconciled with its own row, none doubled, none lost --
    though layers 2 and 4 print the same friction angle."""
    doc, _gt = table
    floor = floor_from_pages(doc, [0], "SYN")

    merged, log = merge_calculation(floor, _calc(_model_rows()))

    names = [v.name for v in merged.inputs]
    assert len(names) == 12 and len(set(names)) == 12
    assert log.reconciled == 12
    assert not log.disagreements
    phi = {v.name: v.value.value for v in merged.inputs
           if v.name.startswith("Phi")}
    assert phi == {f"Phi (deg), Layer {k}": v[2] for k, v in LAYERS.items()}


def test_a_row_the_model_left_out_is_kept_from_the_floor(table):
    doc, _gt = table
    floor = floor_from_pages(doc, [0], "SYN")
    rows = [r for r in _model_rows() if not r[0].endswith("Layer 3")]

    merged, _log = merge_calculation(floor, _calc(rows))

    assert {v.name for v in merged.inputs if v.name.endswith("Layer 3")} == \
        {"Thickness (ft), Layer 3", "Unit weight (pcf), Layer 3",
         "Phi (deg), Layer 3"}
    assert len(merged.inputs) == 12


# ---------------------------------------------------------------------------
# the model-first fold (two calls over one printout)
# ---------------------------------------------------------------------------

def _reading(rows, page=0) -> CalcReading:
    return CalcReading(
        kind="shallow_foundation_bearing", program="", program_version="",
        method="", subject="", summary="",
        inputs=[ReadValue(name=n, value=v, unit=u,
                          prov=ReadProv(page=page)) for n, v, u in rows],
        results=[], pages_read=[page])


def test_the_fold_keeps_two_rows_that_print_one_number():
    """Model-first answers carry no box, so two rows read once per call are
    one slot only when they read one value. Layers 2 and 4 print the same
    friction angle; they are still two rows."""
    first = _reading([("Phi (deg), Layer 1", 30.0, "deg"),
                      ("Phi (deg), Layer 2", 32.0, "deg")])
    second = _reading([("Phi (deg), Layer 3", 34.0, "deg"),
                       ("Phi (deg), Layer 4", 32.0, "deg")])

    folded = _fold_readings(first, second, fresh=True, same_value=True)

    assert [v.name for v in folded.inputs] == [
        f"Phi (deg), Layer {k}" for k in "1234"]


def test_the_fold_still_takes_one_row_read_twice_as_one():
    first = _reading([("Phi (deg), Layer 2", 32.0, "deg")])
    second = _reading([("Phi (deg), Layer 2", 32.0, "deg")])

    folded = _fold_readings(first, second, fresh=True, same_value=True)

    assert len(folded.inputs) == 1


# ---------------------------------------------------------------------------
# the scorer
# ---------------------------------------------------------------------------

def test_other_rows():
    assert other_rows("Phi (deg), Layer 2", "Phi (deg), Layer 4")
    assert other_rows("Unit weight, Sand", "Unit weight, Clay")
    assert not other_rows("Phi (deg), Layer 2", "phi, layer 2")
    assert not other_rows("Unit weight (kN/m^3)", "Unit weight (kN/m3), Layer 2")
    assert not other_rows("Allowable, case 2, Clay",
                          "Allowable Bearing Capacity (kPa), Clay, Case 2")
    assert not other_rows("Ultimate capacity, Scenario 1",
                          "Ultimate capacity (kips) at 30 ft, Scenario 1")
    assert not other_rows("Settlement (in), last row of the table",
                          "Settlement (in), case 3")
    assert not other_rows("Phi", "Phi (deg), Layer 2")


def test_the_scorer_credits_each_row_only_with_its_own_row():
    truth = {"id": "SYN-calc-1", "kind": "shallow_foundation_bearing",
             "inputs": [{"name": f"Phi (deg), Layer {k}", "value": v[2],
                         "unit": "deg"} for k, v in LAYERS.items()]}
    # Layer 4's row missing: layer 2's phi (also 32) must not answer for it.
    record = _calc([r for r in _model_rows() if r[0] != "Phi (deg), Layer 4"])

    score = score_record(truth, record)

    assert score.scores["inputs"].found == 3
    assert score.scores["inputs"].total == 4


def test_the_model_first_calc_instruction_asks_for_every_row():
    said = mf.instruction("calc", "R00", [0])
    assert mf.CALC_TABLE_ROWS in said
    assert mf.CALC_TABLE_ROWS not in mf.PLAIN_INSTRUCTIONS["calc"]
    assert mf.READ_PROMPT_VERSION == "mf-10"


# ---------------------------------------------------------------------------
# review fixes
# ---------------------------------------------------------------------------

def _credited(truth_name, record_name, value=1.25, unit=""):
    truth = {"id": "SYN-calc-2", "kind": "other",
             "inputs": [{"name": truth_name, "value": value, "unit": unit}]}
    score = score_record(truth, _calc([(record_name, value, unit)]))
    return score.scores["inputs"].found == 1


@pytest.mark.parametrize("truth_name, record_name", [
    # a comma inside brackets, or between digits, splits nothing
    ("Fixed head, 1,200-mm pile", "Fixed head (1,200-mm pile, 2% steel)"),
    ("Kh, active and passive", "Kh (active/passive, half the PGA)"),
    ("Surcharge pressure, second", "Surcharge pressure (A-1, left)"),
    # small words, and a tail that is a value and its unit, name no row
    ("Total cumulative settlement (in), north strip footing on compacted "
     "fill", "Total cumulative settlement (in) - North Strip Footing, "
     "Compacted Fill, 2.5 ksf"),
    ("Footing bearing pressure (ksf), on natural soils",
     "Footing bearing pressure (ksf), Spread Footing - Natural Soils"),
    ("Total cumulative settlement (in), north spread footing on natural "
     "soils", "Total cumulative settlement (in), North Spread Footing - "
     "Bearing on Natural Soils"),
    # a place on the printout is not a row's name
    ("Allowable pressure, second sheet",
     "Allowable pressure (from settlement), Weathered rock"),
    # digits of a description are not a row's number; one name, other words
    ("Safety factor, static circular, 6.5 m wall",
     "Safety factor, S-2 - S-2' Static - Circular Failures"),
])
def test_the_scorer_credits_one_row_named_two_ways(truth_name, record_name):
    """Review fix 1: a correct reading the hand names in other words is
    credited, as master's scorer credited it."""
    assert _credited(truth_name, record_name)


@pytest.mark.parametrize("truth_name, record_name", [
    ("Total cumulative settlement (in), north spread footing on natural "
     "soil", "Total cumulative settlement (in) - North Strip Footing, "
     "Compacted Fill, 2.5 ksf"),
    # two words in common of five are not one row's name
    ("Total cumulative settlement (in), north spread footing on natural "
     "soils", "Total cumulative settlement (in), North Wall Footing - "
     "Bearing on New Fill"),
    ("Unit weight (pcf), dense sand", "Unit weight (pcf), loose sand"),
    ("Embedment (D), scenario 1", "Embedment (D) Scenario 2, Clay"),
    ("Phi (deg), Layer 2", "Phi (deg), Layer 4"),
])
def test_the_scorer_still_refuses_another_rows_number(truth_name,
                                                      record_name):
    """Review fix 1 keeps RI-208's point: another row printing the same
    number does not answer for this one, though the labels are alike."""
    assert same_name(truth_name, record_name)
    assert not _credited(truth_name, record_name)


@pytest.mark.parametrize("left, right, other", [
    ("Settlement (in), Depth (ft) 5.0", "Settlement (in), Depth (ft) 5.5",
     True),
    ("Settlement (in), Depth (ft) 0.5", "Settlement (in), Depth (ft) 5.0",
     True),
    ("Settlement (in), Depth (ft) 2.0", "Settlement (in), Depth (ft) 2.2",
     True),
    ("Settlement (in), Depth (ft) 10", "Settlement (in), Depth (ft) 10.0",
     False),
    ("Settlement (in), Depth (ft) 10.0", "Settlement (in), Depth 10 ft",
     False),
    ("Dq, Width (ft) 3.5, row 2", "Dq, Width (ft) 3.5, row 3", True),
    ("qall (ksf), Width B (ft) 2.0", "qall (ksf), Width B (ft) 3.0", True),
    ("qall (ksf), B = 5 ft", "qall (ksf), B = 6 ft", True),
    ("Phi (deg), Layer 1", "Phi (deg), Layer 10", True),
])
def test_row_numbers_are_compared_as_numbers(left, right, other):
    """Review fix 3: 5.0 and 5.5 are two rows, 0.5 and 5.0 too (their
    digits are one set), 10 and 10.0 one row."""
    assert other_rows(left, right) is other
    assert other_rows(left, right, loose=True) is other


def test_the_fold_keeps_two_depth_rows_that_print_one_number():
    """Review fix 3 in the model-first fold: rows at 5.0 ft and 5.5 ft read
    on one page with one value are two rows."""
    first = _reading([("Settlement (in), Depth 5.0 ft", 0.4, "in")])
    second = _reading([("Settlement (in), Depth 5.5 ft", 0.4, "in"),
                       ("Settlement (in), Depth 5.0 ft", 0.4, "in")])

    folded = _fold_readings(first, second, fresh=True, same_value=True)

    assert sorted(v.name for v in folded.inputs) == [
        "Settlement (in), Depth 5.0 ft", "Settlement (in), Depth 5.5 ft"]
