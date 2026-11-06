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
from report_ingest.calc_scoring import other_rows, score_record
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
