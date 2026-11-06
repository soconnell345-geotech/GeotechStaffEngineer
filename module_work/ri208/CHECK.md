# RI-208: the calc reader keeps only the last row of a printed table

Branch `claude/ri-208-calc-table-rows`, cloud helper, 2026-11-06. Commit
38f9fa3 (code and tests) plus this note. Every value in the tests is made up.

## Where the rows were lost

- **The floor reader.** On master, `report_ingest/calc_reader.py:801`
  (`row = body[-1]` in `_values_from_table`, lines 788-814): a table with a
  header row and several filled body rows gave one value per column taken
  from the LAST filled row, named `<heading>, last row of the table`
  (`LAST_ROW_SUFFIX`, line 156). Every other row was dropped. Today's reader
  keeps a floor value the model did not return, so these labels reached the
  saved records: the 2,112 labels of `module_work/dr11/LABELS_SUMMARY.md`.
- **The model-first reader** never sees the floor (`read_calcs` builds with
  no floor), so nothing in its code dropped a row of an answer. Two things
  made rows unsafe there: its instruction never asked for every row of a
  table, and the fold of two answers (`_read_twice`) took two rows as one
  printed value when their labels passed the partial ratio of 80 ("Phi
  (deg), Layer 2" and "Phi (deg), Layer 4" do) on one page with the same
  number, because model-first answers carry no box.
- **The merge** (`_twin`) picked the reader's row for a floor value among
  rows passing the same partial ratio, and among those at the table's box
  took the first with the same value, so with rows present one layer's
  reading could settle another's.
- **The scorer** (`calc_scoring._find`) credited a truth value from any row
  whose label passed the partial ratio and whose value matched: the wrong
  row of a table could answer.

## The fix

- `calc_reader.py:769` `_row_names`: the qualifier each body row is named
  by: its first cell as printed when it has words ("Silty sand", "Case B"),
  a bare number with its column heading ("Layer 2", "Depth (ft) 10.0"),
  `row N` for an empty first cell, and `, row N` added where two rows would
  share one. A one-row table keeps no qualifier, as before.
- `calc_reader.py:829` `_values_from_table`: one value per row and column,
  `<heading>, <qualifier>`; the qualifier's own column is not a value. The
  printed place is the table's box, as today. Rows with one filled cell
  (ruled empty lines) are still left out. `LAST_ROW_SUFFIX` stays defined so
  saved labels still compare as before.
- `calc_scoring.py:281` `other_rows(a, b)`: two labels name different rows
  when both print row numbers and neither set is within the other (bracketed
  units and exponents ignored: "(kN/m^3)" names no row), or both carry a
  qualifier after their last comma and neither qualifier's words are all in
  the other label. A label with no qualifier, or the old ", last row of the
  table", is no row's in particular. `_find` (line 389) uses it, so a truth
  value is credited only from its own row. Program and method matching
  (`same_name` elsewhere) are unchanged.
- `calc_reader.py:1075` `_same_label` = `_same_name` and not `other_rows`;
  used in `_twin` (line 1186), in `merge_calculation`'s split of floor values
  read at their own place and its "added" count (line 1303 on), and in
  `_read_twice` (line 2556). `_twin` now prefers a row whose label names the
  same thing (`_names_same_thing`) before one that only shares the value.
  The merge's rules are as before: the reader's value at its own printed
  line, both where the place is unknown, a floor value the reader did not
  return kept.
- `model_first.py:578` `CALC_TABLE_ROWS`, added to the model-first calc
  instruction only (the plain baseline's text is unchanged): every row's
  values, labelled with the column heading, a comma and the row's own name.

### Before / after (made up)

A printed table `Layer | Thickness (ft) | Unit weight (pcf) | Phi (deg)` with
rows 1-4 (4.0/115/30, 6.5/120/32, 8.0/125/34, 10.0/130/32).

- Before (floor): `Thickness (ft), last row of the table` 10.0, `Unit weight
  (pcf), last row of the table` 130, `Phi (deg), last row of the table` 32:
  three values, rows 1-3 lost.
- After (floor): twelve values, `Thickness (ft), Layer 1` 4.0 ... `Phi (deg),
  Layer 4` 32. Merged with a model-first answer naming the same twelve
  labels and no boxes: twelve rows, twelve reconciled, no disagreement,
  although layers 2 and 4 print the same phi.

## Prompt version

**Moved: `READ_PROMPT_VERSION` mf-9 -> mf-10** (`model_first.py:316`), its
digest recorded in `report_ingest/tests/test_model_first_schema.py`
(`MODEL_FIRST_ASKS["mf-10"]`). The next live run re-reads every saved
model-first answer. Today's reader's prompts (`CALC_READER_SYSTEM`,
`CALC_TOOLS`) and the page fingerprints are unchanged (fingerprints test
passes without a re-record).

## Tests

`report_ingest/tests/test_ri208.py`, 10 tests. Run against master's code
(the test file with the two new helpers stubbed): **9 fail, 1 passes** (the
guard that one row read twice still folds to one, which is meant to pass on
both).

- floor: the 4-row table gives all 12 values, none "last row"; one-row
  tables carry no qualifier; qualifiers from the first cell, distinct;
- merge: a canned model-first answer (no boxes) against the floor keeps all
  12, reconciles 12, no disagreement; a row the model left out is kept from
  the floor;
- model-first fold: two answers whose rows of layers 2 and 4 print one phi
  keep all 4 rows; one row read twice still folds;
- scorer: `other_rows` cases; a truth of four layers credits 3 when layer 4
  is missing although layer 2 prints the same value (master credits 4);
- the model-first calc instruction carries `CALC_TABLE_ROWS`, the plain one
  does not.

Changed: `test_calc_reader.py::test_every_filled_row_of_a_table_is_read`
(was `test_the_last_filled_row_of_a_table_is_read`, which asserted the old
"last row" label); `test_model_first_schema.py::
test_the_model_first_instructions_add_to_the_plain_ones` (calc now adds one
sentence).

Runs (python 3.12 venv per the brief):

- `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py
  webapp/tests/test_page_fingerprints.py`: **4674 passed, 23 skipped
  (pydiggs not installed), 0 failed**.
- `pytest -q -n 4 report_ingest -k calc`: 224 passed before the test
  updates (1 failed: the old last-row test, as expected).
- `pytest -q -n 4 module_work/report_ingest_harness/tests`: **182 passed,
  34 skipped, 0 failed**.

Consumers grepped: `recompute.py`, `cluster_scoring.py`,
`model_first_scoring.py`, `measure_wp5_calc.py`, `dry_run_model_first.py`,
`measure_grouping.py`, `run_plain_baseline.py`; none reads the
"last row" label. `webapp/`, `funhouse_agent/`: only `funhouse_agent/deep/
agent.py` wires `recompute_report_calcs`, whose code is unchanged.

## For the lead

Get the real before/after offline, no model call:

1. Re-score the saved calc runs: `measure_wp5_calc.py` (and
   `score_on_cluster(stages=("calc",), rescore=True)` on the saved `calc/`
   run files) with this branch against master: floor inputs/results found,
   and the counts of floor values, reconciled and disagreements per printout.
   The floor half re-runs over the PDFs; the saved model answers fold and
   merge again under the new label rule.
2. Re-run the DR-11 label measurement locally over re-floored records: the
   count of labels ending "last row of the table" should go to 0 and the
   per-kind label counts should rise by the rows recovered (bearing 31,
   settlement 83, axial 40, slope 52, site response 13, pavement 16 readings
   had them).

## To measure live

- One model-first calc run under mf-10 on a few printouts with layer or case
  tables: does the model give every row, labelled `<heading>, <row>`, and
  does the output stay inside the cap on long tables (a 40-row p-y or
  settlement table is now 40 x columns values)?

## Release note

Calculation tables now keep every row. A printed table inside a calculation
printout (a soil profile, a layer-by-layer settlement, a load schedule)
reaches the record as one value per row and column, labelled by the column
heading and the row ("Unit weight (pcf), Layer 2"), instead of only its last
row labelled "..., last row of the table". Saved calc values gain the other
rows when a report is ingested again; the last row is still there, under its
own row's label. The model-first calc prompt is mf-10, so the next live run
re-reads saved model-first answers. `recompute_report_calcs` sees a
parameter printed once per layer as "printed more than once with different
values" and declines it rather than recomputing from the last row's value.

## What is left / noted, not fixed

- Noted, not fixed: the floor still gives a whole-table box for every row's
  values; planlens cells carry no per-row box here. A per-row box would let
  the merge settle a row at its own line rather than the table's.
- Noted, not fixed: `MAX_FLOOR_KEEP` (300) and `MAX_FLOOR_VALUES` (140, the
  part shown to today's model) are unchanged; a long tabulated printout now
  reaches them sooner, and the cap is warned of as before.
- Noted, not fixed: today's (non-model-first) calc prompt does not ask for
  row-qualified labels; its floor now hands it every row, and rows it reads
  under another label stand beside the floor's.
- `recompute` (DR-11) declines a layered table instead of using one row;
  mapping rows to layers is the later slice the DR-11 summary names.
