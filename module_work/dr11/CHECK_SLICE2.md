# DR-11 slice 2: bearing and settlement recompute on real printouts

Branch `claude/dr-11-slice-2-cases-labels`. Read with `CHECK.md` (slice 1)
and `LABELS_SUMMARY.md` (the label measurement this slice answers). Only
`report_ingest/recompute.py` and its tests changed; the calc reader was not
touched (RI-208 is another session's).

## What changed, and why

### 1. A calculation split by the case its labels print

`LABELS_SUMMARY.md`: 41 bearing readings put several cases (scenario x
stratum x load case) in one Calculation, so each input was "printed more than
once with different values" and nothing ran.

* `_case_of` (`report_ingest/recompute.py:169`) reads the case a label
  prints, wherever it sits, and returns the label without it. Dimensions
  (`_CASE_DIMS`, line 154): `scenario N`, `case N` / `load case N`,
  `stratum N` / `layer N` / `soil layer N`, `short term` / `long term`
  (`long-term`, `longterm`), and the footing shape (`strip`, `continuous`,
  `square`, `circular`, `round`, `rectangular`, with an optional
  `footing`). An id is a number (with a letter), a roman numeral or one
  letter, so `two-layer`, `layer thickness` and `in case of` are not cases.
  - 'phi, stratum 2, long term' -> `{stratum: 2, term: long}`, 'phi'
  - 'B (Case A)' -> `{case: a}`, 'b'
  - 'Stratum 2 undrained shear strength (short-term)' ->
    `{stratum: 2, term: short}`, 'undrained shear strength'
* Every label pattern now matches the label WITHOUT its case words (`_lab`,
  used by `_Spec.matches`, the claim finders and the factor checks), so
  'Stratum 2 phi' is phi. Shape and method words are still read off the
  printed text as before (`_all_words`).
* `_cases` (line 1121): a case is each distinct combination the printed
  RESULTS carry. A case gets the results with exactly its combination, and
  every input whose own case does not contradict it: an input with no case
  word is shared, 'phi, stratum 1' never reaches a stratum-2 case, the
  short-term su never reaches the long-term case (so the slice-1 refusal of
  "an undrained and a drained strength together" no longer stops a
  printout that gives both cases). An input printed for a dimension the
  case does not name stays in, so two such values still show as printed
  twice, never as a choice.
* `recompute_calculation` (line 1154) runs the recipe once per case that
  prints a claim (`_HAS_CLAIM`), and each row carries `case`
  ("scenario 1, stratum 2, long term", "case A", "strip"). A case with no
  claim (a bearing factor printed per stratum) is not a row. A printout with
  no case word, or whose results carry none, runs exactly as in slice 1.
  `table_markdown` prints the case beside the quantity: `q_ult [stratum 2,
  short term]`.

Before: 'Ultimate bearing capacity, stratum 1, long term' 640 kPa and
'Short-term ultimate bearing capacity, stratum 1' 330 kPa, with
'phi, stratum 1, long term' 30 and 'su, stratum 1, short term' 50 kPa ->
one row, not_recomputed (undrained and drained strength together). After:
two rows, case "stratum 1, long term" (phi 30) and "stratum 1, short term"
(su 50 run as phi = 0), each compared with its own claim.

### 2. Labels

Taken wrongly before (the summary's list):

| Label | Was | Now | Where |
|---|---|---|---|
| B/L, L/B, B/Df, 'B/L ratio' | width / length / Df | none of them | `_RATIO` (line 461) on width, length, depth |
| q/Nc, qc/N | applied pressure | not | `_RATIO` and `^q_?c?\s*/` on q_net |
| 'Water pressure at base' | applied pressure | not | `water` on q_net |
| 'Drain thickness' | layer thickness | not | `drain` on thickness |
| 'Overburden stress ratio', su/sigma'v | sigma_v0, su | not | `_RATIO` on sigma_v0, su, cohesion |
| yes/no settings, 'a) ...', sentences, words | Es, B | skipped | `_skip` (line 213) |
| 'Unconfined compressive strength qu' | su | not | `unconfined` on su and cohesion |

`_RATIO` matches 'ratio' or a known symbol before a slash (B, L, Df, q, qc,
su, cu, sigma..., width, length, depth), so a unit's slash (kN/m2, lb/ft2)
is not a ratio. `_skip` drops a value printed as words, a label with
"yes ... no" or "y/n", a label opening with an enumerator ("a) ", "(1) "),
and a label of more than eight words, counted without its case, its
parentheses and small words ("the", "of", "at", "below" ...): the summary
proposed eight words; counting small words would have skipped real labels
such as 'Effective overburden stress at the middle of the clay layer'.

Patterns added (the summary's proposals):

| Label | Now | Where |
|---|---|---|
| rho, rho_i, Se, delta, delta-h (and the Greek ρ, δ, Δ) | a settlement claim, only with a length unit | `_SETTLEMENT_CLAIM` (line 916), `_GREEK["ρ"]` |
| qall, q_all, qa, q_a, 'qa,net' | an allowable claim (net with 'net') | `_ALLOWABLE` (line 515) |
| q_ult, qult | an ultimate claim | `_BEARING_RESULTS` (added beside qall; not in the summary) |
| 'Height of soil layer (H)', 'layer height' | layer thickness | thickness spec |
| 'Depth to water', 'Water depth' | the water depth | gwt spec |
| 'Depth to water below footing' | the water depth + Df, said on the input | `_water_below_footing` (line 633) |
| 'Unit Wt.', 'Soil density' (in a unit-weight unit) | total unit weight | `_UNIT_WEIGHT` (line 473); 'relative density' and Dr excluded |
| 'd q', 'dg', 'd_c' | a printed depth factor (no depth-factors-off option) | `_DEPTH_FACTOR_LABEL` (line 525) |

Every settlement claim now needs a length unit, the word 'settlement'
included: a 'Time for 90 % settlement (years)' or 'Allowable bearing
pressure (settlement controls)' is no longer a second settlement claim that
made the row "more than one settlement printed".

The do-not-alias list holds and is tested: Esur is not Es; Is, If are not
Iw (an 'Influence factor Is' still refuses the elastic row); sigma'zD is not
sigma'v0; an allowable "from settlement" among the inputs is not a claim.
The prefixed Es ('Clay Es') was not added: the summary rates it low value
until layer tables.

### 3. The shape from the case's own labels

`_shape` (line 342): when the printout's words name several shapes or none,
the shape comes from this calculation's labels (in a split, this case's
labels and the shared ones), said as "printed in this case's labels:
strip". With the case split, a shape word in a claim's label is also a case
dimension, so 'q_ult (strip footing)' and 'Square footing: ultimate bearing
capacity' run as two cases, each with its own B.

### 4. gamma' alone (OWN-130, the owner's open decision)

One setting, `GEOTECH_RECOMPUTE_GAMMA_PRIME` (`GAMMA_PRIME_ENV`, line 79),
or the `gamma_prime=` argument of `recompute_calculation` and
`recompute_record`; the table carries `gamma_prime_rule`. Nothing was
picked for the owner:

* `refuse` (default, today's behaviour): the row is `not_recomputed`,
  `inputs_missing` still says "unit_weight: not printed" and adds "water
  level not printed: only the effective unit weight is printed ('γ'' 9 kN/m3)
  ...", and the row carries `water_level: "not printed"` and
  `gamma_prime_rule: "refuse: ..."`.
* `submerged`: gamma' runs as the unit weight with no water table (gamma'
  above and below the footing base), the unit weight input's note and the
  row's `gamma_prime_rule` say so.
* Under both, gamma' with a printed water depth below the ground is refused
  (the total unit weight above the water is not printed), and gamma' beside
  a total unit weight is listed as not used.
* An unknown setting is a `ValueError` (the tool returns it as its error).

`_gamma_prime` is at line 592. The rule is not in the tool's description,
which did not change.

## Tests

`report_ingest/tests/test_recompute_slice2.py`, 30 tests (offline, made-up
labels in the summary's generic shapes, expected values from the modules):

* 28 fail on master (run against master's `recompute.py`): the case split
  (3 cases incl. the Markdown), ratios, q/Nc and water pressure, drain
  thickness and stress ratio, layer height, settings and sentences,
  unconfined strength, five settlement symbols, qall/q_a/qa,net/q_ult,
  depth to water below the footing and from the ground, Unit Wt. and
  density, d q / dg, the shape from the case's labels (2), gamma' refused /
  submerged / with a water depth / beside a total unit weight / unknown
  rule.
* 2 pass on master by design, as guards: a claim naming no case among cased
  inputs is still refused as "printed more than once" (the split never
  guesses), and the do-not-alias list.

Slice-1 `report_ingest/tests/test_recompute.py`: 55 passed, unchanged.

Runs (Python 3.12 venv per the brief):

* `pytest -q -n 4 report_ingest/tests/test_recompute.py report_ingest/tests/test_recompute_slice2.py`: 85 passed.
* `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`: 4,694 passed, 23 skipped (pydiggs not installed), 0 failed.
* No funhouse_agent test names `recompute_report_calcs`; the tool is built
  in `funhouse_agent/deep/agent.py` from `make_recompute_tool`, whose
  signature and description did not change, so the page fingerprints were
  not re-recorded.

No model prompt changed; no prompt version moved.

## For the lead

**Re-run the local label measurement on the saved private records after
merge** (the script behind `LABELS_SUMMARY.md`, at this branch's
`recompute.py`), once with the default and once with
`GEOTECH_RECOMPUTE_GAMMA_PRIME=submerged`, to get the real numbers: how
many of the 217 bearing cases and the 196 claim rows now recompute and
agree, how many rows the label fixes move, and what the gamma' rule adds.
The summary's what-if (181 of 196 within 10 %) was measured with a
different split, so it is a target, not a result.

## To measure live

Nothing calls a model. The measurement above is local and offline.

## What is left

* Schmertmann (76 % of settlement printouts), layer-table settlement and
  piles wait on RI-208 (whole table rows).
* Slice 3: retaining-wall earth-pressure coefficients (static and
  Mononobe-Okabe), then rigid pavement D.
* OWN-130 stays the owner's: the default stays `refuse` until they answer.

## Noted, not fixed

* A label that names the same dimension twice ('scenario 1 vs scenario 2')
  keeps the first.
* Case words outside the list (a static/seismic case, a footing mark F-1,
  'end of construction') do not split a calculation.
* A dry unit weight is still taken as the unit weight, as in slice 1.
* gamma' with the water printed between the ground and the footing base is
  refused even under `submerged` (no total unit weight above the water).

## Release note

The bearing and settlement recompute (`recompute_report_calcs`) now reads
printouts that put several cases in one calculation (scenario, stratum,
load case, short or long term, footing shape): each case is recomputed with
its own inputs and gets its own row, the case named. It reads more label
spellings (qall/qa, q_ult, rho/Se/delta settlements, layer height, depth to
water, Unit Wt., depth factors 'd q') and no longer takes ratios (B/L, L/B,
B/Df, q/Nc, stress ratios), water pressure, drain thickness, unconfined
strength or yes/no settings as inputs. Rows that change: bearing and
settlement calculations that were `not_recomputed` ("printed more than once
with different values", "an undrained and a drained strength", a ratio's
missing unit) or `not_compared` ("more than one settlement printed", no
claim found under qall/qa/rho) now recompute, one row per case. A printout
that gives only gamma' is still refused by default, now listed as 'water
level not printed'; `GEOTECH_RECOMPUTE_GAMMA_PRIME=submerged` runs it as the
submerged case (OWN-130 is the owner's decision).
