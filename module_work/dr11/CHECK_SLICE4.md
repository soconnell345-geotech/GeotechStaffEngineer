# DR-11 slice 4: AASHTO 1993 rigid pavement slab thickness D

Branch `claude/dr-11-slice-4-rigid-pavement`. Read with `CHECK.md` (slice 1),
`CHECK_SLICE2.md`, `CHECK_SLICE3.md` and `LABELS_SUMMARY.md` ("Next-slice
coverage": rigid pavement D, 31 readings / 14 unique calcs over 7 reports).
Only `report_ingest/recompute.py` and a new test file changed. The calc
reader (`calc_reader.py`, `model_first*.py`, `calc_scoring.py`) was not
touched. Line numbers are at this branch's head.

## 1. A new kind: the rigid slab thickness D

`_recompute_pavement` (`report_ingest/recompute.py:2361`),
`_RECIPES["pavement"]`. One row per claim (per case where the printout
prints several, by slice 2's `_cases`, unchanged).

**The module function it calls**: `pavement_design.rigid_pavement_design`
through `funhouse_agent.dispatch.call_agent` (the page agent's tool; the
adapter runs `pavement_design.rigid.design_rigid_pavement`, which solves
`geotech_references.aashto_1993.equations.rigid_d_from_w18`, Figure 3.7).
No new formula. The recomputed value is the module's `d_required_in`
(the unrounded required D); the module's own rounding (`d_provided_in`)
is not compared.

**The claims** (`_pave_claims`, line 2042):

* `D`: a result matching the slab pattern (`_SLAB`, line 1958: slab, PCC,
  concrete or pavement thickness, 'D', 'D required', 'design thickness')
  with a length unit. A base, subbase, asphalt, dowel, joint or frost
  dimension is never the slab.
* `W18` (the inverse, item 1's "only if the module offers it": it does,
  as the check mode `slab_thickness_in` -> `w18_capacity`): a W18 result
  named as a capacity ('allowable ESALs', 'W18 capacity'), or any W18
  result when the slab is printed among the inputs and no D is claimed.
  The slab is the printed input, else the one D the printout claims (the
  input says "the slab the printout claims"). The row adds
  `log10_difference`.
* Results that are not claims (a design W18 printed among the results, a
  composite k) move to the inputs first (`_pave_prepare`), as slice 3 does
  for walls, so a case split carries them.

**The inputs** (`_PAVE_INPUTS`, line 1969; `_pave_params`, line 2138):

| printed | module parameter | if not printed |
|---|---|---|
| W18, ESALs, 18-kip, design traffic (or log10 W18; 'million') | `w18` | refused |
| ZR (standard normal deviate) | `zr` (a positive printed ZR read negative, said) | from R |
| reliability R (%, or a fraction under 1, said) | `reliability_pct` -> ZR by the module's Table 4.1 | refused (neither) |
| So, overall standard deviation | `so` | refused: "the module's 0.35 would be a guess" |
| delta PSI / serviceability loss | `delta_psi` | po and pt |
| po, pt | `po`, `pt` | refused when only one of them and no delta PSI |
| S'c, modulus of rupture, flexural strength | `sc_psi` | refused |
| Ec, concrete / elastic modulus | `ec_psi` | refused |
| k, modulus of subgrade reaction (effective, composite) | `k_pci` | refused; a printed MR is named: "k = MR/19.4 holds only for a slab directly on the roadbed, which the printout does not state" |
| J, load transfer coefficient | `j` | refused: "the module's Table 2.6 midpoint would be a guess" |
| Cd, drainage coefficient | `cd` | refused: "the module's 1.0 would be a guess" |

An input printed twice with different values is refused (slice 1's
rule). With delta PSI and po and pt that disagree, delta PSI is used and
the row's `notes` says so; with ZR and R both, the printed ZR, said.

**Units** (`_pave_si`, line 1896): the module's units, as AASHTO prints
them: ESALs, %, psi, pci, inches. psi, pci (psi/in, lb/in3) and inches
pass as printed. MPa and kPa become psi; MPa/m, MN/m3 and kPa/mm become
pci (1 pci = 0.271447 MN/m3); mm and cm become inches. Each converted
input says "printed 4.3 MPa: 623.7 psi", and the row's `units_note` lists
them under "converted once to the module's US customary units". The
row's `unit_si` is `in` or `ESALs`, and `unit_note` says the row is in
the module's own units, not SI.

**pt with delta PSI alone**: pt enters the equation's strength term as
well as delta PSI. The module's pt 2.5 is the recomputed value, listed
under `assumed`, and pt 2.0 and 3.0 run beside it (slice 2's rule: a claim
inside their range is `within_options`).

**Rounding** (`_rounding`, line 2237): a claimed D that is the required D
rounded UP to the next construction increment in the unit it is printed in
(0.5 or 1 in; 5, 10, 25 or 50 mm; 0.5, 1, 2.5 or 5 cm) gets a `rounding`
line on the row. It turns the flag to `agrees` only when the printout says
it rounds (round, rounded, increment, nearest anywhere in its words);
otherwise the flag is slice 2's and the row says "the printout does not
say it rounds".

**The method**: AASHTO printed: "printed: AASHTO 1993" (AASHTO without a
year: the 1993 equation, "the 1986 Guide prints the same"). None printed:
"its inputs (S'c, Ec, k, J, Cd) are those of AASHTO 1993's rigid equation"
(a row is only run when they are all printed). UFC 3-250, TM 5-822, PCA,
Corps, airfield, FAA or Westergaard printed without AASHTO: refused, "not
AASHTO 1993: the UFC 3-250-01 and PCA rigid designs take other inputs".

**Not recomputed, each saying why**: a flexible printout (SN, asphalt,
HMA, 'flexible') and no rigid word: "a flexible pavement (structural
number SN): not recomputed yet". A pavement printout with neither: the
`NOT_RECOMPUTED["pavement"]` reason, now "only the AASHTO 1993 rigid slab
thickness D ... is recomputed; flexible pavements (SN) and UFC 3-250-01
come in a later slice" (slice 1's guard test reads it). A rigid printout
with no D and no allowable W18: `not_compared`. The module refusing
(W18 too small for its 4-20 in bracket): "the module refused: ...".

**Cases**: `_PAVE_NAME_NOISE` (line 274) takes the pavement's own words out
of a label's name, so 'Slab thickness D (truck lane)' and 'Design ESALs W18
(truck lane)' name the case 'truck lane'.

Before: method 'AASHTO 1993', inputs W18 5.1e6, R 95 %, So 0.29, po 4.5,
pt 2.5, S'c 650 psi, Ec 5.0e6 psi, J 3.2, Cd 1.0, k 72 pci; result
'Required slab thickness D' 9.5 in -> one row `not_recomputed`
("pavement_design ... comes in a later slice"). After: recomputed 9.488
in, -0.13 %, `agrees`, with "the claim is the recomputed D (9.488 in)
rounded up to the next 0.5 in; the printout does not say it rounds".

**Tool description**: `RECOMPUTE_TOOL_DESCRIPTION` now names the rigid
slab thickness D (in inches, the module's units) and the allowable W18,
and "a pavement's unprinted pt" among the options. The page fingerprints do
not pin it (the tool is built only with `GEOTECH_REPORT_INGEST`, off in
the recorded pages): `test_page_fingerprints.py` passes unchanged, nothing
re-recorded.

## Tests

`report_ingest/tests/test_recompute_slice4.py`, **31 tests**, made-up
labels and values, every expected value from the module tool. **All 31
fail on master** (run against master's `recompute.py`: 31 failed). They
cover: D through the module; a wrong D differs; no method printed; a
design W18 among the results; ZR printed and a positive ZR read negative;
R as a fraction; no reliability refused; So, J, Cd, S'c, Ec and W18 each
missing refused by name (6); k missing with MR printed; an input twice;
pt alone refused; delta PSI alone with pt options; an SI printout (MPa,
MPa/m, mm) converted; W18 in millions and as a log; a rounded claim agrees
only when the printout says it rounds; a claim below the required D is no
rounding; the module refusing; allowable W18 for a printed slab; W18
capacity of the claimed slab; allowable W18 without a slab refused;
flexible says why; UFC refused; no claim not compared; the reason text;
two sections split by name; the table and the tool description.

Slices 1-3 pass unchanged: `test_recompute.py` 55, `test_recompute_slice2.py`
36, `test_recompute_slice3.py` 85 (slice 1's "every other kind is a row
saying why" still holds for `pavement` with an FS-only printout).

Runs (Python 3.12 venv per the brief):

* `pytest -q -n 4 report_ingest/tests/test_recompute*.py webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  227 passed, exit 0.
* `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py pavement_design`
  at this branch's head: exit 0, 4,941 passed, 23 skipped (pydiggs not
  installed). (A first run had 1 failure, slice 1's guard on the
  `pavement` reason for a printout with no rigid or flexible word; the
  recipe now gives that printout `NOT_RECOMPUTED["pavement"]`, and the
  code commit carries the fix.) pavement_design's own 87 tests are in it.
* Consumers: only `funhouse_agent/deep/agent.py` builds the tool
  (`make_recompute_tool`); no test outside `report_ingest` reads a
  recompute row.

No model prompt changed; no prompt version moved.

## For the lead

**Re-run the local measurement on the saved private records after merge**
(the script behind `LABELS_SUMMARY.md`, at this branch's `recompute.py`):
for the pavement readings (60 / 37 unique / 8 reports; rigid D 31 / 14 /
7), how many give D rows and how many W18 rows; agrees / within options /
differs and the median difference; how many agree only through the
"printout says it rounds" rule, and how many carry a rounding line that
the printout does not state; and why the rest are refused (each input by
name: So, J, Cd, k with or without MR, reliability, serviceability,
twice-printed, UFC/PCA named, the module refusing).

## To measure live

Nothing calls a model. The measurement above is local and offline.

## What is left

* Flexible SN (16 / 5 / 2 in the summary: thin) and UFC 3-250-01 rigid and
  flexible designs: later slices.
* Piles, liquefaction and Schmertmann: coverage to be re-measured after
  RI-208 (mf-10 re-reads) before choosing slice 5.
* OWN-130 stays the owner's.

## Noted, not fixed

* k from a printed MR (k = MR/19.4) is refused even where the printout
  states the slab sits directly on the roadbed; the composite-k worksheet
  (subbase, depth to rigid foundation, loss of support) is not read: the
  printed effective k is the input.
* The design W18 is read as printed: it is not rebuilt from a traffic
  count, growth or lane factors.
* A printed J or Cd given as a range ('3.2-3.8') is read as whatever the
  reader stored; ranges are not run as options.
* A minimum-thickness floor (the module's 4 in, an agency's 6 in) is not
  treated as rounding.
* The 1972 interim guide's equation is not told apart: a printout naming
  'AASHTO 1972' runs the 1993 equation.

## Release note

The report recompute (`recompute_report_calcs`) now recomputes the slab
thickness D an AASHTO 1993 rigid pavement printout claims, through
`pavement_design.rigid_pavement_design`, from the printed W18,
reliability (ZR, or R by Table 4.1), So, serviceability loss, S'c, Ec, k,
J and Cd, in the module's US customary units (an SI printout converted
once, and said). So, J, Cd and k are never taken from the module's
defaults: a printout missing one is refused with its name. A D printed as
the required D rounded up to a construction increment agrees only when the
printout says it rounds; otherwise the row says the rounding. An allowable
W18 printed for a slab is checked by the module's forward equation. Rows
that change: `pavement` calculations claiming a rigid D (or an allowable
W18) were one `not_recomputed` row and are now one row per claim; flexible
and UFC pavement printouts keep one row, with a reason naming what is
recomputed.
