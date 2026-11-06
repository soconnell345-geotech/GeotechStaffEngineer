# DR-11 slice 4: AASHTO 1993 rigid pavement slab thickness D

Branch `claude/dr-11-slice-4-rigid-pavement`. Read with `CHECK.md` (slice 1),
`CHECK_SLICE2.md`, `CHECK_SLICE3.md` and `LABELS_SUMMARY.md` ("Next-slice
coverage": rigid pavement D, 31 readings / 14 unique calcs over 7 reports).
Only `report_ingest/recompute.py` and a new test file changed. The calc
reader (`calc_reader.py`, `model_first*.py`, `calc_scoring.py`) was not
touched. Line numbers are at this branch's head, after the review fixes
(87c555c); the text below is corrected for them (see "Review fixes").

## 1. A new kind: the rigid slab thickness D

`_recompute_pavement` (`report_ingest/recompute.py:2382`),
`_RECIPES["pavement"]`. One row per claim (per case where the printout
prints several, by slice 2's `_cases`, unchanged).

**The module function it calls**: `pavement_design.rigid_pavement_design`
through `funhouse_agent.dispatch.call_agent` (the page agent's tool; the
adapter runs `pavement_design.rigid.design_rigid_pavement`, which solves
`geotech_references.aashto_1993.equations.rigid_d_from_w18`, Figure 3.7).
No new formula. The recomputed value is the module's `d_required_in`
(the unrounded required D); the module's own rounding (`d_provided_in`)
is not compared.

**The claims** (`_pave_claims`, line 2065):

* `D`: a result matching the slab pattern (`_SLAB`, line 1959: slab, PCC,
  concrete or pavement thickness, 'concrete paving', 'D', 'D required',
  'design thickness') with a length unit. A base, subbase, asphalt, dowel,
  joint (the word: 'jointed plain concrete pavement' is the slab) or frost
  dimension, or a total thickness, is never the slab.
* `W18` (the inverse, item 1's "only if the module offers it": it does,
  as the check mode `slab_thickness_in` -> `w18_capacity`): only a W18
  result named as a capacity ('allowable ESALs', 'W18 capacity'). Any
  other W18 result is the design traffic, an input, even beside a slab
  printed as an input with no D claimed.
  The slab is the printed input, else the one D the printout claims (the
  input says "the slab the printout claims"). The row adds
  `log10_difference`.
* Results that are not claims (a design W18 printed among the results, a
  composite k) move to the inputs first (`_pave_prepare`), as slice 3 does
  for walls, so a case split carries them.

**The inputs** (`_PAVE_INPUTS`, line 1977; `_pave_params`, line 2159):

| printed | module parameter | if not printed |
|---|---|---|
| W18, ESALs, 18-kip, design traffic (or log10 W18; 'million'); not a year's or a day's ESALs ('annual', 'first-year', '/yr', 'daily') | `w18` | refused |
| ZR (standard normal deviate) | `zr` (a positive printed ZR read negative, said) | from R |
| reliability R (%, 'percent', 'per cent', 'pct', or a fraction under 1, said) | `reliability_pct` -> ZR by the module's Table 4.1 | refused (neither) |
| So, overall standard deviation, overall deviation | `so` | refused: "the module's 0.35 would be a guess" |
| delta PSI / serviceability loss | `delta_psi` | po and pt |
| po, pt (also PSIi / PSIo, PSIt / PSIf) | `po`, `pt` | refused when only one of them and no delta PSI |
| S'c, modulus of rupture, flexural strength | `sc_psi` | refused |
| Ec, concrete / elastic modulus | `ec_psi` | refused |
| k, modulus of subgrade reaction (effective, composite); not a dynamic, back-calculated or FWD k | `k_pci` | refused; a printed MR is named: "k = MR/19.4 holds only for a slab directly on the roadbed, which the printout does not state" |
| J, load transfer coefficient | `j` | refused: "the module's Table 2.6 midpoint would be a guess" |
| Cd, drainage coefficient; not a base, subbase or layer's coefficient | `cd` | refused: "the module's 1.0 would be a guess" |

A label naming the flexible design a printout prints beside the rigid one
(`_FLEX_LABEL`: flexible, asphalt, bituminous, HMA, AC, structural number)
is never read as W18, log W18, ZR, R, So, delta PSI, po, pt, Ec or Cd. J
keeps its labels (they name an asphalt shoulder).

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

**Rounding** (`_rounding`, line 2258): a claimed D that is the required D
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
HMA, 'flexible') and no rigid word ('concrete' in 'asphalt concrete' or
'bituminous concrete' is not one): "a flexible pavement (structural
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

`report_ingest/tests/test_recompute_slice4.py`, **51 tests** (31 here and
20 from the review fixes, listed under "Review fixes"), made-up labels and
values, every expected value from the module tool. **All 51 fail on
master** (run against master's `recompute.py`: 51 failed). The first 31
tests
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

Runs (Python 3.12 venv per the brief), at 95f0706, before the review
fixes (the runs after them are under "Review fixes"):

* `pytest -q -n 4 report_ingest/tests/test_recompute*.py webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  227 passed, exit 0.
* `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py pavement_design`:
  exit 0, 4,941 passed, 23 skipped (pydiggs not
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

The review ran it before and after its fixes: see "Review fixes". (It
counts a unique calc by report and first page, which gives 31 for the
pavement readings.)

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

## Review fixes

An independent review (merge after fixes) asked for five fixes. Each is a
commit with its tests in `test_recompute_slice4.py`, made-up labels and
values (none is a label in any saved record or a value in a saved pavement
record; checked). The review's notes-only items are left as they are.
The text above is corrected for the fixes.

* **1, the inverse W18** (75f1d25). A W18 result was the inverse claim
  whenever the slab was printed among the inputs and no D was claimed. A
  check printout that also works out its design W18 from a traffic count
  then compared that design W18 with the slab's capacity: a false
  'differs' (over +500 % in the review's repro). Now only a W18 named as a
  capacity ('allowable', 'capacity', 'predicted', 'carried', ...) is the
  inverse; any other W18 result is the design traffic, an input. Test:
  `test_design_traffic_beside_a_printed_slab_is_no_capacity_claim` (a
  design W18 beside a printed slab is `not_compared`; with an allowable
  W18 printed too, only that one is a claim, and it agrees).
* **2, three labels read as the wrong input** (c0964b3). (a) 'Total
  pavement thickness' (the slab and the layers under it) was claimed as
  D: `_SLAB` leaves out a total. (b) A year's or a day's ESALs ('annual',
  'yearly', 'daily', 'first-year', '/yr', '/day') was read as the design
  W18 and gave a false 'differs': W18 leaves them out, and a printout with
  only them is refused ("design ESALs W18: not printed"). (c) A dynamic,
  back-calculated or FWD k was run as the design k: k leaves it out, so a
  printout with only a dynamic k is refused, and one with both uses the
  static k. Tests: `test_a_total_thickness_is_not_the_slab`,
  `test_a_year_or_a_day_of_traffic_is_not_w18` (4 labels),
  `test_a_dynamic_k_is_not_the_design_k`.
* **3, a flexible design printed beside the rigid one** (c02b432). (a)
  'concrete' in 'asphalt concrete' or 'bituminous concrete' made a
  flexible printout rigid, and its row said `not_compared` instead of "a
  flexible pavement (SN): not recomputed yet": `_pave_kind` takes the
  asphalt concrete layer's words out before looking for a rigid word
  (`_ASPHALT_CONCRETE`). (b) The flexible design's own W18, reliability,
  So, serviceability, asphalt modulus and layer drainage coefficients were
  read as rigid inputs, so W18 and the serviceability were "printed more
  than once" and the row was refused: `_FLEX_LABEL` leaves those labels
  out of W18, log W18, ZR, R, So, delta PSI, po, pt and Ec, and Cd also
  leaves out a base, subbase or layer's coefficient. Tests:
  `test_an_asphalt_concrete_printout_is_flexible`,
  `test_the_flexible_design_beside_the_rigid_one_is_not_read`.
* **4, two spellings** (0e37343). (a) Reliability printed with the unit
  'percent', 'per cent', 'pct' or 'percentage' was refused as "not a
  percentage": the words now read as %. (b) So printed as 'overall
  deviation' was not read ("So not printed"): it now is. Tests:
  `test_reliability_printed_in_percent_words` (5 spellings),
  `test_overall_deviation_is_so`.
* **5, three more labels** (5dd50ff). (a) The slab's exclusion 'joint'
  also left out 'jointed plain concrete pavement thickness': it is now the
  word 'joint' or 'joints'. (b) Serviceability printed as PSIi / PSIo and
  PSIt / PSIf was not read (po refused, pt the assumed 2.5): po and pt now
  read them. (c) 'Concrete paving' and 'PCC paving' were not the slab:
  they now are. Tests: `test_slab_labels_jointed_and_paving` (3 labels),
  `test_serviceability_printed_as_psi_i_and_psi_t`, and
  `test_a_joint_dimension_is_still_not_the_slab` (a guard: a joint spacing
  is still not the slab; it passes before the fix too).

**Tests after the fixes**: `test_recompute_slice4.py` has 51 tests (31 +
20). Run against each earlier `recompute.py`, the failures fall by exactly
each commit's own tests: master 51, 95f0706 19, fix 1 18, fix 2 12, fix 3
10, fix 4 4, fix 5 0. `pytest -q report_ingest/tests/test_recompute.py
report_ingest/tests/test_recompute_slice2.py
report_ingest/tests/test_recompute_slice3.py
report_ingest/tests/test_recompute_slice4.py pavement_design
webapp/tests/test_page_fingerprints.py webapp/tests/test_public_text.py`:
exit 0, 334 passed (314 before the fixes).

**The local measurement after the fixes**: the saved private records,
offline, master (bb97c70) vs this branch before (95f0706) and after the
fixes (5dd50ff, the same rows as at 87c555c, its code unchanged since).
Bearing (54 readings, 220 rows), settlement (255, 478) and retaining-wall
(93, 268) rows are identical to master's, row for row, before and after.

Pavement (unique calc = report and first page):

| measure | master | branch before the fixes | branch after |
|---|---|---|---|
| readings / unique calcs / reports | 60 / 31 / 8 | 60 / 31 / 8 | 60 / 31 / 8 |
| rows | 60 | 139 | 146 |
| read as rigid (readings / unique / reports) | - | 55 / 29 / 8 | 51 / 26 / 8 |
| D claim rows / W18 claim rows | 0 / 0 | 98 / 2 | 115 / 0 |
| readings with a claim (unique) | 0 | 21 (7) | 29 (13) |
| rows recomputed (readings / unique / reports) | 0 | 6 (4 / 3 / 1) | 24 (13 / 5 / 2) |
| agrees / within options / differs / below | - | 6 / 0 / 0 / 0 | 24 / 0 / 0 / 0 |
| median difference | - | 0.50 % | 0.05 % |
| agree only because the printout says it rounds | - | 0 | 0 |
| not_compared | 0 | 34 | 22 |
| not_recomputed | 60 | 99 | 100 |

What moved, measured at each fix commit in turn:

* Fix 1: 1 reading. Its design W18 was taken as 2 inverse claims (both
  refused); it now has no claim and is `not_compared`.
* Fix 2: no saved row moves (the review's repros were made up).
* Fix 3: 4 flexible printouts go from `not_compared` to "a flexible
  pavement".
* Fix 4: 9 readings go from refused to recomputed, all agreeing (the
  two spellings, with fix 3's flexible labels out of the way).
* Fix 5: 9 readings with a slab label it now reads go from
  `not_compared` to a D claim, all refused (below). The 4 readings
  recomputed before the fixes (one report) now take their printed initial
  and terminal serviceability (PSIi / PSIf) where pt was assumed: still
  agreeing, the differences now 0.22 % at most. They also print a
  'jointed' slab label beside each D, now a second D row per case with
  the same value (6 rows, all agreeing): 18 recomputed rows become 24.

No row differs before or after.

Why the other 122 rows are not recomputed after the fixes (100 refused,
22 with no claim; rows / unique calcs; a row can have more than one
reason; before the fixes in brackets):

| reason | rows / unique |
|---|---|
| W18 not printed, serviceability not printed, S'c printed twice (all one catalog page) | 72 / 1 (72 / 1) |
| So not printed (72 of the rows that page) | 73 / 2 (93 / 4) |
| Ec, J, Cd, k each not printed (72 of the rows that page) | 73 / 2 (75 / 3) |
| reliability in 'percent' | 0 (85 / 3) |
| reliability not printed | 0 (2 / 1) |
| only pt printed (no po, no delta PSI) | 0 (2 / 1) |
| S'c not printed | 1 / 1 (3 / 2) |
| W18 printed twice | 6 / 2 (17 / 1) |
| serviceability printed twice | 2 / 1 (15 / 1) |
| So printed twice | 2 / 1 (0) |
| k printed twice | 11 / 6 (0) |
| S'c printed without a unit (or as a formula of f'c) | 7 / 4 (2 / 1) |
| Ec printed without a unit | 2 / 1 (2 / 1) |
| a flexible pavement | 8 / 4 (4 / 2) |
| neither rigid nor flexible words | 1 / 1 (1 / 1) |
| no claim among the results (`not_compared`) | 22 / 20 (34 / 26) |

The new "k printed twice" rows are the newly claimed 'concrete paving'
slabs: those printouts work the composite k through several steps (the
worksheet "Noted, not fixed" names) and print a k at each, so the row is
refused rather than a step's k guessed. W18 printed twice comes from two
printouts: one whose flexible values the reader stored without the word
'flexible' (its So printed twice too), and one that prints each vehicle
class's ESALs beside the total (also refused for its k). Each refusal
names the input.

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
W18 printed for a slab (a W18 named as a capacity: 'allowable',
'capacity', ...) is checked by the module's forward equation; any other
W18 is read as the design traffic. Common spellings are read ('percent'
as %, 'overall deviation' as So, PSIi and PSIt as po and pt, a 'jointed'
or 'concrete paving' thickness as the slab); a total
thickness, a year's or a day's ESALs and a dynamic k are not read as D,
W18 and k, and a flexible design printed beside the rigid one is left out
of the rigid inputs. Rows that change: `pavement` calculations claiming a
rigid D (or an allowable W18) were one `not_recomputed` row and are now
one row per claim; flexible and UFC pavement printouts keep one row, with
a reason naming what is recomputed.
