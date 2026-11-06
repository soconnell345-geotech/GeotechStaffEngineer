# DR-11 slice 3: method names read with spelling tolerance; retaining-wall earth-pressure coefficients

Branch `claude/dr-11-slice-3-earth-pressure`. Read with `CHECK.md` (slice 1),
`CHECK_SLICE2.md` (slice 2) and `LABELS_SUMMARY.md` ("Next-slice coverage").
Only `report_ingest/recompute.py` and its tests changed. The calc reader
(`calc_reader.py`, `model_first*.py`, `calc_scoring.py`) was not touched:
RI-208 is under review there. Line numbers are at this branch's head.

## 1. Method names read with spelling tolerance

`_read_methods` (`report_ingest/recompute.py:463`) reads a printed method
word that is one letter off exactly one known method (two letters only for
a name of eight letters or more, with its last letter and no English
ending: review fix F1, below), with the same first letter, as that method.
A swap of two letters counts as one (`_edits`, Damerau-Levenshtein).

* Known names (`_KNOWN_METHODS`, line 436): Meyerhof, Terzaghi, Vesic,
  Hansen, Rankine, Coulomb, Mononobe, Okabe, Schmertmann, Caquot, Kerisel.
  Hough and Jaky are left out: they are too short to read past English
  words (rough, tough, jake).
* Never read: a word as close to two names; a word whose first letter
  differs; words of under four letters; a short list of English words
  within reach (`_NOT_METHODS`: ranking, ranked, kernel, column, Hanson,
  Meyer...).
* Accents are dropped first (Vesić is Vesic).
* Every row whose method was read so says it, in `method_basis`
  ("printed: Mayerhof (1963); the method name read: 'mayerhof' read as
  Meyerhof (one letter off)") and in a new row field `method_spelling`
  (a list of those notes). A correctly spelled name adds nothing.
* Where it applies: `_method_words` (bearing's method and program, and so
  bearing's method choice), the settlement method from the method words
  and the summary (so a misspelled Schmertmann is refused as Schmertmann,
  with the note), and the wall methods (the claim's label, the method and
  program, the subject and summary).

Before: method 'Mayerhof' -> "the printout names no method the module
offers": Vesic as the recomputed value, Meyerhof and Terzaghi beside it,
so a claim near Meyerhof's value was `within_options`. After: Meyerhof is
run as the printed method, and the row says how the name was read.

## 2. A new kind: retaining-wall earth-pressure coefficients

`_recompute_wall` (line 1793), `_RECIPES["retaining_wall"]`. A
`retaining_wall` calculation is recomputed coefficient by coefficient; one
claimed-vs-recomputed row per printed coefficient, with `coefficient` (Ka,
Kp, K0, KAE, KPE) on the row.

**The module functions it calls** (through `funhouse_agent.dispatch.call_agent`,
as the page's agent does, so the result carries the module's `choices` and
`defaults_used`; no new formula):

| Coefficient | Tool | Underlying module functions |
|---|---|---|
| Ka, Kp (Rankine) | `retaining_walls.earth_pressure_coefficient`, theory `rankine` | `sheet_pile.earth_pressure.rankine_Ka` / `rankine_Kp`; `retaining_walls.earth_pressure.rankine_Ka_sloped` for a sloping backfill |
| Ka, Kp (Coulomb) | same, theory `coulomb` (delta_deg, alpha_deg, beta_deg) | `sheet_pile.earth_pressure.coulomb_Ka` / `coulomb_Kp` |
| Kp (log spiral) | same, theory `caquot_kerisel` (UFC 3-220-20 by default; `caquot_kerisel_1948` when Caquot or Kerisel is printed) | `geotech_common.log_spiral` |
| K0 | same, state `at_rest` (OCR for EN 1997-1 Eq. 9.1, beta for Eq. 9.2) | `sheet_pile.earth_pressure.K0` (Jaky) |
| KAE, KPE | `seismic_geotech.seismic_earth_pressure` (phi, delta, kh, kv, beta, i; include_passive with delta_passive = the printed delta) | `seismic_geotech.mononobe_okabe.mononobe_okabe_KAE` / `mononobe_okabe_KPE` |

**The claims** (`_wall_claim_key`, line 1316): a result whose label prints
a K symbol or the word 'coefficient' and Ka / K_a / Kₐ / 'active', Kp,
K0 / Ko / 'at-rest', KAE, KPE, as a number without a unit (an 'active earth
pressure' in kPa is no claim). A Ka or Kp named seismic, dynamic or M-O is
KAE or KPE. A seismic increment (delta KAE) is a `not_compared` row.

**The inputs** (`_WALL_INPUTS`, line 1357): phi (not a wall or interface
friction); the wall friction delta ('wall friction', 'interface friction',
'delta'), or delta/phi as a ratio (delta = ratio x phi, said on the
input); the backfill slope; a slope in front of the wall (for Kp, KPE
only); a wall batter; a bare 'beta' (under 45 deg the backfill slope, from
45 the wall's angle from horizontal, said); kh, kv, PGA and As (unitless
or in g); OCR. A wall printout's worked-out values that are not
coefficients (a kh derived from the PGA) move to the inputs first
(`_wall_prepare`, line 1343), so a case split carries them.

**The method** (`_wall_method`, line 1499), the claim's own label first,
then the method and program, then the subject and summary:

* printed Rankine, Coulomb or log spiral: that one. Rankine with a
  printed wall friction runs Coulomb (and for Kp the log spiral) beside it.
* several printed: the first is the recomputed value, the others beside it.
* none printed, a wall friction printed: Rankine is the recomputed value,
  Coulomb (and the log-spiral Kp) beside it.
* none printed and no wall friction: refused, "the printout names no
  method (Rankine or Coulomb) and prints no wall friction to run Coulomb
  beside Rankine".
* K0: Jaky (OCR printed: EN 1997-1 Eq. 9.1 with it beside).
* KAE, KPE: Mononobe-Okabe, printed or (none printed) the module's seismic
  coefficient, said.

**kh** (`_wall_kh`, line 1447; `_KH_RULES`, line 1397): as printed; else
from a printed PGA or As by a rule the printout states anywhere in its
text ('kh = 0.5 PGA', 'kh = PGA/2', 'kh = As', 'kh taken as one-half of
the PGA'), said on the input. A PGA with no stated rule is refused: "kh
= PGA/2 or kh = PGA would be a guess". Two different stated rules are
refused.

**Options, by slice 2's rules** ('agree' within 10 %; 'within options'
when another printed convention reproduces it; 'differs' outside them
all):

* the other method (above);
* a wall batter whose sense the label does not state runs both ways
  (`_batter_senses`, line 1518): from vertical, the soil overhanging the
  back face (UFC 3-220-20's theta; the default) and the wall leaning toward
  the backfill; from horizontal (or 45 deg and more), the printed alpha and
  180 - alpha;
* a kv runs with both signs (the module takes kv positive upward);
* a log-spiral Kp named only "log spiral" runs the Caquot-Kerisel (1948)
  chart beside UFC 3-220-20.

**Refusals, each with the reason**: phi not printed or printed twice;
Coulomb, log spiral or M-O without a wall friction ("the module's default
would be a guess"); M-O without kh (above); the module refusing a case
(Rankine Kp behind a sloping backfill, a backfill steeper than phi) as
"the module refused: <its own message>".

**What is not passed**: a backfill slope for a passive coefficient (the
backfill is behind the wall; the passive side's slope is the 'front' or
'toe' slope); a batter for Rankine (a vertical plane) or K0, or for Kp and
KPE (the back face's batter; review fix F4). Each is said
on the input. Defaults the module took (level backfill, vertical wall,
kv 0, OCR 1) are listed under `assumed`, from the module's own
`defaults_used`.

**Cases**: slice 2's split (`_cases`) applies unchanged, with the wall's own
words (`_WALL_NAME_NOISE`, line 251: coefficient, active, seismic, wall,
batter, kh...) taken out of the names, so 'Ka (silty sand)' and 'phi
(silty sand)' name the case 'silty sand', and 'Ka, stratum 2' takes
'Stratum 2 phi' only.

Before: a `retaining_wall` calculation was one row, `not_recomputed`
("retaining_walls, sheet_pile and soe recompute walls in a later slice").
After, e.g. method 'Rankine', inputs 'Backfill friction angle phi' 32 deg,
results 'Ka' 0.307 and 'Kp' 3.25: two rows, Ka recomputed 0.3073 and Kp
3.2546, both `agrees`. A wall printout with no coefficient keeps one
`not_recomputed` row, the reason now naming what is recomputed.

**Shared changes**: `_GREEK` maps β, α, θ and the subscripts ₐ ₚ ₀ ₒ ₕ ᵥ;
`_si` takes a kind `accel` (unitless or g); `_names_of` takes the kind.
None changes a bearing or settlement value (slice 1 and 2 tests pass).

**Tool description**: `RECOMPUTE_TOOL_DESCRIPTION` now says the wall
coefficients are recomputed and that a batter's or kv's sense is among the
options run. The page fingerprints do not pin it (the tool is built only
with `GEOTECH_REPORT_INGEST`, off in the recorded pages), and
`test_page_fingerprints.py` passes unchanged, so nothing was re-recorded.

## Tests

`report_ingest/tests/test_recompute_slice3.py`, 32 tests, made-up labels
and values, the expected values from the module tools:

* 29 fail on master (run against master's `recompute.py`): misspelled
  Meyerhof, Terzaghi, Vesic read (3); English words not read; a word as
  close to two names not read; misspelled Coulomb and Mononobe-Okabe read;
  Rankine Ka/Kp/K0; Rankine on a sloping backfill; Coulomb with a batter of
  unknown sense; Coulomb without delta refused; no method with delta; no
  method and no delta refused; delta/phi; log-spiral Kp; K0 with OCR; a
  wrong Ka differs; M-O KAE with kh; kv both signs; kh from the PGA by a
  stated rule (2); a PGA never halved; M-O KPE; an increment not compared;
  phi missing; phi twice; strata split; soils named split; the tool
  description.
* 3 pass on master by design, as guards: a correctly spelled method says
  nothing about spelling; a pressure in kPa is not a coefficient; a wall
  with no coefficient says why.

Slice 1 (`test_recompute.py`, 55) and slice 2 (`test_recompute_slice2.py`,
36) pass unchanged. Slice 1's "every other kind is a row saying why" still
holds for `retaining_wall` (a FS-only printout keeps its
`NOT_RECOMPUTED` reason).

Runs (Python 3.12 venv per the brief):

* `pytest -q -n 4 report_ingest/tests/test_recompute*.py webapp/tests/test_public_text.py`:
  128 passed.
* `pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  4,732 passed, 23 skipped (pydiggs not installed), 0 failed.
* No funhouse_agent test names `recompute_report_calcs`; the tool is built
  in `funhouse_agent/deep/agent.py` from `make_recompute_tool`.

No model prompt changed; no prompt version moved. The tool description
changed (above); no fingerprint pins it.

## For the lead

**Re-run the local label measurement on the saved private records after
merge** (the script behind `LABELS_SUMMARY.md`, at this branch's
`recompute.py`): how many retaining-wall readings (93, 56 unique, 7
reports) now give coefficient rows, how many agree / within options /
differ, and how many are refused for each reason (no method and no delta,
delta missing, kh rule not printed, phi printed twice); and how many
bearing and settlement rows the method-spelling reader moves (the summary:
a misspelled Meyerhof kept 8 claim rows at 'within options' and hid a phi
reading error in 6 more).

## To measure live

Nothing calls a model. The measurement above is local and offline.

## What is left

* Rigid pavement D (AASHTO 1993) is slice 4.
* The wall's stability checks (sliding, overturning, bearing, the thrust)
  are not recomputed: a later slice.
* Piles, liquefaction and Schmertmann wait on RI-208 (whole table rows).
* OWN-130 stays the owner's.

## Noted, not fixed

* A printout giving a retained-soil phi and a foundation-soil phi with no
  case word or soil name is refused as "printed more than once"; Ka is
  not matched to the retained phi and Kp to the foundation phi by their
  words.
* A sloping-backfill Rankine Kp and a K0 behind ground falling away are
  refused by the module (its own reasons), not run another way.
* AASHTO's kh = 0.5 kh0 with kh0 = Fpga x PGA is read only where the
  printout states kh against As or PGA directly; a rule printed with Fpga
  in it is refused (review fix F2); As is not computed from Fpga and PGA.
* A batter label stating its sense in other words than toward / away from
  the backfill or "soil overhang" runs both senses.

## Review fixes

An independent review (merge after fixes) asked for four fixes. Each is a
commit with its tests in `test_recompute_slice3.py`, made-up values. Line
numbers above are at de4b88c, before them. N1 is left as it is; F5
(Meyerhof's depth term beyond D/B = 1) is BACKLOG GM-309.

* **F1, method spelling** (f6e4d98). One letter off for a name of seven
  letters or fewer (Rankine, Coulomb, Kerisel, Hansen, Caquot, Okabe,
  Vesic). Two letters only for Meyerhof, Terzaghi, Mononobe and
  Schmertmann, and then only with the name's own last letter and no
  English ending (-ing, -ole, -one, -oir). `_NOT_METHODS` gains Rankin,
  Vesica and Hanse. Before, 'ranging', 'raking', 'ravine', 'Racine',
  'monopole', 'monotone', 'couloir' and 'Kessel' were read as a method,
  and a wall printout naming none ran Rankine. 'Coloumb' (two letters off a
  seven-letter name) is no longer read; 'Coulumb' and 'Colomb' are.
  Tests: 20 ordinary words, names and places never read; 15 misspellings
  still read; a wall summary with such words is still refused for naming
  no method.
* **F2, the kh rule** (df78b5e). Read only as the printout states it:
  'kh = <factor> x PGA', 'kh = PGA/2', 'kh = PGA', 'kh taken as one-half
  of the PGA'. A kh value with the PGA beside it ('kh = 0.12 (PGA 0.24
  g)', 'kh = 0.12, PGA 0.24') is no rule; before, it was read as 0.12 x
  PGA. 'as' followed by a word is not As ('kh = 0.15 as recommended' was
  read as 0.15 x As). A rule carrying the site factor ('kh = 0.5 PGA x
  Fpga', 'kh = 0.5(PGA)(Fpga)') is refused with the reason; before, the
  factor was dropped. Tests: six forms read, four not, three refused, and
  a wall row with a kh value beside the PGA refused.
* **F3, labels** (ad6ea9f). The increment sign U+2206 maps as the Greek
  capital delta, so '∆ KAE' is a `not_compared` increment, not a KAE
  claim. 'Eq.' or 'Eq 3' (an equation) no longer makes a static Ka
  seismic; 'EQ' alone still does. Tests for both.
* **F4, the batter** (11f51e4). A printed batter is the back face's: it
  now goes to Ka and KAE only. Kp and KPE run with a vertical face, and
  the batter input says "not used: a passive coefficient acts on the front
  of the wall". Before, Coulomb's alpha, the log spiral's theta and M-O's
  beta took it, and the vertical face was not run. Test: Coulomb Kp and
  M-O KPE with a back-face batter agree with the vertical-face values; KAE
  still takes it.

**Tests after the fixes**: `test_recompute_slice3.py` has 85 tests (32 +
53). `pytest -q report_ingest/tests/test_recompute.py
report_ingest/tests/test_recompute_slice2.py
report_ingest/tests/test_recompute_slice3.py
webapp/tests/test_page_fingerprints.py webapp/tests/test_public_text.py`:
exit 0, 196 passed (143 before the fixes).

**The local measurement after the fixes**: the saved private records,
offline, master (4b67aaa, its code unchanged since 1be5341) vs this
branch, with the gamma' rule at its default and at `submerged`. The wall
rows are the same under both rules.

Retaining-wall coefficients:

| measure | master | branch before the fixes | branch after |
|---|---|---|---|
| readings / unique calcs / reports | 93 / 56 / 7 | 93 / 56 / 7 | 93 / 56 / 7 |
| rows | 93 | 268 | 268 |
| readings recomputed (unique calcs, reports) | 0 | 38 (26, 7) | 38 (26, 7) |
| claim rows (unique) | 0 | 107 (75) | 107 (75) |
| Ka / Kp / K0 / KAE / KPE | - | 40 / 29 / 16 / 16 / 6 | 40 / 29 / 16 / 16 / 6 |
| agrees (unique) | - | 86 (56) | 86 (56) |
| within options | - | 0 | 0 |
| differs (unique) | - | 21 (19) | 21 (19) |
| median difference | - | 0.09 % | 0.09 % |
| not_compared | 0 | 3 | 4 |
| not_recomputed | 93 | 158 | 157 |

One saved row moved, F3's: a seismic increment printed with U+2206, before
refused as a KAE without phi or a kh rule, now a `not_compared` increment.
F1, F2 and F4 move no saved row. Of the 21 rows that differ, 16 trace to
the reading (13 a phi read as 0, 3 a slope taken from a table row), 4 are
KPE values printed above the static Kp (the module takes the inertia away
from the wall, the worst case), and 1 is a K0 taken from a reference table.

Why the 157 rows are refused (rows / unique calcs; a row can have more
than one reason):

| reason | rows / unique |
|---|---|
| phi not printed | 45 / 30 (46 / 30 before) |
| phi printed more than once | 50 / 8 |
| phi in a unit that is not an angle | 13 / 1 |
| kh printed more than once | 39 / 7 |
| a PGA printed but no kh rule | 23 / 17 (24 / 18 before) |
| kh not printed, nor a PGA | 4 / 1 |
| no method and no wall friction | 10 / 3 |
| wall friction not printed (M-O or Coulomb) | 4 / 1 |
| wall friction printed more than once | 2 / 1 |
| the module refused | 4 / 1 |
| no coefficient among the results | 13 / 13 |

Bearing (the same before and after the fixes):

| measure | master default | branch default | master submerged | branch submerged |
|---|---|---|---|---|
| claim rows (unique) | 1 (1) | 1 (1) | 183 (173) | 183 (173) |
| agrees | 0 | 0 | 167 | 173 |
| within options | 0 | 0 | 14 | 0 |
| below | 1 | 1 | 1 | 5 |
| differs | 0 | 0 | 1 | 5 |
| median difference | 87.5 % | 87.5 % | 4.01 % | 0.05 % |

Under `submerged`, 182 rows move, all from 'myerhof' read as Meyerhof
(still read after F1: one letter off an eight-letter name): 8 go from
within options to agrees; 6 from within options to differs or below (the
phi reading error the options had hidden); 2 from agrees to differs
(Meyerhof's depth term beyond D/B = 1, GM-309). Settlement: no row moves.

## Release note

The report recompute (`recompute_report_calcs`) now recomputes the
earth-pressure coefficients a retaining-wall printout claims: static Ka,
Kp and K0 (Rankine, Coulomb, the log-spiral Kp, Jaky) through
`earth_pressure_coefficient`, and Mononobe-Okabe KAE and KPE through
`seismic_earth_pressure`, one row per coefficient (per case where the
printout prints several walls or strata), with kh as printed or from a
printed PGA only by a rule the printout states. A wall batter or kv whose
sense is not printed runs both ways, and a claim matching the other is
'within options'. A printout with no method and no wall friction, no phi,
or a PGA without a kh rule (or with the site factor Fpga in it) is refused
with the reason. Method names one letter off a known method, or two off a
long one (Mayerhof, Terzhagi, Coulumb), are read as that method, and the
row says so; ordinary words, names and places never are. Rows that change: `retaining_wall`
calculations that claim a coefficient were one `not_recomputed` row and
are now one row per coefficient; bearing calculations with a misspelled
method name now run the printed method instead of the module's default
with the others beside it; settlement printouts with a misspelled
Schmertmann are refused as Schmertmann with the note.
