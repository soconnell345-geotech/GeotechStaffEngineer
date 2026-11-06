# REF-60: GEC-13 functions that cite the wrong figure or equation — CHECK

Branch `claude/ref-60-gec13-citations` (refs), 2026-11-06. Read against the
PDFs with PyMuPDF: `docs/GEC 13 Vol 1.pdf`, `docs/GEC 13 Vol 2.pdf`,
`docs/GEC 11 Vol 1.pdf`. No app branch: no function was renamed, and the
app's `gec13_adapter` builds its method list and briefs from these
docstrings, so nothing in the app needed changing (see "App" below).

## What the print says

| Function (name kept) | Was cited as | The print |
|---|---|---|
| `figure_4_3_suitability_number` | "Figure 4-3" (module comment), "Brown 1977" | GEC-13 Vol I **Eq. 4-3**, Ch. 4 §3.2.2 "Materials", p. 4-73: Suitability No. = 1.7 sqrt(3/(D50)^2 + 1/(D20)^2 + 1/(D10)^2), D in mm; it rates the vibro-compaction **backfill**. Rating in Table 4-7, p. 4-73. Fig. 4-3 is the SPT liquefaction chart. |
| `figure_5_2_area_replacement_ratio` | "GEC-13 Figure 5-2" | GEC-13 Vol I **Eq. 5-1**, §4.1.2.2, p. 5-31: alpha_sc = A_sc / A, A the unit cell (p. 5-30, Fig. 5-17: a hexagon for a triangular pattern). Fig. 5-2 is top-feed vibro-replacement. |
| `equation_8_1_groutability_ratio` | "GEC-13 Eq 8-1" | GEC-13 Vol II **Eq. 8-2**, §2.3.2.1, p. 8-29 (King and Bush 1963): N = (D15)SOIL/(D85)GROUT; "Grouting is consistently possible in soils if N > 24 or NC > 11, and is not possible when N < 11 or NC < 6." Eq. 8-1 is Darcy's law. |
| `equation_11_1_ltds` | "GEC-13 Eq 11-1" | **Not in GEC-13** (ten chapters; no Eq. 11-1, a leftover of the chapters 9-11 REF-25 found were not the manual's). GEC-13 Vol II Ch. 6 §4.6, p. 6-37 states it in words and refers to Chapter 3 of GEC 11. The printed equation is **GEC-11 (FHWA-NHI-10-024) Vol I Eq. 3-12, p. 3-28**: T_al = T_ult/RF = T_ult/(RF_ID·RF_CR·RF_D). No FS in it. |
| `equation_7_4_composite_modulus` | "GEC-13 Eq 7-4" | **Not printed in GEC-13.** Chapter 7 (Deep Mixing, Vol II pp. 7-1 to 7-41) numbers no equations at all. Step 5 (§2.3.2.5, p. 7-16) says only that the treated zone "is treated as a composite of the soil-cement elements and the untreated zone"; the chapter defers to the FHWA (2013) deep mixing manual, not in this library. No other held reference was found printing the area-weighted form. The worked example takes E50 = 150 qu (§3.3.3.2, p. 7-32). |

**The D10 term.** The backlog item says the function leaves out the
1/(D10)^2 term. It does not: `figure_4_3_suitability_number` has taken
`d10_mm` and computed the printed formula since the file was written
(`git log -S d10_mm` finds only the file's first commit). So **the answer
does not change**; what was wrong was the citation. The value is now pinned
to a hand-worked example.

## What changed (refs)

- `geotech_references/gec_13/figures.py`: section comments, docstrings and
  each result's `description` now cite Eq. 4-3 p. 4-73, Eq. 5-1 p. 5-31,
  Eq. 8-2 p. 8-29, GEC-11 Eq. 3-12 p. 3-28 (with GEC-13 p. 6-37), and say
  the composite modulus is not printed in GEC-13 (the library's own parallel
  model). Each says why it keeps its old name.
- **One value changes:** `equation_8_1_groutability_ratio` now calls
  N > 24 "Feasible", as p. 8-29 prints (was N > 25). N in (24, 25] moves
  from "Uncertain" to "Feasible". `groutability_ratio` itself is unchanged.
- `ltds`: documented that `fs` is not in Eq. 3-12 and its default 1.0 gives
  the printed equation; `rf_cbd` is GEC-11's RF_D. No value changes.
- Composite modulus docstring: the unprinted "E_col = 50-200 qu" replaced
  by the printed E50 = 150 qu (p. 7-32). No value changes.
- `agents/gec13_agent.py` docstring and `CLAUDE.md`'s GEC-13 line follow.
- The chapter text JSON (`text/chapter04.json`, `chapter05.json`) already
  cited Eq. 4-3 and Eq. 5-1 correctly; no catalog or manifest entry named
  these functions, so no JSON changed.

## Before / after on the hand-worked example

Made-up D50 = 1.0, D20 = 0.5, D10 = 0.25 mm, Eq. 4-3:
3/1.0² + 1/0.5² + 1/0.25² = 3 + 4 + 16 = 23; 1.7·√23 = 1.7 × 4.7958 =
8.153 → **8.2 (Excellent), before and after** (without the D10 term it
would be 1.7·√7 = 4.50). Groutability, made-up D15 = 0.25 mm, D85 = 0.01 mm:
N = 25.0, "Uncertain" before, "Feasible" after.

## Tests

- refs `tests/test_gec13*.py`: **206 passed** (197 before; 9 new: the Eq. 4-3
  hand pin, D10 changing the answer, each function's citation, GEC-11
  Eq. 3-12 by hand; the N = 25 boundary test replaced by N = 24 Uncertain
  and N = 25 Feasible).
- refs full suite `pytest -n 8 tests/`: **6,547 passed, 2 xfailed**, exit 0.
- app (refs branch installed editable): `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py`, `funhouse_agent/tests/test_reference_adapters.py`,
  `test_citations.py`, `test_reference_result_sizes.py`: **393 passed**;
  `test_gm179_unused_inputs.py` + `test_defaults_used.py`: 841 passed,
  23 skipped, **1 failed**, `profile_figure|plot_data` (logx/logy change
  nothing); unrelated to this branch (plot_data does not read GEC-13).

## App

Grepped `funhouse_agent/`, `webapp/`, adapters, dispatch tables and tests
for each function name and the old citations: only
`figure_5_5_settlement_improvement` sits in the alias table
(`_reference_common._CURATED_ALIASES`), and it is untouched. The page
fingerprints pass, so no re-record. No app branch.

## Noted, not fixed

- Table 4-7 prints 30-40 Poor and >50 Unsuitable with no row for 40-50;
  the function calls 30-50 "Poor" and >50 "Not suitable" (docstring says so).
  Fixed in review: see "Review and fixes" below.
- Eq. 8-3 (NC = D10_soil/D95_grout, same page) is not computed.
- `equation_7_4_composite_modulus` returns `stress_concentration_ratio`
  = E_col/E_soil, which is a modular ratio, not a stress ratio.
- Same file family, same REF-25 leftover: `tables.py`'s
  `table_9_2_nail_bond_strength`, `table_10_1_micropile_bond_stress` and
  `table_11_1_geosynthetic_reduction_factors` cite tables from the chapters
  9-11 GEC-13 does not have (two of them are in the app's alias table).
- `test_defaults_used`'s `plot_data` logx/logy failure in the app (above).

## Release note

GEC-13 lookups now cite the printed equations: the backfill suitability
number is Eq. 4-3, the area replacement ratio Eq. 5-1, the groutability
ratio Eq. 8-2, the long-term geosynthetic strength GEC-11 Eq. 3-12, and the
deep-mixing composite modulus is marked as not printed in GEC-13. One
answer moves: a groutability ratio between 24 and 25 now reads "Feasible",
as GEC-13 p. 8-29 prints (N > 24). The backfill suitability number itself
does not change; its rating now uses Table 4-7's own words (above 50 is
"Unsuitable", was "Not suitable"), and between 40 and 50, where Table 4-7
prints no band, it gives no rating and a note saying so (was "Poor").
No other value changes.

## Review and fixes (2026-11-06)

**What the review checked.** An independent read of the branch diff
(`912a99d...e87a779`) checked each changed citation against its printed
page, from the PyMuPDF text and the rendered page where the content is an
image: Eq. 4-3 and Table 4-7 (GEC-13 Vol I p. 4-73), Fig. 4-3 (p. 4-9),
Eq. 5-1 (p. 5-31), the unit cell (pp. 5-29, 5-30), Fig. 5-2 (p. 5-12),
Eq. 8-2 and its limits (Vol II p. 8-29), Eq. 8-1 (p. 8-17), every page of
chapter 7 for a numbered equation (none), pp. 7-16 and 7-32, GEC-13
p. 6-37 and GEC-11 Vol I Eq. 3-12 (p. 3-28). All matched. It also checked
the callers in refs and the app: no signature changed, nothing calls the
functions by name, and the page fingerprints did not move. Verdict:
merge-ready, four low findings.

**The four findings and what was done.**

1. **Stale reference label in `agents/gec13_agent.py`.** Every
   `figures.py` function was labelled "FHWA-NHI-16-027, Chapters 4-5
   Figures", wrong for the ch. 7, ch. 8 and GEC-11 functions. Fixed: a
   small map gives each function its own source as its docstring cites
   it. Examples: "FHWA-NHI-16-027 (GEC-13) Vol II, Ch. 8, Eq. 8-2,
   p. 8-29"; "FHWA-NHI-10-024 (GEC-11) Vol I, Eq. 3-12, p. 3-28; GEC-13
   Vol II Ch. 6, p. 6-37 refers to it"; "Not printed in FHWA-NHI-16-027
   (GEC-13); Vol II Ch. 7, p. 7-16 names a composite without an
   equation". The module fallback label is now "FHWA-NHI-16-027 (GEC-13)".
   A new test holds this.
2. **The suitability rating was not in Table 4-7's words, and 40-50 had
   no printed row.** Re-read Table 4-7 "Backfill Evaluation Criteria"
   from the rendered page (p. 4-73): 0 to 10 Excellent, 10 to 20 Good,
   20 to 30 Fair, 30 to 40 Poor, >50 Unsuitable, Source: Brown 1977.
   There is no row for 40 to 50. Fixed in
   `geotech_references/gec_13/figures.py`:
   - above 50 now reads "Unsuitable" (was "Not suitable");
   - 30-40 is "Poor";
   - for 40 < SN <= 50 the rating is `None`, and the result carries a
     `note`: "No rating: GEC-13 Table 4-7 (p. 4-73) prints no band for a
     suitability number between 40 and 50 ..." (was "Poor");
   - the result's description now says "rated by Table 4-7".
   The suitability number itself is unchanged. The tests: the silt case
   now expects "Unsuitable", and a new test works one made-up gradation
   per band by hand from Eq. 4-3 (16.3 Good, 20.4 Fair, 32.6 Poor,
   47.3 no rating with the note, 81.5 Unsuitable).
3. **This note's location.** Left at the refs root as asked; the lead
   moves it to the app's `module_work/ref60/CHECK.md` at merge. Its
   "Noted, not fixed" list goes to BACKLOG then. The `tables.py` item
   (three REF-25 leftovers citing the wrong tables) is the lead's new
   backlog item; `tables.py` was not touched here.
4. **Refs `CLAUDE.md` test counts.** Recounted by `pytest --collect-only
   -q` after the test changes: `tests/test_gec13*.py` 212 (was 197;
   the builder's 9 new tests plus the 6 here), `tests` 6,555 (was 6,540).
   Updated at all four places.

**The answers that move.**
- Groutability: a ratio N between 24 and 25 now reads "Feasible" (was
  "Uncertain"), because p. 8-29 prints "N > 24". This was the builder's
  change; the review confirmed it.
- Suitability: the NUMBER is unchanged (the D10 term was always there).
  Only its rating words move: above 50 "Unsuitable" (was "Not suitable"),
  and 40 < SN <= 50 gets no rating plus a note (was "Poor"), because
  Table 4-7 prints no band there.
- Nothing else changes. The new reference labels only show in the refs
  agent's describe result.

**Tests after the fixes** (logs in `C:/dev/tmp/ref60_fix/`, judged by exit
code; the import was checked to be the worktree's package):
- refs `test_gec13_figures.py`, `test_gec13_tables.py`,
  `test_cover_citations.py`, `test_figure_captions.py`,
  `test_references_agent.py`: exit 0, 264 passed, 2 xfailed.
- app `webapp/tests/test_page_fingerprints.py` +
  `funhouse_agent/tests/test_reference_adapters.py` with these refs on
  PYTHONPATH: exit 0, 319 passed. No fingerprint moved.
