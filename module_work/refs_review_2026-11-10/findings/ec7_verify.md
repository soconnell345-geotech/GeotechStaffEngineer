# ec7 independent check (branch `fix/review-ec7`, worktree `C:/dev/tmp/wt-fix-ec7`)

Filed by the clerk from checker `v-ec7`'s three messages (2026-11-10). Read
only; nothing in the worktree changed. Renders and scripts:
`C:/dev/tmp/refs_review_2026-11-10/v-ec7/`. Page offsets: EN documents PDF =
printed + 2; FEMA P-2082 PDF = printed + 38.

**Verdict.** Nearly everything as printed: every changed number, formula,
range, quote, footnote and page citation rendered from EN 1997-1, EN 1997-2,
Caltrans and FEMA P-2082 matches. One regression (F2) and one addition not as
printed (F1). Smaller: D.3 and F.3 answer silently above the printed range,
F.3 interpolates with no printed rule; CHANGELOG gaps.

## Not as printed / regression

| ID | Function | Call and output | Print | Suggested fix |
|---|---|---|---|---|
| F1 | FEMA `site_class_from_vs30`, `estimated_vs_site_classes` | `(2500, 'ft/s')` -> `{'vs/1.3': 'C', 'vs': 'BC', '1.3vs': 'B'}`; `(3900, 'ft/s')` -> `'1.3vs': 'A'` | Sec. 20.2.4 (fema_pdf125.png, printed p. 87): "Site Class B can only be assigned to a site on the basis of shear wave velocity measured on site"; Sec. 20.2.5 same for A. Test `tests/test_fema_p2082_tables.py::test_classes_for_an_estimated_velocity` pins 'B'. Docstring's Sec. 20.3 quote stops at "..." before "factors less than 1.3 can be used if approved by local building officials" (fema_pdf126.png, p. 88) | no B/A from an estimated velocity (cap at BC, or B/A with a note that measured velocity is needed); fix the test; complete the quote |
| F2 | Caltrans `heave_factor_of_safety` | c=100, Nc=7.5, H=10, B=10 (heave_main.py): gamma 64.299 Qu/Q 1.49960; 64.29 -> 1.49987; 64.3 -> 1.49957; all fs 1.5; main adequate False, branch True | FS = Qu/Q >= 1.5 (cal_pdf228.png, p. 10-19) | compare the unrounded ratio, round only the reported fs; add a test at FS 1.4996; CHANGELOG does not mention the change |

## Print does not say (answered silently)

- **F3** EN 1997-2 Table D.3 `table_d3_pile_base_resistance` (ec72_pdf117.png, p. 115; columns qc 10, 15, 20, 25 only, no ">= 25"): qc < 10 now refused; qc > 25 returns the 25 column with no note: (0.03, 25.001) and (0.03, 40) -> pb 2.25 MPa; (0.10, 40) -> 4.0 MPa. Docstring mentions the cap, result does not. Already on main: intermediate s/Db (0.025, 0.05) refused though the table's NOTE allows linear interpolation.
- **F4** EN 1997-2 Table F.3 `table_f3_phi_from_density_index` (ec72_pdf128.png, p. 126; rows ID 40, 60, 80, 100): ID < 40 refused; ID > 100 returns the 100 row with no note: (100.001, 'medium', 'uniform') and (120, ...) -> 43 deg. Interpolates between rows with no printed rule (docstring admits), while Table A.10 is refused between printed n for that reason (RULINGS row 6). Result says neither interpolated nor capped.

## Smaller points

- F5 Table P.1 middle row printed "0,95 << r <= 0,98" (typo; ec72_pdf165_p1zoom.png); code "0,95 < r <= 0,98", silently. A comment would cover it.
- F6 Caltrans `table_4_2_wall_friction` (not changed by the branch): 'sheet pile' -> steel sheet piles though "concrete sheet piling" is printed in the formed/precast group; 'steel' -> steel sheet piles though "steel on steel at sheet pile interlocks" (17 deg) is printed and not coded. EN 1997-1 `table_h_1_limiting_relative_rotation` keeps defaults ('acceptable_many_structures', 'sagging') picking one printed row (on main).
- F7a `aep_cohesive_max_ordinate(120, 20, 3, factor=0.3, ka=0.2)` raises (Ka < 0.25 refused even for Ns <= 4, where Ka is unused).
- F7b Table A.11 accepts fractional n (2.5, 4.99 take the lower column); Table A.9 refuses one.
- F7c EN 1997-2 Table D.1 very loose E' "<10" held as (0, 10), pinned by a new test; Table H.1 holds "<10" as (None, 10).
- F7d `heave_factor_of_safety` with Q <= 0: fs None, adequate True (judgment; main returned inf).
- F7e `shape='rectangle'` now refused (main treated it as rectangular); message names accepted shapes.
- F7f Eq. 4-5-2 caution confirmed: `max_allowable_slope_angle(30, c, 120, 20)` -> 30.0, 31.47, 33.41, 40.52, 61.14, 46.91, 30.0 deg for c 0, 50, 100, 200, 400, 800, 1600 psf (slope_sweep.py); unit statements match cal_pdf73.png (p. 4-22).

## Effects on callers (about 185 calls, main vs branch; collateral.py, before.txt, after.txt)

All explained by the diff. Signatures: defaults removed from the 17 `table_a_*` lookups and `design_approach_sets`; Annex D `shape` default 'rectangular' -> None (both functions); aep `factor` 0.3 -> None; `seismic_design_category` `s1` 0.0 -> None. A call relying on any of these now raises. `design_approach_sets('DA1')` now returns `{'combinations', 'description', 'note'}`; top-level keys such as 'action_set' gone (KeyError). No other library code and no app code (grep) calls the changed functions; chapter 11 text points to `design_approach_sets('DA3')`, unchanged.

CHANGELOG: every item matches the code; "before" numbers re-run on main are right (A.10 n=6 1.135; A.11 n=4 1.40, 1.533; Eq. C 82.35, 349.05; G.1 N10=200 1.10; D.3 qc 4 2.0; F.3 ID 10 34, ID 0.5 coarse well graded 41; A.4 no args 1.0; SDC 1.2/0.9 no S1 'D'; Table 3-1 very loose). Gaps: heave change (F2) missing; DA1 listed under "same numbers" though its keys changed; strip example (328.5 vs 277.1 kPa) does not say q = 20 kPa.

## Runs (one at a time, 4.1-4.3 GB free)

test_eurocode_7_1 157, test_eurocode_7_2 164, test_california_trenching_tables 102, test_fema_p2082_tables 88, test_references_agent 36: all passed, exit 0. `audit_chapter_text.py` eurocode_7_1 (21 ch), eurocode_7_2 (9), california_trenching (11), fema_p2082 (4): passed, exit 0. `git status --short` empty; 16 commits, 16 files, +1845/-341; CRLF both sides, no wholesale rewrite (only the C.2 docstring line re-indented); pyproject 1.6.0 unchanged. Gitignored `__pycache__` possibly refreshed by the first run; no tracked file changed.

## Tally (as printed / not / print does not say)

eurocode_7_1/equations.py 15/0/1 (shape spellings refused); eurocode_7_1/tables.py ~47/0/2 (A.11 fractional n; H.1 defaults); eurocode_7_1 text chapter09, 13, 15 (15 lines) all as printed; eurocode_7_2/equations.py 16/0/0 (Eq. 4.5 IDMT < 0,8; G.1 five eqs, 3 <= N10 <= 50; G.3 four eqs, four inclusive ranges); eurocode_7_2/tables.py ~75/1 (F5)/3 (F3, F4, F7c) (D.3 12 cells, NOTE, 0,75; F.3 24 cells; D.1, D.5, H.1 notes; G.1 ranges; P.1, P.2, Q.1, S.1, W.1 pp. 163, 164, 165, 168, 176); california_trenching/equations.py 14/1 (F2)/2 (F7a, F7d); california_trenching/tables.py 8/0/2 (F6); california_trenching text chapter10 (3 lines) as printed; fema_p2082/tables.py 6/1 (F1)/1 (shortened 20.3 quote). Tests: every traced number as printed except the F1 test. Boundaries: every printed boundary answered, every unprinted lower value refused, inclusive/exclusive as printed (G.1, G.3 inclusive; IDMT 0.8 refused; Ns 4 -> 8-3-2, Ns 6 -> 8-3-3; factors 0.2, 0.4 allowed; Ka 0.25 allowed; S1 0.75 -> E/F), except F2, F3, F4, F7b.

## Not checked

EN 1997-2 Table 3.1 quality-class page (only its error message changed); chapter JSON outside changed lines; full suite; the agent's behaviour with the new refusals (no live run).

## Loop

Sent back to `ec7`: F2, F1 (required), F3, F4 (note in the result), CHANGELOG gaps; F5 comment optional. Re-check by `v-ec7` of those items only.
