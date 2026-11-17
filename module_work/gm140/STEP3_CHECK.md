# GM-140 step 3: the printed acceptability checks

Branch `claude/gm-140-step3-acceptability-checks`, from master decec33 (steps
1 and 2 merged; `STEP1_CHECK.md` and `STEP2_CHECK.md` are on master).
2026-11-13.

**The rule.** Once a method has solved a surface (step 2) and the surface is
one sliding mass (GM-122), every further test is a printed acceptability
check whose finding is reported with the surface: which check, which slice,
what value. A search keeps a flagged surface by default and says so; refusing
on a check is a stated option. Nothing is left out silently, and circles and
polylines meet the same checks.

## 0. Read this first

1. **The commit "item 5: remove the three noncircular screens" is
   separable.** The two code commits before it ("the first two commits"
   below) hold items 1 to 4, 6 and 7 with the noncircular screens kept and
   counted. The third removes three screens (item 5), as the brief's rule
   requires: each refused only surfaces the printed checks flag. A fourth
   commit holds this file and `VALIDATION.md`'s entry.
   The removal's consequence is large on random and weak-layer noncircular
   searches: the headline can become a zig-zag surface with a low factor of
   safety, flagged, with the lowest unflagged surface beside it. On ACADS
   1(a) (printed 1.00) a 200-trial random noncircular Spencer search reads
   0.633 flagged (1.096 unflagged) where it read 1.092. If the lead does not
   want that, revert the third commit: the first two stand alone and their
   tests pass (section 9). `DESIGN.md`'s paragraph on the screens goes back
   with it; `VALIDATION.md` B22's last table and this file's sections 4 and
   5 would then need their "all three" values taken out.
2. **Bishop below the Ordinary Method of Slices value happens on ordinary
   circles**, not only on odd ones: step 2's made-up fill slope (c' 40 kPa,
   phi' 22 deg), circle (8, 12) R 12, reads Bishop 1.729 and Ordinary Method
   of Slices 1.750, so the result now reports 1.750 with Bishop's own value
   beside it (OWN-102 as written). No benchmark pinned to print moved for it.
3. **One eval answer key changed** (`funhouse_agent/geotech_test_suite.json`,
   PPG-1): 1.223 to 1.101, the frictional term as printed; the earlier value
   is kept as its stated alternative. It is a module value, not a printed
   one: the lead should confirm.
4. **After the review (section 12).** Item 1's statement that every
   surface the screens refused carried a flag held for Spencer,
   Morgenstern-Price and GLE, not for Janbu's simplified method and the
   Ordinary Method of Slices, which the checks do not cover: the jagged
   screen stays for those two. Every exported search now reports the
   governing surface's flags, and the agent's search result says, when the
   headline is flagged, the lowest surface with no flag and the option in
   one sentence. Master's step 4 was merged first; section 12 has what it
   moved and the branch-against-master table.

## 1. What the print says

Read from the reference library's own text (the installed geotech-references
1.7.0, `dm7_1/text/chapter07.json`, which holds UFC 3-220-10 chapter 7):

- UFC 3-220-10 pp. 361 to 362: "It is important for the engineer to be able
  to examine the forces acting on each slice. In particular, the forces at
  the boundary between slices should be in compression, and the normal force
  at the base of the slice should be in compression. ... Ideally, the line of
  thrust should be located within the soil mass defined by the slices. If
  portions of the line of thrust plot outside of the free body being
  analyzed, it often indicates tensile forces between slices or tensile
  forces at the base of slices." And: "Critical failure surfaces can be
  missed as a result of non-convergence. The engineer should be able to
  identify when convergence issues are present."

Not in a repository (the owner's own copies): taken from the account's
quotations and the brief, and cited by document, page and equation. Nothing
was written from memory as if it were the print.

- Whitman and Bailey (1967, p. 487, Eq. 5): the user is warned if, for any
  slice, the numerator becomes negative or the denominator becomes negative
  or small, "for example, if cos(theta) + tan(phi) sin(theta) / F < 0.2 ...
  the user must examine the result in detail". A warning and an instruction,
  not a rejection.
- Duncan (1996, p. 580): calculate the Ordinary Method of Slices value for
  each circle when Bishop's method is used; where Bishop's F is below it,
  Bishop's solution has numerical problems and the Ordinary Method of Slices
  value is the better answer.
- EM 1110-2-1902 C-10a (pp. C-33 to C-34): calculated normal and interslice
  forces should be examined; tensile forces should not exist on the sides or
  bottom of slices; the line of thrust should be within the sliding mass.
- EM 1110-2-1902 C-10b (pp. C-34 to C-36): four remedies for a steep exit, in
  this order: flatten the toe of the surface (Eq. C-32: an exit at alpha =
  -(45 deg - phi'/2) for a horizontal ground surface and interslice force);
  change the side-force inclination; use the Ordinary Method of Slices; give
  the exit zone a passive strength as cohesion with phi = 0 (the only one
  usable with Bishop's method). The module names them and applies none.
- Bishop (1955) Eq. 13, p. 10: the frictional term is W(1 - B) = W - u b,
  with no lower limit; Eq. 9, p. 10: the Ordinary Method of Slices' N' =
  W cos(alpha) - u l.
- EM 1110-2-1902 C-2b (p. C-16): the Ordinary Method of Slices equals the
  rigorous methods when phi = 0 (used for the tolerance of the comparison).
- OWN-102 and its limit of 2026-11-13 (`CALLS.md`): Duncan, Wright and
  Brandon (2014, p. 93) print no Ordinary Method of Slices equation with
  known forces, so the comparison is made only on circles with no supports
  and no seismic coefficient.

**No printed source was found for**, and the result says so where it shows:
the exemption of phi' = 0 slices from the m_alpha check (the module's own
reasoning); setting a negative frictional term to zero (the option); the span
and slice-count screens; and the GLE solver's settings (D29, unchanged).

## 2. The departures, reproduced

`slope_stability/tests/test_acceptability.py` (21 tests, made-up geometries,
each docstring with the rule and its page). On master the file does not
import (`slope_stability.acceptability` does not exist), so master's
behaviour on each case was measured by script and is given here:

| Departure | Master | Branch |
|---|---|---|
| D20. A Bishop search of a made-up 10 m 1H:1V slope (c' 1 kPa, phi' 36 deg): circles with m_alpha below 0.2 | 5 left out, counted, not located | 5 kept and flagged, counted by family (`trial_flags`); with `low_m_alpha='refuse'` 5 left out, the same count; headline 0.9032 either way |
| D20. One of those circles, (42.928, 10.081) R 15.072, given to `analyze_slope` | a warning with the lowest m_alpha | the finding names slice 30 of 30 (m_alpha 0.19), and Bishop's 151.2 is set beside the Ordinary Method of Slices' 101.3 |
| D20. A steep-exit polyline on that slope, Spencer | a warning citing Slide2's error -112 | analysed (1.043), flagged, the warning names EM C-10b's four remedies in the printed order and says none is applied |
| Duncan (1996, p. 580). Circle (18, 12) R 10 on the made-up slope (c' 5, phi' 30): Bishop 6.1786, Ordinary Method of Slices 6.1799 | Bishop's value reported | the Ordinary Method of Slices value reported, Bishop's own beside it, flagged, said in `method_used` and the warnings |
| OWN-102's limit. A circle an anchor crosses; a circle under kh = 0.1 | no comparison, nothing said | the result says the comparison was not made and why |
| D21. Bishop on an artesian made-up grid (head 6 m above the toe's level): u b above W on 5 slices | 1.524, the term set to zero, no slice named | 1.374 as printed, the 5 slices named; 1.524 with `negative_friction_term='zero'`; each satisfies Eq. 13 with its own term (checked by hand in the test) |
| D21. The Ordinary Method of Slices on the same circle | 1.376 (N' set to zero) | 1.265 as printed; the difference is exactly the negative terms' friction over the driving sum |
| D22. Spencer circle (20, 10) R 10 in the c' 5 slope: tension at the crest | "base tension (N' < 0)" in the solver's own warnings, not in the result | the slices with a negative effective normal force and the slice sides with a tensile interslice force, named, at the crest end |
| D50. Spencer circle (18, 8) R 10: the line of thrust leaves the mass | the clipped line only | the line as solved (`thrust_line_unclipped`), the sides where it is outside, and the figure's legend says so |
| D19. The toe-circle family's below-base test | its own test, then the common one | the common one only |
| Item 4. A Spencer search of the c' 5 slope | nothing on the governing circle's forces | `critical_flags`, `trial_flags`, `lowest_unflagged`, one warning |

## 3. What changed, by the brief's items

New module `slope_stability/acceptability.py`: `surface_findings(geom, slip,
slices, solution, rule)` returns the findings of every check on a solved
surface; `flags()` the names of those not met. New logic is in new functions;
the diff in existing functions is the calls to them.

1. **m_alpha below 0.2 (OWN-102).**
   - `search._trial_checks` (called by `_compute_fos`, every search's scorer):
     the surface is scored and its flags counted (`_note_trial_flags`); it is
     left out only under `low_m_alpha='refuse'` (`search.LOW_M_ALPHA_ACTIONS`),
     where it is counted in `n_rejected_m_alpha` as before.
   - `solution.bishop_against_oms`, inside `solve_method` (the one solver):
     the Ordinary Method of Slices value for every Bishop circle with no
     supports and no seismic coefficient; where Bishop's is below it by more
     than Bishop's own tolerance, `fos` is the Ordinary Method of Slices
     value. `MethodSolution` gains `fos_bishop`, `fos_oms`, `oms_taken`,
     `oms_not_compared`. A result shows `FOS_bishop`, `FOS_fellenius` and
     `method_used`.
   - The phi' = 0 exemption stays (`m_alpha_rule='frictional'`), said in the
     finding's text and in `choices` to be the module's own reasoning;
     `'every_slice'` stays the option.
   - The warning names EM C-10b's remedies where a low slice is in the
     passive zone (its base inclined against the sliding). No remedy is
     built.
2. **The frictional term (D21).** `methods.NEGATIVE_TERM_RULES`
   (`as_printed`, `zero`), held in a context variable so every solve inside
   a call follows it; parameter `negative_friction_term` of `analyze_slope`
   and `search_critical_surface`. `bishop_fos` and `fellenius_fos` use the
   term as printed by default; the finding names each slice. The supports'
   normal-force terms in the Ordinary Method of Slices follow the same rule.
   `gle.py` is untouched: the `base_tension` finding names the slices whose
   effective normal the solver holds at zero.
3. **Compression and the line of thrust (D22, D50).**
   `acceptability.rigorous_findings`: `base_tension`, `side_tension`,
   `thrust_outside`, from the converged state, for Spencer,
   Morgenstern-Price and GLE. `SlopeStabilityResult.thrust_line_unclipped`;
   the figure's legend says when the solved line left the mass
   (`plotting._thrust_label`); the calc package's "Line of Thrust" step is
   made on the solved line.
4. **In a search.** `SearchResult.trial_flags` (by family and method:
   scored, flagged, each flag, the lowest unflagged), `critical_flags`,
   `critical.acceptability`, `lowest_unflagged`, a warning when the governing
   surface is flagged, and three `choices` entries
   (`search_supplements.critical_acceptability`, `acceptability_warning`,
   `acceptability_choices`).
5. **The noncircular screens.** Measured (section 5). Counted by screen
   (`screen_refusals`) in the first commit; three removed in the third
   commit.
6. **One below-base test.** `search_toe_circles`' own `below_base` now calls
   `search._dips_below_base`.
7. **The direct march.** Measured again and not applied (section 6).

Also: the calc package prints a "Printed Acceptability Checks" table, Bishop's
own value in its Bishop step with a "Factor of Safety Reported" step where the
Ordinary Method of Slices value was taken, and no longer says the frictional
term is clamped; the slope packages' adapter returns `flags`, `method_used`
and `FOS_bishop`; the agent's tools take `low_m_alpha` and
`negative_friction_term` (and `m_alpha_rule` on `analyze_slope`), with
descriptions, `applies_when`, the manual's rows and `NOT_REACHED` probe
entries (`funhouse_agent/tests/method_probes.py`: the probes have no flagged
surface and no artesian slice, and slope_stability's probes are frozen).

## 4. The validation set (account section 11): before, after, printed

"Before" is master decec33 in a separate worktree; "after" is the branch's
head (all three code commits). Measured by `module_work/gm140/step2_section11.py`
(unchanged), by the new `step3_section11.py` (rows step 2's script does not
have) and `step3_screens.py`, and by the pinned tests. Where the third commit
alone moves a value, the value after the first two commits is given too.

| Ref | Problem | Before | After | Printed | What moved it |
|---|---|---|---|---|---|
| B1 | F&K (1977) homogeneous, dry and ru | pins hold | same | F&K 1.928, 2.080, 2.041, 2.073, 2.076 | nothing: no slice has a negative term; Bishop is above the Ordinary Method of Slices |
| B2 | F&K composite surface | pins hold | same | F&K 1.288, 1.373, 1.370 | nothing |
| B3 | ACADS 1(a), search: Bishop / Spencer / GLE / Janbu | 0.9851 / 0.9840 / 0.9840 / 0.9830 | same | 1.00 (Slide2 0.987, 0.986, 0.990) | nothing; master left 2 circles out for m_alpha (Bishop), now kept and flagged, far above the minimum |
| B3 | ACADS 1(a), random noncircular, 200 trials, seed 42, Spencer (not a pinned benchmark; `test_search_modern`) | 1.0915 | first two commits: 1.0915, flagged (side tension, thrust outside), lowest unflagged 1.0959. All three: **0.6332**, flagged (base and side tension, thrust outside), lowest unflagged 1.0959 | 1.00 | item 5: the zig-zag was refused by the jagged screen (7 refused, all flagged). It reads 1.645 on 26 slices and 0.634 on 50 |
| B4 | Duncan (2000) reliability | pins hold | same | beta 2.32, about 1 % | nothing |
| B5 | Duncan, Wright & Brandon suite | all pass | same | book not in hand | nothing |
| B5 | Duncan Ex. 4 by Janbu, 30 slices (`test_m_alpha`) | 1.2033 (78 circles left out for m_alpha) | **0.8426**, flagged (m_alpha below 0.2), lowest unflagged 1.2154; 1.2033 with `low_m_alpha='refuse'` | none in hand for Janbu | item 1: the circle Janbu's solve reads low on is kept and flagged |
| B6 | ACADS 1(a), LE against fem2d SRM | pins hold | same | 1.00 | nothing |
| B7 | #95, #96, #98 rapid drawdown (V-037, V-038, V-041) | pins hold | same | 1.347; 1.443; 0.931, 1.039 | nothing moved in the pinned tests; the drawdown search now keeps a surface with m_alpha below 0.2 on a drawn-down slice |
| B8 | Pore-pressure grid (V-029): grid against line | pins hold | same | not pinned to print | nothing |
| B8 | V-029's made-up grid on the ACADS 5 slope, Bishop | 1.2471 | **1.1752**, flagged (negative frictional term); 1.2471 with `negative_friction_term='zero'` | none (the published 1.498 is for a flow net this grid does not reproduce) | item 2: u b exceeds W on the toe slices; the test no longer pins the module's value |
| B9/B16 | ACADS 1(b), crack (V-026) | pins hold | same | Slide2 1.596 / 1.592 / 1.592 | nothing |
| B10 | #54 micro-pile row (V-040) | pins hold | same | 1.193 | nothing (supports: no comparison with the Ordinary Method of Slices) |
| B11 | ACADS 4 (V-028), the declared family: Spencer / M-P / Janbu | 0.7532 / 0.7225 / 0.7284 | same | 0.760 / 0.720 / 0.734 | nothing |
| **B11** | **ACADS 4, weak-layer search, Spencer (seed 3, 700 trials, loads on)** | 0.8301 | **0.8301, flagged: tensile interslice forces and the line of thrust outside the mass; lowest unflagged 1.6666** (343 scored trials, 287 flagged) | 0.760 (Table 9.4) | no value moved; the result now says what the surface carries. The strict xfail at 3 % stays |
| B11 | ACADS 4, random noncircular, loads, seed 3, Spencer | 1.6412 | 1.6412, flagged (side tension), lowest unflagged 1.707 | 0.760 | nothing (step 7's generator) |
| B11 | ACADS 4, the calc-package report of the search (no loads) | 0.879 | same | none | nothing |
| B12 | V-054 steep phi = 0 slope, toe circle, Bishop | 2.3464 | same (Ordinary Method of Slices 2.3464 on it) | 2.16 (FLAC 2.18) | nothing |
| B12 | #86 without supports, Spencer search | 0.9420 | same | 0.942 | nothing |
| B12 | #86, five tiebacks: Bishop / Spencer / GLE | 1.6131 / 1.6065 / 1.6070 | same | 1.629 / 1.620 / 1.622 | nothing |
| B12/B16 | #85, one support, active: Bishop / Spencer / GLE | 1.3715 / 1.5505 / 1.5726 | 1.3713 / 1.5505 / **1.5732** | Bishop 1.531 (D&W 1.51), Spencer 1.884, GLE 1.575 | item 6: the toe-circle family's own below-base test is gone. GLE moves toward print (-0.15 % to -0.11 %); Bishop by 0.0002 on the reverse-curvature circle that is step 4's |
| B12/B16 | #85, one support, passive: Bishop / GLE | 1.2538 / 1.3594 | 1.2534 / 1.3594 | 1.324 (D&W 1.32) / 1.378 | item 6 (Bishop's toe circle, 0.0004) |
| B12/B16 | #85, reverse curvature discarded, Bishop, active: default rule / every slice | 1.5080 / 1.5109 (157 left out) | 1.5080 / **1.5080 flagged, lowest unflagged 1.5109**; 1.5109 with `low_m_alpha='refuse'` | 1.531 (D&W 1.51) | item 1 (OWN-102's own example: 1.508 against 1.511) |
| B12/B16 | #85, reverse curvature discarded, Bishop, passive: either rule | 1.3157 | same (462 trials flagged with every slice looked at, none governing) | 1.324 (D&W 1.32) | nothing |
| (brief) | #85 without its support, Spencer 6 x 6; Bishop 6 x 6 | 1.0468; 0.9095 | same | none | nothing |
| B12 | V-032 (Slide2 #61), entry/exit, Bishop | 1.3149 (3 left out) | 1.3149 (51 trials flagged and kept, 3 of them for m_alpha; on the governing circle Bishop 1.3149, Ordinary Method of Slices 1.2627) | 1.366 (Baker 1.35) | nothing |
| B13 | Duncan Ex. 1, D = 2 and 3, 4 H each side, Bishop | 0.7049 / 0.6774 | same (Ordinary Method of Slices equal on both) | Taylor 0.706 / 0.679 | nothing |
| B13 | Pilarcitos (drawdown) | pins hold | same | 0.823 | nothing |
| B14 | Weak layer over rock, default Bishop search | 0.8856 (116 left out) | 0.8856 (385 trials flagged, kept; Ordinary Method of Slices 0.8504 on the governing circle) | none | nothing |
| **B14** | **Weak layer over rock, DE directed (windows 25.5 to 28 / 29 to 40, seed 1), Spencer** | no surface (93 left out for m_alpha) | first two commits: 0.9605, flagged (m_alpha, base and side tension, thrust), none unflagged. All three: **0.6117**, the same four flags, none of the 19 scored trials unflagged | none (EV-25) | item 1, then item 5 (the moment-axis screen refused 27, the jagged screen 23, all flagged) |
| B14 | Weak layer over rock, random noncircular, default windows | 6.3656 | 6.3656, flagged (thrust outside), lowest unflagged 6.8826 | none | nothing |
| B15 | V-030 Duncan (2000) surface, FOSM, Monte Carlo | pins hold | same | Slide2 1.157; Duncan 1.17 | nothing |
| B15 | V-043 sand fill, DE, seeds 1 to 3 (strict xfail) | 1.234 | 1.2337 / 1.2331 / 1.2335, no flag | Slide2 1.188, Tandjiria 1.192 | nothing: the xfail holds; no DE trial was refused by a removed screen |
| B15 | V-043 sand fill, random 2000, seeds 1 and 2 | 1.2575 (the planar family governs) | same | Slide2 1.188 | nothing governing |
| B16 | V-043 circle; V-031 (Bishop / Spencer / GLE; Spencer search) | 1.2096; 1.4205 / 1.4247 / 1.4091; 1.4174 | same | 1.209; 1.417 / 1.422 | nothing |
| B16 | Vertical cut, Bishop search, default rule | 0.8416 | same, no flag | Taylor 0.851 | nothing |
| B16 | Vertical cut, Bishop search, every slice looked at | 0.9348 (157 left out) | **0.8416**, flagged, lowest unflagged 0.9353; 0.9348 with `low_m_alpha='refuse'` | Taylor 0.851 | item 1: toward print (+9.9 % to -1.1 %) |
| B17 | V-043 Spencer, refined; V-042 Bishop entry/exit | 1.1859; 1.3335 (2 left out) | same (V-042: 537 of 3,942 scored circles take the Ordinary Method of Slices value, none governing; on the governing circle Bishop 1.3335, Ordinary Method of Slices 1.2631) | 1.188 / 1.209; 1.334 | nothing |
| B17 | V-042 FOSM and Monte Carlo (Bishop; Spencer) | cov 0.1228, beta 2.312, pf 0.0165; 0.1232, 2.380, 0.0140 | same | beta 2.336 | nothing |
| B18 | Surface touching the base | pins hold | same | (consistency) | nothing |
| V-035/036 | Slide2 #79, #81 | pins hold | same | 1.44, 1.15 | nothing |
| V-039 | ky of the #104-structure demonstration, Spencer / Bishop / M-P / Janbu | 0.56491 / 0.53519 / 0.55899 / 0.46921 | same | (another section's 0.139) | nothing |
| V-044 | Slide2 #70 submerged: Bishop / Spencer / GLE | 1.5974 / 1.5962 / 1.5933 | same | 1.603 / 1.599 / 1.599 | nothing |
| V-045, V-048, V-052 | Slide2 #34; Lowe-Karafiath; Pilarcitos | pins hold | same | none; 0.823, 1.047, 1.043 | nothing |
| (GM-130) | Circles above the toe of #85's clay face, Janbu | 1.033 | 1.033, under every rule and option | none | nothing: the 1.138 the backlog item records predates GM-134 |
| (step 2) | Made-up fill slope (c' 40, phi' 22), circle (8, 12) R 12, Bishop | 1.7291 | **1.7498 reported** (the Ordinary Method of Slices value; Bishop's own 1.7291 beside it) | independent Bishop 1.7268 | item 1's comparison (Duncan 1996, p. 580) |
| (step 1) | Made-up sloping weak-layer section, weak-layer search, Spencer | 1.2313 | first two commits: 1.2313, flagged (side tension), lowest unflagged 1.5737. All three: **0.2564**, four flags, lowest unflagged 1.5737 | none | item 5: a saw-tooth the jagged screen refused; it has a Spencer state at 20 slices and none at 21 or 40 |
| (GM-131) | w11 cut over sandstone, DE (seed 3), Spencer | no surface (267 left out for m_alpha) | first two commits: 6.8761, no flag. All three: **0.4951**, four flags, none of the 15 scored trials unflagged | none | item 1, then item 5 (59 jagged, 10 moment-axis, 1 slicing refusals, all flagged) |

**Reading it.** Against print, three pinned benchmark values move, each by
item 6 and each by less than 0.05 %: #85's Bishop toe circles (active and
passive) and #85's GLE toe circle, the last toward the printed 1.575. The
vertical cut with every slice looked at moves toward Taylor. No test pinned
to a printed answer failed and no tolerance was widened. The brief expected
movement on every search that left surfaces out for m_alpha; on most of them
(ACADS 1(a), V-032, V-042, B14's default search, #85 with the default rule)
the surfaces formerly left out are far from the minimum, so the headline is
the same and only the counts move from "left out" to "flagged".

Values that are not pinned to print and move a long way are all noncircular
searches or made-up cases, and all are flagged with the lowest unflagged
surface beside them: Duncan's Example 4 by Janbu (item 1), and, by the third
commit, the ACADS 1(a) random search, the weak layer over rock by DE, the
sloping weak-layer section and w11.

**Tests that failed on the branch before any test was edited** (the first
two commits' code; 21 of the module's and validation's tests, 3 outside):

| Test | It asserted | Now |
|---|---|---|
| `test_m_alpha` (7 tests) | surfaces with m_alpha below 0.2 left out and counted; the choice named "left out"; the Slide2 wording of the warning | the default keeps and flags; each test also runs `low_m_alpha='refuse'` and finds the earlier value and count (vertical cut 0.935, Duncan Ex. 4 1.203, the steep-exit polyline left out, the DE counts) |
| `test_search_reinforcement` (3) | `(w,) = r.warnings` on circles with a support; "5.89 m deep" for #85's toe circle; the every-slice discard pin 1.511 | one more warning (the comparison not made); the crack 5.85 m deep (item 6); 1.508 flagged with 1.511 as lowest unflagged, and 1.511 under the option |
| `test_ground_crossings`, `test_search_ranges`, `test_choices` (4) | the choice's name; a missing argument in the drawdown search (a defect of this branch, fixed) | the new choice's name |
| `test_calc_steps_modern::test_absent_without_compare` | no comparison section for a Bishop analysis | Bishop is set beside the Ordinary Method of Slices value; the test uses a method with one value, and a new test holds the Bishop case |
| `test_own_solution_or_no_number` (3 parameters of 1 test) | Bishop 1.7268 on step 2's circle as `FOS` | `FOS_bishop` within 1 % of 1.7268; `FOS` the Ordinary Method of Slices value |
| `test_search_robustness::test_reject_stats_counts_geometry` | the tally's exact keys | one more key (the screen's name) |
| `test_slope_stability::TestBishopArtesianClamp` | Bishop's term floored at zero | under the option; a new test holds the printed term |
| `test_published_v029_slope::...flownet_grid_demonstration` | the module's 1.247 on a made-up grid | 1.247 under the option; the default asserted lower and flagged, not pinned |
| `funhouse_agent/tests/test_calc_package_adapter::test_bishop_solves_the_same_circle` | 1.7268 as the package's `FOS` | `FOS_bishop`; `FOS` the higher value, `method_used` and `flags` |
| `funhouse_agent/tests/test_eval_suite_v54` (PPG-1) | the key 1.223 | the key is 1.101 with 1.223 as its alternative (section 0) |

After the third commit, five more (`test_search_robustness` 3,
`test_search_modern::test_de_refines_random_search`,
`test_one_material_per_point::test_weak_layer_search_runs_in_a_sloping_layer`
2 parameters, `test_choices` 1): each asserted a removed screen's refusal or
a headline that is now a flagged surface. They are rewritten for the rule
(the lowest unflagged surface is what the weak-layer and DE tests look at);
the DE test now also holds ACADS 1(a)'s printed 1.00 within 5 %.

## 5. The noncircular screens, measured (item 5)

`module_work/gm140/step3_screens.py`, on the first two commits (screens in
place), Spencer's method. "Flagged" counts the refused surfaces the printed
checks flag. The two screens applied before any solve were solved by the
script to see what the printed checks say of them.

| Case | Span: refused / of those solved / flagged / not flagged (lowest FOS) | Jagged below 1.5: refused / flagged | One moment axis | One slicing |
|---|---|---|---|---|
| V-028 ACADS 4, weak-layer search, loads, seed 3 | 51 / 2 / 2 / 0 | 1 / 1 | 0 | 0 |
| V-028, weak-layer search, no loads, 40 slices | 51 / 2 / 2 / 0 | 0 | 1 / 1 | 0 |
| V-028, random noncircular, loads, seed 3 | 36 / 0 | 0 | 0 | 0 |
| B14 weak layer over rock, DE directed, seed 1 | 150 / 1 / 1 / 0 | 23 / 23 | 27 / 27 | 0 |
| B14, random noncircular, default windows, seed 1 | 67 / 0 | 0 | 0 | 0 |
| w11 cut over sandstone, DE, seed 3 | 406 / 5 / 5 / 0 | 59 / 59 | 10 / 10 | 1 / 1 |
| V-043 sand fill, DE, seed 1 | 16 / 1 / 0 / 1 (4.382) | 0 | 0 | 0 |
| V-043, DE, seeds 2 and 3 | 22 / 0; 18 / 0 | 0 | 0 | 0 |
| V-043, random 2000, seed 1 | 235 / 18 / 5 / 13 (1.922) | 4 / 4 | 0 | 0 |
| V-043, random 2000, seed 2 | 237 / 19 / 8 / 11 (2.525) | 1 / 1 | 0 | 0 |
| **All** | 48 solved, **25 not flagged** | **88 / 88** | **38 / 38** | **1 / 1** |

The slice-count screen refused no trial in any case.

**Decision, by the brief's rule.**

- **Jagged below 1.5, one moment axis, one slicing: removed** (third commit)
  for the method's own solves. Every surface each refused carried a flag of
  the printed checks.
- **Span: stays**, counted in `screen_refusals` and named in `choices` as the
  module's own, not found in print. What it catches: short surfaces confined
  to part of the entry and exit windows. Of those with a Spencer solution, 25
  of 48 meet every printed check; none of the 25 was below the search's
  minimum (their lowest, 1.922, against V-043's 1.2575).
- **Slice count: stays** as it was; the measurement says nothing of it
  (call 11).
- No screen was added.

**After the removal** (the same script on the branch's head): ACADS 4's two
searches, V-043's five and B14's random search return the same values and
flags. B14's directed DE goes from 0.9605 to 0.6117 and w11's DE from 6.8761
(unflagged) to 0.4951, each on a surface carrying four flags with no
unflagged trial among those scored (19 and 15); both differential evolutions
stop after 465 surfaces where they ran 3,255 and 2,325, because the
population settles on the flagged surfaces.

## 6. The direct march (item 7): decided, not applied

Step 2's review proposed solving a fixed-lambda state by the direct march on
every run where the lagged iteration cannot take a step. Step 2 measured it
and kept it out (STEP2_CHECK.md section 11). The lead's view for this step:
a state that stands only on the m_alpha clamp or on tension on a slice base
is not the method's own solution; with such a state refused, the march could
go in.

Built again here as a temporary switch (reverted; `gle.py` is byte for byte
master's), in two forms:

| Form | Reviewer's phi = 0 circle (independent Spencer 2.9435) | w11 polyline (GM-131: no Spencer solution) | Tests that fail |
|---|---|---|---|
| The march wherever the lagged iteration's denominator vanishes | 2.9458 | 1.559 to 1.562 at 29, 30, 31, 40, 41 and 60 slices, with m_alpha below 0.2, base and side tension and the line of thrust outside the mass | as step 2 found |
| The same, and a marched state in which the m_alpha clamp or the no-friction-in-tension branch is active on any slice is no state | 2.9458 | no solution at any of the six slice counts | 4, below |

With the second form the strict xfail's case passes and the GM-131 surfaces
no longer solve, but four other tests fail:

- `test_own_solution_or_no_number::test_the_robust_lambda_search_runs_without_supports`:
  a GLE solve that converges today (its bracket started from a lambda other
  than zero) has "no equilibrium state at any lambda in range".
- `test_newmark::test_yield_acceleration_self_consistent`: Spencer has no
  solution at a trial kh of 1.0, so an ordinary slope has no yield
  coefficient (static FOS 5.57).
- `test_search_robustness::test_a_refused_lowest_surface_is_reported` and
  `test_m_alpha::test_w11_de_search_runs_its_whole_budget` (465 surfaces
  instead of 2,325): two searches change.

The likely cause of the first two: a marched state at one lambda belongs to
another branch of solutions than the lagged states either side of it, so the
scan's sign changes no longer bracket the crossing it found before. The rule
"refuse a marched state resting on a clamp" removes the false solutions but
not that. So the decision is as step 2's: **not applied.** A rule applied to
marched states and not to lagged ones would also be two rules for one solver;
applied to both, it would take away every Spencer solution with tension at
the crest, which item 3 asks to report, not refuse. The strict xfail
`test_the_march_is_tried_where_the_lagged_iteration_cannot_step` stays and
its reason is rewritten with this.

## 7. Calls made (for the owner to revisit)

The lead's two calls, as the brief asks them listed:

1. **The frictional term as printed is the default (item 2).** Bishop's
   W - u b and the Ordinary Method of Slices' W cos(alpha) - u l are not set
   to zero where negative; the earlier clamp is the option
   `negative_friction_term='zero'`. The GLE solver's clamps are not changed;
   the slices they act on are reported.
2. **The headline stays the lowest solved surface (item 4)**, flagged or not,
   with the lowest unflagged surface beside it. With item 5's removal this is
   what puts a flagged zig-zag in the headline of some noncircular searches
   (section 0).

Calls made here:

3. **"Bishop below" means below by more than Bishop's own tolerance**
   (`tol`, 1e-4 by default). On a phi = 0 circle the two methods are the same
   sum (EM C-2b, p. C-16) and differ in the last digits; without a tolerance
   the flag would be raised by rounding. No printed tolerance.
4. **The Ordinary Method of Slices value compared is the one under the rule
   in force** (as printed by default). Both methods then follow Bishop
   (1955) as printed.
5. **The comparison is made in the one solver**, so it also holds in FOSM and
   Monte Carlo on a Bishop circle (a realization where Bishop's value is the
   lower takes the Ordinary Method of Slices value). The yield acceleration
   never compares: it acts under a seismic coefficient.
6. **m_alpha is judged at Bishop's own F** where the Ordinary Method of
   Slices value was taken (the Ordinary Method of Slices has no m_alpha).
7. **EM C-10b's remedies are named only where a low slice is in the passive
   zone** (its base inclined against the sliding). A base nearly vertical
   where the surface enters is located, no remedy named: C-10b is about the
   steep exit.
8. **A tensile interslice force is one below 0.001 of the largest on the
   surface**, the GLE engine's own threshold for "no force" on the thrust
   line; the line of thrust is checked only on sides that carry a force. A
   negative effective normal is one below 1e-9 of the largest normal
   (rounding). No printed tolerance for either.
9. **`lowest_unflagged` is among the trials the search scored by the same
   method as the governing surface**, not a second search. The result says
   so.
10. **A comparison not made is a finding and a warning**, not a flag: the
    surface meets every check that was made.
11. **Slice count is kept.** It refused no trial in any measured case, so the
    measurement says nothing of it; removing it would change nothing measured
    and it guards against three- and four-slice slivers.
12. **The jagged gate stays for a caller's own per-surface solve** (the
    rapid-drawdown search): the printed checks are not made on those solves
    here (step 9), so the measurement does not cover them. Since the fix
    round it also stays for Janbu's simplified method and the Ordinary
    Method of Slices (call (d) below).
13. **The rapid-drawdown search follows the default** (a surface with
    m_alpha below 0.2 on a drawn-down slice is kept and counted as flagged)
    and takes `low_m_alpha`. Nothing else in `rapid_drawdown.py` changed.
14. **The eval key PPG-1** takes the as-printed value, the earlier one as
    its alternative (section 0).
15. **The parameter names**: `low_m_alpha` ('flag', 'refuse') and
    `negative_friction_term` ('as_printed', 'zero').

Calls made, as the orchestrator ruled them after the review (for the
owner's desk at the merge), each with its other form:

- **(a) The headline.** The headline stays the lowest solved surface,
  flagged, with the lowest surface that carries no flag beside it (the
  spec's item 4), and a flagged headline never comes alone: the agent's tool
  result gives the headline, its flags, the lowest surface with no flag (or
  "none found") and the option that leaves flagged surfaces out, as fields
  and in one sentence (`summary`). Neither number alone is safe: ACADS 1(a)
  by Spencer reads 0.633 flagged and 1.09 with none, printed 1.00; ACADS 4's
  weak layer reads 0.830 flagged and 1.667 with none, printed 0.760. *Other
  form:* the headline is the lowest surface with no flag.
- **(b) Bishop's numerator as printed**, not held at zero (Bishop 1955
  Eq. 13, p. 10): V-029's made-up grid 1.247 to 1.175, the agent suite's
  PPG-1 1.223 to 1.101. *Other form:* `negative_friction_term='zero'` as the
  default.
- **(c) The Ordinary Method of Slices value taken where Bishop's is lower**,
  as Duncan (1996, p. 580) prints. By construction this raises the reported
  value wherever it applies: the hand circle of
  `test_own_solution_or_no_number` 1.7268 to 1.750, +1.3 %. *Other form:*
  report Bishop's value with a flag and the Ordinary Method of Slices value
  beside it.
- **(d) The jagged screen kept for Janbu's simplified method and the
  Ordinary Method of Slices**, counted and named in `choices` as the
  module's own, not found in print: no check of EM 1110-2-1902 C-10a covers
  those methods, so the zig-zags it refuses there would be admitted with no
  flag (ACADS 1(a), 200 random trials: 29 Janbu and 42 Ordinary Method of
  Slices surfaces on step 3's head, as the review counted them). *Other form:* remove it for them as well and admit those
  surfaces unflagged.

## 8. Where the rule now holds

- `acceptability.surface_findings` (new): every method, both kinds of
  surface.
- `solution.solve_method`: Bishop against the Ordinary Method of Slices, for
  every caller of the one solver.
- `methods.bishop_fos`, `fellenius_fos`: the frictional term as printed.
- `analysis.analyze_slope`: `acceptability`, `flags`, `FOS_bishop`,
  `FOS_fellenius`, `method_used`, `thrust_line_unclipped`, warnings;
  `compare_methods_table` through it (its Bishop row is Bishop's own value;
  its warnings carry the findings).
- `search._compute_fos`, so every family and search function: `grid_search`,
  `optimize_radius`, the refinement walk, `search_entry_exit`,
  `search_noncircular`, `search_de`, `search_pso`,
  `search_weak_layer_biased`; `search_supplements.search_toe_circles` (toe
  circles and circles above the toe, with the common below-base test) and
  `check_infinite_slope` (planes, scored by Janbu: m_alpha only).
- `analysis.search_critical_surface`: `trial_flags`, `critical_flags`,
  `lowest_unflagged`, `screen_refusals`, the warning and `choices`.
- `rapid_drawdown.search_rapid_drawdown`: the m_alpha default and option
  (the single-surface `rapid_drawdown_fos` already analysed and warned).
- `newmark.yield_acceleration`: through `solve_method` (no comparison under
  kh; the frictional term as printed at every trial kh).
- `probabilistic.fosm_fos`, `monte_carlo_fos` and
  `reliability.wrappers.slope_reliability`: through `solve_method`.
- `geo_project.builders`: its searches and analyses call the functions
  above; `to_dict()` carries the new fields. Not edited.
- `funhouse_agent/adapters/slope_stability.py` (parameters, descriptions,
  `applies_when`) and `adapters/calc_package.py` (`flags`, `method_used`,
  `FOS_bishop`); `slope_stability/calc_steps.py`, `results.py`,
  `plotting.py`.

## 9. Tests

Judged by pytest's exit code. Python 3.12, the venv of the common brief
(planlens 0.15.1 and geotech-references 1.7.0 from PyPI).

| Run | Result |
|---|---|
| Master, `slope_stability/tests` and the 21 `validation_examples/test_published_*` files that import the module | 810 passed, 17 skipped, 3 xfailed (exit 0) |
| First two commits (screens kept), the same files with the new test file | 833 passed, 17 skipped, 3 xfailed (exit 0) |
| Branch head (screens removed), the same files | 833 passed, 17 skipped, 3 xfailed (exit 0) |
| `slope_stability/tests/test_acceptability.py` alone | 21 passed |
| Branch head: the 46 files under `calc_package/tests`, `funhouse_agent/tests`, `funhouse_agent/deep/tests`, `reliability/tests`, `fem2d/tests`, `geotech_common/tests`, `report_ingest/tests`, `dxf_import/tests` and `geo_project/tests` that name `slope_stability` (`test_defaults_used.py` whole), with `webapp/tests/test_page_fingerprints.py` and `test_public_text.py` | 2,992 passed, 26 skipped, 2 failed (exit 1) |
| The 2 failures | `test_defaults_used.py::...[profile_figure|plot_data]` and `calc_package/tests/test_calc_package.py::TestDm7FigureHelper::test_figure_436_with_query`: both fail the same way on master in this environment; neither is a slope test |
| First two commits, the same 46 files | 2,990 passed, 26 skipped, 4 failed: the 2 above and the 2 this step caused (the calc-package adapter's Bishop circle and the PPG-1 key), fixed before the first commit and re-run with the adapter tests: 207 passed |
| `funhouse_agent/tests/test_defaults_used.py -k slope` | 41 passed, 14 skipped (as on master: the slope probes are frozen) |
| `webapp/tests/test_public_text.py`, `test_backlog.py`, `test_docs_currency.py`, with the new files staged | 22 passed |
| `python module_work/doc10/check_manual.py` (names, parameters and defaults in the manual; no rebuild) | OK, exit 0 |

The three xfails are strict and unchanged in kind: ACADS 4's weak-layer
search against print, V-043's DE search against print, and step 2's march
case (reason rewritten, section 6). The 17 skips are the suite's own.

`python -m webapp.tests.page_builds record` was run and left
`page_fingerprints.json` unchanged, so there is no fingerprint commit.

User manual: the `m_alpha_rule` row of `search_critical_surface` rewritten
and rows added for `low_m_alpha` and `negative_friction_term` (both tables
that list the search's parameters), `m_alpha_rule` and
`negative_friction_term` for `analyze_slope`, `low_m_alpha` for
`search_rapid_drawdown`. No rebuild.

Consumers checked by grep over the whole repository for the keys and
functions added, renamed or removed (`n_rejected_m_alpha`, the choice's
name, `refused_lowest`, `confirm_below`, `_solution_fails`,
`m_alpha_statement`, `FOS_bishop`, `thrust_line`): `funhouse_agent/adapters/
slope_stability.py` and `calc_package.py` (updated), `docs/gallery/
build_gallery.py` (reads `n_rejected_jagged`, which stays), `docs/
user_manual/worked_examples.json` (saved outputs; section 11).

## 10. Release note

slope_stability: the printed acceptability checks are now reported with
every surface and no longer decide which surfaces count. A result lists
`acceptability` and `flags`: m_alpha below 0.2 on a slice (Whitman and Bailey
1967, p. 487), a negative frictional term, and, for Spencer,
Morgenstern-Price and GLE, tension on slice bases and sides and the line of
thrust outside the sliding mass (EM 1110-2-1902 C-10a). A search keeps a
flagged surface: it no longer leaves out surfaces with m_alpha below 0.2
(`low_m_alpha='refuse'` restores that), and it reports the flags of its
governing surface and the lowest scored surface without a flag. Answers that
move: (1) searches that used to leave such surfaces out can report a lower,
flagged minimum (Duncan's Example 4 by Janbu: 0.843 flagged against 1.203,
with 1.215 as the lowest unflagged); (2) with Bishop's method on a circle
with no supports and no seismic coefficient, where Bishop's value is below
the Ordinary Method of Slices value the latter is reported (Duncan 1996,
p. 580), with Bishop's own beside it; (3) Bishop's method and the Ordinary
Method of Slices use W - u b and W cos(alpha) - u l as printed, so where the
pore-water force on a slice base exceeds its weight the factor of safety is
lower than before (`negative_friction_term='zero'` restores the earlier
value) and the slices are named; (4) noncircular searches no longer refuse a
low factor of safety on a jagged surface or a solution that held about one
moment axis or one slicing only: such a surface is scored and flagged, and
can be the headline (ACADS 1(a), 200 random trials: 0.633 flagged, 1.096
unflagged, 1.092 before; printed 1.00); (5) the toe-circle family uses the
common below-base test, which moves Slide2 #85's toe circle by 0.0002 to
0.0004. No value pinned to a printed answer moved outside its tolerance.

## 11. Stopped short of, and noted, not fixed

**Stopped short of.**

- **The remedies of EM C-10b are not built** (the brief: named, not applied).
- **Rapid drawdown beyond the m_alpha default.** `rapid_drawdown_fos` does
  not carry `acceptability` for its stages, and the drawdown search's trials
  are not put through the base, side and line-of-thrust checks (its
  per-surface solve is the caller's). Step 9 rebuilds that file.
- **A general "refuse on any check" option** was not added: only m_alpha has
  the refusal option the brief names. `lowest_unflagged` gives the lowest
  unflagged surface among the trials scored; a search that leaves out every
  flagged surface would be a new option.
- **The user manual's worked-example outputs** (`docs/user_manual/
  worked_examples.json`, printed in `manual.html`) still show a search's
  `choices` as they were ("slip surfaces left out for m_alpha"). Only the
  parameter rows were hand-edited, as the common brief asks; the rebuild at
  release refreshes the outputs.
- **`geo_project/`, `dxf_import/`, BACKLOG, HANDOFF, CLAUDE.md**: not edited.

**Noted, not fixed.**

- D24, D39 (step 4): the flagged tension at the crest of Spencer circles in
  soils with cohesion is what EM C-10c's tension crack is for; #85's toe
  circles are reverse-curvature circles (1.3713 active, 1.2534 passive
  against the printed 1.531 and 1.324).
- D12 (step 7): ACADS 4's weak-layer search reads 0.830 with side tension
  and the line of thrust outside the mass, against the printed 0.760; its
  lowest unflagged trial is 1.667. The strict xfail at 3 % stays. V-043's DE
  search (1.234 against Slide2's 1.188) is unchanged and unflagged; its
  strict xfail stays.
- D42 (step 6): the planar family is scored by Janbu and the toe families
  scanned by Bishop whatever method is named; `lowest_unflagged` therefore
  looks only at trials scored by the governing surface's own method.
- D29: the GLE's settings stand. A surface whose only Spencer state exists
  at one slice count is now scored and flagged (the sloping weak-layer
  section of `test_one_material_per_point`: 0.256 at 20 slices, no solution
  at 21 or 40).
- The differential-evolution search seeds from the random search's lowest
  surface. Where that is now a flagged zig-zag the evolution starts from it;
  its own objective penalises bumps. On the measured cases its results are
  in section 4.
- `SearchResult.refused_lowest` is kept as a field and is always None after
  the third commit; the warning that read it is dead code. Removing the
  field would break `to_dict()` consumers' expectations for no gain here.
- `test_defaults_used.py::...[profile_figure|plot_data]` and
  `calc_package/tests/test_calc_package.py::TestDm7FigureHelper::
  test_figure_436_with_query` fail on master too in this environment
  (geotech-references 1.7.0 from PyPI); neither is a slope test.
- GM-130's Janbu value for the circles above the toe of #85's clay face
  reads 1.033 on master and on the branch, under every rule and option (the
  1.138 the backlog item records was before GM-134).

## 12. Fix round after review

Branch `fix/gm140-step3-review`, cut from 677fce4. The review
(`STEP3_REVIEW.md`, outside the repository) passed items 1 to 4 and 6 to 9
and found three defects; the orchestrator ruled on them and on five more
points. Master (GM-140 step 4 and unrelated work) was merged first.

**The merge (bf2c600, 2ac03e8).** Both rules kept: step 4's discards
(reverse-curvature circles with no crack asked for, surfaces the crack's
bottom does not reach) stay refused and counted in `crack_refused`; step 3's
checks flag and keep. Both sets of counts are in the same result. Step 3's
`VALIDATION.md` entry became B22, after step 4's B21. The manual was rebuilt
with `check_manual.py --rebuild --write` (exit 0). What the merge moved:

- step 4's crack solve passed `confirm_below`, the parameter step 3 removed
  with the moment-axis screen: a NameError on any search with a Rankine
  crack (`test_cracks_as_printed`), fixed;
- the vertical cut by every slice: 0.842 flagged (step 3 alone) to 0.935
  with no flag. The 0.842 was a toe circle closed by a crack nobody asked
  for, which step 4 discards; the circles flagged for m_alpha are kept and
  counted, none below 0.935;
- #85's default Bishop search: the 1.371 toe circle (step 3 alone, with the
  "was not compared" warning) is discarded by step 4; the search reads what
  'discard' reads, and the warning that Bishop's value was not compared with
  the Ordinary Method of Slices value (a support crosses) stays;
- two step 3 test circles are reverse-curvature circles, so each is now
  given its crack: the phi = 0 exemption's circle (8 m crack) and the march
  circle (3.0 m crack, ruling 4).

**Per defect and ruling** (the fix commit; tests named).

1. **Defect 1, the jagged screen.** It stays for every method the module has
   no C-10a checks for, Janbu's simplified method and the Ordinary Method of
   Slices (`search._jagged_screen_applies`, `acceptability.C10A_METHODS`),
   counted in `screen_refusals` and named in `choices` as the module's own,
   not found in print; removed only for Spencer, Morgenstern-Price and GLE
   (Bishop's noncircular surfaces are scored by Spencer). The four
   statements that every refused surface carried a flag now say which
   methods (`search.REMOVED_SCREENS`, the `choices` text in
   `search_supplements`, `DESIGN.md`, the adapter's `screen_refusals`), and
   `results.py` and `analysis.py`'s docstrings with them. Re-measured on
   ACADS 1(a), the reviewer's random search (200 trials, seed 42): no
   surface jagged below 1.5 is admitted by Janbu or the Ordinary Method of
   Slices, and every refusal is counted under the screen
   (`test_search_robustness::test_jagged_screen_stays_for_methods_without_c10a_checks`,
   both methods).
2. **Defect 2, flags from every search function.** One place,
   `search_supplements.report_acceptability` (the governing surface's
   findings and flags, the lowest scored surface with no flag, the warning,
   a comparison with the Ordinary Method of Slices not made and why).
   `search_critical_surface` calls it; `search._reports_acceptability` calls
   it on `grid_search`, `search_noncircular`, `search_pso`,
   `search_weak_layer_biased`, `search_entry_exit` and `search_de` when
   called on their own (a search inside another leaves it to the outer
   one). `search_rapid_drawdown` scores with its own staged solve, on which
   the checks are not made; the `critical_flags` docstring says so. Tests:
   `test_acceptability::test_every_exported_search_reports_its_governing_surface_s_flags`
   and `::test_search_noncircular_called_directly_reports_the_flags`
   (ACADS 1(a) by Spencer called directly: 0.633 with base tension, side
   tension and the line of thrust outside, the lowest with none beside it).
   One existing test moved with it: `test_m_alpha::test_de_reports_the_seeding_search_when_it_has_no_surface`
   (the direct DE call now also carries the acceptability warning).
3. **Defect 3, `test_choices.py`.** Back to 100 trials:
   `test_a_short_search_reports_its_flagged_surface` asserts what the code
   says, 0.462 flagged low_m_alpha, base_tension, side_tension and
   thrust_outside, none of the 2 scored surfaces without a flag. The "no
   surface" message keeps `test_a_search_with_no_surface_says_none_and_points_at_nothing`,
   now on level ground (made up), where no trial has anything driving it and
   Spencer solves none (95 not converged).
4. **The direct march's xfail target.** `_march_circle()` works out the
   target as the moment ratio cu R (arc length) / sum(W x), on 600 equal
   slices from the toe-side end (x 23.139, on the face) to x = 41 where the
   arc turns vertical: 3.084 (the review's 3.084). Since step 4 the circle
   needs a crack asked for (its centre is 3.0 m below the crest), so the
   geometry gives a dry 3.0 m crack, which closes it at x = 41. The strict
   xfail keeps `rel=0.01` and still fails because Spencer has no solution
   (checked with `--runxfail`: `method_solved` False); a new passing test,
   `test_the_march_circle_s_target_is_the_moment_ratio`, holds the target at
   3.084 and Bishop's 30-slice value on the same closed surface within 1 %
   (3.068). The march stays out. 2.9435 is gone from the test and from
   `DESIGN.md` (both places).
5. **The headline, and what the agent is told.** The headline stays the
   lowest solved surface, flagged, with the lowest surface with no flag
   beside it. The adapter's search result now carries, besides `critical`,
   `critical_flags` and `lowest_unflagged` (FOS null and a note when none
   was found), `refuse_option` (the option that leaves surfaces with the
   headline's flags out, or that there is none) and `summary`, one
   sentence with all four (`search_supplements.headline_report`). Before,
   the option was only in the warning list. Only m_alpha has such an
   option; for base tension, side tension and the line of thrust there is
   none, and `refuse_option` says so (the step's section 11 already notes a
   general refusal option was not built). Test:
   `test_slope_stability_adapter::test_a_flagged_headline_never_comes_alone`
   (ACADS 1(a) by Spencer through the adapter).
6. **The table** is below.
7. **Recorded, not pinned**: Duncan's Example 4 by Janbu, master 1.2033,
   branch 0.8426 flagged (low_m_alpha), 1.2154 with none, 1.2033 with
   `low_m_alpha='refuse'`; V-029's grid 1.2471 to 1.1752 (1.2471 with
   `negative_friction_term='zero'`); PPG-1 1.2231 to 1.1007. No printed
   value for any. In `VALIDATION.md` B22 as well.
8. **Calls made**: section 7, (a) to (d).

**The table (ruling 6)**, branch against `origin/master` (ee2e152; its
`slope_stability` is the same at ddebb25), one probe
(`review/step3fix/probe6.py`, outside the repository) run once on each.
"Lowest with none" is given when the headline carries a flag.

| Case | Master | Branch | Branch flags | Lowest with none | Step 4 discards (branch; master) | Step 3 flagged trials (branch) | Printed |
|---|---|---|---|---|---|---|---|
| Vertical cut, cu 20, Bishop search, default | 0.9353 | 0.9353 | none | - | reverse_curvature_no_crack 12; reverse_curvature_no_crack 12 | 0 of 2956 | Taylor 0.851 (3.83 cu / (gamma H), as DESIGN.md gives it) |
| Vertical cut, `m_alpha_rule='every_slice'` | 0.9348 | 0.9353 | none | - | reverse_curvature_no_crack 12; reverse_curvature_no_crack 7 | 288 of 2956 (low_m_alpha 288) | the same |
| ACADS 1(b), given 3.81 m crack full of water, Bishop, Slide2's grid | 1.5896 | 1.5896 | none | - | within_cracked_depth 456; within_cracked_depth 456 | 0 of 6287 | 1.596 (Slide2 Verification #2, Table 2.2, p. 25) |
| ACADS 1(b), the same, Spencer | 1.5849 | 1.5849 | none | - | within_cracked_depth 456; within_cracked_depth 456 | 400 of 6415 (base_tension 296, thrust_outside 137) | 1.592 (the same) |
| #85, reverse curvature discarded, Bishop, default | 1.5080 | 1.5080 | none | - | reverse_curvature_discard_asked 28; reverse_curvature_discard_asked 28 | 0 of 2135 | 1.531 (Slide2 Verification #85); Duncan and Wright (2005, Fig. 6.34) 1.51 |
| #85, discarded, every slice | 1.5109 | 1.5080 | low_m_alpha | 1.5109 | reverse_curvature_discard_asked 28; reverse_curvature_discard_asked 26 | 421 of 2135 (low_m_alpha 421) | the same |
| Taylor 45 deg, D = 2.25, default search | 1.9419 | 1.9419 | none | - | reverse_curvature_no_crack 38; reverse_curvature_no_crack 38 | 0 of 3270 | 1.947 (Taylor's chart, FHWA NHI-06-088 Fig. 6-15, p. 6-28) |
| ACADS 1(a), random noncircular, 200 trials, Spencer | 1.0915 | 0.6332 | base_tension, side_tension, thrust_outside | 1.0959 | 0; 0 | 108 of 144 (side_tension 73, thrust_outside 106, base_tension 1) | 1.00 (Giam and Donald 1989; Slide2 Verification #1) |
| ACADS 4, weak-layer search, 700 trials, Spencer | 0.8301 | 0.8301 | side_tension, thrust_outside | 1.6666 | 0; 0 | 288 of 344 (base_tension 124, side_tension 257, thrust_outside 268) | 0.760 (Slide2 manual p. 52) |

Read: on master the vertical cut by every slice and #85 by every slice leave
the m_alpha circles out (0.9348, 1.5109); the branch keeps and flags them,
which moves #85 to 1.5080 flagged with 1.5109 beside it. The vertical cut
reads 0.9353 with no flag on the branch, 0.0005 above master's 0.9348: no
flagged circle is lower, and the grid refines around a different circle
when the m_alpha circles are left out (the branch with
`low_m_alpha='refuse'` reads 0.9348 as master does). Step 4's
discards are the same on both trees wherever both score the same circles
(the counts differ only where master leaves m_alpha circles out first).
ACADS 1(a)'s 1.0915 to 0.6332 is the removal of the moment-axis screen for
Spencer (step 3's item 5), unchanged by the fix round.

**Factors of safety that moved, in the merge or in the fixes** (before is
the step 3 branch at 677fce4 unless named):

| Case | Before | After | Printed | Why |
|---|---|---|---|---|
| Vertical cut, every slice | 0.842 flagged | 0.9353, no flag | Taylor 0.851 | merge: step 4 discards the crack-closed toe circle |
| #85, one support, default Bishop search | 1.3713 (toe circle) | 1.508 (what 'discard' reads) | 1.531 / D&W 1.51 | merge: step 4 |
| W10 noncircular, 100 trials (`test_choices`) | 0.462 flagged (unpinned; the test ran 20 trials, no surface) | 0.462 flagged, pinned | none | defect 3 |
| Janbu / OMS noncircular searches | jagged zig-zags below 1.5 admitted | refused, counted | none | defect 1; ACADS 1(a) headlines unchanged (1.0357, 0.9170 in the review) |
| The march circle's target | 2.9435 | 3.084 (moment ratio, 600 slices) | none (arithmetic) | ruling 4 |

No value pinned to print moved in the fix round.

**Tests** (single files, judged by pytest's exit code; the full suites are
the lead's gate on GitHub).

| Run | Result |
|---|---|
| After the merge: `test_m_alpha.py` | 17 passed, 1 xfailed, after the every-slice case was updated to 0.935 with no flag (exit 0) |
| After the merge: `test_search_reinforcement.py`, `test_slope_stability.py`, `test_calc_steps_modern.py` | 33, 93, 24 passed (exit 0 each) |
| After the merge: `test_own_solution_or_no_number.py` | 26 passed, 1 xfailed (exit 0) |
| After the merge: `test_cracks_as_printed.py`, `test_acceptability.py` | 41 and 21 passed after the `confirm_below` fix and the exemption circle's crack (exit 0 each) |
| After the fixes: `test_own_solution_or_no_number.py` | 27 passed, 1 xfailed (exit 0); the xfail fails on `method_solved` False (`--runxfail`) |
| After the fixes: `test_choices.py`, `test_acceptability.py`, `test_search_robustness.py`, `test_search_modern.py`, `test_slope_stability.py` | 7, 23, 28, 11, 93 passed (exit 0 each) |
| After the fixes: `funhouse_agent/tests/test_slope_stability_adapter.py` | 93 passed (exit 0) |
| After the fixes: `test_m_alpha.py` | 16 passed, 1 xfailed, 1 failed: the DE seeding test unpacked one warning and now gets the acceptability warning too; updated, that test passes (exit 0) |
| After the fixes: `funhouse_agent/tests/test_defaults_used.py` (whole) | 856 passed, 17 skipped, 2 failed (exit 1): both `ModuleNotFoundError: geotech_references.em_1905`, master's EM 1110-1-1905 adapter against the venv's references 1.7.0; not slope tests |
| `module_work/doc10/check_manual.py --rebuild` | OK (exit 0), manual unchanged by the fix round |
