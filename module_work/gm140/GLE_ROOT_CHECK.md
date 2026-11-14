# GM-140, the GLE solver: a root, or no number (BACKLOG GM-344)

**Fix round after review (2026-11-14): section 12.** It changes section 0
items 3 and 5 (0.918 is now reported WITH a `low_m_alpha` finding on slice
1), corrects the record of master on the r_u case (1.0887 at 31.13 deg, not
0.6065; section 0, 3 and 4), and rewrites check (a) (section 4).

**Printed benchmarks that move under the default** (ruling 3, the equations
as printed; the lead's call R14 keeps it). No pin moved, no tolerance
widened. "Before" is the circle-end branch; the option is
`base_in_tension='no_friction'`.

| Benchmark | Before | As printed (default) | With the option | Printed |
|---|---|---|---|---|
| F&K B1 dry, Spencer (50 slices) | 2.0745 | 2.0712 (-0.16 %) | 2.0744 | 2.073 (Fredlund and Krahn 1977; Slide2 Verification #21) |
| F&K B1 dry, Morgenstern-Price | 2.0764 | 2.0708 (-0.27 %) | 2.0767 | 2.076 |
| F&K B1 r_u 0.25, Spencer | 1.7633 | 1.7567 (-0.37 %) | 1.7629 | 1.761 |
| F&K B1 r_u 0.25, Morgenstern-Price | 1.7651 | 1.7560 (-0.51 %) | 1.7653 | 1.764 |
| ACADS 4 declared family, lowest, Spencer | 0.7532 | 0.7480 (-0.69 %) | 0.7495 | 0.760 (Slide2 #9, Table 9.4, p. 52) |
| ACADS 4 declared family, lowest, Morgenstern-Price | 0.7225 | 0.7157 (-0.94 %) | 0.7222 | GLE 0.720 (the same) |
| V-042, FOSM by Spencer, F at the means | 1.3493 | 1.3346 (-1.09 %) | 1.3482 | Hassan and Wolff 1.334 by Bishop (Slide2 manual pp. 137-139) |
| V-043 sand, DE seed 1, Spencer | 1.2337 | 1.2374 (+0.30 %) | 1.2374 | 1.188 (Slide2 manual pp. 147-150), 1.192 (Tandjiria) |

**Other files changed, and why.** Beside `gle.py`, its tests, the option's
passage through `analysis.py` and the adapter, `DESIGN.md` and this file:
`slope_stability/acceptability.py` (the `base_tension` finding said "the
solver takes no friction on those bases", which is no longer the default: it
now says which rule was in force and names the option) and
`slope_stability/solution.py` (the no-solution reason: the range is 0 to 1.5,
and it carries where the branch followed had no state). Tests of earlier
steps that used surfaces which no longer solve, or solve elsewhere, are
updated (section 8). The search files are not touched.

Branch `claude/gm-140-gle-root`, from master 9c73795; master 3d553a8 (step 9
merged) merged in before the push. 2026-11-14. The "before" columns below
are master 9c73795; step 9 changed no value in them (its tests pass on the
branch, section 2 item 10).

## 0. Read this first

**The step's check (a) does not pass as the brief wrote it, and cannot under
ruling 2.** The brief took 26.20 deg, F 0.6505 to be the root "on the branch
from lambda = 0" of Spencer's equations as printed, and 26.563 deg, F 0.6704
to be what the option gives. Both are roots (my own script finds them, and
the solver's equations have them: section 3). Neither is on the branch from
lambda = 0:

1. On the case with r_u, the force equation has one solution at lambda = 0
   (F 0.6775, Janbu's). For lambda above 0 the toe slice (base inclined
   26.57 deg AGAINST the sliding) has a pole at
   F = tan(phi') tan(theta - alpha_toe), which rises from 0.34 at lambda = 0
   to 0.90 at lambda = 0.5. The branch that starts at lambda = 0 lies above
   that pole at every lambda (0.6775, 0.691, 0.719, 0.799 at lambda 0, 0.2,
   0.3, 0.4) and is pushed up by it: its crossing is **26.32 deg, F 0.918**,
   the toe wedge all but locked (N' on slice 1 +1,381 kN/m, E on side 1
   1,533 kN/m). The roots at 0.6505 (as printed) and 0.6704 (the slice taken
   frictionless) lie on the branch BELOW the toe slice's pole, which starts
   near F = 0 at lambda just above 0, not at Janbu's solution. Section 3 has
   the table of both branches.
2. 0.918 is a root of Spencer's equations as printed, under both settings
   (no base is in tension on it, so the option changes nothing). The probe
   took it for a root of the module's half-applied device only.
3. So under ruling 2 as written the solver reports **Spencer 0.918 at
   26.32 deg on the case, under both settings, with no flag** (the line of
   thrust stays inside the mass with r_u). That is 37 % above the closed
   form 0.6704. It is a solution of the method's equations, which the 1.089
   and the 0.6065 below were not; it is not the answer the brief wanted.
   The two tests that hold check (a) as the brief wrote it are strict xfails
   (`test_check_a_as_printed_as_the_brief_wrote_it`,
   `test_check_a_with_the_option_as_the_brief_wrote_it`).
   **Fix round (R6):** 0.918 is now reported with a `low_m_alpha` finding on
   slice 1 (its own denominator 0.0176), and the two strict xfails are gone;
   check (a) as rewritten is in section 4.
4. **Ruling 2's other form is not a way out.** "The lowest of every root in
   the range" would pick 0.6505 and 0.6704 here, but on an ordinary made-up
   circle (c' 10, phi' 25, r_u 0.25) Spencer's equations as printed have
   eleven roots from -40 to 60 deg: the usual one (17.89 deg, F 1.3742),
   six at F 0.10 to 0.16 and four at F 1.30 to 1.31, each of the ten with
   slices below their poles and bases and sides in tension. The lowest root
   is nonsense on every ordinary circle.
5. What would give the closed form on this case is for the lead and the
   owner: Morgenstern-Price with the half-sine function reads 0.6779 on it,
   flagged for the line of thrust outside the mass
   (no side shear at the ends, so the toe wedge does not lock; EM
   1110-2-1902 C-10b names a change of side-force inclination among its
   remedies for a steep exit). A check that flags a slice whose own
   equations are nearly singular (the toe slice's determinant is 0.02 at
   the 0.918 root; step 3's `low_m_alpha` reads the m_alpha of horizontal
   side forces, 0.57 there, and does not flag) would at least mark it. Not
   built: it is an acceptability check, step 3's, and no number for an
   inclined side force is printed. **Built in the fix round (R6, section
   12).**

**The 1.089 is not reproduced on master.** [Corrected in the fix round
(R10): master 9c73795 and 3d553a8 both read Spencer 1.0887 at 31.13 deg on
the r_u case through `analyze_slope` and `solve_method` (0.8901 at 25.41
deg from `gle_fos` at its default tol), "converged", not a root; no tree
giving 0.6065 was found. The water-line value below was not re-run.] With r_u, master (9c73795) reads
Spencer 0.6065 at 18.00 deg, "converged"; with the water line on the ground
(the full head, the module's reading of a water line) 0.5513 at 16.69 deg.
Neither is a root. The water-table reading of the probe (u = gamma_w z
cos^2 beta from a water line) is step 5a's and is not on master. The
failing-first test is built on the r_u case
(`test_what_master_reported_is_not_a_root`,
`test_spencer_reports_a_root_of_spencers_equations`).

**Check (c) holds to 2.3 %, not exactly.** Dry, Spencer's own solution on
this surface is 26.365 deg, F 1.3805 under both settings (my script: the
same to four figures); the closed form is 1.349. The difference is the
passive wedge of the 2 m toe ramp (about 160 kN/m on a driving force of
about 6,200 kN/m), not the solve.

**Values compared with print that move away from it** (none re-pinned):

| Value | Before | After (default) | With the option | Printed |
|---|---|---|---|---|
| V-043 sand fill, noncircular DE, seed 1 (seeds 2, 3 unchanged at 1.234) | 1.234 | 1.237 | 1.237 | Slide2 1.188, Tandjiria 1.192 |
| ACADS 4 (V-028) declared family, lowest, Spencer | 0.7532 | 0.7480 | 0.7495 | referee 0.78; Slide2 0.760 |
| ACADS 4 declared family, lowest, Morgenstern-Price | 0.7225 | 0.7157 | 0.7222 | Slide2 GLE 0.720 |
| V-042 pinned circle, Spencer (FOSM F at the means) | 1.3493 | 1.3346 | 1.3482 | Hassan and Wolff's circle; the pinned tests hold |

The first is a strict xfail
(`test_v043_sand_de_reaches_the_same_minimum_from_three_seeds`); the pins of
the others hold within their tolerances and were not touched. Each moves
because the friction term is now as printed on bases in tension (the option
column is close to "before").

**A behaviour the lead should see before merging.** A circle with cohesion
whose crest slices are in tension can now have NO Spencer solution where
master read a number. On step 2's made-up fill slope (`_FILL_CIRCLE`) Spencer
solved up to about 40 kPa on master (1.645 at 30 kPa, 1.917 at 38); it now
solves up to about 28 kPa as printed (to under 20 kPa with the option) and
has no solution above. My script finds no root from -5 to 60 deg at 30 kPa
under either setting: master's values rested on the half-applied device.
EM 1110-2-1902 C-10c prints the remedy (a tension crack). Section 6.

## 1. What the print says

- Spencer (1967, pp. 14 to 15; Eqs. 7 and 8, p. 15): the interslice forces
  are parallel; F and theta are found where the curve of F from force
  equilibrium and the curve from moment equilibrium, each against theta,
  cross. Morgenstern and Price (1965, Eq. 18, p. 85): X = lambda f(x) E.
  Neither prints a rule for a negative effective normal force on a base, nor
  which crossing to take where there are several. (From the account's
  section 9 and the brief: the papers are the owner's copies and in no
  repository. No slice equation is transcribed in the code or the script:
  both are the statics of a slice with those assumptions, and say so.)
- EM 1110-2-1902 C-10a (pp. C-33 to C-34): the forces on each slice are
  examined; tensile forces should not exist on the sides or bottom of
  slices; the line of thrust within the mass. C-10b (pp. C-34 to C-36): a
  steep exit gives very large or negative forces; four remedies. C-10c
  (p. C-36): tension at the crest, a tension crack.
- UFC 3-220-10 p. 362 (read from the reference library's own text,
  `dm7_1/text/chapter07.json`): "Critical failure surfaces can be missed as
  a result of non-convergence. The engineer should be able to identify when
  convergence issues are present."
- FHWA NHI-06-088 Sec. 6.3.2, Eq. 6-10, p. 6-8: the infinite slope with
  seepage parallel to the face, c' = 0:
  F = (gamma' / gamma_sat) tan(phi') / tan(beta). The r_u form is not
  printed there as such; the test shows that r_u = (gamma_w / gamma)
  cos^2(beta) gives Eq. 6-10's pore pressure gamma_w z cos^2(beta) on every
  slice of the plane (`test_the_r_u_of_the_case_is_the_pore_pressure_of_eq_6_10`).
- Whitman and Bailey (1967, p. 487): the user is warned when a slice's
  denominator is negative or small.

## 2. What changed

`slope_stability/gle.py`:

1. **One state, one solver.** Every state of the lambda scan is
   `_GLESystem._state`: the direct Morgenstern-Price march with the F that
   closes it, found from the neighbouring lambda's F. The lagged iteration
   is used only for the record of F_m and F_f at lambda = 0 (Bishop's and
   Janbu's values) and by `janbu_fos`.
2. **Poles.** Slice i's two equations are singular at
   F = tan(phi') tan(theta_i - alpha) (`_poles`). The closing force changes
   sign there without being zero, and a root can lie close beside a pole
   (0.918 beside 0.892): the search for a state looks on each side of every
   pole (`_nearest_root_cut`) and takes only a state on the neighbour's side
   of each pole (`side`). A sign change of the closing force that does not
   close (`_CLOSES`) is not a state.
3. **A crossing is judged on the state's net moment**, R = resisting moment /
   F - driving moment with that state's own N, E and X (a couple, the same
   about any point, since the forces are in equilibrium). It is zero to
   rounding at a root. `|F_m - F_f| < max(50 tol, 1e-3)` is gone: on the
   case the fitted moment axis is 15 km away and F_m - F_f is under 1.5e-4
   at every state of both branches, a jump included.
4. **A jump is not a bracket.** A sign change of R is refined (`refine`,
   bisection and false position by turns) until a state has R = 0 or the
   bracket is closed in lambda (1e-12) without one. The second is a jump:
   its lambda goes in `GLEResult.jumps` and the scan goes on.
5. **The factor reported is the state's** (`fos == fos_force`), and the
   per-slice forces reported are that state's, not recomputed at a mean F.
6. **No crossing: no number**, with where the branch had no state and each
   jump in the reason (`solution.gle_failure_reason`).
7. **Ruling 2.** The scan starts at lambda = 0 on the branch on which every
   slice is above its pole and goes up in steps of 0.1 to 1.5; the first
   crossing is reported. `gle.ROOT_CHOICE`, in a search's `choices`.
8. **Ruling 3.** `gle.BASE_TENSION_RULES`: `as_printed` (default) and
   `no_friction`, the latter now in every equation (`_normal_eff`, `_march`,
   the force and moment sums, the E march of the lagged iteration).
   Parameter `base_in_tension` of `analyze_slope` and
   `search_critical_surface`, held in a context variable
   (`gle.base_tension`), so rapid drawdown, the yield coefficient, FOSM and
   Monte Carlo follow the rule in force. Janbu's simplified method is the
   same system at lambda = 0 and follows it too. `GLEResult.base_tension_rule`
   and `.frictionless`.
9. **A crossing with no net driving moment or force is not taken** (step 2's
   'no_driving' rule; the lagged iteration could not step from such a
   state).

10. **The marched solve at one given lambda** (`_solve_marched`). Since
    step 9 it is also the force-equilibrium solution at a fixed side-force
    inclination (`rapid_drawdown.fixed_inclination_solution`). Called that
    way it finds its F as the scan does: the root nearest the start that
    closes the march with every slice above its pole. As the fallback of
    the lagged iteration (Janbu's simplified method, and the record of F
    at lambda = 0) it is called with `above_poles=False` and takes the
    nearest sign change, as before: Janbu's own solve is not this step's
    (with the pole rule there, Duncan Ex. 4's default Janbu search would
    read 1.203 with no flag in place of 0.843 flagged). After master's step 9 was merged, EM 1110-2-1902 G-5's
    example of the Corps 1970 procedure (printed 3.49 and 1.35, side forces
    at 19.4 deg) read 0.526 and 1.97 with the equations as printed and the
    old search, which took the first sign change near F = 1.3, a pole. With
    this the printed values hold as step 9 pinned them (-0.1 % and -0.7 %).
    `rapid_drawdown.py` is not touched.

Adapter (`funhouse_agent/adapters/slope_stability.py`): `base_in_tension` on
`analyze_slope` and `search_critical_surface`, with its description,
allowed values and when it applies; a probe
(`slope_stability|analyze_slope#base_in_tension`, 1.3742 against 1.3756).

## 3. The case: before, after, and the second computation

All values made up. Ground (-20, 0), (0, 0), (400, 200), (420, 200); one
soil, gamma = gamma_sat = 19.5, phi' 34, c' 0; surface (20, 10), (22, 9),
(380, 188), (382, 191); 60 slices; r_u = (9.81 / 19.5) cos^2(26.565 deg).
Closed form 0.6704 (wet), 1.349 (dry).

| | Master 9c73795 | Branch, as printed | Branch, option | Own script |
|---|---|---|---|---|
| r_u, Ordinary Method of Slices | 0.6699 | 0.6699 | 0.6699 | |
| r_u, Janbu (corrected) | 0.6782 | 0.6782 | 0.6782 | |
| r_u, Morgenstern-Price | 0.6779 | 0.6779 | 0.6779 | |
| r_u, Spencer | **1.0887 at 31.13 deg (not a root)** (corrected in the fix round; was recorded as 0.6065 at 18.00 deg) | 0.9182 at 26.319 deg | 0.9182 at 26.319 deg | root 0.9182 at 26.319 deg |
| dry, Spencer | 1.3779 at 25.41 deg | 1.3805 at 26.365 deg | 1.3805 at 26.365 deg | root 1.3805 at 26.365 deg |
| dry, Morgenstern-Price | 1.3564 | 1.3564 | 1.3564 | |

Every root of Spencer's two equations on the r_u case, theta 0 to 75 deg, F
0.2 to 3 (`module_work/gm140/gle_root_spencer_script.py`; the Q form, no E,
no X, no lambda, no fitted axis; Newton's method from a grid of starts, each
root checked to close both sums to 1e-8 of the forces):

| Setting | theta, deg | F | N' slice 1, kN/m | Side 1 (along theta), kN/m | In tension |
|---|---|---|---|---|---|
| as printed | 26.200 | 0.6505 | -110.6 | -160.7 | base 1; sides 1 to 6 and on |
| as printed | 26.319 | 0.9182 | +1,381 | +1,710 | none |
| as printed | 56.959 | 0.6674 | -21.6 | -39.2 | bases 1 and 60; sides |
| slice frictionless | 26.319 | 0.9182 | +1,381 | +1,710 | none |
| slice frictionless | 26.563 | 0.6704 | +40.6 | +29.1 | none |

Against the probe's numbers in the brief: 26.20 deg / 0.6505 / -111 kN/m and
E -144 kN/m (160.7 cos 26.2 deg), 56.96 deg / 0.6674 and 26.563 deg / 0.6704
are the same. **Different:** the probe gave 26.32 deg, F 0.918 as the one
crossing of the module's half-applied equations; it is also a root of the
equations as printed and of the slice taken frictionless.

The two branches of the force equation, as printed (F that closes the march;
R the net moment, kN m/m; E on side 1, kN/m):

| lambda | Pole of the toe slice | Branch above the pole (from lambda = 0) | Branch below the pole |
|---|---|---|---|
| 0 | none | F 0.6775, R +6,400, E 77 | none |
| 0.2 | 0.525 | F 0.691, R +7,531, E 177 | F 0.507, R -38,905, E -1,473 |
| 0.3 | 0.635 | F 0.719, R +10,263, E 379 | F 0.591, R -13,018, E -657 |
| 0.4 | 0.759 | F 0.799, R +12,630, E 900 | F 0.636, R -2,502, E -260 |
| 0.48 | 0.870 | F 0.898, R +3,465, E 1,436 | F 0.6493, R -183, E -154 |
| 0.49 | 0.885 | F 0.912, R +1,165, E 1,502 | F 0.6503, R -29, E -146 |
| 0.495 | 0.892 | F 0.919, R -94, E 1,535 | F 0.6508, R +41, E -142 |
| 0.5 | 0.900 | F 0.926, R -1,427, E 1,568 | F 0.6512, R +107, E -138 |
| 0.6 | 1.061 | F 1.082, R -45,202, E 2,174 | F 0.6572, R +866, E -92 |

(With the slice taken frictionless the upper branch is the same, and a lower
branch appears from lambda about 0.31 at F 0.670, crossing at 0.49995.)
Master's scan stepped between these two branches; where it bisected the
sign change between an upper state and a lower one it closed on the jump.

The solver's own scan started on the lower branch finds the script's other
roots, so the solver's equations are the script's under both settings
(`test_the_other_roots_are_roots_of_the_solver_s_equations_too`).

**Check (b).** At every root the solver reports on the case (r_u and dry,
both settings) and on an ordinary circle with a crest slice in tension
(1.3742 at 17.891 deg as printed, 1.3756 at 17.836 deg with the option), the
script's root is the same to 0.1 % and 0.05 deg
(`test_spencer_reports_a_root_of_spencers_equations`,
`test_a_crest_slice_in_tension_under_each_rule`). Not compared: the case
with a water line on the ground. There the module's slices carry a pond
thrust of 11.7 kN/m a slice at an elevation above the base, which the
script's one horizontal force through the base midpoint does not place (see
"noted, not fixed").

## 4. The checks

| Check | Result |
|---|---|
| (a) master first | 1.0887 at 31.13 deg, converged, not a root (corrected in the fix round, R10; the record had 0.6065 at 18.00 deg) |
| (a) as R6 rewrites it: Spencer 0.918 at 26.32 deg under both settings WITH the `low_m_alpha` finding on slice 1 | **Met**: 0.91818 at 26.319 deg, finding on slice 1 only, denominator 0.0176 (the review's script: 0.0176), side force 26.3 deg, E 1,533 kN/m, N' 1,381 kN/m; EM C-10b and its remedies named (`test_check_a_spencer_is_reported_with_the_finding_on_slice_1`) |
| (a) as R6 rewrites it: Morgenstern-Price half-sine within 2 % of 0.6704 | **Met**: 0.6779, +1.12 % (`test_check_a_morgenstern_price_half_sine_within_2_percent`) |
| (a) as R6 rewrites it: the roots below the toe slice's pole stay as the script's arithmetic | 0.6505 (as printed) and 0.6704 (frictionless) in `test_the_other_roots_are_roots_of_the_solver_s_equations_too`, stated as not what the solver reports |
| (a) as the first brief wrote it (0.6505 flagged; 0.6704 with the option) | Not met, by ruling 2; the two strict xfails are removed (R6) |
| (a) never 1.089 | Met |
| (a) 0.918 only if a true root, carrying its thrust flag, and not the one ruling 2 picks | A true root (the script). No thrust flag with r_u (it carries one with the full-head reading). It IS the one ruling 2 picks |
| (b) own script, 0.1 % at every root reported | Met |
| (c) dry, 1.349 under both settings | 1.3805 under both, the script's root; 2.3 % above the closed form (the toe ramp) |
| (d) section 11, before and after, both settings | Section 5 |
| (e) run time of a default search | Section 7: faster |
| A sign change between two branches is not a bracket | `test_a_sign_change_between_two_branches_is_not_a_bracket` |
| No crossing in the range: no number, with its reason | `test_no_crossing_in_the_range_is_no_number_with_its_reason`, `test_a_branch_that_ends_is_named_in_the_reason` |

## 5. Section 11, before and after

`step2_section11.py` and `step3_section11.py`, unchanged, on master
(9c73795) and on the branch before the merge of step 9, then again with the
option forced. Rows that do not move to four figures: Slide2 #85 without a
support (Spencer 6 x 6: 1.0468) and with one (Spencer 1.5505, GLE 1.5732,
passive GLE 1.3594; on Slide2's circle Spencer 1.5653, GLE 1.5656), #86
(Spencer 1.6065, GLE 1.607; no supports 0.942), ACADS 1(a) circles (Spencer
and GLE 0.984, Janbu 0.983), V-043 circles (default Spencer search 1.1859,
pinned circle 1.2096), V-031 Spencer search 1.4174, V-039 static F, V-030,
V-045, B4, the ACADS 4 weak-layer search (0.8301) and every Bishop row.

| Benchmark | Before | After, as printed | After, option | Printed | What moved it |
|---|---|---|---|---|---|
| V-031 pinned circle, Spencer | 1.4247 | 1.4245 | 1.4247 | Slide2's circle; pin holds | friction as printed on a base in tension |
| V-031 pinned circle, GLE | 1.4091 | 1.4081 | 1.4092 | pin holds | the same |
| V-044 pinned circle, Spencer | 1.5962 | 1.5929 | 1.5961 | pin holds | the same |
| V-039 structure, ky, Spencer | 0.56491 | 0.56448 | 0.56479 | pin holds | the same |
| V-039 ky, Morgenstern-Price | 0.55899 | 0.55771 | 0.55881 | pin holds | the same |
| V-039 ky, Janbu | 0.46921 | 0.46854 | 0.47043 | pin holds | the rule now reaches Janbu |
| V-042 FOSM, Spencer, F at the means | 1.3493 | 1.3346 | 1.3482 | pin holds | friction as printed on bases in tension |
| V-042 Monte Carlo, Spencer, mean | 1.3598 | 1.3452 | 1.3588 | pin holds | the same |
| ACADS 4 family, lowest, Spencer | 0.7532 | 0.7480 | 0.7495 (5 of 81 without a solution) | referee 0.78, Slide2 0.760 | the same; away from print |
| ACADS 4 family, lowest, Morgenstern-Price | 0.7225 | 0.7157 | 0.7222 | Slide2 GLE 0.720 | the same; away from print |
| ACADS 4 family, lowest, Janbu | 0.7284 | 0.7251 | 0.7360 | | the rule now reaches Janbu |
| ACADS 4 calc-package report, Spencer headline | 0.879 | 0.862 | 0.865 | | the search's path |
| ACADS 1(a), random noncircular, 200 trials, Spencer | 0.6332 flagged (1.0959 unflagged) | 1.0916 flagged (1.0959 unflagged) | 1.0916 | 1.00 | the 0.633 surface was a crossing at a negative lambda; toward print |
| V-043 sand fill, noncircular DE, seeds 1 to 3 | 1.234, 1.234, 1.234 | 1.237, 1.234, 1.233 | 1.237, 1.234, 1.233 | Slide2 1.188, Tandjiria 1.192 | the search's path; seed 1 away from print (strict xfail) |
| Duncan Ex. 4, Janbu, 30 slices | 0.8426 flagged | 0.8426 flagged | 1.2033 | | the option reaches Janbu |
| #85 no support, Spencer 6 x 6, surfaces without a solution | 337 | 334 | 334 | | all now "no lambda from 0 to 1.5" |

Pinned tests moved (each with its reason in the test): section 8.

## 6. Where the rule now holds, and what it does to ordinary results

One solver (`gle.gle_fos`) is behind every caller, and the rule for a base
in tension is a context variable read when a system is built, so the rule
holds in: `analyze_slope` (and its `compare_methods` columns),
`compare_methods_table`, every family of `search_critical_surface`
(`search._compute_fos` through `solution.solve_method`), `spencer_fos`,
`morgenstern_price_fos`, `janbu_fos`, rapid drawdown and its search
(`rapid_drawdown.py` calls `gle_fos`), the yield acceleration and Newmark,
FOSM and Monte Carlo, `reliability/wrappers.py`, `geo_project`'s
limit-equilibrium builder, the adapter and the calc packages. The OPTION is
a parameter of `analyze_slope` and `search_critical_surface` (and their
adapter methods); the other entry points run under the default, or under
`with gle.base_tension(...)`.

**Cohesive circles with the crest in tension.** Step 2's made-up fill slope
(10 m, phi' 22), circle (9.5, 11) R 11.2, Spencer:

| c', kPa | Master | Branch, as printed | Branch, option | Bishop |
|---|---|---|---|---|
| 0 | 0.6359 at 34.9 deg | 0.6359 | 0.6359 | 0.6400 |
| 20 | 1.3070 at 23.3 deg | 1.2877 at 18.0 deg | no solution | 1.2830 |
| 28 | | 1.5481 at 9.6 deg | no solution | |
| 30 | 1.6449 at 18.9 deg | no solution | no solution | 1.6042 |
| 38 | 1.9172 at 14.4 deg | no solution | no solution | 1.8622 |
| 50 | no solution | no solution | no solution | 2.2510 |

At 30 kPa F_f is above F_m at lambda = 0 and rises faster: they do not cross
for lambda from 0 to 1.5 (the script: no root from -5 to 60 deg under either
setting; the review's script over -40 to 60 deg finds roots only at negative
theta, -33.7 and -19.8 deg at 30 kPa, with slice denominators below 0.01,
and no crossing on the branch from lambda = 0). Master's values are roots of
neither set of equations. EM C-10c's
remedy is a tension crack (`tension_crack_depth_basis`), which step 4 built.

**Lambda below 0.** Scanning down as well (as master did) finds roots there:
1.7486 at -30.3 deg on the fill circle (8, 12) R 12 at 40 kPa, with bases
and sides in tension (a root by the script too), and, on jagged polylines,
values such as 0.41 on Taylor's D = 2 profile (the circle, as printed, is
0.706) and 0.42 on V-043 (printed 1.188). Ruling 2 says "from lambda = 0
upward"; that is what is built.

## 7. Run time

`search_critical_surface` on the ACADS 1(a) slope, defaults, best of two, the
same machine:

| Search | Master | Branch |
|---|---|---|
| Bishop, default circular | 2.4 s | 2.2 s |
| Spencer, default circular (2,187 surfaces) | 14.6 s | 5.3 s |
| Morgenstern-Price, default circular | 15.2 s | 5.6 s |
| Spencer, random noncircular, 500 trials | 7.9 s | 4.4 s |

And from the section 11 runs: #85 without support, Spencer 6 x 6, 13.7 s to
6.2 s; #85 with its support, Spencer 23.6 s to 14.6 s, GLE 23.2 s to 16.2 s;
the ACADS 4 weak-layer search 10.6 s to 11.6 s.

## 8. Tests

New: `slope_stability/tests/test_gle_root.py` (19 pass, 2 strict xfails
holding check (a) as the brief wrote it; after the fix round 26 pass, no
xfail: section 12).

Earlier tests changed, each with its reason beside the change:

| Test | Was | Now |
|---|---|---|
| `test_own_solution_or_no_number::_FILL_CIRCLE` comment; `test_fosm_meeting_a_surface_with_no_solution_gives_no_result` | mean 38 kPa, std 12 | mean 22, std 8 (Spencer has no solution at 38 any more) |
| `...::test_the_robust_lambda_search_runs_without_supports` | asserted the "bracket started from" note | the scan now has a state at lambda = 0 there; the note is not raised |
| `test_acceptability::test_a_steep_exit_polyline_...` | Spencer | Morgenstern-Price (Spencer has no solution on that polyline for lambda 0 to 1.5) |
| `test_acceptability::test_search_noncircular_called_directly_reports_the_flags`; `funhouse_agent/tests/test_slope_stability_adapter::test_a_flagged_headline_never_comes_alone` | 0.633, three flags | 1.092 (printed 1.00), two flags |
| `test_acceptability::test_base_and_side_tension...` | "no friction" in the text | the rule and the option in the text |
| `test_choices::test_a_short_search_reports_its_flagged_surface` | 0.462 flagged four ways | 9.49, no flag, 84 of 100 trials without a solution (made-up slope) |
| `test_slice_boundaries::test_a_corner_close_to_another_boundary_is_kept` | 2.521 | 2.525 (made-up surface; friction as printed on the bases in tension) |
| `test_search_robustness::test_a_flagged_lowest_surface_is_reported_flagged` | four flags, none unflagged, all trials flagged | two flags, an unflagged surface, 245 of 285 flagged |
| `test_m_alpha::test_de_search_leaves_out_and_counts_negative_m_alpha_surfaces`, `test_w11_de_search_runs_its_whole_budget` | Spencer | Janbu (with Spencer the surfaces those tests are about have no solution now, so none is left out for m_alpha) |
| `validation_examples/test_published_v043...::test_v043_sand_no_reinforcement_noncircular` | pinned the three seeds | pins the minimum (1.234); the three-seed pin is a new strict xfail |

Runs on the branch with master 3d553a8 merged (pytest's exit code, `-n 4`):

- `slope_stability/tests` (all of it) and every
  `validation_examples/test_published_*` file that imports
  `slope_stability`: **926 passed, 17 skipped, 9 xfailed, 0 failed** (exit
  0). The xfails are the two of check (a), V-043's three-seed pin, and six
  that were strict xfails on master.
- The `calc_package`, `funhouse_agent`, `reliability`, `fem2d`,
  `geotech_common`, `report_ingest`, `dxf_import` and `geo_project` test
  files that name `slope_stability`, `funhouse_agent/tests/test_defaults_used.py`
  (all of it), `webapp/tests/test_page_fingerprints.py` and
  `webapp/tests/test_public_text.py`: **2,924 passed, 27 skipped, 1
  failed**. The one failure is
  `test_defaults_used.py::...[profile_figure|plot_data]` (logx / logy),
  which fails the same way on master in this environment and has nothing
  to do with this step.
- The page fingerprints pass unchanged: nothing was re-recorded.
- The user manual: rebuilt by its generator after the merge
  (`check_manual.py --rebuild --write`, then `check_manual.py` and
  `webapp/tests/test_user_manual_current.py`, each exit 0), with
  geotech-references at e1bbc5b, the commit master pins.

## 9. Calls made

1. **Ruling 2, as built.** The first crossing from lambda = 0 UPWARD on the
   branch on which every slice is above its pole. Other forms: (a) the
   lowest of every root in the range (not built; nonsense on ordinary
   circles, section 0 item 4); (b) the scan also below lambda = 0, as
   master (roots with tension, section 6).
2. **Ruling 3, as built.** As printed by default; the slice taken
   frictionless is the option, in every equation. Other form: the device as
   the default. On the benchmark set the option's values are within 0.001 of
   master's (section 5) and the default's move by up to 1.1 % (V-042).
3. **The branch is the one above every pole** (mine; the brief did not
   decide where the branch "starts" when the force equation has several
   solutions at lambda = 0, or none). Other form: the solution nearest
   Janbu's whatever its side. Without the rule a surface on which Janbu's
   method has no solution started from a state with slices below their
   poles and reported 0.07 to 0.4 on jagged polylines.
4. **A crossing is judged on the net moment of the state**, to 1e-7 of the
   forces times the size of the mass, not on |F_m - F_f| < tol (mine;
   ruling 1 says "to the solver's own tol"). F_m = F_f then holds far
   inside tol. With the test in F a jump passes wherever the moment axis is
   far away.
5. **Janbu's simplified method follows the same rule** for a base in tension
   (mine: it is the same system at lambda = 0). Its values move with it
   (section 5: V-039 ky 0.46921 to 0.46854; ACADS 4 0.7284 to 0.7251).
6. **A crossing with no net driving moment or force is not taken** (step 2's
   'no_driving' rule applied to the state).
7. **R6, the finding on an inclined side force** (the orchestrator's R6;
   the lead's call; joins this step's OWN-143 part). `low_m_alpha` reads,
   for Spencer, Morgenstern-Price and GLE, the denominator of the slice's
   own equations at the reported state, cos(alpha - theta_i) +
   sin(alpha - theta_i) tan(phi') / F, against Whitman and Bailey's 0.2
   (p. 487). Other form: the finding on horizontal side forces only (m_alpha
   at the method's F), as step 3 built it; there 0.918 carries no finding
   (m_alpha 0.57 on the toe slice).
8. **R15, the lambda range** (the lead's call; OWN-143). The scan goes
   upward from lambda = 0 to 1.5 on the branch that starts there and stops
   where that branch ends (R12). Other form: where no crossing is met
   upward, a scan downward from 0 (D29's former range, -1.5 to 1.5). Not
   built.
9. **R7** (the orchestrator's; joins OWN-143). A cohesive circle with its
   crest in tension that has no Spencer solution is the method's own
   behaviour where the review's script finds no root on the branch from
   lambda = 0 (30 and 38 kPa as printed; 20 and 28 kPa with the option).
   Other form: a defect to be fixed (none found).
10. **A slice taken frictionless is exempt from `low_m_alpha`** under the
   default rule, as a slice with phi' = 0 is (mine, the fix round; the
   module's own reasoning, no print). Other form: look at it; with the
   option the crest slice of the fill circle at 10 kPa (79 deg) would be
   flagged at 0.189.

## 10. Release note

Spencer's method, Morgenstern-Price and GLE now report a factor of safety
only where one state of the slope meets both the force and the moment
equation; where they do not, there is no number and the result says why.
Before, on some surfaces (long thin planar slides, surfaces with a steep toe
ramp) the solver could report a value that was not a solution. A slice whose
base is in tension is now treated as the method's equations print it, with
the earlier device (the slice taken frictionless) available as
`base_in_tension='no_friction'` and now applied consistently; Janbu's
simplified method follows the same setting. Most answers move by less than
0.5 %; on circles through soil with cohesion whose crest is in tension the
move is up to about 1 %, and some such circles that returned a number now
return none (add a tension crack, as EM 1110-2-1902 C-10c prints). The
solver no longer looks for solutions with a negative lambda. Default
searches with these methods run two to three times faster. On a surface
whose toe is steep against the sliding, Spencer's method can lock the toe
wedge and read high (0.918 where the infinite slope gives 0.670); such a
result now carries the `low_m_alpha` finding, read on the slice's own
equations, with EM 1110-2-1902 C-10b's remedies. Rapid drawdown takes
`base_in_tension` as well.

## 11. Noted, not fixed

- ~~**The case reads Spencer 0.918 with no flag**~~ Fixed in the fix round
  (R6): flagged `low_m_alpha` on slice 1. The half-sine Morgenstern-Price
  reads 0.6779.
- `search_rapid_drawdown` does not take `base_in_tension` (the fix round
  gave it to `rapid_drawdown_fos` only); it runs under the default.
- A search's `m_alpha_min` record (the search files, out of this step's
  scope) still reads m_alpha. Its flags, and its refusal under
  `low_m_alpha='refuse'`, come from the finding (`surface_findings`), so
  with Spencer, Morgenstern-Price and GLE they follow the slice's own
  denominator; no pinned search value moved in the files run.
- The 0.05 m_alpha clamp (D29) is unchanged. With the option, a frictionless
  slice whose base is vertical (m_alpha = cos alpha = 0) rests on it: on the
  4 m step of `test_a_corner_close_to_another_boundary_is_kept` the option
  reads 2.835, which is not a root of the script's unclamped equations (as
  printed, 2.525, is).
- The fitted moment axis of a noncircular surface (D29) is unchanged; the
  crossing no longer depends on it (the net moment is a couple).
- A water line lying on the ground gives each slice a pond thrust (11.7 kN/m
  a slice on the case) at an elevation above the base (D-number of step 5a's
  pore-pressure reading; that step is not on master).
- `test_the_march_is_tried_where_the_lagged_iteration_cannot_step` (step
  2's strict xfail) still fails as it should, but for a new reason: the
  circle now solves, at 2.9458 against the hand value 3.084 (-4.5 %, 30
  chord slices of a circle that turns vertical at its crack). Its reason
  text describes the earlier state.
- `DESIGN.md`'s GM-60 record still says "no lambda in -1.5 to 1.5" of the
  #85 toe circle (history; Spencer and GLE still have no solution there).
- `gle_fos`'s `lambda_bounds[0]` is kept in the signature and not used.
- The device is not self-consistent where the frictionless solve returns a
  positive N' (the slice is "in tension" by its frictional solve only): at
  the 0.6704 root slice 1's N' is +40.6 kN/m. As the device was.
- `funhouse_agent/tests/test_defaults_used.py::...[profile_figure|plot_data]`
  fails on master too in this environment (logx / logy); not this step's.

## 12. Fix round after review (2026-11-14)

Branch `fix/gm140-gle-root-review`, on the lead's brief (the orchestrator's
R6 and R7; the review's defects as R8 to R13; the lead's calls R14 and
R15). BACKLOG is left to the lead at the merge.

**The merge.** `fix/gm140-circle-ends-review` merged first (290fe39).
Conflicts: `VALIDATION.md` keeps both sections (the circle-end step keeps
B23; this step's section is now **B24**: the brief's "B23" for this step
means B24); `funhouse_agent/tests/method_probes.json` keeps both new probe
lines (valid JSON); `docs/user_manual/manual.html` was rebuilt by its
generator. `origin/master` (fd4fe6f) was already in. What the merge moved:
Spencer on Taylor's arc of the vertical cut (the circle-end step's check
(c)), no solution before, now **0.8515 at 48.03 deg** (Taylor's chart, FHWA
NHI-06-088 Fig. 6-14, p. 6-27: 3.83 cu / (gamma H) = 0.8511; +0.04 %;
flagged for a base and sides in tension); the default Spencer search of the
cut 0.8632 to 0.8513 (+0.02 % against 0.8511). Morgenstern-Price still has
no solution on that arc. `test_circle_ends`' test that held "no solution"
now holds the root (`test_spencer_on_taylors_arc_is_a_root_near_taylor`).

**Per ruling.**

- **R6 / R8 (defect 1).** `gle.GLEResult` carries, at the reported state,
  `slice_denominator` (cos(alpha - theta_i) + sin(alpha - theta_i)
  tan(phi') / F with the tan(phi') the solve used), `side_force_theta_deg`
  (theta_i = atan(lambda f) on the side the march solves for; 0 on the last
  slice's open side, as in the march) and `side_force_E`.
  `acceptability.low_m_alpha_finding` reads that denominator for Spencer,
  Morgenstern-Price and GLE (m_alpha for Bishop and Janbu, unchanged); the
  finding gives the slice, the value, theta, the interslice force on that
  side and N' on the base, cites Whitman and Bailey (1967, p. 487) and EM
  1110-2-1902 C-10b (p. C-34), and names C-10b's four remedies as the EM
  prints them (pp. C-34 to C-36; for (2) the EM names the Modified Swedish
  Method and the Morgenstern and Price method, and Morgenstern-Price with the
  half-sine is the module's form). A slice the option takes frictionless is
  exempt under the default rule, as a phi' = 0 slice is (call 10). Tests:
  `test_check_a_spencer_is_reported_with_the_finding_on_slice_1`,
  `test_check_a_morgenstern_price_half_sine_within_2_percent`,
  `test_the_finding_does_not_fire_on_an_ordinary_circle` (least 0.654 as
  printed, the review's 0.654; 0.516 with the option, the frictionless crest
  slice), `test_the_denominator_is_m_alpha_for_horizontal_side_forces`,
  `test_a_slice_taken_frictionless_is_exempt_as_phi_0_slices_are`; the two
  strict xfails are gone. In `test_m_alpha`, Spencer's finding on Duncan's
  steep exit now reads 0.0211 on slice 40 (side force 8.7 deg), where
  m_alpha is 0.176 (the search's `m_alpha_min`, unchanged).
- **R7 / R13.** The no-solution reason names EM 1110-2-1902 C-10c (p. C-36)
  and `tension_crack_depth` (or `tension_crack_depth_basis='rankine'`, Eq.
  C-36) where the state at lambda = 0 has a base or side in tension on a
  slice with cohesion. The fill circle at 30 and 38 kPa gives no number
  under both settings; the review's script finds no crossing on the branch
  from lambda = 0 there, nor at 20 and 28 kPa with the option, so no root is
  lost. Test: `test_a_cohesive_crest_in_tension_with_no_root_names_c10c`.
- **R9 (defect 3).** `test_v043_sand_no_reinforcement_noncircular` is the
  strict xfail in place (seed 1 ends 0.25 % higher, 1.237);
  `test_v043_sand_lowest_de_seed_and_the_random_search_above_it` holds the
  `min(de)` and random-search lines. No pin moved.
- **R10 (defect 4).** Master 9c73795 and 3d553a8 (each extracted alone,
  first on the path) read the r_u case at **1.0887 at 31.13 deg** through
  `analyze_slope` and `solve_method` (30, 59, 61, 120 slices: 1.08874 to
  within 2e-7), and 0.8901 at 25.41 deg from `gle_fos` at its default tol
  1e-4; dry, 1.3779 at 25.41 deg (as section 3 had). No tree giving 0.6065
  was found. Corrected: `test_gle_root.py` (docstring and
  `test_what_master_reported_is_not_a_root`, which now holds both master
  values as non-roots), `VALIDATION.md` B24, sections 0, 3 and 4 here.
- **R11 (defect 5).** `rapid_drawdown.fixed_inclination_solution` asks
  `_solve_marched(lam, 1.3, above_poles=True)` by name and reports
  S = (c' l + N' tan(phi')) / F with the tan(phi') each slice's solve used
  (was max(N', 0)); `rapid_drawdown_fos` takes `base_in_tension` (validated
  in the adapter; named in `choices`) and runs every stage under it. No
  drawdown factor of safety moved (the table); S moves on bases in tension
  only. Tests: `test_base_in_tension_reaches_every_stage_and_is_named`,
  `test_force_equilibrium_stage_reports_the_shear_it_solved` (made-up
  circle; crest N' -2.76 kN/m as printed). `search_rapid_drawdown` does not
  take the input (section 11).
- **R12 (question 4).** `first_crossing` stops at the first lambda with no
  state on its branch; the reason says "stopped where that branch ends,
  between lambda = a and b (no state at b)". No value moved. Test:
  `test_the_scan_stops_where_its_branch_ends`.
- **R14.** The printed benchmarks that move are tabled in this file's first
  lines and in `VALIDATION.md` B24, with the option's values.
- **R15.** `DESIGN.md` records the change from D29's -1.5 to 1.5 (both
  forms); the reason says "the scan went upward from lambda = 0 on the
  branch of solutions that starts there", to 1.5 or to where it ends.

**Check (a) as R6 rewrites it, on the merged tree**: section 4. Spencer
0.91818 at 26.319 deg under both settings, `low_m_alpha` on slice 1 only
(0.0176); Morgenstern-Price 0.6779 (+1.12 % on 0.6704); 0.6505 and 0.6704
kept as the script's roots, not what the solver reports.

**The table** (`review\gle\fix_table.py`; baseline = the circle-end branch's
`slope_stability`, extracted alone; both settings on the branch; "After round 2": the same script on fix round 2's tree, both settings, `review\gle\fix_table_round2.json`):

| Case | Baseline | Branch, as printed | Branch, option | After round 2 | Change | Printed, page | Cause |
|---|---|---|---|---|---|---|---|
| Probe r_u, Spencer | 1.0887 (thrust) | 0.9182, `low_m_alpha` slice 1 (0.0176) | 0.9182, same | same, both settings | -15.7 % | closed form 0.6704 | baseline not a root (ruling 1); R6 finding |
| Probe r_u, M-P half-sine | 0.6779 (lambda 0.750) | 0.6779 (0.747) | 0.6779 | same | 0.00 % | 0.6704: +1.12 % | |
| Fill circle 10 kPa | 0.9698 | 0.9620 | 0.9688 | same F; the option's flags no longer include `low_m_alpha` (round 1's exemption, b0ce78b, made after its table) | -0.81 % | made up | ruling 3 |
| Fill circle 20 kPa | 1.3070 | 1.2877 | none | same | -1.48 % | made up | ruling 3; option: no root (review) |
| Fill circle 28 kPa | 1.5772 | 1.5481 | none | same | -1.85 % | made up | as above |
| Fill circle 30 / 38 kPa | 1.6449 / 1.9172 | none | none | none / none; Spencer meets no crossing below 0 (F2) | | made up | no root on the branch (R7) |
| F&K B1 dry, Spencer / M-P | 2.0745 / 2.0764 | 2.0712 / 2.0708 | 2.0744 / 2.0767 | same | -0.16 / -0.27 % | 2.073 / 2.076 | ruling 3 |
| F&K B1 r_u 0.25, Spencer / M-P | 1.7633 / 1.7651 | 1.7567 / 1.7560 | 1.7629 / 1.7653 | same | -0.37 / -0.51 % | 1.761 / 1.764 | ruling 3 |
| ACADS 4 family, Spencer / M-P | 0.7532 / 0.7225 | 0.7480 / 0.7157 | 0.7495 / 0.7222 | same | -0.69 / -0.94 % | 0.760 / GLE 0.720, Table 9.4 p. 52 | ruling 3 |
| V-042 FOSM, Spencer | 1.3493 | 1.3346 | 1.3482 | same | -1.09 % | 1.334 by Bishop, pp. 137-139 | ruling 3 |
| V-043 DE seed 1 | 1.2337 | 1.2374 | 1.2374 | same | +0.30 % | 1.188, pp. 147-150 | search path (strict xfail) |
| ACADS 1(a) 200-trial Spencer search | 0.6332 | 1.0916 | 1.0916 | 1.0916, same; 6 of the 55 trials without a solution now solve, F 8.05 to 39.12 (F1) | +72 % | 1.00 | no crossing below lambda 0 (R15) |
| Taylor's arc, Spencer / M-P | none / none | 0.8515 / none | 0.8515 / none | same | | 0.8511, Fig. 6-14 p. 6-27 | merge (ruling 2's scan on the arc) |
| Vertical cut, default Spencer search | 0.8632 | 0.8513 | (not run) | same | -1.38 % | 0.8511: +0.02 % | merge |
| EM example, Bishop 12 slices, stages | 2.1925 / 1.5137 / 1.4384 | same | same | same | 0 | 2.20 / 1.52 / 1.44, pp. G-15, G-18, G-20 | |
| #95, force 19.4 deg, 103 ft pool | 1.3379 | 1.3379 | 1.3379 | same | 0 | 1.35, p. G-15 | |
| EM example, Spencer 50 slices, stages | 2.19983 / 1.51281 / 1.43412 | 2.19982 / 1.51281 / 1.43413 | same | same | <0.001 % | as above | |

**Other pinned values that moved on this branch** (`git diff
fix/gm140-circle-ends-review...HEAD` on test files and `VALIDATION.md`; no
pin was widened): ACADS 1(a)'s direct 200-trial search 0.633 to 1.092, two
flags, 1.096 beside it unflagged (`test_acceptability`,
`test_slope_stability_adapter`; printed 1.00; R15); the made-up short
search of `test_choices` 0.462 (four flags) to 9.49 (none; 84 of 100 trials
without a solution; R15); `test_slice_boundaries`' made-up corner 2.521 to
2.525 (ruling 3); `test_search_robustness`' flagged lowest surface, four
flags to two, an unflagged surface now found (2.085), 245 of 285 trials
flagged (R15); `test_own_solution_or_no_number`'s FOSM input 38 to 22 kPa
(an input, so the surface solves); `test_m_alpha`'s two DE tests run Janbu
in place of Spencer, and its steep-exit warning reads 0.0211 in place of
0.18 (R6); V-043 seed 1 1.234 to 1.237 (strict xfail); Taylor's arc by
Spencer, none to 0.8515 (merge). All others are new tests.

**Tests run in the fix round** (single files, pytest's exit code, all 0):
`test_gle_root` (26 passed), `test_circle_ends` (15), `test_rapid_drawdown`
(17), `test_rapid_drawdown_em_appendix_g` (23), `test_acceptability` (23),
`test_own_solution_or_no_number` (27, 1 xfail), `test_choices` (7),
`test_m_alpha` (18), `test_search_robustness` (28), `test_gle` (25),
`test_slope_stability` (93), `test_ground_crossings` (21),
`test_pore_pressure_as_printed` (50),
`funhouse_agent/tests/test_slope_stability_adapter` (94),
`funhouse_agent/deep/tests/test_wave1_geotech` (20; `describe_method
rapid_drawdown_fos` 15,849 characters, under 16,000),
`funhouse_agent/tests/test_defaults_used -k slope_stability` (40 passed, 20
skipped: slope_stability frozen), `validation_examples/
test_published_v043_slope_reinforced` (8, 2 xfail), `..._v052_drawdown_
pilarcitos` (6), `..._v029_slope` (6), `webapp/tests/
test_user_manual_current` (1). After the merge, before the fixes:
`slope_stability/tests/test_validation.py` 20 passed (30 min; not re-run
after the fixes, which moved no value it pins). The manual rebuilt by its
generator (`check_manual.py --rebuild --write`, then `check_manual.py`,
exit 0), geotech-references at e1bbc5b. The full suites are left to the
lead's GitHub run.

**Fix round 2 (2026-11-14; the lead's brief, two items).**

- **F1. A crossing in the last step before the branch ends.** The scan
  (`_GLESystem._scan`, which `first_crossing` and `crossing_below` share):
  where a step of 0.1 meets no state, the step is bisected between the last
  lambda with a state and that lambda, each trial state found from the
  branch's last state (its F and side), until the two are less than 1e-4
  apart (`gle._BRANCH_END_TOL`); each state met on the way is tested as a
  step's state is (agreement, then a sign change of R against the state
  before it, refined by `refine`). `branch_end` holds the narrowed pair and
  the reason prints it to 4 decimals. One scan for both `base_in_tension`
  settings. On ACADS 1(a)'s 200-trial Spencer search (seed 42, 25 slices)
  trial 97 now reads 13.98592 at 23.2500 deg (lambda 0.4296; the review's
  script: 13.9859 at 23.25 deg) and trial 16 8.05148 at 5.9556 deg (lambda
  0.1043; the script: 8.0514 at 5.9556 deg), under both settings, each a
  root of the second computation's equations (its Newton step from the
  module's F and theta lands within 5e-6 of that F, relative; the residuals at the
  module's own values are below 1e-6 of their scales). Trial 167 still has
  no number (the scan goes to 1.5). Of the review's 55 trials without a
  solution, 6 now solve (16, 73, 97, 150, 172, 187; F 8.05 to 39.12, each a
  root of the second computation) and 49 stay without one; the search's
  value 1.0916 does not move. Tests:
  `test_a_crossing_in_the_last_step_before_the_branch_ends_is_found` (both
  trials, both settings), `test_trial_167_still_has_no_number`;
  `test_the_scan_stops_where_its_branch_ends` now reads "between lambda =
  0.4499 and 0.4500 (no state at 0.4500)".
- **F2. The orchestrator's ruling 2, last clause.** Where ONE surface is
  analysed (`analyze_slope` solves through the context
  `gle.single_surface`; a context variable, no new input) and a side-force
  method meets no crossing upward, the branch from lambda = 0 is scanned
  DOWN to -1.5 (`lambda_bounds`' lower bound, D29's former range) with the
  same steps and end rule (`_GLESystem.crossing_below`). A crossing met
  there is named in the no-number reason: F, lambda, the slices with a base
  in tension, the sides in tension, the least slice denominator and its
  slice, and that it is not reported because below lambda = 0 the
  interslice shear acts against the sense the sliding gives it. Off by
  default in `gle_fos`, so never in a search's trials. A search's
  governing surface is solved again (`report_acceptability`,
  `m_alpha_confirmation`), but neither solve prints a reason (the first
  returns where there is no solution, the second records "no_solution"
  without words), so no scan is made there either. Cases (probe
  `review\gle\fix2_probe.py`): step 2's fill circle at 30 kPa by
  Morgenstern-Price (half-sine), as printed, names lambda -0.0342, F 1.6089,
  the crest slice 30's base and sides 24 to 29 in tension, least
  denominator 0.4352 on slice 30; with the option, and at 38 kPa under
  both settings, Morgenstern-Price meets none below 0. Spencer on that
  circle at 30 and 38 kPa meets none below 0 (its branch from 0 ends near
  lambda -0.328 and -0.295 as printed, -0.202 with the option). The
  made-up planar case has its crossings upward (Spencer 0.9182,
  Morgenstern-Price 0.6779), so nothing is scanned below 0 there. The
  search: ACADS 1(a)'s 200-trial Spencer search leaves 49 trials without a
  solution and makes no scan below 0 (calls counted: 0); trial 167
  analysed alone makes one. Tests:
  `test_a_crossing_below_0_is_named_where_one_surface_is_analysed`,
  `test_a_search_makes_no_scan_below_0`. Recorded in `DESIGN.md` (D29, the
  lambda range).
- **Pins moved** (`git diff e44a89c -- '*test*' slope_stability/
  VALIDATION.md`): `test_choices::test_a_short_search_reports_its_flagged_
  surface`, 84 to 83 trials without a solution (F1: a crossing in the last
  step before its branch ends found; 9.49 stands);
  `test_the_scan_stops_where_its_branch_ends`, the reason's "0.4 and 0.5"
  to "0.4499 and 0.4500" (F1). `VALIDATION.md` unchanged; no printed
  benchmark moved (the table's column).
- **Run time.** The default circular Spencer search on ACADS 1(a)
  (`search_critical_surface(acads1a, method="spencer")`), first search in
  its process: 6.4 s on e44a89c, 6.2 s after (0.98383, the same circle).
  Run again later in the same process after round 2 it took 9.3 s and
  10.1 s (that position was not timed on e44a89c).
- **Tests run in fix round 2** (single files, pytest's exit code, all 0):
  `test_gle_root` (33), `test_choices` (7), `test_acceptability` (23),
  `test_search_robustness` (28), `test_validation` (20, 30 min),
  `test_rapid_drawdown` (17), `test_rapid_drawdown_em_appendix_g` (23),
  `funhouse_agent/tests/test_slope_stability_adapter` (94),
  `funhouse_agent/deep/tests/test_wave1_geotech` (20), and, beside the
  brief's list, `test_gle` (25), `test_circle_ends` (15),
  `test_own_solution_or_no_number` (27, 1 xfail), `test_m_alpha` (18);
  the reason's words
  changed, so in full: `funhouse_agent/tests/test_top_level_params` (53),
  `test_defaults_used` (881 passed, 21 skipped, 30 min),
  `test_gm179_unused_inputs` (20), `test_calc_package_tool_parity` (190).
  The manual: `check_manual.py` exit 0 (OK, not moved; geotech-references
  at e1bbc5b), so no rebuild.
