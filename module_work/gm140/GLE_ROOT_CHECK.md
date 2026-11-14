# GM-140, the GLE solver: a root, or no number (BACKLOG GM-344)

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
   inclined side force is printed.

**The 1.089 is not reproduced on master.** With r_u, master (9c73795) reads
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
| r_u, Spencer | **0.6065 at 18.00 deg (not a root)** | 0.9182 at 26.319 deg | 0.9182 at 26.319 deg | root 0.9182 at 26.319 deg |
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
| (a) master first | 0.6065 at 18.00 deg, converged, not a root (the 1.089 needs step 5a's water-table reading) |
| (a) with the option: within 0.5 % of 0.6704, theta within 0.1 deg of beta, no flag | **Not met**: 0.9182 at 26.319 deg, no flag. The root exists on the lower branch (section 3). Strict xfail |
| (a) as printed: 26.20 deg, 0.6505, flagged for base and side tension | **Not met**: 0.9182 at 26.319 deg, no flag. The root exists on the lower branch. Strict xfail |
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
setting). Master's values are roots of neither set of equations. EM C-10c's
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

New: `slope_stability/tests/test_gle_root.py` (21 tests: 19 pass, 2 strict
xfails holding check (a) as the brief wrote it).

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

Runs (pytest's exit code; `-n auto`): the counts are in the final message
and in section 10.

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
searches with these methods run two to three times faster.

## 11. Noted, not fixed

- **The case reads Spencer 0.918 with no flag** (section 0). The half-sine
  Morgenstern-Price reads 0.6779. No check marks a slice whose own equations
  are nearly singular under an inclined side force (D29 / step 3).
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
