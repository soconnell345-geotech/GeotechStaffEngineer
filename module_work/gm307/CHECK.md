# GM-307 — MSE walls, what GM-114 left (check record)

Branch `claude/gm-307-mse-left`, from master `bc8994b`. Touches
`retaining_walls/` (MSE only), its calc package and the adapters;
`slope_stability` untouched.

GEC-11 = FHWA GEC-11, *Design and Construction of Mechanically Stabilized
Earth Walls and Reinforced Soil Slopes*, Vol I FHWA-NHI-10-024 and Vol II
FHWA-NHI-10-025, November 2009, read in `geotech-references/docs/` with
PyMuPDF; pages with equations as pictures were rendered and read
(Eqs. 4-1 to 4-3, 4-5 to 4-7, 4-10 to 4-13, 4-15, 4-16, 4-26; Figs. 4-4,
4-5, 4-9; Tables 4-1, 4-2, 4-5, E4-5.1).

Commits: `de9fa8b` (part 1), `9e002ab` (part 2), `00877c3` (part 3), then
this record and the DESIGN.md section.

## Part 1 — a sloping or broken backslope in the package

**What the print says.** Vol I Sec. 4.4.4: Kab by Eqs. 4-2, 4-3 (p. 4-15)
with θ = 90° and δ = β; a broken backslope takes β = δ = I, the angle to
the ground 2H behind the face (p. 4-15, Fig. 4-4, p. 4-14). F_T = ½ Kab
γ_b h² on h = H + L tan β, h − H not more than S (Eq. 4-7 and its note,
p. 4-20); F_H = F_T cos β (Eq. 4-10, p. 4-21); F_T sin β resists (Eq.
4-13, p. 4-21; Eq. 4-16, p. 4-23); V2 the slope fill over the zone (Figs.
4-3, 4-4, 4-8); bearing for a slope Eq. 4-21 (p. 4-28). Internally S_eq =
½(0.7H) tan β, σ_2 = γ_r S_eq (Fig. 4-11, p. 4-38; Eq. 4-29, p. 4-42),
H1 (Fig. 4-9(a), p. 4-34), Zp (Fig. 4-15, p. 4-50).

**What changed.** `retaining_walls/calc_steps.py`: the level-only guard is
gone; `_mse_backslope_steps` prints the geometry, Kab with Γ, F_T, its
components, the traffic thrust behind a crest and V2 from the run's own
`external_terms["backslope_loads"]`; the overturning and bearing steps cite
Eqs. 4-16, 4-21; the internal steps add S_eq and σ_2, H1 for steel, the
Zp column and the 0.3 H1 surface; the section figure draws the slope.
`funhouse_agent/adapters/calc_package.py`: `_mse_package_scope` (the
refusal) removed.

**Before / after** (made-up wall `SLOPE`: H 6, L 5, Sv 0.6, γ_r 19, φ_r 34,
φ_b 30, γ_b 18, φ_f 30, q_allowable 400, β 18°, geogrid GG-II):

| | master | branch |
|---|---|---|
| package | refused ("level backfill only") | FOS_s 1.68, FOS_ot 4.01, FOS_b 2.38, σ_v 168.3 kPa |
| mse_wall tool | 1.681, 4.008, 168.3 | the same |

Kab 0.3948 (Γ 1.9974, hand), h 7.625 m, F_T 206.6, F_H 196.5, V2 77.2
kN/m, σ_2 12.96 kPa — each pinned by hand in
`test_kab_and_the_thrust_under_a_slope_by_hand`. Broken backslope
(`BROKEN`: β 26.57°, S 2 m, q 12): I = atan(2/12) = 9.46°, h 8.000 m,
Kab 0.3477, F2 33.4 kN/m (hand); package FOS_s 1.80, FOS_ot 2.98, σ_v
194.6. No calculated value changes: the package prints what the tool
already computed. V-056 / V-057 (Examples E3, E1) keep passing.

## Part 2 — a battered face (10° or more)

**What the print says.**
- p. 4-15: Kab for "near vertical walls (defined as walls with a face
  batter of less than 10 degrees from vertical)" with θ = 90°; "Battered
  Wall with or without Backslope: For an inclined front face and reinforced
  zone (i.e., batter) equal or greater than 10 degrees from vertical, the
  coefficient of earth pressure can be calculated using Equations 4-2 and
  4-3 where θ is the face inclination from horizontal, and β the surcharge
  slope angle as shown in Figure 4-5. The wall friction angle δ is assumed
  to be equal to β."
- Fig. 4-5 (p. 4-16): F_T at h/3 on the back of the reinforced zone, at δ
  to its normal, F_H at θ − 90 to that normal.
- p. 4-36: Eq. 4-26, "the following simplified form of the Coulomb
  equation", for face batters of 10° or more; the Ka is "a Coulomb earth
  pressure relationship, assuming no wall friction and a β angle equal to
  zero", which for a vertical wall "reduces to the Rankine equation" (Eq.
  4-25).
- p. 4-33 and Fig. 4-9(b) (p. 4-34): for extensible reinforcement with a
  batter over 10°, the Coulomb failure plane, through the toe, with the
  printed equation for ψ and "with δ = β"; Fig. 4-9(a)'s note for
  inextensible reinforcement: "If wall face is battered, an offset of
  0.3H1 is still required, and the upper portion of the zone of maximum
  stress should be parallel to the wall face".
- Vol II p. 10-36: "Negative (outward leaning) batter is not acceptable."

**Two misprints, and how they are read (stated in the result's choices).**
1. Eq. 4-26 is printed `sin²(θ+φ) / (sin³θ [1 + sin φ / sin θ])`, the
   bracket not squared. At θ = 90° that is 1 − sin φ, not Eq. 4-25 — the
   page's own words say it must be. Eqs. 4-2, 4-3 with δ = β = 0 give the
   bracket squared and Eq. 4-25 at θ = 90°. The code takes that; the
   choices print the as-printed value beside it (0.3281 against 0.2087 at
   φ 34°, θ 102°).
2. Fig. 4-9(b)'s left side reads `tan(ψ − θ)`, with "θ = wall batter
   angle". The right side has no value at θ = 0 (tan 90°); with θ from the
   horizontal (Fig. 4-5's) a vertical face gives tan(45 − φ/2), and only
   `tan(ψ − φ)` then gives the figure's printed "For vertical face ψ = 45 +
   φ/2". The code reads `tan(ψ − φ)` with θ = 90 + ω.

**What changed.** `MSEWallGeometry.face_batter` (default 0; `battered`,
`theta_deg`); `backslope_loads` takes θ, the thrust at δ − ω, the arms on
the parallelogram; `_external_terms` runs that path for a battered face too
(level, a slope, a surcharge); `check_internal_stability` takes
`battered_internal_Ka`, `extensible_failure_angle` and Zp from the top of
the face; `face_batter_choices`; `MSEWallResult.face_batter`; the LRFD
path refuses a battered geometry; the adapter takes `face_batter` (aliases
`wall_batter`, `batter`, `batter_angle`, `face_batter_deg`, GM-114's
refused names); the package's `dropped` entry and `_MSE_BATTER_REASON` are
gone (every part traced to the print); the calc steps print θ, the
components at δ − ω, Eq. 4-26, ψ, and the figure draws the batter.

**What stays refused, and why** (`BATTER_BROKEN_BACKSLOPE_REASON`,
`LRFD_BATTER_REASON`): a battered face under a broken backslope — p. 4-15
gives I for a vertical wall only and Fig. 4-9(b)'s "For wall with a broken
backslope, use δ = β_cs" never defines β_cs; and `mse_wall(lrfd_external=
True)` with a battered face — Eqs. 4-15, 4-16, 4-19 to 4-21 are written for
a vertical face (F_T sin β at L).

**Before / after** (made-up wall `BATTER`: H 6, L 4.5, Sv 0.6, γ_r 19, φ_r
34, φ_b 30, γ_b 18, φ_f 30, q_allowable 400, ω 12°, steel strips Sh 0.9):

| | master | branch, ω 12° | branch, ω 0 |
|---|---|---|---|
| package | refused (face_batter dropped) | FOS_s 3.49, FOS_ot 8.52, FOS_b 3.63, σ_v 110.1 kPa, FOS_po,min 2.11 | 2.74, 5.34, 2.85, 140.2, 1.56 |
| mse_wall tool | face_batter refused (unknown input) | the same as the package | unchanged from master |

By hand (`test_the_external_checks_on_a_battered_face_by_hand`): Kab =
sin²132 / [(1 + sin 30 / sin 102)² sin³ 102] = 0.25841; F_T = 83.72,
F_H = 81.89, F_V = −17.41 kN/m at 4.925 m; FOS_s = (513 − 17.41) tan 30 /
81.89 = 3.494; FOS_ot = (513 × 2.888 − 17.41 × 4.925) / 163.78 = 8.521;
e < 0, σ_v = 110.1. Internally Ka = 0.20874 and, with geogrids, ψ = 56.00°
(tan(ψ − φ) = 0.40403), La at z = 0.3 m = 5.7 (cot 56 − tan 12) = 2.633 m,
Tmax = 0.714 kN/m (geogrid package: FOS_s 2.87 against 2.25 at ω 0, the
lowest sheet governing). No GEC-11 worked example has a battered face
(searched Vol I and Vol II for "batter"; Vol II's examples are vertical,
Example E1's 3° MBW face taken as vertical, Vol II pp. E1-2, E1-3).

## Part 3 — an LRFD calc package

**What the print says.** Vol I Table 4-1 (p. 4-4): Strength I LL 1.75,
Service I 1.00; Table 4-2 (p. 4-4): EH active 1.50 / 0.90, EV retaining
walls 1.35 / 1.00; Table 4-5 (p. 4-19): bearing 0.65, sliding 1.0; p. 4-23:
e within L/4 on soil (3/8 L on rock), "use the minimum EV and maximum EH
load factors" for the critical combination; Vol II Table E4-5.1 (p. E4-6):
Strength I (maximum) 1.35 / 1.50 / 1.75, Strength I (minimum) 1.00 / 0.90 /
1.75, Service I 1.00; Service L/6 (Tables E3-6.3, E4-6.3).

**What changed.** `run_mse_lrfd_external_stability` (adapter) returns
(result, inputs in force); the tool's `_run_` wrapper calls it.
`retaining_wall_package` takes `wall_type="mse_lrfd"` (or the tool's
name), runs it, and the module's `_calc_steps_mse_lrfd` prints the factors
with their printed sources (and names any given otherwise), the
combinations, the unfactored loads, and per combination the tool's sliding
CDR, eccentricity and bearing pressure, L′, q_R and CDR, Strength I (min)
included, local shear and warnings. Nothing is recomputed in the steps.
The response carries `passes`, `checks_failed`, `checks_not_run`, the
governing and Strength I min values.

**Before / after** (made-up wall `LRFD`: H 6, L 4.5, γ_r 19, φ_r 34, q 12,
φ_f 30, γ_f 18): master refused `wall_type="mse_lrfd"`; the branch prints
sliding CDR 1.877 / 2.048 / 1.391 (max / min / critical), e 0.6758 /
0.6456 / 0.9123 m ≤ 1.125, Service 0.5291 ≤ 0.75, bearing CDR 1.825 /
2.508 / 1.889, σ_v Strength I min 178.17 kPa. By hand
(`test_strength_i_min_by_hand`): sliding 513 tan 30 / (0.9 × 114 + 1.75 ×
24) = 2.048; e (Eq. 4-15) = (0.9 × 114 × 2 + 1.75 × 24 × 3) / 513 =
0.6456 m; e_B = 331.2 / 607.5 = 0.5452, σ_v = 178.2 kPa, q_R = 0.65 × 0.5
× 3.4096 × 18 × 22.4 = 446.8, CDR 2.508. The tool's numbers are
unchanged.

## Tests

venv per the brief (Python 3.12, `pip install -e packages/aec-agent-core
-e .`). Exit codes read directly.

- `retaining_walls calc_package funhouse_agent/tests/test_gm114_mse_calc_package.py
  funhouse_agent/tests/test_gm307_mse_package.py
  funhouse_agent/tests/test_calc_package_tool_parity.py
  funhouse_agent/tests/test_calc_package_figure_conventions.py
  validation_examples/test_published_v009_v011.py
  validation_examples/test_published_v021_v022.py
  validation_examples/test_published_v056_v057_gec11_slope.py
  funhouse_agent/tests/test_defaults_used.py
  webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  **2001 passed, 2 failed, 26 skipped**. The 2 failures fail the same on
  master `bc8994b` (checked with the branch's changes stashed):
  `test_large_pile_group_leaves_the_forces_to_the_table` and
  `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  — neither touches MSE walls.
- Every other `funhouse_agent/tests` file naming the MSE tools or the
  package (21 files: adapter ergonomics, calc-package adapter / assumed
  inputs / layout, cited documents, eval suite v54, GM-179, GM-212,
  GM-292, GM-302, the `test_mse_*_adapter` files, render tables, Foundry
  usability, top-level params): **671 passed**.
- New: `retaining_walls/tests/test_gm307_battered_face.py` (9) and
  `funhouse_agent/tests/test_gm307_mse_package.py` (21).
- Changed: GM-114's two refusal tests became "taken since GM-307";
  `test_mse_sloping_backfill.py`'s calc-steps refusal test became a test
  that the steps print the result's Kab, h, F_T; the tool-parity "dropped
  input" test now uses the vibro package's `N1_target` (GM-114's face batter
  was its example). `method_probes.py`: `face_batter` on the MSE probes is
  marked as not moving the result (the probe's step stays under 10°).
- `test_page_fingerprints.py` passed unchanged: no page's recorded prompt
  or tool list changed, so nothing was re-recorded.
- E1–E4 reproductions (V-056 / V-057, GM-114's E4 LRFD) pass.

## Noted, not fixed

- The LRFD package's steps print the tool's per-combination results; the
  tool does not expose its factored V, M and P_d per combination, so the
  steps give the equations and factors without a numeric substitution of
  the factored loads. Exposing them is a refactor of
  `check_external_stability_lrfd`; no saved output changes.
- `mse_lrfd_external_stability` (and `lrfd_external`) still take a vertical
  face only; GEC-11 prints no LRFD worked case for a batter.
- A battered face under a broken backslope stays refused (β_cs undefined).
- On a battered face the section figure's zone labels are placed as for a
  vertical face (cosmetic).

## Release note

- New: the MSE calc package (`retaining_wall_package`, `wall_type="mse"`)
  takes a sloping or broken backslope and prints GEC-11 Vol I Sec. 4.4.4's
  terms (Kab by Eqs. 4-2, 4-3, h and F_T by Eq. 4-7, V2, S_eq and σ_2) with
  the mse_wall tool's numbers; no value changes.
- New: `mse_wall` and the MSE package take `face_batter`. From 10° from
  vertical the wall is battered (GEC-11 Vol I p. 4-15, Fig. 4-5): Coulomb
  Kab with θ = 90 + ω and the thrust at δ − ω, the internal Ka of Eq. 4-26
  (taken with its bracket squared; the print drops the square), and for
  geosynthetics the Coulomb failure plane of Fig. 4-9(b); under 10° the
  face is vertical. A battered face under a broken backslope, or with
  `lrfd_external`, is refused. Vertical walls do not change.
- New: an LRFD MSE package, `retaining_wall_package` with
  `wall_type="mse_lrfd"`: `mse_lrfd_external_stability`'s CDRs for Strength
  I max and min, the critical combination and Service I, with the load and
  resistance factors of GEC-11 Vol I Tables 4-1, 4-2 and 4-5; no value
  changes.
