# GM-302: MSE wall checks GEC-11 prints but the module did not report

Checked against FHWA GEC-11, *Design and Construction of Mechanically
Stabilized Earth Walls and Reinforced Soil Slopes* (FHWA-NHI-10-024 Vol I
and NHI-10-025 Vol II, November 2009), read on the rendered pages
(`geotech-references/docs/GEC 11 Vol 1.pdf`, `GEC 11 VOL 2.pdf`).

Scope:
- `retaining_walls/mse.py` and `results.py`;
- the MSE steps of `calc_steps.py`;
- the MSE text of `funhouse_agent/adapters/retaining_walls.py`;
- `retaining_walls/DESIGN.md`.

For each item: the page, whether it is a design check (a verdict) or advice
(a warning), and whether GEC-11 gives it on the ASD path, the LRFD path or
both. The module's ASD path (`analyze_mse_wall`, `mse_wall`) follows
GEC-11's own checks with factors of safety; its LRFD path
(`check_external_stability_lrfd`, `mse_lrfd_external_stability`) follows
the LRFD worked examples. Pinned in `retaining_walls/tests/test_gm302_checks.py`
and `funhouse_agent/tests/test_gm302_mse_checks_adapter.py`.

## C1. Service I eccentricity, L/6. A verdict; LRFD

Printed in Vol II only, in the external stability tables of the LRFD
examples:
- E3 Table E3-6.3 (p. E3-11);
- E4 Table E4-6.3 (p. E4-9);
- E5 Tables E5-7.1 (p. E5-17) and E5-7.3 (p. E5-20).

Each prints the row "Limiting eccentricity, e = L/4 for strength limit
states and e = L/6 for service limit state" and then "Is the resultant
within limiting value of eL?", answered for Str I (max), Str I (min) and
Ser I. That is a check with a verdict. Vol I's text gives only the
strength limit (p. 4-23: emax = L/4 on soil, 3/8 L on rock).

Table E4-6.3's Ser I column: "Vertical load at base of MSE wall including
LL on top, ΣV = R = V1 + VS" 62.19 k/ft, MRA 559.71, MOA 144.30 k-ft/ft,
eL 2.32 ft, limit 3.00 ft (L = 18 ft), "Yes". The live load is counted, as
in the module's Service I bearing resultant, which already worked eL out
(`eL_s`) and did not report it.

The module now reports `eccentricity.eL_service_m` against
`e_limit_service_m` (`passes_service`), `checks` "eccentricity (Service
I)"; `eccentricity.passes` is both limit states' (`passes_strength` is the
old verdict). `ecc_limit_ratio_service`, default 1/6, sets the limit.

Reproduced: E4 eL 2.32 ft against 3.00 ft; E3 (a 2H:1V slope that
continues, Table E3-6.3) eL 2.61 ft against 4.00 ft.

On a level backfill with GEC-11's factors it cannot fail unless the
critical Strength I check fails: eL_crit = (1.5 MF1 + 1.75 MF2) / V1 against
L/4, and eL_s = (MF1 + MF2) / (V1 + Vs) against L/6, so 1.5 eL_s <= eL_crit.
It can govern with lighter EH or LL factors, and on a sloping backfill.

Not given:
- Rock. Every example is on soil. With a rock `ecc_limit_ratio` (above
  0.25) and the service limit left at L/6 the result warns that no page
  held prints the service limit on rock.
- The ASD path. Its source (AASHTO 2002 / FHWA NHI-00-043) is not held;
  the ASD overturning check stays the FOS of GEC-11's own text.

## C2. Grid transverse spacing St. With the empirical F*: a verdict above 24 in., a warning below 6 in.

Vol I p. 3-20: "For steel grid reinforcements with transverse spacing of
St > 6 inches (150 mm) (see Figure 3-4), F* is a function of a bearing or
embedment factor, Fq, applied over the contributing bearing αβ"
(Eqs. 3-6, 3-7).

Vol I p. 3-21: "St must be uniform throughout the length of the
reinforcement rather than having transverse grid members concentrated only
in the resistant zone. The maximum St distance is 24 in. (610 mm)."

Both bounds sit in pp. 3-20 and 3-21's paragraphs on the empirical F* for
grids (Eqs. 3-6, 3-7). With that F* (no `F_star_tested`) the maximum is a
stated requirement: a verdict, on the path that carries reinforcement (ASD
`analyze_mse_wall`, `mse_wall`; the LRFD external tool takes none).
`grid_spacing` gives the largest St of the levels' grids against
0.6096 m; `checks_failed` "grid spacing".

The 6 in. bound is the domain of the empirical F*, not a design limit.
Example E4 takes the empirical F* at St = 6 in. exactly (0.5 ft), so 6 in.
is inside. Below 6 in., with the empirical F*, the result warns and points
to tests.

Review fix: with F* from pullout tests the 24 in. sentence is about a
different F*, so an St over 24 in. warns ("not a check; the tests should be
on the grid at this spacing") and `grid_spacing` is absent. A grid given no
St (allowed since GM-297 only with F* from tests) is not checked and no
longer warns: St matters only to the empirical F*.

The built-ins carry Example E4's St: 6, 12 and 18 in.; all pass.

## C3. Vertical spacing Sv. A warning; ASD

Vol I p. 4-38 (Sec. 4.4.7.d): "to provide a coherent reinforced soil zone,
vertical spacing of reinforcement should not exceed 32 in. (800 mm)."

"Should" is advice. Vol I p. 3-48: "Although 36-in. reinforcement spacing
has been used successfully on many projects, it is not in agreement with
the 32-in. limit to ensure a coherent MSE mass"; p. 4-39: gabions at 3 ft,
"This spacing slightly exceeds the limit noted above, but this may be
offset by the contributions of the large facing unit". So it is a warning,
not a verdict.

The module warns where Sv (`reinforcement_spacing`), or the largest gap
between the `level_depths` given, is over 32 in. (0.8128 m). ASD path only:
the LRFD external tool has no levels.

## C4. Friction angles. Warnings and the left-out angles' 30°; both paths

Vol I p. 4-10 (and p. 3-21 for the 34°):
- "Per Article 11.10.6.2 (AASHTO, 2007) the maximum friction angle of the
  select granular reinforced fill should be assumed to be 34°, unless the
  project specific fill is tested for frictional strength by triaxial or
  direct shear testing methods. A design friction angle greater than 40°
  should not used, even if the measured friction angle is greater than
  40°."
- "For the foundation soil, Article 11.10.5.3 (AASHTO, 2007) notes that in
  absence of specific data, a maximum friction angle, φ'f of 30° may be
  used. The use of an assumed, non-specific parameter is recommended only
  for preliminary sizing."
- "An assumed friction angle, φ'b of 30° is often used for the retained
  (i.e., behind the reinforced zone) backfill. The use of an assumed,
  non-specific parameter is recommended only for preliminary sizing."

The reinforced fill's limits are advice on what to assume; the module
cannot tell a tested angle from an assumed one. So they are warnings:
- `phi_backfill` above 34 deg: "Use it only from such tests".
- Above 40 deg: the 40° sentence.

The foundation's and the retained fill's 30° are the values the page gives
where there is no data. Review fix: a left-out `phi_foundation` or
`phi_retained` now takes 30 deg (`PHI_ASSUMED`) on every MSE path:
`analyze_mse_wall` and both tools, `check_external_stability_lrfd`,
`check_external_stability`'s retained fill, and the calc steps' terms.
Before, both took `phi_backfill`, which raised the resistance and lowered
the thrust behind the zone. Each left-out angle is a choice ("30 deg,
assumed for preliminary sizing", how "default", the p. 4-10 sentence as
its basis) and the tools' `defaults_used` gives the same text. The first
build's two warnings for them are gone. `gamma_foundation` without
`phi_foundation` is still refused: qn takes the foundation soil's own
angle. A backfill slope steeper than a left-out `phi_retained`'s 30° is
refused with the reason (GEC-11 Eq. 4-3 has no real Kab), saying the 30°
was p. 4-10's default and asking for `phi_retained`; with `phi_retained`
given the refusal keeps its wording (second review).

## C5. Local shear: Eq. 4-24. A verdict; both paths

Vol I p. 4-30, "Local Shear, Punching Shear And Lateral Squeeze" (the
equation read on the rendered page): "To prevent local shear of structures bearing on weak cohesive
soils it is required that: γr H ≤ 3 cu (Equation 4-24), where γr is the
nominal unit weight of the reinforced fill, H is the height of the wall and
cu is the nominal total stress cohesion of the foundation soil."

"Required" makes it a verdict. Its quantities are nominal (unfactored), so
it applies to both paths.

It runs where c_foundation is cu: the undrained sliding rule
(`sliding_cohesion` 'undrained', the default for phi_foundation 0), the
module's one marker of a cohesive foundation strength. On a drained
foundation it does not apply and is not listed.
- `local_shear` gives gamma_r_H_kPa, three_cu_kPa and passes.
- `checks_failed` "local shear"; the verdict prints both numbers.
- On the ASD path it counts in `passes_external`.

Lateral squeeze, "a special case of local shear that can occur when bearing
on a weak cohesive soil layer overlying a firm soil layer", has no check of
its own on the page beyond Eq. 4-24. Two-layer bearing is left to AASHTO
10.6.3.1.2 and NHI-06-088 (p. 4-31), not held for it.

## C6. Reduced strengths where local or punching shear is possible. An option; LRFD

Vol I p. 4-30: "Local shear is a transition between general shear and
punching shear, which can occur in loose or compressible soils, in weak
soils under slow (drained) loading. If local shear or punching shear
failure is possible, Section 10.6.3.1.2b of AASHTO (2007) requires the use
of reduced shear strength parameters for calculating the nominal bearing
resistance. The reduced effective stress cohesion, c* is set equal to
0.67c'. The reduced effective stress soil friction angle, φ* is set equal
to tan⁻¹(0.67 tan φ'f)."

Whether local shear is possible is the engineer's call from the soil. It
is an input, `local_shear_possible` (default False, general shear), on the
LRFD path's qn by Eq. 4-22 (`gamma_foundation`):
- qn takes c* and phi* (phi 30 deg gives 21.15 deg); sliding keeps the
  strengths given.
- `bearing` adds c_star_kPa and phi_star_deg.
- A choice, "foundation strengths in qn (Eq. 4-22)", states which
  strengths were taken, whenever qn is worked out. Review fix: its
  alternative gives the governing Strength I bearing CDR with the other
  strengths, min(qR / sigma_v) over Strength I max and the critical
  combination, each at its own L'. Left at general shear, the default, it
  shows what the reduced strengths would give. On a made-up 6 m wall (L
  4.5 m, phi_backfill 34, phi_f 30, no cohesion, gamma_f 18) the CDR is
  2.335 with the strengths given and 0.659 with c* and phi*.
- Refused without `gamma_foundation`: a given `bearing_resistance_strength`
  is used as given. Refused on the ASD path, whose bearing check takes
  `q_allowable`.

## Before and after: recorded inputs replayed

Recorded inputs:
- 446 calls recorded from the MSE tests (module functions `analyze_mse_wall`
  and `check_external_stability_lrfd`; tools `mse_wall`,
  `mse_lrfd_external_stability` and the calc package's wall route);
- the 6 MSE probes of `method_probes.json`;
- replayed on 7eef829 (before) and on this branch after the review fixes
  (452 calls; the same refusals on both, and 2 more after, below).

The first build moved no verdict. With the review's 30° defaults, 322 of
the 452 calls move: those that leave `phi_retained` or `phi_foundation`
out, nearly all over a 34° reinforced fill.
- Retained fill only: Ka 0.283 to 0.333, so FOS_sliding and
  FOS_overturning fall by 15 % (x 0.848) and sigma_v rises 3 to 5 %. For
  example, a 6 m wall, L 4.2 m, phi_f 28: FOS_sliding 2.344 to 1.988 (the
  lowest sheet governs), FOS_overturning 5.20 to 4.41, sigma_v 141.1 to
  147.4 kPa.
- Foundation left out as well: the base's sliding falls by 27 % (x 0.726,
  tan 30 / tan 34 with the Ka ratio).
- A sloping backfill: Kab rises by about 20 % (0.307 to 0.367 at 2H:1V).
- Under a reinforced fill weaker than 30° the retained fill is now the
  stronger, and FOS rises by up to 22 % (noted, not fixed).

Verdicts, 21 calls:
- 10 calls that failed no check now fail one:
  - GM-297's cu 40 wall (6 m, L 4.2 m, phi_f 0): ASD FOS_sliding 1.738 to
    1.474 (3 calls), LRFD CDR 1.158 to 0.982 (3 calls);
  - a 2H:1V slope, L 5 m: LRFD sliding CDR 1.078 to 0.870; with a 12 kPa
    surcharge, eL_crit 1.050 to 1.258 m against L/4 1.25 m;
  - a 6 m broken backslope with traffic: FOS_sliding 1.938 to 1.409 (tool
    and module).
- 9 calls already failing gain a failed check: local shear (1, the first
  build's), Service I eccentricity (6: the first build's 4 and 2 more on
  the broken backslope), sliding (2 slope walls, FOS 1.504 to 1.193).
- 2 calls are now refused: a 34° backslope over a 34° fill with
  `phi_retained` left out is steeper than the retained fill's 30° (GEC-11
  Eq. 4-3); before, FOS_sliding 1.239. One refusal (a 35° slope, the
  angle left out) now names the 30° as p. 4-10's default and asks for
  `phi_retained`.

GM-297's cases 3 to 8 (its CHECK.md, "What moved"), now with the retained
fill at 30°, at the base:

| Case | GM-297 after | Now |
|---|---|---|
| 3. ASD sliding, φf 28, c 10 | 2.633 | 2.233 |
| 4. ASD sliding, φf 0, cu 40 | 1.738, passes | 1.474, **fails** |
| 5. ASD sliding, φf 0, cu 120 | 3.340 | 2.833 |
| 6. LRFD sliding, φf 28, c 10 | CDR 1.755 | 1.489 |
| 7. LRFD sliding, φf 0, cu 40 | CDR 1.158, passes | 0.982, **fails** |
| 8. LRFD sliding, slope 15°, φf 28, c 10 | CDR 1.425 | 1.211 |

New output:
- Warnings: the reinforced fill above 34° or 40°, 2 grid St of 150 mm
  (5.9 in.) and 2 vertical gaps of 2 m.
- The left-out angles' choices; `grid_spacing` and `local_shear` where they
  apply; the LRFD eccentricity's new keys in every LRFD result; with
  `gamma_foundation`, the strengths choice with the other strengths' CDR.

Release note: the MSE tools also report the Service I eccentricity (L/6),
GEC-11 Eq. 4-24 on a cohesive foundation and, with the empirical F*, a
steel grid's maximum St (24 in.); they warn of GEC-11's advice on the
reinforced fill's friction angle and the vertical spacing. A foundation or
retained-fill friction angle left out is now 30°, as GEC-11 p. 4-10 gives
for no data (it was the reinforced fill's): walls that leave them out
over a 34° fill show sliding and overturning about 15 % lower (27 % for
sliding when the foundation's is left out too), and 10 of 452 recorded
calls go from pass to fail.

## Gates

On this branch, each by its exit code:
- retaining_walls/tests, validation_examples V-009 to V-011 and V-056/V-057,
  the MSE adapter tests, test_gm292/gm212 adapter tests, test_citations,
  test_choices_stated_defaults and test_adapter_ergonomics: 907 passed;
- test_calc_package_tool_parity, test_calc_package_adapter and
  test_render_tables: 277 passed;
- calc_package/tests and the other calc package tests,
  test_gm179_unused_inputs, test_top_level_params, test_eval_suite_v54,
  test_tool_usability_foundry_eval, test_cited_documents and
  test_cited_pages, `-k "mse or MSE or wall or retaining or reinforc"`: 19
  passed;
- test_defaults_used `-k "mse or wall or retaining or citation"`: 108
  passed;
- webapp test_public_text and test_page_fingerprints, after the commit.

Tests changed for the new output:
- test_retaining_walls and test_gm179_unused_inputs: the strengths choice
  follows the qR choice, then a left-out phi_retained's choice;
  `TestMSERetainedFill.test_default_is_30_deg` replaces the phi_backfill
  default's test;
- test_mse_backslope_adapter: phi_retained left out is 30 deg in
  `defaults_used` and a choice; the slope-steeper-than-phi refusal is
  against the retained fill's 30° unless phi_retained is given;
- test_gm297_print and test_mse_sliding_cohesion_adapter: GM-297's sheet
  and cu cases leave phi_retained out, so their values are worked again
  by hand at 30° (Ka 1/3, Pa 114.0 kN/m on the 6 m wall);
- test_mse_live_load and test_retaining_walls' min-phi test: the hand
  checks take one Ka, so they give `phi_retained` as the reinforced
  fill's;
- test_citations: Eq. 4-24 added to `REFERENCE_GAPS`. The library's
  chapter text leaves its number out; it is printed on Vol I p. 4-30.

## Noted, not fixed

- The Service I limit on rock: not printed in anything held. It warns and
  can be set (`ecc_limit_ratio_service`).
- An ASD eccentricity check: its source (AASHTO 2002 / NHI-00-043) is not
  held.
- Lateral squeeze and two-layer bearing beyond Eq. 4-24: AASHTO
  10.6.3.1.2, not held.
- St "must be uniform throughout the length": the module has no input for
  a grid's St along its length.
- A reinforced fill weaker than 30° with `phi_retained` left out: the
  retained fill takes 30° and is the stronger of the two. p. 4-10's 30° is
  "often used"; whether to take min(30, phi_backfill) is not printed.
- The reference library's GEC-11 chapter text leaves out Eq. 4-24's number
  (as it does Eqs. 4-21 to 4-23); for the library to fill.
