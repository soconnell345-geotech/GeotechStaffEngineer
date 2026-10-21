# retaining_walls — Cantilever & MSE Wall Design

## Purpose
Stability analysis for cast-in-place cantilever retaining walls and
mechanically stabilized earth (MSE) walls per GEC-11.

## References
- FHWA GEC-11 (Mechanically Stabilized Earth Walls)
- AASHTO LRFD Bridge Design Specifications
- Coulomb/Rankine earth pressure theory
- FHWA NHI-06-089, Soils and Foundations Reference Manual Vol II (2006),
  Ch. 10 (CIP semi-gravity walls)
- UFC 3-220-20 (2025), App. B-1 (cantilever cut wall example)

## Files
- `geometry.py` — CantileverWallGeometry (auto-sizing), MSEWallGeometry
- `earth_pressure.py` — reuses sheet_pile Ka/Kp + resultant force computation
- `cantilever.py` — sliding, overturning, bearing, eccentricity checks
- `mse.py` — Kr/Ka ratio, Tmax, pullout, F*, external + internal stability,
  level or sloping backfill (`backslope_geometry`, `backslope_loads`)
- `reinforcement.py` — Reinforcement dataclass + built-in products
- `results.py` — CantileverWallResult, MSEWallResult

## Public API
```python
analyze_cantilever_wall(geometry, soil_params, ...) -> CantileverWallResult
analyze_mse_wall(geometry, reinforcement, soil_params, ...) -> MSEWallResult
check_external_stability_lrfd(geometry, soil_params, factors, ...) -> dict  # CDRs
list_reinforcement() -> list of built-in reinforcement products
```

## MSE external stability: ASD FOS (default) vs LRFD CDRs (opt-in)
Two external-stability paths are available; both are default-preserving.
- **ASD** — `check_external_stability` / `analyze_mse_wall` return unfactored
  factors of safety (resistance/demand). This is the historical default. Since
  GM-164 its live load over the reinforced zone is left out of the resisting
  weight in sliding and overturning and counted in bearing, as on the LRFD
  path below (see "A live load over the MSE reinforced zone").
- **LRFD** — `check_external_stability_lrfd` returns the AASHTO/GEC-11
  capacity:demand ratios (CDRs). `analyze_mse_wall(lrfd_external=True)` also runs
  it and attaches the CDR set to `MSEWallResult.external_lrfd` (None otherwise).
  The load-side bookkeeping is built in:
  - Load components about the toe: EV reinforced-mass weight V1 (arm L/2), LL
    vertical surcharge Vs (L/2), EH active thrust F1 (arm H/3), LL surcharge
    thrust F2 (arm H/2), where F1/F2 use the retained-fill Ka.
  - **Sliding** and **eccentricity/overturning** EXCLUDE the LL surcharge from
    the resisting side (unconservative to count a transient stabilizing weight);
    **bearing** INCLUDES it (it raises the bearing stress).
  - Load combinations: Strength I max (EV 1.35 / EH 1.50 / LL 1.75), Strength I
    min (EV 1.00 / EH 0.90 / LL 1.75), and Service I (all 1.0). The **critical**
    ("min-vertical") combination pairs γEV,min on all vertical gravity loads
    (least resistance / largest eccentricity) with the max horizontal/driving
    factors — this governs sliding (CDR 1.37) and eccentricity (eL) in E4.
  - Sliding CDR = φ_sl·(γEV·V1·tanδ + 2/3·c·L)/(γEH·F1 + γLL·F2), δ =
    min(φ_r, φ_fd), φ_sl = 1.0. Eccentricity check: eL = L/2 − (ΣMr − ΣMo)/ΣV
    vs e_max = L/4 (soil). Bearing: σ_v = ΣV/(L − 2eL) vs the factored bearing
    resistance qR (strength) or the service bearing resistance (Service I).
  - All load/resistance factors and the eccentricity ratio are parameters with
    the GEC-11/AASHTO defaults above. Validated vs GEC-11 Example E4
    (FHWA-NHI-10-025) in `validation_examples/test_published_v009_v011.py`:
    sliding 1.85/2.08/1.37, eL 2.87/3.87 ft (< L/4 = 4.50), bearing σ_v 6.70 ksf
    / CDR 1.57, Service σ_v 4.66 ksf — all reproduced through the high-level path.
  - **Bearing resistance worked out (GM-179, 2026-10-19).** Given
    `gamma_foundation` in place of `bearing_resistance_strength`, qR is worked
    out for each Strength I combination: qn = c_f·Nc + 0.5·L'·γ_f·Nγ (GEC-11
    Vol I Eq. 4-22, "for a level grade in front of a MSE wall and no
    groundwater influence", the embedment neglected, p. 4-28) over that
    combination's L' = L − 2eB (L' = L when eB < 0); Nc, Nγ from Table 4-6
    by its own equations (Prandtl Nc, Vesic Nγ; within 0.1 of the table);
    qR = φb·qn, φb = 0.65 (Table 4-5, Eq. 4-23). Reproduces Vol II Example E2
    on Example E1's wall: L' = 12.46 ft, qn = 17.44 ksf, qR = 11.34 ksf
    (`test_v057_e2_nominal_bearing_resistance`). Service I keeps its given
    resistance (a settlement criterion). Giving both is refused, and so is
    `gamma_foundation` without `phi_foundation`: the factors are the
    foundation soil's, never the reinforced fill's `phi_backfill`. The
    result's `choices` state φb (given or default), the Eq. 4-22 basis and
    the other way (a given `bearing_resistance_strength`, e.g. qn by AASHTO
    10.6.3.1.2 for a slope in front of the wall or high water). The ASD
    `mse_wall` tool refuses `gamma_foundation`: its bearing check compares
    with `q_allowable` and it counts no passive resistance, so the input
    changed nothing (GM-184); the library function keeps it and passes it to
    the LRFD path only.

## Wall height, checks not run and failed checks (GM-100, 2026-10-15)

**Wall height.** `CantileverWallGeometry.wall_height` is the TOTAL height H,
from the underside of the base slab (footing) to the top of the stem:
H = base_thickness + stem height. The backfill surface meets the stem at its
top, and the active thrust acts on the vertical plane through the back of the
heel over H (plus the rise of a sloping backfill over the heel). A wall that
retains 4.0 m above the top of a 0.5 m footing is H = 4.5 m. The adapter's
`wall_height` description says this, and its `geometry` block returns the
stem height, base thickness, toe and heel actually analysed. (In live wave
w8 a model read "H" as the retained height, and a figure labelled the 4.5 m
pressure diagram against a 4.0 m retained height.)

**A check that was not run is never reported as a pass.** The cantilever
bearing check compares the larger edge pressure, max(q_toe, q_heel), with
the allowable bearing pressure `q_allowable`. Before this fix, with no
`q_allowable` the module returned `FOS_bearing` 99.9 and `passes_bearing`
True and the calc package printed "OK" for a check it never made. Now:

- `FOS_bearing` and `passes_bearing` are None, `bearing_checked` is False,
  and `checks_not_run` is `["bearing"]`;
- `summary()` says "Bearing: not checked (no q_allowable given); the bearing
  pressures q_toe / q_heel and the middle-third check are still reported";
- the calc package says "NOT CHECKED" in the input echo, the bearing section
  and the summary table;
- `all_checks_pass` is True only when every check ran and passed, False when
  one failed, and None when none failed but one was not run.

The MSE wall does the same for its external bearing check
(`passes_external` is None when sliding and overturning pass but bearing was
not checked), and the LRFD external check lists `bearing (Strength I)` /
`bearing (Service I)` in `checks_not_run` when their resistances are not
given. When a Service I bearing resistance IS given, its CDR now counts in
`bearing.passes` and the overall `passes` (every load combination must pass);
before, a Service I CDR below 1 was reported but never failed anything.

**The demand is the larger edge pressure.** The bearing FOS used to be
q_allowable / q_toe. When the resultant lies on the heel side of the centre
the larger pressure is under the heel, so the check now uses
max(q_toe, q_heel). When the resultant falls outside the base (the wall tips
over the toe; q_toe and q_heel both print 0) the bearing check fails
(FOS 0) instead of returning the old 99.9.

**No driving force is not a number.** When the active thrust is zero (backfill
cohesion cancels it) nothing pushes or tips the wall: sliding and overturning
pass, their FOS is None, and the result says "no driving force (the active
thrust is zero)" instead of a 99.9.

**Failed checks are named where the agent reads the result.** Every result
carries `verdict` (the failed checks first, with their numbers, then those
that pass, then those not checked), `checks_failed` (sliding, overturning,
bearing, eccentricity = resultant outside the middle third; for MSE also
internal) and `checks_not_run`; `to_dict()` puts them first, and the
`retaining_wall_package` response carries them too. `summary()` ends with
"Failed checks: ..." and "Not checked: ...". In live wave w8 a wall failed
sliding (FOS 1.17 < 1.5) and the user never learnt it.

## MSE walls with a sloping backfill (GM-112, 2026-10-16)

Until GM-112 both MSE checks computed a level-backfill Rankine Ka only, and
`mse_wall` accepted `backfill_slope` and gave the same answer at 0 and 26.6
degrees (GM-110 then refused it). `MSEWallGeometry.backfill_slope` (beta,
degrees from horizontal, rising away from the face) and
`broken_backslope_height` (S, the height of the slope crest above the top of
the wall) now drive the GEC-11 method in `check_external_stability`,
`check_external_stability_lrfd` and `check_internal_stability`. With beta = 0
every result is byte-identical to before (checked on 868 level outputs:
module, adapter, summaries).

**Source.** FHWA GEC-11 Vol I, FHWA-NHI-10-024 (`geotech-references/docs/GEC
11 Vol 1.pdf`), read from the PDF; page = printed page (PDF page).

External loads (`backslope_loads`):

- Eq. 4-2 / 4-3, p. 4-15 (PDF 157), "Vertical Wall and a Surcharge Slope":
  Kab = sin²(θ + φ'b) / [Γ sin²θ sin(θ − δ)],
  Γ = [1 + √( sin(φ'b + δ) sin(φ'b − β) / (sin(θ − δ) sin(θ + β)) )]²,
  "δ = Angle of friction between retained backfill and reinforced soil, set
  equal to β", "θ = 90° for vertical, or near (< 10°) vertical, wall". This
  is the repo's `coulomb_Ka(phi, delta, alpha=θ, beta)` with δ = β. "Note
  that the earth pressure force, (FT) in Figure 4-3, is oriented at the same
  angle as the backslope, β, as it is assumed that δ = β."
- Broken backslope, same page: "computed using Equations 4-2 and 4-3, with
  the design β angle and the interface angle δ both set equal to I, as
  defined in Figure 4-4". Fig. 4-4, p. 4-14 (PDF 156): I is the angle from
  the top of the wall to the ground surface 2H behind the face, so
  I = atan(min(S, 2H tanβ) / 2H) (Example E1: I = arctan(9/40) = 12.7°).
- Eq. 4-7, p. 4-20 (PDF 162): FT = ½ Kab γb h², "h = total height of wall,
  H, and slope at the back of the reinforced zone = H + L tan β". "For a
  broken backslope (see Figure 4-4), h - H should not exceed the height of
  the upper crest ... use (H + S) if (L tanβ) > S."
- Figs. 4-3 (p. 4-13, PDF 155) and 4-8 (p. 4-25, PDF 167): V1 = γr H L at
  L/2; V2 = γr L (h − H) / 2 at 2L/3 from the face (L/6 behind the centre);
  FT at h/3. The module takes V2 as the reinforced-fill weight of the slope
  fill over the zone (γr, as the figures do); FT uses the retained fill
  (`phi_retained`, `gamma_retained`, defaulting to the reinforced fill).

Checks:

- Sliding, p. 4-21 (PDF 163): Eq. 4-10 FH = FT cos β; Eq. 4-11
  Pd = γEH FT cos β; Eq. 4-13 Rr = [γEV (V1 + V2) + γEH (FT sin β)] × μ.
- Eccentricity, Eq. 4-16, p. 4-23 (PDF 165):
  e = [γEH-MAX FT cosβ (h/3) − γEH-MAX FT sinβ (L/2) − γEV-MIN V2 (L/6)] /
  [γEV-MIN V1 + γEV-MIN V2 + γEH-MAX FT sinβ] (moments about the centre; the
  module sums them about the toe, the same thing).
- Bearing, Eq. 4-21, p. 4-28 (PDF 170):
  qV-F = [γEV-MAX V1 + γEV-MAX V2 + γEH-MAX FT sinβ] / (L − 2eB), and "(L −
  2eB) is set equal to L when the value of eccentricity is negative". The
  sloping paths apply that rule; the level paths are unchanged (their e is
  never negative).
- LRFD combinations as for the level wall (Strength I max, Strength I min,
  the critical EV-min / EH-max one, Service I), with FT's two components
  carrying the same EH factor, "to maintain consistency between the total
  inclined lateral force and its components" (Vol II Example E3, notes to
  Tables E3-6.1 to E3-6.3). The ASD check (`check_external_stability`,
  `mse_wall`) is the same statics unfactored: FOS_sliding = (V1 + V2 + FT
  sinβ) tan δb / FT cosβ, FOS_overturning = resisting / overturning moments
  about the toe, σv over L − 2e.

Internal stability (`check_internal_stability`):

- Eq. 4-25, p. 4-36 (PDF 178): Kr from Ka = tan²(45 − φ'r/2), "assuming no
  wall friction and a β angle equal to zero" — Ka is not changed by the
  slope. Same page: "The value of Seq should not exceed the slope height for
  broken back sloping fills. A reinforcement length of 0.7H is used to
  compute the sloping backfill stress, σ2".
- Fig. 4-11, p. 4-38 (PDF 180): Seq = (½) 0.7H tan β, σ2 = S γr; Eq. 4-29,
  p. 4-42 (PDF 184): σH = Kr [γr (Z + Seq) γEV-MAX] (unfactored here).
- Fig. 4-9(a), p. 4-34 (PDF 176), inextensible: H1 = H + tanβ × 0.3H /
  (1 − 0.3 tanβ); La = 0.3 H1 over the upper H1/2 (measured from the top of
  H1), tapering to the toe below. For a broken backslope the rise H1 − H is
  not more than S. Fig. 4-9(b), extensible, vertical face: ψ = 45 + φ/2, La
  unchanged.
- Fig. 4-15, p. 4-50 (PDF 192): σV-PO = γr Zp; Vol II Fig. E3-5 (p. E3-12,
  PDF 284): "Use average Zp over the resistance zone ... Zp-ave = Z +
  0.5 tanβ (La + L)". The module averages the surface rise over the
  resistant zone (the same for a slope that continues; for a broken
  backslope the rise is capped at S).

**Traffic is left out of pullout, level backfill too (GM-112 review,
2026-10-16).** GEC-11 p. 4-49 (PDF 191) defines the pullout σv as "the
unfactored vertical stress at the reinforcement level in the resistant
zone, including distributed dead load surcharges, neglecting traffic
loads"; p. 4-51 (PDF 193) and AASHTO C11.10.6.2.1 say the same. The
module's one surcharge, `MSEWallGeometry.surcharge`, is the live load (the
LRFD check factors it with γLL; there is no separate dead-load surcharge),
so `check_internal_stability` keeps it in Tmax (Eq. 4-30) and no longer
adds it to the pullout σv. Before, it did, on the unsafe side: on a 6 m wall
(L 5 m, Sv 0.6 m, γ 19, φ 34°, geogrid UX1600) with q = 12 kPa, the top
level's (z = 0.3 m) pullout resistance is 10.15 kN/m (FOS 3.38), where it
was 31.50 (FOS 10.49); at z = 5.7 m 473.8 where it was 526.3.
`pullout_resistance`'s `q_surcharge` remains, for a dead-load surcharge.

**The traffic thrust's vertical component is an option.** Behind a broken
backslope the traffic gives F2 = Kab q h at h/2, inclined at I (Example
E1). Whether F2 sin I resists sliding and eccentricity is where GEC-11
differs from itself: the general text includes "External loads that
increase sliding resistance ... only if those loads are permanent" (Vol I
p. 4-22, PDF 164), while Example E1 counts it with the LL factor (Vol II pp.
E1-7 to E1-9, PDF 249-251). `traffic_vertical_resists` (default False,
the general text; True, Example E1) chooses, on `check_external_stability`,
`check_external_stability_lrfd`, `analyze_mse_wall` and both adapter
methods; bearing counts it either way, since "the bearing check applies
live load above both the reinforced zone and the retained backfill" (p.
4-26, PDF 168). The result's `backslope.traffic_basis` says which was used,
with the page.

What is refused, with the reason (ValueError, surfaced by dispatch):

- a slope steeper than the retained fill's friction angle: sin(φ − β) < 0
  under the root of Eq. 4-3, so Kab has no real value, and the fill slope
  does not stand (at β = φ, Γ = 1 and the check runs);
- a surcharge on a slope that continues: GEC-11 gives that case "and no
  surcharges" (§4.4.6.b, p. 4-23). Traffic behind the crest of a broken
  backslope is taken as Example E1 does, with the option above;
- the MSE calculation steps (`calc_steps._calc_steps_mse`) refuse a sloping
  backfill: they re-derive a level backfill (and there is no MSE calc
  package on the agent surface).

Beyond the cases GEC-11 draws (the module's extension, stated here): a
broken-backslope crest inside the reinforced length L. V2 is then the actual
fill over the zone (the triangle to the crest and the level part behind it,
at its centroid), and the traffic over the level part is Vs = q (L − x_crest)
— excluded from the sliding and eccentricity resistance and included in
bearing, as the level wall treats its live load (Fig. 4-2) — and counts in
Tmax (not in pullout) as the level wall's surcharge does. As the crest height
goes to zero every result tends to the level wall with its surcharge
(tested). A battered face (10° or more) is still not covered.

**Worked examples reproduced** (V-056 / V-057,
`validation_examples/test_published_v056_v057_gec11_slope.py`), FHWA GEC-11
Vol II (FHWA-NHI-10-025) Appendix E:

| Example E3, 2H:1V slope that continues | published | module |
|---|---|---|
| Kaf | 0.537 | 0.5367 |
| sliding CDR, Str I max / min / critical | 1.35 / 1.60 / 1.07 | 1.349 / 1.598 / 1.074 |
| eL, Str I max / min / critical (ft) | 2.90 / 2.35 / 3.81 | 2.90 / 2.34 / 3.81 |
| σv, Str I max / min / Service / critical (ksf) | 10.19 / 6.83 / 7.16 / 9.02 | 10.19 / 6.83 / 7.16 / 9.02 |
| bearing CDR, Str I max / Service / critical | 1.03 / 1.05 / 1.16 | 1.03 / 1.047 / 1.164 |
| Level 4 (Z = 8.75 ft): σH, La, Le, Zp-ave | 0.99 ksf, 10.59, 13.41, 17.40 ft | 0.99, 10.60, 13.42, 17.39 |

| Example E1, broken backslope (S = 9 ft) + traffic | published | module |
|---|---|---|
| I, Kab | 12.7°, 0.360 | 12.68°, 0.3604 |
| sliding CDR (critical) | 1.12 | 1.119 |
| e (critical), eB (Str I max) (ft) | 3.80, 2.77 | 3.80, 2.77 |
| σV-F, CDR | 6.55 ksf, 1.60 | 6.56, 1.601 |
| the default (FV2 not resisting, Vol I p. 4-22): sliding CDR, e | — (by hand from E1's loads: 1.10, 4.01 ft) | 1.101, 4.01 ft |

The E1 rows above are with E1's choice, `traffic_vertical_resists=True`.

The ASD path is the same statics with every factor 1.0: on E3 it gives the
Service I column (e 2.61 ft, σv 7.16 ksf).

## The terms each cantilever check used (GM-143, 2026-10-18)

`analyze_cantilever_wall` attaches, unrounded: `earth_pressure` (method,
Ka, H_active, Pa_earth / Pa_surcharge / Pa_cohesion, Pa, z_Pa, delta, Ph,
Pv; `earth_pressure.active_force_components` is the arithmetic of
`horizontal_force_active`), `vertical_forces` (each weight, the heel
surcharge and Pv with its arm from the toe and moment, in the order the
checks sum them), `sliding_terms`, `overturning_terms` and `bearing_terms`
(V, M_stab, M_over, x_R, e, distribution, B_eff, q_toe, q_heel, q_max).
`check_sliding` / `check_overturning` / `check_bearing` return the same
rounded dicts as before, built from these terms. The calc package prints
them (a result built by hand carries none; the steps then call the same
term functions). What the printout gained: the stem arm the module uses
(t_toe + t_stem_top / 2; the steps had their own), totals summed exactly,
q_heel = 2V / (3 (B - x_R)) for a resultant beyond the middle third on the
heel side, the signed e in the trapezoid, and P_a with cohesion on the
section figure. No result number changed.

## A surcharge over the cantilever heel (GM-157, 2026-10-18)

Until GM-157 `cantilever.py` counted the surcharge over the heel (q times
the heel length, at the middle of the heel) as a resisting vertical force
in sliding and overturning, and in the middle-third check. Live wave w12
F55 (H 4.5 m, B 2.8 m, toe 0.7 m, stem 0.35 m, base 0.5 m, gamma 19, phi
32, q 10 kPa) gave sliding 1.17 and overturning 3.06 with it, 1.07 and
2.78 without.

**Sources, read from the PDFs (page = printed page (PDF page)).**

- FHWA NHI-06-089 Vol II (`geotech-references/docs/FHWA NHI-06-089 Soils
  and Foundations Vol 2.pdf`). Sec. 10.4.2, p. 10-31 (PDF 321): surcharges
  are "(1) dead load surcharges such as that resulting from the weight of a
  bridge approach slab or concrete pavement; (2) live load surcharges such
  as that due to traffic loadings". Sec. 10.5.3, p. 10-41 (PDF 331): "For
  CIP semi-gravity walls, the surcharge loads are generally assumed to be
  located behind the heel of the wall, and conservatively neglected within
  the width of the base slab since they contribute to overturning and
  sliding resistance. However, the surcharge loads within the width of the
  base slab are considered for the structural design of the wall stem."
  Sec. 10.5.5, p. 10-43 (PDF 333): "the live load surcharge is not
  considered as a stabilizing force over the heel of the wall when sliding
  resistance is being checked"; p. 10-44: the eccentricity limit e <= B/6
  on soil stands for the overturning check. Fig. 10-17, p. 10-39: W is the
  wall and the soil above the footing; no surcharge.
- UFC 3-220-20 (2025) (`ufc_3_220_20_2025.pdf`; no DM 7.2 PDF is in the
  library, this UFC replaces it), App. B-1, cantilever cut wall with a
  traffic surcharge. B-1.2, p. 597 (PDF 635): "For overturning and sliding
  checks, it is assumed that the traffic pressure does not act over the
  wall heel. For the check of bearing capacity, the traffic pressure is
  included. These assumptions increase the conservatism in the analyses."
  B-1.4, pp. 600-601: overturning judged by the resultant in the middle
  third, without the traffic over the heel. B-1.6, p. 602 (PDF 640):
  "Unlike the checks of overturning and sliding, it is conservative to
  include the surcharge over the heel of the wall, if this is possible to
  occur, in the bearing capacity check." The traffic thrust q Ka H is in
  every check.
- GEC-11 Vol I (MSE walls, AASHTO). p. 4-19 (PDF 161): "The live load
  surcharge is not considered as a stabilizing force when checking
  sliding". p. 4-22 (PDF 164): "External loads that increase sliding
  resistance should only be included if those loads are permanent"; "The
  eccentricity limit check only applies the live load above the retained
  backfill". p. 4-26 (PDF 168): "the bearing check applies live load above
  both the reinforced zone and the retained backfill", and "eB =
  eccentricity for bearing calculation (not equal to eccentricity check
  e)". pp. 4-4 / 4-5 (PDF 146-147): use minimum load factors "if permanent
  loads increase stability".

**What the module does.** `CantileverWallGeometry.surcharge` is the live
load (traffic); `dead_load_surcharge` (new, default 0) is a permanent one.
Both thrusts, Ka (q_live + q_dead) H at H/2, drive every check. Three
choices, each in the result's `choices` with its basis and the other
reading:

| Input | Default | Basis of the default | The other reading |
|---|---|---|---|
| `live_load_resists` | False: the live load over the heel, and its share of an inclined thrust's Pv, resist nothing in sliding, overturning and the eccentricity check | NHI-06-089 pp. 10-41 / 10-43; UFC 3-220-20 B-1.2; GEC-11 pp. 4-19 / 4-22 | True: the free body with the load over the heel (the module before GM-157); no reference read counts a live load so |
| `dead_load_resists` | False: neglected within the width of the base slab | NHI-06-089 p. 10-41, the CIP semi-gravity wall text | True: counted as a permanent load, GEC-11 p. 4-22 (AASHTO) |
| `surcharge_in_bearing` | True: both surcharges over the heel load the base in q_toe / q_heel / FOS_bearing | UFC 3-220-20 B-1.6; GEC-11 p. 4-26 | False: taken as located behind the heel (NHI-06-089 p. 10-41) |

The dead-load default is a judgment between two readings: NHI-06-089
speaks of CIP semi-gravity walls and neglects every surcharge over the
base slab; GEC-11 (MSE walls) allows a permanent load that increases
sliding resistance. The default follows the cantilever-wall text.

So the eccentricity check (the result's `eccentricity`,
`in_middle_third`, and `checks_failed` "eccentricity") is on the loads
the overturning check counts, and the bearing pressures on the bearing
check's loads; the result gives `eccentricity_bearing` (to_dict
`eccentricity_bearing_m`) when the two differ. `vertical_forces` marks
each force with `resists` and `bearing`; the calc steps print a "Counted
in" column, a total for each check and the eccentricity for bearing.

F55 after: sliding 1.07, overturning 2.78, e 0.338 m (was 0.268 m, still
within B/6 = 0.467 m), q_toe 122.4 / q_heel 33.1 kPa unchanged. Tests:
`tests/test_heel_surcharge.py` (a hand check under each reading).

`mse.check_external_stability` (the ASD path) still counted the surcharge
over the reinforced zone in sliding and overturning (W = gamma H L + q L),
against GEC-11 pp. 4-19 / 4-22; the LRFD path did not. GM-164 below.

## A live load over the MSE reinforced zone, ASD path (GM-164, 2026-10-18)

Until GM-164 `mse.check_external_stability` took W = gamma H L + q L in
sliding, overturning and bearing: the live load (traffic) over the
reinforced zone was a resisting weight. GEC-11 Vol I (FHWA-NHI-10-024)
reads otherwise, and the LRFD path already followed it:

- Fig. 4-2 (p. 4-12): the live load over the retained backfill only for
  limiting eccentricity and sliding; over both zones for bearing and
  global stability.
- p. 4-17: "Live loads are not used on specific design steps since they
  contribute to stability." p. 4-18: limiting eccentricity "(formerly
  known as overturning)".
- p. 4-19: "The live load surcharge is not considered as a stabilizing
  force when checking sliding". p. 4-21 (Eq. 4-12, R = gamma_EV V1 mu):
  "the live load is excluded since it increases sliding stability".
- p. 4-22: "External loads that increase sliding resistance should only
  be included if those loads are permanent"; "The eccentricity limit
  check only applies the live load above the retained backfill" (Eq. 4-15,
  p. 4-23, with V1 only).
- p. 4-26: "The bearing check applies live load above both the reinforced
  zone and the retained backfill"; "eB = eccentricity for bearing
  calculation (not equal to eccentricity check e)" (Eqs. 4-19 / 4-20,
  p. 4-27, with q L).
- Vol II App. F.1 (p. F-2): the ASD procedure was not updated; its users
  "may want to refer to current LRFD procedures for any updates which may
  also be applicable to ASD procedure designs".

**What the module does.** `MSEWallGeometry.surcharge` is the live load.
Its thrust, Ka q H at H/2, drives every check. Two choices, in the
result's `choices` with the basis and the other reading
(`LIVE_LOAD_BASIS`, `BEARING_BASIS`, `live_load_choices`):

| Input | Default | Basis of the default | The other reading |
|---|---|---|---|
| `live_load_resists` | False: q L (with a broken backslope, q over the level ground on the zone) resists nothing in sliding or overturning | GEC-11 pp. 4-19, 4-21 (Eq. 4-12), 4-22; Fig. 4-2 | True: the free body with the load over the zone, W = gamma H L + q L (the module before GM-164) |
| `surcharge_in_bearing` | True: it loads the base in sigma_v and e_B | GEC-11 p. 4-26; Eqs. 4-19 / 4-20 | False: taken as not over the zone |

The ASD path has no separate eccentricity (middle-third) check: the
overturning FOS stands for it and leaves the live load out. The result's
`eccentricity_m` stays e_B, the bearing eccentricity, on the bearing
loads. MSE geometry has no dead-load surcharge; a permanent load over the
zone (which p. 4-22 would count) is not modelled. The traffic thrust's
vertical component behind a broken backslope keeps its own choice,
`traffic_vertical_resists` (GM-112).

`_external_terms` returns the terms the checks sum: each vertical force
with `resists` and `bearing`, the thrusts, V / M_stabilizing, V_bearing /
M_bearing, Pa / M_overturning, delta_b, e_B, L_eff, sigma_v.
`MSEWallResult.external_terms` keeps them, and the MSE calc steps print
them (GM-143): a "Counted in" column, a total for each check, R with
delta_b = min(phi_r, phi_f) as the module uses (they printed 2/3 phi_f,
part of GM-114), K_a of the retained fill, and e_B.

Numbers (live load left out / before): 6 m wall, L 4.2 m, 18 kN/m3, phi
34/30, q 12 kPa: sliding 2.339 / 2.599, overturning 3.900 / 4.333,
sigma_v 156.0 kPa both. Example E4 inputs on the ASD path: sliding
2.104 / 2.268, overturning 3.594 / 3.875, sigma_v 223.0 kPa both. A
broken backslope with 3 m of level ground on a 5 m zone: sliding 2.428 /
2.561, overturning 4.346 / 4.667. A crest at or beyond L (Example E1) has
no live load on the zone and does not move. Tests:
`tests/test_mse_live_load.py`.

## The coverage ratio Rc, and F* and alpha from tests (GM-256, 2026-10-20)

**The print** (GEC-11 Vol I, FHWA-NHI-10-024, read on the pages). Eq. 3-1
(p. 3-12): Rc = b/Sh, b "the gross width of the strip, sheet, or grid"
(for a grid, centre to centre of the outside longitudinal bars, Fig. 3-3)
and Sh the "center-to-center horizontal spacing between strips, sheets, or
grids"; p. 3-13: "Rc = 1 in the case of continuous reinforcement". Eq. 3-2
(p. 3-16) is the pullout resistance per unit width of reinforcement; Eq.
4-35 (p. 4-49) checks it per unit width of wall, F* alpha sigma_v Le C Rc.
Eq. 3-11a (p. 3-22): Tal = Fy Ac / b, per unit reinforcement width; p.
4-47: the factored tensile resistance per unit width of wall is "the product
of the nominal long-term strength, coverage ratio, and applicable resistance
factor"; Eqs. 4-33 and 4-34 (pp. 4-48, 4-49): Tr = phi Tal >= Tmax.

**What the code took.** `Reinforcement.coverage_ratio` defaulted to 1.0 and
no route set it: the adapter had no spacing input, the built-ins left it at
1, and `width` (default 0.05 m) was never used. So every steel strip and
bar mat was checked as a continuous sheet, in pullout AND in rupture: the
rupture check was `Tallowable / Tmax` with no Rc, and the built-in strip's
43.1 (a representative allowable load per 75 mm strip) was compared with the
tension per metre of wall. Geosynthetics took Rc = 1, which is right for a
continuous layer.

**Now.** `mse.coverage_ratio`: Rc given (0 < Rc <= 1), or b/Sh from the
reinforcement's `width` and `horizontal_spacing`, or 1 for a geosynthetic
given neither (p. 3-13). A steel strip or grid given neither is REFUSED,
naming `coverage_ratio`, `reinforcement_horizontal_spacing` and
`reinforcement_width`; the refusal quotes GEC-11 Vol II's spacings "for
comparison only (the tool assumes none of them)" (E3: 50 mm strips at 0.76 and 0.51 m, 2 or 3 a 1.5 m panel,
"a minimum 2 strips at a horizontal spacing not exceeding 2.5 ft"; E5: 0.40
to 0.52 m; E4: one 4-wire mat, b 0.46 m, a panel). No spacing is assumed.
Pullout is F* alpha sigma_v Le C Rc and the rupture resistance Tal x Rc,
both per metre of wall; each level reports `Rc`, `alpha` and
`Tal_Rc_kN_per_m`, and the result's choices say how Rc was set. `Tallowable`
is Tal per metre of the reinforcement's own width (Eq. 3-11a): the built-in
strip's is 43.1 / 0.075 = 574.7 kN/m, the only reading of 43.1 that is a
steel strip's strength (per unit strip width it would be 3.2 kN a strip);
the built-in grid's 32 kN/m stays per metre of grid width. A built-in strip
carries its own b (a different `reinforcement_width` is refused); the
built-in grid takes one. `mse_lrfd_external_stability` refuses the new
inputs with the reason.

**Tested F* and alpha.** `F_star_tested` (one value, or [[depth_m, F*], ...]
linear between and held beyond) replaces the empirical F* (Eqs. 3-4 to 3-8)
at every level: Table 3-6 note (i) accepts those only "in the absence of
product and backfill specific test data" and for Table 3-1 fill with Cu >= 4.
`alpha_tested` (0 < alpha <= 1) replaces alpha: p. 3-16, alpha "can be
obtained from pullout tests on reinforcements with different lengths as
presented in Appendix B"; note (ii), "Pullout testing to determine alpha is
recommended if alpha shown in table is less than 1.0". Each is a choice
with FOS_pullout at the empirical value beside it. A Cu below 4 with no
tested F* is still computed and flagged (GM-246); refusing it is the
owner's call.

**Numbers.** The 9 m strip wall (H 9 m, L 7 m, Sv 0.75 m, gamma 20, phi
34, 75 mm strips): FOS_pullout 28.76 to 42.10 and FOS_rupture 0.982 to
16.24 before; at Sh 0.75 m (Rc 0.1) 2.876 to 4.210 and 1.309 to 21.65 now
(pullout x Rc; rupture 43.1 kN a strip / Sh). A 6 m geogrid wall (UX1600,
L 5 m, Sv 0.6 m, phi 34): unchanged, FOS_pullout 10.493 to 25.792,
FOS_rupture 1.192 at the bottom. GEC-11 Vol II Example E3 (Table E3-7.3, p.
E3-19) with its own b 0.164 ft and Sh 2.50 ft (Levels 1-8) and 1.67 ft
(9-12): the module's Pr times Sh is E3's Pr per strip and its FOS times
0.90 / 1.35 (pullout) or 0.75 / 1.35 (tension) is E3's Ng phi R / Tmax, at
every level within 0.4 %; Level 4: phi Pr 12.23 k a strip, 2 x 12.23 /
12.35 = 1.98 printed. With Rc = 1 that level read FOS_pullout 45.2 (30.2
against the printed 1.98). E5 is a bridge abutment whose footing loads the
module does not take; its Level 4 Pr = 9.04 k a strip (p. E5-28) is
reproduced by the primitive. Tests: `tests/test_mse_coverage_ratio.py`,
V-056 in `validation_examples/test_published_v056_v057_gec11_slope.py`.

**The scale effect correction factor alpha (second GM-256 commit).** Table
3-6 (p. 3-17), "alpha Default Value": 1.0 for inextensible strips, 1.0* for
inextensible grids (bar mats and welded wire), 0.8 for extensible grids, 0.6
for extensible sheets; note *: "For longitudinal bars/wires spacing greater
than 6 inches, alpha may be less than 1.0 and pullout tests are required";
note (ii): "Pullout testing to determine alpha is recommended if alpha shown
in table is less than 1.0". p. 3-16: "In the absence of test data, alpha =
0.8 for geogrids and alpha = 0.6 for geotextiles (extensible sheets) is
recommended". The module took 1.0 for every reinforcement. `Reinforcement`
now has the types `geogrid` and `geotextile` beside `geosynthetic`, and
`mse.pullout_alpha` takes alpha as follows:
- `alpha_tested` when given;
- else Table 3-6's value for the type (`mse.table_alpha`): 1.0 for steel,
  0.8 for a geogrid, 0.6 for a geotextile;
- a `geosynthetic` of no stated kind, given no tested alpha, is refused,
  and the refusal quotes both printed values.

The built-in geogrids are now type `geogrid`. A choice states the alpha
wherever the table attaches a condition to it: below 1.0 (note (ii)), and a
steel grid's 1.0* (note *). Each choice quotes the table and p. 3-16. Its
alternative gives FOS_pullout at alpha 1.0, or at the printed values beside
a tested alpha. The tool's custom `reinforcement_type` has no default now:
the type sets the alpha. Kr/Ka (1.0) and F* (0.67 tan phi) are the same for
all three geosynthetic types.

Numbers on the 6 m wall (phi 34, L 5 m, Sv 0.6 m), with FOS_rupture
unchanged at 1.192:

| Reinforcement | FOS_pullout before (alpha 1.0) | FOS_pullout now |
|---|---|---|
| Geogrid (UX1600) | 10.493 to 25.792 | 8.394 to 20.633 (alpha 0.8) |
| Geotextile of the same Tal | 10.493 to 25.792 | 6.296 to 15.475 (alpha 0.6) |

GEC-11 Vol II Example E1 takes "alpha = Scale correction factor, = 0.8 for
these geogrids" (Step 7.8, p. E1-16). The module's geogrid alpha and F*
reproduce every pullout CDR of Table E1-7.5 (p. E1-17) within 1.2 %. At
alpha 1.0 each would be 25 % high. Tests:
`tests/test_mse_scale_effect_alpha.py`, and V-057 in
`validation_examples/test_published_v056_v057_gec11_slope.py`.

**Review (2026-10-20).** The pages above were rendered and read; the 9 m
wall's bottom level worked by hand both ways (Tmax 43.87 kN/m; 43.1 / 43.87
= 0.982 before, 43.1 / 0.75 / 43.87 = 1.310 now: 1.33 strips a metre of
wall). The tool refuses two inputs that were taken and not used: a
geosynthetic's width with no spacing (p. 3-12 describes geosynthetic strips
by their width and spacing, so Rc = 1 would overstate a strip layer;
`mse.GEOSYNTHETIC_WIDTH_NEEDS_SH`), and a width beside `coverage_ratio`.
The library keeps Rc = 1 for the first, as the retired Foundry wrapper
gives every custom reinforcement a 0.05 m width, and its choice says the
width was not used. The refusals now open with what is missing ("A
steel strip needs its coverage ratio Rc = b/Sh ...").

**Left for the owner** (BACKLOG):
- the built-in W11 grid's Tallowable 32.0: its source is not recorded. It
  equals Fy times the area of one uncorroded W11 wire (450 MPa x 70.9 mm2 =
  31.9 kN), and read per metre of grid width (times Rc) it is several times
  below E4's W11 mat (Tn 5.2 k a wire at 0.5 ft, b 1.5 ft: about 202 kN/m of
  grid width). At E4's layout (Rc 0.30) the built-in grid shows FOS_rupture
  about 0.4 where E4's mat passes. Conservative; re-derive it or drop the
  built-in strength;
- whether Cu < 4 with no tested F* is refused rather than computed and
  flagged;
- Rc is one value for the wall: E3 changes Sh with depth (2.50 ft to
  1.67 ft), so a wall like it is run once per spacing.

## Key Notes
- CantileverWallGeometry auto-sizes stem thickness, toe/heel from wall height
- Cantilever checks decompose the active thrust per the chosen method (RW-1):
  Coulomb thrust acts at δ = 2/3·φ from horizontal, sloped-Rankine thrust
  parallel to the slope (β). Only P_h = Pa·cos(δ) drives sliding/overturning;
  P_v = Pa·sin(δ) is applied downward at the back of the heel (x = B) and is
  stabilizing. Rankine + level backfill is unchanged (δ = 0).
- Sloped backfill adds the triangular soil wedge above the heel to the
  weight/moment tally (RW-4)
- MSE internal: Kr/Ka stress ratio varies with depth (AASHTO method).
  `Kr_Ka_ratio`/`F_star_metallic` carry three inextensible/extensible curve
  families, selected by `reinforcement_type` (Q4):
  - ribbed metallic STRIP ("metallic"/"metallic_strip", DEFAULT): Kr/Ka 1.7→1.2
    over 0–20 ft (6.096 m); F* 2.0→tan(φ) over 0–20 ft (GEC-11 Fig. 4-10, p.
    4-37, drawn at "20 ft" for strips and bar mats alike; Eq. 3-5, p. 3-20,
    "F* = tan φ at a depth of 20 ft (6 m) and below"). Until GM-112's review
    (2026-10-16) the strip curves ran over 0–6.0 m, the rounded metric value;
    on GEC-11 Example E3 that left F* 0.6 % low.
    **The top value is 1.2 + log Cu, 2.0 maximum (GM-246, 2026-10-20).**
    GEC-11 Vol I Eq. 3-4 (p. 3-20) gives F* = 1.2 + log Cu at the top of
    the structure, Cu the backfill's uniformity coefficient D60/D10, and
    says "If the specific Cu for the wall backfill is unknown at design
    time, a Cu = 4 should be assumed (i.e., F* = 1.8 at the top of the
    wall) for reinforced fills meeting the requirements of Section 3.1".
    Table 3-6 note (i) (p. 3-17) accepts these empirical values only for
    Table 3-1 granular fill with Cu >= 4: "For fill outside these limits,
    tests must be run". A Cu below 4 is computed, and the choice says so.
    Every strip took 2.0 before, the value a backfill reaches only at Cu
    6.3 or more (Vol II Examples E3 and E5, Cu 7.0, take it: 1.2 + log 7.0
    = 2.045, use 2.000). `mse_wall` takes `Cu_backfill` (aliases `Cu`,
    `uniformity_coefficient`); left out, Cu = 4, and the result's choices
    say so, quote the page and give FOS_pullout at the top level and the
    lowest at the other reading (Cu 6.3 or more, or Cu 4 when one was
    given). A steel grid given its t/St and a geosynthetic keep their own
    F*, which takes no Cu; a Cu given with them is checked against note
    (i) only, since that note covers every empirical F* in Table 3-6, and
    the choice says whether it is within it (it was refused at first; the
    GM-246 review kept it, as refusing it dropped the only check the page
    makes with it). `mse_lrfd_external_stability` refuses it. At phi 34,
    F* at 0, 3 and 6 m: 2.000, 1.348, 0.695 before; 1.802, 1.247, 0.692
    now; unchanged at 6.096 m and below (tan 34 = 0.675). On a 9 m wall
    the top level's pullout falls 10 % (123.7 to 111.8 kN/m at 0.38 m)
    and the levels below 6.1 m do not change, so the lowest FOS_pullout
    there is the same. The library's `F_star_metallic` takes `Cu` too. The
    reference library's GEC-11 Table 3-6 rows still give tan phi at the
    top (BACKLOG REF-42); the module does not read them.
  - steel bar-mat / welded-grid ("bar_mat"/"welded_grid"/"metallic_grid"):
    Kr/Ka 2.5→1.2 over 0–20 ft (6.096 m); F* 20(t/St)→10(t/St) over 0–20 ft,
    where t = transverse-bar diameter and St = transverse-bar spacing
    (GEC-11 Fig. E4-5). The grid F* takes t/St from the `Reinforcement`
    geometry (`thickness`/`transverse_spacing`) or the `t_over_St` arg; if a
    grid lacks that geometry F* falls back to the strip curve.
  - geosynthetic ("geogrid", "geotextile", "geosynthetic"): Kr/Ka 1.0; F*
    0.67·tan(φ); alpha by Table 3-6, 0.8 geogrid, 0.6 geotextile, a
    "geosynthetic" of no stated kind given a tested alpha (GM-256, above).
  `check_internal_stability` auto-selects the bar-mat curves for a
  `metallic_grid` reinforcement (e.g. `WELDED_WIRE_GRID_W11`). Q4 left the
  strip default unchanged; GM-246 changed its top value (above).
  - **Source basis (built-in reinforcement products).** The product constants
    (the `WELDED_WIRE_GRID_W11` grid geometry, the 75x4 mm ribbed-strip
    `T_allowable`, etc.) are **representative values typical of the reinforcement
    class, NOT a specific manufacturer's catalog** — they exist to exercise the
    Kr/F* selection logic and give sensible defaults. For a real design the user
    should supply the actual product's certified properties. The Kr/F* CURVES
    themselves are digitized from GEC-11 Figs 4-11/E4-5 (see `mse.py`), anchored
    by the GEC-11 Example E4 reproduction.
  Active-zone length La branches on reinforcement extensibility (RW-2):
  bilinear coherent-gravity surface (0.3H) for metallic/inextensible (strips and
  bar mats), Rankine 45+φ/2 plane for geosynthetic/extensible
  (AASHTO Fig. 11.10.6.3.1-1)
- MSE external bearing uses the Meyerhof uniform pressure over the effective
  width, σ_v = W/(L−2e), per AASHTO 11.10.5.4 (RW-3) — not trapezoidal q_toe
- SoilProfile adapter: to_retaining_wall_input(wall_height, surcharge)

## Sliding-check base interface (verdict, 2026-07-15)

Owner wall session 2026-07-14 (short-heel wall: 0.5 m heel, 2.6 m toe,
H=5.55 m, gamma=15, phi=33, c=5, q=7.5) showed the built-in sliding FoS
(0.60-0.92) far below an independent free-body check (1.01-1.42). ROOT CAUSE
was NOT the module physics: check_sliding hardwired delta_b = (2/3)*
phi_foundation and ca = (2/3)*c_foundation, and the caller passed
phi_foundation=22 intending "delta_b = 22 deg" -> the module applied 2/3
AGAIN (delta_b = 14.7 deg). With phi_foundation passed correctly (33) the
module reproduces the free-body EXACTLY (Coulomb 1.42 / Rankine 1.01) -
vertical thrust component and full-height cohesion conventions agree.

Resolution (additive, default-preserving): `delta_base` / `base_adhesion`
params on check_sliding + analyze_cantilever_wall BYPASS the 2/3 factors for
direct interface control (e.g. delta_b = phi for CIP concrete soil-on-soil
shear per GEC-11/AASHTO); results now report delta_base_deg /
base_adhesion_kPa actually used, and the funhouse adapter returns a
`sliding_basis` block. Regression tests: TestBaseInterfaceOverrides
(session numbers 0.60/1.42/1.01 locked in).

GM-157 (2026-10-18): that free body counted the 7.5 kPa surcharge over the
heel as resisting. By default the module no longer does (section above),
so the session numbers are reproduced with `live_load_resists=True`, and
the defaults are 0.566 / 1.339 / 0.978.

GM-173 (2026-10-19): the result's `choices` state the base friction used,
first in the list (`cantilever.base_friction_choice`,
`BASE_FRICTION_BASIS`). The default stays delta_b = 2/3 phi_f, phi_f the
foundation soil's or the backfill's when `phi_foundation` is not given
(GM-165 declares that absent case in `defaults_used`; the choice does not
repeat it, so it reads the same with phi_foundation given or left out). Its basis: EN 1997-1 Sec. 6.5.3(10) (p. 64), 2/3
phi'cv;d "for smooth precast foundations", and UFC 3-220-20 Fig. 4-27
(p. 220), tan delta "friction factor between soil and base" from Table 4-1
(p. 184). The alternative is delta_b = phi_f (`delta_base` = phi_f) for
concrete cast against the ground: GEC-6 Sec. 5.4 (p. 99), "For concrete
cast against cohesionless or granular material, the coefficient of
friction, tan delta, will be equal to the tangent of the friction angle
(tan phi) for the soil supporting the footing", and EN 1997-1 Sec.
6.5.3(10), phi'cv;d "for cast-in-situ concrete foundations". Wave w13 F55
turned on it: sliding 1.07 at the default, 1.72 at delta_b = phi_f. No
number changes. EM 1110-2-2502 is not in the reference library.

## Passive resistance on a shear key (GM-210, 2026-10-19)

With `include_passive` the sliding check adds Rankine passive resistance in
front of the wall, with the soil in front taken up to the top of the base:
without a key on the base thickness t, Pp = 1/2 Kp gamma_f t^2. With
`has_shear_key` it now acts over the depth of the key only, from the
underside of the base (depth t) to the level of the bottom of the key
(t + k): Pp = 1/2 Kp gamma_f [(t + k)^2 - t^2]. UFC 3-220-20 Sec. 4-5.1.3
and Fig. 4-29 (pp. 223-224): "the passive pressure is applied only along
the depth of the key and not along the entire burial depth of the wall";
the figure draws the trapezoid on the vertical plane below the toe (a-a1,
the front of the intact soil the key carries), not on the key itself, so
where the key sits under the base does not change Pp. GEC-6 Sec. 5.4 Fig.
5-22 / Eq. 5-41 (pp. 100-101) draws the same and adds that keys in granular
soils "are not very efficient in developing passive pressures"; its key
method is for stiff, overconsolidated clays and soft rock. The base
friction stays (W + Pv) tan delta on the base, UFC's drained form in Fig.
4-29. Before, the key took the triangle 1/2 Kp gamma_f k^2, as if the top
of the key were at the ground surface. Wall 5.0 m, B 3.0 m, t = k = 0.6 m,
phi 30, gamma 18: Pp 9.7 -> 29.2 kN/m, FOS_sliding 1.31 -> 1.57. The two
options are different assumptions, not one formula: without a key the
check counts passive over the base's own burial depth t, and with one UFC
counts it over the key depth only, so a key shallower than about 0.41 t
counts less passive than the no-key option. The check's surface is the top
of the base; UFC adds "it is also prudent to consider some amount of future
excavation in front of the wall". `sliding_basis` reports the Pp counted.

**The theory of Kp (GM-242, 2026-10-20).** `passive_theory` 'rankine' (the
default) or 'log_spiral' (the coefficient tool's 'caquot_kerisel' name is
taken too), with `log_spiral_source` (default UFC 3-220-20, Eq. 4-12 for the
vertical face and level ground in front of the toe) and
`passive_wall_friction_deg` (0 to phi_foundation; left out, 0.66 phi',
UFC p. 191; the 1948 chart and Eq. 4-12 by name refuse it, the GM-242
review). A shear key needs its depth: `has_shear_key` with no `key_depth`
> 0 is refused (before, it was taken as no key without a word). UFC 3-220-20 Sec. 4-5.1 (p.
219) says the passive pressure in front "is typically ignored", and with it
considered Fig. 4-27 (p. 220) and Sec. 4-5.1.3 (p. 223) ask FSL >= 2.0;
neither names a theory for its Kp, so the module keeps Rankine's, the
smaller. With include_passive the result's `choices` state the theory with
the other theory's and the no-passive FOS_sliding; with 'log_spiral' also
the source and the wall friction. `sliding_basis` gives `passive_theory`
and `Kp`. Wall 5 m, B 3.5 m, toe 1 m, t 0.6 m, phi 32, gamma 19:
FOS_sliding 1.53 (Rankine, Kp 3.25), 1.65 (log spiral, Kp 5.90 at delta
21.1), 1.37 with no passive.

**The coefficient tool's wall friction (GM-253, 2026-10-20).**
`earth_pressure_coefficient` with theory 'caquot_kerisel' takes no wall
friction that no source prints. Left out, the UFC 3-220-20 sources take
delta/phi' = 0.66: Figs. 4-12 and 4-13 (pp. 192-193) are drawn for that
value only, their headers print it, and p. 191 calls it "a common
assumption" beside Eq. 4-12. The Caquot-Kerisel (1948) chart and Eq. 4-12
by name print no value: the Caltrans T&S procedure for the chart (Sec.
4-6, p. 4-29) begins "Given delta, beta, and phi". So a call that leaves
`delta_deg` out with them is refused and names it, as `soe` and the wall
toe refuse it (GM-242). Before, the 1948 chart took delta = phi, its
largest Kp: 6.30 at phi 30, against 5.06 by default. The refusal gives
the printed guidance: Caltrans Table 4-2 (p. 4-15), UFC Table 4-1 (p.
184), and FHWA NHI-06-089 (p. 10-17), which gives delta = 1/2 phi' as
"the maximum wall friction suggested for design" on the passive side, and
zero for walls retaining soft cohesive soils or granular soils "that will
be subjected to significant vibration", and prints the same chart as its
Fig. 10-10. Every log-spiral result's wall
friction choice gives the Kp at delta = 0 beside the delta used:
- Rankine's Kp before a vertical face with level ground, which Eq. 4-12
  gives at delta = 0;
- the 1948 chart with R from its delta/phi = 0 column for sloping ground;
- Coulomb's Kp at alpha = 90 - theta for a battered face, as no log-spiral
  chart here gives one at delta = 0 (NHI-06-089 p. 10-19 calls Coulomb's
  passive results "unreliable" for wall friction "greater than phi'/3",
  and the failure surface "almost linear" when delta is small).

## The foundation cohesion in sliding (GM-212, 2026-10-20)

Before, the sliding check added ca = 2/3 c_foundation on B to the friction
V tan delta_b. None of the references prints that rule. They print three
rules. `sliding_cohesion` offers each one, and the result's `choices` give
its basis and each other rule's FOS_sliding on the same wall:

- `drained`: R = V tan delta_b, and the cohesion is not counted.
  - GEC-6 Sec. 5.4 (p. 100), Eq. 5-38, "for footings with planar bearing
    surfaces on frictional (drained) materials".
  - GEC-6 again: "Because drained and undrained conditions will not occur
    simultaneously, Equations 5-38 and 5-39 should not be combined for use
    at the same time".
  - EN 1997-1 Sec. 6.5.3(10) (p. 64): "Any effective cohesion c' should be
    neglected".
- `undrained`: R = ca B alone, with no friction.
  - GEC-6 Eq. 5-39: "In practice, ca is typically taken as equal to the
    undrained shear strength (cohesion, cu) of the clay".
  - EN 1997-1 6.5.3(11) Eq. 6.4a.
  - EN 6.5.3(12)P limits Rd to 0.4 Vd "if it is possible for water or air
    to reach the interface". That limit is not applied, and the basis says
    so.
- `friction_plus_adhesion`: R = V tan delta_b + Ca Be.
  - UFC 3-220-20 Fig. 4-27 (p. 220) and Sec. 4-5.1.3 (p. 223).
  - Ca is from UFC Table 4-1 (p. 184) by the undrained shear strength,
    read linearly within its rows. The table stops at 4000 psf (191.5 kPa);
    above that the rule refuses and asks for `base_adhesion`.
  - Be = B - 2e, by the uniform bearing pressure method, with e taken on
    the loads the check counts.
  - UFC adds that the minimum FS "may also need to be increased when base
    adhesion is included".
  - FHWA NHI-06-089 Fig. 10-17 (after NAVFAC 1986b) prints the same sum
    with the full B.

Answers to the three questions:

1. Does any reference say not to combine adhesion and friction? Yes:
   GEC-6.
2. Does any reference reduce the adhesion? Only UFC and NHI, through their
   adhesion tables (Ca below su above about 24 kPa). None uses 2/3.
3. Does any reference say to ignore cohesion for drained sliding? Yes:
   EN 1997-1. GEC-6 does the same by not combining the two equations.

NHI p. 10-17 is about wall-back adhesion (cw = 0.5 c', "no wall adhesion
be allowed"), not the base.

The default depends on phi_foundation:

- phi_foundation = 0 gives `undrained`. A clay layer from
  `SoilProfile.to_retaining_wall_input` gives phi_foundation 0 and
  c_foundation = cu.
- Anything else gives `drained`.

The choice's basis starts with that reason ("phi_foundation 28 deg is
taken as a drained strength, so c_foundation is c'"). Its other rules
read c_foundation as cu and say so; with phi_foundation 0 the drained rule
is listed as not run (no friction angle to use), not as FOS 0.

An input the rule does not count is refused:

- `base_adhesion` with `drained`;
- `delta_base` with `undrained`.

`sliding_basis` reports `sliding_cohesion` and `adhesion_width_m`.

Before and after, on a wall with H 5 m, B 3.0 m, backfill gamma 19 and phi
32, and foundation c' 10 kPa and phi' 28:

| | Resisting R (kN/m) | FOS_sliding | Result |
|---|---|---|---|
| Before (90.0 friction + 6.67 x 3.0 adhesion) | 110.0 | 1.507 | passes |
| Default `drained` now | 90.0 | 1.233 | fails |
| `friction_plus_adhesion` (Ca 10 kPa on Be 2.33 m) | 113.3 | 1.553 | passes |

