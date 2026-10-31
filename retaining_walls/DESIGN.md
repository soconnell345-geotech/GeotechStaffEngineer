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
(L 5 m, Sv 0.6 m, γ 19, φ 34°, the built-in geogrid of the time, Tal
21.9 kN/m) with q = 12 kPa, the top
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
(pullout x Rc; rupture 43.1 kN a strip / Sh). A 6 m geogrid wall (the
built-in geogrid of the time, Tal 21.9 kN/m, L 5 m, Sv 0.6 m, phi 34): unchanged, FOS_pullout 10.493 to 25.792,
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
| Geogrid (Tal 21.9 kN/m, the built-in of the time) | 10.493 to 25.792 | 8.394 to 20.633 (alpha 0.8) |
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

A Cu given beside a tested F* is in no check: the tested F* replaces the
empirical one. It is not refused, as it still bears on what the result
reports: a strip's empirical F* given beside the tested one (Eqs. 3-4, 3-5)
and note (i)'s Cu >= 4 check on the empirical F* (p. 3-17). The F* choice
says so, for a strip and for a grid or geosynthetic, whose empirical F*
takes no Cu.

**Left for the owner** (BACKLOG):
- the built-in W11 grid's Tallowable 32.0: its source is not recorded. It
  equals Fy times the area of one uncorroded W11 wire (450 MPa x 70.9 mm2 =
  31.9 kN), and read per metre of grid width (times Rc) it is several times
  below E4's W11 mat (Tn 5.2 k a wire at 0.5 ft, b 1.5 ft: about 202 kN/m of
  grid width). At E4's layout (Rc 0.30) the built-in grid shows FOS_rupture
  about 0.4 where E4's mat passes. Conservative; re-derive it or drop the
  built-in strength. Done: GM-262 below;
- whether Cu < 4 with no tested F* is refused rather than computed and
  flagged;
- Rc is one value for the wall: E3 changes Sh with depth (2.50 ft to
  1.67 ft), so a wall like it is run once per spacing. Done: GM-261 below.

## The tensile strength of a steel bar mat (GM-262, 2026-10-20)

**What GEC-11 prints.** GEC-11 Vol I (FHWA-NHI-10-024), read on the
rendered pages:
- Eq. 3-11a (p. 3-22): Tal = Fy Ac / b, "in strength per unit reinforcement
  width"; Eq. 3-11b, Ptal = Fy Ac, per strip or per mat;
- Fig. 3-5 (p. 3-24), for metallic grids and bar mats: Ac = (No. of
  longitudinal bars) x (D*)^2 / 4, D* the bar's diameter corrected for
  corrosion loss (the figure prints no pi; Vol II Example E4 takes pi);
  b is the width of the grid, and Tmax <= phi Rc Tal;
- p. 3-23: Ac is the area after corrosion, and phi is 0.75 for strips,
  0.65 for grids connected to a rigid facing and 0.75 for grids connected
  to a flexible facing; Table 4-7 (p. 4-48) prints the same factors;
- p. 4-77 (checklist IX.5): "Is the correct value of nominal strength of
  steel used? (e.g., 0.55 Fy for strips and 0.48Fy for bar mats)".

Vol II (FHWA-NHI-10-025) Example E4, a precast-panel wall on bar mats:
- p. E4-14: W11 D = 0.374 in., W15 D = 0.437 in.; after 75 years
  (galvanizing gone in 16, then 0.47 mil/year/side of carbon steel for 59)
  0.318 in. and 0.381 in. The page's "Steel loss = 0.16 mil/year/side" is
  the zinc's rate; the calculation takes 0.47;
- p. E4-15: Tn = Fy A = 65 ksi x 0.0795 in.2 = 5.17 k a W11 wire, Tr =
  0.65 Tn = 3.36 k; W15 7.42 k and 4.82 k;
- p. E4-17: Nt = Tmax / Tr; Level 4, Tmax 12.75 k a 5 ft panel, Nt = 3.8,
  so 4 wires;
- p. E4-18, Table E4-7.4: every level's mat. Levels 1, 3 and 4 are "4W11 +
  W11x0.5'", 4 W11 wires at 0.5 ft, one mat a 5 ft panel; Level 2 is
  3W11; Levels 5 to 10 are 4W15.

**What was wrong.** The built-in `WELDED_WIRE_GRID_W11` carried
Tallowable 32.0 kN/m with no recorded source: Fy times ONE uncorroded W11
wire. Since GM-256 it is read per metre of grid width and multiplied by Rc,
so at E4's layout (b 1.5 ft, Sh 5 ft) it gave the strength of about 0.9 of
a wire per panel where E4's mat has four.

**What changed.**
- `reinforcement.steel_grid_Tal(n_longitudinal, D_corroded, Fy, b,
  fraction)`: Eq. 3-11a with Fig. 3-5's Ac, pi taken as E4 takes it.
  `fraction` 1 gives the nominal strength; `BAR_MAT_ALLOWABLE_FY` = 0.48
  the allowable.
- The built-in is now E4's mat, named "Bar mat 4W11 + W11x0.5' (GEC-11 Vol
  II Example E4)": 4 W11 wires, **b = 1.5 ft (0.4572 m) carried on the
  built-in** as the strip carries its width (75 mm then, E3's 50 mm since
  GM-267), Fy 65 ksi, W11 transverse at 6
  in. Tal = 0.48 x 65 ksi x 4 pi (0.318 in.)^2 / 4 / 1.5 ft = **96.44 kN/m
  of mat width** (11.02 kN a wire). So `reinforcement_horizontal_spacing`
  alone gives its Rc; a different width is refused.
- Why 0.48 Fy and not phi: the module's rupture check compares Tal x Rc
  with the UNFACTORED Tmax at a factor of safety of 1.0, so Tal must be an
  allowable. E4's LRFD check takes phi 0.65 against Tmax times 1.35, and
  0.65 / 1.35 = 0.481: the same check to 0.3 %.
- The mat's basis (`Reinforcement.basis`) is in the coverage choice of
  every result and in `list_reinforcement` (`Tallowable_basis`). The tool
  text for `reinforcement_Tallowable` now says the check is at FOS 1.0 on
  the unfactored Tmax, so a custom steel Tal is an allowable: 0.55 Fy for
  strips and 0.48 Fy for bar mats (p. 4-77), and a mat's Ac is the number
  of longitudinal bars times one bar's corroded area. It said Fy Ac / b
  before, the nominal.

**Numbers** at E4's layout (H 25.64 ft, L 18 ft, Sv 2.5 ft, 250 psf live
load, Sh 5 ft, Rc 0.30; the module's levels sit at 1.25 + 2.5i ft, E4's at
1.87 + 2.5i ft):

| Level depth | FOS_rupture before (Tal 32.0) | FOS_rupture now |
|---|---|---|
| 1.25 ft | 0.947 | 2.854 |
| 8.76 ft (beside E4's Level 4) | 0.359 | 1.081 |
| 11.25 ft | 0.318 | 0.957 |
| 23.75 ft (bottom) | 0.241 | 0.726 |

Tal x Rc is 28.93 kN/m of wall (9.60 before); FOS_pullout is unchanged
(1.680 to 6.860). Below about 10 ft the 4W11 mat does not carry E4's
loads, as in E4, which changes to 4W15 mats from Level 5; the module takes
one reinforcement for the wall, so such a wall is a custom reinforcement
for its lower levels. (Since GM-266, below, a reinforcement per level.)

**E4 reproduced** (`validation_examples/test_published_v009_v011.py`,
V-010): Tn and Tr a wire for W11 and W15 within 0.2 % of p. E4-15; Nt at
all ten levels of Table E4-7.4 within 0.07 of the printed one-decimal
values, and each level's Ng wires carry its factored Tmax; at Level 4 the
built-in mat's Tal x Rc over the unfactored Tmax is 1.044, against E4's 4 x
3.36 / 12.75 = 1.054. Module tests: `tests/test_mse_bar_mat_strength.py`.

**Left for the owner** (BACKLOG; the review below took the first three):
- the ribbed strip's 43.1 kN a strip has no recorded source either (it is
  about 0.32 x 450 MPa x 75 x 4 mm, uncorroded) - now 0.55 Fy Ec at E3's
  corroded thickness, 47.9 kN (review below);
- a grid connected to a flexible facing takes phi 0.75 (p. 3-23), so 0.55
  Fy; the built-in is E4's rigid-facing mat at 0.48 Fy - now stated, with
  the flexible-facing Tal as the alternative (review below);
- whether the built-in geogrids' Tal (21.9 and 29.2 kN/m) is a long-term
  design strength or an allowable: the check takes it at FOS 1.0 - now
  taken as Tult / RF and checked at 1.5 (review below);
- a reinforcement per level (E4's 3W11 at Level 2 and 4W15 below Level 4)
  is not taken; GM-261 takes the spacing per level. Done: GM-266 below.

## The spacing per reinforcement level (GM-261, 2026-10-20)

**What GEC-11 prints.** Vol II Example E3 sizes the strips at each level
and spaces them to suit: "The last column of Table E3-7.3 provides
horizontal spacing of the reinforcing strips which is obtained by dividing
the panel width, wp, by the governing number of strips, Ng" (p. E3-19).
Ng is 2 a 5 ft panel at Levels 1 to 8 (Sh 2.50 ft) and 3 at Levels 9 to 12
(Sh 1.67 ft). Vol I Eq. 3-1 (p. 3-12), Rc = b/Sh, holds level by level.

**What changed.** `Reinforcement.horizontal_spacing` and
`Reinforcement.coverage_ratio` take one value for the wall, as before, or
a list with one value a level from the top down (kept as a tuple;
`Reinforcement.per_level`). `mse.coverage_ratios(reinforcement, depths)`
gives Rc at each level; `check_internal_stability` takes Rc and Tal x Rc
level by level, and each level's result carries its own `Rc`. A list whose
length is not the number of levels is refused, naming the levels and their
depths (Sv sets them); a level whose b is more than its Sh is refused,
naming the level. `coverage_ratio(reinforcement)` stays the one-value
function and refuses a per-level input. The coverage choice lists the
spacings by runs of levels ("Sh 0.762 m at levels 1-8 (Rc 0.0656); Sh 0.508
m at levels 9-12 (Rc 0.0984)") with Tal x Rc as a range, and cites p.
E3-19. The calc package shows S_h and R_c per level ("0.750 x 8, 0.500 x
4"). The tool takes a list for `reinforcement_horizontal_spacing` or
`coverage_ratio` ("float or list").

**Numbers.** E3 at one spacing for the wall (2.50 ft, the run before
GM-261) against E3's spacing per level, Levels 9 to 12 (the module's
levels at 21.26 to 28.74 ft; Levels 1 to 8 are the same):

| Level | FOS_pullout one Sh | per level | FOS_rupture one Sh | per level |
|---|---|---|---|---|
| 9 | 2.110 | 3.165 | 1.851 | 2.776 |
| 10 | 2.237 | 3.355 | 1.691 | 2.537 |
| 11 | 2.365 | 3.548 | 1.557 | 2.336 |
| 12 | 2.494 | 3.741 | 1.443 | 2.164 |

At Level 12, E3's tension check is Ng phi Tn / Tmax = 3 x 9.75 / 24.36 =
1.20; at one spacing it would be 0.80.

**E3 reproduced in one run** (`validation_examples/
test_published_v056_v057_gec11_slope.py`, V-056): the pullout per strip
and Ng phi R / Tmax for pullout and tension at all twelve levels of Table
E3-7.3, each level equal to the run at its own spacing. Module tests:
`tests/test_mse_per_level_spacing.py`.

**Not covered:** a reinforcement per level (another Tal, another width, or
E4's W15 mats below Level 4) is not taken; only the spacing or the ratio
varies. Done: GM-266 below.

## Review of GM-262 and GM-261 (2026-10-20)

**The pages, rendered and read.**
- Fig. 3-5 (Vol I p. 3-24) prints "Ac = (No. of longitudinal bars) x
  ((D*)^2 / 4)", with no pi, read at 300 dpi. It is a misprint: the area
  of a round bar is pi D^2 / 4, and Vol II Example E4 takes "(pi)(0.318
  in.)^2/4 = 0.0795 in^2/wire" (p. E4-15). The code takes pi, as E4 does;
  without it Tal would be 21 % low.
- p. 4-77 (checklist IX.5) prints "Is the correct value of nominal
  strength of steel used? (e.g., 0.55 Fy for strips and 0.48Fy for bar
  mats)". The checklist calls them "nominal", but Eq. 3-11a's nominal is
  Fy Ac / b: these are the allowables of a check on the unfactored load,
  and they equal phi over the load factor, 0.75 / 1.35 = 0.56 and 0.65 /
  1.35 = 0.48.
- The module's rupture check is that check: `check_internal_stability`
  takes FOS_rupture = Tal x Rc / Tmax with Tmax unfactored (E4 Level 4:
  the module's 9.50 k a 5 ft panel, E4's factored 12.75 k / 1.35 = 9.44 k)
  and requires 1.0, beside FOS_pullout 1.5. So 0.48 Fy is the consistent
  allowable for a bar mat on a rigid facing.
- p. 3-23: phi "is 0.65 when reinforcement is connected to a rigid facing
  element and is 0.75 when connected to a flexible facing"; it points to
  "Table 4-8", but the table is Table 4-7 on p. 4-48, whose note C reads
  "For grid reinforcements connected to a flexible facing mat or which are
  continuous with the facing mat, use the resistance factor for strip
  reinforcements".
- Vol II E3 p. E3-16: "Grade 65 (Fy = 65 ksi), 1.969 in. x 0.157 in. (50
  mm wide x 4 mm) thick galvanized steel ribbed strips"; "EC = 0.157 in. -
  0.055 in. = 0.102 in."; area 0.200 in.^2; "Tn = 65 ksi (0.200 in2) =
  13.00 k/strip"; Tr = 0.75 Tn = 9.75 k. Vol I p. 2-16: "The currently
  commercially available strips are ribbed top and bottom, 2 in. (50 mm)
  wide and 5/32-inch (4 mm) thick".

**E4 Level 4, 1.044 against 1.054.** E4's 4 x 3.36 / 12.75 = 1.054 times
1.0436 / 1.0541, the product of four small differences:
- 0.48 against 0.65 / 1.35 = 0.4815: x 0.9969;
- the module's area pi (0.318)^2 / 4 = 0.07942 in.^2 against E4's 0.0795:
  x 0.9990;
- E4 rounds Tr to 3.36 (0.65 x 65 x 0.0795 = 3.359): x 0.9997;
- Tmax: the module takes sigma_H at the level times Sv, 9.497 k a panel
  unfactored; E4 averages sigma_H over the tributary height and rounds
  each step, 12.75 k / 1.35 = 9.444 (its spreadsheet 12.77): x 0.9944.

**Rupture at the bottom, 0.726.** It comes only from taking the 4W11 mat
at every level. At E4's Level 10 (24.37 ft) the module's 4W11 mat gives
0.709, and E4's own numbers for four W11 wires give 4 x 3.38 / 19.05 =
0.710; a 4W15 mat gives 1.018 against E4's 4 x 4.84 / 19.05 = 1.016. E4
changes to 4W15 from Level 5 (11.87 ft) because four W11 wires carry only
4 x 3.38 / 14.26 = 0.95 of the factored Tmax there.

**What the review changed.**
- The reinforcement's Tal is its own choice in the result,
  `mse.tensile_strength_choice` ("Long-term tensile strength Tal of the
  reinforcement (tensile rupture check)"). It was a sentence appended to
  the coverage choice's basis. It is given for a built-in (how "module",
  with its basis), and for any reinforcement whose FOS_rupture is below
  1.0 somewhere: the result then names those levels, and its alternative
  says that this tool takes one reinforcement (one Tal and one width) for
  every level, so a stronger one is a run of its own and a closer spacing
  is `reinforcement_horizontal_spacing` per level (E3's 3 strips a panel
  below Level 8). The built-in mat's basis now says E4 uses it to Level 4,
  and gives E4's 4W15 mat (Tal 138.4 kN/m of mat width) for the levels
  below.
- The facing is stated, with the other reading: the mat's basis says
  0.48 Fy is for a grid on a rigid facing, as E4's panels (phi 0.65); the
  alternative gives a grid on a flexible facing, or continuous with it,
  phi 0.75 and so the strips' 0.55 Fy: Tal 110.5 kN/m, every FOS_rupture x
  1.146. It is not an input: a custom reinforcement takes that Tal.
- **The built-in strip's Tal is from the print** (it was 43.1 kN a strip,
  574.7 kN/m, of no recorded source): `reinforcement.steel_strip_Tal`,
  0.55 Fy Ec (Ac = b Ec, Fig. 3-5a), Ec = 0.102 in. (E3's 4 mm strip after
  75 years), Fy 65 ksi: 638.6 kN/m of strip width, 47.9 kN a 75 mm strip,
  11 % more than before. E3's printed 0.55 Tn / b is 636 kN/m on its
  rounded 0.200 in.^2. The 75 mm width is the tool's own, and its basis
  says so; the choice's alternative gives GEC-11's 50 mm strip (two thirds
  of the tensile and pullout resistance at the same spacing).
- A built-in `reinforcement_name` given with `reinforcement_Tallowable`,
  `reinforcement_type`, `reinforcement_Fy`, `reinforcement_thickness` or
  `reinforcement_transverse_spacing` is refused: they were taken and not
  used, so a 4W15 mat's 138.4 given beside `welded_wire_grid_w11` ran the
  4W11 mat's 96.44.
- A per-level Rc or Sh whose levels all take the same Tal x Rc prints one
  value, not "57.47 to 57.47".

**Numbers after the review.** E4's layout with the built-in mat is
unchanged (FOS_rupture 2.854 at the top level, 1.081 at 8.76 ft, 0.726 at
the bottom). The 9 m strip wall at Sh 0.75 m: FOS_rupture 1.309 to 21.65
before, 1.455 to 24.06 now (Tal x Rc 57.47 -> 63.86 kN/m of wall);
FOS_pullout unchanged (2.876 to 4.210). A Cu_backfill given beside a
tested F* enters no check; the F* choice says so (caa622f, above in the
GM-256 section) and it is not refused, since it still sets a strip's
empirical F* given beside the tested one. Tests:
`tests/test_mse_bar_mat_strength.py`, `tests/test_mse_coverage_ratio.py`,
and `funhouse_agent/tests/test_mse_coverage_ratio_adapter.py`.

**The factor of safety on rupture, by reinforcement type.** Before the
review the rupture check required Tal x Rc / Tmax >= 1.0 of every
reinforcement. That is right for steel, whose Tal is given as GEC-11's
allowable (0.55 Fy or 0.48 Fy, p. 4-77, the 0.75 / 1.35 and 0.65 / 1.35
already in it), and unsafe for a geosynthetic, whose Tal is its long-term
strength:
- GEC-11 Vol I p. 4-47: Tal is "the nominal long-term reinforcement
  design strength" for steel and geosynthetics alike; for a geosynthetic
  Tal = Tult / RF, RF = RFID x RFCR x RFD (Eq. 3-12, p. 3-28). Eq. 4-33
  (p. 4-48): Tr = phi Tal, phi 0.90 for geosynthetics, static (Table
  4-7). Tmax takes gamma_P-EV = 1.35 (p. 4-40).
- Vol II Example E1 does exactly that for its geogrids: GG-I Tult 3,000
  lb/ft, RF 1.85 x 1.15 x 1.3, Tal 1,085 lb/ft, Tr = 0.90 Tal = 976
  (Table E1-7.3, p. E1-15), against the factored Tmax, CDR 1.00 at Layer
  4 (Table E1-7.4, p. E1-16).
- Against the unfactored Tmax this module uses, that is Tal Rc / Tmax >=
  1.35 / 0.90 = 1.5, the same 1.5 the module already requires of pullout
  (phi 0.90). At 1.0 a geosynthetic passed at two thirds of GEC-11's
  factored resistance.

The module now requires 1.35 / phi of each type
(`mse.rupture_FOS_required`): 1.0 for steel (the phi / 1.35 is in its
Tal) and 1.5 for a geosynthetic (Tal = Tult / RF). The steel and
geosynthetic checks are then the same check, GEC-11's phi Tal Rc >= 1.35
Tmax; only where the phi / 1.35 sits differs, and it sits where each Tal
is printed: in the steel allowable of p. 4-77, and in the required factor
of safety for a geosynthetic, whose published Tal (Tult / RF) carries
none. The 0.48 Fy argument above stands. Each level gives
`FOS_rupture_required`, the calc package prints it, the Tal choice of a
geosynthetic states it with this basis, and `check_internal_stability`
still takes `FOS_rupture` to override it.

The built-in geogrids' 21.9 and 29.2 kN/m (about 1,500 and 2,000 lb/ft)
have no recorded source. The code called them the long-term design
strength after creep, installation damage and durability, i.e. Tult / RF,
and the check now takes them so; their basis says the value is not
checked against the product's published strength.

Numbers: the 6 m geogrid wall (the built-in geogrid of the time, 21.9
kN/m, L 5 m, Sv 0.6 m, gamma 19, phi
34) keeps FOS_rupture 1.192 at the bottom level, and now fails rupture at
5.1 m (1.332) and 5.7 m (1.192), where every level passed before. Steel
results are unchanged. Tests: `tests/test_mse_rupture_factor_of_safety.py`.

**Left for the owner and the lead** (BACKLOG):
- a reinforcement per level (another Tal and width at the deeper levels,
  as E4's 4W15 mats, or E1's GG-II geogrids below Layer 4). Done: GM-266
  below;
- the built-in strip's 75 mm width, where GEC-11's strip is 50 mm.
  Done: GM-267 below;
- the built-in geogrids' Tal (21.9, 29.2 kN/m) and their names, one
  maker's product line: no source is recorded. Done: GM-271 below, E1's
  generic geogrids;
- `reinforcement_Fy` is used in no check, for a custom reinforcement too.
  Done: GM-269 below.

## Example E1's rupture CDRs, and `reinforcement_Fy` (GM-268, GM-269, 2026-10-21)

**E1 reproduced through the module.** The review's test pins E1's Layer 4
by hand. The module's own path is set beside every layer of Table E1-7.4
(p. E1-16) in `validation_examples/test_published_v056_v057_gec11_slope.py`
(V-057):
- Tal and Tr of GG-I, GG-II and GG-III (Table E1-7.3, p. E1-15) to the
  pound, Tr = 1.35 Tal / 1.5.
- At each layer the module's Tmax is taken at E1's own Z and Sv, under the
  slope's S_eq. It is within 0.22 % of E1's TMAX / 1.35 (Table E1-7.2, p.
  E1-14). FOS_rupture / 1.5 is then within 0.6 % of E1's CDR: Layer 4 is
  1.006 against 1.00, Layer 9 is 1.015 against 1.01.
- The exception is Layer 10 (Z 18.67 ft, Sv 1.33 ft). The module's Tmax
  there is 3.0 % over the printed 1,366 lb/ft, and Layer 11 below agrees
  within 0.2 %. So the ratio is 1.39 against 1.43, on the safe side.
- A whole run with GG-I at Sv 2 ft fails rupture from 7 ft down (1.46 at
  7 ft), as GG-I reaches its limit at E1's Layer 4.

E1's text cites "Table 4-8" for the resistance factors (p. E1-15); in Vol
I it is Table 4-7.

**`reinforcement_Fy` is refused (GM-269).** `mse_wall` took it and used it
in no check:
- The rupture check works from the Tal given.
- The tool has no section inputs (number of bars, corroded diameter or
  thickness, design life) to work a Tal out of Fy. Adding them would make
  a second way to give the same number.
- Its description said MPa, while the dataclass is in kPa.

Refusing it is the plainer choice. Dispatch refuses it with or without a
built-in name (`refused_parameters`, the gm179 pattern). The reason says to
give the allowable worked out from Fy and the corroded section: Eq. 3-11a,
0.55 Fy for strips and 0.48 Fy for bar mats. The `Reinforcement` dataclass
keeps `Fy`, since the built-ins record theirs. Tests:
`funhouse_agent/tests/test_gm179_unused_inputs.py`.

## GEC-11's printed strip and geogrids as the built-ins (GM-267, GM-271, 2026-10-21)

Every built-in reinforcement is now one a GEC-11 Vol II worked example
prints, with its Tal as printed and the pages in its basis, which every
result carries in the Tal choice and `list_reinforcement` gives as
`Tallowable_basis`. E4's 4W11 mat was the first (GM-262, above).

**The strip (GM-267): `ribbed_steel_strip_50x4`.** Example E3's strip,
"Grade 65 (Fy = 65 ksi), 1.969 in. x 0.157 in. (50 mm wide x 4 mm)", Ec =
0.102 in. after 75 years, Tn = 65 ksi x 0.200 in.^2 = 13.00 k a strip
(p. E3-16); GEC-11 Vol I p. 2-16 gives the 50 mm strip as the one
commercially available. Tal = 0.55 Fy Ec (p. 4-77; Ac = b Ec, Fig. 3-5a,
Eq. 3-11a) = 638.6 kN/m of strip width, the same per metre of width as the
75 mm strip had, and 31.93 kN a strip against E3's 0.55 Tn = 7.15 k =
31.80 kN on its rounded 0.200 in.^2 (0.4 %). Tal per strip is Tal per unit
width x b, so with Rc = b/Sh the tensile resistance per metre of wall is
the strip's strength / Sh.
- The old `ribbed_steel_strip_75x4` (75 mm, the tool's own width) is
  refused by the tool, naming the 50 mm strip and its Tal, or a custom
  `metallic_strip` with its Tal and width. `RIBBED_STEEL_STRIP_75x4` stays
  importable for the retired `foundry/` wrapper only.
- The Tal choice no longer offers the 50 mm strip as the other reading,
  since it is now the built-in. The `_COMMERCIAL_STRIP` alternative in
  `mse.tensile_strength_choice` is gone with it.
- Every strip pin moves by 50/75 = 2/3 (Rc 0.1 -> 0.0667 at Sh 0.75 m).
  The 9 m strip wall (H 9 m, L 7 m, Sv 0.75 m, gamma 20, phi 34, Cu 4, Sh
  0.75 m): FOS_pullout 2.876 to 4.210 -> **1.917 to 2.807**; FOS_rupture
  1.455 to 24.06 -> **0.970 to 16.04**, Tal x Rc 63.86 -> 42.57 kN/m of
  wall. The bottom level (8.62 m, Tmax 43.89 kN/m) now **fails rupture**;
  the choice gives a closer spacing per level, as E3 takes 3 strips a
  panel below Level 8.
- **E3 reproduced with the built-in** (V-056,
  `test_v056_the_built_in_strip_is_E3s`): at E3's Sh per level, its Pr at
  all 12 levels of Table E3-7.3 (p. E3-19) within 0.3 %, and FOS_rupture
  is E3's tension CDR times 0.55 x 1.35 / 0.75 (0.99) and 31.93 / 31.80.
  The published-example pins are unchanged.

**The geogrids (GM-271): `geogrid_gec11_e1_gg1`, `_gg2`, `_gg3`.** The two
built-in geogrids carried one maker's product-line names, and their Tal
(21.9 and 29.2 kN/m) had no recorded source. GEC-11 prints a full Tal =
Tult / RF for generic geogrids in Example E1, Table E1-7.3 (p. E1-15):

| Built-in | Tult (lb/ft) | RFID | RFCR | RFD | Tal (lb/ft) | Tal (kN/m) |
|---|---|---|---|---|---|---|
| GG-I | 3,000 | 1.3 | 1.85 | 1.15 | 1,085 | 15.83 |
| GG-II | 6,000 | 1.3 | 1.85 | 1.15 | 2,169 | 31.66 |
| GG-III | 9,000 | 1.2 | 1.85 | 1.15 | 3,525 | 51.45 |

`reinforcement.geosynthetic_Tal` is Eq. 3-12 (Vol I p. 3-28). For pullout
E1 takes F* = 0.45 and alpha = 0.8 "for these geogrids", C = 2, Rc = 1
(Step 7.8, p. E1-16); the module takes F* = 0.67 tan phi for a geogrid
(Eq. 3-8, p. 3-21), 0.452 at E1's 34 deg, and alpha 0.8 (Table 3-6, p.
3-17). A product's own Tult (its minimum average roll value from ASTM
D6637 tests, p. 3-28) and reduction factors for the wall's backfill and
design life are a custom reinforcement (`reinforcement_type` 'geogrid'
with `reinforcement_Tallowable`).
- The reduction factors E1 prints: RF = RFID x RFCR x RFD = 2.77 for GG-I
  and GG-II (RFID 1.3) and 2.55 for GG-III (RFID 1.2); RFCR 1.85 and RFD
  1.15 for all three. E1 gives no reason for GG-III's lower RFID. Each
  basis now prints the RF beside its factors (review, 2026-10-21).
- The print disagrees with itself: Table E1-7.1 (p. E1-11) lists the
  grades' ultimate strength as 6,000, 9,000 and 12,000 lb/ft, Table E1-7.3
  as 3,000, 6,000 and 9,000, so Table E1-7.1's values are Table E1-7.3's
  one grade up. Dividing Table E1-7.1's Tult by Table E1-7.3's Tal gives
  5.5, 4.1 and 3.4, which are not RFs E1 prints anywhere; they only show
  the two tables' mismatch. E1's Tal and its rupture CDRs (Table E1-7.4)
  follow Table E1-7.3, and so do the built-ins; the basis says so.
- The built-ins are worked-example values, not products (review,
  2026-10-21). The tool description says so, `list_reinforcement` returns
  a `built_ins` note saying so, and the Tal choice of every result with a
  built-in carries the reinforcement specified for the wall as its other
  way: a custom reinforcement with the product's own Tal, Fy Ac / b at
  0.55 Fy or 0.48 Fy (Eq. 3-11a, p. 4-77) for steel, Tult / RF (Eq. 3-12)
  for a geosynthetic (`mse._SPECIFIED_PRODUCT`).
- `geogrid_ux1600` and `geogrid_ux1700` are refused by the tool, giving
  E1's three geogrids with their Tal, or a custom geogrid Tal = Tult / RF.
  `GEOGRID_UX1600` and `GEOGRID_UX1700` stay importable for the retired
  `foundry/` wrapper only, which imports them; no tool text names them.
- Not taken: Example E9 (p. E9-2) uses a 22 kN/m geogrid, but in a
  reinforced slope, not a wall.
- The 6 m geogrid wall (L 5 m, Sv 0.6 m, gamma 19, phi 34) with GG-I:
  FOS_pullout unchanged (8.394 to 20.633; pullout does not take Tal);
  FOS_rupture 0.862 at the bottom (1.192 on 21.9 kN/m), below 1.5 from 3.3
  m down. With GG-II every level passes (lowest 1.723).
- V-057 checks that each built-in is Table E1-7.3's Tal to the pound, and
  that a whole run with the built-in GG-I is the run with E1's GG-I.

Tests: `tests/test_mse_coverage_ratio.py`,
`tests/test_mse_rupture_factor_of_safety.py`,
`funhouse_agent/tests/test_mse_builtin_reinforcement_adapter.py`, V-056
and V-057.

## A reinforcement per level (GM-266, 2026-10-21)

GEC-11 Vol II Example E4 changes its bar mats with depth (Table E4-7.4,
p. E4-18): 4W11 + W11x0.5' at Levels 1, 3 and 4, 3W11 + W11x0.5' at
Level 2, 4W15 + W11x1' at Levels 5 to 7 and 4W15 + W11x1.5' at Levels 8
to 10. Example E1 takes GG-I at Layers 1 to 4 and GG-II below (Table
E1-7.4, p. E1-16). Before GM-266 the module took one reinforcement for
the wall.

**The input.** `analyze_mse_wall` and `check_internal_stability` take
one `Reinforcement` for every level, as before, or a list of them, one a
level from the top down (`mse.level_reinforcements`). The tool's
`reinforcement_per_level` gives one entry a level: a built-in name, or a
custom reinforcement `{Tallowable, type, width, thickness,
transverse_spacing, name}` (the keys of the `reinforcement_*` inputs
without the prefix, which is also taken). `reinforcement_horizontal_spacing`
and `coverage_ratio` apply level by level, one number or a list (GM-261).
`reinforcement_name`, `reinforcement_width` and the custom
`reinforcement_*` inputs are refused beside it, and a list given as
`reinforcement_name` points to it.

**Refused, with what to give instead:**
- a list whose length is not the number of levels, naming the levels'
  depths, `wall_height` and `reinforcement_spacing` (the module puts the
  levels at Sv (i + 1/2));
- steel at some levels and geosynthetic at others, the tool's choice
  rather than a rule GEC-11 states: GEC-11 Vol I p. 4-33 takes the
  failure surface as "approximately bilinear in the case of inextensible
  reinforcements" and "approximately linear in the case of extensible
  reinforcements" (Fig. 4-9, p. 4-34), through the toe "in both cases",
  and says nothing of a wall with both, so the internal checks would have
  no surface (`mse.ONE_KIND_PER_WALL`);
- an entry with its own spacing or ratio (that is the list input), an
  unknown key (did-you-mean), a built-in with custom values, a custom
  entry with no Tallowable or no type; each message names the entry and
  its level.

**Each level takes its own** Tal, width, Rc, Kr/Ka curve (strip 1.7 to
1.2, grid 2.5 to 1.2), F* (strip 1.2 + log Cu to tan phi; grid 20 t/St
to 10 t/St, its own t/St) and alpha (Table 3-6 by its type). A level run
in a list is the same as the wall run with that level's reinforcement.
Each level of `internal_results` names its reinforcement, the summary's
level lines carry the name, and `reinforcement_name` in the tool result
is the list of names. The choices give Tal, Rc, alpha and the empirical
F* by runs of levels ("Tal per level, from the top: ... at levels 1, 3
...; ... at level 2"); the Tal choice is "module" when every level is a
built-in. The calc package keys each level to a letter (A, B, ...) in
the inputs and the per-level table, with the names and Tal in the notes,
and draws each reinforcement with its own line style.

**E4's other mats are built-ins** (`reinforcement.py`), as E4 prints
them: `bar_mat_e4_3w11_w11x0_5` (3 W11 longitudinal wires, b = 1 ft,
Tal 108.5 kN/m of mat width), `bar_mat_e4_4w15_w11x1` and
`bar_mat_e4_4w15_w11x1_5` (4 W15 wires, b = 1.5 ft, Tal 138.4 kN/m; W11
transverse wires at 1 ft and 1.5 ft, so t/St = 0.374 / 12 and 0.374 /
18). Wires at Sl = 0.5 ft (p. E4-17), D75 0.318 in. (W11) and 0.381 in.
(W15) after 75 years (p. E4-14), Fy 65 ksi and t = 0.374 in. (p. E4-15),
Tal = 0.48 Fy Ac N / b (Eq. 3-11a, p. 3-22; p. 4-77), the rigid facing's
phi 0.65 as for the 4W11 mat.

**E4 reproduced, mats per level** (`validation_examples/
test_published_v009_v011.py`, `test_v010_e4_mats_per_level`), at E4's
level depths (at GM-266 a test geometry put the levels at the printed Z,
each taking Sv = 2.5 ft; since GM-273 they are given as `level_depths`,
and the FOS_rupture table below is GM-266's, before it: GM-273's section
has E4 at its contributory heights). F*, Le and phi_p Pr match Table
E4-7.4 to 0.3 % at all ten levels (F* 0.438 to 0.208 at Levels 5 to 10,
the W15 mats' 1 ft and 1.5 ft transverse spacings). FOS_rupture (Tal x Rc
over the unfactored Tmax, ASD) against Ng phi_s Tn / Tmax (LRFD), the
ratio the table implies:

| Level | Mat (Table E4-7.4) | FOS_rupture | Ng phi_s Tn / Tmax |
|---|---|---|---|
| 1 | 4W11+W11x0.5' | 2.438 | 2.163 |
| 2 | 3W11+W11x0.5' | 1.192 | 1.213 |
| 4 | 4W11+W11x0.5' | 1.044 | 1.059 |
| 5 | 4W15+W11x1' | 1.344 | 1.358 |
| 8 | 4W15+W11x1.5' | 1.215 | 1.208 |
| 10 | 4W15+W11x1.5' | 1.018 | 1.016 |

Levels 2 to 10 within 2 %. Level 1 is 13 % over, from the top level's
tributary depth alone: E4 gives Level 1 the wall from the top down to
midway to Level 2, Svt = 3.12 ft (Table E4-7.1, p. E4-11), and averages
sigma_H over it (p. E4-13); the module gives every level Sv = 2.5 ft and
sigma_H at the level's depth. 3.12 / 2.5 = 1.248 times E4's average over
0 to 3.12 ft against the module's at 1.87 ft, 0.910, is 1.136, against
1.127 from the printed 6.25 k and 3.38 k (GM-273 takes the level depths
as an input). With the 4W11 mat at every
level the check fails from Level 5 down, 0.709 at Level 10 against E4's
4 x 3.38 / 19.05 = 0.71 (`test_v010_e4_level10_needs_its_w15_mat`).

**Not covered at GM-266** (closed by GM-273, below): the module took
one Sv, so a wall's level depths were Sv (i + 1/2); E4's (1.87 ft, then
2.5 ft apart) and E1's were reproduced only through a test geometry.

Tests: `tests/test_mse_per_level_reinforcement.py`,
`funhouse_agent/tests/test_mse_per_level_reinforcement_adapter.py`,
V-010, the probes `retaining_walls|mse_wall#per_level` and `#geogrid`.

**Review (2026-10-21).** E4's mats checked on the pages: W11 and W15
D75 0.318 and 0.381 in. (p. E4-14), Fy 65 ksi, Tn 5.17 and 7.42 k a wire,
W11 transverse wires at t = 0.374 in. and St 6, 12 and 18 in. (p.
E4-15), Table E4-7.3's mat by level (p. E4-16), Sl = 0.5 ft (p. E4-17).
One defect, older than GM-266 (GM-256): with `alpha_tested` left out on a
grid or a geosynthetic, `defaults_used` gave the alpha choice's sentence
("alpha 0.8, GEC-11 Vol I Table 3-6's default value for ...") in place of
the input's value; the GM-158 guard gives a run's `defaults_used` back as
inputs, so it handed that sentence back as `alpha_tested` and the call
failed, which is why the per-level probe had been given alpha_tested 1.0.
`alpha_tested` is an alpha from pullout tests, so the table's value given
back under that name would claim a test: `defaults_used` now gives the
declared case, "Table 3-6's alpha for the reinforcement type" with its
basis (as a strip wall already did), each level gives the value it took
(`alpha`), and the alpha choice names the value and Table 3-6's row. The
per-level probe runs without alpha_tested, and a geogrid probe covers the
0.8 row. No number changed.

## The levels' depths as an input (GM-273, 2026-10-21)

**What GEC-11 prints.** Vol I (FHWA-NHI-10-024), read on the rendered
pages:

- p. 4-46: with equally spaced levels, "sigma_H, calculated at the level
  of the reinforcement, is at the center of the contributory height. The
  contributory height is defined as the midpoint between vertically
  adjacent reinforcement elevations, except for the top and bottom layers
  reinforcement."
- p. 4-47: "For the top and bottom layers of reinforcement, Sv is the
  distance from top or bottom of wall, respectively, to the midpoint
  between the first and second layer (from top or bottom of wall,
  respectively) of reinforcement" (Fig. 4-14). TMAX for those layers,
  "and for intermediate layers that do not have equally spaced adjacent
  layers, is calculated as the product of the contributory height and the
  average factored horizontal stress acting upon that contributory height.
  The average stress can be calculated based upon the tributary
  trapezoidal area (i.e. average of the stress at top and at the bottom of
  the contributory height) or at the midpoint of the contributory height."

Vol II (FHWA-NHI-10-025): Examples E3 and E4 tabulate Svt = Z+ - Z-
(Table E3-7.1, p. E3-13; Table E4-7.1, p. E4-11) and take sigma_H as the
average of its values at Z- and Z+ at every level (p. E3-15, Level 4:
sigma_H-surcharge = 0.5 (0.38 + 0.36) ksf; p. E4-13), Kr/Ka at each. E1
takes its layers' TMAX at Z, Layer 1's at ZAve 0.835 ft, the middle of its
1.67 ft (p. E1-14); for a geogrid (Kr/Ka = 1) the trapezoid and the
midpoint agree, and for Layers 2 to 9 the midpoint is Z.

**What the module does.** `MSEWallGeometry(level_depths=...)` takes the
levels' depths below the top of the wall, from the top down, as the
examples print Z; each must lie inside the wall (0 < z < H) and below the
one above, else it is refused with what to give. `contributory_heights`
gives each level's (z-, z+): the midpoints between levels, 0 for the top
level and H for the bottom. `check_internal_stability` and
`analyze_mse_wall` then take Tmax = sigma_H x Svt
(`contributory_Tmax`), sigma_H averaged over Svt by `sigma_H_average`:
'trapezoidal' (the default, as E3 and E4) or 'midpoint'. A sloping
backfill's sigma_2 and a live load enter sigma_H as before. Each level
also gives `Svt_m` and `sigma_H_kPa` (unfactored), and the result's
`choices` say how Svt was taken and which average, with the other
(`level_depths_choices`). A reinforcement list (GM-266) or a spacing or
coverage ratio per level (GM-261) of another length than the depths is
refused, naming the depths and that `level_depths` gives them.
`sigma_H_average` without depths is refused (the levels at Sv take sigma_H
at the level). Pullout is unchanged: it does not depend on Svt.

The tool (`mse_wall`) takes `level_depths` (alias `reinforcement_depths`)
and `sigma_H_average` (default 'trapezoidal', applies only with depths);
`reinforcement_spacing` with `level_depths` is refused, since the depths
place every level. Left out, `level_depths` is declared (GM-165) as "the
levels at reinforcement_spacing (i + 1/2) below the top, wall_height /
reinforcement_spacing of them rounded down, each taking Sv and sigma_H at
its depth", a trial layout, so `defaults_used` lists it.

**Without depths nothing moved.** Strip, bar mat (E4's wall and mats per
level), geogrid and a sloping strip wall through `analyze_mse_wall`: the
results, summaries and calc-package steps are byte-identical to master.
The tool's result differs only by the `level_depths` line in
`defaults_used`.

**For the owner.** The even layout (no depths) left the strip of wall
between the bottom level's Sv and the base unassigned when H is not a
whole number of spacings: E4's wall at Sv 2.5 ft (H 25.64 ft) has its last
level at 23.75 ft carrying 22.5 to 25.0 ft, so 0.64 ft went to no level,
and its first level sits at 1.25 ft where E4's is at 1.87 ft (Level 1
Tmax 4.69 k a panel against E4's 6.25 k). It was kept at GM-273 for
byte-identity; GM-277 (below) gives the bottom level the strip.

**E4 reproduced at its depths** (`test_v010_e4_in_one_run_at_its_depths`,
`test_v010_e4_mats_per_level`): Tmax x 1.35 over a 5 ft panel against
Table E4-7.4 (p. E4-18), and FOS_rupture against Ng phi_s Tn / Tmax:

| Level | Z (ft) | Svt (ft) | Tmax (k) | Table | FOS_rupture | Ng phi_s Tn / Tmax |
|---|---|---|---|---|---|---|
| 1 | 1.87 | 3.12 | 6.24 | 6.25 | 2.145 | 2.163 |
| 2 | 4.37 | 2.50 | 8.36 | 8.36 | 1.201 | 1.213 |
| 3 | 6.87 | 2.50 | 10.80 | 10.80 | 1.239 | 1.252 |
| 4 | 9.37 | 2.50 | 12.76 | 12.77 | 1.049 | 1.059 |
| 5 | 11.87 | 2.50 | 14.24 | 14.26 | 1.349 | 1.358 |
| 6 | 14.37 | 2.50 | 15.23 | 15.23 | 1.261 | 1.271 |
| 7 | 16.87 | 2.50 | 15.73 | 15.71 | 1.221 | 1.232 |
| 8 | 19.37 | 2.50 | 16.03 | 16.03 | 1.199 | 1.208 |
| 9 | 21.87 | 2.50 | 17.08 | 17.10 | 1.124 | 1.132 |
| 10 | 24.37 | 2.52 | 19.03 | 19.05 | 1.009 | 1.016 |

Tmax within 0.2 % at every level (Level 1 was 13 % over through the test
geometry). FOS_rupture is about 1 % under at every level: the ASD Tal 0.48
Fy against E4's phi_s Tn / 1.35 = 0.481 Fy, and the table's spreadsheet
3.38 and 4.84 k a wire against 3.36 and 4.82 by hand (p. E4-18). With
'midpoint' Tmax is within 1.7 % (Level 1 6.36 k; Level 8 1.3 % under,
where its Svt spans the end of Kr/Ka at 20 ft). The 4W11 mat at Level 10
gives 0.703 (E4: 4 x 3.38 / 19.05 = 0.71).

**E1 and E3 at their depths.** E1's eleven layers with their grades
(`test_v057_e1_in_one_run_at_its_layers`): the contributory heights are
Table E1-7.4's Sv (1.67, 2.0, 1.33, 1.0 ft); TMAX x 1.35 within 0.3 % of
Table E1-7.2 at Layers 1 to 9 and 0.6 % at Layer 11; Layer 10 is 1.4 %
over (1,386 against 1,366 lb/ft), where E1 takes Z = 18.67 ft, not the
middle of 17.67 to 19.0 ft, and prints less than either; its rupture CDR
is 1.5 % under, the others within 0.6 %; La, Le and Zp give Table E1-7.5's
pullout CDRs within 1.1 %. E3's twelve levels (`test_v056_e3_at_its_
depths`) are the module's own at Sv 2.5 ft (H 30 ft), each Svt 2.5 ft; the
trapezoid gives Table E3-7.3's Tmax within 0.4 % and sigma_H at the level
within 0.25 %, the two differing by the curvature of Kr/Ka times depth.
No test geometry is left: E4's `_E4Levels` subclass is gone, and E1 and E3
never had one (their per-layer tests are hand primitives at the printed Z
and Sv).

**Review (2026-10-21).** Checked by hand, formulas written out apart
from the module: a strip wall under a 15 degree slope (sigma_2 = gamma
0.35 H tan beta, constant with depth) at uneven depths, both averages,
and a level bar-mat wall with a 12 kPa live load whose contributory
heights straddle 20 ft. Tmax agrees at every level to rounding (0.03 %).
Where Kr/Ka changes with depth sigma_H is not linear over Svt; the
trapezoid then reads within about 1 % of the true mean of sigma_H over
Svt (under it above 20 ft, 0.7 % over it at the level that straddles
20 ft) and the midpoint within 0.3 % above it: the approximation p. 4-47
offers. Where a geosynthetic wall's FOS_rupture is short, the choices now
name the input in use for closer levels: level_depths when the depths are
given (reinforcement_spacing would be refused beside them).

Tests: `tests/test_mse_level_depths.py`,
`funhouse_agent/tests/test_mse_level_depths_adapter.py`, V-010, V-056,
V-057, the probe `retaining_walls|mse_wall#depths`.

## The even layout's bottom level and its count (GM-277, 2026-10-21)

Without `level_depths` the module places the levels at Sv (i + 1/2) below
the top, H / Sv of them rounded down. Two defects, both on the unsafe
side, found by the GM-273 review:

- **A dropped level.** `int(H / Sv)` on a wall a whole number of spacings
  high fell one short where floating point puts H / Sv under the whole
  number: 4.8 m at Sv 0.8 m is 5.999..., so five levels and the bottom
  0.8 m of wall in none. 22 such heights up to 20 m at Sv 0.4, 0.8 and
  0.9 m, and in feet (as the metres they convert to) 7 ft at 1 ft, 7.5
  ft at 1.5 ft, 14 ft at 2 ft, 15 ft at 3 ft and others. The count now
  adds a tolerance of 1e-9 of a spacing (`geometry.
  WHOLE_SPACING_TOLERANCE`), and `whole_spacings` says whether the wall
  is a whole number of spacings high.
- **The bottom strip in no level.** When H is not a whole number of
  spacings the bottom level took Sv, so the wall below n Sv was in no
  level's Tmax.

**What GEC-11 prints** (Vol I, read on the rendered pages; the GM-273
section quotes them): equally spaced levels take Sv, with sigma_H at the
level "at the center of the contributory height" (p. 4-46); the top and
bottom layers' Sv runs "from top or bottom of wall, respectively, to the
midpoint between the first and second layer" (p. 4-47), Fig. 4-14 drawing
Sv(1) from the midpoint above the bottom layer down to the base, and their
TMAX is the contributory height times the average stress over it, on the
trapezoid or at the midpoint (p. 4-47).

**What the module does now.** `contributory_heights` works for both
layouts: without depths, i Sv to (i + 1) Sv, the bottom level's from
(n - 1) Sv down to H. Every level but the bottom one is unchanged (Sv,
sigma_H at the level, which is the middle of its contributory height).
The bottom level of a wall that is not a whole number of spacings takes
Tmax = sigma_H x (H - (n - 1) Sv), sigma_H at the middle of that height
(`contributory_Tmax(..., 'midpoint')`): the midpoint, so the one average
the whole layout uses, and for steel above 6 m, where sigma_H is concave
in depth, the higher of p. 4-47's two. One level (H under 2 Sv) takes the
whole wall. Pullout, La and Le are unchanged (they do not depend on the
contributory height). A spacing that puts the first level at or below
the base (Sv / 2 >= H) is refused; it had put a level outside the wall.

**The top level** at Sv/2 already matched the print: its contributory
height from the top of the wall to the midpoint between it and the
second level is Sv, the level at its middle, so sigma_H at the level is
p. 4-47's midpoint average. Nothing moved there.

**Said in the result.** `level_depths_choices` gives the even layout a
choice ("Tmax at each level, the levels equally spaced at
reinforcement_spacing ..."): how many levels at what Sv, the top level at
Sv/2, each level's contributory height, and for a wall that is not a
whole number of spacings the bottom level's from the midpoint above it to
the base with its height and the depth sigma_H is taken at; the basis
quotes pp. 4-46 and 4-47 and Fig. 4-14; the other option is the depths as
designed (`level_depths`), with the trapezoid as E3 and E4. The summary
prints it with the other choices; the calc package's Tmax step says where
the bottom level's contributory height ends and cites Eq. 4-32a, pp.
4-46, 4-47 (it had cited "Eq. 4-10", which is the sloping-backfill
thrust). The tool's `defaults_used` case for `level_depths` and the
`reinforcement_spacing` description say the same.

**What moved** (every other number is byte-identical; a sweep of 882
even-layout walls, 14 heights x 7 spacings x 3 reinforcements x level /
surcharged / sloping, moved only the bottom level, always up, or added
the dropped level):

- the bottom level's Tmax, FOS_pullout and FOS_rupture on every wall that
  is not a whole number of spacings: Tmax up by 2.6 % to 130 % in the
  sweep, the most where the strip is nearly a spacing;
- the dropped level restored on walls a whole number of spacings high
  (one more level; the levels above unchanged);
- `test_mse_bar_mat_strength.py`, E4's wall at Sv 2.5 ft (the module's
  layout, not E4's): the bottom level at 23.75 ft takes 22.5 to 25.64 ft,
  3.14 ft, sigma_H at 24.07 ft: Tmax 39.84 -> 50.66 kN/m, FOS_rupture
  0.726 -> 0.571 (the 32 kN/m Tal 0.241 -> 0.189), the flexible-facing
  reading's lowest 0.832 -> 0.654. At Sv the bottom level carried less
  than the 41.2 kN/m E4 prints for its own Level 10 over 2.52 ft (19.05 /
  1.35 k a 5 ft panel, Table E4-7.4, p. E4-18);
- the published examples did not move: E1, E3 and E4 at their depths,
  V-009, V-010, V-011, V-021, V-022, V-056, V-057 (the even-layout E4
  checks there are at the top level, and V-057's walls at Sv 2 ft are a
  whole number of spacings).

Tests: `tests/test_mse_even_layout.py` (the count over heights and
spacings in metres and feet, the 32 walls that lost a level, the bottom
level for strips, mats and geogrids, level, surcharged and sloping, the
top level, one level, the refusal, the choice and the calc package),
`tests/test_mse_level_depths.py`,
`funhouse_agent/tests/test_mse_level_depths_adapter.py`.

## Key Notes
- CantileverWallGeometry auto-sizes the base thickness, toe and stem from
  the wall height, H/10 each, and B = 0.6 H (FHWA NHI-06-089 Fig. 10-16(a);
  GM-288, below)
- Cantilever checks decompose the active thrust per the chosen method (RW-1):
  Coulomb thrust acts at the wall friction on the plane through the heel,
  the backfill slope β by default (GM-288; it was 2/3·φ), sloped-Rankine
  thrust parallel to the slope (β). Only P_h = Pa·cos(δ) drives
  sliding/overturning; P_v = Pa·sin(δ) is applied downward at the back of
  the heel (x = B) and is stabilizing. Rankine + level backfill is
  unchanged (δ = 0).
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
  - **Source basis (built-in reinforcements).** Since GM-262, GM-267,
    GM-271 and GM-266 each built-in is a GEC-11 Vol II worked example's
    reinforcement as printed, with its pages in its basis: E3's 50 x 4 mm
    strip, E4's four bar mats, E1's geogrids GG-I to GG-III (sections
    above). They are not a
    manufacturer's catalog; for a real design the user supplies the actual
    product's certified properties as a custom reinforcement. The Kr/F* CURVES
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
the defaults were 0.566 / 1.339 / 0.978. GM-288 (concrete 23.6, Coulomb's
delta = beta = 0 on the vertical plane, the tension zone not counted) made
them 0.392 / 0.945 / 0.945.

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

**What a left-out wall friction means, by theory (GM-257, 2026-10-21).**
The tool declared `delta_deg` with one default, 0.0, which is Coulomb's
and misdescribed 'caquot_kerisel' (0.66 phi' with the UFC sources, refused
with the 1948 chart and Eq. 4-12 by name). It now declares no default and
a case by theory (`declare_when_absent`): Coulomb takes delta = 0, no wall
friction; caquot_kerisel takes 0.66 phi' with the UFC sources, and the
other two refuse the call. The description says the same, and `delta_deg`
applies to both theories (`applies_when`). Each run lists the value it
took in `defaults_used`: 0 for Coulomb, 0.66 phi' (19.8 at phi' 30) for
the UFC sources; Rankine and K0 take none and list none. No number
changes. Tests: `funhouse_agent/tests/test_earth_pressure_sources.py`,
the probes `retaining_walls|earth_pressure_coefficient#coulomb` and
`#coulomb_passive`.

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

## Checked against GEC-6 and UFC 3-220-20 as printed (GM-288, 2026-10-31)

Every equation, default and limit of the cantilever checks was read
against GEC-6 ch. 5-7, UFC 3-220-20 Sec. 4-5.1 and App. B-1, FHWA
NHI-06-089 Sec. 10.5 and EN 1997-1 6.5.3, the GM-282 way; the item by item
record is `module_work/gm288/CHECK.md`, the hand checks
`tests/test_gm288_printed_check.py`, and two printed worked examples are
V-061 (UFC App. B-1 and NHI Example 10-3, both to the printed digit or
within the examples' rounded Ka). Takes in GM-32 and GM-235.

Fixed to the print:

- Trial proportions: base thickness, toe and stem base H/10 (NHI Fig.
  10-16(a)); they were 0.60 m, 0.1 B and 0.08 H + 0.2 m.
- Concrete 23.6 kN/m3 (150 pcf, UFC App. B-1.3; NHI Example 10-3); it was
  24.
- Coulomb's wall friction on the vertical plane through the heel is the
  backfill slope (NHI p. 10-20, soil on soil); it was 2/3 phi. At that
  default Coulomb's Ka is Rankine's sloped Ka, so `pressure_method` moves
  the result only with `wall_friction_deg` given.
- The tension zone: no negative active pressure (UFC Fig. 4-5 Note A; NHI
  p. 10-15). Pa = 0.5 Ka gamma (H - z0)^2 below z0; the choices give c
  neglected and NHI's 5z kN/m2 minimum as alternatives.
- The soil over a battered back face and over the toe counts as resisting
  weight (UFC Fig. 4-27). New `stem_batter` ("back" or "front", the
  examples' wall) and `embedment_depth` (Df, default the base thickness).
- The passive resistance acts over Df (with a key from Df to Df + key
  depth), and counts cu under the undrained rule (UFC Eq. 4-7, p. 181).
- Required FS: sliding 2.0 with the passive resistance counted (UFC Fig.
  4-27, Sec. 4-5.1.3); overturning 2.0 on soil, 1.5 on rock (NHI p. 10-44).
- `founded_on_rock`: e <= B/4 in place of B/6 (GEC-6 6.4.1; UFC 4-5.1.2).
- The bearing pressure compared with q_allowable is the uniform q_eq =
  V / (B - 2e) on soil (GEC-6 Eq. 6-6; NHI Fig. 10-17; UFC Eq. 4-27), the
  trapezoid's q_max on rock (`bearing_pressure` "linear").
- A descending backfill is refused (GM-32); it was taken as level.

Added as an option: EN 1997-1 6.5.3(12)P's Rd <= 0.4 Vd for an undrained
clay where water or air can reach the base (`interface_gap_possible`).

Noted, kept: UFC Fig. 4-27 prints F_OT = W a / (P_H b - P_V f); the module
counts P_V f as a resisting moment, the smaller FOT when P_V > 0 and the
same eccentricity. GEC-6 Eq. 5-41's a1b (toe to the key) is taken as 0.

Not modelled: water behind the wall (UFC 4-4.1; NHI 10.5.3); the backfill
is taken as drained, as the tool's brief now says.
