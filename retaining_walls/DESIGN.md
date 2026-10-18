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
  factors of safety (resistance/demand, live load folded into the mass weight).
  This is the historical default and is unchanged.
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

`mse.check_external_stability` (the ASD path) still counts the surcharge
over the reinforced zone in sliding and overturning (W = gamma H L + q L),
against GEC-11 pp. 4-19 / 4-22; the LRFD path does not. Not changed here.

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
  - steel bar-mat / welded-grid ("bar_mat"/"welded_grid"/"metallic_grid"):
    Kr/Ka 2.5→1.2 over 0–20 ft (6.096 m); F* 20(t/St)→10(t/St) over 0–20 ft,
    where t = transverse-bar diameter and St = transverse-bar spacing
    (GEC-11 Fig. E4-5). The grid F* takes t/St from the `Reinforcement`
    geometry (`thickness`/`transverse_spacing`) or the `t_over_St` arg; if a
    grid lacks that geometry F* falls back to the strip curve.
  - geosynthetic ("geosynthetic"): Kr/Ka 1.0; F* 0.67·tan(φ).
  `check_internal_stability` auto-selects the bar-mat curves for a
  `metallic_grid` reinforcement (e.g. `WELDED_WIRE_GRID_W11`). Q4 left the
  strip default unchanged.
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
