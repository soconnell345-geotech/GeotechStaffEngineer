# retaining_walls — Cantilever & MSE Wall Design

## Purpose
Stability analysis for cast-in-place cantilever retaining walls and
mechanically stabilized earth (MSE) walls per GEC-11.

## References
- FHWA GEC-11 (Mechanically Stabilized Earth Walls)
- AASHTO LRFD Bridge Design Specifications
- Coulomb/Rankine earth pressure theory

## Files
- `geometry.py` — CantileverWallGeometry (auto-sizing), MSEWallGeometry
- `earth_pressure.py` — reuses sheet_pile Ka/Kp + resultant force computation
- `cantilever.py` — sliding, overturning, bearing, eccentricity checks
- `mse.py` — Kr/Ka ratio, Tmax, pullout, F*, external + internal stability
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
    over 0–6 m; F* 2.0→tan(φ) over 0–6 m (GEC-11 Fig. 4-10/4-11).
  - steel bar-mat / welded-grid ("bar_mat"/"welded_grid"/"metallic_grid"):
    Kr/Ka 2.5→1.2 over 0–20 ft (6.096 m); F* 20(t/St)→10(t/St) over 0–20 ft,
    where t = transverse-bar diameter and St = transverse-bar spacing
    (GEC-11 Fig. E4-5). The grid F* takes t/St from the `Reinforcement`
    geometry (`thickness`/`transverse_spacing`) or the `t_over_St` arg; if a
    grid lacks that geometry F* falls back to the strip curve.
  - geosynthetic ("geosynthetic"): Kr/Ka 1.0; F* 0.67·tan(φ).
  `check_internal_stability` auto-selects the bar-mat curves for a
  `metallic_grid` reinforcement (e.g. `WELDED_WIRE_GRID_W11`). The strip default
  is byte-identical to the pre-Q4 behavior.
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
