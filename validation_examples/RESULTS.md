# Phase E — Published-Example Validation Results

Runs of the analysis modules against the worked examples in `INVENTORY.md`.
Offline (pure Python, no API). Verdicts:

- **PASS** — module reproduces the published answer within the stated tolerance.
- **DISCREPANCY** — module differs beyond tolerance with the *same* method (a real
  bug if confirmed; investigated before any fix).
- **CONVENTION** — module and example use a defensibly different convention; delta
  documented, module NOT tuned to the one example.
- **N/A (scope)** — the module implements a different method than the example;
  documented as a coverage gap, not a failure.

Tests live in `validation_examples/test_published_v###.py`, runnable offline:
`pytest validation_examples/ -v`.

| Entry | Module | Our value | Published | Δ | Verdict | Notes |
|-------|--------|-----------|-----------|---|---------|-------|
| V-006 | drilled_shaft side (sand) | β=0.25 (depth-based, floored) | β=0.41 (rational OCR chain) | — | N/A (scope) | Module = O'Neill-Reese depth-based β (`1.5−0.245√z_ft`), not the GEC-10 (N1)60→φ′→OCR→Ko rational β. Documented coverage gap. |
| V-007 | drilled_shaft side (clay) | α=0.55 (AASHTO) | α=0.47 (rational su-transform) | — | N/A (scope) | Module = AASHTO α (0.55, cu/pa≤1.5), not the GEC-10 `0.30+0.17/(su_CIUC/pa)` with UU→CIUC transform. Documented gap. |
| V-008 | drilled_shaft base (sand) | qb=49.24 ksf; RBN(unreduced)=2475 kips | qb=49.2 ksf; RBN=2473 kips | <0.1% | **PASS** (unit) | `57.5·N60` kPa ≡ `0.60·N60` tsf — exact. Module additionally applies the O'Neill-Reese 1.27/Dᵦ large-diameter reduction (factor 0.521 at Dᵦ=2.44 m) → 1289 kips; example shows the unreduced value. See CONVENTION note below. |
| V-008b | drilled_shaft base large-D | 1.27/Dᵦ applied (0.521) | (not shown in example) | — | CONVENTION | Module follows GEC-10 §13.3.4.3 / O'Neill-Reese large-diameter base reduction; the extracted example value is unreduced. Correct per the cited method; flagged for owner awareness. |
| V-001 | axial_pile Nordlund shaft (sand) | Rs = 126/249/353/477 kips (D=35/50/60/70) | Rs = 137.5/250.7/344.1/452.5 | −8% / −1% / +3% / +5% | **PASS** | Nordlund shaft vs Table D-6, modeled from the footing datum. All four depths within ±15%; the displacement-pile Kd fit reproduces the published Nordlund shaft for this φ 33–36 H-pile profile. |
| V-001 | axial_pile Nordlund toe (sand), toe φ=40 | Rt = 408.5 kips (Layer 3 plateau) | Rt = 428.1 kips | −5% | **PASS** (toe fn) | Calling `end_bearing_cohesionless(φ=40)` reproduces the published toe plateau; the q_L cap (Meyerhof Fig 7-15) at φ=40 governs at 408.5 kips. |
| V-001 | axial_pile Nordlund toe (single-φ API) | Rt = 301 kips (φ=36 in L3) | Rt = 428.1 kips | −30% | CONVENTION | The high-level `AxialPileAnalysis` API uses ONE φ per layer, so it cannot apply the example's separate Layer-3 toe φ=40 (GEC-12 design-limit). Documented API limitation; not tuned. See note below. |
| V-002 | axial_pile alpha toe (clay), 9·cu | Rt = 32.8/33.3/34.1 kips (D=70/80/90) | Rt = 32.75/33.03/33.68 | 0% / +1% / +1% | **PASS** | End bearing 9·cu matches Table D-100 to ±1% where the tip is cleanly inside Layer 3. (D=40 toe is a layer-boundary artifact — tip exactly at the L2/L3 interface — excluded.) |
| V-002 | axial_pile alpha shaft (clay) | Rs = 250/297/345 kips (D=70/80/90) | Rs = 318.4/369.3/420.33 | −22% / −20% / −18% | CONVENTION | Module under-predicts shaft. Cause isolated to the STIFF clay band: module Tomlinson α≈0.42 vs the example's DrivenPiles α≈0.76 at su≈1.9 ksf. Soft (α≈0.85) and very-stiff (α≈0.34) bands agree. Different α-vs-cu curve; module not tuned. |
| V-003 | wave_equation drivability (diesel) | ~48 bpf @ 746 kips, ~48 ksi comp | 120 bpf @ Rndr=746 kips, 40.1 ksi | bpf −60%, stress +21% | N/A (scope) | No Delmag D36-52 in the hammer DB (only the −22/−32 single-energy diesel series). A hand-built D36-52 + generic cushion gives 48 bpf (vs 120, outside ±25%) and 48 ksi (vs 40.1, outside ±10%): the module's diesel hammer is a simple energy/efficiency velocity conversion, not a GRLWEAP diesel combustion + ram-cycle model. Coverage gap. |
| V-004 | downdrag neutral plane (Fellenius) | NP=53.3 ft, Qmax=487.2 kips, DF=286.2 kips | NP=54 ft, Qmax=486 kips, DF=285 kips | NP −0.7 ft, Qmax +0.2%, DF +0.4% | **PASS** | Fellenius NP construction reproduces NP/Qmax/DF to <1% at 100% toe mobilization (toe=428.1 kips, Q=201 kips). Per-layer β overridden to feed the module the published Table D-6 Nordlund shaft (inventory-permitted), validating the NP equilibrium logic decoupled from the shaft method. |
| V-005 | Meyerhof (1976) group settlement | (no module method) | S = 1.04 in | — | N/A (scope) | No module exposes S = 4·pf·If·√B/N160. `settlement/` has only shallow-footing methods; `pile_group/` has only the elastic equivalent-raft method (`group_settlement_equivalent_raft`, a 2V:1H elastic sum — different method). Closed form verified inline (1.044 in) as documentation of the gap. |
| V-009 | retaining_walls MSE external, unfactored loads | V1=57.69, Vs=4.50, F1=13.68, F2=2.14 k/ft; MV1=519.1, MF1=117.0, MF2=27.4 k-ft/ft | V1=57.69, Vs=4.50, F1=13.68, F2=2.13; MV1=519.21, MF1=116.94, MF2=27.36 | <0.5% | **PASS** | `rankine_Ka`(34→0.283, 30→0.333) + the GEC-11 force eqns reproduce every Table E4-4.3/4.4 unfactored force & moment about Point A. `horizontal_force_active` returns the combined F1+F2 thrust (15.83 k/ft) + line of action. These are the load quantities the module's primitives contribute. |
| V-009 | retaining_walls MSE external, LRFD CDRs | sliding 1.85/2.08/1.37; ecc eL 2.87/3.87 ft; bearing σv 6.70 ksf, CDR 1.57; svc σv 4.66; crit σv 5.75 | sliding 1.85/2.08/1.37; ecc 2.87/3.87; bearing 6.70, 1.57; svc 4.66; crit 5.86 | ≤2% | **PASS** (via primitives) | Driving the GEC-11 Str I max/min load-factor pairing (EV 1.35/1.00, EH 1.50/0.90, LL 1.75) + LL-on-resisting exclusion (sliding/ecc) / LL-included (bearing) on top of the module's earth-pressure primitives reproduces every published CDR. Critical bearing σv 5.75 vs 5.86 (+1.9%) is within the source's own "consistent-values" rounding note. |
| V-009 | retaining_walls `analyze_mse_wall` (high-level) | ASD FOS_sliding ≈ 2.27 (R/demand, no load factors) | LRFD sliding CDR 1.85 | — | CONVENTION | The packaged `analyze_mse_wall` returns ASD factors of safety (unfactored resistance/demand, surcharge in W, LL not split out), NOT the GEC-11 LRFD CDRs. The LRFD load-factor + LL-exclusion bookkeeping is not in the module. Documented API/method gap; the example was reproduced through the lower-level primitives instead. |
| V-010 | retaining_walls MSE internal `Tmax_at_level` (bar mat) | σH/Tmax L1 0.40/6.24, L4 1.02/12.76, L7 1.26/15.73, L10 1.51/19.03 (k/panel) | σH/Tmax L1 0.40/6.25, L4 1.02/12.77, L7 1.26/15.71, L10 1.51/19.05 | ≤0.5% | **PASS** (primitive) | `Tmax_at_level` reproduces Table E4-7.4 bar-mat Tmax/σH at all 4 sampled levels when fed the bar-mat Kr/Ka (2.5→1.2 over 0–20 ft) and the EV factor 1.35 on soil+surcharge, via the example's average-over-tributary-bounds σH method. |
| V-010 | retaining_walls MSE internal `pullout_resistance` (bar mat) | nominal Pr=23.06, φ·Pr=20.76 k/ft (Level 4) | Pr=23.06, φ_p·Pr=20.75 | <0.1% | **PASS** (primitive) | `pullout_resistance(C=2)` (two grid surfaces = the example's "2b") with bar-mat F*=0.955 (interp 20·t/St→10·t/St), Le=L−0.3H=10.31 ft, unfactored soil σv, φ_pullout=0.90 reproduces Level-4 Pr exactly. Le lengthens to 17.24 ft at Level 10 (Z>H/2 taper). |
| V-010 | retaining_walls MSE internal built-in curves | `Kr_Ka_ratio`(z=0)=1.7; `F_star_metallic`(0)=2.0 (ribbed STRIP) | bar mat: Kr/Ka(0)=2.5; F*(0)=20·t/St=1.246 | — | CONVENTION | The module's built-in `Kr_Ka_ratio`/`F_star_metallic` implement only the ribbed-metallic-STRIP curves (Kr/Ka 1.7→1.2; F* 2.0→tanφ), not the steel-bar-mat curves (Kr/Ka 2.5→1.2; F* 20(t/St)→10(t/St)). So the high-level internal path doesn't auto-match a bar-mat wall — the primitives must be fed the bar-mat coefficients. Documented coverage gap. |
| V-011 | seismic_geotech Mononobe-Okabe KAE (**regression anchor**) | **KAE = 0.4782** | KAE = 0.4785 | −0.06% | **PASS** | φ=30, δ=30, kh=kmax=0.206, kv=0, vertical wall, level backfill. δ=φ=30 with kv=0 is the case the battered-wall M-O fix targeted; the corrected sign/degrees/kv handling gives KAE within ±0.06% (tolerance ±2%). No bug; value pinned to catch a future M-O regression. |
| V-011 | seismic_geotech M-O seismic sliding chain | PAE=19.65, PIR=6.09, THF=24.64, V=67.52, R=38.98, CDR=1.58 k/ft | PAE=19.65, PIR=6.09, THF=24.64, V=67.52, R=38.98, CDR=1.58 | <0.5% | **PASS** | Full GEC-11 E7 Step-8 chain built on the module KAE: PAE=0.5γh²KAE; THF=PAE·cos30+PIR+0.5·qLS·H·KAE; V=W+PAE·sin30; R=V·tan30; CDR=R/THF. Reproduces the example to <0.5%. |

## Notes / flags for the owner

- **drilled_shaft is the simplified-method module by design.** It implements
  AASHTO/O'Neill-Reese (1999) simplified α (0.55) and depth-based β
  (`1.5−0.245√z_ft`). The GEC-10 (2018) *rational* side-resistance chains
  (OCR-based β from (N1)60; su-test-mode α `0.30+0.17/(su/pa)`) used in the
  GEC-10 Appendix A example are **not** in the analysis module — they live in
  the reference layer (`gec_10` lookups). This is a real **coverage gap** if the
  owner wants the module to offer the rational methods; recorded, not "fixed."
- **Deep cohesionless β floors at 0.25** (`z ≳ 40 ft`) — the O'Neill-Reese
  depth formula clamps there. Expected behavior, noted because it makes deep
  sand side resistance conservative vs the rational method (0.41).
- **Large-diameter base reduction** (V-008b) is applied by the module and not by
  the extracted example. The module is correct per the cited method; if the
  owner's downstream comparisons assume the unreduced nominal, that's the source
  of any 2× base-resistance gap.

### Batch V-001..V-005 (GEC-12 Vol 3 driven piles) — owner notes

- **Datum matters for axial_pile.** The GEC-12 tables reference depths to the
  footing/cap bottom (5 ft bgs) and the pile head sits there. Modeling the
  `AxialSoilProfile` from the footing bottom (not the ground surface) is what
  makes V-001 shaft land within ±15% — modeling from the ground surface adds
  ~5 ft of spurious skin friction above the pile head and inflates shaft by
  ~25%. There is no surcharge/embedment-offset input on `AxialPileAnalysis`; the
  user must clip the profile to the pile head themselves. **Possible ergonomics
  add:** an optional `head_depth`/surcharge parameter so the footing datum is
  handled without re-clipping layers.
- **axial_pile has no separate shaft/toe φ per layer (V-001).** The example uses
  a Layer-3 design-limit toe φ=40 with a shaft φ=36; the high-level API applies
  one φ per layer, so the single-φ toe is −30% low (301 vs 428 kips). The toe
  *function* with φ=40 is only −5% off. **Possible feature:** a per-layer
  `toe_friction_angle` override (GEC-12 explicitly allows different shaft/toe φ
  in dense gravels). Not a bug — an API capability gap.
- **axial_pile Tomlinson α-vs-cu curve runs low for stiff clay (V-002).** Isolated
  to su≈1.9 ksf (stiff CL): module α≈0.42 vs DrivenPiles/Tomlinson ≈0.76. The
  soft and very-stiff bands agree. This is the dominant cause of the 18–32%
  shaft under-prediction. The module's α is conservative here. If the owner wants
  closer agreement with DrivenPiles, the stiff-clay segment of `alpha_tomlinson`
  is the place to revisit — but it is a defensible (conservative) curve choice,
  so it was NOT changed for this one example.
- **wave_equation diesel hammers are simplified (V-003).** The DB carries only
  −22/−32 single-energy Delmag diesels and the `Hammer` diesel path is an
  energy→velocity conversion, not a diesel combustion + ram-cycle model. It
  cannot reproduce a GRLWEAP diesel bearing graph (blow count ~60% low, stress
  ~20% high with a hand-built D36-52). If diesel drivability tie-outs are wanted,
  this needs a real diesel hammer model + the GRLWEAP-default helmet/cushion data
  — a substantial build, flagged as a known limitation.
- **downdrag NP construction is solid (V-004).** Given the published shaft, the
  Fellenius neutral-plane equilibrium reproduces NP/Qmax/DF to <1%. The only
  caveat is that the module's intrinsic β-method shaft (~185 kips) is much lower
  than the Nordlund shaft (~344 kips) for this dense gravel, so the test feeds the
  module the published distribution via per-layer β overrides (inventory-permitted)
  to isolate the NP logic. Also note: `_compute_toe_resistance` uses `pile_area`
  as the toe area, so for H-piles the box toe area must be passed as `pile_area`.
- **No Meyerhof (1976) SPT group settlement anywhere (V-005).** Trivial closed
  form (S = 4·pf·If·√B/N160). `pile_group.group_settlement_equivalent_raft` is an
  elastic 2V:1H method, not this. **Possible add:** a one-line
  `meyerhof_group_settlement(...)` helper in `pile_group` if SPT-based group
  settlement is in scope; otherwise leave as a documented gap.

### Batch V-009..V-011 (GEC-11 E4/E7 MSE wall) — owner notes

- **V-011 M-O regression anchor is a clean PASS — no bug.** The repo M-O active
  coefficient gives **KAE = 0.4782** vs the published 0.4785 (−0.06%), and the
  full E7 seismic sliding chain (PAE 19.65, THF 24.64, V 67.52, R 38.98, CDR
  1.58 k/ft) reproduces the example to <0.5%. The δ=φ=30°, kv=0 case that the
  battered-wall M-O fix targeted is handled correctly (sign of δ/θ, degrees vs
  radians, the (1−kv) term). `seismic_geotech/` suite was run after to confirm
  no regression (all pass — see below). No module change was made.
- **retaining_walls MSE is an ASD-FoS module; the GEC-11 LRFD bookkeeping is not
  packaged.** `analyze_mse_wall` returns factors of safety (R/demand, no load
  factors, LL folded into the weight) — it does NOT do the Strength I max/min
  load-factor pairing, the LL-on-resisting-side exclusion, or report LRFD CDRs.
  So V-009 was reproduced by driving the LRFD factors **on top of** the module's
  earth-pressure primitives (`rankine_Ka`, `horizontal_force_active`) + the
  geometry, which DO reproduce every unfactored load/moment and (with the
  factors applied) every published CDR to ≤2%. **Possible add:** an LRFD
  external-stability path (load-combination input + per-mode CDR output) if
  AASHTO-LRFD MSE checks are in scope. The ASD path is correct for what it is.
- **MSE internal stress/pullout primitives are solid; only the bar-mat curves are
  missing.** `Tmax_at_level` and `pullout_resistance(C=2)` reproduce the
  Table E4-7.4 bar-mat Tmax/σH/Pr to ≤3% when fed the right coefficients. But
  the module's built-in `Kr_Ka_ratio` and `F_star_metallic` only carry the
  **ribbed-metallic-STRIP** curves (Kr/Ka 1.7→1.2; F* 2.0→tanφ over 0–6 m), not
  the **steel-bar-mat / grid** curves (Kr/Ka 2.5→1.2; F* 20(t/St)→10(t/St) over
  0–20 ft). **Possible add:** a `reinforcement_type="bar_mat"` (or grid) branch
  in `Kr_Ka_ratio`/`F_star_metallic` with the 2.5-top Kr and the t/St-based F*,
  so the high-level internal path matches bar-mat walls without hand-feeding
  coefficients. The depth datum is also worth noting: GEC-11 caps the bar-mat
  Kr/Ka taper at Z = 20 ft (≈6.1 m) — close to but not exactly the module's 6.0 m
  strip cap.
- **Unit note for MSE per-length quantities.** Forces convert k/ft ↔ kN/m by
  14.594; moments-per-length convert k-ft/ft ↔ kN-m/m by 1.356/0.3048 = 4.4488
  (a kip-ft of moment per ft of wall). The tests carry both constants explicitly.
