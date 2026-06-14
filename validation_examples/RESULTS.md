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
| V-025 | sheet_pile earth-pressure builder (layered Ka + water) | Ka1=0.249, Ka2=0.333; σ pts 129.5/173.2/377.8/644.2 psf, u=1248; FTOTAL=24,611 lb/ft | Ka1=0.249, Ka2=0.333; 129.48/173.16/377.76/644.16/1248; FTOTAL=24,610.9 | 0.00% | **PASS** | `rankine_Ka` + `active_pressure` reproduce every stress ordinate at the Ka-discontinuity (4-ft boundary, same overburden × Ka1 vs Ka2) and the layered-moist/submerged + hydrostatic water blocks. Total driving force exact to 0.00% with source-rounded Ka (≈359 kN/m). |
| V-012 | sheet_pile cantilever Ka/Kp (soldier pile) | Ka=0.271, Kp=3.69 | Ka=0.271, Kp=3.69 | exact | **PASS** (primitive) | `rankine_Ka`/`rankine_Kp`(φ=35) match the published coefficients exactly. |
| V-012 | soldier-pile arching / simplified toe-moment | (not in module; hand: f=2.8, fb=5.6 ft, D0=12.27, D=14.73 ft, Y=6.0, Mmax=379.7 kip-ft, Vmax=137.7 kips) | D0=12.27, D=14.73, Y=6.0, Mmax=379.7, Vmax=137.7 | <0.1% (hand) | N/A (scope) | `analyze_cantilever` is a CONTINUOUS per-metre free-earth-support solver — no soldier-pile effective-width/arching (active on 2-ft hole width, passive on f·b=0.08φ·b=5.6 ft) and no simplified toe-moment cubic. The published Simplified row is reproduced by hand on the module Ka/Kp to document the method. |
| V-012 | analyze_cantilever (continuous wall) | D_conv ≈ 11.3 ft (per-metre framework) | D0=12.27 ft (per-pile) | — | CONVENTION | Continuous-wall embedment is a different framework than the per-pile simplified D0 (active+passive spread over a continuous wall, no 2-ft/5.6-ft widths). Not comparable; pins the framework gap. |
| V-013 | sheet_pile anchored Rankine Ka | Ka=0.333 | Ka=0.333 | exact | **PASS** (primitive) | `rankine_Ka`(φ=30) matches exactly. |
| V-013 | passive coefficient source | Rankine Kp=3.0 (−36%); Coulomb Kp(δ=15)=4.98 (+6%) | log-spiral Kp=4.7 (Caquot-Kerisel 6.3×0.746) | — | CONVENTION | No Caquot-Kerisel log-spiral passive in the module; published Kp=4.7 lies between Rankine (3.0) and Coulomb (4.98), matching neither tightly. Module not tuned. |
| V-013 | FHWA apparent-diagram anchor quantities | P=11,980; PT=15,574; σ_a=934.4 psf; T1U=6,228; TH=143.9; T=148.95 kips | P=11,980; PT=15,574; σ_a=934.4; T1U=6,228; TH=143.87; T=148.95 | <1.5% (hand) | **PASS** (by hand) | The single-anchor 1.3× apparent trapezoid, max ordinate, upper-tributary anchor force, per-anchor TH (10-ft spacing) and inclined T reproduce by hand on the module Ka. Mmax 22,494 ft-lb/ft ≈ 100 kN-m/m. |
| V-013 | analyze_anchored (classical FES) | D ≈ 10.8 ft (Rankine FES) | D=6.09 ft (apparent + log-spiral Kp) | — | N/A (scope) | `analyze_anchored` is classical free-earth-support with a TRIANGULAR Rankine/Coulomb active diagram (not the FHWA 1.3× apparent trapezoid) and Rankine/Coulomb passive (no log-spiral). Embedment/anchor/moment differ by method, not bug. |
| V-014 | basal heave — Caltrans force balance | (hand: qu=3.8 ksf, F_RS=40.0, W=37.8, +q 3.15, −S 15.0, F_dr=26.0, FS=1.54) | F_RS=40.0, F_dr=26.0, FS=1.54 | <0.5% (hand) | N/A (scope) | Caltrans balances qu·(0.7B) against the block W + 0.7B·q − S with sidewall shear S=c·H. Reproduced exactly by hand. |
| V-014 | check_basal_heave_bjerrum_eide | FOS=0.86 (cu·Nc/(γH+q)); module Nc=6.71 | FS=1.54; chart Nc=7.6 | — | N/A (scope) | Module uses the inverted-footing bearing FOS = cu·Nc/(γH+q) — NO sidewall-shear term, NOT the 0.7B force block — so FOS=0.86 (far more conservative). Its Bjerrum-Eide Nc table reads 6.71 at H/B=2, Be/Le=1/3 vs the Caltrans chart's 7.6. Different formulation. |
| V-016 | soe Ka + FHWA pe + ps | Ka=0.295, pe=43.6, ps=3.2 | Ka=0.295, pe=43.6, ps=3.2 | <1% | **PASS** | `soe.rankine_Ka`(φ=33)=0.295; two-anchor trapezoid pe=0.65·Ka·γ·H²/(H−H1/3−H3/3)=43.6; ps=Ka·qs=3.2. |
| V-016 | tributary anchor loads + R + DL | TH1=168.4, TH2=172.1, R=36.7, DL1=435, DL2=445 | TH1=168, TH2=172, R=37, DL1=435, DL2=445 | <2% | **PASS** | Tributary-area horizontal anchor loads, subgrade reaction R=(3/16)H3·pe+(H3/2)ps, and anchor design loads DL=TH·s/cos15 all reproduce to ≤2%. |
| V-016 | hinge moments M1 / M2,3 | M2,3=66 (exact); M1=70.4 (compact form); M1_earth=65.6 | M2,3=66; M1=Mmax=76 | M2,3 <1%; M1 −7.4% | **PASS** (M2,3) / CONVENTION (M1) | M2,3=(1/10)H2²(pe+ps)=66 exact. M1: the inventory's compact (13/54)H1²(pe+ps)=70.4 vs published 76 — GEC-4 applies the uniform surcharge over a larger tributary than the apparent term in the top region. Dominant earth part (13/54)pe·H1²=65.6 exact; delta is the surcharge tributary only (within 8%). |
| V-015 | slope_stability slice table (pinned circle) | θ_i, W_i exact; ΣW·sinθ=116.49, ΣW·cosθ=137.09 kips/ft | ΣW·sinθ=116.49, ΣW·cosθ=137.09 | 0.00% | **PASS** | The pinned circle (center 0,60 ft directly above the toe, R=60 ft) + the 4V:3H face reproduce the Caltrans Table 10-1/10-2 slice table EXACTLY — all six θ_i=asin(x_i/60), weights W_i, and both force sums. Geometry reconstruction verified before any FS comparison. |
| V-015 | slope_stability Fellenius (OMS) | 0.863 (module pipeline, x=0→57.6, ΣdL→77.2 ft); 0.874 (published formula w/ source L=113.55) | 0.87 | −0.007 / +0.004 | **PASS** | `fellenius_fos` over the geometrically-correct mass converges to 0.863 (6 slices 0.866); feeding the source's hand arc length L=113.55 ft into the published formula gives 0.874. Both within ±0.05. The only spread is the cohesion term: source's hand L=113.55 ft over-states the discretized base length (true lower-arc 0→57.6 ft is 77.2 ft). No module change. |
| V-015 | slope_stability Simplified Bishop | 0.9595 (rigorous fixed point) | 0.90 (source, under-iterated) | +6.6% | CONVENTION | `bishop_fos` returns the proper fixed point FS=FSa=0.9595 on the source's own 6-slice table. The manual shows only two hand iterations (FSa=1.5→1.10; FSa=0.8→0.90, reproduced EXACTLY here incl. the Hb=m_α column) and declares "converges to ≈0.9" — one step short. Re-iterating the source's OWN table to convergence gives 0.96, matching the module. Module is the more-correct value; published 0.90 is not converged. |
| V-017 | lateral_pile FD solver (verification) | fixed-head y & M_head match Reese-Matlock T-method to 4 figures (linear p=k·z·y) | (analytical T-method) | <1% | **PASS** | With a linear soil law the beam-column solver reproduces 0.93·V·T³/EI (fixed-head groundline y) and ≈−0.93·V·T (head moment) exactly. Isolates the solver + fixed-head BC (slope=0) as correct, so the V-017 deflection excess is a p-y construction difference, not a solver bug. |
| V-017 | lateral_pile fixed-head Mmax (Reese sand, axial) | −39.3 kN·m (at head) | −37.3 kN·m | +5.4% | **PASS** | `SandReese` static p-y, 2-layer profile (φ=32/30, k=24430/16287 kN/m³, γ′=18.84/17.64), head 0.305 m below grade (overburden +0.305 m), V=44.482 kN, slope=0, axial P=1423.4 kN with P-delta. Mmax within ±10%. |
| V-017 | lateral_pile fixed-head deflection | 3.92 mm | 3.3 mm | +19% | CONVENTION | Just outside ±15%. Solver verified exact (row above); the excess is the module's documented chart-free `SandReese` simplification (LP-1: 1/3-power parabola anchored at ultimate) being softer than LPILE's full Reese (1974) construction with the B-factor m-point. Not a bug, not tuned. Axial P-delta correctly raises y 3.60→3.92 mm and Mmax −36.8→−39.3. |
| V-018 | ground_improvement / settlement unimproved consolidation | S = 22.0 in (settlement module + closed form) | S = 22 in | 0.0% | **PASS** | `consolidation_settlement_layer` (NC clay, Cc=0.25, eo=0.7, H=15 ft, po=432 psf, dq=2500 psf) reproduces S=Cc/(1+eo)·H·log10((po+dq)/po) to the printed digit (0.559 m). Tight closed-form anchor, independent of any GI method. |
| V-018 | ground_improvement `area_replacement_ratio` (RAP) | as_tri = 0.2744 | Ra = Ac/(π/4·de²) = 0.27 | <0.5% | **PASS** (primitive) | The triangular tributary √3/2·s² numerically equals the de=1.05·s unit cell π/4·(1.05s)² (both 21.65 ft²), so the module's geometric `as` matches the source's de-based Ra to 4 figures. |
| V-018 | RAP improved settlement (upper-zone stiffness modulus qg/kg) | hand: qg=6324 psf, suz=0.68 in (module `as`); module SRF path gives ~9.4 in | qg=6383 psf, suz≈0.7 in | suz <3% (hand) | N/A (scope) | The GEC-13 two-layer upper-zone method (qg=q·ns/(Ra·ns−Ra+1), suz=qg/kg, kg=pier stiffness modulus pci) is NOT in the module. `analyze_aggregate_piers`/`improved_settlement` use the equal-strain SRF=1/(1+as(n−1)) model (no kg) → SRF=0.43, improved ≈9.4 in — a fundamentally different method. Published 0.68 in reproduced by hand on the module `as`; module not tuned. |
| V-019 | ground_improvement / settlement unimproved consolidation | S = 27.2 in (settlement module + closed form) | S = 27 in | <1% | **PASS** | `consolidation_settlement_layer` (Cc=0.2, eo=0.6, H=50 ft, po=1440 psf, dq=1875 psf) reproduces S=27.16 in (0.690 m) exactly. Tight anchor. |
| V-019 | ground_improvement `priebe_basic_improvement_factor` (stone column) | n0=3.06 (as=Ac/A=0.277, φ_col=42.5); n0=2.81 (module geom as=0.251, φ_col=42.5) | chart ratio 2.7 (Fig 5-27) | +13% / +4% | CONVENTION | Module exposes the genuine Priebe (1995) n0. Published 2.7 is a CHART read; the factor depends on the area-ratio convention AND φ_col. With the source's as=Ac/A=0.277 + default φ_col=42.5 → 3.06 (outside ±0.3); with the module's geometric triangular as=0.251 + φ_col=42.5 → 2.81 (WITHIN 2.7±0.3); the chart 2.7 ≈ φ_col 39° at as=0.277. Improved S=27/n0 = 8.8–9.6 in brackets the published 10 in. Formula correct, not tuned. |
| V-020 | ground_improvement wick drains (Barron-Hansbo radial-only, ideal) | t90 = 299 d (s=8 ft), 503 d (s=10 ft) | ~300 d / ~500 d ("on the order of") | <1% | **PASS** | RADIAL-ONLY ideal Barron (F(n)=ln(n)−0.75, no smear/well resistance) via `influence_diameter`(de=1.05s) + `drain_function_F` + `time_for_radial_consolidation` reproduces both published t90 to <1% (well inside ±20%). de(8 ft)=2.560 m, de(10 ft)=3.200 m, dw=0.0635 m (2.5 in), n=40.3/50.4. |
| V-020 | wick drains combined vertical+radial (convention check) | U_total ≈ 93.1% at s=8 ft, t=300 d (Ur≈90%, Uv≈31%) | (radial-only target 90%) | — | CONVENTION | The packaged combined model U_total=1−(1−Uv)(1−Ur) over-predicts U at the published times — the 20-ft clay over rock (single drainage, Hdr=20 ft) adds ~31% vertical on its own, so the combined t90 would be SHORTER. Documents that the source used radial-only; the module supports both. Not a bug. |

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

### Batch V-012..V-025 (Caltrans T&S + GEC-4 SOE / sheet-pile) — owner notes

- **V-025 is the cleanest SOE/sheet-pile check (PASS).** The `sheet_pile`
  earth-pressure builder (`rankine_Ka` + `active_pressure`) reproduces the
  Caltrans Ex 7-2 layered-Ka stress-point diagram and the total driving force
  (24,611 lb/ft ≈ 359 kN/m) to 0.00 % with the source-rounded Ka. The two-sided
  stress point at the 4-ft layer boundary (same overburden × Ka1=0.249 above vs
  Ka2=0.333 below) and the moist→submerged transition + hydrostatic water all come
  out right. No module change.
- **V-012 — the soldier-pile *Simplified method* is not in the module (N/A-scope),
  but the coefficients are exact (PASS).** `rankine_Ka`/`rankine_Kp`(φ=35) give the
  published 0.271 / 3.69 exactly. The Caltrans simplified method, however, puts
  ACTIVE pressure on the 2-ft hole width and PASSIVE pressure on an
  arching-amplified width f·b = (0.08·φ)·b = 5.6 ft, then solves a simplified
  toe-moment cubic — none of which lives in `analyze_cantilever`, which is a
  continuous per-metre free-earth-support solver. The published Simplified row
  (D0=12.27, D=14.73 ft, Y=6.0, Mmax=379.7 kip-ft, Vmax=137.7 kips) is reproduced
  by hand on the module coefficients to document the method. **Possible add:** a
  soldier-pile mode (active-width / passive-arching-width inputs + the AASHTO
  simplified toe-moment) if soldier-pile SOE is in scope. Note the inventory's
  D=13.53 "rigorous" row would need the conventional cantilever method (also not the
  module's continuous solver).
- **V-013 — single-anchor wall: classical FES vs FHWA apparent diagram (N/A-scope)
  + log-spiral passive gap (CONVENTION).** Rankine Ka=0.333 matches the module
  exactly, and the FHWA apparent-diagram quantities (P, PT=1.3P, σ_a=934.4 psf,
  upper tributary T1U=6,228, per-anchor TH=143.9, inclined T=148.95 kips) reproduce
  by hand to <1.5 %. But `analyze_anchored` implements CLASSICAL free-earth-support
  on a TRIANGULAR Rankine/Coulomb active diagram (not the 1.3× apparent trapezoid)
  and has no Caquot-Kerisel log-spiral passive — published Kp=4.7 sits between
  Rankine (3.0, −36 %) and Coulomb δ=15 (4.98, +6 %). So the packaged
  embedment/anchor/moment differ by method, not by a bug. **Possible adds:** an
  FHWA apparent-diagram single-anchor path, and a log-spiral (Caquot-Kerisel)
  passive option. Module not tuned to the example.
- **V-014 — basal heave: module method differs from Caltrans (N/A-scope).** The
  module DOES have `check_basal_heave_bjerrum_eide`, but it computes the
  inverted-footing bearing ratio FOS = cu·Nc/(γH+q) — with NO sidewall-shear term
  and NOT the 0.7B force block — returning FOS≈0.86 here (far more conservative).
  Its Bjerrum-Eide Nc table also reads 6.71 at H/B=2, Be/Le=1/3, vs the Caltrans
  chart's 7.6. The Caltrans force-balance (resistance qu·0.7B vs driving block
  W + 0.7B·q − S, S=c·H side shear → FS=1.54) is reproduced exactly by hand. The two
  are simply different basal-heave formulations. **Possible add:** a Caltrans-style
  force-balance heave option with the side-shear term and the 0.7B block, if that
  convention is wanted; the current bearing-ratio method is defensible (and more
  conservative), so it was NOT changed.
- **V-016 — GEC-4 two-tier anchored wall is largely PASS via primitives.**
  `soe.rankine_Ka`(33)=0.295, the FHWA trapezoid pe=43.6, surcharge ps=3.2, both
  tributary anchor loads (TH1=168, TH2=172), the subgrade reaction R=37, the lower
  hinge moment M2,3=66, and both anchor design loads (DL1=435, DL2=445) all
  reproduce to ≤2 %. The **one** soft spot is the upper hinge moment M1: the
  inventory's compact form (13/54)·H1²·(pe+ps)=70.4 vs the published 76 kN-m/m,
  because GEC-4 applies the uniform traffic surcharge over a larger tributary in the
  top region than the apparent-pressure term. The dominant earth part
  (13/54)·pe·H1²=65.6 is exact; the −7.4 % is purely the surcharge tributary
  (CONVENTION, within 8 %). No module change — these are all hand-built on the
  module's Ka because the FHWA multi-anchor envelope / tributary-load / hinge-moment
  bookkeeping is not packaged as a single `soe` entry point. **Possible add:** a
  multi-anchor apparent-envelope helper (pe, TH_i, M_i, R) in `soe`.
- **No module bugs found or fixed in this batch.** V-012/013/014 are method/scope
  gaps (soldier-pile arching, FHWA apparent diagram + log-spiral passive,
  Caltrans force-balance heave with side shear); V-016 M1 is a surcharge-tributary
  convention; V-025 and the module coefficient/earth-pressure primitives are clean
  PASSes. `soe/` and `sheet_pile/` suites were not modified.

### Batch V-015 / V-017 (Caltrans slope FOS + Micropile LPILE) — owner notes

- **V-015 slope_stability is a clean PASS on the hard part (the geometry + the
  Fellenius/Bishop methods themselves), and surfaces a SOURCE error in the
  published Bishop value.** Driving the module's slice builder with the *pinned*
  circle (center 0,60 ft directly above the toe, R=60 ft) and the 4V:3H face
  reproduces the Caltrans Table 10-1/10-2 slice table to the printed digit — every
  θ_i=asin(x_i/60), every weight, and both sums ΣW·sinθ=116.49 / ΣW·cosθ=137.09
  kips/ft. On that mass:
  - **Fellenius** `fellenius_fos` = 0.863 over the geometrically-correct mass
    (x=0→57.6 ft, where the full circle crosses the face), vs the published 0.87
    (Δ −0.007, well inside ±0.05). The published 0.87 itself is recovered exactly
    (0.874) by feeding the source's hand arc length **L=113.55 ft** into the
    published formula. That L is a hand over-estimate — the true lower-arc length
    from the toe to x=57.6 is **77.2 ft** (the module's self-consistent ΣdL); the
    c′·L cohesion term is the entire ~0.05 spread. The module's smaller, self-
    consistent L is the more accurate one.
  - **Bishop** `bishop_fos` = **0.9595**, the rigorous fixed point (FS=FSa) on the
    source's own 6-slice table — vs the published **0.90**. The manual prints only
    two hand iterations (FSa=1.5→FS=1.10; FSa=0.8→FS=0.90; reproduced here EXACTLY,
    including the m_α=cos θ+sin θ·tanφ/FSa = "Hb" column 1.06/1.15/1.21/1.23/1.20/
    1.06) and loosely declares "converges to ≈0.9" — **one hand step short of
    convergence.** Iterating the source's OWN table to FS=FSa gives 0.96, matching
    the module. So the module is correct and the published 0.90 is under-iterated;
    recorded as CONVENTION (the +6.6% is the source's truncated iteration, not a
    module error). **No module change.**
  - *Geometry-representation note:* this circle's center sits directly above the
    toe, so its lower arc never re-exits the slope on its own (max z=60 ft at
    x=60 ft, while the face there is z=80 ft) and the source's stated exit
    (x=57.6 ft, z=76.8 ft) is on the *upper* arc. `CircularSlipSurface` only models
    the lower arc, so `build_slices`/`find_entry_exit` cannot auto-terminate the
    mass at 57.6 ft for this specific circle. The test therefore builds the slice
    discretization over x=0→57.6 directly and exercises `fellenius_fos`/`bishop_fos`
    on it (the FOS methods are the validation target; they are exact). A general
    auto-handling of toe-centered circles whose exit is above the center elevation
    would need the slip surface to expose the upper arc too — a minor possible
    enhancement, not required here.
- **V-017 lateral_pile: the FD solver is verified EXACT; the one out-of-tolerance
  number (deflection +19%) is the documented Reese p-y simplification, not a bug.**
  - With a linear soil law p=k·z·y the beam-column solver reproduces the Reese-
    Matlock characteristic-length (T-method) closed form to 4 figures for the
    fixed-head groundline deflection (0.93·V·T³/EI) and head moment (≈−0.93·V·T).
    That pins the solver, the fixed-head BC (slope=0), and the discretization as
    correct, so the deflection excess can only come from the p-y curve.
  - With `SandReese` static curves + the LPILE inputs, **fixed-head Mmax = −39.3
    kN·m vs −37.3 (+5.4%, PASS within ±10%)** and **head deflection 3.92 mm vs 3.3
    (+19%, just outside ±15%)**. The deflection overshoot is the module's own
    chart-free `SandReese` simplification (LP-1 in `py_curves.py`: a 1/3-power
    parabola anchored at the ultimate point) being a bit softer than LPILE's full
    Reese (1974) three-part construction with the B-factor m-point and variable
    exponent. Recorded as CONVENTION — not tuned to this one example. (For a
    smoother curve, `SandAPI` gives 5.0 mm / −44 kN·m — also soft; neither matches
    LPILE's exact Reese curve to <15% on deflection.)
  - **Axial P-delta is correctly captured:** P=1423.4 kN raises the fixed-head
    deflection from 3.60→3.92 mm and Mmax from −36.8→−39.3 kN·m. The published case
    has P-delta on, so the axial result is the comparison value.
  - **Head-datum subtlety (flagged):** the LPILE echo's "ground surface 0.305 m
    below top of pile = −0.30 m" and the manual text ("the cap will be embedded
    0.305 m below the ground surface") both mean the head is **0.305 m BELOW grade**
    — i.e. +0.305 m of soil overburden over the head. The INVENTORY note "0.305 m
    above ground (stickup)" is a misreading of the sign; modeling it as an
    above-grade stickup gives ~9 mm (far off). The test honors the below-grade
    overburden with a thin `SandReese` depth-offset wrapper. **Possible ergonomics
    add:** an optional `head_depth` / overburden-offset parameter on
    `LateralPileAnalysis.solve()` so a below-grade pile head needs no wrapper (the
    existing `stickup` only handles the above-grade case).
- **No module bugs found or fixed in this batch.** slope_stability's slice builder +
  Fellenius/Bishop and lateral_pile's FD solver are all verified correct; the gaps
  are a source under-iteration (V-015 Bishop) and a documented p-y construction
  simplification (V-017 deflection). `slope_stability/` and `lateral_pile/` suites
  were run unchanged (109 lateral_pile pass; slope_stability pass).

### Batch V-018..V-020 (GEC-13 ground_improvement) — owner notes

- **The unimproved consolidation anchors are clean PASSes (V-018, V-019).** Both
  GEC-13 examples open with a baseline consolidation S = Cc/(1+eo)·H·log10((po+dq)/po),
  and the `settlement` module's `consolidation_settlement_layer` (NC clay,
  sigma_p=sigma_v0) reproduces both to the printed digit in SI: **22.0 in** (V-018,
  0.559 m) and **27.2 in** (V-019, 0.690 m). These are the tight checks; the
  improvement factors are the module-specific (looser) targets.
- **V-018 — the RAP upper-zone *stiffness-modulus* settlement method is NOT in the
  module (N/A-scope), but the area-ratio primitive is exact (PASS).** GEC-13's RAP
  two-layer method computes a top-of-pier stress qg = q·ns/(Ra·ns−Ra+1) and then
  suz = qg/kg, where **kg is the pier stiffness modulus in pci** (a stress/penetration
  modulus, 75–360 pci range). `ground_improvement` has **no kg input and no qg/kg
  path** — `analyze_aggregate_piers` / `improved_settlement` instead use the
  equal-strain **settlement-reduction-factor** model SRF = 1/(1+as·(n−1)). For this
  example SRF = 0.43, so the module's improved settlement would be ~9.4 in, versus the
  published **0.68 in** from qg/kg — a fundamentally different model (the SRF model
  has no notion of a pier stiffness modulus and gives far less reduction at as≈0.27).
  The published 0.68 in is reproduced by hand on the module's `area_replacement_ratio`
  (which DOES match the source's de=1.05·s unit-cell Ra=0.2744 to 4 figures, because
  √3/2·s² ≈ π/4·(1.05s)²). **Possible add:** a RAP `upper_zone_settlement(q, Ra, ns, kg)`
  helper (the GEC-13 / Lawton-Fox-Wissmann two-layer method) if RAP stiffness-modulus
  design is in scope. Not a bug — a method/coverage gap.
- **V-019 — Priebe improvement factor is in the module; the spread is convention,
  not a bug (CONVENTION).** The module exposes the genuine Priebe (1995) basic
  improvement factor `priebe_basic_improvement_factor(as, phi_col, nu_s)`. The
  published 2.7 is a **chart read** (Figure 5-27, after Wallays et al. 1983), and the
  factor is sensitive to two conventions the source leaves implicit:
  - **Area-ratio convention.** The source uses A/Ac = (s/d)² = 3.6 → as = Ac/A = 0.277
    (a square-grid / unit-cell ratio). The module's geometric `area_replacement_ratio`
    for a *triangular* pattern uses √3/2·s² and gives as = 0.251. Feeding the module's
    own as → Priebe n0 = 2.81 (within 2.7±0.3); feeding the source's as = 0.277 →
    n0 = 3.06 (outside).
  - **Column friction angle.** The module default φ_col = 42.5° (Priebe's compacted-
    stone chart value). At as = 0.277 the published 2.7 corresponds to φ_col ≈ 39°.
  Either way the improved S = 27/n0 = 8.8–9.6 in brackets the published 10 in. The
  formula reproduces the chart family; the ±13% is the area-ratio + φ_col + chart-read
  latitude the inventory flagged. **Module NOT tuned to the one chart value.** (Note:
  `priebe_basic_improvement_factor` takes as = Ac/A — feed it the source's 1/(s/d)²
  directly to match the source's convention, rather than the geometric triangular as.)
- **V-020 — the PVD radial-consolidation primitive is a clean PASS (radial-only,
  ideal Barron).** With ch only (no vertical, no smear, no well resistance) the
  module's Barron-Hansbo radial solution — `influence_diameter`(de = 1.05·s),
  `drain_function_F`(F(n) = ln(n)−0.75), `time_for_radial_consolidation`(t = −F/8·
  ln(1−U)·de²/ch) — gives **t90 = 299 d at s = 8 ft** and **503 d at s = 10 ft**, vs
  the published "on the order of" **~300 / ~500 days** (<1%, well inside ±20%). The
  de = 1.05·s triangular unit cell and the dw = 2.5 in plumbing both check out
  (n = 40.3 / 50.4). **Convention used by the source: RADIAL-ONLY.** The module's
  packaged `analyze_wick_drains` adds the combined term U_total = 1−(1−Uv)(1−Ur); at
  s = 8 ft, t = 300 d that gives U_total ≈ 93% (Ur ≈ 90%, Uv ≈ 31% from the 20-ft clay
  over rock at single drainage), so the COMBINED t90 would be *shorter* than 300 d.
  The example clearly used radial-only — documented, not a bug. The module supports
  both (pass radial-only via the primitives, or combined via `analyze_wick_drains`).
- **No module bugs found or fixed in this batch.** All three are either clean PASSes
  (unimproved consolidation; area-ratio primitive; radial-only Barron t90) or
  documented method/convention gaps (RAP stiffness-modulus method absent; Priebe
  area-ratio + φ_col + chart latitude; PVD radial-only vs combined). `ground_improvement/`
  suite was run unchanged (**49 passed**); `settlement` used read-only.
