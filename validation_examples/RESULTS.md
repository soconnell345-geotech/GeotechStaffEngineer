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
| V-006 | drilled_shaft side (sand), `beta_method="rational"` | β=0.413; fSN=935 psf; RSN=470.2 kips (high-level path) | β=0.41; fSN=936 psf; RSN=470.7 kips (rational OCR chain) | β +0.7%, RSN −0.1% | **PASS** (v5.3) | The GEC-10 Appendix A rational β chain is now BUILT INTO the module ((N1)60→φ′=27.5+9.2·log10; σ′p=0.47·pa·N60^0.6; OCR=σ′p/σ′v_ref; Ko=(1−sinφ′)·OCR^(sinφ′)≤Kp; β=Ko·tanφ′) as the opt-in `beta_method="rational"`. Per-layer `sigma_v_ref` carries the no-scour σ′v (4,645 psf) for OCR while `fs` uses the scoured σ′v (2,266 psf). Reproduced through the high-level `DrillShaftAnalysis` path, no hand-fed coefficients. Default `beta_method="depth"` (O'Neill-Reese, floors at 0.25) byte-identical. Coverage gap CLOSED. |
| V-007 | drilled_shaft side (clay), `alpha_method="rational"` | α=0.475; fSN=977 psf; RSN=367.8 kips (high-level path) | α=0.47; fSN=976 psf; RSN=368.1 kips (rational su-transform) | α +1.1%, RSN −0.1% | **PASS** (v5.3) | The GEC-10 Chen-2011 rational α is now BUILT INTO the module (`alpha_method="rational"`): su(UU)→su(CIUC) via the Chen & Kulhawy UC-pair transform (0.893+0.513·log10, `su_test_type="uc"`) → su(CIUC)=2,057 psf, then α=0.30+0.17/(su_CIUC/pa)=0.475, with `fs = α·su_CIUC`. Reproduced through the high-level path; the full 15-ft clay is active (bottom-1D exclusion lands in the bearing layer below). Default `alpha_method="aashto"` (0.55) byte-identical. Coverage gap CLOSED. |
| V-008 | drilled_shaft base (sand) | qb=49.24 ksf; RBN(unreduced)=2475 kips | qb=49.2 ksf; RBN=2473 kips | <0.1% | **PASS** (unit) | `57.5·N60` kPa ≡ `0.60·N60` tsf — exact. Module additionally applies the O'Neill-Reese 1.27/Dᵦ large-diameter reduction (factor 0.521 at Dᵦ=2.44 m) → 1289 kips; example shows the unreduced value. See CONVENTION note below. |
| V-008b | drilled_shaft base large-D | 1.27/Dᵦ applied (0.521) | (not shown in example) | — | CONVENTION | Module follows GEC-10 §13.3.4.3 / O'Neill-Reese large-diameter base reduction; the extracted example value is unreduced. Correct per the cited method; flagged for owner awareness. |
| V-001 | axial_pile Nordlund shaft (sand) | Rs = 126/249/353/477 kips (D=35/50/60/70) | Rs = 137.5/250.7/344.1/452.5 | −8% / −1% / +3% / +5% | **PASS** | Nordlund shaft vs Table D-6, modeled from the footing datum. All four depths within ±15%; the displacement-pile Kd fit reproduces the published Nordlund shaft for this φ 33–36 H-pile profile. |
| V-001 | axial_pile Nordlund toe (sand), toe φ=40 | Rt = 408.5 kips (Layer 3 plateau) | Rt = 428.1 kips | −5% | **PASS** (toe fn) | Calling `end_bearing_cohesionless(φ=40)` reproduces the published toe plateau; the q_L cap (Meyerhof Fig 7-15) at φ=40 governs at 408.5 kips. |
| V-001 | axial_pile Nordlund toe (single-φ API, default) | Rt = 301 kips (φ=36 in L3) | Rt = 428.1 kips | −30% | DEFAULT | With NO per-layer toe φ set, the high-level `AxialPileAnalysis` API uses ONE φ per layer (φ=36 at the toe too) → 301 kips. Pins the unchanged default; the v5.2 toe-φ feature (next row) is purely additive. |
| V-001 | axial_pile Nordlund toe (high-level API, per-layer toe φ=40) | Rt = 408.5 kips (Layer 3 plateau) | Rt = 428.1 kips | −4.6% | **PASS** (v5.2) | The new `AxialSoilLayer.toe_friction_angle=40` lets the high-level `AxialPileAnalysis` apply the GEC-12 separate Layer-3 design-limit toe φ; the q_L cap (Meyerhof Fig 7-15) at φ=40 governs at 408.5 kips. Shaft unaffected (still φ=36). Closes the old single-φ CONVENTION gap; default-preserving (toe φ unset → byte-identical to before). Also new: `AxialPileAnalysis(head_depth=…)` clips the shaft to a below-grade pile head (e.g. a footing datum) without hand-clipping layers; default 0 = head at surface. |
| V-002 | axial_pile alpha toe (clay), 9·cu | Rt = 32.8/33.3/34.1 kips (D=70/80/90) | Rt = 32.75/33.03/33.68 | 0% / +1% / +1% | **PASS** | End bearing 9·cu matches Table D-100 to ±1% where the tip is cleanly inside Layer 3. (D=40 toe is a layer-boundary artifact — tip exactly at the L2/L3 interface — excluded.) |
| V-002 | axial_pile alpha shaft (clay) | Rs = 250/297/345 kips (D=70/80/90) | Rs = 318.4/369.3/420.33 | −22% / −20% / −18% | CONVENTION | Module under-predicts shaft. Cause isolated to the STIFF clay band: module Tomlinson α≈0.42 vs the example's DrivenPiles α≈0.76 at su≈1.9 ksf. Soft (α≈0.85) and very-stiff (α≈0.34) bands agree. Different α-vs-cu curve; module not tuned. |
| V-003 | wave_equation drivability (diesel) | ~48 bpf @ 746 kips, ~48 ksi comp | 120 bpf @ Rndr=746 kips, 40.1 ksi | bpf −60%, stress +21% | N/A (scope) | No Delmag D36-52 in the hammer DB (only the −22/−32 single-energy diesel series). A hand-built D36-52 + generic cushion gives 48 bpf (vs 120, outside ±25%) and 48 ksi (vs 40.1, outside ±10%): the module's diesel hammer is a simple energy/efficiency velocity conversion, not a GRLWEAP diesel combustion + ram-cycle model. Coverage gap. |
| V-004 | downdrag neutral plane (Fellenius) | NP=53.3 ft, Qmax=487.2 kips, DF=286.2 kips | NP=54 ft, Qmax=486 kips, DF=285 kips | NP −0.7 ft, Qmax +0.2%, DF +0.4% | **PASS** | Fellenius NP construction reproduces NP/Qmax/DF to <1% at 100% toe mobilization (toe=428.1 kips, Q=201 kips). Per-layer β overridden to feed the module the published Table D-6 Nordlund shaft (inventory-permitted), validating the NP equilibrium logic decoupled from the shaft method. |
| V-005 | pile_group `meyerhof_group_settlement` (Meyerhof 1976 SPT) | S = 26.52 mm (1.044 in) | S = 1.04 in | +0.4% | **PASS** | New `pile_group.meyerhof_group_settlement` packages S[in]=4·pf[ksf]·If·√B[ft]/N160 behind an **SI public API** (B in m, load in kN → settlement in mm), converting internally (SI→US, apply US-form, US→mm; equiv. SI coeff C=25.4·4/47.88/√0.3048=3.8435). 50-ft Table D-23 case (B=5 ft, Z=41 ft, Q=1540 kips → pf=7.512 ksf; DB=5 ft → If=0.9167; N160=59) → 26.52 mm = 1.044 in vs published 1.04 in (+0.4%, within ±5%). Coverage gap closed; formula NOT tuned. |
| V-009 | retaining_walls MSE external, unfactored loads | V1=57.69, Vs=4.50, F1=13.68, F2=2.14 k/ft; MV1=519.1, MF1=117.0, MF2=27.4 k-ft/ft | V1=57.69, Vs=4.50, F1=13.68, F2=2.13; MV1=519.21, MF1=116.94, MF2=27.36 | <0.5% | **PASS** | `rankine_Ka`(34→0.283, 30→0.333) + the GEC-11 force eqns reproduce every Table E4-4.3/4.4 unfactored force & moment about Point A. `horizontal_force_active` returns the combined F1+F2 thrust (15.83 k/ft) + line of action. These are the load quantities the module's primitives contribute. |
| V-009 | retaining_walls MSE external, LRFD CDRs | sliding 1.85/2.08/1.37; ecc eL 2.87/3.87 ft; bearing σv 6.70 ksf, CDR 1.57; svc σv 4.66; crit σv 5.75 | sliding 1.85/2.08/1.37; ecc 2.87/3.87; bearing 6.70, 1.57; svc 4.66; crit 5.86 | ≤2% | **PASS** (via primitives) | Driving the GEC-11 Str I max/min load-factor pairing (EV 1.35/1.00, EH 1.50/0.90, LL 1.75) + LL-on-resisting exclusion (sliding/ecc) / LL-included (bearing) on top of the module's earth-pressure primitives reproduces every published CDR. Critical bearing σv 5.75 vs 5.86 (+1.9%) is within the source's own "consistent-values" rounding note. |
| V-009 | retaining_walls `analyze_mse_wall` (high-level) | ASD FOS_sliding ≈ 2.27 (R/demand, no load factors) | LRFD sliding CDR 1.85 | — | CONVENTION | The packaged `analyze_mse_wall` returns ASD factors of safety (unfactored resistance/demand, surcharge in W, LL not split out), NOT the GEC-11 LRFD CDRs. The LRFD load-factor + LL-exclusion bookkeeping is not in the module. Documented API/method gap; the example was reproduced through the lower-level primitives instead. |
| V-010 | retaining_walls MSE internal `Tmax_at_level` (bar mat) | σH/Tmax L1 0.40/6.24, L4 1.02/12.76, L7 1.26/15.73, L10 1.51/19.03 (k/panel) | σH/Tmax L1 0.40/6.25, L4 1.02/12.77, L7 1.26/15.71, L10 1.51/19.05 | ≤0.5% | **PASS** | `Tmax_at_level` reproduces Table E4-7.4 bar-mat Tmax/σH at all 4 sampled levels driven by the BUILT-IN bar-mat `Kr_Ka_ratio`(z,"metallic_grid") (2.5→1.2 over 0–20 ft) and the EV factor 1.35 on soil+surcharge, via the example's average-over-tributary-bounds σH method. (v5.2 Q4: curve now in the module, no hand-fed coefficient.) |
| V-010 | retaining_walls MSE internal `pullout_resistance` (bar mat) | nominal Pr=23.06, φ·Pr=20.76 k/ft (Level 4) | Pr=23.06, φ_p·Pr=20.75 | <0.1% | **PASS** | `pullout_resistance(C=2)` (two grid surfaces = the example's "2b") with the BUILT-IN bar-mat F*=0.955 from `F_star_metallic`(z,"metallic_grid",t/St) (20·t/St→10·t/St), Le=L−0.3H=10.31 ft, unfactored soil σv, φ_pullout=0.90 reproduces Level-4 Pr exactly. Le lengthens to 17.24 ft at Level 10 (Z>H/2 taper). (v5.2 Q4.) |
| V-010 | retaining_walls MSE internal built-in curves (bar-mat branch) | `Kr_Ka_ratio`(0,"metallic_grid")=2.5; `F_star_metallic`(0,…,"metallic_grid",t/St)=20·t/St=1.246; F*(L4)=0.955 | bar mat: Kr/Ka(0)=2.5; F*(0)=1.246; F*(L4)=0.955 | ≤0.5% | **PASS** (v5.2 Q4) | The steel-bar-mat / welded-grid curves are now BUILT INTO `Kr_Ka_ratio`/`F_star_metallic` (reinforcement_type "bar_mat"/"welded_grid"/"metallic_grid"): Kr/Ka 2.5→1.2 over 0–20 ft and F* 20(t/St)→10(t/St) over 0–20 ft. `check_internal_stability` auto-selects them for a `metallic_grid` reinforcement (e.g. `WELDED_WIRE_GRID_W11`), so the high-level internal path matches a bar-mat wall directly. The ribbed-STRIP default (1.7→1.2; 2.0→tanφ) is byte-identical/unchanged. Closes the former coverage gap. |
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
| V-021 | bearing_capacity Vesic N-factors (φ=35) | Nq=33.30, Nγ=48.03 | Nq=33.3, Nγ=48.0 | <0.1% | **PASS** | `bearing_capacity_Nq`/`bearing_capacity_Ngamma(method="vesic")` reproduce the GEC-6 Table 5-1 AASHTO/Vesic factors to 4 figures. (Meyerhof Nγ=37.15, Hansen Nγ=33.92 differ markedly — the Vesic path is the right one.) The direct N-factor match the inventory flagged. |
| V-021 | bearing_capacity Vesic shape factors (square) | sq=1.700, s_γ=0.600 | sq=1.7, s_γ=0.6 | exact | **PASS** | Vesic `shape_factors` for B/L=1: sq=1+(B/L)·tanφ=1.700, s_γ=1−0.4(B/L)=0.600 — GEC-6 Table 5-2, exact. |
| V-021 | bearing_capacity qult/qall (assembled, example convention) | qult=2552+254.2·B; qall(FS=3)=851+85·B → B=3: 1105, 4.6: 1240, 6.1: 1368 kPa | qult=2553+254·B; B=3: 1106, 4.6: 1242, 6.1: 1369 | <0.2% | **PASS** | Strongest bearing check. Driving the example's assembly (q·Nq·sq·Cwq + 0.5·γ·B·Nγ·s_γ·Cwγ, dq=1.0, AASHTO Cwγ=0.9 fixed at the B=6 trial) on the MODULE factor functions reproduces the published closed-form intercept (2552 vs 2553) AND slope (254.2 vs 254) to 4 figures. q=γ·Df=45.1 kPa; Cwq=2.48→cap 1.0; Cwγ(B=6)=0.903. |
| V-021 | bearing_capacity `compute()` (high-level) | qult=3897 kPa at B=3 (dq=1.195, γ_eff=19.6, no Cwγ) | qult=3315 at B=3 | +17.6% | CONVENTION | Packaged `BearingCapacityAnalysis.compute()` runs high for two defensible reasons: (1) it applies Vesic DEPTH factors dq≈1.10–1.20 while the example sets dq=1.0 (cohesive overburden, Table 5-4); (2) its effective-unit-weight GW model sees the GW at 9.1 m below the bearing wedge so γ_eff=γ=19.6 (no reduction), vs the example's AASHTO Cwγ=0.9 correction factor. Same N/shape factors; closed form recovered exactly the example's way (row above). Module NOT tuned. |
| V-021b | bearing_capacity `Footing` effective area (eccentric) | B'=4.746, L'=4.666, A'=22.15 m², q_applied=364.4 kPa; sliding FS=30.9 | B'=4.75, L'=4.67, A'=22.2, q=364; FS=31 | <0.5% | **PASS** | The 4.9 m square footing with eB=0.077, eL=0.117 (P=8070 kN): `Footing.B_eff/L_eff/A_eff` (Meyerhof effective-area) reproduce the example's eccentric-load dimensions and q_applied=P/A' exactly; the 4.6 m trial likewise (B'=4.45, L'=4.37, A'=19.4, q=416). Sliding FS=0.7·(W+P)/V=30.9 (pub 31). |
| V-022 | settlement `approximate_2to1` stress increase (square) | Δσv/q: B=3 → 0.549/0.162/0.070/0.044; B=6.1 → 0.728/0.334/0.179/0.123 | B=3 → 0.55/0.16/0.07/0.04; B=6.1 → 0.73/0.33/0.18/0.12 | ≤2% | **PASS** (primitive) | The example's 2:1 Δσv=q·B²/(B+Z)² for a square footing IS `approximate_2to1` (q·B·L/((B+z)(L+z)) at B=L). The module reproduces every Table B1-2 stress fraction (3 widths × 4 layers) to ≤2%. |
| V-022 | settlement Hough (granular C'-index) via new `hough_settlement` | B=3,q=240 → 15.4+4.4+1.1+0.6=21.5 mm; full Table B1-3 (12 cells) reproduced to within ±4.6% | per-layer 15+4+1+1=21 mm; table B=3:21/25/28/30, B=4.6:28/31/34/37, B=6.1:31/35/38/41 | ≤4.6% | **PASS** | The new `settlement.hough.hough_settlement(layers, q_net, B, L)` method (over `HoughLayer`) packages the Hough (1959) granular C'-index form dH=H/C'·log10[(σ'vo+Δσ)/σ'vo], reusing `approximate_2to1` for the 2:1 stress increase. C' is the Hough bearing-capacity index (NOT Cc/(1+e0)); the Cc/Cr e-log(p) path remains for cohesive soils. The module method reproduces every cell of Table B1-3 (12 q×B cells) within ±15% (largest −4.6% at B=4.6,q=240: 26.7 vs 28; worked B=3,q=240: 21.5 vs 21). Coverage gap closed; module NOT tuned. |
| V-023 | fem2d final drained settlement (confined 1-D Biot column) | w = 2.609 mm (elastic confined-column solve AND `solve_consolidation` end-state) | pz·H/(K+4G/3) = 2.61 mm | 0.0% | **PASS** | Laterally-confined column (oedometric, side rollers), base fixed, top loaded pz=1e5 Pa. fem2d reproduces the drained end-state settlement EXACTLY (ratio 1.0000) both via `solve_elastic` and via the coupled `solve_consolidation`. Units mapped Pa→kPa; Biot M→n_w=M/1000 kPa; mobility k→k_hyd=k·γ_w. |
| V-023 | fem2d analytical anchors (S, c, p0) | S=1.554e-9 1/Pa; c=0.0643 m²/s; p0_consistent=0.839e5 Pa | S=1.554e-9; c=0.0643; p0 (formula) 0.839e5 / (Itasca reported 0.981e5) | exact | (anchors) | Storage S=1/M+α²/(K+4G/3), c=k/S, and the Biot 1-D undrained p0 verified inline. The inventory's stated 0.981e5 is Itasca's reported value, which needs an effective M≈10× the stated 4e9 (near-incompressible fluid); the inventory's own formula gives 0.839e5 (consistent with the stated M). Documented; immaterial since fem2d gives 0 (next row). |
| V-023 | fem2d undrained p0 + p(z,t) consolidation decay | excess pp = 0 at every step; settlement = drained 2.609 mm from t=0 (NO transient) | undrained p0 ~0.84e5 Pa; Terzaghi decay p/p0 vs t̂; FLAC <5% err | — | N/A (scope) | **Structural limitation of the staggered Biot scheme.** `solve_consolidation` applies the surface load as a static F_ext and the staggered displacement step solves the fully DRAINED equilibrium at every time level (top pinned head=0) → no undrained excess pore pressure is generated and the settlement is the drained value from t=0 with no decay. A prescribed-p0 dissipation test (no load) also fails to reproduce the Terzaghi diffusion (non-monotonic, far too fast). Needs a monolithic u-p solve or an undrained predictor — NOT a unit bug. The drained end-state (PASS above) is the only reproducible quantity. |
| V-024 | fem2d MC plastic radius (Salencon cavity) | R0=1.735 m (σ_r=12.01 crossing); R0=1.731 m (σ_θ peak) | R0=1.735 m | 0% / −0.2% | **PASS** (slow) | Quarter-symmetry graded T6 annular mesh (a=1, R_out=20a, ~3700 elem, ~2.5 s), in-situ σ=−P0, cavity unloaded P0→0 via the new `initial_stress_relaxation` driver + `roller_base` symmetry BC (both general solver additions). Both R0 detectors (σ_r boundary crossing on the x-axis; σ_θ peak = elastic-plastic boundary) land within ±5%. psi=0 non-associated. |
| V-024 | fem2d radial stress at elastic-plastic boundary | σ_r(R0) ≈ 11.9 MPa (at the σ_θ peak) | (1/(Kp+1))·(2P0−q) = 12.01 MPa | −0.7% | **PASS** (slow) | Boundary radial stress within ~1% of the Salencon value. Kp=3.0, q=2c√Kp=11.95 MPa. |
| V-024 | fem2d far-field stress profile r/a=2..5 | σ_r/σ_θ follow Salencon elastic branch but ~5% LOW (compression under-predicted) | Salencon elastic σ_r=P0−(P0−σ_r,R0)(R0/r)², σ_θ symmetric | ~−5% | CONVENTION | Far-field domain-truncation: the outer boundary is FIXED at R_out=20a (the inventory flags ~1−2%; a rigid fixed boundary adds a bit more than a far-field traction). Profile SHAPE and elastic decay are correct; asserted ±7% with the low-side bias documented. R0/σ_r(R0) (primary targets) pass at ±5%. A pure-traction outer BC pushes the far field to ±3% but is under-constrained (drifts, non-convergence) — fixed is the robust choice. |

## Notes / flags for the owner

- **drilled_shaft now offers BOTH the simplified defaults AND the GEC-10
  rational chains (v5.3 — gap CLOSED).** The defaults remain the AASHTO /
  O'Neill-Reese simplified α (0.55) and depth-based β (`1.5−0.245√z_ft`), byte-
  identical to before. The GEC-10 (2018) *rational* side-resistance chains from
  the Appendix A example are now built in as **opt-in** methods on
  `DrillShaftAnalysis`: `beta_method="rational"` (OCR-based β from (N1)60/N60 with
  the Ko chain) and `alpha_method="rational"` (Chen-2011 α `0.30+0.17/(su_CIUC/pa)`
  with the UU/UC→CIUC transform, `su_test_type`). V-006/V-007 now reproduce the
  published β=0.41 / α=0.47 through the high-level path within ~1% (see the flipped
  rows above). The reference-layer `gec_10` lookups still hold the same formulas.
- **Deep cohesionless β floors at 0.25** (`z ≳ 40 ft`) — the O'Neill-Reese
  depth formula clamps there. Expected behavior, noted because it makes deep
  sand side resistance conservative vs the rational method (0.41).
- **Large-diameter base reduction** (V-008b) is applied by the module and not by
  the extracted example. The module is correct per the cited method; if the
  owner's downstream comparisons assume the unreduced nominal, that's the source
  of any 2× base-resistance gap.

### Batch V-001..V-005 (GEC-12 Vol 3 driven piles) — owner notes

- **Datum matters for axial_pile — head-offset feature ADDED (v5.2).** The GEC-12
  tables reference depths to the footing/cap bottom (5 ft bgs) and the pile head
  sits there. Modeling the `AxialSoilProfile` from the footing bottom (not the
  ground surface) is what makes V-001 shaft land within ±15% — modeling from the
  ground surface adds ~5 ft of spurious skin friction above the pile head.
  `AxialPileAnalysis(head_depth=…)` now clips the shaft integral to a below-grade
  pile head automatically (layers above the head contribute no skin friction;
  overburden above the head still counts toward σ'v at the tip). Default
  `head_depth=0` = head at surface, byte-identical to before. (For the V-001
  excavated footing the soil above the footing is removed, so σ'v=0 at the head
  and the footing-datum profile remains the physically-correct shaft model; the
  offset is the convenience for a below-grade head with the soil left in place.)
- **axial_pile separate shaft/toe φ per layer — feature ADDED (v5.2).** The example
  uses a Layer-3 design-limit toe φ=40 with a shaft φ=36. The optional
  `AxialSoilLayer.toe_friction_angle` now carries a distinct end-bearing φ for a
  layer (GEC-12 explicitly allows different shaft/toe φ in dense gravels). With it
  set the high-level API toe reaches **408.5 kips (−4.6%** vs 428.1), closing the
  old single-φ gap (which was −30% at 301 kips). Unset → falls back to the layer's
  `friction_angle`, preserving prior behaviour exactly. Adapter param:
  `layers[].toe_friction_angle` (alias `toe_phi`).
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
- **Meyerhof (1976) SPT group settlement added — gap CLOSED (V-005).** The
  closed form (S = 4·pf·If·√B/N160) is now packaged as
  `pile_group.meyerhof_group_settlement`, distinct from the elastic 2V:1H
  `group_settlement_equivalent_raft`. The coefficient "4" is US-calibrated
  (S in inches, pf in ksf, B in feet), so the helper takes **SI inputs**
  (B in m, load in kN / pf in kPa) and returns **mm**, converting internally
  (SI→US, apply the US-form, US→mm). The 50-ft Table D-23 case reproduces the
  published 1.04 in (26.52 mm = 1.044 in, +0.4%). Adapter method
  `meyerhof_group_settlement` wired with alias/require/reject discipline.

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
- **MSE internal stress/pullout primitives are solid; the bar-mat curves are now
  built in (v5.2 Q4 — gap CLOSED).** `Tmax_at_level` and `pullout_resistance(C=2)`
  reproduce the Table E4-7.4 bar-mat Tmax/σH/Pr to ≤3%. The module's built-in
  `Kr_Ka_ratio` and `F_star_metallic` now carry BOTH the **ribbed-metallic-STRIP**
  curves (Kr/Ka 1.7→1.2; F* 2.0→tanφ over 0–6 m) AND the **steel-bar-mat / welded-grid**
  curves (Kr/Ka 2.5→1.2; F* 20(t/St)→10(t/St) over 0–20 ft), selected by
  `reinforcement_type` ("bar_mat"/"welded_grid"/"metallic_grid"). The grid F*
  needs t (transverse-bar diameter) and St (transverse spacing) — taken from the
  `Reinforcement` geometry (`thickness`/`transverse_spacing`, e.g.
  `WELDED_WIRE_GRID_W11`) or supplied to `F_star_metallic(t_over_St=…)`.
  `check_internal_stability` auto-selects them for a `metallic_grid` reinforcement,
  so the high-level internal path now matches a bar-mat wall directly — no
  hand-fed coefficients. The ribbed-STRIP default path is byte-identical/unchanged.
  Depth datum: the bar-mat Kr/Ka and F* tapers cap at Z = 20 ft (= 6.096 m),
  distinct from the 6.0 m strip cap (both handled in the branch).
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

### Batch V-021 / V-022 (GEC-6 Ex B-1 spread-footing) — owner notes

- **V-021 is the strongest bearing check in the library — a clean PASS on the
  bearing_capacity factors, no bug.** The example's Vesic/AASHTO bearing-capacity
  factors at φ=35 (**Nq=33.3, Nγ=48.0**) and Vesic square shape factors (**sq=1.7,
  s_γ=0.6**) are reproduced by the module's `bearing_capacity_Nq` (33.30),
  `bearing_capacity_Ngamma(method="vesic")` (48.03) and `shape_factors` (1.700 /
  0.600) **to 4 figures**. Assembling the example's own equation
  `qult = q·Nq·sq·Cwq + 0.5·γ·B·Nγ·s_γ·Cwγ` (q=γ·Df=45.1 kPa, dq=1.0, Cwq capped
  at 1.0, Cwγ=0.9) on those module factors gives **qult = 2552 + 254.2·B** vs the
  published **2553 + 254·B** — intercept and slope both within 0.04%. qall (FS=3)
  at B=3/4.6/6.1 m = **1105/1240/1368 kPa** vs published **1106/1242/1369** (all
  <0.2%, inside the ±3% tolerance). The eccentric effective-area follow-on
  (V-021b) is also a clean PASS through the module's `Footing`: the 4.9 m square
  with eB=0.077, eL=0.117 gives B'=4.746, L'=4.666, A'=22.15 m², q_applied=364.4
  kPa (pub 364) and sliding FS=30.9 (pub 31). **No module change.**
- **V-021 high-level `compute()` is a CONVENTION (depth factor + GW model), ~+17%.**
  The packaged `BearingCapacityAnalysis(...).compute()` returns qult=3897 vs the
  published 3315 at B=3 m for two defensible reasons, NOT a bug: (1) it applies the
  Vesic **depth factor** dq≈1.10–1.20, while the example deliberately sets **dq=1.0**
  because the overburden above the footing is the cohesive Unit-1 lean clay (GEC-6
  Table 5-4 guidance — "set dq=1.0 when the overburden is cohesive"); and (2) its
  groundwater model averages the effective unit weight over the bearing wedge — and
  for these footing widths the GW at 9.1 m lies *below* the wedge (Df+1.5B), so
  γ_eff=γ=19.6 with **no reduction**, whereas the example applies the AASHTO **Cwγ
  groundwater-correction factor** Cwγ=0.5+0.5·[Dw/(Df+1.5B)]=0.9 (a width-dependent
  factor it then holds fixed at the B=6 m trial to get a linear qult). The module's
  N/shape factors are identical to the example's; the closed form is recovered
  exactly by assembling them the example's way (above). **Possible adds:** (a) an
  optional flag to suppress depth factors / pass dq=1.0 (cohesive-overburden case),
  and (b) an AASHTO Cwq/Cwγ groundwater-correction-factor option as an alternative
  to the effective-unit-weight averaging. Both are GEC-6 conventions; the current
  module path is defensible and was NOT changed.
- **V-022 — Hough / C'-index granular settlement method ADDED (v5.2), now a PASS.**
  The example's Hough form `dH = H/C'·log10[(σ'vo+Δσ)/σ'vo]` uses a *bearing-capacity
  index* C' (from the corrected SPT N', GEC-6 Fig 5-19), which is **not** the module's
  Cc/(1+e0) consolidation index. This is now packaged as `settlement.hough`
  (`hough_settlement(layers, q_net, B, L=None)` over `HoughLayer`, plus
  `hough_settlement_layer`/`HoughResult`), additive alongside the Cc/Cr e-log(p) and
  Schmertmann/elastic methods. It REUSES `approximate_2to1` for the 2:1 stress
  increase (square footing q·B²/(B+Z)² = q·B·L/((B+z)(L+z)) at B=L). It reproduces the
  published Table B1-2 stress fractions (e.g. B=3 → 0.55/0.16/0.07/0.04) to ≤2%, and
  the full Table B1-3 settlement matrix (4 stresses × 3 widths) within ±15% / the
  source's mm rounding (worked B=3,q=240: 15.4+4.4+1.1+0.6=21.5 mm vs pub 15+4+1+1=21;
  largest spread −4.6% at B=4.6,q=240). The funhouse adapter exposes it as the
  `hough_settlement` method. Module NOT tuned (validated, not fit).
- **No module bugs found or fixed in this batch.** V-021 is a clean factor-level PASS
  (the high-level `compute()` delta is the dq-and-GW-model convention); V-022 is a
  method/scope gap with the shared 2:1 primitive validated. `bearing_capacity/` and
  `settlement/` suites were run unchanged (**136 passed** combined); both used
  read-only.

### Batch V-023 / V-024 (Itasca FLAC — fem2d Biot consolidation + MC cavity) — owner notes

- **V-024 (cylindrical hole in MC, Salencon) is a clean PASS — but it required
  two small, GENERAL solver capabilities that fem2d was missing.** This is the
  hardest FE problem in the library (elasto-plastic stress redistribution around
  an unloaded cavity), and the win is squarely on the analysis module. The
  quarter-symmetry graded-T6 annular model (a=1, R_out=20a, ~3700 elem, ~2.5 s)
  reproduces:
  - plastic radius **R0 = 1.735 m** (σ_r=12.01 crossing) / **1.731 m** (σ_θ peak)
    vs analytic **1.735 m** (0% / −0.2%);
  - **σ_r(R0) = 11.9 MPa** vs analytic **12.01** (−0.7%).
  The far-field profile (r/a=2..5) runs ~5% low — domain truncation from the
  FIXED outer boundary (inventory flags ~1−2%; the rigid boundary adds more).
  **Two capabilities were ADDED to `fem2d` (general, not benchmark-tuned):**
  1. **`roller_base` BC key** (v = 0, u free) in `assembly.apply_bcs_penalty` and
     `solver.build_nl_context` — a horizontal-roller / symmetry-plane support
     (the BC vocabulary previously had only `fixed_base` and the vertical
     `roller_left/right`, so a quarter-symmetry model with v=0 on a horizontal
     axis was impossible). Needed by ANY half/quarter-symmetry model.
  2. **`solve_nonlinear(initial_stress_relaxation=True)`** — an initial-stress
     release / excavation driver. Given a pre-stressed `sigma_init` (in-situ
     field), it drives the analysis by the release load
     `F_ext = −∫B^T σ_init` (ramped over `n_steps`) with the residual offset by
     that initial internal force (`run_nl(f_int_offset=…)`), so a traction-free
     boundary (cavity wall, excavated face) relaxes to equilibrium while the
     MC/HS yield check sees the TRUE total stress (`σ_init + Δσ`) at every step.
     fem2d's `solve_nonlinear` previously only equilibrated gravity/surface loads
     from a STRESS-FREE reference (or staged construction) — it had no
     in-situ-stress + boundary-release path, which is exactly what tunnel/cavity
     and excavation-unloading problems need. **Possible follow-up:** expose a
     thin high-level `analyze_cavity(...)` / `analyze_tunnel(...)` wrapper (mesh
     + symmetry BCs + relaxation) so users don't hand-build the annular mesh.
  - The fem2d **groundwater + core suites still pass (138)**; the broader fem2d
    non-slow suite was run after the edits (see final pytest line). No existing
    behavior changed (both additions are opt-in: a new BC key and a default-False
    flag).
  - *Inventory slip (documented, harmless):* the inventory's `nu ≈ 0.313` is
    wrong — K=3.9, G=2.8 GPa give **nu = 0.210**. nu does not enter R0 or the
    stresses (only displacements), so it is immaterial to the verdict.

- **V-023 (1-D Terzaghi/Biot consolidation) is a SPLIT verdict: final drained
  settlement is an EXACT PASS; the undrained p0 and the consolidation decay are
  N/A (a structural limitation of fem2d's staggered Biot scheme).**
  - **PASS:** the analytical final drained settlement `w = pz·H/(K+4G/3) =
    2.609 mm` is reproduced to **ratio 1.0000** both by the elastic
    confined-column solve and by the coupled `solve_consolidation` end-state.
    The analytical storage `S = 1/M + α²/(K+4G/3) = 1.554e-9`, coefficient of
    consolidation `c = k/S = 0.0643 m²/s`, and the Biot undrained `p0` are
    verified inline.
  - **N/A (scope):** `solve_consolidation` does NOT generate a load-induced
    undrained pore pressure and has NO consolidation transient. It applies the
    surface load as a static `F_ext`, and the STAGGERED displacement step solves
    the fully DRAINED equilibrium at every time level (the top is pinned to
    head=0), so the max excess pore pressure stays **0** (vs the analytical
    p0 ≈ 0.84e5 Pa) and the settlement is the drained value already at t=0.
    A prescribed-p0 dissipation probe (load bypassed) further shows the staggered
    split does not reproduce the Terzaghi diffusion (pressure history is
    non-monotonic and far too fast). The fix is NOT a unit/mesh tweak — it needs
    a **monolithic (coupled u-p) Biot solve, or an undrained predictor** (apply
    the load with drainage temporarily closed to set p0, then dissipate). This
    is a real solver limitation; the module was **not** re-architected for one
    benchmark. **Possible follow-up:** a monolithic u-p consolidation option, or
    an `undrained_load=True` predictor in `solve_consolidation`.
  - *Storage-term note (for the future upgrade):* fem2d's compressibility uses
    `S = 1/n_w` only — it has no Biot α and omits the skeleton storage term
    `α²/(K+4G/3)`. With `n_w = M`, fem2d's storage would be `1/M`, vs the true
    `1/M + α²/(K+4G/3)` (≈ 17% higher here). Map `n_w` to the FULL storage
    (`n_w = 1/S`) if matching `c` is wanted once the transient works.
  - *p0 reconciliation (documented):* the inventory quotes both a formula
    (`α·M/(K+4G/3+α²·M)·pz` → **0.839e5**, consistent with the stated M=4e9) and
    Itasca's reported **0.981e5** (which needs an effective M ≈ 10× larger, i.e.
    a near-incompressible fluid). Both recorded; immaterial since fem2d gives 0.

- **Module edits this batch (fem2d only):** `assembly.apply_bcs_penalty`
  (+`roller_base`), `solver.build_nl_context` (+`roller_base` DOF),
  `solver.run_nl` (+`f_int_offset`), `solver.solve_nonlinear`
  (+`initial_stress_relaxation`). All additive/opt-in. No consolidation-solver
  changes were made (the V-023 limitation is documented, not patched).
