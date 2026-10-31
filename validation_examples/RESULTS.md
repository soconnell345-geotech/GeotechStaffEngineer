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
| V-006 | drilled_shaft side (sand), `beta_method="rational"` | β=0.413; fSN=935 psf; RSN=470.2 kips (high-level path) | β=0.41; fSN=936 psf; RSN=470.7 kips (rational OCR chain) | β +0.7%, RSN −0.1% | **PASS** (v5.3) | The GEC-10 Appendix A rational β chain is now BUILT INTO the module ((N1)60→φ′=27.5+9.2·log10; σ′p=0.47·pa·N60^0.6; OCR=σ′p/σ′v_ref; Ko=(1−sinφ′)·OCR^(sinφ′)≤Kp; β=Ko·tanφ′) as the opt-in `beta_method="rational"`. Per-layer `sigma_v_ref` carries the no-scour σ′v (4,645 psf) for OCR while `fs` uses the scoured σ′v (2,266 psf). Reproduced through the high-level `DrillShaftAnalysis` path, no hand-fed coefficients. Since GM-241 (2026-10-20) `beta_method="rational"` is the default; the O'Neill-Reese depth β stays by name (`"depth"`). Coverage gap CLOSED. |
| V-007 | drilled_shaft side (clay), `alpha_method="rational"` | α=0.475; fSN=977 psf; RSN=367.8 kips (high-level path) | α=0.47; fSN=976 psf; RSN=368.1 kips (rational su-transform) | α +1.1%, RSN −0.1% | **PASS** (v5.3) | The GEC-10 Chen-2011 rational α is now BUILT INTO the module (`alpha_method="rational"`): su(UU)→su(CIUC) via the Chen & Kulhawy UC-pair transform (0.893+0.513·log10, `su_test_type="uc"`) → su(CIUC)=2,057 psf, then α=0.30+0.17/(su_CIUC/pa)=0.475, with `fs = α·su_CIUC`. Reproduced through the high-level path; the full 15-ft clay is active (bottom-1D exclusion lands in the bearing layer below). Default `alpha_method="aashto"` (0.55) byte-identical. Coverage gap CLOSED. |
| V-008 | drilled_shaft base (sand) | qb=49.24 ksf; RBN(unreduced)=2475 kips | qb=49.2 ksf; RBN=2473 kips | <0.1% | **PASS** (unit) | `57.5·N60` kPa ≡ `0.60·N60` tsf — exact. Module additionally applies the O'Neill-Reese 1.27/Dᵦ large-diameter reduction (factor 0.521 at Dᵦ=2.44 m) → 1289 kips; example shows the unreduced value. See CONVENTION note below. |
| V-008b | drilled_shaft base large-D | 1.27/Dᵦ applied (0.521) | (not shown in example) | — | CONVENTION | Module follows GEC-10 §13.3.4.3 / O'Neill-Reese large-diameter base reduction; the extracted example value is unreduced. Correct per the cited method; flagged for owner awareness. |
| V-001 | axial_pile Nordlund shaft (sand) | Rs = 139.1/253.5/347.8/457.4 kips (D=35/50/60/70) | Rs = 137.5/250.7/344.1/452.5 | +1.2% / +1.1% / +1.1% / +1.1% | **PASS** | Since GM-121 (2026-10-18): GEC-12's charts (δ/φ from Fig. 7-9 curve f at V = 0.151 ft³/ft, K_δ from Tables 7-6/7-7, C_F from Fig. 7-14), profile from the ground surface with the head at the footing (`head_depth` 5 ft): Table D-6's σ′ keeps the 5 ft above the footing (its toe at 0.01 ft, 16.4 kips, needs it; module 16.1). Before: a fitted K_δ, δ/φ 0.75 and the footing-datum profile, −8 % to +5 %. |
| V-001 | axial_pile Nordlund toe (sand), toe φ=40, σ′p not limited | Rt = 429.3 kips (D=50/60/70) | Rt = 428.1 kips | +0.3% | **PASS** (`toe_stress_limit=None`) | The example (DrivenPiles) did not apply GEC-12 Step 8's 3 ksf limit on σ′p: σ′v at its Layer-3 toe is 3.6–4.8 ksf and its plateau is q_L(40) (Fig. 7-15) × box area. With the limit off the module reproduces it (q_L governs at 50–70 ft since GM-282; Eq. 7-9 gave 425.9 at 50 ft before). |
| V-001 | axial_pile Nordlund toe below the plateau (sand), σ′p not limited | Rt within 0.7 % (toe φ 33, D=0–10 ft; toe φ 40, D=46–48 ft), 1.9–2.3 % low (toe φ 36, D=23–34 ft) | Table D-6 | ≤ 0.7% / ≤ 2.3% | **PASS** (GM-282) | α_t N′q σ′p × box area (Eq. 7-9, Fig. 7-16). N′q read again off Fig. 7-16's own image (GM-282): the earlier nodes were 1–3.5 % low from 29°, and the toe 2.0–2.3 %, 4.6–4.9 % and 2.7 % low here. |
| V-001 | axial_pile Nordlund toe (sand), toe φ=40, GEC-12 Step 8 (default) | Rt = 371.5 kips | Rt = 428.1 kips | −13% | CONVENTION (GEC-12 Step 8) | Since GM-121 the module applies Step 8 by default: α_t N′q × 3 ksf × box area. The published value moves away by the limit the example did not apply (previous row). |
| V-001 | axial_pile Nordlund toe (single-φ API, default) | Rt = 154.8 kips (φ=36 in L3) | Rt = 428.1 kips | −64% | DEFAULT | With NO per-layer toe φ set, the high-level `AxialPileAnalysis` API uses ONE φ per layer (φ=36 at the toe too). Pins the default; `AxialSoilLayer.toe_friction_angle` is the example's separate toe φ. |
| V-002 | axial_pile alpha toe (clay), 9·cu | Rt = 32.8/33.3/34.1 kips (D=70/80/90) | Rt = 32.75/33.03/33.68 | 0% / +1% / +1% | **PASS** | End bearing 9·cu matches Table D-100 to ±2% where the tip is cleanly inside Layer 3. (D=40 toe is a layer-boundary artifact — tip exactly at the L2/L3 interface — excluded.) |
| V-002 | axial_pile alpha shaft (clay), the example's Fig. 7-17 choices | Rs = 320.2/372.1/424.0 kips (D=70/80/90) | Rs = 318.4/369.3/420.33 | +0.6% / +0.8% / +0.9% | **PASS** (`adhesion_curve="smooth_steel"`, `adhesion_beyond_chart="chart_end"`) | Since GM-121 the adhesion is GEC-12 Fig. 7-17 (by su, pile type, D/b with D from the ground surface). The example used the smooth-steel curves for the H-pile and held Ca at the chart's end in Layer 3 (su 3.1–3.7 ksf); with those two choices the module reproduces it. Before: a simplified α curve, −18 % to −32 %. |
| V-002 | axial_pile alpha shaft (clay), module defaults with a highly overconsolidated line of 1.0 | Rs = 362.7/418.2/473.7 kips (D=70/80/90) | Rs = 318.4/369.3/420.33 | +14% / +13% / +13% | CONVENTION (`highly_overconsolidated_su_ratio=1.0`) | The defaults take the corrugated curve × box area for H-piles (GEC-12 p. 247). Past Fig. 7-17's curves GEC-12 recommends API α for highly overconsolidated clays (alpha method, p. 247 and Step 1, p. 250) and gives no su/σ′vo that marks one, so since GM-229 a default run is refused until the engineer gives the line. With 1.0, Layer 3 (su/σ′vo 0.32–0.41) is below it and Ca is held at the chart's end, as the example did; the difference is the corrugated curve's higher adhesion. No V-002 layer reads α above 1, so reading Fig. 7-17 as drawn (GM-229) moves none of these numbers. |
| V-002 | axial_pile alpha shaft (clay), API α past the chart | Rs = 396.7/516.4/645.0 kips | Rs = 318.4/369.3/420.33 | +25% / +40% / +53% | CONVENTION (`adhesion_beyond_chart="api"`) | GEC-12 p. 247 recommends API α (Eqs. 7-13/7-14) for highly overconsolidated clays whose su exceeds the chart; in Layer 3 API α ≈ 0.7 at σ′v ≈ 7 ksf nearly doubles the chart-end Ca the example held. Both are choices reported in `choices`. |
| V-003 | wave_equation drivability (diesel) | ~48 bpf @ 746 kips, 39.0 ksi comp | 120 bpf @ Rndr=746 kips, 40.1 ksi | bpf −60%, stress −3% | N/A (scope) | No Delmag D36-52 in the hammer DB (only the −22/−32 single-energy diesel series). A hand-built D36-52 + generic cushion gives 48 bpf (vs 120, outside ±25%): the module's diesel hammer is a simple energy/efficiency velocity conversion, not a GRLWEAP diesel combustion + ram-cycle model, and its result warns so (GEC-12 Vol 2 p. 206). Coverage gap. The stress was 48.5 ksi (+21%) until GM-294's review took the pile-top force without the helmet's inertia; 39.0 ksi now, within ±10%, on a blow count that is not reproduced. |
| V-004 | downdrag neutral plane at 100% toe mobilization | NP=54.0 ft, Qmax=486.6 kips, DF=285.6 kips (pile weight left out, as the example does: the default since 2026-10-15); NP=53.6 ft, Qmax=490.1, DF=289.1 with `include_pile_weight=True` | NP=54 ft, Qmax=486 kips, DF=285 kips | NP 0.0 ft, Qmax +0.1%, DF +0.2% (weight in: −0.4 ft, +0.8%, +1.4%) | **PASS** | The intersection of the load and resistance curves reproduces NP/Qmax/DF at 100% toe mobilization (toe=428.1 kips, Q=201 kips). Per-layer β overridden to feed the module the published Table D-6 Nordlund shaft (inventory-permitted), validating the neutral plane construction decoupled from the shaft method. 2026-10-14 (GM-01): the plane search, the reported Qmax and the calibration now integrate the same shaft profile the same way; before, NP=53.3 ft, Qmax=487.2, DF=286.2 kips (weight in). |
| V-005 | pile_group `meyerhof_group_settlement` (Meyerhof 1976 SPT) | S = 26.52 mm (1.044 in) | S = 1.04 in | +0.4% | **PASS** | New `pile_group.meyerhof_group_settlement` packages S[in]=4·pf[ksf]·If·√B[ft]/N160 behind an **SI public API** (B in m, load in kN → settlement in mm), converting internally (SI→US, apply US-form, US→mm; equiv. SI coeff C=25.4·4/47.88/√0.3048=3.8435). 50-ft Table D-23 case (B=5 ft, Z=41 ft, Q=1540 kips → pf=7.512 ksf; DB=5 ft → If=0.9167; N160=59) → 26.52 mm = 1.044 in vs published 1.04 in (+0.4%, within ±5%). Coverage gap closed; formula NOT tuned. |
| V-009 | retaining_walls MSE external, unfactored loads | V1=57.69, Vs=4.50, F1=13.68, F2=2.14 k/ft; MV1=519.1, MF1=117.0, MF2=27.4 k-ft/ft | V1=57.69, Vs=4.50, F1=13.68, F2=2.13; MV1=519.21, MF1=116.94, MF2=27.36 | <0.5% | **PASS** | `rankine_Ka`(34→0.283, 30→0.333) + the GEC-11 force eqns reproduce every Table E4-4.3/4.4 unfactored force & moment about Point A. `horizontal_force_active` returns the combined F1+F2 thrust (15.83 k/ft) + line of action. These are the load quantities the module's primitives contribute. |
| V-009 | retaining_walls MSE external, LRFD CDRs (high-level `check_external_stability_lrfd`) | sliding 1.85/2.07/1.37; ecc eL 2.87/3.87 ft (< L/4 4.50); bearing σv 6.71 ksf, CDR 1.57; svc σv 4.66; crit σv 5.75 | sliding 1.85/2.08/1.37; ecc 2.87/3.87; bearing 6.70, 1.57; svc 4.66; crit 5.86 | ≤2% | **PASS** (v5.3, high-level) | The NEW high-level `check_external_stability_lrfd` computes the full CDR set directly from the wall/soil inputs with the GEC-11 Str I max/min + Service I load-factor combinations BUILT IN (EV 1.35/1.00, EH 1.50/0.90, LL 1.75; φ_sliding 1.0; e_max=L/4), incl. the LL-on-resisting exclusion (sliding/ecc) vs LL-included (bearing) and the min-vertical "critical" combination. Reproduces every published CDR through the module path — no hand-assembled factors. Critical bearing σv 5.75 vs 5.86 (−1.8%) is within the source's own "consistent-values" rounding note. |
| V-009 | retaining_walls `analyze_mse_wall(lrfd_external=True)` (high-level) | LRFD sliding CDR 1.85/2.07/1.37 attached as `result.external_lrfd`; ASD FOS_sliding ≈ 2.10 by default (live load left out, GM-164; 2.27 with live_load_resists=True) | LRFD sliding CDR 1.85 | ≤2% | **PASS** (v5.3) | The LRFD external path is now packaged: `analyze_mse_wall(lrfd_external=True)` runs BOTH the ASD FOS path (default, unchanged) AND the LRFD path, attaching the CDR set (sliding/eccentricity/bearing) to `MSEWallResult.external_lrfd`. Closes the former API/method gap — the GEC-11 LRFD load-factor + LL-exclusion bookkeeping now lives in the module. Default (flag off) keeps the ASD-only behavior byte-identical. |
| V-010 | retaining_walls MSE internal `Tmax_at_level` (bar mat) | σH/Tmax L1 0.40/6.24, L4 1.02/12.76, L7 1.26/15.73, L10 1.51/19.03 (k/panel) | σH/Tmax L1 0.40/6.25, L4 1.02/12.77, L7 1.26/15.71, L10 1.51/19.05 | ≤0.5% | **PASS** | `Tmax_at_level` reproduces Table E4-7.4 bar-mat Tmax/σH at all 4 sampled levels driven by the BUILT-IN bar-mat `Kr_Ka_ratio`(z,"metallic_grid") (2.5→1.2 over 0–20 ft) and the EV factor 1.35 on soil+surcharge, via the example's average-over-tributary-bounds σH method. (v5.2 Q4: curve now in the module, no hand-fed coefficient.) |
| V-010 | retaining_walls MSE internal `pullout_resistance` (bar mat) | nominal Pr=23.06, φ·Pr=20.76 k/ft (Level 4) | Pr=23.06, φ_p·Pr=20.75 | <0.1% | **PASS** | `pullout_resistance(C=2)` (two grid surfaces = the example's "2b") with the BUILT-IN bar-mat F*=0.955 from `F_star_metallic`(z,"metallic_grid",t/St) (20·t/St→10·t/St), Le=L−0.3H=10.31 ft, unfactored soil σv, φ_pullout=0.90 reproduces Level-4 Pr exactly. Le lengthens to 17.24 ft at Level 10 (Z>H/2 taper). (v5.2 Q4.) |
| V-010 | retaining_walls MSE internal built-in curves (bar-mat branch) | `Kr_Ka_ratio`(0,"metallic_grid")=2.5; `F_star_metallic`(0,…,"metallic_grid",t/St)=20·t/St=1.246; F*(L4)=0.955 | bar mat: Kr/Ka(0)=2.5; F*(0)=1.246; F*(L4)=0.955 | ≤0.5% | **PASS** (v5.2 Q4) | The steel-bar-mat / welded-grid curves are now BUILT INTO `Kr_Ka_ratio`/`F_star_metallic` (reinforcement_type "bar_mat"/"welded_grid"/"metallic_grid"): Kr/Ka 2.5→1.2 over 0–20 ft and F* 20(t/St)→10(t/St) over 0–20 ft. `check_internal_stability` auto-selects them for a `metallic_grid` reinforcement (e.g. `WELDED_WIRE_GRID_W11`), so the high-level internal path matches a bar-mat wall directly. The ribbed-STRIP default (1.7→1.2; 2.0→tanφ) is byte-identical/unchanged. Closes the former coverage gap. |
| V-011 | seismic_geotech Mononobe-Okabe KAE (**regression anchor**) | **KAE = 0.4782** | KAE = 0.4785 | −0.06% | **PASS** | φ=30, δ=30, kh=kmax=0.206, kv=0, vertical wall, level backfill. δ=φ=30 with kv=0 is the case the battered-wall M-O fix targeted; the corrected sign/degrees/kv handling gives KAE within ±0.06% (tolerance ±2%). No bug; value pinned to catch a future M-O regression. |
| V-011 | seismic_geotech M-O seismic sliding chain | PAE=19.65, PIR=6.09, THF=24.64, V=67.52, R=38.98, CDR=1.58 k/ft | PAE=19.65, PIR=6.09, THF=24.64, V=67.52, R=38.98, CDR=1.58 | <0.5% | **PASS** | Full GEC-11 E7 Step-8 chain built on the module KAE: PAE=0.5γh²KAE; THF=PAE·cos30+PIR+0.5·qLS·H·KAE; V=W+PAE·sin30; R=V·tan30; CDR=R/THF. Reproduces the example to <0.5%. |
| V-056 | retaining_walls MSE, sloping backfill (GEC-11 E3, 2H:1V slope that continues), `check_external_stability_lrfd` | Kaf 0.5367; sliding 1.349/1.598/1.074; eL 2.90/2.34/3.81 ft; σv 10.19/6.83/7.16/9.02 ksf; bearing CDR 1.03/1.54/1.047/1.164 | Kaf 0.537; sliding 1.35/1.60/1.07; eL 2.90/2.35/3.81; σv 10.19/6.83/7.16/9.02; CDR 1.03/1.54/1.05/1.16 | ≤0.01 | **PASS** (GM-112, high-level) | Built for GM-112 (2026-10-16): Coulomb Kab with δ = β (GEC-11 Eqs 4-2/4-3), FT = ½·Kab·γ·h² on h = H + L·tanβ (Eq 4-7), FT cosβ driving and FT sinβ resisting with the same EH factor (Eqs 4-11/4-13/4-16/4-21), V2 the slope fill over the zone. Values: Str I max / min / Service / critical. The ASD path gives the Service I column (e 2.61 ft, σv 7.16 ksf). |
| V-056 | retaining_walls MSE internal under the slope (E3 Level 4, Z = 8.75 ft) | σH 0.99 ksf (×1.35); Kr 0.419; La 10.60, Le 13.42, Zp 17.39 ft; F* 1.420 | σH 0.99 ksf; Kr 0.419; La 10.59, Le 13.41, Zp-ave 17.40 ft; F* 1.420 | ≤0.1% | **PASS** | Seq = ½(0.7H)tanβ added to σv for Tmax (Fig 4-11, Eq 4-29), Ka at β = 0; H1 = 35.29 ft for the inextensible active zone (Fig 4-9a); σv = γr·Zp for pullout, traffic left out (Fig 4-15, p 4-49, Fig E3-5). The strip Kr/Ka and F* reach their deep values at 20 ft as E3's do (before the GM-112 review at 6.0 m: F* 1.411, −0.6 %). |
| V-056 | retaining_walls MSE internal pullout and tension, every level (E3 Table E3-7.3, p. E3-19; GM-256) | phi Pr 9.40 to 20.24 k/strip; Ng phi Pr / Tmax and Ng phi Tn / Tmax at Levels 1-12, e.g. Level 4: 2 x 12.23 / 12.35 = 1.98 and 2 x 9.75 / 12.35 = 1.58 | phi Pr within 0.3 %; FOS x 0.90/1.35 (pullout) and x 0.75/1.35 (tension) within 0.4 %; Level 4 1.978 / 1.577 | ≤0.4% | **PASS** (GM-256) | The module now takes the coverage ratio Rc = b/Sh (GEC-11 Vol I Eq. 3-1, p. 3-12) in pullout (Eq. 4-35, p. 4-49) and in rupture (Tal x Rc, p. 4-47), with E3's own b = 0.164 ft and Sh = 2.50 ft (Levels 1-8) or 1.67 ft (9-12), Tal = Tn / b. Before GM-256 every strip took Rc = 1: Level 4 read FOS_pullout 45.2 (30.2 against the printed 1.98). E5's Level 4 Pr = 9.04 k/strip (p. E5-28) is reproduced by the primitive; the abutment's footing loads are outside the module. GM-267: the tool's built-in strip is E3's 50 x 4 mm strip, Tal 0.55 Fy Ec = 31.93 kN a strip (E3: 0.55 Tn = 31.80 kN); at E3's Sh per level it gives E3's phi Pr at all 12 levels within 0.3 %, and FOS_rupture is E3's tension CDR x 0.55 x 1.35 / 0.75 within 0.4 % (`test_v056_the_built_in_strip_is_E3s`). |
| V-057 | retaining_walls MSE pullout, every layer (E1 Table E1-7.5, p. E1-17; GM-256) | pullout CDR = 0.90 Pr / TMAX at Layers 1-11, 11.4 to 31.1, with F* 0.45, alpha 0.8 'for these geogrids', C 2, Rc 1 (Step 7.8, p. E1-16) | every CDR within 1.2 % from E1's own Zp and Le, with the built-in geogrid's alpha 0.8 and F* 0.67 tan 34 = 0.452 | ≤1.2% | **PASS** (GM-256) | The module now takes GEC-11 Vol I Table 3-6's alpha by type (p. 3-17): 1.0 steel, 0.8 geogrid, 0.6 geotextile. Before GM-256 every geosynthetic took 1.0 and each CDR would read 25 % high. GM-271: the tool's built-in geogrids are E1's GG-I to GG-III, Table E1-7.3's Tal (1,085, 2,169 and 3,525 lb/ft) to the pound (`test_v057_the_built_in_geogrids_are_table_E1_7_3s`). |
| V-010 | retaining_walls MSE internal, E4's whole wall at its level depths (`level_depths`; GM-273) | Tmax 6.24 / 8.36 / 10.80 / 12.76 / 14.24 / 15.23 / 15.73 / 16.03 / 17.08 / 19.03 k a 5 ft panel (x 1.35); Svt 3.12, 2.50 x 8, 2.52 ft | Table E4-7.4 (p. E4-18) 6.25 / 8.36 / 10.80 / 12.77 / 14.26 / 15.23 / 15.71 / 16.03 / 17.10 / 19.05; Svt Table E4-7.1 (p. E4-11) | ≤0.2% | **PASS** (GM-273, high-level) | The levels at E4's printed Z (1.87 ft, then 2.5 ft apart), each over its contributory height (GEC-11 Vol I pp. 4-46, 4-47) with sigma_H the average at Z- and Z+ (p. E4-13), mats per level (GM-266). Replaces the test geometry that put the levels at Z with Sv 2.5 ft, Level 1 13 % over. FOS_rupture about 1 % under Ng phi_s Tn / Tmax at every level (0.48 Fy against 0.65 Fy / 1.35; spreadsheet 3.38 k against 3.36 by hand). |
| V-057 | retaining_walls MSE internal, E1's whole wall at its layers (`level_depths`, grades per layer; GM-273) | TMAX x 1.35 within 0.3 % at Layers 1-9, +0.6 % Layer 11, +1.4 % Layer 10 (1,386 lb/ft); rupture CDRs within 0.6 % (Layer 10 -1.5 %); pullout CDRs within 1.1 % | Tables E1-7.2 / E1-7.4 / E1-7.5 (pp. E1-14, E1-16, E1-17): Sv 1.67, 2.0 x 8, 1.33, 1.0 ft; Layer 10 TMAX 1,366 | ≤1.5% | **PASS** (GM-273, high-level) | The contributory heights are E1's Sv. Layer 10: E1 takes sigma_H at Z = 18.67 ft, not the middle of its 17.67-19.0 ft, and prints less than either (at Z the module gives 1,407); the module is on the safe side. |
| V-057 | retaining_walls MSE, broken backslope + traffic behind the crest (GEC-11 E1) | I 12.68°, Kab 0.3604; sliding 1.119; e 3.80 ft; eB 2.77 ft; σV-F 6.56 ksf, CDR 1.601 | I 12.7°, Kab 0.360; CDR 1.12; e 3.80; eB 2.77; σV-F 6.55 ksf, CDR 1.60 | ≤0.2% | **PASS** (GM-112, high-level) | `broken_backslope_height` = S = 9 ft: Kab with β = δ = I = atan(S/2H) (Fig 4-4), h = H + min(L·tanβ, S) (p 4-20); traffic F2 = Kab·q·h inclined at I with the LL factor. E1 counts FV2 as sliding and eccentricity resistance: pinned with `traffic_vertical_resists=True`. The default follows Vol I p 4-22 (resisting loads only if permanent): sliding 1.10, e 4.01 ft, bearing unchanged (hand-checked from E1's loads). Seq 3.50 ft (E1 3.51 with β rounded to 26.6°). |
| V-058 | drilled_shaft side (sand), GEC-10 (2018) Example 10-1, the defaults (Eq. 10-13; sand from the ground surface), `sigma_p_correlation="silty_sand"` | β 0.967 / 0.976; fSN 835 / 2,233 psf; RSN 196.6 / 631.5 kips; Σ 828.1 kips | β 0.97 / 0.98; fSN 837 / 2,248 psf; RSN 197.2 / 635.6 kips; Σ 833 kips | ≤0.7 % | **PASS** (GM-241, high-level) | The gap is the example's rounding: OCR to one place (6.7 / 6.1 against 6.69 / 6.07), Ko to two (1.24 / 1.13 against 1.238 / 1.123) and Layer 1's β to two places before fSN = β σ′v; taken the same way Eq. 10-13 gives 837 / 2,248 psf and 833 kips (`test_v058_the_printed_rounding_gives_833`). σ′p 5,768 / 13,890 psf (m = 0.8). Its RSN2 line prints '698,659 lb' for 635.6 kips. Side resistance from the ground surface, the tool's default in sand since the GM-241 review; φ′ from (N1)60 by Eq. A-2 gives 38.0 / 40.8° (printed 38 / 41). |
| V-059 | axial_pile layered (Nordlund in sand and gravel, α in clay), GEC-12 Vol 3 Pier 2, HP 12x74, Table D-51, with the example's choices (smooth steel curves, Ca held at the chart's end, σ′p not limited) | Rs 18.7 / 163.7 / 344.2 kips; Rp 265.2 / 31.5 / 275.1 kips (D = 9.99 / 39.99 / 59.99 ft) | Rs 18.5 / 161.0 / 339.6; Rp 273.0 / 31.4 / 275.4 | shaft +1.1 % to +1.8 % at every depth; toe −2.9 % (φ 43), +0.5 % (9 su), −2.1 % to −0.1 % (φ 38) | **PASS** (GM-282) | Profile from Tables D-49/D-50 and Figs. D-37/D-39: water table at the ground surface, pile cap 10 ft below it, Layer 3 in six 5 ft sublayers. With the module's defaults the clay shaft is +11 % (178.5 kips at 39.99 ft: Fig. 7-17's corrugated curve for H-piles, p. 247) and the Layer 4 toe −13 % (240.3 kips: Step 8). GEC-12 factors it layer by layer (Vol 3 p. 111: 0.45 Nordlund, 0.35 α), checked by its parts (the factored resistance is printed only as Fig. D-54). |
| V-060 | pile_group rigid cap, p-multipliers, group efficiency, Meyerhof group settlement, block failure: GEC-12 Vol 3 North Abutment (Tables D-13, D-14, D-23; pp. 50, 56, 136, 137, 203) and UFC 3-220-20 Appendix B-5.4 and Table B-32 | rows 2908.75 / -403.75, 2069 / -231, 1810.5 / -270.5 kips; per pile 323.2 / 264.4 / 223.8 kips; 6-DOF with equal springs the same, Vuy 47.0 / 38.5 / 32.5 kips; p_m 0.90 / 0.625 / 0.5 and 1.0 / 0.85 / 0.7; settlement within 0.45 % at all ten Table D-23 entries; block 14,494 kips, factored (0.60) 8,696; Mokwa 0.82 / 0.67 / 0.58 / 0.52, 0.593 | rows 2908 / -406, 2069 / -231, 1811 / -271; per pile 323 / 265 / 224, Vuy 47 / 39 / 33; the same p_m; Table D-23; 14,494 and 8,696 kips; Table B-32 | exact but for the print's rounding | **PASS** (GM-286) | Two print slips kept as printed: Table D-13's Strength I back row -406 against -403.75 by its own loads, and Table D-14's -46 kips per pile against -44.9. UFC B-5.4 measures B and Z to the centres of the outer piles (18 ft); to their outer faces, as GEC-12 Fig. 7-31 draws, 19 ft and 15,813 kips. No held document prints a 6-DOF (battered) group benchmark: the 6-DOF method is pinned only on this vertical-pile case. |
| V-061 | cantilever_wall external stability, UFC 3-220-20 App. B-1 (cut wall, front batter, soil over the toe, traffic surcharge) and FHWA NHI-06-089 Example 10-3 (Coulomb with δ = β on the plane through the heel, front batter, `bearing_pressure="linear"`) | UFC: R = 2540 B − 4560 at every B; Ph 6,795; FSL 1.500 / 1.584 (B 9.09 / 9.5 ft); FOT 2.487 / 1.000 (9.5 / 6.21 ft); e = B/6 at 9.11 ft; x0/B 0.353; bearing x0 1.805 ft, θ 23.686° (7.5 ft). NHI: Ka 0.3495; Pa 9,550; W 28,112; Mw 211,870; FSs 1.719; d 5.55; e 1.00; q_max 3,312 / q_min 1,233 psf; FS bearing 6.04 | UFC: R = 2540 B − 4560; Ph 6,795; FSL 1.50 / 1.58; FOT 2.49 / 1; middle third at 9.11 ft; x0/B 0.35; x0 1.805, θ 23.686°. NHI: Ka 0.35; Pa 9,564.2; W 28,112.6; Mw 211,881.1; FSs 1.72; d 5.55; e 1.0; q 3,318.1 / 1,227.3 psf; FS 6.03 | UFC exact; NHI ≤0.5 % | **PASS** (GM-288) | The examples round Ka (UFC 0.25, NHI 0.35 against Coulomb's 0.3495, which puts NHI's thrust 0.15 % above the module's). Table B-4's x0/B is the resultant without the surcharge (B-1.8's x0 = 3.36 ft), not B-1.6's with it. The module's default bearing pressure on soil is the uniform q_eq = V/B′ (UFC Eq. 4-27; NHI Fig. 10-17; GEC-6 Eq. 6-6), so NHI's FS on q_max is reproduced with `bearing_pressure="linear"`; B-1.6's R/(2 x0) is the module's q_eq. Both draw the stem with the front face battered (`stem_batter="front"`). |
| V-062 | site_coefficients (Fpga, Fv, As = Fpga PGA) and Mononobe-Okabe KAE and PAE, FHWA GEC-11 Vol II Example E7 (pp. E7-3 to E7-6) | Fpga 1.00; Fv 1.0; kmax 0.206 g; KAE 0.4782; PAE 19.65 k/lft | Fpga 1.0; Fv 1.0; As 0.206 g; KAE 0.47823; PAE 19.649 k/lft | exact to the printed digit | **PASS** (GM-293) | The example gives no Ss; Site Class B's Fa is 1.0 at every Ss (GEC-12 Vol I Table 7-28). Its KAE is UFC 3-220-20 Eq. 4-15 with delta = phi' = 30 deg, a vertical face and level backfill; through the tool as well. |
| V-025 | sheet_pile earth-pressure builder (layered Ka + water) | Ka1=0.249, Ka2=0.333; σ pts 129.5/173.2/377.8/644.2 psf, u=1248; FTOTAL=24,611 lb/ft | Ka1=0.249, Ka2=0.333; 129.48/173.16/377.76/644.16/1248; FTOTAL=24,610.9 | 0.00% | **PASS** | `rankine_Ka` + `active_pressure` reproduce every stress ordinate at the Ka-discontinuity (4-ft boundary, same overburden × Ka1 vs Ka2) and the layered-moist/submerged + hydrostatic water blocks. Total driving force exact to 0.00% with source-rounded Ka (≈359 kN/m). |
| V-012 | sheet_pile cantilever Ka/Kp (soldier pile) | Ka=0.271, Kp=3.69 | Ka=0.271, Kp=3.69 | exact | **PASS** (primitive) | `rankine_Ka`/`rankine_Kp`(φ=35) match the published coefficients exactly. |
| V-012 | soldier-pile arching / simplified toe-moment | (not in module; hand: f=2.8, fb=5.6 ft, D0=12.27, D=14.73 ft, Y=6.0, Mmax=379.7 kip-ft, Vmax=137.7 kips) | D0=12.27, D=14.73, Y=6.0, Mmax=379.7, Vmax=137.7 | <0.1% (hand) | N/A (scope) | `analyze_cantilever` is a CONTINUOUS per-metre free-earth-support solver — no soldier-pile effective-width/arching (active on 2-ft hole width, passive on f·b=0.08φ·b=5.6 ft) and no simplified toe-moment cubic. The published Simplified row is reproduced by hand on the module Ka/Kp to document the method. |
| V-012 | analyze_cantilever (continuous wall) | D_conv ≈ 11.3 ft (per-metre framework) | D0=12.27 ft (per-pile) | — | CONVENTION | Continuous-wall embedment is a different framework than the per-pile simplified D0 (active+passive spread over a continuous wall, no 2-ft/5.6-ft widths). Not comparable; pins the framework gap. |
| V-013 | sheet_pile anchored Rankine Ka | Ka=0.333 | Ka=0.333 | exact | **PASS** (primitive) | `rankine_Ka`(φ=30) matches exactly. |
| V-013 | log-spiral passive Kp (`caquot_kerisel_Kp`, v5.3) | Kp=4.70 (φ=30, δ/φ=0.5) | log-spiral Kp=4.7 (Caquot-Kerisel 6.3×0.746) | 0% | **PASS** (v5.3) | The Caquot-Kerisel log-spiral passive is now BUILT IN (`caquot_kerisel_Kp`, mirrored in sheet_pile + soe): Kp'=R·Kp0, base Kp0(δ=φ)=6.30 (Caltrans Fig 4-20), R=0.746 (Caltrans Matrix 4-1 / NAVFAC DM-7.2) → 4.70. Sits between Rankine (3.0) and Coulomb (4.98). Selectable as `pressure_method="log_spiral"` in analyze_cantilever/analyze_anchored (Coulomb active + log-spiral passive). Not tuned (digitized-chart values). |
| V-013 | FHWA apparent-diagram anchor quantities (`fhwa_apparent_pressure_anchored_wall`, v5.3) | σ_a=934.4 psf; PT=15,573; T1U=6,229 (high-level) | P=11,980; PT=15,574; σ_a=934.4; T1U=6,228 | <0.1% | **PASS** (v5.3, high-level) | The single-anchor FHWA apparent envelope is now NATIVE: `soe.fhwa_apparent_pressure_anchored_wall` computes pe = 0.65 Ka γ H²/(H−H1/3−Hn+1/3) = σ_a = 934.4 psf, the total apparent load PT = 1.3·P = 15,573 lb/ft, and the upper-tributary T1U = 6,229 lb/ft directly from the wall/soil inputs. The single-anchor TOTAL anchor force and embedment are native too since 2026-09-15 (next row). |
| V-013 | analyze_anchored embedment (log-spiral now available) | D ≈ 7.2 ft (classical FES + log-spiral Kp=4.7) | D=6.09 ft (apparent-diagram FES + log-spiral Kp) | — | CONVENTION | With the new `pressure_method="log_spiral"`, `analyze_anchored`'s classical (TRIANGULAR-active) FES embedment drops from ~10.8 ft (Rankine) to ~7.2 ft — closer to the published 6.09 ft, which additionally uses the FHWA 1.3× APPARENT-diagram active distribution (not the triangular one). The apparent-diagram FES solver is now built in `soe` (next row); `sheet_pile.analyze_anchored` stays classical. The log-spiral passive coefficient itself is exact (row above). |
| V-013 | apparent-diagram FES, single anchor (`soe.fhwa_apparent_pressure_anchored_wall(Kp=4.7, FOS_embedment=1.3)`, 2026-09-15) | D=6.09 ft; D′=4.90 ft; T1=14,243 lb/ft; M=22,494 ft-lb/ft at T1; zero shear 19.67 ft | D=6.09 ft; D′=4.89 ft; T1=14,254 lb/ft; M=22,494; zero shear 19.69 ft | ≤0.2 % | **PASS** | Free earth support about the anchor (`soe.free_earth.solve_about_support`): FHWA trapezoid above the dredge line, Rankine active + log-spiral passive below, MR = 1.3·MD for D, MR = MD for D′ and the anchor load. Closes the V-013 residual. Built for field feedback N2/N3 (the old single-anchor result stopped at T1U without saying so). |
| V-014 | basal heave — Caltrans force balance (`check_basal_heave_caltrans`, v5.3) | qu=3.79 ksf, F_RS=39.8, W=37.8, +q 3.15, −S 15.0, F_dr=25.95, FS=1.533 (high-level) | F_RS=40.0, F_dr=26.0, FS=1.54 | FS −0.5% | **PASS** (v5.3, high-level) | The new `soe.check_basal_heave_caltrans` reproduces the Caltrans force balance NATIVELY, INCLUDING the sidewall-shear resistance S=cu·H (the term the Bjerrum-Eide bearing ratio omits): F_RS=cu·Nc·(0.7B) vs F_dr=0.7B·H·γ+0.7B·q−cu·H → FS=1.533 (Nc=7.575 from Fig. 10-15 as read, 8.48 × (0.84 + 0.16/3), GM-290; Skempton's formula, 7.68 and FS 1.55, before; exactly 1.538 with the chart Nc=7.6 override). All block terms (W=37.8, q=3.15, S=15.0, F_dr=25.95 k/ft) match. Not tuned. |
| V-014 | check_basal_heave_bjerrum_eide | FOS=0.97 (cu·Nc/(γH+q)); module Nc=7.575 | FS=1.54; chart Nc=7.6 | — | N/A (scope) | Module uses the inverted-footing bearing FOS = cu·Nc/(γH+q) — NO sidewall-shear term, NOT the 0.7B force block — so FOS=0.97 (far more conservative). Its Nc is the Caltrans chart's since GM-290 (an unprinted table, 6.71 and FOS 0.86, before). Different formulation. |
| V-016 | soe Ka + FHWA pe + ps | Ka=0.295, pe=43.6, ps=3.2 | Ka=0.295, pe=43.6, ps=3.2 | <1% | **PASS** | `soe.rankine_Ka`(φ=33)=0.295; two-anchor trapezoid pe=0.65·Ka·γ·H²/(H−H1/3−H3/3)=43.6; ps=Ka·qs=3.2. |
| V-016 | tributary anchor loads + R + DL | TH1=168.4, TH2=172.1, R=36.7, DL1=435, DL2=445 | TH1=168, TH2=172, R=37, DL1=435, DL2=445 | <2% | **PASS** | Tributary-area horizontal anchor loads, subgrade reaction R=(3/16)H3·pe+(H3/2)ps, and anchor design loads DL=TH·s/cos15 all reproduce to ≤2%. |
| V-016 | hinge moments M1 / M2,3 | M2,3=66 (exact); M1=75.6 | M2,3=66; M1=Mmax=76 (75.6) | <1% | **PASS** | M2,3=(1/10)H2²(pe+ps)=66 exact. M1 as printed (p. A-6) = 13/54 H1² pe + ps H1²/2 = 65.6 + 10.0 = 75.6, the surcharge's uniform pressure on the cantilever (GM-290; the module took (13/54)H1²(pe+ps) = 70.4 before, filed here as a CONVENTION). |
| V-015 | slope_stability slice table (pinned circle) | θ_i, W_i exact; ΣW·sinθ=116.49, ΣW·cosθ=137.09 kips/ft | ΣW·sinθ=116.49, ΣW·cosθ=137.09 | 0.00% | **PASS** | The pinned circle (center 0,60 ft directly above the toe, R=60 ft) + the 4V:3H face reproduce the Caltrans Table 10-1/10-2 slice table EXACTLY — all six θ_i=asin(x_i/60), weights W_i, and both force sums. Geometry reconstruction verified before any FS comparison. |
| V-015 | slope_stability Fellenius (OMS) | 0.863 (module pipeline, x=0→57.6, ΣdL→77.2 ft); 0.874 (published formula w/ source L=113.55) | 0.87 | −0.007 / +0.004 | **PASS** | `fellenius_fos` over the geometrically-correct mass converges to 0.863 (6 slices 0.866); feeding the source's hand arc length L=113.55 ft into the published formula gives 0.874. Both within ±0.05. The only spread is the cohesion term: source's hand L=113.55 ft over-states the discretized base length (true lower-arc 0→57.6 ft is 77.2 ft). No module change. |
| V-015 | slope_stability Simplified Bishop | 0.9595 (rigorous fixed point) | 0.90 (source, under-iterated) | +6.6% | CONVENTION | `bishop_fos` returns the proper fixed point FS=FSa=0.9595 on the source's own 6-slice table. The manual shows only two hand iterations (FSa=1.5→1.10; FSa=0.8→0.90, reproduced EXACTLY here incl. the Hb=m_α column) and declares "converges to ≈0.9" — one step short. Re-iterating the source's OWN table to convergence gives 0.96, matching the module. Module is the more-correct value; published 0.90 is not converged. |
| V-017 | lateral_pile FD solver (verification) | fixed-head y & M_head match Reese-Matlock T-method to 4 figures (linear p=k·z·y) | (analytical T-method) | <1% | **PASS** | With a linear soil law the beam-column solver reproduces 0.93·V·T³/EI (fixed-head groundline y) and ≈−0.93·V·T (head moment) exactly. Isolates the solver + fixed-head BC (slope=0) as correct, so the V-017 deflection excess is a p-y construction difference, not a solver bug. |
| V-017 | lateral_pile fixed-head Mmax (Reese sand, axial) | −36.8 kN·m (at head) | −37.3 kN·m | −1.2% | **PASS** | GM-287 (2026-10-31): −39.3 kN·m (+5.4%) before the sand coefficients came from GEC 9 Eqs. A-22 to A-24 and the four-segment curve became the default. `SandReese` static p-y, 2-layer profile (φ=32/30, k=24430/16287 kN/m³, γ′=18.84/17.64), head 0.305 m below grade (overburden +0.305 m), V=44.482 kN, slope=0, axial P=1423.4 kN with P-delta. Mmax within ±10%. |
| V-017 | lateral_pile fixed-head deflection | 3.32 mm (composite EI); 3.57 mm (casing-only) | 3.3 mm | +0.6% (composite) / +8% (casing-only) | **PASS** (GM-287) | **GM-287 (2026-10-31):** with C1, C2, C3 from GEC 9 Eqs. A-22 to A-24 (a table no page prints before) and GEC 9's four-segment Reese curve as the default, both EIs read within ±10%: 3.57 mm casing-only, 3.32 mm composite; the 'simplified' curve now reads 2.83 mm. The history below is before GM-287. **Root cause confirmed (v5.3 A4): the gap is the section BENDING STIFFNESS, NOT the p-y construction — and v5.4 E5 closes it.** The published LPILE run used NONLINEAR/composite EI (extract mp_sp2.txt "Computed Pile Response Using Nonlinear EI"); the section is a grout-filled steel casing (OD=0.1969, wall=0.0151, grout f'c=27.6 MPa). The NEW `composite_section_ei` helper computes the uncracked transformed EI = 8109 kN·m² (×1.13 the 7171 casing-only), and feeding it through the SAME p-y model/loads gives **3.60 mm — within ±10% of 3.3 (PASS)**, NOT tuned (EI is pure geometry + ACI 318 Ec + the published steel modulus). The casing-only EI still reads 3.92 mm (+19%); the full Reese (1974) construction is SOFTER (5.02 mm), so the p-y construction was never the cause. The uncracked composite is an upper-bound stiffness; the residual +9% is the cracked/moment-curvature (M-φ) EI LPILE actually used, out of scope here (owner decision pending). k (24430.244/16286.830 kN/m³, LPILE echo) used verbatim. Mmax row PASS unchanged. |
| V-018 | ground_improvement / settlement unimproved consolidation | S = 22.0 in (settlement module + closed form) | S = 22 in | 0.0% | **PASS** | `consolidation_settlement_layer` (NC clay, Cc=0.25, eo=0.7, H=15 ft, po=432 psf, dq=2500 psf) reproduces S=Cc/(1+eo)·H·log10((po+dq)/po) to the printed digit (0.559 m). Tight closed-form anchor, independent of any GI method. |
| V-018 | ground_improvement `area_replacement_ratio` (RAP) | as_tri = 0.2744 | Ra = Ac/(π/4·de²) = 0.27 | <0.5% | **PASS** (primitive) | The triangular tributary √3/2·s² numerically equals the de=1.05·s unit cell π/4·(1.05s)² (both 21.65 ft²), so the module's geometric `as` matches the source's de-based Ra to 4 figures. |
| V-018 | RAP improved settlement (upper-zone stiffness modulus qg/kg) | hand: qg=6324 psf, suz=0.68 in (module `as`); module SRF path gives ~9.4 in | qg=6383 psf, suz≈0.7 in | suz <3% (hand) | N/A (scope) | The GEC-13 two-layer upper-zone method (qg=q·ns/(Ra·ns−Ra+1), suz=qg/kg, kg=pier stiffness modulus pci) is NOT in the module. `analyze_aggregate_piers`/`improved_settlement` use the equal-strain SRF=1/(1+as(n−1)) model (no kg) → SRF=0.43, improved ≈9.4 in — a fundamentally different method. Published 0.68 in reproduced by hand on the module `as`; module not tuned. |
| V-019 | ground_improvement / settlement unimproved consolidation | S = 27.2 in (settlement module + closed form) | S = 27 in | <1% | **PASS** | `consolidation_settlement_layer` (Cc=0.2, eo=0.6, H=50 ft, po=1440 psf, dq=1875 psf) reproduces S=27.16 in (0.690 m) exactly. Tight anchor. |
| V-019 | ground_improvement `priebe_improvement_factor` (stone column; GEC-13 Fig. 5-21 curve (4) as printed) | settlement ratio = 2.55 at A/A_sc = (s/d)² = 3.61 (as = 0.277) → S = 27/2.55 = 10.6 in | 2.7 → 10 in (Fig 5-27, p. 5-49) | −5.6 % (−0.15) / +0.6 in | **PASS** (chart read, ±0.3) | Since GM-276 the module reads GEC-13's printed Priebe curve only (the Priebe formula and its 42.5° default, printed nowhere in GEC-13, removed; owner OWN-70). The curve reads 2.55 at 3.61 on both printed copies; Fig. 5-27's arrows are drawn at A/A_sc 3.53, where the curve reads 2.63, and labelled 3.6 and 2.7. The module's geometric triangular as = 0.251 (A/A_c 3.98) gives 2.31: the example takes (s/d)². |
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
| V-023 | fem2d analytical anchors (S, c, p0) | S=1.554e-9 1/Pa; c=0.0643 m²/s; p0_consistent=0.839e5 Pa | S=1.554e-9; c=0.0643; p0 (formula) 0.839e5 / (Itasca reported 0.981e5) | exact | (anchors) | Storage S=1/M+α²/(K+4G/3), c=k/S, and the Biot 1-D undrained p0 verified inline. The inventory's stated 0.981e5 is Itasca's reported value, which needs an effective M≈10× the stated 4e9 (near-incompressible fluid); the inventory's own formula gives 0.839e5 (consistent with the stated M). The monolithic Taylor-Hood scheme reproduces this 0.839e5 p0 exactly (next row). |
| V-023 | fem2d undrained p0 + p(z,t) consolidation decay | monolithic u-p (Taylor-Hood T6/T3): interior undrained p0 = 83.92 kPa; base decay p/p0 within <1% of Terzaghi (θ=0.5) / ≤3.1% (θ=1); end-state 2.596 mm | undrained p0 ~0.84e5 Pa; Terzaghi decay p/p0 vs t̂; FLAC <5% err | <1% (θ=0.5) | **PASS** | v5.3 A5 (commit d1e27a7): NEW monolithic (coupled u-p) Biot scheme `solve_consolidation(scheme="monolithic", theta=…)` with a **Taylor-Hood T6-displacement / T3-pressure** pairing (LBB-stable, no stabilization term). The t=0 block solve splits the applied load into effective stress + excess pore pressure: interior/base p0 = **83.92 kPa** EXACTLY matches the formula-consistent 0.839e5 Pa (Itasca's reported 0.981e5 needs an effective M≈10×; tuned to NEITHER). The field then dissipates as the Terzaghi/Biot transient with c=k_mob/S=0.0643, matching the series to **<1%** (θ=0.5, fine steps) / **≤3.1%** (θ=1) at t̂=0.05–1.0 — inside the FLAC <5% envelope. Drained end-state **2.596 mm** (−0.5% vs 2.61). Equal-order CST monolithic shows the classic LBB drained-boundary overshoot (max ≈173 vs 84); Taylor-Hood removes it (residual t=0 boundary-layer ≈119, dissipates immediately). Units for the transient: k=mobility m²/(kPa·s), n_w=M kPa. Default `scheme="staggered"` byte-identical. Tests: `test_v023_monolithic_*`. |
| V-024 | fem2d MC plastic radius (Salencon cavity) | R0=1.735 m (σ_r=12.01 crossing); R0=1.731 m (σ_θ peak) | R0=1.735 m | 0% / −0.2% | **PASS** (slow) | Quarter-symmetry graded T6 annular mesh (a=1, R_out=20a, ~3700 elem, ~2.5 s), in-situ σ=−P0, cavity unloaded P0→0 via the new `initial_stress_relaxation` driver + `roller_base` symmetry BC (both general solver additions). Both R0 detectors (σ_r boundary crossing on the x-axis; σ_θ peak = elastic-plastic boundary) land within ±5%. psi=0 non-associated. |
| V-024 | fem2d radial stress at elastic-plastic boundary | σ_r(R0) ≈ 11.9 MPa (at the σ_θ peak) | (1/(Kp+1))·(2P0−q) = 12.01 MPa | −0.7% | **PASS** (slow) | Boundary radial stress within ~1% of the Salencon value. Kp=3.0, q=2c√Kp=11.95 MPa. |
| V-024 | fem2d far-field stress profile r/a=2..5 | σ_r/σ_θ follow Salencon elastic branch but ~5% LOW (compression under-predicted) | Salencon elastic σ_r=P0−(P0−σ_r,R0)(R0/r)², σ_θ symmetric | ~−5% | CONVENTION | Far-field domain-truncation: the outer boundary is FIXED at R_out=20a (the inventory flags ~1−2%; a rigid fixed boundary adds a bit more than a far-field traction). Profile SHAPE and elastic decay are correct; asserted ±7% with the low-side bias documented. R0/σ_r(R0) (primary targets) pass at ±5%. A pure-traction outer BC pushes the far field to ±3% but is under-constrained (drifts, non-convergence) — fixed is the robust choice. |
| V-029 | slope_stability pore-pressure GRID + ponded (ACADS 5 / Slide2 #10) | — (not run) | Bishop 1.498, Spencer 1.500, GLE 1.500, Janbu-c 1.457; referee 1.53 | — | **N/A (scope)** | The module has NO pore-pressure-GRID input — only a piezometric surface (`gwt_points`) and per-layer `ru`. The #10 pore pressure comes from a TIN-interpolated flow-net grid (Figure 10.2), and neither the grid values nor the excavation height survived the text extraction, so an ru/piezo approximation is not defensible. Capability gap for B2 (pore-pressure-grid + TIN interpolation). No test. **v5.4 (E3): capability BUILT** — `SlopeGeometry.pore_pressure_points` (scattered (x,z,u), Delaunay-linear TIN + nearest fallback, suction clamped ≥0, wired through search; additive, None = byte-identical). Validated rigorously by construction (hydrostatic grid reproduces the piezometric-line FOS to machine precision; TIN exact on linear fields; 6 tests). The published Bishop 1.498 is still NOT pinned — Fig 10.2's flow-net grid values did not survive extraction and inventing them is forbidden; the ACADS-5-style run is a capability demonstration (dry 1.89 → grid 1.25). Not tuned. |
| V-030 | slope_stability Duncan (2000) LASH underwater slope (Slide2 #29) — deterministic FOS + reliability | det. Spencer 1.218, GLE 1.220, Janbu-c 1.207 on Duncan's noncircular surface (1.190 / 1.192 / 1.182 before GM-131); reliability F=1.17, COV_F≈0.16 → Pf 18.2% | det. Slide2 Spencer 1.157 / GLE 1.160 / Janbu-c 1.168 (Duncan quote 1.17); Pf 18% (Taylor) | Spencer +5.3% (+2.9% before GM-131); Pf +0.2% | **PASS** (det. FOS + reliability + input-COV FOSM, v5.4 F1) | Geometry recovered from labeled Fig 29.1. (1) DETERMINISTIC: on Duncan's specified noncircular failure surface, with the linearly-increasing Bay-Mud undrained strength (su=100 psf @ el −20, +9.8 psf/ft) discretized into thin undrained sub-layers and the ocean at el 0 (ponded buttress), the rigorous methods reproduce Spencer 1.157 / Duncan 1.17 to +2.9% (insensitive to the near-surface su floor); only OMS shows the submerged-noncircular over-estimate (1.44). (2) RELIABILITY: the Taylor-series helpers reproduce Duncan's F=1.17→Pf 18% and bracket the Slide2 per-method Pf table. (3) **INPUT-COV FOSM — gap CLOSED (F1).** The new `linear_su` law treats su(z)=a+b·(datum−z) as ONE correlated (a,b) pair applied coherently across the sub-layers, so a shared su-gradient is no longer over-counted; `gamma_sat` was added as a var, since the dry γ is inert on a fully submerged slope. Propagating the STATED Table 29.2 inputs (unit-weight COV 0.033 via gamma_sat + su-gradient std 1.2 psf/ft) → COV_F 0.133 → Pf ~11% (Pf ~13% at Duncan's F=1.17, in the Slide2 13–14% band); the unit weight is buoyancy-amplified (gamma_sat ~42% of Var[F] despite COV 0.033, because the buoyant weight COV is ~0.088). Duncan's higher closed-form COV_F 0.16 / Pf 18% is recovered when the su uncertainty is the full coherently-scaling profile COV (a,b both 0.16) → COV_F 0.160 → Pf 18.2% at F=1.17 — the regression-SLOPE std alone (1.2 psf/ft) under-captures the total su COV (intercept+scatter). FOSM≈MC. Tests: `test_published_v030_fosm_slope.py` (5). NOTE the Slide2 beta_LN & Pf columns are Latin-hypercube outputs that do NOT obey pf=Φ(−β); only Duncan's quote is the closed-form anchor. **GM-131 (2026-10-17):** a noncircular surface is sliced at its corners. Duncan's surface reads 1.218 at any slice count; the 1.190 was 60 even slices across its corners (the old slicing read 1.226 at 240, 1.215 at 960, 1.218 at 3,840). The +5.3 % on Slide2 is not slicing (insensitive to the su floor and to the number of sub-layers) and is open (BACKLOG). |
| V-031 | slope_stability 2-layer + water, circular (Pockoski-Duncan / Slide2 #57) | Bishop 1.411, Spencer 1.415, GLE 1.398 (Janbu-c 1.374, OMS 1.159) | Table 57.3 circular: Bishop 1.417, Spencer 1.422, Janbu-simp 1.263, Ordinary 1.319 | Bishop −0.4%, Spencer −0.5% | **PASS** | Recovered geometry (labeled Fig 57.1): 2.5:1 50-ft slope, sandy-clay over highly-plastic-clay band (el 90-85), water table. Bishop & Spencer reproduce the Slide2 circular values to <0.6%. Janbu (module f0-CORRECTED, 1.374) sits above Slide2 Janbu-SIMPLIFIED (1.263); OMS-with-water (1.159) below Slide2 Ordinary (1.319) — usual method-definition conventions, not surface disagreement. **GM-134 (2026-10-17):** circles cut at the layer and water-table crossings, each slice its chord: Bishop 1.421 / Spencer 1.425 / GLE 1.409 on 50 slices (1.411 / 1.415 / 1.398 before; 1.399-1.446 on 25-45 slices, now 1.420-1.425), within 0.3 % of Slide2; the Bishop search 1.416 (1.397). Pinned for slice-count independence in slope_stability/tests/test_circle_slicing.py. **GM-138 (2026-10-18):** the default Spencer search (30 slices) 1.4177 → 1.4174, against 1.4173 for the critical circle by an independent brute force (1.4200 on 480 slices; Slide2 1.422); pinned within 0.01 of it. |
| V-032 | slope_stability homogeneous M-C (Baker 2003 ex3 / Slide2 #61) | Spencer 1.365, Bishop 1.367, Janbu-c 1.373 | Spencer 1.366, Janbu-simp 1.291 (M-C) | Spencer −0.07% | **PASS** | Recovered geometry (labeled Fig 61.1): 45°, 6 m homogeneous clay (c'=6/φ'=32/γ=18). Spencer matches to <0.1%. The power-curve (nonlinear-envelope) variant (FS 1.48) is out of the module's scope (no power-curve strength model). **Caution (2026-10-14):** the <0.1% match is on circle (-1.3, 9.7, R 9.3), not Slide2's printed critical circle (-1.665, 9.968, R 10.106), which gives Bishop 1.324 / Spencer 1.322 here (independent hand Bishop 1.3244), 3.2% below its printed 1.366. The search (with the GM-13 toe-circle family) finds a toe circle at Bishop 1.315 (was 1.320). |
| V-033 | slope_stability pseudo-static SEISMIC, homogeneous (Loukidis 2003 ex1 / Slide2 #62) | at kc=0.432: Spencer 1.005, Bishop 0.994, GLE 1.005 (static Bishop 2.55) | Spencer FOS = 1.000 at kc=0.432 (Slide2 circular 1.001) | +0.5% | **PASS** | Recovered geometry (labeled Fig 62.1): 3:1 25-m homogeneous clay. Reproduces Loukidis' critical-seismic-coefficient FOS=1.0 at kc=0.432 to +0.5%. **KEY: this clean fully-labeled geometry validates the module's pseudo-static seismic engine** — so the ACADS #4 3-layer seismic shortfall (V-027) is a geometry-reconstruction limit, not a kh bug. |
| V-027 | slope_stability 3-layer + seismic 0.15g (ACADS 1(d) / Slide2 #4) | static Bishop 1.02, seismic 0.77 (soil assignment VERIFIED correct) | static (#3) Bishop 1.405 / Spencer 1.375; seismic (#4) Bishop 1.016 / Spencer 0.991; referee 1.00 | static −28%, seismic −24% | **N/A (geometry discrepancy)** | Discrepancy analysis (not forced): the per-slice soil assignment was verified correct vs the figure (toe/base = Soil#3 φ20, wedge = Soil#2, cap = Soil#1). The critical toe circle is DOMINATED by the weak Soil#3 (φ20) → static 1.02. A full soil-interpretation sweep found NONE reaching the published static 1.405 — the target implies an EFFECTIVE φ≈29 critical surface (it sits between homogeneous all-φ23=1.305 and all-φ38=1.562). The target seismic/static ratio 0.723 matches the HOMOGENEOUS cases (0.712-0.725), not the layered one (0.760); the seismic engine is independently validated to +0.5% by Loukidis #62 (V-033). So the module is correct and the published critical surface must AVOID the weak Soil#3 toe (Soil#3 deeper than the figure read implies) — a geometry-reconstruction limit, hyper-sensitive to the weak-layer boundary. No forced test. |
| V-028 | slope_stability weak seam + piezo, noncircular (ACADS 4 / Slide2 #9) | weak_layer search → Spencer 0.808 (GLE 0.802, Janbu 0.819; 0.792 / 0.786 / 0.804 before GM-131); noncircular_de → ~0.69 | no-opt Spencer 0.760 / GLE 0.720 / Janbu 0.734; opt Spencer 0.707 / GLE 0.683; Slope-2000 GLE 0.6878; referee 0.78 | Spencer +3.6% vs referee (+1.5% before GM-131) | **PASS** (v5.3 B2) | RE-VALIDATED after the SS-6 weak-layer search fix. Before the fix the `weak_layer` search returned spurious DEGENERATE surfaces (Spencer ~0.05-0.18 — a solver-non-convergence artifact); it now reliably finds a smooth seam-following critical at Spencer 0.792 (+1.5% vs referee 0.78), bracketed by Slide2's no-opt 0.760 / optimized 0.707. The optimized `noncircular_de` reaches ~0.69, matching Slide2's optimized 0.707 / Slope-2000 0.6878. Fix: `_compute_fos` now requires GLE convergence for noncircular trials + a low-FOS jaggedness gate (a low FOS on a non-concave surface is a solver artifact). Robust across slice/trial counts (was 0.15 at ns=40). Module still supports only ONE surcharge zone (bench + crest — a separate B2 gap; secondary to the seam-governed FOS). **v5.4 (E8): gap CLOSED** — `SlopeGeometry.surcharges` (list of zones, summed at each x). Both ACADS-4 loads applied: Spencer 0.792 → 0.788, GLE 0.786 → 0.782 (~0.4% — the weak seam governs, as documented). Tests: test_published_v028_e8.py (4). **GM-131 (2026-10-17):** sliced at its corners the seam surface reads 0.808 on 30 to 1,600 slices; the 0.792 was 50 even slices across its corners (0.830 on 51). With both loads Spencer 0.806, GLE 0.799. |
| V-034 | slope_stability 3-layer + seismic (Loukidis 2003 ex2 / Slide2 #63) | seismic Spencer 0.82 at kc=0.155 | Spencer 1.000 at kc=0.155 (Slide2 0.991) | −18% | **N/A (geometry-precision)** | Fig 63.1 is UNLABELED; the pixel-reconstructed 3-layer geometry (~±1.5 m) gives seismic ~0.82 vs 1.0 — the same imprecise-3-layer-geometry issue as V-027. Seismic engine validated by V-033. No passing test. |
| V-035 | slope_stability `infinite_slope_fos` (Duncan-Wright / Slide2 #79) | FOS = tan(30)/0.4 = 1.443 | Bishop/Spencer/GLE 1.443-1.444; referee 1.44 | 0.0% | **PASS** | New `infinite_slope_fos` closed form. Cohesionless 2.5:1 embankment (c'=0, φ'=30) — the shallow surface-parallel mechanism FOS = tan φ'/tan β reproduces the Duncan-Wright / Slide2 referee exactly. |
| V-036 | slope_stability `infinite_slope_fos` (Duncan-Wright / Slide2 #81) | FOS = tan(30)/0.5 = 1.155 | Bishop/Spencer/GLE 1.155; referee 1.15 | 0.0% | **PASS** | Cohesionless 2:1 embankment (c'=0, φ'=30). Exact. The new method also covers c-φ, seepage-parallel and ru cases (unit-tested in slope_stability/tests/test_infinite_slope.py). |
| V-037 | slope_stability rapid drawdown — USACE 2-stage (EM 1110-2-1902 / Slide2 #95) | flat-phreatic 1.207 (bound); steady-seepage stage-1 **1.34** | Army-Corps 2-stage 1.347; referee 1.35 | seepage −0.6%; flat −10% (bound) | **PASS (steady-seepage stage-1)** | `rapid_drawdown_fos(method='corps_2stage')` on the EXACT two-segment face (Fig 95.1, calibrated to the published circle: entry (72,24)/exit (354,110) to <0.5 ft). The default flat full-pool phreatic is the conservative no-through-seepage bound (1.21); supplying the steady-seepage phreatic via the new `stage1_phreatic_points` (declining 110→80 ft, ~0.08 gradient) recovers 1.34 vs 1.347 (~0.6%). The combined R/effective (`min`) envelope is confirmed correct (pure R overshoots to ~1.42). Exact geometry alone does NOT close the gap (≈1.21) — the residual was the stage-1 flow net. Undrained drawdown ≪ drained drawdown (2.01) of the same dam. |
| V-038 | slope_stability rapid drawdown — Duncan-Wright-Wong 3-stage (Slide2 #96) | flat 1.235; steady-seepage stage-1 1.273 | DWW 3-stage 1.443; referee 1.44 | seepage −12% | **CONVENTION (approximate)** | Same dam, `method='duncan_3stage'`: Kc-interpolated undrained strength (Kc=1 R-envelope → Kc=Kf drained) + stage-3 drained substitution (Duncan-Wright-Brandon 2014 Ch.9). Published ORDERING holds (3-stage ≥ 2-stage at the default) and the seepage stage-1 raises it (1.235→1.273), but a ~12% residual to 1.443 remains at the SAME phreatic that validates the 2-stage. Isolated to the Kc anisotropic-consolidation interpolation for a c'=0 soil (drained envelope below the R envelope at low σ'_fc → the anisotropic strength gain is under-captured); NOT geometry/seepage. Documented follow-up (DESIGN.md). **v5.4 (E2) re-diagnosis: the Kc interpolation is sound (~1.45 in isolation); the residual is the STAGE-3 Fellenius effective-normal estimate (inconsistent with stage 1's rigorous GLE normal, over-fires the drained substitution 17/50 slices). Opt-in `stage3_effective_normal='gle'` → flat 1.306 / seepage 1.370 vs 1.443 (~5% residual); default preserved, flip owner-gated.** #98 search wrapper landed as V-041. Framework tested (slope_stability/tests/test_rapid_drawdown.py). |
| V-039 | slope_stability Newmark seismic sliding block (Slide2 #104 / Tutorial 28) | integrator == closed-form rectangular pulse (exact); Jibson(2007) == published eq (exact); ky bisection self-consistent (FOS(ky)=1.000) | #104: no-seismic FS 1.359, k=0.15 FS 0.978, Ky=0.139, Newmark 5.042 cm | integrator/Jibson 0%; #104 record-limited | **PASS (integrator + Jibson) / #104 documented** | New `newmark.py`: `yield_acceleration` (bisect pseudo-static FOS→1, ay=ky·g), `newmark_displacement` (rigid-block double integration, downslope-only, both-polarity), `newmark_jibson2007` (Eq.6 regression 0.215/2.341/−1.438, σ0.51). #104's Tutorial-28 acceleration RECORD and geometry are unpublished, so the integrator is validated against the exact rectangular-pulse closed form D=ap(ap−ay)T²/(2ay) and Jibson against its published equation. #104 structure reproduced qualitatively (published FS 1.359/0.978 ⇒ FS=1 at k≈0.141 ≈ Ky 0.139; Jibson at Ky=0.139 brackets 5.042 cm between PGA 0.35–0.60g). Tests: slope_stability/tests/test_newmark.py + validation_examples/test_published_v039_slope.py. |
| V-040 | slope_stability stabilizing micro-piles (Yamagami 2000 / Slide2 #54) | no-pile 1.114; with-pile (10.7 kN/m at crest) 1.223; pile-force integration exact (phi=0 closed form) | no-pile 1.102 (Yamagami 1.10); with-pile 1.193 (Yamagami 1.20) | no-pile +1.1%; with-pile +2.5% | **CONVENTION (pile reinforcement validated)** | New `StabilizingPile` (specified shear OR Ito & Matsui 1975 plastic force) on `geom.stabilizing_piles`, resolved via `compute_reinforcement_forces` (active convention) where the pile crosses the slip surface. #54 uses a specified 10.7 kN/pile shear at 1 m spacing on the labeled Fig 54.1 geometry. The +2.5% over-prediction of the pile benefit = active-vs-passive support convention + figure-read pile location + single-pile search subtlety (re-search finds a pile-avoiding surface at ~1.11, so the pile is applied to the recovered critical surface, as Slide reports). The Ito-Matsui c-φ FORMULA (`ito_matsui_lateral_force`/`ito_matsui_pressure`) implements the ORIGINAL 1975 Eq. 13/23 (exp arg tan(π/8+φ/4), coeff 1/(Nφ·tanφ)) and is unit-tested against the paper's exact hand-check values (c=10/φ=20/z=5/D1=2/D2=1.5 → 105.079; φ=0 c=25/z=4/D1=2/D2=1 → 146.683) PLUS the #106 spacing trend (force/m falls as spacing/diameter grows). #106's cross-section is not in the manual so a single-surface FOS is not reproduced. Tests: slope_stability/tests/test_reinforcement.py::TestStabilizingPile + validation_examples/test_published_v040_slope.py. **v5.4 (E6):** `StabilizingPile.support_convention='passive'` (Method B — force added to the resisting side; circular moment methods) narrows the with-pile residual: Bishop 1.223 (+2.5%) → 1.203 (+0.8%) vs published 1.193. Default 'active' unchanged. Tests: test_published_v040_e6.py (5). |
| V-041 | slope_stability rapid-drawdown SEARCH minima (Walter Bouldin / Slide2 #98) — v5.4 E1 | Corps 2-stage search min 0.837; DWW 3-stage 0.938 (0.959 with `stage3_effective_normal='gle'`) | Corps 0.931; DWW 1.039 (search minima) | ~−10% both, ordering DWW > Corps correct | **PASS (geometry-limited)** | New `search_rapid_drawdown` wrapper: drawdown strengths (2-stage/3-stage) evaluated per trial inside the existing circular/entry-exit/noncircular searches via an additive `fos_fn` hook (default None = byte-identical). Robustness gate rejects degenerate near-flat trials (non-converged / <min-slice / stage-1 FOS <0.5) that would win with spurious ~0.18 FOS. The ~10% shortfall is GEOMETRY-LIMITED: the #98 section is recovered from a raster figure with flat-stacked layers (pinch-outs simplified, riprap omitted). Wrapper exactness proven on the exact #95/#96 section (search min ≤ published specified-circle FOS; winning-surface stage detail reproduces the search FOS to machine precision). NOT tuned. Tests: validation_examples/test_published_v041.py (7). **GM-87 (2026-10-15):** the search now refines its lowest grid centre; the #98 minima move down by at most 0.3 % (Corps 0.8370 → 0.8341), inside the pins (slope_stability VALIDATION.md B13). **GM-131 (2026-10-17):** on the exact #95/#96 dam the Corps 2-stage search is governed by a shallow planar slide on the upper face, 0.783 against the infinite slope (gamma'/gamma) tan(phi')/tan(beta) = 0.772 (+1.5 %); before GM-131 no planar surface solved the full-pool stage and the circle (0.895; 0.902 now) was reported. Its noncircular search 0.962 → 0.972. The #98 minima did not move. |
| V-048 | slope_stability rapid drawdown — Lowe & Karafiath 2-stage (Slide2 #98 published column; #95/#96 cross-check) — v5.4 F4 | #98 search min **0.964**; #95/#96 exact circle 1.357 flat / **1.450** seepage | Lowe-Karafiath #98 search min **1.075**; (DWW #96 1.443) | #98 −10% (geometry, ordering correct); #95/#96 seepage +0.5% vs published DWW 1.443 | **PASS (mechanics/ordering) / CONVENTION (#98 geometry)** | New `method='lowe_karafiath'`: the SAME Kc-interpolated stage-2 undrained strength as Duncan-Wright-Wong but WITHOUT stage 3 (Lowe & Karafiath 1960; DWW 1990 = L&K + the stage-3 drained check), so `lowe_karafiath` FOS ≥ `duncan_3stage` always. On the exact #95/#96 dam the L&K **seepage** FOS (1.450) lands on the PUBLISHED DWW 1.443 — a cross-check that the Kc anisotropic-consolidation stage-2 is sound and the default 3-stage's residual is entirely the Fellenius stage-3 normal (V-038 E2), not the interpolation. On #98 the L&K search minimum (0.964) reproduces the published method ORDERING L&K (1.075) > DWW (1.039) > Corps (0.931) and inherits V-041's ~10% recovered-geometry deficit (Corps 0.837 / DWW 0.938 / L&K 0.964 — same offset). Additive; existing methods byte-identical. NOT tuned. Tests: validation_examples/test_published_v048_drawdown_methods.py (4). |
| V-042 | slope_stability probabilistic reliability index (Li-Lumb 1987 / Hassan-Wolff 1999 / Slide2 #36) — v5.4 F2 | det. Bishop critical FOS 1.332 (search, GM-57 windows; 1.336 on the pinned circle the RI uses, 1.325 before GM-122); FOSM COV_F 0.1228, beta_LN 2.308 (Pf 1.05%; 2.244 before GM-122); MC beta_LN 2.320 | Slide Bishop det 1.340 (RI_LN 2.482) / overall 1.350 (2.393); Hassan-Wolff 1.334 (2.336) | FOS −0.1% vs H-W (search; +0.2% on the pinned circle); RI −1.2% vs H-W | **PASS** | Labeled homogeneous slope (Fig 36.1): surface (0,5),(5,5),(15,15),(20,15), c'=18±3.6 / φ=30±3 / γ=18±0.9 / ru=0.2±0.02. Validates the FOSM + Monte Carlo + lognormal-RI path against the ORIGINAL Hassan-Wolff (1999) benchmark. The FOSM COV_F 0.1225 REPRODUCES Hassan-Wolff's implied FOS-COV: `lognormal_beta` at the published deterministic FOS (1.334) gives RI 2.29 — within 1.8% of 2.336 (module's own F=1.336 → 2.308, within 1.2%). FOSM≈MC (beta 2.308 vs 2.320, COV 0.1228 vs 0.1227). GM-122 (2026-10-16): the pinned circle passes 1.3 cm over the toe and dips 0.2 mm under the ground in front of it; taken from that touch to its exit, the slice at the toe fell in the air and was dropped (F 1.325, beta 2.244); now the face mass alone, every slice kept. Variance c'-dominated (~70%) then φ (~23%) — physically sensible for COV_c'=0.20. Needed the additive `ru` probabilistic variable (added alongside `gamma_sat`). Tests: test_published_v042_slope_probabilistic.py (3). |
| V-043 | slope_stability geosynthetic-reinforced embankment (Tandjiria 2002 / Slide2 #39) — v5.4 F2 | sand no-reinf: circ Spencer 1.186 on the critical circle the default search finds (1.190 on 480 slices; the pinned circle 1.210) / noncirc DE 1.234; sand reinf force (FS=1.35) 57 kN/m on the critical circle, 41 on the pinned circle; clay no-reinf no-crack 1.041 | sand no-reinf circ 1.209/1.219, noncirc 1.188/1.192; reinf circ 44-45, noncirc 56 kN/m; clay 0.975 (water crack) | sand circ −1.6 % (critical circle, 480 slices) / noncirc DE +3.8 %; reinf 57 vs noncirc 56 (the same slide), pinned circle 41 vs circ 44-45 | **PASS (sand no-reinf)** / **CONVENTION (reinf force + clay crack)** | Labeled 2-material embankment (Fig 39.1/39.2): crest el 9 x0-10 → (20,3), 3 m soft-clay foundation el 0-3. SAND fill (c'=0/φ=37/γ=17 over soft clay c'=20/φ=0/γ=20), no reinforcement: circular Spencer 1.180, noncircular ~1.19 — within ~2.5% (dry crack barely intersects). Geosynthetic force (active, el 3) to reach FS=1.35 = ~55 kN/m circ / ~50 kN/m noncirc, bracketing published 44-56 but with the circ/noncirc split inverted — reinforcement force is highly critical-surface-sensitive and the search finds a slightly different surface (the manual attributes such gaps to search differences). CLAY fill (both layers c=20/φ=0/γ=19.4, undrained) needs a WATER-FILLED crest crack whose depth is NOT labeled; without it 1.039 (+6.6% vs 0.975), and a full 2c/γ=2.06 m water column overshoots to ~0.27 — not reproduced, NOT tuned to a fitted depth. Tests: test_published_v043_slope_reinforced.py (4). **GM-131 (2026-10-17):** sliced at its corners, differential evolution reaches 1.234 from seeds 1-3 (+3.8 % on Slide2 1.188), the same on 31 to 120 slices; before, DE read 1.09-1.10 and the random draws 1.15-1.27 on surfaces diving into the soft clay that read 1.24-1.32 with one slice more. The random search (2000 trials) now stays above DE (1.34 / 1.46), and the infinite slope on the sand face (1.258) governs those runs. The noncircular comparison is a KNOWN GAP (GM-117; a strict xfail): DE's 1.234 sits above our own converged critical circle (1.213 on 60 slices or more) and a 41-point polyline inscribed in it (1.2145). **GM-134 (2026-10-17):** the circle is cut where it crosses the fill's base and each slice takes its chord: circular Spencer 1.210 (+0.05 % on Slide2 1.209; 1.208-1.211 on 25-45 slices, 1.213 on 3,840 either way); the 1.180 was a 30-slice value (1.246 / 1.190 / 1.180 / 1.232 on 25 / 29 / 30 / 31). The force to raise that circle to 1.35 is 40.9 kN/m, 7 % below the published circular 44-45. Clay no-crack 1.041, on the circle with its radius cut 0.1 mm to touch the base (it came 3.5 mm under it, refused at any count now). **GM-138 (2026-10-18):** that circle (from an old entry/exit search) is not the sand fill's critical one. An independent brute force finds a circle tangent to the soft clay's base, centred over the face's mid-length, at Spencer 1.186 on 30 slices and 1.190 on 480: −1.6 % on Slide2's circular 1.209 (Tandjiria 1.219), because Slide2's circular search stopped short of a circle tangent to the model's base. The published values pair up by surface: Slide2's circular 1.209 and 44-45 kN/m match the pinned circle (1.2096 on 30 slices, 41 kN/m); its noncircular 1.188 and 56 kN/m match the critical circle (1.190, 57 kN/m). A dry crack cannot explain the lower value: truncated by a dry crack 1, 2 or 3 m deep the critical circle reads 1.189 / 1.198 / 1.221, higher. The default Spencer search reported 1.238 (its refinement walk moved on Bishop's FOS and followed shallow circles on the sand face) and now finds 1.186 (M-P / GLE 1.246 → 1.196); pinned within 0.01. On that circle the force to reach 1.35 is 57 kN/m (55 on 240 slices): it matches the published noncircular 56 kN/m (the same slide); the published 44-45 belongs to Slide2's own circle, close to the pinned one; the pinned test keeps the pinned circle's 41. |
| V-106 | slope_stability Ito-Matsui stabilizing pile (Cai & Ugai 2000 / Slide2 #106) | — | Bishop FS: no-pile 1.13/1.14; S/D=2 1.54; 3 1.37/1.43; 4 1.31/1.33; 6 1.25 | — | **N/A (source)** | The Slide2 Fig 106.1 is an unlabeled RASTER image (no coordinates, no soil/pile properties), and the geometry + Ito-Matsui pile parameters (D1/D2/position/limit-strength) live only in the external Cai & Ugai (2000) paper, not the manual — cannot recover honestly from the source without reconstructing/fitting parameters. The module DOES implement the Ito-Matsui 1975 force (unit-tested vs the paper's hand values) and the spacing trend + a with-pile FOS are validated at V-040 (Slide2 #54), so this is a source-coverage gap, not a capability gap. Recorded, not pinned. |
| V-044 | slope_stability submerged slope + ponded water (Duncan & Wright 2005 Fig 6.27 / Slide2 #70) — v5.4 F2 | circular Bishop 1.597 / Spencer 1.598 / GLE 1.595; FOS water-level-independent (el 75/105/45 → 1.5974 identical) | Bishop 1.603 / Spencer 1.599 / GLE 1.599; ref 1.60; Case 1 = Case 2 | −0.4% Bishop; independence reproduced | **PASS** | Labeled homogeneous slope (Fig 70.1): surface (0,15),(30,15),(105,45),(140,45), γ=128 pcf/c'=100 psf/φ'=20°, fully submerged with ponded water. On the Bishop centre-grid critical circle the three methods land within 0.4% of published. The headline physics — FS INDEPENDENT of ponded-water depth for a submerged slope — is reproduced to ~7 sig figs: the same circle gives Bishop 1.5974 at water el 75 (Case 1), 105 (Case 2) AND 45 ft (at crest), validating the ponded buttress + submerged-weight handling. NOTE a free Spencer/entry-exit search converges to a spurious near-zero degenerate surface on this ponded geometry; the robust path is the Bishop centre-grid search (matches Slide's auto-refine). Tests: test_published_v044_slope_submerged.py (2). |
| V-045 | slope_stability correlated c'-φ' probabilistic (Wolff & Harr 1987 Cannon Dam / Slide2 #34) — v5.4.1 train item 1 | feature consumes #34's published rho (Table 34.1: Phase I +0.11, Phase II −0.51): FOSM COV_F 0.396 (indep) → 0.413 (corr), MC 0.381; corr(...) variance entries surfaced | Slide GLE 2.333 / Spencer 2.383 / Wolff-Harr 2.36; Pf 3.55e-3 (Slide) | full FOS/Pf N/A(geometry) | **PASS (correlated-pair capability) / N/A-scope (full FOS+Pf)** | New correlated scalar-pair reliability capability (generalizes the F1 su-law (a,b) correlation to arbitrary scalar pairs — FOSM cross-term 2·rho·(dF1/2)(dF2/2) + bivariate-normal MC, adapter `correlations`). Validated against #34's OWN published correlation coefficients on a representative 2-material embankment: the feature accepts the (c',φ') rho pairs, propagates them (COV_F shifts 0.396→0.413; corr(k1,k2) shares total 100%), and FOSM≈MC in the high-COV band (~8% gap = FOSM Taylor limit at #34's extreme φ' COV≈1.2, not a defect — machinery matches MC to <1% at moderate COV). The FULL #34 slope FOS 2.36 / Pf is N/A-scope: Fig 34.2 is a rendered multi-material section with NO labeled coordinates, a figure-only non-circular surface, omitted layers, and Slide-tuned γ (manual's own note) — geometry does not survive honest extraction. Tests: test_published_v045_slope_correlated.py (2); machinery unit-tested in test_probabilistic.py + the slope adapter. **GM-134 (2026-10-17):** FOSM mean 3.482 (3.445 before; 3.484 on 3,840 slices either way), COV_F 0.387 / 0.404 (0.396 / 0.413). |
| V-046 | slope_stability probabilistic dyke (El-Ramly 2003 Syncrude / Slide2 #33) — v5.4.1 | — | Slide FOS 1.305, Pf 1.54e-3; El-Ramly 1.31, Pf 1.6e-3 | — | **N/A (geometry)** | Multi-material tailings dyke, Bishop + Monte Carlo Pf. Table 33.1 gives c'=0 for all layers with only two φ variables (Sandy till std 2, Kca clay-shale std 2.1) — INDEPENDENT (no c'-φ' correlation, so it doesn't exercise the new feature). Fig 33.2 is a rendered 4-material section with phreatic surfaces and NO labeled coordinates — geometry does not survive honest extraction, so the Pf is not reproduced. Recorded, not pinned. |
| V-047 | slope_stability Cannon Dam reliability indices (Hassan-Wolff 1999 / Slide2 #35) — v5.4.1 | — | Table 35.2: 9 surfaces, RI_LN 4–11 | — | **SKIPPED (source)** | The manual states outright: "the paper does not provide all the required input parameters, we selected values for the missing parameters that allowed us to match factors of safety." Slide tuned missing inputs to hit the target, and the published RIs are physically scattered (4–11 across surfaces). Not a clean independent anchor — skipped per the manual's own admission. |
| V-049/050 | slope_stability grouted tiebacks (Duncan & Wright 2005 Fig 6.34/7.28 / Slide2 #85/#86) — v5.4 F2/F2c | guard removes the spurious below-base surfaces: #85 no-support 0.72→0.912; #86 no-support 0.55→1.549 (both now base-admissible) | #85 Bishop 1.531 active / 1.324 passive (ref 1.51/1.32); #86 Bishop 1.629 (ref 1.61) | with-support not reproduced | **CONVENTION (guard-improved; with-support blocked)** | Both geometries recovered (labeled): #85 saturated clay c=350 psf/φ=0, single 9,000 lb/ft support at mid-height on a steep 0.5:1 20-ft face; #86 c'=0/φ=37 fill on rigid rock, five 800 lb/ft tiebacks. **The F2c search-admissibility guard (SS-5 generalized) FIXED the spurious degenerate surfaces** these exposed — a circle plunging below the rigid base and re-emerging near the exit (one-sided fragment) is now rejected, so the no-support baselines are physical (#85 0.912, #86 1.549 vs the old 0.72 / 0.55). The published WITH-support FOS is still NOT cleanly reproduced, for reasons BEYOND the search guard: (#85) the single mid-height support is partially avoided by the critical surface — the anchor force IS computed and engages surfaces crossing it (a 10× anchor saturates the FOS) but the search minimum is a support-avoiding circle, so matching Slide's 1.531 needs its support-application convention; (#86) the c'=0 circular search settles on a deep circle (1.549) rather than the true planar infinite-slope critical (0.94), which the tiebacks don't engage (Bishop unchanged with/without them). Not tuned/pinned. The below-base guard is validated at slope_stability/tests/test_search_robustness.py; the Anchor mechanic at V-043. **GM-14 (2026-10-14):** without supports the search now reports the infinite slope, 0.942 = tan 37°/(24/30), with `governing_surface='infinite_slope'` (was 1.566 on the default grid); pinned in slope_stability/tests/test_search_supplements.py. **GM-53 (2026-10-14):** searches now score reinforcement (before, every search returned the unreinforced critical). #86 WITH tiebacks, global-failure windows: Bishop 1.612 / Spencer 1.606 / GLE 1.606 vs Slide2 1.629 / 1.620 / 1.622, D&W 1.61 — **reproduced** (pinned, test_search_reinforcement.py). #85 active: GLE 1.581 vs 1.575 — reproduced; Bishop 1.304 vs 1.531 — convention (small toe circle near the support; the module's Bishop counts the support only by its moment arm about the centre). **GM-59/GM-60 (2026-10-14):** supports now act at the base of the slice they cross in every equation the method satisfies (Wright & Duncan 1991; Bishop: moment + vertical component), active or passive in every method. The Bishop 1.304 is a reverse-curvature toe circle (5.9 ft dry crack, as Slide2's default closes it, but its grid never generates it), now flagged; with `reverse_curvature='discard'` #85 Bishop is 1.503 active / 1.305 passive vs Slide2 1.531 / 1.324, D&W 1.51 / 1.32 — **reproduced**. Spencer/GLE converge on Slide2's printed circle (1.564 = Bishop, phi = 0; Slide2 GLE 1.575; before: fallback to the unreinforced 0.913); searches active Spencer 1.581 / GLE 1.609, passive 1.346 / 1.379 (Slide2 GLE passive 1.378). **GM-134 (2026-10-17):** circles cut at the crack's base and sliced by their chords: #85 Bishop search 1.372 active (1.304 before; the old slicing reads 1.368 on 3,840 slices, the tangent's error at the circle's vertical end), `'discard'` 1.508 active / 1.316 passive against D&W 1.51 / 1.32; Spencer / GLE searches 1.551 / 1.573. |
| V-052 | slope_stability rapid drawdown — Pilarcitos Dam method ordering (Duncan-Wright-Wong 1990 / GEO-SLOPE Ex3 / Slide2 #97) — v5.4 F4 | search minima on a representative 2.5:1 face: Corps ~0.79 / DWW ~1.35 / LK ~1.35 (ordering Corps<DWW<LK, DWW~LK) | Corps 0.823 / LK 1.047 / DWW 1.043 | Corps −4%; LK/DWW overshoot | **CONVENTION (ordering validated; magnitudes geometry/φ'-limited)** | Third published case history (homogeneous rolled earth-fill, γ=135 pcf, c'=0/φ'=45, R-envelope c_R=60/φ_R=23; drawdown El 692→657). The upstream SLOPE is NOT published (assumed 2.5:1); the asserted ORDERING Corps<DWW<LK is slope-invariant (holds 2:1–3:1), so it is not geometry-tuned. Corps 2-stage (R-governed) ~0.79 lands near published 0.823 and <1.0, reproducing the 1969 upstream failure. Lowe-Karafiath/DWW ~1.35 OVERSHOOT published ~1.05: assumed slope + the steep published φ'=45 (Kf=5.83) inflate the Kc-interpolation gain vs Slide2's narrow LK-vs-Corps spread (the moderate-φ' #98 dam matched to ~10%, V-048). Not tuned. Tests: test_published_v052_drawdown_pilarcitos.py (2). |
| V-053 | slope_stability rapid drawdown — Pumped-Storage Project Dam (Duncan-Wright-Wong 1990 / GEO-SLOPE Ex2 / Slide2 #99) — v5.4 F4 | — | Corps 1.345 / LK 1.620 / DWW 1.534 | — | **N/A (geometry)** | ~300 ft ZONED core/shell/bench dam; all soil properties + elevations recovered (rockfill γ=142/φ'=37 FREE-DRAINING; silty-clay core+random-zone γ=140/φ'=36, R c_R=2000/φ_R=18; drawdown El 545→380) but the zone BOUNDARIES (core width, rockfill-shell slopes, bench) are figure-only — the free-draining-shell / undrained-core mechanism depends on where the core sits relative to the slip surface, so the section cannot be reconstructed honestly from prose. Data recorded (INVENTORY) for a future figure-digitization pass. Not pinned. |
| V-054 | slope_stability anisotropic undrained strength su(α) (Casagrande-Carrillo 1944; isotropic=Frontiers 13:1581457; Bakklandet=IOP EES 1523(2025)012038) — v5.4 F6 | isotropic H=5/45°/su=40 critical Bishop 2.287; anisotropic all-equal = mohr EXACTLY; Bakklandet ratios drop FOS to 1.796 (21.5% crit-vs-crit) | isotropic 2.16 (FLAC 2.18); Bakklandet 24% drop (1.43→1.09 FEM) | isotropic +5%; aniso drop 21.5% vs 24% | **PASS (isotropic + exact reduction) / CONVENTION (Bakklandet magnitude)** | New `strength_model='anisotropic'`: su varies with the slice-base inclination α (ADP active/DSS/passive, φ=0) via the Casagrande-Carrillo `su_H+(su_V−su_H)sin²(i)` with the failure plane at 45° to σ1 (i=α+45°) → clean sin(2α) on the base angle; su_active (α≥+45°), su_dss (α=0), su_passive (α≤−45°); held constant beyond ±45°. α SIGN CONVENTION documented (α>0 = active/crest for the standard toe-at-low-x slope; failure-PLANE angle, ≠ σ1 inclination by ~45°). Equal su's reduce to mohr_coulomb cu EXACTLY (isotropic identity, all methods). Isotropic critical toe circle Bishop 2.287 within ~5% of published 2.16-2.2 (the auto centre-grid search alone over-estimates to ~2.78 by missing the small toe circle — a search-coverage limit, not the model; since GM-13 the toe-circle family finds 2.286). Most of the remaining gap is the model starting at the toe: with level ground beyond it the search gives 2.237 (GM-63's profile-end check; GM-88). Bakklandet DSS/A=0.63, P/A=0.35 → 21.5% FOS drop, bracketing the published FEM 24% (CONVENTION: FEM vs LEM, depth-varying su vs constant). Slide2 #105 geometry is in Tutorial 32 not the manual → N/A(source). Additive; mohr/shansep/hoek_brown byte-identical. Not tuned. Tests: test_published_v054_anisotropic_su.py (4) + test_strength_models.py::TestAnisotropicSu (5). **GM-134 (2026-10-17):** the toe circle (2.5, 4, 4.72) reads 2.379; its 2.287 was a slicing error at the arc's vertical end (the old slicing reads 2.374 on 3,840 slices and 2.378 on 10,240). The search's toe circle is 2.346; with level ground 40 m past the toe a base circle gives 2.222 (+2.9 % on 2.16, +1.9 % on FLAC 2.18), now pinned. Bakklandet ratios: 1.942 on the toe circle, 1.886 on the anisotropic critical (a 20.7 % drop against the published 24 %). |
| V-026 | slope_stability tension crack + water (ACADS 1(b) / Slide2 #2) | dry crack: Bishop 1.553 / Spencer 1.555 / GLE 1.551; water crack: Bishop 1.497 / Spencer 1.498 / GLE 1.494; no-crack base 1.686 | water crack: Bishop 1.596, Spencer 1.592, GLE 1.592, Janbu-c 1.489; referee 1.65 | dry −2.7%; water −6.2% (Bishop) | **CONVENTION** | ACADS #1 geometry (validated in B3) MIRRORED so the module's entry-side tension crack lands at the crest; soil c'=32/φ'=10/γ=20. The tension-crack strength-truncation mechanic is validated — the module's **dry** crack (1.553) matches Slide2's **water** crack (1.596) within ~3%. The module's **water-filled** crack (1.497 pinned; ~1.465 free search) is ~6-8% below Slide2 and ~9-11% below the referee 1.65: it is MORE CONSERVATIVE, keeping the cracked wedge as zero-strength driving soil AND applying the full hydrostatic thrust F_w=0.5·γ_w·zc²≈71 kN/m, whereas Slide2 truncates the sliding mass at the crack. All four Slide2 methods (1.489-1.596) and the module bracket the referee 1.65 from below. NOT tuned. Two B2 candidates surfaced: (1) exit-side tension crack (currently entry-only → forces the mirror); (2) mass-truncation crack model. **v5.4 (E4): BOTH BUILT and the residual RESOLVED** — `tension_crack_side='exit'` (un-mirrored slope reproduces the mirrored FOS to ~1e-16, mirror workaround obsolete) and `tension_crack_model='truncation'` (removes the cracked wedge from the sliding mass, as Slide2 does): water-crack Bishop 1.597 / Spencer 1.593 / GLE 1.594 vs published 1.596 / 1.592 / 1.592 — <0.1%, not tuned. Defaults ('entry','strength') byte-identical; the strength model remains the documented conservative convention. Tests: test_published_v026_e4.py (5). **GM-134 (2026-10-17):** the crack's edge is a slice boundary: water crack, strength model 1.485 / 1.486 / 1.482 on 5 crack slices (1.497 / 1.498 / 1.494 on 4 before); truncation 1.603 / 1.599 / 1.600, within 0.5 % of Slide2; the old slicing reads the same on 3,840 slices, so the <0.1 % was in part a 60-slice value. |

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

- **Datum (revised by GM-121, 2026-10-18).** The GEC-12 tables reference depths to
  the footing/cap bottom (5 ft bgs) and the pile head sits there. Table D-6's σ′
  keeps the 5 ft of soil above the footing (its toe at 0.01 ft below the footing is
  16.4 kips, which needs σ′ > 0 there), and Fig. 7-17's D runs from the ground
  surface (GEC-12 p. 246). So both entries are modeled from the ground surface with
  `AxialPileAnalysis(head_depth=5 ft)`: the shaft starts at the head, the overburden
  above it still counts. (Before GM-121 the profiles started at the footing with
  that overburden removed; with GEC-12's charts that model falls 17–27 % short of
  Table D-6's shaft.)
- **axial_pile separate shaft/toe φ per layer — feature ADDED (v5.2).** The example
  uses a Layer-3 design-limit toe φ=40 with a shaft φ=36. The optional
  `AxialSoilLayer.toe_friction_angle` carries a distinct end-bearing φ for a layer
  (GEC-12 allows different shaft/toe φ in dense gravels). Adapter param:
  `layers[].toe_friction_angle` (alias `toe_phi`).
- **GEC-12 Step 8's limit on σ′p (GM-121).** The module limits σ′p at the toe to
  3 ksf by default, as GEC-12 Step 8 (p. 244) gives it; the example did not, so its
  428.1-kip plateau is reached only with `toe_stress_limit=None` (371.5 kips with
  the limit, −13 %; 359.7 before GM-282 read N′q again off Fig. 7-16).
- **Fig. 7-17 choices (V-002, GM-121).** The module reads Ca off GEC-12 Fig. 7-17
  (measured off the printed figure). The example's DrivenPiles run used the smooth
  steel curves for the H-pile and held Ca at the chart's end above su ≈ 3.05 ksf;
  GEC-12's text gives the corrugated curve for H-piles (p. 247), and p. 247
  recommends API α for highly overconsolidated clays past the chart. The module
  takes the corrugated curve. Past the chart, since GM-229 (the owner's OWN-63
  rulings), it takes API α in a layer whose su/σ′vo is at or above the engineer's
  `highly_overconsolidated_su_ratio` and holds Ca at the chart's end below it;
  GEC-12 gives no such line, so a run without one is refused. With a line of 1.0
  Layer 3 holds Ca at the chart's end, like the example (shaft +13 % to +14 %);
  API α in every layer past the chart is the option `adhesion_beyond_chart="api"`
  (+25 % to +53 %: Ca jumps at the chart's end). It reproduces the example within
  1 % with `adhesion_curve="smooth_steel"`, `adhesion_beyond_chart="chart_end"`.
  Within the chart Ca is read off Fig. 7-17 as drawn (alpha method, Steps 1-2,
  p. 250), so α may exceed 1 (about 1.1 at su 1.0 ksf on the D = 40b solid
  curve); API's α ≤ 1.0 applies only where API α is taken. No V-002 layer reads
  α above 1.
- **wave_equation diesel hammers are simplified (V-003).** The DB carries only
  −22/−32 single-energy Delmag diesels and the `Hammer` diesel path is an
  energy→velocity conversion, not a diesel combustion + ram-cycle model. It
  cannot reproduce a GRLWEAP diesel bearing graph (blow count ~60% low with a
  hand-built D36-52; the stress, ~20% high until GM-294's review took the
  pile-top force without the helmet's inertia, is now 3% low). If diesel drivability tie-outs are wanted,
  this needs a real diesel hammer model + the GRLWEAP-default helmet/cushion data
  — a substantial build, flagged as a known limitation.
- **downdrag NP construction is solid (V-004).** Given the published shaft, the
  intersection of the load and resistance curves reproduces NP/Qmax/DF to
  0.2% with the pile weight left out as the example does, the module's
  default since 2026-10-15 (<2% with `include_pile_weight=True`;
  GM-01, 2026-10-14, made the search and the reported loads one integral). The only
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
- **retaining_walls MSE now has BOTH an ASD-FoS path AND a packaged GEC-11 LRFD
  external-stability path (v5.3 — gap CLOSED).** `analyze_mse_wall` still returns
  ASD factors of safety by default (unchanged). The NEW
  `check_external_stability_lrfd` (also reachable via
  `analyze_mse_wall(lrfd_external=True)` → `result.external_lrfd`, and the funhouse
  `mse_lrfd_external_stability` adapter method) does the full AASHTO/GEC-11
  bookkeeping in the module: Strength I max/min + Service I load-factor
  combinations, the LL-on-resisting-side exclusion (sliding/eccentricity) vs
  LL-included (bearing), the min-vertical "critical" combination, and per-mode
  CDRs. It reproduces every V-009 published CDR (sliding 1.85/2.08/1.37, eL
  2.87/3.87 ft < L/4, bearing σv 6.70 ksf / CDR 1.57, Service 4.66 ksf) through
  the high-level path — no hand-assembled factors. Load/resistance factors and the
  eccentricity ratio are documented parameters with GEC-11 defaults. The ASD path
  is byte-identical by default.
- **MSE internal stress/pullout primitives are solid; the bar-mat curves are now
  built in (v5.2 Q4 — gap CLOSED).** `Tmax_at_level` and `pullout_resistance(C=2)`
  reproduce the Table E4-7.4 bar-mat Tmax/σH/Pr to ≤3%. The module's built-in
  `Kr_Ka_ratio` and `F_star_metallic` now carry BOTH the **ribbed-metallic-STRIP**
  curves (Kr/Ka 1.7→1.2; F* 2.0→tanφ over 0–20 ft, 0–6.0 m until 2026-10-16) AND the **steel-bar-mat / welded-grid**
  curves (Kr/Ka 2.5→1.2; F* 20(t/St)→10(t/St) over 0–20 ft), selected by
  `reinforcement_type` ("bar_mat"/"welded_grid"/"metallic_grid"). The grid F*
  needs t (transverse-bar diameter) and St (transverse spacing) — taken from the
  `Reinforcement` geometry (`thickness`/`transverse_spacing`, e.g.
  `WELDED_WIRE_GRID_W11`) or supplied to `F_star_metallic(t_over_St=…)`.
  `check_internal_stability` auto-selects them for a `metallic_grid` reinforcement,
  so the high-level internal path now matches a bar-mat wall directly — no
  hand-fed coefficients. The ribbed-STRIP default path is byte-identical/unchanged.
  Depth datum: the bar-mat Kr/Ka and F* tapers cap at Z = 20 ft (= 6.096 m);
  since the GM-112 review (2026-10-16) the strip tapers do too (GEC-11 Fig.
  4-10, Eq. 3-5), where they capped at 6.0 m before.
- **Unit note for MSE per-length quantities.** Forces convert k/ft ↔ kN/m by
  14.594; moments-per-length convert k-ft/ft ↔ kN-m/m by 1.356/0.3048 = 4.4488
  (a kip-ft of moment per ft of wall). The tests carry both constants explicitly.
- **V-056 / V-057 (GM-112, 2026-10-16): the sloping backfill is built and both
  GEC-11 sloping examples reproduce through the high-level path.** Until then
  the MSE checks computed a level-backfill Rankine Ka only. E3 (a 2H:1V slope
  that continues, steel strips) and E1 (a broken backslope with traffic behind
  the crest, geogrid) are pinned in
  `test_published_v056_v057_gec11_slope.py`; the method and page references
  are in `retaining_walls/DESIGN.md` "MSE walls with a sloping backfill". Level
  backfill results are byte-identical (V-009..V-010 unchanged).
- **GM-256 (2026-10-20): the coverage ratio.** A steel strip or grid now
  needs its horizontal spacing (or Rc) and is refused without it. E3's strips
  are given their own width and spacing, and the module reproduces Table
  E3-7.3's pullout and tension at all twelve levels (row above). V-009/V-010's
  bar mat is given E4's (p. E4-18: 4 wires at 0.5 ft, b 1.5 ft, one mat a 5 ft
  panel); the external values and the Kr/Ka and F* pinned there do not depend
  on it and did not move.
- **GM-256, second commit: alpha.** The built-in geogrids take Table 3-6's
  alpha 0.8 (p. 3-17). E1's pullout CDRs now reproduce (row above). V-057's
  external and S_eq values do not depend on alpha and did not move.

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
- **V-013 — log-spiral passive + FHWA apparent-pressure now BUILT IN (v5.3 A3 —
  gaps CLOSED); single-anchor embedment is the one residual.** The Caquot-Kerisel
  log-spiral passive is now `caquot_kerisel_Kp` (sheet_pile + soe): Kp=4.70 exact,
  selectable as `pressure_method="log_spiral"`. The FHWA apparent-diagram anchor
  quantities are now native via `soe.fhwa_apparent_pressure_anchored_wall`
  (σ_a=pe=934.4 psf, PT=15,573, T1U=6,229 — all high-level). **Residual CLOSED
  2026-09-15:** `soe` now solves the apparent-diagram free earth support, so D=6.09 ft,
  T1=14,254 lb/ft and M=22,494 ft-lb/ft reproduce to ≤0.2 % (row in the table). Before that,
  the single-anchor embedment D=6.09 ft was NOT fully reproduced —
  `pressure_method="log_spiral"` brings the classical (triangular-active) FES
  embedment from ~10.8 ft (Rankine) to ~7.2 ft, but the published 6.09 ft
  additionally uses the FHWA 1.3× APPARENT-diagram active distribution in the
  free-earth balance, which needs an apparent-diagram FES solver (tracked in
  V5.3_PLAN). The single-anchor TOTAL anchor force TH/T likewise comes from that
  solve. Module not tuned.
- **V-014 — Caltrans force-balance basal heave now BUILT IN (v5.3 A3 — gap
  CLOSED).** `soe.check_basal_heave_caltrans` reproduces the Caltrans force
  balance NATIVELY, INCLUDING the sidewall-shear term S=cu·H and the 0.7B block:
  F_RS=cu·Nc·(0.7B) vs F_dr=0.7B·H·γ+0.7B·q−cu·H → FS=1.55 (Nc=7.68 from the
  Skempton/Bjerrum-Eide form; exactly 1.538 with the chart Nc=7.6 override). All
  block terms (W=37.8, q=3.15, S=15.0, F_dr=25.95 k/ft) match. The existing
  `check_basal_heave_bjerrum_eide` (inverted-footing bearing ratio, FOS=0.86, no
  side shear, Nc=6.71) is UNCHANGED — the two are distinct, defensibly-different
  formulations now both available. Not tuned.
- **V-016 — GEC-4 two-tier anchored wall is largely PASS via primitives.**
  `soe.rankine_Ka`(33)=0.295, the FHWA trapezoid pe=43.6, surcharge ps=3.2, both
  tributary anchor loads (TH1=168, TH2=172), the subgrade reaction R=37, the lower
  hinge moment M2,3=66, and both anchor design loads (DL1=435, DL2=445) all
  reproduce to ≤2 %. The upper anchor's moment M1 was the soft spot: the
  compact form (13/54)·H1²·(pe+ps)=70.4 vs the published 76 kN-m/m. GEC-4 prints
  M1 = 13/54 H1² pe + ps H1²/2 = 75.6 (p. A-6), and the module takes it since
  GM-290. No module change — these are all hand-built on the
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
- **GM-287 (2026-10-31) update:** the sand's C1, C2, C3 now come from GEC 9
  Eqs. A-22 to A-24 and the four-segment curve is the default, so V-017 reads
  3.57 mm (casing-only EI, +8%) and 3.32 mm (composite EI, +0.6%) against 3.3,
  Mmax −36.8 kN·m against −37.3. The notes below are the earlier history.
- **V-017 lateral_pile: the FD solver is verified EXACT; the deflection +19% is
  the section BENDING STIFFNESS (composite/nonlinear EI), NOT the p-y curve —
  root-caused in v5.3 A4.**
  - With a linear soil law p=k·z·y the beam-column solver reproduces the Reese-
    Matlock characteristic-length (T-method) closed form to 4 figures for the
    fixed-head groundline deflection (0.93·V·T³/EI) and head moment (≈−0.93·V·T).
    That pins the solver, the fixed-head BC (slope=0), and the discretization as
    correct.
  - **A4 root cause (v5.3):** the published LPILE run used **nonlinear/composite
    EI** — the micropile section is a steel casing + grout + bar (extract
    `mp_sp2.txt`: "Computed Pile Response Using Nonlinear EI") — while the test uses
    the casing-only elastic EI (I=3.58667e-5, per the LPILE echo's linear section).
    A modest, physically-expected composite stiffening (**EI×1.3**) lands the
    simplified curve on **exactly 3.28 mm ≈ 3.3**. So the gap is section stiffness,
    not the p-y construction.
  - **The full Reese (1974) four-segment construction is now available**
    (`SandReese(construction="reese1974")`, v5.3 A4) but for V-017 gives **5.02 mm
    — SOFTER** than the simplified 3.92 mm and further from 3.3, DISPROVING the old
    "simplified is too soft a p-y" hypothesis. This is faithful to Reese: the
    working deflection sits at the m-point ym=b/60=3.28 mm where Reese fixes the
    resistance at pm=B·pu (B→0.5 at depth), genuinely softer than the simplified
    1/3-power parabola. The A/B asymptotes (A_s=0.88/B_s=0.50 etc.) were NOT tuned
    to force 3.3 mm. **fixed-head Mmax = −39.3 kN·m vs −37.3 (+5.4%, PASS)**;
    **deflection 3.92 mm (simplified) / 5.02 mm (full) vs 3.3 — CONVENTION.** The
    documented k (24430.244/16286.830 kN/m³) is used verbatim from the LPILE echo,
    so not a k-selection issue either. Reproducing 3.3 mm honestly needs a
    composite/nonlinear-EI capability (flagged as an A4 follow-up in V5.3_PLAN).
  - **RESOLVED (v5.4 E5): the composite-EI capability is built and V-017
    deflection is now a PASS.** `lateral_pile.composite_section_ei(...)` computes
    the uncracked transformed-section EI for a concrete-filled pipe / cased grout
    (+ optional bar ring) / reinforced concrete. For the SP-2 grout-filled casing
    (OD=0.1969, wall=0.0151, grout f'c=27.6 MPa, casing E=199,948 MPa) the
    composite EI = **8109 kN·m²** (steel annulus 88.5% + grout core 11.5%), ×1.13
    the casing-only 7171. Feeding it through the SAME p-y model / loads /
    overburden offset gives a fixed-head head deflection of **3.60 mm vs the
    published 3.3 mm (+9.2%, PASS — within the ±10% band the Mmax test uses)**.
    NOT tuned: the EI is pure section geometry + the ACI 318 Ec correlation + the
    published steel modulus. This confirms the A4 diagnosis (section stiffness,
    not the p-y curve) and closes ~half the original +19% gap. The uncracked
    composite is an *upper-bound* stiffness; the residual +9% is the cracked /
    moment-curvature (M-φ) EI LPILE actually used ("Nonlinear EI"), which is
    softer and is out of scope here (owner decision pending). Test:
    `test_v017_composite_ei_upgrades_deflection_to_pass`.
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
- **V-019 — the settlement ratio off GEC-13's printed Priebe curve (PASS on the
  chart read).** Since GM-276 (owner, OWN-70, 2026-10-21: "just the printed curve.
  discard your formula. don't trust formulas from your memory!") the module's only
  Priebe basis is the curve GEC-13 prints, Fig. 5-21 curve (4), φ = 37.5° (p. 5-36),
  read off the page (`settlement_ratio.PRIEBE_CURVE_4`, ±0.03 in the settlement ratio). The Priebe
  (1995) formula and its 42.5° default, printed nowhere in GEC-13, were removed; the
  tool refuses `phi_column` and `nu_soil`. Earlier rows of this file (the formula
  at 42.5°: n0 = 3.06, +13 %) no longer apply.
  - **The example's own area ratio.** A/A_sc = (s/d)² = (5.7/3.0)² = 3.61 (p. 5-48),
    as = 0.277: the curve reads **2.55** on both printed copies (Fig. 5-21 p. 5-36,
    Fig. 5-27 p. 5-49), so S = 27/2.55 = **10.6 in**.
  - **What GEC-13 prints.** "The settlement ratio is 2.7 … 10 inches" (p. 5-49).
    Fig. 5-27's arrows are drawn at A/A_sc 3.53, where the curve reads 2.63, and
    labelled 3.6 and 2.7: the printed read is 0.15 (6 %) above the curve at 3.61,
    within the inventory's 2.7 ± 0.3.
  - **Area-ratio convention.** The module's geometric `area_replacement_ratio` for a
    triangular pattern (√3/2·s²) gives as = 0.251 (A/A_c 3.98) and a settlement ratio of 2.31; the
    example takes (s/d)², so feed as = 1/(s/d)² to follow it.
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
  area-ratio + φ_col + chart latitude, since GM-276 the printed curve only; PVD
  radial-only vs combined). `ground_improvement/`
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

- **V-023 (1-D Terzaghi/Biot consolidation) is now a full PASS (v5.3 A5,
  commit d1e27a7): final drained settlement AND the undrained p0 + consolidation
  decay all reproduced, via the NEW monolithic u-p (Taylor-Hood) Biot scheme.**
  - **PASS (drained end-state):** the analytical final drained settlement
    `w = pz·H/(K+4G/3) = 2.609 mm` is reproduced to **ratio 1.0000** by the
    elastic confined-column solve and the coupled `solve_consolidation` end-state
    (both staggered and monolithic; monolithic transient converges to 2.596 mm,
    −0.5%). The analytical storage `S = 1/M + α²/(K+4G/3) = 1.554e-9`, coefficient
    of consolidation `c = k_mob/S = 0.0643 m²/s`, and the Biot undrained `p0` are
    verified inline.
  - **PASS (undrained p0 + decay) — the monolithic scheme.** The original
    staggered split could not create a load-induced undrained pore pressure (it
    solved the fully drained equilibrium at every step → excess pp ≈ 0). v5.3 A5
    adds an additive, opt-in **monolithic (coupled u-p) Biot solve**,
    `solve_consolidation(scheme="monolithic", theta=…)` (default stays
    `"staggered"`, byte-identical). The t=0 block solve
    `[[K, −Q],[Qᵀ, S+θΔtH]]` splits the applied total-stress increment into
    effective stress + excess pore pressure in one solve: interior/base
    **p0 = 83.92 kPa**, EXACT vs the formula-consistent `0.839e5 Pa`. The field
    then dissipates as the Terzaghi/Biot transient, matching the analytical
    series at the base to **<1%** (θ=0.5, Crank–Nicolson, fine step schedule) /
    **≤3.1%** (θ=1, backward Euler) across t̂ = 0.05–1.0 — inside the FLAC <5%
    envelope.
  - **LBB / Taylor–Hood (the key numerical point).** An equal-order (linear
    u / linear p) monolithic solve violates the inf-sup (LBB) condition and shows
    a spurious pressure overshoot at the drained boundary (max ≈ 173 vs the
    correct 84). The monolithic path therefore uses a **Taylor–Hood T6-
    displacement / T3-pressure** pairing: the CST input mesh is converted to T6
    internally, pressure DOFs live on the corner nodes only (so H and S reuse the
    CST corner-skeleton matrices), and the mixed coupling
    `Q_e = ∫ B_u^T m N_p dA` is integrated against the linear corner-pressure
    shape. This removes the overshoot (residual t=0 boundary-layer max ≈ 119,
    dissipating immediately) with **no stabilization term and no fudge on the
    physics**.
  - *Storage / units (resolved):* the monolithic transient needs the **mobility**
    `k_mob = k/γ_w` (m²/(kPa·s)) passed as `k` and the Biot modulus **M** (kPa)
    passed as `n_w`; the coupling supplies the skeleton storage `α²/(K+4G/3)`, so
    the effective storage is the full `S = 1/M + α²/(K+4G/3)` (the earlier
    staggered note about `S = 1/n_w` only applied to the uncoupled scheme).
  - *p0 reconciliation (documented):* the inventory quotes both a formula
    (`α·M/(K+4G/3+α²·M)·pz` → **0.839e5**, consistent with the stated M=4e9) and
    Itasca's reported **0.981e5** (which needs an effective M ≈ 10× larger, i.e.
    a near-incompressible fluid). fem2d reproduces the **0.839e5** value that is
    consistent with the stated M; **tuned to neither** target.

- **Module edits (V-024 batch, fem2d only):** `assembly.apply_bcs_penalty`
  (+`roller_base`), `solver.build_nl_context` (+`roller_base` DOF),
  `solver.run_nl` (+`f_int_offset`), `solver.solve_nonlinear`
  (+`initial_stress_relaxation`). All additive/opt-in.
- **Module edits (V-023, v5.3 A5, commit d1e27a7):** `porewater.solve_consolidation`
  (+`scheme="staggered"|"monolithic"`, +`theta`), new
  `porewater._monolithic_consolidation` / `_monolithic_taylor_hood` /
  `assemble_coupling_taylor_hood`; `analysis.analyze_consolidation` and
  `fem2d_adapter` (+`consolidation_scheme` allowed_values + `theta`). All
  additive/opt-in — the default staggered path is byte-identical (369 fem2d
  tests unchanged; +3 `test_v023_monolithic_*`). The V-023 limitation is now
  patched, not just documented.
