# Das "Principles of Foundation Engineering" 6e — Instructor's Solutions Manual sweep

> **INTERNAL-ONLY SOURCE.** This sweep verifies module behavior against the
> Instructor's Solutions Manual for Das, *Principles of Foundation Engineering*,
> 6th ed. (Thomson, 2007) — a copyrighted document held locally
> (`...\Braja M. Das\Principles of Foundation Engineering\6th Edition\
> Principles of Foundation Engineering 6th - Solution Manual.pdf`, scanned,
> read visually page-by-page; problem statements cross-read from the textbook
> PDF in the same folder). Inputs/answers are quoted here strictly as
> verification data. **Nothing from this source may enter the shipped corpus,
> worked_examples.json, docstrings, or any published anchor set.**
>
> Method: eight solved problems with complete numeric inputs and answers,
> chosen to exercise module paths WITHOUT existing published anchors
> (bearing inclined / two-way eccentricity / water table; Schmertmann strip;
> consolidation magnitude + time-rate; cantilever wall full checks; anchored
> sheet pile in clay; alpha/beta pile in layered clay). US→SI conversions:
> 1 ft = 0.3048 m, 1 lb/ft2 = 0.047880 kPa, 1 lb/ft3 = 0.157087 kN/m3,
> 1 kip = 4.4482 kN, 1 lb/in2 = 6.89476 kPa. Runs go through
> `funhouse_agent.dispatch.call_agent` (module time-rate helpers called
> directly where the dispatch layer has no method).
>
> Classification: MATCH (<5%), CONVENTION-GAP (explainable, reconciliation
> shown), SUSPECTED-DEFECT (unexplainable after honest reconciliation).

Date: 2026-07-19. Sweep agent: defect-detection (verification only, no module edits).

---
## Problem 1.17 — consolidation magnitude + time-rate (settlement module) — SALVAGED FROM PRIOR RUN

**Statement (manual p. -6-):** Normally consolidated clay, lab curve e1=0.91 @ 21 kN/m2 (converted context: manual works in lb/ft2), e2=0.792 @ 42 → Cc = 0.392. Field: Hc = 15 ft (two-way drainage), sigma'0 = 1553.6 lb/ft2 (from Problem 1.12 profile), e0 = wGs = 0.953, delta-sigma = 1000 lb/ft2, cv = 1.45e-4 in2/s.
**Published answers:** (a) Sc = 7.8 in; (b) t(U=50%, Tv=0.197) = 509.5 days.
**Module run (prior interrupted session, settlement/consolidation + time-rate helper):** both parts reproduced at **−0.03%** (magnitude) and **−0.3%** (time-rate; module Tv(50) analytic vs the manual's chart read 0.197).
**Classification: MATCH (both parts).**

---
## Problem 3.4 — inclined load, square footing (bearing_capacity)

**Statement (text p. 165):** Square footing B = 5.5 ft, Df = 4 ft, γ = 107 lb/ft3, φ' = 25°, c' = 350 lb/ft2; load inclined 15° from vertical; FS = 4; §3.6 factors (Table 3.3 N's + De Beer shape + Hansen depth + Meyerhof inclination).
**Published (manual p. -21-):** Fcs=1.514, Fqs=1.466, Fγs=0.6, Fcd=1.29, Fqd=1.226, Fci=Fqi=0.694, Fγi=0.16 → **Qall = 119.7 kip** (= 532.4 kN).
**Module run** (`bearing_capacity_analysis`, SI: B=L=1.6764 m, Df=1.2192 m, c'=16.76 kPa, γ=16.81 kN/m3, `load_inclination=15`, `factor_method="vesic"`, FS=4; with `vertical_load` omitted the module warns and falls back to the angle-based Meyerhof inclination factors — exactly Das's family): every factor matches the manual (Nc 20.72/Nq 10.66/Nγ 10.88; sc 1.5146/sq 1.4663/sγ 0.6; dq 1.2261; ic=iq 0.6944; iγ 0.16) except **dc: module 1.2495 (Vesic exact dc = dq−(1−dq)/(Nc·tanφ)) vs Das 1.29 (simplified 1+0.4·Df/B)**. Module Qall = q_all·B² = **522.6 kN → −1.85 %**.
**Reconciliation:** substituting Das's dc=1.2909 into the module's own term decomposition gives Qall = 533.2 kN (+0.16 %). Sole difference = printed dc-form choice.
**Classification: MATCH** (−1.85 % raw; ±0.2 % after dc-form alignment).

---

## Problem 3.10 — one-way eccentricity + water table above base (bearing_capacity)

**Statement (text p. 167, Fig. P3.10):** 8 × 8 ft footing, Df = 6.5 ft, GWT 3 ft below surface; γ = 110 lb/ft3 above GWT, γsat = 122(.4) below; c' = 500 lb/ft2, φ' = 26°, e = 0.65 ft (one-way). Meyerhof effective-area method.
**Published (manual pp. -25/-26-):** B' = 6.7 ft; q = 3(110)+3.5(122−62.4) = 538.6 psf; q'u = 34,213 psf; **Qu = 1833.8 kip** (= 8157 kN).
**Module runs** (`bearing_capacity_analysis`, B=L=2.4384 m, Df=1.9812 m, `gwt_depth=0.9144`, `eccentricity_B=0.19812`, c'=23.94 kPa, φ'=26): the adapter takes a SINGLE unit weight per layer, so the two-γ profile can't be encoded exactly; runs with γ=110 pcf and γ=122.4 pcf bracket the truth: Qu = q_ult·B'L = **7969 kN (−2.3 %)** and **8470 kN (+3.8 %)**. Factors match the manual (Nc 22.25/Nq 11.85/Nγ 12.54; sc 1.4461 = Das Fcs 1.446; sq 1.4085 = Fqs 1.408; sγ 0.665 = Fγs).
**Reconciliation:** hand-assembling the module's own factor functions with the exact overburden q = 25.79 kPa and γ' = 9.36 kN/m3 (and Das's full-B depth factors) gives Qu = 8163 kN (**+0.07 %**). Residual bracket spread is purely the single-γ input-fidelity limit (module depth factors also use B' where Das uses full B — worth ~+1 % here, inside the bracket).
**Classification: MATCH** (−2.3 % on the conservative encoding; +0.07 % at exact inputs). *Ergonomics note (not a defect): `bearing_capacity_analysis` cannot take separate moist/saturated unit weights around a GWT.*

---

## Problem 3.13 — two-way eccentricity (bearing_capacity)

**Statement (text p. 168):** 4 × 6 ft footing, Df = 3 ft, eB = 0.4 ft, eL = 1.2 ft, γ = 115 lb/ft3, φ' = 35°, c' = 0, FS = 4; Highter & Anders (1985) effective-area (Case II charts: eB/B=0.1, eL/L=0.2 → L1/L=0.865, L2/L=0.22).
**Published (manual p. -28-):** A' = 13.02 ft2, B' = 2.51 ft, L' = 5.19 ft, Fqs=1.339, Fγs=0.806, Fqd=1.191 (full B), q'u = 23,908 psf, **Qall = q'u·A'/FS = 77.86 kip** (= 346.3 kN).
**Module run** (`bearing_capacity_analysis`, eccentricity_B=0.12192 m, eccentricity_L=0.36576 m, vesic factors): the module uses the **Meyerhof (1953) rectangular effective area B'=B−2eB, L'=L−2eL** (B'=3.2 ft, L'=3.6 ft, A'=11.52 ft2) — NOT the Highter-Anders charts. q_all = 344.5 kPa → Qall = q_all·A' = **368.7 kN → +6.5 %** vs published.
**Reconciliation:** feeding the module's own factor kernels the Highter-Anders dims (B'=2.51 ft, L'=5.19 ft, depth on full B) reproduces q'u = 1144.6 kPa vs printed 1144.7 (−0.01 %) and Qall = 346.1 kN (**−0.06 %**). The whole +6.5 % is the documented effective-area convention (Meyerhof rectangle: higher shape factors, smaller A'; net less conservative here).
**Classification: CONVENTION-GAP** (Meyerhof-1953 rectangle vs Highter-Anders charts for two-way eccentricity; module kernels exact under the printed convention). *Coverage note: no Highter-Anders option exists in the module.*

---
## Problem 5.18 — Schmertmann strain-influence, strip footing, variable Es (settlement)

**Statement (text p. 268, Fig. P5.18):** Continuous footing B = 8 ft, Df = 5 ft, gross q̄ = 4000 lb/ft2, γ = 115 lb/ft3, creep C2 at 10 yr; Es profile below base: 875 psi (0–6 ft), 1740 psi (6–20 ft), 1450 psi (20–32+ ft).
**Published (manual p. -49-):** simplified strip diagram (Iz0 = 0.2, peak Iz = **0.5 fixed** at z = B, zero at 4B); ΣIz/Es·Δz = 0.0756; C1 = 0.916, C2 = 1.4 → **Se = 2.31 in** (58.7 mm).
**Module runs** (`schmertmann_settlement`, q_net = 164 kPa, q_overburden = 27.5 kPa, shape="strip", time_years=10, manual's four sublayers 0-6/6-8/8-20/20-32 ft): Se = **83.1 mm → +41.6 %**. Cause: the module implements the full Schmertmann-1978 peak correction Izp = 0.5 + 0.1·√(Δq/σ'vp); with the adapter's default σ'vp = q0 (base) it gets Izp = 0.744. Direct call with `gamma_soil` (σ'vp at the peak depth, the documented SET-1 behavior) gives Izp = 0.651 → 73.8 mm (+25.7 %).
**Reconciliation:** forcing Izp = 0.5 in the module's own diagram geometry reproduces the manual **exactly**: per-sublayer Iz = 0.312/0.463/0.375/0.125 (printed: 0.313/0.463/0.375/0.125), C1 = 0.916, C2 = 1.4, Se = 58.5 mm (**−0.25 %**). The entire discrepancy is the printed simplified-diagram (fixed 0.5 peak) vs the module's primary-source Schmertmann-1978 Izp correction.
**Classification: CONVENTION-GAP** (module is the more faithful 1978 formulation; Das 6e teaches the pre-correction diagram). *Two ergonomics notes: (1) the dispatch adapter does not expose `gamma_soil`, so agent runs silently fall back to σ'vp = q0, overstating Izp (0.744 vs 0.651 here); (2) the integrator takes ONE Iz sample per user layer — a coarse layer straddling the Iz peak loses the kink (83.1 vs 84.1 mm between 4- and 3-layer encodings); it should subdivide internally at z_peak.*

---
## Problem 8.3 — cantilever retaining wall, full external checks (retaining_walls + bearing_capacity chain)

**Statement (text pp. 404-405, Fig. P8.1):** Cantilever wall, SI: H(stem) = 6.5 m, stem 0.3 m top / 0.6 m bottom, toe 0.8 m, heel 2.0 m, base 0.8 m thick (B = 3.4 m, total H' = 7.3 m), D = 1.5 m, α = 0; backfill γ1 = 18.08, φ1' = 36°; foundation γ2 = 19.65, φ2' = 15°, c2' = 30 kPa; γconc = 23.58; k1 = k2 = 2/3, Pp = 0; Rankine.
**Published (manual pp. -81/-83-):** Ka = 0.26, Pa = 125.25 kN/m, ΣV = 368.15, ΣMr = 753.52, Mo = 304.78 → **FS_ot = 2.47**; **FS_sliding = 1.06**; e = 0.481 m, **q_toe = 200.19 kPa**; qu (inclination ψ = 18.79°, B' = 2.438) = 346.85 → **FS_bearing = 1.73**.
**Module run** (`retaining_walls.cantilever_wall`, wall_height=7.3, base_width=3.4, toe 0.8, stem 0.3/0.6, base 0.8, phi_foundation=15, c_foundation=30 — 2/3 factors applied internally, include_passive=False): **FS_sliding = 1.063 (+0.3 %)**, **FS_ot = 2.446 (−1.0 %)**, **q_toe = 204.7 kPa (+2.3 %)**, e = 0.505 m. ΣV identical (368.15 reproduced in components); the small FS_ot/e drift is the stem-taper orientation: the module puts the rectangular stem panel on the toe side and the taper on the heel side (arms 0.95/1.20 m) where Fig. P8.1 has the taper on the toe side (arms 1.25/1.00 m) — ΔM = 9.1 kN·m/m explains Δe = +0.024 m exactly.
**Bearing chain** (`bearing_capacity_analysis`, strip B = 3.4, Df = 1.5, ecc_B = e, load_inclination = 18.79°, vesic factors + angle-based inclination fallback): qu = 346.1 kPa with Das's e (−0.21 %), 347.3 with the module's e (+0.14 %) → **FS_bearing = 1.73 / 1.70** vs published 1.73.
**Classification: MATCH** (all four checks ≤ 2.3 %; taper-orientation arm difference documented, conservative here).

---
## Problem 9.14 — anchored sheet-pile bulkhead penetrating clay, free earth (sheet_pile)

**Statement (text p. 463, Fig. P9.14):** L1 = 3 m (moist sand, γ = 17, φ' = 36°), L2 = 8 m (submerged sand, γsat = 19.5), water at L1 both sides (waterfront bulkhead, balanced); below dredge line: clay, c = 40 kPa (φ = 0); anchor at l1 = 1.5 m. Free earth support.
**Published (manual pp. -104/-105-):** P1 = 206.53 kN/m, σ6 = 4c − (γL1 + γ'L2) = 31.48 kPa, D² + 19D − 72.69 = 0 → **D_theory = 3.3 m**; **F = 102.6 kN/m**.
**Module run** (`sheet_pile.anchored_wall`, excavation_depth=11, anchor_depth=1.5, FOS_passive=1.0; balanced water encoded by passing the submerged layer's EFFECTIVE unit weight 9.69 kN/m3 with no gwt — the adapter's gwt_depth_active/passive parameters model an UNBALANCED head, which is not this problem): **D = 3.41 m (+3.3 %)**, **F = 99.5 kN/m (−3.0 %)**. The module's numeric free-earth balance (pa = σv−2c, pp = σv+2c, net = 4c − (γL1+γ'L2)) is algebraically identical to the printed Eq. (9.75) treatment; the +3.3 % on D is embedment-search quantization — D is scanned on np.linspace(0.5, 4H, 300) (~0.145 m steps here) and the FIRST grid point past moment balance is returned (bias high by ≤1 step; 3.41 is a grid point, true balance in (3.26, 3.41]). The −3 % on F follows (larger D → more passive → less anchor).
**Classification: MATCH** (both ≤3.3 %, fully explained by grid quantization). *Two ergonomics notes: (1) no "balanced water table both sides" input mode — waterfront problems require the effective-γ workaround above (undocumented); (2) the embedment grid has a 0.5 m FLOOR and ~(4H−0.5)/300 resolution: a shallow-embedment problem (e.g. the manual's US variant of this same wall, D_theory = 1.15 ft = 0.35 m) cannot be resolved at all — worth an adaptive refinement pass in `sheet_pile/anchored.py`.*

---
## Problem 11.10 — pile skin friction in layered clay, alpha / lambda / beta (axial_pile)

**Statement (text p. 585, Fig. P11.10):** Concrete pile 16 × 16 in, L = 60 ft; silty clay 0–20 ft (γsat = 118 pcf, cu = 700 psf), 20–60 ft (γsat = 122.4 pcf, cu = 1500 psf); GWT at 20 ft; φR = 20°, normally consolidated. Ultimate skin friction by (a) α, (b) λ, (c) β.
**Published (manual pp. -119/-120-):** (a) α1 = 0.6, α2 = 0.59 (Das Fig. 11.23, α as a function of cu/σ'v: 0.59, 0.63) → **Qs = 233 kip** (1036 kN); (b) λ = 0.18 → Qs = 301 kip; (c) β = (1−sin20°)tan20°, σ'av = 1180/… → fav = 282.6 / 852.6 psf → **Qs = 212 kip** (943 kN).
**Module runs** (`axial_pile_capacity`, concrete_square w=0.4064 m, L=18.288 m, gwt 6.096 m):
- **(a) auto (Tomlinson α):** Q_skin = **1124.6 kN → +8.5 %**. The module's GEC-12 Fig 7-17 Tomlinson curve is α(cu): α1 = 0.881, α2 = 0.585 vs Das's α(cu/σ'v) chart 0.6 / 0.59 — the low-cu layer drives the gap. Kernel check: the module's own Qs = p·Σα·cu·ΔL with the PRINTED α's gives 1039 kN (+0.3 %). Different published α-family, same kernel.
- **(c) beta:** first run with per-layer `friction_angle=20` on the cohesive layers returned **+12.4 %** — routing finding: for cohesive layers the beta path IGNORES the per-layer friction angle and always uses the global `cohesive_phi` (default 25°; β(25)/β(20) = 1.126 explains the offset exactly, axial_pile/capacity.py lines 296-300). With `cohesive_phi=20`: Q_skin = **942.9 kN vs 943.0 → −0.01 %** (exact: same (1−sinφ)tanφ β, same effective-stress profile).
- **(b) λ (Vijayvergiya-Focht):** not implemented in the module (no comparison; coverage note).
**Classification: (a) CONVENTION-GAP** (α-curve family; module kernel exact under printed α's), **(c) MATCH** (−0.01 %). *Two findings: (1) adapter doc for `cohesive_phi` says "Nordlund tip term" but it also silently drives beta-method SKIN (and tip) in every cohesive layer, and a supplied per-layer `friction_angle` on a cohesive layer is ignored — one global φR for all clay layers, misleading doc; (2) no λ-method option.*

---

## Sweep summary

| # | Problem | Module path | Raw diff | After reconciliation | Class |
|---|---------|-------------|----------|----------------------|-------|
| 1 | 1.17 consolidation magnitude + t50 | settlement (consolidation + time-rate) | −0.03 % / −0.3 % | — | MATCH |
| 2 | 3.4 inclined load | bearing_capacity (Meyerhof-inclination fallback) | −1.85 % | +0.16 % (dc form) | MATCH |
| 3 | 3.10 eccentric + GWT | bearing_capacity (effective area + gwt) | −2.3 % | +0.07 % (exact q, γ') | MATCH |
| 4 | 3.13 two-way eccentricity | bearing_capacity | +6.5 % | −0.06 % (H-A dims) | CONVENTION-GAP (Meyerhof rectangle vs Highter-Anders) |
| 5 | 5.18 Schmertmann strip | settlement (strain influence) | +41.6 % | −0.25 % (Izp=0.5) | CONVENTION-GAP (1978 Izp correction vs simplified diagram) |
| 6 | 8.3 cantilever wall (4 checks) | retaining_walls + bearing_capacity | +0.3/−1.0/+2.3/−0.2 % | arm geometry explained | MATCH |
| 7 | 9.14 anchored SP in clay | sheet_pile (free earth) | +3.3 % (D), −3.0 % (F) | grid quantization | MATCH |
| 8 | 11.10 α/β layered clay | axial_pile | +8.5 % (α) / −0.01 % (β) | +0.3 % (printed α) | CONVENTION-GAP (α) / MATCH (β) |

**No suspected defects.** Every discrepancy reconciled to ≤0.3 % once the printed convention was fed to the module's own kernels — the signature of correct arithmetic under different published conventions, not digitization errors. Actionable ergonomics/coverage items surfaced (not defects): bearing single-γ around GWT; no Highter-Anders two-way option; Schmertmann adapter missing `gamma_soil` + single Iz sample per user layer; sheet-pile embedment grid floor 0.5 m / no refinement + no balanced-water mode; axial_pile beta ignores per-layer φ for cohesive layers (global `cohesive_phi`, misdocumented as tip-only) + no λ method.
