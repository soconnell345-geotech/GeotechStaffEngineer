# Wave 7 (final) — EM 1110-2-2906 pile_group / PTI DC35.1-14 bond stresses / Mononobe-Okabe anchor

**Summary:** (1) pile_group 6-DOF rigid-cap stiffness basis **EQUIVALENT-NOTATION** vs EM 1110-2-2906
Ch. 4 (¶4-5, printed matrix eqs {q}=[B]{u} and [K]=Σ[K]i on p. 4-29 confirmed by render; module is a
documented subset — axial+lateral springs only, no per-pile rotational stiffness; minor citation
flag: calc_steps.py cites "Eq. 4-1" but the EM's Ch. 4 equations are unnumbered). (2) PTI DC35.1-14
presumptive bond stresses **FUTURE-ANCHOR** — soe codes a GEC-4-attributed table (correctly cited,
values differ from PTI's); PTI Tables C6.1/C6.2/C6.3 located at printed pp. 47–49 (PDF 55–57) with
representative values recorded; the §6.7 bond-length equation + §6.6 FS=2.0 the module implements
DO match PTI. (3) Mononobe-Okabe K_AE **CONFIRMED** — Whitman (1990) GSP 25 Fig. 11 (printed
p. 831) numeric anchor reproduced digit-for-digit: P_A = 8892 and ΔP_AE = 4088 lb/ft at φ=35°,
δ=0, k_h=0.2, H=25 ft, γ=105 pcf (our K_A=0.2710, K_AE=0.3956); Steedman (1998) is formulas-only
(NOT-FOUND numeric there, seismic-angle definition consistent). No code discrepancies found.

---

## Item 1 — EM 1110-2-2906 (1991) basis for pile_group 6-DOF rigid-cap stiffness method

**Source:** `C:\Users\socon\OneDrive\Lib\02 Tech\Military\ACoE\Design of Pile Foundations.pdf`
(EM 1110-2-2906, 15 Jan 91, 186 PDF pages; text layer good, equations are images —
p. 4-29 rendered visually to read them).

**Module:** `pile_group/rigid_cap.py` (`analyze_group_6dof`, `analyze_vertical_group_simple`),
`pile_group/DESIGN.md`, `pile_group/calc_steps.py`. Docstrings cite "CPGA User's Guide (ITL-89-4)"
and "USACE EM 1110-2-2906, Chapter 4".

### What the EM prints (page citations = EM page / PDF page)

| EM statement | EM page (PDF page) |
|---|---|
| ¶4-5b(1): Stiffness method basis — Hrennikoff (2-D), Aschenbrenner (3-D), **Saul (matrix methods, incorporates position and batter of piles)**; "The Saul approach is the basis for the pile analysis presented in the following paragraphs… implemented in the computer program CPGA (Item 5)." | 4-28 (PDF 51) |
| ¶4-5b(2): Pile-soil model — structure supported by "sets of six single degree-of-freedom springs"; each pile: **{q}_i = [B]_i {u}_i**, q_i = (F1,F2,F3,M1,M2,M3) pile forces/moments, u_i = (U,V,W,Θ1,Θ2,Θ3) displacements/rotations; **[K] = Σ_{i=1..n} [K]_i** with "[K]_i = stiffness of ith pile **transformed to global coordinates**" (printed equations verified by page render; they carry NO equation numbers). | 4-28/4-29 (PDF 51/52) |
| ¶4-5b(3): Rigid vs flexible base — "applied loads are distributed to each pile on the basis of **rigid body behavior** (Figure 4-8) as is the case in CPGA". | 4-30 (PDF 53) |
| ¶4-5d: Axial pile-head stiffness **b33 = C33·AE/L**, C33 = soil-pile interaction constant (1.0–2.0 typical for compression). | 4-31/4-32 (PDF 54/55) |
| ¶4-5e: Lateral stiffness from E, I, pile-head fixity constant C1, and Es or nh (subgrade-reaction, secant-linearized at an estimated deflection); group/cyclic reduction factors Rg, Rc. | 4-33 to 4-37 (PDF 56–60) |
| ¶4-5f: Torsional stiffness **b66 = C66·JG/L** (C66 ≈ 2 fixed-head, 0 free-head). | 4-38 (PDF 61) |
| ¶4-7d(2): "For a pile group that contains only vertical piles, the **rigid cap assumption requires that the plane of the pile heads remains plane** when loads are applied"; batter couples the axial/lateral reaction components. | 4-45 (PDF 68) |

### Module vs EM

- **Rigid-cap equilibrium system:** module assembles K_group(6×6) = Σ B^T k B and solves
  K_group·U = F for the 6 cap DOFs, then back-substitutes the SAME rigid-cap kinematic matrix B
  (`ux = dx − rz·y`, `uy = dy + rz·x`, `s = dz − rx·y + ry·x`) for pile forces. This is exactly the
  EM's [K] = Σ [K]_i global assembly + rigid-body-behavior load distribution (EM 4-29/4-30), written
  in explicit congruence-transform notation. The plane-of-pile-heads-remains-plane kinematics match
  EM 4-45. **EQUIVALENT-NOTATION.**
- **Pile-head stiffness coefficients:** module takes per-pile `axial_stiffness` (ka) and
  `lateral_stiffness` (kl) as inputs — the EM's b33 = C33·AE/L and the Es/nh-derived lateral
  stiffness are exactly what the user supplies; the module does not compute C33/C1 internally
  (consistent scope split; EM methods are the recipe for the inputs).
- **Batter handling:** module projects ka along the pile axis via direction cosines
  (kxx = ka·lx² + kl·(1−lx²), etc., incl. the kxy in-plane coupling) — Saul's "position and batter"
  transformation to global coordinates (EM 4-28/4-29). Matches in kind.
- **Documented subset (not a discrepancy):** the module models only translational pile-head springs
  (3×3 k per pile) — no per-pile rotational/torsional stiffnesses (EM b44/b55 via C1, b66 = C66·JG/L)
  and no kxz/kyz batter force↔rotation coupling. `rigid_cap.py` states this PG-1 limitation
  explicitly ("for heavily battered groups under combined load, a full CPGA-style coupled
  formulation is required"), and DESIGN.md repeats it. Moment resistance comes from axial-spring
  eccentricity, which the rigid-cap kinematics provide.
- **Simplified method** (`analyze_vertical_group_simple`, P_i = Vz/n + My·x_i/Σx² − Mx·y_i/Σy²):
  this is the classic "approximate/elastic" distribution the EM *describes and discourages* for
  laterally loaded groups (¶4-5a, EM 4-27/PDF 50: distribute "based on pile location, batter, and
  cross-sectional area… should not be used except for very simple 2-D structures where lateral
  loads are small"). The specific formula is not printed in the EM; the module's own docstring
  correctly warns and ignores nothing silently.
- **Minor citation flag (docs only, no code impact):** `calc_steps.py` line 372 cites
  "USACE EM 1110-2-2906, **Eq. 4-1**" for the simplified elastic distribution. The EM's Chapter 4
  equations are unnumbered (verified: no "(4-x)" labels in the text layer; the rendered p. 4-29
  equations carry no numbers), and the P_i formula is not printed in the EM at all. A better cite
  would be "EM 1110-2-2906 ¶4-5a (approximate methods)" or a foundations text. Also
  `calc_steps.py` line 353 labels the 6-DOF method "CPGA" — fair as a basis label given the
  documented subset caveat.

**VERDICT: EQUIVALENT-NOTATION** (module's K = Σ B^T k B matrix form = EM's Saul/CPGA
[K] = Σ [K]_i rigid-cap stiffness method, printed on EM pp. 4-28/4-29 (PDF 51/52) with the
rigid-cap distribution on 4-30 (PDF 53) and pile-head stiffness definitions on 4-31 (PDF 54) and
4-38 (PDF 61); rotational-stiffness/batter-coupling subset is explicitly documented in the module.
One minor doc-citation flag: "Eq. 4-1" does not exist as a printed equation number.)

---

## Item 2 — PTI DC35.1-14 presumptive grout-bond stresses vs soe

**Source:** `C:\Users\socon\OneDrive\Lib\02 Tech\PTI\2014 PTI Recommendations.pdf`
(PTI DC35.1-14, *Recommendations for Prestressed Rock and Soil Anchors*; 120 PDF pages,
text layer usable but table row/value alignment scrambled — the three tables were read from
150-dpi page renders).

**Module:** `soe/anchor_design.py`. The presumptive bond table `_BOND_STRESS_TABLE`
(lines 26–57) is explicitly attributed to **FHWA GEC-4 Table 4**, NOT to PTI; PTI DC35.1 is used
in this module only for tendon strand/bar data (line 61 comment) — and `soe/DESIGN.md`
lines 146–156 says exactly that (with its own RED FLAG that the anchor tables are unbenchmarked).
So the module codes **no PTI-sourced presumptive bond values** → per the audit protocol this item
records the PTI printed table locations + representative values as a FUTURE-ANCHOR.

### Where the presumptive tables are printed in DC35.1-14

- **Table C6.1 — Typical average ultimate bond strengths—rock/grout:** printed page 47
  (PDF page 55), in the C6.7.1 Rock-anchors commentary column.
- **Table C6.2 — Typical average ultimate bond strengths—cohesive soils:** printed page 48
  (PDF page 56), C6.7.2 commentary.
- **Table C6.3 — Typical average ultimate bond strengths: non-cohesive soils:** printed page 49
  (PDF page 57), C6.7.2 commentary. (Bond-length equation Lb = FS·P/(π·d·τu) is in §6.7,
  printed p. 45 / PDF p. 53 — this IS what `compute_bond_length` implements, with the same FS
  placement; §6.6 sets min FS = 2.0 on the ground-grout interface for permanent anchors, matching
  the module's `FOS_bond = 2.0` default.)

### Representative printed values (read from renders, digit-verified)

| PTI table / row | Printed value, MPa (psi) |
|---|---|
| C6.1 Granite and basalt | 1.7 to 3.1 (250 to 450) |
| C6.1 Sandstones | 0.8 to 1.7 (120 to 250) |
| C6.1 Soft shales | 0.2 to 0.8 (30 to 120) |
| C6.1 Weathered marl | 0.15 to 0.25 (25 to 35) |
| C6.2 Gravity-grouted anchors (straight shaft), cohesive | 0.03 to 0.07 (5 to 10) |
| C6.2 Pressure-grouted, very stiff clay, medium plasticity | 0.14 to 0.35 (20 to 50) |
| C6.3 Gravity-grouted anchors (straight shaft), non-cohesive | 0.07 to 0.14 (10 to 20) |
| C6.3 Pressure-grouted, fine-medium sand, medium dense to dense | 0.08 to 0.38 (12 to 55) |
| C6.3 Pressure-grouted, sandy gravel, dense to very dense | 0.28 to 1.38 (40 to 200) |

C6.7.1 commentary also prints the rule of thumb: rock ultimate bond ≈ **10% of UCS up to a
maximum of 4.2 MPa (600 psi)** (p. 47 / PDF 55).

### Cross-check of the coded GEC-4 values against PTI (informational)

The module's GEC-4-attributed bins do NOT coincide with PTI's tables digit-for-digit — e.g. coded
`rock_hard` "granite, basalt" 1400–2100 kPa vs PTI granite/basalt 1700–3100 kPa; coded
`sand_dense` 190–310 kPa vs PTI fine-medium sand (pressure-grouted) 80–380 kPa. One coincidence:
coded `clay_stiff` range 30–70 kPa = PTI C6.2 gravity-grouted cohesive 0.03–0.07 MPa. Takeaway:
the coded table must keep its GEC-4 citation and should not be cross-cited to PTI DC35.1-14.

**VERDICT: FUTURE-ANCHOR** (module codes no PTI presumptive bond values — coded table is GEC-4
Table 4 basis, correctly cited; PTI DC35.1-14 printed values recorded above at printed pp. 47–49 /
PDF pp. 55–57 for a future benchmark. The §6.7 bond-length equation and §6.6 FS = 2.0 that the
module DOES implement match the PTI text — printed pp. 45–46 / PDF pp. 53–54.)

---

## Item 3 — Mononobe-Okabe K_AE numeric anchor

**Sources (both scanned, no text layer — read from page renders):**
- `C:\Users\socon\OneDrive\Lib\01\1. Geotechnical\3. ERS\9. Seismic ERS\Seismic Design of ERS.pdf`
  = Steedman (1998), "Seismic design of retaining walls," Proc. ICE Geotech. Engng 131, pp. 12–22
  (11 PDF pages).
- `C:\Users\socon\OneDrive\Lib\01\1. Geotechnical\3. ERS\9. Seismic ERS\Seismic Design of Gravity ERS.pdf`
  = Whitman (1990), "Seismic Design and Behavior of Gravity Retaining Walls," ASCE GSP 25,
  pp. 817–ff (14 PDF pages).

**Module:** `seismic_geotech/mononobe_okabe.py` — `mononobe_okabe_KAE(phi, delta, kh, kv, beta, i)`
(full M-O with seismic inertia angle θ = atan(kh/(1−kv)); Seed-Whitman fallback
`Ka_static + 0.75·kh` when φ−θ−i < 0). `retaining_walls` has no independent M-O implementation
(grep clean — it consumes seismic_geotech).

### What the sources print

- Steedman (1998) ¶6 explicitly does NOT present the M-O equations ("summarized in many
  references … and therefore are not presented here", printed p. 13 / PDF p. 2). It DOES print the
  seismic-angle definitions Eq. 1(a)/1(b): θf = tan⁻¹[γd·kh/(γb(1−kv))], θr = tan⁻¹[γt·kh/(γb(1−kv))]
  — the free/restrained water generalizations of the module's θ = tan⁻¹[kh/(1−kv)] (dry case:
  γd = γb ⇒ identical). No numeric K_AE example. **NOT-FOUND (numeric) in this paper.**
- Whitman (1990) prints (p. 819 / PDF p. 2): **Eq. (1)** P_AE = (1/2)γ(1−kv)H²K_AE; **Eq. (2)**
  Seed-Whitman approximation **K_AE ≈ K_A + (3/4)k_h**; Fig. 1 K_AE chart (φ = 35°, δ = 0 and ½φ).
  And — the usable numeric anchor — **Figure 11** (p. 831 / PDF p. 8): "Wall designed by
  traditional approach, with friction angles of 35 degrees and seismic coefficient of 0.2":
  25-ft wall (2.5-ft top, 15.5-ft base), printed force diagrams **STATIC = 8892** lb/ft and
  **DYNAMIC increment = 4088** lb/ft (plus wall inertia 6769 and dynamic base value 10859).

### Anchor run (our implementation, exact inputs)

Run: `.venv/Scripts/python.exe -X utf8` importing `seismic_geotech.mononobe_okabe` directly.
Inputs implied by Fig. 11: φ = 35°, δ = 0, k_h = 0.2, k_v = 0, vertical wall, level backfill, H = 25 ft.

| Quantity | Printed (Whitman Fig. 11) | Our implementation | Match |
|---|---|---|---|
| K_A (static, φ=35°, δ=0) | 8892 = ½γH²K_A ⇒ K_A = 0.2710 at γ = 105 pcf | `mononobe_okabe_KAE(35, 0, 0.0)` = 0.2710 | exact (γ backs out to 105.00 pcf — clean round number, confirms the interpretation) |
| P_A | **8892 lb/ft** | ½·105·25²·0.2710 = **8892 lb/ft** | digit-for-digit |
| K_AE (φ=35°, δ=0, kh=0.2) | implied 12980/32813 = 0.3956 | `mononobe_okabe_KAE(35, 0, 0.2)` = **0.3956** | 4 significant digits |
| ΔP_AE = P_AE − P_A | **4088 lb/ft** | 12980 − 8892 = **4088 lb/ft** | digit-for-digit |
| Wall inertia k_h·W | 6769 lb/ft | 0.2·(225 ft²·150 pcf) = 6750 lb/ft | 0.3% (their γ_c rounding; not a module quantity) |

Formula cross-checks: Whitman Eq. (1) = the module's `seismic_earth_pressure` PAE_total =
½γH²K_AE (kv = 0 case) — same expression. Whitman Eq. (2) K_A + (3/4)k_h is symbol-for-symbol the
module's documented Seed-Whitman fallback (`Ka_static + 0.75*kh`, mononobe_okabe.py line 90).
Steedman Eq. 1(a) reduces to the module's θ for dry backfill. Sanity: our K_AE(35°, δ=½φ, 0.2)
= 0.3797 sits on Whitman Fig. 1's δ = ½φ curve at k_h = 0.2 (chart-consistent).

**VERDICT: CONFIRMED** (printed numeric anchor found in Whitman 1990 GSP 25 Fig. 11, printed
p. 831 / PDF p. 8; our `mononobe_okabe_KAE` reproduces the printed static 8892 and dynamic
increment 4088 lb/ft digit-for-digit with K_A = 0.2710 / K_AE = 0.3956 at φ = 35°, δ = 0,
k_h = 0.2, γ = 105 pcf. Steedman 1998: formulas-only, NOT-FOUND for a numeric K_AE, seismic-angle
definition consistent.)
