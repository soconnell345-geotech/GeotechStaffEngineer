# Duncan (2000) — published-COV database + formula verification

**Summary: all 15 Duncan-attributed COV ranges and all 3 Duncan-attributed formulas are digit-for-digit / symbol-for-symbol CONFIRMED against the published paper; the only findings are citation-level: the paper's COV table is Table 3 (module says "Table 1" throughout), the in-situ-test rows cite "Kulhawy & Trautmann (1996)" which does not appear anywhere in the paper (Table 3 credits Harr 1984 / Kulhawy 1992 only), and two docstring nits (su source list omits Duncan 2000; "Duncan (2000) discusses N=4" is not in the paper).**

## Source

- Duncan, J. M. (2000). "Factors of Safety and Reliability in Geotechnical
  Engineering." *J. Geotech. Geoenviron. Eng.*, ASCE, 126(4), 307–316.
- PDF: `C:\Users\socon\OneDrive\Lib\01\1. Geotechnical\27. Safety and Reliability\Factors of Safety and Reliability in Geotechnical Engineering.pdf`
  (10 pages, full text layer; text extraction via PyMuPDF — no rendering needed).
- COV table = **Table 3**, "Values of Coefficient of Variation (V) for
  Geotechnical Properties and In Situ Tests", p. 310. (Table 1 in the paper is
  the retaining-wall Taylor-series analysis, p. 308; Table 4 is the Bay-mud
  Cc/(1+e) estimation study, p. 310.)
- Equations: (2a)/(2b) Taylor-series σ_F and V_F (p. 308); "2N calculations"
  (p. 309); (5) three-sigma rule (p. 310); (10) lognormal β_LN with the
  F=1.50 / V=0.17 → β_LN=2.32 / Pf=0.0102 anchor (Appendix I, p. 315).

Module under test: `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\reliability\cov_database.py`
(rows 74–105) and `reliability\stats.py`, `reliability\fosm.py`,
`reliability\VALIDATION.md`. No code was modified.

## Row-by-row: COV values (module vs paper Table 3, p. 310)

Module stores percent ranges; paper prints percent ranges. Both endpoints compared.

| # | Module key / label | Module range | Paper Table 3 row | Paper range | Paper source column | Verdict |
|---|---|---|---|---|---|---|
| 1 | `gamma` Unit weight | 3–7 | Unit weight (γ) | 3–7% | Harr (1984), Kulhawy (1992) | **CONFIRMED** |
| 2 | `gamma_b` Buoyant unit weight | 0–10 | Buoyant unit weight (γb) | 0–10% | Lacasse and Nadim (1997), Duncan (2000) | **CONFIRMED** |
| 3 | `phi` Effective stress friction angle | 2–13 | Effective stress friction angle (φ′) | 2–13% | Harr (1984), Kulhawy (1992) | **CONFIRMED** |
| 4 | `su` Undrained shear strength | 13–40 | Undrained shear strength (Su) | 13–40% | Harr (1984), Kulhawy (1992), Lacasse and Nadim (1997), Duncan (2000) | **CONFIRMED** (values); module source bracket omits "Duncan 2000" — see finding C3 |
| 5 | `su_ratio` Undrained strength ratio Su/σ′v | 5–15 | Undrained strength ratio (Su/σ′v) | 5–15% | Lacasse and Nadim (1997), Duncan (2000) | **CONFIRMED** |
| 6 | `Cc` Compression index | 10–37 | Compression index (Cc) | 10–37% | Harr (1984), Kulhawy (1992), Duncan (2000) | **CONFIRMED** |
| 7 | `pc` Preconsolidation pressure | 10–35 | Preconsolidation pressure (pp) | 10–35% | Harr (1984), Lacasse and Nadim (1997), Duncan (2000) | **CONFIRMED** |
| 8 | `k_sat` Permeability, saturated clay | 68–90 | Coefficient of permeability of saturated clay (k) | 68–90% | Harr (1984), Duncan (2000) | **CONFIRMED** |
| 9 | `k_unsat` Permeability, partly saturated clay | 130–240 | Coefficient of permeability of partly saturated clay (k) | 130–240% | Harr (1984), Benson et al. (1999) | **CONFIRMED** |
| 10 | `cv` Coefficient of consolidation | 33–68 | Coefficient of consolidation (cv) | 33–68% | Duncan (2000) | **CONFIRMED** |
| 11 | `N` SPT blow count | 15–45 | Standard penetration test blow count (N) | 15–45% | Harr (1984), Kulhawy (1992) | **CONFIRMED** (values); module source string differs — finding C2 |
| 12 | `qc` Electric CPT tip resistance | 5–15 | Electric cone penetration test (qc) | 5–15% | Kulhawy (1992) | **CONFIRMED** (values); finding C2 |
| 13 | `qc` Mechanical CPT tip resistance | 15–37 | Mechanical cone penetration test (qc) | 15–37% | Harr (1984), Kulhawy (1992) | **CONFIRMED** (values); finding C2 |
| 14 | `q_dmt` Dilatometer test tip resistance | 5–15 | Dilatometer test tip resistance (qDMT) | 5–15% | Kulhawy (1992) | **CONFIRMED** (values); finding C2 |
| 15 | `su` Vane shear test undrained strength | 10–20 | Vane shear test undrained strength (Sv) | 10–20% | Kulhawy (1992) | **CONFIRMED** (values); finding C2 |

**15 / 15 ranges digit-for-digit correct (both endpoints).** The paper's Table 3
footnote "aDuncan (2000) refers to the present paper" matches the module's
"[... Duncan 2000]" bracket usage.

## Formula checks

| Formula (module) | Module form | Paper form (page) | Verdict |
|---|---|---|---|
| Taylor-series FOSM (`fosm.py` header + engine) | Var[g] ≈ Σ(Δg_i/2)²; ΔF_i = F⁺_i − F⁻_i with ±1σ per variable, others at most-likely values; V_F = σ_F/F_MLV; "2n+1 evaluations" | Eq. (2a): σ_F = √[(ΔF1/2)²+(ΔF2/2)²+(ΔF3/2)²+(ΔF4/2)²]; Eq. (2b): V_F = σ_F/F_MLV; ΔF1 = (F⁺1 − F⁻1); "This involves 2N calculations" + step-1 F_MLV evaluation (pp. 308–309) | **CONFIRMED** symbol-for-symbol. Module's added correlation cross-term 2Σρ_ij(Δg_i/2)(Δg_j/2) is NOT in Duncan (paper assumes independence) — it is the module's documented USACE/UFC extension, clearly presented as such. |
| Three-sigma rule (`stats.py` `std_from_range`) | σ = (HCV − LCV)/N, default N=6 | Eq. (5): σ = (HCV − LCV)/6, HCV/LCV = highest/lowest conceivable value (p. 310; rule credited to Dai and Wang 1992) | **CONFIRMED** (N=6 default, HCV/LCV symbols exact). Docstring nit — finding C4. |
| Lognormal reliability index (`stats.py` `beta_lognormal`) | β_LN = ln(F/√(1+COV_F²)) / √(ln(1+COV_F²)); anchor F=1.5, COV=0.17 → β=2.32, pf≈1% | Eq. (10): β_LN = ln(F_MLV/√(1+V²)) / √(ln(1+V²)); Appendix I worked value: F_MLV=1.50, V=0.17 → β_LN=2.32, reliability 0.9898, Pf=0.0102 (p. 315) | **CONFIRMED** symbol-for-symbol; anchor digit-for-digit. |
| Normal reliability index (`stats.py` `beta_normal`, cited "UFC Eq. 7-7; Duncan 2000") | β = (μ − threshold)/σ | Not printed in Duncan (2000) — the paper gives only β_LN (Eqs. 10, 11) | **NOT-FOUND in Duncan 2000** (formula is standard and correctly covered by the primary UFC Eq. 7-7 citation; the "Duncan 2000" co-citation is loose) — finding C5. |

Bonus anchors (VALIDATION.md §1, rows 1–3) also re-verified against the paper text:
retaining wall F=1.50/V=17%→Pf≈1% (pp. 308–309, 315); LASH slope F=1.17/V=16%→Pf=18%,
~22% of the 2,000-ft slope failed (450 ft, p. 313); consolidation settlement 1.07 ft,
V=21%, SR(1%)≈1.6 (p. 313). All **CONFIRMED**.

## Citation-level findings (no numeric errors)

- **C1 — Table number: DISCREPANCY.** `cov_database.py` (docstring line 7, line 27
  reference entry, `_DUNCAN` constant line 63, hence every row's source string) and
  `VALIDATION.md` anchor #9 cite "Duncan (2000) **Table 1**". The COV table in the
  published paper is **Table 3** (p. 310); Table 1 is the retaining-wall Taylor-series
  example. The quoted title ("Values of COV for geotechnical properties and in situ
  tests") matches Table 3's title, so this is purely a wrong table number.
- **C2 — In-situ-test source string: DISCREPANCY (attribution).** The five in-situ rows
  (`_KT`, lines 69–70) cite "Kulhawy (1992); Kulhawy & Trautmann (1996); reproduced in
  Duncan (2000) Table 1". **"Kulhawy & Trautmann (1996)" appears nowhere in the paper**
  (not in Table 3's source column, not in the reference list). Paper Table 3 credits:
  SPT — Harr (1984), Kulhawy (1992); electric CPT — Kulhawy (1992); mechanical CPT —
  Harr (1984), Kulhawy (1992); DMT — Kulhawy (1992); VST — Kulhawy (1992). Harr (1984)
  is also dropped from the module's SPT / mechanical-CPT attributions.
- **C3 — su source bracket: minor omission.** Module row 4 lists
  "[Harr 1984; Kulhawy 1992; Lacasse & Nadim 1997]"; the paper adds "Duncan (2000)"
  (the present paper) as a fourth source for the 13–40% range.
- **C4 — `std_from_range` docstring: NOT-FOUND claim.** "Duncan (2000) discusses N=4"
  — the published paper never proposes N=4; it prints only /6 (Eq. 5). What the paper
  does discuss (Table 4 / Folayan et al. 1970) is that engineers underestimate the
  HCV–LCV range "by about a factor of two", which supports the docstring's parallel
  "2× judgment ranges" remark. (An N<6 recommendation exists in the later
  Christian–Baecher discussion/closure of this paper, not in the paper itself.)
- **C5 — `beta_normal` co-citation.** β = (μ−threshold)/σ is not printed in Duncan
  (2000); the UFC Eq. 7-7 primary citation carries it. Cosmetic.

## Out of scope (attributed to other sources — not verified here)

- ISSMGE TC304 (2021) Tables 1.2/1.3/1.4 rows: LL, PL, PI, w, LI, OCR, Cc(clay),
  phi(clay), su(clay), su_ratio(clay), St, qt, N(clay), K0(clay); e, phi(sand),
  qc(sand), N(sand), N160, K0(sand); gamma(rock), sigma_ci, Ei, RQD, RMR, GSI, Is50
  — **OUT-OF-SCOPE** (26 rows).
- Phoon & Kulhawy (1999b) via UFC 3-220-20 transformation rows: su from VST
  (7.5–15), su from CPT (29–35), su from SPT (15) — **OUT-OF-SCOPE** (3 rows).

## Verdict roll-up

- COV rows attributed to Duncan (2000): **15 CONFIRMED, 0 DISCREPANCY, 0 NOT-FOUND** (values).
- Formulas: Taylor-series 2N/σ_F/V_F **CONFIRMED**; three-sigma (HCV−LCV)/6 **CONFIRMED**;
  lognormal β_LN + 2.32 anchor **CONFIRMED**; `beta_normal`'s Duncan co-citation NOT-FOUND
  (UFC citation valid).
- Citation fixes worth a small follow-up edit: C1 (Table 1 → Table 3, several files),
  C2 (drop Kulhawy & Trautmann 1996 / restore Harr 1984 on in-situ rows), C4 (reword
  the N=4 sentence). None affect numerics or tests.

*Verified 2026-07-18 by text extraction (PyMuPDF) from the owner-library PDF; scratch
script at the session scratchpad (`extract_duncan.py`). No module code changed; no git
operations run.*
