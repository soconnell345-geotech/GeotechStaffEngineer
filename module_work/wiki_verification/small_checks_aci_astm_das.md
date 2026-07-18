# Wave-5 small checks: ACI 318 Ec, ASTM A615 rebar areas, Das anisotropic su

**Summary: 3/3 CONFIRMED.** (1) Ec correlation confirmed in-hand at ACI 318-08 §8.5.1 (psi form 57,000√f'c; 4700 MPa is the SI-edition coefficient — the "-19 §19.2.2.1(b)" clause number itself is still not-in-hand). (2) All 11 `_REBAR_AREAS` mm² entries match ASTM A615/A615M-12 Table 1 digit-for-digit. (3) slope_stability anisotropic su matches Das 3rd Ed Eq. (7.51) (Casagrande & Carrillo 1944) — the coded sin(2α) form is the exact algebraic reduction of SuH+(SuV−SuH)sin²i under φu=0 (i=α+45°), generalized with an independent su_dss anchor (documented).

---

## 1. ACI 318 concrete modulus Ec — VERDICT: CONFIRMED (in-hand: ACI 318-08 §8.5.1, psi form)

- **Source in hand:** `C:\Users\socon\OneDrive\Lib\02 Tech\ACI\318_08.pdf` — ACI 318-08 (inch-pound edition), 471 pp, text layer present.
- **Clause found:** **§8.5.1** (PDF p. 111), printed verbatim (code column):
  > "8.5.1 — Modulus of elasticity, Ec, for concrete shall be permitted to be taken as wc^1.5 · 33 · sqrt(fc') (in psi) for values of wc between 90 and 160 lb/ft3. For normalweight concrete, Ec shall be permitted to be taken as **57,000 sqrt(fc')**."
  (Commentary R8.5.1, same page: Ec = slope from zero stress to 0.45 fc'.)
- **Module claim** (`lateral_pile/composite_section.py`, `_ACI_EC_COEFF = 4700.0`, `aci_concrete_modulus`): Ec = 4700 sqrt(f'c) MPa (normalweight), cited "ACI 318-19 §19.2.2.1(b)", edition flagged unverified-in-hand.
- **Comparison:** In-hand 318-08 prints the inch-pound coefficient 57,000 (psi). Conversion: 57,000·sqrt(f'c[psi]) = 57,000·sqrt(145.038)·sqrt(f'c[MPa]) psi = 686,466·sqrt(f'c[MPa]) psi = **4733·sqrt(f'c[MPa]) MPa**, which the ACI SI editions (318M) round to the standard **4700**. The coded correlation is therefore the standard ACI normalweight simplified modulus, now anchored to an ACI edition in hand.
- **Record for the docstring:** in-hand anchor = **ACI 318-08 §8.5.1** (Ec = 57,000√f'c psi, normalweight; SI rendering 4700√f'c MPa per ACI 318M). The 318-19 clause number §19.2.2.1(b) remains not-verified-in-hand (no 318-19 PDF in the library); formula and provenance are confirmed.

---

## 2. ASTM A615 rebar nominal areas — VERDICT: CONFIRMED (11/11 exact)

- **Source in hand:** `C:\Users\socon\OneDrive\Lib\02 Tech\ASTM\A615.pdf` — designation line verbatim: "Designation: A615/A615M – 12" (AASHTO M 31). Table 1 "Deformed Bar Designation Numbers, Nominal Weights [Masses], Nominal Dimensions, and Deformation Requirements", PDF p. 2.
- **Module:** `bearing_capacity/concrete_design.py` `_REBAR_AREAS` (mm², lines 23-35).

| Bar No. [M] | Printed area, in.² [mm²] | Module (mm²) | Match |
|---|---|---|---|
| 3 [10]  | 0.11 [71]    | 71.0   | exact |
| 4 [13]  | 0.20 [129]   | 129.0  | exact |
| 5 [16]  | 0.31 [199]   | 199.0  | exact |
| 6 [19]  | 0.44 [284]   | 284.0  | exact |
| 7 [22]  | 0.60 [387]   | 387.0  | exact |
| 8 [25]  | 0.79 [510]   | 510.0  | exact |
| 9 [29]  | 1.00 [645]   | 645.0  | exact |
| 10 [32] | 1.27 [819]   | 819.0  | exact |
| 11 [36] | 1.56 [1006]  | 1006.0 | exact |
| 14 [43] | 2.25 [1452]  | 1452.0 | exact |
| 18 [57] | 4.00 [2581]  | 2581.0 | exact |

- All 11 entries match the printed SI [mm²] column digit-for-digit. (Module dict covers exactly the 11 standard A615 sizes; no extras, none missing.)

---

## 3. Das anisotropic undrained strength — VERDICT: CONFIRMED (exact reduction of Eq. 7.51; documented ADP generalization)

- **Source in hand:** `C:\Users\socon\OneDrive\Lib\02 Tech\_Books\Books\_From Garino's IPAD\Braja M. Das\Advanced Soil Mechanics\3rd Edition\Advanced Soil Mechanics.pdf`, §7.18 "Anisotropy in undrained shear strength", book pp. 433-435 (PDF pp. 460-462), text layer present.
- **Printed equation (verbatim, book p. 434, Eq. 7.51, Casagrande & Carrillo 1944):**
  > Su(i) = SuH + (SuV − SuH) sin² i
  Convention as printed: **i** = angle the specimen axis / major principal stress makes **with the horizontal**; Su(i=90°) ≡ **SuV** (axis vertical), Su(i=0°) ≡ **SuH** (axis horizontal); K = SuV/SuH ranges 0.75-2.0 in natural deposits (Eq. 7.52). (Eq. 7.53 is the alternative Richardson et al. 1975 vane form — not what the module codes.)
- **Module** (`slope_stability/geometry.py`, `_anisotropic_su`, lines 245-281): ADP model,
  - α ∈ [0°, 45°): su = su_dss + (su_active − su_dss)·sin(2α)
  - α ∈ (−45°, 0]: su = su_dss + (su_passive − su_dss)·sin(2|α|)
  - held constant beyond ±45°; α = slice-base (failure-plane) inclination from horizontal.
- **Symbol-for-symbol comparison:** the docstring cites exactly the printed C&C form "su = su_h + (su_v − su_h)·sin²(i), i = major-principal-stress inclination from horizontal" and reduces it with φu = 0 (failure plane at 45° to σ1, so i = α + 45°): sin²(α+45°) = (1 + sin 2α)/2, giving su(α) = (su_v+su_h)/2 + [(su_v−su_h)/2]·sin(2α). This is **identical** to the coded form when su_active = SuV (i = 90°, active/crest, σ1 vertical), su_passive = SuH (i = 0°, passive/toe) and su_dss = (su_active+su_passive)/2 — verified: at α = 0 both give (SuV+SuH)/2; at α = ±45° both give SuV / SuH; the sin(2α) coefficient matches term-for-term. The 2·sin²(α+45°) − 1 = sin(2α) identity stated in the docstring is correct.
- **Deliberate generalizations (documented in the docstring, not discrepancies):** (a) su_dss is an independent third anchor (strict C&C forces the DSS value to the SuV/SuH midpoint — the module recovers C&C exactly when su_dss is set to the midpoint); (b) su is clamped constant beyond ±45° (C&C's ellipse is periodic; the clamp confines the interpolation to the physically meaningful TC/TE range); (c) su_active/su_passive mapping to ±α is tied to the module's slope orientation, with the mirror-slope swap warning stated explicitly. Convention mapping (α = failure-plane angle vs printed i = σ1 inclination, offset 45° for φu=0) is stated correctly and prominently.
