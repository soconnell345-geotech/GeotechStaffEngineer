# pavement_design — AASHTO 1993 Pavement Structural Design

## Purpose

Complete flexible (asphalt) and rigid (PCC) pavement structural design per
the **AASHTO Guide for Design of Pavement Structures (1993)** — the
authoritative empirical basis for US highway pavement design. This module
is an *orchestrator*: every equation, table, and chart value comes from the
digitized, worked-example-verified reference layer
`geotech_references.aashto_1993` (equations/tables, plus the follow-on
`lef` closed-form Appendix D module and the `composite_k` Section 3.2
module). Nothing is re-derived here; each result carries the printed-page
provenance strings of every reference call it made. Three exceptions are
read off the printed pages in this module because the reference layer's
reads departed from the print (GM-298, `module_work/gm298/CHECK.md`): the
Figure 2.8 and 2.9 a2 scales (`flexible.py`) and the Figure 3.3 composite
k chart (`composite_k.py`).

Direct import of `geotech_references` by a calc module follows the
`settlement/calc_steps.py` precedent (the reference package is a shared
library, exempt from the no-cross-module-imports rule).

## UNITS — US customary (documented exception)

psi, pci, inches, feet, kips, 18-kip ESALs. The 1993 Guide is US-customary
native and the structural number SN is tied to inch-based layer
coefficients (do not convert it). Same precedent as
`geotech_references.aashto_1993` and GEC-12. All parameter names carry unit
suffixes (`mr_psi`, `thickness_in`, `load_kips`).

## Files

| File | Contents |
|------|----------|
| `traffic.py` | `growth_factor`, `compute_design_esals` (axle spectrum / truck factors / direct, growth + DD + DL) |
| `performance.py` | environmental (swelling/frost) dPSI resolution + `estimate_performance_period` (Table 3.1 iteration) |
| `flexible.py` | `PavementLayer`, `design_flexible_pavement` (Figure 3.1 solve + Figure 3.2 layered split) |
| `rigid.py` | `design_rigid_pavement` (Figure 3.7 solve; direct / MR-19.4 / composite k) |
| `composite_k.py` | Figure 3.3 read off the printed chart (`composite_k_inf`) and the Table 3.2 worksheet (`effective_k`) |
| `ufc.py` | UFC 3-250-01 flexible, rigid, mixed traffic, and the AASHTO/UFC comparison |
| `ufc_e1.py` | UFC Figure E-1 read off the printed chart (`figure_e1_thickness`) |
| `ufc_f1.py` | UFC Figure F-1 read off the printed chart (`figure_f1_thickness`) |
| `common.py` | input resolution helpers (ZR, So, ΔPSI, effective MR, round-up) |
| `results.py` | `DesignTrafficResult`, `FlexiblePavementResult`, `RigidPavementResult` |

## Design procedures

### Flexible (Figures 3.1 / 3.2, Sections 2.3 / 3.1)

1. ZR from reliability (Table 4.1); So (default 0.45, the guide's total
   standard deviation with traffic for flexible pavements, printed
   II-9/10; range 0.40–0.50, I-62); ΔPSI = po − pt (po 4.2).
2. Effective roadbed MR: direct, or the Figure 2.3/2.4 seasonal
   relative-damage average (`uf = 1.18e8·MR^-2.32`).
3. Layer coefficients: a1 from EAC (Figure 2.5 chart read), a2/a3 granular
   from the printed regressions, cement-/bituminous-treated read straight
   across the Figure 2.8 / 2.9 parallel scales (read off the print at each
   labelled tick, GM-298: UCS 200–1,000 psi → a2 0.126–0.25; Marshall
   200–1,800 lb → 0.126–0.32; below the lowest tick is refused, above the
   top tick holds the top value with a warning). Direct `a` override always
   available.
4. Drainage m (Table 2.4) applies to UNBOUND layers only; when given as
   quality + %-saturation-time, the **midpoint of the printed range** is
   used and echoed (selection policy for every range-valued table in this
   module).
5. Required SN over each successive foundation (Figure 3.2): re-solve the
   Figure 3.1 equation using the base modulus, subbase modulus, then the
   roadbed MR — so design mode requires `modulus_psi` on granular layers
   below the surface. Over a base or subbase stiffer than 40,000 psi no SN
   is solved: the guide says the procedure "should not be applied" there
   and the layers above are set by "cost effectiveness and minimum
   practical thickness" (Section 3.1.5, printed II-37). The layer above
   takes the larger of `min_thickness_in` and the Section 3.1.4 minimum
   (error if neither exists). A treated base with no `modulus_psi` counts
   as stiffer than 40,000 psi (every modulus on Figures 2.8/2.9 is).
6. Thickness split is **round-as-you-go top-down**: size D1 from SN(over
   base)/a1, round UP to the increment (default 0.5 in), carry the rounded
   surplus down, size D2 from the remaining need, etc. Section 3.1.4
   traffic-based minimum AC/base thicknesses are enforced (design mode).
7. Forward check: `flexible_w18_from_sn` of the as-built SN must carry the
   design W18. `adequate` requires SN ≥ SN_required, capacity ≥ W18, and
   the Figure 3.2 note-2 criteria over each solved foundation (SN1* ≥ SN1,
   SN1* + SN2* ≥ SN2; `sn_stack[i]['sn_provided_above']`, `['adequate']`).

Check mode (every layer has `thickness_in`): computes provided SN + the
same forward check, and the layered criteria over each layer whose
modulus (≤ 40,000 psi) is given; changes nothing.

### Environmental serviceability loss (Section 2.1.4 / 3.1.3, Appendix G)

Both design functions accept `swelling={vr_in, ps_pct, theta}` (the
PRINTED Figure G.4 equation ΔPSI_SW = 0.00335·VR·Ps·(1−e^(−θt))) and
`frost={phi_mm_day, pf_pct, delta_psi_max}` (the PRINTED Figure G.8
equation ΔPSI_FH = 0.01·PF·ΔPSI_MAX·(1−e^(−0.02φt))), evaluated at
`design_period_yr` and subtracted from the design ΔPSI before the
structural solve (Table 3.1 Step 4). Errors if the environmental loss
consumes the whole budget; warns above 50%. VR, θ, φ and ΔPSI_MAX are the
caller's inputs; the module computes none of them. The reference layer
has functions for θ (Fig G.2), VR (Fig G.3), φ (Fig G.6) and ΔPSI_MAX
(Fig G.7), reachable through the reference agent, not called here. Its
Fig G.3 VR read is wrong against the print (GM-298: the printed example
is PI 50, average conditions, 5-ft layer → VR 0.83 in; the reference
overstates VR up to about 4 times; REF candidate).
`estimate_performance_period` runs the full printed Table 3.1 iteration
on a DESIGNED section (fixed SN or D), converting allowable traffic to
calendar years via the compound-growth inverse; reproduces the guide's
printed 3-row example (po=4.4 case) within its own rounding.

Sections supported: 1–3 layers (full-depth AC / AC + base / AC + base +
subbase). Top layer must be `asphalt`; a subbase requires a base above it.

### Rigid (Figure 3.7, Section 3.2)

1. ZR / So (default 0.35, the guide's total standard deviation with
   traffic for rigid pavements, printed II-9/10; range 0.30–0.40) / ΔPSI
   (po 4.5); pt also enters the strength-term exponent.
2. J from Table 2.6 (pavement type × shoulder × dowels; midpoint) or
   direct. Cd from Table 2.5 (midpoint) or direct (default 1.0).
3. Effective k — exactly one of:
   - `k_pci` direct;
   - `mr_psi` → simplified k = MR/19.4 (slab directly on roadbed only);
   - `composite_k` spec dict → the full Section 3.2 worksheet
     (`composite_k.effective_k`, same keys as the reference layer's
     `effective_modulus_subgrade_reaction`): seasonal composite k from
     subbase properties (Fig 3.3, read off the printed chart in this
     module, Dsb 6–18 in, Esb 15,000–1,000,000 psi, MR 1,000–20,000 psi),
     rigid-foundation correction (Fig 3.4), relative damage u_r (Fig 3.5),
     loss-of-support (Fig 3.6 / Table 2.7), Figures 3.4–3.6 from the
     reference layer. Warns when Esb/MR > 4 (the guide's limit for an
     unbound subbase, printed II-37/44). Because u_r depends on the slab
     thickness, k and D are **fixed-point iterated** (converges when the
     rounded D changes < 0.25 in; `iterations` reported).
4. Solve D by bisection (`rigid_d_from_w18`), round UP to 0.5 in, forward
   check. The guide: "generally, the layer thickness is rounded to the
   nearest inch, but the use of controlled grade slip form pavers may
   permit 1/2-inch increments" (Section 3.2.2, printed II-44); give
   `thickness_increment_in=1.0` for whole inches.

## UFC 3-250-01 alternative method (`ufc.py`)

The DoD roads/parking procedure as a first-class alternative to AASHTO.
Figures E-1 and F-1 are read off the print in this module (GM-298); the
tables, Eq 13-1, the frost tables and the other Appendix E vehicle curves
come from `geotech_references.ufc_pavement` (Appendix-G-anchored). The
reference layer's E-1 and F-1 are exact at their Appendix G anchors and
wrong between them (E-1 up to 1.7 in under the curves at CBR 5–20; F-1
barely moves with flexural strength): REF candidates.

- `design_flexible_pavement_ufc` — Corps CBR method: Figure E-1 cover
  curves (`ufc_e1`, every Appendix G read within 0.1 in) applied at each
  interface CBR, Table 7-2 surface/base minimums, interfaces past the end
  of the curves handled (minimums govern, noted); a subbase is at least
  4 in (p. 20; Appendix G-3 ends 3.5 / 4 / 4 in, reproduced). Optional
  Ch 19 seasonal frost via the reduced-subgrade-strength method (Table
  19-2 frost group → Table 19-3 frost support index replaces the subgrade
  CBR; the governing section is reported).
- `design_rigid_pavement_ufc` — Figure F-1 (`ufc_f1`, worked as para 13-3
  prints it: R across to k, down to the pass level, across to thickness;
  G-7, G-8.11 and G-8.12 reproduced) + the Eq 13-1 stabilized-foundation
  reduction; k direct or from Table 10-1 (k above 500 taken as 500, Table
  10-1 note 2); slab rounded up to 0.5 in and to the 6 in minimum of para
  13-3.
- `compare_flexible_pavement_methods` — runs BOTH guides on one 18-kip
  single-axle pass count (identical physical count: AASHTO LEF(18k)=1.0
  by definition; E-1 is natively that axle). Differences are expected —
  the damage models differ, and reliability exists only on the AASHTO
  side — and every cross-guide assumption (FHWA Mr=2555·CBR^0.64
  correlation for the subgrade, defaults) is stated in `notes`. The
  AASHTO base and subbase moduli come from their CBRs read straight
  across the guide's own Figure 2.6 (base, II-19) and Figure 2.7
  (subbase, II-21) scales, which hold the Road Test pairs (base CBR 100
  with 30,000 psi, subbase CBR 30 with 15,000 psi, II-20). The FHWA
  correlation on a base would give CBR 80 → 42,205 psi and a2 0.175,
  where Figure 2.6 reads about 0.13 (GM-298). UFC functions return dicts
  (reference-layer style); the AASHTO side keeps its dataclasses.

## Validation ledger

| Anchor | Source | Result |
|--------|--------|--------|
| Flexible SN | Guide Figure 3.1 printed example (W18=5e6, R=95%, So=0.35, MR=5000, ΔPSI=1.9 → SN=5.0) | orchestrator returns SN 4.95–5.0 ✔ |
| Rigid D | Guide Figure 3.7 printed example (W18=5.1e6, k=72, Ec=5e6, Sc'=650, J=3.2, Cd=1.0, So=0.29, ΔPSI=1.7 → D=10.0) | orchestrator returns D 9.7–10.0 ✔ |
| Effective MR | Guide Figure 2.4 12-month example → 5000 psi | reproduced through `monthly_mr_psi` ✔ |
| Layer coefficients | printed checks a2(30000)=0.14, a3(15000)=0.11; a1(400k)=0.42 chart | reproduced ✔ |
| Growth factor | Table D.20 (D-24): nine printed cells, 0–10 %, 5–35 yr (4 %/20 yr → 29.78) | every cell within 0.006 (V-075) |
| Truck-factor traffic | Table D.21 (D-25) worksheet, 13 vehicle classes, 2 %/20 yr | per-row ESALs exact; column sums 43,797,514 where the total line prints 43,772,314 (a 25,200 slip in the print; text D-22 says 43.8 million) (V-075) |
| Appendix H flexible example | H-1…H-7: traffic 18.6 million, Fig H.3 effective MR 5,700, Table H.2 performance period 13.2 yr, SN 3.2/4.5/5.6, D 7.6/6.8/11.04 | 18,598,900; MR 5,695; 13.21 yr; SN 3.18/4.53/5.60; D 8/7/12 (the guide rounds 11.04 to 11; the module rounds up) (V-076) |
| Base / subbase from CBR (comparison) | Figures 2.6 / 2.7 straight across; Road Test pairs base CBR 100 ↔ 30,000 psi, subbase CBR 30 ↔ 15,000 psi (II-20) | 29,800 / 14,800 psi (GM-298) |
| LEF identity | 18-kip single axle → LEF 1.000 at any SN/pt (definition) | reproduced (table and closed-form paths) ✔ |
| Composite-k worksheet | Guide §3.2 Table 3.3 printed example (avg u_r 0.60, eff. k 540 pci, LS-corrected 170 pci) | module worksheet with its own Fig 3.3 read: avg u_r 0.613, k 522 (−3.3%), LS-corrected 165 (−2.7%) (V-074) |
| Fig 3.3 read | printed example (Esb 20,000, MR 7,000, Dsb 6 → 400) and Table 3.3 rows (1,100 / 160 / 230 / 410) | 391 / 1,085 / 154 / 230 / 391, within −4.7…+0.2% (V-074) |
| Fig 2.8 / 2.9 | the printed scales: a2 0.22 at UCS 800 psi (the dashed line); 0.25 at 1,000 psi; Marshall 1,600 lb → 0.30 | read at every labelled tick (GM-298) |
| UFC Fig E-1 | Appendix G-1 (CBR 3, 1,000,000 → 16.4 in; CBR 4, 1,395,400 → 14 in) and G-3 (5,000,000: CBR 10 / 7 / 4 / 25 → 7.8 / 9.8 / 14.5 / 3.4 in) | 16.4, 13.9; 7.8, 9.8, 14.5, 3.4 (GM-298). G-8.10's 16 in at CBR 3.5, 1,200,000 is 15.1 on the curves |
| UFC G-3 section | 5,000,000 passes, base CBR 80, subbase 25, subgrade 10 → 3.5 in AC, 4 in base, 4 in subbase | reproduced (GM-298) |
| UFC Fig F-1 | G-7 (650 psi, k 100, 20,000,000 → 8.1 in), G-8.11 (k 325, 1,200,000 → 6.3), G-8.12 (k 50, 1,200,000 → 8 in slab) | 8.1, 6.3, 7.8 → 8.0 (GM-298) |
| Appendix D LEF tables | FULL Tables D.1–D.18 transcribed (~5,850 cells; the derivation equations live in the 1986 edition's separate Volume 2 App. MM — the 1993 Guide is a single volume and only carries the citation); lead visual QC of 4 tables (~130 cells) zero discrepancies; exact match to the 4 pre-existing independent curves | 143 tests in `geotech-references/tests/test_aashto_1993_lef.py` |
| Swelling loss (Fig G.4) | printed example t=15, θ=0.10, Ps=60%, VR=2 in → 0.3 | 0.312 through the design function ✔ (printed equation, lead page-verified) |
| Frost heave loss (Fig G.8) | printed example t=15, φ=5, PF=30%, ΔPSI_MAX=2.0 → 0.47 | 0.466 ✔ (printed equation incl. the 0.02 constant, lead page-verified) |
| Table 3.1 iteration | printed 3-row example (po=4.4, ΔPSI=1.9, trial 13.0 → 9.7 → 8.5 yr) | rows reproduced within the guide's rounding (63 tests in `geotech-references/tests/test_aashto_1993_environmental.py`) |

Composite-k chart-read confidence: Fig 3.3 is read off the printed chart
in `composite_k.py` (every curve at every Dsb gridline; within 5 % of the
five printed anchors, all at Dsb 6 in). The reference layer's Fig 3.3 fit
departs from the chart away from the Table 3.3 rows (at Dsb 6, Esb
400,000, MR 5,000 it gives 3,026 pci, the chart about 490; its Dsb
scaling is too flat): REF candidate. Fig 3.4 (reference layer) is exact
at Dsg = 5 ft (4 printed Table 3.3 triples) with chart-read depth scaling
(±15–20 %, not re-read in GM-298). Fig 3.5's closed form matches a pixel
read within about ±2 % for D 6–14 in, k 15–1,950 pci; Fig 3.6's slopes
within about ±8 % for k ≥ 30 pci (GM-298).

## Scope limits (out of scope, by design)

- **Overlays / rehabilitation (Part III)** — text-only in the reference
  layer; not computed.
- **Rigid joint / reinforcement / prestressed design** (Section 3.3+) —
  not computed.
- **Low-volume road catalog** (Part II Ch 4) — only the aggregate-loss
  equations exist (reference layer).
- ESAL fallback path (closed-form `lef` module absent) is limited to the
  digitized SN=5 / D=9-in, pt=2.5 tables, no triple axles — the module
  says so in notes/errors.

The reference agent (`consult_references` → `aashto_1993`) remains the
path for guide text, figures, and anything not computed here.

## Conventions & gotchas

- Range-valued tables (m, Cd, J, DL) resolve to the **midpoint** unless the
  caller passes a value; every default/midpoint decision is echoed in
  `notes`.
- Thicknesses always round **UP** (never down) to `thickness_increment_in`.
- `delta_psi` bounds differ by type: flexible (0, 2.7), rigid (0, 3.0) —
  enforced in the reference equations.
- `chart_read` provenance (a1, treated-base a2, composite-k charts) is
  surfaced as warnings so reports can flag read-off tolerance.
- Check mode vs design mode is inferred from `thickness_in` /
  `slab_thickness_in`; mixed layer thickness specs are rejected.
