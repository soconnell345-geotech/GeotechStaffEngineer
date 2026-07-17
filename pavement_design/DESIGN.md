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
provenance strings of every reference call it made.

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
| `common.py` | input resolution helpers (ZR, So, ΔPSI, effective MR, round-up) |
| `results.py` | `DesignTrafficResult`, `FlexiblePavementResult`, `RigidPavementResult` |

## Design procedures

### Flexible (Figures 3.1 / 3.2, Sections 2.3 / 3.1)

1. ZR from reliability (Table 4.1); So (default 0.45 = midpoint of the
   0.40–0.50 flexible range); ΔPSI = po − pt (po 4.2).
2. Effective roadbed MR: direct, or the Figure 2.3/2.4 seasonal
   relative-damage average (`uf = 1.18e8·MR^-2.32`).
3. Layer coefficients: a1 from EAC (Figure 2.5 chart read), a2/a3 granular
   from the printed regressions, cement-/bituminous-treated from the
   sparse Figure 2.8/2.9 chart reads (flagged in warnings). Direct `a`
   override always available.
4. Drainage m (Table 2.4) applies to UNBOUND layers only; when given as
   quality + %-saturation-time, the **midpoint of the printed range** is
   used and echoed (selection policy for every range-valued table in this
   module).
5. Required SN over each successive foundation (Figure 3.2): re-solve the
   Figure 3.1 equation using the base modulus, subbase modulus, then the
   roadbed MR — so design mode requires `modulus_psi` on every non-surface
   layer.
6. Thickness split is **round-as-you-go top-down**: size D1 from SN(over
   base)/a1, round UP to the increment (default 0.5 in), carry the rounded
   surplus down, size D2 from the remaining need, etc. Section 3.1.4
   traffic-based minimum AC/base thicknesses are enforced (design mode).
7. Forward check: `flexible_w18_from_sn` of the as-built SN must carry the
   design W18. `adequate` requires both SN ≥ SN_required and capacity ≥
   W18.

Check mode (every layer has `thickness_in`): computes provided SN + the
same forward check, changes nothing.

### Environmental serviceability loss (Section 2.1.4 / 3.1.3, Appendix G)

Both design functions accept `swelling={vr_in, ps_pct, theta}` (the
PRINTED Figure G.4 equation ΔPSI_SW = 0.00335·VR·Ps·(1−e^(−θt))) and
`frost={phi_mm_day, pf_pct, delta_psi_max}` (the PRINTED Figure G.8
equation ΔPSI_FH = 0.01·PF·ΔPSI_MAX·(1−e^(−0.02φt))), evaluated at
`design_period_yr` and subtracted from the design ΔPSI before the
structural solve (Table 3.1 Step 4). Errors if the environmental loss
consumes the whole budget; warns above 50%. Supporting inputs come from
the reference layer: θ (Fig G.2 nomograph, exact reconstruction), VR
(Fig G.3, coarse chart read ±20–40%), φ (Fig G.6 chart read),
ΔPSI_MAX (Fig G.7, exact linear slopes by drainage quality).
`estimate_performance_period` runs the full printed Table 3.1 iteration
on a DESIGNED section (fixed SN or D), converting allowable traffic to
calendar years via the compound-growth inverse; reproduces the guide's
printed 3-row example (po=4.4 case) within its own rounding.

Sections supported: 1–3 layers (full-depth AC / AC + base / AC + base +
subbase). Top layer must be `asphalt`; a subbase requires a base above it.

### Rigid (Figure 3.7, Section 3.2)

1. ZR / So (default 0.35, rigid range 0.30–0.40) / ΔPSI (po 4.5); pt also
   enters the strength-term exponent.
2. J from Table 2.6 (pavement type × shoulder × dowels; midpoint) or
   direct. Cd from Table 2.5 (midpoint) or direct (default 1.0).
3. Effective k — exactly one of:
   - `k_pci` direct;
   - `mr_psi` → simplified k = MR/19.4 (slab directly on roadbed only);
   - `composite_k` spec dict → the full Section 3.2 worksheet
     (`geotech_references.aashto_1993.composite_k.
     effective_modulus_subgrade_reaction`): seasonal composite k from
     subbase properties (Fig 3.3), rigid-foundation correction (Fig 3.4),
     relative damage u_r (Fig 3.5), loss-of-support (Fig 3.6 / Table 2.7).
     Because u_r depends on the slab thickness, k and D are **fixed-point
     iterated** (converges when the rounded D changes < 0.25 in;
     `iterations` reported).
4. Solve D by bisection (`rigid_d_from_w18`), round UP to 0.5 in, forward
   check.

## UFC 3-250-01 alternative method (`ufc.py`)

The DoD roads/parking procedure as a first-class alternative to AASHTO,
built on the rebuilt `geotech_references.ufc_pavement` (Appendix-G-anchored):

- `design_flexible_pavement_ufc` — Corps CBR method: Figure E-1 cover
  curves (7 exact Appendix G anchors, ±10% elsewhere) applied at each
  interface CBR, Table 7-2 surface/base minimums, off-curve high-CBR
  interfaces handled (minimums govern, noted). Optional Ch 19 seasonal
  frost via the reduced-subgrade-strength method (Table 19-2 frost group →
  Table 19-3 frost support index replaces the subgrade CBR; the governing
  section is reported).
- `design_rigid_pavement_ufc` — Figure F-1 (flexural strength × k ×
  18-kip passes) + the Eq 13-1 stabilized-foundation reduction; k direct
  or from Table 10-1.
- `compare_flexible_pavement_methods` — runs BOTH guides on one 18-kip
  single-axle pass count (identical physical count: AASHTO LEF(18k)=1.0
  by definition; E-1 is natively that axle). Differences are expected —
  the damage models differ, and reliability exists only on the AASHTO
  side — and every cross-guide assumption (FHWA Mr=2555·CBR^0.64
  correlation, defaults) is stated in `notes`. UFC functions return dicts
  (reference-layer style); the AASHTO side keeps its dataclasses.

## Validation ledger

| Anchor | Source | Result |
|--------|--------|--------|
| Flexible SN | Guide Figure 3.1 printed example (W18=5e6, R=95%, So=0.35, MR=5000, ΔPSI=1.9 → SN=5.0) | orchestrator returns SN 4.95–5.0 ✔ |
| Rigid D | Guide Figure 3.7 printed example (W18=5.1e6, k=72, Ec=5e6, Sc'=650, J=3.2, Cd=1.0, So=0.29, ΔPSI=1.7 → D=10.0) | orchestrator returns D 9.7–10.0 ✔ |
| Effective MR | Guide Figure 2.4 12-month example → 5000 psi | reproduced through `monthly_mr_psi` ✔ |
| Layer coefficients | printed checks a2(30000)=0.14, a3(15000)=0.11; a1(400k)=0.42 chart | reproduced ✔ |
| Growth factor | 4%/yr, 20 yr → GF 29.78 (standard value) | reproduced ✔ |
| LEF identity | 18-kip single axle → LEF 1.000 at any SN/pt (definition) | reproduced (table and closed-form paths) ✔ |
| Composite-k worksheet | Guide §3.2 Table 3.3 printed example (avg u_r 0.6042, eff. k 540 pci, LS-corrected 170 pci) | reproduced +1.2% / −2.7% / −2.2% (71 tests in `geotech-references/tests/test_aashto_1993_composite_k.py`); Fig 3.3 printed example 400 pci → 390 (−2.5%) |
| Appendix D LEF tables | FULL Tables D.1–D.18 transcribed (~5,850 cells; the derivation equations live in the 1986 edition's separate Volume 2 App. MM — the 1993 Guide is a single volume and only carries the citation); lead visual QC of 4 tables (~130 cells) zero discrepancies; exact match to the 4 pre-existing independent curves | 143 tests in `geotech-references/tests/test_aashto_1993_lef.py` |
| Swelling loss (Fig G.4) | printed example t=15, θ=0.10, Ps=60%, VR=2 in → 0.3 | 0.312 through the design function ✔ (printed equation, lead page-verified) |
| Frost heave loss (Fig G.8) | printed example t=15, φ=5, PF=30%, ΔPSI_MAX=2.0 → 0.47 | 0.466 ✔ (printed equation incl. the 0.02 constant, lead page-verified) |
| Table 3.1 iteration | printed 3-row example (po=4.4, ΔPSI=1.9, trial 13.0 → 9.7 → 8.5 yr) | rows reproduced within the guide's rounding (63 tests in `geotech-references/tests/test_aashto_1993_environmental.py`) |

Composite-k chart-read confidence (documented per function): Fig 3.4 is
EXACT at Dsg=5 ft (4 printed Table 3.3 triples) with chart-read depth
scaling (±15–20%); Fig 3.3's subbase-thickness scaling away from
DSB=6 in is a 3-point chart read (±20–25%, the weakest piece); Fig 3.6
LS curves are multi-point reads (±7–16%); Fig 3.5 is a literature
closed form validated on the printed example (0.6%) + 6 chart reads.
Results carry `chart_read` flags and these tolerances in docstrings.

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
