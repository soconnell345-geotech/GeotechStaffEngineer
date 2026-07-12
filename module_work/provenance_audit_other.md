# Source-Provenance Audit — non-slope modules

**Purpose.** For the 5.5.0 quality pass and the owner's upcoming private
reference-wiki integration. Two goals: (a) make explicit, where a future
engineer would see it, every implementation that was built WITHOUT the original
source in hand or from a risky PDF/vision/figure read; (b) build the input map
(WIKI-WISHLIST) of specific documents/editions that would upgrade a module's
provenance tier.

**Scope (this file).** The non-slope custom modules. `slope_stability` and
`validation_examples` are audited separately (slope-qc). Covered here: fem2d,
reliability, lateral_pile (+ composite_section), soe, retaining_walls,
drilled_shaft, axial_pile, wave_equation, downdrag, seismic_geotech, settlement,
bearing_capacity, ground_improvement, pile_group, calc_package, pdf_import,
drawing_ir, geotech_common.

**This audit changed NO analysis code.** It reads provenance and, where a
method's basis was undocumented, adds additive doc-only notes (DESIGN.md /
docstring). Doc fixes are listed per module and in the change log at the end.

## Classification legend

**SOURCE TIER** — how close the implementation is to the primary source:
- `original-in-hand` — coded from the primary document (equations/tables read
  directly from it).
- `authoritative-secondary` — coded from a faithful secondary (a standard
  textbook / FHWA GEC restating the method), primary not directly consulted.
- `memory+anchored` — built from recalled formulation BUT pinned to a
  reproduced published result (worked example / benchmark), so errors surface.
- `memory-only` — recalled formulation with NO reproduced published anchor.
  **RED FLAG.**

**EXTRACTION RISK** — how numbers entered the code:
- `text` — closed-form equations / values typed from text (low risk).
- `vision-raster` — coefficients/curves read off a raster figure or scanned
  table (digitization risk; a wrong read is silent). **List each item.**
- `OCR` — machine-OCR of a scanned table (risk of transcription error).
- `hand` — hand-tabulated by the author from a chart.

**ANCHOR STATUS** — what the result is validated against:
- `worked-example` — the source's OWN worked example reproduced.
- `secondary` — a different textbook/tool's example.
- `limiting-behavior` — only limiting/sanity checks (e.g. phi->0, symmetry).
- `none` — no reproduced numeric anchor.

**DOCUMENTED?** — is the provenance stated where an engineer would look
(DESIGN.md / VALIDATION.md / docstring)? `yes (loc)` / `FIXED (this audit)` /
`GAP`.

---

<!-- Per-module tables populated below from the evidence sweep. -->

## Foundations & capacity

| Item | Source tier | Extraction risk | Anchor | Documented? |
|---|---|---|---|---|
| bearing_capacity — Nc/Nq/Nγ + shape/depth/incl/base/ground factors | authoritative-secondary (Vesic 1973/Meyerhof/Hansen/GEC-6) | **text — all closed-form Python expressions, NOT table lookups** | **worked-example** — factor tables match GEC-6 Table 6-1 / Das Table 3.4 within 1-2 %; textbook qult examples only loosely bounded | yes (strong); minor: `sg` floor 0.6 & `_REBAR_AREAS` uncited |
| bearing_capacity — `_REBAR_AREAS` (concrete_design) | representative (standard ASTM A615 bar areas) | transcribed standard values | none | **GAP → note** (ASTM A615) |
| settlement — Schmertmann Iz, C1/C2, Boussinesq/Newmark/Westergaard, Tv-U | authoritative-secondary (Schmertmann 1978, Boussinesq, Terzaghi, GEC-6) | text — closed-form (Iz from triangular geometry; no chart lookup) | **worked-example** — GEC-6 Ex B-1 reproduced (~21 mm, 41 mm) | yes; **calc-pkg display bug FIXED** (Iw 1.0→actual 1.122) |
| axial_pile — Nordlund Kδ/CF/Nq'/αt/q_L charts; Tomlinson α; δ/φ Table 7-1 | authoritative-secondary (Nordlund 1963/79; GEC-12) | **vision-raster — the chart-heaviest module**: Kδ (Fig 7-5), CF (7-8), Nq' (7-14), αt (7-13), **q_L arrays (Fig 7-15)**, Tomlinson α (Fig 7-17); source figs all cited | **none** — DESIGN.md states plainly "no numeric GEC-12 worked example available; flagged for future tie-out"; tests re-assert the transcribed values | yes — **the chart-fit disclosure (`nordlund.py` warning block + "Known Simplifications") is the HOUSE MODEL for honesty** |
| axial_pile / downdrag — Nt(φ) piecewise (GEC-12 Table 7-9), **duplicated** | authoritative-secondary | transcribed interpolation | **verified-CONSISTENT (not identical)** — see note | **FIXED** — consistency pin tests added in both modules |
| drilled_shaft — GEC-10 α (Fig 13-5) + rational α/CIUC chains; fs cap 200 kPa; RQD multiplier schedule; Nc | authoritative-secondary (GEC-10; O'Neill & Reese 1999) | mixed — α linearization digitized (Fig 13-5/10-6); **fs cap 200 kPa uncited**; **RQD schedule no table#** | **worked-example** — GEC-10 App A Steps 11.5-11.6 reproduced (rational chains, ≤7 %) | yes; **calc-pkg display bug FIXED** (Nc formula); dual GEC-10 editions (10-016 vs 18-024) |
| pile_group — rigid-cap 6-DOF statics; Converse-Labarre; p-multipliers (GEC-12 Table 9-2) | original-in-hand-equation (statics) / authoritative-secondary (tables) | text (statics closed-form); **p-multiplier table transcribed** (Table 9-2, no per-row test) | **worked-example** — GEC-12 Vol3 Table D-23 (~5 %); Converse-Labarre textbook | partial — 6-DOF engine's real basis (CPGA ITL-89-4, EM 1110-2-2906) is in code but **absent from DESIGN.md refs** |
| downdrag — neutral-plane force-equilibrium solve; UFC 3-220-20 eqs; β=0.3 default | authoritative-secondary (Fellenius; UFC 3-220-20) | text — neutral plane SOLVED, not charted | **none** — all internal hand-calc/consistency checks | yes (UFC eq-by-eq map); β=0.3 default uncited |

**Notes.**
- **Table 7-9 verification (in-house).** GEC-12 Table 7-9 in the in-house
  reference (`geotech_references/gec_12/tables.py`) gives Nt as **ranges by soil
  type** (clay 3-30, silt 20-40, sand 30-150, gravel 60-300). The code's
  `Nt_from_phi` / `_Nt_from_phi` is a piecewise Nt(φ) **interpolation**, NOT a
  literal transcription — but its values sit **within** those published ranges
  in the overlapping φ bands (verified 2026-07-12; above the sand max φ it
  enters gravel territory ≤300). Per the "if different, don't change" rule the
  code was **NOT modified**; instead a consistency pin test now guards each copy.
- **Positives.** bearing_capacity factor tables match GEC-6/Das to 1-2 %;
  settlement, drilled_shaft (rational chains) and pile_group have real published
  worked-example anchors; **axial_pile's chart-fit disclosure is the model** the
  other chart-digitized modules (Reese p-y, Caquot-Kerisel) should copy.
- **Two calc-package DISPLAY BUGS fixed** (code contradicted its own displayed
  formula): settlement rendered `I_w=1.0` while using Schleicher 1.122;
  drilled_shaft displayed & recomputed the superseded `Nc=min(6+L/D,9)` instead
  of the current `min(6(1+0.2·L/D),9)`. Both pinned by regression tests.
- **Deferred in-house verifications** (honest): GEC-12 Table 7-1 δ/φ &
  Table 9-2 p-multipliers, GEC-10 Fig 13-5 + the 200 kPa fs-cap origin, and
  GEC-6 Fig 5-19/Table 6-1 were NOT cross-read this pass — carried on the
  wiki-wishlist for a follow-up in-house lookup.

## Lateral & dynamic piles

| Item | Source tier | Extraction risk | Anchor | Documented? |
|---|---|---|---|---|
| lateral_pile — Reese (1974) sand p-y A/B coeff. charts (`_REESE_A_*`/`_REESE_B_*`, py_curves.py) | authoritative-secondary (Reese & Van Impe 2001 / COM624P restate Reese 1974) | **vision-raster** — A & B curves digitized off Figs 3.30/3.31 (also C1–C3, the k-table, stiff-clay As/Ac) | memory+anchored — p-y regression tests + A≥B monotonicity + V-017; ordinates NOT directly anchored to the chart | **FIXED** — Source-basis note added; DESIGN.md A_c stale 0.53→0.55 |
| lateral_pile — Matlock (1970) soft-clay p-y, API sand | authoritative-secondary | text (closed-form) | worked-example / behavior | yes (DESIGN.md/docstrings) |
| composite_section — ACI 318 `Ec=4700√f'c` | authoritative-secondary (textbook-standard) | text | limiting/formula (uncracked upper bound; V-017 flip documented) | **FIXED** — clause ACI 318-19 §19.2.2.1(b) added, edition flagged unverified-in-hand |
| wave_equation — Smith (1960) spring-dashpot algebra | authoritative-secondary | text | drivability/bearing-graph behavior | yes |
| wave_equation — default quake/damping constants (`soil_model.py`) | authoritative-secondary (Smith/GRLWEAP defaults, GEC-12 Table 12-3) | transcribed (user-overridable inputs) | behavior only | **FIXED** — Source-basis note added |
| seismic_geotech — site classification (Vs30/N/su boundaries + site factors) | authoritative-secondary (AASHTO LRFD 9th Ed §3.10.3 / NEHRP FEMA P-1050) | text (boundaries) + transcribed site-factor tables (Fpga/Fa/Fv) | class-boundary sanity | **FIXED** — DESIGN.md "ASCE 7-22" corrected to the implemented AASHTO/NEHRP standard |
| seismic_geotech — Mononobe-Okabe | original-in-hand-equation (AASHTO §11.6.5 closed-form) | text | **internal numerical trial-wedge cross-check only — NOT a published K_AE table** | **FIXED** — anchor gap noted honestly |
| seismic_geotech — liquefaction triggering (Youd et al. 2001 / NCEER) | authoritative-secondary | text + published curve-fits (NCEER CRR/MSF, Liao-Whitman rd) | good — NCEER-vs-B&I distinction is exemplary and explicit | yes (DESIGN.md is explicit it is NOT B&I-2014) |

**Notes.** The Reese A/B tables are the clearest elevated-risk item in this
cluster: digitized from charts, and the anchors verify curve SHAPE not the exact
ordinates. They were mis-digitized once (cyclic A dipped below B) and corrected
in the v5.3 review — a concrete demonstration of why chart-digitized tables need
wiki verification. The seismic liquefaction NCEER-vs-B&I documentation is a model
to emulate elsewhere.

## Earth retention & ground improvement

| Item | Source tier | Extraction risk | Anchor | Documented? |
|---|---|---|---|---|
| soe — Caquot-Kerisel log-spiral Kp (Kp0 base + R reduction grid; `earth_pressure.py`, byte-identical copy in `sheet_pile`) | authoritative-secondary (Caltrans T&S / NAVFAC DM-7.2 restating Caquot-Kerisel 1948) | **vision-raster** — Kp0 column + full R(δ/φ) grid read off charts; only **φ=30→Kp0=6.30 is a verified read** | worked-example for the one verified value (V-013 Ex 8-1); **none** for the other ~5 Kp0 + R grid | **FIXED** — Source-basis added; **RED FLAG** for the unverified Kp0/R values |
| soe — grouted-anchor bond-stress table (`_BOND_STRESS`, `anchor_design.py`) | authoritative-secondary (FHWA GEC-4 Table 4 presumptive nominal) | transcribed (nominal presumptive) | **none — UNBENCHMARKED** (no validation_examples anchor) | **FIXED** — moved out of "Future Work", provenance + UNBENCHMARKED RED FLAG stated |
| soe — FHWA apparent-earth-pressure envelopes | authoritative-secondary (Terzaghi-Peck / FHWA) | text | limiting/behavior | yes (docstrings) |
| retaining_walls — MSE Kr/K0 & F* pullout curves (`mse.py`) | authoritative-secondary (GEC-11 Vol I NHI-10-024 method; E4 in Vol II NHI-10-025) | **vision-raster** — Kr/F* curves digitized from Figs 4-11/E4-5 | **worked-example** — GEC-11 Example E4 reproduced (CDRs, F* within ~3%) | **FIXED** — two-volume note + Source-basis added |
| retaining_walls — built-in reinforcement product constants (W11 grid, 75x4 strip T_allow) | representative-of-class (NOT a catalog) | transcribed representative values | n/a (defaults, user should override) | **FIXED** — "representative, not a catalog" note added |
| ground_improvement — Priebe stone columns; Barron drain theory | authoritative-secondary (Priebe 1995; Barron 1948 closed-form) | text | limiting/behavior; **Priebe first-order disclaimer is exemplary** | yes (exemplary) |
| ground_improvement — vibro-compaction feasibility thresholds + probe spacing (`vibro.py`) | authoritative-secondary (Brown 1977 / GEC-13) for screening; **empirical rule-of-thumb** for spacing | transcribed thresholds; **unsourced-empirical** spacing endpoints | none (spacing not benchmarked) | **FIXED** — spacing annotated unsourced-empirical |

**Notes.** `ground_improvement` is the **lowest-risk** module in this half (mostly
closed-form + honest disclaimers). The **soe Caquot-Kerisel Kp0/R grid** and the
**UNBENCHMARKED anchor-design bond table** are the two genuine red flags here.

## Numerical, probabilistic, ingestion & common

| Item | Source tier | Extraction risk | Anchor | Documented? |
|---|---|---|---|---|
| fem2d — element formulations, SRM, seepage/consolidation | original-in-hand-equation (standard FE closed-form) | text (closed-form; no chart reads) | **worked-example** — Prandtl bearing ~2 %, Griffiths-Lane slope | yes (DESIGN.md/VALIDATION.md) |
| fem2d — `gamma_w=9.81`, `n_w=2.2e6` defaults | physical constants | text | n/a | **FIXED** — source-basis one-liners added |
| reliability — COV knowledge base (`reliability/`) | authoritative-secondary — **GOLD STANDARD**: per-row source cites (Duncan 2000 Table 1; TC304 Tables 1.2–1.4/3.1), provenance-locked VALIDATION rows | mixed, HONESTLY LABELLED: Duncan rows "verified against the published paper" (text); **TC304 rows "stored verbatim from the report PDF"** (transcription/OCR) | published examples / engine-agreement | yes — **model for the whole codebase**; TC304 transcription already flagged honestly |
| reliability — FOSM/Duncan, Rosenblueth PEM, FORM, Vanmarcke spatial averaging | authoritative-secondary (coded from the papers' equations) | text | worked-example / engine cross-check | yes |
| calc_package — embedded engineering constants | n/a — **embeds ZERO engineering constants** (formatting only) | n/a | n/a | n/a |
| pdf_import / drawing_ir — extraction confidence tiers | n/a (self-reported provenance) | per-entity source + confidence fields — **exemplary** | n/a | yes — author-judgment heuristics, appropriately uncited |
| geotech_common — N60→φ (Peck) | authoritative-secondary (PHT 1974) | **hand-digitization** — breakpoints trace the chart, intermediate slopes author-chosen, NOT per-value verified | limiting/behavior | **FIXED** — Source-basis + "Correlation provenance" DESIGN.md section |
| geotech_common — N60→cu, LL→Cc, Cc→Cr, cu→ε50; GAMMA_W | authoritative-secondary (textbook) / physical constant (GAMMA_W) | text | limiting | **FIXED** — documented in DESIGN.md section |

**Notes.** `reliability/` is the provenance **gold standard** to emulate:
every COV row names its source table and states whether it was verified against
the published paper or transcribed from the report PDF. `drawing_ir`'s per-entity
source+confidence is the ingestion-side model. The only common-layer flag is the
hand-digitized Peck N→φ curve.

## Numerical, probabilistic, ingestion & common

_(fem2d, reliability COV DB, calc_package, pdf_import, drawing_ir, geotech_common)_

---

## RED FLAGS (elevated-risk digitization / weak anchor / unbenchmarked)

No item in this half is **memory-only with no anchor**. The genuine flags are
chart-digitized tables whose anchors verify behavior rather than the exact
ordinates, plus one unbenchmarked table. All are now documented in place; each
is a wiki-verification candidate.

1. **soe Caquot-Kerisel Kp0 base column + R(δ/φ) reduction grid** — only
   φ=30→6.30 is a verified chart read (V-013); the other ~5 Kp0 values and the
   full R grid are chart-attributed without per-value in-hand confirmation.
   *(Byte-identical copy lives in `sheet_pile` — same flag.)*
2. **soe `anchor_design.py` bond-stress table** — GEC-4 nominal presumptive
   values, **UNBENCHMARKED** (no worked-example validation anywhere).
3. **lateral_pile Reese A/B coefficient charts** (+ C1–C3, k-table, stiff-clay
   As/Ac) — digitized from figures; anchors check curve shape/behavior, not the
   ordinates; the cyclic A curve was mis-digitized once (fixed v5.3 review).
4. **seismic_geotech Mononobe-Okabe** — implementation anchored to an INTERNAL
   numerical trial-wedge cross-check, not to a published K_AE table.
5. **geotech_common Peck N→φ** — hand-digitized chart with author-chosen
   interpolation slopes; not verified per-value against PHT 1974 Table 10-3.
6. **reliability TC304 COV rows** — honestly labelled "stored verbatim from the
   report PDF" (transcription); a pre-existing, already-disclosed flag — listed
   here only for the wiki re-verification queue, not as a doc gap.
7. **ground_improvement vibro probe-spacing endpoints** — empirical
   rule-of-thumb, not tied to a specific GEC-13 figure and not benchmarked.
8. **composite_section ACI Ec edition/clause** — the §19.2.2.1(b) clause was
   NOT confirmed against ACI 318 in hand this campaign (correlation itself is
   textbook-standard).
9. **axial_pile Nordlund chart fits (Kδ/CF/Nq'/αt/q_L) + Tomlinson α** — the
   chart-heaviest module and **it has NO published worked-example anchor**
   (DESIGN.md says so plainly); tests re-assert the transcribed values. Best
   chart-fit *disclosure* in the codebase, but genuinely unanchored numbers.
10. **drilled_shaft `fs` cap 200 kPa + RQD multiplier schedule** — uncited (no
    figure/table number for either).
11. **downdrag default β=0.3** — uncited default.
12. **pile_group 6-DOF engine references** — its real basis (CPGA ITL-89-4,
    EM 1110-2-2906) lives in code comments but is **absent from DESIGN.md**.

### Resolved this audit (were live defects, now fixed + pinned)
- **settlement calc package displayed `I_w = 1.0`** while the analysis used the
  Schleicher value (~1.122 for a square). Fixed to render the actual Iw.
- **drilled_shaft calc package displayed & recomputed `Nc = min(6 + L/D, 9)`**,
  the superseded formula, vs the code's `min(6(1+0.2·L/D), 9)`. Fixed.

## WIKI-WISHLIST — documents that would upgrade tiers

Ranked by how much a verified copy would de-risk the code:

1. **Caquot & Kerisel (1948)** passive-pressure tables; **NAVFAC DM-7.2** Kp
   charts; **Caltrans Trenching & Shoring Manual** Fig 4-20 / Matrix 4-1 — to
   verify the soe/`sheet_pile` Kp0 column and R grid *(top priority)*.
2. **Reese, Cox & Koop (1974)** OTC 2080 originals; **Reese & Van Impe (2001)**
   Figs 3.30/3.31; **COM624P (FHWA-SA-91-048)** Tables 2.1/2.2 (Figs 2.19/2.20)
   — Reese sand p-y A/B (+C, k) coefficients.
3. **FHWA GEC-13 (FHWA-NHI-16-027)** vibro-compaction + ground-modification
   chapters — vibro probe-spacing charts and other GI empirics.
4. **Peck, Hanson & Thornburn (1974)** Table 10-3 / their N–φ chart —
   geotech_common N→φ breakpoints and slopes.
5. **FHWA GEC-4 (FHWA-IF-99-015)** Table 4 grout-bond stresses; **PTI DC35.1**
   strand/bar tables; **ASTM A722/A416** — soe anchor design (and to give it a
   worked-example anchor).
6. **AASHTO LRFD Bridge Design Specifications, 9th Ed. §3.10.3**
   Tables 3.10.3.1-1/2/3 — seismic site factors (Fpga/Fa/Fv); plus a published
   **Mononobe-Okabe K_AE table** to anchor the M-O implementation.
7. **ACI 318-19 §19.2.2.1(b)** — confirm the Ec=4700√f'c clause/edition.
8. **TC304 (2021)** state-of-the-art report + **Duncan (2000)** JGGE paper —
   per-row re-verification of the reliability COV database.
9. **Manufacturer product catalogs** (certified reinforcement properties) — to
   replace the representative retaining_walls product constants for real design.

10. **FHWA GEC-12 (FHWA-NHI-16-009)** — Table 7-1 (δ/φ), Table 9-2
    (p-multipliers), and the **Nordlund charts Figs 7-5/7-8/7-13/7-14/7-15** +
    q_L Fig 7-15 (axial_pile is the highest chart-digitization density and has
    NO worked-example anchor — top foundations priority).
11. **FHWA GEC-10** — Fig 13-5 (α) and **§13.3.3** for the origin of the
    200 kPa `fs` cap and the RQD multiplier schedule (both currently uncited);
    note the two editions (NHI-10-016 simplified vs NHI-18-024 rational).
12. **FHWA GEC-6 (FHWA-IF-02-054)** — Fig 5-19 / Table 6-1 (bearing/settlement).
13. **Vesic (1973), Meyerhof, Hansen** originals — bearing-capacity factors;
    **Schmertmann (1978)** — settlement Iz figure.
14. **CPGA (ITL-89-4)** + **USACE EM 1110-2-2906** — pile_group 6-DOF engine
    basis (also: add these to `pile_group/DESIGN.md` refs).
15. **ASTM A615** — standard rebar areas (bearing_capacity `_REBAR_AREAS`).

_(Most in-house-verifiable now: GEC-12/GEC-10/GEC-6 text layers are already in
the `geotech-references` submodule — items 10-12 can be cross-read in-house.)_

## Change log — doc fixes made by this audit (additive, doc-only)

- `lateral_pile/DESIGN.md` — corrected stale cyclic asymptote A_c 0.53→0.55;
  added Source-basis note for the Reese A/B tables.
- `lateral_pile/composite_section.py` — added ACI 318-19 §19.2.2.1(b) clause to
  the Ec citation; flagged edition unverified-in-hand.
- `seismic_geotech/DESIGN.md` — corrected site-class standard "ASCE 7-22" →
  AASHTO LRFD 9th Ed §3.10.3 / NEHRP FEMA P-1050 (the implemented standard);
  added M-O anchor-gap note + site-factor Source-basis.
- `wave_equation/DESIGN.md` — Source-basis for the default quake/damping
  constants (GEC-12 Table 12-3).
- `retaining_walls/mse.py` — GEC-11 two-volume (Vol I 10-024 / Vol II 10-025)
  clarification; Kr/F* digitization + product-constant Source-basis.
- `retaining_walls/DESIGN.md` — "representative product, not a catalog" note.
- `soe/DESIGN.md` — Caquot-Kerisel Kp Source-basis (only φ=30 verified); moved
  ground-anchor design out of "Future Work" and added its provenance +
  UNBENCHMARKED flag.
- `ground_improvement/vibro.py` — annotated probe-spacing as unsourced-empirical.
- `geotech_common/water.py` — GAMMA_W physical-constant basis.
- `geotech_common/soil_properties.py` — Peck N→φ hand-digitization Source-basis.
- `geotech_common/DESIGN.md` — added "Correlation provenance" section.
- `fem2d/porewater.py` — gamma_w / n_w physical-constant Source-basis one-liners.
- `settlement/calc_steps.py` — **BUG FIX**: elastic-settlement step now renders
  the actual influence factor `Iw` used (Schleicher, ~1.122) instead of a
  hardcoded `1.0`; added `_immediate_Iw` helper + regression pin test.
- `drilled_shaft/calc_steps.py` — **BUG FIX**: cohesive-tip `Nc` display + value
  now use `min(6(1+0.2·L/D), 9)` (matching `end_bearing.py`) instead of the
  superseded `min(6+L/D, 9)`; regression pin test.
- `axial_pile/tests/test_axial_pile.py`, `downdrag/tests/test_downdrag.py` —
  added GEC-12 Table 7-9 consistency pin tests (Nt(φ) within the in-house
  reference ranges; verified 2026-07-12).
- `funhouse_agent/tests/test_calc_package_adapter.py` — the two calc-package
  display-fix regression tests.
