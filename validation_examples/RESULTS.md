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
