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
