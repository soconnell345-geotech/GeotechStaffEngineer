# Verification: Bray & Travasarou (2007) — `slope_stability.bray_travasarou_2007`

**SUMMARY: ALL coefficients CONFIRMED-PRIMARY against the original ASCE paper (digit-for-digit), including the previously flagged rigid-branch a0 = -0.22 (printed in full as Eq. 6, p. 387); both published flexible worked examples reproduced by the module to printed precision; no rigid worked example exists in either source (rigid branch verified instead by hand-evaluating printed Eq. 6 — module matches to 1e-5 in ln D, i.e. rounding of the echoed output only). Zero discrepancies.**

Verified: 2026-07-18. Implementation: `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\slope_stability\newmark.py` (`bray_travasarou_2007`, lines 406–489). No module code changed.

## Sources (hydration status: BOTH HYDRATED, good text layers)

| Tag | File | Status |
|---|---|---|
| PAPER | `C:\Users\socon\OneDrive\Lib\_To Be Sorted\2022-11-28 Schnabel Backup\_Research Etc\VT\VT\Superseded\Earthquake Engineering\Resources and Slides\11 Seismic Slope Stability\Bray and Travasarou - 2007.pdf` | 270,494 B, 12 pp (journal pp. 381–392), text extracted via PyMuPDF. Bray & Travasarou (2007), "Simplified Procedure for Estimating Earthquake-Induced Deviatoric Slope Displacements," ASCE JGGE 133(4). |
| CHAPTER | same folder, `Bray 2007-ICGEE-Seismic Slope Stability-FINAL.pdf` | 672,361 B, 20 pp, text extracted. Bray (2007), "Simplified Seismic Slope Displacement Procedures" (4ICEGE / book-chapter version — the source previously cited as "Ch. 14"). |

Note: the folder is `...\Resources and Slides\11 Seismic Slope Stability\` (the task brief's "Resources and Software" does not exist).

PDF-page → journal-page map for PAPER: journal page = PDF page + 380 (footer-confirmed).

## Coefficient-by-coefficient comparison

### P(D=0) model — module `p_arg` vs PAPER Eq. (3), journal p. 386 (= CHAPTER Eq. (4), PDF p. 12)

Printed: `P(D=0) = 1 - Phi(-1.76 - 3.22 ln ky - 0.484 Ts ln ky + 3.52 ln Sa(1.5Ts))`

| Term | Module | Printed | Verdict |
|---|---|---|---|
| intercept | -1.76 | -1.76 | CONFIRMED-PRIMARY |
| ln ky | -3.22 | -3.22 | CONFIRMED-PRIMARY |
| Ts·ln ky | -0.484 | -0.484 | CONFIRMED-PRIMARY |
| ln Sa(1.5Ts) | +3.52 | +3.52 | CONFIRMED-PRIMARY |
| "zero" threshold | docstring "<= ~1 cm" | paper p. 386/CHAPTER p. 12: D <= 1 cm | CONFIRMED-PRIMARY |

### Median displacement — module vs PAPER Eq. (5) (recommended, with magnitude term), journal p. 387

Printed: `ln D = -1.10 - 2.83 ln ky - 0.333 (ln ky)^2 + 0.566 ln ky ln Sa(1.5Ts) + 3.04 ln Sa(1.5Ts) - 0.244 (ln Sa(1.5Ts))^2 + 1.50 Ts + 0.278(M-7) ± eps`, D in cm.

| Term | Module | Printed (p. 387) | Verdict |
|---|---|---|---|
| a0 flexible | -1.10 | -1.10 | CONFIRMED-PRIMARY |
| ln ky | -2.83 | -2.83 | CONFIRMED-PRIMARY |
| (ln ky)^2 | -0.333 | -0.333 | CONFIRMED-PRIMARY |
| ln ky · ln Sa | +0.566 | +0.566 | CONFIRMED-PRIMARY |
| ln Sa | +3.04 | +3.04 | CONFIRMED-PRIMARY |
| (ln Sa)^2 | -0.244 | -0.244 | CONFIRMED-PRIMARY |
| Ts term | +1.5·Ts | +1.50·Ts | CONFIRMED-PRIMARY |
| magnitude | +0.278(M-7) | +0.278(M-7) | CONFIRMED-PRIMARY |
| sigma_ln | 0.66 | sigma = 0.66 for Eq. (5) (p. 387; paper's magnitude-free Eq. (4) has sigma = 0.67, which the module correctly does NOT use since it implements Eq. (5)) | CONFIRMED-PRIMARY |

### Rigid branch — the audit-flagged item

- **a0 = -0.22: CONFIRMED-PRIMARY.** PAPER p. 387 states twice that the first term of Eqs. (4)/(5) "should be replaced with -0.22" for nearly rigid masses, and prints the full rigid equation as **Eq. (6)**: `ln D = -0.22 - 2.83 ln ky - 0.333(ln ky)^2 + 0.566 ln ky ln PGA + 3.04 ln PGA - 0.244(ln PGA)^2 + 0.278(M-7) ± eps` — i.e., a0 = -0.22, the +1.50 Ts term dropped, and Sa(1.5Ts) → PGA (= Sa at Ts=0). This is exactly the module's rigid branch (`a0 = -0.22`, `ts_term = 0`, caller passes PGA as `sa_1p5ts`). CHAPTER p. 13 states the same replacement.
- **Threshold: CONFIRMED-PRIMARY.** CHAPTER p. 13 (clean glyphs): "replaced with -0.22 when **Ts < 0.05 s**" and "use Eq. (5) for cases where Ts ranges from 0.05 s to 2 s". Module auto-selects rigid when `ts < 0.05` (strict), i.e., flexible at exactly 0.05 s — matches. (The PAPER's <=/< glyph is lost in text extraction; the CHAPTER wording is unambiguous.)
- **Rigid worked example: NOT-FOUND-IN-SOURCE.** Neither PDF contains a published rigid-branch numerical example (both illustrative examples are flexible). Substitute check: printed Eq. (6) hand-evaluated at three points vs the module —

| ky | PGA (g) | M | Eq. (6) hand ln D | module ln D | diff |
|---|---|---|---|---|---|
| 0.10 | 0.4 | 7.0 | 2.734568 | 2.734562 | 6e-6 |
| 0.05 | 0.3 | 6.5 | 3.158113 | 3.158106 | 7e-6 |
| 0.20 | 0.6 | 7.5 | 2.459895 | 2.459931 | 4e-5 |

  (diffs are purely the 3-decimal rounding of `displacement_cm` in `to_dict()`; branch flag `rigid=True` echoed; ts=0.049 → rigid, ts=0.05 → flexible confirmed.) The rigid branch is now equation-verified against the primary source; only a published rigid numeric example remains nonexistent, which is a property of the literature, not a gap in the module.

## Worked-example reproductions (module run via `funhouse_agent.dispatch.call_agent("slope_stability","bray_travasarou_2007",...)`)

### PAPER example (earth dam, journal pp. 390–391): ky=0.14, Ts=0.33 s, Sa(0.5 s)=1.07 g, Mw=6.9

| Quantity | Paper prints | Module | Agreement |
|---|---|---|---|
| P(D=0) | 0 (Eq. 3) | 0.0000 | exact |
| ln D | 3.77 | 3.7733 | exact at printed precision |
| median D | exp(3.77) ~ 40 cm | 43.5 cm | matches (paper rounds exp(3.77)=43.4 to "~40"; its 16/84% pair 20/80 cm brackets identically) |
| 16% / 84% | ~20 / ~80 cm | 22.5 / 84.2 cm | matches at the paper's ~half/~double rounding |

### CHAPTER example (30 m fill, PDF pp. 16–17): ky=0.14, Ts=0.4 s, Sa(0.6 s)=0.48 g, Mw=7.2

| Quantity | Chapter prints | Module | Agreement |
|---|---|---|---|
| P(D=0) (Eq. 9 first line) | 0.01 | 0.0089 | exact at printed precision |
| ln D | 2.29 | 2.2866 | exact at printed precision |
| median D | ~10 cm | 9.84 cm | exact at printed precision |
| 16–84% range | 5–20 cm | 5.1–19.0 cm | matches |

(This is the example already cited as the module's V-062bt anchor; it is now re-confirmed against the source PDF itself, and the ASCE-paper example is a NEW, independent reproduction.)

## Verdict roll-up

| Item | Verdict |
|---|---|
| P(D=0) coefficients (-1.76, -3.22, -0.484, +3.52) | CONFIRMED-PRIMARY (PAPER Eq. 3, p. 386) |
| Flexible ln D coefficients (-1.10, -2.83, -0.333, +0.566, +3.04, -0.244, +1.50 Ts, +0.278(M-7)) | CONFIRMED-PRIMARY (PAPER Eq. 5, p. 387) |
| sigma_ln = 0.66 | CONFIRMED-PRIMARY (PAPER p. 387) |
| Rigid a0 = -0.22, Ts term dropped, Sa→PGA | CONFIRMED-PRIMARY (PAPER Eq. 6, p. 387) — audit flag CLOSED |
| Rigid/flexible threshold 0.05 s (strict <) | CONFIRMED-PRIMARY (CHAPTER p. 13) |
| PAPER worked example (ky=0.14, Ts=0.33, Sa=1.07, M=6.9) | REPRODUCED (ln D 3.7733 vs 3.77; P(D=0) 0) |
| CHAPTER worked example (ky=0.14, Ts=0.4, Sa=0.48, M=7.2) | REPRODUCED (ln D 2.2866 vs 2.29; P(D=0) 0.0089 vs 0.01) |
| Published rigid worked example | NOT-FOUND-IN-SOURCE (neither PDF has one; Eq. 6 hand-evaluation used instead, module matches) |

Follow-up (docs only, not done here per no-edit instruction): the docstring's "treat the rigid coefficient as UNVERIFIED" caveat in `slope_stability/newmark.py` (lines ~456–462) and the corresponding `module_work/provenance_audit_slope.md` flag can now be retired — the rigid branch is confirmed against printed Eq. (6) of the primary source.
