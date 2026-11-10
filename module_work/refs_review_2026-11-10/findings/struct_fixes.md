# struct fix table (branch `fix/review-struct`, worktree `C:/dev/tmp/wt-fix-struct`)

Filed by the lead from fixer `struct`'s two messages (2026-11-10). Base
88bf0bc, HEAD fe313a3, 14 commits, not pushed. No public function removed or
renamed. struct-16 (wind speed conversion) and the re-read of every formula
against rendered pages were asked for after this table was sent: addendum
pending. Independent verification: after the addendum.

| ID | Status | What changed | Commit |
|---|---|---|---|
| struct-01 | fixed | UFC 4-023-03 Table 4-1 rows ii-iv acceptance criteria as printed (pdf 79, printed 63) | 34bbca4 |
| struct-02 | fixed (units named) | RC beam flexure tables say psi units; results add `vw_ratio`, `vw_ratio_units`, `vw_ratio_row`; values unchanged; psi unit comes from GSA's worked example only | ec9e0b3 |
| struct-03 | fixed | EM 2107 Eq 10.2-10.14 descriptions are the printed load-combination headings (pdf 155-157) | 5e07827 |
| struct-04 | fixed | steel connection tables name the inch; a depth at which a printed expression is <= 0 raises (762 "in" gave m -59.0); m between 0 and 1 returned with a note | a9b1b36 |
| struct-05 | fixed | em_2104: every parameter described with its unit (167 of 230 undescribed -> 0); `aci_beta1` "Not MPa" | 4c4675e, 0cf34de |
| struct-07 | fixed | EM 2107 Table 4.1 load factor for T is 1.2 in all three categories (was an error) | bbd1031 |
| struct-09 | fixed | GSA Tables 6 and 8 single printed acceptance column (Table 6 primary 0.063 -> 0.10; Table 8 0.015 -> 0.020); module docstring and chapter04.json no longer say IDENTICAL to UFC | 7c5893b, 6cb215b |
| struct-10 | fixed | five page citations | d72c640, 7ab014c, 4c4675e |
| struct-11 | fixed | `bolt_grade_check` raises on an unlisted grade (was "not permitted") | a0f0864 |
| struct-06, -08, -12, -13, -14, -15 | left | 0-based `pdf_page`; bare numbers; METHOD_INFO types; `lp` default 3.3 ft; `is_minimum` echo | |

Tests: every touched file run on its own, exit 0 (counts in the fixer's
message); cover_citations 29 passed 1 xfailed; references_agent 36; chapter
audit for gsa_collapse passes. CHANGELOG "Unreleased": d864bf4, fe313a3.

New result keys: `acceptance`; `vw_ratio`, `vw_ratio_units`, `vw_ratio_row`;
`d_units`/`dbg_units`, `note`. New ValueErrors: steel depth (eight functions);
unknown bolt grade. No app file names the changed functions.

Not settled from the print: steel m-factors between 0 and 1 (returned with a
note; refusal at <= 0 is the fixer's reading); em_2104 f'c units (Notation
lists psi; Appendix examples and code use ksi; docstrings say which);
struct-02's psi unit (worked example only); the print's "ASTM 3148" (no F)
raises as typed.

Observations not fixed: em_2107 Table 4.1 EQ extreme returns the string
"1.0 or 1.25"; GSA Table 7 rows ii-iv not compared with the print in this
pass; one bare `git stash`/`pop` was run early (stash list empty afterwards).
Commit trailers name the fixer's own model.
