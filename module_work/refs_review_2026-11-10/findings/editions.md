# DM 7.2: the held 2025 UFC against the 2026 UFS — comparison

Filed by the lead from helper `edition`'s messages (2026-11-10). Held: UFC
3-220-20, 16 January 2025 (`docs/ufc_3_220_20_2025.pdf`, 724 PDF pages,
printed = PDF - 38). Newer: UFS 3-220-20, 26 March 2026 (downloaded to
`C:/dev/tmp/refs_review_2026-11-10/editions/pdf/`, 720 PDF pages, printed =
PDF - 34). Data: `editions/page_diff.tsv` (725 rows), `editions/png/`.

**The 2026 UFS is the 2025 UFC text under a new designation and status; no
engineering content changed.** Across the 686 body pages there is one wording
change (a corrected cross-reference on printed p. 442). No equation,
coefficient, table value, figure, limit or recommendation changed, Appendix B
included.

Status: the document is now a Unified Facilities SUPPLEMENT. Its foreword,
word for word: "Unified Facilities Supplements (UFS) provide non-mandatory,
supplemental information in support of Unified Facilities Criteria (UFC). Only
UFC establish enforceable criteria. UFS are For Information Only (FIO) and
exist to help users understand and apply UFC requirements." No record of
changes, no change markers, no signature.

Numbering: same 718 printed page labels on the same content; every chapter,
section, equation, figure and table number the same and on the same printed
page, except Eq. 4-28 (same number, now on p. 222 instead of 221: its last
line reflowed). Every running header changed from "UFC 3-220-20 / 16 January
2025" to "UFS 3-220-20 / 26 March 2026". If the PDF file is swapped, PDF page
indices drop by 4.

Method: pages aligned by printed label; five independent checks per pair
(word text; every character's font, size and flags; embedded images by pixel
digest and position; vector drawing items; a 100 dpi pixel comparison); every
flagged page viewed rendered. Table 5-2 and some other tables and figures are
drawn as vector paths with no text layer, so a text-only diff cannot see
them. Counts: header-footer only 704; reflow only 12 (TOC pp. ii-xi; pp.
221-222); small text change 3 (cover, TOC p. i, p. 442); added 1 (the UFS
foreword); removed 5 (two blanks, the inner title page with the empty Record
of Changes, the UFC foreword with signatures, a blank).

## The one wording change

Ch 6, Sec. 6-5.8.2 "Empirical Method for Pile Group Settlement in
Coarse-Grained Soil", printed p. 442 (PDF 480 in 2025, 476 in 2026). 2025:
"For pile groups in silty sand, the settlement estimate from Equation 6-51
should be doubled." 2026: "...from Equation 6-46 should be doubled." (Eq. 6-51
is the 2V:1H stress-spreading equation; Eq. 6-46 is Meyerhof's group
settlement on the same page.) Evidence `editions/png/p442_eqref_*.png`.

## Specific places checked in the 2026 edition: all the same as 2025

- Table 5-2: still prints Nc(18) = 15, Nq(27) = 17, Nc(44) = 160, Nq(49) = 350 in the Terzaghi columns (pixel for pixel at 200 dpi; all 1,132 vector items match).
- Equations 3-4, 3-9, 3-10, 5-11, 5-12, 5-26, 6-22, 6-23, 6-30, 6-31, 6-33, 6-40, 6-41, 6-47, 6-48, 6-49, 6-50, 6-76 (keeps its trailing minus sign after the bracket in both).
- Table P-6 DSS row: "DSS: USR = sin phi' - 0.5 sin^2 phi'". Table 6-17 Nq equation: "Nq ~ 1.2 (Z/b)^-0.437 exp[6.34 (Z/b)^0.0486 tan phi'] <= 200 unless supported by project-specific information".
- Tables 6-18, 6-21, 6-22, 6-23, 6-36, 6-37 (6-18 viewed rendered; the others by the automated checks and saved PNGs).
- Figure 4-36: raster byte-identical; curve equations M_DESIGN/M_MAX = 2.03/ln rho - 0.076 (medium dense and dense) and 1.58/ln rho - 0.015 (very dense); note 2 "No reduction in M_MAX is permitted for walls in fine-grained soils, or very loose to loose coarse-grained soils."; legend still cites "Figure 3-37".
- Figure 5-18 and Table 5-11. Appendix B (printed pp. 596-659): every page header-only.

## What a move to 2026 touches in the library

No coded equation, table or figure. It touches: the designation and date in
the citations ("UFC 3-220-20" appears 298 times in 17 files under
`geotech_references/dm7_2`); one printed-page citation (`chapter4.py:1651`,
Eq. 4-28, p. 221 -> 222); one text field (`text/chapter06.json:1291`, the
Sec. 6-5.8.2 body with the 2025 "Equation 6-51" wording;
`meyerhof_group_settlement` does not apply the silty-sand doubling, so no
answer changes); and only if the PDF file is swapped, the 252
`pdf_page_index` values in `figures_catalog.json` and the page ranges in
`scripts/manifests/dm7_2.json`, each -4.

Side findings (not edition differences): `chapter6.py` docstrings cite 2025
PDF page numbers, not printed ones (Table 6-17 "p. 459" is printed 421; Eq.
6-76 "p. 512" is printed 474), while chapters 4 and 5 cite printed pages;
`chapter6.py:1568` cites Eq. 6-51 at "p. 485" (printed 445, PDF 483).

Owner's call: whether "For Information Only" changes how the app cites DM 7.2.
