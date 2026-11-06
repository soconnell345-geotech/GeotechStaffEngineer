# REF-12: the AASHTO ESAL lookups through the full Appendix D tables

Branches: refs `claude/ref-12-esal-lef` (cee5ad7); app `claude/ref-12-esal-lef`
(moves the `geotech-references` pointer from c0ed2b9 to cee5ad7). No version
or pin bump.

## What the print says

AASHTO Guide for Design of Pavement Structures (1993), Appendix D, Tables
D.1-D.18 (pdf_page 362-379, printed D-3 to D-20): load equivalency factors for
single, tandem and triple axles, flexible by SN = 1-6, rigid by D = 6-14 in,
each at pt = 2.0, 2.5 and 3.0. No equation is printed (lef.py's docstring
gives the reasons). `geotech_references/aashto_1993/lef.py` already held every
table cell; `tables.py`'s four `esal_*` lookups held only Tables D.4, D.5,
D.13 and D.14 (SN = 5 / D = 9 in, pt = 2.5).

## Before / after

| Call | Before | After |
|---|---|---|
| `esal_flexible_single_axle(18)` | 1.0 | 1.0, `table` "Table D.4" |
| `esal_flexible_single_axle(24, sn=3)` | NotImplementedError | 3.09 (Table D.4) |
| `esal_flexible_single_axle(18, pt=2.0)` | NotImplementedError | 1.0 (Table D.1) |
| `esal_flexible_tandem_axle(40, sn=3)` | NotImplementedError | 2.06 (Table D.5) |
| `esal_rigid_single_axle(18, d_in=10)` | NotImplementedError | 1.0 (Table D.13) |
| `esal_rigid_tandem_axle(36)` | 2.43 | 2.43 (Table D.14) |
| `esal_flexible_single_axle(18, pt=2.2)` | NotImplementedError | ValueError, lef's message |
| `esal_rigid_single_axle(60)` | 67.8 | 67.8 plus lef's clamp `note` |

- The four functions now call `lef.load_equivalency_factor`; return keys kept
  (`axle_load_kips`, `sn`/`d_in`, `pt`, `lef`, `reference`), `table` added,
  and lef's `note` passed on when a load or column is clamped (a column
  outside SN 1-6 / D 6-14 is clamped, as lef does; intermediate SN or D is
  interpolated linearly, as lef does).
- What lef refuses (pt not 2.0/2.5/3.0, a load or SN/D <= 0) they refuse with
  lef's ValueError and the same message (tested string-equal). The exception
  type changes from NotImplementedError to ValueError; nothing in either repo
  caught NotImplementedError from these calls (grepped; the app's
  `pavement_design/traffic.py` uses lef directly and the esal_* only as a
  fallback for a refs install without lef.py).
- **SN = 5 / D = 9 in, pt = 2.5 unchanged to the digit**: before the change
  the old curves and lef agreed exactly at every quarter-kip from 0.5 to 95
  kips (all four functions, clamps included); 64 load points captured from the
  old code are pinned in refs `tests/test_aashto_1993_esal.py`.

## Spot checks off the printed page (rendered scan, docs/aashto1993.pdf)

- Table D.4 (pdf_page 365, printed D-6), SN = 5: 12 kips "189" = 0.189; 30 kips
  "7 0" = 7.0. Also SN = 3, 24 kips "3 09" = 3.09.
- Table D.5 (pdf_page 366, printed D-7), SN = 5: 36 kips "1 38" = 1.38; 90 kips
  "46 8" = 46.8. Also SN = 3, 40 kips "2 06" = 2.06.

All match the code; they are pinned in `test_spot_checks_off_the_printed_page`.

## docs/UFC_4-023-03_2024_c4.pdf

A 1,359-byte HTML page (a web site's shell) under a .pdf name. Grep found no
reader (no manifest, script or catalog names it; the real document is
`docs/UFC_4-023-03.pdf`, which `ufc_collapse` uses). Deleted.
`tests/test_docs_pdfs.py` checks every git-tracked `docs/*.pdf` starts as a
PDF and opens with PyMuPDF (skipped where PyMuPDF is not installed): 38
files pass (39 tests with the listing check).

## Tests

Refs (branch head):
- `tests/test_aashto_1993*.py tests/test_docs_pdfs.py`: 544 passed.
- full refs suite `pytest -n 8 tests`: 6,658 passed, 2 xfailed.

App (refs branch on PYTHONPATH):
- `pavement_design` + `webapp/tests/test_page_fingerprints.py` +
  `webapp/tests/test_public_text.py`: 106 passed (fingerprints unchanged: the
  reference adapters' text is not in the page builds, so no re-record).
- `funhouse_agent/tests/test_reference_adapters.py`: 309 passed.
- `funhouse_agent/tests/test_defaults_used.py -k "pavement or aashto"`: 34 passed.
- the other files naming aashto/esal (see the app commit message for counts).

## App side

- `docs/user_manual/manual.html`: the four esal_* rows re-rendered with the
  builder's own `method_param_table` from the adapter's METHOD_INFO (no full
  rebuild).
- `pavement_design/traffic.py`: the fallback's comment no longer says the
  lookups are SN = 5 / D = 9 only. No code change.

## Release note

The AASHTO 1993 ESAL lookups (`esal_flexible_single_axle`,
`esal_flexible_tandem_axle`, `esal_rigid_single_axle`,
`esal_rigid_tandem_axle`) now answer any printed SN (1-6) or slab thickness D
(6-14 in) at pt 2.0, 2.5 or 3.0 from the full Appendix D tables, instead of
refusing everything but SN = 5 / D = 9 in at pt = 2.5; each answer names its
printed table. SN = 5 / D = 9 in at pt = 2.5 are unchanged. An off-grid pt is
now refused with a ValueError (was NotImplementedError). Design ESALs from
`pavement_design` do not change (it already read the full tables).

## Noted, not fixed

- `aashto_1993/figures_catalog.json` (the Appendix D truck-factor worked
  example entry, line 215) still says "see tables.esal_flexible_single_axle/tandem_axle for
  the digitized SN=5, pt=2.5 factors". Left alone so the reference library
  the page fingerprints record does not move for a description; wanted
  wording: "see lef.load_equivalency_factor (or the tables.esal_* lookups)
  for the Appendix D factors".
- Refs CLAUDE.md / README not edited, per the task. Wanted wording for the
  refs CLAUDE.md GEC/AASHTO table row (aashto_1993): append "the
  tables.esal_* single/tandem lookups read lef.py's full Appendix D grid
  (REF-12)"; the aashto_1993 test files now run 505 tests
  (`pytest tests/test_aashto_1993*.py -q`); total refs
  tests 6,541 -> 6,660 collected; and a docs line "every tracked docs/*.pdf is
  checked to open as a PDF (tests/test_docs_pdfs.py)".
