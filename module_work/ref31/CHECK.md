# REF-31: cited designations against the PDF covers

Branches: refs and app `claude/ref-31-cover-citations` (refs `d9dadf3`; the
app's `geotech-references` pointer moves to it). Method as REF-28: read each
cover in `docs/`, check which edition the data was built from, then correct
the citation. Citations only; no calculated value changes.

## What each cover prints

| Package | PDF | Cover (PDF page 1) | Built from |
|---|---|---|---|
| gec_10 | `docs/GEC 10.pdf` | "Publication No. FHWA-NHI 18-024, FHWA GEC 010, September 2018 ... Drilled Shafts: Construction Procedures and Design Methods" | 2018: all 19 text files already said FHWA-NHI-18-024; the figure catalog's entries are the 2018 numbering (Fig 8-1 "Drilled Shaft Design and Construction Process", Fig 10-1 load transfer; the 2010 manual's chapters differ). Only the manifest and catalog headers said FHWA-NHI-10-016 |
| gec_13 Vol I | `docs/GEC 13 Vol 1.pdf` | "Publication No. FHWA-NHI-16-027, April 2017 ... Volume I" | unchanged, correct |
| gec_13 Vol II | `docs/GEC 13 Vol 2.pdf` | "Publication No. FHWA-NHI-16-028, April 2017 ... Volume II" | chapters 6-10 (`volume: 2`, rebuilt from this PDF in REF-25) and its 30 catalogued figures (`pdf_path` Vol 2) |
| ufc_backfill | `docs/ufc_3_220_04fa_2004.pdf` | "UFC 3-220-04FA, 16 January 2004, Backfill for Subsurface Structures" (supersedes TM 5-818-4, 1983) | **not this PDF**, see below |

### ufc_backfill: an edition gap, nothing changed

The held 04FA (60 pages; the reproduced TM 5-818-4 text) prints:
- densities as a percent of **CE 55** maximum dry density by MIL-STD-621
  (p. 5-2: 90 % in open areas; 95-100 % cohesionless and at least 95 %
  cohesive under structures) and Table 5-1 "Summary of Compaction Criteria"
  (p. 5-3, a picture); the package's compaction table is **ASTM D698** with
  rows (85 % above pipe, 90 % pipe haunch, ...) the print does not have;
- lift thicknesses in inches in prose (p. 5-1/5-2: 10 in loose to 6 in for
  cohesive, 8 in for cohesionless, 12-15 in with heavy vibratory equipment,
  4-6 in confined cohesionless, at most 4 in with rammers); the package's
  nine-equipment millimetre table is not printed;
- compaction-induced pressure only as Fig. 2-5 (p. 2-7) with no equation;
  the package's Broms (1971) sigma_h,max = sqrt(2 q gamma K0 / pi) is not in it;
- no filter ratios (it sends filter design to TM 5-818-5, p. 2-6); the
  package's D15/d85 <= 5 etc. are not in it;
- Tables 3-1 and 5-1 only; the package's tables carry no printed numbers.

The package's figure catalog was built from this PDF (Figs 2-1 to B-8 at
their pages), but its tables and equations were not. No UFC 3-220-04N is
held. Per the task, the citation is unchanged and the cover row stays
`xfail` (strict), its reason now naming the edition gap. **For the lead:**
the strict xfail was kept rather than removed, because removing it would
make the test fail; the package's tables/equations need either a held
source or the treatment REF-61 gave GEC-13's unprinted tables (that would be
a new item, values may go; not done here).

## Before / after

| Where | Before | After |
|---|---|---|
| refs `scripts/manifests/gec_10.json` reference_id | FHWA-NHI-10-016 | FHWA-NHI-18-024 |
| refs `gec_10/figures_catalog.json` reference_id | FHWA-NHI-10-016 | FHWA-NHI-18-024 |
| refs gec_10 reference_title (manifest, catalog, 19 text files) and `__init__`/`figures`/`tables` docstrings | Drilled Shafts: Construction Procedures and LRFD Design Methods (the 2010 title) | Drilled Shafts: Construction Procedures and Design Methods (chapter 10 keeps its "(2018)") |
| refs `gec_13/text/chapter06-10.json` reference_id | FHWA-NHI-16-027 | FHWA-NHI-16-028 |
| refs `gec_13/figures_catalog.json`, 30 Vol II figures | (catalog's FHWA-NHI-16-027) | per-figure `"reference_id": "FHWA-NHI-16-028"` |
| refs `scripts/manifests/gec_13.json` volume 2 | (manifest's -027) | `"reference_id": "FHWA-NHI-16-028"` |
| refs `_figures_db.py` | catalog-level reference_id and volume for every figure | a figure's own reference_id and volume win (`figure_get` on a GEC-13 figure now reports volume 1 or 2, not null) |
| refs `scripts/build_figure_catalog.py` | | writes a volume's reference_id onto its figures, so a rebuild keeps -028 |
| refs `gec_13/__init__.py`, `figures.py` Eq. 8-2, package `__init__` list | FHWA-NHI-16-027 | Vol II named as FHWA-NHI-16-028 |
| refs `scripts/retrieval_gold.json` (Sec. 7.2.3.2.4), `tests/test_query_expansion.py` comment (Ch. 7) | FHWA-NHI-16-027 | FHWA-NHI-16-028 |
| app `adapters/__init__.py` gec13 brief | Ground modification reference (FHWA-NHI-16-027) | Ground modification reference (FHWA-NHI-16-027/-028) |
| app `adapters/gec13_adapter.py` (figures, tables, text citation) | FHWA-NHI-16-027 | FHWA-NHI-16-027 / -028 (GEC-13 Vols I / II) |
| app `ground_improvement/DESIGN.md` reference line | FHWA-NHI-16-027 ... Vols. I and II | Vol. I FHWA-NHI-16-027 and Vol. II FHWA-NHI-16-028 |
| app ufc_backfill brief and adapter | UFC 3-220-04N | unchanged (edition gap) |

Left as printed: GEC-10 chapter 1's "Replaces FHWA-NHI-10-016 (2010
edition)" (the manual's own text); `drilled_shaft`'s deliberate 2010
citations; `funhouse_agent/cited_documents.py` already maps both -027 and
-028 (and 10-016) to their GEC. The app's `gec10` brief and adapter already
said FHWA-NHI-18-024. App tests and fixtures carry none of the old strings
except `test_module_followup_adapters.py`, which asserts "10-016" is absent
from GEC-10 method docs (still true).

## Test changes (refs)

`tests/test_cover_citations.py`: GEC-10's strict xfail removed (it is a
plain row); GEC-13 is checked one volume at a time: `"gec_13/1"`
(FHWA-NHI-16-027: manifest, catalog header, 3 Vol I figures, 5 text files,
the references-agent citation; 11 citations) and a new `"gec_13/2"` row
(FHWA-NHI-16-028 against the Vol II cover: manifest volume 2, 30 figures, 5
text files; 36 citations). ufc_backfill keeps its strict xfail.
`tests/test_gec13_text.py`: reference_id by volume, and a test that Vol II
figures carry -028 and Vol I figures no override.

## Tests run

refs (refs branch, Python 3.12 venv, PyMuPDF present so the cover tests ran):
- `tests/test_cover_citations.py`: 29 passed, 1 xfailed (ufc_backfill)
- `test_ufc_backfill`, `test_references_agent`, `test_gec13_text`, gec10 /
  figure / query-expansion tests: 939 passed
- whole refs suite (`-n 8`): 6,542 passed, 1 xfailed, exit 0

app (refs branch on PYTHONPATH):
- `test_reference_adapters`, `test_citations`, `test_reference_result_sizes`,
  catalog / cited-documents / module-followup tests,
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  560 passed (fingerprints and public text: 20 passed; no re-record needed,
  the briefs are not in the page fingerprints)
- `test_reviewers`, `test_reference_db_dispatch`, `test_method_aliases`,
  `test_gm179_unused_inputs`, figure / cited tests, `ground_improvement/`:
  574 passed, 3 skipped, 1 failed:
  `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
  fails identically with this branch's changes stashed (pre-existing on
  master in this environment; unrelated to citations).

## For the lead (files this branch did not touch)

- refs `geotech_references/gec_13/tables.py` (REF-62 has it open): the
  module docstring cites "FHWA-NHI-16-027 (GEC-13)" for all its tables, but
  `table_7_2_deep_mixing_strength` (line ~842/907, "Table 7-2", ch. 7) and
  the jet grouting systems (line ~941/979, "Table 8-2", ch. 8) are Vol II
  chapters, so -028 if they stay.
- refs `agents/gec13_agent.py` (REF-62): line 223 "Not printed in
  FHWA-NHI-16-027 (GEC-13); Vol II Ch. 7, p. 7-16" and line 226
  "FHWA-NHI-16-027 (GEC-13) Vol II, Ch. 8, Eq. 8-2, p. 8-29" should read
  FHWA-NHI-16-028; line 210-211's category citations name -027 only.
- app `funhouse_agent/adapters/_reference_common.py`: no GEC-13 Vol II
  designation in it (one alias line, `figure_5_5_settlement_improvement`, a
  Vol I function); nothing to carry.
- refs `CLAUDE.md` / `README.md` wording wanted: the GEC-13 rows "FHWA-NHI-16-027
  Ground Modification" -> "FHWA-NHI-16-027 / -028 (Vols I / II) Ground
  Modification"; the gec_10 architecture line already says FHWA-NHI-18-024;
  README's "Drilled Shafts: Construction Procedures and LRFD Design Methods"
  -> "... and Design Methods" for the 2018 manual; the GEC-13 section heading
  "FHWA-NHI-16-027" -> "FHWA-NHI-16-027 (Vol I) / -028 (Vol II)".
- `docs/user_manual/`: left for DOC-23.

## Noted, not fixed

- `scripts/manifests/gec_10.json`'s chapter list (titles and page ranges)
  is the 2010 manual's (e.g. "10 LRFD for Drilled Shaft Design"); the text
  files are the 2018 chapters. Used only by the retired text pipeline and the
  weekend QC page extraction; no output value depends on it.
- The ufc_backfill figure catalog was built from the 04FA PDF yet carries
  the 04N id with the rest of the package; left with the package.

## Release note

Citations only, no calculation changes: GEC-10 is cited as FHWA-NHI-18-024
(September 2018, "Drilled Shafts: Construction Procedures and Design
Methods") everywhere, as its data always was; GEC-13 Volume II (chapters
6-10 and their figures) is cited as FHWA-NHI-16-028, its own cover's number,
and a GEC-13 figure lookup now reports its volume. ufc_backfill keeps its
UFC 3-220-04N citation: the held UFC 3-220-04FA (2004) does not print its
tables or equations (REF-31's edition gap).
