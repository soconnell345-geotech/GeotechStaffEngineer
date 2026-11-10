# core: the library's shared machinery (search, fetch, catalogs, packs, interpolation) — findings register

Filed by the lead from reviewer `core`'s messages (2026-11-10). `refs/` =
`geotech-references/geotech_references/`; `app/` = `funhouse_agent/`.
Evidence: `C:/dev/tmp/refs_review_2026-11-10/core/` (every probe with its
output; `clamping_functions.md`; `catalog_pages*.tsv`). Counts: **A 3, B 10,
C 6.** Lead reproduced core-01, -02, -06, -08 through `dispatch.call_agent`.

## A

- **core-01** Two sections of one reference share a number; the fetch returns the other one. `refs/_retrieval_db.py:747-751` (`ORDER BY rowid LIMIT 1`), `refs/_retrieval.py:536-568`, `refs/_schema.py:105-156`. 22 ids (gec_13 16, aashto_1993 4, ufc_stabilization 2); 23 sections unreachable by `reference_get`; the two fetch layers return different sections for one id. `reference_get('gec_13','8.2.7')` -> chapter 1 "Step 7: Consider Project Risks"; `gec13.retrieve_section('8.2.7')` -> "Jet Grouting". Cause in GEC-13: chapter 1 keeps the print's local numbers bare. Data: `gec_13/text/chapter01.json`; `aashto_1993/text/chapter06.json, 09, 10, 11`; `ufc_stabilization/text/appendix_a/b/c.json`.
- **core-02** A figure number held by two catalog rows; the read-off goes to the first row's page. `refs/_figures_db.py:538-542`; `app/adapters/figure_db_adapter.py:99-107`; `app/reference_vision.py:218-219`. ufc_pavement F-21 (print repeats F-21: PDF 264 M1078 cargo truck, PDF 265 P-22 crash truck; no F-22); gec_13 6-14 (print repeats, Vol 2 PDF 52 and 54); gec_11 "5-14" row 63 is printed Figure 5-11 (Vol 1 PDF 243, printed 5-17), so 5-11 is not found; gec_12 D-1 to D-4: 10 continuation rows all given the first page's index.
- **core-03** (pattern) 81 of 1,234 public functions reach a clamping primitive (`_linterp` 72, `min(max())` 20, endpoint return 7, `min(input, limit)` 4). Out of range: 9 raise at both ends, 2 flag, 7 no clamp seen, **42 return a clamped answer with nothing said** (16 with hard evidence), 21 not probed. Not print-checked: a printed open-ended row makes a clamp right. List: `core/clamping_functions.md`.

## B

- **core-04** `refs/_interpolation.py:22-30`: a NaN input returns the table's last value; 37 lookups give a finite answer (GEC-12 `figure_7_16b_nq(phi=nan)` -> 475).
- **core-05** Search relaxes to any-word and synonym matching and labels the hits "more match": `refs/_retrieval_db.py:563-607, 359-372`; `refs/_figures_db.py:465-473`; `refs/_retrieval.py:179-194`.
- **core-06** `figure_search` and the chapter filters answer an unknown scope with an empty list: `refs/_figures_db.py:413-414, 263-266`; `app/adapters/figure_db_adapter.py:84-93`; `refs/_retrieval_db.py:526-531`. (REF-41 fixed the reference name for `reference_db` only.)
- **core-07** Per-reference `search_sections` is a substring AND with no fallback (`refs/_retrieval.py:648-650, 725-750`): "what is the nordlund method" -> empty; "Ka" matches inside "shrinkage".
- **core-08** A section id sent as a JSON number returns another section (7.10 -> 7.1): 34 ids in 10 references.
- **core-09** A loose reference name ("FHWA", "dm7", "pile", "UFC") resolves by prefix or title and the read-off echoes the caller's word; "GEC-13" always fails (catalog listed twice). `refs/_figures_db.py:243-311, 234-240`; `app/reference_vision.py:273, 311`.
- **core-10** A skipped local pack, or a skipped chapter of one, is invisible to the helper (`refs/_packs.py:160-163, 181-203`; `app/reference_packs.py:100-119`).
- **core-11** The search databases can serve text no longer in the package (deleted chapter file; a file restored with an older date); key is sha1(package folder | installed version); the figure database has no layout version. Pip installs are safe.
- **core-12** A built-in chapter file or catalog the build cannot read is left out without a word (`refs/_retrieval_db.py:265-268`; `refs/_figures_db.py:134-137`).
- **core-13** Second volumes are cited under the first volume's report number: gec_11 (33 Vol 2 figures), gec_12 (251 Vol 2 and 135 Vol 3 figures; all 19 text files), fhwa_soils_foundations (165 Vol 2 figures); gec_13 `chapter06.json` title says Volume I. `list_indexed_references` lists four references twice.

## C

- core-14 the 2010 GEC-10 report number resolves to the 2018 text with no word about the edition.
- core-15 `load_chapter` cannot load 17 files `list_chapters` lists (`refs/_retrieval.py:277-283`).
- core-16 four stripped aliases reach the first of several tables (`dm7.clm`, `ufc_structural.pipe_span`, `note`, `footnote`).
- core-17 247 text parameters are advertised as `float` (`app/adapters/_reference_common.py:36-39`).
- core-18 table and equation entries (1,561 and 2,357; 837,000 characters) are outside the text index; `reference_get` drops `implemented_in` and `cross_references`.
- core-19 `agents/references_agent.py` (retired route): unshaped results, a same-name overwrite.

## Patterns

1. Empty means two things ("nothing matches" and "the scope does not exist or failed to load"). 2. First of several, silently. 3. Clamp at the table end; NaN through the same door. 4. The result does not say how it was got (all words or one; which catalog, volume, edition).

## Right, though it looked wrong

Catalog pages: 3,651 of 3,736 rows confirmed on their PDF page by text, no wrong page found (54 rows on pages with no text layer: aashto_1993 29, ufc_expansive 17, gec_12 7, gec_8 1; 23 neighbouring-page rows not looked at). Multi-volume PDFs and pages right. Section tree: no wrong parent. FTS5 syntax characters handled. Size shaping never drops or cuts without a note; no section fetch exceeds the app's 16,000 cap (largest 14,237; two exceed the adapter's 14,000 budget). Near-miss ids raise. `_linterp` tables all ascending. A half-built database is never served. The three empty folders are advertised nowhere except `cited_documents.MODULE_DOCUMENTS` (lines 226, 236, 462). No same-name collision in any adapter. REF-41 holds.
