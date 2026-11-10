# Coverage audit of the reference library (2026-11-10, reviewer `coverage`)

*Copied on 2026-11-10 from the review's working folder, which stays local (`C:/dev/tmp/refs_review_2026-11-10/`); the files named below as links or paths there (lists, scripts, JSON, renders) are not in this repository.*

The owner's question: "I think we may not have digitized all the references and mapped all the figures for
agentic vision calls." Short answer: **the mapping that exists is sound, and it is not complete.**

## Headline numbers

| | Built | Printed | |
|---|---|---|---|
| Chapters and appendices as text | 333 | 514 | 181 printed chapters or appendices have no text; 27 of the 33 references are missing at least one |
| Figures with a catalog row | 3,694 | 4,598 (AASHTO 1993 not counted: a scan with no list of figures) | 904 printed figures cannot be read by `read_reference_figure`; 435 of them are in the geotechnical manuals |
| Catalog rows whose page shows the figure | 3,716 | 3,736 rows | 20 rows lead to a page without the figure; 3 more have the right page but cannot be fetched |
| Printed tables that are a lookup, or typed into the text | about 900 | about 1,950 | about 1,050 printed tables are values the helper cannot get: the vision tool takes catalog figures only |
| Numbered equations present in the chapter text | 1,780 | 2,598 | seven references carry fewer than half (section 7) |

All 3,736 rows resolve (`resolve_pdf`) to a PDF that exists in `docs/`, in the right volume. No row has
`page_estimated`, so the vision prompt never asks the model to confirm the figure is on the page.

The two gaps that matter most to a geotechnical user:

1. **GEC-5 (site characterization) stops at Figure 11-9**: 90 printed figures have no row (the rest of
   chapter 11, all of chapters 12 and 13, Appendix 2). Not in BACKLOG. GEC-13's 207 missing figures are
   REF-08.
2. **Four DM 7.1 rows point at the page after the figure** (1-13, 3-1, 5-4, 6-9): the manual prints its
   captions under the figures, and these four captions fell onto the next page. For Figure 6-9 that page
   shows Figure 6-10 under the 6-9 caption, so the model is asked to read 6-9 and is shown another figure.

## How it was checked

Scripts are in this folder (`s01` to `s19`, `common.py`; Node helpers `toc.js`, `bm.js`, `pk.js`). Nothing
was written to either repo.

- Every page of every PDF in `docs/` (ASCE 7-22 excluded) was cached as text blocks (`s02`): 42 PDFs,
  15,614 pages.
- **Rows on their page** (`s04`, `s06b`, `s06c`, `s14`, `s16`, `s19`): the row's page must hold a line that
  starts "Figure <number>" followed by the caption's first words. 3,466 rows pass in the text layer. 233
  pages whose caption is not in the text layer (scans, and FHWA PDFs where the figure and its caption are
  one image with alt text: GEC-8, GEC-11, UFC 3-220-07, AASHTO 1993) were read with RapidOCR at 150 dpi: 221
  more pass. The last 27 were read line by line (26 are on their page). A second, stricter pass and a
  caption-position pass (is there a drawing or image on the caption's side of the page?) found the rows
  where the caption is on the page and the figure is not; 44 pages were rendered and looked at.
- **Printed figures and tables** (`s07`, `s13`): the List of Figures / List of Tables pages plus caption
  lines in the body. **Tables as data** (`s08`, `s15`): an estimate, explained in section 6.
- **Limits**: AASHTO 1993 is a 624-page scan; only its contents pages and the 29 catalog pages were read by
  OCR, so its printed figures, tables and equations are not counted. The table and equation counts come
  from caption and label patterns, not from reading each one. Figure descriptions (REF-09) and missing
  lookups (REF-10) were not re-examined. No vision model was called.

## 1. Summary by reference

Chapters = printed chapters and appendices with text / printed (reference lists, glossaries and
bibliographies not counted). Figures = distinct catalog numbers / printed. Rows = rows whose page shows the
figure / rows. Tables = lookup or typed in / printed (estimate). Equations = numbered equations in the
chapter text / printed (label and "Eq." patterns). n/c = not counted.

| Reference | Chapters + appendices | Figures catalogued | Rows on their page | Tables as data | Equations in text |
|---|---|---|---|---|---|
| dm7_1 | 8 / 10 | 223 / 223 | **219 / 223** | 40 / 139 | 114 / 116 |
| dm7_2 | 8 / 9 (App B: 1 of 7 examples) | 252 / 252 | 252 / 252 | 43 / 172 | 134 / 136 |
| fhwa_soils_foundations | 11 / 16 | 331 / 392 | 331 / 331 | 99 / 136 | 201 / 255 |
| gec_4 | 9 / 15 | 73 / 90 | 73 / 73 | 17 / 23 | 50 / 50 |
| gec_5 | 13 / 15 | **256 / 346** | 256 / 256 | 78 / 127 | 227 / 283 |
| gec_6 | 10 / 13 | 57 / 86 | 57 / 57 | 17 / 50 | 76 / 77 |
| gec_7 | 10 / 17 | 78 / 89 | 78 / 78 | 38 / 68 | 86 / 88 |
| gec_8 | 8 / 10 | 132 / 132 | 132 / 132 | 6 / 15 | 16 / 18 |
| gec_9 | 14 / 18 | 106 / 106 | **105 / 106** | 12 / 22 | 103 / 103 |
| gec_10 | 19 / 26 | 380 / 381 | **379 / 380** | 24 / 48 | 64 / 65 |
| gec_11 | 11 / 17 | 141 / 148 | **141 / 142** (1 more unreachable) | 26 / 121 | 40 / 52 |
| gec_12 | 19 / 20 | 536 / 536 | **536 / 546** | 193 / 262 | 266 / 266 |
| gec_13 | 10 / 10 | **32 / 239** | 33 / 33 (1 unreachable) | 60 / 82 | 70 / 70 |
| gec_14 | 4 / 8 | 5 / 5 | **2 / 5** | 0 / 0 | 0 / 0 |
| micropile | 10 / 15 | 132 / 142 | 132 / 132 | 31 / 37 | 50 / 53 |
| california_trenching | 11 / 14 | 178 / 178 | 178 / 178 | 9 / 39 | **42 / 421** |
| eurocode_7_1 | 21 / 21 | 23 / 23 | 23 / 23 | 18 / 19 | 47 / 55 |
| eurocode_7_2 | 30 / 30 | 19 / 21 | 19 / 19 | 19 / 22 | 5 / 5 |
| ufc_expansive | **0 / 11** | 42 / 42 | 42 / 42 | 0 / n/c (image pages) | 0 / 15 |
| ufc_backfill | **0 / 8** | 14 / 14 | 14 / 14 | 0 / n/c | 0 / 0 |
| fema_p2082 | 4 / 21 | 32 / 166 | 32 / 32 | 7 / 56 | 31 / 84 |
| fhwa_pavements | 5 / 13 | 179 / 179 | 179 / 179 | 20 / 158 | 3 / 80 |
| aashto_1993 | 9 / 27 | 29 / n/c | 29 / 29 | n/c | n/c |
| ufc_pavement | 21 / 27 | 126 / 126 | 127 / 127 (1 unreachable) | 19 / 28 | 29 / 30 |
| ufc_concrete_practice | 16 / 16 | 9 / 9 | 9 / 9 | 14 / 16 | 1 / 1 |
| ufc_flexible_practice | 8 / 8 | 17 / 17 | 17 / 17 | 29 / 39 | 0 / 0 |
| ufc_stabilization | 6 / 6 | 2 / 2 | 2 / 2 | 10 / 10 | 0 / 0 |
| em_2104 | 5 / 11 | 32 / 32 | 32 / 32 | 11 / 27 | 14 / 14 |
| em_2107 | 6 / 21 | 114 / 173 | 114 / 114 | 5 / 38 | 9 / 64 |
| ufc_structural | 7 / 14 | 13 / 13 | 13 / 13 | 12 / 14 | 0 / 2 |
| ufc_collapse | 8 / 15 | 39 / 87 | 39 / 39 | 17 / 29 | 21 / 22 |
| gsa_collapse | 8 / 12 | 35 / 105 | 35 / 35 | 12 / 34 | 20 / 22 |
| wood_handbook | 4 / 20 | 86 / 244 | 86 / 86 | 18 / 125 | 61 / 151 |

## 2. Printed contents against the text (step 1)

Full table: `gaps_chapters.md`. What a geotechnical user loses most:

- **The worked examples.** Almost no example appendix is text: DM 7.2 Appendix B Examples 1, 2, 4, 5, 6, 7
  (only Example 3 is built); NHI-06-089 Appendix A (the Apple Freeway project carried through the whole
  manual); GEC-4 App A; GEC-6 App B; GEC-7 App C; GEC-8 App B; GEC-9 App B; GEC-11 App E; GEC-5 App 2;
  Micropile App D and E. GEC-10 App A is built in 3 condensed sections; GEC-12 App D is built (89 sections).
- **Design content in appendices**: GEC-7 App B (charts for preliminary design of soil nail walls), GEC-10
  App B (challenging geomaterials) and App C (axial load-deformation closed forms), GEC-11 App B to D
  (pullout, steel dimensions, creep reduction), GEC-4 App B and C.
- **Two manuals have no text at all**: UFC 3-220-07 (expansive soils, 99 pages) and UFC 3-220-04FA
  (backfill, 60 pages). Their lookups are reachable; nothing can be searched or quoted.
- FEMA P-2082: 4 of 21 chapters; none of the commentary (C11, C19 to C22 are the geotechnical part).
- California Trenching: chapter 1, chapter 9 (railroads) and Appendix D (sheet pile sections).
- Complete: GEC-13, both Eurocodes, UFC 3-250-03 / -04 / -11.

Also found: three DM 7.2 chapter files carry titles the print does not use ("Earth Pressures" for the
print's Chapter 2 EXCAVATIONS), and its Chapter 1 is filed inside the Prologue as chapter 0. Detail at the
end of `gaps_chapters.md`.

## 3. Printed figures against the catalog (step 2)

Full list with pages: `figures_printed_not_catalogued.md`.

| Reference | Printed, no row | Which |
|---|---|---|
| gec_13 | 207 | all of chapters 2, 3, 4, 5, 7, 8, 9, 10 (the catalog holds 3 figures of chapter 1 and chapter 6). REF-08 |
| gec_5 | 90 | 11-10 to 11-26, all of chapter 12 (25) and chapter 13 (11), Appendix 2 (37). **New** |
| fhwa_soils_foundations | 61 | Appendix A figures A-1-1 to A-6-10 and A7-2, plus D-10 |
| gec_6 | 29 | Appendix B (21) and Appendix C (8) |
| gec_4 | 17 | A-1 to A-10, B-1 to B-5, D-1, D-2 |
| gec_7 | 11 | 3-22, A-1, **B-1 to B-4 (preliminary design charts)**, C-1 to C-3, G-1, G-2 |
| micropile | 10 | sample problem figures S1-1 to S1-7, S2-1 to S2-3 |
| gec_11 | 7 | B-1, D-6-1, D-6-2, E1-2, E4-1, E4-2, F-1; and Figure 5-11 (catalogued as 5-14, section 4) |
| eurocode_7_2 | 2 | B.4, B.5 |
| gec_10 | 1 | 16-17 (List of Figures only; caption not found in the body) |
| fema_p2082, wood_handbook, gsa_collapse, em_2107, ufc_collapse | 134, 158, 70, 59, 48 | structural commentary and example figures, and the 16 unbuilt wood chapters; each catalog's note says it is a chosen subset |
| aashto_1993 | not counted | 29 rows by hand; the catalog's own note puts the guide at 150 or more |

Catalogued and printed in full: dm7_1, dm7_2, california_trenching, gec_8, gec_9, gec_12, gec_14,
fhwa_pavements, eurocode_7_1, the UFC pavement set, em_2104, ufc_structural, ufc_backfill, ufc_expansive.

**Duplicated numbers.** No number repeats across the volumes of a multi-volume manual: each of gec_11,
gec_12, gec_13 and fhwa_soils_foundations numbers by chapter, and every row carries its own volume's PDF.
Five numbers repeat inside one volume, and `figure_get` returns only the first row
(`_figures_db.py:538`, `LIMIT 1`):

- gec_12 D-1 (6 rows), D-2 (3), D-3 (2), D-4 (3): continuation sheets, section 4.
- gec_11 5-14, gec_13 6-14, ufc_pavement F-21: the print itself numbers two figures the same.

## 4. Every catalog row's page (step 3)

Every failing row: `catalog_rows_failing.md`. 3,716 of 3,736 rows show their figure
on the catalog page.

**A. The page does not show the figure (20 rows).**

| Reference, figure | Catalog page (1-based PDF) | The figure is on | What the vision model is shown |
|---|---|---|---|
| dm7_1 1-13 (karst map) | 76 | 75 (printed 47), offset -1 | a blank page with the caption |
| dm7_1 3-1 (direct shear elements) | 174 | 173 (printed 145), -1 | the caption, then text |
| dm7_1 5-4 (evaluation of existing conditions) | 255 | 254 (printed 226), -1 | the caption, then text |
| dm7_1 6-9 (erosion and stoping) | 334 | 333 (printed 305), -1 | the 6-9 caption, then **Figure 6-10** |
| gec_9 6-17 (ground improvement for lateral resistance) | 100 | 101 (printed 97), +1 | text that cites the figure |
| gec_10 4-5 (crane mounted drilling rig) | 77 | 78 (printed 4-6), +1 | Figure 4-4 |
| gec_14 A-1, A-2, A-3 | 9 | 46, 47, 48 | the contents page that lists them |
| gec_11 "5-14" | 243 | this page is Figure 5-11 | Figure 5-11 under the name 5-14; "5-11" is not found |
| gec_12 D-1 continued (5 rows), D-2 pages 2-3, D-3 page 2, D-4 pages 2-3 | 36, 45, 48, 50 | 37 to 41; 46, 47; 49; 51, 52 | the first sheet, for every continuation sheet |

Reproduced (`s18_repro.py`): `resolve_pdf('dm7_1', '6-9')` -> `ufc_3_220_10_2026.pdf`, index 333;
`resolve_pdf('gec_14', 'A-1')` -> index 8; `resolve_pdf('gec_11', '5-11')` raises KeyError naming 5-1,
8-11, 6-11, 5-31, 5-21. Pages rendered: `png/nographic.png`, `png/nographic2.png`, `png/check3.png`.

**B. The page is right, the row cannot be fetched (3 rows).** gec_13 6-14 "Beam method" (PDF p. 54:
`resolve_pdf('gec_13', '6-14')` returns index 51, the triangular column arrangement); ufc_pavement F-21
"P-22 Crash Truck, plain concrete and RCCP" (PDF p. 265; index 263, the M1078 cargo truck curve, comes
back); gec_11 5-14, the real one (PDF p. 248). A search finds these by caption, hands the helper the
number, and the read returns the other figure with no signal.

**C. The catalog number is not the printed one (7 rows, page right).** ufc_pavement `C145B` is printed
"Figure C14-5B"; `D-3 (Sheet 1 of 4)` to `(Sheet 4 of 4)` are not found as "D-3";
fhwa_soils_foundations `9-37a`, `9-37b` are printed "9-37(a)", "9-37(b)". The not-found error names the
near numbers, so a careful caller recovers.

**Sideways charts (14 rows, page right).** The chart is printed turned on an upright page, so the model
receives it rotated a quarter turn: fhwa_pavements B-3, B-4, B-5, B-6, B-18, B-19;
fhwa_soils_foundations 10-3; gec_5 9-25; micropile 6-6, 6-14, 6-24, 6-36, 7-8, 8-1. (14 more
fhwa_soils_foundations pages are landscape pages with a rotation flag and render upright.)

**No row unresolvable.** With `GEOTECH_REFERENCES_DOCS` unset and the checkout's `docs/` present, all
3,736 rows resolve to an existing file.

What looked wrong and is right: 80 GEC-11 and 80 GEC-8 rows and 39 UFC 3-220-07 rows have no caption in
the text layer; the caption is inside the figure's image and OCR finds it on the catalog page. GEC-5
captions use mathematical alphabet characters, and GEC-8 and the Wood Handbook word a few body captions
differently from their lists; all are on their pages.

## 5. Figures the text points to that the tool cannot reach (step 4)

Full list: `text_figures_without_catalog_row.md`.

| Reference | Figure numbers in the chapter text | With no catalog row | |
|---|---|---|---|
| gec_13 | 241 | 208 | 202 are printed figures; the text lists them section by section and none can be opened |
| gec_5 | 314 | 55 | all 55 printed (chapters 11 to 13) |
| fhwa_soils_foundations | 354 | 7 | appendix and cross-manual numbers |
| gsa_collapse | 22 | 6 | all printed |
| gec_7 | 101 | 3 | 1 printed |
| eurocode_7_2 | 22 | 3 | 1 printed |
| dm7_2, gec_6, gec_10, gec_11, eurocode_7_1, ufc_structural, aashto_1993 | | 1 to 2 each | not found printed: figures of other manuals, or a number the file has wrong |

Every other reference: each figure its text lists has a row.

## 6. Tables (step 5)

Per-reference lists for the geotechnical manuals: `tables_without_data.md`.

How the estimate was made: for each printed table, the text under its caption on its page gives its
distinctive tokens (numbers with a decimal point or three or more digits; shorter numbers or long words
for a table with few). *Typed in* means 60 % or more of those tokens are in the chapter-text sections that
list or cite the table. *Lookup* means a public function named `table_<n>` or citing "Table <n>" in its
docstring. It is a screen, not a cell-by-cell check, and it judges a multi-page table on its first page.

| Reference | Printed | (a) lookup | (b) typed in | (c) partly (25-60 %) | (c) caption only | not mentioned in the text | (d) image |
|---|---|---|---|---|---|---|---|
| dm7_1 | 139 | 10 | 32 | 46 | 47 | 6 | 0 |
| dm7_2 | 172 | 16 | 30 | 50 | 45 | 34 (all Appendix B) | 0 |
| fhwa_soils_foundations | 136 | 0 | 99 | 3 | 8 | 26 | 0 |
| gec_4 | 23 | 4 | 15 | 2 | 4 | 0 | 0 |
| gec_5 | 127 | 0 | 78 | 18 | 14 | 12 | 5 |
| gec_6 | 50 | 10 | 16 | 2 | 3 | 28 (appendices) | 0 |
| gec_7 | 68 | 13 | 36 | 2 | 11 | 17 | 0 |
| gec_8 | 15 | 1 | 6 | 1 | 0 | 8 | 0 |
| gec_9 | 22 | 5 | 12 | 0 | 1 | 8 | 1 |
| gec_10 | 48 | 5 | 24 | 10 | 9 | 4 | 1 |
| gec_11 | 121 | 19 | 13 | 7 | 9 | 76 (appendix examples) | 3 |
| gec_12 | 262 | 9 | 187 | 35 | 33 | 1 | 0 |
| gec_13 | 82 | 12 | 58 | 11 | 11 | 0 | 0 |
| micropile | 37 | 13 | 29 | 3 | 2 | 1 | 0 |
| eurocode_7_1 | 19 | 17 | 7 | 0 | 1 | 0 | 0 |
| eurocode_7_2 | 22 | 13 | 15 | 1 | 0 | 2 | 0 |
| california_trenching | 39 | 7 | 2 | 2 | 3 | 25 | 0 |

(a) and (b) overlap, so the "as data" column of section 1 is their union; the four columns to the right
count only tables with no lookup. The remaining references are in `table_summary.json`.

- **DM 7.1 and DM 7.2 are the weak point**: 99 and 129 printed tables are neither a lookup nor typed in.
  Their chapter text is a condensed summary, so a table's numbers are mostly not there. REF-34 covers
  DM 7.2; DM 7.1 is the same.
- Tables printed as images are few (GEC-5 five, GEC-11 three, GEC-10 and GEC-9 one each, and all of UFC
  3-220-07). NHI-06-088/089's two picture tables (4-8, 10-1) are already typed in.

**Is there any route to a printed table or an uncatalogued page?**

- **The references helper: none.** `build_references_subagent` (`funhouse_agent/deep/agent.py:236-248`)
  gives it `call_agent` / `list_methods` / `describe_method` over the reference modules, `calculate`, and
  from the vision kit only `read_reference_figure` (`include={"read_reference_figure"}`). That tool takes
  `(reference, figure_number)` and nothing else (`reference_vision.py:205-213`): no page argument, no
  table argument. A table with no lookup and no typed values cannot be read, and a "Table" request returns
  the figure not-found error.
- **The main agent has two side doors, neither described to it** (read from code, not run):
  (1) `read_pdf_text` and `analyze_pdf_page` open a reference PDF by its bare file name when the file is
  in a reference folder on disk (`aec_agent_core/vision_tools.py:627-656`, `find_readable_file`); it needs
  the file name and page, which the helper's prose does not carry, and on a host that fetches PDFs on
  first use the file is there only after a figure from it was read. (2) `view_worked_example_source`
  takes a free `pdf_page` (`reference_vision.py:124`), so any page of the 11 PDFs that are the source of a
  worked example can be rendered and read.

## 7. Equations (step 6)

Counts are in section 1 (numbered equations found printed, by label lines such as "(5-2)", "[Eq. 2-1]"
and by "Eq." / "Equation" citations, against the chapter JSON). References where fewer than half are in
the text:

| Reference | In text / printed | Why |
|---|---|---|
| california_trenching | 42 / 421 | the manual numbers by section ("(4-3-2)"); most are in the worked examples of chapters 7 to 10 |
| fhwa_pavements | 3 / 80 | 5 of 8 chapters built, as summaries |
| em_2107 | 9 / 64 | chapters 7 to 16 and the appendices have no text (lookups cover chapter 10 and Appendices D, F) |
| fema_p2082 | 31 / 84 | 4 chapters built |
| wood_handbook | 61 / 151 | 4 of 20 chapters built |
| ufc_expansive | 0 / 15 | no text (5 equation lookups exist) |
| ufc_structural | 0 / 2 | |

In lookups (function name or docstring cites the number): dm7_1 114, dm7_2 130, wood_handbook 67,
ufc_pavement 25, california_trenching 21, gsa_collapse 17, ufc_collapse 16, em_2104 12, em_2107 11;
under 10 elsewhere. The FHWA manuals' equations live in the text, not in functions (REF-10).
The NHI-06-088/089 printed count (255) includes some table ranges read as labels; its 201 tagged
equations are what REF-29 built.

## 8. PDFs not digitized, and the three empty folders (step 7)

- **`geotechnical-tgm.pdf`**: FHWA Federal Lands Highway, DRAFT Geotechnical Technical Guidance Manual,
  May 2007. 291 pages, a text PDF (6 pages without text), bookmarked: Sections 1 to 6 (introduction,
  guidance and references, investigations, analysis and design, documentation, consultants) and
  Appendices A to C (source lists, topical bibliography). About 55 "Exhibits" stand in for figures and tables
  (boring criteria, test selection, allowable bearing pressures, factor-of-safety criteria). It is mostly a
  guide to which primary source governs what. Building it: `build_printed_text.py` reads it as is; the
  exhibits need a catalog keyed "Exhibit 3.1-B", which the builder and `figure_get` do not handle today.
  About a day. Untracked in git.
- **`nchrp_syn_467.pdf`**: NCHRP Synthesis 467, Visualization of Geotechnical Data for Hazard Mitigation
  and Disaster Response (TRB, 2015). 89 pages, text with 15 image-only pages, 52 figure labels, no numbered
  tables or equations. A survey of software and practice, with no design method or value. A TRB
  publication, so check its terms before committing it. Lowest value of the three. Untracked.
- **`ufc_3_260_02_2001.pdf`**: UFC 3-260-02 Pavement Design for Airfields, 30 June 2001. 538 pages, text
  PDF, 21 chapters and Appendices A to N, about 180 figure captions and about 200 equation labels (CBR and
  layered elastic flexible design, plain / reinforced / fibrous / prestressed concrete, overlays, frost).
  **Tracked in git** (commit 6869631) and named by no catalog or manifest; `ufc_pavement` was once coded
  from it by mistake and rebuilt from UFC 3-250-01. Building it is the largest job here, the size of
  UFC 3-250-01 twice over, and of use only for airfield work.
- The ASCE 7-22 copy is in `docs/`, untracked, and was not opened. The 2025 EM 1110-1-1905 is the lead's.

**The three empty folders** (`fema_p2192/`, `noaa_frost/`, `ufc_dewatering/`): not tracked
(`git ls-files` returns nothing); they are leftovers on disk. The packages were added 2026-02-24 (commits
d01c0ee, 5c6ac4b: FEMA P-2192 tables, NOAA frost depth tables and equations, UFC dewatering tables and
equations) and deleted 2026-06-05 by d4b9679, "Remove low-value/incorrect references". The library and
`scripts/README.md` say they are gone. **Four app strings still advertise NOAA**:
`funhouse_agent/deep/agent.py:162` (the references helper's description, read by the main agent),
`funhouse_agent/agent.py:101`, `funhouse_agent/native_tools.py:450`, `funhouse_agent/reviewer.py:6`. The
only NOAA content left is DM 7.1 Figure 1-11 (frost penetration map). FEMA P-2192 and dewatering are not
advertised anywhere current.

## 9. The app side (step 8)

- **Reachability**: all 33 packages are reachable. Each has an adapter in `REFERENCE_MODULES`
  (`dm7` covers dm7_1 and dm7_2); `reference_db.list_indexed_references` returns every package that has
  `text/`; `figure_db.list_indexed_figures` returns all 33 catalogs. Every public function of every
  package is an adapter method, except 11 DM 7.2 chapter 7 functions that take a function as an argument
  (FOSM, Monte Carlo), which a JSON call cannot pass.
- **Text reachable only through `reference_db`**: the `dm7` and `ufc_pavement` adapters register lookups
  only (`funhouse_agent/adapters/ufc_pavement_adapter.py`, `dm7_adapter.py`), so their 903 and 317
  sections have no `list_chapters` / `retrieve_section` of their own. A helper that lists the `dm7`
  module's methods sees no text tools; `reference_search` and `reference_get` do reach the text.
- **Advertised, not built**: NOAA (section 8).
- **Figure search by chapter fails silently for three catalogs** (reproduced, `s18_repro.py`):
  gec_7 (78 rows) and gec_8 (132 rows) have `chapter: null` on every row, so
  `figure_search('bond strength', reference='gec_7', chapter=4)` returns `[]` where the same search
  without `chapter` returns five figures; gec_4's `chapter` is the figure's own number (Figure 25 is
  "chapter 25"), so `chapter=5` returns Figure 5 only. An empty list reads as "no such figure". Not in
  BACKLOG.

**A host that installs only the wheel.** The wheel ships `*/text/*.json`, `*/figures_catalog.json` and
the schema (`pyproject.toml` package-data); no PDF. `read_reference_figure` then does this
(`reference_vision.py:215-232`, `_figures_db.py:594-638`, `reference_docs.py`,
`webapp/reference_fetch.py`):

1. `figure_get` finds the row (the catalog is in the wheel).
2. `resolve_pdf` looks for `<GEOTECH_REFERENCES_DOCS>/<file name>` when that variable is set, else
   `<site-packages>/docs/<file name>`, which does not exist. It raises FileNotFoundError.
3. `reference_docs.fetch(file name)` then looks in the cache (`GEOTECH_REFERENCES_CACHE`, else
   `<temp>/geotech_reference_pdfs/`), and if a fetcher is registered calls it once per file. The geotech
   page registers one when SharePoint is configured: it downloads
   `<SharePoint base>/primary_references/<file name>` (folder name from
   `GEOTECH_REFERENCES_SHAREPOINT_DIR`) into the cache. On Foundry there is no fetcher; the PDFs must be
   in a folder named by `GEOTECH_REFERENCES_DOCS`.
4. If the file is nowhere, the helper is told: `source PDF for <reference> <figure> not found:
   <path>. Set GEOTECH_REFERENCES_DOCS to the folder containing the reference PDFs.` and, with a fetcher,
   `It is not available from SharePoint primary_references/ either (looked for '<file name>'; the name
   must match exactly).` That is an error result, not a wrong value.

The file name is the base name of the row's `pdf_path`, the same for every reference, and case matters on
Linux. The 38 names: `aashto1993.pdf`; `California Trenching and Shoring Manual.pdf`;
`ufc_3_220_10_2026.pdf` (dm7_1); `ufc_3_220_20_2025.pdf` (dm7_2); `EM_1110-2-2104_2025.pdf`;
`EM 1110-2-2107_2022.pdf`; `eurocode_7_-_geotechnical_designen.1997.1.2004.pdf`; `en.1997.2.2007-1.pdf`;
`fema_p2082_2020_nehrp.pdf`; `FHWA-NHI-05-037 - Geotech Pavements.pdf`;
`FHWA NHI-06-088 Soils and Foundations Vol 1.pdf` and `FHWA NHI-06-089 Soils and Foundations Vol 2.pdf`;
`GEC 4.pdf` to `GEC 10.pdf` and `GEC 14.pdf` (one each); `GEC 11 Vol 1.pdf`, `GEC 11 VOL 2.pdf`;
`GEC 12 vol 1.pdf`, `GEC 12 Vol 2.pdf`, `GEC 12 Vol 3.pdf`; `GEC 13 Vol 1.pdf`, `GEC 13 Vol 2.pdf`;
`GSA_Alternate_Path_2016.pdf`; `FHWA Micropile Manual.pdf`; `ufc_3_220_04fa_2004.pdf`;
`ufc_3_220_07.pdf`; `UFC_4-023-03.pdf`; `ufc_3_250_04_2024.pdf`; `ufc_3_250_03_2018.pdf`;
`ufc_3_250_01_2016.pdf`; `ufc_3_250_11_2020.pdf`; `UFC_3-301-01_2023_c4.pdf`;
`USDA_Wood_Handbook_FPL-GTR-282.pdf`. The Wood Handbook PDF (104 MB) is in `.gitignore`, so a checkout
does not have it either. The page's connection diagnostics (`webapp/geotech_page.py:328`) lists every
missing name.

## 10. What to build first, for a geotechnical user

| # | What | Why | Size |
|---|---|---|---|
| 1 | Correct the 20 wrong-page rows and make the 3 shadowed rows fetchable | DM 7.1 6-9 and the two same-number pairs return another figure with no signal | 23 rows; under a day with a test that every row's page holds its caption and a drawing |
| 2 | GEC-5 catalog: Figures 11-10 to 13-11 and Appendix 2 | 90 site-characterization figures (chapter 12 is the hazards chapter) unreadable; the text already lists 55 of them | one builder run and a page check; 90 rows |
| 3 | GEC-13 catalog (REF-08) | 207 figures of a manual the ground-improvement module cites | one builder run over two volumes; 207 rows |
| 4 | A way to read a printed table or page: a table catalog (number, caption, page) and a page argument on the read tool, for the helper | about 1,050 printed tables are neither lookup nor text, 228 of them in DM 7.1 / 7.2 (REF-34) | catalog from the Lists of Tables is mechanical (pages are in `tables.json`); tool change and tests: 2 to 3 days |
| 5 | Appendix figures into the catalogs: GEC-7 B-1 to B-4 and the rest, GEC-4, GEC-6, NHI-06-089 App A, Micropile, GEC-11 | 135 figures, the GEC-7 preliminary design charts among them | builder change to read appendix captions; 135 rows |
| 6 | Worked-example appendices as text: DM 7.2 App B (6 examples), NHI-06-089 App A, GEC-6 App B, GEC-4 App A, GEC-11 App E, Micropile App D and E, GEC-9 App B, GEC-8 App B, GEC-5 App 2, GEC-7 App C | the examples are how an engineer checks a method; about 650 printed pages | `build_printed_text.py` per appendix; 1 to 2 days each for the large ones |
| 7 | GEC-10 App B and C, GEC-11 App B to D, GEC-4 App B and C as text | design content: challenging geomaterials, load-deformation closed forms, pullout and creep factors | about 80 pages |
| 8 | UFC 3-220-07 (expansive soils) as text | 99 pages with no text; 32 pages are images and need reading by eye | 2 to 3 days |
| 9 | FEMA P-2082 commentary C11, C19 to C22 and chapter 22; DM 7.1 / 7.2 tables as data | site class and site-specific procedure commentary; the DM7 tables if item 4 is not built | about 95 pages; the tables are a week by hand |
| 10 | Tidy: catalog `chapter` values (gec_7, gec_8, gec_4), DM 7.2 chapter titles and chapter 1, the NOAA wording in four prompts, the three empty folders, `ufc_3_260_02_2001.pdf` (build or remove) | each misleads a search or a reader | hours |

Lower: California Trenching chapters 1 and 9, the FLH guidance manual, AASHTO Part III, UFC 3-220-04FA
text, the airfield UFC, the structural references' appendices.

## Files in this folder

- `catalog_rows_failing.md`: every failing row.
- `figures_printed_not_catalogued.md`, `text_figures_without_catalog_row.md`, `figures_catalogued_not_found_printed.md` (the last is the text scan's raw list; OCR settled it).
- `tables_without_data.md`, `tables.json`, `table_summary.json`.
- `gaps_chapters.md`.
- `rows_final.json` (every row's result), `summary.json`, `equations.json`, `printed_sets.json`,
  `ocr_items.json` (OCR of 254 pages), `png/` (rendered pages), `s18_out.txt` (reproductions).
