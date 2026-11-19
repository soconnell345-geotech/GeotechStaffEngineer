<!-- Filed here by the lead on 2026-11-18 from planlens' branch `claude/audit-reading` (planlens keeps no review file on its main). Repairs 2 and 3 (e359bd8, c466ccf) are merged to planlens' main on 2026-11-18 after a green run of the whole suite on GitHub (after 0.17.0, not released, no tag). Repair 1 (db75390, a bare number in a footer read as the printed page) is HELD on the branch: it takes the first bare number in a band, and it needs a sequence check across pages and a count on real reports first (the app's BACKLOG PL-46). So "fixed" in row 1 below is true of the branch, not of main. -->

10 findings, 3 fixed here, 2 suspected and not reproduced

# Audit: how planlens reads a report (run 2026-11-19 03:13-03:31 UTC, from main f9b5b38)

Every finding below comes from a made-up PDF built with PyMuPDF and read back through the
review tools as an agent calls them (`planlens.review` toolkit via the test host:
`open_document`, `read_document`, `search_document`, `document_structure`, `document_page_map`).
Made-up words and numbers only (Zorvex Ridge, Quillfen, blorite, QX-n borings).

| # | File and function | The made-up page | What the agent is handed | What is on the page | Fixed / noted | How often |
|---|---|---|---|---|---|---|
| 1 | `document/structure.py` `printed_numbers` | 17-page report, footer "Quillfen Associates" left and a bare "3" centred | no `printed_page` on any page, no `printed_pages` on any segment; `read_document` and hits cite only the counted page | printed 1-10 at the foot | **fixed** (db75390): a whole number <= 999 alone in the header/footer band, not on a sheet or a band saying SHEET | nearly every report (bare footer numbers are the common form) |
| 2 | `document/search_text.py` `fold` | "The contractor's schedule" set with a curly apostrophe; "-1.2" with a minus sign; "2-4 ft" with an en dash | `search_document("contractor's schedule")` -> 0 hits, hint points at the scan and the cover | the words, on page 4 | **fixed** (e359bd8): curly quotes, prime, minus, en dash fold to ASCII on both sides; hits still quote the page | every word-processor report |
| 3 | `review/conversation.py` `_is_pages_key` | "Page 1 of 3" footers | segment `printed_pages "1-3"` plus `pdf_printed_pages "2-4"` | printed 1-3; no page prints 2-4 | **fixed** (c466ccf): `printed_` keys get no viewer twin | every report whose numbers are read (all of them after #1) |
| 4 | `document/document.py` + `reading_order.py` (PL-09, `column_warning` off by default) | two columns drawn row by row in the content stream | "Left0 / Right0 / Left1 / Right1 ...": two texts shuffled, no warning | two columns | noted (the check exists; the host must ask for it) | some two-column reports (depends on the writer) |
| 5 | `document/document.py` `search` | "drainage" last word of page 4, "blanket" first of page 5 | `search_document("drainage blanket")` -> 0 hits; hint says the term may be on a scan | phrase across the page break | noted (search groups are per page; the hint does not say a phrase can cross a page) | any long phrase search in narrative |
| 6 | `document/structure.py` (int `printed_page`) | TOC page "ii", appendix "A-1", "A-2" | no printed number; the "ii" page joins segment "1-8"; restart from roman to arabic not split | ii, A-1, A-2 | noted (model holds an int; roman / letter-dash numbers need a str field) | most reports with front matter or appendices |
| 7 | `document/document.py` `search` | "a capacity of 1,250 kips" | `search_document("1250 kips")` -> 0 hits | 1,250 | noted (thousands separator not folded; `regex` or the printed form finds it) | engineering reports, often |
| 8 | `tools/toolkit.py` search `pages_not_searchable_as_text` | cover page with 4 words of text | the cover listed as "no usable text layer", so a miss "may be" on it | a text cover | noted | every report with a sparse cover |
| 9 | `document/document.py` page reading | a garbage text layer (mojibake "Ã¢â‚¬ ¿¿¿") | kind `mixed`; only the 4 U+FFFD characters are flagged; the rest is handed as text | unreadable text layer | noted | poorly OCR'd scans, now and then |
| 10 | `read_document` text | header and footer on every page | running header, footer and page number repeated in every page's text, in content-stream order (footer before body when drawn first) | same | noted (not false; costs budget, and a footer line can sit above the body) | every report |

Checked and correct (no finding): columns drawn column by column; side note; figure caption
between two paragraphs; footnote; landscape page; /Rotate 90 page (text upright, rotation 90 in
the page map); ruled table over three pages with repeated heading (three tables of 14 x 5,
rows and columns as printed, empty cell kept empty, thousands separators and minus signs kept,
merged heading "Laboratory results" in its own row); hyphen at a line end ("Excava-/tion" found
as "excavation"); other case; ligature (fi, fl); no-break space; word inside a table cell;
blank page (kind `blank`) and scanned page (kind `scanned`, `needs_ocr`, listed in
`pages_to_view` and `pages_without_text_layer`); a figure-only page (kind `mixed`, no text).
Unruled table and two tables side by side: no table is made; the cells come one per line, in
row order, interleaving the two side-by-side tables row by row (not false, but noted here).
Tables continued over pages are not linked (`continued_tables` is a host option, off).

## Size (300 pages, 45 lines each)

`open_document` 3.2 s. `read_document` with no pages returned pages 0-3 (11,281 chars) with
`next: {pages: "3-299", start_line: 16}` and a note saying the size limit was reached: the cut
is said and the rest can be asked for. `search_document` stopped at `max_hits` 100 with
`max_hits_reached: true`, and cut the list to 68 hits for size with `hits_omitted_for_size` and
a hint to narrow `pages=`. A word on page 287 was found (0.0 s). `document_structure` 0.0 s.

## Suspected, not reproduced

1. A ruled table whose second page has no repeated heading row may be read as a different
   table (not built: my continued table repeated its heading).
2. `pages_by_kind` classes text report pages `mixed` when they carry ruling lines (header rule,
   table); I did not find what the agent loses by it.

## What I ran

- Made-up PDFs: a 17-page report (cover, roman TOC, two columns, side note, figure + caption,
  footnote, search page, 3-page ruled table, landscape, rotated, unruled and side-by-side
  tables, A-1, blank, scan, figure-only, garbage text); a 300-page report; a one-page
  row-by-row two-column page. 23 searches, 6 `read_document` calls, `document_structure`,
  `document_page_map`, `open_document` on each.
- New tests: `planlens/document/tests/test_audit_reading.py` (10), `planlens/review/tests/test_audit_reading.py` (2);
  each fails before its fix (6, 1 and 2 failures).
- `planlens/document/tests` after fix 1-2: one existing failure found
  (`test_pl37_printed_numbers.py::test_a_sheet_count_is_never_the_sheet_number`, a "SHEET / 3 /
  OF 24" band), fixed by not reading a bare number in a band that says SHEET.
- Whole suite (`pytest planlens`, 4 workers): 2758 passed, 5 skipped, exit code 0 (146 s), with all three fixes, pinned-surface test included, nothing re-recorded.

## Files read

Read whole: none of the listed files end to end in the time. Read in the parts that decide
the findings: `document/search_text.py` (lines 1-90 of 135), `document/structure.py`
(`printed_numbers`, band texts, constants), `document/reading_order.py` (module notes and
constants), `document/document.py` (`search`, page summary build), `review/tools.py` (lines
40-215), `review/conversation.py` (`_viewer_pages`), `tools/toolkit.py` (search trailer),
`review/DESIGN.md` (first 60 lines). Not reached: `pdf/extractor.py`, `pdf/labels.py`,
`document/pdf_text.py`, `model.py`, `budget.py`, `findlike.py`, `loggrid.py`, `ir/ingest.py`,
`ir/queries.py`, `ir/results.py`, `review/specs.py`, `review/find_like.py`, `review/host.py`.
