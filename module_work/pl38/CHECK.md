<!-- Moved from planlens branch claude/pl-38-printed-numbers (task P2a), where it sat at the repo root; planlens keeps no records. -->

# CHECK: PL-37 `printed_numbers` and PL-38 (task P2a, planlens)

Branch `claude/pl-38-printed-numbers` from planlens main cc2a728. Pieces 2
and 3 of "Task P2" of `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`.
Piece 1 (the rest of PL-37: the index at open, sheet numbers as addresses,
scale and markup ids in the rows) is task P2b and is not built here. Both
pieces are repairs, for every host; there is no setting.

**Scope.** The files of P2's row in "Who edits what" that I changed:
`planlens/document/structure.py`, `sheets.py`, `document.py`,
`planlens/tools/toolkit.py`, `planlens/review/marked_copy.py`,
`planlens/review/tools.py`, `CHANGELOG.md`, `planlens/document/DESIGN.md`,
`planlens/review/DESIGN.md`; and `planlens/review/conversation.py`, **added
by the lead's leave**: the viewer-page note lives there and nothing on P2's
list sees the result on the app's path, and the list was there to keep
parallel tasks apart while no other planlens branch is being built. Commits:
16c738b (piece 2), 73d8e2b and e780242 (piece 3), and this file's.

## Confirmed on today's code (cc2a728)

- **`printed_numbers`** (`planlens/document/structure.py`), called only by
  `Document.summary` with the text of the top and bottom 8 % of the page
  (`band_texts`). Its one pattern `_SHEET_ID` missed "SHEET NO. S-201A" (the
  trailing letter fails its `\b`), "DWG. NO.: A-101" (the colon),
  "DRAWING NO." (not in it) and a value in the cell beside or under its
  label; a title block down the right edge was never read; "SHEET 3 OF 24"
  became `sheet = "3 of 24"`; and a note "1. REFER TO DWG NO. S-501" in the
  footer gave `sheet = "S-501"`. Each is a new test that fails on cc2a728
  (below). Where its result goes: `PageSummary.sheet` -> the page-map row's
  `sheet` (`document_page_map`), and `structure.segments` -> a drawing
  segment's `sheets` list and its title "Drawing sheets <first> .. <last>"
  (`document_structure`, and the segment titles `open_document` returns).
  Nothing in the app reads `PageSummary.sheet` itself (grep of `packages/`,
  `webapp/`, `funhouse_agent/`, `report_ingest/`, `tinyapps/`: the app's
  `cited_pages` reads `printed_page`, the label, header and footer); the
  model reads it in those three results.
- **Two annotate calls on one copy** (`planlens/review/marked_copy.py`
  `_write_marked_copy`): copy the marked copy to a temporary file, add the
  marks, `os.replace` it in, with no lock. Two calls at once lose the first
  swap's marks; the new test reproduces it on cc2a728 ("made-up mark three"
  missing, both results saying `n_written: 1`).
- **Viewer page numbers near the limit** (`planlens/review/conversation.py`
  `with_viewer_pages`, called by `dispatch_document_tool`): a result the
  `pdf_page` additions would push past the limit is returned as it came,
  with no word. Confirmed in the code, and by the new test that fails on
  cc2a728 (below). `conversation.py` is not in P2's file list and no file
  that is sees the result on the app's path (the app calls
  `conversation.dispatch_document_tool` directly); the lead gave leave to
  edit it for this repair.
- **Copies of the file in an annotate call** (`planlens/tools/toolkit.py`
  `_tool_annotate_document`): `entry.doc.tobytes()` serialises the whole
  file on every call, also when adding to an existing copy, where
  `write_markups` never reads it (`base = output if appended else source`).
  `write_markups` then opens its base from bytes and saves it (about one
  more copy while it saves); `markup_check.describe_copy` and `check_marks`
  each read the whole copy into memory, one after the other.
- **Open and the text caches** (`planlens/document/document.py`): opening
  runs `page_map()` on every page; `_text`, `_pages`, `_summaries` and
  `_annots` are plain dicts with no bound.

## PL-38 measured (before and after, same generator and seed)

Made-up scanned sets: 17 x 11 in pages, each one 1700 x 1100 px grey image
of random noise (incompressible), seed 7: 27 pages = 50.5 MB, 80 pages =
149.6 MB. Both measured (4.0-4.4 GB free at each run). Each line is one
fresh process: read the file (the app holds an upload as bytes), configure
a `ReviewHost` with a look that always confirms, `review.call(
"open_document")` on it as an attachment, then 0, 1 or 2
`review.call("annotate_document")` calls adding one box each to
`set_marked.pdf` (the second adds to the first's copy). Peak = psutil
`peak_wset` (counts MuPDF's native memory), cumulative in the process.

| set | step | before: time, peak | after: time, peak |
|---|---|---|---|
| 50 MB | file read (+ imports) | -, 75 MB | -, 76 MB |
| 50 MB | open | 0.34 s, 168 MB | 0.33-0.35 s, 168 MB |
| 50 MB | one annotate | 0.37 s, 273 MB | 0.20 s, 243 MB |
| 50 MB | second annotate on the copy | 0.37 s, 277 MB | 0.21 s, 246 MB |
| 150 MB | file read (+ imports) | -, 174 MB | -, 175 MB |
| 150 MB | open | 0.56-0.60 s, 367 MB | 0.45 s, 367 MB |
| 150 MB | one annotate | 1.13-1.20 s, 673 MB | 0.35-0.36 s, 541 MB |
| 150 MB | second annotate on the copy | 2.08 s, 677 MB | 0.49 s, 543 MB |

Where it goes (150 MB, a thread sampling resident memory every 5 ms inside
each step, over the process after reading the file): open's `page_map`
+167-179 MB, which stays; one annotate's peak: before, `tobytes` +344 and
`write_markups` +490 (the open's +187 plus two copies of the file); after,
`write_markups` +340, `describe_copy` +340 and `check_marks` +351 (one copy
each, one after the other). Emptying MuPDF's resource store
(`pymupdf.TOOLS.store_shrink(100)`) at the end took the open's growth from
+94 to +38 MB (50 MB set) and from +193 to +38 MB (150 MB set): most of what
open keeps is MuPDF's store of decoded images, which is one per process and
shared by every document, not per document.

**Extrapolated to 300 MB, linear in file size from the two sets (an
extrapolation, not a measurement):** open 0.9 s; process peak after open
about 670 MB (about 300 of it the app's own upload bytes; the store part
stops at the store's limit, MuPDF's default 256 MiB as I understand it, not
measured here); one annotate before 2.3 s and a peak of about 1,280 MB (the
call's own growth over the open document about 610 MB), after 0.6 s and
about 990 MB (its growth about 320 MB); the second call on the copy before
4.7 s, after 0.9 s.

**Text caches** (the scanned set has no text, so they were measured apart):
a made-up 400-page letter report, 60 lines of made-up words a page
(0.7 MB file): open 1.13 s and +28.6 MB; a search, nothing more; reading
every page in full, +7.3 MB more. About 90 KB a text page with everything
cached, nothing for a scanned page.

Which bound I chose, and why, is under "Calls made".

## What changed

Piece 2, `printed_numbers` (commit 16c738b):
- `planlens/document/sheets.py`: the piece-finding of `_read_page` moved,
  unchanged, into `_pieces(page, lines, width, height, where=None)`; new
  `page_sheet_number(lines, width, height, is_sheet, where)` reads one page
  by the index's rules for a sheet no slot covers: `_by_label`, then on a
  sheet `_by_size`. The slot needs the whole set and stays the index's.
  `sheet_index` results are unchanged (its tests pass as they were).
- `planlens/document/structure.py` `printed_numbers(texts, lines=None,
  width=0, height=0, is_sheet=False)`: the page number as before; the
  number by `page_sheet_number`, on a sheet in the title-block bands (right
  32 %, bottom 24 %, the index's) and the header band, on any other page in
  the header and footer bands as before; without lines, in the texts (a
  label's value on its line, or the next text after a label alone). The
  old "SHEET A-101" form (a bare SHEET, SHT or DWG starting a line) is still
  read. A count goes to `sheet_count`, never to `sheet`; a page's own
  "Page 3 of 24" is masked before a count is looked for.
- `planlens/document/document.py` `summary`: passes the page's lines, size
  and whether it is a sheet (drawing sheet, or 11 x 17 in or larger: the
  index's yardstick); keeps the count in `evidence["sheet_count"]`.
- `planlens/tools/toolkit.py` `_tool_document_page_map`: a row with a
  count carries `sheet_count` (taken out of `evidence` when
  `with_evidence`, so it is said once).

Piece 3, PL-38 (commit 73d8e2b):
- `planlens/review/marked_copy.py`: `copy_lock(path)`, one re-entrant lock
  per marked copy (normalised absolute path), in this process, dropped when
  no call holds or waits for it. `write_marked_copy` holds it for the write.
- `planlens/review/tools.py` `annotate_document`: holds the same lock from
  naming the marks to remove by their words, through the write, to
  `describe_copy`; the look's check (`check_marks`, model calls) runs after
  it is released. The locked part is `_annotate_locked`, moved code.
- `planlens/document/document.py`: `Document.pdf_bytes()`, the bytes the
  document was opened from (PyMuPDF keeps that very object as
  `Document.stream`; an image's is the PDF it became), else `tobytes()`.
  `render(marks=...)` draws on a copy opened from it instead of from
  `tobytes()`.
- `planlens/tools/toolkit.py` `_tool_annotate_document`: writes from
  `pdf_bytes()`, and, when the marks go onto an existing copy (append), from
  the document's file name (not read) where it has one.
- `planlens/review/conversation.py` `with_viewer_pages` (commit e780242):
  where the `pdf_page` additions would push a result past the limit, it
  goes out without them and with `viewer_pages` added through `_with_note`
  (`VIEWER_PAGES_LEFT_OUT`: "the pages in this result count from 0, as the
  tools take them: it was too near its size limit to add pdf_page beside
  each, so a PDF viewer shows page n as page n + 1"), only where a page
  number was left out and the note still fits. A result under the limit is
  byte-identical; one with no page to add to goes out as it came.

## Made-up example

- Piece 2. A 36 x 24 in sheet whose right-hand title block prints
  "SHEET NO. S-201A" (at 85 % of its height) and whose bottom edge prints
  "SHEET 3 OF 24". `document_page_map` row before: `"sheet": "3 of 24"`;
  after: `"sheet": "S-201A", "sheet_count": "3 of 24"`. The drawing
  segment's `sheets` were `["3 of 24"]`, now `["S-201A"]`.
- Piece 3. `set_marked.pdf` holds "mark one"; two annotate calls at once
  add "mark two" and "mark three". Before: the copy holds one and two, and
  both calls said `n_written: 1`. After: it holds all three.
- Piece 3, the note. A 40-page page map under a host cap that leaves room
  for the note but not for 40 `pdf_page` keys. Before: 40 rows with no
  `pdf_page` and nothing said. After: the same rows and
  `"viewer_pages": "the pages in this result count from 0, ..."`.

## Tests

New, each failing on cc2a728 (run there from a `git archive cc2a728`
export with the new files copied in; reason in brackets) and passing here:

- `planlens/document/tests/test_pl37_printed_numbers.py`:
  `test_a_label_and_number_on_one_line_in_a_right_title_block` (None, not
  S-201A); `test_the_dwg_no_forms` (None); `test_a_label_and_its_value_in_
  separate_cells` (None); `test_a_sheet_count_is_never_the_sheet_number`
  ("3 of 24" as sheet); `test_a_note_naming_another_sheet_is_not_the_number`
  (S-501, not A-101); `test_a_large_number_alone_in_the_corner_of_a_sheet`
  (None); `test_the_page_map_row_keeps_number_and_count_apart` (no
  `sheet_count`).
- `planlens/review/tests/test_pl38_large_files.py`:
  `test_two_annotate_calls_on_one_copy_keep_both_calls_marks` (mark three
  lost); `test_the_copy_lock_is_one_per_file_and_re_entrant` (no
  `copy_lock`); `test_a_document_opened_from_bytes_is_not_copied_again` (no
  `pdf_bytes`); `test_annotate_writes_the_copy_without_another_whole_copy`
  [bytes] and [file] (`tobytes` called on every write).
- `planlens/review/tests/test_pl38_viewer_pages_note.py`: on cc2a728 the
  file does not import (`VIEWER_PAGES_KEY` is new). Run there with the two
  names defined in the test instead:
  `test_page_numbers_left_out_near_the_limit_are_said_to_be` and
  `test_a_document_tool_near_the_host_cap_says_so` fail (no note);
  `test_a_result_under_the_limit_is_what_it_was` and
  `test_no_note_where_no_page_number_was_left_out` pass there, as they
  should: they pin what must not change.

Existing assertions edited, because the repair makes them untrue (each
asserted the count as the sheet number):
`planlens/document/tests/test_structure.py` lines 58 (`s.sheet` is now
None, the count in `evidence["sheet_count"]`), 90 (the drawing segment has
no `sheets`: counts only), 121 (`printed_numbers(["SHEET", "2", "OF 7"])`
is `{"sheet_count": "2 of 7"}`); `planlens/tools/tests/test_structure_tools.py`
line 63 (the row has `sheet_count`, no `sheet`).

Runs (PowerShell, `PYTHONPATH` = this checkout, `planlens.__file__`
printed inside it each time; `C:\dev\GeotechStaffEngineer\.venv\Scripts\
python.exe -m pytest -q -p no:cacheprovider <files>`):

- 26 files: the three new ones; the brief's `planlens/tools/tests/
  test_sheet_tools.py`, `planlens/review/tests/test_sheet_tools.py`,
  `test_host_specs.py`, `test_end_to_end.py`, `test_marked_copy.py`,
  `planlens/document/tests/test_structure.py`, `test_sheets.py`; those of
  what I changed (`test_document.py`, `test_markup_writer.py`,
  `test_structure_tools.py`, `test_toolkit.py`, `planlens/review/tests/
  test_tools.py`); and the ones that read summaries, segments or copies
  (`test_crossrefs.py`, `test_conversation.py`,
  `test_host_per_conversation.py`, `test_page_numbers.py`,
  `test_find_like_tool.py`, both
  `test_pl39_misleading_results.py`, `test_markup_check.py`,
  `test_text_reliable.py`, `test_duplicates_distinct_sheets.py`,
  `test_markup_divider_and_ids.py`): **673 passed, 2 skipped**, exit 0.
- The app as a consumer, from `packages/aec-document-review`, `PYTHONPATH`
  = this checkout + the two packages: `tests/test_review_sheet_tools_offline.py
  tests/test_review_binding_offline.py tests/test_review_harness_offline.py`
  (the last reads `with_viewer_pages`): **31 passed, 1 failed**:
  `test_the_process_host_is_the_apps` (asserts "Document Review" is in the
  process host's `own_mark_words`, which hold `('AEC agent', '(AI draft)')`).
  It fails the same way with planlens main cc2a728 (1 failed, 6 passed for
  that file alone), so it is not this branch's; see "For the lead".

## Fingerprints

Not moved: no tool name, description or argument schema changed, and
`planlens/review/tests/app_tool_specs.json` is untouched. The page-map
description already says "sheet reference"; its rows now carry the number
as `sheet` and the count as `sheet_count`.

## Calls made

- **Where the count goes.** `PageSummary` lives in
  `planlens/document/model.py`, which is not in P2's file list, so the
  count is kept in `PageSummary.evidence["sheet_count"]` and the page-map
  row lifts it into `sheet_count`. A `sheet_count` field on `PageSummary`
  would be cleaner; it needs model.py.
- **Which bands.** On a sheet, the index's title-block bands plus the
  header band; on any other page only the header and footer bands, as
  before: prose low on a report page that opens with "SHEET NO." is not a
  sheet number (tested). The size rule only on a sheet, as the index.
- **Kept from the old reading:** a bare "SHEET A-101" at the start of a
  line, so nothing it found right before is lost. A drawing segment of a
  set that prints only counts is now titled "Drawing sheets" (or by its
  labels) with no `sheets`; the counts stay on its rows.
- **The lock** is per file and in this process (one app process serves
  every conversation), re-entrant, held through `describe_copy` (cheap: it
  counts the copy) and released before the look's check (model calls,
  seconds), so a second call on the same copy waits only for the write.
- **Copies.** Written from the bytes the document was opened from; for a
  document opened from a file the first write still serialises it
  (reading the file instead costs the same), and adding to a copy reads
  neither.
- **The bound: none on the text caches, none on open's per-page work.**
  By the numbers, an annotate call's whole-file copies were what grew with
  the file (two copies, about 610 MB at 300 MB by extrapolation, per call
  and per conversation calling at once); they are now one at a time. What
  open keeps on a scan is MuPDF's store, one per process with its own limit,
  not per document. The text caches cost about 90 KB a text page and
  nothing on a scan: a bound would trade every later search of a long
  report re-reading its pages for memory that 8 x 8 documents of 400 text
  pages (about 2.3 GB) would only reach if every live conversation held
  eight such reports at once; with the conversation idle close (30 min)
  and 8 open documents each, I left them. See "For the lead".
- **Sizes.** 50 and 150 MB as the rules file sets for this laptop; 300 MB
  is a linear extrapolation, labelled so.

## Release note

For the planlens release that carries this (with the app's pin raised): A
drawing's sheet number in the page map now comes from where its title block
prints it ("SHEET NO. S-201A", "DWG. NO.: A-101", or a number in the cell
under its label), so the reviewer cites "S-201A" rather than a page index;
a sheet count ("SHEET 3 OF 24") is no longer given as the sheet number and
appears beside it as the count. Two marks added at the same moment to one
marked copy are both kept (one could be lost without a word). Marking up a
large uploaded PDF takes a little over half the extra memory and about a
third of the time it did (on a made-up 150 MB scanned set, the call's extra memory fell
from about 306 MB to 174 MB and its time from about 1.1 s to 0.35 s). A
result too long to carry the PDF viewer's page numbers now says its pages
count from 0, so the reviewer no longer cites the page before without
knowing it.

## For the lead

- `conversation.py` was edited by your leave (see "Scope"); the change is
  that one repair.
- **Two more whole-file copies per annotate call** are in files not on
  P2's list: `planlens/document/markup_writer.py` `write_markups` opens its
  base from bytes and saves it, and `planlens/review/markup_check.py`
  `describe_copy` and `check_marks` each read the whole copy into memory
  (and `_check_one` opens a `Document` per mark on those bytes). Opening
  the copy by path would let MuPDF read it lazily. Each is one copy at a
  time now, about 150 MB on the 150 MB set.
- **If the testers' host is short of memory with long text documents**, a
  process-wide bound on the text caches (by pages held) is the next step;
  it costs re-reading pages on later searches. Numbers above.
- **`test_the_process_host_is_the_apps`** (Document Review's
  `tests/test_review_binding_offline.py`) fails with this branch and with
  planlens main cc2a728 alike, from the app checkout at hand: an app-side
  matter to look at.
- The planlens release and the app's pin; whether the app's
  `GEOTECH_REVIEW_SHEET_TOOLS` passes the new parts through is P2b's
  question (nothing here is behind a setting).

## Noted, not fixed

- A lower-case label and number ("dwg no. s-101") were read before (the old
  pattern ignored case) and are not now: the index's sheet-number shape is
  upper case. Title blocks print upper case; worth a line only if a saved
  run shows one.
- The count is read in the header and footer bands, and on a sheet also as
  "SHEET n OF m" on a title-block line; a bare "3 OF 24" in a right-hand
  title block with "SHEET" in another cell is not read as a count.

## The generator and the probe

Run in a fresh process per line of the table; the files went to a scratch
folder and were deleted after measuring. Condensed from the script that was
run, which also printed resident memory and had the two other modes used
above (the per-step sampling thread, and the 400-page text report).

```python
import json, os, random, sys, time
PAGE_W, PAGE_H = 1224.0, 792.0        # 17 x 11 in, a scanned sheet
IMG_W, IMG_H = 1700, 1100             # grey noise, about 1.87 MB a page

def make(out, target_mb, seed=7):
    import pymupdf
    rng = random.Random(int(seed))
    per_page = IMG_W * IMG_H
    n = max(1, int(round(float(target_mb) * 1e6 / per_page)))
    doc = pymupdf.open()
    for _ in range(n):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        pix = pymupdf.Pixmap(pymupdf.csGRAY, IMG_W, IMG_H,
                             rng.randbytes(per_page), 0)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(out)
    doc.close()

def measure(path, mode, workdir):          # mode: open|annotate1|annotate2
    import psutil
    from planlens import review
    proc = psutil.Process()
    mem = lambda: round(proc.memory_info().peak_wset / 1e6)
    os.makedirs(workdir, exist_ok=True)
    look = lambda image, question, kind: json.dumps(
        {"inside": "x", "same_thing": True, "encloses": True, "sure": True})
    review.configure(review.ReviewHost(look=look,
        working_folder=lambda: workdir, signer="Probe Reviewer (AI draft)"))
    data = open(path, "rb").read()
    att = {"set.pdf": data}
    out = {"read_peak_mb": mem()}
    t = time.perf_counter()
    handle = json.loads(review.call("open_document", {"source": "set.pdf"},
                                    attachments=att))["handle"]
    out.update(open_s=round(time.perf_counter() - t, 2), open_peak_mb=mem())
    for i in range({"open": 0, "annotate1": 1, "annotate2": 2}[mode]):
        t = time.perf_counter()
        review.call("annotate_document", {"handle": handle,
            "output_path": "set_marked.pdf", "markups": [{"kind": "box",
            "page": i, "comment": f"probe mark {i + 1}",
            "bbox": [100, 100, 300, 200]}]}, attachments=att)
        out[f"annotate{i + 1}_s"] = round(time.perf_counter() - t, 2)
        out[f"annotate{i + 1}_peak_mb"] = mem()
    print(json.dumps(out))

if __name__ == "__main__":
    make(*sys.argv[2:]) if sys.argv[1] == "make" else measure(*sys.argv[2:5])
```
