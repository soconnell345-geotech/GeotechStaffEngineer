# RI-185: a partial copy says which logs, lab results and appendices it lacks

Branch `claude/ri-185-partial-copy-logs-lab`, from master 50ef609. Brief: `module_work/cloud_briefs/INGEST_2026-11-13.md` section 3.

## What changed and why

RI-184 flags a copy whose page numbering runs short or whose text cites figures, tables or appendices the file holds none of. This adds, in the same module and with the same caution, three checks of what a geotechnical report names and a partial copy drops. Text and the record only; no model call; nothing a model is sent changes.

| Where | What |
|---|---|
| `report_ingest/completeness.py:457` `named_not_logged` | Explorations the report's own pages name at least twice that the record holds no log for. |
| `report_ingest/completeness.py:594` `lab_said_not_held` | Places the text says laboratory results are in this document, when the record holds no laboratory test and the file no laboratory page. |
| `report_ingest/completeness.py:726` `listed_appendices_not_held` | Appendices a contents page lists by letter and title that the file plainly lacks, where it holds others. |
| `report_ingest/completeness.py:814` `named_not_held_qa` | The ONE `partial` QA entry, ranked `open`, `where="document.completeness"`, naming all three with their pages. |
| `report_ingest/graph.py:1420-1423` | The graph calls it after RI-184's `completeness_qa`, with the record and the bound pages. The pages' text is read once for both. |

`completeness_qa` itself is not changed (its signature, its entries, its tests), so RI-184's behaviour is exactly as before.

### 1. Explorations named, with no log

- **A name** is a series of one to four capital letters and a number: `B-1`, `TP-3`, `CPT-2`, `MW-02A`, `B1`. A range of one series is expanded (`B-1 through B-8`).
- **It is a hole's name only in a known series**: one the record's own holes have, or one an exploration word introduces on the report's own pages ("borings B-1, ...", "test pits (TP-1 through TP-6)"; `_RE_HOLE_WORD`, line 381). The series keeps whether it is written with a hyphen, so `D50` beside borings named `D-1` is not a hole.
- **Not a hole's**: a name after Figure, Table, Appendix, Plate, Sheet, Drawing, Section, Sample, Zone, Type, Class, Grade, Detail, Photo, Exhibit, Attachment, ASTM or AASHTO, including later names of such a list ("Figures B-9 and B-11").
- **The report's own pages** are those labelled narrative, letter, cover or toc (`CITING_LABELS`, RI-184's), outside any bound document.
- **Held** when a log in the record carries the name by the existing matcher `log_holes.names_match` (leading zeros folded, a site prefix dropped, a bare number against a series), or when a page labelled a log, a field test or an appended report, or any bound document's page, prints the name. No new matcher.
- **Not missing** when any sentence naming it says it was not drilled, could not be drilled, was abandoned, omitted, cancelled, relocated, by others, previous, prior, earlier, former, historic, preliminary, proposed, planned, future or recommended, or names a year's investigation (`_RE_HOLE_NOT_MISSING`, line 398).
- **Nothing is flagged** when: the text says the logs are under separate cover, in a separate volume or a data report (line 407); a bound document has a page with no text (a scan may hold them); or the holes missing are no more than the record's holes under names the text never uses plus the scanned log pages no recorded hole was read from (the reader may have read them under other names, which RI-144's `missing_holes_qa` already says).
- **A single mention is not flagged** (`MIN_MENTIONS = 2`, line 358).

### 2. Laboratory results named, with none held

Flagged only when the record holds no `lab_tests` at all AND no page of the file (a bound document's included) is labelled `lab_test` AND no page outside the report's own text leads with a laboratory sheet's words AND no bound page is an unread scan.

Wordings taken as "included in this document" (`_RE_LAB_INCLUDED`, line 552), in a sentence that is about laboratory results, data or tests:

- "are / is / were / was / been presented | included | provided | shown | contained | summarized | given | tabulated | reported | plotted | compiled | attached | enclosed | appended in / on / as / at / to (the) (attached) Appendix | Appendices | Attachment | Enclosure";
- "are / is attached | enclosed | appended";
- "Appendix X presents | contains | includes | provides ...";
- a contents line titling an appendix or attachment with the word laboratory ("Appendix C - Laboratory Test Results").

Wordings NOT taken as included:

- results in a **Table or a Figure** ("summarized in Table 2"): the table may sit on a narrative page the readers took as text;
- "shown **on the boring logs**": those values are the logs';
- "described in Appendix C" (a description is not results);
- any sentence with will, to be, should be, shall, recommend, planned, proposed, pending, forthcoming, in progress, under separate cover, separate report / volume / letter / document, not (yet) performed / conducted / included / available / completed, no laboratory, by others, previous, prior, earlier, upon request, on file, if (`_RE_LAB_NOT_HERE`, line 562).

Two such places are needed (two sentences, or a sentence and a contents line). The entry names the tests the text names (sieve or grain size, Atterberg limits, consolidation, triaxial, direct shear, unconfined compression, compaction, corrosion, moisture content) and quotes each place.

### 3. Appendices in the contents with no pages

- **Listed**: lines of a `toc` page "Appendix X - Title ..... 12" (the title may be on the next line). At least two listed.
- **Held** when a page that is not a list leads a line with its letter (RI-184's own rule), when a page outside the report's text leads with a line alike its title (two thirds of the title's words, at least two; plurals and "-ing" folded), or when a page labelled for its content follows the last appendix held before it in the list. Content by title (`_TITLE_CONTENT`, line 681): CPT → `cpt_log`; test pit → `test_pit_log`; boring / borehole / log → any log label; laboratory → `lab_test`; calculation / computation → `calculation`; photo → `photos`; figure / drawing / plan / map → figure, plan, profile, photos.
- **Never flagged**: an appendix whose title names no content the labels know; one its own line says is not used, not included, reserved, omitted, under separate cover or by others; anything when NO listed appendix is held (RI-184 says that one, so the two never repeat each other).

## Made-up before / after

A 6-page file: a contents page listing Appendix A (Site Plan and Figures), B (Boring Logs) and C (Laboratory Test Results); a narrative saying twice that borings B-1 through B-8 were drilled and that "The laboratory test results are presented in Appendix C."; dividers for A and B; logs B-1 to B-6. The record holds B-1 to B-6 and no lab test.

- **Before (master):** no completeness entry (the file holds Appendices A and B, so RI-184 says nothing about C; the numbering is not printed).
- **After:** one entry, `partial` / `open` / `document.completeness`:
  > this copy looks incomplete: the text names the explorations B-7 and B-8 (page 2), and the record holds no log for them; the text says laboratory results are in this document (pages 1-2), naming sieve or grain size and Atterberg limits, and the record holds no laboratory page or test; the contents (page 1) list Appendix C, which the file does not hold

  with `values` listing each hole and its pages, each laboratory wording quoted with its page, and each appendix with its title and contents page.

## Tests added

`report_ingest/tests/test_ri185.py`, 37 tests, made-up pages and records. On master the file fails at collection (the four functions do not exist): checked by stashing the two source files, `1 error during collection`.

- Explorations: some missing; all missing; a whole report; padded names; a single mention; not drilled; abandoned, by others, previous investigation, a year's investigation (parametrized); a bound report's holes; a scanned bound document; holes read under other names and unread scans; figure, appendix and sample names and `D50`; logs under separate cover; test pits and soundings.
- Laboratory: flagged; a record with a lab test; a file with a laboratory page (by label, and by a sheet's leading words); a single mention; planned, recommended, separate cover, by others, "described in", "on the boring logs" (parametrized).
- Appendices: one lacking; a whole report; content pages following with no divider; a slightly different title; none held (RI-184's); not used / separate cover / unknown content; one listed line is not a list.
- The entry: all three in one entry (exact text); a complete report with every kind gives none from either function; no labels or no record gives none.
- The graph hands the check the record and the bound pages.

RI-184's tests (`test_completeness.py`) are unchanged and pass.

## Test runs

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`: **4911 passed, 23 skipped (pydiggs not installed), 0 failed**, exit code 0. Graph snapshots green with no change.
- `python -m pytest -q report_ingest/tests/test_ri185.py report_ingest/tests/test_completeness.py`: 63 passed.

## Prompt versions

None moved. Nothing a model is sent changes.

## Consumers

Grepped the whole repo for `completeness_qa`, `document.completeness` and "looks incomplete": the only caller is `graph.py`; nothing outside `report_ingest` reads the entry's text. No record field, result key or export is added, so the tables export (RI-209) needs nothing.

## Release note

A record ingested after this change may carry one more open `partial` QA entry at `document.completeness`, saying which named explorations have no log, that laboratory results the text says are in the document are not held, and which listed appendices the file lacks. No value of the record other than its QA list changes, and its open-entry count rises by at most one. Records already saved are not changed until re-ingested; a resumed run re-computes the entry (it is not cached).

## To measure live

No model call is needed: the check reads the PDF's text, the saved labels and the saved record.

For each saved report folder (`report.record.json` and its PDF):

```python
from planlens.document import open_document
from report_ingest.bound import parse_range
from report_ingest.completeness import page_texts, named_not_held_qa
from report_ingest.model import ReportRecord

record = ReportRecord.model_validate_json(open(folder / "report.record.json").read())
doc = open_document(str(pdf))                # as graph._open opens it; pass text_source= where a DI result was used
labels = {p.page: p.label for p in record.page_labels}
bound = [p for b in record.bound_documents for p in parse_range(b.pages, doc.n_pages)]
print(report_id, [e.detail for e in named_not_held_qa(page_texts(doc), labels, record, bound)])
```

(`BoundReport.pages` is the 0-based span the child occupied, as the graph's bound ranges gave it.) Count: reports with an entry, and within entries the number of holes, laboratory places and appendices named. Then open the cited pages of every entry.

A false entry looks like one of these; each is worth a line back to the backlog only if it shows on a complete report:

- a "hole" that is a sample, a figure, a grid line, a soil unit or a standard's number in a series that happens to be a hole series;
- a hole whose log is in the file but is a scan the reader named differently (RI-144's entry should be beside it);
- holes of an earlier investigation named without any of the "previous / prior / by others" words in the same sentence;
- laboratory sheets present but labelled something other than `lab_test` with no sheet title in their first lines;
- an appendix whose pages are there under a title sharing fewer than two thirds of the listed title's words and whose content labels do not follow.

Also worth counting the other way on the known partial copies (the ones RI-184 flags): does the new entry name what a person sees is missing.

## Calls made

1. **One entry for all three, not one per kind.** The brief says "one `partial` QA entry". The other reading is one entry per kind (as RI-184 makes one per short run), which would let a reviewer close them one at a time.
2. **A separate function and graph call, not a new argument on `completeness_qa`.** A test of RI-184's replaces `completeness_qa` with a two-argument stand-in; changing its call would have changed that test, and the brief says RI-184's tests stay unchanged.
3. **Laboratory: flagged only when the record holds no laboratory page or test at all**, with the kinds named in the entry. The other reading of "flag by kind" is to flag, say, consolidation when the text says its results are in an appendix and the record holds other kinds but no consolidation test. Not built: a summary table's rows, a sheet read as `other` or a sheet not read would each make that a false note on a whole report.
4. **Results "in Table N" or "in Figure N" are not taken as included.** Such a table is often on a narrative page and never becomes a lab test, so a complete report would flag. The other reading catches a copy missing its results tables; RI-184 already flags tables when the file holds none of the cited ones.
5. **"Mentioned twice" for a hole counts each naming on the report's own pages**, a range counting for every hole in it. So "B-1 through B-8" written once and never again is not flagged. The other reading (one range in a field-work sentence is plain enough) would flag more partial copies and more whole ones.
6. **Logs "under separate cover" / in a "data report" suppress the exploration check for the whole report.** The copy is then the whole of its document though it holds no logs. The other reading is that the record still lacks them and should say so.
7. **Bare-number holes ("Boring 3") are not read.** "boring 3 feet" and "borings 10 m apart" read the same way. A report that numbers its holes without letters gets no exploration check.
8. **An appendix whose title names no content the labels know is never flagged** (for example a seismic hazard printout, or "Important Information"), even when it is last in the list and plainly cut off. Only its divider could show it is held, and dividers are often missing.
9. **A hole printed on any log-labelled page counts as held**, even though the record does not carry it. That absence is the reader's, not the copy's, and RI-144 reports it.
10. **Pages of a bound document never cite** and always count as possibly holding a hole or a laboratory sheet. A report bound into the file that itself lacks its logs is not checked: the check runs on the parent only, as RI-184's does.

## Noted, not fixed

- `completeness_qa` and the new check run before the bound documents are read, so a volume of the SAME investigation bound in (RI-94) is covered only by its pages' text and labels, not by its record. With text pages this is enough; a scanned one suppresses the exploration and laboratory checks.
- The new check does not run on a bound document's own record (neither does RI-184's).
- `budgets.max_items` leaving logs unread does not cause a false exploration entry where the log pages have text (their names are printed); with scanned logs the scan allowance covers it.
- `report_ingest/README.md` does not describe RI-184's check, so nothing was added there for this one.
