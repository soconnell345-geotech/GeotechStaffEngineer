# CHECK: RI-199, RI-201, RI-202 (MF-3 trace review fixes)

Branch `claude/ri-199-201-202-trace-fixes`. The source is
`module_work/field_feedback/2026-11-02_foundry_build_g/TRACE_REVIEW.md`
§3.1, §3.2, §6 and §7. Every test value is made up.

## RI-199: a log sheet with no depth ruler is judged on the hole's other sheets

**What was wrong.** `log_reader._depth_window` judged every depth against
one window: the union of the sheets that have a ruler. A boring whose first
or last sheet has no ruler then puts that sheet's depths outside the window.
The first sheet runs from the ground to the second sheet's top. The last
sheet runs past the last ruled sheet to the total depth. On R15 today's
readers (`refuse`) dropped 5 samples and 2 SPT records, and model first
(`flag`) wrote 19 false `out_of_range` entries at confidence 0.5.

There was a second gap in model first. A call whose own pages had no ruler
(say a ruler-less last sheet read in a call of its own) got `window=None`.
That kept every depth with no check at all.

**What changed.**
- `report_ingest/log_reader.py:846` `_ruled_windows`: each ruled sheet's own
  window, the same edges and slack as before.
- `log_reader.py:869` `_depth_window`: now the union of `_ruled_windows`.
  Its result is unchanged.
- `log_reader.py:889` `_sheet_windows(ruled, pages, total_depth, slack)`
  gives every sheet of the hole a window:
  - A ruled sheet keeps its own window.
  - A sheet with no ruler runs from the top of the nearest earlier ruled
    sheet (or `-slack`, the ground) to the bottom of the nearest later ruled
    sheet.
  - After the last ruled sheet, it runs to the printed total depth plus
    slack. With no total depth, the last ruler's sheet length is carried on
    once per sheet.
  - With no ruler anywhere it returns `{}`, and the old RI-18 printed-depth
    path is unchanged.
- `log_reader.py:932` `_hole_total`: the hole's printed total depth.
- `_Builder` takes two new keywords, `sheets=` and `ruled_pages=`:
  - `_window_for(page)` (`log_reader.py:1096`) judges a depth on a sheet
    WITH a ruler, or on a page that is not one of the hole's, exactly as on
    master: on the union of the ruled sheets. A depth on a sheet with no
    ruler is judged on the whole hole, the union of every sheet's window:
    from the ground (first sheet ruler-less) to the printed total depth
    (last sheet ruler-less). A depth is attached to one page, so a layer
    whose base is on the next sheet, or a row cited one page off, must not
    be judged on its own sheet alone (review Finding 1).
  - `_no_window()` (`log_reader.py:1091`) is True only when neither this
    answer's pages nor the hole's other sheets have a ruler.
  - The refusal and `out_of_range` wording names which window was used: "the
    scale on this log reads (a to b)" for a ruled sheet, "what this log's
    other sheets leave for page N, which has no depth scale of its own
    (a to b)" for a ruler-less one.
  - With no `sheets` the builder behaves exactly as before, so every other
    caller is untouched.
- `log_reader.py:2131`: today's reader passes the sheets for the item.
- `report_ingest/model_first_read.py:827`: model first builds the grids per
  call as before, then builds the sheet windows over the WHOLE hole, across
  every call of its group. The hole's total depth is the first reading that
  gives one.

**Before/after (made up).** Hole QX-9 has four sheets. Rulers were found on
sheet 2 (10-20 ft) and sheet 3 (20-30 ft); the total depth is 38 ft.

| Sample | Before (refuse) | Before (flag) | After (both modes) |
|---|---|---|---|
| Sheet 1 at 2 ft | dropped ("outside 9.5 to 30.5") | kept at 0.5 with `out_of_range` | kept, no entry |
| Sheet 4 at 36.5 ft | dropped | kept at 0.5 with `out_of_range` | kept, no entry |
| Sheet 4 at 41 ft (past the total depth) | dropped | kept at 0.5 with `out_of_range` | still refused / flagged: "outside what this log's other sheets leave for page 13 ... (-0.5 to 38.5)" |

A hole with a ruler on every sheet is judged exactly as on master.

**Tests:** `report_ingest/tests/test_ri199.py`, 13 tests.
- Unit tests of the windows (first sheet, last sheet with and without a
  total depth, a sheet between two rulers, no ruler anywhere).
- The builder in both modes:
  - the ruler-less ends are kept as good;
  - the old one-window gate on the same reading loses them;
  - a value outside the whole hole (past the total depth) still refuses or
    flags; 25 ft on a ruled sheet, inside the ruled union, is kept as on
    master;
  - a hole ruled on every sheet: a layer, water and sample cited on sheet 1
    that run into sheet 2's depths are kept at 0.9, as on master (fails on
    the branch before the review fix).
- A model-first `read_logs` run of two calls. The ruler-less last sheet is
  read alone: its 33 ft sample is kept good, and its 41 ft sample (past the
  total depth) is flagged.

On master the file does not import (`_sheet_windows` and `_ruled_windows`
do not exist), so every test fails.

**Prompt versions:** none moved. The windows are never sent to a model.

## RI-201: an excerpted photo log is not an incomplete copy

**Note on the task wording.** The routine prompt says "an excerpted run
flags". The backlog item, the build f review (§3.5) and the MF-3 review
(§3.2) say three things:
- R30's six false notes are runs that are excerpted on purpose: selected
  pages of earlier investigations' photo logs.
- R28's real case, a laboratory certificate printed as its pages 3-6 of 13,
  is also an excerpt, and it must still flag.
- Calculation runs are never checked today ("as calculation pages' already
  do").

So "every excerpted run flags" cannot fix R30, and "no excerpted run flags"
would break R28. What separates the two is whose numbering it is. I fixed
the item as the backlog and the reviews describe it:
- Every attachment's run goes through one rule.
- A run printed only on pages that reproduce ANOTHER document's pages
  (calculations, photos) carries that document's numbering and is not
  checked.
- Every other run (the report's own, a lab's certificates) flags when it is
  excerpted or short, and never when every X of Y is present.

**What changed.** `report_ingest/completeness.py:86`:
`UNCHECKED_RUN_LABELS = ("calculation", "photos")`, with its comment and the
`short_runs` docstring explaining why. A run only partly on photo pages (the
report's own run whose last page is a photo plate) is still checked.

**Before/after (made up).** Two narrative pages, then pages 3, 4, 9 and 10
of a 12-page photo log, labelled photos.
- Before: an open `partial` entry, "the document numbered 'Page X of 12' ...
  lacks its pages 1-2, 5-8 and 11-12".
- After: no entry.

A lab certificate printed as pages 3-6 of 13 flags before and after.

**Tests:** `report_ingest/tests/test_completeness.py::TestEveryAttachmentsRun`,
8 tests:
- an excerpted photo log does not flag (fails on master);
- a complete photo log does not flag;
- an excerpted lab certificate flags (R28's shape);
- a certificate lacking its last page flags (R28's other);
- a complete certificate does not flag;
- the report's own run lacking its last page flags (R05's shape);
- a calculation excerpt does not flag;
- a run only partly on photo pages is checked.

The other seven are guards and pass on master as well.

**Prompt versions:** none. The check reads text only and calls no model.

## RI-202: model first's describer and range labeller in `calls_by_step`

**What changed.** `report_ingest/cluster_scoring.py`:
- `_ingest_steps` (around line 7452) reads the report folder's `labels.json`
  `cost.describe.calls` and `cost.ranges.calls` into two new steps,
  `describe` and `ranges`. These are the costs graph.py already records in
  `_labels_blob`.
- A bound document's `labels.json` is NOT read for them. graph.py writes it
  with the parent's labelling (`dataclasses.replace(labelling, ...)`), so it
  carries the same whole-file describer and labeller cost, and reading it
  would count those calls twice.
- `_STEPS` (line 7584) gains `describe` and `ranges` after `vision`, so the
  `## Calls by step` table in RESULTS.md has two more columns.
- The run's `cost` (the totals) is not read differently and does not change.
  Only `unattributed` shrinks, to what no file holds.

**Before/after (made up).** A model-first report has 233 calls: 180 describer,
23 labeller, 2 review, 20 log, 5 narrative and 3 in a bound document's lab.
- Before: the report's row totals 27, with `unattributed +203`.
- After: `describe 180`, `ranges 23`, total 230, `unattributed +0`; the bound
  row is 3.

**Tests:** in `report_ingest/tests/test_cluster_scoring.py::TestCallsByStep`:
- new: `test_model_first_s_describer_and_labeller_are_steps` (fails on
  master);
- updated: `test_the_report_and_each_bound_report_are_counted_apart`, for
  the two new empty columns (its column list fails on master by design).

**Prompt versions:** none.

## Test runs

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py
  webapp/tests/test_page_fingerprints.py`: **4416 passed, 8 skipped (pydiggs not installed), 0 failed**, exit 0.
  The first run had 2 failures, both pinning what this branch changes on
  purpose, and both were updated: `test_printed_depths.py::...continuation_sheet_with_no_ruler...`
  (a ruler-less page's refusal now says the other sheets' window was used;
  the 65 ft value is still refused) and `test_ri125.py` (the review cell
  moved two columns right for describe and ranges).
- Before it: the log reader, model-first read, ground-surface,
  graph-model-first, RI-143 and checks suites: 309 passed.
- `report_ingest/tests/test_completeness.py`: 26 passed.
- `TestCallsByStep`: 17 passed.
- Consumers: `calls_by_step`, `_STEPS`, `_ingest_steps`, `_depth_window`,
  `UNCHECKED_RUN_LABELS` and `_Builder` have no consumer outside
  `report_ingest` (grepped the whole repo, `*.py`).

## Release note

Saved records change in three ways:
- **Boring logs** whose first or last sheet has no fitted depth ruler keep
  the depths read on that sheet when they lie between the hole's other
  sheets (or the ground, or the printed total depth). Today's readers no
  longer drop those samples, layers and SPT records. Model first no longer
  writes `out_of_range` entries for them or lowers them to confidence 0.5.
  A depth outside the whole hole (above the ground, past the printed total
  depth) is still refused or flagged; a hole with a ruler on every sheet is
  judged exactly as before. A model-first
  call whose own pages have no ruler is now checked against the hole's other
  sheets instead of not at all.
- **The incomplete-copy note (RI-184)** no longer flags a photo appendix's
  "Page X of Y" run.
- **RESULTS.md and run.json:** the calls-by-step table gains `describe` and
  `ranges` columns, and `unattributed` falls by those calls. Run totals are
  unchanged.

## To measure live

- **R15, both arms:**
  - RI-199: the six-sheet boring's 5 samples and 2 SPT records are back on B,
    with no refusal "outside what the scale".
  - RI-199: on A, the 19 `out_of_range` entries are gone and those values
    keep their confidence.
  - RI-202: `calls_by_step.report.describe + ranges` is about 203 on A, and
    `unattributed` is near 0.
- **R30, both arms:**
  - RI-201: none of the six open `partial` "this copy looks incomplete"
    entries on the photo-log excerpts (pp119-129, 197-226). Check which label
    each arm gave those pages; if an arm labelled them other than `photos`,
    the note stays on that arm and the label is the cause.
  - RI-199: the three one-page driller's site logs (pp66, 74, 80) have no
    ruler on any sheet, so `_sheet_windows` is empty and B still refuses.
    That case is unchanged (see below).
  - RI-202: describe + ranges is about 559 on A.
- **R05 and R28:** the RI-184 notes still appear: R05's last body page; R28's
  certificate pages 3-6 of 13 and the one lacking its page 10 of 10. Both are
  laboratory runs, not photos, provided the labels call them `lab_test`.
- **R21:** describe + ranges is about 735 on A.

## Noted, not fixed

- A log with no ruler on ANY sheet (R30's one-page driller's logs) is not
  helped. With nothing to continue from, B still refuses depths not backed
  by a printed number (RI-18's path). Out of RI-199's scope; it changes B's
  output on those three holes. Worth an item only if B stays a live arm.
- A ruler-less sheet's depths are accepted inside the continuation window
  without a change entry, as a ruler-less middle sheet's already were.
- The incomplete-copy rule is by label: a photo-log excerpt labelled anything
  but `photos` still flags.
