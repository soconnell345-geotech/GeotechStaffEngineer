# CHECK: RI-165 and RI-157 (narrowed)

Branch `claude/ri-165-157-carried-link-probe`, from master 25c2525.
Commits: 32f9d00 (RI-165), 8f16ffb (RI-157), and this file.

## RI-165: the carried hole and sample link on the model-first lab route

### What changed and why

Today's route: `split_lab_items` (`report_ingest/lab_reader.py:1258`) gives a
later part of a cut lab item whose pages print no boring or sample the first
part's link, `evidence["carried_link"]`. `graph._read_lab` passes it to
`read_lab_sheet`, which puts ONE sentence in the brief (`_carried_words`):
"a test on these pages that names no boring or sample is that one's, unless
the sheet says otherwise". Today's route applies it through the prompt only;
no code puts it on a test.

The model-first lab route never saw it. It reaches a model-first lab group two
ways, and the link was lost on both:

1. **Labels not model first:** each of today's items (already cut by
   `split_lab_items`) becomes a lab group (`graph._model_first_units`). The
   item's `carried_link` was dropped. Now copied onto the group:
   `report_ingest/graph.py:3251`.
2. **Labels model first (the default):** `lab_groups` cuts a run of lab pages
   into windows of at most `MAX_LAB_PAGES`; a later window whose pages print
   no boring lost the one printed over the run's first sheet. Now each later
   window names the run's pages before it, nearest first, bounded by the
   page descriptions: `PageGroup.carried_from`, set in `lab_groups`
   (`report_ingest/model_first_pages.py`). See "Review fixes" below; the
   first version named the first window's pages.

`PageGroup` gains `carried_link` and `carried_from`
(`model_first_pages.py:937-941`); `to_dict()` writes them only when set, so a
group without them serialises as before.

`model_first_read.read_lab` applies the link **by code after the answer**
(`model_first_read.py:979`, `_carried` at `:999`, `lab_reader.carry_link` at
`lab_reader.py:1314`). The model-first route sends no per-item brief, so
putting the sentence in the request would have been a new prompt and a
version bump that re-reads every saved lab answer; the code form gives the
outcome today's sentence asks for without changing any request. The rule
mirrors today's sentence and never overrules what the pages print:

- the group's own pages print a boring or sample (`_printed_link`): nothing
  is carried (as `split_lab_items` carries nothing to such a part);
- a test that names a boring OR a sample of its own keeps it, untouched;
- a summary table whose rows name their own borings is untouched;
- a test naming neither takes the carried boring and sample, and its note
  says `boring 'QX-7' carried from page 0: these pages print no boring or
  sample of their own (RI-165)`;
- the result blob gains `carried_link` (`{..., "tests": n}`) when any test
  took it.

### Made-up before/after

Pages 1-2 are Atterberg sheets printing no boring; page 0, the summary before
them in the same run, prints `Boring: QX-7`. The model reads a test with no
boring on page 1.

- master, model first: `investigation_id ""`, `sample_id ""`; the reconciler
  cannot link it to a hole.
- branch: `investigation_id "QX-7"`, with the note above; a second test on the
  same page printed `QX-9 / S-8` keeps `QX-9 / S-8`.

### Tests (`report_ingest/tests/test_ri165.py`, 6)

- a cut part takes its carried link, and the request sent is the same as
  without it (text blocks and the cut PDF's page text compared);
- a test naming only its sample keeps it, no boring added;
- a later window of a run carries what its earlier pages print;
- a group printing its own boring carries nothing;
- `lab_groups` names the pages before each later window, nearest first;
  `to_dict` only when set (changed by the review fixes);
- today's cut items give their carried link to the model-first group
  (`_model_first_units` with the labels not model first).

All six fail on master (the new `PageGroup` fields do not exist there; the
two guard tests fail for that reason only).

### Prompt version

None moved. No text a model is sent changes; `test_page_fingerprints` and
the model-first schema digests pass unchanged.

## RI-157 (narrowed 2026-10-29): a two-page probe sheet joined into one record

### What changed and why

`_fold_continuations` (`report_ingest/reconciler.py`) folds a log's
continuation sheets but left soundings out (`_FOLDED_KINDS`). A dynamic probe
or cone sheet printed over two pages, page 2 starting where page 1 stops,
became two records of one name.

`_continues` now hands a `cpt`/`dcp` pair to `_sounding_continues`
(`reconciler.py:704`, called at `:781`). The two are joined only when **all**
of these hold:

- the later record is on the next page, same kind, same depth unit and
  units-known flag, same vertical axis;
- one name: `hole_key` equal after a trailing "continued" / "cont'd" /
  "cont." is dropped (`DP-03` and `DP-3 (continued)` are one);
- a printed sheet number on the later page, if any, is not sheet 1 and not a
  sheet of another set;
- page 2's series starts where page 1's stops: |top(2) - bottom(1)| within
  `SOUNDING_JOIN_TOL_M` = 0.10 m (`:664`) or 1.5 × page 1's median reading
  spacing, whichever is larger, and page 2 reaches deeper. top(2) is the
  first point's printed interval top where it prints one, else its depth; an
  elevation axis is turned over (`_series_m`, `:681`).

The 1.5 × spacing is there because page 2's first reading is one step below
page 1's last (a 0.10 m increment probe: 3.00 then 3.10).

Overlapping depths (page 2 starting above page 1's bottom by more than the
tolerance, for example twins both from the ground) or a larger gap stay two,
as twins do today.

The fold (`_fold_series`, `:739`, from `_fold_into`) appends page 2's
points. A page-2 reading not deeper than the last one kept (by more than
5 mm) is a reading printed again: it is dropped and counted in the QA note.
The first version dropped only a repeat of the last reading; see "Review
fixes". Blank series header fields (cone type,
hammer, increment ...) are filled from page 2, and page 2's refusal depth is
kept. The record-level fields go through the same `_fold_into` as a log's.
The QA note is the fold's usual `<name>.sheets` entry.

`_check_counts` runs after `_fold_continuations` in `reconcile` (and in
`graph._recheck_counts` over the folded record), so a joined sounding counts
once. Note: `cptCount` counts `cpt` only; no narrative count field covers
`dcp`.

### Made-up before/after

`DP-3` on page 4, readings 0.10-3.00 m every 0.10 m; `DP-3` on page 5, readings
3.10-5.20 m.

- master: two investigations `DP-3` (pages [4] and [5]); shared_names notes
  them as two parts of one hole.
- branch: one `DP-3`, pages [4, 5], 52 readings 0.10-5.20 m, with a QA note
  "DP-3 was read as 2 logs ... its series starts at 3.10 m, where the first's
  stops (3.00 m; within 0.15 m)".

### Tests (`report_ingest/tests/test_ri157.py`, 10)

Joined (all fail on master):
- a probe continued on the next page;
- a title adding only "continued";
- a reading printed on both pages, kept once;
- a cone sounding continued;
- the count check counts a joined sounding once (`cptCount` 1, no
  `count_mismatch`).

Not joined (pass on master and on the branch, as guards):
- a gap of 0.5 m;
- twins starting again at the surface;
- overlapping depths;
- another name, or a page between;
- a later page printing "1 of 1".

Changed: `report_ingest/tests/test_ri135.py::test_a_soundings_readings_carried_on_to_the_next_page`
pinned master's two-record state for exactly this shape. It now asserts the
joined record (pages [50, 51], eight readings in order). It is the only
existing test that moved.

### Prompt version

None moved (reconciler code only).

## Test runs

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  - before the test_ri135 update: 4550 passed, 1 failed (that test), 8
    skipped (pydiggs missing);
  - after it: see the final run below.
- `report_ingest/tests/test_graph_model_first.py` (the graph snapshot) and
  `test_ri135.py`: 134 passed. The snapshot did not move.
- Consumers: a repo-wide grep for `carried_link`, `carried_from`,
  `carry_link`, `_continues`, `_fold_into` and `_FOLDED_KINDS` finds no use
  outside `report_ingest`.

Final run on the branch (both items, the test_ri135 update): **4551 passed, 0 failed, 8 skipped** (pydiggs missing), exit code 0.

## Review fixes

From the independent review (verdict "merge after fixes"). A wrong hole link
is worse than none.

1. **RI-165, default route: a later lab window carried the wrong boring.**
   `lab_groups` gave every later window the run's FIRST window, so it took
   QX-1 when QX-2 was printed nearer (S1), skipped the window before (S2),
   and carried into a window whose descriptions name a new hole that its
   text prints without "Boring" beside it (S3). Now (`lab_groups`,
   `model_first_pages.py`):
   - a later window names the run's pages before it, **nearest first**, so
     `_printed_link` finds the nearest printed boring;
   - a window any of whose pages the descriptions say logs an exploration
     (`page_identifiers`) names **no** pages: nothing is carried there;
   - the pages named **stop at the nearest earlier page the descriptions
     say logs an exploration** (that page included). This goes one step
     past the review's code: with only the first two rules, a page between
     that logs QX-2 without the word "Boring" beside it would be passed
     over and QX-1, printed further back, carried. Now nothing is carried in
     that case. The cost: if every page of the window before is described
     with its hole and only an older page prints "Boring: ..." in text, the
     carry is missed. That loses a link; it never adds a wrong one.

   `_carried` (`model_first_read.py`) is unchanged in code: it reads
   `carried_from` in the order given. Docstrings of `PageGroup.carried_from`,
   `lab_groups`, `read_lab` and `_carried` updated. The item route
   (`carried_link`) is unchanged.

   Tests (`test_ri165.py`): the lab_groups test now checks the reversed
   earlier pages; new tests for S1 (carries QX-2, through `read_lab` too,
   note "carried from page 6"), S2 (the third window carries QX-3), S3
   (carries nothing), the stop at a described page (carries nothing, not
   QX-1), and RI-150's shape (summary prints QX-7, sheets none: still
   carries QX-7, with the descriptions and without). All but the RI-150
   test fail on the branch before the fix.
2. **RI-157: readings repeated on page 2 were kept twice, out of order.**
   `_fold_series` now drops every page-2 reading not deeper than the last
   one kept (more than 5 mm deeper is new), turned over on an elevation
   axis. The unused `_same_reading` helper is removed. Tests
   (`test_ri157.py`): a probe whose page 2 repeats two readings (40 points,
   strictly deeper, "2 reading(s) printed again"); a cone repeating three
   (600 points, in order) and an elevation cone repeating two (200 points,
   running down). Both fail before the fix.
3. **RI-135's sounding note is tested again.** The branch had turned
   `test_a_soundings_readings_carried_on_to_the_next_page` into a test of
   the joined record. New `test_a_soundings_readings_carried_on_past_a_page_between`
   puts the old assertions on pages [50] and [52]: the fold leaves them two,
   and RI-135 says "its readings (1.80-2.40 m) carry on", one hole.

No request text changes; `test_model_first_schema.py` passes unchanged.

Runs (from the worktree, free RAM checked at 3 GB or more before each):

| Command | Result | Exit |
|---|---|---|
| `pytest -q report_ingest/tests/test_ri165.py test_ri157.py test_ri135.py test_model_first_schema.py` | 126 passed | 0 |
| `pytest -q report_ingest/tests/test_graph_model_first.py` | 63 passed (snapshot unchanged) | 0 |
| `pytest -q report_ingest/tests/test_model_first_pages.py` | 141 passed | 0 |

## Release note

Two changes to what a saved record holds; neither changes a request sent to a
model, so no saved answer is re-read.

- **Lab tests on the model-first route** that print no boring or sample now
  carry the boring and sample their run prints earlier. On a cut part of a
  lab item, that is the item's first part's. On a later window of a long
  lab run, it is the boring printed **nearest before** the window. It is
  never carried into a window whose page descriptions name a hole, nor
  from beyond an earlier page the descriptions say logs a hole. The
  test's own `investigation_id` / `sample_id` stay empty where nothing
  carries, and a note says where the link came from. The reconciler can
  then link those tests to their hole, so `linked_investigation_id`,
  `linked_sample_id` and the lab results written into the logs gain entries.
  The `unlinked` QA entries for those tests go away.
- **A dynamic probe or cone sounding printed over two pages**, page 2 carrying
  on from page 1, is now one investigation, not two of one name. Its series
  holds both pages' readings, each once and in depth order. Hole counts and
  the "two holes under one name" notes drop by one for each such pair, and
  the DIGGS file has one `DynamicProbeTest` / `StaticConePenetrationTest`
  where it had two.

## To measure live

- Re-run the reconciler offline over the saved corpus records (no model):
  count the sounding pairs now joined, and check each against its pages.
  The 2026-10-27 review's R28 probe record should be one. Confirm no true
  twin of one name was joined.
- On the next model-first run with the labels model first: count lab windows
  with `carried_link` in their result, and check the carried boring against
  the pages. A summary page that prints several borings would carry only the
  first. That is today's rule too (`_printed_link`), so this measures whether
  it bites.
- The lab-to-hole link rate on the model-first route, before and after,
  over the saved model-first lab answers (no new model calls: `carry_link`
  can be applied to the saved blobs offline).

## Noted, not fixed

- A window halved inside one call (cut-off reply or over the request limit,
  `_Caller.read`) whose half prints no boring does not carry the group's
  first half's link. Today's route does not carry it across its own windows
  either. Not fixed, per the no-edge-case rule.
- `carry_link` fills both boring and sample from the carried link, or
  neither. A test printing only a sample keeps it with no carried boring, as
  today's sentence ("names no boring or sample") implies.
- No narrative count field covers dynamic probes (`_COUNT_KINDS` has
  `cptCount` only), so a joined `dcp` pair changes no count check.
