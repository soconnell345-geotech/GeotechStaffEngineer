# RI-103 parts (1) and (2): every drawn line an interval, written changes as notes, stratum labels as printed

Branch `claude/ri-103-log-intervals`, from master 50ef609. Everything is behind
one switch, `GEOTECH_INGEST_LOG_INTERVALS`, off by default, on the model-first
log path only. Today's log reader and the floor are not changed. Part (3),
mapping intervals to the narrative's units, is not built.

Nothing here was run against a model or a real report. Every log, name, depth
and label in the tests is made up.

## What changed, and why

| What | Where |
|---|---|
| The switch: read from the environment by one function, a context manager for tests, the version suffix `+li` | `report_ingest/log_intervals.py:78`, `:84`, `:87`, `:98` |
| The paragraph the model-first log instruction gains when on | `report_ingest/log_intervals.py:115` (`MODEL_FIRST_PARAGRAPH`); put in after "say which unit." at `report_ingest/model_first.py:678` |
| The answer schema when on: today's `LogReading`, its `layers` retyped to a `ReadLayer` with seven more fields and a `ReadIntervalNote` | `report_ingest/log_intervals.py:141`, `:154`, `:200`; chosen at `report_ingest/model_first.py:461` |
| `prompt_version("logs")` carries the suffix `+li` when on, on whatever `READ_PROMPT_VERSION` is (mf-10 on this branch's base, mf-11 on master today: `mf-11+li`; with the water switch too, `mf-11+wnm+li`) | `report_ingest/model_first.py:333` |
| The record: `LayerNote`, and seven optional fields on `Layer`, each left out of the saved record when empty | `report_ingest/model.py:1112` (`LayerNote`), `:1162` to `:1220` (fields and serializer), `:1224` (`LAYER_INTERVAL_FIELDS`), `:162` (why `SCHEMA_VERSION` stays 4.0) |
| The new fields put onto the layers today's builder made | `report_ingest/log_intervals.py:270` (`apply_interval_additions`), called at `report_ingest/model_first_read.py:861` |
| A label marked carried that no interval above has: listed, kept as read | `report_ingest/log_intervals.py:372` (`carried_label_entries`), called on the joined log at `report_ingest/model_first_read.py:922` |
| A continuation sheet that prints an interval again keeps the new fields its second printing adds | `report_ingest/reconciler.py:846` (`_restated` leaves them out of its comparison), `:854` (`_same_note`), `:891` to `:917` (`_fold_into`) |
| The library's search words hold a note's words and a stratum label | `report_ingest/library.py:1235` |
| The question list's soft-layer reading sees "soft" written in a note | `report_ingest/question_list.py:1203` |
| The log scorer's interval groups | `report_ingest/log_scoring.py:121` (`INTERVAL_METRICS`, the rules), `:771` (`_score_interval`); RESULTS table at `report_ingest/cluster_scoring.py:3841` to `:3868` |
| An answer sheet's layer turned into the record's fields | `module_work/report_ingest_harness/truth_records.py:210` (`_interval`) |
| The switch in the README | `report_ingest/README.md:1463` |

### The record's new fields (`Layer`)

- `top_elevation`: a `Quantity` in the header's elevation unit (empty unit when
  the log prints none), only where the log prints an elevation for the
  interval's top.
- `notes`: a list of `LayerNote` (`depth`, a `Quantity` in the log's depth
  unit with the interval's page, or absent; `text`, as printed; `prov`).
- `top_line`: the kind of the top line in the legend's or the line's own
  words; empty when the log says nothing.
- `top_line_full_width`: true, false, or absent. False is a reading and is
  saved.
- `stratum_label`, `stratum_label_source` (the column's heading, or "in the
  description"), `stratum_label_carried`.

A layer with none of them saves with exactly the keys it had. A record saved
before them loads and re-saves unchanged.

**Each interval stays addressable for part (3).** An interval is one entry of
`Investigation.layers`, in the log's order, with its own top depth. Nothing in
this branch joins or reorders them, so a later step can name an interval by
its hole, its position and its top, and give it one unit or none. No
identifier field was added.

### Where layers are joined, dropped or de-duplicated (item 1)

I looked at every place on the model-first log path and in what it reuses.

| Place | What it does to layers | Under the switch |
|---|---|---|
| `log_reader._Builder.layers` (`report_ingest/log_reader.py:1397`) | One `Layer` per layer read, in order. No merge. Under `checks="flag"` it keeps a layer out only when its top is impossible (negative, past the depth bound), with an entry that keeps the whole layer as read. | Unchanged. The additions of the layers after a kept-out one are paired by description and top, so they do not shift (`log_intervals._pairs`). |
| `model_first_read._join_logs` (a log read in several calls) and `graph._join_by_hole` (one hole logged in two places) | Append the later call's layers. No de-duplication. | Unchanged; tested. |
| `reconciler._fold_into` (a continuation sheet of the same hole) | Leaves out a later sheet's layer when it is the first's printed again: the same top, a base that does not differ, the same description, and no other field that both set and that differs. | Kept: it is one interval printed twice, not two intervals. Two adjacent intervals have different tops and are never taken as one. Changed: `_restated` leaves the RI-103 fields out of its comparison, so two printings that carry different notes, lines or label marks are still one interval; the kept copy then takes each note of the second printing it does not already hold (same words and depth), the second printing's label with its source and carried mark only where it has no label, its line kind with its width only where it has no line kind, and its top elevation where it has none. Before the review fixes the comparison included the new fields and a restated interval was kept twice. |
| `checks.ground_investigation` | Drops a layer whose top is above the ground surface by more than the tolerance, with an entry. | Unchanged. |
| `log_holes.drop_other_hole_rows`, `take_rows`, `sort_rows_by_hole` | Move a row to the hole it belongs to. No merge. | Unchanged. |
| `diggs_writer.lithology` | Skips a layer whose top has no SI value; writes a ditto-mark description as the one above (the interval is kept). | Unchanged. |
| The summary, the library page | Count layers; do not join them. | Unchanged. |

No place merges two adjacent layers by description or classification, so
there was nothing to turn off. The one de-duplication (a restated interval on
a continuation sheet) stays, for the reason above.

## A made-up example

The log (drawn in the test, `report_ingest/tests/test_log_intervals.py:49`):
a line at 4, 9 and 14 ft, the one at 9 ft dashed; a legend saying a dashed
line is an approximate boundary; "FILL" in a STRATUM column against the first
interval; "wet below 7 ft" written inside the second with no line. The
intervals at 4 and 9 ft have the same words and the same symbol.

Before (switch off), a reading may give, and nothing stops it:

```
0-4 ft   MADE-UP FILL, brown sandy clay                     CL
4-14 ft  MADE-UP lean CLAY, gray, stiff; wet below 7 ft     CL
14-20 ft MADE-UP silty SAND, dense                          SM
```

After (switch on, the answer built as the instruction asks):

```
{"top": "0 ft", "bottom": "4 ft", "description": "MADE-UP FILL, brown sandy clay", "uscs": "CL", "top_elevation": "212 ft", "stratum_label": "FILL", "stratum_label_source": "STRATUM"}
{"top": "4 ft", "bottom": "9 ft", "description": "MADE-UP lean CLAY, gray, stiff", "uscs": "CL", "top_elevation": "208 ft", "notes": [{"depth": "7 ft", "text": "wet below 7 ft"}, {"text": "as above, but brown"}], "top_line": "solid", "top_line_full_width": true}
{"top": "9 ft", "bottom": "14 ft", "description": "MADE-UP lean CLAY, gray, stiff", "uscs": "CL", "top_line": "approximate", "top_line_full_width": false}
{"top": "14 ft", "bottom": "20 ft", "description": "MADE-UP silty SAND, dense", "uscs": "SM", "top_line": "solid", "top_line_full_width": true}
```

(Quantities shortened to "value unit" and provenance left out, for reading.)

## What each consumer does with a switched-on record (item 6)

| Consumer | With more, shorter intervals | With the new fields |
|---|---|---|
| Reconciler (`reconcile`) | Runs; keeps every interval, like ones included. | Does not read them, except the continuation-sheet fill above. |
| Summary (`summary_markdown`) | Runs; it counts layers. | Shows none. The summary of a record with them is byte for byte the summary without them (tested). |
| Library page and `write_outputs` | Run. The saved record carries the fields and loads back equal. | The page shows none. |
| Library (`library.py`) | Lists intervals as before (top, bottom, description, USCS). | The search words now hold each note's words and each stratum label, so they stay findable. The rows returned do not carry the new fields. |
| Question list (`question_list._soft`) | Runs. | Reads "soft" in a note's words as well as in the description. No other reading uses layers. |
| DIGGS writer | Four intervals in, four `LithologyObservation` out; the schema gate and the round-trip gate both pass (tested). | Left out of the file in this branch. |

**DIGGS 2.6 elements each new field would go to** (names read from the
bundled 2.6 schema's `Kernel.xsd`; none was written or validated here):

- a note: a `remark` (`Remark` with `content`) on the interval's
  `LithologyObservation`; its depth has no slot of its own in a remark, so
  either in the words or as a second, placed observation;
- the stratum label: a `StratigraphySystem` with one
  `StratigraphyObservation` per interval, the label in
  `stratigraphicUnitName`, placed over the interval; the source and the
  carried mark have no element, so a `remark`;
- the line kind and its width: the `baseBoundary` (`Boundary`) of the
  interval above. `BoundaryType` holds only a location, so the kind would be
  the boundary's `gml:description` or a `remark`;
- the top elevation: no element is needed where the hole's reference point
  carries its elevation; a printed per-interval elevation that disagrees with
  it has no slot, so a `remark`.

**What an export of the new fields would need** (the report tables export is
another branch's, RI-209; nothing was added to any export or table here): a
layers table would take `top_elevation` (value, unit), `top_line`,
`top_line_full_width`, `stratum_label`, `stratum_label_source`,
`stratum_label_carried` as columns of the layer row, and the notes as a table
of their own keyed by hole and interval position (hole, interval top, note
depth, unit, text, page), since an interval has any number of notes.

## The scorer and the answer sheets (item 7)

A sheet without the new keys scores exactly as today: pinned by
`TestTheScorer::test_a_sheet_without_the_new_keys_scores_exactly_as_today`,
which also checks that a record holding notes, lines and labels scores the
same on such a sheet. The groups are not in `METRICS`, so `LogScore.total`
and the RESULTS table's OVERALL row are today's. The RESULTS table prints the
groups in a block of their own under OVERALL, only where a sheet has them.

The record's interval "at that top" is the one whose top is nearest the
sheet's layer top, within the layer-top tolerance (0.30 m).

| Group | Sheet key on a layer | Found when |
|---|---|---|
| `interval_note` | `notes`: `[{"depth": 7, "text": "..."}, {"text": "..."}]` | that interval holds a note with those words, equal after case, spacing and punctuation are set aside (`_same_words`; not `_same_text`, the comparison `log_scoring` uses for USCS symbols and header fields, which also credits one text inside the other) and, where the sheet gives a depth, a depth within 0.30 m. A sheet note with no depth is found only in a record note with no depth. |
| `line_kind` | `line` (`""` = the log says nothing of its lines) | `Layer.top_line` is the same after whitespace and case; with `""`, the record must be empty too. |
| `line_width` | `line_full_width` (true or false) | `Layer.top_line_full_width` is the same. |
| `stratum_label` | `label` (`""` = the log prints none here) | `Layer.stratum_label` is the same after whitespace and case, whatever the carried marks say. |
| `label_source` | `label_source` | scored only where the label was found: `_same_text` against `Layer.stratum_label_source`. |
| `label_carried` | `label_carried` (absent = false, printed against this interval) | scored only where the label was found and is not empty: the record's carried mark is the same. A label the record carries down that the sheet marks as printed, or the reverse, misses here and not on the label. |

A saved run is re-scored with the groups by `rescore_saved`, no model call
(tested).

## Tests

New file `report_ingest/tests/test_log_intervals.py`, 51 tests. On master the
file does not import (`report_ingest.log_intervals` does not exist), so every
one fails there; checked by running the file in a worktree of master (see
"Test runs").

- The switch (12): off by default and read from the environment; off, the
  instruction and schema are today's to the byte and the version is not
  raised; on, the instruction gains one paragraph and nothing else, the
  schema is today's plus the interval fields (new fields last, everything
  else equal), it passes the strict-mode rewrite, and the version carries
  `+li`; a saved answer resumes only under its own switch (four stages); a
  real call through `read_logs` offers the paragraph and saves its version;
  off, the same page is read as today; the README lists the switch.
- Intervals (4): a line at every change of description, every interval kept
  and none merged, two adjacent intervals with the same words and symbol
  among them; the printed elevation; the reconciler keeps like intervals
  apart; a log read in two calls keeps every interval.
- Notes (5): a written change with a depth and one without; the depth in the
  log's unit with its page; a negative depth; a depth outside its interval;
  a note with no words.
- Lines (2): a legend that defines a dashed line; a log that says nothing.
- Stratum labels (5): a stratum column and a tag in the description; a label
  carried over three intervals; a label whose extent is not plain; a carried
  mark with nothing above; a source or mark with no label.
- The builder it reuses (1), the saved record (3), a continuation sheet (1).
- Consumers (6): the reconciler, the summary and library page with
  `write_outputs`, the summary unchanged by the new fields, the DIGGS file
  (schema gate and round trip), the library's search words, the question
  list.
- The scorer (9) and the answer sheet's conversion (1).

## Test runs

All with the venv the common brief gives (Python 3.12), judged by pytest's
exit code.

| Run | Result |
|---|---|
| `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py` | 4925 passed, 23 skipped, exit 0 (51 of them the new file) |
| `report_ingest/tests/test_log_intervals.py` alone | 51 passed, exit 0 |
| The same file in a worktree of master 50ef609 | 1 error at collection (`cannot import name 'LayerNote'`), exit 2: every test fails on master |
| Consumers outside `report_ingest`: `module_work/report_ingest_harness/tests`, `funhouse_agent/tests/test_subsurface_write_diggs.py`, `funhouse_agent/tests/test_report_checklist_shim.py`, `funhouse_agent/deep/tests/test_report_ingest_chat_model.py`, `test_report_ingest_page.py`, `test_report_ingest_offline.py`, `test_report_library_offline.py`, `webapp/tests/test_report_library_wiring.py`, `webapp/tests/test_backlog.py`, `webapp/tests/test_docs_currency.py` | 325 passed, 34 skipped, exit 0. The skips: 33 tests that need the private corpus, labels or hand-truthed sheets (gitignored, not on this machine), among them the six of `test_diggs_truth.py` that turn the real log sheets into records and gate their DIGGS files, and one that needs `python_ags4` |

The 23 skips in the first run are the `pydiggs` tests (not installed here);
the DIGGS schema gate the new consumer test uses did run and pass. The graph
snapshot tests are in the first run and pass unchanged.

## Review fixes (2026-11-13)

From the review of 0248175 (verdict merge). Each does nothing with the
switch off and no record carrying the new fields.

1. **A restated interval folds once.** `reconciler._restated` leaves the
   names in `LAYER_INTERVAL_FIELDS` out of its comparison; `_fold_into`
   copies a label with its source and carried mark only where the first
   printing has no label, a line kind with its width only where it has no
   line kind, and appends each later note whose words and depth it does not
   hold (`_same_note`).
2. **The instruction counts intervals:** "Give one layer for every interval,
   from one drawn line to the next, with its depths, ...". The default (off)
   digests in `test_model_first_schema.py` pass unchanged.
3. **A carried-label entry the join makes untrue is dropped.** After
   `graph._join_by_hole` joins a hole's parts, `_recheck_carried_labels`
   asks `carried_label_entries` again of the joined log (switch on only),
   drops a part's entry whose label is printed above on an earlier part and
   lists none twice. Reproduced first with a made-up two-part hole. The
   reconciler's fold of continuation sheets and `sort_rows_by_hole` are not
   re-checked.
4. **Note text is scored by equality** after case, spacing and punctuation
   are set aside (`log_scoring._same_words`), not by one text inside the
   other: "wet" against "wet below 7 ft" is a miss. `_same_text` is
   unchanged.

Six new tests in `test_log_intervals.py` (three for fix 1, two for fix 3,
one for fix 4), now 57; one words check gained the new sentence.

## Prompt versions

`READ_PROMPT_VERSION` did not move (`mf-10`). With the switch off the log
instruction and every answer schema are byte for byte master's; the digest
tests in `report_ingest/tests/test_model_first_schema.py` pass unchanged.
With the switch on, `prompt_version("logs")` is `READ_PROMPT_VERSION` with
the suffix `+li` (`mf-10+li` on this branch's base, `mf-11+li` on master
today). No page
fingerprint changed (`webapp/tests/test_page_fingerprints.py` passes; nothing
re-recorded).

## Release note

None with the switch off: no saved record's values change, and no default
prompt changes. Several code paths run whatever the switch says, and none
changes a value on a record made without it: the library's search words and
the question list's soft-layer reading add words only from fields that are
empty unless the switch was on; the reconciler's `_restated` leaves the new
fields out of its comparison and its fold loop copies them only where a
later printing sets them; the `Layer` serializer leaves the empty new fields
out; the log scorer's interval hook returns at once for a sheet without the
new keys; the results table's OVERALL filter keeps every metric the base
scores; and the carried-label re-check after `graph._join_by_hole` returns at
once with the switch off.

With the switch on (not a default, not measured): a log's `layers` become one
per drawn line, so a record has more, shorter layers than the same log read
with it off, and layers carry notes, a line kind and a stratum label where
the log prints them.

## To measure live

Nothing below was run. Each needs the switch on
(`GEOTECH_INGEST_LOG_INTERVALS=1`) in a fresh output folder (the answers save
under `READ_PROMPT_VERSION` plus `+li`, `mf-11+li` on master today, and are
never mixed with the plain version's).

1. **Today's totals, switch on against off**, on the open logs first (R04,
   R21), then the rest: `layer_top`, `uscs` and OVERALL from the logs stage,
   model first. Intervals are expected to add layer tops, not lose any; a
   fall in `layer_top` or `uscs` is a reason not to turn it on.
2. **Interval counts.** For each log, how many layers with the switch on and
   off, against the number of lines the page draws (from the corrected
   sheet). On the two firms that write changes as notes, the count should
   not rise above the drawn lines.
3. **The interval groups**, once the sheets carry the keys below:
   `interval_note`, `line_kind`, `line_width`, `stratum_label`,
   `label_source`, `label_carried`, per log and summed.
4. **Notes made into intervals, and the reverse.** Read the misses of
   `interval_note` in full: a note found as its own layer is the failure the
   rule exists to stop.
5. **Labels never printed.** On the five logs that print no label, the count
   of layers with a `stratum_label` should be zero (give those sheets
   `"label": ""` on every layer and read `stratum_label`).
6. **Line kinds guessed.** On logs whose legend defines no line style, the
   count of layers with a `top_line` should be zero (`"line": ""`).
7. **Cost**: output tokens per log, on against off (more intervals and notes
   are more output; a long log may now hit the output cap and be halved).
8. **The QA entries** `interval_note_depth` and `label_carried`: how many,
   and whether each is a real misreading.

### The answer sheet format

A layer row keeps its keys of today (`top`, `bottom`, `description`, `uscs`).
One row per drawn line. It may also carry, each optional:

- `notes`: a list, one per change written inside the interval with no line:
  `{"depth": <number, in the sheet's depth unit>, "text": "<as printed>"}`;
  leave `depth` out where the log writes none;
- `line`: the kind of the interval's top line in the legend's or the line's
  own words; `""` to say the log says nothing of its lines (then a record
  that gives a kind misses); leave the key out to not score it;
- `line_full_width`: `true` or `false`; leave out to not score it;
- `label`: the stratum label printed for this interval, as printed; `""` to
  say the log prints none here; leave the key out to not score it;
- `label_source`: the heading of the label's column as printed, or
  `"in the description"`;
- `label_carried`: `true` where the label is printed once above and plainly
  runs down over this interval; leave out (false) where it is printed against
  this interval.

A made-up sheet:

```json
{
  "id": "R00_QX-7",
  "depth_unit": "ft",
  "layers": [
    {"top": 0, "bottom": 4, "description": "MADE-UP FILL, brown sandy clay",
     "uscs": "CL", "label": "FILL", "label_source": "STRATUM", "line": ""},
    {"top": 4, "bottom": 9, "description": "MADE-UP lean CLAY, gray, stiff",
     "uscs": "CL", "line": "solid", "line_full_width": true,
     "label": "Unit 2", "label_source": "STRATUM",
     "notes": [{"depth": 7, "text": "wet below 7 ft"},
               {"text": "as above, but brown"}]},
    {"top": 9, "bottom": 14, "description": "MADE-UP lean CLAY, gray, stiff",
     "uscs": "CL", "line": "approximate", "line_full_width": false,
     "label": "Unit 2", "label_source": "STRATUM", "label_carried": true},
    {"top": 14, "bottom": 20, "description": "MADE-UP silty SAND, dense",
     "uscs": "SM", "line": "solid", "line_full_width": true, "label": ""}
  ]
}
```

`truth_records._layers` turns such a row into a `Layer` with the same fields,
so the DIGGS gate on the sheets keeps working (the DIGGS file leaves the new
fields out).

## Calls made

Each is a choice between two defensible readings, for the lead to revisit.

1. **`top_elevation` added to `Layer`.** Item 1 asks the reading for "the
   elevation where the log prints one"; item 5's list of new record fields
   does not name it. I kept it on the record, since asking a model for a
   value and dropping it helps nobody. The other reading: ask and do not
   keep, or do not ask.
2. **The elevation's unit is the header's `elevation_unit`.** The reading has
   one elevation unit. A log whose elevation column is in another unit than
   its header would be wrong; I have not seen a page that says so, and gave
   the note no unit field of its own.
3. **The line kind is free text, in the legend's words.** The drawn test log's
   legend says a dashed line is "an approximate boundary", and the test
   answer gives "approximate". "dashed" is as defensible. The scorer compares
   exactly (after whitespace and case), so the sheets and the instruction
   must agree on which word: the instruction says "names the kind of line",
   which I read as the legend's meaning word where it gives one. The lead may
   prefer the drawn style.
4. **The restated interval on a continuation sheet stays de-duplicated.**
   The brief says to keep apart every place that de-duplicates layers. I read
   that as "two intervals", and one interval printed on two sheets is one. The
   other reading keeps both copies, which would give the hole two layers at
   one depth.
5. **A note whose depth is outside its interval is kept, flagged**
   (`interval_note_depth`), not moved to the interval that holds the depth and
   not dropped. Moving it is a guess about what the log meant.
6. **A carried mark with no label above is kept as read and listed**
   (`label_carried`), not cleared.
7. **Scorer: a sheet note with no depth is found only in a record note with
   no depth.** The other reading scores words only. I took a depth the log
   never wrote as not a reading.
8. **Scorer: `""` for `line` and `label` scores that the record is empty**,
   so a guessed line kind or a made-up label is a miss. The other reading
   scores only what the sheet prints, and never catches a guess.
9. **Scorer: three groups beyond the brief's two** (`line_kind`,
   `line_width`, `label_source`). The brief says a sheet may carry the line
   kind and the source, and gives rules for notes, labels and the carried
   mark only. `label_source` uses the lenient `_same_text`, since a heading is
   often abbreviated; the label itself is exact.
10. **Scorer: the nearest interval, not any interval within tolerance.**
    Today's `uscs` check takes any layer within 0.30 m. With shorter
    intervals two can sit inside the tolerance, so the new groups take the
    nearest one. Today's groups are unchanged.
11. **The library's search words and the question list's soft-layer reading
    now read notes.** The brief says not to add a field to an export or
    table; these are neither (no row or column changes), and without them a
    word written as a note stops being findable and "soft below 12 ft" stops
    answering the soft-layer row when the switch goes on.
12. **The paragraph's examples** are the brief's own ("wet below 7 ft", "as
    above, but brown"). The paragraph is 1,279 characters (after the review
    fixes) on an instruction of
    1,660; I could not say what a line is, what a written change is, the line
    kind and the label rule in fewer.
13. **Commit trailer**: the round's brief says to use the model I run as, so
    the commits end with that line, not the common brief's.

## Noted, not fixed

- **A continuation sheet's restated layer never fills the first copy's
  base.** In `reconciler._fold_into`, a first sheet's last layer with no
  printed base and the next sheet's copy of it with a base are taken as one,
  and the later copy is left out without its base, so the record keeps the
  layer with no base. That is today's behaviour with the switch off and would
  change a saved record's value (a base depth), so it is worth an item; I did
  not change it because it is outside the switch. Samples and driven records
  are filled this way already (`_join_rows`).
- **No check that intervals are contiguous or in depth order.** Not asked;
  a gap between two intervals may be a real "no recovery" stretch.
- **`log_holes` and `locate` were not given the new fields.** A note has no
  box and is not placed on the page by its words; its provenance is its
  interval's page. If notes need their own placement, `locate.
  investigation_targets` is where.
- **Today's log reader (model-first off) has no interval reading.** By the
  brief.
- **The model-first scoring arm** (`score_on_cluster(model_first=...)`) reads
  the switch from the environment like the graph; no parameter was added.
