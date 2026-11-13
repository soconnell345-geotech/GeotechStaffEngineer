# RI-99 parts (3) to (6): placement and the lab reading's loose ends

Branch `claude/ri-99-placement-3-6`, from master 50ef609. Scope: BACKLOG
RI-99 parts (3) to (6), as section 5 of
`module_work/cloud_briefs/INGEST_2026-11-13.md` decides them. Parts (1) and
(2) are in `module_work/ri99/CHECK.md`. No model and no API was called; every
page and value below is made up.

## (3) Which box a reviewer sees for a lab value

**The rule (the lead's):** the box is where the value itself is printed.

**What model-first placement gives, anchor by anchor** (`lab_targets`,
`report_ingest/locate.py`, read in order; the first that finds one place
wins):

| anchor | the box | on the values? |
|---|---|---|
| `sample column` | the sample's column, over the cells printing the test's results | yes |
| `sample results` (new) | the one row printing two or more of the test's results | yes |
| `sample label` | the sample's label, on a row of a table printed one row per sample | the results' row (the label's cell on it) |
| `result values` | the row printing the most of the test's results | yes |
| `boring label` | the boring's label, only on a row that also prints the test's depth or a result | the results' row |
| `test title` | the sheet's title: page level, or the place of the page's only test | no: nothing on the page names the test or prints two of its results |

**One anchor boxed the label where the values were findable**, so placement
code changed for it. `sample label` runs before `result values`. On a sheet
that names the sample in a heading line (`Sample: S-2`) and prints its
results on a row of their own further down, the label is printed once, so
`sample label` won and the reviewer was shown the heading line, which prints
none of the values.

**Change.** A new anchor `sample results` (`_sample_results`,
`report_ingest/locate.py`, before `_sample_line` in `lab_targets`; added to
`WORD_ANCHORS`, so a test still moves to another page of its call by it, as
it did by its label). It gives the results' row when all of these hold:
the test has two or more result numbers; its sample label, narrowed as
`sample label` narrows it, names it in one place; that label's row prints
none of the results; and one row of the page prints two or more of them
(`_values_row`, today's `result values` test). In any other case it gives
nothing and the anchors after it place the test exactly as before.

**Made-up page** (`ATTERBERG_ONE` in `report_ingest/tests/test_locate.py`):

```
ATTERBERG LIMITS TEST RESULTS
Boring: TB-4      Sample: S-2      Depth (ft): 5.0        (y 90)

Liquid Limit      Plastic Limit      Plasticity Index     (y 150)
41                19                 22                   (y 164)
```

| | anchor | box |
|---|---|---|
| master | `sample label` | `S-2` in the heading line, y about 87 |
| branch | `sample results` | the row `41 19 22`, y about 161 |

**The placement measure** (`module_work/report_ingest_harness/measure_placement.py`).
A lab pair on one page is now also put in one of three groups
(`where_beside`, `sample_block`): `same line`; `same sample block`
(different lines; one box on a line printing the test's own sample label,
or its boring's where it names no sample, the other on a line printing that
label or at least two of the test's results); `elsewhere`. `RESULTS.md` gains the
table "Lab: where the two boxes sit" (counts and shares of the pairs on one
page, per stage or arm and set), `results.json` carries `where` on each lab
pair and `lab_same_line`, `lab_same_block`, `lab_elsewhere` in each
summary's `compare`. The existing "Against the other box" table and its
numbers are unchanged for every stage, so earlier results stay comparable.
The listing "Same page, another line than the other box" leaves out the
`same sample block` pairs: only `elsewhere` is a disagreement to read.

## (4) Two merges that lose or double a value

Both fixes are in `report_ingest/model_first_read.py`, the model-first
window merge. Today's readers are not touched.

**(a) A later remark whose words the first already holds.** The only place
the model-first merge drops a later remark as a substring is the log join
(`_join_logs`: `remarks not in out.remarks`). The lab merge joins no
remarks: a window's tests are listed end to end (see Calls made, 1).
- **Change.** `_join_logs(built, apart=None)` keeps such a remark where
  `apart` says the pages print it separately. `_printed_apart(doc)` is that
  test, with placement's own page search (`locate.text_hits`): the remark
  is found on a page only the later call was shown, or in two places over
  the pages of both. Found once, on a page the earlier call read, it is one
  printed text read twice and is dropped, as before. Found nowhere (a scan
  with no words), nothing can be told and it is dropped, as before.
  `read_logs` passes `_printed_apart(doc)`.
- **Made-up example.** Sheet 1 prints "Remarks: Hole caved at 12 ft after
  the casing was pulled. Backfilled with made-up grout."; sheet 2 prints
  "Remarks: Hole caved at 12 ft after the casing was pulled." Before: the
  joined log's remarks were sheet 1's only. After: sheet 1's, then sheet
  2's. If sheet 2 does not print it (the second answer repeated sheet 1's
  words), the remarks are sheet 1's only, before and after.

**(b) A second calc window's misreading of a value the first read.** Both
windows of a long printout carry its first and last page. With no boxes,
`_fold_readings(same_value=True)` takes two rows for one printed value only
when they read the same value, so a misreading stood beside the right
reading as a second value, with nothing said.
- **Change.** `_printed_once(first, later, doc)` runs before each fold in
  `read_calcs`. A row of each answer under one label on one page, with
  different numbers, neither with a twin in the other answer and no other
  such pairing for either, is set against the page with placement's test
  (`locate._labelled_cells`: the number where the printout around it
  carries the label). Where the page prints one of the two numbers at that
  label, in one place, and the other nowhere at it, the row that does not
  match the print is taken out. `_beside` then puts it on the kept value as
  a `Provenance.alternatives` entry (`field="value"`, `method="model"`) and
  adds one `unresolved` entry (`refused_by: python`, with `read`).
  Anything else is left as it was: both numbers printed at the label (one
  label per load case), neither found (a scan), or more than two readings.
- **Made-up example.** Page 27 prints `Footing width B (ft)   6.5` once.
  Window 1 reads 6.5, window 2 reads 8.5.

  | | inputs | alternative | entry |
  |---|---|---|---|
  | master | 6.5 and 8.5 | none | none |
  | branch | 6.5 | 8.5 (model) | "two calls read this label on page 27 as 6.5 and 8.5; the page prints it once, as 6.5, …" |

  The same holds the other way round (window 1 misreads, window 2 is right).

## (5) A summary row's `sample_id`, described, behind a switch

**What the schema holds today** differs from the BACKLOG line: the test's
own `sample_id` is described ("as printed", `lab_reader.ReadTest`); the
summary ROW's `sample_id` (`lab_reader.ReadRow`) has no description. The
undescribed one is the row's (see Calls made, 2).

**Change** (`report_ingest/model_first.py`): switch
`GEOTECH_INGEST_LAB_SAMPLE_ID_AS_PRINTED`, off by default, read by
`lab_sample_id_as_printed_enabled()` (`use_lab_sample_id_as_printed()` for
a block). On, `answer_schema("lab")` gives three classes with the same
names, docstrings and fields as today's, whose row `sample_id` carries the
test's own description, "as printed"; that one description is the whole
difference in the schema (pinned). `prompt_version("lab")` is then
`READ_PROMPT_VERSION + "+sid"`. Off, `answer_schema("lab")` is
`model_first.LabSheetReading` itself. The lab instruction's words do not
change either way. Today's lab reader is not changed. Listed in
`report_ingest/README.md` beside the other lab switches.

## (6) Entries that repeat their sheet's tests

**Not built as wording.** See "Noted, not fixed".

**The builder defect, found and fixed as a merge defect.** Today's lab
builder keeps every entry of an answer: two entries alike in everything
(kind, boring, sample, depth, every result) on one page stood as two tests
with no QA entry. `_tests_read_twice(tests, doc)`
(`report_ingest/model_first_read.py`, called in `read_lab` for each
answer, before placement) sets such entries against the page with
placement's `_values_row`: where the page prints those results on fewer
rows than the answer has entries, the entries past that count are kept out
and listed (one `unresolved` entry each, with `read`). Entries the page
prints as often as the answer gives them stay; so do a test with fewer than
two result numbers and a page that prints the results on no row (a scan).
- **Made-up example.** The page prints `S-3   4.0   41   19   22` once; the
  answer gives that Atterberg test twice. Master: two tests, no entry.
  Branch: one test, one entry.

## Tests added

All on made-up pages drawn with PyMuPDF or made-up answers.

| test | file | on master |
|---|---|---|
| `test_a_sample_named_apart_from_its_results_lands_on_the_results` | `report_ingest/tests/test_locate.py` | **fails** |
| `test_a_sample_named_apart_moves_to_the_page_printing_it` | same | **fails** |
| `test_results_on_no_one_row_keep_the_samples_label` | same | passes (pins today) |
| `test_a_later_remark_printed_on_its_own_sheet_is_kept` | `report_ingest/tests/test_model_first_read.py` | **fails** |
| `test_a_later_remark_its_own_sheet_does_not_print_is_dropped` | same | passes (pins today) |
| `test_a_later_remark_no_page_can_be_searched_for_is_dropped` | same | passes (pins today) |
| `test_a_second_window_s_misreading_of_a_value_printed_once_is_set_beside` | same | **fails** |
| `test_the_first_window_s_misreading_gives_way_to_the_printed_value` | same | **fails** |
| `test_a_label_printed_twice_with_two_values_keeps_both_with_no_entry` | same | passes (pins today) |
| `test_two_readings_the_page_prints_neither_of_keep_both` | same | passes (pins today) |
| `test_one_printed_test_given_twice_is_kept_once_with_an_entry` | same | **fails** |
| `test_a_test_printed_twice_and_given_twice_is_two_tests` | same | passes (pins today) |
| `test_entries_alike_on_a_page_that_cannot_be_searched_are_kept` | same | passes (pins today) |
| `test_two_tests_that_differ_are_never_one` | same | passes (pins today) |
| the switch: off is today's; on moves one description; saved under its own version (four stages); reads the environment; a window read with it on builds the same record (8 tests) | `report_ingest/tests/test_lab_sample_id_as_printed.py` | the module does not import (the switch is new) |
| three groups; reported apart and only `elsewhere` listed; no lab table without lab pairs (3 tests) | `module_work/report_ingest_harness/tests/test_measure_placement.py` | fail (the functions are new) |

"On master" was run here: the two test files with master's
`model_first_read.py` and `locate.py` give 6 failed, 121 passed, the six
marked above.

## Test runs

- Gate, `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  **4896 passed, 23 skipped (pydiggs not installed), exit 0.** The graph
  snapshot tests and `test_model_first_schema.py` pass unchanged.
- Consumer outside `report_ingest`, `python -m pytest -q -n 4 module_work/report_ingest_harness`:
  **185 passed, 34 skipped, exit 0.**
- Consumers searched over the whole repository: `lab_targets`,
  `WORD_ANCHORS`, `_join_logs`, `_fold_readings`, `answer_schema` and the
  anchor names. Outside `report_ingest` only the harness
  (`measure_placement.py`, which counts anchors by name, so `sample
  results` appears in its tables with no change) reads them. `webapp/` and
  `funhouse_agent/` read none.

## Prompt versions

None moved. The branch does not raise `READ_PROMPT_VERSION`: with the
switch off, every instruction and answer schema is byte for byte what it
was (the digest tests pass unchanged). With
`GEOTECH_INGEST_LAB_SAMPLE_ID_AS_PRINTED` on, `prompt_version("lab")` is
`+sid` on `READ_PROMPT_VERSION` (mf-11 on master today). Page fingerprints
are unchanged.

## Record fields and exports

No record field was added. A misreading set beside its value uses the
existing `Provenance.alternatives`; the entries use the existing
`unresolved` shape. Nothing for the report tables export (RI-209) to add.

## Release note

Under model-first ingest (the default), a newly read report can differ in
four ways; saved items are reused as saved until they are read again.
(1) A laboratory test whose sample is named in a heading line apart from
its results is boxed on the row printing its results (anchor `sample
results`), not on the heading line; its page and confidence do not change.
(2) A log read in more than one call keeps a later sheet's remark that the
sheet prints on its own, even when the first sheet's remarks hold the same
words, so a log's `remarks` may now hold a second remark printed separately
on a later sheet and repeat words already there (never the same remark
twice: a line printed on every sheet is joined once). (3) A calculation
printout read in two windows no longer keeps a second window's misreading
of a value on a shared page as a value of its own: a calc value read two
ways where the page prints it once keeps the printed reading, with the
other reading as an alternative and one QA entry, so such a printout has
one input or result fewer. (4) A laboratory test given twice where the
sheet prints it once (alike in everything, its results on one row) is kept
once, with one QA entry, so such a sheet has one test fewer.

## To measure live

No model call is needed for 1 to 3.

1. **Part (3), the measure.** Re-run
   `python -m module_work.report_ingest_harness.measure_placement` on the
   RI-86 sets. Read "Lab: where the two boxes sit": expect most lab pairs
   that were "same page, another line" (in sample 13 of 24) under `same
   sample block`. Open every `elsewhere` pair on its page: that is the list
   of real disagreements. In "Placed by anchor", count `sample results` and
   open five to confirm the row is the test's own.
2. **Part (4b) and (6), on saved model-first runs.** For each saved
   `items/calc/*.json` of a printout read in two windows, count pairs of
   inputs or results under one label on one page with different numbers.
   For each saved `items/lab/*.json`, count tests alike in everything but
   provenance on one page. These are the cases the two fixes act on at the
   next read; a count of zero means no saved record would change.
3. **Part (4a).** Over saved model-first logs read in more than one call
   (`unresolved` has "read in N calls"), count logs whose later sheets print
   a remarks line: the ones whose remarks can grow.
4. **Part (5), with a model.** On the held-out lab sheets, run the lab
   stage twice under model first, with
   `GEOTECH_INGEST_LAB_SAMPLE_ID_AS_PRINTED` off and on, each in a fresh
   folder. From each run's `placement.json`, count the lab tests placed by
   an anchor that reads the sample id (`sample column`, `sample results`,
   `sample label`) against all lab tests, and the summary rows whose
   `sample_id` is printed on the page as written (`locate.label_hits` finds
   it). The first count cannot move for summary rows: `lab_targets` reads
   only the test's `sample_id`, never a row's (the review's check 8). The
   second count is the measure: the switch earns its place if it rises and
   no test's link is lost.

## Calls made

1. **(4a) is fixed in the log join, though the item says "lab".** The only
   place the model-first merge drops a later remark as a substring is
   `_join_logs`. The lab merge has no remark to join. The other reading:
   the item meant a `LabTest.note` merged somewhere I did not find; I
   searched `report_ingest` for every substring test on a note or remark
   and found none on the lab path.
2. **(5) describes the row's `sample_id`, not the test's.** The code is the
   reverse of the BACKLOG line: the test's is described, the row's is not.
   The brief's rule ("an undescribed field is a schema defect") points at
   the row's. The other reading: strengthen the test's description beyond
   "as printed"; that would be new wording chosen for one model's habit, so
   it is not done. The review agreed: the brief had the two `sample_id`
   levels swapped, and the builder followed the code.
3. **(5) uses the test's own words, "as printed".** No example ids were
   added to the description. The other reading: a longer description with
   examples.
4. **(3) changed placement code.** The brief allows it only for an anchor
   that boxes the label where the values are findable; `sample label` on a
   one-test sheet is that case. "Findable" is today's `result values` test
   (one row printing two or more of the results). A test with one result
   number, or results each on a line of their own, keeps its label line.
5. **(3) "same sample block" is defined by printed words, not distance:**
   one box on a line printing the test's label, the other on a line
   printing the label or at least two of its results (review fix 2: one
   number alone can be another sample's row). The other reading: any two
   boxes within some points of each other. Words do not depend on a
   sheet's spacing.
6. **(4a) when no page can be searched, the later remark is dropped, as
   today.** The other reading: keep it, since it cannot be shown to be the
   same text read twice. Dropping changes nothing on scans, and the later
   words are already in the first remark.
7. **(4a) a remarks line repeated on every sheet of a log is now kept once
   per sheet** where a later one is shorter than the first; identical ones
   are joined only once. That is the lead's rule (printed separately is
   kept). The review found identical ones doubled on the branch (found
   apart on the later sheet, the same words were appended again); review
   fix 1 restored joining them once.
8. **(4b) acts only on a clean pair:** one row from each answer, no twin,
   no third reading, and the print settling it. Everything else keeps both
   values, as today, with no new entry.
9. **(6) fixed in the model-first merge, not in the shared builder.** The
   builder also serves today's reader, whose output must not change.
   The other reading: fix `lab_reader._Builder` for both paths.
10. **(6) entries must be alike in everything.** A `summary_table` entry
    beside a per-row test, or a density entry carrying a compaction sheet's
    points (RI-98's two shapes), are different kinds and are not touched:
    whether those are one printed test is a reading question.

## Review fixes (2026-11-13)

The lead's review of 025d6d5 (verdict "fix first") found two defects, both
fixed; no model or API was called.

1. **A remark printed on every sheet is joined once**
   (`report_ingest/model_first_read.py`, `_join_logs`). The join keeps the
   remarks already joined, starting with the first answer's own, and drops
   a later remark whose whitespace-normalised text equals one of them
   before asking whether it is printed apart. Before: a line printed on
   both sheets and read by both calls came out twice ("X; X"). Test
   `test_a_remark_printed_on_every_sheet_is_joined_once` (made-up). The
   review's cases A, B, E, F, H and I give what they gave before; case C
   is now "once".
2. **The measure's same sample block needs the label or two of the test's
   numbers** (`module_work/report_ingest_harness/measure_placement.py`,
   `sample_block`). A row that does not name the sample's label counts as
   its block only when it prints at least two of the test's numbers.
   Before: placement on another sample's row sharing one number with the
   test counted as `same_block`, hiding a disagreement part (3) should
   list. Test `test_another_samples_row_sharing_one_number_is_elsewhere`
   (made-up): now `elsewhere`.

The review's two low findings (cases D and G) are not fixed; see "Noted,
not fixed".

## Noted, not fixed

- **(6) wording for the lab instruction is not written.** Entries that
  repeat their sheet's tests under another kind were one model's habit on
  one set. A defect that holds on any model would be one of: the builder
  keeping two entries for one printed test with no entry (found, fixed
  above for entries alike in everything); or two entries of different kinds
  whose every value is the same printed cell, which needs boxes the
  model-first answers do not carry.
- The summary row's `investigation_id` is also undescribed
  (`lab_reader.ReadRow`). Not in the item.
- `measure_placement`'s row test compares a displayed-frame box with
  upright-frame rows, as `printed_under` already does; on a sheet scanned
  sideways the three groups can be wrong. The same limit as the existing
  listing.
- `sample results` needs two result numbers. A single-value sheet (a water
  content alone) with its sample named apart keeps the label line.
- (4a) searches a remark by its first six words (`locate.lead`), as
  placement does; two different remarks opening with the same six words
  count as one text.
- (4b) leaves a misreading in place when its number happens to be printed
  at the same label elsewhere on the page.
- Review, case D (low): the earlier sheet prints two remarks that open with
  the same six words, and the later answer repeats one of them, which its
  own sheet does not print. The six-word search finds it in two places, so
  the remark is doubled. It needs an answer to give words its pages lack
  (inferred rare). No fix.
- Review, case G (low): in (4b), a true second value printed as feet and
  inches (`7'-9"`) cannot be read by the number search, so it is set beside
  the first value as an alternative, with an entry saying the page prints
  it once; it is not lost. It needs each window to read only one of two
  values on a shared page (inferred rare). No fix.
