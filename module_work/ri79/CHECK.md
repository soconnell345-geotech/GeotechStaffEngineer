# RI-79: the log scorer's conventions for a paired public answer

Branch `claude/ri-79-scorer-conventions`, from master 50ef609. Section 2 of
`module_work/cloud_briefs/INGEST_2026-11-13.md`.

Nothing here changes how a reader reads. The changes are in the scorer
(`report_ingest/log_scoring.py`), the logs RESULTS writer
(`report_ingest/cluster_scoring.py`, one line when something was set aside),
the AGS4 converter (`module_work/report_ingest_harness/ags_truth.py`) and
their tests. No prompt, schema, record field or export changed. The public
blind set was not opened or fetched; every record, answer sheet and AGS4
text in the tests is made up.

## What changed, item by item

### 1. Comma-joined increments

`log_scoring._blow_numbers` read a string entry through planlens'
`numbers_in`, whose thousands rule stops at the first comma: `"3,6"` gave 3,
`"12,15,20"` gave 12, `"1,2/2,3,3,4"` gave 1 and 2.

Now `_blow_entries` (`log_scoring.py:192`) reads every increment of an
entry, however joined (comma, slash, dash, plus, spaces), and
`_blow_numbers` (`log_scoring.py:232`) returns their counts.

| made-up record entry | master | branch |
|---|---|---|
| `"3,6,8"` | 3 | 3, 6, 8 |
| `"1,2/2,3,3,4"` | 1, 2 | 1, 2, 2, 3, 3, 4 |

Answer blows `[3, 6, 8]` against a record that kept `"3,6,8"` as printed:
missed on master, found on the branch.

### 2. A count over a penetration

A drive stopped short is one increment whose count is the count. The
penetration is kept beside it, never among the counts
(`_blow_entries`, same place). Read as such: `50/5"`, `50/20mm`, `50/0.1m`,
`21 for 60mm`, `50 blows for 20 mm`, and a lone `50/20` with no unit. Units
are folded (`"`, `in`, `inch` are one). `model.SPT`'s docstring is the basis:
`"50/5\""` is "50 blows over 5 in, a drive stopped short of its increment,
not the number fifty".

| made-up | master | branch |
|---|---|---|
| record `[10, 15, "50/20"]`, counts read | 10, 15, 50, 20 | 10, 15, 50 |
| answer blows `[10, 15, 50]` against it | found only by the sub-run rule | found |
| answer blows `[10, 15, 50]`, N fallback sees | 4 increments | 3 increments |

Penetrations are still compared where both sides state one in the same unit
(`_penetrations_agree`, `log_scoring.py:239`): answer `50/20mm` against a
record's `50/60mm` is a miss, as it was on master (where both numbers were
compared as counts). A side that states none (the AGS4 answer carries counts
only) is not held against the other.

### 3. Blows on a driven tube sample

**The record has a slot, so no "no slot" group was added.** `model.SPT` is
"one driven penetration record", its docstring says a ring or California
sampler's drives keep `n` None, the reader's answer class `ReadSPT`
(`log_reader.py:207`) is "one driven record, as the log prints it", and the
model-first log instruction asks for "every SPT or other blow-count record"
(`model_first.py:480`). `truth_records._samples` already builds a drive
record from any answer sample that has blows. So a tube's blows belong on a
drive record of their own at the sample's depth, and the scorer already
looked there.

What was wrong was the other direction. The scorer credited a tube's single
count from any drive record in the sample's window whose increments
contained that number. A tube at 2.00 to 2.45 m with 8 blows was "found"
among the six increments of the penetration test that follows it at 2.45 m.

- The converter marks the tube's count: `"blows_kind": "sampler"`
  (`ags_truth.py`, `_samples`).
- The scorer, for a sample so marked, finds the blows only on a drive record
  in the window whose counts are those blows and no others
  (`log_scoring.py:796`, in `score_record`). It stays in
  the `blows` metric; its miss line reads `sampler blows [...]`.
- A sheet without the mark (every hand-written sheet) is scored exactly as
  before, pinned by `test_a_sheet_without_the_mark_is_scored_as_before`.

| made-up: tube `U2` 2.00-2.45 m, 8 blows | master | branch |
|---|---|---|
| record: drive record at 2.00 m, blows `["8 blows"]` | found | found |
| record: only a test at 2.45 m, `[1,2,2,3,8,4]`; "8 blows" in the sample's note | found (wrongly) | miss |

Where a reader put the count in the sample's note and nowhere else, that is
a miss, and it is the reader's: the record has the place for it.

### 4. N from the increments

What is held in print (searched `geotech-references/` at the pinned
88bf0bc):

- **Three increments.** FHWA NHI-06-088 Vol. I, Sec. 3.7, p. 3-51
  (`fhwa_soils_foundations/text/chapter03.json`): "three successive 6 in
  (150 mm) increments ... The first recorded increment is considered as a
  'seating' penetration, while the number of blows to advance the second and
  third increments are summed to give the N-value".
- **Six increments: no rule held.** EN 1997-2:2007 Sec. 4.6.1 (2)P, p. 49
  (`eurocode_7_2/text/section4.json`, the library's own wording of the
  clause) gives N as the blows "for 300 mm of sampler penetration ... below
  a seating drive" and prints no increments; Sec. 4.6.2 sends the procedure
  to EN ISO 22476-3, which the library does not hold. No copy of the AGS4
  data dictionary is in the repository (searched for `ISPT_INC`, `ISPT_SEAT`:
  only `ags_truth.py` itself).
- **Four increments: not settled by the print.** Fig. 3-20 (p. 3-52) prints
  both "Occasional Fourth Increment Used to provide additional soil
  material" and "total number of blows to drive sampler last 300 mm".

So:

- (a) `N_RULES = {3: (1, 2)}` with the cite in the code
  (`log_scoring.py:257-270`); `_n_by_rule` computes an N for three
  increments only.
- (b) For any other number of increments nothing is computed. Where the
  answer has an N, the record prints none, and the answer's drives stand at
  that depth on a drive record with no N, the N goes to
  `LogScore.set_aside["n_no_rule"]` ("N not printed, no rule held for this
  number of increments"): counted and listed, in no metric and no total.
  Same in `score_grid`. An N the record prints is scored as ever, right or
  wrong; an N with no such drives at that depth is still a miss.
- (c) The record's own printed-N check was **left alone**: see "Calls made".

| made-up: answer N=12, increments 1,2/2,3,3,4 | master | branch |
|---|---|---|
| record prints the six drives, no N | miss (2+2=4 is not 12) | set aside |
| record prints N=12 | found | found |
| record prints N=11 | miss | miss |
| answer N=21 on 5, 9, 12, record has the drives only | found | found |

`set_aside` is in the run file's `after` (and in `model_alone` /
`floor_alone`) only when not empty, as `{group: {count, what, items}}`.
`RESULTS.md`'s logs section prints one line per group per set when there is
any (`cluster_scoring.py`, `_score_logs` / `_render_logs`); with none, the
file is byte for byte what it was.

### 5. Sample ids

The log scorer does not score sample ids (no metric reads a sample's `id`),
so there was no comparison to adjust. The id reaches the record that
`truth_records` builds from a sheet (`Sample.sample_id`, `SPT.sample_id`),
which the placement measure reads.

The converter now builds the id as a log prints a sample
(`ags_truth.sample_id`): type and reference together, `D` + `1` = `D1`. A
reference that already leads with its type (`D1`) is kept; with no reference
there is no id. The bare reference stays on the sheet as `ref`, and the
laboratory rows find their sample by it.

| made-up SAMP row | master `id` | branch `id` / `ref` |
|---|---|---|
| type `D`, reference `1` | `1` | `D1` / `1` |
| type `U`, reference `2` | `2` | `U2` / `2` |

### 6. Header fields the form may not print

- The sheet carries `fields_from_data` (every scored header field, all taken
  from the data) and `fields_struck: []`. Nothing is struck by the
  converter. `build` keeps a reviewer's strikes when it rewrites a sheet.
- The scorer leaves a struck field out of `fields` in both `score_grid` and
  `score_record`, and counts it under `set_aside["fields_struck"]`.

Made-up: fields `client`, `project`; record holds the project only.
Unstruck: fields 1/2. With `"fields_struck": ["client"]`: fields 1/1, one
set aside.

## Tests

New file `report_ingest/tests/test_log_scoring_ri79.py` (33 tests) and seven
new tests in `module_work/report_ingest_harness/tests/test_ags_truth.py`
(13 with parameters); two assertions there changed (`"1"` to `"D1"`, `"9"`
to `"B9"`).

Run against master's `log_scoring.py` and `ags_truth.py` (the two source
files stashed, the test file's import made tolerant of the missing names):
36 failed, 26 passed. Failing on master for the behaviour, not for a missing
name:

| item | test |
|---|---|
| 1 | `test_every_increment_of_a_joined_record_is_read[3,6]`, `[1,2/2,3,3,4]`, `[12,15,20]`; `test_a_record_written_as_one_string_matches_its_increments`; `test_an_answer_written_as_one_string_is_read_whole_too` |
| 2 | `test_a_refusals_penetration_is_not_a_blow_count` |
| 3 | `test_a_tubes_blows_are_not_found_among_another_tests_increments`; `test_a_tubes_blows_are_marked_as_the_samplers` |
| 4 | `test_six_increments_with_no_printed_n_are_set_aside_not_missed`; `test_a_six_increment_n_is_never_credited_by_the_three_increment_rule` |
| 5 | `test_a_sample_id_is_its_type_and_reference_together`; `test_a_borehole_fills_the_logs_schema` |
| 6 | `test_a_struck_field_is_in_no_total_and_an_unstruck_one_is_scored`; `test_a_struck_header_field_is_left_out_of_the_total`; `test_the_header_fields_taken_from_the_data_are_listed` |

The rest of the 36 fail on master only because they read `set_aside`,
`N_RULES`, `_blow_entries` or `rescore_run`, which master does not have
(for example `test_three_increments_with_no_printed_n_are_credited_by_the_rule`
pins behaviour master already had).

Runs on the branch, by pytest's exit code:

- `python -m pytest -q -n 4 report_ingest module_work/report_ingest_harness webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py webapp/tests/test_backlog.py`:
  **5,130 passed, 56 skipped, 0 failed** (exit 0). The skips are
  dependencies this venv lacks (pydiggs and others), named in the run.
- `python-ags4` was installed in the venv so the converter's tests ran
  (they skip without it).
- The existing `report_ingest/tests/test_log_scoring.py` (49 tests) passes
  unchanged.

Consumers checked: `LogScore.to_dict`, `_scored` and `rescore_saved` are read
by `cluster_scoring.py`, `model_first_scoring.py`, `sounding_scoring.py` and
the harness (`run_plain_baseline.py`, `measure_templates.py`,
`measure_wp2b_logs.py`, `dry_run_model_first.py`). Every one sums `scores`
by name; `set_aside` is a separate key so no total picks it up. Nothing in
`webapp/` or `funhouse_agent/` reads the scorer or the converter.

## Prompt versions

None moved. No text a model is sent changed.

## Release note

None. No saved record's values change: the scorer and the converter are not
part of a record. Scores of saved runs change when re-scored (below).

## How a saved run is re-scored without a model call

A logs run file (`<out_dir>/logs/<log id>.json`) keeps the record the reader
built. After rebuilding the answer sheets with this branch:

```
python -m module_work.report_ingest_harness.ags_truth build --ids P01,P02,P03
python -m module_work.report_ingest_harness.ags_truth rescore <out_dir> \
    --truth-dir <set>/raw/truth/logs
```

`rescore` prints, per metric and overall, found/total as saved and as
re-scored, and the set-aside counts. It reads no PDF and calls no model. Add
`--write` to replace each run file's `after` (the saved one is kept beside
it as `after_as_run`, and writing into a tracked path is refused). In code:
`report_ingest.log_scoring.rescore_run(logs_dir, truth_dir, write=False)`.
It prints counts only.

Rebuild the sheets first: items 3, 5 and 6 are on the sheet
(`blows_kind`, `id`, `fields_from_data`). Items 1, 2 and 4 apply to the old
sheets as they are.

## To measure live

1. Re-score the paid public-set run as above. Report, saved against
   re-scored: `blows`, `n_value`, `fields`, overall, and the counts of
   `n_no_rule` and `fields_struck`.
2. For `n_no_rule`: how many of those records print the N on the scan after
   all. This needs someone allowed to open the pages. If most do, the reader
   is dropping a printed N and the group is hiding misses; if most do not,
   the form prints drives only and a six-increment rule page is what is
   needed (below).
3. Tube blows: of the `sampler blows` misses, how many records hold the
   count in the sample's note and how many nowhere. Both are reader misses;
   the split says whether the log instruction's "or other blow-count record"
   is being followed.
4. Header fields: a reviewer who may open the scans strikes, per sheet, the
   fields in `fields_from_data` the form does not print, then re-score.
5. The hand-truthed corpus logs, re-scored the same way, to confirm their
   totals move only where a record kept increments in one string or a
   four-increment record prints no N (expected: few or none).

## Calls made

Each is a judgement between two defensible readings.

1. **Four increments have no rule.** The repository already reads Fig. 3-20
   as "the N is still the 2nd + 3rd" for a fourth increment
   (`log_floor.py`, RI-50 / RI-70), and master's scorer credited any record
   of three or more increments by 2nd + 3rd. I took the stricter reading:
   the figure prints "last 300 mm" as well, and a four-count record is also
   what a six-increment test halted early looks like. Other reading: add
   `4: (1, 2)` to `N_RULES` with the Fig. 3-20 cite; one line.
2. **The bundled AGS4 dictionary was not used as a held page.**
   `python-ags4` (an optional dependency, installed here only to run the
   tests) ships the AGS4 standard dictionary. It names `ISPT_INC1` and
   `ISPT_INC2` "(Seating)" and `ISPT_INC3` to `ISPT_INC6` "(Test)", and
   `ISPT_MAIN` "Number of blows for main test drive". The brief allows the
   dictionary only if a copy is held in the repository, and none is. It also
   does not print that N is the sum of the test increments. If the lead
   takes it as held, the entry would be `6: (2, 3, 4, 5)`; a halted record
   of fewer increments would still have none.
3. **The record's own printed-N check was not changed** (item 4 (c)). It is
   in `log_floor.py` (the floor's and the merge's 2nd + 3rd checks and
   remarks), which is reader and merge code: changing it would change QA
   entries, `n_basis` and in places `n` in saved records. On a six-increment
   log it will set a printed N against the 2nd + 3rd of the first three
   increments and may remark that they differ. That is worth a backlog item
   of its own if the re-score shows such remarks on the public set.
4. **A set-aside N needs the answer's drives standing in the record.** The
   same condition the three-increment credit has always had. Other reading:
   set aside every answer N on a non-three-increment record that the record
   does not print, drives or no drives. I kept those as misses: with no
   drive record at the depth the reader has missed the test, whatever the
   rule.
5. **A lone `A/B` with no unit is a count over a penetration**, as the brief
   says of `50/20`. So a two-increment record printed `3/6` reads as 3 blows
   over 6. Three or more (`3/6/8`) are increments. Inside a longer string a
   slash needs a unit to be a penetration (`10, 15, 50/20mm`); without one
   (`10, 15, 50/20` as one string) the 20 is read as a count, because
   `1,2/2,3,3,4` must read as six. A record that keeps one entry per
   increment, as `model.SPT` asks, does not meet this.
6. **Penetrations are compared only where both sides state one in the same
   unit.** Other reading: ignore them altogether. Kept, so a hand sheet's
   `50/5"` is still not matched by a record's `50/3"`.
7. **A tube's blows stay in the `blows` metric**, not a group of their own,
   because the record has a slot for them. Other reading: report them
   apart so the SPT blows rate is comparable across sets; `counts()` in the
   converter already reports `tube_blows` separately on the sheet side.
8. **The sample id is joined with no space** (`D1`). The scorer does not
   score ids; the placement measure folds spacing and punctuation. A log
   that prints the type and number in two columns would be `D` and `1`.
9. **`fields_from_data` lists the scored fields only** (those in
   `FIELD_MAP`): an unscored field needs no strike.

## Noted, not fixed

- `log_floor.py` has its own `_blow_numbers` (reader side, the floor's
  comparison of two blow records). Not touched: nothing here may change how
  a reader reads.
- `measure_wp2a_loggrid.py` restates the scorer's rules for the grid alone
  and still has the old 2nd + 3rd fallback and comma reading. It is a
  harness script over the hand corpus and was left as it is.
- The converter does not keep the seating/test split of an `ISPT_REP`
  bracket apart (`1,2/2,3,3,4` becomes six counts); the report string is on
  the sheet as `spt_report`. A six-increment rule, once a page is held,
  could be applied from the count alone.
- The page needed for a six-increment rule: the test procedure clause of
  EN ISO 22476-3 (seating drive and test drive increments, and how N is
  reported), or the AGS4 data dictionary's ISPT group held in the
  repository with the lead's ruling that it prints the rule.
