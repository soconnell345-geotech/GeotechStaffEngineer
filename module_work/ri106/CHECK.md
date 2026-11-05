# CHECK: RI-106 and RI-107 (branch `claude/ri-106-107-groundwater-composites`)

Two distinctions the record must keep, both from the owner's checklist
markup (OWN-31, folded by RI-105). Commits: `5f8a146` (RI-106), `d05e961`
(RI-107), then this file. No model or API was called. Every log page, sheet,
hole name and depth in the tests is made up.

## Switches, and what the default run does today

| Behaviour | Switch | Default run |
|---|---|---|
| RI-106 prompt text: a paragraph after today's log reader's water bullet, a sentence in the model-first log instruction | `GEOTECH_INGEST_WATER_NOT_MEASURED` (new; `water_status.use_water_not_measured()`) | **OFF**: both prompts byte-identical to master (tested) |
| RI-106 record value `when='not_measured'`, the floor reading it off a groundwater field, the reconciler's `settle_water`, DIGGS, checklist | none (pure Python, works whichever reader filled the field) | **ON** |
| RI-107 combined-sample links (label and depth) | none (pure Python in the reconciler, no prompt text) | **ON** |
| RI-07's label prompt (`GEOTECH_INGEST_LAB_LINK_FROM_LABEL`) | unchanged | OFF as before; RI-107 links a hole read out of a label whether the label reached `investigation_id` or `sample_id` |

So today's default run: the models are asked exactly what they were asked
on master; a model that answers `not_measured` anyway is now kept (it was
turned into `unknown` with an "outside" entry); a floor or reconciler that
finds "not measured" in the printed words, or a fluid drilling method with
no water row, files the hole as not measured; combined lab tests link to
every sample.

## RI-106: groundwater 'not measured' apart from 'not encountered'

**What and why.** A boring drilled with water or mud and no well has no
reading; that is not water below the bottom of the hole.

- `report_ingest/model.py:1228`: `WaterWhen` gains `not_measured` (no depth).
- `report_ingest/water_status.py` (new): `says_not_measured` (:154; "water
  level not measured", "not recorded", "not obtained", "obscured by drilling
  fluid", "no readings", N/M; "not observed" is deliberately NOT one, logs
  print it for none seen), `fluid_method` (:176; mud rotary, wash boring,
  rotary wash, drilling fluid, slurry, bentonite, water added), and
  `settle_water` (:197): a row with no depth whose own note says not
  measured is refiled; a hole with no reading whose water header field or
  remarks say so gains a `not_measured` row (prov `derived`, 0.6); a
  boring/well with NO water row at all, drilled by a fluid method, gains one
  (prov `derived`, 0.5, the method in the note). A reading is never
  overruled; a printed "not encountered" on a mud-rotary boring is kept.
- `report_ingest/reconciler.py:531,556`: `_settle_water` runs on every hole
  right after the continuation fold, so both readers' records get it; one
  `note` QA entry (`where="investigations.water"`) names each change.
- `report_ingest/log_floor.py:3400`: a groundwater field saying not
  measured is `not_measured` (was: no row at all).
- `report_ingest/log_reader.py:1391`: the builder (shared by today's reader
  and model first) keeps `not_measured`.
- Prompt text: `log_reader.py:640` `with_water_not_measured` inserts
  `water_status.LOG_READER_PARAGRAPH` (:87) after the water bullet;
  `model_first.py:614` appends `MODEL_FIRST_SENTENCE` (:98) after "and the
  water levels." (only for `INSTRUCTIONS`, never the plain baseline). Both
  behind `GEOTECH_INGEST_WATER_NOT_MEASURED` (:59).
- DIGGS: `diggs_writer.py:447` `_not_measured`; `:1152` writes a remark
  `Groundwater not measured.` on a borehole or a pit, NEVER `notEncountered`
  and no `WaterStrike`; a `not_measured` row with a depth is not a level.
  `subsurface_characterization/diggs26.py:846` `NOT_MEASURED_REMARK`, `:830`
  reads it back as `gwl_not_measured`; `site_model.py:286` the field
  (written by `to_dict` only when True), summary and soil-profile note;
  `plots_plan.py` labels; `subsurface_adapter.py` passes it only when True.
  The round-trip gate (`diggs_writer.py:3778`) says when it is lost or
  appears from nowhere.
- Checklist: `checklist.py:825` `check_groundwater` groups the logs into a
  reading / not encountered / NOT MEASURED / nothing; `:933` a judge item
  naming a `check` gets its finding as `code_said`.
  `report_review_checklist.json`: `completeness.groundwater` stays `judge`,
  gains `"check": "groundwater"`; its text is unchanged; `how_to_edit` says
  judge items may name a check.
- `question_list.py` (`_reading`), the library page's water column: say
  "not measured", not "not encountered".

**Before/after (made up).** Log QB-1, header "Advancement Method: Mud
Rotary", footer "Groundwater: water level not measured". Master: floor
gives no water row; DIGGS has no `waterStrike`; checklist hands
"groundwater" to the agent with nothing. Branch: `water=[{when:
not_measured, depth: null, note: "Groundwater: water level not
measured"}]`; DIGGS remark "Groundwater not measured."; read back
`gwl_not_measured=True, gwl_not_encountered=False`; checklist code_said
"groundwater NOT MEASURED ... 1 (QB-1)". Log QB-2, "Mud Rotary" and no
water line at all: master no row; branch a derived `not_measured` row,
confidence 0.5, note "no water reading printed; drilled by 'Mud Rotary'...".

**Tests.** `report_ingest/tests/test_water_not_measured.py`, 48 tests:
the floor on three made-up log pages (planlens' imperial fixture with its
groundwater note and method rewritten), the printed words (positive and
negative cases incl. "not observed"), `settle_water` (9 cases), the
reconciler note, today's reader (model answer kept; default system prompt
== `LOG_READER_SYSTEM`; switch inserts the paragraph and nothing else; env
switch), model first (default instruction unchanged; switch adds the
sentence; plain baseline untouched; model answer kept through
`read_logs`), DIGGS (borehole, pit, not-encountered unaffected, gate
catches a loss, depth not a level, site_model round trip), checklist (3).
On master: 18 fail (every behaviour test); the 30 that pass are the
`water_status` word tests (the module is new; run against master's package
they resolved it from the branch) and the default-unchanged / regression
guards. `test_checklist.py`'s pin of the item map updated for the new
`check`.

**Prompt versions.** None moved: with the switch off every prompt is byte
for byte master's (tested); `READ_PROMPT_VERSION` stays `mf-8`. The page
fingerprints test passes unchanged.

**Release note.** Groundwater "not measured" is now its own answer. A
boring whose log prints that the water level was not measured (or not
recorded, or hidden by drilling fluid) gets a water row `not_measured`
with no depth instead of no row, or instead of "not encountered" where a
reader had filed it so with those words in its note; a boring or well with
no water row at all whose drilling method is mud rotary, wash boring or
otherwise fluid gets a derived `not_measured` row (confidence 0.5). The
DIGGS file says it in a remark, never as `notEncountered`, and the app's
DIGGS reader returns `gwl_not_measured`. The report gains one QA note
listing the holes changed.

## RI-107: lab tests on combined samples

**What and why.** A test run on two or more samples combined was linked to
one sample or to none (RI-71 sent a composite meeting several named samples
to no sample with a conflict).

- `report_ingest/model.py:2172` (LabTest), `:2010` (SummaryRow):
  `linked_sample_ids` (shallowest first) and `combined`, written only when
  set (no schema bump; `SCHEMA_VERSION` comment says why).
  `linked_sample_id` keeps its meaning and holds the first sample, so a
  reader of that field alone (checklist, writers) still sees the test linked.
- `report_ingest/reconciler.py:1499` `_combined_samples`, asked before
  `_sample_for` for every sheet (`:1302`) and summary row (`:2077`):
  - by the label: two or more of the hole's samples named (`_named_samples`,
    `:1470`, factored out of `_sample_for` unchanged; ranges of names as
    RI-71), with a combine word (`_COMBINE_WORD`, `:1494`: `&`, `+`, `and`,
    composite, comp., combined, mixed, blend) or a printed interval meeting
    every one; a hole read out of the label (RI-07) works the same;
  - by the depth: where the label names none of the hole's samples, a
    printed top AND bottom holding two or more samples with no one sample
    spanning it.
  `_link_combined` (`:1567`) fills the fields; one QA note per report
  (`where="lab_tests.combined"`, `:1334`). Links are reset on every
  reconcile.
- The library page's lab table shows "S-3 + S-4 (combined)"; the library's
  lab rows give the joined samples and `combined: true`.

**Before/after (made up).** Sheet "QB-3, S-3 & S-4, 3.0 m"; log S-3
3.0-3.45 m, S-4 4.5-4.95 m. Master: `linked_sample_id="S-3"` (the depth
meets S-3 only). Branch: `linked_sample_ids=["S-3","S-4"], combined=true,
linked_sample_id="S-3"`. Sheet with no label, 3.0-4.95 m: master S-3
(depth_top only); branch S-3 + S-4. "S-2,S-3", 7-16 ft meeting both:
master no sample + conflict; branch both, no conflict.

**Tests.** `report_ingest/tests/test_combined_samples.py`, 22 tests: seven
printed label forms, no depth, a range of names, comma names with a depth
meeting both, the hole inside the label, the report note, three depth cases
(two samples; a bulk bag spanning the interval stays one; a short interval
stays one), unchanged single-sample cases, re-reconcile clears the flag,
summary rows, library page, checklist, round trip. On master 21 fail, 1
passes (a regression guard). `test_reconciler.py::test_an_interval_meeting_several_named_samples_is_a_composite`
pinned RI-71's "to no sample"; updated to RI-107's behaviour.

**Prompt versions.** None: no model-facing text changed. I did not add a
lab-reader prompt about combined samples; the readers already copy the
label as printed, and whether they keep "S-3 & S-4" whole is listed below
to measure.

**Release note.** A laboratory result run on samples combined ("S-3 &
S-4", "S3+S4", "composite of", "comp.", or a printed depth range holding
two or more of the log's samples) is now linked to every sample it
combined (`linked_sample_ids`, `combined: true`; `linked_sample_id` the
shallowest). Before, it was linked to the one sample its top depth met, or
(a composite whose interval met several named samples) to none, with a
conflict entry, which is gone. The report gains one QA note listing them.

## Test runs

- Baseline on master before any change:
  `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`
  4394 passed, 8 skipped (pydiggs absent), exit 0.
- Same command on the branch: first 4463 passed + 1 failed (the RI-71 pin);
  after updating it, the final run: **4464 passed, 8 skipped, 0 failed,
  exit 0** (70 new tests).
- Consumers: `subsurface_characterization`, `funhouse_agent/tests/test_subsurface_adapter*.py`,
  `module_work/report_ingest_harness`: 481 passed, 58 skipped, exit 0.
  Every webapp/funhouse test that imports report_ingest, the checklist or
  the DIGGS reader (17 files): 456 passed, 2 skipped, exit 0.
- `webapp/tests/test_public_text.py` before each commit: passed.

## To measure live

RI-106, on the open logs (no model call was run here):
1. Count the logs that print a drilling method with mud, wash, fluid or
   water added, and for each, what the log prints about water. How many get
   a derived `not_measured` row, and is any of them a hole with a printed
   reading the readers missed (then the row is wrong and the reader is the
   fix)?
2. Grep the logs' groundwater lines and remarks for not-measured wording the
   word list misses (other languages, "WL N.R.", "not gauged").
3. Run the logs stage with `GEOTECH_INGEST_WATER_NOT_MEASURED=1` beside off,
   both readers (model first and today's), and compare water rows per hole
   against the hand truth: does the switch move rows from `unknown`/
   `not_encountered`/nothing to `not_measured` correctly, and does it move
   any true "not encountered"? Only then turn it on by default.
4. The log truth files may need `not_measured` where they hold
   "not encountered" for such holes (`truth_records.py` maps phrases to
   `not_encountered` only).

RI-107, on the open lab sheets:
1. Search the sheets and summary tables for "&", "+", "and", "comp",
   "composite", "combined", "mixed" in sample labels, and for depth ranges
   longer than one sample: list the printed forms actually present.
2. Check the lab readers keep "S-3 & S-4" whole in `sample_id` (or the hole
   and both samples, with the RI-07 switch on) rather than collapsing it to
   one name; if they collapse it, a prompt line behind a switch is the next
   step.
3. For tests linked by depth alone to two samples, check against the sheet
   that the test really was combined (a test on a bulk bag the log does not
   list, inside a range holding driven samples, would read as combined).

## Noted, not fixed

- A printed "not encountered" on a mud-rotary boring is kept as printed; no
  QA flag says it is doubtful.
- A hole with both a "not encountered" and a "not measured" row and no
  reading is written to DIGGS as `notEncountered` only.
- `fluid_method` reads the method field only; a fluid method printed only
  in the remarks does not derive a row.
- "S-3 & 4" (second sample as a bare number) is not read as combined (bare
  numbers never name a sample in `_named_samples`, as before).
- Library rows keep one `linked_sample` text (joined for a combined test)
  rather than a list column.
