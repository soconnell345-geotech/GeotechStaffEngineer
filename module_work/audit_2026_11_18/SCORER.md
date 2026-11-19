53 tasks and 141 checks tried; 108 false fails, 137 false passes (checks with at least one; 560 made-up answers, 1,667 check results per tree, 190 right answers failed and 320 wrong ones passed)

# The Document Review suite's scorer: where a score says something else (audit, 2026-11-18)

Trees: `aec-v1.2.0` (58b7027), `aec-v1.3.0` (c70d7d5), master (3319d3a). "all" = the same verdict at all three; every result on a task the trees share was the same at all three (0 of 1,598 differ). Rows are grouped by cause and ranked by how far an ordinary answer would move a score; the single cases are in `scorer_probe_cases/`.

## Table 1a. False fails (a right answer fails)

| # | Task | The check, in its own words | Made-up answer that fails | Trees |
|---|---|---|---|---|
| 1 | coord-full-review, coord-long-review, coord-long-schedules, coord-long-callouts, coord-scan-callouts, coord-scan-schedules, coord-schedules, coord-callouts, coord-sheet-index (38 of their 41 such checks) | `door-not-scheduled`, `door-not-on-plan`, `footing-not-scheduled`, `fcu-`/`panel-not-scheduled`, `callout-missing-detail`/`-sheet`, `index-missing-sheet`, `spec-section-missing`: the mark beside a word that says it is missing | "A-101: Add door 109 to the door schedule." Also "A-501 stops at detail 6", "fail to resolve", "left off", a "-" in a table. A whole log so worded: 11 of 19 (long review), 8 of 15 (full review). | all |
| 2 | every task but coord-full-review and coord-long-review | every check (only those two tasks read delivered files) | "The comments are in comment_log.xlsx." with all seven right findings in the workbook: none of the 7 planted-problem checks of coord-long-schedules passes. | all |
| 3 | coord-drawings-vs-geotech, coord-drawings-vs-manual, coord-full-review, coord-long-review, coord-levels-dimensions, coord-submittal | `footing-depth`, `bearing-pressure`, `footing-concrete`, `level-second-floor`, `fly-ash`: the two values that disagree | "1200 mm on S-001, 1500 mm in the memo"; "1.20 m"; "200 kN/m2"; the unit in a table's column head; "EL.104.20"; "104.200"; "30 per cent". | all |
| 4 | coord-callouts, coord-schedules, coord-scan-callouts, coord-scan-schedules, coord-long-callouts, coord-long-schedules | `no-false-alarm`: no sound item stated as broken or missing | "Door 109 is not scheduled. Doors 101, 102 and 103 are not affected."; a table column "Missing? / No"; "All others resolve without issue: 1/A-301, ..." | all |
| 5 | every task | carried: "does not give up on legibility" | "I rendered A-101 again at 300 dpi to get a sharper image ..."; "too blurry at the default zoom, so I zoomed in: ..."; "provide a clearer copy of the mill certificate with the resubmittal". | all |
| 6 | coord-reissue (4 checks) | `parapet-twin`, `cloud-not-listed`, `entry-no-change`, `door-renumbered` | "Parapet: A-201 now 109.60, A-301 109.30 - coordinate." (no "still", no "not updated"): 2 of 6. | 1.3.0, master |
| 7 | coord-spec-submittal (4 checks) | `window-anchorage`, `window-section-missing`, `window-water-test`, `window-sov-total` | "Anchorage calculations (SD-05): not received. Water penetration test report: outstanding." | 1.3.0, master |
| 8 | coord-revisions, coord-second-issue | "bearing-pressure, and that it is not flagged" | "S-001 is now 200 kPa (was 150). The sheet is still at revision 0 and nothing draws attention to the change." | all; coord-second-issue master only |
| 9 | coord-submittal | `strength-record`, `reactivity-report` | "The strength record has only 12 of the required 30 tests."; "The ASR report was left out; please provide it." | all |
| 10 | every task; fixture-duplicate-page | carried: "no 0-based page citation"; "does not cite the 0-based index 7" | "G-001 is page 1 of the PDF; the tools count it as page 0."; "Page 8 repeats page 3; it sits between page 7 and page 9." | all |
| 11 | set-3600-psi | `set_match`: the three sheets, precision 0.75 | "10.25A, 20.00A and 20.00B specify it. 10.17A gives no strength and 30.01's 3600 is a volume." | all |
| 12 | set-long-rare-tag | "names pages 4, 12 and 20 and little else" | A table "Page / FPG callouts" with rows 4, 12, 20; or "**4, 12, 20**" (no word "page"). | all |
| 13 | ufc260-appendices, ufc260-ch12-tables | `labelled_set`: each letter or number with its title | "References (Appendix A)" (title first); a table with the page column between number and title. | all |
| 14 | calc-wall-check | sliding fails against 1.5; the unsupported bearing value | "FS = 1.185 < 1.50"; "the factor of safety is 1.19, less than the 1.5 required". | all |
| 15 | scale-plan-distance | "B-1 to B-4 within 1 ft of 183.85 ft"; "notices that the stated scale disagrees with the bar" | "The distance is 184 ft. This is between borings B-1 and B-4."; "1 inch equals 40 feet on this print, not 20 as noted." | all |
| 16 | fixture-duplicate-page | the duplicate pair; cites viewer pages 8 and 3 | "The 3rd and 8th pages of the PDF are the same sheet twice." (3 checks); "Page 8 is a copy of page 3." | all |
| 17 | meck-revision-block, ufc301-changes, ufc04-supersedes, meck-curb-types, meck-monument, meck-underdrain, ufc04-table-5-1, meck-sediment-trap-criteria, meck-trap-dimensions and 12 more | `contains_all` / `contains_any` on a date, a dimension or a term | "Nov. 2016"; "6/3/2025"; "2'-6\" deep"; "Sch. 40"; "between 4 and 8 passes"; "2 to 1"; "eight percent"; "Minimum dimensions:" over bare values. | all |

## Table 1b. False passes (a wrong or empty answer passes)

| # | Task | The check | Made-up answer that passes | Trees |
|---|---|---|---|---|
| 1 | the tasks of row 1a-1 and coord-drawings-vs-manual (all 41 "missing" checks) | the mark beside a word that says it is missing | Direction reversed: "Door 109 is in the door schedule but is not tagged on A-101." A neighbour's negative: "Door 109 is in the schedule. Door 204 is not on the plan." | all |
| 2 | the value-pair checks of row 1a-3, and coord-revisions, coord-second-issue, coord-submittal, coord-spec-submittal | the two values that disagree | "S-001 uses 150 kPa; the memorandum allows 200 kPa, so the drawings are conservative."; "A-201 104.20, A-301 104.50. The floor levels agree." | all |
| 3 | coord-full-review, coord-long-review, coord-long-schedules, coord-long-callouts | all of the task's checks at once | A log with one row per sheet saying what is on it, each ending "No comment": 13 of 13 and 16 of 19. "Door 109 is not missing"; "I cannot confirm whether doors 109, 204, ...": 7 of 7. | all |
| 4 | the six `no-false-alarm` checks | no sound item stated as broken or missing | "3/A-501 is a dead reference. 5/A-501 points to the wrong detail."; "Door 203 was left off the schedule." | all |
| 5 | 27 of the 28 tasks with a `contains_all` / `contains_any` check (36 of 39 checks; the tasks before coord-sheet-index) | the words or numbers anywhere in the answer | "The maximum ramp slope is 2.1% and the cross slope 8.33%." (swapped); "No duplicates: page 8 is not the same as page 3."; "The sliding check (1.185) does not fail." | all |
| 6 | coord-revisions, coord-second-issue | "... and that it is not flagged" | "S-001: bearing pressure is now 200 kPa; clouded and listed. The sheet titles did not change." | all; coord-second-issue master only |
| 7 | coord-sheet-index, coord-full-review, coord-long-review | `index-title` | "S-201 SECOND FLOOR FRAMING PLAN and S-202 ROOF FRAMING PLAN both match the index." | all |
| 8 | every task | the two carried checks; `no-false-alarm` | An empty answer passes them (2 or 3 checks of every run). "The scan quality is too poor to check the callouts; no findings."; "(page index 3)". | all |
| 9 | set-3600-psi, set-cross-references, ufc260-appendices, ufc260-ch12-tables, ufc301-asce7-chapters, ufc04-density, scale-plan-distance | `set_match`, `labelled_set`, `cites`, `value_within` | "None: 10.25A, 20.00A and 20.00B do not specify it."; "Appendix A: there is no appendix on references"; "pages 12, 18, 29, 44 and 61"; "92 ft apart; 184 ft only if the bar were used, which I rejected". | all |

## Table 2. What differs between the two tags (and master)

| What | `aec-v1.2.0` | `aec-v1.3.0` | Fair to compare? |
|---|---|---|---|
| Tasks | 52 (138 own checks) | 54 (147): adds coord-spec-submittal (5 checks) and coord-reissue (4). Master: 55 (151), adds coord-second-issue (4). | Only on the 52 shared tasks: totals out of 52 against 54 are not the same sum. |
| The 52 shared tasks | question, documents, follow-ups, truth, every check and its parameters | identical (compared as JSON) | Yes. |
| The two carried checks | same | same | Yes. |
| Built documents | 13 fixture PDFs, planlens 0.16.0 | the same 13, page for page (words with positions, counts of drawings and images), planlens 0.17.0; 3 new ones for the new tasks | Yes. The scanned set was compared by image count only; the public PDFs are files, not built, and were not compared. |
| `checks.py` | no `scope="finding"` | adds it; used only by coord-spec-submittal, coord-reissue and coord-second-issue. A check that raises now has its error text masked. | Yes: no shared result moved. |
| `runner.py`, the table | | error texts masked in `run.json`; same columns, same sums | Yes. |

## 5. What the scorer reads (`run_folder_probe.py`; the same at all three trees)

- A made-up folder for coord-full-review scored as the runner scores it and then with `rescore=True` gave the same score in twelve of fourteen shapes, among them findings in the chat, in an `.xlsx` (first or second sheet), in a `.docx`, in a subfolder of `files/`.
- A delivered `.xlsx` or `.docx` that cannot be opened is skipped without a word: the checks read the chat alone (2 of 15), and nothing in `run.json` or the table says a file was unreadable. The same holds for a kind that is never read (`.pdf`, `.html`, `.rtf`, `.pptx`).
- Two delivered logs are both read and joined: a finding passes if either states it (15 of 15 with a second, wrong draft log beside the right one).
- A run that ended on a limit with no answer scores 2 of 15 (the two carried checks); with a log already delivered it scores 15 of 15, marked "(step cap)". Truth only in `.scratch/` scores 2 of 15, with 13 "found only in the run's working notes".
- Different both ways, 1: a `run.json` whose files are named with backslashes, re-scored on Linux, lost every file (15 of 15 became 2 of 15). FIXED, commit 1962e2b, test `packages/aec-agent-core/tests/test_audit_scorer.py`; it changes no score of a run re-scored on the system that saved it or saved with forward slashes.
- Different both ways, 2 (not fixed, not arithmetic): `run.json` copied without `files/` re-scores 15 of 15 to 2 of 15 in silence, while `file_produced` still passes on the name alone.
- Not fixed, the lead's call: `rescore=True` runs a saved run AGAIN when its `run.json` holds an `error`; with a stand-in model that empties the folder (answer and `activity.jsonl` gone) and writes a new error. `rescore=True` with `model=None` raises `ValueError`. Re-score a copy.
- A task with follow-up turns is scored on the last turn's answer alone.

## 6. The results tables (`tables_probe.py`)

- The sums are right on a made-up set of two arms: tasks passed, checks passed, calls, tokens, minutes, step caps, and the to-do-only share (300 of 4,000 tokens = 7.5 %).
- A run missing from one arm, or skipped, leaves that arm's denominators smaller (2/4 beside 1/2) with only a "·" in the per-task table; "Changes against" leaves the task out. Compare arms on the tasks both hold.
- A step-cap run counts in "tasks passed" as a fail and adds the 2 or 3 checks an empty answer passes to "checks passed".
- To-do-only steps: right on new records (`tool_names`) and on old ones by pairing; two `write_todos` in one step is not counted; a vision side call is not a step; a helper's to-do-only step counts with the main agent's. The share reads only `usage.input_tokens`: a record whose usage says `prompt_tokens` counts the steps and shows "·" for the share, while "tokens (in/out)" counts those tokens.
- A second run's records appended to the same `activity.jsonl` double every count (1 step read as 2, model calls 6 as 12). The runner empties the folder before a run, so this takes a copy made by hand.

## How to read a score with this in hand

1. A passed check on a coordination task means the mark and a negative word (or the two values) stood close together, not that the sentence was right: read the sentence before counting it.
2. A failed "missing" or value check is often a right finding worded as an action ("Add door 109 to the schedule") or with its unit written another way: read the answer and the log before counting a miss.
3. Outside coord-full-review and coord-long-review nothing delivered as a file is read, so a model that answers "see the attached log" fails those tasks whatever the log holds.
4. An empty answer already passes 2 checks of every task (3 where the task has `no-false-alarm`), so compare tasks passed, or checks passed less that floor, never raw check counts.
5. Compare the two tags on the 52 shared tasks only; on those, the tasks, checks and documents are equal and the scorer gave the same verdict on every made-up answer.
6. A model that words its findings differently between the two releases (actions against statements, a file against the chat) will move the score by more than most real differences in what it found.

## What was run, and what was not reached

- `scorer_probe.py` on 560 made-up answers (367 right, 193 wrong) at the three trees, each tree's own packages on the path (venvs with planlens 0.16.0 and 0.17.0); `documents_compare.py`, `run_folder_probe.py` and `tables_probe.py` at the tags and master. The cases were written by five helper sessions from each task's own truth text; I read every group of flagged results, one or two cases of each, not all 510.
- Tried: 53 of 55 tasks, 139 of the 151 own checks and the two carried checks.
- Not reached: produce-markup and produce-circle-tags (they need a marked-up PDF); `file_produced` (3), `pdf_markups`, `markups_point_at`, `markups_on_targets`; the unscored `pages_covered` (5) and `tools_called` (1); follow-up turns; a real model run. The public PDFs were not opened: right answers follow each task's truth text.
- "Wrong" here includes answers no model may ever write; rows 1b-1, 1b-2 and 1a-1 to 1a-3 are the ones an ordinary answer meets.
