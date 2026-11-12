# Foundry brief 7: the full read of the 84 coordination runs

This is the write-up of the full read of Foundry brief 7 (BACKLOG EV-42), done on 2026-11-12. Six readers each read one group of tasks, every run in full, and wrote a part file under `parts/`; the parts keep the per-run detail. This file merges them, settles the two places where they disagreed, and checks every proposed follow-up against the code. The first pass and the plan it led to are in `FIRST_PASS_AND_PLAN.md`, and the score worked out again on what each run delivered is in `RESCORED.md`, both beside this file. The follow-ups in section 10 go into `BACKLOG.md` by ID; this file keeps the evidence, not the list.

## 1. What was run and how it was read

- **The tasks.** Fourteen coordination tasks on one made-up building's documents: a drawing set of 15 sheets, the same set as a scan with no text layer, a 40-sheet set, a project manual, a geotechnical memorandum and a concrete mix submittal. Problems are planted on purpose (`coordination_fixture.py` in `aec-document-review`), and each task's checks look for them.
- **The runs.** GPT-5.6 Sol in five set-ups ("arms"): `baseline` (the page as shipped), `show_me` (the main model may look at a page itself), `method` (a written review method given to the agents), `sheet_tools` (two extra tools, `sheet_index` and `check_cross_references`) and `all_three`. GPT-5.4 in `baseline` only. That is 70 Sol runs and 14 GPT-5.4 runs. No run errored or hit the step cap. The code was app commit b95e5b5 and planlens commit d7f43d8; none of the files checked in section 10 has changed on master since.
- **The readers**, by task group: `wholeset15` (coord-full-review), `wholeset40` (coord-long-review), `revisions` (coord-revisions and coord-submittal), `callouts` (the three callout tasks), `schedules` (the three schedule tasks) and `singles` (sheet index, levels and dimensions, drawings against the manual, drawings against the memo).
- **What "read in full" meant.** For every run, the reader wrote a small script that printed each record of `activity.jsonl` in order, trimmed: who made it (the main agent, which helper, or a vision side call), the kind of record, the tool and its arguments, the start of each result, and the model's own text and reasoning summary. The reader paged through every record and printed whole any result that mattered (a page reading that named a planted problem, a tool result the model then acted on). `run.json`, the console log and any `FEEDBACK.md` were read, the delivered Word and Excel files were opened and read row by row, and every value a run acted on was checked against the fixture. Three readers went further. `wholeset40` rebuilt the two note files missing from the hand-back out of `activity.jsonl` and re-ran its task's 114 planted-problem checks with the package's own `run_check`. `schedules` compared, by script, every mark every page reading reported with the fixture, and read each disagreement by hand. `singles` timed every one of the suite's 15,545 tile calls.
- **The coordinator** read the six parts, the first pass and the re-score, and checked each proposed follow-up in the code, read-only. It went to the raw records only three times: to count the event kinds across the 84 records, to set one run's delivered files beside its `run.json`, and to read the engine field of one run's first record.

## 2. Results

The score on what each run delivered (`RESCORED.md`). Only the two whole-set tasks read files, so only they moved.

| set-up | checks, recorded | checks, delivered | tasks passed, recorded | tasks passed, delivered |
|---|---|---|---|---|
| Sol `baseline` | 92 of 95 | 87 of 95 | 12 of 14 | 12 of 14 |
| Sol `show_me` | 93 of 95 | 91 of 95 | 12 of 14 | 12 of 14 |
| Sol `method` | 91 of 95 | 89 of 95 | 12 of 14 | 12 of 14 |
| Sol `sheet_tools` | 92 of 95 | 92 of 95 | 12 of 14 | 12 of 14 |
| Sol `all_three` | 93 of 95 | 92 of 95 | 12 of 14 | 12 of 14 |
| GPT-5.4 `baseline` | 80 of 95 | 74 of 95 | 10 of 14 | 10 of 14 |

Planted problems delivered on the two whole-set reviews, by the corrected scorer and by the readers' hand count of the delivered comment logs. The scorer's figure is the checks passed less the two general checks (no giving up on legibility, no 0-based page citation), which every run passed.

| run | coord-full-review (13 planted): scorer / hand | coord-long-review (19 planted): scorer / hand | both (32): scorer / hand |
|---|---|---|---|
| Sol `baseline` | 11 / 11 | 13 / 13 | 24 / 24 |
| Sol `show_me` | 10 / 10 | 18 / 18 | 28 / 28 |
| Sol `method` | 11 / 11 | 15 / 15 | 26 / 26 |
| Sol `sheet_tools` | 13 / 13 | 17 / 17 | 30 / 30 |
| Sol `all_three` | 12 / 12 | 17 / 17 | 29 / 29 |
| GPT-5.4 | 7 / 6 | 6 / 6 | 13 / 12 |

- **The recorded score read the runs' working notes.** On the two whole-set tasks the scorer read each run's page notes (`.scratch/page_readings.json`, `.scratch/reads.json`) and kept helper work as if they had been delivered. That is fixed (EV-41: `checks.is_working_file` in `packages/aec-agent-core/aec_agent_core/review_eval/checks.py`), and 16 verdicts changed from pass to fail. Two readers found the defect on their own and confirmed all 16 flips against the delivered logs: none of the passages that had passed those checks states the planted problem. Some were neutral transcriptions; three rested on pixel coordinates, such as door 204 "found" in `[648, 162, 757, 204]` next to "project no." (`wholeset40.md` §1).
- **Where hand count and scorer still differ: one check.** GPT-5.4 coord-full-review passes `door-not-scheduled` on its delivered workbook from "T.O. PARAPET EL. 109.30" in row 23, near "not" in "sheet status is not coordinated". The workbook never mentions door 109. The check is wrong: its number pattern matches a whole number at the front of a decimal (F-2). The first pass's quick re-score (24, 28, 26, 30, 29 and 13 of 32) is reproduced exactly, except that GPT-5.4's 13 is 12 by hand for this reason.
- **The single-question tasks.** Every verdict on every Sol run of the twelve other tasks is right; each reader checked them one by one. GPT-5.4's are generous in at least five places: `index-title` passes on a reply that says the index "matches the set exactly" (F-4); `door-not-on-plan` passes on a scan answer with five false alarms (F-3); on coord-long-callouts two checks pass among 18 false alarms, one for the wrong reason (F-3); on coord-scan-callouts `callout-missing-sheet` passes on an answer that says it does not count 2/A-302; and the decimal above. Sol's corrected numbers stand; GPT-5.4's 74 of 95 overstates it.

## 3. What each arm did, from the record

**`show_me`: offered, never used.** The option was in the tools the model saw: `run.json`'s run-level `fingerprint.tools.sha`, a hash of every tool's name, description and argument schema, is 14030f98 for baseline and 88ad49e3 for show_me, with the same system prompt (all_three differs from sheet_tools the same way). No call in any of the 70 Sol runs passes `show_me=true`, and the model never mentions it. So show_me against baseline is two runs of the same harness, and the gap between them measures run-to-run chance: 87 against 91 checks of 95 delivered; on coord-long-review 13 against 18 planted problems; on coord-full-review 11 against 10; 23.9 against 27.0 million input tokens. Most of show_me's extra long-review problems came in one long helper reply (7,529 characters). Where it lost one (07 84 00 on the 15-sheet review), the cause was how the main agent split the sheets among helpers, not the switch.

**`method`: no visible effect on any answer.** On the twelve single-question tasks it found the same planted problems as baseline. On the whole-set reviews it delivered 11 of 13 (as baseline) and 15 of 19 (baseline 13). Its long review was the fastest and cheapest Sol run of that task (13.9 minutes, 3.27 million input tokens): three helpers, one of them given only the manual and the memo, so the helper that read G-002 never set note 6 beside the manual and the firestopping section was lost. The method's own clause on revisions points only at flagged changes ("what the cloud and the revision table say changed"), yet the method run found the unflagged bearing-pressure change. End-of-run "independent" helpers appeared in `method` and `all_three`, but `show_me`, with no method text, started them too, so one run each cannot tie them to the method.

**`sheet_tools`: two correct tools, used and never contradicted.**
- `sheet_index` returned every sheet's number, title, revision and view marks exactly as the title blocks print them, on the 15- and 40-sheet sets (4,073 characters for 40 sheets, in under half a second). On the scan it returned nothing, honestly: "no text layer on this sheet: read its title block by looking". It does not read the cover's index table, so both index problems still needed the model's own reading of G-001, which every run made.
- `check_cross_references` returned, on the 15-sheet set, 17 callouts: 15 resolved, 7/A-501 unresolved ("view marks recognised on A-501: 1 to 6; none for 7") and 2/A-302 not in the document. On the 40-sheet set it returned 39 callouts, 36 resolved, with 9/A-502 also unresolved. There was no false flag anywhere. On the scan it found no callouts and said their text was not readable.
- Use: no run ignored or contradicted either tool, and every flagged callout was looked at before it was reported. Zooms went to the tool's boxes: 6 and 12 zooms in the two runs with the callout check on coord-long-callouts, against 46 in baseline.
- What it changed: on the whole-set reviews 7/A-501 was delivered in 4 of 4 runs that had the callout check and in 2 of 6 Sol runs that did not (0 of 3 on the 15-sheet review, where all three runs without it read both halves in separate side calls). In the 40-sheet sheet_tools run the main agent recovered 7/A-501 from the tool after its helper had dropped it. On the three callout tasks Sol was right with or without the tool. It saved no looking anywhere, because the page's rule is to look at every page in scope, and it was not faster: sheet_tools spent the most minutes of any Sol arm (106.0), much of it waiting on stalled vision calls (section 8).
- The revision miss: the sheet_tools coord-revisions run used the index's revision column to choose which sheets to read as text (PDF pages 1, 8 and 10 only), then asked its issue-1 page readings what had changed. The unflagged change on S-001 (150 to 200 kPa, no cloud, no revision entry) was never transcribed. One baseline helper made the same miss without the tool, by the same second step (`revisions.md`, "Each run's path"). The cause is a method that finds only flagged changes, which would hold on any model, plus a reading that answers a comparison it cannot make (F-8); the tool's description does not suggest choosing sheets by revision.

**`all_three`: as sheet_tools wherever the sheet tools applied; show_me unused.** 92 of 95 delivered. On coord-full-review it delivered 12 of 13: the main model stated the bay conflict (record #897) and dropped it two steps later. On coord-callouts it looked at 12 of 15 pages and wrote "I visually examined all 15 PDF pages"; the three skipped sheets carry no callouts, so the answer was right.

**What one run each can and cannot show.** The record can show a mechanism: a tool result that went straight onto the comment list, a page prompt that happened to point a reading at the overall plans, a split that put two halves of a problem with different helpers. It cannot show that an arm raises a score: two runs of the same harness (baseline and show_me) differ by five planted problems on coord-long-review, as large as any difference between the arms. The one arm effect with a visible cause is the callout check, and it rests on 4 runs against 6.

## 4. Where findings were lost in the whole-set reviews

Sol missed 23 planted problems over the ten whole-set runs (160 possible, 137 delivered). The readers traced every one.

| where it was lost | coord-full-review | coord-long-review | total |
|---|---|---|---|
| never read by any call | 0 | 0 | 0 |
| only in a text result that no agent ever mentioned | 0 | 3 | 3 |
| read in page readings, never set beside the other half | 5 | 5 | 10 |
| in a helper's page readings, left out of its reply | 0 | 6 | 6 |
| named in the main model's reasoning, then left out of the log | 3 | 1 | 4 |
| **total** | **8** | **15** | **23** |

- **Only in a text result:** door 204 in the long baseline (A-601's row in a helper's text read, A-102's tag list in the main agent's first text read), and 07 84 00 in the long method and sheet_tools runs (G-002 note 6 in a helper's text read; the manual read by another helper or not compared).
- **Never paired:** on the 15-sheet review, 7/A-501 in baseline (the A-101 reading said "Confirm both details exist", A-501's reading showed six details, in separate side calls) and in show_me, 07 84 00 and the bay in show_me (all three seen by helpers whose ranges or briefs held only one half), and the bay in method. On the 40-sheet review, the bay in all five runs: 7500 was read on A-401 by a tile call in every run and 7200 off the plans many times, yet no agent's own text ever names the two together. In three of those runs one helper had both A-101 to A-105 and A-401 in its own scope.
- **Dropped by a helper:** in the long baseline, 7/A-501, 9/A-502, door 109 and door 509 (helper H1 read each in its tiles; its reasoning assumed "Door 109 appears on sheet A601", which is false); in the long method, doors 204 and 509 (the helper checked plan to schedule and never schedule to plan).
- **Dropped at the write:** 15-sheet baseline, the bay ("a conflict between core dimensions in different drawings that needs closer inspection", #413, never inspected); 15-sheet method, 7/A-501 (listed among comments to log at #878, absent from the write at #882); 15-sheet all_three, the bay (#897, dropped at #901 and #905); 40-sheet all_three, door 509 (a line in a helper's sheet inventory; the main reasoning said "doors like 509 might be missing plans, so I should check those", then checked 204 only).
- **The bay B to C (7200 against 7500)** was missed in 9 of 10 Sol whole-set reviews and found in all six runs of coord-levels-dimensions, whose question names enlarged plans and grid dimensions. No run ever looked at A-401 and an overall plan together; where the pairing happened at all it was in the main model's text, from two separate readings. The one whole-set run that delivered it (15-sheet sheet_tools) had an A-401 page prompt that asked for "coordination back to overall plans".
- **Two patterns.** A cross-sheet problem whose halves come from two separate side calls is the one that gets lost. In the long reviews the helpers checked from plan to schedule more than from schedule to plan: the two "scheduled but not on a plan" doors were delivered in 3 and 2 of 5 runs, the five "on a plan but not scheduled" marks in 4 or 5 of 5.
- **GPT-5.4** missed 7 of 13 and 13 of 19. Of those, 6 and 9 were in its own results and never compared (it compared nothing across sheets except the S-201 title, whose two halves both arrived as text); 1 and 4 were never read, because it never looked at the sheets that hold them (section 9).

## 5. False alarms

Every row of the twelve whole-set comment logs, classed against the fixture (`wholeset15.md` "Hand count of every log row"; `wholeset40.md` §3). "True, not planted" means accurate about the made-up documents but not a planted problem; "request" asks for confirmation and asserts no defect; "duplicate" repeats another row (wholeset40 counts the same point repeated on a sibling sheet).

| log | rows | planted rows (problems) | wrong statement | true, not planted | request | duplicate |
|---|---|---|---|---|---|---|
| 15-sheet, Sol baseline | 30 | 11 (11) | 2 | 14 | 3 | 0 |
| 15-sheet, Sol show_me | 26 | 10 (10) | 0 | 13 | 3 | 0 |
| 15-sheet, Sol method | 17 | 11 (11) | 0 | 6 | 0 | 0 |
| 15-sheet, Sol sheet_tools | 18 | 12 (13) | 0 | 6 | 0 | 0 |
| 15-sheet, Sol all_three | 14 | 12 (12) | 0 | 2 | 0 | 0 |
| 15-sheet, GPT-5.4 | 28 | 6 (6) | 1 | 7 | 1 | 13 |
| 40-sheet, Sol baseline | 32 | 12 (13) | 0 | 19 | 1 | 0 |
| 40-sheet, Sol show_me | 40 | 18 (18) | 1 | 16 | 1 | 4 |
| 40-sheet, Sol method | 17 | 14 (15) | 0 | 3 | 0 | 0 |
| 40-sheet, Sol sheet_tools | 29 | 16 (17) | 0 | 9 | 0 | 4 |
| 40-sheet, Sol all_three | 24 | 14 (17) | 0 | 9 | 1 | 0 |
| 40-sheet, GPT-5.4 | 7 | 6 (6) | 0 | 1 | 0 | 0 |
| **total** | **282** | **142** | **4** | **105** | **10** | **21** |

Most "true, not planted" rows are about the fixture's stand-in drawing: untagged windows, no spot grades, ceiling plans with no devices, detail notes that do not match their titles. Each was checked against the fixture.

**Every wrong statement in a whole-set log:**
- Sol baseline, 15-sheet, row 23: rigid insulation "on A-501 Details 2, 3, 5 and 6"; the fixture prints it on details 3 and 6 only.
- Sol baseline, 15-sheet, row 24: sealant, backer rod and flashing "on Details 2, 3, 5 and 6"; they are on 2 and 5 only.
- GPT-5.4, 15-sheet, row 14: door 107's hardware revision "is shown only by a cloud"; A-601's revision block carries "REVISED DOOR HARDWARE".
- Sol show_me, 40-sheet, CR-08: A-401 "includes a 1/A-601 reference"; it calls 1/A-501. One zoom misread the sheet number (#1690), a helper reported it as "Seen by zoomed look", and the main agent copied it; the sheet's text layer holds A-501.

Close to the line, counted true: the 40-sheet baseline's row 26 adds FCU-3-4 into the fan-coil totals with 360 L/s and 7.5 kW taken from the other floors' rows and does not say the values are assumed; three 15-sheet rows are questionable (show_me rows 10 and 23, baseline row 20; `wholeset15.md`). Wrong statements in helper text that never reached a log: "Door 109 appears on sheet A601" (40-sheet baseline H1), GROUT FRAME SOLID placed in S-501 detail 2 (40-sheet method H3), and "LP-4B omitted by the plan's text extraction" (coord-long-schedules sheet_tools).

**False statements on the single-question tasks:**
- GPT-5.4 coord-long-callouts: 20 callouts called broken because "A-301 / A-501 / S-501 not in set"; 18 are good callouts, and 7/A-501 is broken for another reason than the one given (`callouts.md`, the failed run).
- GPT-5.4 coord-callouts: names C-101, A-201, A-601 and S-202 as sheets with callouts (they have none), calls 4/A-501 and 6/A-501 broken and then valid, and opens with only 2/A-302 broken before a closing "Correction".
- GPT-5.4 coord-scan-callouts: opens "I did not find any ... callouts that point to a non-existent view", lists 7/A-501 as resolving and then flags it.
- GPT-5.4 coord-scan-schedules: doors 201, 202, 203, 205 and 206 "scheduled but not on the checked floor plan", and "I did not find a second floor architectural plan sheet".
- GPT-5.4 coord-long-schedules: door 409 "listed in the door schedule on A-602"; no door 409 exists.
- GPT-5.4 coord-revisions: "Issue 1 changes are limited to three sheets", then a fourth; "No cloud seen" on G-001, which it never looked at (true, but unseen).
- GPT-5.4 coord-sheet-index: "the sheet index matches the set exactly".
- Sol sheet_tools coord-revisions, Word file: S-001 listed under "no substantive content change", and "Text extraction was used to confirm" when text was read for three pages only.
- Sol all_three coord-callouts: "I visually examined all 15 PDF pages" (it looked at 12).
- Sol baseline coord-scan-callouts, Word file: 2/A-302 placed at the "upper-right / right edge" of A-102; it is near the top middle.
- Sol Word files with wrong counts: coord-long-schedules baseline ("four" plan marks for five) and show_me ("eight ... six" for seven and five); coord-drawings-vs-manual show_me and all_three ("five" sections issued separately, for four). The chat answers count correctly.

No Sol chat answer on a single-question task states a fact about the documents that the fixture contradicts, apart from all_three's coverage sentence.

## 6. The scorer

Every defect the readers reported, with the run that shows it.

1. **Working notes read as delivered** (fixed, EV-41). 16 false passes on the two whole-set tasks (`RESCORED.md`).
2. **A heading carries over the appended file text** (F-1). Sol show_me coord-full-review passed `callout-missing-detail` and `spec-section-missing` from neutral transcriptions that followed the reply's last bold bullet. EV-41 removed the notes, not the join, so a delivered file still sits under the reply's last heading.
3. **A whole number matches inside a decimal** (F-2). GPT-5.4 coord-full-review, `door-not-scheduled`, on "EL. 109.30"; also "204.9", a box coordinate in the notes.
4. **Pixel coordinates and table headers matched as findings.** Door 204 in `[648, 162, 757, 204]`; a revision table's "NO." header and a "Not visible" column read as absence words. Only in the notes, so fixed by EV-41; noted.
5. **No check against false alarms** on the callout and schedule tasks (F-3). GPT-5.4 coord-long-callouts (18 good callouts called broken), coord-scan-schedules (five) and coord-long-schedules (one).
6. **`stated_together` passes on a plain ordered list** (F-4). GPT-5.4 coord-sheet-index, `index-title`, on two adjacent lines of a sheet list 55 characters apart.
7. **Two truth texts disagree with the fixture** (F-5): coord-long-callouts' sheet count, and coord-drawings-vs-manual's "the other cited sections are in the manual".
8. **Direction-blind checks.** "204 is on the plan but not scheduled" would pass; no run made that error. Noted.
9. **The Word file is not scored on the single-question tasks**, so four Sol files with wrong counts went unseen. Noted.
10. **`door-not-scheduled-4` passes on a numbering gap.** GPT-5.4 coord-long-review wrote "verify whether door 407 is missing from the schedule" without looking at A-104. Noted.
11. **The callout check's 200-character window is not too strict.** The near-misses of 249 and 209 characters were measured over page notes; neither delivered log mentions 7/A-501 (`wholeset15.md`, "Detail 7/A-501"). Noted.
12. **An untested truth.** coord-submittal's truth says the transmittal marks no deviations; three runs reported "DEVIATIONS: NONE" against Section 01 33 00 and no check scores it. The question points at Section 03 30 00 alone. Noted.

## 7. Tools and the record

**Tool results that misled a run:**
- `pages_to_view` (planlens `open_document`) lists only pages classed as drawing sheets, scans, figures or forms: 6 of 15 and 18 of 40, leaving out every callout destination. GPT-5.4 coord-long-callouts took it as the set and its 18 false alarms follow from that (F-6). Sol all_three and GPT-5.4 on coord-callouts skipped exactly the three "text" pages, harmlessly.
- Contact-sheet labels are bare 0-based page numbers. GPT-5.4's reading called them PDF pages, and on coord-scan-schedules the model looked at the wrong sheets and never at A-102 (F-7).
- A single page reading asked what changed against the other issue answered "No substantive change versus Issue 0 is identifiable in this tile"; two agents took it as "unchanged" (F-8).
- One zoom misread 1/A-501 as 1/A-601 and the run delivered it (show_me coord-long-review). Whole-sheet overviews misread small numbers several times, and tiles read them right. A reading error, not a tool defect; noted.
- The page tool's merged list of found things keeps one reading where the overview and a tile disagree (GPT-5.4 coord-long-schedules #421: overview "409", tile "408"). It did not cause the 409, which started in the model's own prompt ("range 401-409"). Noted.
- `search_document`: a pattern with "|" and no `regex=true` is matched literally, and the zero-hit hint blames drawing sheets (four times); its `pages_not_searchable_as_text` names pages the same search just found hits on (F-9). No answer was changed by either.
- `find_quantities` read "10-20M EACH WAY" as 10 to 20 m (F-10). Not copied into any answer.
- `read_document` returns three to five drawing sheets per call with a `next` pointer, and most agents did not follow it; this works as designed. GPT-5.4's three refused `analyze_pdf_page` calls with a `pdf_pages` range got a clear refusal; the model changed its plan rather than calling page by page. Noted.

**Gaps in the record:**
- **Stalled vision calls.** The 84 records hold 20,153 `model_start` events and 20,109 `model_end`: 44 calls started and never ended, 34 of them tiles (`singles.md`). A side call with no answer in 180 s is abandoned and asked once more with only a log warning (`vision_engine.invoke_side_call`); there is no event, the abandoned call keeps its thread and one of the eight vision slots until the engine itself gives up (three tiles in sheet_tools coord-long-review returned after 905 s), and the results table says "failed calls 0" (F-12).
- **Unrecorded waits.** A helper model call in Sol baseline coord-revisions took 904 s for 25,000 tokens in and 324 out, half of the run. The agent-level busy wait writes a `model_busy` event for each wait and none occurs in any record, so it was not that. On the Foundry engine a timed-out or dropped call is retried inside the call with only an info log line (`palantir_sdk_engine.call_with_retries`, read timeout 900 s), and the per-call HTTP record kept since EV-14 is installed only on the Tiny Apps route; that fits 904 s, but nothing in the record says. The run's model was also wrapped by a `Throttled` class that is not in the packages (it came with the run's set-up), so its waits are outside the record too. (A note in `revisions.md` that pointed at the busy-turn limit now says this.) Also unexplained: 123 s between process start and turn start and 56 s before the first model call (coord-sheet-index all_three), and 37 s before the first call and 75 s after the turn (coord-long-callouts sheet_tools) (F-14). Helper queueing is recorded: 25 `helper_queued` events.
- **The per-turn tool hash covers names only.** `tools_offered_sha` on `turn_end` is the same in baseline, show_me and method, so the activity record alone cannot show the show_me argument was offered; `run.json`'s fingerprint can. Nothing counts looks made with `show_me=true` (F-15). The `callouts.md` and `singles.md` sentences that said the record could not confirm the option was offered are corrected in place.
- **Files the scorer read that the hand-back lacked.** `run.json` lists `files/.scratch/page_readings.json` and `files/.scratch/reads.json`; the hand-back's `files/` does not hold them (checked on Sol baseline coord-long-review, reported for every whole-set run). The EV-41 scorer still reads them, to mark a failed check "found only in the run's working notes" (F-16).
- **Refused tool calls are not counted.** GPT-5.4 coord-long-review shows 0 errors and 0 failed calls although three of its tool calls were refused and it filed tool feedback (F-12).
- GPT-5.4 records no reasoning text, so its omissions cannot be explained from the record. That is the model, not the logging.

## 8. Cost and time

Sol used 23.3 to 27.0 million input tokens and 92 to 106 minutes per arm (122.7 million in all), against the brief's estimate of 8 to 15 million; GPT-5.4 used 11.6 million and 60 minutes. Each Sol arm made 3,300 to 4,200 model calls for 765 to 997 tool calls. Where it went:

- **Vision side calls and tiles.** `analyze_pdf_page` makes one overview call plus one call per tile (3 x 3 by default on a text-layer set, 4 x 4 on a scan); a zoom is one call, a contact sheet one, and five probe calls measure the vision model once per run. These are the "extra" model calls. Viewing one sheet costs about 35,000 to 37,000 side-call input tokens on Sol, so 15 sheets cost about 560,000. On coord-long-review tile calls are 80 % of the calls and about half the tokens. A 4 x 4 choice where 3 x 3 would do raised show_me coord-callouts to 1.45 million tokens (baseline 1.09) and all_three coord-scan-callouts to 1.65 million; 2 x 2 tiles on plain text pages added about 100,000 tokens to show_me and method coord-submittal.
- **The main agent re-sends the whole conversation each call**, about 105,000 to 146,000 input tokens a call by the end of a 15-sheet review, so its 11 to 17 calls are 19 % to 66 % of a run's tokens.
- **To-do-only steps.** On coord-submittal, steps that only updated the to-do list used 25 % to 42 % of each run's input tokens (113,000 to 167,000) on both models; on coord-revisions 4 % to 18 % on Sol and 37 % on GPT-5.4, whose five such steps took 52 of its 201 s (F-17).
- **End-of-run "independent" helpers.** On coord-full-review, method, sheet_tools and all_three each ended with a helper that re-read all 15 sheets (about 150 vision calls, 0.18 to 0.31 million helper tokens, 3 to 4 minutes) and added no planted problem. On coord-drawings-vs-manual, show_me, method and all_three did the same (110 to 170 s and 0.4 to 0.7 million more side-call tokens each), on coord-submittal show_me and all_three (about 87,000 tokens, 51 to 73 s), and on coord-drawings-vs-geotech sheet_tools (117 s). No reply changed.
- **Duplicated helpers.** Baseline coord-revisions gave two helpers the same whole-set job: 3.79 million tokens and 1,697 s, 1.6 to 1.8 times the other Sol arms.
- **Helpers queue two at a time** (a deliberate setting): waits of 192 and 198 s (coord-long-callouts baseline), 188 and 329 s (its show_me), 274 s (coord-scan-callouts baseline), 107 to 284 s (coord-long-schedules), 222 s (coord-full-review show_me).
- **Stalled calls.** Across the suite a tile takes 5 s at the median and under 22 s at the 99th percentile, but each of the 34 stalled tiles held its batch for the full 180 s: about 350 s of the 841 s of sheet_tools coord-long-callouts, 180 of the 269 s of baseline coord-sheet-index, 175 s of sheet_tools coord-drawings-vs-geotech, and the extra time of the two slowest 15-sheet callout runs. The arms' minutes cannot be compared without taking these out (F-13).

## 9. GPT-5.4

GPT-5.4 passed 10 of 14 tasks (74 of 95 checks delivered, generous in places; section 2). It failed both whole-set reviews, coord-long-callouts and coord-long-schedules. What it did differently:

- **It looked at far fewer sheets** and leaned on text: 10 of 15 sheets on coord-full-review, 5 of 40 on coord-long-review, 13 of 40 on coord-long-callouts, 12 of 15 on coord-callouts, 6 of 15 on coord-drawings-vs-manual, and 3 sheets in each issue on coord-revisions, after reading all 30 pages as text. It never looked at A-102 on the scan or at S-101 and S-501 on coord-long-schedules, which is exactly its F4 miss there, and the answer did not say the footing schedule went unchecked.
- **It compared almost nothing across sheets.** On coord-full-review it had both halves of six planted problems in its own results and delivered none of them; 13 of its 28 rows repeat another row.
- **What misled it:** `pages_to_view` (coord-long-callouts, 18 false alarms, F-6); the 0-based contact-sheet labels (coord-scan-schedules, five false alarms, F-7); its own prompt to the page tool ("the scheduled fourth-floor door range 401-409"), which a whole-sheet view then obliged by reading door 408 as "409"; and a refused page-range argument on coord-long-review, after which it handed the review to one helper that made four page looks and wrote a 7-row log in 5.2 minutes.
- **Its answers contradict themselves** on coord-callouts, coord-scan-callouts, coord-sheet-index and coord-revisions.
- **Where text was enough it did well and cheaply:** on coord-revisions it read every page of both issues as text first and found all four changes in 79 model calls and 0.73 million tokens, under a third of any Sol run; on coord-submittal it read Section 01 33 00 and reported "DEVIATIONS: NONE".
- No proposal is made for any of its misses (the owner's steer); the tool defects it ran into are F-6 and F-7, and they would mislead any model that trusts them.

## 10. Follow-ups

### Proposed

Filed in `BACKLOG.md` on 2026-11-12: F-1 to F-5 as EV-44; F-6, F-7, F-9 and F-10 as PL-39; F-8 and F-13 as VIS-31; F-11 and F-17 as DR-43; F-12, F-14 and F-15 as INF-75; F-16 as a line on EV-43 (the next brief's packing step).

Each is a defect in the scorer, a tool or the record, or a behaviour that would hold on any model. Paths are under `packages/aec-agent-core/aec_agent_core/`, `packages/aec-document-review/aec_document_review/` and planlens' `planlens/`.

**F-1. A heading in the reply carries over the delivered file's text.** Kind: scorer. Belongs to: the review suite and its checks.
- Backlog text: `check_stated_together` lets a term in a heading count for every line under it. `run_check` appends the delivered files' text to the reply, and `_sections` neither starts a new section at the join nor ends one at a blank line, so the reply's last all-bold line (or short line ending in a colon) heads the whole file. In brief 7 this passed two checks on Sol show_me coord-full-review from neutral transcriptions under the bullet "F4 is used on S-101 but absent from the footing schedule." Start a fresh section at each file boundary.
- Evidence: `parts/wholeset15.md`, "Sol show_me" and "Follow-ups" F2 (the reader reproduced it).
- Confirmed in code: yes; `review_eval/checks.py`, `run_check` and `_sections` (after EV-41).

**F-2. A whole number matches inside a decimal.** Kind: scorer. Belongs to: the review suite and its checks.
- Backlog text: `_num("109")` builds `(?<![\w.,/-])109(?!\d)`, which refuses a following digit but not a following ".digit", so door 109 matches in "EL. 109.30" and door 204 in a coordinate "204.9". GPT-5.4 coord-full-review still passes `door-not-scheduled` on its delivered workbook this way, the one verdict where the corrected scorer and the hand count differ. Refuse a following ".digit" for a whole-number term.
- Evidence: `parts/wholeset15.md`, "GPT-5.4 baseline" and "Follow-ups" F3.
- Confirmed in code: yes; `review_eval/tasks.py`, `_num`.

**F-3. No check penalises a false alarm on the callout and schedule tasks.** Kind: scorer. Belongs to: the review suite and its checks.
- Backlog text: the callout and schedule tasks score only the planted problems, so an answer that calls good callouts broken or reports unplanted marks still passes. GPT-5.4 coord-long-callouts called 18 good callouts broken and passed two of three checks; coord-scan-schedules reported five doors that are on the plan and passed all three; coord-long-schedules invented door 409. The fixture knows every good callout (15 and 36) and every real mark; add a scored check that fails a good callout or an unplanted mark stated with an absence word in the same line or table row (not a 200-character window: Sol's answers put "All other callouts resolve, including ..." close to the broken ones).
- Evidence: `parts/callouts.md`, proposed follow-up 1; `parts/schedules.md` §8 P1.
- Confirmed in code: yes; `review_eval/tasks.py`, `COORDINATION_TASKS` (only `_planted` checks, an information-only `pages_covered`, and the two automatic checks).

**F-4. `stated_together` passes on a plain ordered list.** Kind: scorer. Belongs to: the review suite and its checks.
- Backlog text: `index-title` asks for S-201, "roof" and "second floor framing" within 300 characters, and in this fixture S-201's own title and S-202's ROOF FRAMING PLAN sit on adjacent lines of any ordered sheet list. GPT-5.4 coord-sheet-index said the index "matches the set exactly" and passed on two lines 55 characters apart. Require the terms in one line or list item for this check, then review the level and bay checks, which could pass on an inventory of values the same way.
- Evidence: `parts/singles.md`, "The six questions" 1 and proposed follow-up 1.
- Confirmed in code: yes; `review_eval/tasks.py`, `_PLANTED_CHECKS["index-title"]` (no window, so the default 300), and `review_eval/checks.py`, `check_stated_together`.

**F-5. Two task truth texts disagree with the fixture.** Kind: scorer. Belongs to: the review suite and its checks.
- Backlog text: coord-long-callouts' truth says "39 callouts on 22 of them"; the fixture puts the 39 on 17 sheets. coord-drawings-vs-manual's truth says "the other cited sections are in the manual", but the manual binds only 01 33 00, 03 30 00 and 08 71 00 and lists 04 20 00, 07 92 00, 08 11 13 and 09 91 00 as issued separately; every run correctly listed those four as not resolving, and the truth text would make a reviewer count them as false alarms. Correct both sentences.
- Evidence: `parts/callouts.md`, proposed follow-up 5; `parts/singles.md`, Task 3 and proposed follow-up 6.
- Confirmed in code: yes; `review_eval/tasks.py`, the two `Task(... truth=...)` entries, against `review_eval/coordination_fixture.py` (`_project(floors=5)` gives 39 callouts on 17 sheets; the manual's contents note).

**F-6. `pages_to_view` leaves out drawing sheets classed as text or mixed.** Kind: tool. Belongs to: planlens' document tools.
- Backlog text: `open_document` lists in `pages_to_view` only pages whose kind is scanned, figure, drawing sheet or form, with the note "these pages are scans, figures or drawing sheets". On the coordination sets that is 6 of 15 and 18 of 40 sheets, leaving out every callout destination (A-301, A-501, A-502, S-501), though every page is a titled drawing sheet. GPT-5.4 coord-long-callouts took the list as the set and wrote that A-301, A-302, A-501 and S-501 are absent "anywhere in this 40-page set", the source of its 18 false alarms. List every drawing-size page with a title block (`sheet_index` already finds all 40), or say in the note that text and mixed pages can be drawing sheets.
- Evidence: `parts/callouts.md`, proposed follow-up 2 and "GPT-5.4, baseline (254 s): the failed run".
- Confirmed in code: yes; `planlens/tools/toolkit.py`, `open_document`'s `pages_to_view` built from `VIEW_KINDS`.

**F-7. Contact-sheet labels are bare 0-based page numbers.** Kind: tool. Belongs to: planlens' document tools.
- Backlog text: `render_thumbnails` labels each thumbnail "<index> <kind>" with the 0-based index, and its legend ("each thumbnail is labelled '<page> <kind>'") does not say which count it is, while the page tools take both `page` (from 0) and `pdf_page` (from 1). GPT-5.4's reading of the coord-scan-schedules contact sheet called "3 drawing_sheet" "PDF page 3"; the model then looked at C-101 and A-101 for the two floor plans, never at A-102, and reported five false alarms and "no second-floor plan". Print the PDF page (or both), and say so in the legend.
- Evidence: `parts/schedules.md` §5.6 and §8 P2.
- Confirmed in code: yes; `planlens/document/document.py`, `Document.render_thumbnails`.

**F-8. A page reading asked what changed answers as if it had compared.** Kind: tool. Belongs to: vision calls.
- Backlog text: a vision reading sees one image of one issue, yet asked about changes "versus issue 0" it answered "No substantive change versus Issue 0 is identifiable in this tile" (baseline helper) and "Substantive changes: None identifiable in this tile" (sheet_tools), and both agents took S-001 as unchanged; sheet_tools missed the unflagged 150 to 200 kPa change. The tile and page prompts add only "Report only what is IN this tile" and the user's question. Tell every reading that it sees one image: asked to compare with another issue or sheet, it transcribes what is there and says the comparison needs the other image.
- Evidence: `parts/revisions.md`, coord-revisions "Each run's path" and proposed follow-up 1.
- Confirmed in code: yes; `vision_tools.py`, the tile jobs' prompt, and `turn_question.with_question`; nothing there says a reading cannot compare with an image it was not given.

**F-9. `search_document`'s result misleads in two ways.** Kind: tool. Belongs to: planlens' document tools.
- Backlog text: (a) `pages_not_searchable_as_text` lists every page classed as a drawing sheet, scan, figure or form, including pages with a full text layer and pages on which the same search just found hits (GPT-5.4 coord-long-review #98; Sol method coord-long-schedules). (b) A pattern holding "|" without `regex=true` is matched literally, and the zero-hit hint says the term may be on a drawing sheet and offers `fuzzy=true`, never `regex` (four times, in GPT-5.4 and Sol method coord-long-review). List only pages without a usable text layer and drop those with hits; when a literal pattern holds "|", say that `regex=true` reads it as "or".
- Evidence: `parts/wholeset40.md` §10 item 3; `parts/schedules.md` §9; `parts/singles.md`, "Noted, not proposed".
- Confirmed in code: yes; `planlens/tools/toolkit.py`, `_tool_search_document`.

**F-10. `find_quantities` reads a rebar callout as a length range.** Kind: tool. Belongs to: planlens' document tools.
- Backlog text: "10-20M EACH WAY" and "8-15M EACH WAY" on S-501 came back as lengths of 10 to 20 m and 8 to 15 m (Sol baseline coord-levels-dimensions). It reached no answer here, but a wrong value in a tool result is the kind that gets copied into a deliverable. Read a count and a bar size ("10-20M") as one rebar callout, not a range with a unit.
- Evidence: `parts/singles.md`, Task 2 "Sol, baseline" and proposed follow-up 3.
- Confirmed in code: yes; `planlens/document/quantities.py`, `_RE_VALUE` with `_RANGE_TAIL`, case-blind, so "M" is read as metres.

**F-11. A `sandbox:` link reaches the user as a dead link.** Kind: tool. Belongs to: the Document Review page and its tools.
- Backlog text: two coord-submittal answers (Sol baseline and method) link the Word file as `sandbox:/mnt/data/...`, which goes nowhere on the page; a text search finds such links in 15 Sol runs of the suite. `method_call`'s note to its writers says "a /tmp path or a sandbox: link would not reach them"; the note after `write_docx` and the other saves does not, and nothing rewrites such a link. Give every writer's note that line, or point such a link at the attached file of the same name.
- Evidence: `parts/revisions.md`, proposed follow-up 3.
- Confirmed in code: yes; `aec_document_review/core.py`, `ATTACHMENT_NOTE`, against `method_call.py`'s note.

**F-12. A timed-out vision call leaves no event, and the results table counts neither it nor a refused tool call.** Kind: record. Belongs to: the run record.
- Backlog text: a vision side call with no answer in 180 s is abandoned and asked once more with only a log warning; the abandoned call leaves a `model_start` with no `model_end` (44 in the 84 records) and keeps one of the eight vision slots until the engine gives up. RESULTS' "failed calls" counts only `model_error` events and "errors" only failed runs, so both read 0 here, though GPT-5.4 also had three tool calls refused. Write an event when a side call times out (the limit, the call's kind and page, when it was asked again) and count timeouts and refused tool calls in the table.
- Evidence: `parts/callouts.md`, proposed follow-up 3; `parts/revisions.md`, proposed follow-up 2; `parts/wholeset40.md`, "Noted, not proposed".
- Confirmed in code: yes; `deep/vision_engine.py`, `invoke_side_call` and `_invoke_once`; `review_eval/runner.py`, the results table.

**F-13. The 180-second side-call limit is far beyond the measured tail.** Kind: tool. Belongs to: vision calls.
- Backlog text: over the suite's 15,545 tile calls the median took 5 s and 99 % took under 22 s, but each of the 34 stalled tiles held its batch for the full 180 s before the retry, which then answered in 1 to 2 s; one stall cost about 350 s of an 841 s run. Brief 5 chose 180 s when the 99th percentile was 74 s. Set the limit from the measured tail (for example 60 s) with the existing single retry, once F-12 records timeouts so the change can be measured.
- Evidence: `parts/singles.md`, proposed follow-up 2; `parts/callouts.md`, proposed follow-up 3.
- Confirmed in code: yes; `deep/vision_engine.py`, `DEFAULT_TIMEOUT_S = 180.0` and `TIMEOUT_RETRIES = 1`.

**F-14. A model call's own retries and the time before the first model call are not recorded.** Kind: record. Belongs to: the run record.
- Backlog text: a helper model call in Sol baseline coord-revisions took 904 s for 25,000 tokens in and 324 out, half the run, and the record cannot say why; 123 s before the turn and 56 s before the first model call (coord-sheet-index all_three) and 37 s and 75 s (coord-long-callouts sheet_tools) are blank too, and the arms are compared on those seconds. The Foundry engine retries a timed-out or dropped call with only an info log line, and the per-call HTTP record is installed only on the Tiny Apps route. Write an event for each engine-level retry and for what the time before the first model call was spent on.
- Evidence: `parts/revisions.md`, "Cost and time"; `parts/singles.md`, proposed follow-up 5; `parts/callouts.md`, "Noted, not proposed".
- Confirmed in code: yes for the retries (`palantir_sdk_engine.call_with_retries`; `http_trace.note_response` installed only in `tinyapps_engine`); no for the gaps before the turn, whose cause is not in the packages.

**F-15. The record does not show which arguments were offered or used.** Kind: record. Belongs to: the run record and the review suite.
- Backlog text: `tools_offered_sha` on `turn_end` hashes the tool names only, so it is identical in baseline and show_me; only `run.json`'s fingerprint (names, descriptions and schemas) shows the arm differed, and neither says what changed. RESULTS does not count looks made with `show_me=true`, so an arm whose switch was never used (0 of 70 Sol runs) reads as a measurement of it. Hash the schemas in the per-turn value (or record argument names), and count `show_me` looks per run in RESULTS.
- Evidence: `parts/wholeset40.md` §10 item 4; `parts/singles.md`, proposed follow-up 4; `parts/schedules.md` §9; `parts/callouts.md`, proposed follow-up 4.
- Confirmed in code: yes; `activity_log.py`, `ActivityLogger._turn_facts`; `aec_document_review/core.py`, `tools_fingerprint`; `review_eval/runner.py` has no such count.

**F-16. The hand-back lacked the working notes the scorer reads.** Kind: record. Belongs to: the review suite (the Foundry hand-back).
- Backlog text: each whole-set `run.json` lists `files/.scratch/page_readings.json` and `reads.json`, and the hand-back's `files/` lacks them. After EV-41 the scorer still reads them, to mark a failed check "found only in the run's working notes", the measure of what was lost between reading and writing; on this hand-back it cannot be worked out, and readers had to rebuild the notes from `activity.jsonl`. The runner writes and lists them; they were lost between the run folder and the hand-back (the whole `.scratch` folder is missing, contact sheets included). Keep `files/.scratch/*.json` in the packing step of the next brief.
- Evidence: `parts/wholeset15.md`, "Follow-ups" F4; `parts/wholeset40.md` §10 item 2.
- Confirmed in code: yes; `review_eval/checks.py`, `run_check` and `working_text` (after EV-41); `review_eval/runner.py`, `run_task` lists every file under `files/`; the packing step is in the brief, not the package.

**F-17. To-do-only steps cost a quarter to two fifths of a short task's input tokens.** Kind: behaviour on any model. Belongs to: the Document Review page and its tools.
- Backlog text: on coord-submittal, steps that only updated the to-do list used 25 % to 42 % of each run's input tokens on both models, and on GPT-5.4 coord-revisions 37 % and 52 of 201 s, because each step re-sends the whole conversation. Two texts push this way: the framework's to-do tool ("Mark tasks complete IMMEDIATELY after finishing (don't batch completions)") and the page's prompt ("Plan multi-step work with `write_todos` ... and keep it updated"). Measure the share over all 84 runs first; whether to keep `write_todos` for short tasks, or allow several items to be marked at once, is the owner's call.
- Evidence: `parts/revisions.md`, "Why tokens range from 0.30 to 0.61 million" and proposed follow-up 4.
- Confirmed in code: yes; `aec_document_review/review_prompt.py` (the `write_todos` rule) and LangChain's `agents/middleware/todo.py`.

### Noted, not proposed

- A planted problem was named in the main model's reasoning and left out of the log four times (section 4).
- Helpers dropped what their own page readings found: 12 of the 15 Sol losses on coord-long-review.
- Splitting sheets among helpers by page range separates the halves of cross-sheet problems; the bay was missed even when one helper held both halves.
- Each tile call sees one ninth of one sheet with the whole question, so no single vision call can see a conflict between sheets.
- End-of-run "independent" helpers re-read the set and added nothing planted; it was the model's own choice each time.
- Helpers run two at a time, a deliberate setting; later helpers waited up to 329 s.
- Per-page prompts were written from guessed sheet contents (15-sheet baseline).
- A tile that cuts a dimension makes its reading hedge ("cannot be verified here").
- The bay was found by every run of the narrow task and missed in 9 of 10 whole-set reviews: breadth of attention, not ability.
- Schedule-to-plan doors were delivered less often than plan-to-schedule marks.
- A single zoom reading was taken as fact against the sheet's own text (show_me coord-long-review CR-08).
- Whole-sheet overviews misread small sheet numbers; the tiles read them right.
- The page tool's merged "found" list hides a disagreement between the overview and a tile.
- `read_document` returns three to five drawing sheets per call; agents rarely followed `next`.
- `sheet_index` does not read the cover's index table, returns nothing on a scan, and is called again by each helper (cheap).
- `sheet_index`'s revision column invites choosing sheets by revision; its description does not suggest it.
- `check_cross_references` on a scan gives no pointer to look instead.
- The page's rule to look at every page in scope means the sheet tools cannot save looking; it is deliberate.
- The draft review method's clause on revisions covers only flagged changes.
- A tool comparing the text of two issues would make an unflagged change mechanical to find; five of six runs found this one by reading text.
- The baseline coord-revisions main agent gave two helpers the same whole job.
- Direction-blind checks, the unscored Word files and the untested "DEVIATIONS: NONE" (section 6).
- The `door-not-scheduled-4` check cannot tell a schedule's numbering gap from a mark on a plan.
- The callout check's 200-character window is not too strict.
- The "pages read" information line names PDF numbers only, not the sheets missed.
- 2 x 2 tiles were too coarse for the scan's door schedule; a zoom rescued it.
- One wasted text-continuation call (coord-schedules baseline).
- GPT-5.4 records no reasoning text; told the page tool what to expect ("401-409"); judged 12 of 15 pages unchanged from text alone; and turned a refused argument into a change of plan.

## 11. Index of tasks

| task | part file and section |
|---|---|
| coord-sheet-index | `parts/singles.md`, "Task 1: coord-sheet-index" |
| coord-callouts | `parts/callouts.md`, "coord-callouts: the 15-sheet set with a text layer" |
| coord-schedules | `parts/schedules.md` §3 and §5.1 to §5.5 |
| coord-levels-dimensions | `parts/singles.md`, "Task 2: coord-levels-dimensions" |
| coord-drawings-vs-manual | `parts/singles.md`, "Task 3: coord-drawings-vs-manual" |
| coord-drawings-vs-geotech | `parts/singles.md`, "Task 4: coord-drawings-vs-geotech" |
| coord-revisions | `parts/revisions.md`, "coord-revisions" |
| coord-submittal | `parts/revisions.md`, "coord-submittal" |
| coord-full-review | `parts/wholeset15.md`, "Run by run" and the tables after it |
| coord-scan-callouts | `parts/callouts.md`, "coord-scan-callouts: the same 15 sheets as a scan" |
| coord-scan-schedules | `parts/schedules.md` §3, §4.2 and §5.6 |
| coord-long-callouts | `parts/callouts.md`, "coord-long-callouts: the 40-sheet set with a text layer" |
| coord-long-schedules | `parts/schedules.md` §3 and §6 |
| coord-long-review | `parts/wholeset40.md` §2 to §9 |
