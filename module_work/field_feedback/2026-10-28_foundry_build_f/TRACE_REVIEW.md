# Trace review: Foundry build f, MF-1b and MF-2 (model-first vs today's pipeline on Claude)

Reviewed 2026-10-29 with no model calls. This file is PUBLIC: report and item IDs, field names, counts and scores only. No hole names, depths, values, firm, person or post names, page text, Foundry dataset names, resource IDs, branch names or usernames. Detail that needs any of those is in the gitignored `raw/TRACE_NOTES_private.md`; the narrative misses, row by row, are in `raw/NARRATIVE_MISSES.md`.

**What ran** (plan of record `module_work/FOUNDRY_BRIEF_MODEL_FIRST.md`, section 0):
- Build f: app `5.33.0+foundry.20261028.1` (master `1c0c3ae`, RI-171 to RI-177), planlens `0.13.0+main.82660d3`, references `1.4.1+main.ceeef53`; numpy 1.26.4. `import scipy.optimize` fails on the host (INF-60).
- MF-1b: eight arms on Claude through Foundry's `palantir_models` chat class, pictures route, `checks='flag'`: logs today (Sonnet, Opus), lab today and first (Opus), calc today (Sonnet, Opus), narrative today and first (Opus). A two-item pilot build, the full build, then one resume build, 14:01 to 16:26 UTC.
- MF-2: the `ingest` stage over R05, R36, R37, R13, R16 and R28, off (today's readers, the vision voter and the label review) and on (model first: Haiku describes every page, Opus labels ranges, each stage reads its groups in one call). Four builds, each stopping at the 2.5 h build budget between reports, 16:30 to 01:42 UTC.

**What was read:** the FDE's hand-back note; the five results tables; every arm's `RESULTS.md`, `results.json`, `*_build.json`, `*_stdout.txt`, `*_retries.jsonl` and meter; every MF-1b run file (logs 15 x 2, lab 31 x 2, calc 10 x 2, narrative 8 x 2); for MF-2 every report's `run.json`, `triage.json`, `labels.json`, `qa.json`, record and bound record, the item files, the on arm's page descriptions and range labels. The glue code was read, never run. Everything was re-scored with today's scorers (`rescore_saved`, `_score_ingest_record`, $0); every saved score was reproduced.

---

## 1. Plumbing: build f's fixes held; the 0-failed claim is right for MF-1b, not for MF-2's on arm

| check | MF-1b | MF-2 |
|---|---|---|
| `is_error` on tool results (RI-172) | today's arms ran whole: 0 tool-round failures | the off arm's label review and readers ran whole |
| meter tokens in = package tokens in (RI-171) | equal on all 8 arms | equal on every report of both arms |
| failures reach the totals and print (RI-173) | yes: the one failure printed by stage | "ingest none" on every build, but see below |
| failed items | 1, then 0 after the resume | off 0. **On: 9 page groups, 45 pages of content unread** (below) |
| retries / gave up / rate limits | 2 (connection) / 0 / 0 | 18 (connection) / 0 / 0; 16.7 s of retry waits in all; throttle wait 1.0 s in one build |
| cached answers | no call under 3.4 s, no repeated signature | no call under 1.1 s. 57 groups of calls share model, tokens in and out: the vision voter's fixed-size page pictures, each 2.5 to 3.5 s, so not cache hits |
| peak 60 s | 6 starts, 136k tokens in | 24 starts (Haiku or Sonnet), 281k tokens in; calls ran one at a time (5 of 2,347 overlapped another) |
| largest request | 101,909 (narrative first) | off 94,160; on 100,011 |

**The one MF-1b failure.** Logs today on Sonnet lost R37_p26 in the full build to an HTTP 500 from Foundry's model service (32 s). Neither the engine (it re-sends rate limits and connection errors) nor the glue's retry wrapper (connection, timeout, rate limit, 503) re-sends a 500. RI-173's failures line printed it, and the resume build re-read it in 2 calls (37k in). The resume worked as designed: the other seven arms restored their files and made no call.

**MF-2's on arm left 45 content pages unread and its failures line says "ingest none".** Two kinds of `skipped` entry, both from `report_ingest/model_first_read.py`:
1. **A half still cut off at its output cap is given up.** The module says so: "Nothing is asked a third time." Lab pages on R13 (4), R28 (8) and R16's bound volume (5).
2. **"The call gave no structured answer ... does not match the schema (inputs / results: Input should be a valid list)".** Six calc calls, 28 pages: R16's bound volume (4 calls, 24 pages), R28 (1), R36 (3 pages). The engine unwraps a wrapper object sent as JSON text, but not a field whose list arrives in another shape. The run files keep no raw answer, so the shape is not known (most likely JSON text).

RI-110 counts an item that "could not be read" as failed and re-asks it on resume. These entries are `skipped` with `refused_by: "budget"`, so neither the failures line nor the glue's stop rule saw them. The half that was read keeps no item file of its own; its tests are in the record. A further 19 `skipped` entries on 5 reports of the on arm ("labelled X belong to no work item, so nothing read them") hold only divider pages that RI-177 set aside on purpose: false alarms. Two more of that kind repeat the bound volume's calc failures in the parent.

**`bound_together`, compared between the arms.** R13 is the same in both: an appended earlier report on pages 63-76. On R16, triage lists four bound documents in both arms (two volumes of the report itself, an earlier firm's logs and lab inside volume I, a seismic report); the package reads them as one bound record. That record starts at page 3 on the off arm (planlens and triage) and at page 4 on the on arm (triage alone: planlens' vote does not run model first). There is none on the other four reports.

---

## 2. MF-1: per stage, model-first against today

Scores are with today's scorers. "First" for logs and calc is build e's arms (not re-run). "Local" is the 2026-10-24 runs on the Anthropic API. The spread is between today's runs of the same stage, which is the noise.

| stage | model-first | today | spread of today's runs | verdict |
|---|---|---|---:|---|
| logs /521 | Sonnet 494, Opus 494 (e) | Foundry Sonnet 477, Opus 473; local 475, 478 | 5 | **first +16 to +21**: 12 of it is header fields only first's schema asks for. Reading: +4 to +9, about the noise |
| lab /826 | Opus 759 (f), 752 (e) | Foundry Opus 778; local 773 | 5 | **today +19**: one sheet whose grading series the builder refused (R15_p82, -19). The rest level |
| calc /203 | Sonnet 176, Opus 176 (e) | Foundry Sonnet 173, Opus 179; local 172, 178 | 1 | **level** |
| narrative recall /235 | Opus **206** (f, one call a report) | Foundry Opus 196 (f), 194 (e); local 196 | 2 | **first +10**, five times the spread |

### 2.1 Logs
- Today's Foundry arms are valid this time (build e's were void) and land on the local runs: Sonnet 477 against 475, Opus 473 against 478.
- Where first gains over today: header fields +12 (schema only), water +7 (R15_p46: today's floor and merge lose rows the model reads), layer tops +2 to +4.
- USCS on R07_p30: first leaves `uscs` blank where the log prints non-USCS codes, as its instruction says. Today's Opus arm did the same this time (28/35); today's Sonnet arm copied the codes (35/35).
- Cost: today 3.4 times first (522k and 519k against 152k tokens in).

### 2.2 Lab
- RI-175's sentence worked: R17_p114 56/56 and R28_p163 47/47 on first (build e 43 and 32).
- The -19: R15_p82's grading series came back empty. The model said its x values were sieve designations, not openings, and the builder refuses a series that "runs the wrong way" whole. The run file keeps the count of refusals (1), not the reason, so this is the likely cause, not a proven one. Build e's first arm read the same sheet 10/10.
- Gains R15_p184 +4, R27_p85 +4; losses R17_p155 -3 (read as `other`), R35_p45 -5.
- Placement 52 of 70 values at a spot (74 %); this stage places about two values a sheet.
- Cost: today 1.5 times first (786k against 510k).

### 2.3 Calc
- Level on both models. Item swings either way are 2 to 7 points (R20_p344, R30_p542, R23_p386, R18_p50). Cost: today 1.7 to 1.9 times first.

### 2.4 Narrative

| report | first (f) | today (f) | first (e) |
|---|---:|---:|---:|
| R05 | 22/27 | 23/27 | 25/27 |
| R06 | 23/29 | 22/29 | 15/29 |
| R15 | 30/31 | 29/31 | 23/31 |
| R21 | 31/33 | 28/33 | 26/33 |
| R26 | 25/28 | 24/28 | 17/28 |
| R28 | 27/32 | 22/32 | lost |
| R36 | 25/28 | 25/28 | 18/28 |
| R37 | 23/27 | 23/27 | 16/27 |
| **all** | **206/235** | **196/235** | 140/203 |

- Precision: first 206/235, today 195/231.
- Under the owner's rulings of 2026-10-29: first 209/234, today 201/234, still +8 (section 4.1).
- RI-175 turned build e's -33 into +10. One call a report, 0 cut off, 248 of 256 citations placed at a spot. 529k tokens in against today's 650k (build e's first: 1.67 M).

---

## 3. MF-2: the whole report, off against on

### 3.1 Against the answer sheets

| | off | on |
|---|---:|---:|
| logs (4 items) | **216/225** | 206/225 |
| lab (12 sheets) | **258/268** | 216/268 |
| calc (R16_p162, in the bound record) | 23/25 | 23/25 |
| soundings (3 sheets on R28) | 139/143 | 140/143 |
| narrative recall (R05, R36, R37, R28) | 95/114 | **98/114** |

- Lab -42 is one sheet: R28_p163 0/47 on the on arm, its pages in the unread half (section 1). Everything else equal.
- Logs -10: R13_p45 -5 (water contents); R36_p38 -4 (ring-sample blows not recorded; the reader's own QA entry says why it held back); R28_p57 -1. MF-1's model-first logs (build e) read R36_p38 44/44.
- Narrative: off R05 26, R36 24, R37 23, R28 22; on 24, 26, 22, 26.

### 3.2 Holes printed against holes in the record

| report | printed (log sheets) | off | on |
|---|---:|---:|---:|
| R36 | 13 (11 borings, 2 CPT) | 13 | 13 |
| R37 | 5 | 5 | 5 |
| R13 | 28 (15, plus 13 in the bound earlier report) | 27 | **28** |
| R16 | 9 (in the bound volume) | 9 | 9 |
| R28 | 74 (15 borings, 4 CPT, 22 probes, 33 test pits) | 74 | 74 |

- Wave 2's R28 lacked 18 of its holes. Build f lacks none in either arm (RI-117, RI-135, RI-145 hold).
- Extras: off +4 test-pit records on R28 off a photo-and-sketch appendix, each repeating a pit already logged. On +1 probe read off a plot of all the probes. Both arms split two probes of two pages into two records each (today's sounding reader, on both arms).
- Off's R13 miss is one of the earlier report's borings, set aside as "a different boring" of the log before it.

### 3.3 Page labels: pages sent to the wrong reader (R13, R16, R28; 1,078 hand-labelled pages)

| | rules alone | final, off (vision + review) | final, on (ranges) |
|---|---:|---:|---:|
| R13 | 26 | 12 | 5 |
| R16 | 9 | 7 | 10 |
| R28 | 32 | 7 | 17 |
| **all** | **67** | **26** | **32** |

Eleven of the on arm's 17 on R28 are one appendix of foundation sketches that the hand labels call test-pit logs and the range labeller calls photos.

### 3.4 Unread pages and other differences
- On: 45 content pages unread (section 1). Off: none.
- **R16's narrative, on arm, reads 3 pages.** With `narrative_skips_bound` off on both arms, today's narrative reads the parent across the bound pages (77 pages; 25 of 29 general fields answered). Model first reads only the parent's own 3 cover pages (12 of 29). It does not honour the switch. The bound record answers 28 of 29 in both arms.
- **DIGGS:** on R13 the on arm's file is valid but "read back DIFFERS": a summary-table row's resistivity is not written (5 values). Every other DIGGS file is equal on read-back.
- The on arm's records (parent and bound) hold fewer QA entries (1,199 against 2,151) because the floors and voters that raise disagreements do not run.

### 3.5 Is each report's file complete?

Checked from the text layer (Document Intelligence text on scanned pages) and from the page describer's reading of each page's printed numbering. A printed page counts as missing only where the file has no page that could hold it, and every gap was checked by hand. Citations of a code's or a textbook's own figures and tables are left out: the design-map printouts, a pavement design sheet, and textbook pages reproduced in a calculation.

| report | pages in the file | pages printing "Page X of Y" | printed pages not in the file | cited by the text | cited, not in the file |
|---|---:|---:|---|---|---|
| R05 | 15 | 14 | 1: the body's last page (it prints "of 16") | figures 6, appendices 7 | **figures 6, appendices 7** |
| R36 | 94 | 19 | 0 | exhibits 30, appendices 5 | 0 |
| R37 | 48 | 5 | 0 | tables 8, appendices 3, drawings 1 | 0 |
| R13 | 197 | 0 (scanned; 50 pages carry a sheet number) | 0 | figures 10, tables 8, appendices 4 | 0 |
| R16 | 426 | 78 | 0 | figures 5, tables 12, appendices 5 | 0 |
| R28 | 455 | 39 | 10, all in two laboratory certificates: one is printed as its pages 3-6 of 13, the other lacks its page 10 of 10 | figures 22, tables 14, appendices 4 | 1 table: a laboratory sheet's "attached" point table |

- R05 is the only incomplete copy of a report itself, as RI-182's ruling 6 found.
- R28's gaps are in attachments. RI-184's footer rule would flag them too, so its note should name the document whose numbering is short.
- Two runs that first looked short were misread numbering, both checked by hand: R13's handwritten calculation-sheet numbers, and a describer note giving the PDF's own page count.

---

## 4. Narrative misses, verdicts (private rows in `raw/NARRATIVE_MISSES.md`)

Runs: **A** MF-1b today (8 reports), **B** MF-1b first (8), **C** MF-2 off (4), **D** MF-2 on (4). Where build f gave build e's answer to the same question, build e's verdict is carried. This table scores against the answer sheets as they stand. Section 4.1 applies the owner's rulings of 2026-10-29 (RI-182).

| verdict | A | B | C | D | all |
|---|---:|---:|---:|---:|---:|
| ours wrong | 12 | **2** | 8 | **1** | 23 |
| convention | 17 | 19 | 8 | 11 | 55 |
| sheet wrong | 6 | 6 | 3 | 3 | 18 |
| wording | 7 | 4 | 3 | 2 | 16 |
| **rows** | 42 | 31 | 22 | 17 | 112 |

**Model first's real misreads are 2 of 31 on eight reports; today's are 12 of 42.** Today's 12: 4 repeated items, 3 counts where the caption count replaced the model's answer, 2 summaries just over 100 words, 1 floor slab given as a foundation, 1 hazard-analysis "unclear", 1 free hazard phrase. Model first's 2: a summary at 103 words; one "unclear" saying an appendix is missing that the copy prints.

**Build e's twelve patterns:**

| # | pattern | build f |
|---|---|---|
| 1 | structures: buildings only | ruled (4.1): the 22 rows are ours wrong |
| 2 | repeated items | **gone on model first**; still on today (A and C, 7 rows) |
| 3 | summaries over 100 words | 5 rows, each 101 to 105 words |
| 4 | project phase "other" outside the estate | ruled (4.1): the 10 rows are ours wrong |
| 5 | the caption count overrides the model | **gone on model first**; still on today (5 rows) |
| 6 | model first without the house rules | **mostly gone (RI-175)**; one rule left: today's prompt rule 4 (another client's report: `propertyType` "other") is not in the model-first instruction (6 rows, plus 2 `projectPhase` answered "design") |
| 7 | count scope: this investigation or all | ruled (4.1): the 8 count rows pass; hole lists cover this investigation |
| 8 | foundation and bearing list grain | 16 wording rows; RI-181 (no ruling needed) |
| 9 | hazard phrases | **gone on model first** (build e's 9 free phrases); the sheet-wrong rows corrected (4.1) |
| 10 | earlier investigations | ruled (4.1): a neighbouring site's investigation counts |
| 11 | figures cited but not in the copy | ruled (4.1): the report is the file |
| 12 | hole names only in the calcs | **gone** in all four runs |

New: `soilCorrosion` "mixed" on R36 (A, C). The glossary's draft rule gives "mixed" for soil corrosive to metal and not to concrete; the sheet says "yes". That is an owner question, not a misread.

### 4.1 Under the owner's rulings (RI-182, 2026-10-29)

Scored a second time, with the rulings and corrections applied to the answer sheets in memory. The sheet files are unchanged until the product build.

**What changed in the sheets:**
- Ruling 1 (buildings only; a parking garage counts; site works do not; no building gives 0): no change, since the sheets already count that way.
- Ruling 2 (the phase the report names, else "other"): no change, since each "other" is a report that names no phase. The new companion question is not scored.
- Ruling 3 (counts and hole lists cover this investigation; the all-investigations count is also accepted):
  - R28's and R21's hole lists are cut to their own investigations.
  - R28's accepted all-investigations counts are the totals the pages print, which the sheet's alternates were not.
- Ruling 4: the hazard corrections on R06 and R36.
- Ruling 5: no change, since R05's sheet already counts the neighbouring site's investigation.
- Ruling 6 (the report is the file; RI-182 records it as ruled): R05's figure count is 0.
- Ruling 7: no change, since the scorer already folds spelling variants.
- The other sheet corrections: R06's corrosivity, R15's mat contact pressure, R28's ASCE 7 edition.

| run | recall, sheets as they stand | recall, under the rulings | precision, as they stand | precision, under the rulings |
|---|---:|---:|---:|---:|
| A MF-1b today | 196/235 | 201/234 | 195/231 | 200/231 |
| B MF-1b first | 206/235 | **209/234** | 206/235 | **209/235** |
| C MF-2 off | 95/114 | 97/114 | 94/111 | 96/111 |
| D MF-2 on | 98/114 | **100/114** | 98/113 | **100/113** |

- Recall's total falls by one on R21: its own investigation drilled no borings, so its boring list has no hand answer.
- Model first leads today's reader by 8 under the rulings, against 10 on the sheets as they stand. The spread of today's runs (2) was measured on the sheets as they stand.
- Per report under the rulings, first / today: R05 21/24, R06 25/24, R15 31/30, R21 29/26, R26 25/24, R28 29/24, R36 26/26, R37 23/23.

**Which rows change:**
- 23 misses pass: 14 sheet-wrong rows (hazards 6, the ASCE edition 4, corrosivity 2, the mat pressure 2) and 9 convention rows (R28's counts 8, R05's figure count 1).
- 11 passes become misses:
  - hole lists that also name earlier investigations' holes: R28's boring list in all four runs, R21's two lists in A and B;
  - R05's figure count wherever a run counted the figures the text cites (B, C, D).

| verdict | A | B | C | D | all |
|---|---:|---:|---:|---:|---:|
| ours wrong | 28 | 19 | 15 | 10 | 72 |
| convention | 2 | 5 | 2 | 3 | 12 |
| sheet wrong | 0 | 0 | 0 | 0 | 0 |
| wording | 7 | 4 | 3 | 2 | 16 |
| **rows** | 37 | 28 | 20 | 15 | 100 |

**Where the 72 ours-wrong rows come from:**
- carried from the table above: 23 (A 12, B 2, C 8, D 1)
- ruling 1, structures: 22 (A 9, B 8, C 2, D 3)
- ruling 2, project phase: 10 (A 3, B 3, C 2, D 2)
- ruling 3, hole lists: 12 (A 4, B 4, C 2, D 2)
- ruling 5, an earlier investigation: 2 (B, D)
- ruling 6, the figure count: 3 (B, C, D)

17 of model first's 19 are rules the build f readers had never been given: RI-182 puts them in the glossary.

**The 12 rows still a convention after the rulings:**
- `propertyType` for another client's report, on model first: 6 (follow-up 5)
- R36's corrosivity "mixed": 2
- on R37, a blank against 0 earlier investigations: 2
- on R37, whether an appendix drawing is a figure: 2

Ruling 3 accepts the all-investigations count but not the all-investigations list. 11 of its 12 list rows would pass if the list were accepted the same way: an owner question.

---

## 5. Spend (the meters: plain + cache write + cache read)

**MF-1b:** 192 calls, 4.16 M in, 0.69 M out.

| arm | calls | in | out |
|---|---:|---:|---:|
| logs today, Sonnet | 30 | 522,307 | 125,345 |
| logs today, Opus | 29 | 519,128 | 113,817 |
| lab today, Opus | 40 | 786,174 | 110,177 |
| lab first, Opus | 31 | 510,139 | 98,861 |
| calc today, Sonnet | 15 | 304,896 | 44,067 |
| calc today, Opus | 16 | 342,092 | 60,802 |
| narrative today, Opus | 23 | 649,733 | 85,906 |
| narrative first, Opus | 8 | 529,014 | 49,254 |

**MF-2 by report:** calls / tokens in, and the share of the brief's stop line.

| report (pages) | off | share | on | share |
|---|---|---:|---|---:|
| R05 (15) | 8 / 52,844 | 9 % | 18 / 138,341 | 42 % |
| R36 (94) | 60 / 903,050 | 24 % | 117 / 1,143,340 | 54 % |
| R37 (48) | 31 / 480,270 | 25 % | 60 / 538,039 | 49 % |
| R13 (197) | 147 / 2,718,424 | 34 % | 252 / 2,716,356 | 63 % |
| R16 (426) | 231 / 3,898,387 | 23 % | 496 / 5,670,474 | 60 % |
| R28 (455) | 319 / 5,395,450 | 30 % | 608 / 6,674,665 | 67 % |
| **all (1,235)** | **796 / 13.45 M**, out 2.50 M, 4.4 h | | **1,551 / 16.88 M**, out 2.15 M, 3.9 h | |

**MF-2 by stage.** The step files leave out cache writes (RI-171's noted line), so each step file was matched to the run's per-call rows by output and cache-read tokens. Every call was matched except the on arm's 19 calls that left no step file: the cut-off and unanswered calls, and the read halves of groups whose other half was given up.

| stage | off: calls / in | on: calls / in |
|---|---|---|
| page describer (Haiku) | | 1,235 / 8.75 M (52 %) |
| vision voter + label review | 242 / 1.27 M | |
| range labels | | 12 / 0.65 M |
| triage + identity | 8 / 0.15 M | 8 / 0.15 M |
| logs and soundings | 269 / 4.75 M | 163 / 2.16 M |
| lab | 85 / 2.42 M | 17 / 0.88 M (17 pages unread) |
| calc | 171 / 4.24 M | 90 / 3.20 M (28 pages unread) |
| narrative | 21 / 0.61 M | 7 / 0.32 M |
| calls with no step file | | 19 / 0.79 M |

- By model: off Sonnet 9.04 M in and Opus 4.41 M. On Haiku 8.75 M, Sonnet 4.89 M and Opus 3.25 M.
- The model-first readers use 7.3 M against today's 12.0 M. The page describer more than gives the saving back in tokens, but on the cheapest tier. Dollars have to come from Foundry's usage report for the window; the hand-back carries tokens only.
- Per page: off 10.9k, on 13.7k (the brief estimated about 15k for on).

---

## 6. Recommendation: a product build's default

Per stage, from sections 2 and 3:
- **Narrative: model first.** +10 on MF-1 (five times the spread) and +3 on MF-2, or +8 and +3 under the owner's rulings; one call a report; fewer tokens.
- **Lab and calc: today's readers until two model-first defects are fixed** (items 1 to 3 below). Both stages read level or better when nothing is dropped (MF-1 calc level; lab level apart from one refused series), but in a whole report the on arm lost 45 pages without a failure.
- **Logs: no winner on these numbers.** First leads on items (MF-1, about the noise once the schema-only header fields are set aside); today leads on the whole report (MF-2, -10 on two items). Today's reader costs 3.4 times as much.
- **Page labels: today's** (26 pages to the wrong reader against 32; the describer is half the on arm's input).

The package switches model first on or off for every stage at once (`GEOTECH_INGEST_MODEL_FIRST`). The default above needs a per-stage setting (item 6).

---

## 7. Noted, not fixed (one line each)

- The engine and the glue do not re-send an HTTP 500; the resume recovered the one item it cost.
- R15_p82: the lab builder refuses a grading series whole when sieve numbers stand in for openings; the run file keeps no refusal reason. One sheet in one run.
- Model-first logs left out one log's ring-sample blows (R36_p38) and five water contents on another (R13_p45). Both are run-to-run differences: build e's model-first logs read R36_p38 whole.
- The step files' spend leaves out cache writes (RI-171's noted line); per-stage tokens above were matched from the per-call rows.
- Both arms split two two-page probe sheets into two records each (today's sounding reader).
- R28's foundation-sketch appendix: the hand labels call it test-pit logs; the range labeller calls it photos; the off arm reads 4 repeat pits from it.
- 36 MuPDF colour-profile messages in one MF-1b build log; harmless.
- The on arm runs no label review by design; its 151 split pages stay as `label_disagreement` entries.

---

## 8. Follow-ups for BACKLOG (the lead files them; IDs are not assigned here)

Each changes a saved report's output or what a live run returns.

1. **Model first: a half cut off at the output cap is cut again**, down to one sheet, instead of given up. 17 lab pages on 3 reports; R28_p163 0/47.
2. **Model first: an answer refused as "not a valid list".** Keep the raw answer in the skipped entry, and accept the list's shape if it is JSON text. 6 calc calls, 28 pages on 3 reports.
3. **A model-first `skipped` group counts as a failed item** (`item_errors`, the failures line, re-asked on resume), as RI-110 does for "could not be read". The on arm reported "ingest none" with 45 pages unread.
4. **Model-first narrative honours `narrative_skips_bound`.** R16's parent read 3 pages (12 of 29 general answered against 25).
5. **Model-first narrative carries today's prompt rule 4** (another client's report: `propertyType` "other", `outsideProject` "yes", no post). 6 misses on 4 reports.
6. **A per-stage model-first setting**, so a product build can read the narrative model first inside today's pipeline (section 6).
7. **Divider pages set aside by RI-177 get no `skipped` entry.** 19 false entries on 5 reports, each a stop under RI-110's rule.
8. **The DIGGS writer writes a summary-table row's resistivity.** R13's on-arm file fails its read-back on 5 values.

Owner questions (no code). Build e's patterns 1, 4, 7, 10 and 11 are ruled (section 4.1); these are left:
- whether a hole list may name earlier investigations' holes, as the count may (ruling 3; 11 rows);
- R36's `soilCorrosion` "mixed" against the sheet's "yes";
- on R37, a blank against 0 earlier investigations;
- on R37, whether an appendix drawing is a figure.
