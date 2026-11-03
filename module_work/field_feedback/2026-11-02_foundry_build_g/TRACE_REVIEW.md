# Trace review: Foundry build g, MF-3 (model first, the new default, against today's readers on R15, R30, R21)

Reviewed 2026-11-02 with no model calls. **This file is PUBLIC.** It holds report and item IDs, page numbers, field names, counts and scores only. It has no hole names, depths, values, firm, person or post names, page text, Foundry dataset names, resource IDs, branch names or usernames. Detail that needs any of those is in the gitignored `raw/TRACE_NOTES_private.md`. The three readers' full sections are beside it: `raw/section_R15.md`, `raw/section_R30.md`, `raw/section_R21.md`.

**What ran** (plan of record `foundry_handoff/gse_app_2026-11-02/RUN_PLAN.md`):
- **Build g:**
  - app `5.33.0+foundry.20261102.1`: master `0fa2ed5` plus a version-only commit. `report_ingest/` is unchanged on master since then, so every line number below holds on master.
  - planlens `0.13.0+main.82660d3`, references `1.4.1+main.ceeef53`, numpy 1.26.4.
- **MF-3:** six `ingest` jobs in parallel on Claude through Foundry's `palantir_models` chat class, 15:15Z to 18:01Z.
  - **Arm A**, model first (the build's default, RI-186): Haiku describes every page, Opus labels ranges under the owner's conventions B, and each stage reads its page groups in one call.
  - **Arm B**, `model_first="off"`: today's readers, the vision voter and the label review.
  - Each arm read three held-out reports: R15 (202 pages), R30 (556) and R21 (729, with an appended prior report on pp120-591).
- **Then the optional question-list step (RI-101):** on arm A's R15 and R30 records.

**What was read:**
- the FDE's hand-back note and `HANDOFF_brief1027.md`;
- `mf3_calls.csv` (2,832 calls), `mf3_summary.csv` and `install_check.csv`;
- every job's `run.json`, `arm.json`, `triage.json`, `labels.json`, `ranges-*.json`, `qa.json`, items, records, bound records and DIGGS files;
- arm A's page descriptions;
- the build logs, retry logs and store logs.

The FDE's glue (`code/`) was read, never run. Everything was re-scored with today's scorers and truth ($0), and every saved score was reproduced. The one exception: narrative scores moved where the owner's rulings of 2026-10-29 amended the answer sheets after the run.

---

## 1. Plumbing: sound; the only failures are two items with one cause (section 4)

| check | result |
|---|---|
| meter = package (calls, tokens in) | equal on all six jobs |
| refused / errors in the meter | 0 / 0 of 2,832 calls |
| failed items | A_R21 1, B_R21 1 (section 4); the other four jobs 0 and "ingest none" |
| stop lines (§5) | none passed. A jobs used 55-61% of theirs, B jobs 21-29%; no token brake |
| retries / gave up / rate limits | 11 connection drops (A_R15 1, A_R21 2, A_R30 2, B_R15 0, B_R21 4, B_R30 2), each re-sent once, 10.2 s of waits in all / 0 / 0 |
| cached answers | none. No call under 1.38 s. The 69 repeated (model, in, out) signatures are describer and vision calls on fixed-size pictures at normal durations |
| prompt cache across jobs | each B job's triage read the triage cache A had written minutes earlier on the same report. The outputs differ, so these are not cached answers |
| overlaps | 0 within any job; at most 6 calls in flight (one per job) |
| peak 60 s | 88 starts across the wave (Haiku 58, Sonnet 45, Opus 6) and 0.83 M tokens in; 20-26 starts per job against build f's "about 7 a minute per arm" |
| largest request | 121,863 tokens in (A_R30's range labeller). 8 requests were over 100k, all range-labeller or Opus reading calls, and none failed |
| output cut at 32k | A_R15 2, A_R30 3, B_R15 1; every one halved (RI-191) or re-asked |
| DIGGS | written, valid and equal on read-back in all six jobs |
| store (§4) | 10-33 snapshot puts per job, 0 warnings, 0 restores |

### 1.1 Against the brief's estimates (§3)

| job | calls (est) | tokens in (est) | out (est) | run h (est) | share of line |
|---|---|---|---|---|---:|
| A_R15 | 230 (250) | 2.47 M (2.8 M) | 0.48 M (0.35 M) | 0.82 (0.6) | 55% |
| A_R30 | 713 (700) | 7.67 M (7.6 M) | 1.22 M (1.0 M) | 2.15 (1.8) | 61% |
| A_R21 | 908 (920) | 9.63 M (10 M) | 1.14 M (1.3 M) | 1.93 (2.3) | 59% |
| B_R15 | 83 (130) | 1.69 M (2.2 M) | 0.42 M (0.4 M) | 0.74 (0.7) | 21% |
| B_R30 | 376 (360) | 6.63 M (6.1 M) | 1.23 M (1.1 M) | 2.18 (2.0) | 29% |
| B_R21 | 522 (470) | 8.56 M (7.9 M) | 1.45 M (1.5 M) | 2.61 (2.6) | 29% |
| **all** | **2,832 (2,830)** | **36.6 M (36.6 M)** | **5.94 M (5.65 M)** | wave 2.77 h | 39% of 94 M |

- The estimates held to 0.2% on calls and tokens in; output came in 5% over.
- By arm: A made 1,851 calls with 19.76 M in and 2.84 M out; B made 981 calls with 16.88 M in and 3.10 M out.

### 1.2 Tokens in by stage, three reports (step files matched to the per-call rows)

| stage | A | B |
|---|---:|---:|
| page describer (Haiku, 1,488 calls) | 10.25 M | |
| range labels (Opus) | 0.94 M | |
| vision voter + label review | | 1.57 M |
| triage + bound identity | 0.15 M | 0.15 M |
| logs and test pits | 2.74 M | 7.58 M |
| lab | 3.30 M | 5.31 M |
| calc | 2.02 M | 1.94 M |
| narrative | 0.25 M | 0.33 M |
| failed calls | 0.11 M | 0.01 M |
| **all** | **19.76 M** | **16.88 M** |

- **A's readers** (logs, lab, calc, narrative) took 8.3 M in against B's 15.2 M, 45% less.
- **A's page describer** is 52% of A's input, on the cheapest tier.
- **Per page:** A 13.3k, B 11.4k. The brief estimated about 15k and 27k high.

### 1.3 The FDE's glue against §4 ("every output kept as it is written")

- **Kept as written: met.** Every 300 s, and at the start and end, each job's folders are zipped into a store whose writes survive a failed build. The FDE checked this with a probe.
- **Restored at start: not wired.**
  - Each job restores only from its own output dataset, which is written at the end of a build that succeeded.
  - The store's docstring names a restore transform that is not in the code, and no job reads one.
  - A build that died mid-report would restart from its last good build unless someone restored it by hand.
  - It was not exercised, because no build died. This is RI-203.
- **Also:** the package's own mirror (`sharepoint=`) is not used, and `durable_dir` is a temp folder.

### 1.4 Install check

- Hashes and versions matched. `scipy.optimize` and `scipy.stats` import, so INF-60 is closed on this host.
- Model first reads `on` for every stage, with prompt mf-5.
- **Range-label convention:** the FDE's probe came back empty, but all three arm-A `run.json` files record the hash of conventions **B**.
  - The fingerprint is `61251c1d08`, the labeller prompt `7d0464c386`.
  - The probe looked for a module attribute; the default is the `range_conventions="B"` keyword of `score_on_cluster`.
- `report_ingest.mirror` resolving to `funhouse_agent/run_mirror.py` is the intended alias (INF-30).
- The host's recipe still installs 5.31.0 beneath the test wheels; the wheels win every import that is made.

---

## 2. Per stage, three reports, both arms

| stage (truth) | R15 A / B | R30 A / B | R21 A / B | **all: A / B** |
|---|---|---|---|---|
| logs (R15_p46, R30_p65, R21_p96) | 64/64 / 64/64 | 21/22 / 19/22 | 20/22 / 16/22 | **105/108 / 99/108** |
| lab (R15_p82, p112, p184) | 75/75 / 71/75 | | | **75/75 / 71/75** |
| calc (R30_p542, p555) | | 32/37 / 33/37 | | **32/37 / 33/37** |
| test pit sheets (R15_p68, R30_p152, R21_p43) | 15/16 / 14/16 | 17/18 / 18/18 | 11/11 / 10/11 | **43/45 / 42/45** |
| narrative recall, today's sheets (R15, R21) | 30/31 / 30/31 | (no sheet) | 32/32 / 31/32 | **62/63 / 61/63** |
| page labels, final (1,487 hand pages) | 0.985 / 0.955 | 0.989 / 0.957 | 0.978 / 0.967 | **0.983 / 0.962** |
| harmful labels | 0 / 5 | 6 / 11 | 15 / 24 | **21 / 40** |
| failed items | 0 / 0 | 0 / 0 | 1 / 1 | **1 / 1** |
| holes in the record (parent + bound) | 12 / 12 | 96 / 96 | 58 + 42 / 58 + 54 | |

- **Calc:** B's 4 misses on R30_p542 are the scorer. A printed ratio in B's result labels puts digits in them, and the word-subset match needs digits to agree. A's misses are one result block whose name it shortened, the same digits, and `program`, where A named a program the pages do not print.
- **Test pits:** A's two misses are both the pit width (RI-198).
- **Narrative:** on the sheets Foundry scored, before the rulings, R15 is 29/31 in both arms and R21 is A 31/33, B 30/33.
- **Page labels:** all 21 of A's harmful pages are hand-label questions: owner questions 4 and 5 in section 10, and one appendix tab already on RI-104's list. A has no real misroute.

---

## 3. Per report

### 3.1 R15 (202 pages)

- **Lab:** B reads one sheet's four density tests as one summary table (R15_p184, kind 0/4).
- **Test pit sheet R15_p68:**
  - A misses the pit width (RI-198).
  - B misses `contract_number`. Today's floor also puts the equipment text in `method`, which is scored.
- **Narrative:** both arms miss `recommendedFoundations` the same way. The sheet has two items, one naming two foundation types; both readers give three, one per type as RI-181 rules (owner question 7).
- **Labels:**
  - A is wrong on 3 pages: two volume title pages, and p67, a boring log's last sheet that the hand labels call a test-pit log (owner question 6).
  - B's 5 harmful labels: one figure and three pages of the chemical lab's report set to `other` (those three go unread), and two `other` pages set to narrative.
- **The depth window (RI-199):**
  - One boring's log runs to six sheets, and planlens found no depth ruler on the first or last.
  - `log_reader._depth_window` (log_reader.py:845) unions the ranges of the sheets that do have one. Model first uses the same window (model_first_read.py:741).
  - So the first sheet's top and the hole's bottom fall outside the window.
  - **B** (`checks="refuse"`) dropped 5 samples and 2 SPT records.
  - **A** (`checks="flag"`) kept them, with 19 false `out_of_range` entries at confidence 0.5.
  - This log has no truth file, so no score shows it.
- **Lab tests:** A 137, B 149.
  - B repeats the chemical lab's seven samples about three times and adds seven moisture contents; A repeats two of them.
  - The values are the same. There is no chemical truth.
- **Cut-offs:**
  - A halved one cut-off lab group twice (RI-191). Every page was read, at 111k in and 64k out for the two cut calls.
  - B re-asked one cut-off item. The second answer dropped a summary table's elevation bottoms "for output length", as the reader itself said.
- **Spend:**
  - A: 230 calls, 2.47 M in, 49 min. 54% of the input is Haiku page descriptions.
  - B: 83 calls, 1.69 M in, 44 min.

### 3.2 R30 (556 pages)

- **"0 failed" is true on both arms.**
  - RI-191 halved three cut-off Opus lab calls on A, and every half answered.
  - RI-187 was not exercised. RI-193 raised no false divider entries.
- **Seven pages read by nothing on A** (487, 491, 494, 498, 505, 509, 516). These are the slope-stability appendix's section sub-dividers.
  - The range labeller labels them calculation, under conventions B rule 7, as the hand labels do.
  - RI-177 then sets them aside inside the calc grouping with no record entry.
  - The calc item that starts on the next page names its section, so nothing printed is lost. RI-197 gives such pages a note.
- **Holes:** 44 borings and 52 test pits in both arms, with the same IDs.
  - **Three one-page driller's site logs (pp66, 74, 80):** B's log reader found no depth ruler and refused every depth, so those holes have 0 layers and 0 SPT records. A keeps 15 layers and 21 SPT records, the same depths B's model read.
  - **Layer splitting:** on logs that print one description per sample, A writes a layer per description and B per drawn stratum line. There is no truth to say which is right.
- **Test pit widths:** A writes a width on 0 of 52 pits, B on 45 of 52 (RI-198).
  - Model first's log schema inherits the `pit` field, but its instruction lacks today's paragraph saying the bucket is the width.
  - R30_p152: A 17/18, B 18/18.
- **Lab:** A 273 tests, B 346. Matched by kind, hole and depth, the core test sets are the same. B's 73 extra are:
  - 47 summary tables built from repeated header cells (B's own QA calls two of them "not a real summary table");
  - 14 `other` entries from dividers and certificate pages;
  - 5 CBR swells as separate tests;
  - 3 hydrometer sheets as separate gradation tests;
  - 4 chemical and moisture rows.
- **Calc:** A has 26 items, one per printout; B has 8, with several printouts per item. A placed 95.8% of 2,042 values at a spot.
- **Labels:** A 0.989 with 6 harmful, against B 0.957 with 11.
  - A's 6: five specimen photo pages inside the lab appendix (owner question 4), and one appendix tab that is a known hand-label mismatch on RI-104's list.
  - B's review fixed 7 pages and broke 1. It rejected 13 changes as "not looked at", 6 of which would have helped.
- **Narrative** (no answer sheet; checked against the pages):
  - The arms agree on the identity fields, all five exploration and structure counts, site class, seismic code, liquefaction and two foundation types.
  - Rulings held: RI-181 (one item per foundation type), RI-182 (the project phase companion field) and RI-189 (the other-client rule correctly not triggered).
  - **A right, B wrong:** `tableCount` (A 15, B 12) and `figureCount` (A 9, B 0). `count_captions` counts 0 here, and today's reader hands that 0 to the model as a given fact.
  - **B right, A null:** `asceSevenVersion`. The edition is printed only on an appendix calculation sheet (p555). Today's reader searches the whole report for seven appendix-answered questions; model first reads only the narrative's pages (RI-200, owner question 3).
  - **Differences with no rule yet:** `soilCorrosion` (owner question 1), `earthHazardsExposed` (owner question 2), `projectNumber` (two printed forms), `structureList` (abbreviations against full names).
  - **RI-178:** B dropped 7 `bearingCapacity` items as repeats, and the item it kept names one structure where the value covers about ten. A's answer writes one item per value and names every structure.
  - **RI-179:** A's summary was sent back once and came back inside the limit.
  - **RI-184:** both arms carry six open "this copy looks incomplete" entries, and all six are false. R30 reproduces selected pages of earlier investigations' photo logs (pp119-129, 197-226), and their "Page X of Y" runs look short (RI-201).
- **Spend:**
  - A: 713 calls, 7.67 M in, 7,727 s.
  - B: 376 calls, 6.63 M in, 7,860 s.
  - A sends 3.78 M to Opus and Sonnet against B's 6.63 M.

### 3.3 R21 (729 pages, prior report bound on pp120-591)

- **The bound report is kept apart in both arms:**
  - No value in a parent record cites a bound page, and none in a bound record cites a parent page.
  - The parent narrative read 48 (A) and 51 (B) pages, none of them bound.
  - Bound counts: A 42 investigations and 98 lab tests; B 54 and 131. There is no bound truth.
- **A lost 12 of the bound report's 36 test pits** (pp223, 224, 227, 230, 235, 237, 238, 239, 242, 254, 257, 258; RI-196, RI-197).
  - **The groups:** rebuilding A's log groups from its own page descriptions reproduces them exactly.
  - **Pages that open a log:** a page whose describer `exploration` field holds only the name opens its own log.
  - **Pages read as no name:** a page whose field adds a note after a comma (a depth, an elevation, water) reads as no name. `exploration_id` (model_first_pages.py 508-544) refuses two numbers, and `_folded` refuses any comma. So `log_groups` (894-967) adds the page to the log before it.
  - **The reader:** it read one pit per group and said the rest were other logs, or returned them in `pages_left`.
  - **The record:** only 2 of the 12 pages got a QA entry, worded wrongly for model first ("did not get to this page within its model-call budget", log_reader.py ~1428-1434). None is an item error, so a resume would not re-read them.
  - **The missing-holes check** (RI-144) reads names from the text layer, and these pages are scans.
  - B, reading one item per page, kept all 36.
- **A's parent: a photo-only test pit page (p45) was set aside as a divider.** Its describer kind mentions a divider in a note, and `is_divider` (model_first_pages.py 800-809) matches the word anywhere. The page holds photographs only, so A loses one empty investigation.
- **Scores:**
  - Log R21_p96: A 20/22 (one layer top, `sheet`); B 16/22 (two `pp_kpa` values, `project`, `project_number`).
  - Test pit: A 11/11, B 10/11.
  - Narrative: A 32/32, B 31/32 (`figureCount`).
- **Labels:** A 0.978 with 15 harmful, against B 0.967 with 24.
  - The 15 in both arms are pp714-728, which the hand labels call calculation and both arms read as a laboratory chemistry report (owner question 5).
  - B split 496 pages, mostly bound pages (rules `appended_report` against vision), and leaves 492 `label_disagreement` entries in the parent's QA. A split 12.
- **Calc** (no truth): A cut the parent's printouts at printout boundaries (14 items, RI-176); B split long printouts into page runs (20). In the bound report A has 30 items plus the failed window, B 23.
- **Spend:**
  - A: 908 calls, 9.63 M in, 6,954 s.
  - B: 522 calls, 8.56 M in, 9,382 s.
  - A's readers took 3.9 M against B's 7.7 M.
  - 13 describer answers arrived empty and were recovered by the engine's text fallback; all 729 pages are described.

---

## 4. The failures and their one cause

**The two failed items** (A_R21: bound calc pages 453 and 464-468; B_R21: item_47, test pit log p81) **have one cause.** Claude's own tool-parameter markup arrived inside the JSON argument values. The answer failed the schema, and neither reader asks again.

- **stop_reason `tool_calls` is not the fault.** The engine takes every answer through an answer tool, so every call ends that way.
- **A, call 889 (Sonnet, 35k in):** window 2 of a pile-analysis printout.
  - The raw answer the engine kept (`answer_args`) has `"inputs": "<parameter name=\"0\">{...}"`, with the list's later items spilled out as top-level keys "1" to "14". `results` has the same.
  - Refused as "inputs / results: Input should be a valid list".
- **B, call 222 (Sonnet, 12k in):** `"pit": "null</pit>\n<parameter name=\"remarks\">..."`, the next field swallowed into this one.
  - Refused as "pit: Input should be a valid dictionary".
- **Nothing asks again.**
  - Model first: `_Caller.read` (model_first_read.py 500-507) turns a reply with no structured answer into a `skipped` row at once.
  - Today's log reader (log_reader.py ~1892-1900) raises on its first call, although its budget is 6.
  - Today's calc reader does ask again (calc_reader.py 2767-2771). It recovered a refused answer in this same job (B calls 263 then 264).
- **The same leak elsewhere in the wave** (a grep of all six jobs):
  - A_R21's describer p255 was refused, re-asked by the describer, and recovered.
  - Four descriptions were **accepted** with one field swallowed into another: A_R21 p372, and A_R30 p124, p314, p359 (`issuer_date`, `exploration`, `identifiers`, `continues`, `language`). The `exploration` field feeds log grouping (section 3.3).
  - In all, about 7 in 2,832 calls: Sonnet 2 of 1,058, Haiku 5 of 1,488, Opus 0 of 286.
- **Build f's six "not a valid list" calc refusals were most likely the same thing.**
  - Build f kept no raw answer, so RI-192 assumed JSON text. `_json_text` (engine.py 3325) reads only text that starts with `[` or `{`.
  - Build g's kept answers show the markup instead, which RI-192 does not read.
- **The FDE's stop was right** (§5 point 4: the same error failed items in two jobs). A resume would very likely recover both items: A in 2 Sonnet calls (about 98k in), B in 1 (about 12k). But a wave of about 2,800 calls should expect about 7 leaks and about 2 failed items, so RI-195 belongs before the next wave.
- **After the hand-back, the FDE ran the resume-once (2026-11-02, reported in a note; not yet in a hand-back):**
  - **Both items recovered:** item_errors 0 and failures "ingest none" on both jobs; 0 refusals, retries or give-ups.
  - A_R21: 2 Sonnet calls, 98k in and 3k out, as estimated. B_R21: 2 Sonnet calls, 38k in and 5k out.
  - This confirms the cause and RI-187's resume.
  - **The glue, not the package:**
    - The first resume attempt made no model call: the glue read the old run file before loading build g's package. The FDE fixed its glue.
    - A_R21's job read to the end, then timed out writing its folder back. Its job output kept the old state. The store's last snapshot, taken after the read, held the finished folder, and a no-model step unpacked it. This is RI-203's case.

---

## 5. The question-list step (RI-101)

**What came back:**
- 4 calls on arm A's R15 and R30 records, 166k in and 34k out. That is over the estimate (0.10-0.13 M and 0.02-0.03 M) and under the step's 0.2 M line.
- R15's main answer was not taken; R30's was.
- The run files and answer sheets are private and stayed in Foundry.

**What the error shows** (from the error text, the private list v3 and `engine._validate` at 0fa2ed5):
- **Every main answer was the wrong shape.** The main call asks 64 questions; R15's screening closes no rows. The error names the list's first three main keys "and 61 more": 64 errors, one per key, so no key held an answer object.
- **Not a wrapper.** The engine unwraps a single wrapper key (RI-80), and a missed wrapper would read "Field required".
- **Not nulls.** The engine's retry without nulls would also read "Field required".
- **Not JSON text of the objects.** The engine reads a string that starts with `{` or `[` as JSON (RI-192).
- **That leaves two shapes, both strings or bare values:**
  - each key holding its bare value, without `status`, `pdf_pages` or `note`; or
  - each key holding the model's parameter markup, as in section 4.
- **The run cannot tell them apart.** The engine keeps the raw answer on its cost row (`answer_args`), but `question_list._ask` copies only the stop reason and token counts into `run.calls`, and that is all the run file keeps.
- **One thing that would settle it:** the engine's warning line in the question-list build log. It names any keys it dropped, and dropped `status` and `pdf_pages` keys would mean markup spill.
- **Settled after the hand-back (the FDE sent the build log's lines):** the engine's only line in that window is the "not taken" warning above, with no dropped-key line. So the main answer held **bare values**, not markup. RI-195 sends that shape to the re-ask; it cannot be repaired without guessing each answer's status and pages.

**What it cost:** the R15 sheet lost all 64 model answers. The rows answered by both keep code's reading only. `_ask` makes one call, and nothing re-asks.

**RI-101's open question is answered for this route.**
- On Claude through Foundry, the schema (about 58k characters, 64 questions) is **accepted**: no refusal, no request error.
- It is **not enforced**: this route sends tools without strict mode.
- So the risk is the answer's shape, and it happened on one of two main calls.
- RI-195 covers it: repair, re-ask once, and keep the raw answer.

---

## 6. Record gaps

- **`calls_by_step`** (run.json and RESULTS) leaves out arm A's page describer and range labeller: 203 of R15's 230 calls, 559 of R30's 713 and about 735 of R21's 908. Their counts are only in `labels.cost` (RI-202).
- **The question-list run file** keeps no raw answer and no engine notes (RI-195).
- **A log group reader's reason for leaving a page** ("another log") stays in the item's unresolved list and never reaches the record's QA. The model-first wording "did not get to this page within its model-call budget" is wrong (RI-197).
- **Pages set aside inside a group** (R30's seven calc sub-dividers) leave no entry (RI-197).
- **The package's `cost.retried`** reads 0 on every job though the glue re-sent 11 calls. The glue retries below the engine, and `*_build.json` has the count.
- **Page groups and why each opened** are not saved, so R21's lost pits had to be rebuilt from the descriptions. Worth adding only if it is cheap.
- **Good news:** refused answers' raw arguments are now kept (`answer_args`). That is what showed the markup, which build f could not show.

---

## 7. Fix list (priority order; the lead files RI-195 onward in this order)

Each changes a saved report's output or what a live run returns.

| ID | fix | evidence | section | changes output |
|---|---|---|---|---|
| **RI-195** | **Repair an answer whose values carry the model's parameter markup, and ask again once when an answer is refused.**<br>(a) The engine (`_validate`) splits `</field>` + `<parameter name="next">` back into fields, and `<parameter name="0">` plus numbered top-level keys back into a list.<br>(b) Where the repair does not validate, ask again once with the validation error, as today's calc reader does. The callers are model first's `_Caller.read`, today's log reader's first call and `question_list._ask`.<br>(c) `question_list` keeps each call's engine notes and raw answer in `run.calls`. | Both failed items of the wave (R21: A bound calc pp453-468, B item_47 p81). Four descriptions accepted with a swallowed field (R21 p372; R30 p124, p314, p359). R15's question-list main answer refused whole, losing 64 rows. Build f's six calc refusals are most likely the same. | 4, 5 | yes: about 2 failed items a wave, wrong description fields that feed log grouping, and a whole question-list sheet |
| **RI-196** | **Model first: a describer `exploration` field with a note after the name still names the page.** `exploration_id` reads the name before a comma or semicolon when what follows is a note, and the describer prompt keeps the field to the name. | R21 bound: 12 of 36 test pits merged into the log before and lost. | 3.3 | yes: 12 investigations |
| **RI-197** | **Model first: a page no reader read is a failed item, never silent.**<br>(a) Pages a log group's reader says are other logs, or returns in `pages_left`, go back as their own groups.<br>(b) A content page still unread is a `skipped` "could not be read" entry with the reader's reason: an item error, re-asked on resume, not "did not get to this page within its model-call budget".<br>(c) The missing-holes check (RI-144) also reads the describer's names on scanned pages.<br>(d) `is_divider` matches only a description that says the page is a divider.<br>(e) A divider set aside inside a group gets a note, not a failure. | R21: the same 12 pages, where only 2 got a QA entry, both worded wrongly, and none got an item error; p45 set aside, one investigation missing. R30: 7 calc sub-dividers with no entry. | 3.2, 3.3 | yes |
| **RI-198** | **Model-first logs carry today's rule that the bucket is the width** (`pit.width`; today's paragraph is in `log_reader.py` ~l. 570). | R30: A writes a width on 0 of 52 pits, B on 45 of 52. R30_p152 A 17/18; R15_p68 A 15/16. | 2, 3.1, 3.2 | yes: every pit whose equipment names a bucket |
| **RI-199** | **A log page with no depth ruler is not judged against the other sheets' window** (`_depth_window`, log_reader.py:845; model_first_read.py:741). Judge a depth on its own page's ruler, else against 0 to the hole's printed total depth. | R15, one boring: B drops 5 samples and 2 SPT records; A writes 19 false `out_of_range` entries and lowers 19 good values to confidence 0.5. | 3.1 | yes, both arms |
| **RI-200** | **The model-first narrative gets today's whole-report search for the seven appendix-answered questions** (siteClass, asceSevenVersion, seismicCodeUsed, soilCorrosion, bearingCapacity, liquefactionPotential, reportDate). Waits on owner question 3. | R30: `asceSevenVersion` null on A; printed only on p555, which B found. | 3.2 | yes |
| **RI-201** | **RI-184 does not flag an excerpted attachment as an incomplete copy.** For example, photo pages' "Page X of Y" runs go unchecked, as calculation pages' already do. Check that build f's real cases (R05, R28) still flag. | R30: six false open `partial` entries on a complete report, both arms. | 3.2 | yes (QA) |
| **RI-202** | **`calls_by_step` counts arm A's page describer and range labeller.** | About 1,500 of A's 1,851 calls are missing from the step table. | 6 | yes (RESULTS, run.json) |
| **RI-203** | **Next brief, glue (FDE side, not package code): each job restores its folder from the store at start,** or the glue passes a store adapter as `sharepoint=` so the package restores itself. | §4's third requirement is unmet; not exercised this wave. | 1.3 | what a live run returns after a crash |

---

## 8. Verdict: arm A stays the default

**On the evidence**, A reads better or level on every stage with truth, across 1,487 pages:
- logs 105/108 against 99/108;
- lab 75/75 against 71/75;
- test pits 43/45 against 42/45;
- narrative 62/63 against 61/63;
- labels 0.983 with 21 harmful (all of them hand-label questions) against 0.962 with 40;
- calc level (32/37 against 33/37, where B's edge is the scorer).

**Failures:** one in each arm, from the same cause, which hits both arms.

**Cost:**
- A's readers use 45% fewer tokens in than today's, on Opus and Sonnet.
- The page describer adds 10.3 M on Haiku, so A's total input is 17% higher.
- Output is 8% lower, and the three reports finished in 0.6 h less run time in all.

**Before the next wave:**
- A's real defects are model-first only, and sit where there is no truth: 12 lost bound test pits (RI-196, RI-197) and no pit widths (RI-198). Fix those three and RI-195 first.
- Today's readers keep their own losses (section 9), and they are off by default since build g.

---

## 9. Noted, not fixed (one line each)

- The range labeller does not apply `max_request_tokens_in`: requests up to 121.9k tokens in were accepted by the route, and none failed.
- The page describer writes each page into the prompt cache, where it is never read: about 4.5k cache-write tokens a page, on Haiku.
- Each B job's triage reads A's prompt cache across jobs (same prompt, same report).
- The calc scorer's digit agreement trips on printed ratios inside labels: 4 of B's misses on R30_p542. Scorer only.
- A named a program for one R30 printout where none is printed, and shortened one result block's name.
- A records a pit's no-water note as two water readings with no depth (R15); B records not encountered.
- A names three bound R21 pits by their number alone, as the model read them, and kinds three second records `other`.
- A's R21 bound labels leave a photo report bound inside the bound report as `appended_report`. No reader reads it either way.
- 13 empty describer answers on R21 were recovered by the engine's text fallback.
- RESULTS.md leaves a failed report out of every number and then prints "No hand truth"; run.json has the numbers.
- B's `build.json` prints model first `on` against `arm.json`'s `off`; the glue reads the environment first.
- The FDE's range-convention probe looked for a module attribute. Read `run.json`'s `ranges.conventions` instead.
- `max_total_tokens` prints an in-plus-out ceiling.
- **Today's readers only** (arm B, off by default since build g):
  - the depths on three R30 site logs are refused wholesale;
  - the narrative hands a 0 caption count to the model (R30 `tableCount`, `figureCount`; R21 `figureCount`);
  - a stale "over the word limit, kept as written" QA row survives the summary call that fixed it (R15, `_apply_summaries`);
  - the bearing-value dedupe keeps the first structure's wording (R30);
  - the label review broke one sub-divider and rejected 13 unlooked changes (R30);
  - summary tables are built from header cells (R30);
  - CBR swells are recorded as separate tests (R30);
  - the chemical samples are repeated (R15);
  - a density sheet is read as one summary table (R15_p184);
  - elevation bottoms are dropped after a cut-off (R15);
  - 7 spurious `direct_shear` tests are kept though the model flagged them (R21 bound);
  - 492 bound-page `label_disagreement` entries and 85 QA rows citing page 0 (R21);
  - the log reader raises on its first refused call (part of RI-195).

---

## 10. Owner questions

| # | report, pages | the choice | what changes either way | recommendation |
|---|---|---|---|---|
| 1 | R30, p35 | `soilCorrosion` "no" or "mixed", when the main strata test non-corrosive and one fill sample tests corrosive | "no": A is right on R30. "mixed": B is right. Either way it becomes the rule for both readers; it is the same pattern as build f's open R36 question | **"mixed"**: the owner's v3 question list added "mixed" to its pick list (2026-10-29 rulings) |
| 2 | R30, pp4, 32-34 | `earthHazardsExposed`: does a slope-stability concern that the report analyses count as landslide? | yes: A is right (it adds landslide). No: B is right | **yes**, when the report calls slope stability a concern for the site; not for a routine design check of a cut slope |
| 3 | R30, p555 (body pp0-56 names no edition) | `asceSevenVersion`: does the edition an appendix calculation uses answer it when the body names none? | yes: B is right and RI-200 is needed on model first. No: A's null is right, and today's reader over-reads | **yes**: the report's own seismic calculation used that edition. Cite the calc page |
| 4 | R30, pp418, 419, 471-473 | specimen photo pages inside the lab appendix: `lab_test` (conventions B rule 1, photos inside a section take its label) or photos (the hand labels) | `lab_test`: A's R30 harmful labels go from 6 to 1, and the hand labels join RI-104's list. Photos: the range labeller needs an exception. No values are lost either way | **`lab_test`** |
| 5 | R21, pp714-728 | the hand labels say calculation; the pages are a laboratory chemistry report that both arms read as lab tests | `lab_test`: A's harmful labels go from 15 to 0 and B's from 24 to 9. Calculation: both arms misroute 15 pages to the lab reader | **`lab_test`**: a hand-label slip; add it to RI-104's list |
| 6 | R15, p67 | the hand label says test-pit log; the page is a boring log's last sheet | boring log: A's R15 labels go from 0.985 to 0.990, and B gains the page too | **boring log**: a hand-label slip; add it to RI-104's list |
| 7 | R15, pp4, 22-24 | `recommendedFoundations`: the answer sheet's two items, one naming two foundation types, or three items, one per type as RI-181 rules | split: both arms 31/31. Keep: both stay 30/31 and the sheet disagrees with RI-181 | **split**: the sheet predates RI-181 and is Claude's transcription (RI-72) |
