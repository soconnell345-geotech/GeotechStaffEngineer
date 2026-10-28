# Trace review: Foundry build e, MF-0 and MF-1 (model-first vs today's pipeline on Claude)

Reviewed 2026-10-27 with no model calls. This file is PUBLIC: report and item IDs, field names, counts and scores only. No hole names, depths, values, firm, person or post names, page text, Foundry dataset names, resource IDs, branch names or usernames. Detail that needs any of those is in the gitignored `raw/TRACE_NOTES_private.md`.

**What ran** (plan of record `module_work/FOUNDRY_BRIEF_MODEL_FIRST.md`):
- Build e: app `5.33.0+foundry.20261027.1` (master `ea373e8`), planlens `0.13.0+main.a221a7e`, references `1.4.1+main.ceeef53`; Python 3.12, numpy 1.26.
- MF-0: wheel checks, the PDF-route probe and the pictures probe on Opus 5.5, Sonnet 5.5 and Haiku 5.5, a usage probe.
- MF-1: twelve arms, each stage off ("today": floor + model reader) and on ("first": the model answers each item's pages against the answer schema), on Claude through Foundry's `palantir_models` chat class, pictures route, `checks='flag'`, instructions `mf-2`. A two-item pilot per arm, then the full build. Arms ran one after another, 18:02 to 20:20 UTC (138.5 min).

**What was read:**
- The FDE's hand-back note, the results tables (summary, per-call meter, usage probe), every arm's `RESULTS.md`, `results.json`, `foundry_logs/*_build.json`, `*_stdout.txt` and `*_retries.jsonl`, and every run file (logs 15 x 4, lab 31 x 2, calc 10 x 4, narrative 8 x 2), with their durable twins.
- The FDE's glue code (meter, arm settings, run loop), read, never run.
- Every saved arm re-scored with today's scorers (`rescore_saved`, $0) beside the 2026-10-24 local Claude runs, the plain-Opus baseline (RI-86) and wave 1's GPT-5.4 runs (`2026-10-24_foundry_brief_1023`).

Model-first run files keep, per call, the pages sent, the stop reason and whether an answer came back, but not the prompt or the model's text. Today's errored items keep no partial record. So "why" below is read from stop reasons, saved answers and the code (noted in section 7).

---

## 1. MF-0: the route

| check | result |
|---|---|
| wheels, sha256, imports | ok |
| PDF route (document block) | refused client-side on all three models: the class carries text and images only |
| pictures route | read the probe's code word on Opus, Sonnet and Haiku (0.9 to 1.8 s) |
| usage | `input_tokens` is 4 a call; the rest of the input comes back as cache write and cache read (about 7,530 on the probe). The class's own `usage_metadata.input_tokens` gives the whole 7,534 |
| repeat call, cache | Haiku and Sonnet read the cache on the repeat. Opus wrote it twice: the two calls went to different backends (Bedrock, then Vertex) |

So D1 = Sonnet vision, D2 = Haiku describes, and every model-first call goes as pictures. Section 5 is what the usage split does to the package's brakes.

---

## 2. Plumbing: the "today" arms were broken by one argument

The FDE reported "12/12 ok, 0 refusals". The builds were all `ok`, but **47 items failed**:

| arm | items | failed | error |
|---|---:|---:|---|
| logs today, Sonnet | 15 | 14 | `TypeError: ...ClaudeChatToolResultContent.__init__() missing 1 required positional argument: 'is_error'` |
| logs today, Opus | 15 | 15 | same |
| lab today, Opus | 31 | 9 | same |
| calc today, Sonnet | 10 | 3 | same |
| calc today, Opus | 10 | 5 | same |
| narrative today, Opus | 8 | 0 | |
| narrative first, Opus | 8 | 1 (R28) | `ValidationError ... GeneralFacts outsideProject` (section 3.4) |
| the five other first arms | 79 | 0 | |

**Cause.** `webapp/palantir_sdk_engine.py` builds a tool result in two places (`_claude_tool_result`, about line 1224; `_marked_block`, about line 1333) and passes `is_error` only when the result is an error. Foundry's SDK class takes it as a required argument. So every reader turn that follows a tool call fails before it is sent:

- In every failed item, call 1 is a reply with stop `tool_calls` (a zoom or a look) and call 2 is the error, at 0.0 s, classed `other`, never retried.
- The items that succeeded are the ones where the reader called the answer tool on its first turn without looking. **The today arms' partial scores are on a self-selected easier subset**, so the shared-item comparisons in section 3 are weak evidence; the 2026-10-24 local runs (same models, Anthropic API, full sets) are the fair "today".
- The error happens before the request reaches the FDE's meter, so the per-call meter shows 0 errors.
- It is unchanged on master `7126fd7`. **The same break waits for MF-2's off arm** (the label review is a tool loop and the readers zoom) **and for the GSE agent on Foundry's Claude SDK path** (any tool message).
- The fix is one line at each site: always pass `is_error=bool(...)`, False by default. The offline fake (`webapp/tests/test_palantir_claude_engine.py`, the `_ToolResult` stand-in) gives `is_error` a default, which is why no test caught it; the fake should require it, as the SDK does.

**Why nobody stopped.** The pilot's own printed log already showed both today-log pilot items (and both calc-today-Opus ones) as `ERROR` with "0 model calls". The plan's stop points say to check each pilot run file for "no error" and to stop an arm when two items fail with the same error (§5, points 2 and 5). The glue reads the arm's status only. `results.json` is no help either: its top-level `totals.failed` reads **0** while the stage's own `logs.cost.failed` reads 15.

**The rest of the plumbing was clean:**

| | |
|---|---|
| calls (full build, meter) | 191: Sonnet 45, Opus 146; the pilot builds' 31 calls are on top |
| retries / gave up / throttle wait | 3 (all `connection`: logs first Opus, calc today Sonnet, calc today Opus) / 0 / 0 s |
| rate-limit errors | 0 |
| fastest / slowest call | 3.3 s / 104.6 s; no call under 0.5 s, no repeated in/out signature, so no cached answers |
| largest request | 99,153 tokens in (narrative first), under the 100k limit |
| peak 60 s | 5 calls overall; Opus 5 calls, 178,828 tokens in, 20,480 out; Sonnet 4 calls, 86,565 in |
| resume | worked: failed pilot items were retried in the full build |

---

## 3. Per stage

Scores are re-scored with today's scorers. "Local" is the 2026-10-24 Claude runs on the Anthropic API.

### Verdict

| stage | model-first (Foundry) | today, fair comparison | verdict |
|---|---|---|---|
| logs /521 (15 logs) | Sonnet 494, Opus 494 | local Sonnet 475, Opus 478; GPT-5.4 478; plain Opus (PDF) 454 | **first ahead, +16 to +19**, but +12 of it is header fields only model-first asks for; reading about +4 to +7, inside noise; Sonnet = Opus |
| lab /826 (31 sheets, Opus) | 752 | local Opus 773 (its repeat on 11 sheets 186/234); local Sonnet 746; plain Opus 756 | **level**, apart from one instruction gap (-31 values, below) |
| calc /203 (10 items) | Sonnet 176, Opus 176 | local Sonnet 172, Opus 178; GPT-5.4 184; plain Opus 176 | **level** |
| narrative recall | 140/203 on 7 reports (R28 lost) | Foundry today 194/235; on the same 7 reports 173 | **today clearly ahead, -33** (about 4.7 fields a report against a noise of about 1 in 28) |

Foundry today, where it ran: logs void (Sonnet 14/14 on the one log that finished); lab 578/602 on 22 sheets against first's 574 on the same 22; calc Sonnet 118 against first's 121 (7 items), Opus 94 against 94 (5 items).

### 3.1 Logs

| arm | open /307 | blind /214 | all /521 | placed on the page |
|---|---:|---:|---:|---:|
| first, Sonnet | 297 | 197 | 494 | 219/221 |
| first, Opus | 296 | 198 | 494 | 215/218 |
| local today, Sonnet | 295 | 180 | 475 | |
| local today, Opus | 296 | 182 | 478 | |
| GPT-5.4 (wave 1) | 295 | 183 | 478 | |
| plain Opus, PDF | 279 | 175 | 454 | |

Where first gains over local today:
- **header fields +12**: project and project number on R02, R03, R04, R21, R30 and R34. Only the model-first schema asks for them (RI-82's added fields), so this is schema, not reading.
- **water +7** (15/17 against 8/17), mostly R15_p46: groundwater rows whose unit is printed against the number (the item scores 64/64 against 58/64). Plain Opus also gets 15/17, so this is today's floor and merge losing what the model reads.
- **layer tops +2 to +4.**

Where it loses:
- **USCS -5 / -6** on R07_p30. The log prints soil codes that are not USCS symbols. Model-first leaves `uscs` blank, as its instruction says ("the USCS symbol where one is printed"); today's reader and the answer sheet copy the code. This is a convention, not a misreading.

Level in every arm: R25_p19 6/14 (no ruler the scan can use) and R37_p26 42/46 (the known answer-sheet suspect). One call per log.

### 3.2 Lab

- First 752/826. Link 76/79, kind 76/79.
- Against local today (773), **-21 overall is one instruction gap**. Today's lab prompt says a grading curve with Atterberg limits on the same sheet is TWO entries (`report_ingest/lab_reader.py`, about line 806). The model-first instruction does not, so the limits on gradation sheets were not returned: 31 values on R17_p114 (-13), R28_p163 (-15) and R15_p82 (-3). Without that gap first is about +10, which is inside noise.
- Gains: R06_p67 +4, R27_p85 +4, R15_p184 +4. Loss: R17_p155 0/6 (kind read as `other`; local today got 3/6, the link only).
- **Placement 41/59 (69.5 %), below the plan's 90 % gate.** The count covers only about two values a sheet (the sheet's title and link); R27_p45's 17 values are placed at page level only. So the gate number is measuring very little on this stage (section 6).

### 3.3 Calc

- First 176 on both models. Local today 172 (Sonnet) and 178 (Opus); with today's merge re-run 175 and 181; the model alone 176 and 182. GPT-5.4 184. Placement 263/276 (Sonnet), 277/286 (Opus).
- Level. Item-level notes: R20_p344, both first arms record 12 inputs and no results (Opus 9/22, Sonnet 16/22 by credit on misplaced labels); R30_p542 Sonnet 18/26, a label variance; R23_p371 Claude 20 to 21 in every arm against GPT-5.4's 29.

### 3.4 Narrative

| report | set | first | today (Foundry) |
|---|---|---:|---:|
| R05 | open | 25/27 (1 call) | 22/27 |
| R06 | blind | 15/29 (5 calls) | 22/29 |
| R15 | blind | 23/31 | 29/31 |
| R21 | blind | 26/33 | 28/33 |
| R26 | blind | 17/28 | 22/28 |
| R28 | blind | ERROR | 21/32 |
| R36 | open | 18/28 | 26/28 |
| R37 | blind | 16/27 | 24/27 |
| **7 reports** | | **140/203** | **173/203** |

- Precision: first 138/186, today 193/231. List-item precision 49 % against 73 %. Placement 393/416 (94.5 %). Local today Opus 196/235 and Sonnet 194/235, GPT-5.4 189/235.
- **R05 is the one report that fits one call, and there first is ahead (+3).** Everywhere it split, first is behind. That is the decisive reading.

Causes, from the code and the saved answers:
1. **The pictures route splits a report.** `MAX_PICTURES_PER_CALL = 20` (`report_ingest/model_first_pages.py`, line 107) cuts a narrative of about 50 pages into 3 or 4 parts of 16 to 19 pages, about 78k to 99k tokens each. Today's builder joins them: the first part's answer is kept and lists are joined, then a summary call. Each part sees only part of the report.
2. **documentType "partial report"** in 5 of the 6 split reports: the part calls answer about the part they were shown. Four score wrong; R26's scores right only because its answer sheet lists that answer as an alternate.
3. **Counts null instead of 0**: cptCount missed 5, testPitCount 4 (today: 1 and 0). The model-first instruction lacks today's conventions (`narrative_glossary.py`: a count of explorations the report did not do is 0).
4. **propertyType** wrong 3 (today 0): free text instead of the vocabulary.
5. **List fields**: boringDictionary wrong 4 (today 0); items annotated or unioned across parts. earthHazardsExposed invented 4 (today 0), soilCorrosion invented 2 (today 0).
6. **R26**: the first part (17 pages) came back with stop `stop` and no structured answer, so those pages counted for nothing.
7. **R28 lost whole**: `ValidationError: GeneralFacts outsideProject Input should be 'yes', 'no' or 'unclear' [input_value='mixed']`. The prompt's vocabulary for `outsideProject` is `YES_NO_UNCLEAR` (`report_ingest/narrative_reader.py` line 165; `model.py` line 2359 includes "mixed") and the field is a CLOSED one, but `GeneralFacts.outsideProject` is `Literal["yes", "no", "unclear"]` (`model.py` line 2537). One answer the prompt invites throws the report's whole narrative away. Today's path uses the same builder, so the defect is latent there too.

Cost: first used 1.67 M tokens in, today 0.48 M (3.5 times), over the plan's 0.5 to 1 M for this arm.

---

## 4. Spend (per-call meter, full build)

| arm | calls | tokens in | out |
|---|---:|---:|---:|
| logs today, Sonnet | 15 | 217,658 | 64,532 |
| logs today, Opus | 15 | 217,658 | 61,409 |
| logs first, Sonnet | 13 | 152,237 | 55,605 |
| logs first, Opus | 13 | 152,237 | 56,130 |
| lab today, Opus | 29 | 534,102 | 63,038 |
| lab first, Opus | 29 | 474,665 | 84,501 |
| calc today, Sonnet | 9 | 187,574 | 24,986 |
| calc today, Opus | 10 | 202,084 | 37,705 |
| calc first, Sonnet | 8 | 182,299 | 22,191 |
| calc first, Opus | 8 | 182,299 | 21,975 |
| narrative today, Opus | 18 | 483,519 | 68,031 |
| narrative first, Opus | 24 | 1,669,004 | 95,122 |
| **total** | **191** | **4,655,336** | **655,225** |

- "Tokens in" = plain input 764 + cache write 3,693,365 + cache read 961,207. By model: Sonnet 45 calls, 739,768 in, 167,314 out; Opus 146 calls, 3,915,568 in, 487,911 out.
- Against plan (about 240 calls, 4.5 to 5.0 M in, 0.9 M out): input on plan, output lower, because the today arms stopped at their first tool round (the plan expected 29 to 37 calls for each today log and lab arm).
- **Spent for nothing:** the 47 failed items cost 50 calls, about 1.10 M tokens in and 0.17 M out (run-file figures), about a quarter of the input.
- Dollars: the hand-back carries tokens only; read the dollars off Foundry's own usage report for the window.

---

## 5. The token brakes do not see Foundry's Claude input (RI-171 confirmed)

`report_ingest/engine.py` `_usage_of` (about lines 2987 to 3022) sets `input_tokens` to the class's input minus cache read minus cache write, which is about 4 a call on this route, and keeps the two cache counts in their own fields. Every brake reads `input_tokens` alone:

- `cluster_scoring._tokens_of` (input + output) behind `max_total_tokens`;
- `_TokenBrake.check` (the report limit and the run's total);
- `_attempts_in` and the per-report `report_in` (line 639 on).

The RESULTS cost line (`CostMeter.summary`) prints "(+N the provider cached)" for the cache reads and leaves the cache writes out. Example: the logs-today arm printed "60 input tokens (+51,828 the provider cached)" for 217,658 tokens in.

On narrative first the brakes saw about 0.10 M of the 1.76 M the arm spent (in + out). The Anthropic-API `ClaudeEngine` (about line 1645) splits usage the same way.

**The brief's own MF-2 stop check has the same blind spot.** It reads `all_attempts.input_tokens` against the per-report lines (plan §5, point 3), and on this route that field is about 4 a call. Until the fix lands, that check must read input + cache read + cache write, or the FDE's meter.

**Smallest fix:** a tokens-in total (input + cache read + cache write) on `CostMeter` and in the cost dict, used by `_tokens_of`, `_TokenBrake.check`, `_attempts_in` / `report_in` and the RESULTS line. Keep the three fields apart in the record.

---

## 6. The FDE's work

- Arm settings match the plan: models per stage, `model_first` off/on, page route auto (resolved to pictures), `max_request_tokens_in` 100000, the lab label-link switch on, budgets (log 6, lab 4, calc 2, narrative 8, front 50), `max_total_tokens` at 2.5 times the estimate (at least 0.75 M), one `out_dir` per arm, `checks='flag'` recorded, prompt `mf-2`.
- The meter counts input + cache write + cache read + output off the raw response, which is right. It wraps the retrying handle, so it writes one row per successful logical call and never sees a client-side failure. That is fine for spend, but blind to failed items.
- Noted, not fixed: the meter's ceiling resets per build, not per arm folder; `durable_dir` sits inside the same temporary root as `out_dir`, so only the dataset write-back actually persists.
- The hand-back's own note names a Foundry resource ID and a branch with a username. Neither is copied here.

---

## 7. Noted, not fixed (one line each)

- Model-first confidence is constant by method (0.8 for a picture read, 0.9 from the text layer, one 0.5 or 0.6 a stage), so it cannot steer a review.
- Model-first run files keep no prompt or model text, and today's errored items keep no partial record. The record cannot say why R26's first part returned nothing.
- Lab placement counts about two values a sheet, so its 90 % gate measures little on that stage.
- R07_p30's `uscs` convention (blank where the log prints non-USCS codes) is the instruction's reading, and differs from the answer sheet's.
- Opus calls go to either Bedrock or Vertex, so a repeat can write the cache twice (cost only).
- R25_p19, R37_p26, R17_p155 and R20_p344 as in section 3.

---

## 8. Recommendations (ranked)

1. **Fix `is_error` in both tool-result builders** in `webapp/palantir_sdk_engine.py`, make the test fake require it, then **re-run MF-1's today arms on Foundry** (logs Sonnet and Opus, lab Opus, calc Sonnet and Opus) before MF-2. This also unblocks the GSE agent's tool round trip on Foundry's Claude SDK path.
2. **Brakes count cache tokens (RI-171)** before any further Foundry run, and the brief's MF-2 token check reads the same total (section 5).
3. **The glue and the brief stop on item errors, not arm status.** After the pilot, read each arm's failed-item count and stop on two of the same error, as the plan says. Make `results.json`'s top-level `totals.failed` carry the stages' failures.
4. **Narrative model-first, before its repeat:**
   - one call per report where it fits: pages with a reliable text layer go as text, pictures only where needed, so a report no longer splits into 3 or 4 picture parts;
   - the instruction carries today's conventions (counts 0, vocabulary values, list-item form);
   - the "mixed" vs `Literal` mismatch on `outsideProject` is fixed on both paths.
5. **The model-first lab instruction** says Atterberg limits on a gradation sheet are a separate entry (31 values on 3 sheets); then check lab placement against its gate.

**Repeats:** logs, lab and calc today on Foundry are needed, not optional: today's Foundry arms are void or a self-selected subset. Narrative first again after item 4. Logs, lab and calc first need no repeat for the method decision: logs first is ahead and lab and calc first are level with today's local runs, on both models where both ran.

## 9. Follow-ups for BACKLOG (the lead files them; IDs are not assigned here)

- Recommendations 1, 3, 4 and 5 (2 is RI-171).
- The `outsideProject` vocabulary/`Literal` mismatch is a defect on today's path as well (section 3.4, cause 7). It changes a saved report's output: R28's narrative is lost whole.

---

## 10. MF-2 partial (R05, R36)

**What ran:** build e, the `ingest` stage over R05 (15 pages) and R36 (94 pages), off and on, 22:40 to 23:12 UTC. Both arms stopped after R36 at the plan's stop point (two items failed with the same `is_error` error) and wrote `STOP.json`. This time the glue stopped where it should: it now reads `run.json`'s `item_errors` and `qa.json`'s "could not be read" entries.

**What was read:** the updated hand-back note, the three results tables, both arms' `RESULTS.md`, `results.json`, build and stop records, and for each report `run.json`, `triage.json`, `labels.json`, `qa.json`, the record, every item file, and on the on arm every page description (109), `ranges-plain.json` and `placement.json`. Both arms' records were re-scored with today's scorers (`_score_ingest_record`, $0). The scores matched the saved ones.

### The off arm

The off arm is void where `is_error` struck: 14 items on R36 (logs, CPT, lab, calc). **Its label review also failed on `is_error`, on both reports.** That shows only as a `partial` QA entry ("the label review did not finish"), not as an item error. So the glue counts R05's off run as finished, and a resume will not ask for its review again. Build f's off arm should start R05 and R36 afresh.

### The on arm, whole report

| | R05 | R36 |
|---|---|---|
| describer (Haiku) | 15/15 pages, 0 failed | 94/94 pages, 0 failed |
| range labels vs hand labels | 5/5 | 5/5 |
| pages split with the rules | 1 | 25 (no review on this arm; each is a QA entry, confidence 0.0) |
| triage `bound_together` | none | none |
| units read | narrative 1 (one call) | narrative 1 (3 picture parts + a joining call), logs 11 (one per hole), lab 2, calc 2; CPT 2 by today's sounding reader |
| holes printed / in the record | none | borings 11/11; CPT 0/2 (`is_error`) |
| placed | 17/17 | 291/309, of which 288 at a spot of the 308 on pages with text (93.5 %); placed by stage: narrative 58/62, logs 183/185, lab 16/17, calc 34/45 |
| logs vs truth | | R36_p38 44/44 |
| lab vs truth | | 5 sheets 46/48 |
| narrative recall / precision | 24/27, 22/27 (off 24/27, 24/27) | 18/28, 18/26 (off 24/28, 24/27) |
| could not be read | 0 | 2 (the CPT items, `is_error`) |

- **Where truth exists, the whole-report path reproduces MF-1's item-level scores exactly** (logs 44/44, lab 46/48, narrative R36 18/28). The describer's exploration names put every log page on its hole, and the labeller's sections found the same pages the hand-truthed items use.
- **Narrative R36 is MF-1's split again**: 20 `partial` entries say "part N and part M answer this differently; the first answer is kept", and parts report the document as "truncated" or "sections not among the pages provided". RI-175 (one call, text pages as text) is the fix.
- **Calc page groups ignore printout boundaries.** R36's calculation pages hold four printouts (a seismic design-maps printout of six sheets, two liquefaction analyses, one per CPT, and shaft capacity charts) with two appendix dividers between them. `calc_groups` saw no sheet-number start and no text-layer head change, so it made a balanced 14/13 cut. That gave two records, each a mix: one is recorded as site response, the other as an axial pile (a liquefaction analysis plus the shaft charts). The reader flagged the mix itself in its own `partial` entries. The describer's titles and kinds show every boundary (`model_first_pages.py`: `calc_starts` about line 898 reads sheet numbers only; `calc_groups` about line 930).
- **Appendix dividers are read as content.** The labelling conventions give a divider its section's label (`range_labels.py` line 146), and `model_first_pages` never sets one aside. So R36's lab divider was read alone, at one Opus call, and put an empty `other` lab test in the record. Its two calc dividers sit inside the calc groups.
- Soundings are not model-first: CPT items go to today's sounding reader on the main model (Opus), after the other readers (`graph.py` line 2965). With `is_error` fixed they will run.

### Spend against the lines (the FDE's meter: plain + cache write + cache read)

| arm | report | calls | tokens in | out | line | share of line |
|---|---|---:|---:|---:|---:|---:|
| off | R05 | 6 | 42,207 | 7,105 | 0.6 M | 7 % |
| off | R36 | 45 | 600,797 | 89,596 | 3.8 M | 16 % (14 items not read) |
| on | R05 | 18 | 197,648 | 14,030 | 0.33 M | 60 % |
| on | R36 | 119 | 1,372,835 | 144,810 | 2.1 M | 65 % |

R36's on-arm input by step:

| step | calls | tokens in | share |
|---|---:|---:|---:|
| describer | 94 | 649,366 | 47 % |
| narrative | 4 | 263,774 | 19 % |
| calc | 4 | 164,913 | 12 % |
| logs | 11 | 138,496 | 10 % |
| lab | 2 | 82,534 | 6 % |
| labeller | 1 | 37,634 | 3 % |
| CPT, first turns | 2 | 18,902 | 1 % |
| triage | 1 | 17,216 | 1 % |

- Per page that is 13.2k (R05) and 14.6k (R36), against the plan's estimate of about 15k a page. At R36's rate R37, R13, R16 and R28 would land at about 65 % of their lines.
- The describer alone is about 6.9k tokens in a page, so about 7.4 M of input over R13, R16 and R28.
- Largest request 90,381 tokens (a narrative part). No rate limits, 1 connection retry. R36 took 15.5 min, 4.8 of them describing.

### Noted, not fixed

- The run's `calls_by_step` leaves the describer's and labeller's calls "unattributed" (+16 on R05).
- Each arm ran its own triage. The shared review folder came back empty because the off arm's review never finished.
- The narrative's front 50 pages include the logs and CPT plots as pictures. RI-175's one-call narrative changes this.
- **For RI-175 on R13 (every page scanned):** the "text pages as text" rule has to count Document Intelligence text as a text layer, or R13's narrative still goes as picture parts.

## 11. Narrative misses by verdict (MF-1 and MF-2, re-scored at $0)

Every miss under the current strict scorer was read against its pages and given a verdict. The row-by-row file, with page evidence and a blank column for the owner, is private under `raw/`. Five more rows fail strictly but an alternate on the sheet accepts them, so they are not counted here. The lenient view passes 2 of the 68.

Runs:
- **A:** MF-1 today, Opus, 8 reports.
- **B:** MF-1 model-first, R05.
- **C:** MF-2 model-first, R05 and R36.

Build e's model-first instruction did not carry the house rules; build f adds them (RI-175).

| verdict | A (48) | B (4) | C (16) | all (68) |
|---|---:|---:|---:|---:|
| ours wrong | 16 | 2 | 7 | 25 |
| convention (a house rule decides) | 17 | 1 | 8 | 26 |
| sheet wrong | 6 | 0 | 0 | 6 |
| wording (same substance) | 8 | 1 | 1 | 10 |
| either fine | 1 | 0 | 0 | 1 |

| question | rows | ours wrong | convention | sheet wrong | wording | either fine |
|---|---:|---:|---:|---:|---:|---:|
| structures (count and list) | 13 | 4 | 9 | | | |
| recommended foundation types | 9 | 3 | | | 6 | |
| summaries over 100 words | 7 | 7 | | | | |
| earth hazards | 6 | 2 | 2 | 2 | | |
| allowable bearing pressure | 5 | 1 | | 1 | 3 | |
| figure count | 4 | 2 | 2 | | | |
| seismic hazard analysis done | 3 | 1 | 2 | | | |
| soil corrosivity | 3 | 2 | | 1 | | |
| project phase | 3 | | 3 | | | |
| test pit count | 2 | | 2 | | | |
| earlier investigations count | 2 | | 2 | | | |
| boring identifiers | 2 | 1 | 1 | | | |
| outside the estate | 2 | | 2 | | | |
| boring count | 1 | | 1 | | | |
| table count | 1 | 1 | | | | |
| test pit identifiers | 1 | | | 1 | | |
| kind of document | 1 | 1 | | | | |
| soil profile | 1 | | | | 1 | |
| ASCE 7 edition | 1 | | | 1 | | |
| seismic code used | 1 | | | | | 1 |
