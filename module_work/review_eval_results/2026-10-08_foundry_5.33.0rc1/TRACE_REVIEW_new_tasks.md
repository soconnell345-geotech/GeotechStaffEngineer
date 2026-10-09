# Foundry brief 5: the five new tasks, every run read in full

Brief 5 ran the test wheels **geotech-staff-engineer 5.33.0rc1** (app `39716a8`,
same code as master `fcc5c38`) and **planlens 0.13.0rc1** (`c5bdb8d`). This
section covers the five tasks new in this round:

* `report-extract-all`
* `report-extract-diggs`
* `report-summary-vs-sheets`
* `scale-log-depth`
* `scale-plan-distance`

That is 28 runs:

* **GPT-5.6 Sol, part C:** 5 baseline, 5 coverage, 3 checklist. The checklist arm ran only the three report tasks.
* **GPT-5.4, part D:** 5 baseline, 5 coverage, 5 checklist.

**Sources.** The raw hand-back is git-ignored:
`module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/`
(`part_c/sol/runs/<arm>/<task>/`, `part_d/gpt54/runs/<arm>/<task>/`). The fixtures
are public, built by `funhouse_agent/review_eval/report_fixture.py` and
`planlens.testing.visual_scale_fixtures`. Code is cited at `39716a8`, and
planlens at `c5bdb8d`.

**How the runs were read.**
* Every event of the 28 `activity.jsonl` files: 3,733 records, including 332 helper (sub-agent) model calls.
* Every vision side call's text, every tool argument and result, every final answer and every check verdict.
* The 18 `coverage.json` ledgers.
* The delivered files: every docx, csv and md was opened, and the "values" check was re-run on the answer and the files together.
* Two behaviours were reproduced offline:
  * the tool layer silently dropping an unknown `pages` argument;
  * `parse_pages` dropping dict-shaped page lists.

`Lnn` below is a line of that run's `activity.jsonl`.

---

## In short

1. **The DIGGS file was written and valid in 6 of 6 diggs runs, but always to `/tmp`.**
   * Every model followed the tool's own example path (`'/tmp/site.diggs.xml'`).
   * The suite runner collects only the run's `files/` folder, so `file_produced` failed 6 of 6.
   * In the app the file would have reached the user: the app's turn job copies a reported `output_path` into the conversation folder. The runner skips that step.
   * So this is mainly a scorer/harness gap (S). The writer's path handling and description are a plumbing flaw (P) that other `call_agent` writers share. One of them (DXF) would not be rescued even in the app.
2. **W2 cross-checks ran in all 6 diggs runs (8 of 8 `write_diggs` calls that wrote a file).** The planted LL/PL swap was caught in 3 of 3 Sol runs (pages 17 and 19, both values) and relayed in all 3 answers. It was caught in 0 of 3 GPT-5.4 runs:
   * Twice GPT-5.4 wrote the summary row with the sheet's values ("corrected" it), so there was nothing to disagree.
   * Once its summary was refused because the tool's kind list omits `summary_table`. It resent the data without the summary.
   * The checklist then reported "summary values match the sheets" as **pass** in that run.
3. **The "last sheets" values check is a scorer issue.** 43, 41.7, 1.94 and 1,850 are present in every delivered table: all 5 runs that wrote files, plus GPT-5.4 baseline's inline tables. The check reads only the answer text. "shell" is a soil description, and the question asks for none.
4. **GPT-5.4 and the 2011 scans had three different causes:**
   * **Baseline:** it chose to leave them out, and said so.
   * **Coverage:** it called `analyze_pdf_page(pages=12)`. The tool layer silently dropped the unknown `pages` key and looked at **page 0, the cover**, three times. The agent then marked the scans "skipped" with an invented reason.
   * **Checklist:** it read them all, but named them as ranges ("BH-1 to BH-3"), which the boring check cannot parse.
   * "PDF pages 8–11 not read" in GPT-5.4 diggs: it read those pages only through `log_grid` (all 120 rows), which neither the scorer nor the ledger counts.
5. **Coverage arms:**
   * **Coverage stated as counts:** 8 of 8 extraction answers in the coverage and checklist arms (baseline 0 of 4).
   * **The gate:** spoke in 10 of the 18 switched runs and never stood down. It cost 1–24 s each, plus one 69 s outlier from latency on a single model call.
   * **Step limit:** the 150-step extraction cap was in force, but no turn came near the ordinary 50. The longest run had 17 primary model calls.
   * **Answers after the gate:** **every gated answer (10 of 10) is the pre-gate answer glued to the post-gate answer with no separator**. Two GPT-5.4 answers repeat the whole finding twice.
6. **The checklist added little, and its code checks went wrong three ways:**
   * GPT-5.4 never called `report_checklist` (0 of 5).
   * Sol called it 3 of 3 but never reported the judge items.
   * **A false fail:** "sheets not on the summary" for test kinds the summary does not carry.
   * **A false pass:** see point 2.
   * **A stuck "unsure":** the 2011 logs stayed "unsure" even after they were looked at and written.
   * On the Document Review page its consistency items cannot run at all, because they need `write_diggs`.
7. **Sol's `scale-plan-distance` failure in the coverage arm is not about coverage.**
   * No gate fired and no coverage tool was called.
   * `measure` returned the same 183.85 ± 0.12 ft with the same "stated scale disagrees with the bar" note.
   * The answer simply did not repeat that note (M: run variance).
   * Separately, 3 of the 5 plan runs lost a call: `measure`'s `to` takes only a box on the first view, not a second zoom's view + box.
8. **`scale-log-depth`: GPT-5.4 never called `measure` or `log_grid` (0 of 3).**
   * It took one look, whose answer said "starts at exactly 4.0 m", and answered 4.00, 3.90 and 3.80.
   * The tools are findable from their descriptions: GPT-5.4 used `measure` 3 of 3 on the plan task. So this is a model habit (M).
   * Sol called `log_grid` 2 of 2 and answered 3.60 (truth 3.62).
9. **Layers: Sol's `n_lithology_intervals: 0` is right.**
   * The fixture's logs are tables of sample rows with no stratum lines.
   * `log_grid` returned 24, 8 and 9 "layers" on those vector logs. They are the table's row rules, read as stratum lines (a planlens P).
   * GPT-5.4 wrote 25–29 layers of its own, built from sample depths. The writer asks for values "AS PRINTED (never inferred)", so this is M.

---

## 1. The 28 runs at a glance

Score is checks passed of the scored checks. Time is the turn's seconds. Tokens are input tokens, helpers included.

| model | task | baseline | coverage | checklist |
|---|---|---|---|---|
| Sol | report-extract-all | 4/6 · 202 s · 0.71 M | 5/6 · 219 s · 1.08 M | 5/6 · 234 s · 1.02 M |
| Sol | report-extract-diggs | 4/6 · 157 s · 0.77 M | 5/6 · 449 s · 1.68 M | 5/6 · 244 s · 0.88 M |
| Sol | report-summary-vs-sheets | 5/5 · 62 s · 0.25 M | 5/5 · 129 s · 0.47 M | 5/5 · 80 s · 0.38 M |
| Sol | scale-log-depth | 3/3 · 69 s | 3/3 · 51 s | — |
| Sol | scale-plan-distance | 4/4 · 52 s | **3/4** · 46 s | — |
| GPT-5.4 | report-extract-all | 2/6 · 127 s · 0.26 M | 3/6 · 122 s · 0.25 M | 4/6 · 169 s · 0.33 M |
| GPT-5.4 | report-extract-diggs | 3/6 · 721 s · 0.89 M | 5/6 · 696 s · 1.04 M | 5/6 · 578 s · 0.83 M |
| GPT-5.4 | report-summary-vs-sheets | 5/5 · 140 s | 5/5 · 156 s | 5/5 · 85 s |
| GPT-5.4 | scale-log-depth | 2/3 · 25 s | 2/3 · 27 s | 2/3 · 85 s |
| GPT-5.4 | scale-plan-distance | 4/4 · 56 s | 4/4 · 67 s | 4/4 · 40 s |

**Where the failures came from.** Of the 28 failed scored checks in the slice:

* **14 are scorer or harness (S):**
  * `file_produced` × 6;
  * the values check × 6, five of them on "shell" or answer-only reading, every value present in the delivered files;
  * GPT-5.4 checklist's boring ranges × 1;
  * GPT-5.4 baseline diggs' `log_grid`-only pages × 1.
* **The rest are model behaviour (M):**
  * depth by eye × 3;
  * no coverage counts in the baseline arm × 4;
  * Sol's plan answer leaving out the note;
  * GPT-5.4 baseline leaving out the 2011 logs;
  * plus one that a plumbing bug (P) caused: GPT-5.4 coverage's scans, through the dropped `pages` argument.

**What the coverage arms cost, on the three report tasks together:**

| model | baseline | coverage | checklist |
|---|---|---|---|
| Sol | 421 s, 1.73 M | 797 s, 3.23 M | 558 s, 2.28 M |
| GPT-5.4 | 988 s, 1.38 M | 974 s, 1.50 M | 832 s, 1.36 M |

Most of Sol's coverage-arm increase is one run's own choices, not the gate. In `report-extract-diggs` Sol sent three helpers. One of them asked for forced 3×3 tiles on 14 typed lab sheets: 229 vision side calls, most of them on blank tiles. The gate itself cost 1–24 s per run.

---

## 2. Q1. `report-extract-diggs`: no .xml in 6 of 6

### What happened

Every diggs run wrote the file and got a valid verdict. Every path was absolute and under `/tmp`:

| run | output_path | verdict |
|---|---|---|
| Sol baseline L166 | `/tmp/harbour_road_subsurface.diggs.xml` | valid DIGGS 2.6, read back equal, 201 kB |
| Sol coverage L599, L603 (a helper, twice) | same | valid, 192 kB |
| Sol checklist L138 | same | valid, 201 kB |
| GPT-5.4 baseline L100 | same | valid, 234 kB |
| GPT-5.4 checklist L110, L114 | same (L110 refused, see §3) | valid, 197 kB |
| GPT-5.4 coverage L176 | `/tmp/harbour_road_geotechnical_report_extracted.diggs.xml` | valid, 207 kB |

**What the answers did with the file:**
* Sol linked it as `sandbox:/tmp/…` (3 of 3).
* GPT-5.4 printed the path (3 of 3).
* Sol checklist even ran `list_files /tmp` (L148) to confirm the file was there.
* By contrast, every `write_docx`, `save_file` and `annotate_document` call in the whole hand-back was given a bare name and landed in `files/`.

### Why: four sites

* **Example and claim.** `funhouse_agent/adapters/subsurface_adapter.py:889` describes `output_path` as "Where to write the .xml (e.g. '/tmp/site.diggs.xml'); it is attached to the conversation."
  * Six of six models copied the example.
  * "It is attached" is true only through the app's post-turn import (below).
* **Path written as given.** `_run_write_diggs` writes it exactly as given (`subsurface_adapter.py:463-466`, `open(out, "w")`).
  * It never calls `funhouse_agent._fileio.resolve_output_path` (`_fileio.py:78`). So even a bare name would land in the **process working directory**, not in the conversation's folder.
  * Compare `write_docx`: its description says "a bare name lands in the working folder" (`deep/tools.py:1155`). Its result carries the "This file is now ATTACHED …" note (`_with_saved_note`, `deep/tools.py:92`).
  * `write_diggs` returns neither, so models invent `sandbox:` links. Those are dead in the app: `displayable_markdown` rewrites only images.
* **Prompt rule.** The geotech page's prompt still says "Prefer `/tmp/...` or a Unity Catalog `/Volumes/...` path for `output_path`" (`funhouse_agent/deep/prompt.py:150`). That rule comes from the Databricks era, and the same section also says "A bare filename lands in the working folder".
* **The runner** collects only `files/` (`funhouse_agent/review_eval/runner.py:248-252`). Its `stream_turn` call passes only the activity logger (`runner.py:222`).
  * The app's turn job also runs `OutputCollector` (`webapp/turn_jobs.py:145`) and then `core.import_reported_outputs` (`turn_jobs.py:206`).
  * That copies any `"output_path"` a tool reports outside the conversation folder into `files/` (`webapp/output_capture.py:40-43`, `webapp/core.py:1153-1192`).
  * So **in the app the user would have received this file as a card.** The suite measures less than the app delivers.

### Other writers reached through `call_agent` (same flaw or worse)

| writer | path handling | captured by the app's import? |
|---|---|---|
| `subsurface.write_diggs` | absolute honoured; relative → process working directory (`subsurface_adapter.py:463`) | yes (`output_path`) |
| `calc_package` packages and `generate` | explicit path used as given; relative → working directory (`calc_package.py:215, 233`) | yes |
| `calc_package.html_to_pdf` | same (`calc_package.py:1459-1463`) | yes |
| `dxf_export.export_to_dxf` | relative → working directory (`dxf_export.py:65`); reports the path as **`filepath`** (`dxf_export.py:67`) | **no**: `output_capture` only knows `output_path`, `saved` and `plotly_json_path`, so a DXF in `/tmp` never reaches the user, even in the app |
| `drawing_ir.snip_region` | `output_path` passed straight to `render_region_to_file` (`drawing_ir_adapter.py:443`) | yes (`saved`) |
| `subsurface.plot_*`, `profile_figure.*` | `resolve_output_path`: bare → working folder; absolute honoured (`adapters/__init__.py:238`, `profile_figure_adapter.py:46,132`) | yes |

### The general fix

A writer saves into the conversation's folder whatever path the model gives (§9, N1–N2):
* One helper used by every `call_agent` writer. When the host has set a working folder, the file is written there under the file name the model gave.
* An explicit durable location (`/Volumes/…`, `/Workspace/…`) gets a copy as well.
* The result carries the same "attached" note `save_file` returns.
* The descriptions drop the `/tmp` example, and the prompt rule is dropped or kept only for hosts with no working folder.
* The runner reuses the app's output import, so the suite scores what the user receives.

---

## 3. Q2. The `write_diggs` cross-checks (W2)

They fired in every one of the 8 `write_diggs` calls that wrote a file. The 9th call was refused before writing.

| run | summary row written for B-2 S-3 | cross-check entries | answer relayed the swap? |
|---|---|---|---|
| Sol baseline | LL 20, PL 32 (as printed) | conflict LL, conflict PL (pages [16, 18] = PDF 17, 19); mg/L unconverted | yes, both values (no page numbers) |
| Sol coverage | LL 20, PL 32 | same (both calls) | yes, both values |
| Sol checklist | LL 20, PL 32 | same | yes, twice (draft and post-gate); the gate also gave PDF 17, 19 |
| GPT-5.4 baseline | **LL 32, PL 20** (the sheet's values) | partial (BULK-1 at 0.5 m has no logged sample); mg/L | no |
| GPT-5.4 coverage | **LL 32, PL 20** | partial ×2 (BULK-1; W-1 water sample); mg/L | no |
| GPT-5.4 checklist | 1st call: summary as kind `other` → **refused** (L111); 2nd call: **no summary at all** | mg/L only | no |

### What this shows

* **The reconciler works on what it is given.** With the summary as printed, it named both values and both pages (3 of 3).
* **It is blind to a summary the model has "corrected".**
  * GPT-5.4 had the printed row in context: `read_document` of page 16 returned "B-2 S-3 … 20 32 12".
  * It still wrote 32 and 20.
  * That breaks the description's own rule ("numbers AS PRINTED", `subsurface_adapter.py:867`). The behaviour is M, but the check depends on it.
* **The refusal is partly the tool's doing (P).**
  * The `lab_tests` description lists the kinds as "atterberg|gradation|…|other", without `summary_table` (`subsurface_adapter.py:877-879`).
  * `summary_table` appears only later in the same text.
  * The refusal ("a 'other' test cannot carry a 'summary_table' result", `report_ingest/model.py:1563`) does not say which kind to use.
  * GPT-5.4 then dropped the summary instead of fixing it.
* **Noise in the entries:**
  * "mg/L unconverted" came up in all 8 calls. Groundwater chemistry is normally reported in mg/L, so this entry is noise.
  * The pages come back 0-based and unlabelled (`"pages": [16, 18]`). The gate converted them to PDF 17, 19; a model relaying them directly could cite the wrong pages.

---

## 4. Q3. `report-extract-all`: the "last sheets" values, and the pages GPT-5.4 skipped

### Where the values are

| term | value | PDF page | page kind |
|---|---|---|---|
| `43` | B-3 S-5, N at 7.0 m | 11 | vector log, newest boring |
| `shell` | BH-2 description, "Grey silty SAND with shell fragments" | 14 | scan |
| `41.7` | B-2 S-5 fines | 24, and on Table B-1 (17) | lab sheet |
| `1.94` | B-1 BULK-1 maximum dry density | 27 | lab sheet |
| `1,850` | B-2 W-1 sulfate, mg/L | 30 | lab sheet, last page |

### Were they read, and where did they go?

**Sol (3 of 3)** read all 21 target pages. It wrote the tables to a docx and one or two CSVs, and answered with a short pointer to the files.

The check (`contains_all`, `review_eval/checks.py:79-83`) reads only the answer. Re-running it on the answer plus the delivered files:

| run | answer only | answer and files |
|---|---|---|
| Sol baseline, coverage, checklist | 5 of 5 missing | only `shell` missing |
| GPT-5.4 baseline (tables inline, no file) | only `shell` missing | — |
| GPT-5.4 coverage (`.md` file) | 5 of 5 missing | only `shell` missing |
| GPT-5.4 checklist (docx) | 5 of 5 missing | only `shell` missing |

So:
* **The four numbers reached the deliverable in 6 of 6 runs.**
* **`shell` reached it in none**, because the question asks for date, total depth, groundwater and SPT N, and no description (`review_eval/tasks.py:988`).

Both are scorer issues (S). The fix:
* The extraction checks read the answer plus any text file produced, as `docx_contains` already does for docx.
* `shell` is replaced by a scan-only value the question does ask for. BH-1's "Water struck: 3.8 m" is printed only on the scan; the narrative gives only the 3.4–4.1 m range.
* The same file-blind reading fails GPT-5.4 checklist's "names all six borings". Its answer says "B-1 to B-3" and "BH-1 to BH-3", and its docx has a row for each of the six.

### GPT-5.4 and PDF pages 13–15 (report-extract-all)

* **Baseline (L18–L54).**
  * Its thumbnail look listed pages 12–14 as "scanned table/form; appears to be boring/sample field record".
  * It read 7–10 and 16–29 and never looked at 12–14.
  * It said so: "I did not include the scanned 2011 logs on viewer pages 13–15."
  * A scoping choice (M).
* **Coverage (L16–L39).**
  * Its first look at the scans was `analyze_pdf_page(attachment_key=…, pages: 12)`, and likewise for 13 and 14.
  * The tool takes `page` (`deep/tools.py:808-812`). LangChain's `StructuredTool.from_function` (`deep/tools.py:1293`) silently ignores the unknown `pages` key, so `page` defaulted to 0.
  * All three calls returned the **cover page**: "This image is a report cover page, not a borehole log page … If you meant to send the actual boring log sheet, please upload that page." The result even says `"page": 0`.
  * The agent did not retry. It marked 12–14 "skipped" with "Older 2011 boring logs were not included because the request asked for one row per boring…" (L58), and its answer said so.
  * Reproduced offline: a `StructuredTool` built the same way returns page 0 for `{"pages": 12}`.
  * A plumbing bug (P), with a rationalised skip on top (M).
* **Checklist (L19).**
  * The same mistake with `pages: "12-14"`: cover page again.
  * This time the agent went on to `log_grid` and `document_page_map` on 12–14, then looked at each page with `page=`. All 21 pages read, all six borings in the docx.

### GPT-5.4 and PDF pages 8–11 (report-extract-diggs, baseline)

**What it read.** It read pages 2–4 and 15–29 as text, looked at 12–14, and took 7–10 only through `log_grid`:
* one call over `"7-10,12-14"`, paged six times with `offset` (L64–L85), returning all 120 rows;
* it never ran a text read or a look of 7–10.

The `pages_covered` check and the coverage ledger count only `read_document`, `read_pdf_text`, `analyze_pdf_page` and `render_region` (`funhouse_agent/coverage.py:81-83`). So 8–11 count as unread, although every printed value on them was returned. That part is S/P.

**The real loss came from `log_grid` itself:**
* Over seven pages of six logs it returned **one** set of header fields, B-1's. `_collect_fields` keeps the first value per field (planlens `document/loggrid.py:2186-2210`).
* Its `groundwater` field is the whole header line ("Date drilled: … Water level: 3.1 m Total depth: 14.9 m").

**The result in the file:**
* No ground elevations for B-1, B-2 or B-3.
* "B-3 groundwater … assigned 3.6 m based on [the report text] because the detailed log text layer did not separately expose the water entry".
* "B-2 and B-3 total depths were taken from the report text".

The values happen to be right, but they came from the narrative, not the logs. `log_grid`'s description says "Give it the pages of ONE log". The tool does not notice when it is given six.

### GPT-5.4 diggs, BH-1 to BH-3

All three GPT-5.4 diggs runs looked at 12–14 and wrote BH-1 to BH-3, and the borings check passed 3 of 3.

---

## 5. Q4. Coverage (`GEOTECH_COVERAGE`) and checklist (`GEOTECH_REVIEW_CHECKLIST`)

### The ledgers (18 `coverage.json`)

* **The inventory was right in every report run:**
  * logs = 7 (pages 7–10, 12–14);
  * lab = 14 (pages 16–29);
  * plan 5, narrative 2–4;
  * front = 0, 1, 6, 11, 15.
* **Reads were attributed to helpers as well** (Sol coverage extract-all and diggs).
* **"Read as text but no text layer"** was recorded for GPT-5.4 checklist's `read_pdf_text` over the scans (`look+text_empty`), and the scans counted only once they had been looked at.
* **`write_diggs` marked its cited pages extracted** (4 of 4 runs with the gate).
* **The `pages=` bug is in the ledger too.** It recorded GPT-5.4 coverage's three cover-page looks as reads of page 0, which is honestly what happened. The scans stayed unread until the agent marked them skipped.
* **Marking pages is fragile** (P, `coverage.py:110-137`). `document_coverage(extracted=…)` silently lost pages for dict-shaped arguments, as `parse_pages` stringifies the dict:

  | run | `extracted` given as | pages marked |
  |---|---|---|
  | Sol checklist extract-all L206 | `{"logs": "7-10,12-14", "lab": "16-29"}` | none |
  | Sol checklist diggs L146 | `{"logs": [7, …, 14], "lab": [16, …, 29], …}` | `3,8-10,12-13,17-28` |
  | GPT-5.4 checklist extract-all L88 | `[{"pages": [7, …, 14], "group": "logs"}, …]` | `8-10,12-13,17-28`: first and last of each list lost |

  Reproduced offline. Marks affect only the "extracted" column, not the read counts, so no score changed.

### The gate

It spoke in **10 of 18** switched runs:

| run | what it said | what the agent did | cost after the note |
|---|---|---|---|
| Sol coverage extract-all | plan and narrative unread | read them both | 24 s |
| Sol checklist extract-all | plan unread | looked at it | 21 s |
| Sol checklist diggs | all read; 3 checklist fails | restated with the checklist points | 5 s |
| Sol coverage summary | 6 logs unread | marked them skipped, reason given | 6 s |
| Sol checklist summary | 6 logs unread | marked them skipped, reason given | 6 s |
| GPT-5.4 coverage extract-all | plan and narrative unread | marked them skipped, reason given | 4 s |
| GPT-5.4 checklist diggs | plan unread; units fail | said why it was not needed | 69 s (one slow 67 k-token call) |
| GPT-5.4 coverage summary | 7 logs unread | said why not needed | 4 s |
| GPT-5.4 checklist summary | logs, plan and narrative unread | said why not needed | 4 s |
| GPT-5.4 checklist scale-log-depth | all read; state counts | appended "other pages 1 of 1 read" | 1 s |

**Where it was silent:**
* Sol coverage diggs, GPT-5.4 coverage diggs and GPT-5.4 checklist extract-all: every target page was read or skipped, and the answer already stated counts.
* The five other scale runs: no data group was read and no coverage task was declared.

**Stand-down and step limit:**
* The gate never stood down (no `skipped` entry in any ledger).
* **The extraction step allowance:**
  * The runner calls `core.stream_turn` with the default cap of 50 (`runner.py:210`).
  * For a coverage build, `stream_turn` raises the run's cap to 150 and passes 50 as the ordinary turn's allowance (`webapp/core.py:765-771`). So the limit was in force on both switched arms.
  * No turn came near either cap. The most primary model calls in a run was 17, no answer is the tool-less "[Step limit reached]" reply, and RESULTS reports 0 step caps.
  * The activity log does not record the graph step count, so the margin cannot be stated more exactly. Logging `langgraph_step` at `turn_end` would settle it next time.

**Two findings.**

* **Every gated answer is two answers glued together (P).**
  * `stream_turn` joins all model text of one graph pass with no separator (`webapp/core.py:790-799`). The gate jumps back to the model inside the same pass (`deep/coverage_tools.py:343-347`).
  * All 10 gated runs here, and all 16 in parts C and D, have answer = pre-gate draft + post-gate reply.
  * Examples: "…recorded as **Not reported**.The remaining pages have now been reviewed:" (Sol coverage extract-all). GPT-5.4 coverage and checklist summary state the whole B-2 S-3 finding twice.
  * A reader sees a doubled or run-on reply.
* **On a narrow question the gate behaved as the owner decided:**
  * One extra model call, and the agent said why the logs were not needed (5 summary runs).
  * The cost was a coverage paragraph appended to an answer that did not need one, in every summary run and in one depth question. GPT-5.4 had itself declared that one-page depth question a coverage task with `document_coverage`.

**What changed against baseline:**
* Coverage stated as counts in **8 of 8** coverage/checklist extraction answers, against 0 of 4 in baseline.
* Pages covered: Sol 6 of 6 in every arm. GPT-5.4: 0 of 2 in baseline; 3 of 4 in the switched arms (the miss is the cover-page bug).

### `document_coverage`

| model | called in |
|---|---|
| Sol | 8 of 8 switched report runs (primary or helper) |
| GPT-5.4 | 6 of 6 switched report runs except coverage summary; also checklist scale-log-depth |

### The checklist

| model | `report_checklist` called | judge items reported |
|---|---|---|
| Sol | 3 of 3 (at the start in extract-all and summary; after writing in diggs) | never, in any of the three answers |
| GPT-5.4 | 0 of 5 | — |

The code checks are the substance. Here is what they said:

* **`completeness.explorations_have_logs`: "unsure" in 3 of 3.**
  * "BH-1, BH-2, BH-3 named with no log in the text layer … look at them".
  * In the diggs run this was after the agent had looked at 12–14 and written BH-1 to BH-3 with those pages. The check reads only the text layer (`funhouse_agent/report_checklist.py:197-231`), not the ledger's looks or the written data.
* **`completeness.summary_and_sheets`: a false fail (P).**
  * "Sheets not on the summary: B-1 BULK-1 (compaction), B-1 S-3 (chemical), B-2 S-2 (density), B-2 W-1 (chemical), B-3 S-3 (chemical)".
  * Table B-1 carries only Atterberg and fines columns, and prints "See the individual test sheets for the compaction, density and chemical tests."
  * The check expects every sheet on every summary (`report_checklist.py:326-355`).
  * Sol rightly answered that they were "intentionally not included".
* **`consistency.summary_vs_sheets`: a false pass (P).**
  * When no summary rows were written (GPT-5.4 checklist diggs), the item returns **pass**, "raised nothing of this kind" (`report_checklist.py:265-275`). It should say it did not run.
  * The same item also passes after a "corrected" summary, which code cannot detect from the written data alone.
* **On the Document Review page** (extract-all, summary-vs-sheets), the three consistency items all report "not run: … no data has been written". The only check that can compare a summary with its sheets never runs where a reviewer would ask for that comparison.

### Sol `scale-plan-distance`, coverage arm: why the one check failed

* **No coverage involvement.** The ledger shows one read and no gate, and no coverage tool was called.
* **Same method and value as baseline:**
  * `measure` listed the scales: bar 0.5556 ft/pt; stated 1" = 20' with "disagrees with the bar scale by 50.0 %".
  * It snapped both dots and returned 183.85 ± 0.12 ft with "reconciled: the stated scale disagrees with the bar … the bar is used" (L45).
* **The answers differ:**
  * Baseline: "The printed scale note conflicts with the graphic scale, so the graphic scale was used."
  * Coverage: "…measured center-to-center in a straight line using the site plan's graphic scale bar." The note is gone.
* M, run variance. GPT-5.4 relayed the disagreement 3 of 3.
* **A plumbing note** from the same task. Three runs passed `to: [view, image_box]` for the second dot, seen in a different zoom, and got "to must be four numbers" (`funhouse_agent/measure_tool.py:167-212`): Sol coverage L40, GPT-5.4 baseline L38, GPT-5.4 coverage L40.
  * GPT-5.4 baseline's eventual box left one end unsnapped (182.7 ± 1.7 ft). It still scored, at 0.85 ft off.

---

## 6. Q5. `scale-log-depth`: GPT-5.4 skipped the tools

**The three GPT-5.4 runs (baseline, coverage, checklist):**
* Each took one `analyze_pdf_page`, with tiles on auto.
* Each answered from it. The whole-page readings were "Its top aligns with the 4.0 depth mark … starts at exactly 4.0 m", "about 3.9 m" and "about 3.8 m".
* The tile readings disagreed (4.0, and 8.0 from a tile without the description column).
* The answers were 4.00 ("Confidence: high"), 3.90 and 3.80. Truth is 3.62; a reading that takes label centres for depths gives about 3.72.
* GPT-5.4 checklist added, unprompted: "I would not report it more precisely than that from this log image alone." It knew the reading was approximate and still did not look for a way to measure.

**What it was offered:**
* `measure`: "Measure a position on a page through the page's own scale: a log's depth or elevation ruler … A position read off an image by eye is approximate; this finds the drawn thing itself (a line…) and converts its position" (`measure_tool.py:39-60`).
* `log_grid`: "Read a boring log … the LAYERS the description column is cut into, each with its top depth" (planlens' text, `measure_tool.py:255-270`).
* Both were in GPT-5.4's tool list on this page.

**How Sol did it:**
* It called `log_grid` in both runs. One vision call read the ten scan labels.
* The layers came back at 1.64, **3.62**, 4.87 and 6.93 ± 0.016 m, with empty descriptions because the scan has no text.
* It then matched the description by looking, and in baseline confirmed with `measure` (an ambiguous list with 3.62 first).
* It answered 3.60 both times.

**Verdict: a model habit (M), not a description a model cannot find.**
* GPT-5.4 found and used `measure` in 3 of 3 plan runs, from its first step, when the sheet showed a scale bar and a note.
* On the log, the look came back with a confident number, and GPT-5.4 stopped.

The general lever is the descriptions' first words. Lead `measure` and `log_grid` with the reviewer's questions ("at what depth…", "how far apart…", "where does a layer start"), not with the method. That is to be measured on Foundry (GPT-5.4 × 3 per wording). A result note pointing at the tool is off the table by the owner's rule.

---

## 7. Q6. Layer intervals

* **The fixture.** Its 2026 logs are ruled tables (`report_fixture.py:322-347`): one row per sample, rows every 34 pt, a DEPTH column holding sample depths 1.0, 2.5, 4.0 … and a description per sample. The 2011 scans are the same as images. Neither draws a stratum line.
* **Sol's answer was correct.** `parse_diggs` read back `n_lithology_intervals: 0` (L175 in the baseline run), and the answer said "The report provides no numeric boring coordinates or explicit layer-boundary depths."
* **Sol's helpers asked the question directly** (checklist diggs L79–L84): "Determine whether true lithologic contacts are shown separately from sample-row lines". The looks answered: "No true lithologic-contact lines are shown separately from the sample-row grid lines … no exact lithologic contact depths are present on this sheet."
* **`log_grid` did return layers**:
  * 24 for B-1's two sheets, 8 for B-2, 9 for B-3, each `source: "stratum_rule"` at confidence 0.85;
  * tops at 0.24, 1.74, 3.24 … m.
  * These are the table's horizontal row rules, read through a "depth ruler" fitted to the sample-depth column ("label centres taken as the value they mark; nothing on the page confirms it").
  * planlens takes every horizontal rule covering the description column as a stratum rule (`document/loggrid.py:1800-1810`).
  * No run copied them, and nothing in the code drops layers on the way to the file. The model passes layers to `write_diggs` itself; Sol passed none.
* **GPT-5.4 invented layers.** It wrote 28, 25 and 29 layers of its own, built from sample depths (0–1.0, 1.0–2.5 … in baseline; tops shifted to start at 1.0 in coverage). Each run's scheme was different. That is invention against the writer's "never inferred" rule (M).
* **On the scanned scale fixture**, which does draw stratum lines, `log_grid` is right (3.62).
* **The 2011 report scans** returned no ruler and no layers ("no column holds an even run of label-sized marks; if the page is a sketch not drawn to scale, use the depths it prints"). That is the right answer for a table.

**The fix:** `log_grid` should tell the vector table apart in the same way (§9, N8).

---

## 8. Per-run findings

### Sol, baseline

* **report-extract-all (4/6).**
  * Read all 21 target pages: `log_grid`, plus looks of the vector logs, the scans with tiles, and all 14 lab pages.
  * Wrote a docx and a CSV with every value, and flagged the B-2 S-3 swap with PDF 17 and 19.
  * Failed: coverage counts (M; it counted rows, not pages) and the values check (S).
  * The thumbnail look mislabelled the logs ("B-4"), which was harmless.
* **report-extract-diggs (4/6).**
  * Read everything. One `write_diggs`: valid, two conflicts and the mg/L entry. `parse_diggs` read back 6 investigations and 0 intervals.
  * The answer relayed the conflict without page numbers.
  * Failed: coverage counts (M) and file (S/P).
* **report-summary-vs-sheets (5/5).** Looked at all 14 lab pages, wrote a docx memo, and got the right answer.
* **scale-log-depth (3/3).** `log_grid` gave 3.62, `measure` confirmed it, and the answer was 3.60. The vision guesses ranged from 2.0 to 8.0 m.
* **scale-plan-distance (4/4).** `measure`: 183.85 ± 0.12 ft. The answer said 184 ft and noted the conflicting scale note.

### Sol, coverage

* **report-extract-all (5/6).**
  * Declared a coverage task and sent three helpers. One helper passed the document handle as `attachment_key` to `open_document` and four `analyze_pdf_page` calls, got errors, then recovered.
  * `render_page_thumbnails(pages="all")` failed: "invalid literal for int()".
  * Files written; then the gate (plan and narrative) → it read them. The answer is two glued parts.
  * Failed: values (S).
* **report-extract-diggs (5/6).**
  * 449 s and 1.68 M tokens. The lab helper forced 3×3 tiles on 14 typed sheets, and most tiles came back blank.
  * The writing helper's first call used ids "B1, B2, B3". The writer said it had synthesised three empty boreholes, and the helper rewrote with B-1 to B-3, a good catch by the writer.
  * Conflict relayed, counts stated, no gate.
  * Failed: file (S/P).
* **report-summary-vs-sheets (5/5).** Gate on the logs → skipped with a reason. Answer glued.
* **scale-log-depth (3/3).** `log_grid` first, gave 3.62; answered 3.60.
* **scale-plan-distance (3/4).** One `measure` `to` error, then 183.85 ft. The answer left out the scale disagreement (M).

### Sol, checklist

* **report-extract-all (5/6).**
  * `report_checklist` at the start (pages 0 of 14 "fail"; explorations "unsure").
  * Read all pages and zoomed the scans' boring IDs. A dict-shaped `extracted` marked nothing.
  * Gate (plan) → it looked at the plan. Answer glued.
  * Failed: values (S).
* **report-extract-diggs (5/6).**
  * Asked the looks whether contacts are drawn: "no".
  * One `write_diggs` with the conflict; `list_files /tmp`; `report_checklist` gave the false "sheets not on summary" and "unsure" for the BH logs.
  * The gate passed on the checklist fails, and the agent explained them.
  * Failed: file (S/P).
* **report-summary-vs-sheets (5/5).** Checklist at the start; gate on the logs → skipped with a reason. Answer glued.

### GPT-5.4, baseline

* **report-extract-all (2/6).** Left out the 2011 logs by choice (M); tables inline.
  * Failed: pages 13–15, coverage counts, borings, values ("shell" only, S).
* **report-extract-diggs (3/6).**
  * Logs 7–10 taken only through a seven-page `log_grid`.
  * Summary "corrected", so no conflict. 28 invented layers, no elevations, B-3's water taken from the narrative.
  * Failed: pages (S/P), coverage counts (M), file (S/P).
* **report-summary-vs-sheets (5/5).**
  * Its thumbnail look called page 28 a "HCL DIFFERENTIAL - AUTOMATED" summary table, a hallucination from the contact sheet. It was ignored.
  * The answer was correct.
* **scale-log-depth (2/3).** One look; 4.00 m.
* **scale-plan-distance (4/4).**
  * One `to` error, then an ambiguous list from a whole-page box, then one unsnapped end.
  * Got 182.7 ± 1.7 and answered 183 ft, noting the scale conflict.

### GPT-5.4, coverage

* **report-extract-all (3/6).** `pages=12/13/14` → the cover page ×3 → scans marked skipped. The gate (plan and narrative) → they were marked skipped too. Answer glued.
  * Failed: pages, borings, values.
* **report-extract-diggs (5/6).**
  * `log_grid` on 7–10 and 9, 10; scans looked at. A helper drafted a 44 k-character JSON.
  * Summary "corrected". Two "partial" links relayed. 29 invented layers.
  * Failed: file (S/P).
* **report-summary-vs-sheets (5/5).** Gate on the logs → said why not needed. The answer repeats the finding in full.

### GPT-5.4, checklist

* **report-extract-all (4/6).**
  * `pages="12-14"` → cover page, then recovered and read all pages.
  * A list-of-dicts `extracted` lost pages.
  * Answer gave ranges of borings, with counts.
  * Failed: borings (S), values (S).
* **report-extract-diggs (5/6).**
  * Summary refused as kind `other`, then dropped. No word of the swap anywhere.
  * Gate (plan and units) → duplicated answer.
  * Failed: file (S/P).
* **report-summary-vs-sheets (5/5).** Gate → duplicated answer.
* **scale-log-depth (2/3).** `document_coverage` on a one-page question; one look; 3.80. The gate appended "1 of 1 read".
* **scale-plan-distance (4/4).** Used the text layer's label boxes, got 183.85, answered 184 ft with the note.

---

## 9. Fix table

Classes:
* **P:** plumbing or code; reproducible offline with a fake model or any model.
* **M:** specific to GPT-5.4 or Sol; only Foundry can measure it.
* **S:** scorer or task.

On a Claude API check: no P item here needs one. Every call shape is in these traces (`pages=12`, `to=[view, box]`, dict-valued `extracted`, `/tmp` paths, a summary given as kind `other`), so a scripted fake model replays each path exactly. An optional Haiku 5.5 end-to-end smoke of the five tasks after the fixes would cost well under $1. It would catch integration slips, but it is not needed to confirm any fix, and it says nothing about GPT.

| id | failure | class | code site | proposed general fix | how to measure | Claude check? |
|---|---|---|---|---|---|---|
| N1 | DIGGS written to `/tmp`, never in `files/`; the suite fails `file_produced` 6/6 though the app would import it | S + P | `review_eval/runner.py:222, 248-252`; `adapters/subsurface_adapter.py:463-466, 889`; `deep/prompt.py:150` | (a) The runner reuses the app's output import (`OutputCollector` + `core.import_reported_outputs`), so the suite scores what the user gets. (b) `write_diggs` goes through one writer helper: with a host working folder set, the file lands there under the model's file name (a copy kept at an explicit durable path), and the result says "attached" as `save_file` does. (c) The description says "a bare file name; it lands in the conversation's folder"; the `/tmp` example goes; prompt.py:150's rule applies only with no host folder. | Offline: a fake model calls `write_diggs(output_path='/tmp/x.diggs.xml')`, the file is in `files/` and `file_produced` passes. Rescore nothing (the file is gone). Foundry: diggs `file_produced` 6/6. | n: a fake model replays the exact call |
| N2 | Other `call_agent` writers keep the model's path; DXF reports `filepath`, which the app never imports | P | `adapters/calc_package.py:215, 233, 1459-1463`; `adapters/dxf_export.py:65-67`; `adapters/drawing_ir_adapter.py:443`; `webapp/output_capture.py:40-43` | The same writer helper for every file-writing method; every writer reports `output_path` | Offline: one test per writer, `/tmp` path and bare name → `files/` | n |
| N3 | `analyze_pdf_page(pages=12)` silently looks at page 0 (2 GPT-5.4 runs, 4 calls); one run then skipped the scans | P | `deep/tools.py:808-827` (`page: int = 0`); `deep/tools.py:1293` (`StructuredTool.from_function` ignores unknown keys) | Every app tool refuses unknown argument names with the valid ones listed (strict schema or a wrapper). For one-page tools, `pages` → "this tool takes one page; call it once per page". | Offline: the fake call `{"pages": 12}` returns the error, never page 0. Foundry: GPT-5.4 extract-all pages covered and borings. | n |
| N4 | After the gate, the answer is the draft and the reply glued, with no separator (10/10 gated runs; GPT-5.4 repeats whole findings) | P | `webapp/core.py:790-799`; `deep/coverage_tools.py:321-351` | When the gate speaks, the turn's answer is the post-gate reply alone (the draft stays in the trace), and the gate's note says its reply replaces the draft, so it must be complete | Offline: a fake model answers, the gate fires, it answers again; assert one answer, not two. Foundry: gated answers read once. | n |
| N5 | Values and borings checks read only the answer; tables were in files; `shell` is not asked for | S | `review_eval/checks.py:79-83, 110-147`; `review_eval/tasks.py:986-990` | Extraction checks read the answer plus every text file produced (docx, csv, md), as `docx_contains` does. Replace `shell` with a scan-only value the question asks for (BH-1 "Water struck 3.8 m"). | `rescore_saved` on these 6 runs offline: values pass 6/6 with files; borings pass for GPT-5.4 checklist | n |
| N6 | Pages read only through `log_grid` count as unread (scorer and ledger) | S/P | `funhouse_agent/coverage.py:81-83, 171-242` | Count `log_grid` as a text read of each page with a text layer whose rows it returned in full (every offset reached) | Offline rescore of GPT-5.4 baseline diggs: 21 of 21 | n |
| N7 | `log_grid` over several logs returns the first log's header fields; `groundwater` is the whole header line | P (planlens) | planlens `document/loggrid.py:2186-2210`, `:2652` | When the pages carry more than one boring number, return fields per log, or refuse with "these pages hold B-1, B-2, B-3: one call per log". Split a multi-pair header line into its pairs. | Offline on `report_fixture` pages 7–10: three field sets; `groundwater = "3.1 m"` | n |
| N8 | On a ruled table log, `log_grid` reads the row rules as stratum lines (24/8/9 false layers) | P (planlens) | planlens `document/loggrid.py:1800-1810` (and the ruler fit on a sample-depth column) | When each depth label sits alone in its own ruled row band at the label step, treat the column as a table of sample depths: rows with printed depths, no ruler, no layers, and a warning saying so | Offline: `report_fixture` page 9 gives 0 layers; the scale fixture still gives 1.64/3.62/4.87 | n |
| N9 | Summary table refused as kind `other`; the kind list omits `summary_table`; the refusal does not name the fix | P | `adapters/subsurface_adapter.py:877-879`; `report_ingest/model.py:1563` | List `summary_table` in the kind enum; the refusal names the kind to use | Offline: replay GPT-5.4 checklist L110's call; the error names `summary_table` | n |
| N10 | Summary-vs-sheet check depends on the model writing the summary as printed (GPT-5.4 corrected it 2/3 and dropped it 1/3); the checklist then passes | P (model part M) | `report_checklist.py:265-275`; `coverage.py:574-641` | (a) The item is "not run" when no summary rows were written but the inventory has a lab summary page, and the gate says so. (b) Where that page has a text layer, code reads the summary from the page itself and compares it with the sheets, independent of the model's transcription. | Offline: a fake model writes the corrected row and the check still flags B-2 S-3 from page 16's text. Foundry: GPT-5.4 diggs names the swap. | n |
| N11 | Checklist false fail: "sheets not on the summary" for kinds the summary does not carry | P | `report_checklist.py:326-355` | Expect on a summary only sheets of the kinds its columns carry | Offline on Sol checklist diggs' ledger: no fail | n |
| N12 | "explorations have logs" stays "unsure" after the scans were looked at and written | P | `report_checklist.py:197-231` | Settle from the ledger's looks and the investigations written with their pages | Offline on the same ledger: pass | n |
| N13 | `document_coverage` marks silently lose pages given as dicts or lists of dicts | P | `coverage.py:110-137` | Accept `{group: pages}` and `{"pages": […]}`; report any part it could not read | Offline: the three shapes in §5 mark all their pages | n |
| N14 | `measure`'s `to` takes only a box on the first view; 3 runs lost a call, one ended unsnapped | P | `measure_tool.py:167-212` | Accept `to` as `[view, image_box]` (or `to_view` + `to_image_box`) | Offline: replay GPT-5.4 baseline L38 → 183.85 | n |
| N15 | A document handle is refused as `attachment_key` / `source` (one helper lost 5 calls) | P | `funhouse_agent/document_tools.py:213-227` (`_resolve`); vision tools' key lookup | Resolve an open document's handle to its source | Offline: replay Sol coverage extract-all L29, L84 | n |
| N16 | "mg/L unconverted" in every write; cross-check pages are 0-based and unlabelled | P (minor) | `report_ingest` reconciler unit table; `subsurface_adapter.py:503-508` | Store concentration units verbatim without a point; return `pdf_pages` beside `pages` | Offline | n |
| M1 | GPT-5.4 answers a depth by eye and never calls `measure` or `log_grid` (0/3) | M | descriptions `measure_tool.py:39-60, 255-270` | Lead both descriptions with the reviewer's questions ("at what depth…", "how far apart…") | Foundry: GPT-5.4 `scale-log-depth` × 3 per wording; tool call recorded | n: model habit |
| M2 | GPT-5.4 "corrects" a summary row; invents layers from sample depths | M (N10 is the code side) | `write_diggs` description | None in prose; N10 makes the check independent of it | Foundry diggs × 3 | n |
| M3 | GPT-5.4 scopes out the 2011 logs (baseline) and rationalises a skip (coverage) | M (N3 is the code side) | — | None beyond N3; the ledger already reports "3 skipped" in the answer | Foundry extract-all × 3 | n |
| M4 | Sol's coverage plan answer left out the scale disagreement | M | — | None (`measure` already returns it) | Foundry repeat × 3 | n |

**Priority, in this order:**
* N3, N4 and N1(a, b): they change what users and the suite see.
* N10 and N9: the W2 purpose.
* N5 and N6: the scorer, so these tasks measure what they mean.
* N7 and N8: `log_grid`'s data quality.

None of the fixes is tuned to the Harbour Road fixture. Each names a general shape: unknown arguments, writer paths, glued answers, tabular logs, multi-log calls, summaries that carry some kinds, and dict-shaped page lists.
