# Foundry brief 5, part C: the 23 other older tasks, every run read in full

**Slice:** every older suite task except `meck-*` and the five new ones
(`report-*`, `scale-*`): `set-*` (6, including `set-long-rare-tag`),
`ufc04-*` (4), `ufc07-*` (2), `ufc301-*` (2), `ufc260-*` (2), `calc-*` (2),
`fixture-*` (2) and `produce-*` (3). GPT-5.6 Sol, arms `baseline` and
`coverage` (`GEOTECH_COVERAGE=1`): 46 runs. Wheels
**geotech-staff-engineer 5.33.0rc1** (app `39716a8`) and **planlens
0.13.0rc1** (`c5bdb8d`). Code is cited at those commits.

**Sources.** Raw hand-back (git-ignored):
`module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/part_c/sol/runs/{baseline,coverage}/<task>/`.
Each run is set beside its **run 6** twin from brief 4 (5.32.1rc3, Sol,
baseline): `.../2026-10-08_foundry_brief4_5.32.1rc3/raw/part_c/sol/runs/baseline/<task>/`,
reviewed in `../2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md` (cited as
**brief 4**). The cost split in §7 covers all 42 tasks and uses only
`results_baseline_coverage.json` and `run.json`.

**Method.** For the 46 runs I read every event of `activity.jsonl`
(5,180 records), every vision side call's text (1,342 page, tile and zoom
answers, plus the `find_like`, `analyze_image` and markup-check reads),
every tool result, every final answer and check verdict, and the 21
`coverage.json` ledgers. The run-6 twins were read in compact form and in
full where a comparison needed them. Scripts then counted, per run and arm:
tokens, time and calls; page looks, tiles and zooms, each zoom's window and
px per pt; pixel boxes against 0-999 grid boxes; bracketed readings;
`found` / `found_note` / `aim_note` and what each pointed at; gate events.
Offline, with no model: the coverage inventory (`Inventory.from_document`)
over the suite's public documents; the ground truth of the long tag set
(`build_synthetic_tag_set`); page rotation of the Mecklenburg PDFs; a
600 dpi render of the 11.01 pavement note.

`Lnn` is a line of that run's `activity.jsonl`.

---

## In short

1. **Outcomes: 23 of 23 in run 6; 22 of 23 in baseline; 23 of 23 in
   coverage.** The only change is `set-long-rare-tag`, which failed in
   baseline (pages 3, 9 and 19 extra, precision 0.50) and passed in coverage.
   That pass was **not the gate's doing** (the gate never fired there) and is
   mostly luck (§2). In all three runs every wrong zoom reading was a tag
   **turned 90°**: 6 of 15 turned look-alikes that were zoomed were read as
   FPG, against 0 of 11 upright ones. The other 22 answers
   carry the same facts as run 6.
2. **The gate fired on 6 of these 23 ordinary tasks and helped none.** Four
   firings came from planlens page roles that call prose pages "laboratory
   sheets" or "exploration logs". UFC 3-220-04 has 33 of its 60 pages so
   labelled (30 of them plain text), UFC 3-260-02 has 22, UFC 3-220-07 has 7,
   and the submittal's duplicated text page is labelled a boring log. The
   answers then told the user things like "laboratory sheets: 3 of 33 read"
   about a backfill manual (§3.1).
3. **When the gate fires, the user gets the answer twice, or with an
   addendum glued on.** The pre-gate answer and the post-gate answer are
   joined with no separator (`webapp/core.py:780-799`). In 4 of the 6
   firings the whole answer was repeated (a 14-row appendix table twice in
   `ufc260-appendices`); in one it was glued on mid-line (§3.2).
4. **Most of the coverage arm's extra cost on ordinary tasks came from the
   agent calling `document_coverage` on its own, not from the gate.** It
   opened 7 of 23 ordinary tasks with a coverage declaration (sheet index,
   cross-references, memo, calc check, "list Chapter 12's tables"…). Those 7
   cost +1.86 M input tokens; the 4 gate-only firings cost +0.28 M.
   `ufc260-ch12-tables` alone went from 92 K to 1,090 K for the same eight
   tables: it sent 4 helpers to look at 49 pages, then the gate listed 400+
   unread pages of a 538-page manual, because a declaration has no page
   scope (§3.3).
5. **Part C cost split:** the extra +3.82 M input tokens break down as:
   - 39 % in the 3 extraction tasks;
   - 56 % in 11 ordinary tasks where coverage engaged (all in this slice);
   - 5 % in the 28 ordinary tasks where it did not.

   Of the extra +2,036 s, 1,702 s are two single events: a tile call that
   stalled 907 s, and one `find_like` run of 795 s (§7). **On the 28 untouched
   tasks the arm costs what it should, about nothing.**
6. **Brief 4's fixes held on these tasks.**
   - A, B, J: comments and rings were confirmed, and the comment is 0.0 pt
     from note 4.
   - C: no agent `dpi` in 104 zooms.
   - D: no refused `annotate_document` call.
   - G: 1 bracketed reading in 1,342 answers.
   - H: the flood guard stopped a leader-tip example in 5.5 s (brief 4: 289 s).
   - L: half held. `run.json` has commits and the vision profile, but 0 of
     1,873 `model_end` records carry reasoning.

   **Two of the fixes are noisy:**
   - **F**'s `aim_note` fired on 11 of 22 padded zooms, all false alarms.
   - **E**'s `found_note` named 441 "tiles-only" things in 139 of 181 tiled
     looks. 30 % of them are lines saying nothing is there. On the long tag
     set it did its job (§4).
7. **Two general weaknesses showed up across tasks.**
   - **Turned lettering.** 8 of the 10 Mecklenburg sheets are drawn sideways
     on portrait pages, with no `/Rotate`.
     - Whole-page reads there drop characters from sheet numbers: "10.1",
       "20.0A", "503".
     - Zooms there misread titles: "POST STANDARDS", "REFERENCE CONTROL
       MONUMENT".
     - Vision calls there write 1.6× the output tokens of the upright
       sheets.
   - **Stalled side calls.** Side calls inherit the engine's 900 s read
     timeout (`webapp/palantir_sdk_engine.py:639`). The p99 side call takes
     74 s, yet calls stalled for 907 s and 343 s (and 903 s in brief 4).

---

## 1. Per task: run 6 → baseline → coverage

Input/output tokens in thousands. "Page looks (tiled)" counts
`analyze_pdf_page` calls and how many were tiled. In the last column,
"declared" means the agent called `document_coverage` itself; "gate fired"
means a `coverage_gate` event is in the log.

| task | outcome r6 / base / cov | input K | output K | seconds | model calls | page looks (tiled) | zooms | coverage arm |
|---|---|---|---|---|---|---|---|---|
| calc-bearing-consistency | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 58 → 33 → 51 | 1.9 → 0.7 → 0.8 | 34 → 15 → 25 | 10 → 3 → 4 | 1 (0) → 0 → 0 | 0 → 0 → 0 | declared, gate fired |
| calc-wall-check | ✓ 5/5 / ✓ 5/5 / ✓ 5/5 | 242 → 270 → 392 | 21.2 → 29.4 → 43.0 | 179 → 161 → 247 | 23 → 23 → 30 | 6 → 7 → 7 (0) | 0 → 0 → 5 | declared |
| fixture-duplicate-page | ✓ 6/6 / ✓ 6/6 / ✓ 6/6 | 41 → 32 → 154 | 0.3 → 0.3 → 6.1 | 10 → 9 → 52 | 4 → 3 → 24 | 0 → 0 → 4 (2) | 0 → 0 → 0 | gate fired, then marked |
| fixture-markups | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 28 → 122 → 32 | 0.6 → 13.8 → 0.7 | 14 → 126 → 16 | 3 → 27 → 3 | 0 → 2 (1) → 0 | 0 → 0 → 0 | — |
| produce-circle-tags | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 286 → 190 → 287 | 88.6 → 12.9 → 18.0 | 428 → 134 → 198 | 60 → 37 → 48 | 1 (1) each | 9 → 8 → 15 | — |
| produce-markup | ✓ 4/4 / ✓ 5/5 / ✓ 5/5 | 75 → 106 → 108 | 3.8 → 3.2 → 4.5 | 49 → 59 → 64 | 14 → 19 → 19 | 1 (1) each | 0 → 2 → 2 | — |
| produce-memo | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 469 → 653 → 857 | 92.1 → 89.5 → 119.6 | 273 → 266 → 300 | 71 → 74 → 80 | 10 (10) each | 6 → 6 → 10 | declared |
| set-3600-psi | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 252 → 266 → 328 | 32.8 → 52.8 → 70.1 | 82 → 120 → 232 | 58 → 58 → 63 | 10 (10) each | 0 → 0 → 3 | — |
| set-cross-references | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 595 → 630 → 676 | 120.6 → 124.8 → 119.5 | 1340 → 370 → 323 | 72 → 73 → 73 | 11 (10) → 10 (10) → 10 (10) | 3 → 5 → 4 | declared |
| set-find-bioretention | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 123 → 343 → 351 | 22.8 → 37.5 → 52.2 | 154 → 156 → 1091 | 23 → 63 → 65 | 2 (2) → 10 (10) → 10 (10) | 1 → 2 → 2 | — |
| **set-long-rare-tag** | ✓ 3/3 / **✗ 2/3** / ✓ 3/3 | 997 → 1128 → 1520 | 43.9 → 52.1 → 261.2 | 173 → 228 → 1033 | 258 → 264 → 322 | 24 (24) each | 8 → 11 → 16 | declared; `find_like` |
| set-ncdot-vs-county | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 197 → 281 → 276 | 31.9 → 56.0 → 49.7 | 168 → 413 → 114 | 38 → 59 → 59 | 5 (5) → 10 (10) → 10 (10) | 1 → 0 → 0 | — |
| set-sheet-index | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 297 → 321 → 403 | 19.9 → 46.9 → 43.0 | 97 → 148 → 163 | 64 → 65 → 70 | 10 (10) each | 4 → 5 → 7 | declared |
| ufc04-confined-zones | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 49 → 69 → 100 | 0.4 → 0.6 → 0.7 | 11 → 12 → 24 | 4 → 4 → 6 | 0 | 0 | gate fired, then marked |
| ufc04-density | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 46 → 62 → 85 | 0.4 → 0.4 → 0.5 | 11 → 13 → 13 | 4 → 5 → 5 | 0 | 0 | gate fired |
| ufc04-supersedes | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 42 → 33 → 34 | 0.2 → 0.2 → 0.1 | 13 → 8 → 7 | 4 → 3 → 3 | 0 | 0 | — |
| ufc04-table-5-1 | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 69 → 61 → 62 | 0.7 → 0.5 → 0.9 | 22 → 27 → 27 | 11 → 10 → 10 | 1 (0) each | 0 | — |
| ufc07-drilled-shaft-table | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 117 → 119 → 123 | 3.0 → 2.9 → 2.8 | 39 → 54 → 40 | 17 → 17 → 17 | 2 (1) each | 0 | — |
| ufc07-figure-1-1 | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 85 → 107 → 139 | 5.5 → 10.0 → 11.7 | 46 → 61 → 71 | 18 → 19 → 22 | 2 (2) → 2 (2) → 3 (2) | 0 → 0 → 1 | — |
| ufc260-appendices | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 90 → 87 → 189 | 1.4 → 1.0 → 2.0 | 36 → 32 → 40 | 5 → 5 → 7 | 0 | 0 | gate fired, then marked |
| ufc260-ch12-tables | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 95 → 92 → **1090** | 0.9 → 0.7 → 24.3 | 22 → 26 → 133 | 5 → 5 → 90 | 0 → 0 → **49** (0) | 0 | declared, 4 helpers, gate fired |
| ufc301-asce7-chapters | ✓ 3/3 / ✓ 3/3 / ✓ 3/3 | 92 → 98 → 96 | 1.6 → 1.3 → 1.5 | 26 → 24 → 28 | 5 → 6 → 5 | 0 | 0 | — |
| ufc301-changes | ✓ 4/4 / ✓ 4/4 / ✓ 4/4 | 36 → 40 → 41 | 0.4 → 0.4 → 0.5 | 9 → 11 → 18 | 3 → 3 → 3 | 0 | 0 | — |
| **slice total** | 23 / 22 / 23 | **4,383 → 5,141 → 7,392** | 495 → 538 → 833 | 3,236 → 2,474 → 4,258 | 774 → 845 → 1,028 | 86 → 100 → 152 | 32 → 39 → 65 | |

Model calls include the 5-call probe in the first vision tool of every task
that looks (14 baseline runs, 15 coverage runs), as in brief 4. Run 6's 3,236 s include the
903 s hung call in `set-cross-references`.

**Baseline against run 6:**
- Input is +17 %, most of it agent choices:
  - `set-find-bioretention` and `set-ncdot-vs-county` now look at all 10
    sheets (run 6 looked at 2 and 5);
  - `fixture-markups` looked at both pages although the markup record
    answers it.
- `produce-circle-tags` dropped from 88.6 K to 12.9 K output. `find_like`
  no longer floods (H, §4).

## 2. `set-long-rare-tag`: why baseline failed and coverage passed

**Truth** (`planlens.testing.tag_fixtures`, 24 pages, read offline):
- one upright FPG callout on each of pages 4, 12 and 20;
- 72 FBG look-alikes, 10 of them callouts **turned 90°**;
- turned GPE, GCE and GCG callouts too;
- no FPG is ever turned.

**Baseline (fail).**
- Whole-page answers named FPG callouts on pdf pages 3, 4, 9, 12 and 20.
- `found_note` added six "tiles-only" FPG callouts. The agent zoomed all 11
  candidates at 12-17 px/pt (L556-L599).
- The zooms correctly rejected 5 look-alikes: 3 upright FBGs, a turned FBG
  and a turned GPE.
- **They accepted three turned FBGs as FPG**:
  - pdf 3, window [430,410,530,500], answered "FPG, rotated vertically", L592;
  - pdf 9, [360,495,440,575], L586;
  - pdf 19, [660,270,745,350], L598.
- Answer: 3, 4, 9, 12, 19, 20 (precision 0.50).

**Coverage (pass).** The gate never fired: the 24 sheets inventory as
"other", the agent declared coverage itself, read all 24, and its answer
says "all 24".

Three things differed, none of them coverage:
- **Page reads.** The whole-page reads on pdf 9 and 19 happened to call
  the turned FBG an FBG this time.
- **The pdf 3 zoom prompt.** The zoom on pdf 3's turned FBG asked
  *"Distinguish FPG from FBG"* and was answered "F-B-G", character by
  character (L680). Baseline's prompt on the same tag did not name the
  look-alike.
- **Where the misreads landed.**
  - Two more zooms read turned tags as FPG:
    - pdf 4's turned FBG at [384,645] (L692);
    - pdf 20's turned **GPE** at [760,369] (L688).
  - Both pages hold a true FPG, so the page list stayed right.
  - The answer's per-page count is still wrong ("pages 4 and 20 each
    contain two FPG leader callouts"). The check does not look at counts.

The coverage arm also ran `find_like` (L564-L663) on a loose example box: 47 × 18 pt, the callout with its leader, round a 10 × 4 pt tag.
- **Size.** 978 candidates over 24 pages (about 41 a page, so under
  `FLOOD_PER_PAGE = 200`, `funhouse_agent/find_like.py:57`).
- **Cost.** 49 contact-sheet reads, 795 s, 129 K input and 195 K output
  tokens.
- **Result.** 28 "FPG instances", 3 of them real. The agent zoomed the
  false ones on pdf 5, 7, 8, 14, 15, 17, 18, 19 and 24 and rejected them all
  (one turned GPE came back as "GDE", still not FPG).
- **Verdict.** It changed no answer and is most of the arm's extra time and
  output on this task.

**Run 6 (pass at 0.75):** it zoomed four turned look-alikes and misread one
(pdf 9, "rotated tag reads FPG").

**Across the three runs:**

| | turned look-alikes zoomed | read as FPG | upright look-alikes zoomed | read as FPG |
|---|---|---|---|---|
| run 6 | 4 | 1 | 1 | 0 |
| baseline | 5 | 3 | 3 | 0 |
| coverage | 6 | 2 | 7 | 0 |
| **all three** | **15** | **6** | **11** | **0** |

Each zoom window was matched to the fixture's truth offline; the turned
look-alikes are FBG and GPE callouts at 90°.

So the pass/fail of this task is decided by how many turned look-alikes the
whole-page reads happen to flag, and how each zoom of one happens to read.
**It is one coin per turned tag, not a measure of the coverage arm.** The
general weakness is turned lettering (fix R).

## 3. What the coverage arm did on ordinary tasks

### 3.1 The gate fired 6 times; 4 times on mislabelled pages

| task | why armed | what the note listed | what the agent did | answer |
|---|---|---|---|---|
| ufc04-density | data (read pages 4, 5, 28) | "laboratory sheets 30 of 33 not read" | re-answered | answer + "Laboratory sheets: 3 of 33 read", glued on mid-line |
| ufc04-confined-zones | data (page 28) | "laboratory sheets 32 of 33 not read" | marked 32 skipped, re-answered | whole answer twice + "1 of 33 pages categorized by the app as laboratory sheets" |
| ufc260-appendices | data (one appendix page) | 1 "exploration log", 20 "laboratory sheets" | marked skipped, re-answered | 14-appendix table twice (5,692 chars against 3,592) + "Exploration logs: 0 of 1 read" |
| fixture-duplicate-page | data (looked at the 2 log pages) | "exploration logs 1 of 3 not read: PDF page 8" | read page 8, re-answered | answer twice + "Exploration logs 3 of 3 read… page 8 was read" (page 8 is the duplicated **text** page) |
| calc-bearing-consistency | declared by the agent | "every page read; state counts" | re-answered | whole answer twice + "Plans, profiles and figures: 1 of 1 read…" |
| ufc260-ch12-tables | declared by the agent (whole manual) | 157 figures, 186 text pages, 83 other… not read | marked 0-189, 267-537 skipped, re-answered | table twice + "Recorded whole-manual coverage: exploration logs 0 of 1…" |

**The cause of the four "data" firings** is in the inventory, not the gate
logic. `Inventory.from_document` (`funhouse_agent/coverage.py:269-309`)
takes planlens' role for every page and ignores its strength. planlens
already marks these pages as weak:
- UFC text pages: `lab_test` with confidence 0.6, "laboratory test name
  'compaction test' on a page of working" (`planlens/document/roles.py:1003-1009`);
- the submittal's page 7: `boring_log` 0.6, "INHERITED from its appendix tab;
  the page says nothing about itself" (`roles.py:1660`).

The `data` arming in `armed()` (`coverage.py:796-832`) then holds the turn to
every such page.

Offline, the inventory over the suite's public documents gives these "data"
pages (logs, lab, field tests):

| document | data pages | of which `text` kind | real data pages |
|---|---|---|---|
| UFC 3-220-04FA (60 pp) | 33 | 30 | 0 |
| UFC 3-220-07 (99 pp) | 7 | 6 | 0 |
| UFC 3-260-02 (538 pp) | 22 | 15 | 0 (worksheets and figures in appendices) |
| UFC 3-301-01 | 0 | — | 0 |
| submittal fixture | 3 | 1 | 2 |
| report fixture | 21 | 0 | 21 |
| Mecklenburg set, tag sets, calc packages | 0 | — | 0 |

**Worth it?** In none of the six did the gate cause a read that the answer
needed:
- 4 answers got worse: a duplicated answer, plus a misleading count about
  "laboratory sheets" in a criteria manual;
- 1 is neutral but duplicated (`calc-bearing-consistency`);
- 1 is the same answer at 12× the cost (`ufc260-ch12-tables`).

The owner's ruling on narrow questions (one extra call, the agent says why
the rest are not needed) held in cost on the four "data" firings: +1 to +2
calls, +18 K to +31 K tokens, except `fixture-duplicate-page`. But the
firings should not have happened.

### 3.2 The answer is delivered twice

`stream_turn` collects every streamed token of a graph pass into one string
(`webapp/core.py:780-799`). The gate jumps back to the model inside the same
pass (`deep/coverage_tools.py:321-351`). So the reply before the note and the
reply after it are concatenated with no break:
- `ufc04-density`: "…“Density requirements.”**The unread pages were not
  needed…";
- `fixture-duplicate-page`: "…among the 11 pages.Yes. **PDF viewer page 8
  duplicates…".

The gate note ends "This note comes once; then finish your answer"
(`coverage.py:893`). The model reads that two ways: 5 of 6 times it
restated the whole answer, once it wrote only an addendum. So simply keeping
the last message would have lost `ufc04-density`'s answer. The run records
and the app show the same doubled text, since `turn_done` carries it.

### 3.3 The agent declares coverage on whole-set questions

`document_coverage` is described as for "a task that takes data out of it
or reviews all of it: … a whole-set review. Call it when such a task starts"
(`deep/coverage_tools.py:89-104`).

**Where Sol declared.** It opened 7 of the 23 ordinary tasks with it:
- sheet index;
- cross-references;
- memo;
- the long tag set;
- both calc checks;
- "list Chapter 12's tables".

Each call declares the **whole document** (`ledger.declare(key)` on every
call, `coverage_tools.py:414`), including the calls made only to mark pages
skipped after a gate.

**Effects:**
- **ufc260-ch12-tables.** The first `document_coverage` result listed all
  538 pages "not read" (L20). The agent's plan became "inspect every Chapter
  12 page" (L27). It sent four general-purpose helpers over pages 190-266:
  49 page looks and 22 `read_document` calls.
  - At the end the gate listed 400+ unread pages of chapters the question
    never touched (L364), and the agent spent a call marking them skipped.
  - Baseline read the list of tables and answered in 5 calls (92 K).
- **calc-wall-check.** After declaring, the agent made 5 deep zooms. They
  produced a fuller review: the stem rectangle's arm should be 0.45 m, not
  0.60 m; 5.0 m active height against 4.4 m drawn. Arguably better, at +45 %.
- **set-sheet-index, set-cross-references, produce-memo.** The same answers
  plus "All 10 of 10 sheets were visually reviewed" — which baseline also
  says, in words.

These 7 voluntary declarations account for +1.86 M input tokens. The 4
gate-only firings account for +0.28 M (§7).

### 3.4 The new tools on ordinary tasks

- **`measure` and `log_grid`:** not called in any of the 46 runs. They did
  not distract.
- **`document_coverage`:** 15 calls in 11 coverage runs, as above.
- **`find_like`:**
  - twice in baseline `produce-circle-tags` (one refused with "pass bbox OR
    view + image_box, not both", one stopped by the flood guard);
  - once in coverage `set-long-rare-tag` (§2).

## 4. Brief 4's fixes A-M on these tasks

| fix | held? | evidence here |
|---|---|---|
| A. markup check judges the target | **yes** | `produce-markup`, both arms: the agent passed `target`, the check answered `same_thing: true, comment_fits: true`, confirmed 1/1 (baseline L46, coverage L46). Brief 4: 0 of 10 comment checks confirmed. |
| B. "encloses" from geometry | **yes** | `produce-circle-tags`: 7/7 rings confirmed in each arm, every look `encloses: true`. The suite scores 7/7 on target in both. |
| C. agent `dpi` never shrinks a zoom | **yes** | 0 of 104 zooms carried `dpi`. Windows of 120 pt or less render at 14-16.7 px/pt. The low px/pt values left (2.3-4 px/pt) are the agent's own wide or tall windows, e.g. 240 × 562 pt title-block strips. |
| D. `annotate_document` forgives guesses | **yes** (nothing to forgive) | 4 calls, 0 refused, 0 adjusted. One `find_like` call was refused for passing both `bbox` and `view` (baseline `produce-circle-tags` L43); the same family of guess, not covered by D. |
| E. say what only the tiles found | **works, noisy** | On the long tag set every tiles-only "FPG" became a zoom (baseline L556-L570), as designed. Elsewhere `found_note` fired on 139 of 181 tiled looks, naming 441 items. 134 of them are lines like "Sheet number/title: Not visible in this tile…" or partial titles across a tile edge, all with "treat these as found, and zoom on each" (`vision_tools.py:927-1007`). `found` + `found_note` are 17 % of page-look result text. |
| F. say how far a zoom's answer is from its aim | **noisy** | `aim_note` fired on 11 of 22 padded zooms. I judge all 11 false: 6 aimed at a leader tip and answered with the same callout's tag 35-51 pt away (baseline `produce-circle-tags` L67-L81); the rest answered with a word inside a long note whose centre is 18-280 pt from the aim. The rule compares centres (`vision_tools.py:865-912`). Sol r3's two-candidate case did not recur. |
| G. best reading | **yes** | 1 bracketed reading ("[8/3]") in 1,342 answers; `reading_note` once. |
| H. `find_like` stops flooding | **half** | The per-page guard stopped a leader-tip example at 400 candidates in 5.5 s, unread (baseline `produce-circle-tags` L47; brief 4 took 289 s). A loose example spread at 41 a page passed it and read 49 sheets in 795 s (§2). |
| I. duplicate marks | **yes** | no duplicates written or flagged |
| J. suite checks where a comment points | **yes** | `produce-markup`: arrow tip 0.0 pt from note 4 in both arms |
| L. reasoning summary; run.json records | **half** | `commits` and `vision_profile` are in every `run.json`. **0 of 1,873 `model_end` records carry `reasoning`** (the README's cause: `SummaryConfig` is not in `_SUMMARY_TYPES`, `webapp/palantir_sdk_engine.py:463-465`). |
| K, M | off / not touched | |

**Pixel boxes:** 866 of 1,342 vision answers gave `px=` boxes and none fell
back to the 0-999 grid. The other 476 located nothing.

## 5. Other findings

**R. Turned lettering.** Eight of the ten Mecklenburg sheets
(10.17A, 10.25A, 20.00A/B, 21.01, 30.00, 30.01, 50.03) are drawn sideways on
portrait pages with no `/Rotate` (checked with PyMuPDF). The vision model is
sent them sideways.

What that costs:
- **Sheet numbers misread on whole-page looks** (tiles read them right):
  "10.1 / REV. 7A" (10.17A), "20.0A", "20.0B", "503", every run of
  `set-3600-psi`, `set-sheet-index` and `set-ncdot-vs-county`.
- **Titles misread on zooms:**
  - "POST STANDARDS FOR USE IN MECKLENBURG COUNTY", L158;
  - "REFERENCE CONTROL MONUMENT", L154;
  - "NCDOT STANDARDS FOR USE IN MECKLENBURG COUNTY TOWNS", L156.

  All three are in `set-sheet-index` coverage. They were fixed only by a
  second zoom whose prompt said "portrait sheet with a rotated vertical title
  block".
- **Note 3 read "MCDOT" in one arm and "NCDOT" in the other.**
- **Effort.** Over the six set tasks in the three runs, page and tile
  answers on sideways sheets have a median of **1,065 output tokens and
  13.4 s**, against **605 tokens and 8.2 s** on the two upright sheets. The
  content differs too, so this is suggestive, not proof.
- **Locations.** One tile on 21.01 put "2'-0" TO 4'-0"" 290 pt from where it
  is in x (baseline L112). The zoom made on that box found a blank (L142).

With §2's turned tags, the pattern is general: **Sol reads turned lettering
markedly worse than upright lettering.**

**T. Stalled side calls.**
- **What stalled:**
  - one tile call in coverage `set-find-bioretention`: **907.5 s** for 586
    output tokens (L152);
  - one tile call in baseline `set-ncdot-vs-county`: 343 s;
  - brief 4 had a 903 s hang.
- **The distribution:** the 2,204 side calls in the three runs have p50
  6 s, p95 39 s and p99 74 s.
- **The cause:** the engine raises every call's HTTP read timeout to 900 s
  for long reasoning (`webapp/palantir_sdk_engine.py:639, 685-699`). Side
  calls are served by the same engine, so one stalled tile holds the whole
  turn for 15 minutes.

**The 11.01 "%" note is a true finding.** Both memo runs flagged that "FINAL
LIFT TO BE APPLIED AFTER MEETING % DEVELOPMENT OCCUPANCY" has no number. A
600 dpi render of the public sheet shows the number is not printed. Baseline
says "omits the numeric percentage" (right). Coverage says "illegible"
(slightly wrong). Coverage's zoom prompt invented a "75%" to test, and the
zoom correctly refused it.

**Author.** Baseline `produce-markup` set `author: "AI Draft Review"`
although the user named no author and the tool note says to leave it alone
(`deep/tools.py:596-598`). The comments still say AI draft, but the
signed-in user's name is lost (`tools.py:1064` takes the agent's value
first).

## 6. Per-run notes

- **ufc04-supersedes, ufc04-table-5-1, ufc07-drilled-shaft-table,
  ufc301-changes, ufc301-asce7-chapters:** the same path and the same answer
  in all three runs. Coverage costs +1 % to +3 % here (the tool description
  is about 250 tokens on every primary call).
- **ufc07-figure-1-1:**
  - Coverage's answer is better. It zoomed the scanned text under Figure 1-1
    and cited the 5° (9 %) creep sentence **on that page**, as the question
    asked (L58-L63).
  - Baseline cited §2-1 on PDF 17.
  - Not coverage's doing: no declaration, no gate.
- **calc-bearing-consistency:** declared on a 5-page package; the gate then
  asked for counts, and the answer was repeated with "1 of 1 read" lines.
- **fixture-markups:**
  - Baseline looked at both pages (16 tiles) though `document_markups`
    answers the question; run 6 and coverage did not.
  - Its page look read the turned reply as "Embedment per Structural is
    updated to 6m". The answer correctly used the markup record's "revised
    to 6 m per updated calcs".
- **fixture-duplicate-page:**
  - Coverage looked at 4 pages first (the "look" pages); baseline answered
    from the page map.
  - The gate then sent it to page 7.
- **produce-circle-tags:**
  - **Baseline.** The page answer boxed the leader tips, not the tags. One
    `find_like` was refused for a guess and the next was stopped by the flood
    guard. The 8 padded zooms answered with the tag text (6 false
    `aim_note`s). QCE was rejected at zoom.
  - **Coverage.** Its first 5 zooms asked for "circular GCE tags" and all
    answered "not enclosed in a circle" (L60-L66), a wasted round from the
    agent's own reading of "circle every GCE tag". A tile called the turned
    GCE "legend"; a zoom corrected it (L106).
  - Both arms: 7/7 rings on target.
- **produce-markup:** one zoom pair and one `annotate_document` call each;
  the comment was confirmed by the new check (A).
- **produce-memo:** same three standards cited in all three runs.
  Coverage's horizontal zoom windows cut the vertical notes on sideways
  sheets ("cropped at the bottom"), so it zoomed twice more (R).
- **set-3600-psi:**
  - All three runs name 30.01 as "3600 cu ft per acre, not concrete" and
    score precision 0.75.
  - `check_set_match` counts a sheet named to rule it out by design
    (`review_eval/checks.py:110-128`), so this is not a fault.
  - Coverage added 3 verifying zooms (2 false `aim_note`s).
- **set-find-bioretention:**
  - Baseline: one zoom from a tile box landed on a blank (R, location); a
    wider zoom read the ponding note; it answered from the page read.
  - Coverage: the 907 s stall (T), then a correct zoom read "2'-0" TO 4'-0"
    FILTER MEDIA" (L160).
- **set-ncdot-vs-county:** baseline's 413 s is the 343 s stalled tile.
- **set-cross-references:** same 15-17 references in both arms. Both treat
  10.17A as not the cited "STD. NO. 10.17", which is a judgement. No hung
  call this time.
- **set-sheet-index:** the sideways title-block misreads, then
  self-correction (R). Both arms end with the right 10-row index.
- **calc-wall-check:**
  - Both arms: sliding fails, the printed 460.1/166.7 is not 2.733, and the
    bearing "99.9 / OK" is unsupported.
  - Coverage adds the stem-arm error and the 5.0 m against 4.4 m height.
- **ufc260-appendices, ufc260-ch12-tables, ufc04-confined-zones,
  ufc04-density:** see §3.1-3.3.

## 7. Part C cost split (all 42 tasks, from `run.json` / `results_baseline_coverage.json`)

**Totals:**
- input 8.18 M → 12.00 M (+3.82 M, +47 %);
- output 0.78 M → 1.21 M;
- time 4,508 s → 6,544 s (+2,036 s, +45 %);
- model calls 1,231 → 1,660.

"Engaged" means the coverage arm called `document_coverage` or (in this
slice's traces) the gate fired. For tasks outside this slice, "not engaged"
means no `document_coverage` call in `run.json`.

| group | tasks | Δ input | share of +input | Δ output | Δ seconds | share of +time | Δ input vs its baseline |
|---|---|---|---|---|---|---|---|
| A. extraction (`report-*`) | 3 | +1,504 K | 39 % | +147 K | +375 | 18 % | +87 % |
| B. ordinary, coverage engaged | 11 | +2,140 K | 56 % | +274 K | +1,072 | 53 % | +63 % |
| — of which the agent declared it (7) | | +1,862 K | 49 % | | | | |
| — of which only the gate fired (4) | | +278 K | 7 % | | | | |
| C. ordinary, not engaged | 28 | +180 K | 5 % | +12 K | +589 | 29 % | +6 % |

**The extra is concentrated, not spread.**
- **Input:** a third of the extra (+1.0 M) is one ordinary task,
  `ufc260-ch12-tables`. Extraction plus the seven declared tasks are 88 %.
- **Time:** two single events are 84 % of the extra time, and neither is
  the gate:
  - the 907 s stalled tile (group C);
  - the 795 s `find_like` (group B, an agent tool choice).

  Without them group C is −318 s (noise; it includes baseline's own 343 s
  stall) and the arm is +334 s (+7 %).
- **Group C** (+6 % input) is at noise level. It includes the tool
  description, about 250 tokens on every primary call.

Per task, sorted by the input delta:

| task | input K, base → cov (Δ) | Δ output K | Δ s | Δ calls | where it went |
|---|---|---|---|---|---|
| ufc260-ch12-tables | 92 → 1090 (+997) | +23.6 | +107 | +85 | B: declared at start → 4 helpers, 49 page looks; gate on the whole 538-page manual → skip + re-answer |
| report-extract-diggs | 772 → 1680 (+908) | +124.1 | +291 | +188 | A: extraction; `document_coverage` ×6 |
| set-long-rare-tag | 1128 → 1520 (+392) | +209.1 | +806 | +58 | B: declared; `find_like` 795 s, 49 reads (not the gate) |
| report-extract-all | 711 → 1084 (+374) | +22.5 | +16 | +39 | A: extraction; `document_coverage` ×11 |
| report-summary-vs-sheets | 252 → 474 (+222) | +0.3 | +67 | +11 | A: extraction; `document_coverage` ×3 |
| produce-memo | 653 → 857 (+205) | +30.1 | +34 | +6 | B: declared; +4 zooms (vertical notes on sideways sheets) |
| fixture-duplicate-page | 32 → 154 (+122) | +5.8 | +42 | +21 | B: 4 page looks (agent); gate on a mislabelled "log" page → read + re-answer |
| calc-wall-check | 270 → 392 (+122) | +13.6 | +86 | +7 | B: declared; +5 deep zooms (fuller review) |
| ufc260-appendices | 87 → 189 (+102) | +1.0 | +8 | +2 | B: gate on mislabelled "lab/log" pages → skip + re-answer (answer doubled) |
| produce-circle-tags | 190 → 287 (+97) | +5.1 | +64 | +11 | C: first zoom round asked for "circular" tags; 15 zooms against 8 + 2 `find_like` |
| set-sheet-index | 321 → 403 (+82) | −3.9 | +15 | +5 | B: declared; +2 zooms |
| set-3600-psi | 266 → 328 (+62) | +17.3 | +112 | +5 | C: +3 verifying zooms |
| set-cross-references | 630 → 676 (+46) | −5.3 | −46 | 0 | B: declared; −1 zoom |
| meck-trap-dimensions | 90 → 135 (+45) | +4.0 | +62 | +5 | C: no `document_coverage` call |
| meck-underdrain | 97 → 135 (+37) | +5.0 | +51 | +4 | C: no `document_coverage` call |
| ufc07-figure-1-1 | 107 → 139 (+32) | +1.8 | +10 | +3 | C: one more look and a zoom (better answer) |
| ufc04-confined-zones | 69 → 100 (+31) | +0.1 | +12 | +2 | B: gate on mislabelled "lab" pages → skip + re-answer (answer doubled) |
| ufc04-density | 62 → 85 (+23) | +0.1 | 0 | 0 | B: gate on mislabelled "lab" pages → re-answer (glued on) |
| scale-plan-distance | 80 → 101 (+21) | −0.7 | −6 | +1 | C: no `document_coverage` call |
| meck-pavement-section | 84 → 104 (+20) | +0.8 | +12 | +2 | C: no `document_coverage` call |
| calc-bearing-consistency | 33 → 51 (+18) | +0.1 | +10 | +1 | B: declared; gate asked for counts → re-answer (answer doubled) |
| meck-driveway-notes | 64 → 83 (+18) | −3.1 | −53 | +3 | C: no `document_coverage` call |
| meck-sediment-trap-criteria | 64 → 80 (+16) | −2.1 | +11 | +2 | C: no `document_coverage` call |
| set-find-bioretention | 343 → 351 (+8) | +14.8 | +934 | +2 | C: one tile call stalled 907 s (host); tokens level |
| ufc07-drilled-shaft-table | 119 → 123 (+4) | 0.0 | −14 | 0 | C |
| produce-markup | 106 → 108 (+2) | +1.2 | +5 | 0 | C |
| meck-ramp-detail-callouts | 84 → 86 (+1) | −2.3 | −28 | 0 | C |
| ufc04-table-5-1 | 61 → 62 (+1) | +0.4 | −1 | 0 | C |
| meck-revision-block | 76 → 77 (+1) | −0.4 | −5 | 0 | C |
| ufc04-supersedes | 33 → 34 (+1) | 0.0 | −1 | 0 | C |
| ufc301-changes | 40 → 41 (0) | +0.1 | +7 | 0 | C |
| meck-curb-types | 63 → 63 (0) | +2.1 | +19 | 0 | C |
| meck-ramp-warning-mat | 67 → 66 (−1) | −4.1 | −35 | 0 | C |
| meck-ramp-slopes | 65 → 64 (−1) | +0.2 | −7 | 0 | C |
| ufc301-asce7-chapters | 98 → 96 (−2) | +0.2 | +4 | −1 | C |
| meck-bioretention-section-dims | 98 → 93 (−4) | +3.8 | −23 | +1 | C |
| set-ncdot-vs-county | 281 → 276 (−5) | −6.3 | −299 | 0 | C: baseline had a 343 s stalled tile call |
| meck-row-sidewalk | 82 → 64 (−17) | +0.2 | −8 | −3 | C |
| meck-bioretention-access | 85 → 65 (−20) | −6.7 | −39 | −2 | C |
| scale-log-depth | 113 → 91 (−22) | −0.8 | −18 | −2 | C |
| meck-monument | 91 → 66 (−25) | −5.2 | −58 | −3 | C |
| fixture-markups | 122 → 32 (−90) | −13.1 | −109 | −24 | C: baseline looked at both pages (16 tiles), coverage did not |

## 8. Fixes

None is fitted to one sheet or one task. Classes:
- **P:** plumbing or code; reproducible offline with a fake model or any
  model.
- **M:** model behaviour, measurable only on Foundry.
- **S:** scorer or task.

**The Claude column.** Claude numbers never count as a measurement of GPT
or Sol. "y" means a cheap local live check (Haiku 5.5, public documents)
would exercise a P path that a fake model cannot.

| id | problem | class | code site (39716a8 / c5bdb8d) | proposed general fix | how to measure | Claude check? |
|---|---|---|---|---|---|---|
| CV1 | The gate arms on prose pages that planlens marks as **weak** lab or log pages (0.6, "on a page of working", "inherited from its tab"). 4 of 6 firings here; the answers then report "laboratory sheets: 3 of 33 read" in criteria manuals. | P | `funhouse_agent/coverage.py:269-309` (`Inventory.from_document` drops role strength), `:796-832` (`armed`); planlens `document/roles.py:1003-1009, 1656-1680` | Keep the role's confidence and source in `PageInfo`. Let only self-declared, non-weak data pages arm the implicit `data` path, and count weak or inherited ones as "other" for arming (a declared or written-out extraction still covers them). | No model: rerun the inventory over the suite's public documents. Expect UFC 3-220-04 33→0, 3-220-07 7→0, 3-260-02 22→~0, submittal 3→2, report fixture 21→21. A fake-model replay of the four narrow tasks should give 0 gate events. | n |
| CV2 | When the gate fires, the delivered answer is the pre-gate reply plus the post-gate reply, joined with no break: the whole answer twice in 5 of 6, glued on mid-line once. | P | `webapp/core.py:780-799` (`_run_passes` joins every token of a pass); gate note `coverage.py:887-893` | Make the contract explicit both ways. The note says "your reply above will be replaced: give the complete answer, with the coverage line". `stream_turn` drops the text streamed before a `coverage_gate` event of the same pass (or, if the post-gate reply lacks it, keeps it as a separate paragraph). | Offline: a fake model that answers, gets the note and answers again; the delivered answer is the final one, once. On Foundry: count delivered answers containing the same paragraph twice (target 0). | n |
| CV3 | A declaration has no page scope. Every `document_coverage` call declares the **whole** document (also calls made only to mark skips), so a chapter task on a 538-page manual is held to 400+ pages. | P | `deep/coverage_tools.py:398-427` (`declare` at :414); `coverage.py:796-832` | Add a `pages` scope to a declaration (the targets are the scope's pages). Calls that only mark `extracted` / `skipped` should not (re)declare. | Fake-model replay of `ufc260-ch12-tables` with `pages="190-266"`: the gate lists nothing outside 190-266. Unit test that a mark-only call leaves `declared` unchanged. | n |
| CV4 | Sol opens ordinary whole-set questions with `document_coverage` (7 of 23 tasks). That is +1.86 M input tokens, and on `ufc260-ch12-tables` a 12× plan (4 helpers, 49 looks) for an answer baseline gave in 5 calls. | M (lever: the tool's own description) | `deep/coverage_tools.py:89-104` (`DOCUMENT_COVERAGE_DESCRIPTION`: "or reviews all of it … a whole-set review") | The tool describes itself as for **taking data out of a document** (all logs, all lab sheets, a DIGGS file, a table of every value), and says a question answered from a list of titles or a few pages does not need it. No prompt rule. | Foundry, the same 23 ordinary tasks, coverage arm: number of declarations (target 1-2: produce-memo, set-sheet-index at most), and input versus baseline (target within +10 %). Extraction tasks must still declare. | n |
| R | Turned lettering is read badly: 6 of 15 turned look-alike tags zoomed were read as FPG (0 of 11 upright), and 8 sideways Mecklenburg sheets lose sheet-number characters and title words, needing an extra zoom round at 1.6× output tokens per call. | M (lever P) | `funhouse_agent/vision_view.py:353` (`render_view`), `vision_tools.py:748` (`_dispatch_render_region`), `:1241` (`_dispatch_analyze_pdf_page`) | Render upright. Use the text layer's line direction where there is one. Where there is none, ask the look for the lettering direction or let a zoom answer that says "vertical/rotated" trigger one re-render turned 90° (the image is rotated, the boxes are converted back). Offer `rotate` on `render_region` in its own description. | Offline: boxes from a rotated render convert back exactly (fixture). Foundry: long tag set ×3, share of zoomed turned look-alikes read as FPG (6 of 15 now; target ≤ 1 in 15), and Mecklenburg set sheet-number reads on sideways sheets plus output tokens per call against upright. | y, only if orientation is found by a vision call: Haiku on the public Mecklenburg sheets checks the detection path (≈ $0.05) |
| T | A stalled side call holds the turn up to 15 min: 907 s, 343 s here, 903 s in brief 4. The p99 side call is 74 s. | P | `webapp/palantir_sdk_engine.py:639, 685-699` (`read_timeout_s=900` raised on every call, side calls included) | Give vision side calls their own read timeout (about 150 s), with one retry, and keep 900 s for the primary's long reasoning. A tile that still fails is reported as unread, not waited on. | Offline: a fake SDK handle that never answers; the tile returns as unread within the side-call budget, and the turn completes. Foundry: count side calls over 300 s (target 0). | n |
| H2 | `find_like` with a loose example (callout plus leader) gave 978 candidates at 41 a page, under the per-page flood guard. It read 49 sheets in 795 s for 3 true of 28 "instances". | P | `funhouse_agent/find_like.py:57` (`FLOOD_PER_PAGE`), read pass `:157-205` | Add a whole-search budget: past about 100 candidates in total, or more than k × the pages searched, do not read. Return the count and "box only the lettering of one copy". Also warn when the example box is much taller than lettering the page holds (here 18.7 pt for 4.3 pt letters). | No model: replay planlens `find_like` on the long tag set with the coverage arm's box (p3 [592.8,364.3,639.7,382.7]). Candidates, and whether the read pass would run. Foundry: `find_like` seconds per call. | n |
| E2 | `found_note` names 441 "tiles-only" things in 139 of 181 tiled looks. 30 % are lines saying nothing is there, and many are partial titles cut by a tile edge, each with "treat these as found, and zoom". | P | `vision_tools.py:927-1007` (`_merge_found`, built on `vision_view.answer_boxes` `:789`) | Count as found only boxed things whose line names something present: drop negated or "not visible / cropped / partial" lines. Merge a tile's partial reading into the page item it overlaps. Name tiles-only items in the note only when they carry a code or a value. | Offline replay over this run's saved tile answers: tiles-only items 441 → (expect < 100). Every tiles-only FPG candidate on the long tag set must remain. | n |
| F2 | `aim_note` fired on 11 of 22 padded zooms, all false: the answer named the aimed callout's tag, or a word inside the aimed note. | P | `vision_tools.py:865-912` (`_say_how_far_from_the_aim`, centre to centre), `vision_view.py:760` (`aim_tolerance`) | Measure the gap between boxes, 0 when they overlap or one holds the other, against the aimed box padded by the source view's error. Warn only when no answer box comes within it, or when two boxes do and they carry different labels. | Offline replay of the 22 padded zooms from their saved `aim` and answer boxes: notes 11 → 0. Brief 4's Sol r3 two-candidate zoom must still warn. | n |
| L2 | No reasoning summary is recorded: 0 of 1,873 `model_end` records. | P | `webapp/palantir_sdk_engine.py:463-465` (`_REASONING_TYPES`, `_SUMMARY_TYPES`) | As the hand-back README found: put `SummaryConfig` first in `_SUMMARY_TYPES` and `ReasoningConfig` first in `_REASONING_TYPES`. | First Foundry call: `generation_info.reasoning_summary == "requested"` and `model_end.reasoning` present. | n |
| AU | The agent overrode `author` without being asked ("AI Draft Review"), losing the signed-in user's name on the comments. | P | `deep/tools.py:1064` (agent's `author` wins), note `:596-598` | Keep the app's identity and the "(AI draft)" suffix always. An agent-given author is accepted only as an addition ("for <name>"), never as a replacement. | Unit test: `annotate_document(author="X")` writes "<user> via GeotechStaffEngineer (AI draft)". | n |
| S1 | `set-long-rare-tag` is a coin per turned look-alike: precision 0.75 / 0.50 / 1.00 in three runs, decided by which turned FBGs the page reads flag. Its pass in the coverage arm is not evidence for the arm, and the check ignores the per-page counts the answer states ("two on page 4"). | S | `funhouse_agent/review_eval/tasks.py` (`set-long-rare-tag`, `pages_listed` check) | Run this task ×3 per arm and report the mean precision. Score separately how each zoomed turned look-alike was read, from the trace. Optionally check stated counts against truth. | Rescore saved runs: no model. | n |
