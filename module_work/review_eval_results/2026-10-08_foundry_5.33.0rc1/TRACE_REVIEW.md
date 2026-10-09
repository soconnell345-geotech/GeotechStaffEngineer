# Foundry brief 5 (5.33.0rc1 + planlens 0.13.0rc1): every run read in full

## What ran

The AI FDE ran brief 5 on Palantir Foundry on 2026-10-08.

**Test wheels:**
- **geotech-staff-engineer 5.33.0rc1:** app `39716a8`, which is master's code.
- **planlens 0.13.0rc1:** `c5bdb8d`.

**Environment and setup:**
- numpy 1.26.4 and Python 3.12.14.
- The package's own `PalantirSdkChatModel(route="responses")`.
- `GEOTECH_VISION_*` unset, except part F's patch alignment.
- Every setup check passed. No traceback came from inside the package.

| part | what | GPT-5.4 | GPT-5.6 Sol |
|---|---|---|---|
| A | label crops (needs_values), 3 fixtures, 32 labels | 32/32 | 32/32 |
| B | `produce-circle-tags` + `produce-markup`, × 3 | circles 2/3, markup 1/3 by the suite | circles 2/3, markup 3/3 |
| C | the suite, 42 tasks, `baseline` and `coverage`; `checklist` on the 3 `report-` tasks | — | baseline 39/42, coverage 39/42, checklist 1/3 |
| D | the 5 new tasks, `baseline` / `coverage` / `checklist` | 2/5 in every arm | — |
| E | GEC-12 Fig 7-15, second reading | 5 reads | 5 reads |
| F | 32-px patch alignment, off / on (location re-measure) | one sample per look | — |
| G | report ingest: G1 rescore of brief 4's run files (no model); G2 visual scales off / on (GPT-5.4) | G2 run | — |

**Sources.**
- **Raw hand-back (git-ignored):** `module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/`. Part G's RESULTS stay there; this review quotes only report IDs, counts and rates.
- **Comparison runs:** run 6, in brief 4 raw at `module_work/field_feedback/2026-10-08_foundry_brief4_5.32.1rc3/raw/`.

**Method.** Four Opus readers read every event of every agent run: 12 in part B, 87 in part C and 15 in part D. Each wrote one slice:

| file | slice |
|---|---|
| [TRACE_REVIEW_new_tasks.md](TRACE_REVIEW_new_tasks.md) | the 5 new tasks, both models, every arm (28 runs, 3,733 records) |
| [TRACE_REVIEW_suite_meck.md](TRACE_REVIEW_suite_meck.md) | the 14 Mecklenburg sheet tasks, both arms, against run 6 (28 runs) |
| [TRACE_REVIEW_suite_rest.md](TRACE_REVIEW_suite_rest.md) | the other 23 older tasks against run 6 (46 runs), plus the coverage arm's cost split over all 42 |
| [TRACE_REVIEW_B_AEFG.md](TRACE_REVIEW_B_AEFG.md) | part B (12 runs, every ring against truth), A, E (Fig 7-15 measured with no model), F, G and the engine |

Fix IDs below are the slice files' own IDs.

## In short

1. **A wrong reference table is in the released app. This is the most important finding.**
   - GEC-12 Fig 7-15 (Meyerhof limiting toe resistance) was measured off the figure with no model, and both models agree with that reading.
   - The table is **+41 % at φ 30°, +25 % at 32°, +11 % at 34°**, right at 36°, and 3–6 % low from 38° to 43°.
   - Its nodes at 26, 28, 44 and 45° are not on the chart: the axis starts at 30° and the curve ends at 43.75°.
   - The same table, in kPa, sets the Nordlund toe limit in `axial_pile/nordlund.py:312-318`. That limit is unconservative for loose sands.
   - Exact 1° correction: B_AEFG §3.2 (E5).
2. **Most of the new tasks' failures are the harness, not the models.**
   - **The DIGGS file:**
     - It was written and was schema-valid in 6 of 6 runs, always to `/tmp`, because the tool's own example says `/tmp`.
     - The suite runner collects only `files/`. The app's turn job copies a reported `output_path`, the runner does not.
     - Other `call_agent` writers keep the model's path too. The DXF export is not rescued even in the app (N1, N2).
   - **The "last sheets' values":** they are in every delivered table. The check reads only the answer text (N5).
3. **The `write_diggs` cross-checks (W2) work when they get the data.**
   - **Sol:** the planted LL/PL swap was caught 3 of 3 times and relayed in every answer.
   - **GPT-5.4:** caught 0 of 3 times.
     - Twice it "corrected" the summary row before writing, which is model behaviour.
     - Once the summary was refused because the tool's kind list omits `summary_table` (N9). The checklist then passed that run.
4. **Coverage (switch OFF) is not ready to switch on.**
   - **What it does well:** it makes extraction answers state their counts (8 of 8 runs, against 0 of 4 without it).
   - **Every gated answer reaches the user doubled, or glued mid-line:** 10 of 10 on the new tasks, 5 of 6 on ordinary tasks (CV2 / N4, `webapp/core.py:780-799`).
   - **It fired on 6 ordinary tasks and helped none.**
     - 4 of those came from planlens page roles that call prose pages "laboratory sheets" or "logs": 33 of 60 pages of UFC 3-220-04 (CV1).
     - Sol also opened 7 ordinary tasks with its own `document_coverage` declaration. A declaration has no page scope, so `ufc260-ch12-tables` went from 92 K to 1.09 M tokens for the same answer (CV3, CV4).
   - **Cost:** +3.82 M input (+47 %). 39 % of that went to the 3 extraction tasks and 56 % to the 11 ordinary tasks where coverage engaged. On the other 28 tasks it cost nothing.
5. **The checklist (switch OFF) is not ready.**
   - GPT-5.4 never called it (0 of 5 runs).
   - Its code checks gave a false fail, a false pass, and a stuck "unsure" (N10–N13).
   - This goes to the owner's checklist session.
6. **Ingest visual scales (switch OFF) cannot be measured on this corpus.**
   - Logs went 466 → 468 of 521, and that is variance.
   - Blind `layer_top` stayed at 22 of 36, and no label call was made.
   - **Why:** the voters act only on scans whose labels have no text, and Azure DI gives this corpus's scans text.
   - Leave it off. G3 prints the voters' counts next time.
7. **Visual scales on the Document Review page:**
   - **Sol:** used them and passed both new tasks (3.60 m against 3.62; 183.85 ± 0.12 ft).
   - **GPT-5.4:** used `measure` 3 of 3 on the plan, but never on the log. It read 3.8–4.0 m by eye (M).
   - **Two planlens faults:**
     - `log_grid` reads a table's row rules as stratum lines (N8).
     - The raster gate (image coverage ≥ 0.5) blinds `find_scales` on textbook pages that hold two figures, so `code_reading` never measured Fig 7-15 (E1–E3).
8. **Markups.**
   - **The circles:** GPT-5.4 marked 6, 6, 5 of 7; Sol 7, 2, 7. Neither model fails consistently: each has an occasional failed run.
   - **GPT-5.4's markup arrow is not a coordinate bug.** Both wrong tips are the centre of a different label, "RAMP SLOPE UP TO 7.5% (8.3% MAX.)", to within 1 pt. `found` dropped the tile that boxed note 4 (B1). Then the check confirmed the comment against the agent's own `target` text (B3).
   - **The check is wrong in both directions:** Sol r2's 13 exact rings were rejected, and 12 of 12 over-large rings were confirmed.
9. **Older suite tasks: no outcome changed against run 6** (37 of 37, except one task that varies between runs).
   - **Sideways lettering is the main cause of the remaining misreads:**
     - 8 of the 10 Mecklenburg sheets are drawn sideways.
     - 6 of 15 turned look-alike tags were misread, against 0 of 11 upright ones (R).
   - **Two of brief 4's new notes are mostly false alarms:**
     - `aim_note` was false on 11 of 22 padded zooms (F2).
     - `found_note` is often noise (E2).
10. **The engine never requested a reasoning summary** (L1).
    - The first-match name lookup picks the wrong SDK classes. It should use `ReasoningConfig(summary=SummaryConfig.AUTO)`.
    - The offline test fakes the same wrong names, which is why it passed.
11. **Report ingest, brief 4's open items (G1, with no model call):**
    - **Blind logs index 0/16:** the floor never held those values, so nothing was overruled. Per-value evidence is not needed.
    - **Gradation:** 3 of the 4 collapsed sheets are link failures and one is a reading failure.
    - **Linking flips from run to run:** one sheet linked 9 of 10 in one arm and 1 of 10 in the other.
12. **Patch alignment (part F) is undecided.** It looks like the predicted pure stretch: y scale 1.010–1.015 off, 0.994–1.000 on. But with three samples per arm it doesn't beat the noise (F1).
13. **Side calls can hang for 15 minutes.**
    - They inherit the engine's 900 s read timeout. Stalls were seen at 907 s and 343 s, while the p99 side call is 74 s (T).
    - One `find_like` run took 795 s (H2).

## Switch verdicts

| switch | verdict | what has to happen first |
|---|---|---|
| `GEOTECH_COVERAGE` | **stay OFF** | CV1 (weak roles must not arm it), CV2 (no doubled answer), CV3 (declarations need a page scope), N6 (`log_grid` reads count); then re-measure |
| `GEOTECH_REVIEW_CHECKLIST` | **stay OFF** | the owner's checklist session (week of 2026-10-12), then N10–N13 |
| `GEOTECH_INGEST_VISUAL_SCALES` | **stay OFF** | nothing to measure on a DI corpus; G3 prints its counts. Measure on a scan set without DI, or never |
| `GEOTECH_VISION_PATCH_ALIGN` | **stay OFF** | F1: a stretch statistic and ≥ 10 repeats per arm |

## The fix list

Classes:
- **P:** a code bug, reproducible offline.
- **S:** the scorer or the task is wrong.
- **M:** model behaviour; only Foundry measures it.

### 1. In released code, and a user would hit it

| id | fix | class |
|---|---|---|
| **E5** | **DONE 2026-10-08** (refs `d9dab7a`, app `7d8771e`). Fig 7-15 replaced by the measured 1° nodes in `geotech_references/gec_12/figures.py` and `axial_pile/nordlund.py`. Owner's rule: the reference refuses outside 30–43.75°, and Nordlund completes at the end value with a warning carried on `AxialPileResult.warnings`. V-001's toe is now +0.3 % against the published plateau (was −4.6 %). Reaches users with the next release | P |
| N2 | Every `call_agent` writer saves into the conversation's files folder, whatever path is given. Fix the DXF export, which is not copied even in the app (`dxf_export.py:67`) | P |
| N3 | An unknown argument (`pages=12`) is refused with the right name, never silently dropped (`deep/tools.py:808,1293`) | P |
| T | Side calls get their own read timeout, near the p99, and retry once (`palantir_sdk_engine.py:639`) | P |
| H2 | `find_like` gets a budget for the whole search (`find_like.py:57`) | P |
| N9 | `write_diggs`' kind list takes `summary_table` (it is in the description but refused) | P |
| AU | The comment author is the app's, not the agent's choice (`deep/tools.py:1064`) | P |
| L1 | Request reasoning summaries with the SDK's real class names; the test fakes the real names | P |

### 2. Before coverage can be switched on

| id | fix | class |
|---|---|---|
| CV2 / N4 | A gated turn delivers one answer, not the draft glued to the second reply (`webapp/core.py:780-799`) | P |
| CV1 | Only strong page roles arm the gate; planlens' weak roles do not (`coverage.py:269-309, 796-832`) | P |
| CV3 | A coverage declaration carries a page scope; calls do not re-declare (`coverage_tools.py:398-427`) | P |
| CV4 | The tool description keeps `document_coverage` to extraction-shaped asks | M |
| N6 | Reads through `log_grid` count as reads in the ledger and the scorer (`coverage.py:81-83`) | S/P |

### 3. Vision and markups

| id | fix | class |
|---|---|---|
| R | Render turned lettering upright: detect it from the text layer's direction where there is one, and rotate the view before the look (`vision_view.py:353`) | M, with a P lever |
| B1 | `found` judges region against text line by aspect, not width > 300 on the grid | P |
| B2 | `found` rows carry the text read, not "?" or "Tag" | P |
| B3 | The markup check judges size and enclosure against the label on the page, not against the agent's `target` | P |
| F2 / MK1 | `aim_note` measures from the aim box, not region centres | P |
| E2 / MK2 | `found_note` matches on quoted text and drops "nothing here" lines | P |
| MK3, MK5 | Say when a page look and a tile disagree, and when a zoom is coarser than the tiles already read | P |

### 4. planlens visual scales

| id | fix | class |
|---|---|---|
| N8 | `log_grid` does not read a table's row rules as stratum lines (`loggrid.py:1800`) | P |
| N7 | A multi-log `log_grid` call gives each log its own header fields (`loggrid.py:2186`) | P |
| E1 | The raster decision is per region (the figure's image), not per page (`raster.py:282-291`, `scalefinder.py:449-456`) | P |
| E2b | Find a y scale whose gridlines are labelled only every 4th line | P |
| E3 | `code_reading` on a single-curve chart follows the input line, not ±6 pt of the model's box (`chart_reading.py:62,240,270`) | P |
| N14 | `measure`'s `to` takes a second view + box | P |
| G2 | Bind log index cells to the nearest sample | P |

### 5. Scorer and suite

| id | fix | class |
|---|---|---|
| N1 | The runner copies a reported `output_path` into `files/`, as the app does (`runner.py:222,248`) | S |
| N5 | The values and borings checks read the delivered files too, and read ranges ("BH-1 to BH-3"). Drop "shell" | S |
| S1 | Tasks that vary between runs (`set-long-rare-tag`, circles) are judged over repeats | S |
| F1 | Part F: score a stretch, not an intercept; ≥ 10 repeats | S |
| G3 | Ingest RESULTS print the visual-scales counts and the QA entries | P |

### 6. Model habits: Foundry only, with no prompt rule fitted to them

| item | behaviour |
|---|---|
| M1 | GPT-5.4 reads log depths by eye |
| M2 | GPT-5.4 "corrects" a summary table before writing, and invents layers |
| M3 | GPT-5.4 scopes the scans out |
| M4 | Sol omits the scale warning once |
| B4 / B5 | GPT-5.4 follows the wrong note |
| MK4 / MK6 | Sol completes cut-off text, and once gave boxes in a rotated frame |

Watch these on the next round, and act only through general fixes such as R, B1 and B3.

## Where the Claude API credit helps

Almost everything above replays offline from these traces, at no cost: a scripted fake model can reproduce every call shape. Claude numbers never stand in for GPT or Sol. The credit is worth spending on three things only:

1. **B3, the markup check against the label:** Sonnet 5.5 on the public tag fixture, a few runs, under $1. A fake look cannot show whether a real model's rings and comments are now judged right.
2. **R, turned lettering:** only if orientation has to come from a vision call; where the text layer gives the direction, it is free. Haiku 5.5, cents.
3. **A smoke run before brief 6:** the next rc on Haiku 5.5 over the touched tasks, about $1–3, to catch crashes before a day-long Foundry round.

## Next

1. ~~E5~~ DONE 2026-10-08, with the owner's rule: the reference refuses, and Nordlund warns.
2. **Build groups 1–2,** plus N1 and N5 so the suite tells the truth. Each change is tested offline from these traces.
3. **Then groups 3–4.** Re-measure on Foundry (brief 6) with repeats where results vary between runs.
4. **The checklist** waits for the owner's session.
