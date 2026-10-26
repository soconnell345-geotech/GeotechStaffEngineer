# Foundry brief, model-first waves (RI-87): DRAFT

**Cost, in tokens first.** MF-1 is about 4.5 to 5 M tokens in and 0.9 M
out (about $30 to $33 at the API rates of the local ledger, §7). MF-2 is
about 30 to 52 M in and 5 to 10 M out (about $120 to $230). MF-3 runs only
on the owner's word: about 28 to 48 M in more (about $120 to $220).
Foundry's meter reads $0 for these models: a Foundry model name has no
price, so `max_total_dollars` never stops a run. The per-report token check
in §5 is the only brake on spend.

A draft of the model-first section of the INF-40 brief (plan §8.7.1 and
§8.8). It covers what to run, the settings, the calls and tokens to expect,
where to stop and what comes back. Packaging is the run package's job and is
not covered here. Reports are named by id only. The estimates come from the
saved measurements listed in §7, and the wave's own token counts replace
them.

MF-2 runs on wave 2's six reports so the two waves line up, but it does
not rely on wave 2's results: wave 2 may stop early on the 10-23 build, and
the off arm run on the new build in this wave is the baseline. The models,
decisions D1 to D3, are in §2.

## 1. Before the wave

- **MF-0, the probe:** `report_ingest.model_first.probe_pdf_route(chat_model)`,
  two calls each, about 8 calls in all:
  - Opus 5.5 on the proxy class and on the `palantir_models` class, for the
    PDF route. Set `GEOTECH_FOUNDRY_CLAUDE_PDF=1` only if the proxy brought
    back the code word from the PDF. Without it, the model-first arms send
    pictures with each page's text (`page_route="auto"` resolves to
    `pictures`).
  - Sonnet 5.5 and Haiku 5.5 on the class the wave uses. A pictures call
    that brings back the picture word shows the model is offered and reads
    pictures. If Haiku passes, it describes the pages; if not, Sonnet does
    (D2). If Sonnet does not pass, stop and send the probe back: Sonnet
    reads the logs and printouts and is the off arm's vision voter (D1,
    D3).
- **Ask the AI FDE how to bypass the response cache.** Wave 1 served 12
  repeat calls from it. MF-2's off arm sends wave 2's calls again on wave
  2's reports, so the cache could answer them with wave 2's replies.
- **Do not set `GEOTECH_INGEST_MODEL_FIRST`** in the environment. Each cell
  passes `model_first="off"` or `"on"`, the run resolves it once, and every
  run file records it.

## 2. Settings for every arm

- **Models.** Both arms of a comparison read each stage on the same model
  (D3), so a difference between them is the method's, not the model's.
  - `stage_models={"logs": "<Sonnet 5.5>", "lab": "<Opus 5.5>", "calc":
    "<Sonnet 5.5>", "narrative": "<Opus 5.5>"}` in both arms of MF-2 and
    MF-3. Under `off`, `stage_models` puts today's reader of each stage on
    its model. The triage, the label review, the soundings and the bound
    reports' identity call read on `model="<Opus 5.5>"`.
  - Logs on Sonnet and lab on Opus: RI-78. Logs came out level on the two
    models at half the cost on Sonnet; lab gained 26 values on Opus and
    held it on the repeat.
  - Calc on Sonnet: plan §8.7.1. RI-78 found the two models level on calc
    (158 and 158) and left the choice until RI-74 (5).
  - Narrative on Opus: the owner, 2026-10-25. The model-first narrative
    asks every question in one call, and in the owner's runs only Opus
    managed that; Sonnet needed one call per category. RI-78 found today's
    narrative reader level on the two models, so the off arm loses nothing
    on Opus.
  - **D1:** the off arm's vision voter (`vision_chat_model`) is Sonnet 5.5,
    unless MF-0 shows Sonnet is not offered.
  - **D2:** `describe_model` is Haiku 5.5 if MF-0 shows Foundry offers it
    (RI-73: as good as Sonnet at describing), else Sonnet 5.5.
  - `range_model="<Opus 5.5>"`.
- `GEOTECH_INGEST_LAB_LINK_FROM_LABEL=1` in both arms (RI-55's winner). It
  changes only today's lab reader's prompt, so under `on` it does nothing.
  RESULTS.md says so.
- **One `out_dir` per arm.** A run refuses to read or overwrite an item
  saved in the other arm: the item is listed as a failure and left as it
  is.
- `durable_dir=` per arm, so the run files survive a restart (wave 1 passed
  none).
- `page_route="auto"`. Each run file records the route it resolved to.
- The same `reports_dir`, `di_dir`, truth folders, label files and
  `chat_model` factory as wave 2.

## 3. MF-1: items handed over

This wave uses the `logs`, `lab`, `calc` and `narrative` stages of
`score_on_cluster`. Each arm reads one stage, so `model=` sets that arm's
model in both settings, and both arms of a row read on one model (D3).
Under `on`, each item is read from its own pages (the narrative from the
graph's narrative pages). The record is built with `checks='flag'`, each
value is placed on its page, and the stage's own scorer scores the record.

Items per stage: logs 15 (9 blind), lab 31 (15 blind), calc 10 (open; no
blind set), narrative 8 reports (6 blind).

| arm | stage | `model_first` | model | calls | tokens in / out | API rates |
|---|---|---|---|---:|---:|---:|
| logs today | logs | off | Sonnet 5.5 | 29 | 0.53 M / 0.12 M | $2.29 |
| logs today | logs | off | Opus 5.5 | 30 | 0.54 M / 0.11 M | $4.43 |
| logs first | logs | on | Sonnet 5.5 | about 15 | about 0.13 M / 0.06 M | about $0.85 |
| logs first | logs | on | Opus 5.5 | about 15 | about 0.13 M / 0.06 M | about $1.70 |
| lab today | lab | off | Opus 5.5 | 37 | 0.71 M / 0.10 M | $4.91 |
| lab first | lab | on | Opus 5.5 | about 31 | about 0.43 M / 0.10 M | about $3.70 |
| calc today | calc | off | Sonnet 5.5 | 14 | 0.29 M / 0.04 M | $1.01 |
| calc today | calc | off | Opus 5.5 | 15 | 0.33 M / 0.05 M | $2.40 |
| calc first | calc | on | Sonnet 5.5 | about 10 | about 0.13 M / 0.03 M | about $0.60 |
| calc first | calc | on | Opus 5.5 | about 10 | about 0.13 M / 0.03 M | about $1.20 |
| narrative today | narrative | off | Opus 5.5 | 23 | 0.64 M / 0.08 M | $4.11 |
| narrative first | narrative | on | Opus 5.5 | 8 to 16 | about 0.5 to 1 M / 0.1 M | about $3 to $6 |

- The Sonnet rows for logs and calc are the production models (§2); the
  Opus rows are the plan's second model for those stages (§8.8). The
  narrative runs on Opus in both arms, as §2 says.
- **Totals:** about 240 calls, 4.5 to 5.0 M tokens in and about 0.9 M out.
  At the API rates the local ledger used, that is about $30 to $33.
- **Repeats:** if two arms land within one run's noise, re-run both arms on
  the items where they differ (§8.8). The worst case is the same totals
  again.
- **Pictures route:** if the probe passes on one route only, run lab first
  again on the pictures route over the 15 blind sheets. Estimate: about 15
  to 20 calls and about 0.1 to 0.15 M tokens in.

## 4. MF-2: whole reports

The `ingest` stage, run off and on over wave 2's six reports (1,235 pages):

| report | pages | what is there |
|---|---:|---|
| R05 | 15 | small public report; narrative answer sheet |
| R36 | 94 | small public report; narrative, log and lab answer sheets |
| R37 | 48 | small public report; narrative and log answer sheets |
| R13 | 197 | hand-labelled; every page scanned; one bound document |
| R16 | 426 | hand-labelled; a bound volume; calc answer sheet R16_p162 |
| R28 | 455 | hand-labelled; narrative, log, lab and sounding answer sheets |

Scoring uses `_score_ingest_record` against the answer sheets, plus pages
sent to the wrong reader against the hand labels (RESULTS.md column
`wrong rdr`, rules / final) on R13, R16 and R28.

R13 is all scanned. On the PDF route the model sees each page's image with
no text layer, while placement reads the Document Intelligence words where
the corpus has them; on the pictures route the model gets those words as
well (plan §8.7). The run files say which route ran.

- **Off arm:** `model_first="off"`, the models of §2: each reader on its
  stage's model, the triage and label review on Opus, the vision voter on
  Sonnet (D1).
- **On arm:** `model_first="on"`, the same `stage_models`, the describer
  of D2 and the labeller on Opus. A saved label run seeds the triage alone.
  The on arm has no vision voter and no label review.
- Each `run.json` records the arm and the model of each reader. Under `on`
  it also records the route each reader resolved to, the describer's and
  labeller's models, the instruction version (`mf-1`), the describer's and
  labeller's versions, and the placement counts.
- **Order:** R05 and R36 first, as wave 2's pilot ran them, then R37, R13,
  R16 and R28.

Estimates for the six reports, at the API rates of the local ledger:

| arm | calls | tokens in / out | API rates | time |
|---|---:|---:|---:|---|
| off | about 710 (run 5's count; the earlier 1,800 was too high) | about 9.6 M in on GPT-5.4, so about 10 to 18 M on Claude (1.0 to 1.9 times); out and dollars to be recomputed from run 5 | to be recomputed | stage b took 3.9 h for R13, R16 and R28 |
| on, Haiku describing | about 1,650 to 2,100 | about 12 to 18 M / 2.1 to 3.7 M | about $45 to $70 | about 1 h describing, then the readers |
| on, Sonnet describing | about 1,650 to 2,100 | about 17 to 23 M / 2.1 to 3.7 M | about $70 to $95 | about 1.7 h describing, then the readers |

How the rows are built:

- **Off:** stage b ran the same pipeline on GPT-5.4, with the vision voter
  on every page:

  | report | calls | tokens in / out |
  |---|---:|---:|
  | R13 | 206 | 2.67 M / 0.43 M |
  | R16 | 897 | 8.39 M / 1.08 M |
  | R28 | 537 | 5.08 M / 0.75 M |

  R05, R36 and R37 (157 pages) were not in stage b and are taken at its
  average per page (about 1.1 calls, 11.5 k tokens in and 1.6 k out). That
  gives about 1,800 calls, 17.6 M tokens in and 2.5 M out on GPT-5.4. On
  the same reader items, Claude read 1.0 to 1.9 times GPT-5.4's input
  tokens and wrote 1.1 to 2.6 times its output tokens (wave 1 against the
  local runs); those ratios give the range. Stage b's own files split the
  tokens by step: on these reports the printouts, the logs and the vision
  voter, which run on Sonnet, take about three quarters of the input; the
  lab, the narrative, the label review and the triage, on Opus, the rest.
  Wave 2's own plan for its first arm is lower (about 750 calls and 10 M
  in); the run's own counts settle it.
- **On:**
  - Each page is described in one call. Per page, Haiku used about 5.1 k
    tokens in and 0.57 k out, and Sonnet about 9.1 k in (RI-73's full runs,
    labeller calls included). At RI-73's measured cost per page, the
    describer comes to about $3 on Haiku and $30 on Sonnet.
  - The labeller is about 10 calls on Opus, about $2.50 (plan §8.8: $0.30
    for a 150-page report). Triage and the bound reports' identity are a
    few calls on Opus.
  - The readers use the plain call's tokens per item (CLAUDE LOCAL RUN 1).
    The low end takes the plan's median mix of items per page (§8.8):
    about 0.32 calls, 4.2 k tokens in and 1.1 k out a page, or 5 M in. The
    high end takes stage b's own item counts for these reports, which hold
    many printouts: about 860 items and 11 M in. Priced per stage on §2's
    models, the readers come to about $40 to $60.
- **Time:** inside the ingest stage, the describer runs one page at a time
  (the graph's step has no thread pool). Over the API that took about 3
  seconds a page on Haiku and 5 on Sonnet. The clock binds before
  Foundry's rate limit does.

**Both arms together:** about 30 to 52 M tokens in and 5 to 10 M out,
about $120 to $230 at API rates with Haiku describing. MF-2 is most of the
wave.

### 4a. MF-3: held-out reports, counts only

Run last, and only on the owner's word, after MF-2's six reports are done
and checked: the same two arms, on the same models, on R15 (202 pages),
R21 (729) and R30 (556), 1,487 pages. These are stage-b reports with blind
answer sheets (R15: 3 lab, 1 narrative; R21: 1 log, 1 narrative; R30: 1
log). Report counts only, as for every blind set.

- **Off:** stage b measured R15 at 88 calls, 1.22 M tokens in and 0.21 M out
  and R30 at 402 calls, 4.16 M and 0.61 M. Stage b failed on R21 after 5
  calls; at its average per page R21 is about 800 calls and 8.4 M in. In
  all, about 1,300 calls and 14 M in on GPT-5.4, or 14 to 26 M in and 2.2
  to 5.2 M out on Claude (about $65 to $140 on §2's models).
- **On:** about 2,000 to 2,500 calls, 14 to 22 M tokens in and 2.5 to 4.5 M
  out with Haiku describing (about $55 to $80), MF-2's on arm scaled by
  pages.
- **Order:** R15, R30, then R21 last.

## 5. Stop points

1. **MF-0:** if neither route returns the code word, stop and send back the
   probe's output. If only the pictures route does, run on pictures (the
   run files record it). If Sonnet 5.5 is not offered, stop (§1).
2. **Each MF-1 arm:** run `max_reports=2` first. Check that both run files
   record the arm (`model_first`), the route and no error, then run the rest
   in the same `out_dir` (it resumes).
3. **Each MF-2 arm, the token check:** run R05 and R36 first. Check each
   `run.json` (arm, models, routes, placement counts) and its tokens in
   (`all_attempts.input_tokens` where the file has it: `cost` is the last
   attempt alone).
   Stop and send the report back if its tokens in pass the line below: 1.5
   times the high estimate per page (off about 27 k a page, on about 15 k).
   Then run R37, R13, R16 and R28 with `max_reports`, and check each report
   against its line as it finishes. MF-3 takes the same lines per page.

   | report | pages | off: stop above | on: stop above |
   |---|---:|---:|---:|
   | R05 | 15 | 0.6 M | 0.33 M |
   | R36 | 94 | 3.8 M | 2.1 M |
   | R37 | 48 | 1.9 M | 1.1 M |
   | R13 | 197 | 7.9 M | 4.3 M |
   | R16 | 426 | 17 M | 9.4 M |
   | R28 | 455 | 18 M | 10 M |

4. **If wave 2 stops early:** nothing here changes. This wave's off arm is
   the baseline; wave 2's results are not used for the comparison.
5. **Failures:** stop an arm when two of its items or reports fail with the
   same error, or when more than two items fail.
6. **What stops a run on Foundry:** `max_reports` and the per-step call
   budgets cap the work. Nothing caps the spend: the meter reads $0, so the
   token check in point 3 is the only brake. The token ceiling for the
   whole wave is the lead's call (suggested: 1.5 times this brief's high
   estimates).

7. **Added after Foundry run 5 (RI-120):**
   1. **Stop lines per describer.** The page-group stop line is per page: on
      Haiku describing about 15 k tokens in a page, on Sonnet describing
      about 28 k (run 5's Sonnet page).
   2. **Time limit.** Check it between reports, not only at the start, and
      expect more than one build per arm: an arm is not one sitting.
   3. **After each report**, read the run's failed count and its 'could not
      be read' QA entries as well as its errors. A report with a skipped
      item is not done (RI-110).
   4. **One `review_dir` for both arms**, and compare each report's
      `bound_together` between the arms.
   5. **The off arm waits for RI-117 and RI-123.** RI-108, RI-109 and
      RI-110 are merged (2026-10-26). Wave 2's second hand-back showed
      today's log reader reads several holes' pages as one hole (R28 lacks
      18 of the 77 holes it prints) and lets depths above the ground
      surface through; model-first groups log pages by hole, so it would
      win on grouping alone. If the wave cannot wait, score hole recall
      (holes printed against holes in the record) as its own line.
   6. **Token brake.** Set `max_report_tokens_in` and `max_total_tokens`
      (RI-120 (1)); the dollar meter reads $0 on Foundry. Set them to the
      stop lines above, and the wave's ceiling to the lead's number.
   7. **Rate limits (OWN-99).** Before the wave, ask the AI FDE for each
      model's tokens-per-minute and requests-per-minute limits on the
      route, any per-request input cap, and whether they are shared with
      other Foundry work. Run 5's lost log item was stopped by the
      per-minute limit, not a per-request one: a 126,558-token request
      passed after 4 retries, and the next small call waited 72 s. Set
      `max_request_tokens_in` (RI-110) and the page groups' split (RI-121)
      from the answer, and expect slow minutes after any large request.
   8. **Cost from every attempt.** RESULTS' Cost block counts every
      attempt of every report (RI-124): the finished reports, then a line
      for the reports not finished (braked, given up or failed, all listed
      under failures), then the total for every report. Read the arm's
      spend off that total line. `max_total_tokens` counts one build only,
      so the lead adds the builds' totals for the wave's ceiling (RI-133).
      Return one summary row per build (the glue's table listed only some).
   9. **Narrative repeats.** The same narrative read twice differed by one
      field in 28. Compare the arms' narratives only on differences larger
      than one field a report, or read each twice.
   10. **Arm stamp.** Each report folder carries `arm.json`; a folder
      started by one arm is refused to the other (RI-120 (3)). Use one
      `out_dir` per arm, as in section 2.

## 6. What goes back

For each arm folder, the same as wave 1:

- `results.json` and `RESULTS.md`;
- the run files of each stage;
- for each report: `ingest/<ID>/run.json`, `labels.json`, `qa.json`,
  `ranges-*.json` and `items/`;
- the calls and tokens by step;
- the full traces, for the owner's trace review of every run.

Also send back:

- the probe's printed outcome for each route and model;
- each cell's printed log.

Private content stays in the workspace, as in wave 1.

At home, the two arms are compared at $0:

- `rescore_saved` brings both arms to the current scorers.
- `run_plain_baseline report --model-first --pipeline-arm <arm>` sets the
  local model-first arm beside any saved arm.

## 7. Where the numbers come from

All counts only, from the saved runs, reported in MEASUREMENTS:

- **FOUNDRY RUN 4, wave 1** (GPT-5.4, 2026-10-23): 223 calls, 2.96 M tokens
  in and 0.38 M out, 0 retries.
- **Stage b** (GPT-5.4, the whole pipeline, 13 reports, 3,418 pages): 3,737
  calls, 39.4 M tokens in and 5.6 M out.
  - By step, from the cost each saved file carries, share of tokens in:
    printouts 36 %, logs and soundings 21 %, lab 18 %, narrative 14 %,
    label review 5 %, vision voter 5 %, triage 1 %. On R13, R16 and R28 the
    printouts take 55 %.
- **CLAUDE LOCAL RUN 1 and 2** (2026-10-24): today's readers on Sonnet 5.5
  and Opus 5.5, and the plain Opus call per item.
  - Plain call totals: logs 15 calls, 0.13 M tokens in; lab 31 calls, 0.43 M;
    calc 10 calls, 0.13 M; narrative one report, 0.04 M.
  - API rates on the local ledger: Opus $4 / $20 and Sonnet $2 / $10 per
    million tokens in / out.
- **RI-73's full describing runs** (13 reports, 3,418 pages): Haiku $8.40,
  Sonnet $83.36.
- **Dry run** (RI-87 piece 9, $0,
  `module_work/report_ingest_harness/dry_run_model_first.py`): both arms run
  end to end through `score_on_cluster` on the first two answer sheets of
  each stage, with every call answered by the plain call's saved reading.
  - Each arm's run files record their arm.
  - The before column is the same in both arms.
  - Both arms give the same after score on the same reading: logs 13 / 17,
    lab 132 / 132, calc 40 / 49, narrative 24 / 27.
  - Of the values on pages with text, the first arm placed 3 of 3 (logs),
    2 of 2 (lab), 72 of 78 (calc) and 32 of 32 (narrative).
  - One narrative report has no saved reading and fails in both arms, as
    expected.
