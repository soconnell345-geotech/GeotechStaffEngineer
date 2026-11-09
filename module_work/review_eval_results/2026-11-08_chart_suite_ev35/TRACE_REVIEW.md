# EV-35 chart suite on CharXiv: full trace review (2026-11-08)

What was read: the hand-back of the chart suite (`funhouse_agent/chart_eval/`, app commit 8f1f8c3) run
on two models, Opus 5.5 and GPT-5.6 Sol (called Opus and Sol below). Each model ran the same 100 CharXiv
reasoning tasks on five arms: `none`, `zoom`, `zoom_code`, `page_baseline` and `page_minimal`. That is
1,000 runs. Four readers each took one model and half the tasks (ids 14 to 1312, and 1318 to 2390). They
read every run in full: the direct arms' `run.json`, `transcript.jsonl` and images, and the page arms'
`run.json` and `page/activity.jsonl`. For the page arms that means every side call's prompt, text and
reasoning, and every tool result. Every judge verdict was checked against the gold. The chart was opened
for every task with a wrong or doubtful run.

The raw traces are in `module_work/field_feedback/2026-11-08_chart_suite_ev35/raw/`. That folder is
gitignored and holds the hand-back's `REPORT.md`, each model's `RESULTS.md` and every run's files. The
hand-back carries no questions or gold answers, so the readers took them from CharXiv's public files at
the commit the suite pins. This file names tasks by id only. Code is cited as `file:line` at 8f1f8c3.

## 0. Verdict for the owner

1. **Opus: nothing separates the arms.** After three gold fixes every arm scores 95 to 97 of 99. Each gap is one or two tasks, and each of those tasks is an ambiguous question or a single misread. The side-call design does not limit Opus on this set.
2. **Sol: the direct arms beat Document Review's default page, and the gap is real.** Corrected, `zoom_code` scores 95 of 99 and `page_baseline` 84 (12 to 1 discordant, exact p = 0.003). `none` scores 91 against 84 (9 to 2, p = 0.065).
3. **Why Sol's page loses:** its main model never sees the chart. In 89 of 100 baseline runs it asked one vision side call and copied the reply. Every page loss traces to that reply, to a re-ask that replaced a right reply, or to the side call answering the main model's rewording of the question.
4. **`page_minimal` recovers part of the gap (89 corrected), but its pixels account for little of it.** Only 2 of its 7 wins over baseline come from its main model looking at the image itself. Sol looked in 16 of 100 runs.
5. **Two product faults changed results:** a failed vision probe silently lowers the run's image detail (in the app the lowered setting is cached for the whole process), and the side call never sees the user's question. Fix both, then re-measure Sol with repeats before changing any default. Gaps of 1 to 3 tasks between the other arms are within rerun noise.

## 1. The scores

Accuracy per arm. Reported figures are out of 100. Corrected figures are out of 99, with 193 dropped and
1322 and 1544 re-scored.

| arm | Opus reported | Opus corrected | Sol reported | Sol corrected |
|---|---|---|---|---|
| `none` | 93 | 95 | 89 | 91 |
| `zoom` | 94 | 95 | 89 | 91 |
| `zoom_code` | 94 | 95 | 93 | 95 |
| `page_baseline` | 93 | 95 | 82 | 84 |
| `page_minimal` | 95 | 97 | 87 | 89 |

**Gold fixes** (the same in both models):
- **193 dropped.** The gold reads a right-axis curve against the left axis. The value the question asks
  for is not printed, so the task cannot be re-scored cleanly. Every Sol arm was marked wrong. On Opus,
  `zoom` and `zoom_code` were marked right, but their answers say they matched the gold's reading
  knowingly. REPORT's "+193" for those two arms is therefore an artefact.
- **1322 re-scored.** The gold counts a line lying exactly on the threshold as above it. Every run in
  both models gave the strict count.
- **1544 re-scored (the softer fix).** The gold needs each month label read as centred on its month. The
  chart's labels sit on month-start gridlines, and under that reading all ten runs across both models
  are right. Without this fix, subtract 1 from every corrected figure.

**Sensitivities** (kept as scored):
- **1060.** If the legend's sample marker counts toward "the total", Opus gains 1 on each page arm. Sol
  gains 1 on `none`, `zoom`, `page_baseline` and `page_minimal`, which removes `zoom_code`'s lead on this
  task.
- **160.** The wording is ambiguous about which curve "reaches" the steady value. Accepting both readings
  gives Opus +1 on `zoom`, `zoom_code` and `page_baseline`, and Sol +1 on `none`.
- **1455.** The count depends on whether points on a peak's flank count as "on the peak". Accepting both
  readings gives Opus +1 on `none`, `zoom`, `zoom_code` and `page_minimal`, and Sol +1 on `zoom`,
  `zoom_code` and `page_minimal`.
- **640** is a one-pixel near tie at a crossing. The gold is defensible.

**The judge** made no errors in 1,000 runs. Every verdict follows CharXiv's rule for its answer type. The
lenient-looking passes are right: an expanded abbreviation the chart itself defines, a misspelling copied
from the chart's own legend, and a tick mark read as a trailing dash. The verdict JSON comes back under
two key spellings, and `funhouse_agent/chart_eval/judge.py:119-125` reads both.

**McNemar on the corrected Sol scores** (exact, two-sided). The corrections apply to every arm alike, so
the discordant pairs are the same as REPORT's.
- `zoom_code` against `page_baseline`: 12 to 1, p = 0.003.
  - `zoom_code` alone right on 241, 445, 640, 741, 897, 967, 1043, 1060, 1169, 2083, 2089, 2108.
  - Baseline alone right on 1177.
- `none` against `page_baseline`: 9 to 2, p = 0.065.
  - `none` alone right on 241, 445, 897, 967, 1043, 1169, 1455, 2089, 2108.
  - Baseline alone right on 1318, 1494.
- **Sensitivity.** Count 2108 as a harness loss (section 4.1) and accept 1060's legend reading. Then
  `zoom_code` against baseline is 10 to 1 (p = 0.012) and `none` against baseline is 8 to 2 (p = 0.109).
- **For contrast, Opus corrected:** `none` against baseline is 3 to 3 and `zoom_code` against baseline
  is 2 to 2 (both p = 1.0).

**Rerun noise.** Several passes are coin flips: the same chart got different side-call answers in the
two page arms, and two of `zoom_code`'s wins over `none` were single calls that used no tool. At one
repeat, only the Sol baseline gap is large enough to read.

## 2. How Document Review's page loses on Sol

**What the page's main model sees.**
- `analyze_image` is a one-shot vision side call. It receives the image key and the main model's
  `prompt`, and nothing else (`funhouse_agent/deep/vision_kit.py:764-796`, the arguments at `:782`).
- The page's own tool note says an upload is not shown to the main model
  (`funhouse_agent/deep/minimal_agent.py:99-101`).
- In `page_baseline`, `render_region` is another side call.
- In `page_minimal` (`funhouse_agent/review_flags.py:440`), `render_region` returns the view to the main
  model inline (`funhouse_agent/deep/inline_images.py`). That is the only way a page arm's main model
  sees pixels.
- Neither arm set `GEOTECH_VISION_INLINE`. That switch acts on the lean agent only, and the minimal agent
  turns the same inline display on by itself (`funhouse_agent/deep/minimal_agent.py:137`).
- In every build, `analyze_image` shows an image inline only when it is a working-folder file opened by
  path (`funhouse_agent/vision_tools.py:2471`). The suite's chart is an upload (an attachment), so its
  first look is always a side call.

**Sol asks once and copies.**

| | Sol baseline | Sol minimal | Opus baseline | Opus minimal |
|---|---|---|---|---|
| runs with one `analyze_image` and nothing else | 89 | 84 | 33 | 23 |
| runs with any `render_region` | 11 | 16 | 66 | 77 |

In every Sol page failure, the final answer is a side call's conclusion, often word for word. The
passes copy too. In 34 of the 35 tasks that were right on every arm in the first half, both page answers
are the first side call's answer.

**Sol baseline's 15 wrong runs after the gold fixes, by mechanism:**
- **A side call's reading, copied (10):** 160, 241, 640, 684, 741, 967, 1043, 1169, 2083, 2108. On
  2108 the side call ran at the lowered detail from section 4.1.
- **A re-ask replaced a right first reply (3):** 445, 897, 1455. The main model asked again with its own
  reading of the question and adopted the newest reply. On 1455 it took the last of several conflicting
  side calls.
- **The side call answered a reworded question (1): 2089, both page arms.** The main model folded
  CharXiv's answer-format line into the question it sent. The side call, which never sees the real
  question, answered that narrower question, and the main model copied it (section 4.2).
- **The ambiguous gold (1):** 1060.

On Sol, each time baseline's main model re-asked and departed from its first side call (445, 897 and
1455), it moved to a wrong answer. Only once did a check keep a right answer:
1760, where a `render_region` side call confirmed the first reply.

**Sol minimal's 10 wrong runs after the gold fixes, by mechanism:**
- **A side call's misreading, copied (5):** 640, 741, 1312, 1760, 2083.
- **Looked at pixels and still wrong (3):** 684 moved from one wrong count to another. 967 kept the side
  call's answer, and one of its three crops was dropped by the inline cap (section 4.4). 1455 took the
  apex-only reading.
- **The reworded question (1):** 2089.
- **The ambiguous gold (1):** 1060.

**What `page_minimal`'s pixels did and didn't do.** Sol's main model looked in 16 of 100 runs: 193, 306,
377, 684, 897, 911, 967, 1043, 1169, 1431, 1455, 1684, 1784, 1985, 2009, 2108.
- **It changed the side call's answer in 5 of them:** it fixed 1043 and 1169, moved 684 from one wrong
  count to another, and made two roundings (306, a pass, and 193, whose gold is dropped).
- **In the other 11 it kept the side call's answer:** 9 passes and the wrong 967 and 1455.

**Reconciling the two Sol halves.** The first half read "mostly yes" to REPORT's "the main model seeing
and zooming" (minimal 43, baseline 38). The second half read "partly" (44 each). Over the full sample,
minimal's 7 wins over baseline split like this:
- **its own pixels:** 1043, 1169;
- **baseline's re-asks:** 445, 897;
- **the two arms' side calls answering the same chart differently, each copied:** 160, 241;
- **baseline's lowered image detail:** 2108.

Baseline's 2 wins over minimal are 1312 and 1760, where minimal's side call misread. Six of minimal's
seven wins fell in the first half. So the halves differ in where those cases fell, not in mechanism.
Looking is real but small on this sample: +2 tasks, from a model that looks in one run in six.

## 3. Why Opus doesn't lose

Opus runs the same page code, and its side calls also run on the main model. Four things differ.
- **Its side calls read better.** The side call is the model reading the chart once, and Opus's
  one-look `none` arm scores 95 corrected against Sol's 91.
- **It checks.** Opus used `render_region` in 66 baseline and 77 minimal runs (Sol: 11 and 16). When a
  first side call was wrong, both Opus page arms checked again and recovered: on 640, baseline through
  more side calls and a computed tick position, minimal by looking at a 4x view. On 241 (baseline) its zooms
  overrode the first reading.
- **Its page losses are interpretation, not reading.** The side call's wording framed an ambiguous
  question on 848 (baseline), 1060 (both page arms) and 160 (baseline). On 1455 (baseline) every arm was
  wrong anyway.
- **The side calls also won one task.** On 2083 the page arms' side-call counts were right, while all
  three direct arms over-read the chart.

On Opus, `page_minimal` is level with baseline at about a fifth of the input tokens: 1.52M against 6.97M
per 100 runs, without the probe calls (section 5.1).

## 4. Faults in the product (Document Review and its vision code)

### 4.1 A failed vision probe lowers the image detail, silently (changed one result)

**What the code does.**
- `funhouse_agent/vision_probe.py:259-264` sends the four probe calls at once.
- `_call` (`:179-197`) does not retry.
- An error on any of the first three calls aborts the probe (`:265-267`), and the budget falls back by
  model name (`:300-307`). For an unknown answerer that means the default budget and detail `high`.
- An error on the fourth call is taken as "detail=original refused" (`:279-284`), so a transient server
  error marks the model as not supporting full resolution.
- `profile_for` caches the profile from a failed probe like any other (`:357-366`).

So in the app, one transient error at a process's first look would set every later vision call for that
model, in every conversation, to the lower detail until a restart. The suite only shows it per run,
because each forked page child probes afresh.

**Evidence.** Eight Sol page runs had a probe call fail and ran at detail `high` instead of `original`:
baseline 160, 239, 1122, 1196, 1322, 2108, 2167 and minimal 848. Opus had none. Sol's probe measured
`high` at a flat 692 tokens for both test squares, against about 4,900 at `original`, so `high` is a much
smaller look. Elsewhere the same 409 is retried: direct calls wait 15 s and retry
(`funhouse_agent/chart_eval/arms.py:222-226`), and page tile calls are retried.

**Effect.**
- **2108 baseline:** the chart and its zoom were large enough for `high` to shrink them. The side call
  then misread which series is highest, and the main model copied it. Minimal, at `original`, read it
  right, as did all three direct arms. Counted as a harness loss in the section 1 sensitivity.
- **848 minimal** was shrunk too but still answered right.
- In the other six the lower detail did not decide the answer. Four are right, 160 is wrong for a
  question-reading reason, and 1322 is a gold fix.

### 4.2 The side call sees only the main model's rewording (changed two results)

`analyze_image` and `render_region` pass the main model's `prompt` to the vision call, not the turn's
question (`funhouse_agent/deep/vision_kit.py:782`).
- **2089, both Sol page arms:** the rewording made the answer-format line part of the question. The
  side call answered the narrower question, and both main models copied it. All three direct arms, given
  the same line beside the question, were right.
- **Related, no verdict changed:** on Opus 1455 (baseline) the rewording framed the names as printed
  labels, and the side call answered that none sits on a peak. On Sol 445 and 897 (baseline) the re-ask
  carried the main model's own reading of the question.

### 4.3 Answers claim first-hand looking that did not happen (no result changed)

Opus's page answers often say "I looked", "I zoomed" or "read by eye" when only a side call saw the chart.
- Baseline's main model is never shown pixels. In at least 10 of the second half's baseline runs it made
  such a claim with no zoom of any kind, and minimal did so in 5 runs across the sample.
- A loose word search finds look wording in up to 61 baseline and 10 minimal answers where no image
  reached the main model.
- Sol's answers are terse and make no such claim.

This misstates the evidence to the engineer reading the answer.

### 4.4 The inline view keeps only the newest two images (no result changed)

`funhouse_agent/inline_store.py:31` (`DEFAULT_KEEP = 2`) and `funhouse_agent/deep/inline_images.py:45-60`
set the limit. On Sol 967 (minimal) the main model asked for three crops at once, and the oldest was
never shown to it. The answer stayed wrong, and the panel that decides it was among the two shown.

### 4.5 The record does not keep the images the main model saw (no result changed)

The inline images go into the request only, never into the saved state
(`funhouse_agent/deep/inline_images.py:12-14`). The page arms save no image files in either model; only
each view's box and size are logged. What minimal's main model actually saw (for example on 684 and 967)
cannot be rechecked from the record.

One more tool result misled without effect: on Sol 1060 (baseline), `find_like` reported no instances of
the marker. The main model ignored it.

## 5. Faults in the suite and its record

### 5.1 Probe calls are counted in the page arms' calls and tokens

Each forked page child starts with an empty probe cache and probes again: 4 calls on Opus, 5 on Sol (the
fifth measures the host's size limit, `funhouse_agent/vision_probe.py:294-297`). The activity log labels
them `kind: probe` (`:191`), but `funhouse_agent/chart_eval/runner.py:103` takes `model_calls` from the
page run as it stands, and `:309` averages it.

| per 100 runs | Opus baseline | Opus minimal | Sol baseline | Sol minimal |
|---|---|---|---|---|
| input tokens, as reported | 8.04M | 2.59M | 4.00M | 1.52M |
| input tokens, without probes | 6.97M | 1.52M | 3.06M | 0.55M |
| model calls per run, as reported | 8.7 | 7.7 | 8.3 | 8.0 |
| model calls per run, without probes | 4.8 | 3.8 | 3.6 | 3.2 |

The probe is 64 % of Sol minimal's input. REPORT §1's cost lines overstate the page arms by these
amounts. No accuracy effect.

### 5.2 The direct arms do not record Sol's reasoning (F4)

- **Scale.** None of Sol's 358 direct-arm model calls carries `reasoning`, though the output token
  counts show it was produced (about 1,559 output tokens for a one-number answer, for example).
- **Cause.** The Responses engine puts the summary in `additional_kwargs["reasoning"]`
  (`webapp/palantir_sdk_engine.py:860`), which the page's activity log reads
  (`webapp/activity_log.py:629`). The direct recorder reads content blocks only
  (`funhouse_agent/chart_eval/arms.py:127-144`), and `:240` stores `reasoning or None`.
- **The page records do keep it:** 116 of 215 baseline and 138 of 218 minimal main calls, and every side
  call.
- **Effect.** No result changed, but the record cannot say why any of Sol's 20 wrong direct-arm runs
  (after the gold fixes) went wrong. Those causes were read from the chart alone. On Opus the route
  returns no reasoning blocks, so this does not apply.

### 5.3 What the run record leaves out

- `run.json` holds neither the question nor the gold, and the judge's prompt is not recorded, only its
  reply (`funhouse_agent/chart_eval/runner.py:156-170` passes them to the judge; `_save`, `:172-178`,
  writes the result without them). Every reader had to fetch CharXiv separately to grade a run.
- The direct transcripts do not keep the prompt text, which is known only from
  `funhouse_agent/chart_eval/arms.py:284-300`.
- `page_minimal`'s `turn_start` records no engine class or model; baseline's does.

### 5.4 The run_python tool

- Its description (`funhouse_agent/chart_eval/arms.py:65-80`) does not say each call starts a fresh
  process. Opus assumed variables carried over and hit a NameError on 317, 640, 697 and 955. Each cost one
  step, and all recovered.
- Printed output keeps only its last 3,000 characters, with no mark
  (`funhouse_agent/plot_box/__init__.py:693-696`). Sol 2083 (`zoom_code`) lost the start of its printed
  counts and was still right.

### 5.5 REPORT's 409 line

REPORT §2 says Sol had two 409 "model busy" replies on page side calls, each answered on the second try.
The record has 13 409 replies on Sol, plus 2 server errors:
- 8 on page probe calls (7 baseline, 1 minimal), never retried, which caused the lowered detail in
  section 4.1;
- 2 on page tile calls, retried;
- 3 on direct calls, retried after 15 s;
- 1 server error on a probe call, which then read as "original refused";
- 1 server error on a direct call, retried.

The 409 body is a write-conflict error from the completion endpoint, not "model busy". Opus had no error
replies. Neither run.json nor RESULTS counts errors, retries or vision fallbacks per run, which is why the
eight lowered-detail runs did not show in the hand-back.

### 5.6 The earlier code review's flags

- **F3** (forked page children inherit live HTTP pools; the code is still at
  `funhouse_agent/chart_eval/runner.py:204-210`): no Opus run had an error. Sol's 409s hit direct runs
  too, so they are not shown to come from F3.
- **F6** (a retried run keeps the failed attempt's images): in all 300 direct runs, the images on disk
  are exactly the ones the transcript names.
- **F7** (image size): every chart is at most 1,024 px, so every arm saw it at its own size. Only zoom
  magnification differs: direct crops up to 2,048 px on Sol and 2,576 on Opus, page views 2x to 4x. No
  wrong run traces to image size.
- **F8** (failed runs dropped from the denominator): no run failed or went unscored, so all 100 count on
  every arm.

## 6. What this means for Document Review's design, and what to measure next

**What the evidence supports:**
- **On Sol, the default page's answer is one side call's reply, copied.** The two product faults
  (sections 4.1 and 4.2) cost three runs here and are fixes on their own merits.
- **Letting the main model see pixels helps when the model uses it.** Minimal's inline views fixed two
  Sol runs and broke none, but Sol used them in 16 of 100 runs. The direct arms, where the model sees the
  chart from its first step, lead baseline by 7 (`none`) to 11 (`zoom_code`) corrected tasks.
- **What this sample does not show** is whether showing an uploaded image to the page's main model from
  its first step would recover that lead. No build does that today: `analyze_image` of an upload is
  always a side call, with or without `GEOTECH_VISION_INLINE` (section 2). That is the arm to add.
- **On Opus there is no accuracy case for changing how the page looks.** Minimal matches baseline at
  about a fifth of the input tokens. That is a cost finding, not an accuracy one.
- **Neither `zoom` nor `zoom_code` against `none` is a tool effect here.** On Sol, `zoom` and `none` are
  level (4 wins each way), and `zoom_code`'s lead rests partly on single calls that used no tool.

**What to measure next, in this order:**
1. **The chart suite on Sol,** after the probe retry, the question passed to the side call and the
   probe-free accounting. Arms: `page_baseline`, `page_minimal`, a page arm that shows the upload to the
   main model from the first step, and `none` and `zoom_code` as reference. Use at least three repeats on
   the same 100 tasks, or the full 1,000 at one repeat, so that a 3-task gap can be told from rerun noise.
2. **The Document Review suite** (`funhouse_agent/review_eval/`, `score_review_suite`) on Sol, before
   any default changes. CharXiv asks for one short answer about one chart; Document Review reviews
   documents. A switch goes on by default only when that suite says so.
3. **Opus** needs no accuracy re-run; the cost comparison without probes is enough.

## 7. Backlog items

These items go into `BACKLOG.md`. This file keeps the evidence.

Changes a measured result or the owner's decision:
- **DR-36** Retry a vision-probe call on a transient error (409, 5xx); never cache, or read as "original refused", a profile from a failed probe; record the fallback on the vision result. (Section 4.1) Changes a measured result: 2108 on Sol, and 8 Sol page runs at lowered detail; in the app, the lowered setting stays for the process.
- **DR-37** Give the vision side call (`analyze_image`, `render_region`) the turn's question beside the main model's prompt. (Section 4.2) Changes a measured result: 2089 on Sol, both page arms.
- **DR-38** Let `analyze_image` show an uploaded image (an attachment) to the main model inline, behind `GEOTECH_VISION_INLINE`, so EV-39 can measure a page whose main model sees the chart from its first step. Today only a working-folder file opened by path is shown (`funhouse_agent/vision_tools.py:2471`); turning the switch on alone does not do this. (Sections 2 and 6) Changes the owner's decision: it is the arm that tests where Sol's lost tasks come from.
- **EV-36** Report the page arms' calls and tokens without the probe calls (`kind: probe` in `activity.jsonl`), and re-derive REPORT §1's cost lines. (Section 5.1) Changes a measured result: Sol minimal's input is 0.55M, not 1.52M, per 100 runs.
- **EV-37** Record the direct arms' reasoning summary from `additional_kwargs` (F4). (Section 5.2) Changes the owner's decision: without it, the record cannot say why 20 direct Sol runs failed, and the next run is meant to explain the gap.
- **EV-38** Put per-run error, retry and vision-fallback counts in `run.json` and RESULTS, and correct REPORT §2's 409 line. (Section 5.5) Changes the reading of a measured result: the eight lowered-detail runs, and the one result they changed, did not show in the hand-back.
- **EV-39** Re-measure Sol after DR-36, DR-37 and EV-36: the chart suite with at least three repeats (or all 1,000 tasks), adding the DR-38 arm, then the Document Review suite before any default changes. (Section 6) Changes the owner's decision.

Noted only, not items:
- Opus page answers claim first-hand looking that did not happen (section 4.3).
- The two-image inline cap dropped one crop once (section 4.4).
- The record does not keep the images shown inline (section 4.5).
- `find_like` reported no instances on one run; the model ignored it (section 4).
- `run.json` lacks the question and gold; the judge prompt and the direct prompt text are not recorded; `page_minimal`'s `turn_start` names no engine (section 5.3).
- run_python does not say each call starts fresh, and it clips printed output unmarked (section 5.4).
- F3, F6, F7 and F8: nothing to fix on this run (section 5.6).
