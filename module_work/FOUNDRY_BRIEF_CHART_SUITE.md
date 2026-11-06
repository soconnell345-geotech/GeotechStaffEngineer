# Foundry brief: the chart suite on CharXiv (EV-35), 2026-11-06

**The question (owner, 2026-11-06).** The Opus 5.5 system card reports
Chartography at 64.4 % without tools and 89.0 % "with tools": the chart in a
container with standard libraries, plus an image-cropping tool, at max
effort. Anthropic published the zoom half of that setup as a Claude Cookbook
recipe, not the harness itself. Does that setup read charts better than this
app's Document Review page? This brief runs one set of chart questions
through five setups on the same model, graded by the same judge, so the
setups can be compared directly.

**The questions are CharXiv's, not Chartography's.** Chartography turned out
to be private on Hugging Face ("Repository not found" for the owner,
2026-11-06). CharXiv (Wang et al., 2024, Princeton) is the nearest public
set: 1,000 real charts from arXiv papers in eight fields, each with one
hand-written reasoning question and its answer (a term, a short phrase or a
number read off the chart). Its authors also published the prompts their
evaluation asks with and the grading prompt for each answer type, and the
suite uses both word for word.

The suite is `funhouse_agent.chart_eval` (in the test wheel). It never calls
a model in its own tests. Everything below runs on Foundry's models.

## 0. What is delivered

| Item | What | Notes |
|---|---|---|
| App test wheel `geotech_staff_engineer-5.34.0+foundry.20261106.2-py3-none-any.whl` | master 8daed85 (5.34.0 plus the unreleased master fixes in HANDOFF, "Master after 5.34.0") plus `funhouse_agent/chart_eval/`, from branch `claude/anthropic-vision-testing-harness-22m8ae`; only the version label was changed for the build | set `GEOTECH_APP_COMMIT` to the commit in `BUILD_NOTES.md`, so every run records it |
| `planlens 0.14.0`, `geotech-references 1.5.0` | PyPI (wheels in the delivery folder too) | what 5.34.0 pins |
| `charxiv_data.zip` (57 MB) | the data folder, made in a Claude Code cloud session from CharXiv's pinned sources (§1) | private: CC BY-SA questions, charts owned by the papers' authors |
| `BUILD_NOTES.md` | wheel versions, commit, install check | |

## 1. The data folder

Already made; nothing to do unless it has to be rebuilt. It holds:

- `tasks.jsonl`: the 1,000 validation reasoning questions. Each prompt is
  the question as CharXiv's own generation script asks it, with the
  response instruction for its answer type;
- `charts/`: the 1,000 charts (JPEG, at most 1,024 px on the long side);
- `judge_prompts.json`: CharXiv's grading prompt for each of its four answer
  types (text in chart, text in general, number in chart, number in
  general);
- `sample_100.txt`: 100 task ids drawn at random (seed 0) in proportion to
  the answer types (44 / 10 / 23 / 23);
- `PROVENANCE.json` (the pinned GitHub commit and Hugging Face revision) and
  `ATTRIBUTION.md`.

The questions are CC BY-SA 4.0, the prompts Apache 2.0, and the charts
belong to the papers' authors. Keep the folder out of every repo; a Foundry
workspace repo has no `.gitignore` entry for it, so unzip it outside the
repo.

To rebuild it where GitHub and huggingface.co can be reached (public, no
token): `python -m funhouse_agent.chart_eval.charxiv --out charxiv_data
--zip`.

## 2. Install (the AI FDE)

Run in a **Code Workspace (JupyterLab)**, as the review suite was run for
briefs 3 to 5. The suite is a notebook cell, and its page arms fork one
child process per run.

1. Install the three wheels as for build i: `--no-deps --target <folder>`,
   with that folder first on `sys.path` (docs/FOUNDRY.md §1, step 3). On a
   numpy 1.26 environment, add the constraints of §1, step 4. Nothing new
   is needed for this suite: Pillow and matplotlib are already app
   dependencies.
2. Check the install:

   ```python
   import funhouse_agent.chart_eval as ce
   from importlib.metadata import version
   print(version("geotech-staff-engineer"), ce.DEFAULT_ARMS)
   from funhouse_agent import plot_box; print("box", plot_box.available())
   ```

   It should print `5.34.0+foundry.20261106.2`, the five arms and `box True`.
   If the box is not available, the `zoom_code` arm fails every run, and the
   failures are recorded.
3. Unzip `charxiv_data.zip` outside the workspace's git repo and set `DATA`
   to that folder.

## 3. Models

| Role | Model | How it is built |
|---|---|---|
| Model under test 1 | Opus 5.5 | `ce.foundry_model("anthropic:<Opus 5.5 RID>")` (the RID imported in the workspace's Models panel, as build i) |
| Model under test 2 | GPT-5.6 Sol | `ce.foundry_model("GPT_5_6_SOL")` (the API name brief 3 used; the Responses route) |
| Judge (both runs) | Sonnet 5.5 | `ce.foundry_model("anthropic:<Sonnet 5.5 RID>")` |

`foundry_model` builds each model exactly as the app does
(`webapp.engine_config.resolve_engine`). It sends the same proxy fixes and
retries, and raises with the app's own message if a model cannot be built.
The judge never sees a chart. It reads CharXiv's grading prompt for the
question's answer type, with the bare question, the golden answer and the
response. CharXiv's authors graded with GPT-4o. One judge for both runs
keeps the grading identical.

## 4. The arms

| Arm | What the model is given |
|---|---|
| `none` | the chart and the question in one message, no tools (CharXiv's own setup; the system card's "without tools") |
| `zoom` | the Claude Cookbook's zoom tool: the chart shown at exactly the size the model reads; the model asks for a region in pixels; the region comes back cropped from the original and magnified to fill the model's image limits; up to 20 tool steps |
| `zoom_code` | zoom, plus `run_python` on the chart in the app's locked plotting box (numpy, scipy, matplotlib; no PIL, OpenCV, network or processes). The nearest the app comes to the system card's container. |
| `page:baseline` | the chart uploaded to the **Document Review page as users get it today**, the question asked there. The main model never sees the image: one-shot vision calls describe it and `render_region` zooms. |
| `page:minimal` | the same page's looking-only agent (`GEOTECH_REVIEW_AGENT=minimal`): zooms are shown to the model itself |

CharXiv's charts are at most 1,024 px on the long side. Every model reads
them at full size, so a zoom adds magnification, not new pixels; the
cookbook's own gain came partly from charts larger than the model's limit.

## 5. The runs

The main run is the 100-question sample on both models: 500 runs each. Set
the folders once:

```python
import funhouse_agent.chart_eval as ce
DATA = "<the unzipped charxiv_data>"
OUT = "<a writable folder>"          # one subfolder per model below
opus = ce.foundry_model("anthropic:<Opus 5.5 RID>")
sol = ce.foundry_model("GPT_5_6_SOL")
judge = ce.foundry_model("anthropic:<Sonnet 5.5 RID>")
```

**A. Dry run** (no model call): lists the sample's tasks with their chart
sizes, the arms, the judge prompts in use and the runs still to do.

```python
ce.score_chart_suite(opus, data_dir=DATA, subset="sample_100.txt",
                     out_dir=f"{OUT}/opus", dry_run=True)
```

**B. Smoke, 3 tasks, every arm, Opus.** Check four things before going on:
`RESULTS.md` shows 0 failed runs; every run is graded (no `?`); a `zoom`
run's `transcript.jsonl` lists `zoom` calls and `images/` holds the crops;
and a `page:*` run has `page/activity.jsonl`.

```python
ce.score_chart_suite(opus, judge_model=judge, data_dir=DATA,
                     subset="sample_100.txt", limit=3,
                     out_dir=f"{OUT}/opus", workers=4, page_workers=1)
```

**C. The sample, Opus.** Same folder, so the smoke runs are reused.

```python
res = ce.score_chart_suite(opus, judge_model=judge, data_dir=DATA,
                           subset="sample_100.txt", out_dir=f"{OUT}/opus",
                           model_label="Opus 5.5", judge_label="Sonnet 5.5",
                           workers=8, page_workers=4)
```

`workers` direct runs go at once on threads. Then the page runs go,
`page_workers` at once, each in a forked child process (stopped after
`page_timeout_s`, default 1800 s). If the forked children misbehave on this
host, use `page_workers=1`, which runs the page arms one at a time in the
notebook process. Where Foundry's rate limit bites, lower `workers`: a busy
model is waited for and asked again (15 s to 2 min, five times), and a run
that still fails is recorded as failed.

**D. The sample, GPT-5.6 Sol.** The same cell with `sol`,
`out_dir=f"{OUT}/sol"` and `model_label="GPT-5.6 Sol"`.

**E. Optional, all 1,000 questions** on the arms the sample says matter:
the same cell without `subset=` and with `arms=(...)`. The 100 already done
are reused. This is about ten times the sample's tokens.

**F. Optional, repeats on the direct arms (Opus).** Three runs each of
`none`, `zoom` and `zoom_code` on the sample, to see how much a score moves
between runs: add `arms=("none", "zoom", "zoom_code"), repeats=3`.

**Resuming.** Every cell can be run again. A finished run is never asked
again, a failed one is retried, and `RESULTS.md` is rewritten after every
run. `rejudge=True` grades every saved answer again without repeating a
run. `redo=True` repeats everything.

## 6. Cost, in tokens, and time

Estimated from the cookbook's measured costs and brief 3's review-suite
tokens; not measured on this suite. Per model, the 100-question sample:

| Arm | Input tokens | Output tokens | Notes |
|---|---|---|---|
| `none` | ~0.3 M | ~0.3 M | one call per question; CharXiv's charts are small (about 1,000 image tokens) |
| `zoom` | ~3 to 10 M | ~0.5 to 1 M | each zoom re-sends the conversation and adds a magnified crop |
| `zoom_code` | ~5 to 15 M | ~1 to 2 M | |
| `page:baseline` | ~5 to 15 M | ~0.3 to 1 M | brief 3: about 130 k input tokens per review task |
| `page:minimal` | ~20 to 50 M | ~0.5 to 1 M | brief 3: about 4× baseline |
| judge | ~0.3 M | ~0.05 M | 500 short calls |

That is roughly 35 to 90 M input tokens per model. Foundry's meter reads $0
for these models, so the per-arm token totals in `RESULTS.md` are the
record of spend. A direct run is capped at 20 tool steps, and a page run at
the page's 40-call budget. Time: the direct arms take about 1 to 2 hours at
`workers=8`; the page arms take about 2 to 4 hours at `page_workers=4`.

**Stop and send back** if the smoke (B) shows any failed run or any `?`, or
if more than 5 runs of an arm fail in C or D. Send the error text from
`run.json` (`error`, `traceback`) with it.

## 7. What comes back

The **whole output folder of each model** (`{OUT}/opus`, `{OUT}/sol`):

- `RESULTS.md` and `results.json`;
- every run's folder, `runs/<arm>/<task>/`:
  - `run.json`: the answer, the judge's verdict and reply, the model calls
    and the tokens;
  - for the direct arms, `transcript.jsonl` (each model call's text,
    reasoning, tool calls and usage; each tool call's arguments and result)
    and `images/` (every image the model was shown);
  - for the page arms, `page/` (the app's own `activity.jsonl`, `run.json`
    and files).

Every run is read in full on the way back (CLAUDE.md, "REVIEW EVERY MODEL
RUN IN FULL"), so the folder matters more than the scores. It holds the
questions, the answers and copies of the charts, so it is as private as the
data folder: hand it to the owner, not into any repo.

Also send the AI FDE's own notes on what had to change in the environment.

## 8. Reading the numbers

- **Compare arms within a model.** Each model's arms share the same model,
  questions and judge, so differences between them are the setup's. Across
  models, the same judge also makes the two runs comparable.
- **Published CharXiv reasoning scores (validation set, GPT-4o grading,
  2024):** human 80.5 %, Claude 3.5 Sonnet 60.2 %, GPT-4o 47.1 %. Current
  models are expected well above these.
- **The system card's numbers are on Chartography, not CharXiv:** Opus 5.5
  at 64.4 % without tools and 89.0 % with them, at max effort. Their size
  and direction may not carry over. CharXiv's charts are smaller and its
  answers exact, so the gain from tools may be smaller.
- **This suite differs from the system card's setup in four ways:**
  - Foundry's Claude runs at the model's default effort (medium on Opus
    5.5), not max.
  - The code box takes no PIL or OpenCV.
  - The judge is Sonnet 5.5 with CharXiv's own grading prompts.
  - A zoom's image comes back in a user message right after the tool
    result, the form every Foundry route takes, rather than inside Claude's
    tool result as in the cookbook.
- **What would change the app:**
  - `zoom` or `zoom_code` well above `page:baseline` on the same model
    would say the main model should see and zoom the image itself. That is
    the `GEOTECH_VISION_INLINE` / minimal direction, so compare
    `page:minimal` too.
  - `page:baseline` level with them would say the side-call design is not
    what limits chart reading.
