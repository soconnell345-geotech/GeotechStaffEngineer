# Foundry brief: the Chartography chart suite (EV-35), 2026-11-06

**The question (owner, 2026-11-06).** The Opus 5.5 system card reports
Chartography at 64.4 % without tools and 89.0 % "with tools": the chart in a
container with standard libraries, plus an image-cropping tool, at max
effort. Anthropic published the zoom half of that setup as a Claude Cookbook
recipe, not the harness itself. Does that setup read charts better than this
app's Document Review page? This brief runs all 100 Chartography questions
through five setups on the same model, graded by the same judge, so the
setups can be compared directly.

The suite is `funhouse_agent.chart_eval` (in the test wheel). It never calls
a model in its own tests. Everything below runs on Foundry's models.

## 0. What is delivered, and by whom

| Item | From | Notes |
|---|---|---|
| App test wheel `geotech_staff_engineer-5.34.0+foundry.20261106.1-py3-none-any.whl` | commit 0b8caf6 on `claude/anthropic-vision-testing-harness-22m8ae`: master 8daed85 (5.34.0 plus the unreleased master fixes in HANDOFF, "Master after 5.34.0") plus `funhouse_agent/chart_eval/`; only the version label was changed for the build | set `GEOTECH_APP_COMMIT=0b8caf6` so every run records the commit |
| `planlens 0.14.0`, `geotech-references 1.5.0` | PyPI (wheels in the delivery folder too) | what 5.34.0 pins |
| `chartography_data.zip` | **the owner** makes it on a laptop (§1) | the dataset: gated, NOT FOR TRAINING |
| `BUILD_NOTES.md` | the delivery folder | wheel versions, commit, install check |

## 1. The data folder (the owner, on a laptop; about 10 minutes)

Chartography is gated on Hugging Face, so it cannot be downloaded from
Foundry without a token, and it must never be committed anywhere.

1. Sign in at huggingface.co and open `datasets/surgeai/chartography`. Click
   **Agree and access** (Surge may approve the request by hand). Then make a
   **read** token under Settings → Access Tokens.
2. On the laptop:

   ```
   pip install huggingface_hub pandas pyarrow
   set HF_TOKEN=<the read token>            (PowerShell: $env:HF_TOKEN="...")
   python funhouse_agent/chart_eval/prepare.py --out chartography_data --zip
   ```

   `prepare.py` needs only those three packages and runs from a checkout of
   this branch. It writes `chartography_data/` and `chartography_data.zip`:
   `tasks.jsonl` (100 lines), `charts/`, `DATASET_CARD.md`,
   `PROVENANCE.json` and `judge_prompt.txt`. That last file is the judge
   prompt of Surge's own harness, read out of their `scorer.py` on GitHub.
   Their repository has no licence, so the prompt goes only into this
   private folder.
3. Check `PROVENANCE.json` says `"tasks": 100` and `"judge_prompt": true`.
   If the judge prompt could not be fetched, the suite grades with its own
   wording of the same rules, and every result says which prompt graded it.
4. Hand the zip to the AI FDE beside the wheels. Keep it out of every repo.
   `.gitignore` lists `chartography_data*`, but a Foundry workspace repo
   does not have that file.

## 2. Install (the AI FDE)

Run in a **Code Workspace (JupyterLab)**, as the review suite was run for
briefs 3 to 5. The suite is a notebook cell, and its page arms fork one
child process per run.

1. Install the three wheels as for build i: `--no-deps --target <folder>`,
   with that folder first on `sys.path` (docs/FOUNDRY.md §1, step 3). On a
   numpy 1.26 environment, add the constraints of §1, step 4. Nothing new
   is needed for this suite: Pillow, matplotlib and pandas are already app
   dependencies.
2. Check the install:

   ```python
   import funhouse_agent.chart_eval as ce
   from importlib.metadata import version
   print(version("geotech-staff-engineer"), ce.DEFAULT_ARMS)
   from funhouse_agent import plot_box; print("box", plot_box.available())
   ```

   It should print `5.34.0+foundry.20261106.1`, the five arms and `box True`.
   If the box is not available, the `zoom_code` arm fails every run, and the
   failures are recorded.
3. Unzip `chartography_data.zip` into a folder **outside the workspace's
   git repo**, and set `DATA` to that folder.

## 3. Models

| Role | Model | How it is built |
|---|---|---|
| Model under test 1 | Opus 5.5 | `ce.foundry_model("anthropic:<Opus 5.5 RID>")` (the RID imported in the workspace's Models panel, as build i) |
| Model under test 2 | GPT-5.6 Sol | `ce.foundry_model("GPT_5_6_SOL")` (the API name brief 3 used; the Responses route) |
| Judge (both runs) | Sonnet 5.5 | `ce.foundry_model("anthropic:<Sonnet 5.5 RID>")` |

`foundry_model` builds each model exactly as the app does
(`webapp.engine_config.resolve_engine`). It sends the same proxy fixes and
retries, and raises with the app's own message if a model cannot be built.
The judge never sees a chart. It reads the question, the golden answer and
the response, as in Surge's harness (they use Gemini 3.5 Flash). One judge
for both runs keeps the grading identical.

## 4. The arms

| Arm | What the model is given |
|---|---|
| `none` | the chart and the question in one message, no tools (Surge's harness; the system card's "without tools") |
| `zoom` | the Claude Cookbook's zoom tool: the chart shown at exactly the size the model reads; the model asks for a region in pixels; the region comes back cropped from the full-resolution original and magnified; up to 20 tool steps |
| `zoom_code` | zoom, plus `run_python` on the full-resolution chart in the app's locked plotting box (numpy, scipy, matplotlib; no PIL, OpenCV, network or processes). The nearest the app comes to the system card's container. |
| `page:baseline` | the chart uploaded to the **Document Review page as users get it today**, the question asked there. The main model never sees the image: one-shot vision calls describe it and `render_region` zooms. |
| `page:minimal` | the same page's looking-only agent (`GEOTECH_REVIEW_AGENT=minimal`): zooms are shown to the model itself |

## 5. The runs

Set the folders once:

```python
import funhouse_agent.chart_eval as ce
DATA = "<the unzipped chartography_data>"
OUT = "<a writable folder>"          # one subfolder per model below
opus = ce.foundry_model("anthropic:<Opus 5.5 RID>")
sol = ce.foundry_model("GPT_5_6_SOL")
judge = ce.foundry_model("anthropic:<Sonnet 5.5 RID>")
```

**A. Dry run** (no model call): lists the 100 tasks with their chart sizes,
the arms, the judge prompt in use and the runs still to do.

```python
ce.score_chart_suite(opus, data_dir=DATA, out_dir=f"{OUT}/opus",
                     dry_run=True)
```

**B. Smoke, 3 tasks, every arm, Opus.** Check four things before going on:
`RESULTS.md` shows 0 failed runs; every run is graded (no `?`); a `zoom`
run's `transcript.jsonl` lists `zoom` calls and `images/` holds the crops;
and a `page:*` run has `page/activity.jsonl`.

```python
ce.score_chart_suite(opus, judge_model=judge, data_dir=DATA,
                     out_dir=f"{OUT}/opus", limit=3,
                     workers=4, page_workers=1)
```

**C. Everything, Opus.** Same folder, so the smoke runs are reused.

```python
res = ce.score_chart_suite(opus, judge_model=judge, data_dir=DATA,
                           out_dir=f"{OUT}/opus", model_label="Opus 5.5",
                           judge_label="Sonnet 5.5", workers=8,
                           page_workers=4)
```

`workers` direct runs go at once on threads. Then the page runs go,
`page_workers` at once, each in a forked child process (stopped after
`page_timeout_s`, default 1800 s). If the forked children misbehave on this
host, use `page_workers=1`, which runs the page arms one at a time in the
notebook process. Where Foundry's rate limit bites, lower `workers`: a busy
model is waited for and asked again (15 s to 2 min, five times), and a run
that still fails is recorded as failed.

**D. Everything, GPT-5.6 Sol.** The same cell with `sol`,
`out_dir=f"{OUT}/sol"` and `model_label="GPT-5.6 Sol"`.

**E. Optional, repeats on the direct arms (Opus).** Three runs each of
`none`, `zoom` and `zoom_code`, to see how much a score moves between runs:

```python
ce.score_chart_suite(opus, judge_model=judge, data_dir=DATA,
                     out_dir=f"{OUT}/opus",
                     arms=("none", "zoom", "zoom_code"), repeats=3,
                     workers=8)
```

**Resuming.** Every cell can be run again. A finished run is never asked
again, a failed one is retried, and `RESULTS.md` is rewritten after every
run. `rejudge=True` grades every saved answer again without repeating a
run (after the judge or `judge_prompt.txt` changes). `redo=True` repeats
everything.

## 6. Cost, in tokens, and time

Estimated from the cookbook's measured costs and brief 3's review-suite
tokens. These are not measured on this suite. Per model, all 100 questions:

| Arm | Input tokens | Output tokens | Notes |
|---|---|---|---|
| `none` | ~0.5 M | ~0.3 M | one call per question |
| `zoom` | ~5 to 15 M | ~0.5 to 1 M | each zoom re-sends the conversation and adds a magnified crop |
| `zoom_code` | ~8 to 20 M | ~1 to 2 M | |
| `page:baseline` | ~5 to 15 M | ~0.3 to 1 M | brief 3: about 130 k input tokens per review task |
| `page:minimal` | ~20 to 50 M | ~0.5 to 1 M | brief 3: about 4× baseline |
| judge | ~0.3 M | ~0.1 M | 500 short calls |

That is roughly 40 to 100 M input tokens per model. Foundry's meter reads
$0 for these models, so the per-arm token totals in `RESULTS.md` are the
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
  - `run.json`: the answer, the judge's verdict and reasoning, the model
    calls and the tokens;
  - for the direct arms, `transcript.jsonl` (each model call's text,
    reasoning, tool calls and usage; each tool call's arguments and result)
    and `images/` (every image the model was shown);
  - for the page arms, `page/` (the app's own `activity.jsonl`, `run.json`
    and files).

Every run is read in full on the way back (CLAUDE.md, "REVIEW EVERY MODEL
RUN IN FULL"), so the folder matters more than the scores. It holds the
dataset's questions and golden answers inside `run.json`, so it is as
private as the dataset: hand it to the owner, not into any repo.

Also send the AI FDE's own notes on what had to change in the environment.

## 8. Reading the numbers

- **Compare arms within a model.** Each model's arms share the same model,
  questions and judge, so differences between them are the setup's. Across
  models, the same judge also makes the two runs comparable.
- **The published numbers do not line up with these.**
  - System card: Opus 5.5 at 64.4 % without tools and 89.0 % with them; a
    five-run average at max effort, on Anthropic's own harness and grading.
  - Cookbook: Fable 5 from 29 % to 73 % and Sonnet 5 from 13 % to 44 %,
    model-judged, single pass.
  - Surge's README: no frontier model above 45 % without tools.
- **This suite differs from the system card's setup in four ways:**
  - Foundry's Claude runs at the model's default effort (medium on Opus
    5.5), not max.
  - The code box takes no PIL or OpenCV.
  - The judge is Sonnet 5.5, with Surge's prompt when `judge_prompt.txt` is
    present.
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
