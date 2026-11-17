# Foundry section: the figure test set (GM-314), plot code OFF against ON

A section ready to drop into the next Foundry brief. It assumes that
brief's own delivery and install (the app test wheel, planlens and
geotech-references at the pinned versions, installed by the AI FDE as for
the previous builds).

**The question.** No measurement of figure quality exists, and none with and
without model-written plot code. The geotech page can draw a figure two
ways. With plot code OFF it uses the figure helper and the tools' own
figures (`plot_data`, the calc packages' figures). With plot code ON the
helper may also write its own matplotlib script, run in the locked box
(`plot_code`). This section asks the same 22 figure requests under both
settings. A vision model marks every figure against the figure checklist,
and the owner looks at all of them side by side. The test measures whether
ON earns its place.

**Which deployment each setting is for.** Model-written plot code is
acceptable on the hosted platform, whose sandbox is tight. It stays OFF on
the shared small-apps host (the owner's rule, BACKLOG OWN-133). This test
runs on the hosted platform only, and its result decides nothing for the
small-apps host.

The suite is `funhouse_agent.figure_eval` (in the test wheel). It never
calls a model in its own tests. Everything below runs on Foundry's models.

## 1. What is delivered

| Item | What | Notes |
|---|---|---|
| The app test wheel of the brief | includes `funhouse_agent/figure_eval/` | as the brief's other sections |
| `figure_set_data.zip` | the data folder, made where the repository is checked out: `python -m funhouse_agent.figure_eval.request_set --out figure_set_data`, no model, $0 | `requests.json` (the 22 requests, resolved) and `references/<id>/` (the sample calc package built on the same inputs, with the numbers the calculation returned) |

The requests are made-up projects. Their inputs are the sample calc
packages' own (`docs/examples/sample_packages.json`, BACKLOG GM-328): the
published worked examples' inputs, and two made-up method probes. Nothing in
the folder is private.

## 2. Install check

```python
import funhouse_agent.figure_eval as fe
from funhouse_agent import plot_box
print(fe.ARMS, fe.ARM_ENV)
print("box", plot_box.available())
```

It should print the two arms and `box True`. Without the box the ON arm
draws exactly as OFF does, and `RESULTS.md` shows that ("ON runs that
called plot_code: 0 of 22"). Fix the box before the paid runs.

Unzip `figure_set_data.zip` outside the workspace's git repo and set `DATA`
to that folder.

## 3. Models

| Role | Model | How it is built |
|---|---|---|
| The agent (both arms) | the app's own agent model there (Opus 5.5 for the main agent; the helpers as the app sets them) | `chart_eval.foundry_model("anthropic:<Opus 5.5 RID>")`, as for the chart suite |
| The vision check | GPT-5.4 by the Responses route | `chart_eval.foundry_model("<the GPT-5.4 API name>")` |

The check sends each figure at image detail **AUTO**, not HIGH. HIGH caps
the image near 768 px on that route, and AUTO keeps the lettering legible.
The suite sets AUTO itself (`detail="auto"`); leave it.

## 4. The arms

The runner sets each arm's switches exactly, for the run only, and unsets
every other figure and review switch. Do **not** set `GEOTECH_PLOT_CODE` or
`GEOTECH_FIGURE_HELPER` in the notebook.

| Arm | Switches | What the page may do |
|---|---|---|
| `off` (the baseline) | `GEOTECH_FIGURE_HELPER=1`, `GEOTECH_PLOT_CODE=0` | the figure helper, `plot_data`, the calc packages' figures |
| `on` | `GEOTECH_FIGURE_HELPER=1`, `GEOTECH_PLOT_CODE=1` | the same, plus `plot_code` scripts in the locked box |

## 5. The runs

One run first, to see that a figure comes back and is marked:

```python
from funhouse_agent import chart_eval, figure_eval as fe
agent = chart_eval.foundry_model("anthropic:<Opus 5.5 RID>")
eyes = chart_eval.foundry_model("<the GPT-5.4 API name>")
DATA = "<the unzipped figure_set_data>"
OUT = "<a folder outside the repo>/figure_set"
fe.score_figure_set(agent, vision_model=eyes, data_dir=DATA, out_dir=OUT,
                    ids=["R01"])
```

Open `{OUT}/side_by_side.html`: R01 should show one figure in each arm with
the marks under it.

**Check that the final figures were marked, before the rest.** The checker
is sent the PNGs the agent's reply names by file name (at most four). When
the reply names none, it gets the four newest PNGs of the run instead, but
never the figure helper's look copies (`.scratch/..._look.png`). A draft
the helper retired is deleted, and a redraw under the same name overwrites
its file. But a draft drawn under another name stays in `files/`, and the
runner cannot tell it from a final figure by the file alone. So, for each
arm of R01, open `{OUT}/runs/R01-bearing-width/<arm>/check.json` and
compare its `figures` and `picked_by` with the PNGs in `agent/files/` and
with the figure the reply presents. If a draft was marked, or `picked_by`
is `newest` while `files/` holds more PNGs than the reply presents, stop
and report it with that folder before running the rest. `RESULTS.md` also
notes every run whose marks came from the newest PNGs. The hand-back keeps
every PNG a run wrote (`files` in `run.json`, and `agent/files/`); only the
presented ones (`figures` in `run.json`) are marked and shown on the
side-by-side page.

Then the whole set, in the same folder (R01 is not asked again):

```python
res = fe.score_figure_set(agent, vision_model=eyes, data_dir=DATA,
                          out_dir=OUT)
print(res["comparison"]["requests"])
```

That is 22 requests x 2 arms = 44 agent runs, one after another. Each is a
fresh conversation on the GeotechStaffEngineer page through the review
suite's own runner (`run_task`, page `geotech`). Then come 44 vision checks,
one per run, with all of that run's figures in one call (at most the four
newest). A run that finishes and draws nothing is marked `no` on every item
("no figure appeared") and is not sent to the vision model. A run that
fails (an error, a rate limit such as 429, a timeout, a crash) is "run
failed": it is kept out of better, same and worse and named in the
headline of `RESULTS.md`. So is a request where neither arm drew a figure,
and a vision reply that cannot be read even when asked a second time
("not scored"). The runs resume: a finished run is never asked again, and
a failed one is asked again on the next call. `RESULTS.md`, `results.json`
and `side_by_side.html` are rewritten after every run.

**Failed runs: retry once.** When the first call ends, read the headline
of `RESULTS.md`. If its row "run failed, not compared" names any request,
wait about ten minutes and run the same cell once more, with the same
`DATA` and `OUT`: only the failed runs are asked again. Report each run
that still fails in the hand-back (the request, the arm and the error in
its `run.json`).

**numpy 1.26.4.** That host's numpy predates `np.trapezoid`. Master takes it
from `geotech_common.numpy_compat`, so these requests should run. Three
requests run calculations that integrate with it: R11 (downdrag), R18 (the
cantilever soldier pile) and R19 (the anchored wall). They are marked
`(numpy)` in `RESULTS.md`. If one of them fails, look for `trapezoid` in its
`run.json` error first. R13 asks only for the liquefaction factor of safety
(the SPT evaluation of `seismic_geotech`), not for the lateral displacement
index.

## 6. Cost, in tokens, and time

A geotech question on that host measured about 100 thousand model tokens.
A figure request takes less when one tool figure answers it (about 70
thousand). It takes more when the figure helper draws two or three times
and looks at its PNG after each draw. Each look sends the image (about 1.5
to 3 thousand image tokens) on top of the helper's own context of 20 to 30
thousand. ON may add a script or two (a few thousand each).

| Part | Arithmetic | Tokens |
|---|---|---|
| Agent runs | 44 runs x 70 to 130 thousand | 3.1 to 5.7 M |
| Vision checks | 44 checks x about 6 thousand (prompt about 2.5 thousand, up to four images at AUTO, a short JSON reply) | about 0.3 M |
| **All told** | | **about 3.4 to 6 M** |

Time: 2 to 6 minutes per agent run, so about 1.5 to 4.5 hours for the 44.
The checks add a few minutes. Their 44 calls sit well inside the route's
rate limit.

## 7. What comes back

The **whole output folder** `{OUT}`:

- `RESULTS.md` and `results.json`. They hold the comparison request by
  request and item by item, as counts of better, same and worse for each
  item and overall, with the requests named. "No difference" is shown as
  plainly as a gain or a loss. Runs that failed, vision replies that could
  not be read and requests neither arm drew are counted apart, by request.
  They also hold the tokens and time per arm,
  and whether each ON run called `plot_code` at all;
- `side_by_side.html`: each request's two figures, OFF left and ON right,
  the vision marks under each, and a box for the owner's own mark;
- for every run, `runs/<request>/<arm>/`:
  - `run.json`: the request, the final reply, the figures, the tool calls,
    token counts and time;
  - `check.json`: the vision model's marks, one line of reason each;
  - `agent/`: the page run's own record (`activity.jsonl`) and `files/`
    with every PNG and file the run wrote.

Every run is read in full on the way back (CLAUDE.md, "REVIEW EVERY MODEL
RUN IN FULL"), so the folder matters more than the counts. Also send the AI
FDE's notes on anything that had to change in the environment.

## 8. Reading the result

One run per request and arm is a measurement, not a verdict. A request is
"better" when ON passes more items than it fails against OFF, "worse" when
fewer, and "same" otherwise. An item one arm marks not applicable is
counted apart, never as a gain or a loss, and so is an item the vision
model left unmarked. A request with a failed run, an unreadable vision
reply or no figure from either arm is not compared at all. A "same" where
ON never called
`plot_code` says ON was not used, not that it gained nothing. The owner's
own marks on the side-by-side page come before the vision model's.
