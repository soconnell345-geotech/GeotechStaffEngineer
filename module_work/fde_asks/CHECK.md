# The hosted platform's small asks: check (2026-11-16)

Branch `fix/fde-small-asks` from master 6941d9d, one commit per ask. The
evidence is build j's conversations, which ran app 5.35.0 with
aec-agent-core 1.0.1. Each point below was checked against today's master
before anything changed.

## 1. Open in new tab cannot work on the hosted platform

- **What master did.** On the hosted platform every download is the
  in-page component (`webapp/file_widgets.py`), which has two buttons:
  Download and **Open in new tab**. The platform's app frame does not let
  the page open a tab, so the second button did nothing. The PDF preview's
  note still told the user to use it. Other hosts use Streamlit's own
  button, which has no Open in new tab.
- **What changed.** There is a new setting, `GEOTECH_OPEN_IN_NEW_TAB`: `1`
  shows the button and `0` hides it. Unset, or set to any other value, the
  button is hidden when `GEOTECH_DEPLOYMENT` names the hosted platform and
  shown elsewhere. When hidden, the component carries neither the button
  nor its script, and no text names it. The PDF note then says "Download
  it above to read the whole file." (or "Prepare the download above to
  save the whole file." for a file over 2 MB). The setting is documented in
  `docs/FOUNDRY.md` §4a, beside the other download settings.
- **Tests.** `webapp/tests/test_file_widgets.py` has eight cases of the
  switch and the PDF note hidden and shown, with made-up bytes. All 30
  tests in the file pass.

## 2. A cap on plot retries

- **The run.** One request for a Mohr circle figure took 401 s and 588,394
  tokens: 206,746 on the main agent, 294,330 on the figure helper and
  87,318 on the references helper. Here is the sequence:
  1. The main agent ran one calculate.
  2. Figure helper, call 1: it drew, looked, drew, looked, drew and looked
     (3 plot_code draws and 3 looks, about 187 s), then answered.
  3. The main agent called the helper again to shorten one line: draw 4,
     look 4.
  4. It called the helper a third time to move one label: draw 5, look 5.
  5. A references check, then the answer.

  Every draw was plot_code under the same figure name.
- **Why it went on.** Master already counted each figure's draws across
  the turn, whoever drew it. From the third draw, the result said "Do not
  draw again", and a re-called helper's brief said its draws count on from
  3. But this was only a note:
  - nothing refused a draw;
  - looks had no cap;
  - a new helper call for the same figure ran in full;
  - there was no time cap.

  Today's master behaves the same on these points.
- **What changed** (`aec_agent_core/plot_box/__init__.py`, `figures.py`,
  `deep/vision_kit.py`, `deep/figure_helper.py`). The draw cap `DRAW_CAP =
  3` is unchanged but now binding, and there is a new look cap, `LOOK_CAP =
  3`. Both apply behind the figure switches.
  - **Draws.** A plot_code or plot_data draw is refused without running in
    two cases. The first is when the figure has already had 3 draws this
    turn, by any helper. The figure is matched by its name, a name the
    code saves under, a name the draw replaces, or the figure it is a
    version of (`plan_v2` of `plan`). The second is when the figure helper
    has already drawn 3 times in its run.
  - **Looks.** An analyze_image look at a figure drawn this turn (its
    picture or its look copy) is refused after its third look. A file that
    is not a figure drawn this turn is never refused.
  - **What the helper is told.** The refusal is one line: the last drawing
    stays as the best version, so hand it back and say that the cap was
    reached and what is left. The helper's record, written in code for
    the main agent, adds one line: "The cap was reached … Show the figure
    as last drawn; do not ask for it to be drawn again this turn." A
    redraw brief that names a capped figure says that plot_code and
    plot_data refuse another draw of it.
- **The numbers.** One figure costs at most 3 draws and 3 looks per turn,
  however many times the helper is called. Each helper run was already
  capped at 14 model calls (`DEFAULT_FIGURE_MAX_MODEL_CALLS`); that cap is
  unchanged. No time cap was added, because time follows from the draws,
  looks and calls. In the run above, draws 4 and 5 and looks 4 and 5 would
  have been refused. Calls 2 and 3 would have had the cap line in their
  brief and a refused draw.
- **Tests.** There is a new test in
  `funhouse_agent/deep/tests/test_figure_turn.py`: three draws in one
  helper call, then a second helper call and a version drawn by another
  helper are refused. The file is unchanged, a look at the look copy is
  refused, an upload is not, and the record names the cap. In
  `test_figure_runs.py`, the figure helper's fourth draw is now refused.
  Results: `test_figure_turn.py` 21 passed, `test_figure_runs.py` with
  `test_plot_box_drafts.py` 32 passed, `test_plot_box.py` 70 passed.

## 3. The diagnostics versions line

- **What master did.** The line listed python, geotech-staff-engineer,
  geotech-references, planlens, numpy, streamlit, the langchain packages,
  deepagents and websockets. It did not list aec-agent-core.
- **What changed** (`webapp/diagnostics.py`). aec-agent-core now follows
  geotech-staff-engineer, and shows NOT INSTALLED when it is missing.
  aec-document-review is listed only when it is installed.
- **Tests.** There is a new test with made-up version numbers in
  `webapp/tests/test_diagnostics.py`; all 36 tests in the file pass.

## 4. WE-SF-1 and depth factors

- **What master did.**
  - **The tool.** For an embedded footing, the result already gives `dc`,
    `dq` and `dgamma`. Its "depth factors" choices entry gives their values
    (1.311, 1.263 and 1.000 for WE-SF-1, under the default `'hansen'`).
    `'none'` and `'gec6'` are listed as cited alternatives, and the
    `depth_factor_method` parameter text offers `'none'` with GEC-6 Sec.
    5.2.3.3.
  - **The record.** WE-SF-1 explained the gap only by dividing the terms
    by the depth factors. It did not name the input that reproduces the
    print, and it called the default "GEC-6-style". The default is Brinch
    Hansen; `'gec6'` takes 1.0 at Df/B = 0.83.
- **What changed** (`funhouse_agent/worked_examples.json`, the record
  only).
  - `report_notes` now says the printed Eq. 8-1 has no depth factors and
    names `depth_factor_method 'none'`.
  - Two calls with that input were added, one dry and one with the water
    table at the base. They give 643.4 kPa against the printed 642.8, and
    595.0 kPa against the printed 594.4 (+0.1 % each).
  - `computed_result` gains those two values, and its comparison now names
    the default correctly.
  - No number the tool gives changed: the default run still gives 804.2
    and 755.9 kPa.
- **Tests.** There are two new tests in
  `funhouse_agent/tests/test_worked_examples.py`. One checks that the
  record names the input and that its `'none'` calls give the record's own
  values. The other uses a made-up footing to check that the result names
  the depth factors with their values and offers `'none'`. Both pass.

## Checks

- `webapp/tests/test_public_text.py`, `test_user_manual_current.py` and
  `test_page_fingerprints.py`: 19 passed.
- `module_work/doc10/check_manual.py --rebuild --write` left the manual
  unchanged, and `--rebuild` exits 0.
- Only the single test files named above were run here. The suites run
  on GitHub.

## 5. The main agent's step cap, and no empty turn at the cap

- **What master did.**
  - **The default cap.** The main agent's default step cap was 50
    (`DEFAULT_BEHAVIOR["recursion_limit"]`; the sidebar allows 5 to 200).
    A normal submittal review on the hosted platform hit 50 and ended with
    no answer after 292 s. At 150, the next turn answered well, and
    another real review used 42 of its 50 steps.
  - **The message at the cap.** It advised "(e.g. 50-100)", which is
    wrong when the cap is already 50.
  - **The step-cap answer.** `GEOTECH_STEP_CAP_ANSWER` (TA-33) was off and
    never run live. When on, it had two faults:
    - it replaced the plain message, so the setting to raise was lost;
    - it sent the turn's tool calls and results as tool messages in a call
      with no tools, which some model services refuse.
- **What changed.**
  - **(a) The default cap is now 150** (`webapp/core.py`). A new
    constant, `MAX_RECURSION_LIMIT = 200`, also sets the sidebar's
    maximum. The comments, the default test and `funhouse_agent/DESIGN.md`
    are updated. No user doc named 50, and the manual rebuild is
    unchanged. The lean agent's floor is untouched.
  - **(b) The message names the cap it hit**, read from the error's
    "Recursion limit of N":
    - "The agent ran out of its steps for this turn (all 150 of them)
      before it finished. … raise 'Primary step cap (recursion limit)' …
      above 150 (up to 200) and re-ask, or split the request across
      turns".
    - At 200 it says the cap is already at the most the sidebar allows,
      and to split the request.
    - Without a number in the error, it says "(up to 200)".
  - **(c) The step-cap answer is on by default in the app**
    (`webapp.core.step_cap_answer_on`): on unless
    `GEOTECH_STEP_CAP_ANSWER` is `0` (or false, no, off). The library
    default and the eval's `baseline` stay off; `review_flags` changed
    only in its docstring. Two small changes made it sound:
    - The turn's replies, tool calls and tool results go as plain text in
      one message. Each result is cut to 4,000 characters, the whole to
      120,000 characters with its start and end kept. Image data is not
      sent.
    - The plain message, with the cap hit and the setting to raise, stays
      under the written answer.

    If the extra call fails or returns nothing, the plain message shows
    alone, as before. The cost is one more model call with at most about
    30 thousand tokens of input.
  - **(d) The "Primary step cap" help text in the sidebar now says what a
    turn can cost.** Each step sends the whole turn again; a real document
    review that reached 50 steps used about 1.5 million tokens; a turn that
    runs to 150 can cost several times that; a turn at the cap answers from
    the work so far.
- **Tests** (made-up values).
  - `webapp/tests/test_turn_jobs.py`, three new tests with a fake model
    through the real turn job:
    - a turn at the cap answers from its work, with the plain message
      under it;
    - if that call fails, the plain message still shows;
    - `0` turns the answer off.
  - `webapp/tests/test_core.py`:
    - the message names the cap (50, 150, 200 and no number);
    - the switch is on unless `0`;
    - the written-out work stays within its bound and sends no image data;
    - the existing step-cap answer test now checks the one plain-text
      message;
    - the default test now expects 150.
  - Results: `test_core.py` and `test_turn_jobs.py` 104 passed;
    `test_record_ev06.py`, `test_app_smoke.py` and
    `test_review_harness_offline.py` 55 passed; public text, manual and
    fingerprints 19 passed. `check_manual --rebuild --write` left the
    manual unchanged, and `--rebuild` exits 0.

## Review round (2026-11-16)

The independent review found one B in the figure cap (F1), one B to decide
(F2) and three C items. The lead's rulings and what changed:

- **F1, the cap refused a different figure.** Inside one figure-helper call
  the fourth draw of any figure was refused, the first draw of another
  figure included, and the note said that figure had had 3 draws. The caps
  are now kept per figure: `plot_box.draw_refusal` refuses only when the
  figure itself (its name, a name the code saves under, a figure it
  replaces, the figure it is a version of) has had 3 draws this turn. A
  backstop per call stays, `CALL_DRAW_CAP` = 10 draws of any figures; when
  it refuses, the result carries `"cap": "call"` and the note says the
  call's total refused it, not the figure's own draws, and that figures
  not yet drawn need a new call. `looks_report` then names the figures not
  drawn and sends them to a new call; a figure at its own cap is named in
  "The cap was reached for ...". The draw note at draw 3 of a call
  (`draw_cap_note`, still counting the call's draws, against renamed
  drafts) now says a different figure the brief asks for may still be
  drawn.
  - The Mohr run: all five draws were `plot_code` with name `mohr_circle`
    (`save_figure(fig, 'mohr_circle')`), and all five looks were at
    `mohr_circle.png`. One name, so the per-figure key caps that run at 3
    draws and 3 looks; no suffix stripping was needed.
  - Tests (`test_figure_turn.py`): two figures in one call both drawn (plan
    three times, then section); four figures in one call all drawn; a
    fourth draw of one figure in one call refused, named in the record; the
    backstop's note and record. `test_figure_runs.py`'s plot_data case
    goes back to master's expectation: the figure helper's fourth draw of a
    new figure is drawn (draw 4, with the note).
- **F2, the eval on GSE pages now runs at 150 steps in every arm.**
  Accepted: the eval measures the app as shipped. The default comes from
  `webapp/core.py` `DEFAULT_BEHAVIOR["recursion_limit"]` (150), read by
  `webapp/suite_page.default_recursion_limit()` and used by the eval runner
  when no `recursion_limit` is passed. No code change.
- **C1:** `core.answer_at_step_cap` takes `usage=` and fills `"tokens"`
  from the reply's usage; `turn_jobs` adds them to the turn's tokens, so
  the cost shown for a capped turn includes the last call.
- **C2:** a refused look or draw with no helper running (the main agent)
  is worded for it: show the figure as it stands and describe it from the
  figure helper's record; no "hand it back".
- **C3:** the app's turn record (`turn_jobs`, `meta.json`, the activity
  log's engine) shows `GEOTECH_STEP_CAP_ANSWER=1 (app default)` while the
  variable is unset (`core.with_app_switches`); the eval's record, which
  never answers at the cap, keeps the library's line.
- Checks, one file at a time: `test_figure_turn.py` 25, `test_figure_runs.py`
  13, `test_plot_box_drafts.py` 19, `test_figure_helper.py` 15,
  `test_review_harness_offline.py` 23, `test_turn_jobs.py` 10,
  `test_core.py` 97, `test_public_text.py` 5, `test_user_manual_current.py`
  1, `test_page_fingerprints.py` 13: all passed. `check_manual --rebuild
  --write` left the manual unchanged and `--rebuild` exits 0.
