# The hosted platform's four small asks: check (2026-11-16)

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
