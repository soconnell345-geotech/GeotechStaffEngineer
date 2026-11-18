# EV-49: the review suite clears the image ceiling between arms (check notes, 2026-11-18)

## What the item asked

`review_flags.switches()` makes each arm exactly what it says by unsetting
every switch and switch setting the arm does not name, and `describe()`
names the switches that are set (the turn's `build.switches`). The turn's
image-call ceiling, `GEOTECH_TURN_MAX_IMAGE_CALLS` (TA-43), was in none of
those sets: a value set in the notebook's environment applied to every arm,
the baseline included, and showed only on `turn_end.image_limit`. A look the
ceiling refuses changes what a run returns. Done when an arm that does not
name the setting runs with no ceiling though the environment sets one, and
a test says so.

## What I built, and where

`packages/aec-agent-core/aec_agent_core/review_flags.py`:

- `LATER_SETTINGS_ENVS = ("GEOTECH_TURN_MAX_IMAGE_CALLS",)`, beside
  `SETTINGS_ENVS` and apart from it: the geotech application's
  `funhouse_agent/deep/tests/test_review_switches_offline.py` pins
  `SETTINGS_ENVS` whole, so the setting waits outside it as `LATER_ENVS`
  does for the later switches, until DR-51 folds both in. Named as a string,
  as the other settings are, so the module still imports nothing; a test
  holds it equal to `deep.limits.TURN_MAX_IMAGE_CALLS_ENV`.
- `switches()` clears it unless the arm names it, and restores the
  notebook's value afterwards.
- `describe()` names it when it is set (`GEOTECH_TURN_MAX_IMAGE_CALLS=3`),
  after the switches; the turn's `build.switches` then shows it. Unset, the
  line is what it was.
- In `__all__`.

Nothing the model is handed changes, and nothing changes for a tester: the
page reads the ceiling as before; only the suite's arms and the turn's
record of its switches differ.

## Tests

New: `packages/aec-agent-core/tests/test_ev49_image_ceiling_between_arms.py`
(each fails before the change):

- `test_the_ceiling_is_a_later_setting_named_as_the_limits_name_it`.
- `test_an_arm_that_does_not_name_the_ceiling_runs_without_one`: with the
  environment at 5, `baseline`, `sheet_tools` and an arm naming another
  setting each run with no ceiling and do not name it; 5 is back after.
- `test_an_arm_that_names_the_ceiling_runs_with_it_and_says_so`.
- `test_a_suite_run_clears_the_ceiling_the_environment_sets`: the
  done-when through the runner, on a made-up page host: with the
  environment at 5, a task under `baseline` runs with no ceiling and its
  switches line does not name one; under an arm `{ceiling: 3}` the turn
  has 3 and names it.

Run through the runner, each exit 0: from the root the new file with
`test_vis33_photo_pages_one_look.py`, `test_vis34_tile_answers.py`,
`test_dr48_photo_pages.py`, `test_core_suite_and_records.py` (the core's)
and `funhouse_agent/deep/tests/test_review_switches_offline.py`,
`test_judgment_calls.py` (the geotech application's, which pin the switch
sets); in Document Review `tests/test_review_switches_offline.py`,
`tests/test_review_fingerprints.py`, `tests/test_dr48_photo_pages_turn.py`.

## For the lead

- DR-51 gains one line when it is done: fold `LATER_SETTINGS_ENVS` into
  `SETTINGS_ENVS` with `LATER_ENVS` into `ALL_ENVS`, in the same renewal as
  the geotech application's two tests (its text already names EV-49's
  setting).
- Brief 9's note in EV-45 ("with `GEOTECH_TURN_MAX_IMAGE_CALLS` unset in the
  notebook, since the suite does not clear it between arms") no longer
  holds once this is released: the suite clears it, and an arm that wants a
  ceiling names it.
