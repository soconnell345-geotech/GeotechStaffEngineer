# INF-33, the app split's step 13: Document Review as `aec-document-review`

Branch `claude/inf-33-step13`, from master `1caf160`. What governs is BACKLOG
INF-33's step-13 text (owner 2026-11-09: a public package on PyPI, a wheel by
path as the backup), with plan `module_work/APP_SPLIT_PLAN.md` Step 13 read
for its content ("private repository" read as `packages/aec-document-review`).

Nothing here was tagged or released. The lead's release order is section 9.

## 1. What was copied, from where

Copies, not moves: GSE keeps every file below until step 15. The copy is
mechanical (`module_work/inf33/copy_step13.py`: each file under its new
import name; GSE's old names of core modules (its `CORE_ALIASES`) become
`aec_agent_core.*`); the hand edits that follow are in commit `60cb917`
(section 2).

| From | To (`packages/aec-document-review/aec_document_review/`) | Plan |
|---|---|---|
| `funhouse_agent/review_prompt.py` | `review_prompt.py` | 2.2 |
| `funhouse_agent/review_build.py` | `review_build.py` | 2.2 |
| `funhouse_agent/deep/review_agent.py` | `review_agent.py` | 2.2 |
| `funhouse_agent/deep/minimal_agent.py` | `minimal_agent.py` | 2.2 |
| `funhouse_agent/review_eval/tasks.py` | `review_eval/tasks.py` (its 38 review-page tasks) | 2.2 |
| `webapp/review_page.py` (the review profile, since INF-31) | `review_page.py` | 2.2 |
| `webapp/app.py`, `core.py`, `turn_jobs.py`, `turn_record.py`, `output_capture.py`, `budget_panel.py`, `feedback.py`, `version_guard.py`, `diagnostics.py`, `profiles.py`, `page_profile.py`, `pages_entry.py`, `file_widgets.py`, `page_errors.py`, `conversation_export.py`, `conversation_restore.py`, `sharepoint_store.py`, `sharepoint_tools.py`, `sharepoint_log.py`, `identity.py`, `suite_page.py` | the same names | 2.9 |
| `webapp/helper_models.py`, `ws_upload.py`, `ws_upload_component/index.html` | the same names | 2.9 (see 2) |
| `webapp/tinyapps_settings.py`, `tinyapps_keys.py`, `graph_sharepoint.py`, `tinyapps_entry.py` | the same names | 2.9, host files |
| `funhouse_agent/deep/notebook.py`, its first part (the stream formatters the page uses) | `stream_format.py` | 2.9 |

New in the package: `__init__.py` (version, hooks, app details), `notice.py`
(the page's notice), `entry.py` (the Streamlit script), `launch.py` (the
`aec-document-review` command), `review_eval/__init__.py` (the suite's door).

Left out, as plan 2.9 says: `webapp/geotech_page.py`; the Databricks and
Foundry host files (`databricks_launcher.py`, `foundry_entry.py`,
`foundry_dataset_store.py`, `reference_fetch.py`); the e-mail tool
(`email_tools.py`, OWN-69; the shell's call to it is inside a `try`, so it
is simply absent); `webapp/README.md` and `requirements.txt`. From the tasks:
GSE's four geotech-page tasks (`report-extract-all`, `report-extract-diggs`,
`report-summary-vs-sheets`, `scale-log-depth`) and the report fixture, which
stay in GSE. `tinyapps/TINYAPPS.md` and `tinyapps/wrapper_repo/` stay where
they are (step 14 revises them; the README links them by repo path).

**Not left out, against plan 2.9:** `helper_models.py` (the per-helper model
choice) and `ws_upload.py` (the websocket uploader). Each has a dozen call
sites in `app.py`; taking them out is page work, not a copy, and on Tiny
Apps neither does anything (helper models are offered only on Foundry; the
websocket uploader only when `GEOTECH_UPLOAD_MODE=ws`, which only the
Databricks launcher sets). The package can drop them as its own change.

## 2. The hand edits after the copy (`60cb917`)

- **One page.** `profiles.py` serves only `DOCUMENT_REVIEW`, which is the
  default. Its single-user folder stays `<data root>/pages/document_review`
  and multi-user `<data root>/users/<key>/document_review`, as beside the
  geotech page, so a deployment finds its conversations; `app.py` still tags
  every conversation with its page. `pages_entry.py` declares one page at the
  app root (Streamlit draws no page list for one page).
- **The page's own parts** (`review_page.py`): its own notice (`notice.py`,
  a placeholder until OWN-78: GSE's speaks of geotechnical analysis and SI
  units); no judgment-call check at a turn's end (GSE's; the review build
  never asked for a consult, so it only recorded); no reference-PDF line in
  the diagnostics; `version_dists=("aec-document-review",)`.
- **The shell:** the geotech-page name fallbacks in `core.py` and
  `diagnostics.py` removed; the diagnostics' optional-packages line (the
  analysis back ends) removed; the versions line names `aec-document-review`,
  `aec-agent-core` and planlens; feedback records this distribution's
  version; the shell's fallback notice is the page's; the default markup
  signer in `identity.py` is "Document Review (AI draft)".
- **The review build** (`review_build.py`): no chart-lines registration (the
  deliberate change, section 4).
- **The suite:** `suite_page.py` names `aec-document-review` /
  `aec_document_review` (run.json) and serves only the review page;
  `review_eval/tasks.py` holds the 38 review-page tasks, `PAGES = ("review",)`.
- **Entries:** `tinyapps_entry.main()` keeps its name and runs the one page;
  `entry.py` is the script for `streamlit run`; `aec-document-review`
  (`launch.main`) runs it, passing its arguments to Streamlit.

## 3. The hooks the package fills

Loading `aec_document_review` (its `__init__`, strings imported only when
the core first asks):

| Hook | Filled with |
|---|---|
| `prompter.get_setting` | `aec_document_review.tinyapps_settings:get_setting` (env, then Key Vault by `KV_NAME`, then `.env`) |
| `prompter.resolve_for`, `prompter.tester_keys_possible` | `aec_document_review.tinyapps_keys:*` (per-tester keys) |
| `app.details` (new, section 5) | name "Document Review", dist `aec-document-review`, package `aec_document_review` |

Loading `aec_document_review.review_eval`: `review_eval.open_tasks` (its 38
tasks) and `review_eval.page_host` (`aec_document_review.suite_page:host`).

Left unfilled, on purpose: `review_eval.document_dirs`, `.fetch_document`,
`.fixture` (documents come from the suite's `docs_dir` or
`$GEOTECH_REVIEW_EVAL_DOCS`; the synthetic ones from `planlens.testing`);
`held_records.record_fields`, `cited_documents.reference_modules` and the
three `vision_tools.*` reference hooks (no record shapes, no reference
library). The chart-lines registration
(`aec_agent_core.charts.register_reference_charts`) is left empty.

## 4. The one deliberate fingerprint change

The package has its own fingerprints, `tests/review_fingerprints.json`,
recorded with `python -m tests.page_builds record` (from the package folder;
`tests/page_builds.py` is GSE's `webapp/tests/page_builds.py` with the
geotech build and the boundary half taken out), and its own test,
`tests/test_review_fingerprints.py`.

Compared with GSE's `webapp/tests/page_fingerprints.json` (the review builds'
app parts), the only differences are the plan's:

```
review_default 4
/calls[0]/subagents[0]/tools/plot_code/args
/calls[0]/subagents[0]/tools/plot_code/description
/calls[0]/subagents[0]/tools/plot_data/args
/calls[0]/subagents[0]/tools/plot_data/description
review_minimal 0
```

That is the figure helper's `plot_data` without `reference_chart` and its
`plot_code` without `chart_lines()` and the note. (The primary agent's own
`plot_data` was already built with `plot_reference_charts=False`, so it does
not change.) `app_environment` records `reference_charts_registered: false`,
and a test keeps it so. The planlens part is recorded under planlens 0.15.0
from PyPI (surface `709cbea6e23e8aaa`).

GSE's `webapp/tests/page_fingerprints.json` is byte-identical to master's
(not touched on the branch; `test_page_fingerprints.py` passes, section 8).

## 5. The core names no application

From step 11 (c)'s review (`module_work/inf32/CHECK_STEP11C.md` section 12),
commit `5f7b5de`. New core module `aec_agent_core/app_details.py`: the page
hands in its name, distribution and package once, through the hook
`app.details` (`hand_in(name=, dist=, package=)`); nothing handed in, a
neutral name ("AEC agent"), no distribution.

| Place | Before | Now |
|---|---|---|
| `activity_log._app_version()` (`process_start.app_version`) | version of `geotech-staff-engineer` | version of `app_details.dist()`; None if none |
| `document_tools.DEFAULT_MARKUP_AUTHOR`, `OWN_MARK_WORDS`, `markup_author()` | "GeotechStaffEngineer (AI draft)", ("GeotechStaffEngineer", "(AI draft)") | "<name> (AI draft)", (<name>, "(AI draft)"), read at each use (module `__getattr__`, merged into the existing one) |
| `docx_renderer.DEFAULT_AUTHOR` | "GeotechStaffEngineer" | <name>, at each use |
| `review_eval.runner.SUITE_AUTHOR`; `score_review_suite(sharepoint_folder=)` default | "Review suite via GeotechStaffEngineer (AI draft)"; "GeotechStaffEngineer/review_eval" | "Review suite via <name> (AI draft)"; "<name>/review_eval" when not given |
| `runtime_check` messages | "funhouse_agent.deep needs / tried to reload / reloaded…" | "<package> needs…" |

GSE hands in today's values (name "GeotechStaffEngineer", dist
`geotech-staff-engineer`, package `funhouse_agent.deep`) from each of its
three doors, whichever loads first: `funhouse_agent/__init__.py`,
`webapp/__init__.py`, `calc_package/__init__.py` (calc_package writes Word
files through the core without loading funhouse_agent). So GSE's output is
unchanged: `webapp/tests/test_app_details.py` checks each door in a fresh
interpreter; GSE's existing tests of these values
(`deep/tests/test_document_tools_offline.py`,
`calc_package/tests/test_docx_roundtrip.py`, `test_tinyapps_wiring.py`)
pass unchanged; core test `packages/aec-agent-core/tests/test_app_details.py`
covers the default, a handed-in app and a broken hook. Document Review hands
in its own (`tests/test_package.py::test_the_core_names_this_app`, and the
turn test checks the Word file's author).

`aec-agent-core` is 1.0.0 (commit `e0d168d`): its pyproject, `__version__`
and GSE's exact pin; the core's README and `hooks.py` name the new hook.
`webapp/tests/test_wheels.py` checks the three agree (passes).

## 6. The freeze (`f2da01d`)

`webapp/tests/test_review_freeze.py` holds the sha256 (LF line ends) of
this repository's review files at the copy: `funhouse_agent/deep/
review_agent.py`, `deep/minimal_agent.py`, `review_prompt.py`,
`review_build.py`, `webapp/review_page.py` (the review profile's file),
`webapp/tinyapps_entry.py`, and `funhouse_agent/review_eval/tasks.py` (the
review tasks). A change fails with a message saying review-page changes go
to `packages/aec-document-review/aec_document_review/`, and that GSE's own
change (its four geotech-page tasks, or a fix the two-page app needs before
step 15) updates the hash in the same commit, saying why. A second test
checks each frozen file has its copy in the package.

## 7. CI and publishing (`7b576f6`, `f774764`)

- **New job `document-review`** in `.github/workflows/tests.yml`, Python
  3.11 and 3.12, no submodule, nothing geotech: builds the package's sdist
  and wheel and checks the wheel's file list (no `tests`, no `module_work`;
  the entry modules, the tasks and the uploader page present); installs the
  wheel with the checkout's core, planlens at its pin and the fleet pins
  (`tinyapps/wrapper_repo/requirements-thirdparty.txt`); fails if anything
  geotech is installed; imports every module and builds the page from the
  installed wheel outside the checkout; runs the package's tests from its
  folder (page and review tests, `tests/test_boundary.py`, the
  fingerprints; on CI a planlens with no recorded entry fails).
- The existing `core-only` job is unchanged: it still builds today's review
  page from this checkout's (frozen) files, until step 15.
- **`publish-document-review.yml`**: the tag-matches-version check was
  already there; added a step that refuses to publish while either exact pin
  (`aec-agent-core`, `planlens`) is not on PyPI, as `publish.yml` does for
  GSE. Run locally against PyPI today: planlens 0.15.0 found, aec-agent-core
  1.0.0 not, exit 1 (as it should be until the core is published).

## 8. Tests (judged by pytest's exit code)

**Fresh venv, nothing geotech** (`python3.12 -m venv`; installed: the core
wheel `aec_agent_core-1.0.0`, `planlens==0.15.0` from PyPI, the package
wheel `aec_document_review-0.1.0`, the fleet pins; `pip list` has no
geotech-staff-engineer or geotech-references; pandas 3.1.1, streamlit
1.62.0):

| Check | Exit | Result |
|---|---|---|
| every package module imports from the installed wheel (outside the checkout, `python -I`), the page builds through `core.build_agent`, no geotech module loaded | 0 | `built CompiledStateGraph ; geotech loaded: []` |
| the package's tests (`packages/aec-document-review`, `python -m pytest -q`) | 0 | 803 passed, 3 skipped |
| of which: the page's real turn path on both builds (`tests/test_page_turn.py`: `core.stream_turn` to `turn_done`, `plot_data`, `calculate`, `write_docx`, the helper) | 0 | in the above |
| of which: the boundary (`tests/test_boundary.py`) and the fingerprints | 0 | in the above |
| of which: the suite's dry run (`tests/test_package.py::test_the_suite_dry_run_lists_its_tasks`: "# Dry run", 38 tasks) | 0 | in the above |
| `aec-document-review --server.headless true` (the installed command), then the page in Chromium | | page served: the "No engine configured" banner, "aec-document-review 0.1.0 · engine: none", the Document Review header |

The package's tests: its own (fingerprints 7, turn 3, boundary 2, package
12) and GSE's page and review tests copied through the same rename
(`module_work/inf33/copy_tests_step13.py`; then by hand: the app path, the
geotech-page monkeypatches and the optional-packages tests taken out).
Left out, as needing GSE code or testing the geotech page, listed by file in
`module_work/inf33/tests_left_out.json`: whole files
`test_review_eval_coverage_offline.py` (GSE's report fixture) and
`test_download_mime.py`; the tests of other files whose code imports GSE;
and 57 that failed in the package for that reason (the specialist picker,
the geotech page's step cap and folder layout, GSE's version and name,
GSE's builders), removed by `module_work/inf33/prune_failing_step13.py`.
The 3 skips are the copied tests' own skips.

**This repository** (GSE venv: `pip install -e packages/aec-agent-core -e .`,
Python 3.12, four workers):

| Suite | Exit | Result |
|---|---|---|
| `webapp/tests funhouse_agent/deep/tests funhouse_agent/review_eval packages/aec-agent-core/tests` (one run, `-n 4`) | 0 | 2588 passed, 9 skipped |
| of which, run again on their own: `test_page_fingerprints.py` (GSE's fingerprints unchanged; `git diff origin/master -- webapp/tests/page_fingerprints.json` is empty), `test_review_freeze.py`, `test_review_boundary.py`, `test_app_details.py`, `test_public_text.py`, `test_wheels.py` | 0 | 42 passed, 5 skipped (the wheel builds, opt-in) |
| `GEOTECH_REQUIRE_WHEEL_TEST=1 pytest webapp/tests/test_wheels.py` (builds and installs GSE's and the core's wheels) | 0 | 11 passed |
| `PYTHONPATH=. python -m webapp.tests.core_only` in the geotech-free venv (the core-only job's page check, today's review files) | 0 | "core-only page check passed: review_default, review_minimal build, run and match" |
| `packages/aec-agent-core/tests` alone, after the app-details change (GSE venv) | 0 | 230 passed (with `webapp/tests/test_app_details.py`) |

Nothing failed, so no failure had to be compared with unmodified master.

**Wheels** (`python -m build`, sdist then wheel):

- `aec_agent_core-1.0.0-py3-none-any.whl`: 75 entries, the
  `aec_agent_core` package only (now with `app_details.py`), LICENSE,
  metadata.
- `aec_document_review-0.1.0-py3-none-any.whl`: 48 entries:
  `aec_document_review/` (39 modules: `__init__`, `app`, `budget_panel`,
  `conversation_export`, `conversation_restore`, `core`, `diagnostics`,
  `entry`, `feedback`, `file_widgets`, `graph_sharepoint`, `helper_models`,
  `identity`, `launch`, `minimal_agent`, `notice`, `output_capture`,
  `page_errors`, `page_profile`, `pages_entry`, `profiles`, `review_agent`,
  `review_build`, `review_page`, `review_prompt`, `sharepoint_log`,
  `sharepoint_store`, `sharepoint_tools`, `stream_format`, `suite_page`,
  `tinyapps_entry`, `tinyapps_keys`, `tinyapps_settings`, `turn_jobs`,
  `turn_record`, `version_guard`, `ws_upload`, `review_eval/__init__`,
  `review_eval/tasks`), `ws_upload_component/index.html`, and the
  dist-info (LICENSE, METADATA, WHEEL, `entry_points.txt` with the
  `aec-document-review` script, `top_level.txt`, RECORD). No tests, no
  `module_work`, no data files that name anything real (the task truths are
  the public Mecklenburg sheets, public UFCs and planlens' synthetic
  fixtures, as in GSE's public repo; `webapp/tests/test_public_text.py`
  walks the package too and passes).
- Metadata requires `aec-agent-core==1.0.0`, `planlens==0.15.0`,
  `streamlit>=1.39`, `requests>=2`, `websockets>=14,<16`; extra `azure`
  (azure-identity, azure-keyvault-secrets, cryptography).

## 9. What the lead runs to release (pre-approved, owner 2026-11-09)

In this order; each tag publishes through its own workflow (trusted
publishing, OWN-128):

1. Merge `claude/inf-33-step13` to master; CI green, including
   `document-review` on 3.11 and 3.12.
2. **The core:** tag `core-v1.0.0` on that master commit and push the tag
   (`publish-core.yml`; it checks the tag equals the version). Check
   https://pypi.org/project/aec-agent-core/1.0.0/ exists.
3. **GSE pinned to it:** GSE's release as the lead cuts it (the version bump,
   CLAUDE.md state block, HANDOFF §0a-current, the install guide's history
   table, the BACKLOG sweep, the manual rebuild, per the READ-FIRST release
   row), then tag `v<version>` (`publish.yml`; its core check passes once
   step 2 is on PyPI). This is step 12's exit: GSE on Foundry from the new
   wheels (INF-25's list).
4. **Document Review:** tag `document-review-v0.1.0`
   (`publish-document-review.yml`: tag equals version; refuses while
   aec-agent-core 1.0.0 or planlens 0.15.0 is missing from PyPI). Check
   https://pypi.org/project/aec-document-review/0.1.0/.
5. The Nexus mirror carries the three a day or two later.

The backup, if PyPI or Nexus fails: the two wheels of section 8, installed
by path (`pip install aec_agent_core-1.0.0-py3-none-any.whl
aec_document_review-0.1.0-py3-none-any.whl planlens==0.15.0`).

## 10. What step 14 (the Tiny Apps wrapper) needs

- **The wrapper's `app.py` import line** becomes
  `from aec_document_review.tinyapps_entry import main` (then `main()`),
  in place of `from webapp.tinyapps_entry import main`. Same function name,
  same behaviour (local `.env`, `GEOTECH_DEPLOYMENT=tinyapps`, the Prompter
  engine), one page at the root.
- **`packages.txt`:** `aec-document-review==0.1.0` (it pulls
  `aec-agent-core==1.0.0` and `planlens==0.15.0`; pinning those two as well
  is harmless), `-r requirements-thirdparty.txt`, and `pandas==3.0.2` can be
  restored: nothing in this install caps pandas (python-ags4 came only with
  geotech-staff-engineer). The fresh venv above resolved pandas 3.1.1 with
  streamlit 1.62.0. The Key Vault libraries are in the fleet pins (or
  `aec-document-review[azure]`).
- **Route B (wheels by path)**: the two wheels, as above.
- **Settings unchanged:** the package reads the same names (`PROMPTER_*`,
  `KV_NAME`, `GRAPH_*`, `SHAREPOINT_SITE_URL(S)`, `GEOTECH_SHAREPOINT_ROOT`,
  `GEOTECH_WEBAPP_DATA`, `DEV_IDENTITY`, the identity header). The
  SharePoint default root is still `Shared Documents/GeotechStaffEngineer`,
  which is where the owner said testers' conversations go (INF-34); a
  tester's conversations land in the same `users/<key>/document_review`
  folder as on today's two-page app.
- `webapp/tests/test_tinyapps_wrapper.py` and the template still name the
  GSE route; step 14 revises them, `.env.example`, the README and
  `TINYAPPS.md` (the test-machine recipe loses "switch to the geotech page").
- The notice (OWN-78) is a placeholder in `notice.py`.

## 11. Noted, not fixed

- `review_prompt.py` (the prompt, in both copies) still tells the agent to
  point a person with a geotechnical report to "the GeotechStaffEngineer
  page", which the package's app does not have. Changing it changes the
  prompt's fingerprint, so it is left for the package's first own change.
  **Fixed in the package by review fix F4 (section 12); GSE's frozen copy
  keeps it.**
- `runtime_check._restart_hint` still says "see the funhouse agent guide";
  left so GSE's message is unchanged.
- The core's `review_eval/documents.py` `DOCUMENTS` still lists GSE's
  `fixture_report` (task data; resolved through GSE's fixture hook, unfilled
  in the package, so a task naming it there would report the document
  missing). No package task names it.
- The suite's default SharePoint folder for Document Review is
  "Document Review/review_eval" (from its name).
- Plan 2.9's `helper_models.py` and `ws_upload.py` are in the package
  (section 1).
- The copied tests left out (section 8) are GSE's still; the package's
  coverage of the Foundry-only paths in the copied shell is thinner. (Seven
  that test this page came back, adapted: section 12, F7.)

## 12. Review fixes

An independent review of the branch at `291fd34` (verdict: merge after
fixes) found seven things. All are fixed here, one commit each, so testers
get them in 0.1.0. The core's messages are unchanged, so GSE's output stays
as it was; none of the frozen files changed (`git diff 291fd34` of the seven
files and of `webapp/tests/page_fingerprints.json` is empty).

| Finding | Commit | Fix |
|---|---|---|
| F1: the `document-review` CI job failed on both Pythons (run 42732084307) | `d2ad659` | `python -I -` in the "nothing geotech" check. Run from the checkout root, plain `python -` put the root first on `sys.path`, so `find_spec` found the repository's own `webapp/` and `funhouse_agent/`. The wheel-listing step, also run from the root, gets `-I` too. The page-build step already left the checkout with `-I`; the tests step runs from `packages/aec-document-review`, so the root is not on its path. |
| F6: `requires-python >=3.10`, but every deepagents in the core's range needs 3.11 | `15d7d73` | `>=3.11` in both pyprojects (the core drops its 3.10 classifier); the package README says 3.11 or later. GSE's own pyproject still says `>=3.10` (not this step's; GSE pins the core, so on 3.10 it fails to resolve as before). |
| F2: the sdists carried the tests, and with them field-session place names | `189ee3d` | `MANIFEST.in` with `prune tests` in both packages; the guard below; the place names in the package's copied tests and in the GSE tests they came from replaced (test values by made-up names; comments that date a field session describe the job instead). None of those GSE files is frozen. |
| F4: the prompt sent the person to the GeotechStaffEngineer page | `0ca377a` | below: the second deliberate fingerprint change |
| F5: GSE's name on three texts a Document Review user sees or is sent | `4befd86` | the SharePoint report a tester copies (`diagnostics`), `MOVED.txt` (`sharepoint_store`) and the connection test's probe file (`graph_sharepoint`) say Document Review. The SharePoint default root stays (the owner's folder, INF-34). A test in `tests/test_package.py` reads every string in the package's code (docstrings aside) and fails on GeotechStaffEngineer anywhere else. |
| F7: copies of review-page tests dropped although they test this page | `97a533f` | seven back, adapted to the package's own name, record and cap: the markup author, the own-mark words, run.json's version, turn_end's step cap (the page's budget raises the host's 40 to `geotech_min_recursion_limit`), a turn stopped at its cap (budget off), the entry scripts' page-error route and a stop or rerun (the embedded page script imported `webapp.page_errors`). The dead reference-PDF test (importorskip `geotech_references`) and its section are removed. `tests_left_out.json` lists 49 left out after a run (was 56). |

**The second deliberate fingerprint change (F4).** "What you are not"
told the model to point a person whose question turns into an analysis to
the GeotechStaffEngineer page. A Document Review deployment has no such
page, so the line now says: say plainly that this page does not run
analyses, so that check needs an engineer's own calculation. Re-recorded
with `python -m tests.page_builds record` from the package folder, in the
fresh venv (nothing geotech, planlens 0.15.0, surface `709cbea6e23e8aaa`):
only `review_default`'s `system_prompt` hash and length change (9839 to
9854 characters); `review_minimal`'s prompt does not carry the line.
`test_review_fingerprints.py`'s docstring names the change. GSE's copy
(`funhouse_agent/review_prompt.py`, frozen) and its `page_fingerprints.json`
are unchanged, so section 4's comparison with GSE now also differs by this
one prompt hash.

**The sdist guard.** `webapp/tests/test_sdists.py` builds each package's
sdist (`python -m build --sdist --no-isolation`) from a copy of its folder
in a temporary folder, so nothing is written into the source tree (seconds),
and checks that no `tests/` is in it and that the package files are. It runs
where `build` and setuptools 77 or later are installed and skips otherwise;
CI's Wheels job runs it with `GEOTECH_REQUIRE_WHEEL_TEST=1`, so there a
missing tool fails. Checked both ways: without the two MANIFEST.in files it
fails for both packages.

**The minimal build gets no deployment instructions.** The page's default
build goes through the core's `assembly.compile_agent` (`review_build.py`),
so it picks up what the core adds there: REF-64 (B)'s deployment
instructions, once both are in the core release the package pins. The
`minimal` build (`GEOTECH_REVIEW_AGENT=minimal`, `minimal_agent.py`) calls
`langchain.agents.create_agent` directly and gets none, here as in GSE.
**It is never the default on Tiny Apps:** `review_flags.minimal_agent()` is
true only for exactly `minimal`, and nothing sets it: not the wrapper
(`tinyapps/wrapper_repo/app.py`, `.env.example`), not the package's
`tinyapps_entry` or `tinyapps_settings`, not the core. It is a measuring
stick for the review suite's `minimal` arm; testers get the default build.

**Re-runs** (foreground, one at a time, no other pytest, at least 2 GB free;
judged by exit code):

| Check | Exit | Result |
|---|---|---|
| both packages built from `git archive HEAD` (sdist, then wheel) | 0 | sdists: no `tests/`; wheels: the same file lists as at `291fd34`; `Requires-Python: >=3.11` |
| the CI job's three checks against those wheels in the fresh venv: the wheel list; nothing geotech, run from the checkout root; the page build outside the checkout | 0, 0, 0 | `not to ship: [] missing: []`; `geotech installed: [] []`; `built CompiledStateGraph ; geotech loaded: []` |
| the package's tests, fresh venv, from `packages/aec-document-review` as CI runs them | 0 | 808 passed, 5 skipped (was 800 and 6: the seven restored and the name test; the dead skip gone) |
| `packages/aec-agent-core/tests` (GSE venv, the worktree's core) | 0 | 227 passed |
| `test_page_fingerprints`, `test_review_freeze`, `test_review_boundary`, `test_public_text`, `test_wheels` (fast part), `test_sdists`, and the four GSE test files whose names changed | 0 | 114 passed, 5 skipped (the wheel builds, opt-in) |
| `GEOTECH_REQUIRE_WHEEL_TEST=1`: `test_wheels.py` alone, then `test_sdists.py` alone | 0, 0 | 11 passed; 2 passed |

A local gotcha: run the opt-in wheel test without `PYTHONPATH` pointing at
the checkout. Its builds leave `*.egg-info` in the source tree; with the
checkout on `PYTHONPATH`, pip in the test's fresh venv takes those for the
wheels already installed and skips the install, and two tests fail (no
`aec_agent_core`, no `geotech-disclaimer.exe`). CI uses editable installs,
not `PYTHONPATH`.

## 13. Master merged (REF-64 (B))

`6490aa6` merges master at `588f0ef` (REF-64 (B), merge `4f22c00`): no
conflicts; the two branches share no files. REF-64 (B) appends a
deployment's instructions (`GEOTECH_INSTRUCTIONS`) in the core's
`compile_agent`, which the package's default review build goes through, so
core 1.0.0 carries it for Document Review too.

`b7c3576`: REF-64 (B) also added a "deployment instructions" line to GSE's
connection diagnostics (`webapp/diagnostics.py`); the package's copy of that
file, made before, lacked it. The same `_instructions_check` now runs in the
package's diagnostics, and the package README's settings table names
`GEOTECH_INSTRUCTIONS` (this page's roles: `main`, `general-purpose`,
`figure`). New `tests/test_deployment_instructions.py` keeps what the probe
below showed.

**Probe** (fresh venv, the core and the package rebuilt from the merged tree
and reinstalled; a made-up instructions folder with `all.md`, `main.md`,
`general-purpose.md`):

- the default review build: the block once, last, on the main prompt and on
  each helper (`figure`, `general-purpose`), one heading and one copy of the
  protected rules each; the build record names each role's files and
  characters, and `prompt_chars` is the compiled prompt's;
- a real turn through the page's turn job writes `turn_start` whose
  `engine.build.appended` names `main`, `general-purpose` and `figure`, with
  the main block's sha;
- the minimal build gets nothing (no heading in any prompt, no `appended`);
- unset: no heading anywhere, no `appended`, and the package's fingerprints
  equal `tests/review_fingerprints.json` for both builds (no differences;
  the planlens surface is the recorded one).

**Re-runs** (foreground, no other pytest, at least 2 GB free; exit codes):

| Check | Exit | Result |
|---|---|---|
| both packages built from the merged tree: sdists, wheels | 0 | no `tests/` in either sdist; the core's wheel has `deep/instructions.py` |
| the CI job's three checks against those wheels | 0, 0, 0 | as in section 12 |
| the package's tests, fresh venv, from its folder | 0 | 813 passed, 5 skipped (808 and the 5 new) |
| `packages/aec-agent-core/tests` | 0 | 237 passed, 1 skipped (POSIX permissions, in REF-64's own test) |
| the gates of section 12, `test_deployment_instructions.py` (GSE's) and `test_app_details.py` | 0 | 127 passed, 5 skipped (the wheel builds, opt-in) |
| `GEOTECH_REQUIRE_WHEEL_TEST=1`: `test_wheels.py` alone, then `test_sdists.py` alone | 0, 0 | 11 passed; 2 passed |
