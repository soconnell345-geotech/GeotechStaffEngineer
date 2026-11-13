# Cloud helper brief: Document Review, the shared core and planlens

Binding for every one-time cloud session the lead starts on Document Review, the shared core or planlens. The routine's own prompt names a task file and one section of it; this file holds the rules that do not change between tasks.

You are a cloud helper for the GeotechStaffEngineer project. Its repository builds a geotechnical agent app (GSE) and two packages: the shared core `aec-agent-core` (model connections, the vision and file tools, the agent's guards, the run record) and Document Review `aec-document-review`, an LLM agent that reviews architecture, engineering and construction documents as a Streamlit page. Testers use Document Review on a shared host (Tiny Apps), several people in one process, so one tester's turn must never hold up, see or pay for another's. The document layer, `planlens`, is a separate repository. All the repositories are PUBLIC on GitHub. The lead session reviews your branch independently before anything is merged. Work on your own and ask no questions. If you are blocked, write down why in CHECK.md, push, and stop.

## Where things are

| What | Where |
|---|---|
| The shared core | `packages/aec-agent-core/aec_agent_core/`; its own tests in `packages/aec-agent-core/tests/`. Many tests of core modules live in GSE's `funhouse_agent/**/tests/` and `webapp/tests/`, which reach the core through old module names (`CORE_ALIASES` in `funhouse_agent/__init__.py` and `webapp/__init__.py`). |
| Document Review, what testers run | `packages/aec-document-review/aec_document_review/`; tests in `packages/aec-document-review/tests/`. |
| GSE's own copy of the two-page app | `webapp/` and `funhouse_agent/`. Its review files are frozen by hash (`webapp/tests/test_review_freeze.py`). Its other copies of page files (`webapp/app.py`, `turn_jobs.py`, `identity.py`, `tinyapps_keys.py`, `sharepoint_store.py` and the rest) leave the repository with BACKLOG INF-35: leave them as they are unless your section says otherwise. |
| planlens | its own repository (`https://github.com/soconnell345-geotech/planlens`); the document tools the page uses are `planlens.review` and `planlens.tools`. |

## What you deliver

- Commits on ONE new branch, named in your section, from `master` (this repository) or `main` (planlens), pushed to origin after each piece of work.
- Never do any of these: open a pull request; push to `master` or `main`; push a branch named `ci/...` (that name is how the lead starts the full suite on GitHub); create a tag (here a `v*`, `core-v*` or `document-review-v*` tag publishes to PyPI, in planlens a `v*` tag does); edit a GitHub release; publish anything.

## Read first

1. The task file's top part and your section.
2. This repository's `CLAUDE.md`, only the sections "READ-FIRST TRIGGERS", "Architecture Patterns" and "Packages, pages and hosts" (the file is about 170 KB: do not read the rest).
3. The `BACKLOG.md` items your section names, in full (search for `**ID**`), and the evidence they cite: `module_work/docreview_review_2026-11-10/REVIEW.md` (the deployment review of Document Review) and `module_work/review_eval_results/2026-11-12_foundry_brief7/TRACE_REVIEW.md` sections 7, 8 and 10 (the full read of 84 suite runs on Foundry, "brief 7"). Do NOT edit `BACKLOG.md`, `HANDOFF.md` or `CLAUDE.md`: the lead folds your results in.
4. The README of each package you change. In planlens: its `CLAUDE.md`, `README.md` and the `DESIGN.md` of the subpackage you change.
5. For anything that runs on the testers' host: the part of `tinyapps/TINYAPPS.md` about the area you change.

## The owner's rules (binding)

- **Nothing fitted to one model.** The owner expects newer models to do better and does not want prompts tuned to today's misses. A task changes a tool, the record, the scorer or the page's behaviour for every model, or it is not a task. Never word a prompt or a tool description around one model's habit; never pin a number that changes from run to run.
- **No switch on by default.** A new behaviour that changes what testers see sits behind a setting that is off unless set; with it unset, the page looks and behaves as before and its fingerprints do not move. The exception is the repair of a defect: a tool result that is wrong or misleading, a file or part of a file lost without a word, a tester seeing or holding up another tester's data or turn, a wait with no bound. Your section says which pieces are repairs. A setting is an environment variable `GEOTECH_...`, read where it is used, documented beside the code and listed in CHECK.md with its default.
- **No edge-case chasing.** Fix the item as written. Anything else you notice goes in CHECK.md under "Noted, not fixed"; it is worth a line only if it would change what a saved run or a live turn returns.
- **Public repositories.**
  - Made-up test values only: sign-in names like `corp\j.doe`, sheets, numbers and documents built inside the test.
  - No names of any government department, embassy post, client, host, workspace, tenant or person, and no Foundry or FDE identifiers.
  - Before every commit in this repository, run the guards below (`webapp/tests/test_public_text.py` is one). planlens has no such test: read your diff for those words yourself.
- **Files you never edit:** `BACKLOG.md`, `HANDOFF.md`, `CLAUDE.md`; every file listed in `webapp/tests/test_review_freeze.py`; the geotech calc modules (the analysis packages at this repository's root), `slope_stability/` and `geotech-references/`; the files the task file's top part names; another task's files (the task file's table "Who edits what").
- **Fingerprints.** The page's system prompt, tool names, descriptions and argument schemas, helpers, middleware and budgets are pinned as hashes. If your change moves them, the fingerprint tests fail: re-record, in a commit of its own that says what moved and why. Never hand-edit either file.
  - Document Review's: from `packages/aec-document-review`, `python -m tests.page_builds record` (test `tests/test_review_fingerprints.py`, file `tests/review_fingerprints.json`; its planlens part is recorded under the installed planlens version).
  - GSE's, for both of its pages: from the repository root, in venv (b) below, `python -m webapp.tests.page_builds record` (test `webapp/tests/test_page_fingerprints.py`, file `webapp/tests/page_fingerprints.json`).
  - A core change can move both.
- **Consumers.** For any function, result key, event, setting or argument you add, rename or remove, grep the WHOLE repository (`packages/`, `webapp/`, `funhouse_agent/`, `report_ingest/`, `calc_package/`, `tinyapps/`, `docs/`, tests and JSON fixtures) and run the tests that read it. A core module serves GSE's geotech page as well as Document Review.
- **Calls made.** Where a value or a design choice is yours to make (a limit, a default, an order, which of two fixes an item offers), make it, say why, and list it in CHECK.md under "Calls made" for the lead to revisit. Do not stop to ask.
- **Tests.**
  - Judge a run by pytest's exit code, never through a pipe that hides it (no `| tail`, no `| grep`).
  - Run only the test files your change touches or that name a module you changed, plus the guards below. The full suites run on GitHub when the lead pushes your branch as `ci/<name>`.
  - Put your new tests in new files named for your item (for example `test_ta39_testers_apart.py`). Edit an existing test only where your change makes one of its assertions untrue, and list each such edit in CHECK.md.
  - Each new test fails on master and passes on your branch; say so, test by test, in CHECK.md.

## Setup: this repository

Python 3.12 (3.11 if 3.12 is missing). Two virtual environments, because CI tests Document Review with nothing geotechnical installed, and the core together with the geotech app.

(a) Document Review and the core alone, as CI's `document-review` job:

```
python3.12 -m venv /tmp/dr
/tmp/dr/bin/pip install --upgrade pip
/tmp/dr/bin/pip install -e packages/aec-agent-core -e packages/aec-document-review pytest
/tmp/dr/bin/pip install -r tinyapps/wrapper_repo/requirements-thirdparty.txt
```

`aec-document-review` pins its core and planlens exactly (`aec-agent-core==1.0.1`, `planlens==0.15.1` on master). The core comes from this checkout, at the pinned version; planlens comes from PyPI. Run the package's tests from its folder:

```
cd packages/aec-document-review && MPLBACKEND=Agg /tmp/dr/bin/python -m pytest -q -p no:cacheprovider tests/<file>
```

(b) The core with the geotech app, as CI's main job, for any task that changes a core file:

```
python3.12 -m venv /tmp/gse
/tmp/gse/bin/pip install --upgrade pip
/tmp/gse/bin/pip install -e packages/aec-agent-core -e . pytest pytest-xdist
```

planlens and geotech-references come from PyPI. Run from the repository root, `/tmp/gse/bin/python -m pytest -q <files>`; the root `conftest.py` puts this checkout's core first on the path. The core's own tests are `packages/aec-agent-core/tests/<file>`.

(c) planlens' unreleased main, only where your section says you need it: after (a) or (b), `pip install --no-deps --force-reinstall "planlens @ git+https://github.com/soconnell345-geotech/planlens@main"`. `--no-deps` because Document Review pins planlens exactly; `--force-reinstall` because main may still report the released version number. Say in CHECK.md which tests ran on it.

## Setup: the planlens repository

```
python3.12 -m venv /tmp/pl
/tmp/pl/bin/pip install --upgrade pip
/tmp/pl/bin/pip install -e ".[mcp]" pytest
```

That is planlens' CI install; the OCR extra is left out on purpose (its README, "Install"). Run tests with `/tmp/pl/bin/python -m pytest -q -p no:cacheprovider planlens/<area>/tests/<file>`; the whole suite runs on GitHub when the lead pushes your branch as `ci/<name>`.

**A planlens session cannot see this repository unless it clones it.** It is public. Clone it read-only, never change or push it:

```
git clone --depth 1 https://github.com/soconnell345-geotech/GeotechStaffEngineer /tmp/gse-src
```

Then read this file (`/tmp/gse-src/module_work/cloud_briefs/COMMON_DOCREVIEW.md`), the task file and your section, and `BACKLOG.md` and the evidence documents, all from that clone.

**The app as a consumer.** Where your change alters a document tool's result or specification, check the app against your branch: build venv (a) from `/tmp/gse-src`, then `/tmp/dr/bin/pip install --no-deps --force-reinstall -e <your planlens checkout>`, and run the Document Review tests your section names. Report what fails and why; you cannot fix the app from there. Its fingerprint test fails wherever a planlens tool specification changed: that is expected, and the lead re-records it when the app's planlens pin moves.

## Guards, before every commit (this repository)

```
/tmp/dr/bin/python -m pytest -q --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py
```

Both need only Python and git; `--noconftest` keeps out the root conftest, which imports the geotech app. Before your last commit, also run Document Review's `tests/test_boundary.py` and `tests/test_review_fingerprints.py` (venv a) when you changed the core or the package, and `webapp/tests/test_page_fingerprints.py` (venv b) when you changed the core.

## Finish with

1. **CHECK.md**, where your section says: in this repository `module_work/<item-id-lowercase, no dash>/CHECK.md`; in planlens `CHECK_<ID>.md` at the repository root, in a last commit of its own (the lead files it under this repository's `module_work/` and drops that commit before merging). Cover each BACKLOG item and each piece of it:
   - what you confirmed on today's code before changing it (file and function), and anything already fixed or wrong;
   - what you changed and why (file and function), which pieces are repairs, and which sit behind a setting (its name and default);
   - a made-up before and after example;
   - the tests you added (each fails on master) and every test run: the command and its passed/failed counts;
   - the fingerprints: moved or not, and the commit that re-recorded them;
   - **Calls made**;
   - **Release note**: one paragraph if a tester or a GSE user would notice anything; "None" otherwise;
   - **For the lead**: live checks, owner questions and work left, each with what it needs;
   - **Noted, not fixed**.
2. **Commit messages** that say what changed and why, each ending with the trailer of the model that runs the task: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` or `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
3. **A final message** of under 150 words: the branch, its commits, the tests and their counts, anything you stopped short of, and what the lead must do or check live.
