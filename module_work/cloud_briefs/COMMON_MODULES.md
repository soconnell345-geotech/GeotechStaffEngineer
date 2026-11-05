# Cloud helper brief: the geotech calc modules

Binding for every one-time cloud session the lead starts on the geotech calc modules. The routine's own prompt names the task; this file holds the rules that do not change between tasks.

You are a cloud helper for the GeotechStaffEngineer project: a Python toolkit of geotechnical analysis modules and an LLM agent app. Its repos are PUBLIC on GitHub. The lead session will review your branch independently before anything is merged. Work autonomously and ask no questions. If you are blocked, write down why and stop.

## What you deliver

Commits on ONE new branch from master, pushed to origin. Name it `claude/<item-id-lowercase>-<short-slug>`.

Never do any of these: open a PR, push to master or main, create a tag, or edit a release.

## Read first

- The repo's `CLAUDE.md` top block: READ-FIRST and Architecture Patterns.
- The module's `DESIGN.md`, before you change its code.
- The BACKLOG.md item your task names. Do NOT edit BACKLOG.md, HANDOFF.md or CLAUDE.md; the lead folds your results in.

## The owner's rules (binding)

- **Print only.** Every formula, default, coefficient and curve must trace to a printed page you can cite: manual, edition, and page, equation or table.
  - The printed manuals' text and figure catalogs are in the geotech-references repo, cloned beside this one or reachable with `git submodule update --init geotech-references`. Search its JSON text there.
  - If no held reference prints what you need, do not invent a value. Say so in CHECK.md and stop short of that part.
- **Open methods.** Where references differ, offer the cited options and state which one the code uses by default and why. Never use a silent basis.
  - A module's result lists its `choices` and `defaults_used`; keep both true.
  - Tool descriptions must say what the code does.
- **No edge-case chasing.** Fix the item as written. A follow-up is worth recording only if it would change a value in a saved report's output. Note anything else in CHECK.md as "noted, not fixed".
- **Public repo:**
  - made-up test values only (never values copied from a real project's documents);
  - no names of any government department, embassy post, client or person;
  - run `python -m pytest -q webapp/tests/test_public_text.py` before every commit (or the repo's equivalent test, if it has one).
- **Fingerprints.** If a tool description or a prompt changes, `webapp/tests/test_page_fingerprints.py` fails. Re-record with `python -m webapp.tests.page_builds record`, in a commit of its own.
- **Tests:** judge them by pytest's exit code, never through a pipe that hides it. Run:
  - the module's own tests;
  - its calc_package tests if it has a calc package;
  - `webapp/tests/test_page_fingerprints.py`;
  - `webapp/tests/test_public_text.py`;
  - `validation_examples` if any example uses the module.

  - `funhouse_agent/tests/test_defaults_used.py -k <module>` and any `funhouse_agent/tests` file that names the module: a changed default or a newly inert input fails there (add a `LABEL_ONLY` entry in `funhouse_agent/tests/method_probes.py` with the reason when an input rightly stops changing the result).

  The full suite is not needed.
- **Consumers.** For any result key, parameter or function you add, rename or remove, grep the WHOLE repo for consumers (`webapp/`, `funhouse_agent/`, `calc_package/`, `report_ingest/`, `docs/`, `validation_examples/`, tests and JSON fixtures) and run their tests. A session missed a failing test outside its module before.
- **User manual.** If the tool text changes, hand-edit only the matching rows of `docs/user_manual/manual.html` and the worked example inputs in `docs/user_manual/gen_worked_examples.py` / `manual_content.py` if the example would now be refused. Do not run a full manual rebuild (it moves the cover version).
- **Setup (app repo):** `python3.12 -m venv /tmp/v && /tmp/v/bin/pip install -e . pytest pytest-xdist` (planlens 0.14 and geotech-references 1.5.0 come from PyPI); use that venv's python for everything.

## Finish with

1. **`module_work/<item-id-lowercase, no dash>/CHECK.md`** in the branch (for example `module_work/gm296/CHECK.md`), covering:
   - what the print says, with page cites;
   - what you changed and why;
   - before and after numbers on at least one worked example;
   - the tests you ran, with passed/failed counts;
   - a one-paragraph **Release note**, if any calculated value changes;
   - anything noted, not fixed.
2. **Commit messages** that say what changed and why, each ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
3. **A final message** of under 150 words: the branch, its commits, the tests and their counts, and anything you stopped short of.
