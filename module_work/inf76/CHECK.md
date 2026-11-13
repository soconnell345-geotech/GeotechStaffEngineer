# INF-76: one release unit, prepared (task G, 2026-11-13)

Branch `claude/inf-76-one-release-unit`, from master 50ef609. This prepares
INF-76 only. Nothing was tagged, published or started on GitHub. The item
stays open until the owner gives the word and the release goes out.
Moving no code and changing nothing for testers: the only edits are
workflows, tests, documents and version numbers.

Commits, in order:

1. the workflows and `webapp/tests/test_inf76_release_unit.py`;
2. the documents and the pyproject comments;
3. one assertion in `webapp/tests/test_wheels.py`;
4. this file;
5. **the version bump, alone and last**: both packages at 1.1.0.

## Confirmed on today's code, before changing it

- `publish-core.yml` publishes `packages/aec-agent-core` on a `core-v*` tag.
  Its only guard is that the tag equals `core-v<version>`.
- `publish-document-review.yml` publishes on `document-review-v*`. It
  refuses while its exact pins (`aec-agent-core==1.0.1`,
  `planlens==0.15.1`) are missing from PyPI. Nothing ties its version to
  the core's.
- Versions: the core is 1.0.1 (pyproject and `__version__`) and Document
  Review is 0.1.1. GSE pins `aec-agent-core==1.0.1` (root `pyproject.toml`).
- `publish.yml` (GSE, `v*`) checks only that the pinned core exists on
  PyPI.
- The pins are repeated in `tinyapps/wrapper_repo/packages.txt`, its
  `README.md` and `tinyapps/TINYAPPS.md`. They are checked by DR
  `tests/test_package.py`, `webapp/tests/test_wheels.py` and
  `webapp/tests/test_tinyapps_wrapper.py`.
- PyPI (`pip index versions --pre`):
  - aec-agent-core: 1.0.1, 1.0.0, 1.0.0.dev0;
  - aec-document-review: 0.1.1, 0.1.0, 0.1.0.dev0.
- Remote tags: `core-v1.0.0`, `core-v1.0.1`, `document-review-v0.1.0` and
  `document-review-v0.1.1`. The last two point at one commit, 464f98f.
- **INF-76's 2026-11-12 note is confirmed.** `git diff --stat core-v1.0.1
  HEAD -- packages/aec-agent-core` shows 13 files changed, among them the
  new `foundry_dataset_store.py`. A GSE release on master's pin (1.0.1)
  would install a core that lacks that module.
- The other session's branch `claude/gm-316-317-warnings-allowed-values`
  is no longer on origin, so it lists no files. None of its named files
  were touched.
- Found while editing: TINYAPPS.md route B named planlens 0.15.0 where the
  pin is 0.15.1. Corrected in commit 2.

## What changed, and why

### 1. One version, one tag (`publish-core.yml`, `publish-document-review.yml`)

- **One trigger.** Both workflows now start on the same tag pattern,
  `aec-v*`, and nothing else. GSE's `v*` never matches `aec-v…`, and
  `aec-v*` never matches `v5.36.0`.
- **One guard, before anything is built.** Both workflows carry the same
  first step, word for word: "One version, one tag (the core and Document
  Review)". It refuses unless:
  - both pyprojects carry one version;
  - the tag is `aec-v<that version>`;
  - Document Review pins exactly `aec-agent-core==<that version>`.
- **The core goes up first.** Document Review's last check, "… pins are on
  PyPI", now waits for the core:
  - it polls PyPI every `CORE_POLL_SECONDS` (30), for at most
    `CORE_WAIT_SECONDS` (1800), for `aec-agent-core==<version>`, which the
    twin workflow is publishing at the same time, then refuses;
  - planlens must already be on PyPI, with no wait;
  - the job has `timeout-minutes: 45` as a backstop;
  - `PYPI_JSON_URL` (default `https://pypi.org/pypi`) is a step setting so
    the test can point it at a local stand-in.
- **No change on pypi.org.** Trusted publishing names the repository, the
  workflow file and the `pypi` environment. It does not name the tag. Both
  files keep their names and environment, so neither project's publisher
  changes.
- The old `core-v*` and `document-review-v*` triggers are gone, so neither
  package can go out alone. The old tags stay as the record of those
  releases.

### 2. The core a release goes out with (`publish.yml`, `publish-document-review.yml`)

- **A new step: "The pinned core is the core in this checkout".** It is
  the same text in both workflows; only `PYPROJECT` differs.
- **What it checks.** It reads the exact core pin and looks for the tag of
  that version: `aec-v<pin>` first, then `core-v<pin>` for releases before
  INF-76. It refuses:
  - when no such tag exists;
  - when `git diff --name-only <tag> HEAD -- packages/aec-agent-core/`
    lists any file. The message names the count and the first five files.
- **Fetching tags.** Both checkouts now use `fetch-depth: 0`, so the tags
  and the history are there.
- **Today it refuses.** A GSE release from master pinned to 1.0.1 (13
  files changed since `core-v1.0.1`) would stop at this step. On this
  branch, with the pin at 1.1.0 and no `aec-v1.1.0` tag yet, it would stop
  too. That is the intended order: release the unit, then GSE.
- **For Document Review inside the unit,** the pinned core's tag is the
  tag being released. The check compares HEAD with itself and passes; it
  is there in case the version guard is ever changed.
- **`publish.yml` also:** its comments now name the `aec-v` tag, and the
  "not on PyPI" error points at INF-76 instead of the split's step 12.

### 3. Tests and documents

- **New test:** `webapp/tests/test_inf76_release_unit.py`, 26 tests. They
  need Python, PyYAML and git only, with no network.
- **Documents updated:**
  - both packages' READMEs (a "Releases" paragraph);
  - both packages' pyproject comments, and GSE's comment on its core pin;
  - `tinyapps/TINYAPPS.md` (the status paragraph, the route-B planlens
    version);
  - `tinyapps/wrapper_repo/README.md` ("Update the deployed app");
  - `docs/FOUNDRY.md` section 6.
- **The proposed `CLAUDE.md` text** is in "For the lead" below.

### 4. The version bump (last commit, alone)

- **Files moved to 1.1.0:**
  - the core: pyproject and `__version__`;
  - Document Review: pyproject, `__version__` and its core pin;
  - GSE's pin;
  - the wrapper's `packages.txt` and `README.md` (pins, wheel names, the
    install command);
  - TINYAPPS.md (pins, wheel names, the checklist's expected install
    line). Its status paragraph and "Before you start" now say that 1.1.0
    is not on PyPI until the lead releases it.
- **Why 1.1.0.** It is above both 1.0.1 and 0.1.1 and above everything on
  PyPI for either package (pre-releases included). It is also the version
  brief 8's test core wheel carried as 1.1.0rc2.
- **The core README.** "Since 1.0.1 (not released yet)" became "1.1.0 (not
  released yet; …)".

Every piece is new release machinery. Nothing is behind a setting, because
nothing reaches a tester's page.

## Example, made up

**Before.** Someone tags `core-v1.0.2` and the core goes up alone. Days
later someone tags `document-review-v0.1.2`, still pinning
`aec-agent-core==1.0.1`, and it goes up too. Next, `v5.36.0` goes out
pinned to `aec-agent-core==1.0.1`, whose wheel lacks
`foundry_dataset_store`. On Foundry the app then keeps nothing, and says
nothing.

**After.**

- Tagging `aec-v1.1.0` starts both workflows. The core's workflow builds
  and publishes. Document Review's prints "aec-agent-core 1.1.0 is not on
  PyPI yet; waiting for publish-core.yml." a few times, then publishes.
- Tagging `aec-v1.1.0` while Document Review still says 1.0.9 stops both
  workflows before any build: "aec-agent-core is 1.1.0 but
  aec-document-review is 1.0.9".
- `v5.36.0` pinned to 1.0.1 stops with "packages/aec-agent-core/ has
  changed since core-v1.0.1 … (13 files, e.g. …)".

## Tests

New tests, each run against master's tree (master's workflows and
pyprojects, with the new test file copied in): 24 fail on master. Two
pass there: `test_the_release_workflows_parse[publish-core.yml]` and
`[publish.yml]`. They are YAML parse checks the brief asks for, and those
two workflows already parsed on master. On this branch the six one-version
tests fail until commit 5 (the bump) and pass after it.

| Test | On master | On the branch (after the bump) |
|---|---|---|
| `test_the_core_and_document_review_carry_one_version` | fails | passes |
| `test_every_pin_of_the_core_names_that_version` | fails | passes |
| `test_the_wrapper_files_state_no_other_version` ×3 | fail | pass |
| `test_the_release_workflows_parse` ×3 | 1 fails, 2 pass | pass |
| `test_one_tag_pattern_starts_both_and_never_gse` | fails | passes |
| `test_both_workflows_carry_the_same_guards` | fails | passes |
| `test_the_guard_runs_before_anything_is_built` | fails | passes |
| `test_the_checks_needing_tags_fetch_them` | fails | passes |
| `test_document_review_waits_for_the_core_with_a_bound` | fails | passes |
| `test_the_version_guard_passes_this_tree_at_its_tag` | fails | passes |
| `test_the_version_guard_refuses` ×4 | fail | pass |
| `test_the_version_guard_passes_a_matching_tree` | fails | passes |
| `test_the_core_guard_passes_an_unchanged_core` | fails | passes |
| `test_the_core_guard_reads_a_tag_from_before_inf76` | fails | passes |
| `test_the_core_guard_refuses_a_core_changed_since_its_tag` | fails | passes |
| `test_the_core_guard_refuses_without_a_tag` | fails | passes |
| `test_document_review_waits_until_the_core_is_up` | fails | passes |
| `test_document_review_stops_waiting_at_the_bound` | fails | passes |
| `test_planlens_missing_is_refused_without_waiting` | fails | passes |

**Existing test edited:** `webapp/tests/test_wheels.py::test_publish_builds_the_root_only`.

- It refused any `packages/` in `publish.yml`.
- The new guard's `CORE = "packages/aec-agent-core/"` line is now exempt,
  like the comment already was.
- The assertion that `publish.yml` builds only the root is unchanged.

**Runs.** Venv (a) is `/tmp/dr`; venv (b) is `/tmp/gse`, with `build`,
`setuptools>=68`, `wheel` and `trove-classifiers`. Each count is pytest's
exit code and summary line.

- Before the bump:
  - venv (a), `--noconftest webapp/tests/test_inf76_release_unit.py`: 6
    failed, 20 passed (the one-version tests, as expected).
  - venv (a), the guards (`test_public_text.py`, `test_review_freeze.py`,
    `--noconftest`): 13 passed. Run before every commit.
  - venv (a), DR `tests/test_package.py`: 13 passed.
- After the bump:
  - venv (a), `--noconftest`, `test_inf76_release_unit.py` and both
    guards: 39 passed.
  - venv (a), DR `tests/test_package.py`, `tests/test_boundary.py` and
    `tests/test_review_fingerprints.py`: 22 passed.
  - venv (b), `GEOTECH_REQUIRE_WHEEL_TEST=1`, `webapp/tests/test_wheels.py`
    and `webapp/tests/test_sdists.py`:
    - 1 failed, 12 passed first: `test_publish_builds_the_root_only`, as
      above;
    - 13 passed after the edit.
  - venv (b), `webapp/tests/test_tinyapps_wrapper.py`,
    `test_docs_currency.py`, `test_page_fingerprints.py` and
    `test_inf76_release_unit.py`: 70 passed.

## Fingerprints

Not moved: DR's `test_review_fingerprints.py` and GSE's
`test_page_fingerprints.py` both pass at 1.1.0. No re-record commit.

## Calls made

- **The tag pattern is `aec-v<version>`.** It is short and names both
  `aec-` packages. GSE's `v*` cannot match it.
- **The old `core-v*` and `document-review-v*` triggers are removed**
  rather than kept beside the new one. Keeping them would let one package
  go out alone, which is what INF-76 ends.
- **The core's wait is 1800 s, polling every 30 s,** with a 45-minute job
  timeout. The core's build and upload take a few minutes, and PyPI's JSON
  page can lag behind an upload. Half an hour is generous without leaving
  a stuck run for hours.
- **planlens is not waited for.** It releases on its own and must already
  be on PyPI.
- **The pinned-core check compares the whole `packages/aec-agent-core/`
  folder,** as INF-76's note says, so a change to the core's tests or
  README also asks for a core release first. Narrowing it to
  `aec_agent_core/` and `pyproject.toml` is a one-line change if that
  proves too strict.
- **The pinned-core check accepts a `core-v<version>` tag** as well as
  `aec-v<version>`, so a GSE release can still pin 1.0.1 or 1.0.0 if their
  folder were ever unchanged.
- **The pinned-core check is also in Document Review's workflow,** where it
  passes by construction, so the two release workflows that pin the core
  carry the same check.
- **The new version is 1.1.0 for both.** Document Review jumps from 0.1.1
  to 1.1.0; a version only has to rise, and one number is the point.
- **The wrapper and TINYAPPS.md pins move to 1.1.0 in the bump commit.**
  `test_tinyapps_wrapper.py` already requires `packages.txt` to follow
  Document Review's pyproject. The prose now says 1.1.0 is not on PyPI
  yet.

## Release note

For the core and Document Review release (1.1.0): From 1.1.0,
`aec-agent-core` and `aec-document-review` carry one version number and go
out together. `aec-document-review==X` always requires
`aec-agent-core==X`, and planlens keeps its own versions. An update on
Tiny Apps moves both pins to the same number. Nothing changes on the page.

For GSE's next release: none for users. `publish.yml` now also refuses
while the core has changed since the release of the version GSE pins.

## For the lead

1. **The owner's word, then the tag.** The tag is
   `git tag aec-v1.1.0 && git push origin aec-v1.1.0` on the merged commit.
   Watch both runs: Document Review's should wait, then publish after the
   core. Check `pip index versions aec-agent-core` and the same for
   `aec-document-review`.
2. **pypi.org: nothing to do,** if both projects' trusted publishers name
   `publish-core.yml` and `publish-document-review.yml` with environment
   `pypi`, as their headers say. Please confirm on pypi.org (Manage →
   Publishing) that neither publisher was set up with a different
   environment. If the `pypi` environment on GitHub needs a reviewer's
   approval, approve both runs; Document Review's wait (30 min) starts
   only after its build job is approved, and so does the core's.
3. **GSE's release after this one** must pin 1.1.0. The new guard will
   refuse it until `aec-v1.1.0` exists and the core folder is unchanged
   since then.
4. **Keeping the bump commit.**
   - If you drop it, the six one-version tests in
     `test_inf76_release_unit.py` fail until the release bump puts both
     packages on one version.
   - The release bump must also move the core pin in GSE's
     `pyproject.toml`, in Document Review's pyproject and in the
     wrapper's files, all checked.
5. **The proposed `CLAUDE.md` text,** for the "bumps the version / cuts a
   release" trigger row. Replace its last two sentences with: "The shared
   core and Document Review release as ONE unit (INF-76): one version in
   `packages/aec-agent-core` and `packages/aec-document-review` (Document
   Review pins the core at exactly that version), and one tag,
   `aec-vX.Y.Z`, that starts `publish-core.yml` and
   `publish-document-review.yml`; Document Review's run waits (at most 30
   min) until the core of that version is on PyPI, and both refuse a tag
   that does not match both versions. The app pins the core exactly, so
   the unit is released first: `publish.yml` stops the app's release while
   the pinned core is missing from PyPI or while `packages/aec-agent-core/`
   has changed since the pinned version's tag (`aec-vX.Y.Z`, or `core-vX.Y.Z`
   before INF-76)."
   - The "Packages, pages and hosts" table rows for the two packages could
     add "released together, one version (INF-76)".
   - `webapp/tests/test_docs_currency.py` needs no change: it follows
     GSE's version, which this branch does not move.
6. **Not run here:** the workflows themselves. Their steps were run
   locally from the parsed YAML, against a throwaway git repository and a
   local stand-in for PyPI. The `${{ github.ref_name }}` input and
   GitHub's tag fetch were not exercised. The first real run is the
   release.

## Noted, not fixed

- `docs/DATABRICKS_INSTALL.md` history rows and `report_ingest/README.md`
  mention `core-v1.0.0` and `aec-agent-core==1.0.0` as history. They are
  left alone: the first is outside this task, and both are correct as
  history.
- `publish.yml` keeps `workflow_dispatch`. A manual run on a branch is
  checked by the same guards.
