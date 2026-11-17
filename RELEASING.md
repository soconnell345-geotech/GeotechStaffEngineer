# Releasing GeotechStaffEngineer

How a release is made here, step by step, with the files each step edits
and the test that checks it; then how to build wheels for a closed platform
without a release; then how to take an upstream release. `MAINTAINERS.md`
explains the repository.

Two facts shape everything below:

- **A pushed tag publishes at once.** The publish workflows run on the tag,
  and the package index never takes the same version twice. A mistake found
  after the push needs a new version. So the tree is green first, the
  version and the notes are read once more, and only then the tag.
- **The app pins its core exactly.** `geotech-staff-engineer` requires
  `aec-agent-core==X.Y.Z`, so the core must be on the index before the
  app's tag; `publish.yml` stops the app's release until it is.

## 1. What is released, and in which order

| Package | Its version lives in | Tag | Workflow |
|---|---|---|---|
| `geotech-references` | its own repository | `vX.Y.Z` there | its own (see its repository) |
| `planlens` | its own repository | `vX.Y.Z` there | its own |
| `aec-agent-core` | `packages/aec-agent-core/pyproject.toml` and `__version__` in `packages/aec-agent-core/aec_agent_core/__init__.py` | `core-vX.Y.Z` | `.github/workflows/publish-core.yml` |
| `aec-document-review` | `packages/aec-document-review/pyproject.toml` and `__version__` in its `__init__.py` | `document-review-vX.Y.Z` | `.github/workflows/publish-document-review.yml` |
| `geotech-staff-engineer` | `pyproject.toml` (`[project]` `version`) | `vX.Y.Z` | `.github/workflows/publish.yml` |

The order is the order of the pins: the reference library, planlens, the
core, Document Review, then the app. The core and Document Review are
upstream's to release (`MAINTAINERS.md` section 2); a maintainer of the app
releases the app and the reference library.

## 2. An app release, step by step

As release 5.35.0 was made (2026-11-10; its record is in `HANDOFF.md`
section 0a-current).

1. **Upstream first.** Every version the app's pins name is on the index:
   the reference library, planlens, the core.

2. **A release branch** from master:

   ```bash
   git fetch origin
   git checkout -b release/X.Y.Z origin/master
   ```

   Work that lands on master in the meantime is merged in
   (`git merge origin/master`).

3. **The pins.** In `pyproject.toml`: `geotech-references>=`, `planlens>=`
   and `aec-agent-core==` (exact). The submodule at the library's release
   tag (`git -C geotech-references checkout vX.Y.Z`, then
   `git add geotech-references`). Where `tests.yml` installs planlens by its
   own line (`"planlens>=0.15"` in the `core-only` and numpy 1.26 jobs), the
   same floor. Check: `webapp/tests/test_wheels.py` (the core pin equals the
   core's version).

4. **The version.** `version = "X.Y.Z"` in the `[project]` table of
   `pyproject.toml`.

5. **The notes a version bump must update.** `webapp/tests/test_docs_currency.py`
   fails until each of the first three names the new version. Refresh the
   prose, not only the number.
   - `CLAUDE.md`, section "CURRENT WORKING STATE": what is released, with
     its pins.
   - `HANDOFF.md`, section "0a-current": a new item 0 for the release: the
     tags and branches, the gate's numbers, and the **release notes by
     module** (each change that moves an answer: for which inputs, a
     before-and-after number, the commit). The previous release becomes
     item 0a.
   - `docs/DATABRICKS_INSTALL.md`, section 11 "History": a row beginning
     `| X.Y.Z |` saying what an install should show ("Not yet installed"
     until one is confirmed).
   - `docs/user_manual/manual_content.py`: the version history row
     (`"5.4.0 &ndash; X.Y.Z"`).

6. **The backlog.** Every item the release closed is under "Done since the
   last sweep" with its commit. Then the sweep moves that list, unchanged,
   to the top of `BACKLOG_DONE.md` under a dated heading:

   ```bash
   python module_work/backlog_dashboard/sweep_done.py --dry-run --note "app master <sha>"
   python module_work/backlog_dashboard/sweep_done.py --note "app master <sha>"
   ```

   Update the "Last Done sweep" line near the top of `BACKLOG.md` by hand.
   Check: `webapp/tests/test_backlog.py`.

7. **The worked examples, then the manual.** With the submodule at the
   pinned commit and the reference library first on the import path
   (`MAINTAINERS.md` section 4). The manual quotes real outputs, so first
   rerun them, then rebuild:

   ```bash
   python docs/user_manual/gen_worked_examples.py
   git diff --stat docs/user_manual/worked_examples.json
   python module_work/doc10/check_manual.py --rebuild
   python module_work/doc10/check_manual.py --rebuild --write
   ```

   `gen_worked_examples.py` rewrites `docs/user_manual/worked_examples.json`;
   read its diff: every number that moved should be one the release notes
   explain. `check_manual.py --rebuild` rebuilds the manual in memory and
   exits 1 if the committed `manual.html` or `toc.json` differ (the cover
   carries the version, so after a bump it always does); `--write` writes
   them. Read the diff, commit `worked_examples.json`, `manual.html` and
   `toc.json`. Check: `webapp/tests/test_user_manual_current.py`. The PDF
   (`docs/user_manual/render_pdf.py`, needs Chrome or Edge) is made on
   demand and not kept in the repository.

8. **The gate, on the exact tree to be tagged.** The full suite and every
   CI job green on the release commit (5.35.0: one `tests.yml` run on the
   release commit, all eight jobs green). CI's "Wheels" job builds the
   wheels the way a release does (the sdist first, then the wheel from the
   sdist), installs them in a fresh virtual environment and imports them
   there; it also checks that neither package's sdist carries tests and that
   every classifier is one the index accepts. The same checks by hand:

   ```bash
   GEOTECH_REQUIRE_WHEEL_TEST=1 python -m pytest -q webapp/tests/test_wheels.py
   python -m pytest -q webapp/tests/test_sdists.py webapp/tests/test_classifiers.py
   ```

   Run `test_wheels.py` with `PYTHONPATH` unset: the build leaves
   `*.egg-info` folders in the tree, and with the tree on `PYTHONPATH` pip
   takes them for installed packages and installs nothing into the test's
   fresh environment (two tests then fail with `No module named
   'aec_agent_core'`).

9. **Read once more, then tag.** The version line, the pins, the release
   notes. `publish.yml` does not compare the tag with the version, so a
   wrong tag publishes whatever `pyproject.toml` says. Then:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

10. **What the tag triggers.** `publish.yml` (tags `v*`) builds the root's
    sdist and wheel (`python -m build --outdir dist/ .`), stops if anything
    but `geotech_staff_engineer-*` is in `dist/`, stops if the pinned core
    is not on the index, then publishes by trusted publishing (the GitHub
    environment `pypi`; no token is stored in the repository).
    `publish-core.yml` (`core-v*`) and `publish-document-review.yml`
    (`document-review-v*`) do the same for their package, and each stops if
    the tag does not match the package's version; Document Review's also
    stops if the core and planlens it pins are missing from the index.

11. **After.** Merge the release branch into master
    (`git merge --no-ff release/X.Y.Z`; 5.35.0 was `ce49b16`). A package
    mirror can take a day or two to carry the new versions. When an install
    on a host is confirmed, replace that version's row in the install
    guide's history.

## 3. A build for a closed platform

A platform that cannot reach the public index installs wheels from files.
Such a build is not a release: no tag, nothing published. It carries a
**local version label** so that every install says exactly what it is:

```
5.36.0+<label>.<date>.<n>        for example 5.36.0+site.20261120.1
```

The public release the build starts from, then a short name for the
platform, the build date and that day's build number. A labelled version
sorts after its public release, and the diagnostics report's versions line
shows it in full. A pin without a label accepts a labelled build:
`geotech-references>=1.7.0` is met by `1.7.0+site.20261120.1`, and
`aec-agent-core==1.0.1` by `1.0.1+site.20261120.1`.

**The label never goes on master.** `test_docs_currency.py` would fail
(the state docs name the plain version), and the label belongs to one
build. Put it in an exported copy instead:

```bash
git archive --format=tar --prefix=gse/ <commit> | tar -x -C /path/to/scratch
cd /path/to/scratch/gse
sed -i 's/^version = "5.36.0"$/version = "5.36.0+site.20261120.1"/' pyproject.toml
grep -n '^version' pyproject.toml
python -m build --wheel --outdir ../dist .
```

- `git archive` exports only the tracked files at that commit, so nothing
  untracked reaches the wheel. In tar form it keeps the commit's line
  endings. In zip form, on a machine with `core.autocrlf=true` (the Windows
  default), it writes Windows line endings; there run
  `git -c core.autocrlf=false archive --format=zip ...`.
- The export leaves `geotech-references/` empty; the app's wheel does not
  include the library, so the build does not need it.
- `python -m build` fetches `setuptools>=68` into an isolated build
  environment from the configured index. Where only a private package index
  is reachable, point pip at it (`PIP_INDEX_URL`), or install `setuptools`
  and `wheel` and add `--no-isolation`.
- The reference library builds the same way from an export of its own
  repository (its version line, its own release notes). Upstream wheels
  (the core, planlens) go into the set as published, unchanged.

**Look inside before shipping:**

```bash
python -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); m = [n for n in z.namelist() if n.endswith('/METADATA')][0]; print(*[l for l in z.read(m).decode().splitlines() if l.startswith(('Version:', 'Requires-Dist: aec', 'Requires-Dist: planlens', 'Requires-Dist: geotech'))], sep='\n')" ../dist/geotech_staff_engineer-*.whl
```

It prints the labelled version and the three pins.

**Sums.** Put every wheel of the set (yours and upstream's) in one folder,
then:

```bash
cd ../dist
sha256sum *.whl > SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
```

**Installing from files.** Some zip tools drop the `+` from a wheel's file
name, and pip refuses a wheel whose name does not match its version: put
the `+` back, copying the version from the wheel's own `METADATA`. Where an
environment manager provides the third-party packages, install the four
packages without their dependencies, into the environment or into a folder
put first on `sys.path` and `PYTHONPATH` at start-up (`docs/FOUNDRY.md`
section 1 describes this route):

```bash
pip install --no-deps --no-index --target /path/to/site geotech_staff_engineer-*.whl aec_agent_core-*.whl planlens-*.whl geotech_references-*.whl
```

In an environment that holds numpy at 1.26, the third-party requirements
come from the script, under the constraints (with the folder on the path):

```bash
python constraints/foundry_requirements.py > requirements.txt
pip install -r requirements.txt -c constraints/foundry-numpy126.txt
pip check
```

`pip check` then reports only that PyNiteFEA is missing: the script leaves
it out on purpose (`constraints/foundry-numpy126.txt` says why, and
`docs/FOUNDRY.md` section 1 how to keep the `pynite` tool anyway; without
it the app runs without that tool). Last, confirm what is installed:

```bash
python -c "import importlib.metadata as m; print(*[m.version(p) for p in ('geotech-staff-engineer', 'aec-agent-core', 'planlens', 'geotech-references')])"
```

## 4. Taking an upstream release

What arrives: the released wheels of the core, Document Review and planlens,
and, for the core, its source folder (the public repository's
`packages/aec-agent-core` at tag `core-vX.Y.Z`). The source folder matters:
the root `conftest.py` makes the tests import the tree's
`packages/aec-agent-core`, not the installed wheel, so a new core wheel
without its folder leaves the tests on the old core.

**A new core:**

1. Replace `packages/aec-agent-core/` whole with the release's copy. Never
   merge it by hand.
2. Set `aec-agent-core==X.Y.Z` in `pyproject.toml`.
3. `python -m pytest -q webapp/tests/test_wheels.py` (the pin equals the
   folder's version).
4. `python -m pytest -q webapp/tests/test_page_fingerprints.py`. If what
   the agent is handed changed and the core's release notes say so,
   re-record in a commit of its own that names the release
   (`python -m webapp.tests.page_builds record`). An unexplained difference
   is a request upstream, never a re-record.
5. The full suite; then build the set (section 3) with the new core's wheel.

**A new planlens:** install its wheel (`pip install --no-deps planlens-X.Y.Z-py3-none-any.whl`)
and run `webapp/tests/test_page_fingerprints.py`. The fingerprints keep one
entry per planlens version; for a planlens with no entry the test skips off
CI and fails on CI. `python -m webapp.tests.page_builds record` adds the
entry for the installed planlens and keeps the others; commit it on its
own. Raise `planlens>=` only when the app needs the new release. Then the
suite.

**A new Document Review:** not a dependency of the app. Replace
`packages/aec-document-review/` with the release's copy to keep the tree
whole; nothing in the app changes.

**The reference library** is maintained with the app: its release reaches
the app through the submodule and the pin (`MAINTAINERS.md` section 3).

## 5. How the commands here were checked

Each command was run once on 2026-11-17 against master `6941d9d` (version
5.35.0) on a Windows laptop with Python 3.14, by exit code, unless it is
listed as not run. Files a command rewrote were restored afterwards.

- Step 2: `git fetch origin` and a branch from `origin/master`; `git merge
  origin/master` ("Already up to date").
- Step 6: both `sweep_done.py` lines (131 entries moved).
- Step 7: `gen_worked_examples.py` (18 of 18 examples ran; its JSON
  changed), the `git diff --stat`, `check_manual.py --rebuild` (exit 1: the
  new worked examples move 46 lines of the manual) and `--rebuild --write`
  (exit 0).
- Step 8: `test_wheels.py` with the build switch (11 passed, `PYTHONPATH`
  unset), `test_sdists.py` and `test_classifiers.py` (5 passed).
- Section 3, with 5.35.0 in place of 5.36.0: the export, the version edit,
  `python -m build --wheel` (built
  `geotech_staff_engineer-5.35.0+site.20261117.1-py3-none-any.whl`), the
  METADATA check, both sum commands (with the published wheels of the core
  1.0.1, planlens 0.15.1 and the library 1.7.0), the install into a folder,
  the requirements script (32 lines, none of the four packages) and the
  versions line (`5.35.0+site.20261117.1 1.0.1 0.15.1 1.7.0`). The two pin
  statements were checked with `packaging`.
- Section 4: `test_wheels.py` without the switch (6 passed, 5 skipped),
  `test_page_fingerprints.py` (13 passed), `page_builds record` (exit 0; the
  recorded file unchanged).

Not run here:

- Step 3's submodule move: a release step (last done in `bd11539`, the
  library at 1.7.0).
- `render_pdf.py`: optional, and the PDF is not part of a release.
- Step 9's `git tag` and `git push`: a pushed tag publishes at once.
- Step 11's merge: a release step (5.35.0's is `ce49b16`).
- Section 3's `pip install -r requirements.txt -c ...` and `pip check`:
  they install about thirty packages into an environment that holds numpy
  at 1.26.4; CI's numpy 1.26 job runs the same install on every push that
  changes code, and `docs/FOUNDRY.md` section 1 records the `pip check`
  line.
- Section 4's `pip install --no-deps planlens-X.Y.Z...`: no planlens newer
  than the installed 0.15.1 exists yet.
