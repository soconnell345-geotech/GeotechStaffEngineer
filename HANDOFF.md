# HANDOFF — GeotechStaffEngineer (current state, read this first)

**Last updated: 2026-07-05.** This is the authoritative, always-current handoff
for a fresh LLM session (any model). The older `HANDOFF_2026-06-14.md` is kept
only for the detailed Phase-E history; this file supersedes it.

---

## 1. One-screen status

| Item | Value |
|------|-------|
| Repo | github.com/soconnell345-geotech/GeotechStaffEngineer (private) |
| **master HEAD** | **`04cc6dd`** — pushed to origin, clean |
| Submodule `geotech-references` | `99e67b7` (pushed; pointer synced) |
| **Version string** | `5.1.0rc5` in `pyproject.toml` — **committed string only, NO git tag** |
| Test wheel | `C:\Users\socon\OneDrive\dev\v5_test_wheel\geotech_staff_engineer-5.1.0rc5-py3-none-any.whl` (only rc5 + the 5.0.0rc8 baseline are kept) |
| Validation suite | `validation_examples/` — **90 passed** (offline, no API) |
| Full repo suite | ~8000 tests green (`pytest -q`, ~35–40 min) |
| **Publish status** | **UNRELEASED.** 5.1.0 final tag/PyPI publish is **owner-gated.** |

**⚠️ Release gate (do not trip):** a `v*` git tag push **auto-publishes to PyPI**
via `.github/workflows/publish.yml` (OIDC trusted publishing). All rc versions
are committed as version strings with **no tag**. Do **not** push a tag, bump to
a final version, or publish without the owner's explicit OK.

---

## 2. What's on master (done + durable)

Chronological, all merged and pushed. Full detail: `docs/V5.1_SUMMARY.html`,
the per-module `DESIGN.md`/`VALIDATION.md`/`UPGRADE_PLAN.md`, and the memory
note `project_le_fem_modernization` (auto-loaded).

- **v5.1 line** (the bulk of the work): v5.1 TODO sweep; LE+FEM modernization
  (rigorous GLE/Morgenstern-Price `slope_stability`; T6 + 3D-MC-return + GL99
  SRM `fem2d`); the `reliability/` module (FOSM/PEM/MC/FORM + COV database);
  calc-package figures/tables + plotly viewers (`calc_package/interactive.py`);
  the staged model-setup agent (`geo_project/` + `deep/setup_agent.py`, OFF by
  default). Intentional behavior changes vs 5.0 are listed in the summary page.
- **Post-v5.0 field-failure fixes** (from a real Funhouse session): the
  lateral-pile calc-package bug (package re-ran the analysis; an invented
  `E_GPa` was silently dropped → steel-default; the deep agent's virtual FS
  couldn't see real files) → adapter-ergonomics sweep across ~30 adapters
  (`require_params`/`reject_unknown_params`, documented params, self-verifying
  file writes); and the **Databricks `/Workspace` placeholder-write** fix
  (`funhouse_agent/_fileio.py`: content-verified writes + rescue-to-`/tmp`).
- **Phase E — published-example validation (DONE):** 25 worked examples from
  GEC/Caltrans/FLAC implemented as **87 offline pytest checks**
  (`validation_examples/test_published_v0*.py` + `RESULTS.md`). **Zero
  analysis-result bugs** — every discrepancy resolved to units / method-variant
  / convention. One additive `fem2d` capability gained en route (`roller_base`
  BC + `initial_stress_relaxation` for excavation/cavity unloading; 369 fem2d
  tests, 0 regressions).
- **v5.2 coverage Batch 1 (DONE):** four additive, strictly default-preserving
  capability adds, each validated against its published example and flipping a
  `RESULTS.md` row to PASS — see `module_work/V5.2_COVERAGE.md`:
  - Q1 `settlement/hough.py` — Hough granular (C′-index) settlement.
  - Q2 `pile_group.meyerhof_group_settlement` — Meyerhof (1976) SPT group settlement.
  - Q3 `axial_pile` — per-layer `toe_friction_angle` + `head_depth` offset.
  - Q4 `retaining_walls.mse` — steel bar-mat/welded-grid Kr (2.5→1.2) + F* curves.

---

## 3. What's in flight / next

- **NOW (owner):** validating the **rc5 wheel** in Funhouse. Health check passed
  (offline 1a–1c PASS, live self-check 2/2). Owner is running the **71-question
  eval suite** (`run_suite`) and will review the results in a separate session.
  When satisfied, the owner gives the go for the `v5.1.0` tag (auto-publishes).
- **NEXT (open, not started) — v5.2 coverage Batch 2** (the bigger builds, all
  still offline with exact published targets in `validation_examples/INVENTORY.md`;
  list + rationale in `module_work/V5.2_COVERAGE.md`):
  drilled_shaft rational GEC-10 chains (V-006/007); MSE LRFD external-stability
  path (V-009); soe basal-heave-with-sidewall-shear + FHWA apparent-pressure
  anchored wall + log-spiral passive (V-013/014); lateral_pile full Reese (1974)
  sand p-y (V-017); fem2d monolithic u-p consolidation for the undrained
  transient (V-023 — the hardest). Deferred/owner-call items are also listed there.

---

## 4. How to run things (all offline unless noted)

**Environment:**
- Work in the git worktree `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.claude\worktrees\v5.1-todos`
  (branch `v5.2-coverage`), NOT the main checkout. Merge to master with
  `git merge --ff-only` from the main checkout, as every milestone has.
- **Venv python (system `python` has no pytest):**
  `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.venv\Scripts\python.exe`
- Git identity: `-c user.name=soconnell345 -c user.email=soconnell345@gmail.com`;
  end commit messages with the `Co-Authored-By` trailer.
- Tests: `pytest <module>/ -q`; validation: `pytest validation_examples/ -q`.

**Rebuild the test wheel** (from the main checkout, after merging to master):
`python -m build --wheel` → copy to `v5_test_wheel/`. **CRITICAL: bump the
version first** (`pyproject.toml`) — pip on Databricks skips a same-version
reinstall ("already installed with the same version… use `--force-reinstall`").
Also **verify the wheel contents** after building (a past rc shipped without a
just-added file): unzip and confirm the new files/tokens are present.

**Funhouse (Databricks) — notebook, needs API/`fh_prompter`:**
- Install: `%pip install "/tmp/...whl[deep]"` then `dbutils.library.restartPython()`
  (mandatory — else the `typing_extensions`/`extra_items` error).
- Health check: `from funhouse_agent.deep.rc_wheel_check import run_rc_check; run_rc_check(fh_prompter)`.
- 71-Q eval: `from funhouse_agent.deep.eval_harness import run_suite;
  run_suite(model, out="/tmp/eval")` (writes `.json` + a readable `.md`). Model =
  `PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")`. Real API
  calls; use `limit=` for a subset first. Correctness is PARTLY auto-scored
  (questions with `expected` keys) + partly eyeballed from the `.md`; process
  metrics (P1 hallucination rate, tool-error rate, rounds, latency, tokens) are
  always computed.
  - **Full coverage needs the optional-dependency extras:** install
    `%pip install "/tmp/...whl[deep,full]"` — with `[deep]` alone, ~12 questions
    (the gstools/pygef/ags4/pydiggs/ezdxf/SALib/pystrata/eqsig/liquepy/openseespy
    modules) fail honestly with "not installed" errors. `run_suite` runs an
    optional-dependency preflight and prints a "Missing optional packages" banner
    at the top of the `.md` when any are absent.
- Save outputs to `/tmp` or `/Volumes`, then `dbutils.fs.cp` out — NOT
  `/Workspace` (FUSE writes are non-durable / permission-blocked on the cluster).

---

## 5. Standing constraints & gotchas (carry forward)

- **No version bump→tag→publish without explicit owner OK.** A `v*` tag push
  auto-publishes to PyPI. rc strings are committed without tags on purpose.
- **`ANTHROPIC_API_KEY`** is read from the Windows *User* env at runtime — never
  pass it through chat/transcripts. Live tests are opt-in (`RUN_LIVE_TESTS=1`).
- **Owner is not a developer** — skip git/dev-procedure rationale; report
  actionable results. Prefers autonomous milestone-level operation, big batches.
- **Stagger subagents** — one at a time (usage-window limits); commit at
  milestones so a cutoff is cheap. Every module change must be **additive +
  default-preserving** (mirror the v5.2 Batch-1 pattern): new params/methods
  default to prior behavior; existing module tests stay green byte-for-byte.
- **Validate against published targets, don't tune to them.** A discrepancy is
  far more likely units / method-variant / convention than a module bug;
  investigate before "fixing." Record CONVENTION / N-A(scope) when the module is
  defensibly correct.
- **Edit/Write tools are pinned to the worktree** — to change a main-checkout
  file, author in the worktree + `cp`, or use a Python heredoc via Bash.
- **geotech-references is an editable install** resolving to the *main*
  checkout's submodule; after a submodule pointer change, `git submodule update`
  the main checkout.
- **`np.trapezoid`** (not `np.trapz`); SI units throughout.

---

## 6. File map (where the important things live)

- `HANDOFF.md` (this file) — current authoritative handoff.
- `HANDOFF_2026-06-14.md` — older, detailed Phase-E history (superseded).
- `module_work/V5.2_COVERAGE.md` — v5.2 plan/board (Batch 1 done, Batch 2 list).
- `module_work/WEEKEND_QC_2026-06-13.md` — the QC board (Phases A–F).
- `validation_examples/INVENTORY.md` — the 25 published problems (+ targets).
- `validation_examples/RESULTS.md` — per-problem verdicts + owner notes (the
  v5.2 coverage-gap backlog lives in these notes).
- `validation_examples/test_published_v0*.py` — the 87 offline validation tests.
- `funhouse_agent/deep/rc_wheel_check.py` — one-cell Funhouse health check.
- `funhouse_agent/deep/eval_harness.py` — `run_suite` (71-Q eval) + scorers.
- `funhouse_agent/geotech_test_suite.json` — the 71 eval questions.
- `funhouse_agent/_fileio.py` — verified-write + `/Workspace` rescue helpers.
- `docs/V5.1_SUMMARY.html` — everything-since-5.0 summary (owner-facing).
- `docs/funhouse_agent_guide.md` — install + Databricks gotchas + snippets.
- `CLAUDE.md` — project instructions (auto-loaded); its "v5.1 status" block is
  kept current and points here.
- Reliability / geo_project / slope_stability / fem2d each have their own
  `DESIGN.md` + `VALIDATION.md` + `UPGRADE_PLAN.md`.
