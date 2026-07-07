# HANDOFF — GeotechStaffEngineer (current state, read this first)

**Last updated: 2026-07-07.** This is the authoritative, always-current handoff
for a fresh LLM session (any model). The older `HANDOFF_2026-06-14.md` is kept
only for the detailed Phase-E history; this file supersedes it.

---

## 1. One-screen status

| Item | Value |
|------|-------|
| Repo | github.com/soconnell345-geotech/GeotechStaffEngineer (private) |
| **master HEAD** | `d41540e` = **v5.3.0 release commit** (tag `v5.3.0`) — pushed, PUBLISHED to PyPI |
| **Branch `v5.4`** | **PUSHED to origin, NOT merged** — all 6 owner directives BUILT (this file's §3); worktree `.claude/worktrees/v5.1-todos` is checked out on it |
| Submodule `geotech-references` | `fe7fe9d` = **v1.3.0 on PyPI** (unchanged since 5.2.0) |
| Version string | `5.3.0` in `pyproject.toml` (on both master and `v5.4` — no bump yet) |
| Validation suite | `validation_examples/` — 136+ passed (offline; V-001..V-040) |
| Full repo suite | **8279 passed / 48 skipped** on `v5.4` (2026-07-07); 8218/48 on master |
| **Publish status** | 5.3.0 PUBLISHED (owner OK'd 2026-07-06). **v5.4 release = OWNER-GATED**: owner is validating 5.3 in Funhouse; on their OK → merge `v5.4`→master, bump 5.4.0, tag. |

**⚠️ Release gate (still applies):** a `v*` git tag push **auto-publishes to
PyPI** via `.github/workflows/publish.yml` (OIDC trusted publishing). Do **not**
push a tag, bump the version, or merge `v5.4` to master without the owner's
explicit OK.

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
- **5.2.0 released 2026-07-05** (with geotech-references 1.3.0; 5.1.0 never
  shipped) after the rc5 71-Q eval review (archived
  `docs/geotech_eval_20260705.json`/`.md`; fixes: drilled_shaft per-layer
  breakdown, +6 dispatch aliases, P1 recovered-split, optional-dep preflight).
- **v5.3 train (released 2026-07-06 as 5.3.0):** Batch-2 coverage 5/5
  (drilled_shaft rational GEC-10 chains; MSE LRFD external-stability CDRs; soe
  basal-heave-sidewall-shear + FHWA apparent-pressure anchored + log-spiral
  Caquot-Kerisel Kp; full Reese-1974 sand p-y; fem2d monolithic Taylor-Hood
  u-p Biot consolidation); slope_stability round 2 (15 new Slide2/ACADS/Duncan
  validation problems V-026..V-040; SS-6 noncircular-search robustness fix +
  rejection diagnostics; rapid drawdown 2/3-stage; Newmark + Jibson; infinite
  slope; Ito-Matsui piles verified vs the ORIGINAL 1975 paper); pdf_import
  round 2 (scale calibration, label→region, cleanup, vision grid overlay,
  vision↔vector cross-check); + **12 adversarial-review fixes** (headline:
  log-spiral Kp δ=0 Rankine anchor — the clamp was ~44% unconservative).
  Plans: `module_work/V5.3_PLAN.md` (+ its review-outcome section).

---

## 3. What's on branch `v5.4` (BUILT + gated, awaiting owner release call)

All six owner directives (2026-07-06) are DONE on `v5.4` (13 commits over
master; final gate 8279 passed / 48 skipped). Plan of record + per-item log:
**`module_work/V5.4_PLAN.md`** (read it before continuing v5.4 work).

- **D1 PDF user manual** — `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`
  (132 pp) + regenerable `docs/user_manual/build_manual.py`; problem catalog is
  auto-generated from MODULE_REGISTRY/METHOD_INFO so it can't drift from code.
- **D2 no-restart Databricks** — `funhouse_agent/runtime_check.py` hot-reloads
  a stale pre-imported `typing_extensions` before any langchain import;
  `dbutils.library.restartPython()` is now only the documented fallback;
  cluster-scoped install documented as the avoid-entirely alternative.
- **D3 layered disclaimers** — DISCLAIMER.md (ships in wheel), prominent README/
  PyPI section, ONE-TIME first-import stderr notice (marker
  `~/.geotech_staff_engineer/disclaimer_ack`; suppress `GEOTECH_NO_DISCLAIMER=1`;
  silent under pytest), `geotech-disclaimer` console script, and a standing
  basis-&-limitations block in both calc-package templates. Honest constraint:
  pip runs NO code on wheel install — these are the legitimate equivalents.
- **D4 visualization gallery** — `docs/gallery/index.html` + `build_gallery.py`:
  12 exhibits, every figure from a REAL validated run (ACADS search, drawdown
  3-stage, Newmark polarity, p-y families, GL99 SRM, V-023 consolidation,
  Duncan reliability, GEC-11 MSE CDRs, bearing/settlement, bearing graph,
  pdf_import demo).
- **D5 `drawing_ir/` module** — LLM-ready drawing digitization: unified IR
  (Line/Polyline/Arc/Circle/Text/Region w/ coords, layer/color, provenance
  dxf|pdf_vector|raster_trace, per-entity confidence), OpenCV raster leg
  (`[raster]` extra), and an agent query surface (digitize_drawing → handle;
  query_drawing: entities_in_bbox / lines_by_angle / text_near /
  candidate_ground_surface(proposal) / …; get_entities). Deterministic
  extractor owns coordinates; the LLM asks for slices. Deferred follow-up:
  geo_project ingestion wiring (flagged in drawing_ir/DESIGN.md).
- **D6 seismic reviewer** (first narrow reviewer) — shared checklist in
  `funhouse_agent/review_checklists.py`; surfaces: `.claude/agents/
  seismic-reviewer.md` (Claude Code) and `funhouse_agent.make_seismic_reviewer
  (engine)` / `make_seismic_reviewer_deep(model)` (Funhouse; scope = 10 seismic
  modules + 7 seismic references via allowed_agents). Template for the
  reviewer-family rollout (V5.4_PLAN F8).

**NEXT:**
1. **Owner Funhouse feedback on 5.3** (and optionally the v5.4 pieces). On the
   owner's OK: merge `v5.4`→master, bump 5.4.0, tag (auto-publishes).
2. **E1–E11 QC carryovers + F1–F8 creative builds** — all scoped in
   `module_work/V5.4_PLAN.md` (rapid-drawdown search wrapper, #96 Kc, pore-
   pressure grid, composite-EI, eval refresh w/ new-tool questions, more Slide2
   problems, Bray-Travasarou, reviewer family…).
3. 71-Q eval re-runs: `docs/geotech_eval_20260705.json`/`.md` hold the 5.1rc5
   baseline; re-run on 5.3+ with `[deep,full]` clears the 4 env-blocked fails.

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
- Install: `%pip install "/tmp/...whl[deep]"`, then just `import funhouse_agent.deep`.
  `dbutils.library.restartPython()` is **no longer required** in the normal flow:
  `funhouse_agent/runtime_check.py` (run at the top of `funhouse_agent.deep`) reloads
  the freshly-installed `typing_extensions>=4.13` in place, curing the old
  `typing_extensions`/`extra_items` PEP 728 error without a restart. Restart is kept
  only as the fallback the auto-fix points to if an even older copy is winning at
  cluster scope; installing `typing_extensions>=4.13` as a **cluster-scoped library**
  avoids the situation entirely.
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
- **`module_work/V5.4_PLAN.md`** — CURRENT plan of record (owner directives
  D1–D6 all done; QC carryovers E1–E11 + creative F1–F8 = the open backlog).
- `module_work/V5.3_PLAN.md` — v5.3 board incl. the adversarial-review outcome.
- `module_work/V5.2_COVERAGE.md` / `module_work/WEEKEND_QC_2026-06-13.md` —
  historical boards (complete).
- `validation_examples/INVENTORY.md` + `RESULTS.md` — published problems,
  verdicts, owner notes (coverage-gap backlog lives in the notes).
- `validation_examples/test_published_v0*.py` — the 136+ offline validation tests.
- `drawing_ir/` — NEW (v5.4): LLM-ready drawing IR + query surface; DESIGN.md
  carries the geo_project-wiring follow-up flag.
- `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf` + `docs/user_manual/` —
  the 132-pp manual + regenerable builder (rebuild each release).
- `docs/gallery/` — 12-exhibit module visualization gallery + build_gallery.py.
- `DISCLAIMER.md` + `funhouse_agent/_disclaimer` surfaces — professional-use
  terms (README/PyPI section, first-import notice, geotech-disclaimer script).
- `funhouse_agent/runtime_check.py` — Databricks no-restart typing_extensions fix.
- `funhouse_agent/review_checklists.py` + `funhouse_agent/reviewers.py` +
  `.claude/agents/seismic-reviewer.md` — the narrow-reviewer pattern (seismic
  first; family rollout = F8).
- `funhouse_agent/deep/rc_wheel_check.py` — one-cell Funhouse health check.
- `funhouse_agent/deep/eval_harness.py` — `run_suite` (71-Q eval) + scorers.
- `funhouse_agent/geotech_test_suite.json` — the 71 eval questions.
- `funhouse_agent/_fileio.py` — verified-write + `/Workspace` rescue helpers.
- `docs/V5.1_SUMMARY.html` — everything-since-5.0 summary (owner-facing).
- `docs/funhouse_agent_guide.md` — install + Databricks gotchas + snippets +
  reviewer-agent usage.
- `CLAUDE.md` — project instructions (auto-loaded); its status block is kept
  current and points here.
- Reliability / geo_project / slope_stability / fem2d each have their own
  `DESIGN.md` + `VALIDATION.md` + `UPGRADE_PLAN.md`.
