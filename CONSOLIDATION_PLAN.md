# Module Consolidation Plan

Source of truth for the scope-narrowing effort kicked off from `MODULE_OVERVIEW.html`.
Owner: fleet agent `module-review-and-consolidation`. Created 2026-06-09.

**Operating notes**
- A QC agent is editing the repo concurrently (mostly geotech-references chapter text). Low file overlap expected with this work; verify before touching shared metadata (`pyproject.toml`, `CLAUDE.md`, `README.md`).
- **No version bumps / PyPI publishes** until the user explicitly says so.
- Removals are git-recoverable; still done carefully (delete dir + de-wire from pyproject, funhouse adapters/dispatch, system_prompt, foundry, README/CLAUDE) with `pytest` after each.

## Decisions (from user feedback 2026-06-09)

| # | Item | Decision | Concrete action | Gating | Status |
|---|------|----------|-----------------|--------|--------|
| 1 | 1-D site response ×3 | Keep `pystrata_agent` + `opensees_agent`; **cut `pyseismosoil_agent`** | Remove pyseismosoil module, optional-dep, adapter, dispatch entry, foundry wrapper, doc refs | — | READY |
| 2 | Two continuum solvers | **Cut `fdm2d`** (keep `fem2d`) | Remove fdm2d module + pyproject (find/testpaths), adapter, dispatch ANALYSIS_MODULES, system_prompt, doc refs | — | READY |
| 3 | 4 data-format parsers | **Merge** `subsurface_characterization` + `pygef_agent` + `ags4_agent` + `pydiggs_agent` → one data-I/O module | Design unified ingest API w/ format adapters (DIGGS parse+validate, GEF/BRO CPT, AGS4) | after Phase 1 | DESIGN |
| 4 | Classification/bearing dup | **Remove `geolysis_agent`** (no new value vs base + groundhog) | Remove module, optional-dep, adapter, dispatch, foundry, doc refs | — | READY |
| 5 | Liquefaction two paths | **One unified API** over SPT + CPT — but confirm methods first | SA-5 subagent: are `seismic_geotech` (Seed-Idriss SPT) and `liquepy` (B&I 2014 CPT) genuinely distinct, or does B&I's SPT update already subsume Seed? Then build unified API | SA-5 ✓ | API-READY |
| 6 | Lateral pile two engines | **Retire opensees lateral-pile overlap** — but harvest first | SA-6 subagent: review opensees BNWF lateral-pile code for non-cyclic methods/features worth porting into native `lateral_pile`; then retire that pathway. **Module stays for site response (#1).** | SA-6 ✓ | READY (nothing to harvest except optional stickup feature) |
| 7 | Reliability + sensitivity | **Backlog — bigger eval.** Assess whether `pystra`/`salib` even apply; user wants: statistical variability of subsurface properties + probabilistic geotech analyses (FOSM, Monte Carlo); possibly build own library. `gstools` may fold in here. | Scoping/build-or-buy evaluation (separate milestone) | — | BACKLOG |
| 8 | Scope outliers | **Remove `wind_loads`**; **keep `gstools_agent`** (may fold into #7 later) | Remove wind_loads module + pyproject + adapter + dispatch + foundry + doc refs | — | READY |
| 9 | Two deployment surfaces | **Conditional.** Retire foundry wrappers for retired modules; assess overall staleness. Keep general Foundry functions (user may build Foundry *reviewer* adapters later — simpler than the Funhouse ReAct harness). | (a) delete foundry wrapper alongside each retired module; (b) assess foundry adapter staleness vs current APIs → report | partial | PARTIAL + ASSESS |
| 10 | Two GUI stacks | **Retire both** Dash + Qt (future: proper web app) | Remove slope/fem Dash GUIs, Qt GUIs, `qt_panels/`, `gui` optional-dep, GUI doc sections. Keep underlying analysis modules. | — | READY |
| 11 | Legacy equation layer | **Cut `DM7Eqs`** after confirming coverage | SA-11 subagent: verify everything in `DM7Eqs/` (UFC 3-220-10/20 eqs) is already covered in geotech-references; report gaps that would be lost. Then delete. | SA-11 ✓ | READY (safe to delete; superset confirmed) |

## Open question for user
- **#1 vs #6 opensees:** interpreting as "keep opensees_agent for site response, retire only its lateral-pile (BNWF) overlap after harvest." If you meant remove the whole module, say so — that would also drop a site-response engine (then #1 = pystrata only).

## Execution phases
- **Phase 0 (in progress):** this plan + 3 read-only investigations (SA-5, SA-6, SA-11).
- **Phase 1 (ungated removals, batchable):** #1 cut pyseismosoil · #2 cut fdm2d · #4 cut geolysis · #8 cut wind_loads · #10 retire GUIs · #9 (retire foundry wrappers for those modules).
- **Phase 2 (gated by investigations):** #6 harvest + retire opensees lateral-pile · #11 cut DM7Eqs · #5 build unified liquefaction API.
- **Phase 3 (build):** #3 merge data parsers into one I/O module.
- **Backlog:** #7 probability/reliability module (build-or-buy eval; gstools fold-in) · #9 foundry staleness assessment.

## Investigation findings

**SA-5 (liquefaction) — done.** The two paths are genuinely distinct, on BOTH axes (SPT vs CPT *and* old vs new method).
- `seismic_geotech/liquefaction.py` implements the **NCEER/Youd-2001 SPT** procedure (NCEER CRR fit, NCEER MSF `10^2.24/M^2.56`, Youd α/β fines, Liao-Whitman rd) — i.e. the predecessor that B&I-2014 updates.
- The "Boulanger & Idriss (2014)" line in its docstrings/`__init__` is **cited but NOT implemented** (the only real B&I code is `residual_strength.py` = Idriss & Boulanger 2008, a separate post-failure calc). → **Side bug to fix: correct the misleading citation.**
- `liquepy_agent` implements B&I-2014 on CPT via `liquepy.trigger.run_bi2014`; LPI/LSN/LDI/strain/settlement live ONLY on the CPT path.
- **Unified-API rec (option b):** one B&I-2014 surface routing on SPT-or-CPT input; re-point SPT at B&I-2014-SPT (liquepy supports it) so both share one CRR/MSF/rd basis; optionally keep Youd-2001-SPT behind an explicit `method="nceer2001"` legacy flag for code-compliance. Common core out = depth/CSR/CRR/FOS/liquefiable; surface LPI/LSN/LDI conditionally (CPT-only for now).

**SA-6 (opensees lateral-pile) — done.** Clean to retire; essentially nothing to harvest.
- `opensees_agent/bnwf_pile.py` is thin/derivative — it **imports native `lateral_pile/py_curves.py`** (no unique p-y model), samples (pult,y50) into OpenSees `PySimple1` springs. No new static physics; monotonic LoadControl only; no cyclic/dynamic (Cd=0).
- Native `lateral_pile` is the superset (variable EI / cracked-RC, HP/filled-pipe/RC sections, partial head fixity, richer outputs).
- **Only harvestable idea:** native module lacks an **above-ground free-length (stickup)** option — re-implement natively in `solver.py`/`analysis.py` (extend mesh above z=0 with zero soil resistance, apply head load at top). Skip BNWF's crude placeholder t-z/Q-z springs. → added to backlog.
- Action: retire the opensees lateral-pile pathway (`bnwf_pile.py` + its dispatch/adapter exposure); KEEP opensees site-response + PM4Sand.

**SA-11 (DM7Eqs coverage) — done. Safe to delete.**
- `DM7Eqs/` = ~354 LLM-callable functions from UFC 3-220-10 (ex-DM7.01) + UFC 3-220-20 (ex-DM7.02): index props, strength, compaction, Boussinesq, seepage, consolidation/settlement, in-situ correlations, earth pressures, bearing capacity, deep foundations, subgrade reaction, reliability (FOSM/PEM/MC).
- Cross-checked every public `def` (name + body) vs `geotech-references/geotech_references/{dm7_1,dm7_2}`: **byte-identical** public functions; references is a strict superset (e.g. adds `table_4_9_tunnel_behavior`). Only diff = private `_linterp` helper placement (already present in references). **Zero gaps.**
- Caveat: DM7Eqs ships its own 1852 tests; references has its own suite. Belt-and-suspenders before delete: confirm references suite exercises the equivalent chapters. Action: delete `DM7Eqs/` + remove its `pyproject` exclude entry + DESIGN refs.
- Bonus: DM7Eqs's reliability functions (FOSM/PEM/Monte Carlo) are relevant to the #7 probability/reliability backlog — note as prior art before building anew.

## Net effect on module count (analysis layer)
Removed: pyseismosoil, fdm2d, geolysis, wind_loads (+ opensees lateral-pile pathway). Merged: 4 parsers → 1. GUIs + DM7Eqs retired. Roughly 36 → ~30 analysis modules, fewer third-party deps.
