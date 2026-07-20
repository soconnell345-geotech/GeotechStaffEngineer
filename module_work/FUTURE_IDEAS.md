# Future possibilities — considered 2026-07-18 (owner-requested ideas pass)

> **UPDATE 2026-07-20 — the sample-calc-as-defect-detector doctrine (proven).**
> Items 1-2 below evolved into an operating doctrine that FOUND FOUR REAL
> DEFECTS in three days (Reese cyclic tables, drawdown Kf c'>0, shaft
> depth-beta units, Nordlund qL — two conservative, two UNCONSERVATIVE up to
> 10x), all invisible to the self-consistent test suite. The working recipe:
> (a) onboard a public worked-example source (PDF -> refs docs/ -> text-search
> the example pages -> ONE curation agent drafts execution-verified corpus
> entries), (b) run internal-only sweeps against solved-problem sources (Das
> manuals: 8/8 reconciled <=0.3% — clean bill confirming the fixes), (c) treat
> every unexplained discrepancy as a defect investigation, verify against the
> printed page before changing code, and pin the corrected value with tests.
> NEXT TARGETS: SCDOT design examples (public web), USACE EM appendices,
> remaining NHI manuals in the owner's library, CGPR #20-style program
> comparisons; plus the six-item ergonomics backlog in
> `wiki_verification/TIER_A_LEDGER.md` (top: axial_pile beta global
> cohesive_phi doc-vs-behavior trap, +12.4%).

Written at the owner's request ("consider and build future possibilities...
creative ideas welcome") during the final Fable session. Ordered roughly by
value-per-effort. Items 1 and 2 got a v1 BUILT this weekend; the rest are
specs for a future session. Standing rules apply (additive, owner-gated
releases, validate-don't-tune).

## 1. Worked-examples corpus (BUILT — v1 shipped this weekend)

`funhouse_agent/worked_examples.json` + `worked_examples` dispatch module +
prompt wiring. Validated calculations from real published design reports
(GEC-12 pile abutment, GEC-10 shaft, GEC-11 MSE, GEC-6 footing, Caltrans
shoring, GEC-13 ground improvement, slope benchmarks incl. Pilarcitos Dam,
FLAC consolidation, AASHTO/UFC pavements) as agent exemplars: problem →
dispatch calls → published vs computed answer → report notes. Every entry is
mechanically verified by test (its calls RUN, offline, in the gate).

**Phase 2 — the owner's own reports as exemplars.** The owner asked for "real
ones in real reports": let the firm's actual calc packages join the corpus.
Sketch: a `GEOTECH_EXAMPLES_DIR` of PDFs; harvest with `pdf_import` text
extraction into per-report JSON stubs (problem narrative + key numbers +
which module methods reproduce them); a curation step where the agent
PROPOSES the dispatch-call reconstruction and a human confirms before the
entry is trusted (provenance: "firm report, unverified" vs "verified
reproduction"). Keep firm reports strictly local — never packaged to PyPI.
The existing FTS5 retrieval layer in geotech-references is the model for
scaling past ~50 entries (swap keyword scoring for an index).

## 2. Playbooks — standard multi-step workflows as data (spec)

The worked-examples corpus answers "how was THIS problem solved"; playbooks
answer "what is the standard sequence for this TYPE of job": e.g. shallow
foundation: site class → bearing (2 methods) → settlement (elastic +
consolidation) → sliding → report; or MSE wall: external stability → internal
→ global (slope module) → report. Implementation mirrors worked_examples: a
JSON registry (`playbooks.json`: steps, each with module/method hints, checks,
report section), one adapter with `find_playbook`/`get_playbook`, prompt nudge.
The calc agent then plans real jobs against a vetted sequence instead of
improvising step order. Seed 6-8 playbooks from the GEC report structures the
reference layer already holds.

## 3. Recompute-from-report QC mode (spec)

Feed a finished calc package (ours or a third party's PDF) back to the agent:
extract the claimed inputs/results (`read_pdf_text` + vision for charts),
re-run the calculation through the modules, and produce a diff table
(claimed vs recomputed, flag > tolerance). The reviewer-agent family already
exists (webapp review mode); this adds the extraction+diff harness. Killer
app for the owner's actual job (reviewing others' geotech reports). Start
narrow: our own calc-package PDFs (known layout), then generalize.

## 4. Single-namespace restructure (ASSESSED — parked as 6.0.0)

35 top-level modules → one `geotech_staff_engineer.*` package; kills the
name-squat risk (`reliability`, `settlement`, `webapp` collide with real PyPI
packages). Mechanical, scriptable, needs compat shims (old top-level names as
re-export stubs for a deprecation window) + full-gate QC + coordinated update
of the owner's Foundry app.py and Databricks notebooks. Do as a dedicated
major-version train; shrinking the dependency tree at the same time would cut
Foundry's slow environment-restore.

## 5. Reliability program (long-standing next big build)

Memory: "reliability module is the next long-term build" (post LE+FEM
modernization). Direction: system reliability across modules (not just slope)
— FORM/SORM on any module's limit state via a generic wrapper, correlated
inputs, spatial variability (Vanmarcke averaging already in), target-β design
iteration ("find B such that β ≥ 3.0"), and probabilistic PAVEMENT design
(reliability beyond the AASHTO ZR·So lump — owner declined for now, revisit).

## 6. Foundry/Databricks deployment hardening

- Pending Monday: admin answer on the gov-enclave LLM proxy (401 saga —
  FOUNDRY.md troubleshooting section has the full story + ticket text). The
  fix lands as one GEOTECH_FOUNDRY_HOST line.
- Conversations on a durable store (Foundry dataset / DBFS) instead of
  container-local disk, so published-app restarts keep history.
- Claude RIDs via the Anthropic proxy path when the enrollment enables them
  (code already routes by RID text; zero change expected).
- run_on_databricks launcher still NEEDS-LIVE-VERIFICATION on the owner's
  cluster.

## 7. Eval + CI

- Owner's GPT-5.x live eval rerun on the 108-question suite (PAV questions
  new); triage vs the 68 answer keys (eval_harness `--ids`, results md).
- CI mock-eval subset (old backlog): run the harness --dry-run + a keyed
  subset through a stub in GitHub Actions so dispatch regressions surface on
  push, not at release.

## 8. Smaller punch-ups (old backlog, unchanged)

- slope_stability toe-circle search under-sampling; steep-φ' Kc validation.
- SRM mesh-consistency follow-up (fem2d).
- foundry/ retired AIP-wrapper directory cleanup (+ foundry_test_harness) —
  deletes ~50 files; owner-sanctioned housekeeping, do in a quiet moment with
  a full gate after.
