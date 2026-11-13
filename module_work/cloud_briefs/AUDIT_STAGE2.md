# Cloud helper brief: the module audit's stage 2 (2026-11-12)

Read `COMMON_MODULES.md` (beside this file) first: its rules bind every task here. This file adds what is common to the audit's fixes, then has one section per task. Your routine's prompt names your section; do that section only.

## Common to every task here

- **The audit** is `module_work/module_audit_2026-11-12/AUDIT.md`. Its section 4 has each module's finding table (severity A, B, C, with path:line); `results/` beside it has the scouts' full rows. Your BACKLOG item says what to fix; the audit's rows are the evidence. Line numbers are from master b5579d2 and may have moved.
- **The yardstick:** the agent can do everything it is told it can, and a result says what it is.
- **Confirm first.** Open the cited lines and confirm each finding on today's master before changing anything. If a finding is wrong, say so in CHECK.md and leave the code.
- **Three ways to close a finding**, in this order of preference:
  1. make the code do what the tool says, where a printed page (COMMON_MODULES, "Print only") or plain arithmetic gives the rule;
  2. refuse the input, with a message that says what is missing or not supported;
  3. keep the behaviour and say it: in the result (`warnings`, `choices`, `defaults_used`) and in the tool description.
  Never a silent basis.
- **Calls.** Where engineering judgment is needed between printed options, do not stop for the owner: choose, state both forms in the tool description and the result, and list the call in CHECK.md under "Calls made" for the owner to revisit (the owner's rule of 2026-11-10).
- **Scope.** Fix the findings your section names and the B findings in the same functions. Everything else is "noted, not fixed" in CHECK.md.
- **Two things another task is building; do not build them yourself.** Branch `claude/gm-316-317-warnings-allowed-values` makes a Python warning raised inside a method reach the agent in the result's `warnings`, and refuses a value outside a parameter's `allowed_values`. So where a finding is only "this `warnings.warn` never reaches the agent", leave the warning as it is. Where the result must say something even with warnings delivered (a flag, a count, a refusal, an entry in `choices`), do that.
- **A changed tool description:** hand-edit the matching rows of `docs/user_manual/manual.html`; re-record fingerprints in a commit of its own (COMMON_MODULES).
- **Do not edit:** `BACKLOG.md`, `HANDOFF.md`, `CLAUDE.md`, `slope_stability/` (change hold), `packages/aec-document-review/`, `packages/aec-agent-core/` (the dispatch task owns the core this round), `geotech-references/`.
- **CHECK.md also carries:** a table of every finding you took (the audit's row, and what you did: fixed, refused, stated, wrong finding, or not done and why); "Calls made"; a release note if any returned value changes.

## DOC-24: quick fixes

Branch `claude/doc-24-audit-quick-fixes`. CHECK at `module_work/doc24/CHECK.md`.

BACKLOG DOC-24, and the audit's section 5: Q1 to Q22. Each is a wording or document change; tool description strings count as wording. No behaviour changes.

- Skip Q10 and the CLAUDE.md half of Q8 (the lead does those files), and Q11 (folders on one machine, not in git).
- For each of the rest: open the cited lines, confirm the claim is still true, then make the text true. If the claim no longer holds, skip it and say so.
- Q1 to Q6 (DESIGN.md public-API sections): list what the package's `__init__.py` really exports, with real signatures read from the code. Do not describe behaviour you have not read.
- Q9 (the two empty extras in `pyproject.toml`): grep the whole repo for `hvsrpy` and `swprocess` as extras (docs, the manual's extras table, install guides, tests) and fix every consumer.
- Tests: `webapp/tests/test_page_fingerprints.py` (re-record if a description moved it), `webapp/tests/test_public_text.py`, `webapp/tests/test_docs_currency.py`, and every test file under `funhouse_agent/tests` that names an adapter you touched.
- CHECK.md: a table Q1 to Q22: done or not done (why), and the commit.

## GM-318: piles longer than the soil profile, and failures that vanish

Branch `claude/gm-318-piles-short-profile`. CHECK at `module_work/gm318/CHECK.md`. Modules: `axial_pile`, `drilled_shaft`, `lateral_pile` (audit section 4, those three).

- **A profile that does not cover the pile** (pile tip below it; a shaft longer than it; lateral soil that starts below the ground or stops above the toe). Today the last layer is extended and only a Python warning says so. Read what each module's printed method says about the soil at and below the tip. Unless a page prints the extension, the default becomes a refusal that names the depth not covered, with one explicit input that extends the last layer and says so in `choices` and `warnings`. End bearing in soil nobody explored is the case that matters most; say that in the message. Every probe, worked example and validation example that relied on the silent extension is a finding: list each, and either give it a profile that covers the pile or pass the explicit input. This default is a call: record it under "Calls made".
- **`capacity_vs_depth`** (axial_pile) drops a depth that fails: name the skipped depths and why, as the drilled shaft version does.
- **The cracked-section stiffness loop** (lateral_pile): when it does not converge, the result carries a flag, the number of passes and the last change.
- Leave the drilled-shaft uplift wording to DOC-24.

## GM-319: pile_group, three things the tools say and do not do

Branch `claude/gm-319-pile-group-interface`. CHECK at `module_work/gm319/CHECK.md`. Module: `pile_group` (audit section 4).

1. The manual offers an elastic equivalent-footing settlement; the function exists and no tool reaches it. Check the function against its printed source first, then add the tool (adapter method, its description, a probe, the manual row). If it does not match the print, fix it or say what differs.
2. `pile_group_6dof` accepts batter angles and leaves out the batter coupling terms. Read the method's printed source. Add the terms if the print gives them; otherwise say in the description and the result that they are left out.
3. `pile_group_simple` accepts per-pile batter and stiffness and ignores them. Use them or refuse them.

## GM-320 and GM-321: settlement and support of excavation

Branch `claude/gm-320-321-settlement-soe`. CHECK at `module_work/gm320/CHECK.md`. Modules: `settlement`, `soe` (audit section 4).

- **Settlement.** `consolidation_settlement`: refuse `layers` and `delta_sigmas` of different lengths. Schmertmann: when the profile stops above the bottom of the influence zone, today the missing depth adds nothing and nothing is said. Read the printed procedure; unless it prints a rule, refuse and name the depth not covered, with one explicit input for what lies below (the last layer continued, or incompressible), stated in the result. Record the default under "Calls made".
- **soe.** `braced_excavation` accepts each support's `spacing`, `angle_deg` and `preload_kN_per_m` and reads none. Use spacing and angle as GEC-4 and Caltrans Trenching and Shoring print them (load per support from the load per metre of wall and the spacing; axial load of an inclined support from its angle), with page cites, and return both the per-metre and the per-support loads, named. Preload: use it if a held page prints how; otherwise stop accepting it, with a message.
- Leave the blow-out unit-weight wording and ground_improvement's feasibility wording to DOC-24.

## GM-322 and GM-323: reliability, pystra, salib

Branch `claude/gm-322-323-reliability-stack`. CHECK at `module_work/gm322/CHECK.md`. Modules: `reliability`, `pystra_agent`, `salib_agent` (audit section 4). Do what the two BACKLOG items list. Notes:

- FOSM and PEM with `lower` / `upper` bounds: refuse (those two methods take no truncation), saying which engines do.
- Monte Carlo: report how many samples were dropped as non-finite, and refuse to report a probability when the share dropped is large (state the threshold you choose and why).
- pystra: a failed SORM returns no SORM numbers and says it failed; `converged` is pystra's own flag; nothing falls back to an empty value without saying so; the seed is local to the call.
- salib: `n_samples` and `n_trajectories` are checked against the design the samples imply, or dropped from the tool; the expression compiler takes the reliability engine's limits.
- Add one made-up truncated case run through FOSM, PEM, Monte Carlo and FORM that shows which agree and which refuse.

## GM-324 and GM-325: liquefaction outputs, site response, signals

Branch `claude/gm-324-325-seismic-outputs`. CHECK at `module_work/gm324/CHECK.md`. Modules: `liquepy_agent`, `seismic_geotech`, `pystrata_agent`, `seismic_signals_agent` (audit section 4). The optional libraries these wrap must be installed in your environment (see `pyproject.toml`'s extras); if one will not install, say so and do the rest.

- **Told returns.** The CPT tool and `field_correlations` say they return values the results do not carry. The values are computed: return them under the names the tool gives (or fix the names in the text where the result's are better), and add a test that every key a tool promises is in its result.
- **One total-stress rule.** The same `gamma` input gives two total stresses: summed by layer in one route, the point's own unit weight times depth in the other. Total vertical stress is the sum of the layers above; make both routes do that unless a printed procedure says otherwise, show before and after on a made-up layered profile, and write the release note.
- **pystrata.** The mean effective stress behind the Darendeli and Menq curves is computed from total stress with K0 = 0.5 and no water table: take the water table and K0 as inputs, state both in the result, and say in the description what is assumed when they are not given. Return the spectra and the profiles with depth that are already computed.
- **Signals.** A duration that could not be computed is returned as None with the reason, never 0.0.

## GM-326 and GM-327: site data and drawing import

Branch `claude/gm-326-327-subsurface-dxf`. CHECK at `module_work/gm326/CHECK.md`. Modules: `subsurface_characterization`, `dxf_import` (audit section 4). Do what the two BACKLOG items list. Notes:

- `compute_trend`: fewer than two points is refused; the log fit reports how many points it dropped; the unknown trend type is the dispatch task's (leave it).
- The GEF reader keeps a missing or non-finite reading missing (not 0.0) and reports the count per column; check what the readers of its output do with a missing value.
- `dxf_import`: every default that stands in for a soil or nail property is listed in `defaults_used`; a missing strength is refused (the module's own documents say strengths come from the user); the drawing's own units are compared with `units`, and a mismatch is refused or converted, said either way; text annotations are converted like geometry.

## GM-310: the default changes from the 2026-11-12 judgement calls

Branch `claude/gm-310-judgement-call-defaults`. CHECK at `module_work/gm310/CHECK.md`.

The source, binding: `module_work/owner_calls_2026-11-12/CALLS.md` section B (the calls for OWN-84, OWN-85, OWN-82, OWN-68, OWN-108, OWN-112, OWN-114, OWN-118 and OWN-131) and, from its section A, the call for OWN-105 (the downdrag summary's wording). BACKLOG GM-310 lists the ten changes. Moment-curvature EI (OWN-19) is GM-311, not this task. Nothing in `slope_stability/`.

- One commit per default: the code, its tests, true `choices` and `defaults_used`, the tool description, the manual row. The fingerprints commit comes last.
- Every one of these moves answers. For each default, CHECK.md gives before and after on one made-up case, and a release note. List every probe, worked example, validation example and saved fixture whose answer moves or which is now refused; an example that is now refused gets the input it now needs, and you say so.
- Where a call says "refuse without X", the message names X and what to give.
- Other sessions are editing the same modules at the same time, in other functions (piles, pile groups, liquefaction outputs). Keep each change inside the function it belongs to and do not reformat or reorder a file.
