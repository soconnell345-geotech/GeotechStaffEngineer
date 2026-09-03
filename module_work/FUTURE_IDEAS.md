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
- run_on_databricks launcher LIVE-VERIFIED 2026-07-31 (5.10.0 first run:
  engine + driver-proxy + token counter all working).
- **TODO (owner 2026-07-31): Prompter model picker.** The deployment-provided
  engine fixes the model at launch (`run_on_databricks(model=...)`) and the
  sidebar picker is inert ("Model is fixed by the deployment"). Owner wants an
  in-app choice incl. the cheaper `funhouse-gpt-medium` (GPT 5.1). Design:
  registered-builder hook accepts an optional model_id (resolve_engine
  inspects the builder's signature), launcher exposes e.g.
  GEOTECH_PROMPTER_MODELS ("Label=id,..." like the Foundry env) to populate
  the picker; PrompterChatModel(model=picked) per conversation.
- **TODO (owner 2026-07-31): file-upload 403 through the driver proxy.**
  Streamlit uploads use HTTP PUT (/_stcore/upload_file); first live run
  403'd on both drag-drop and file-picker. Launcher now force-sets the
  STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION/CORS env vars (in case the flag
  spelling drifted on the uvicorn-server streamlit). If the localhost PUT
  probe shows the proxy itself blocks PUT, the workaround is the
  agent-facing SharePoint tools (files staged in GSE_app; agent downloads
  them) — document as the supported attach path on Databricks.
- **SDK SURVEY 2026-09-01 (full report in the session; source =
  Funhouse_for_Reference/funhouse-sdk-python + examples_python, 23 topics).**
  Key items, ranked:
  1. **_stream landmines (for the streaming build):** the SDK never streams
     tokens — no recipe exists. `wrap_all_openai_methods` INJECTS
     `collect_usage=True` on any `stream=True` call (prompter_api.py:113) →
     TypeError on naive streaming; bypass via `create.__wrapped__` or a clean
     `OpenAI(http_client=prompter.http_client, base_url=...)` — and then
     RE-ADD usage metering (`logger.meter_log` + `stream_options={"include_
     usage": True}`) because unmetered AI = Terms breach per funhouse-gotchas.
     Steal the `_last_chat_error` re-raise (langchain_prompter_chat.py:452).
     NOTE our databricks_bridge already sends real messages arrays (the SDK's
     own bridge string-flattens; ours is ahead).
  2. **Tiny Apps = the sanctioned durable tier and it SUPPORTS STREAMLIT**
     ("Scaffold Flask/Streamlit for Data.State Tiny Apps") — the original
     webapp design target. Driver-proxy = demo tier (their words). Need the
     owner to export `18. Web App Development/Tiny App Development/` (incl.
     sharepoint-chatbot-app OAuth example) — missing from the zip.
  3. **Auto-continue taxonomy** (apps/agents/agent_auto_continue.py):
     goal-evidence auditing (incomplete_todos, false_completion_claim,
     deferral_after_execution_request...) → richer than our bounded
     auto-continue; portable in an afternoon.
  4. **Azure Document Intelligence** (`fh_doc`): `extract_tables_as_
     dataframes` + the Searchable OCR Overlay recipe (04\04) written for
     engineering drawings — candidate upgrade for scanned boring logs.
     CAVEAT: High-Res OCR add-on DISABLED in tenant; run the folder-14 OCR
     bake-off on real logs first. `fh_prompter.analyze_image` "may silently
     fix typos" — never use it to transcribe measured values.
  5. **Budget sidebar**: `fh_budget.get_current_spend()` + config
     `budget.monthly_budget` (default $50) → "$X of $Y used" caption;
     catch BudgetExceededError explicitly (budget exhaustion = the next
     mystery-failure ticket otherwise). Cache per session.
  6. **Email calc packages**: `FunhouseEmail.send_email(attachments=[(name,
     bytes)])` — shared no-reply mailbox, .gov/.mil/.sbu recipients only;
     `fh_outlook` sends as the user instead.
  7. **fh_secrets** for launch-time credential reads (driver-side only — the
     app subprocess still needs the env hop until/unless Tiny App).
  8. **Skills system**: their deepagents fork loads Claude-Code-style
     SKILL.md files; an example `geotech-deepagent-demo` skill ALREADY
     references a "geotechnical_engineering_query" tool — someone in the org
     is wiring geotech into their framework; reconcile tool naming. Do NOT
     import their deepagent_builder (global monkeypatches for a
     notebook-reload problem we don't have).
  9. **Environment facts**: clusters idle out at 30 min (the detach cause,
     documented); packages from CfA Nexus only; egress firewalled (API-First
     gateway = only external-data route); Plotly static export BROKEN in
     tenant (Kaleido incompat) — matplotlib savefig for emailed PNGs;
     `fh_web` = public static blob publishing (shareable report links,
     non-sensitive only); DuckDB sanctioned for driver-side table queries;
     local embeddings (LocalEmbeddingService) don't hit the budget —
     domain-adapted geotech embeddings = retrieval win.
- **PrompterChatModel true streaming (`_stream`) — conditional on live
  evidence post-5.10.2.** The engine implements only `_generate`: every model
  call is one long silent HTTP request, so with a slow reasoning model
  (funhouse-gpt-high) the browser<->app websocket can idle out through the
  driver proxy ("Connecting" flaps, observed 2026-08-03 alongside the
  write_todos loop). IF flaps persist once 5.10.2 kills the loop: implement
  `_stream` driving `prompter.client.chat.completions.create(stream=True)`
  with an OpenAI-delta accumulator for tool-call chunks and automatic
  fallback to `_generate` on proxy rejection. Benefits regardless: live
  token-by-token output instead of a long "Working...". Offline-testable
  with canned chunk sequences.
- **Ref-agent text refusals (owner observation 2026-07-31, GPT engine):**
  reference subagent answered a capability question but refused/omitted
  actual chapter text in the same session. Collect a concrete failing
  exchange next run (turn-details trace) — likely model-level
  copyright-style refusal on funhouse-gpt-high; consider a prompt line
  distinguishing licensed in-library retrieval from reproduction.

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

## STRUCTURAL SURVEY 2026-09-03 (web-verified; owner asked re openseespy-pedigree peers)

Candidates for the structural direction, filtered for TinyApps posture
(pip-installable, permissive license, no GUI toolkits, light deps):
- **pelicun** (NSF NHERI SimCenter/Stanford, BSD-3, v3.10 Aug 2026, light
  numpy/scipy/pandas): FEMA P-58 loss assessment; v3.10 bundles the
  SimCenter Damage & Loss library in the pip install (no runtime fetch).
  STRONGEST institutional candidate; bridges our seismic outputs to
  performance-based structural risk.
- **sectionproperties + concreteproperties** (R. van Leeuwen, MIT, both
  active 2026): section analysis + RC section capacity. Natural
  module-wrapper targets. concreteproperties needs Py>=3.12.
- **PyNite** (MIT, v3.0, individual maintainer, slower cadence): elastic
  3D frame FE; viz deps now optional extras. Good everyday-frame layer
  below openseespy.
- **structuralcodes** (fib GitHub org, Apache-2.0, v0.7.1 Jun 2026):
  EC2-2004/EC2-2023/fib MC2010 provisions in Python (MC2020 = roadmap;
  pre-1.0 API). The one association-adjacent open library that exists.
AVOID for license (DT GPL sensitivity): anaStruct (GPL-3, though active),
XC (GPL-3, Docker-only, small-practice pedigree not university), FEMA
HazPy (GPL-3, stagnant). Not pip-fit: Code_Aster (EDF), CalculiX, OOFEM
(official 3.0 not on PyPI; only a dev pre-release wheel).
ASSOCIATIONS: AISC Shapes DB v16.0 free BUT readme forbids reproduction
without written permission — CANNOT embed tables in a package without an
AISC letter (worth requesting if we build steel modules). ACI: nothing
open (318 PLUS = subscription). ASCE Hazard Tool API = paid/key-gated;
free substitute for seismic = USGS design-maps web services (keyless,
ASCE 7 values by coordinate) + USGS RTGM calculator Python source — note
TinyApps egress is firewalled, external APIs must route via API-First.
NIST: BAM-DB archetype dataset (OpenSees + P-58 models) — data, not lib.

### Structural round 1 decisions (owner, 2026-09-03)
BUILD NOW: sectionproperties + concreteproperties + PyNite wrappers.
DEFERRED (promising, next round): (a) **pelicun** — FEMA P-58 loss layer on
top of our seismic outputs (BSD-3, SimCenter; bundles the P-58 damage/loss
library since v3.10); (b) **ASCE 7 values via the free keyless USGS
design-maps web services** (+ USGS RTGM Python source) — needs the
TinyApps API-First egress route, so wait for pilot networking answers.
DROPPED: XC, OOFEM, Code_Aster/CalculiX, anaStruct (GPL), HazPy.
