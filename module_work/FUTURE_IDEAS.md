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

*Status 2026-09-14:* the EXTRACTION half now exists — planlens 0.3.0's
document layer and the app's seven document tools (5.16.0) give the agent a
page map, the document's structure with printed page numbers, located text,
tables and the review markups, with look cues into vision. The
recompute-and-diff harness is still the open part.

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

## 9. App feedback train follow-ups (2026-09-11; ledger in module_work/field_feedback/2026-09-11_app-usage_v5.14.0/)

- **Ensoft-style TABLES of calculation data (owner ask — the next train).**
  `calc_package.export_tables` as a sibling of `render_figures`; each
  module's `calc_steps` gains `get_tables(result, analysis)` (several already
  build `TableData`); CSV + an HTML table block per report; the calc prompt's
  deliverable skeleton gains a Tables section. Per-depth p-y / load-transfer
  tables, per-slice tables, per-layer capacity contributions, iteration
  histories — what LPILE/APILE/GROUP print.
- **Attach from SharePoint** sidebar box: filename/path in, staged like an
  upload, no LLM turn, no size cap (the large-file route without a turn).
- **Chunked websocket upload** (raise the 25 MB cap) — each chunk is widget
  state that reruns; needs the one-shot-key dance per chunk.
- **Specialist agents cannot draw or report**: foundations / earth-retention /
  slope-fem / seismic scopes exclude both `calc_package` and `profile_figure`.
- **Plotly -> PNG twin** for `subsurface.plot_*` — blocked while static export
  is broken in the tenant.
- WON'T: browser -> SharePoint direct upload (delegated Graph token would
  have to live in page JS).

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

## PLANLENS PACKAGE SURVEY 2026-09-14 (web-verified; PARKED — todo, not a train)

**2026-09-16 status: SHIPPED.** Part A (scale / layers / fill), Tier 1 (fuzzy search,
quantities, duplicate scans) and the MCP server are in planlens 0.4.0 on PyPI;
the app wiring + interactive plot_data are in app 5.18.0 on PyPI (see HANDOFF §0a-current).
App wiring (find_quantities + fuzzy search, feature-detected) is BUILT on app
branch `feature/planlens-survey-wiring` (25ba9ed, unpushed, unmerged; carries the
HANDOFF survey paragraph). Pin bump to plain `planlens>=0.4` (0.4.0 is LIVE on PyPI 2026-09-16; raster and
rapidfuzz are in its core now, so no extras).

**APP MERGE CHECKLIST — DONE 2026-09-16 as 5.18.0** (both branches merged --no-ff,
gate 11,773 / 33 / 0, tag pushed, live on PyPI). Kept for the record:
1. `feature/planlens-survey-wiring` (25ba9ed) — find_quantities + fuzzy search,
   feature-detected; carries the HANDOFF survey paragraph.
2. `feature/plotly-plot-data` (d07fe81, a48e723) — interactive Plotly sidecar for
   `plot_data`, PNG card hidden when a sidecar exists, honest figure wording in
   the deep prompt; plan `~/.claude/plans/whimsical-nibbling-scroll.md`; N6 in the
   2026-09-15 field-feedback ledger re-dispositioned.
At merge: pin `planlens>=0.4`; refresh CLAUDE.md state block (planlens 0.4, eight
document tools, rapidfuzz now a planlens core dep — no new direct app deps); bump
version; HANDOFF §0a + install guide §11 row; full gate in chunks; then tag.
Edge noted by the Plotly agent (pre-existing since 5.4, not fixed): a re-imported
sidecar that collides with different content is renamed `plot.plotly_1.json` by
`_unique_dest`, which stops classifying as plotly and stops suppressing the PNG —
degrades to a static card, no crash. Live check after release: one interactive
card, no broken image link, both files in the SharePoint folder, PDF still embeds.
 **Still todo:** Tier 2 scanned-page engines
(local OCR / table / layout models — needs Funhouse to confirm model vendoring
through Nexus; owner may pick up next week) and revision comparison (A4).
**A cheap HOSTED vision model is being scored first, in 5.20.0.** The local
route needs Funhouse to bless vendored weights and cannot be measured until
they do; a hosted model on the tier list needs nothing and can be measured
this week. `report_ingest/vision_labels.py` sends the page IMAGE to
`funhouse-gpt-low` with structured output and the eighteen-label vocabulary,
and `score_on_cluster(stages=("vision_labels",))` scores it against the same
hand labels, with the same scorer, as the rules and the review. If a cheap
hosted model reads a scanned page well enough, the local-model question is
smaller than it looks; if it does not, the measurement says what a local
model would have to beat.

**TODO (owner agreed 2026-09-16): toolkit MCP server.** One thin MCP server over
the DISPATCH layer — `list_agents` / `list_methods` / `describe_method` /
`call_agent` plus `find_worked_examples` / `get_worked_example` — generated
from the existing native tool specs (`funhouse_agent/native_tools.py`), the
way `planlens/mcp_server.py` is generated from `planlens.tools.specs`. NOT one
server per module. Build it right after (a) the Funhouse team names an MCP host
that exists inside the enclave and (b) the planlens server has been proven live
in it. Decisions to make when building: calc packages write PDFs to a working
folder (return a path, a resource, or attach?); reference figures need the
source PDFs on disk (`GEOTECH_REFERENCES_DOCS`); the vision-backed tools
(`read_reference_figure`, `analyze_pdf_page`) must hand the image to the HOST's
model as MCP image content instead of calling the app's engine; the layered
engineering disclaimers must appear in the server's instructions so a host
model sees them before quoting a capacity. Optional extra `[mcp]` on the app
(mcp cleared the Nexus probe 2026-09-16 as 2.2.0 with helper-pinned idna /
pydantic / pyjwt / starlette — check those against the streamlit stack first).
Size: one Opus agent, ~150 lines + tests, same shape as the planlens one.

Full survey: `module_work/PLANLENS_PACKAGE_SURVEY.md` (copy of the plan file;
private repo only). Headline: the three biggest wins need NO new package —
read the calibrated scale that Bluebeam/Acrobat store in the PDF (viewport
`/Measure` dictionaries and measurement markups), carry PDF layer names from
`get_drawings()`, keep path fill. Then, in order: rapidfuzz (fuzzy search over
OCR/SHX text), quantulum3 (quantities out of narrative for narrative-vs-drawing
reconciliation), imagehash (duplicate scans / revision matching); ONNX-only
engines for scanned tables and page layout (RapidAI TableStructureRec,
rapid-layout, OnnxTR with a headless-OpenCV extra); an MCP server over the
existing ReviewToolkit. Rejected with reasons: pymupdf-layout / pymupdf4llm
(Polyform NONCOMMERCIAL), surya/marker (restricted weights), MinerU and
DocLayout-YOLO (AGPL), docling (torch unavoidable — optional only).

**Two conditions before pickup (owner, 2026-09-14):**
1. Translate the survey into plain English for the owner first.
2. It was NOT checked against the org's banned-package list (Nexus inventory
   404s from outside; DT list never saved). Gate = on-cluster
   `nexus_pip_install` probe per package (survey Part E). Cluster egress is
   firewalled: any package that downloads model weights at runtime is unusable
   as published — vendor the models or skip.

## REPORT INGEST TRAIN (2026-09-17) — WP0–WP5 BUILT; 5.20.0 RELEASED; the calc reader landed 2026-09-21

Plan: `module_work/REPORT_INGEST_PLAN.md`. Goal, in the owner's words:
well-organised data about each geotechnical report, usable either as a
sub-agent in this app or as a WikiLLM-style report-library agent. The
**record** is the product; the one-page summary and the DIGGS file are two
exports of it.

**Shipped in planlens 0.5.0 + app 5.19.0 (2026-09-17):**
- **WP0** — the two planlens bugs (a filled-in form read as a duplicate of the
  sheet beside it; a text layer that is there and WRONG now says so), and the
  38-report corpus harness (`module_work/report_ingest_harness/`, dev-only).
- **WP1** — page roles in planlens: what each page of a report IS over
  eighteen roles with its evidence, the work items its pages make, the
  document's printed outline and the per-page ledger.
- **WP1b** — `report_ingest/`: document triage and the label review, the
  engine abstraction (`PrompterEngine` counts; `ClaudeEngine` is for
  development), the scorecard arithmetic, and
  `cluster_scoring.score_on_cluster` for the run on the cluster.

**Built and released as planlens 0.6.0 + app 5.20.0 (2026-09-17):**
- **WP2** — `log_grid` in planlens (a boring log read as the coordinate system
  it is), the record model, the log reader, and a real DIGGS 2.6 writer the
  app's own subsurface reader reads back.
- **WP3** — the lab reader, with a typed result per test kind refused at
  construction when the result does not match the kind, and the DIGGS lab
  elements.
- **WP4** — the narrative reader on the owner's two standing query schemas,
  the reconciler that records a disagreement rather than settling it, the
  writers (record, summary page, WikiLLM-style library page, SQLite index),
  the deterministic resumable graph, the folder runner, and the
  `report_ingest` sub-agent on the app's tool surface behind
  `enable_report_ingest=False`.

**Next:** WP5 — calculation printouts and the long tail. It is the one work
package of the plan not built.

**What WP5 waits on, and so does turning the sub-agent on:** the owner's
four-stage `score_on_cluster` run. Every reader's accuracy is still unmeasured
against a model — the ledger holds the grid-alone and tables-alone baselines
and nothing more, and the development-engine checkpoints were never run for
the logs, the lab or the narrative. The gate that matters is measured through
Prompter, on the tier the app actually runs on.

**Parked with it:** the four disputed hand labels stay disputed (the
spreadsheet is never edited — a hand label records what a person decided);
Azure DI results for four reports would help and do not exist yet; one label
sheet matches no corpus report and stays unmatched rather than being
force-fitted to one of the same length. The owner's published answer
vocabulary in `report_ingest/model.py` was raised against the privacy word
list and ruled generic taxonomy rather than private report text; it stays
verbatim so the hand answers still match, and the owner can object before the
tag (HANDOFF §0a-current).

**Follow-up — the sub-agent's middleware enforces nothing.** deepagents 0.6.8
reads only name/description/runnable from a `CompiledSubAgent` spec, so
`ScratchFilesystemGuard` and `ModelCallBudgetMiddleware` declared on the
`report_ingest` sub-agent enforce nothing; the real ceilings are the
per-reader `Budgets` in Python and the graph exposes no filesystem tool to any
model. Follow-up: enforce the guard and the budget inside the compiled graph
(or wrap the runnable) so the sub-agent has the same protections as the calc
and references sub-agents.

**Landed in 5.20.0 (commit 6e8750e):** the fifth cluster-scoring stage,
`vision_labels` — page images read by GPT-4.1 on the cheap tier with
structured output as a first-pass page classifier, scored beside the rules and
the review. Its number, like every model number, comes from the owner's run.

**PARKED (2026-09-18, built the mode but not the use): vision as a third
voice the review consults.** 5.21.0 adds `mode="document"` to
`report_ingest/vision_labels.py` — the whole report in thumbnail beside a
window of stamped full-size pages — and it is still scored as a RIVAL to the
rules and the label review, a third column in one table. That is the right way
to measure it and the wrong way to use it. The three disagree in a patterned
way: page-mode vision beat both on narrative and figure recall and lost on
plan and lab_test, and the review's own worst pages are the ones no rule could
fire on at all. A label the rules, the review and vision all agree on needs
nobody's attention; a label where vision and the rules disagree is exactly the
page the review should spend a tool call rendering. So the shape worth trying
is not a fourth column but a CHEAPER REVIEW: run document-mode vision first at
low detail (about a cent a hundred pages), hand the review the pages where the
two answers differ as its work list instead of the rules' own low-confidence
flags, and let it spend its budget there. That would be measured the same way
everything else here is — against the same hand labels, with the same scorer,
reporting fixed / broke / still_wrong — and its claim would be a cost claim as
much as an accuracy one. Not started; it needs the document-mode numbers from
the cluster first, because a consultant whose advice is worse than the rules'
own confidence is a consultant worth not calling.

### Disagreement as a signal (owner, 2026-09-18)

The owner's direction after the first cluster day, standing: "even if
something scores worse, it could still be useful. Something like 'wisdom of
the crowd' or random forests; if multiple methods say different things, it
could trigger an extra review or something. Would be good to have confidence
values associated with the classifications and data extractions."

What that means for this train, in order:

1. **Page labels become a vote. BUILT, 5.22.0** --
   `report_ingest/vote.py` and `stages=("vote",)` on `score_on_cluster`,
   with `module_work/report_ingest_harness/measure_wp6_vote.py` as the
   development twin. Three voters exist already -- planlens' rules (with
   their own confidence and evidence, now saved in the run files as
   `rules_confidence`), the label review (a change with a reason and the tool
   that showed it), and the vision pass (label + confidence per page,
   page/sheet/document mode). A page all three agree on is settled at high
   confidence; a page they split on is the page that gets the extra look.
   The stage calls NO model: it reads the run files already on disk and
   reports the agreement rate, the accuracy of agreed against disagreed
   pages, a per-label trust table learned on the in-sample reports ALONE,
   three combining policies (`trust`, `structural`, `confidence`) scored by
   the same scorer as the voters, the disagreement set as a fraction, and
   the accuracy a targeted review of it would need to carry the set over the
   0.98 gate. Every split is listed per report in `vote/<ID>.json`.
   **AND IT IS NOW THE PRODUCTION LABEL PATH (unreleased, on master since
   5.24.0).** `report_ingest/label_vote.py` holds ONE copy of the policies
   and both callers use it: the `vote` stage scores them, and
   `graph.ingest_report` runs them. Inside the ingest the rules, one vision
   pass on the cheap tier (`sheet` mode, about $0.05 a report) and the
   printed form where a fingerprint file is in force vote on every page
   under `label_policy` (default `structural`; `rules` reproduces the old
   behaviour); `review_mode="disagreements"` gives the label review ONLY the
   split pages plus two either side, on a budget of
   `max(20, 0.5 x split pages)`; a split the review does not settle is a
   `QAEntry(kind="label_disagreement")` carrying both voters and both
   confidences; and `ReportRecord.page_labels` carries, per page, the chosen
   label, its confidence, `agreed` and every voter. The `vote` stage saves
   the in-sample trust table to `vote/trust_table.json` and the graph takes
   it by path for `label_policy="trust"`, falling back to `structural` with
   a printed note when there is none.
   **What is left of this item:** running the vote over the 38 saved
   sheet-mode vision runs beside the label runs, which is what says whether
   `structural` stays the default; and the `ingest` stage's RESULTS block
   now prints the split count and the final labels beside the rules alone,
   so that comparison is one run away.
2. **Every extracted value carries a confidence and a method. BUILT for
   the readers, 5.23.0, and for the page labels since** -- `Provenance.method` names the voter (`grid`,
   `tables`, `model`, `model_from_picture`, `reconciled`), every value has a
   confidence, and a slot the voters split on carries the loser in
   `prov.alternatives` beside a `QAEntry(kind="disagreement")` with both
   values and both confidences -- the triggered second look. The page
   labels have the same field now: `ReportRecord.page_labels` is one
   `PageLabel` per page with its confidence, its `agreed` flag and every
   voter's own label and confidence. **What is left:** the reconciler's
   narrative-vs-appendix count check is still a `count_mismatch` rather
   than a vote.
3. **The floor is the first voter for the readers. BUILT, 5.23.0** --
   `report_ingest/floor.py`, `log_floor.py`, `lab_floor.py`: the grid and
   the tables are seeded into the record before any call, the model is
   shown them and may add, correct with evidence, never drop. **What is
   left:** the cluster run that measures it (`stages=("ingest",)` or the
   `logs`/`lab` stages, which now print the grid, the model alone and
   floor+model), and reading the disagreement list off `qa.json` to see
   which slots a reviewer is actually sent to.


### A BLIND calculation truth set is OWED (2026-09-21) -- TODO, not built

The calculation reader shipped with ten hand-truthed runs
(`raw/truth/calc/`, six reports, seven kinds, 44 pages) and **every one of
them is in sample**: they were read while the reader's prompt was written.
So the `calc` stage's `OPEN.txt` names all six reports, `RESULTS.md` prints
"THERE IS NO BLIND SET" where the open/blind line would go, and none of the
numbers it reports is evidence about a report nobody has looked at.

The other three readers all have a blind half and it is the half that means
anything: the lab reader scored 68 % open and 55 % blind on the tables
alone, and the gap between those two is the whole reason the split exists.

**What to build.** Hand truth for six to eight calculation runs in reports
whose calculation pages nobody has opened. Three such reports are named as
the obvious candidates in the private `raw/truth/calc/README.md`: between
them they carry over four hundred hand-labelled calculation pages, in at
least four kinds this set does not cover (liquefaction, axial pile, pile
group, ground improvement) and in shapes it does not cover either — a 1990s
fixed-width slope printout, a scanned sheet the floor reads nothing off at
all, and the largest calculation appendix in the corpus. Written by somebody
who has NOT read the reader's prompt, to the protocol in that README, into
`raw/truth/calc/` with the open reports still named in `OPEN.txt`. The stage
then splits open from blind on its own and the blind number is the one to
quote.

**Until it exists**, quote the calc stage as "in sample" everywhere, exactly
as the scorecard does.


### The ingest ledger (owner, 2026-09-21) -- TODO, not built

One function the ingest calls at the END of every report, whether it ran in
the app, on the cluster or from `run_folder`, writing ONE row about that
report to every sink it can reach. Nothing about a report should have to be
reconstructed from a folder of run files later.

**The sinks, in the order they matter.**

1. An append-only **JSON-lines file in the workspace folder**, mirrored to
   **SharePoint**. That pair is the source of truth: it is what
   `report_ingest.mirror` already does for run files, it survives a cluster
   restart, and a plain text file cannot be broken by a schema change.
2. A **Delta table** for SQL, rebuilt from that file rather than written
   alongside it, through the Funhouse SDK's `DeltaTableManager`. Rebuilt, so
   the file stays the thing that is true and the table is a view of it.
3. A **SharePoint list** through the SDK's list manager, as the human view,
   if the client supports one. This is the optional sink and its absence
   must never fail an ingest.

**The row.** File hash and file name; pages; ingested-at; package version;
the models actually served (the deployments, not the tiers asked for);
document type and workflow; counts found by kind (investigations, samples,
driven records, laboratory tests by kind); narrative fields answered and
null; QA counts by kind; DIGGS written and validated; tokens and dollars;
where the outputs live; and the run id. **A re-ingest appends a new row** --
it never updates one -- so the history of how a report was read is the
history, and a row is never rewritten to look like it always said the new
thing.

**Why it is worth building.** Right now "how many reports have we ingested,
what did they cost, and which ones have a DIGGS file that failed its schema
gate" is answerable only by walking folders of `run.json` files, and only on
whichever machine still has them. One row per report per ingest, appended
somewhere durable, answers all of it in a line of SQL and makes the corpus
itself a thing that can be reported on.
