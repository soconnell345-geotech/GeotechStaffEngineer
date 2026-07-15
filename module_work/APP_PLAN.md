# App workstream plan (owner directive 2026-07-13; plan of record post-5.5.2)

**Focus: the webapp (`webapp/`).** The reference-wiki integration is PARKED
(owner: wiki needs more work before it's turned on this repo). Working rules
unchanged: additive + default-preserving, owner-gated releases, explicit-path
commits, lead reviews everything.

## Context (state at plan start)
- 5.5.2 on PyPI (launcher, eval fixes, fem2d schema/guards). Gate 8557/48.
- App: Streamlit `webapp/` — persistent conversations (dep-free
  persist+replay, per-conversation dir under `~/.geotech_webapp/`), model
  picker, artifact cards (native Plotly via .plotly.json sidecar; HTML iframe;
  PDF), Attach uploads, save hardening (v5.5.1 crash fixed), Databricks
  Prompter launcher (NEEDS-LIVE-VERIFICATION).
- Owner's live sessions: Bamako BTL mat review ×2 (PDFs in docs/).

## A-items (owner's list, 2026-07-13)
- [ ] A1 **Inline Plotly by default** — every agent-produced plot should render
      natively in the chat (extend the #57/#61 output_path+sidecar path so
      plot tools default their output INTO the conversation files dir and the
      sidecar always gets written; agent prompt nudge to save plots rather
      than describe them; cover calc-package figures if feasible).
- [ ] A2 **Calc-agent dispatch / context economics (DESIGN FIRST)** — owner:
      "dispatch a separate calculation agent (or a bunch) so back-and-forth
      doesn't reload unnecessary data." Today each turn REPLAYS the full
      message history (persist+replay). Design options to evaluate:
      (a) deepagents SUBAGENTS for tool-heavy calc work (context isolation —
      the references consultant already works this way); (b) turn ON the
      existing deepagents summarization middleware (build_deep_agent
      enable_summarization) so long chats compress instead of replaying whole;
      (c) durable checkpointer instead of replay. Deliverable: a short design
      memo w/ token measurements on a real transcript, then build the chosen
      combination. This is the industry "context isolation + compaction"
      pattern — see A7 notes.
- [ ] A3 **Never lose a conversation on error** — extend v5.5.1 hardening:
      per-turn atomic persistence INCLUDING mid-turn partial transcript
      (persist user msg + streamed agent text incrementally), crash-recovery
      on next boot ("recovered conversation" banner), guard the streaming
      loop itself (exception → transcript kept + error entry appended, never
      a red-box wipe). Regression: simulate exceptions at every stage.
- [ ] A4 **Permanent files + durable chat links (VERIFY + FIX GAPS)** —
      artifacts already persist in the conversation dir and cards re-render
      from disk on resume; VERIFY weeks-later resume end-to-end (incl. after
      app restart + version upgrade), fix any gap (e.g. cards referencing
      absolute temp paths), and add an explicit test. Document where files
      live for the owner.
- [ ] A5 **Behavior pickers** (sidebar, per conversation, recorded in meta):
      (1) references consult ON/OFF (agent supports reference_mode — check
      build_deep_agent's surface; wire the off option); (2) round caps
      (references_max_model_calls exists; also a primary-agent recursion cap);
      (3) THINKING EFFORT low/med/high — two layers: (a) API-level extended
      thinking IS selectable on Claude models (thinking budget_tokens on
      ChatAnthropic) — wire if our pinned langchain-anthropic supports it;
      (b) prompt-level effort preset (how many methods to run, cross-checks,
      annexes). (4) AUDIT build_deep_agent's kwargs for other worth-surfacing
      options (summarization, setup agent, max_result_chars...) — report.
- [ ] A6 **Working-folder picker** — sidebar setting per conversation
      (default = the conversation's files/ dir): the agent's saves (calc
      packages, plots, save_file) default INTO it (thread the default output
      dir into the agent build — today calc packages default to system temp,
      which is why 5.5.1 wrote to C:/tmp; clunky). Persisted in meta.
- [ ] A7 **Industry-patterns review → recommendations memo** — how agent apps
      are commonly built (context management, subagents, streaming, session
      stores, observability/tracing, eval loops, HITL); compare to ours; DO
      NOT rearchitect without owner discussion. Initial lead take (2026-07-13):
      our stack (LangGraph deep agent + typed tool registry + Streamlit +
      checkpoint/persist + evals) IS the mainstream shape; the biggest
      consensus practices we DON'T yet use: context compaction/summarization
      + subagent isolation (=A2), and lightweight run tracing/observability.
      Streamlit remains the right UI for single-user + TinyApp; FastAPI+React/
      Chainlit only becomes relevant if multi-user is ever needed.
- [ ] A8 **Suppress the Streamlit dev-mode "Rerun / Always rerun" prompt** —
      it's Streamlit's source-file-change watcher (fires because we deploy
      into the folder the app runs from), NOT a project feature. Production
      config (`.streamlit/config.toml`: fileWatcherType none, toolbarMode
      viewer/minimal) + keep dev behavior available via a flag.

## Parked / deferred
- Reference-wiki integration (owner 2026-07-13: wiki needs more work first).
  Priority map when revived: module_work/provenance_audit_*.md wishlists.
- Deferred follow-ups list in V5.4_PLAN.md (toe-circle search sampling,
  steep-φ' Kc, SqliteSaver extra, E2 default).
- **A5(c) API extended-thinking layer — DEFERRED.** The budget_tokens shape the
  directive named (`thinking={type:enabled,budget_tokens}` + temperature) 400s on
  Opus 4.8 / Sonnet 5 (removed API-wide on 4.7+). The modern lever
  (`thinking={type:adaptive}` + `output_config.effort`) IS supported by our pinned
  langchain-anthropic 1.4.4, but it is model-dependent (Haiku 4.5 rejects `effort`;
  Opus 4.8 already defaults to effort "high") and needs a per-model mapping + a live
  smoke test per model before shipping (else 400s / silent default-depth changes).
  A5 ships the engine-agnostic **Analysis-depth prompt preset** now (Screening /
  Standard / Comprehensive; all engines). "Thinking" is reserved for this future
  API control. Follow-up: wire adaptive-thinking+effort with per-model gating,
  owner-gated live verification.
- **A2(ii) summarization backstop — SKIPPED by owner (2026-07-14).** In persist+
  replay the `/memories` key-fact pinning needs a durable store (the parked
  checkpointer), and enabling summarization without the pinning ships the
  compaction risk without its mitigation. Revisit only if a store is adopted (or
  a webapp-level compaction+pinning mechanism is built). A2 shipped part (i), the
  calc sub-agent, only.
- **A2(iii) durable LangGraph checkpointer / drop-replay — PARKED** (owner verdict
  2026-07-13). Extra ~6% context saving over the calc-subagent; touches persistence
  + adds a dependency; also what would unblock A2(ii). Revisit after the calc
  sub-agent beds in.
- **Run tracing — SHIPPED (A7 rec 1, owner-approved 2026-07-14):** optional,
  off by default — LangSmith via env (`LANGCHAIN_TRACING_V2`) + a local per-turn
  JSONL tracer behind `GEOTECH_TRACE=1` (webapp/README.md § Tracing).

## Awaiting externals
- Owner GPT-5.4 eval rerun (quota). TinyApp env answer (form 2026-07-10).
- Databricks launcher live verification (owner's next Funhouse session).

## Record correction (2026-07-14)
- Commit 90b1451's message overstates: A2 part (i) calc sub-agent SHIPPED (default ON via webapp toggle); part (ii) summarization backstop NOT built - /memories pinning requires the parked durable store; shipping compaction without pinning = risk without mitigation. Backstop = owner decision, likely LOW VALUE now: with calc isolation, sessions peak ~16k/turn and would rarely reach any trigger.
- A7 memo: module_work/A7_INDUSTRY_MEMO.md (d20136b).

## Status close-out (2026-07-14) — A-workstream COMPLETE
All A-items DONE + deployed locally (NOT released to PyPI yet; next release =
5.6.0 on owner word). Commits fc1a2e6..09f2fa6 on v5.4, merged to master
(aad984e): A1 inline plotly, A2 calc sub-agent (default ON, -84% measured;
backstop SKIPPED per owner; checkpointer parked), A3 crash-proof turns,
A4 durable files/links, A5 pickers (Agent full/4 reviewers, Analysis depth
Screening/Standard/Comprehensive, references on/off, caps; API thinking layer
deferred), A6 working folder (+GEOTECH_DEFAULT_OUTPUT_DIR), A7 memo
(rec 1 tracing BUILT: GEOTECH_TRACE=1 local JSONL + LangSmith envs;
rec 3 CI eval subset = open backlog), A8 rerun prompt suppressed.
Post-A fixes from owner wall session (2026-07-14): bounded auto-continue for
mid-turn "Let me..." stops (fd93f57, owner wording = regression test);
recursion cap + limit visibility (a0e7dd7); download MIME fix .md-as-.bin
(09f2fa6); streamlit first-run credentials nag fixed (~/.streamlit).

## Open backlog (next cycle)
- INVESTIGATE retaining_walls sliding-check conventions: owner wall session
  showed built-in sliding FoS (Coulomb c=5: 0.92) well below the agent free-body
  check (1.42) — inclined-thrust vertical component + cohesion crediting differ.
  Agent bracketed honestly; module convention needs a documented verdict.
- A7 rec 3: small mock eval subset in the normal test gate.
- Deferred: API thinking layer (per-model gating + live smoke); durable
  checkpointer; SqliteSaver extra; wiki integration (parked).
- Externals: TinyApp env answer; owner GPT-5.4 eval rerun; Databricks
  launcher live verification; 5.6.0 release word.

## Post-close-out fixes (2026-07-15, from owner wall-session follow-up)
- calc_package.html_to_pdf: generic verified HTML->PDF render (PyMuPDF Story)
  for reports no canned *_package covers (ee44509).
- Calc sub-agent framing: REAL DISK ONLY rule — scratch-FS writes are sandboxed
  and invisible to the user; deliverables via output_path/save_file/html_to_pdf.
- Backlog add: retaining_wall_package parameter surface too narrow for atypical
  walls (no toe/heel dims, footing thickness, c=0 override, foundation-δ) — the
  wall session had to bypass it; widen alongside the sliding-convention
  investigation.

## Backlog close-out (2026-07-15)
- SLIDING DISCREPANCY RESOLVED: not a module bug — the double-2/3 trap
  (phi_foundation=22 meant as delta_b -> module applied 2/3 again). Module
  matches the session free-body EXACTLY when driven correctly (1.42/1.01).
  Fixed by delta_base/base_adhesion direct overrides + loud METHOD_INFO
  warnings + sliding_basis echo in results. Verdict: retaining_walls/DESIGN.md.
- retaining_wall_package param surface WIDENED: full geometry (base_width,
  toe, stem, base thickness, shear key), c_backfill, foundation params,
  delta_base/base_adhesion, include_passive now documented + passed through.
