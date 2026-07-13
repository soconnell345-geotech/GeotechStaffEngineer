# A2 — Context economics: design memo (measure-first)

**Status:** design memo for team-lead sign-off. **No build yet.** (APP_PLAN.md A2.)
**Author:** app-dev. **Date:** 2026-07-13.

## Problem

Each webapp turn **persist+replays the full message history** (`ss.messages` →
`core.stream_turn`). Input tokens per turn therefore grow with everything said
and every tool result produced so far. Calc turns are the expensive ones: a
single bearing/slope calc dump, a `describe_method` listing, and a reference
excerpt are ~10k tokens of tool output that then ride along in *every*
subsequent turn's input.

## Method (measured, not guessed)

Offline model of a **10-turn calc-heavy session** (the Bamako BTL-mat style; the
owner's 2026-07-13 transcript ran ~40–80k tok/turn). Content blocks are realistic
geotech payloads sized with the **real Claude tokenizer** (`anthropic
messages.count_tokens`, model `claude-sonnet-5`); script:
`module_work/a2_measure.py` (reproducible; falls back to a char/3.7 proxy with no
key). Fixed per-turn system+tool-schema overhead assumed 6,000 tok (affects all
strategies equally). Model input window 200k (Opus 4.8 / Sonnet 5).

Measured block sizes (real tokenizer): calc result **7,758**, method dump 1,332,
reference excerpt 1,250, final answer 761, user question 53, **compact subagent
return 168**, summary block 431. New history per calc-turn: **11,512** when tools
run in the main thread vs **982** when the calc runs in a subagent.

## Numbers

Per-turn **input** tokens:

| turn | (a) replay | (b) sum @160k default | (b) sum @30k aggressive | (c) calc-subagent | (c)+checkpointer |
|-----:|-----------:|----------------------:|------------------------:|------------------:|-----------------:|
| 1  |  17,512 |  17,512 | 17,512 |  6,982 | 6,982 |
| 3  |  40,536 |  40,536 | 40,967 |  8,946 | 6,982 |
| 5  |  63,560 |  63,560 | 40,967 | 10,910 | 6,982 |
| 8  |  98,096 |  98,096 | 40,967 | 13,856 | 6,982 |
| 10 | 121,120 | 121,120 | 40,967 | 15,820 | 6,982 |

Cumulative input over the 10-turn session (what you actually pay for):

| strategy | cumulative input tok | vs (a) |
|---|---:|---:|
| (a) full persist+replay | **693,160** | — |
| (b) summarization, default trigger (0.8×200k) | 693,160 | **0%** |
| (b) summarization, aggressive trigger (30k) | 374,272 | −46% |
| (c) calc-subagent isolation (+ today's replay) | 114,010 | **−84%** |
| (c) calc-subagent + durable checkpointer | 69,820 | **−90%** |

## Findings

1. **The default summarizer is inert for real sessions.** deepagents attaches a
   summarizer, and our `enable_summarization` path defaults its trigger to
   `("fraction", 0.8)` = ~160k on a 200k model. A 10-turn calc session peaks at
   121k, so **summarization never fires** — (b)-default is byte-for-byte (a).
   Turning `enable_summarization=True` with defaults changes nothing at these
   lengths.
2. **Aggressive summarization helps moderately but is the riskiest lever.** A 30k
   trigger plateaus per-turn at ~41k (−46% cumulative), but it compacts away the
   very numbers an engineering thread depends on (accepted soil params, chosen
   dimensions, prior FS results) unless they're pinned elsewhere.
3. **Subagent trace isolation is the dominant lever (−84%), and it works from
   turn 1** regardless of window size, because the ~10k of calc/method/reference
   output per turn never enters the accumulating main thread — the main agent
   sees only a ~170-token result summary. It also *composes* with everything
   else. The **references consultant is the in-repo precedent** for exactly this
   (`build_references_subagent`, scoped tools, `ModelCallBudgetMiddleware`
   bounding the subagent's own internal cost).
4. **Dropping replay (durable checkpointer) adds the last −6%** (−90% total) and
   flattens per-turn input to ~7k. It's a smaller, riskier structural change than
   (3) (persistence rework, LangGraph SqliteSaver dependency) and can follow.

## Recommendation

**Build (c) first: a calc/analysis subagent that keeps tool-heavy traces out of
the main thread — mirroring the references consultant.** It is the largest,
lowest-risk, additive win (−84%), needs no persistence rework, and reuses a
proven in-repo pattern.

Pair it with a **summarization backstop tuned for pathological length** — enable
`GeotechSummarizationMiddleware` at a *moderate* absolute trigger (≈60–80k, not
the inert 160k default and not the parameter-eating 30k), **with key facts pinned
to `/memories`** so compaction can never drop them. This only activates on
unusually long threads; normal sessions never reach it.

**Defer** the durable-checkpointer / no-replay change (extra −6%) to a separate,
owner-discussed step — it touches persistence and adds a dependency, for a small
marginal gain once (c) is in.

## Risks & mitigations

- **Summarization drops engineering parameters mid-project** → pin accepted
  values (soil params, geometry, governing FS, chosen sections) to the deep
  agent's `/memories` store (`use_longterm_memory`) and/or working notes; keep
  the trigger moderate so it rarely fires; keep-last-8 preserves recent exchanges.
- **Subagent loses context the main agent needed** → the calc subagent returns a
  structured, complete result (values + saved artifact path + citation), the same
  contract the references consultant already satisfies; the artifact (calc
  package) remains on disk as a card, so nothing is truly lost.
- **Subagent double-spend** → its internal cost is bounded by
  `ModelCallBudgetMiddleware` (the existing `references_max_model_calls` lever);
  set a calc budget too.
- **Measurement is a model, not a live trace** → block sizes are real-tokenizer;
  the *shape* (linear vs isolated) is robust to reasonable size changes. A live
  before/after on one real transcript can confirm post-build.

## Build plan (on your OK)

1. `build_calc_subagent(...)` in `funhouse_agent/deep/agent.py` mirroring
   `build_references_subagent`: scoped to the calc/analysis tools, a concision
   framing that returns values + saved-artifact path + basis, a
   `ModelCallBudgetMiddleware` budget. Route heavy calc through it (primary
   dispatches to `calc` like it dispatches to `references`).
2. Wire a `calc_mode`/subagent toggle through `build_deep_agent` (default ON;
   off = today's inline behavior) — additive, default-preserving surface.
3. Summarization backstop: expose a moderate tuned trigger + `/memories` pinning
   of key facts; keep `enable_summarization` default False unless we adopt the
   backstop as default (your call).
4. Tests: offline (mock engine) proving heavy tool results do NOT appear in the
   main-thread message list, and the subagent return is compact; token-growth
   regression assertion on a synthetic session.
5. Gate + one commit per piece.

**Open questions for you:** (i) subagent ON by default, or opt-in picker (ties to
A5)? (ii) adopt the summarization backstop as default-on at ~70k, or leave it
opt-in? (iii) is the durable-checkpointer step in scope for this train or parked?
