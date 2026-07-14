# A7 — Industry patterns vs. ours: how production LLM-agent apps are built

**Status:** research memo for owner/lead discussion (APP_PLAN.md A7). **Recommendations
only — nothing here is built without your go-ahead; rearchitecture items are flagged
explicitly for discussion, not action.** Author: app-dev, 2026-07-14.

## TL;DR

Our stack is the **mainstream 2026 shape** — a LangGraph/deepagents agent over a
typed tool registry, a Streamlit UI, dependency-free persist+replay for session
state, and an offline eval suite. The lead's initial take holds up against the
literature. The two consensus practices we were **missing** are (1) **context
isolation / compaction** — which **A2 now addresses** (the calc sub-agent shipped
this train; summarization backstop pending a store decision) — and (2)
**lightweight run tracing / observability**, which is our clearest, cheapest gap.
Streamlit remains correct for single-user + TinyApp; a FastAPI+React/Chainlit move
is only warranted if multi-user is ever needed.

## How production agent apps are built (2026 consensus)

| Dimension | Industry consensus | What we do | Verdict |
|---|---|---|---|
| **Orchestration** | ReAct + tool-use + planning + reflection loop, often on LangGraph | deepagents (LangGraph) primary + planning (`write_todos`) + scratch FS | **On-pattern** |
| **Context mgmt** | Keep the model context *ephemeral*; write intermediate data to a file/DB; pass only **compressed summaries** back. Compaction/summarization when near the window. | Was: full persist+replay (grows linearly). **Now: calc sub-agent** (A2) isolates the tool-heavy trace + saves full payload to a file, returns a compact result. Summarization backstop pending. | **Closing the gap (A2)** |
| **Sub-agents** | Isolate focused work in a sub-agent with a *clean* context + **strict tool scoping**; lead keeps only high-level view. ~90% gains reported on research tasks. | references consultant + F8 reviewers + **new calc sub-agent**, all scoped via `allowed_agents` + call-budget middleware | **On-pattern** |
| **Streaming UX** | Token streaming + tool-activity surfacing + interim progress | app streams tokens + tool/status lines + token counts; A3 crash-safe partials | **On-pattern** |
| **Session / file store** | Durable session store (often a checkpointer/DB) + a file store for artifacts | persist+replay (dep-free) + per-conversation `files/`; durable checkpointer **parked** | **Adequate single-user; see rec 4** |
| **Observability / tracing** | Per-run traces (spans for model/tool/subagent), token/latency/error metrics, eval-on-traces. Tools: **LangSmith** (LangGraph-native, ~0 overhead), **Langfuse**/**Phoenix**/**MLflow** (OSS, OpenTelemetry GenAI conventions emerging) | **None** — we read token counts per turn, no trace store | **Clearest gap (rec 1)** |
| **Evals** | Offline **golden dataset in CI** (reproducible regression catch) + online production scoring + a failures→dataset feedback loop | 100-Q keyed offline suite (`funhouse_agent/deep/eval_harness.py`), owner-gated live; 8k+ unit gate | **Good offline; add a CI subset (rec 3)** |
| **HITL** | Approval gates for consequential/irreversible actions; human review queues | `geo_project` staged human-gated model setup; layered disclaimers; reviewer family | **On-pattern for our risk profile** |
| **Model routing** | Router = highest-ROI pattern: classify → cheapest capable model | manual in-app model picker (Opus/Sonnet/Haiku) | **Fine for expert single-user (rec 5)** |
| **UI** | Streamlit for single-user/internal; FastAPI+React or Chainlit for multi-user | Streamlit | **Correct until multi-user (rec 6)** |

## Recommendations (ranked by value / effort)

1. **Lightweight run tracing — HIGH value, LOW effort, additive.** Wire *optional*
   tracing behind an env var: **LangSmith** is LangGraph-native, opt-in via
   `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY` with near-zero overhead and no code
   change; **Langfuse** or **Arize Phoenix** are OSS/self-host (OpenTelemetry) if we
   want to avoid a SaaS. Gives per-turn token/latency/tool-trace visibility, makes
   the A2 savings *measurable*, and enables eval-on-traces. No default dependency;
   off unless the env var is set. **Recommend building** (small, reversible).
2. **Context isolation + compaction — HIGH value, MED effort — IN PROGRESS (A2).**
   The calc sub-agent (shipped this train) is the isolation half. The compaction
   half (summarization backstop) is pending your store decision (see A2 report /
   rec 4). This is the single most-cited practice we were missing; A2 is the right
   response.
3. **A CI-runnable eval subset — MED value, LOW-MED effort.** We have the 100-Q
   golden set but run it owner-gated/live. Add a small **offline/mock subset** that
   runs in the normal gate to catch behavioral regressions reproducibly (industry:
   golden-set-in-CI). Keep the full live run owner-gated for cost.
4. **Durable checkpointer / drop-replay — MED value, MED effort — PARKED, needs
   discussion.** Industry uses durable session stores. Our persist+replay is fine
   for single-user but (a) re-sends full history each turn and (b) blocks true
   cross-turn compaction and `/memories` fact-pinning (this is exactly what stalls
   A2's summarization backstop). A LangGraph SQLite checkpointer would unlock both.
   **Rearchitecture-adjacent — discuss before building.**
5. **Auto model-routing — LOW/MED value for us.** The top industry ROI pattern, but
   aimed at high-volume cost control. For a single expert user who picks the model
   deliberately, low priority. **Report, don't build.**
6. **Multi-user UI (FastAPI+React / Chainlit) — N/A now.** Only relevant if the app
   goes multi-user/hosted beyond TinyApp's single-user model. **Discuss only if that
   need appears.**

## Bottom line

No rearchitecture is warranted now. The mainstream-shape assessment is confirmed.
Do rec 1 (tracing) as the cheap high-value win, finish rec 2 (A2), and add rec 3
(CI eval subset) when convenient. Recs 4–6 are owner-discussion items, not build
items — flagged here so the decision is explicit rather than drifted-into.

## Sources

- [Effective context engineering for AI agents — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [How to orchestrate autonomous sub-agents without blowing your context window — DEV](https://dev.to/programmingcentral/how-to-orchestrate-autonomous-sub-agents-without-blowing-your-llm-context-window-jpo)
- [Agentic AI Architecture: 2026 Production Patterns + Stack — Internative](https://internative.net/insights/blog/agentic-ai-architecture-2026)
- [Agent Observability: Monitor & Evaluate LLM Agents in Production — LangChain](https://www.langchain.com/blog/production-monitoring)
- [15 AI Agent Observability Tools in 2026 — AIMultiple](https://aimultiple.com/agentic-monitoring)
- [Top 5 LLM and Agent Observability Tools in 2026 — MLflow](https://mlflow.org/top-5-agent-observability-tools/)
- [AI Agent Evaluation (2026): Metrics, Frameworks, Production Failures — MorphLLM](https://www.morphllm.com/ai-agent-evaluation)
- [8 best human-in-the-loop LLM evaluation platforms in 2026 — Braintrust](https://www.braintrust.dev/articles/best-human-in-the-loop-llm-evaluation-platforms-2026)
