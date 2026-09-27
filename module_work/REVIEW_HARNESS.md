# Document Review harness — plan of record (2026-09-26)

Owner's ask: "take a big picture view of the harness workflow … the Claude Code
sessions have been overfitting for more narrow use cases." The full review and
approved plan: `C:/Users/socon/.claude/plans/we-ve-been-working-on-synchronous-lemon.md`.

## What the review found (measured offline, 2026-09-26)

1. **No evaluation set.** Every release from 5.27 to 5.29.1 answered one tester
   transcript; nothing could show a fix that helps one question and breaks three.
2. **The reasoning model never sees a page.** Every look is a separate one-shot
   vision call with only the image and the agent's prompt (no conversation, no
   text layer, no legend). The agent reasons from what that call chose to write.
3. **The page's agent was the geotech builder with modules subtracted.** 29
   tools / ~38K chars of schema, ~22K of it deepagents' generic tools (`task`
   7.6K, `write_todos` 4.4K, a shell `execute` that cannot work here, a scratch
   filesystem); two geotech chart tools that cannot work on this page;
   descriptions pointing at drawing_ir tools the page lacks; ~7.7K of the 14K
   system prompt was deepagents' coding-agent prompt; the general-purpose
   helper had a 286-character prompt and none of the review rules.
4. **Whole-document questions exceed a turn.** 17 model calls fit under the
   default step cap of 50, then a GraphRecursionError — no answer.
5. **Incident rules accreting** ("[G/Q]CE" in every vision prompt, etc.), and a
   prompt about tool mechanics rather than how a reviewer works.
6. **Citations one page early**: tools number pages from 0, nothing said +1.

## What was built (UNRELEASED — on top of master after 5.29.1)

Everything that changes behaviour is behind a switch in
`funhouse_agent/review_flags.py`, OFF by default. With every switch off the page
is what it was, except the unswitched fixes at the end.

| Switch | What it does | Code |
|---|---|---|
| `GEOTECH_REVIEW_AGENT=lean` | The page builds its own agent: 19 tools / ~16.6K chars (read, look, write only), the review prompt without the scratch filesystem, no coding-agent prompt, a `task` → `page_reader` helper with the SAME reading rules and no writing tools, a 40-call budget (`GEOTECH_REVIEW_MAX_MODEL_CALLS`) whose last call answers from what was gathered — never a GraphRecursionError (verified on deepagents 0.6.8 and 0.7.13) | `funhouse_agent/deep/review_agent.py`; routed by `build_deep_agent(review_page=True)` |
| `GEOTECH_VISION_TEXT_CONTEXT=1` | Each page/region vision call is told the PDF's own text-layer lines inside its view (exact strings + 0-999 boxes), or that there are none | `vision_view.page_lines/text_context`, `vision_tools._vision_prompt` |
| `GEOTECH_VISION_STRUCTURED=1` | Vision calls end with `LOCATED: [...]`; results carry `located` items with `page_bbox` in PDF points (pass straight to `render_region`) | `vision_view.split_located` |
| `GEOTECH_VISION_INLINE=1` | (lean only) the main model looks: page/region tools store the image and `InlineImageMiddleware` shows the newest 2 to the model at its next call (request only, never saved) | `funhouse_agent/inline_store.py`, `funhouse_agent/deep/inline_images.py` |
| `GEOTECH_REVIEW_SWEEP=1` | (lean only) `sweep_pages`: one question asked of every page in a range, in parallel, per-page answers with citations — the GENERAL path for every/all/count questions | `funhouse_agent/deep/sweep.py` |

**Unswitched fixes** (legacy page too): results carry `pdf_page` (= page + 1)
beside every `page` and `[pdf_page N]` in `=== page N` headers
(`document_tools.with_viewer_pages`; planlens budget margin 200 → 1500 to hold
them); the review prompt's citation rule (cite sheet / printed page / viewer
page, never the 0-based index); `[G/Q]CE` replaced by a neutral example; tile
reads now carry the run's context (their model calls were invisible to the
activity log and token count); planlens `advice.py` no longer names
"drawing-geometry tools" (planlens branch `fix/advice-names-no-missing-tools`,
Unreleased in its CHANGELOG — needs a planlens release; the app does not
depend on it).

## The suite — `funhouse_agent/review_eval/`

29 open tasks over 16 documents, 8 document types, 8 kinds of question. Truth
for each is recorded in the task (`truth`) and every truth note passes its own
checks (test-pinned); an empty answer fails every task.

- Ten Mecklenburg County standard-detail sheets (public; lettering drawn as
  lines, NO text layer) — truth from the DWG text. Also bound as one 10-page set
  for sheet index / cross-reference / count / memo tasks.
- UFC 3-220-04FA (60 pp; its Table 5-1 is an image), UFC 3-220-07 (99 pp; 33
  image-only pages), UFC 3-301-01 (228 pp).
- Two calc packages (a failing retaining wall with an unsupported summary
  value; a consistent bearing package).
- Two SYNTHETIC planlens fixtures only (review markups; a duplicated page) —
  kept small on purpose: the tools were built against them.

Checks are deterministic (terms with alternatives and guarded regexes, sheet
set recall/precision, citations, produced .pdf/.docx read back). Auto-checks on
every task: never "too blurry / send a better file", never "page 0".

### Owner's cells (Funhouse)

```python
# once, on a machine with the source checkout: gather the public PDFs, upload the folder
from funhouse_agent.review_eval import collect_public_docs
collect_public_docs("review_eval_docs")          # then copy to /Volumes/... or SharePoint

# on the cluster
from funhouse_agent.deep.inline_images import probe
from funhouse_agent.deep.databricks_bridge import PrompterChatModel
probe(PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high"))   # ok: True before trusting the inline arm

from funhouse_agent.review_eval import score_review_suite
res = score_review_suite(prompter=fh_prompter, model_name="funhouse-gpt-high",
                         docs_dir="/Volumes/.../review_eval_docs",
                         out_dir="/tmp/review_eval",
                         arms=("baseline", "lean", "grounded", "inline", "sweep"),
                         sharepoint=fh_sp_client)   # durable copy; a restart resumes
print(res["results_md"])
```

Start small (`ids=["meck-ramp", "ufc04-"]`, one or two arms) to see cost per
task; `dry_run=True` checks every document resolves without calling a model.

## Independent review of the build (2026-09-26) — all 9 bugs fixed, regression-tested

A reviewer sub-agent found, and experiments confirmed: a retry scored against
the failed attempt's files; custom arms dropping non-switch variables; located
lists pushing vision results past the cap (mid-JSON cuts); the budget's last
call not seeing inline images; a reading-helper error ending the turn; baseline
step-cap errors re-run (re-paid) on every resume; false passes/fails in the
checks (unit regexes, "step. 12" as a page, Oxford-comma lists, "20.00A/B",
"higher resolution", the duplicate-page task); small host caps; `run_task`
raising on a broken document. Each has a test in
`funhouse_agent/deep/tests/test_review_harness_offline.py` or
`funhouse_agent/review_eval/tests/test_review_eval_offline.py`. A
GraphRecursionError is now an OUTCOME (`outcome_error`, "step caps" column),
never retried.

Known limits left as they are: files the agent writes OUTSIDE the working
folder are not collected (the app imports them; the suite does not); on a
Claude model the budget's tool-less final call may be refused (the app runs
OpenAI); inline mode sends up to `GEOTECH_VISION_INLINE_KEEP` (default 2)
full-budget images per call — turn it to 1 if the gateway refuses a request as
too large (report_ingest hit that limit once).

Gate on this tree (foreground, exit codes): webapp + deep + suite 803 passed /
1 skipped; the rest of funhouse_agent 1,460 / 4; the new tests on the drift
stack (deepagents 0.7.13, langchain 1.3.18) 105 / 2; planlens 1,430.

## Rules going forward

- **Tester feedback becomes a TASK, not a prompt rule.** Private tasks (SBU
  documents) go in a JSON file on SharePoint: `score_review_suite(extra_tasks=[path])`.
- **A blind set** — tasks nobody tuning the harness sees — should be written by
  someone else and kept off the repo (`split: "blind"`).
- **A switch is flipped on by default only when the suite says so**: suite total
  not down and no category regressed. Record the RESULTS.md numbers here.

## Next

1. Owner: run the probe and a first small suite run on the cluster (baseline vs lean).
2. Full run of all arms; decide which switches become defaults.
3. Phase 5 (after numbers): rewrite the review prompt around the review method
   (scope, governing documents, completeness, coordination, cross-references,
   severity, a findings format); drop every incident rule the suite does not need.
4. Candidates to earn their place on the suite: a cross-reference resolver
   (detail bubbles, sheet index vs sheets present); revision compare.

## Results ledger

_(empty — fill in from RESULTS.md after each cluster run; do not tune to the blind set)_
