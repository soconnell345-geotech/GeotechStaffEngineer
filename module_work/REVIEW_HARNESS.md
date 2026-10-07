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

## What was built (RELEASED in 5.30.0, 2026-09-27; extended in 5.31.0 and 5.32.0)

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
| `GEOTECH_REVIEW_FINDINGS=1` (5.31) | (lean only) the shared finding format: `record_finding`, `update_finding`, `list_findings`, `findings_report` | `funhouse_agent/review_findings.py`, `deep/findings_tools.py` |
| `GEOTECH_REVIEW_OVERVIEW=1` (5.31) | contact sheets requested in the orientation above `GEOTECH_REVIEW_OVERVIEW_PAGES` (20); with inline too, shown to the agent | `webapp/profiles.py` |
| `GEOTECH_REVIEW_GEOMETRY=1` (5.31) | (lean only) `drawing_callouts`, `drawing_dimensions`, `title_block`, `revision_clouds` from planlens.ir | `deep/geometry_tools.py` |
| `GEOTECH_REVIEW_DIGEST=1` (5.31) | (lean only) the digest's free layer: `document_inventory`, `digest_search`, `digest_pages`, `digest_references` | `funhouse_agent/review_digest/`, `deep/digest_tools.py` |

Grounded's text-layer half is belt-and-suspenders next to the resolution
fixes (image budget, tiling, zoom, "read the text layer first") and adds
nothing on sheets drawn as lines; its located-items half (page boxes) is new.
Arms added in 5.31: `overview`, `geometry`, `digest` (all include lean +
grounded). The forward plan is `module_work/REVIEW_ARCHITECTURE.md`.

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

**The forward plan is `module_work/REVIEW_ARCHITECTURE.md`** (2026-09-28): three
job shapes (small review / large single-scope review with a digest /
mega-reviews by discipline), shape-1 refinements (geometry tools on the review
page, contact sheets standard above ~20 pages, look policy, the finding
format), and milestones M0-M4.

**M0 is DONE** (runs 1–4 below): every arm measured on Sol and GPT-5.4; the
5.32.0 decision was to keep every switch off. **The open to-do list is
`HANDOFF.md` §0a-current ("Open to-dos after 5.32.0")** — one list for the
whole project. The vision call's box precision was measured on 2026-10-07
(run 5 below; `module_work/harness_theory/locating_things_on_a_page.md`),
and the location fix built from it awaits live check 6. The other
review-harness items on the list: the look-alike /
self-verification and box-carrying boundary found by
`module_work/harness_theory/`; an image crop/zoom tool for raster uploads;
then, as before, Phase 5 (rewrite the review prompt around the review method
and drop every incident rule the suite does not need) and the candidates that
must earn their place on the suite (a cross-reference resolver; revision
compare).

## Results ledger

_(fill in from RESULTS.md after each cluster run; do not tune to the blind set)_

**Run 1 — 2026-09-30, cluster, 5.31.0** (planlens 0.10.1, deepagents 0.7.13,
langchain 1.3.18; `funhouse-gpt-high`). Dry run: all 35 tasks resolve. Probe:
`ok: True` ("Red.") — the endpoint takes an image mid-conversation, so the
`inline` arm is live. Eight-task cost check, baseline vs lean:

| arm | tasks | checks | model calls | tokens in/out | ~$ (2.50/15.00 per M) | min | errors | step caps |
|---|---|---|---|---|---|---|---|---|
| baseline | 8/8 | 28/28 | 60 | 510,863 / 9,205 | 1.42 | 3.4 | 0 | 0 |
| lean | 7/8 | 27/28 | 52 | 357,696 / 10,903 | 1.06 | 3.3 | 0 | 0 |

About $0.15 and 25 s a run on these documents. The Funhouse budget moved
$2.47 against $2.48 from the token counts, so the suite's tokens capture the
whole cost, vision side calls included. Lean read 30 % fewer input
tokens. Its one miss: `set-3600-psi` (count over the 10-sheet stroke-lettered
set) found 2 of 3 sheets, missing 10.25A — a coverage question, which is what
the `sweep` arm is for; one run, not yet a pattern. Baseline's `set-3600-psi`
used 17 model calls, right at its measured ceiling of about 17 — the bigger
tasks will hit step caps there.

**Run 2 — 2026-10-01, Palantir Foundry, run by the AI FDE** (lightweight
Python transform; same package versions; `webapp/palantir_sdk_engine.py`
with the image leg as a glue file, `from_handles`). Image check `ok: True` on
both models. Foundry's `ImageDetail` has AUTO/HIGH/LOW/UNKNOWN and NO
ORIGINAL, so `original` goes as HIGH. GPT-5.4 there is served by a Bedrock
backend that refuses chat-completion requests (404 LanguageModelNotAvailable);
the FDE wrote a Responses-route wrapper. GPT-5.4 is limited to 55
requests/min per project.

| model / arm | tasks | checks | model calls | tool calls | tokens in/out | min | errors | in-run model errors |
|---|---|---|---|---|---|---|---|---|
| GPT-5.4 baseline | 8/8 | 28/28 | 99 | 44 | 543,651 / 16,264 | 5.0 | 0 | 7 (rate limit, one task) |
| GPT-5.4 lean | 8/8 | 28/28 | 115 | 46 | 515,508 / 17,451 | 4.9 | 0 | 0 |
| GPT-5.6 Sol baseline | 8/8 | 28/28 | 235 | 82 | 882,824 / 138,351 | 14.1 | 0 | 6 (dropped connections) |
| GPT-5.6 Sol lean | 8/8 | 28/28 | 267 | 71 | 734,056 / 138,362 | 12.2 | 0 | 3 (dropped connections) |

Model calls run 1.7-2.2x the Funhouse run on the same tool calls. Likely
cause (to confirm from the `tiling` fields): with no ORIGINAL the vision probe
measures a smaller image budget, so `tiles="auto"` splits drawing sheets into
more tiles, and each tile is a model call. Arm-vs-arm comparisons on Foundry
stay fair; absolute call counts and time do not transfer to a host that honours
`original`. In-run model errors are in `run.json` (`model_errors`); since the
release after 5.31.0 RESULTS.md shows them as a "failed calls" column beside
"errors".

**Run 3 — 2026-10-01/02, Foundry, GPT-5.6 Sol, the full suite** (7 arms × 35
tasks, plus grounded vs overview on the 10 UFC tasks with `orientation=True`).
Files: `module_work/review_eval_results/2026-10-02_foundry_5.31.0/` (RESULTS
as run and RESCORED, the run/vision/retry tables; internal URLs stripped).
The FDE's retry layer absorbed 106 dropped connections / rate limits /
timeouts with none given up; 2 failed calls remained inside finished runs.
The vision probe measured Sol on this route as `gpt-4.1-high` (a TILE budget,
~768 px short side): 469 of 705 page looks were split 3×3 or 4×4.

TWO CHECK FAULTS, found reading the failures and fixed in 5.32 (then every
run rescored from its saved answer and files, no model calls):
`produce-markup` required "marked" in the file name the question never asked
for (3 arms named it otherwise and passed everything else); plain-text terms
did not match a hyphenated compound ("Edge-of-pavement elevation", 4 arms).

| arm (Sol, rescored) | tasks | in / out tokens | minutes | vs baseline |
|---|---|---|---|---|
| baseline | **35/35** | 3.62M / 0.63M | 49 | — |
| lean | 34/35 | 3.61M / 0.91M | 86 | breaks bioretention-section-dims |
| grounded | 34/35 | 4.31M / 1.33M | 75 | breaks calc-bearing-consistency |
| inline | 32/35 | 5.08M / 1.13M | 54 | breaks 3 locate tasks |
| sweep | **35/35** | 4.85M / 1.20M | 86 | — |
| geometry | **35/35** | 5.17M / 1.44M | 70 | — |
| digest | 34/35 | 5.85M / 1.70M | 89 | breaks ramp-detail-callouts (read 3/4 as 34) |
| UFC: grounded / overview | 10/10 vs 9/10 | 1.38M vs 1.49M | 12 vs 9 | overview breaks ufc04-table-5-1 |
| GPT-5.4 parity, 8 tasks: baseline / lean | 8/8, 8/8 | — | — | — |

**Reading.** On GPT-5.6 Sol the suite is SATURATED: the page as released
(baseline, every switch off) passes all 35 and is the cheapest and fastest.
No switch can show a gain here; single-task differences are within one run's
noise. The signals that do stand: `inline` (the main model looking at images
itself) is worse on locating small things (3 of 15 locate tasks lost) — keep
it off; `overview` buys nothing — keep it off. The suite does NOT contain the
failure the field session showed (a long drawn-lettering set where unlooked-at
sheets were missed; markups at invented places), so it cannot decide the
coverage settings. 5.32 adds two tasks that do — `set-long-rare-tag` (which
of 24 drawn-lettering sheets carry a rare FPG callout; its look-alike FBG is
on every sheet) and `produce-circle-tags` (rings scored on their targets) —
for the 5.32 Foundry run: baseline vs `sweep`, from local wheels, before any
publishing.

**Run 4 — 2026-10-02/04, Foundry, GPT-5.6 Sol at FULL resolution (Responses
route), 5.32.0rc2/rc3, 37 tasks.** Files and the full reading:
`module_work/review_eval_results/2026-10-04_foundry_5.32.0rc3/` (NOTES.md,
RESULTS_main / _repeats / _resolution). Original 35 ran on rc2 and were
rescored on rc3; the two new tasks ran 3 times per arm on rc3 with
`find_like` hidden (OpenCV aborts the process on that host). Three more check
faults found and fixed (LaTeX numbers, "pages:" and bulleted page lists).

| arm | 35 originals | 2 new tasks × 3 | input tokens (orig / new) | minutes (orig / new) |
|---|---|---|---|---|
| baseline | **35/35** | 5/6 | 4.53M / 2.38M | 45.5 / 30.5 |
| sweep | 35/35 | 6/6 | 10.81M / 2.50M | 124.6 / 11.1 |
| minimal | 33/35 | 4/6 | 18.45M / 9.46M | 67.6 / 34.2 |
| baseline at HIGH (~768 px) | 15/15 drawn-lettering | 1/2 | 0.88M / 0.70M | 18.6 / 20.8 |

**Reading — and the 5.32.0 decisions.** Switches stay OFF: sweep ties
baseline at 2.4x the tokens and 2.7x the time, and its extra new-task pass is
the markup case, not coverage. `minimal` (looking only) loses what the text
and markup tools carry (a whole-set spec search; who wrote a reply) at ~4x
the tokens. Full size kept: HIGH added false pages on the 24-sheet task. The
one baseline miss: told six times its rings were misplaced, the agent WIDENED
them until the vision check confirmed (70–125 pt rings round 10 pt tags) —
the check now also asks whether a mark is drawn close (unverified live:
`module_work/LIVE_TEST_QUEUE.md` item 1). Page 19 (FBG read as FPG) is named
in most runs: a look-alike the reader "verifies" with the same kind of call
that misread it (`module_work/harness_theory/document_review.md`).

**Run 5 — 2026-10-07, Funhouse (live checks 1 and 4, run by the owner),
GPT-5.4.** These were the two markup tasks × 3, on 5.32.0 and again on the
5.32.1rc1 test wheel. Every run was read in full:
`module_work/review_eval_results/2026-10-07_funhouse_5.32.0_check1/`
(TRACE_REVIEW.md, RESULTS.md) and
`module_work/harness_theory/locating_things_on_a_page.md`.

| wheel | `produce-markup` | `produce-circle-tags` (tags marked of 7) | tokens in per run | minutes per run |
|---|---|---|---|---|
| 5.32.0 | 3/3 | 0/3 (0, 1, 2) | 198k–297k | 2.0–3.4 |
| 5.32.1rc1 | 3/3 | 0/3 (0, 5, 0) | 158k–359k | 1.6–3.8 |

**Reading.** There were two causes.
- **5.32.0's "drawn close" question was too strict on GPT-5.4.** It
  rejected correct rings. The size is now measured instead (`cc697e6`), and
  rc1 confirmed that check behaves.
- **Circles were placed from whole-page looks.** Live check 4 measured this
  directly: GPT-5.4's 0-999 boxes on a whole sheet were 57–91 pt off, while
  its pixel boxes, converted with the true image size, were 1–6 pt off.
  Funhouse also caps images at 2,048 px.

The location fix (app `cf43c90`, planlens `08a1d53`) asks for pixel boxes,
caps renders at 2,048 px, and refuses small marks read off wide views. It is
unreleased; live check 6 re-measures it.
