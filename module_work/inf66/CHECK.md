# INF-66: the prompt-cache split in `activity.jsonl`

Two parts. The code part of 2026-11-05 (branch `claude/inf-66-cache-tokens`,
kept below as it was written; the file it names, `webapp/activity_log.py`,
is `packages/aec-agent-core/aec_agent_core/activity_log.py` since the app
split) records the split. The part of 2026-11-18 (branch
`claude/inf-66-cache-split`) makes the record tell a reported count from
none. The live check is the maintainer's: "The check on the hosted
platform", below.

## 2026-11-18: a reported count told from none

**What was asked.** Where a reply reports how many input tokens were read
from the prompt cache and how many written to it, the record keeps them per
call and the turn sums them, for every route the core has; a reply that does
not report them must not look as if it did ("not reported", never a guess).
Nothing the model or the user is handed changes.

**What was wrong.** A reply with no cache count recorded `cache_read: 0` and
`cache_creation: 0`, the same as a reply that reported zeros. So on the
check below, a call with `cache_read: 0` could mean "the cache was not read"
or "the proxy's reply did not say", and the two call for different repairs.
Two readers also blurred the line:

- the provider usage dict that langchain-anthropic keeps in
  `response_metadata` has `cache_read_input_tokens: None` when the reply had
  no cache fields, and the record read the None as a reported 0
  (`_cache_from_raw`); an OpenAI-style `cached_tokens: None` the same;
- the Foundry Claude SDK model (`palantir_sdk_engine._claude_usage`)
  dropped a reported 0 (it set `input_token_details` only when a count was
  above 0), so a reported zero looked like no count.

**What changed** (`packages/aec-agent-core/aec_agent_core/`):

- `activity_log.py`:
  - `_cache_from_details` and `_cache_from_raw` treat a None count as not
    reported; a None count near the message no longer stops the search for a
    real count further out (the `llm_output` usage).
  - `_usage_from_response` returns `(usage, cache_reported)`; its one
    caller, `ActivityLogger.on_llm_end`, writes `cache_reported: true` on
    the `model_end` beside `usage` when the reply carried a cache count (a
    reported 0 included), and nothing when it did not.
  - `turn_end` gets `cache_reported_calls` (how many of the turn's calls
    carried a count) when at least one did (`_add_usage`, `_turn_facts`).
  - `usage` itself is unchanged on every call and every turn: the same keys
    and the same numbers as before (the app's own tests pin both shapes).
- `palantir_sdk_engine._claude_usage`: sets `input_token_details` whenever
  the SDK's usage has a cache field, 0 included; `input_tokens` unchanged.

**Judgement calls.**

- The mark sits beside `usage`, not in it: the app's tests pin the exact
  `usage` dicts of reporting and non-reporting calls and of `turn_end`, and
  a turn without cache counts had to write today's record byte for byte. So
  the mark is written only where a count was reported; "no mark" reads
  "not reported", and every older record (no mark) reads the same way.
- OpenAI-style routes report a read and no write; `cache_creation` stays 0
  there with the mark on, as before (OpenAI caches without a write count).
- A Claude SDK reply with only one of the two counts keeps the other as 0
  (the provider sends both or neither).
- The run report is not changed: it already shows "(N from cache)" on a call
  that read the cache.

**Settings.** None. This is a record addition with no switch, as the count of
image calls was in 1.2.0. Nothing the model is handed moves: the Document
Review fingerprints test passes unchanged.

**Tests** (`packages/aec-agent-core/tests/test_inf66_cache_reported.py`,
made-up usage):

- a turn of four calls whose replies carry no cache count (LangChain usage
  alone; langchain-anthropic's dict with the fields None; an OpenAI-style
  `cached_tokens: None`; `llm_output` alone): every `model_end` line equals,
  byte for byte, the line the record wrote before, and `turn_end` has the
  same `usage` and no `cache_reported_calls`. This test also passes on the
  code before this change (run with the two code files set back);
- a reported zero through langchain-anthropic's own builder is marked; the
  same builder with the fields None (a proxy that drops them) is not;
- the Claude SDK converter keeps a reported 0 and leaves out an absent one;
- a None count near the message does not hide a real count in `llm_output`;
- a turn of a write, a read and an unreported call: the sums as before and
  `cache_reported_calls: 2`.
- Also run: `webapp/tests/test_activity_cache_tokens.py`,
  `test_activity_log.py`, `test_palantir_claude_engine.py`,
  `test_palantir_sdk_engine.py`, the run report's tests and
  `packages/aec-document-review/tests/test_review_fingerprints.py`.

**Noted, not fixed.** The Prompter route that reports a combined total only
(no input/output split) reports no cache count, so it is never marked; that
is the reply, not the record.

## The check on the hosted platform (the maintainer's)

Run on the release that carries this change (the next aec-agent-core
release; the mark does not exist before it).

1. Leave prompt caching on (`GEOTECH_FOUNDRY_PROMPT_CACHE` unset; `0` or
   `off` sends no cache marks). Open a new conversation on a Claude model by
   the Anthropic proxy route and ask a one-method question that takes at
   least three model calls (the agent calls a tool, reads it, answers).
   Repeat on a GPT model if one is in use.
2. Read that conversation's record from its folder (the one holding
   `activity.jsonl`):

   ```python
   from aec_agent_core import activity_log as al
   for r in al.load("<the conversation's folder>"):
       if r.get("event") == "model_end":
           u = r.get("usage") or {}
           print(r.get("agent"), u.get("input_tokens"), u.get("cache_read"),
                 u.get("cache_creation"), r.get("cache_reported", False))
       elif r.get("event") == "turn_end":
           print("turn", r.get("usage"), r.get("cache_reported_calls", 0))
   ```

3. What it should show, and what each outcome means:

   | What the record shows | Meaning | Next |
   |---|---|---|
   | every call `True`; call 1 `cache_creation` well above 0 (about the shared prefix: system prompt and tools); calls 2 on `cache_read` near the previous call's prefix | the proxy reads the cache | cost the turn at the cache rates; INF-66 closes |
   | every call `True`, but `cache_read` 0 on calls 2 on (and, on Claude, `cache_creation` 0 on call 1) | the provider says nothing was cached: the marks did not reach it, or the prefix is under the provider's minimum cacheable length | a repair in the request (the core's `engine_config` / `palantir_sdk_engine`), the developer's |
   | no call `True` | the proxy's replies carry no cache count; the record cannot tell | ask the hosted platform whether its proxy passes the provider's cache counts through, or read the conversation's cost in its own usage view |
   | some calls `True`, some not | the calls went by different routes (a helper on another model, the Responses route) | name the route of each unmarked call |

   On a GPT model `cache_creation` stays 0 with the mark on: OpenAI reports
   a read only. The run report's line for a call that read the cache says
   "N tokens in (M from cache)".

## The code part of 2026-11-05 (as written then)

### What changed

`webapp/activity_log.py`:

- **Each `model_end`'s `usage`** now has two more keys, `cache_read` and
  `cache_creation`. These are the names LangChain's `input_token_details`
  uses, and the ones report ingest reads (`report_ingest.engine._usage_of`, RI-171).
  - Both are **part of** `input_tokens`, not added to it. `input_tokens` is
    still LangChain's count, with cache reads and writes included, so
    `turn_tokens` and every existing total read as before.
  - **No cache fields in the response:** both are recorded as `0`, as report
    ingest's `Usage` defaults them. A response with no usage at all still
    records `usage: null`.
- **Where the counts come from**, in this order:
  1. `usage_metadata.input_token_details`. Every LangChain route sets it:
     - langchain-anthropic (the Foundry proxy route, direct Claude). It also
       reads the per-TTL write keys `ephemeral_5m_input_tokens` and
       `ephemeral_1h_input_tokens`. langchain-anthropic sets `cache_creation`
       to 0 when it uses those keys, so all three are added.
     - langchain-openai (the Prompter route on Tiny Apps, Foundry OpenAI). It
       may put a service-tier prefix on the keys, e.g. `priority_cache_read`.
     - the Foundry Claude SDK model (`palantir_sdk_engine._claude_usage`).
     - the Prompter on Databricks/Funhouse and Foundry GPT named by API
       name (since the review fixes below).
  2. The provider's own usage dict, found in the message's
     `response_metadata`, the generation's `generation_info` or
     `llm_output`:
     - Anthropic: `cache_read_input_tokens` and `cache_creation_input_tokens`;
     - OpenAI: `prompt_tokens_details.cached_tokens`. This is a read; OpenAI
       reports no write, so `cache_creation` is 0.
- **`turn_end` now has `usage`:** `input_tokens`, `output_tokens`,
  `cache_read` and `cache_creation`, each summed over every `model_end` of the
  turn. That covers the primary, its helpers and side calls, the same calls
  `turn_tokens` counts. It also has `calls` (the number of calls that
  reported usage) and, if any call reported none, `unmetered_calls`. A turn
  with no model call has no `usage` key.
- **Readers:**
  - `activity_log.load` is unchanged, so old files load.
  - `funhouse_agent/review_eval/runner._activity` already leaves out keys
    containing "cache" from its input total, so it does not count cache
    tokens twice.
  - `module_work/live_smoke/arms.py` reads only `input_tokens` and
    `output_tokens`.
  - None of these needs the new keys.

### Tests

- `webapp/tests/test_activity_cache_tokens.py` (new, 14 tests, made-up usage):
  - one test per route: langchain-anthropic's own builder, its per-TTL writes,
    langchain-openai's own builder, a service-tier prefix, the Foundry Claude
    SDK converter, the Prompter model (generate and stream) and the Foundry
    GPT SDK model (chat and Responses routes);
  - the raw-usage fallbacks: Anthropic in `response_metadata`, both in
    `llm_output`;
  - a response with no cache fields records 0, and one with no usage records
    None;
  - **a turn summed over three calls**: one cache write, then two reads, the
    third call inside a helper;
  - the unmetered count, and a turn with no call;
  - an old activity file loads, and the review-suite reader's totals are
    unchanged.
- `webapp/tests/test_activity_log.py`: the one exact-dict assertion now
  includes the two zeros.
- Run in a clean venv:
  - the two files: 25 passed;
  - `webapp/tests`, `funhouse_agent/review_eval` and `module_work/live_smoke`:
    1,275 passed and 11 failed. The 11 also fail on master in that venv,
    because the geotech-references submodule and plotly are missing there
    (page fingerprints, diagnostics' reference docs, card colours, the review
    boundary).
  - `test_public_text.py`: passed.

### Review fixes

The review found two OpenAI-style routes the app builds itself that dropped
the provider's cached count, so they recorded `cache_read: 0` after a read.

- **Prompter on Databricks/Funhouse** (`PrompterChatModel`, generate and
  stream) and **Foundry GPT named by API name, chat route**
  (`PalantirSdkChatModel`): the bridge's `_to_usage_metadata` now copies
  `prompt_tokens_details.cached_tokens` into
  `usage_metadata.input_token_details = {"cache_read": n}`, as
  langchain-openai does. `model_end` records it as `cache_read`;
  `cache_creation` stays 0 (OpenAI reports no write).
- **Foundry GPT named by API name, Responses route**: a cached count in the
  SDK's `usage.input_tokens_details.cached_tokens` is passed on the same
  way. Whether the SDK reports one is not yet known; if it does not, the
  record says 0.
- `input_tokens` is unchanged on every route: the cached count is part of
  it. Report ingest's `ChatModelEngine` on these routes now splits the read
  out of its input, as `PrompterEngine` already does; the sum is the same.
- Tests: the old test that claimed the bridge was covered (it fed the logger
  a dict the bridge never makes) is replaced by three that drive the real
  models with made-up usage: the Prompter's generate path with an openai
  `ChatCompletion`, its stream path, and both Foundry GPT routes over the
  stand-in SDK of `test_palantir_sdk_engine.py`.
  - Run in the app venv: the two activity-log files, 27 passed;
    `funhouse_agent/deep/tests/test_deep_token_tracking_offline.py` and
    `webapp/tests/test_palantir_sdk_engine.py`, 40 passed.
  - The three new tests fail on the code before these fixes
    (`cache_read` 0).
- The branch's BACKLOG.md edit is dropped; the item's state is written when
  it merges.

### To measure live (superseded by the check above)

The next Foundry geotech turn shaped like build i's (a one-method question,
four model calls, caching on: `GEOTECH_FOUNDRY_PROMPT_CACHE` unset):

1. In `activity.jsonl`, read `usage.cache_read` and `usage.cache_creation` on
   each primary `model_end`. Expected:
   - call 1 writes about 33.7k tokens (`cache_creation`);
   - calls 2-4 read about 33.7k, 34.3k and 38.4k tokens (`cache_read`), each
     writing only the new tail.
2. Check that `turn_end.usage.cache_read` is about 106k of about 145k input.
3. If calls 2-4 show `cache_read: 0`:
   - check whether the proxy dropped the marks or the response's usage has no
     cache fields (all three keys at 0 on every call, including a write on
     call 1, points at the response);
   - read the raw response usage through the stand-in proxy test
     (`webapp/tests/test_foundry_model_family.py`) against a real Foundry
     response.
4. If the cache is read, cost the turn at the cache rates (about $0.23
   against $0.60 with no read; build i TRACE_REVIEW section 2.3) and close
   INF-66.
