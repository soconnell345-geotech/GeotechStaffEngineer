# INF-66: the prompt-cache split in `activity.jsonl`

Branch `claude/inf-66-cache-tokens`. Code part only; the live check is below.

## What changed

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
  2. The provider's own usage dict, found in the message's
     `response_metadata`, the generation's `generation_info` (where the
     Databricks bridge puts the Prompter's usage) or `llm_output`:
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

## Tests

- `webapp/tests/test_activity_cache_tokens.py` (new, 12 tests, made-up usage):
  - one test per route: langchain-anthropic's own builder, its per-TTL writes,
    langchain-openai's own builder, a service-tier prefix, the Foundry Claude
    SDK converter;
  - the raw-usage fallbacks: OpenAI in `generation_info`, Anthropic in
    `response_metadata`, both in `llm_output`;
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

## To measure live

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
