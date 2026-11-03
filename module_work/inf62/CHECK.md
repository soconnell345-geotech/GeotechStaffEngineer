# INF-62 check: a rate limit Foundry names in a 400 is a rate limit

Branch `wip/inf62` off master `a372646`. Evidence: `module_work/foundry_parity/FINDINGS.md`
row 1 and its "Noted" lines.

## What changed

- **One reading of the names.** `funhouse_agent/error_text.py` gains
  `rate_limit_named(exc)` and `RATE_LIMIT_NAMES` (`ratelimitsexceeded`,
  `qosexception`, matched inside the full name). It reads, on the error and
  each error it was raised from:
  - `error_name` / `_error_name` (the Foundry SDK's errors);
  - the body: `errorName`, `error_name`, `code` or `type` at the top, under
    `error`, or one level further down; `error` itself when it is the name;
    a `message` that writes the name out. This covers the whole body that
    `anthropic.BadRequestError` carries and the `error` object that the
    OpenAI SDK's carries;
  - the response's JSON, else its text;
  - the message, where the name is written out (`ErrorName: '...'`,
    `'errorName': '...'`, `"errorName":"..."`).

  Only the name counts. A 400 that names something else, or carries the word
  only in someone's text, is not a rate limit. The helper never raises.
- **Every route asks the helper.**
  - `vision_engine.busy_kind` returns `rate limit` for a named 400. It serves
    the agents' busy ladder (`AgentBusyRetry`: the primary and every helper,
    on every engine), the vision side calls' ladder, and the turn-error text.
  - `webapp/core._busy_kind`'s fallback asks the helper too.
  - `report_ingest/engine.is_rate_limit` replaces its own `error_name` check
    with the helper. It now also reads the body form, which it missed before:
    the message is read only up to the body, and the body fields it read were
    `message` / `code` / `type`.
  - `webapp/palantir_sdk_engine.transient_kind` replaces `_RATE_LIMIT_ERRORS`
    (now removed) with the helper. Any QoS exception counts, and so does a
    name written out in the message.
- **A spent allowance still wins.** `budget_exhausted` is still asked first
  everywhere. A named 400 whose message says the usage limit is reached gets
  the one plain budget line and is never retried (tested).
- **Foundry wording.**
  - `webapp/diagnostics.py`: with `GEOTECH_REFERENCES_DOCS` unset on Foundry,
    the reference-PDF row names the `references/` folder beside the app file,
    the variable, and FOUNDRY.md §3d. It does not mention SharePoint,
    `stage_sharepoint` or `run_on_databricks`. Funhouse's message is
    unchanged.
  - `docs/FOUNDRY.md` §3: the `GEOTECH_WEBAPP_DATA` row no longer says that
    saved conversations survive restarts. In a published app, the folder is
    inside the container and is lost when the container is rebuilt (Notes;
    INF-38).
  - `docs/FOUNDRY.md` Notes: the busy ladder's list now includes the named
    400.
- `webapp/app.py` is untouched, as the brief asked.

## What a live Foundry turn now does

A 400 named `RateLimitsExceeded` or a QoS exception can come from:

- the Claude proxy;
- the OpenAI proxy;
- the SDK route;
- a vision side call.

The agent now waits it out on the busy ladder: the provider's retry-after,
else 15/30/60/120 s, 5 tries in all, inside the turn's 900 s busy bound.
Vision side calls use their own ladder, with 3 tries. Before this change, the
turn stopped on the raw 400.

If the ladder runs out, the turn says: "The AI model's rate limit was reached
... The app asked 5 times over N minutes". The raw text follows as details.

## Tests (all MADE-UP error bodies; exit codes from pytest)

| Run | Result | Exit |
|---|---|---|
| `funhouse_agent/deep/tests/test_foundry_rate_limit_offline.py` (new; real `anthropic.BadRequestError` / `openai.BadRequestError` with JSON bodies, conjure-style, response-JSON, wrapper, other 400s, spent allowance, the ladder through a compiled deep agent, the turn text) | 23 passed | 0 |
| the same file with the helper switched off (shows the defect) | 14 failed, 9 passed | 1 |
| `report_ingest/tests/test_ri139.py`, `webapp/tests/test_palantir_sdk_engine.py`, `webapp/tests/test_diagnostics.py` (+ the new file, first run) | 124 passed, 1 failed (my own over-strict assertion, removed: see noted) | 1, then 0 on re-run |
| `report_ingest/tests/test_chat_model_engine.py test_engine.py test_prompter_engine.py test_ri190.py`, `module_work/report_ingest_harness/tests/test_run_claude_local.py` | 267 passed | 0 |
| `funhouse_agent/deep/tests` first 36 files | 555 passed, 1 skipped | 0 |
| `funhouse_agent/deep/tests` other 35 files | 619 passed, 1 skipped | 0 |
| `webapp/tests/test_core.py` + `test_foundry_engine/mode/model_family/route.py` | 180 passed | 0 |
| `webapp/tests/test_auth_refusal, test_budget_panel, test_kept_turn_work, test_palantir_claude_engine, test_wave2b_turns, test_wave2c_fixes`, `funhouse_agent/tests/test_reference_docs.py` | 128 passed | 0 |
| `funhouse_agent/deep/tests/test_wave2c_offline.py`, `webapp/tests/test_record_ev06.py`, `webapp/tests/test_public_text.py` | 51 passed | 0 |
| `webapp/tests/test_page_fingerprints.py` | 15 passed (nothing agent-facing changed; no re-record) | 0 |

## Noted, not fixed

- **Report ingest on the proxy route has no wait-and-retry ladder for a rate
  limit** (lead to decide).
  - `ChatModelEngine` now records the named 400 as `rate_limit` (before, it
    was `other`), and its pacer holds the next request, as for a 429. The
    failed request is still raised.
  - Nothing in `ChatModelEngine` waits and asks again: only
    `PrompterEngine` has `RATE_LIMIT_WAITS`. A 429 gets the SDK's two quick
    retries; the 400 gets none.
  - This changes a live run only if ingest runs on the Claude proxy route on
    Foundry. `engine_for` assumes the page's model there is the
    `palantir_models` one, whose own layer retries for five minutes.
- **Which form Foundry's spent allowance takes is unknown (check U1).**
  - If it comes as a named 400 in words that none of `BUDGET_PHRASES`
    carries, it is retried as a rate limit until the ladder or the turn bound
    runs out, and is then called a rate limit.
  - The fix is a phrase in `BUDGET_PHRASES` once U1 shows the words.
- `vision_engine.busy_kind` raises if an error's `body` property raises. This
  was already so before this change. Its callers in `busy_retry.busy` and
  `core._busy_kind` catch it. No known SDK error does this.
- `report_ingest.RATE_LIMIT_WORDS` still lists the two names, to catch a
  message that only says them. A name in an error's attribute, body or
  response is read by the one helper.
- The missing-PDF list's hint "names must match capital letters exactly on
  the cluster" says "cluster" on Foundry as well. This is wording only.
- `webapp/app.py:537` ("try Permanent storage below" on Foundry) is left
  alone: another builder is working in `app.py`.
