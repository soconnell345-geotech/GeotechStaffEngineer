# Review: the run record masks secrets (`claude/record-masks-secrets`)

**Verdict: MERGE AFTER FIXES**, reviewed head `d910d90` (base `a7a48d8`). Eight fixes are on
`claude/rv-record-masks`; merge that branch, not the reviewed one. Findings 9 to 13 are open.

## 1. Findings

| # | Severity | Where | What is wrong, and the made-up input that shows it | State |
|---|---|---|---|---|
| 1 | must-fix | `error_text.py` `_PEM_KEY` | Every `-----BEGIN PRIVATE KEY-----` with no END line searched to the end of the text. 8,000 of them (216,000 characters): 8.2 s; a 2 MB text: 268 s, inside the turn. A key block cut short by a length limit was also shown whole. | fixed `6d1e97f` |
| 2 | must-fix | `error_text.py` `_NAMED_SECRET` | A name was not matched after `_` or `-`, so this code's own settings went through: `GRAPH_CLIENT_SECRET=<v>`, `PROMPTER_API_KEY=<v>`, `GRAPH_CLIENT_KEY_PASSWORD: <v>`, `PROMPTER-API-KEY--J.DOE=<v>`. Also no rule for `secret_access_key`, `secret_key`, `private_key`, `passphrase`, `x-auth-token`. | fixed `568459d` |
| 3 | must-fix | `error_text.py` | `{"key": "<v>"}`, `{"token": "<v>"}`, `{'secret': '<v>'}` and a form body starting `token=<v>&...` or `code=<v>&...` went through: these names were masked only after `?` or `&`. | fixed `34dac50` (a call made, below) |
| 4 | must-fix | `error_text.py` | `Cookie: sessionid=<v>; theme=dark` and `Set-Cookie: SESSION=<v>; Path=/` went through. | fixed `7105738` |
| 5 | must-fix | `error_text.py` | A percent-encoded URL: `redirect_uri=https%3A%2F%2Fapp.example.test%2Fcb%3Faccess_token%3D<v>%26state%3D1` and `...%26sig%3D<v>%26se%3D2` went through. | fixed `a9eca6a` |
| 6 | should-fix | `error_text.py` `_URL_PASSWORD` | `http://someone:made@up<v>@proxy.example.test` was shown from the first `@` on. | fixed `70aa3b0` |
| 7 | should-fix (item 5) | `deep/tool_guards.py` | A tool's own `{"error": str(exc)}` reached the model unmasked (the record masked it). `ToolErrorGuard` now masks the top-level `"error"` string only. | built `f48d32e` |
| 8 | should-fix | `sharepoint_tools.py` `_logged` | `mask_error_secrets(message) or message`: the masker returns `""` when it fails, so the raw message went to the model. CHECK.md did not say so. | fixed `be67a13` |
| 9 | should-fix, open | `activity_log.mask_payload`, `masked_record` | Only top-level strings are masked. Error text one level down is written raw: `vision.probe_fallback` on `model_start` / `model_end` / `model_error` (`vision_probe.py:500-609` through `deep/vision_engine.py:1018`), `cited_pages` → `photo_pages.why` (`cited_pages.py:1721`), `turn_start.engine.build.appended.error` (`deep/instructions.py:173`). The run report shows them too. | not fixed: which nested keys count as error text is a design decision |
| 10 | should-fix, open | `review_eval/runner.py` | `_masked_errors` leaves `checks[].detail` (`checks.py:1575` `check error: {exc}`), `vision_profile.summary` and `profiles[].error` (`runner.py:794-797`); the re-score / "done already" path (`runner.py:1109-1118`) rewrites an old `run.json` without `_masked_errors`. | not fixed: same decision as 9 (which keys), and no time left for the tests |
| 11 | should-fix, open | `sharepoint_tools.py:1530` | `shared = f"SharePoint upload error: {type(exc).__name__}: {exc}"` skips `_logged`; plain text, so neither `mask_error_result` nor finding 7 catches it. Also `minimal_agent.py:234` ("The helper failed: ...") and `feedback.py:210`. | not fixed: no test written in the time allowed; one line each through `mask_error_secrets` |
| 12 | note | `diagnostics.py:104` | `mask_error_secrets(text) or text`: the scrubbed report goes out if the masker fails (CHECK.md call 8 says so). `sharepoint_download` (`diagnostics.py:280`) passes `scrub` only, which has no rule for `tempauth`, `sig`, `code` or `key`. | the lead's call |
| 13 | note | `error_text.py` | Still not caught: a value split by a line break (`access_token=<half>\n<half>` shows the second half); a quoted password with a space in it (shown after the space); `Authorization: Digest ... response="<v>"`; `Pwd=`, `session_id=`, `auth=`, `credentials=`; a twice-encoded URL (`%253D`). Over-masked: `?code=404&key=A-101` in a plain URL loses `A-101` (as on the reviewed head). | noted |

**Call made in finding 3.** Outside a query, `key` / `token` / `secret` / `signature` / `sig` / `code` are masked
only when the value is 16 characters or more with a letter and a digit. So `"key": "sheet_index"` and
`code=context_length_exceeded` are kept, but a hash or UUID labelled `key:` or `token:` (`cache_key=<sha>`) is
now masked. The lead may prefer a different threshold.

**Finding 7, what it leaves.** Only a JSON object's top-level `"error"` string. A plain-text error, a nested
`{"error": {"message": ...}}` and error text under another key (`vision_probe`, `stat_error`, `detail`) still
reach the model as they came. When a secret is masked the result is re-written with `json.dumps` (key order
kept, spacing normalised); every other result is the same object.

## 2. Paths checked

- **Misses (item 1).** Masked on the reviewed head: signed link `sig=` (also first in the query), `access_token=` /
  `code=` / `client_secret=` in a URL and after `&` in a form body, `Authorization: Bearer` / `Basic` (header, dict,
  escaped JSON), `api_key` / `apiKey` / `password` in JSON, `user:password@host`, a JWT alone, `X-Amz-Signature`,
  `AccountKey=`, a secret in `args[1]`, in an `OSError`'s text and in a `KeyError`. Missed: findings 1 to 6 and 13.
- **Nested exceptions.** `error_line` and the record use `str(exc)`: a `__cause__` and an exception group's
  members are not written at all, so nothing leaks; a `traceback` field is masked as text.
- **Over-masking (item 2).** The five plain errors of the tests and 16 of mine (long path, request and trace ids,
  a 40-hex commit, a UUID, base64 of an image, a 50-digit number, a DXF handle with `key=LAYER code=8`,
  `KeyError: 'token'`, `max_tokens=4096`, a plain URL with `?page=3`, a sha256, `host:8443/a/b@c.pdf`,
  `authorization: required`): all byte for byte before and after the fixes, except `?code=404&key=A-101`.
- **Hostile input (item 3).** 27 inputs timed (2 MB of prose, 200,000 characters with no separator, runs of
  spaces after each name, repeated `://`, `Bearer`, `sk-`, `eyJ`, `&key=`, cookie pairs, `%3D`). After the fixes
  the worst are 2 MB of prose 455 ms (268 ms on the reviewed head; the new rules add a pass) and 2 MB of mixed
  secrets 395 ms; every 200,000-character input is under 110 ms. Nothing is worse than linear.
- **Writers (item 4).** Masked: `tool_error`, `model_error.error`, `turn_end.error` and `error_detail`,
  `tool_refused`, a JSON error in `tool_end` / `tool_delivered`, app events' top-level `error` / `why` /
  `traceback`, `page_error`, the suite's `error` / `outcome_error` / `traceback` / `turns[].error`, the failed
  turn's shown error with `transcript.jsonl`, `trace.jsonl` and the history note (`turn_jobs.py:365-381`; the raw
  exception is written nowhere else), `format_report`, `_logged`, `KeepWork._stopped`. Not masked: findings 9
  to 12, plus text shown only on the page (`app.py:183/797` agent build, `:248/:460/:1785` save error, `:684`
  sync errors, `:800` engine message, `:1291`, `:1438`), `sharepoint_store.py:1266` (`not_sent[].why` in the
  local manifest), a stopped turn's kept tool results (`turn_record.py`), the kept helper file's body
  (`keep_work.py:948`), server log lines (`palantir_sdk_engine.py:318`, `vision_probe.py:482-627`,
  `graph_sharepoint.py:1126`, `databricks_bridge.py:526`) and the runner's console lines (`runner.py:1063-1125`).
  CHECK.md lists the page-only, log and console groups under "Noted, not fixed"; it does not list findings 9 to 11.
- **Masking failing (item 6).** With `_mask` made to raise: the record keeps `""` for an error field and for a
  JSON error result, an ordinary result as it came; `error_line` gives the type name; `ToolErrorGuard` still
  returns a `status="error"` message and the turn goes on; `append_event` returns True. Raw only at
  `diagnostics.py:104` (finding 12) and, before `be67a13`, `_logged`.
- **Shared host.** No lock, thread or module table is added; the maskers are pure functions.

## 3. Tests

- Reviewed head: core `test_record_masks_secrets.py` 40 passed; Document Review
  `test_record_masks_secrets_page.py` + `test_review_fingerprints.py` + `test_boundary.py` 14 passed;
  `webapp/tests/test_page_fingerprints.py` 13 passed; `test_public_text.py` + `test_review_freeze.py` 13 passed.
- New tests on the base `a7a48d8`: the core file stops at its import (all 40); the page file 4 failed, 1 passed
  (`test_a_failed_turns_error_with_no_secret_is_as_before`, a guard by design).
- After the fixes: `packages/aec-agent-core/tests` with `test_page_fingerprints.py`, `test_auth_refusal.py`,
  `test_core_move_11c.py`, `test_wheels.py` (venv b): 748 passed, 5 skipped. Document Review (venv a): the two
  masks files, fingerprints, boundary: 15 passed; `test_ta38_turn_ceilings.py`, `test_auth_refusal.py` with them
  27 passed; three SharePoint tool files 86 passed. **Both fingerprint files unmoved.**
- My tests: `packages/aec-agent-core/tests/test_record_masks_review.py` (41) and
  `packages/aec-document-review/tests/test_record_masks_review_page.py` (1). Each case of a fix failed before it
  (the probe output on the reviewed head); the "kept" cases are guards.
- The test values (item 8): made-up shapes only (`sk-fake-00000000-not-real`, a 14-character JWT header,
  `example.test` hosts); none has a live provider's key format. No rewording needed.
- **Order clash (item 9).** On the base too: `tests/test_foundry_host.py` then `tests/test_core.py` in one
  process, 3 failed, 92 passed; `test_core.py` alone 86 passed. The state that leaks is `os.environ`:
  `GEOTECH_DEPLOYMENT=foundry` (and `GEOTECH_PLOT_CODE=0`) are still set after the file. The `_clean` fixture
  pops them by hand, then monkeypatch's own undo, which runs later, puts back the value the entry had written
  straight into `os.environ`. Not one line; not fixed.
- **Merge with `origin/claude/b13-integration` (`1826bdf`), item 10.** One conflict, in `activity_log.py`
  `turn_end`: both sides changed the `error_detail` lines. Keep this branch's masked `error_detail` and then
  b13's `calls_not_started` block. The docstring and `run_report.py` merge by themselves. Tried on the reviewed
  head, not on the fix commits (they do not touch those files). Nothing pushed.

## 4. Scope

Every file on the reviewed branch is one the claims name, a new test or CHECK.md; no existing test was edited.
My fixes touch `error_text.py`, `deep/tool_guards.py`, `sharepoint_tools.py` and two new test files.

## 5. CHECK.md

- "If masking itself fails ... nothing raw": untrue for `sharepoint_tools._logged` (fixed) and, as it says, the
  diagnostics report.
- "Every place an error's text is kept or shown": the table omits the nested fields of finding 9, the suite's
  `checks[].detail` and re-score path (10) and `sharepoint_tools.py:1530` (11).
- The list of what is "still not caught" omits the setting-style names (finding 2) and JSON `key` / `token`
  (finding 3), which were the two widest misses.
- The count of 46 own-built error strings was not recounted. Run counts of existing tests not repeated in full.

## 6. Could not check

- A real failing turn on the testers' host (an expired SharePoint link): download the conversation and search
  `activity.jsonl`, `transcript.jsonl`, `messages.json` and the run report for the link's `tempauth` value.
- Whether any model's vision tool error carries `key:` or `token:` with a hash (the call in finding 3): search a
  saved suite run's `activity.jsonl` for `"…"` after re-reading it with `run_report`.
- The full suite: push `claude/rv-record-masks` as `ci/<name>` (the lead's step).
