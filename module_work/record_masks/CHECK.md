# The run record masks secrets in the text of an error: check notes

A repair, on, no switch (the lead's ruling). Commits `fd7d3df` and
`5e16c0c` (the record's own, narrower masker, on the lead's later rule that
an unlabelled id or hash is kept and `mask_secrets` stays as it was) on
`claude/record-masks-secrets`.

## What the item asked

The handlers for a failed tool and a failed model call in
`aec_agent_core/activity_log.py` (`on_tool_error`, `on_llm_error`) wrote
the exception's text into `activity.jsonl` as it came, while the core has a
masker for this (`error_text.mask_secrets`). An exception from a model
client or an HTTP library can carry a key, a bearer token or a signed URL;
the record is what a tester downloads and sends on, and what the run report
prints. Asked: find every place the two packages write an exception's text
or a tool's error string to something kept or shown; mask at the write, one
place per writer, with the existing masker; extend the masker where it
misses a kind an exception can plausibly carry, without eating ordinary
text; keep every record's shape; an error with no secret written as before.

Then, from the lead: `mask_secrets` also replaces any run of 32 or more
letters, digits, `-` and `_`, which would hide the request ids, run ids
and hashes someone needs to trace a failure. Rule for the record: mask a
secret by its form or its place, keep an unlabelled id or hash; if that
needs a second, narrower function, build it and keep `mask_secrets` byte
for byte for the places that use it today, including `_side_prompt`.

## What was built, and where

- `aec_agent_core/error_text.py`: `mask_error_secrets`, the record's
  masker (below): a secret masked where its name or its form says it is
  one, a run of characters with no name beside it kept however long.
  `mask_error_result` (a tool's result masked with it only when it is the
  tool's error: a JSON object with an `"error"` string). `mask_secrets` and
  `html_gist` keep their rule `_SECRETS`, byte for byte.
- `aec_agent_core/activity_log.py`: `ERROR_FIELDS`, `mask_payload` (the one
  rule) and `masked_record`. Called where the logger takes in error text,
  before any cut: `_tool_finish` (`tool_error`, `tool_end`), `_delivered`
  (`tool_refused`, `tool_delivered`), `_graph_delivered`, `on_llm_error`,
  `turn_end`, `on_custom_event` (app events), `append_event`.
- `aec_agent_core/run_report.py`: `RunRecord` masks the records it reads
  (`masked_record`); `_whole_text` masks a whole copy read from beside the
  log (`mask_payload`).
- `aec_agent_core/review_eval/runner.py`: `run_task` returns its result
  through `_masked_errors` (the body is now `_run_task`).
- `aec_document_review/turn_jobs.py` (`_run_turn_job`): the failed turn's
  error masked once, after `friendly_turn_error`.
- `aec_document_review/diagnostics.py` (`format_report`):
  `mask_error_secrets` after the existing `sharepoint_log.scrub`.
- `aec_document_review/sharepoint_tools.py` (`_logged`, which every
  SharePoint tool failure passes): the returned message masked.

## Which masker each writer uses, and why

| Writer | Masker | Why |
|---|---|---|
| core `activity_log.mask_payload`: every error field, `tool_refused`, a tool's error result, app events; so `tool_error`, `model_error`, `turn_end`, `append_event` | `mask_error_secrets` | the record: a request id, a run id or a hash is what someone needs to trace the failure |
| core `run_report` (`RunRecord`, `_whole_text`) | `mask_error_secrets`, through `mask_payload` | it prints the record |
| core `review_eval/runner._masked_errors` | `mask_error_secrets` | the suite's records |
| Document Review `turn_jobs._run_turn_job` (the failed turn's error) | `mask_error_secrets` | shown and kept; a tester quotes its request id |
| Document Review `diagnostics.format_report` | `mask_error_secrets`, after `scrub` | sent to the app owner to trace a failure |
| Document Review `sharepoint_tools._logged` | `mask_error_secrets` | the tool's result, kept in the record |
| `page_errors._clean` (Document Review's, and the geotech page's `webapp/page_errors.py`) | `mask_secrets`, as before | used it before the branch; unchanged on the lead's rule. Its `page_error` record then passes `append_event`, which adds the narrower masker's named secrets |
| core `error_text.html_gist` (the sign-in refusal shown on both pages) | `_SECRETS`, the rule of `mask_secrets`, as before | not the record |
| core `activity_log._side_prompt` (a vision side call's prompt excerpt) | `mask_secrets`, as before | the lead's rule: left as it is |

## Every place an error's text is kept or shown

"Before" and "after" say whether a secret in the text is masked.

| Where it lands | Writer (file, function) | Before | After |
|---|---|---|---|
| `activity.jsonl` `tool_error.error`, and its whole copy in `activity_full/` | core `activity_log`: `on_tool_error` → `_tool_finish` | no | yes |
| `activity.jsonl` `model_error.error` | core `activity_log.on_llm_error` | no | yes |
| `activity.jsonl` `turn_end.error` (what the user was shown) | core `activity_log.turn_end` | Document Review: paths only (a 401's gist by the old rule); geotech page: as its page wrote it | yes |
| `activity.jsonl` `turn_end.error_detail` (a spent budget's provider text; a step cap's raw error) | core `activity_log.turn_end` | paths only | yes |
| `activity.jsonl` `tool_refused.text` (the error message the model was given for a tool that raised; a guard's refusal) | core `activity_log._delivered` | no | yes |
| `activity.jsonl` `tool_end.result` when it is the tool's error (`tool_error()`, the tools' own `{"error": ...}` results) | core `activity_log._tool_finish` | no | yes; ordinary results unchanged |
| `activity.jsonl` `tool_delivered.text` when it is an error result | core `activity_log._delivered`, `_graph_delivered` | no | yes |
| `activity.jsonl` app events' `error` / `why` / `traceback` (`helper_kept.why` "stopped on an error (...)", `cited_pages` / `cited_documents` "the check failed: ...", `engine_retry.error`, a class name) | core `activity_log.on_custom_event` | no | yes |
| `activity.jsonl` `page_error.error` / `.traceback` | Document Review `page_errors.error_facts` (and the geotech page's `webapp/page_errors.py`, same masker), then `activity_log.append_event` | yes (URLs cut to their host, `mask_secrets`) | yes (`mask_secrets` as before, then `append_event` with `mask_error_secrets`) |
| `activity.jsonl` `turn_stop_requested`, `turn_stopped`, `restored`, `process_start`, `turn_files` | various | no error text | n/a |
| Document Review: the error shown under a failed turn, `transcript.jsonl` `error`, `trace.jsonl` `error`, the history note in `messages.json`, the job's result | `turn_jobs._run_turn_job` (`turn_error`) | paths only | yes |
| `messages.json`: a stopped turn's kept tool results (TA-19) | `turn_record` | as the model got them | unchanged (noted below) |
| Review suite `run.json`, `results.json`, `RESULTS.md`: `error`, `outcome_error`, `traceback`, each turn's `error`, `vision_profile.error` | core `review_eval/runner.run_task` | no | yes |
| Review suite `turn_end` in the run's `activity.jsonl` | the logger | no | yes |
| Run report (`run_report.html`): turn notes, a failed model call, a tool's error, the error message the model was given, a tool's error result, app events | core `run_report.RunRecord`, `_whole_text` | no | yes, for records written before this change too |
| Connection diagnostics report (shown, sent to the app owner) | Document Review `diagnostics.format_report` | `sharepoint_log.scrub` | scrub, then `mask_error_secrets` |
| SharePoint report download | `diagnostics.sharepoint_download` | scrub | unchanged |
| SharePoint tool failure messages (the tool's result: the record and the model) | `sharepoint_tools._logged` | no (the SharePoint log's copy scrubbed) | yes |
| The SharePoint log | `sharepoint_log.record` | scrub | unchanged |
| Feedback records (`feedback.jsonl`, `FEEDBACK.md`) | `feedback.py` | the packages write no exception text there (the note is the agent's own words; the tool's own failure goes back to the model) | n/a |
| A vision side call's prompt excerpt (`model_end.vision.prompt`), not an error | core `activity_log._side_prompt` | `mask_secrets` | `mask_secrets`, unchanged |

## The record's masker, `mask_error_secrets`

`mask_secrets` (kept as it was): `sk-` / `pk-` / `rk-` keys, `Bearer
<anything>`, and any run of 32 or more letters, digits, `_` and `-` (which
also masks hashes, UUIDs, request ids and long file names). Before this
branch the record's error fields went unmasked, the `page_error` record
(through `page_errors`) excepted.

The secrets this code can meet (the two READMEs and
`tinyapps/wrapper_repo/.env.example`): the model endpoint's key and the
per-user keys (`PROMPTER_API_KEY`, `PROMPTER-API-KEY--<NAME>`,
`PROMPTER_KEYS`), the SharePoint app's client secret, its certificate's
private key and password (`GRAPH_CLIENT_SECRET`, `GRAPH_CLIENT_KEY`,
`GRAPH_CLIENT_KEY_PASSWORD`), bearer tokens (Graph's, a hosted platform's),
signed links (SharePoint download and upload links carry `tempauth` in the
query; object stores sign theirs), and a provider's own `sk-` keys.

`mask_error_secrets` masks:
- a private key in PEM form;
- a URL's password (`scheme://user:…@host`; the user is kept);
- an `Authorization` / `Proxy-Authorization` header's value with its scheme
  (`Bearer`, `Basic`, ...), in a header line, a dict or JSON;
- a JSON web token;
- the value after a name that says it is a secret, as `name=value`,
  `name: value`, `"name": "value"` or JSON-escaped: `api_key` (also "API
  key" and "API key provided", as a provider's refusal words it), `x-api-key`,
  `subscription-key`, `ocp-apim-subscription-key`, `access_token`,
  `refresh_token`, `id_token`, `auth_token`, `session_token`,
  `security_token`, `client_secret`, `client_assertion`, `password`,
  `passwd`, `tempauth`, `x-amz-signature`, `x-amz-credential`,
  `x-amz-security-token`, `x-goog-signature`, `x-goog-credential`,
  `account_key`, `shared_access_key`, `shared_access_signature`;
- `sig`, `signature`, `token`, `secret`, `code`, `key` as a URL's query
  parameter only (in prose they are ordinary words);
- `sk-` / `pk-` / `rk-` keys and `Bearer` tokens, as `mask_secrets` does
  (a bearer token's value here stops at a quote or a bracket; in
  `mask_secrets` it runs to the next space).

It has no rule for a run of characters with no name beside it.

Kept: request ids and run ids of any length and mix (a 36-character
`req_` id in upper and lower case and digits), hashes, UUIDs, page
numbers, file names, long words, `max_tokens=4096`, "token limit",
`api_key=None`, `password: ***`, "Authorization failed", and a value under
four characters. Every value stops at a quote, a backslash, a space or a
closing bracket, so a secret inside JSON text is masked and the JSON stays
whole; a full stop at a value's end is kept as the sentence's. A masked
text masks to itself.

Still not caught:
- a key standing alone with no name beside it and none of the forms above
  (a bare gateway key, a random token quoted in prose: "refused the key
  ..."), however random it looks: it cannot be told from a request id,
  and the lead's rule keeps those. It is masked when named
  (`subscription-key=`, `api-key: `, a query parameter, an `Authorization`
  header).
- a secret under a name the list does not hold (`my_value=...`), a cookie
  (`Set-Cookie: ...`), a sentence such as "the token is ..." (no `:` or
  `=`);
- a configured secret by its value (the SharePoint log masks the configured
  Graph secrets by value; this masker works by shape only).

## Calls made

1. **Two maskers.** `mask_secrets` keeps its rule byte for byte for the
   places that used it before (the shown page error through `page_errors`
   on both pages, the vision prompt excerpt in `_side_prompt`; `html_gist`
   has the same rule). The writers this branch masks use
   `mask_error_secrets`, which has no long-run rule: a request id, a run id,
   a hash, a long file name and a long word are kept, so the failure can be
   traced and an error with no secret is written as before (the lead's
   rule). The price: a bare key with no name and no known form is not
   masked (it is when named).
2. **Masked in what is kept, not in what the model is handed.**
   `error_text.error_line` / `tool_error` (the tool guard's message, the
   document tools' errors) are unchanged: masking there would change what
   the geotech page's model is handed, beyond its record. The record masks
   the same text as `tool_refused` / `tool_end`. The exception is
   SharePoint's failure messages (`_logged`, Document Review only), plain
   text the record cannot recognise as an error.
3. **`html_gist` keeps its rule**: it builds the sign-in refusal shown on
   both pages, which is not the record.
   A Document Review failed turn's error therefore passes both rules when
   the turn failed on a refused sign-in (`html_gist` inside
   `friendly_turn_error`, then `mask_error_secrets`); any other failed
   turn's error passes only the narrower one.
4. **A tool's error result is recognised by shape**: a JSON object with a
   top-level `"error"` string, the shape `tool_error()` and the tools' own
   errors take. A plain-text error result is not recognised (the SharePoint
   ones are masked at their source).
5. **`chars` of a masked record is the length of the masked text**, the
   same text as its whole copy beside the log.
6. **Placeholders and values under four characters beside a secret's name
   are kept**: `api_key=None` says the key is missing, which is the
   diagnosis.
7. **The run report masks on reading**, so a record written before this
   change prints masked; the records on disk are not rewritten.
8. **If masking itself fails**, the log keeps `""` (never raw), the turn
   worker shows the exception's class name, and the diagnostics report shows
   today's scrubbed text.

## Settings

None. A repair, on for every page.

## Tests

`packages/aec-agent-core/tests/test_record_masks_secrets.py` (31 tests).
On master the file stops at its import (`mask_error_secrets` is new); "on
master" below means with master's rule (copied into the file as
`OLD_RULE`) or master's code in its place:
- `test_each_shape_of_secret_is_masked` (16 cases): each shape, with the
  exact masked text. Master's rule gives 3 of them exactly (an `sk-` key, a
  bearer token, a 32-hex value); 13 fail.
- `test_an_ordinary_error_comes_back_character_for_character`: hashes, a
  UUID, two request ids (one of 36 characters in mixed case), page numbers,
  file names, a long word and secret-like words survive. Fails with
  master's rule (it masks the hashes, ids and names).
- `test_a_key_is_masked_and_a_request_id_and_a_hash_beside_it_survive`:
  the lead's case, `api-key: <made-up key>` beside a 36-character request
  id and a 40-character hash: the key masked, both ids kept. Master's rule
  does the reverse (the key kept, both ids masked); the test also states
  that `mask_secrets` still hides both ids.
- `test_a_run_with_no_name_beside_it_is_kept_however_random`: an unnamed
  random token is kept (what the record's masker does not catch, by the
  lead's rule). Fails with master's rule.
- `test_mask_secrets_keeps_its_rule_byte_for_byte`: `mask_secrets` against
  master's rule on every text in the file. A guard; passes on both.
- `test_a_masked_text_masks_to_itself_and_none_is_empty`: a guard.
- `test_a_tools_error_result_is_masked_and_its_ordinary_output_is_not`:
  fails on master (no `mask_error_result`).
- `test_a_failed_tool_and_a_failed_model_call_are_kept_masked`:
  `tool_error`, `model_error`, `turn_end` error and detail. Fails on master.
- `test_an_error_with_no_secret_is_written_as_before`: the record holds
  `f"{type}: {message}"` exactly, as before. A guard; passes on both.
- `test_a_raised_errors_key_is_masked_and_its_ids_are_kept`: the lead's
  case raised through a failed tool, a failed model call and `turn_end`:
  the key masked, the request id and the hash kept in each record. Fails
  on master (the key is written raw).
- `test_the_records_keep_their_shape`: the same keys and types with and
  without a secret. A guard; passes on both.
- `test_a_long_error_is_masked_before_its_cut_and_in_its_whole_copy`: the
  copy in `activity_full/` is masked. Fails on master.
- `test_a_tools_error_text_is_masked_and_its_ordinary_output_kept`: an error
  result, a `tool_refused`, a `tool_delivered` error masked; an ordinary
  result with secret-shaped text kept as the model got it. Fails on master.
- `test_app_events_mask_the_account_of_a_failure`: `helper_kept.why`,
  `page_error` through `append_event`. Fails on master.
- `test_the_run_report_masks_a_record_written_before_the_masking`: an
  unmasked record and its whole copy print masked. Fails on master.
- `test_the_suites_result_keeps_its_errors_masked`: `_masked_errors`, keys
  kept, an ordinary result unchanged. Fails on master.

`packages/aec-document-review/tests/test_record_masks_secrets_page.py`
(4 tests):
- `test_a_failed_turns_error_is_shown_and_kept_masked`: a model that raises,
  through the page's agent and the turn worker; the shown error, the
  transcript, `turn_end`, `messages.json`, `trace.jsonl` and
  `activity.jsonl` hold no secret, and the shown error keeps its
  36-character request id. Fails on master.
- `test_a_failed_turns_error_with_no_secret_is_as_before`: a guard; passes
  on both.
- `test_the_diagnostics_report_masks_what_the_record_masks`: masked, setting
  names and the request id kept, an ordinary report identical with the
  masker switched off. Fails on master.
- `test_a_sharepoint_failure_message_is_masked`: a signed link's token
  masked; an ordinary failure identical to its unmasked text. Fails on
  master.

Existing tests run after the second commit, every file that reaches the
changed code (grep of the test folders for the changed modules):
- the core's 11 files: 128 passed;
- Document Review's 39 files, with `tests/test_review_fingerprints.py`:
  538 passed; 3 failed, the `legacy`, `minimal` and `legacy+switches`
  cases of `test_review_page_vocabulary_offline.py` (`NoPageHost`), the
  laptop-only failures the app's CLAUDE.md lists as green on GitHub;
- the geotech page's 18 files: 332 passed, 5 skipped;
- `webapp/tests/test_public_text.py`: 5 passed.
No existing test was edited.

## Noted, not fixed

- What the model is handed for a tool error built by `error_text.tool_error`
  is unmasked (call 2); the model can quote it in an answer, which is then
  kept. Masking there changes the geotech page's model input: the lead's
  call.
- The geotech page's own transcript and shown turn error
  (`webapp/turn_jobs.py`, `webapp/core.py`) are its code, quiet here: a
  request for the maintainer if wanted.
- A stopped turn's kept tool results in `messages.json` (TA-19) are kept as
  the model got them.
- The shown page error (`page_errors._clean`, both pages) and its
  `page_error` record keep `mask_secrets`, which hides a request id or a
  hash there as it did before: unchanged on the lead's rule. Moving it to
  `mask_error_secrets` is the lead's call.
- Lines shown only on the page and not kept (`app.py`'s `agent_error`,
  `save_error`; a restore's message; a helper model's note; SharePoint save
  errors in the sidebar) and server log lines are not masked.
- The review suite runner's console lines (`say`, mirror errors) and the run
  report's line for its own failure are not records.
- planlens' own vision records (`reads.json`, `page_readings.json`) are
  planlens' writers, not checked here.

## For the lead: what a measurement on the hosted platform should compare

Nothing behavioural to measure. On one real failing turn (an expired
SharePoint link, or a wrong key on a test deployment), download the
conversation and check that `activity.jsonl`, `transcript.jsonl` and the run
report hold no key, token or signature, and that the error otherwise reads
as before; a record of an ordinary run is identical in keys and text to one
from 1.2.0.
