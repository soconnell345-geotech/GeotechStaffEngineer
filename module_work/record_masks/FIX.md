# Fix: the review's open findings 9, 10 and 11 (`claude/fx-record-masks`)

From `claude/rv-record-masks` at `40f8d9d`. One commit per finding, each with tests that fail before it.
Findings 12 and 13 are untouched. Both fingerprint files are unmoved; no prompt, tool description or
setting changed.

| Finding | Commit | Files |
|---|---|---|
| 9 | `6fc2bcc` | `error_text.py`, `vision_probe.py`, `cited_pages.py`, `deep/instructions.py`; `tests/test_record_masks_fx9.py` (core) |
| 10 | `bf0ad2a` | `review_eval/checks.py`, `review_eval/runner.py`; `tests/test_record_masks_fx10.py` (core), `tests/test_record_masks_fx10_page.py` (Document Review) |
| 11 | `4f59d93` | `sharepoint_tools.py`, `minimal_agent.py`, `feedback.py` (Document Review); `tests/test_record_masks_fx11_page.py` |

`mask_payload` and `masked_record` are as they were: top-level strings only. Text is masked where it is
made, with the masker the record uses (`error_text.mask_error_secrets`).

## 1. What changed

### Finding 9: error text one level down in the record

- **The helper** (`error_text.masked_error_text(text, limit=None)`, beside the masker): masks, then cuts
  to `limit`. A text with no secret comes back character for character. If the masker fails, the fixed
  line `TEXT_NOT_CHECKED` stands for a text that was not empty; the raw text is never returned.
- **The vision probe** (`vision_probe._error_text`, used in `_Busy.__init__`, the two `prof.error`
  assignments from a plain exception in `probe` and `_probe_host_edge`, and the belt in `profile_for`).
  The profile's `error` is what `fallback_note()` and `summary()` quote, so this covers
  `vision.probe_fallback` on `model_start` / `model_end` / `model_error` (through
  `deep/vision_engine.py`, which is unchanged), the line a vision result carries, and the suite's
  `vision_profile.summary` and `profiles[].error`. Masked before the 300-character cut.
- **`photo_pages.why`** (`cited_pages.TurnPageCheck._photos`).
- **`turn_start.engine.build.appended.error`** and `file_errors` (`deep/instructions._reason`, which
  makes both). The page's warning lines built from them (`instructions.describe`) are masked with them.

### Finding 10: the review suite's result

- **`checks[].detail` of a check that raised** (`checks.run_check`): masked where it is made. The words
  it begins with are now the constant `checks.CHECK_ERROR` (`"check error:"`, the same text).
- **`vision_profile.summary`, `profiles[].error`**: masked where the probe sets the error (finding 9).
- **A result saved before the masking** (`runner._masked_errors`): now also covers
  `vision_profile.summary`, `vision_profile.profiles[].error` and the detail of a check whose detail
  begins `check error:`.
- **The re-score path** writes the re-scored result through `_masked_errors`.
- **The done-already path** reads the saved result through it (`runner._masked_saved`), so
  `results.json` and the table are masked, and writes `run.json` again only if masking changed it.

### Finding 11: three plain-text messages the model reads

- **`sharepoint_tools.py`**: the upload error for a conversation's own file goes through `_masked`, the
  masking half of `_logged` split out; `MESSAGE_NOT_CHECKED` if the masker fails.
- **`minimal_agent.py`**: "The helper failed: ..." through `_masked_reason` (masked, then cut to 300).
- **`feedback.py`**: "Feedback could not be recorded (...)" masked the same way.
- In the last two, the fixed words `REASON_NOT_CHECKED` stand for the text if the masker fails; the
  exception's type name is still shown.

## 2. Made-up before and after

| Where | Before | After |
|---|---|---|
| `vision.probe_fallback` | `...could not be measured (Refused: ...sheet.pdf?sig=Made0UpValue1For2Test&se=2 ...)` | `...(Refused: ...sheet.pdf?sig=…&se=2 ...)` |
| `checks[].detail` | `check error: Refused: ... api_key=Made0UpValue1For2Test` | `check error: Refused: ... api_key=…` |
| the upload tool | `SharePoint upload error: Refused: ...?download=1&tempauth=Made0UpValue1For2Test` | `SharePoint upload error: Refused: ...?download=1&tempauth=…` |
| any of them, no secret | `Refused: ...sheet.pdf?page=3 (request 0f3a)` | the same, character for character |

## 3. Tests

New tests, and what they do on the commit before their fix:

| File | Tests | Before the fix |
|---|---|---|
| core `test_record_masks_fx9.py` | 13 | On `40f8d9d` the file stops at its import (the helper does not exist). With only the helper added: 6 failed (two probe cases, the probe that raised outright, the photo check, the instructions folder and file errors), 7 passed. |
| core `test_record_masks_fx10.py` | 6 | On `6fc2bcc`: 3 failed (a raised check, an old result read back, the rewrite of a saved file), 3 passed. |
| Document Review `test_record_masks_fx10_page.py` | 3 | On `6fc2bcc`: 2 failed (re-score and done-already), 1 passed. |
| Document Review `test_record_masks_fx11_page.py` | 9 | On `bf0ad2a`: 6 failed (masked and masker-fails, for each of the three sites), 3 passed. |

The tests that pass before a fix are the "reads as before" guards, one per site: a text with no secret
is compared whole with what the old code wrote. `test_a_measured_profiles_summary_keeps_no_secret`
(finding 10) also passes on `6fc2bcc`, because finding 9's commit already masks that text where it is
made; it fails on `40f8d9d`.

Test values are made up: `Made0UpValue1For2Test`, `example.test` hosts, `sig=`, `api_key=` and
`tempauth=` as names. No existing test was edited.

Runs on the last finding commit (`4f59d93`), judged by exit code:

| Run | Result |
|---|---|
| venv (b): all of `packages/aec-agent-core/tests`, `webapp/tests/test_wheels.py`, `test_core_move_11c.py`, `test_page_fingerprints.py` | 761 passed, 5 skipped |
| venv (b): the four core masks files and `webapp/tests/test_page_fingerprints.py` | 113 passed |
| venv (a), Document Review: the four masks files, `test_review_fingerprints.py`, `test_boundary.py`, `test_feedback.py`, `test_review_minimal_offline.py`, `test_sharepoint_tools.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_privacy.py`, `test_sharepoint_ta35.py`, `test_ta40_sharepoint_off_turn.py`, `test_review_eval_offline.py` | 284 passed, 2 skipped |
| `webapp/tests/test_public_text.py`, `test_review_freeze.py`, before each commit | 13 passed |

Also run on the way: after finding 9, `test_dr48_photo_pages.py` and `funhouse_agent/tests/test_cited_pages.py`
(62 passed), `test_ta39_probe_refusal.py`, `test_vision_probe_retry.py` and `test_instructions.py` (in a
run of 150 passed); after finding 10, `test_inf75_results_counts.py`, `test_ev48_working_text_said.py`
and `test_core_suite_and_records.py`.

**Fingerprints:** `tests/review_fingerprints.json` and `webapp/tests/page_fingerprints.json` are unmoved.

## 4. Calls made

1. **The helper's name and fallback.** `masked_error_text`, with `TEXT_NOT_CHECKED` when the masker
   fails. Document Review's three sites call `mask_error_secrets` directly, as the brief said, so they
   do not depend on the new helper.
2. **`_Busy`'s own text is masked** when the exception is built, so the probe's two `log.warning` lines
   that print it are masked as well.
3. **`_masked_errors` reads old results more widely** (summary, `profiles[].error`, raised checks). It
   is the result-level reader, not `mask_payload`. Only a detail that begins `check error:` is touched:
   any other check's detail quotes the answer and is left alone.
4. **The done-already path rewrites `run.json` when masking changed it**, so a secret does not stay on
   disk. The rewrite uses the runner's layout (`indent=1`); a result with no secret is not rewritten.
5. **The upload error is masked, not logged.** Routing it through `_logged` would add a line to the
   SharePoint log on that failure; the brief asked for masking only.
6. **`instructions._reason` masks both `error` and `file_errors`**, since it makes both.

## 5. Still not covered

- Findings 12 and 13, as the lead decided.
- `cited_pages.TurnPageCheck._check`'s `why` is not masked where it is made; it is a top-level `why`
  in the `cited_pages` event, which the record already masks by name.
- The probe's `log.info` lines that print a raw exception (`detail=original refused`, `host edge not
  measured`) and the runner's console lines (`previous attempt failed (...)`, `mirror failed`): server
  log and console text, already listed in REVIEW.md section 2.
- Finding 7's remainder (a nested `{"error": {"message": ...}}`, error text under another key of a
  tool's result) is unchanged.
- `minimal_agent._mark_up_tool`'s `{"error": f"could not open {source}: {exc}"}` is a JSON error
  result: `ToolErrorGuard` and the record's error-result rule mask it where that guard is in the
  page's middleware. Not checked for the minimal page.
- An old `run.json` is cleaned only when the suite reads it again. `results.json` of an old run is
  rewritten whole on the next run of that folder; old `activity.jsonl` files are not rewritten (the run
  report masks top-level fields when it reads them, not `vision.probe_fallback` inside an old record).
- GSE's own copies under `webapp/` and `funhouse_agent/` (frozen, or leaving with INF-35) were not
  changed.
- No live run: a real failing turn on the testers' host (REVIEW.md section 6) is still the lead's
  check, and the full suite on GitHub (`ci/<name>`) was not started from here.
