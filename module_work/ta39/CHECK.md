# TA-39: testers told apart (cloud task A, 2026-11-13)

Branch `claude/ta-39-testers-apart`, from master 50ef609. Commits: 4b31a85 (keys and folders, Document Review), 990bc3d (the vision probe, the core), then this file. Closes BACKLOG TA-39 except keying the running-turn table and the conversation-folder lookup by folder and id, which stays with INF-71 and was not touched.

DR = `packages/aec-document-review/aec_document_review/`; core = `packages/aec-agent-core/aec_agent_core/`. All three pieces are repairs. No setting was added.

## 1. Keys

**Confirmed on master** (DR `tinyapps_keys.py`). `tester_suffix` took `Identity.key` and turned every run of characters other than a letter or digit into one hyphen: `corp\j.doe`, `corp\j-doe` and `corp\j_doe` all gave `CORP-J-DOE`. `resolve_for` tried the per-tester secret before the `PROMPTER-KEYS` mapping. So the docstring's own advice for such testers ("give such testers a `PROMPTER-KEYS` entry instead") did not work while the shared secret existed: the secret won.

**Changed.**
- `tester_exact_secret_name` (new): a sign-in name whose domain or user part holds anything other than the letters A-Z and digits has a second secret name, `PROMPTER-API-KEY--<SUFFIX>--<HASH>`. `<HASH>` is the first ten hex digits, upper case, of SHA-256 over the name spelt `domain\user`, case-folded, in UTF-8 (`Identity.name_hash`). The readable part is cut if the whole would pass Key Vault's 127 characters, so a long name now has a secret name too (it had none).
- `resolve_for`: for such a name the order is the exact secret (its `PROMPTER-MODEL--...` and `PROMPTER-URL--...` beside it), then the `PROMPTER-KEYS` entry, then the old secret name. For a name of letters and digits the order and the names asked are as before.
- `tester_suffix` now reads `Identity.legacy_key` (the folder key as made before this change), so no suffix moved when the folder key gained a hash.
- `advertised_secret_name`, `tester_secret_names`, `suffix_is_exact` (new helpers). The refusal ("this person's key goes in the Key Vault secret ...") and the diagnostics line name the exact secret where there is one.
- `python -m aec_document_review.tinyapps_keys "DOMAIN\user" ...` prints a name's folder and secret names. It reads no setting and prints no key.
- The module docstring says how a deployment provisions two such testers.

**Before and after** (made up). `corp\j.doe` has the secret `PROMPTER-API-KEY--CORP-J-DOE`. `corp\j_doe` joins.
- Before: `corp\j_doe` is served `corp\j.doe`'s key and spends that budget. A `PROMPTER-KEYS` entry for `corp\j_doe` changes nothing.
- After: the owner adds `PROMPTER-API-KEY--CORP-J-DOE--A8FB6C89A0` (or a `PROMPTER-KEYS` entry for `corp\j_doe`); `corp\j_doe` is served that key, and `corp\j.doe` keeps theirs with nothing changed in the vault.

## 2. Folders

**Confirmed on master** (DR `identity.py`, `Identity.key`; used by `profiles.session_root` and, through `app.py`, as the owner segment of the SharePoint folders). Every run of characters outside `A-Z a-z 0-9 . _ -` became `_`: `o'brien`, `o brien` and `o_brien` all gave `corp__o_brien`; `josé` gave `corp__jos`, the folder of `jos`. `j.doe`, `j-doe` and `j_doe` already had different folders, as the brief says.

**One thing the brief did not list:** the old rule also trimmed underscores from each part's ends, so `corp\_svc`, `corp\svc` and `corp\svc_` shared `corp__svc`, although all three are in the safe alphabet. See "Calls made", 3.

**Changed.**
- `Identity.legacy_key`: the old rule, unchanged.
- `Identity.key_is_exact`: true when the old key gives the name back (each part only `a-z 0-9 . _ -` in lower case, no `_` at either end, no `__` inside).
- `Identity.key`: the old key when it is exact; otherwise the readable part, `__`, and `Identity.name_hash` (ten hex digits, lower case; the same digits as in the secret name). An exact key holds `__` once at most and a hashed one twice or more, so the two kinds cannot meet, whatever name a person has.
- `Identity.exact_name`, `Identity.name_hash`, `NAME_HASH_LEN` (new).
- `profiles.py` needed no change: `session_root` takes `identity.key`.
- No old folder is moved.

**Before and after** (made up).

| Sign-in name | Folder before | Folder after |
|---|---|---|
| `CORP\alice` | `users/corp__alice/` | the same |
| `corp\j.doe` | `users/corp__j.doe/` | the same |
| `corp\o_brien` | `users/corp__o_brien/` | the same |
| `corp\o'brien` | `users/corp__o_brien/` | `users/corp__o_brien__3210ff1719/` |
| `corp\_svc` | `users/corp__svc/` | `users/corp__svc__<hash>/` |

## 3. The probe

**Confirmed on master** (core `vision_probe.py`). `profile_for` stores a failed probe in `_FAILED` under `_cache_key(model)` (class, name, address) for `FAILED_HOLD_S` = 120 s, whatever failed it. A second defect of the same kind, not in the brief: a non-busy error on the probe's fourth call (`detail="original"`) was read as the model refusing `original` and CACHED for the process. If a tester's budget ran out between calls 3 and 4, every tester's charts were held to detail `high` until a restart.

**Changed.**
- `key_refusal(exc)` (new): "budget used up" (`error_text.budget_exhausted`), "sign-in refused" (`error_text.auth_refused`, a 401) or "no access" (a 403 or `PermissionDeniedError`), read from the error and what it was raised from; else `None`.
- `probe`: a failed profile carries `key_refused`; a key refused on call 4 fails the probe instead of being read as `original` refused.
- `profile_for`: a failed profile with `key_refused` is returned to its caller and kept in neither `_FAILED` nor `_CACHE`. A call that waited on that probe loops and measures with its own key instead of going ahead unmeasured. A model failure (busy, anything else) is held as before; a successful probe is cached and shared as before.
- `VisionProfile.to_dict()` gains `"key_refused"` (`None` or the reason).

**Before and after** (made up). Tester A's budget is spent; A looks at a page at 10:00:00, tester B at 10:00:30, same deployment.
- Before: B's look uses the fallback image budget and carries "the vision model could not be measured", until 10:02:00.
- After: A's look uses the fallback and says so; B's look measures the model with B's key and uses the full size.

## Tests

New files; nothing else added.

`packages/aec-document-review/tests/test_ta39_testers_apart.py` (53 tests). On master's code: 41 fail, 12 pass.
- Fail on master because of the defect: the folder pairs for apostrophe, accent, two accents and trimmed underscore (8); `test_one_person_spelt_two_ways...`; every `test_two_sign_in_names_can_be_given_two_keys` pair (17); `test_the_exact_secret_is_asked_for_before_the_shared_name`; `test_the_exact_mapping_entry_comes_before_the_shared_secret`; `test_the_refusal_names_the_exact_secret`; `test_a_long_name_still_has_an_exact_secret_name`; `test_the_name_printer...`.
- Fail on master only because they read a name that is new (`legacy_key`, `tester_exact_secret_name`): `test_an_ordinary_names_folder_has_not_moved` (8), `test_a_name_outside_the_folder_alphabet_carries_its_hash`, `test_a_name_of_letters_and_digits_has_one_secret_name`. They pin what must not change.
- Pass on master, on purpose (they pin what must not change): the folder pairs that were already apart (punctuation, domain or user, the hash mark; 9), `test_a_single_user_host_keeps_its_folder`, `test_a_tester_with_a_key_today_still_has_it`, `test_a_single_user_host_keeps_the_shared_key`.

`packages/aec-agent-core/tests/test_ta39_probe_refusal.py` (9 tests). On master's code: 8 fail, 1 pass.
- Fail because of the defect: `test_a_refused_key_leaves_no_verdict_for_the_next_caller` (3), `test_a_key_refused_on_the_original_call...`, `test_a_caller_waiting_on_a_refused_probe_measures_for_itself`.
- Fail only through the new name `key_refused` or `key_refusal`: `test_a_failure_about_the_model_is_still_held`, `test_a_detail_the_model_refuses_is_still_a_shared_verdict`, `test_key_refusal_reads_the_error...`.
- Passes on master, on purpose: `test_a_refused_key_does_not_displace_the_shared_measurement`.

**Existing test edited** (its assertions became untrue): `packages/aec-document-review/tests/test_tinyapps_wiring.py`, two lines. `test_an_account_name_is_whatever_windows_allows` expected `corp__o_brien` for `o'brien`; `test_key_is_case_insensitive_and_folder_safe` expected `corp__j_doe` for `J Doe`. Both now expect the hashed key (and still that the two spellings by case agree).

**Runs** (each judged by pytest's exit code, on the branch's final code):

| Venv | Command (from) | Result |
|---|---|---|
| a | `tests/test_ta39_testers_apart.py tests/test_tinyapps_keys.py tests/test_app_profiles.py tests/test_tinyapps_wiring.py tests/test_sharepoint_privacy.py tests/test_boundary.py tests/test_review_fingerprints.py` (DR package folder) | 145 passed |
| a | `tests/test_foundry_host.py tests/test_review_eval_offline.py tests/test_tinyapps_entry.py tests/test_diagnostics.py` (DR package folder) | 118 passed, 2 skipped |
| a | `--noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` (root) | 13 passed |
| b | `webapp/tests/test_page_fingerprints.py packages/aec-agent-core/tests/test_ta39_probe_refusal.py packages/aec-agent-core/tests/test_vision_probe_retry.py funhouse_agent/tests/test_vision_probe.py funhouse_agent/deep/tests/test_wave2b_guards_offline.py` (root) | 80 passed |
| b | the other test files that name `vision_probe`: `packages/aec-agent-core/tests/test_vision_side_call_timeout_offline.py`, `funhouse_agent/review_eval/tests/test_review_eval_offline.py`, `funhouse_agent/tests/test_vision_turned.py`, `test_find_like_tool.py`, `test_vision_locations.py`, `test_find_like_brief4.py`, `test_vision_brief4.py`, `test_vision_view.py`, `test_vision_images.py`, `funhouse_agent/deep/tests/test_strict_args_offline.py`, `webapp/tests/test_palantir_claude_engine.py` (root, `-n 4`) | 342 passed |

Not run here: `webapp/tests/test_wheels.py` (it builds wheels) and the full suites; they run on GitHub as `ci/<name>`.

## Fingerprints

Not moved. `tests/test_review_fingerprints.py` (venv a) and `webapp/tests/test_page_fingerprints.py` (venv b) pass unchanged; no re-record commit.

## Calls made

1. **The key route: a hash in the secret's name, not the mapping first.** It keeps one secret per tester, which is how the pilot provisions; a name of letters and digits is asked for exactly as before; and nobody who resolves today changes source unless they hold both a shared-suffix secret and a mapping entry (call 2). The mapping-first route would have changed the source for every tester holding both.
2. **For a name with punctuation, the `PROMPTER-KEYS` entry is tried before the old secret name** (after the exact secret). Otherwise a mapping entry could never separate two such testers, which is what the old docstring promised. Cost: a tester with punctuation in the name who has BOTH an old secret and a mapping entry is now served the mapping's key (their own entry) instead of the secret's. Revert by dropping `_mapped` back to last in `resolve_for`.
3. **Folders: names with `_` at an end of a part, or `__` inside one, also get the hash**, though they are in the safe alphabet. The brief's "must not change" says a safe-alphabet name keeps its folder; its change 2 says two names never share one. They conflict for `corp\_svc` and `corp\svc`, and I chose never sharing. To restore the brief's letter, make `key_is_exact` test the alphabet only (one condition); the last three "trimmed underscore" test pairs then fail.
4. **Ten hex digits (40 bits)** in both the folder and the secret name, the same digits in each so the owner can match them.
5. **The hash is of the case-folded name in Unicode's composed form (NFC)**, so an accented letter sent as one character or as a letter and a combining accent is one person, one folder, one key. Two accounts that differ only in that form would share; I know of no directory that allows it.
6. **A 403 counts as the caller's key refused.** If a deployment answers 403 to every key (the model itself is closed), the failed probe is no longer held, so each look probes again: four quick refused calls, no cost. The look itself fails the same way either way.
7. **A dotted name's refusal now names the exact secret** (`...--CORP-J-DOE--<HASH>`), so a new tester is never provisioned under a name another can share. Existing secrets under the old name keep working.
8. **`key_refused` is always in `to_dict()`** (as `None` when not refused), so the record's shape does not vary.

## Release note

Document Review on a shared host: two testers whose sign-in names differ only in punctuation or accents can now have separate AI keys, and never share a folder. Nothing changes for a tester whose sign-in name is letters, digits, `.`, `-` and `_` (with no `_` at either end of the user name and no `__`): same key, same folder, same SharePoint mirror. A tester whose name holds anything else (an apostrophe, a space, an accented letter) gets a new, empty folder, `users/<old name>__<10 hex digits>/`, and a new SharePoint folder of the same name under `conversations/`; their earlier conversations are still in the old folder, `users/<old name>/document_review/` on the host and `conversations/<old name>/document_review/` on SharePoint, and are not moved, because another person may have shared it. Ask the app owner to copy what is yours. For the app owner: a tester with punctuation in the name can be given a key under an exact secret name, shown in the refusal, in Connection diagnostics and by `python -m aec_document_review.tinyapps_keys "DOMAIN\user"`; existing secrets keep working. One tester's spent or refused key no longer lowers the image size of other testers' looks.

## For the lead

1. **`tinyapps/TINYAPPS.md` (task G's file).** Its key section, at "would share a secret name; give either a `PROMPTER-KEYS` ..." (about line 206), needs this in place of that sentence:

   > Sign-in names that differ only in punctuation or accents (`CORP\j.doe`, `CORP\j-doe`, `CORP\j_doe`; `o'brien` and `o_brien`) give the same `<TESTER>`. So a name holding anything other than letters and digits also has an exact secret name, `PROMPTER-API-KEY--<TESTER>--<HASH>`, which the app looks up first; then that person's `PROMPTER-KEYS` entry; then `PROMPTER-API-KEY--<TESTER>`, so a key set up earlier still works. To give two such testers different keys, give each the exact secret (or a `PROMPTER-KEYS` entry) and delete the plain `PROMPTER-API-KEY--<TESTER>` once both are served: while it exists, whichever of them has neither is served by it. The app shows the exact name, never a key: in the refusal a tester without a key sees, in the sidebar's Connection diagnostics, and from `python -m aec_document_review.tinyapps_keys "CORP\j.doe"`. As an app setting or `.env` line the name is spelt with underscores (`PROMPTER_API_KEY__CORP_J_DOE__<HASH>`). A tester's own model and URL sit beside the key under the same ending.

   The sentence before it there, "A name longer than 127 characters is never looked up", is no longer whole: such a name is now looked up by its exact secret name, whose readable part is cut to fit.

   The same sentence belongs in `tinyapps/wrapper_repo/README.md` (its `PROMPTER-API-KEY--<TESTER>` row) and `packages/aec-document-review/README.md` (its settings table), both task G's.
2. **Live check on the test host, two colliding sign-in names** (for example a test account `x.y` and one `x_y`, or set the header by hand on dosdev's proxy): (a) with only `PROMPTER-API-KEY--<DOMAIN>-X-Y` in the vault, both are served and Connection diagnostics shows each one's exact name; (b) add the exact secret for the second with a different key: diagnostics shows "key source" per-tester for both and the second's budget is its own; (c) each has its own `users/...` folder and its own SharePoint `conversations/<key>/` folder; (d) check Key Vault accepts the exact name (letters, digits, hyphens, at most 127: it should).
3. **Live check of the probe** needs a key the service refuses: with one tester on a revoked or spent key and another on a good one, the second tester's first look after the first's failure should show no "could not be measured" line. Also read what the Prompter really sends for a spent budget and for a key with no access; if it is not a 401, a 403 or one of `error_text`'s budget signals, `key_refusal` will not see it and the old hold applies.
4. **Testers whose folder moves.** If any pilot tester's sign-in name holds an apostrophe, a space, an accented letter or a leading or trailing underscore, their past conversations stay in the old folder. Nothing copies them.
5. **GSE's own copies** (`webapp/identity.py`, `webapp/tinyapps_keys.py`) are unchanged and still have both defects. They matter only where GSE's two-page app runs behind the sign-in header (`Identity.multi_user`, source "header"): the Tiny Apps wrapper running the GSE package. Databricks (the launcher's email), Foundry (one person at a time) and a laptop are single-user and use neither the per-tester keys nor the per-user folders. The probe fix is in the core, so GSE's pages have it.

## Noted, not fixed

- `error_text` is read through its public `budget_exhausted` and `auth_refused`; the 403 test in `key_refusal` reads the status itself, because the core has no public classifier for "no access" (`error_text.py` is task C's file this round).
- A refused probe's `error` text is kept as it came, cut to 300 characters, as for any failed probe. `error_text.mask_secrets` is not applied to it (before this change either).
- A tester on a spent key now probes again at every look (four refused calls) instead of once in two minutes. Each of their looks fails at the vision call anyway.
- `cached_profiles()` (the run record's list) no longer holds a probe that failed on a refused key; the look's own result still carries the note.
