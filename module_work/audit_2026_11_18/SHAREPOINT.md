8 findings, 4 fixed here, 4 suspected and not reproduced

# Audit: the testers' SharePoint store (Document Review, 2026-11-18/19)

Branch `claude/audit-sharepoint` from master 3319d3a. Offline only: Graph played by the `urlopen` stand-in of
`tests/test_ta40_sharepoint_off_turn.py` and by probe scripts with the same stand-in; made-up secrets only.
New tests: `packages/aec-document-review/tests/test_audit_sharepoint.py` (4 tests, each fails before its fix).

## Findings, by how often they would be met

| # | File, function | Made-up situation | What happens | What should | |
|---|---|---|---|---|---|
| 1 | `sharepoint_store.py`, `SharePointStore._bg_run` | A turn's background save meets one 503 (or 429, a timeout, a reset); the tester then stops for the day | One 503 sets a hold of at least 30 s in the ONE process; the rest of that save, and every tester's calls meanwhile, are refused unsent. Nothing tried the save again until that conversation's next turn: the last turn stayed on the host's disk only | The store tries a failed background save again by itself | fixed 81ac47e (3 tries, 60/120/240 s, never before the service's wait is over) |
| 2 | `graph_sharepoint.py`, `GraphFileManager._send` (`timeout=300.0`) | The network to the service stalls (no answer, no reset) while a tester presses "Show past conversations" or restores one | Each call waits up to 300 s (the token 30 s) and none is retried; a listing is 4 to 5 calls, all on the page's own run under a spinner: the tester's page waits up to about 20 minutes | A page-side call gives up in tens of seconds and says so | noted (a timeout is the lead's call; uploads in parts need the long one) |
| 3 | `sharepoint_store.py`, `SharePointStore.not_kept_reason` | Three of the four app-registration settings set; the secret's name misspelled or its value empty | The store is not configured, nothing is saved to SharePoint, and the page said nothing (the "NOT kept" line spoke only on Foundry); only Connection diagnostics named it | The page's "Conversations are NOT kept across restarts" line names the missing setting | fixed 045018d |
| 4 | `graph_sharepoint.py`, `GraphFileManager.ls` | The site address has a mistyped path (the site lookup answers 404) | `ls` read that 404 as an empty folder: past conversations "none", the agent's list "empty or missing folder", while every save failed | A failed listing that says why | fixed 97a6459 (a token endpoint's 404, from a wrong `GRAPH_LOGIN_URL`, still reads as empty: noted, not fixed) |
| 5 | `sharepoint_tools.py`, `_conversations_needle` | (held: detail in the session's last message, not here) | The check that keeps each tester inside their own conversation folders missed some spellings of a path under one base-folder setting | (as stated) | fixed fa8504a |
| 6 | `graph_sharepoint.py`, `GraphFileManager.download_file` / `sharepoint_tools._fetch` | The agent is asked for a 3 GB file on the site | No size check before a download: the whole body is read into memory in the process all testers share, then written | A size check against a limit before reading, as uploads have | noted (a new limit is a setting default) |
| 7 | `graph_sharepoint.py`, `_request` / `_get_token_locked` | A proxy or gateway answers 200 with an HTML page | `json.loads` raises `JSONDecodeError`, not `GraphError`: the tester reads "JSONDecodeError: Expecting value"; a small upload the service did take counts as failed and is sent again | A `GraphError` naming the non-JSON answer | noted |
| 8 | `graph_sharepoint.py`, `_send` | 500 on any call; 503 on a POST (new folder, upload session, token) | Not retried at all; the save fails at once | Retried as a GET or PUT is | noted (finding 1's retry now covers the save) |

## Question 1: each failure (list, read, create folder, small upload, large upload in parts, token)

| Failure | Retried? how long | What the tester is told | Work kept? |
|---|---|---|---|
| 401, token expired | Graph calls: a new token and one more send; upload-session piece and the token request: not retried | sidebar: the save's error and "SharePoint refused the app (401/403)…"; agent tool: "REFUSED … not a missing file" | on the host's disk; SharePoint copy at the next save (now also by itself, finding 1) |
| 403 | no | same as 401 | as 401 |
| 404 | list: `[]` (folder missing; site 404 now an error); read: error; uploads: folders made once, sent again; folder POST: error | read: "…failed (HTTP 404…)"; list: empty folder | as 401 |
| 409 name exists | upload: no, returns False (caller picks another name); folder: looked up first, so not met | nothing (handled) | kept |
| 429 with Retry-After <= 20 s | up to 4 sends, each wait as named (at most 60 s) | after the last: "throttling (429)" | as 401 |
| 429 with Retry-After > 20 s | not retried; every call in the process refused unsent for min(max(RA, 30), 300) s, for ALL testers | "SharePoint asked the app to wait … about N s more" | as 401 |
| 429 without Retry-After | 4 sends, waits 1, 2, 4 s; then a 30 s hold for all | as above | as 401 |
| 500 | no (large upload: the piece is resumed, up to 5 times, 2 s apart) | "…failed (HTTP 500…)" | as 401 |
| 503 | GET/PUT: 4 sends, 1, 2, 4 s, then a 30 s hold for all; POST and token: no | as 500 | as 401 |
| timeout | no (large upload: resumed up to 5 times); each call waits up to 300 s (token 30 s) | "…failed: TimeoutError" | as 401; the page can wait (finding 2) |
| reset in a large upload | resumed from the session's `nextExpectedRanges`, up to 5 times | nothing when the resume succeeds | kept |
| not JSON | no | "JSONDecodeError" (finding 7); a read returns the page's bytes as the file | as 401 |

No call loops: the longest a turn's tool can wait on throttling is about 60 s plus four sends; saves run off the turn.

## Questions 2 to 6, in one line each

- Q2: a listing of 2,000 folders in pages of 200 (`@odata.nextLink`) came back whole, 13 calls. Files named with a space, `#`, `%`, `'`, `é`, a trailing dot and 250 characters were saved and restored under their own names (the trailing dot under `trail_<hash>`, mapped back by `sp_names.json`). A 300-character name cannot exist on the host's disk (Linux allows 255 bytes), so it never reaches the store.
- Q3: `DOMAIN\user` and `user@domain` in any case give one key (`corp__jdoe`); empty, blank or slashed names are refused before a folder is made; `..`, `.`, `__`, `'`, spaces and accents get a hashed folder that cannot climb or meet another. Folder keys of 15 made-up names (3 more refused): all different except one person's spellings. One finding (5).
- Q4: no secret, token or upload-session query in any `GraphError`, the process log, `sharepoint_log` or a tool result in the 78 failure cases run with a made-up secret and token; only a service that echoes the secret in its error body puts it in an error (suspected list).
- Q5: tool paths: open, own, container or private, judged on every spelling a path takes; uploads take only this conversation's files. Missing file: "failed … says nothing about whether it exists"; old Office formats: named as unreadable; too large: finding 6.
- Q6: a missing, empty or misspelled setting: the page starts; before fix 3 nothing said conversations were not kept.

## Suspected, not reproduced

1. SharePoint's 400-character path limit: `<root>/conversations/<owner, <=64>/<folder, <=75>/files/<name up to 255>` can exceed it; the fake does not enforce the limit, so a file refused on every save was not shown.
2. An apostrophe in a path-addressed Graph URL (`root:/o'brien.pdf:`): the fake accepts it; the service's handling was not checked.
3. A service error body that echoes a secret would show unscrubbed in the sidebar's save and listing errors (only `sharepoint_log` and tool results are scrubbed); played only with a made-up echo.
4. `sharepoint_tools._working_dir` falls back to the shared temp folder when no working folder is bound; no turn path found where that happens.

## What was run (exit codes)

- `probe.py` (78 failure cases x 6 calls, made-up secret): ran, 0. `ident.py`, `scope.py`, `names.py`, `page.py`: 0.
- After each fix: `pytest tests/test_review_fingerprints.py tests/test_boundary.py`, the eight SharePoint and Graph test files, `test_tinyapps_wiring.py`, `test_tinyapps_entry.py`, `test_foundry_host.py`, `tests/test_audit_sharepoint.py`: exit 0 each time (225, 226, 277, 276 passed). Each new test failed with its fix stashed (exit 1).
- Guards from the root before each commit: `webapp/tests/test_public_text.py test_backlog.py test_docs_currency.py`: exit 0 (24 passed).
- Venv: `/tmp/dr` (Python 3.12, core and Document Review editable). Full suites not run (the gate runs on GitHub).

## Read whole, in part, not reached

- Whole: `tinyapps_entry.py`; `identity.py` (all but its first 55 lines).
- In part: `graph_sharepoint.py` (100-230, 636-1230), `sharepoint_store.py` (118-500, 593-1075, 1285-1400, 1734-1790), `sharepoint_tools.py` (86-93, 310-560, 640-730, 812-970, 1013-1130, 1240-1316), `sharepoint_log.py` (1-210), `tinyapps_settings.py` (60-181), `tinyapps_keys.py` (220-250).
- Not reached: `graph_sharepoint.py` sites, clouds, certificate and check (230-636, 1230-1614); `sharepoint_store.py` restore internals (1400-1730); `sharepoint_tools.py` upload and search bodies (1316-1762); the rest of `tinyapps_keys.py`.
