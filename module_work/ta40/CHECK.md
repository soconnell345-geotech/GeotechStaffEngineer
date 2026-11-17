# TA-40 CHECK: SharePoint saves off the turn and off the page

Branch `claude/ta-40-sharepoint-off-turn`, from master 1c69d95. Task D of
`module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`. Model: Fable
(`claude-fable-5-2`). Venv (a) only. Nothing here called a model or a real
SharePoint: every test runs against a fake Graph.

Files changed (all within task D's list, plus new files of its own):

| File | What |
|---|---|
| `packages/aec-document-review/aec_document_review/graph_sharepoint.py` | pieces 2, 3, 4, 7 |
| `packages/aec-document-review/aec_document_review/sharepoint_store.py` | pieces 1, 4, 5, 6, 7 |
| `packages/aec-document-review/aec_document_review/sharepoint_tools.py` | two small parts of pieces 2 and 6 |
| `docs/SHAREPOINT_SETUP.md` | a new step 8 and one settings row |
| `packages/aec-document-review/tests/test_ta40_sharepoint_off_turn.py` | new, 32 tests |
| `packages/aec-document-review/tests/test_tinyapps_wiring.py` | one fake's address parsing (below) |
| `packages/aec-document-review/tests/test_ta40_owner_folder.py` | new with the review fixes, 5 tests (fix 5 below) |

`sharepoint_log.py` is not changed. GSE's copies (`webapp/sharepoint_store.py`,
`webapp/graph_sharepoint.py`) are not changed.

All seven pieces are repairs; none sits behind an on/off setting. One
tunable was added: `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB`, default 2048 (piece 4).

## Review fixes (2026-11-13)

1. `graph_sharepoint._send`: a timeout, any other `OSError` or an
   `http.client.HTTPException` becomes a `GraphError` with no status, so a
   timed-out piece of a large upload is resumed from the session's position.
2. `sharepoint_store._upload_changed`: a file held back for its size is
   skipped only while it is above today's limit; when the limit is raised it
   is sent and no longer named in the note.
3. `sharepoint_store._steady_meta`, used by `folder_name`, `_pin_folder`,
   `session_folder` and `_mirror_locked`: the page and the turn write
   `meta.json` in place, and a save in the background can meet it half
   written. EVERY read the save makes of it is made again while the file
   exists but does not load (up to four reads, 0.1 s apart). The first form
   of this fix checked once, at the start of the save; the reads that make
   the folder came after the check and were still open, so one of them
   taken as "no meta" still filed the conversation outside its owner's
   folder (`test_a_record_half_written_at_any_one_read_keeps_its_folder`
   fails on that form). If the file never loads, the save ends with an
   error and sends nothing. That check runs in `_mirror_locked` before a
   batch is opened: inside `_mirror_files` it left the Foundry dataset's
   manager with a batch begun and never ended
   (`test_a_record_that_stays_unreadable_opens_no_batch`).
4. `graph_sharepoint.upload_file`: the 404 fallback writes one `log.warning`
   (no file, folder or person named).
5. **One person, one owner folder** (`sharepoint_store.owner_folder`; found
   by the review of TA-39).

   **Confirmed.** `conversations_base` made a person's folder,
   `<root>/conversations/<owner folder>/<page>/`, as
   `sanitize_folder_name(key)` of their folder key (`Identity.key`,
   `<domain>__<user>`). That function turns `__` into `_`, drops dots at
   the ends and cuts at 64 characters, so these pairs of sign-in names
   shared one folder, one listing and one restore: `corp_a\b` and
   `corp\a_b` (keys `corp_a__b`, `corp__a_b`, both `corp_a_b`);
   `corp\jdoe.` and `corp\jdoe`; two long names that differ after the
   64th character; a key that carries a hash and a name made up to spell
   it. Over about 11,000 made-up keys, 1,378 folders were shared.

   **The rule.** A plain key keeps the folder it has today; every other
   key gets its letters, digits and dots (`-` for the rest, at most 52
   characters), then `__` and 10 hex digits of a SHA-256 of the exact key:
   `corp_a__b` -> `corp-a-b__<hash>`.
   - Plain: `<domain>__<user>` in lower case from `a-z 0-9 . -`, with no
     underscore in the domain and only single underscores inside the user,
     and nothing else lost by the old function (no dot at either end, 64
     characters or fewer, no `_vti_`); or a key with no underscore at all.
     `corp__j.doe` -> `corp_j.doe`, `corp__john_smith` -> `corp_john_smith`,
     as before.
   - Why no two keys share: a plain key's folder gives the key back (its
     first `_` is the join), so two plain keys differ; every other folder
     holds `__`, which a plain key's folder never does; and among those the
     hash of the exact key tells them apart (40 bits). Names are compared
     without case, as SharePoint does: a plain folder is all lower case.
   - It reads the key only, whatever made it: it holds for the keys of
     this branch's `identity.py` and for TA-39's.

   **Every place that builds or reads the owner folder** goes through
   `conversations_base`, so all agree: the save (`session_folder`), the
   folder-name check (`_pin_folder`, `_unclaimed_remotely`), the listing,
   the restore, the folder link, and the SharePoint tools' own-folder rule
   (`sharepoint_tools._scope`). `listing_cache_key` (the sidebar's cache of
   a listing and the store's record of a failed listing) used the old
   function too and now uses the same one, so one person's listing failure
   is not shown as another's. `_log_save_errors` reads whatever follows
   `/conversations/`. `app.py` hands the key and spells no folder.

   **Who moves.** Only a person whose key is not plain: an underscore in
   the DOMAIN (`corp_a\b`); a doubled underscore in the user part, which
   on the TA-39 branch is every name that carries a hash (an apostrophe, a
   space, an accented letter; TA-39 moves those keys anyway) and before it
   a name with `__` or with an odd character beside an underscore; a name
   that ends in a dot; a key over 64 characters; `_vti_` in the name. A
   name of letters, digits, dots and hyphens, with single underscores in
   the user part, does not move.

   **What a person who moves sees.** What the store already does when a
   conversation's folder changes (a changed root or owner): at each of
   their conversations' next save on this host the whole conversation is
   sent to the new folder, MOVED.txt is left in the old one, the sidebar
   says "Renamed — the folder before the rename keeps a copy and a
   MOVED.txt pointing here", and the folder link is the new one. A
   conversation of theirs that is only in SharePoint (not on the host's
   disk) stays in the old folder: it is no longer in their "Find a past
   conversation" list, and the SharePoint tools now treat the old folder
   as another person's. The old folder is still the folder of the person
   who shares the old name and whose key is plain, so that person still
   lists what the mover left there until the host's administrator moves or
   deletes it. Nothing is moved or deleted in SharePoint by the app.

   **Tests** (`tests/test_ta40_owner_folder.py`; each fails before the
   fix): the pairs get two folders, and a name made up to spell another
   key's folder gets another; about 11,000 made-up keys get as many
   folders, each a name SharePoint accepts, none over 64 characters; plain
   keys keep `sanitize_folder_name`'s folder and the sidebar's cache key
   (a guard: it fails before the fix only because `owner_folder` is not
   there); for a pair, each one's listing holds their own conversation
   only, a restore of the other's by name finds nothing, the tools' rule
   calls the other's folder private, and a failed listing is one person's;
   a mover's next save sends everything to the new folder with MOVED.txt
   in the old one, and the plain key of the pair does not move.

   **Runs** (this laptop, the package's tests from its folder, by pytest's
   exit code): `tests/test_ta40_owner_folder.py`,
   `test_ta40_sharepoint_off_turn.py`, `test_tinyapps_wiring.py`,
   `test_sharepoint_store.py`, `test_sharepoint_tools.py`,
   `test_sharepoint_privacy.py`: 162 passed, 1 skipped (the refused-name round trip, which Windows cannot hold), exit 0; and four more files that use what changed, `test_wave2a_bookkeeping.py`, `test_sharepoint_ta35.py`, `test_sharepoint_links_and_failures.py`, `test_foundry_host.py`: 79 passed, exit 0. The seven tests added with
   fixes 3 and 5 all fail on the commit before them (021aac1's store).

## Piece by piece

### 1. Saves ran inside the turn under one process-wide lock

**Confirmed on master.** `SharePointStore.mirror_conversation` took
`self._lock` (one `threading.Lock` per store, and the store is process-wide)
around the whole mirror. `SharePointStore.background` was True only for the
Foundry backend, so `save_after_change` (called by `turn_jobs.py` before the
job's `_finish`) ran the mirror at once, inside the turn. "Sync now" in
`app.py` calls `mirror_conversation` in the page's run.

**Changed** (`sharepoint_store.py`):
- `SharePointStore.per_conversation` (new): True when the file manager says
  `PER_CONVERSATION` (the Graph one, `GraphFileManager`), or, before the
  manager is built, when the Graph settings are complete and Foundry's
  dataset is not selected.
- `SharePointStore._lock_for(thread_id)` (new): the conversation's own lock
  where `per_conversation`, else the store's one lock.
  `mirror_conversation` takes that.
- `SharePointStore.background`: now also True where `per_conversation`, so
  `save_after_change` queues the save on the Graph route too. Neither
  `turn_jobs.py` nor `app.py` was edited: the turn gets `None` back, and the
  sidebar's existing status fragment reads `save_state` and
  `last_sync_for`, as it does on Foundry.
- `save_soon` / `_bg_take` / `_bg_run` / `save_state` / `flush`: up to
  `BG_WORKERS` (4) workers where `per_conversation`, one elsewhere; a worker
  never takes a conversation another worker is saving.
- `_pin_folder`: the choice of a NEW folder name is made under a short lock
  (`_pin_lock`) and held as a claim until it is in the conversation's meta;
  the remote check (`_unclaimed_remotely`, one listing) runs outside the
  lock. Without this, two conversations with the same title first mirrored
  at the same moment could both take one folder (the one lock used to
  prevent that).
- `GraphFileManager._get_token` renews under a lock, since several threads
  now use one manager.

**Not changed:** the Foundry dataset and the Funhouse SDK's file manager
(and any test stand-in) keep the one lock, one worker, and for the SDK's
manager the save inside the turn.

**"Sync now" still needs a change in `app.py` (task C's file).** It now
waits only for its own conversation's mirror (never another person's), but
it still runs that mirror in the page's run, so a large first mirror holds
the clicker's own page under the spinner. The change, at the "Sync now"
button:

```python
if st.button("Sync now", key=f"spsync_{ss.thread_id}"):
    if getattr(_sp, "background", False) is True:
        sharepoint_store.save_after_change(_sp, ss.thread_id)   # queued
    else:
        with st.spinner(f"Saving to {_sp_label}…"):
            ss.sp_sync = _sp.mirror_conversation(ss.thread_id)
    ss.pop(sharepoint_store.listing_cache_key(**_sp_where), None)
    st.rerun()
```

The status fragment then shows "saving the latest turn…" and the result.
Two comments in `app.py` are now out of date and are also task C's to fix:
`_storage_busy` ("Never on SharePoint, which saves within the turn") and
`_storage_panel` ("Foundry only, in practice").

### 2. Each upload walked the folder chain

**Confirmed on master.** `GraphFileManager.upload_file` called
`_ensure_folder(drive_id, folder)`: one GET of the library root, then one
children listing (all pages) per level, for every file. The store's own
`_ensure_folder` also called `create_folder` once per new directory per
process, which walks the same way.

**Changed.** `GraphFileManager.upload_file` / `_upload_by_path`
(`graph_sharepoint.py`): `PUT drives/{id}/root:/{path}:/content`, and
`POST drives/{id}/root:/{path}:/createUploadSession` above 4 MB; the
service makes the folders. If the PUT answers 404, the folders are made once
(the old walk) and the file is sent again, in case a tenant does not make
them. `GraphFileManager.MAKES_FOLDERS = True`; where a file manager says so,
`SharePointStore._upload_changed` and the upload tool
(`sharepoint_tools._upload`'s `fm.create_folder(folder)`) no longer make
the folders first.

Measured on the fake: a 40-file first mirror of a titled conversation is 45
calls (sign-in, site, library, the folder-name check, and 41 uploads: the
40 files and `meta.json`), with no folder made and none listed. The
review's count for master is about 300.

### 3. A large upload had no retry of a failed piece

**Confirmed on master.** `_upload_large` sent each piece with `_send` and
let the first failure end the upload.

**Changed.** `GraphFileManager._upload_large` / `_session_next`: after a
failed piece it asks the session where it stands (`GET uploadUrl`,
`nextExpectedRanges`) and goes on from that byte (for a cut connection, an
HTTP 5xx and, since the review fixes, a timeout), up to `RESUME_TRIES` (5)
times per upload, `RESUME_PAUSE_S` (2 s) before each. Not tried again: 401,
403, 404, 409, 429 (and a 503/504 that `_send` has already waited out sets
the hold, so the session cannot be asked and the piece's own error is
raised). The error text still never carries the session's token.

### 4. Whole files held in memory, no size limit

**Confirmed on master.** `upload_file` did `content = fh.read()` for every
file, whatever its size.

**Changed.** A file up to 4 MB is read whole (bounded); a larger one is read
one `CHUNK_SIZE` piece (3.2 MB) at a time inside `_upload_large`. A file
above `max_upload_bytes()` is not sent: `FileTooLarge` (a `GraphError`)
names the file, its size and the limit, before anything is asked of the
service. In the mirror (`_upload_changed`, `_mirror_files`): the first sync
reports it in `errors` (so it is counted, shown and logged once); the
manifest records it under `not_sent` with its stamp; later syncs do not try
it again until the file changes or the limit is raised, do not count it as
failed, and name it in
`summary["note"]` (which the sidebar shows) and `summary["not_sent"]`.

Downloads are still read whole into memory (`download_file`); the item is
about uploads. Noted below.

### 5. A listing that fails read as "nothing mirrored yet"

**Confirmed on master.** `list_remote_conversations` returned `[]` from
`except Exception`.

**Changed.** It returns a `RemoteListing` (a `list`): empty with
`error = "SharePoint could not be listed: <type>: <message>"` when the
library could not be read, `error = None` when it was read (a missing
conversations folder is still an empty list with no error). The store keeps
the reason per person and page (`listing_error(owner, page)`) until a
listing succeeds, and writes it to the SharePoint log.

**The sidebar's text is in `app.py` (task C's) and is not changed.** It
still shows "Nothing mirrored yet (or SharePoint could not be listed…)".
The change needed there, where that caption is written:

```python
_sp_list_error = getattr(ss.get(_sp_list_key), "error", None)
if _sp_list_error:
    st.warning(_sp_list_error + " Refresh to try again.")
elif _sp_listed and not _remote:
    st.caption("Nothing mirrored yet.")
```

"In the summary": a restore whose listing fails already reported the error
in its summary (`restore_conversation` catches what `_walk_remote` raises);
left as it was.

### 6. A refused file name failed the same way at every sync

**Confirmed on master.** `_upload_changed` sent the local name as it was;
the error came back at every sync (the file is never stamped).

**Changed** (on the SharePoint backend only, not the Foundry dataset):
`remote_name(name)` maps a refused name to an accepted one, the same on
every host and with no need of the manifest: each refused character becomes
`_`, spaces at the ends and dots at the end go, and six characters of a
SHA-1 of the original name go before the extension. `remote_rel` does that
per path segment. `_upload_changed` uploads under the kept name; the
manifest records `names` (own path -> kept path); the summary's `note` says
so the first time; `sp_names.json` is written into the conversation's
remote folder (`_send_names`; sent only when it changed) so a restore on
another host (`_restore_from`, `_read_names`) gives each file its own name
back. `mirrored_rel(conv_dir, rel)` gives the kept path, used by the
mirrored-copy link in `sharepoint_tools._share_mirrored_copy`.

Refused: one of `" * : < > ? / \ |` or a control character, a space at
either end, a dot at the end, the reserved names (`CON`, `PRN`, `AUX`,
`NUL`, `COM0`-`COM9`, `LPT0`-`LPT9`, `.lock`, `desktop.ini`, also as the
part before the first dot), a leading `~$`, `_vti_` inside.

### 7. Throttle waits under the process-wide lock

**Confirmed on master**: `_send` sleeps (up to three waits of at most 20 s)
inside the mirror, which held the one lock. **After piece 1** the sleep is
taken under the throttled conversation's own lock; no code change in `_send`
beyond the comment on `BUSY_HOLD_*`. The hold that follows a throttle that
did not clear (later calls refused unsent for 30-300 s) is kept and still
applies to the whole manager, since the service throttles the app
registration as a whole; see "Calls made".

## Before and after (made up)

Two testers, `corp\j.doe` and `corp\a.lee`. `j.doe` attaches a 60-sheet set
and asks a question; `a.lee`'s short turn ends ten seconds later.

- Before: `j.doe`'s turn stays at "running" for the minutes its first mirror
  takes (about 7 calls per file). `a.lee`'s finished turn also stays at
  "running" until `j.doe`'s mirror is done. `j.doe`'s file
  `plan: rev 2.pdf` fails at this and every later sync with HTTP 400.
- After: both turns end when their answer is written. The sidebar says
  "SharePoint: saving the latest turn…" for each, then "61 uploaded".
  `a.lee`'s save runs while `j.doe`'s is still going.
  `plan: rev 2.pdf` is kept as `plan_ rev 2_<six characters>.pdf`, with a
  note under the save line the first time.

## Tests

New: `packages/aec-document-review/tests/test_ta40_sharepoint_off_turn.py`,
32 tests (17 of them the cases of one parametrized test). On master's code
(the three package files and the edited fake stashed) all 32 fail; on the
branch all 32 pass.

| Test | On master it fails because |
|---|---|
| `test_two_conversations_mirror_at_once` | the fake Graph answers uploads by path only; master uploads by parent id, so both mirrors report errors. (With a fake that also answered the old form it would fail on the lock, as the next two do.) |
| `test_save_after_a_turn_is_queued_not_run_in_the_turn` | `store.background` is False |
| `test_a_throttled_wait_holds_only_its_own_conversation` | the test's throttle is on an upload by path, which master never sends: the wait never starts |
| `test_two_new_conversations_of_one_title_get_two_folders` | "the second mirror waited on the first's" (the one lock) |
| `test_other_file_managers_keep_one_lock_and_save_in_the_turn` | no `per_conversation` (a guard of the unchanged route) |
| `test_first_mirror_of_40_files_walks_no_folders` | uploads by parent id after walking the chain |
| `test_the_upload_tool_makes_no_folders_first` | no `MAKES_FOLDERS`; folders walked and made |
| `test_a_failed_piece_is_resumed` | session made by parent id; no resume |
| `test_a_piece_that_keeps_failing_ends_the_upload` | the same |
| `test_a_large_file_goes_from_disk_in_pieces` | the same (and the file is read whole) |
| `test_a_file_above_the_limit_is_named_once_and_skipped` | no limit: the file is sent |
| `test_a_failed_listing_reads_as_a_failure` | a bare list has no `error` |
| `test_a_refused_name_maps_to_an_accepted_one` (17 cases), `test_accepted_names_are_left_alone` | no `remote_name` |
| `test_a_refused_name_does_not_fail_twice` | the name is refused at each sync |
| `test_a_conversation_with_plain_names_has_the_manifest_it_always_had` | uploads by parent id (a guard: the manifest keeps its two keys) |

Existing test edited (one): `tests/test_tinyapps_wiring.py`, the fake
Graph `_FakeGraph`'s PUT branch read the file name from the old address
(`/items/{parent}:/{name}:/content`); it now reads it from the path
(`/root:/GSE/{name}:/content`). No assertion of the test changed.

Runs, venv (a), from `packages/aec-document-review`,
`MPLBACKEND=Agg /tmp/dr/bin/python -m pytest -q -p no:cacheprovider …`,
each judged by pytest's exit code:

| Files | Result |
|---|---|
| `tests/test_ta40_sharepoint_off_turn.py` on master's code | 32 failed, exit 1 |
| the new file, `test_sharepoint_store.py`, `test_sharepoint_tools.py`, `test_sharepoint_privacy.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_ta35.py`, `test_graph_sharepoint_setup.py`, `test_past_conversations_pages.py`, `test_foundry_host.py`, `test_tinyapps_wiring.py`, `test_wave2a_bookkeeping.py`, `test_wave2c_fixes.py`, `test_boundary.py`, `test_review_fingerprints.py` | 295 passed, exit 0 |
| `test_output_capture.py`, `test_titles_cards_scratch.py`, `test_record_ev06.py` (the other tests that name the changed modules) | 41 passed, exit 0 |
| guards, from the root: `/tmp/dr/bin/python -m pytest -q --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | 13 passed, exit 0 |

The full suites were not run here (the lead's `ci/` push).

## Fingerprints

Not moved. `tests/test_review_fingerprints.py` passes unchanged; no core
file was changed, so GSE's page fingerprints were not run.

## Consumers checked

- `save_after_change`: `turn_jobs.py` (the turn) and `app.py` (a feedback
  note); both already handle `None` (the Foundry path).
- `mirror_conversation`: `app.py` "Sync now" and
  `sharepoint_tools._share_mirrored_copy` (inside a tool, which needs the
  result); both still run at once, under their conversation's lock.
- `list_remote_conversations`: `app.py` only; a `RemoteListing` is a list.
- `upload_file` on the Graph manager: the mirror, the upload tool,
  `sharepoint_log._write`, `graph_sharepoint.check_connection` (which uses
  `_ensure_folder` itself, kept). The core's `run_mirror.py` drives the
  Funhouse SDK's manager, not this one.
- The store's `_bg_saving` is now a set and `_bg_worker` became
  `_bg_workers`; nothing outside the store read either.

## Calls made

1. **Four background workers** (`BG_WORKERS`). A fifth conversation's save
   waits for a free worker; no turn waits. A thread per save would have no
   bound.
2. **Background saves for the Graph route only.** The Funhouse SDK's
   SharePoint manager (one user, thread safety unknown) keeps the save in
   the turn and the one lock.
3. **The size limit: 2048 MB**, tunable by
   `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB`. High enough for the several-hundred-MB
   file of the live check; at about 3 MB a request it bounds one file to
   some hundreds of requests. A tunable rather than a constant so the lead
   can see the message on a small file.
4. **Too large: an error once, then a note at every sync** (not an error at
   every sync, which would show "1 failed" for ever; not silence).
5. **Refused names are mapped, not skipped**, and the hash is always added
   to a mapped name, so the result does not depend on what else is in the
   folder or on the manifest.
6. **`sp_names.json` in the conversation's remote folder**, so a restore
   gives files their own names back. Existing mirrors have none and restore
   as before. An older app version restoring such a mirror gets the kept
   names and `sp_names.json` as an ordinary file.
7. **Five resumes per large upload**, two seconds apart; a throttled piece
   is not resumed.
8. **The 404 fallback** in `upload_file` (make the folders once, send
   again): the service making folders on a path upload could not be checked
   here.
9. **The hold after a throttle that did not clear stays process-wide**
   (`_busy_until`): the other conversations' saves in the next 30-300 s fail
   unsent, are reported, and are sent at their next sync. Sending them
   would only be throttled again.
10. **`RemoteListing`** (a list with `error`) rather than a changed return
    shape, so `app.py` works unedited.

## Release note

SharePoint on the testers' host: the save after a turn now runs in the
background, so a turn ends when its answer is written and one person's large
first save no longer holds anyone else's turn; the sidebar shows "saving the
latest turn…" and then the result. A first save of 40 files is about 45
requests where it was about 300. A large file is sent in pieces from disk and a failed
piece is taken up again. A file above 2048 MB
(`GEOTECH_SHAREPOINT_MAX_UPLOAD_MB`) is not sent and is named. A file whose
name SharePoint refuses (a colon, a question mark and the like) is now
saved, under a nearby accepted name. Two sign-in names can no longer share
one SharePoint folder of conversations: a name with an underscore in its
domain, a doubled underscore, a dot at its end or more than 64 characters
gets a folder of its own (its conversations on the host are sent there at
their next save, with MOVED.txt left in the old folder); every other
tester's folder stays where it is. A GSE user notices nothing: GSE's own
copies are unchanged.

## For the lead

**Live checks on a real tenant** (settings as `docs/SHAREPOINT_SETUP.md`;
watch the sidebar's Permanent storage block and
`<root>/_app_log/sharepoint-YYYY-MM.log`):

1. *A 40-file first mirror.* Attach a 40-sheet set in a new conversation
   and ask one question. The turn ends without waiting for the save; the
   sidebar goes from "saving the latest turn…" to "N uploaded, 0 failed".
   In SharePoint the folder `conversations/<owner>/document_review/<title>_<date>/files/…`
   exists with its subfolders, though the app never made them: **this is
   the check that the service makes folders on a path upload.** If the
   process log shows the 404 fallback instead, say so.
2. *A file of several hundred MB.* Attach one (under the 2048 MB limit).
   It arrives whole (compare sizes); the host's memory does not rise by the
   file's size. If the network can be interrupted for a few seconds
   mid-upload, the upload should go on and finish.
3. *The limit.* Set `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB=1`, attach a 2 MB
   file: the first save shows one failure naming the file and "1 MB limit";
   the next turn's save shows no failure and the blue note still names it.
4. *A refused name.* Have the agent save a file named with a colon or a
   question mark (a Linux host). It appears in SharePoint as
   `<name>_<six characters>.<ext>` with `sp_names.json` beside `meta.json`;
   the second save uploads nothing for it. Restore the conversation on a
   clean host: the file has its own name.
5. *Two testers at once.* Tester A starts a first mirror of a large set;
   tester B finishes a short turn meanwhile. B's turn ends at once and B's
   save line appears while A's still says "saving".
6. *A failed listing.* With the site grant removed (or the secret wrong),
   open "Find a past conversation": until `app.py` is changed the caption
   is the old one; the log has a "list conversations" line.

**`app.py` changes left for task C or the lead** (both written out above):
"Sync now" through `save_after_change`; the listing's `error` in the
sidebar; two out-of-date comments.

**`tinyapps/wrapper_repo/.env.example` and `tinyapps/TINYAPPS.md`** (task
G's files) do not list `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB`.

**GSE's copies** (`webapp/sharepoint_store.py`, `webapp/graph_sharepoint.py`)
keep the old behaviour until they leave with INF-35, the shared owner
folders included (review fix 5).

**After TA-39 is merged with this branch:** run TA-39's pairs over
`sharepoint_store.owner_folder(Identity.key)` once (the rule reads any key,
but the two were never run together here).

## Noted, not fixed

- A file held back for its size whose limit is changed but is still above
  it keeps the note with the limit it was first refused under.
- What a person who moves to a new owner folder (review fix 5) left in the
  old one stays there, listed by the person who keeps that folder; the app
  does not sort an old shared folder by each conversation's owner.
- `GraphFileManager.download_file` reads a whole file into memory (a
  restore of a several-hundred-MB file, the download tool).
- A failed large upload leaves its upload session to expire by itself (no
  DELETE of the session).
- A queued save is lost if the process is stopped more than a minute into
  the exit flush, and a save still queued is lost at once when the process
  is killed; the files are not stamped, so the next sync sends them. After
  the restart the sidebar says only that mirroring is on, and that
  conversation's next turn or "Sync now" sends it. A conversation never
  opened again after that is not saved until then.
- If a large upload's last piece arrives but its answer is lost, the upload
  is reported as failed and the file is sent again at the next sync.
- A name SharePoint refuses for a reason outside the list above (a path
  over 400 characters) still fails at every sync, with its error shown each
  time.
- `_unclaimed_remotely`: when the library cannot be listed for the
  folder-name check, the conversation takes the name with its thread-id
  shard, as before, and the summary does not say the check failed.

## The page edits (2026-11-13)

Applied on task C's fix branch (`fix/ta-37-review`, which holds `app.py`)
after master, with this task's merge, was merged into it. DR `app.py`:

- **"Sync now"** goes through `sharepoint_store.save_after_change` where
  the store's `background` is True (Foundry, and SharePoint by the Tiny
  Apps route): the sync is queued and the status lines follow it, as after
  a turn. Elsewhere (the Funhouse SDK's SharePoint) it mirrors at once
  under the spinner, as before. On Foundry this changes too: the click no
  longer waits for the save.
- **"Find a past conversation"**: a listing whose `RemoteListing.error` is
  set shows that reason as a warning with "Refresh to try again."; an
  empty listing that was read says "Nothing mirrored yet." (live check 6
  above now sees the reason).
- **The two comments** in `_storage_busy` and `_storage_panel` now say
  saves run in the background on Foundry and on SharePoint by the Tiny Apps
  route.

Tests: `tests/test_ta40_page_edits.py` (new, 4 tests, the real page through
AppTest over a stand-in store): a background store's "Sync now" is queued
and not mirrored, a store that is not background mirrors at once, a failed
listing shows its reason and not "Nothing mirrored yet", an empty one says
"Nothing mirrored yet.". GSE's copy (`webapp/app.py`) is not changed.
