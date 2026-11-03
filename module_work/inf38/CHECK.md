# INF-38: conversations that survive a rebuild on Foundry (workspace dataset library)

Built 2026-11-03 on branch `wip/inf38-transforms`. It replaces the backend
merged that morning, which drove the dataset through `foundry-platform-sdk`.
The AI FDE then tested from a real published app and found that route closed
inside app containers: the platform's datasets API answers
`403 FoundryContainerService:EndpointDisabledForContainers` through the
local connection, and the public host does not resolve. What works is the
workspace dataset library, `foundry.transforms.Dataset`, found by the
dataset's **workspace alias**. The calls below are the ones the FDE ran and
timed. The app's own class has run them only against a fake that models
what the FDE measured.

Removed with the old backend: the platform-SDK client, its host and token
lookup, the leftover-transaction abort and its marker file (the library
cannot abort a transaction), and `GEOTECH_FOUNDRY_CONVERSATION_BRANCH` (the
library takes no branch).

## What a user gets

On a Foundry deployment with `GEOTECH_FOUNDRY_CONVERSATION_DATASET` set to
the dataset's workspace alias:

- After every turn, once the record is complete, the conversation folder
  (`meta.json`, the transcript, `activity.jsonl`, `trace.jsonl`,
  `FEEDBACK.md` / `feedback.jsonl`, `files/`) goes to the dataset under
  `GeotechStaffEngineer/conversations/<title>_<date>/`, in ONE transaction,
  in the background. The chat does not wait for it.
- The sidebar's **Permanent storage** block says "Foundry storage: saving
  the latest turn…" while a save runs, then "Foundry storage: N uploaded,
  M up-to-date · Xs". **Sync now** saves at once, with a spinner, and
  reads the past-conversations list again.
- **Find a past conversation (permanent storage)** lists the dataset's
  conversation folders and restores one (record and files), downloading
  six files at a time.
- **Connection diagnostics**, line *conversation storage*: PASS with the
  alias and how many files are the app's (one listing, no write); FAIL with
  a plain reason; SKIP when the variable is not set.
- **When nothing keeps the conversations, the sidebar says so.** On master
  a Foundry app with no usable dataset drew no storage block at all; only
  Connection diagnostics said why, so the owner could take conversations
  for kept. Now, on Foundry only, the **Permanent storage** block says
  **"Conversations are NOT kept across restarts."** with the diagnostics
  line's own reason and fix (`foundry_dataset_store.not_kept_reason`, one
  text for both). The four cases:
  - no alias set: the reason, the docs section, and "Download this
    conversation" in the meantime;
  - a resource ID (`ri.…`) where the alias belongs;
  - the library missing;
  - an alias the workspace does not define (`AliasNotDefined`).
  The first three say to import the dataset into the workspace (Data
  panel, read and write), push, and set the variable to its alias; the
  library case says to publish from a Code Workspace. An unknown alias is
  found by one `Dataset.get` per process (`check_alias`), so the block is
  right from the first draw if that call resolves the alias; else the
  first listing or save finds it. A call that works clears it.
- A save that meets another transaction open retries for about 30 s, then
  says another save holds the dataset; the turn is unaffected, and the
  files go up with the next save.

Funhouse and Tiny Apps are unchanged: SharePoint still saves at the end of
the turn, and the dataset setting is read only when
`GEOTECH_DEPLOYMENT=foundry`.

**One user** (the owner's decision, 2026-11-03: the owner alone uses the
app on Foundry for about the next year). Every conversation goes in one folder,
`GeotechStaffEngineer/conversations/`, as before. No per-viewer folders
and no identity work; they wait on the app knowing who is viewing.

## The library calls the app makes

All in `webapp/foundry_dataset_store.py`: the `_lib_*` methods,
`DatasetFileManager._dataset` and `DatasetFileManager._fetch`.

```python
from foundry.transforms import Dataset      # imported lazily; missing -> storage off, diagnostics says why
ds = Dataset.get(ALIAS)                      # once per process (the store keeps one file manager)

ds.files(show_hidden_files=True)             # the whole dataset, filtered on File.path; plain files() if that keyword is refused
    # each: .path, .size_bytes, .updated_time, .download() -> local path of a copy
ds.upload_directory("/foundry/outputs")      # one save = one transaction
ds.upload_file(local_path, logical_path)     # only when the outputs folder holds something the app did not put there,
                                             # and for report ingest's mirror (no batch)
```

The rules the app keeps:

- **One save per turn, one transaction.** The store hands the save every
  changed file; each is copied at once into a private temporary folder, so
  a file that changes or is removed before the upload still goes as it was.
  At the end, under one process-wide lock, the copies are placed under
  `/foundry/outputs/<their dataset path>` and `upload_directory` is called
  on the outputs folder. In a `finally`, only the files placed there and
  the folders that held only them are removed.
- **Never upload or delete what the app did not stage.** Before placing
  anything, the app lists the outputs folder. Its own top folder
  (`GeotechStaffEngineer/`, the first segment of the store's root) can only
  be left from a save that stopped, so it is cleared and never uploaded. If
  anything else is there, or the folder cannot be used, the save goes file
  by file (`upload_file`, one transaction each) and the sidebar says so
  once per process.
- **One open transaction per dataset.** A write refused with
  `SimultaneousOpenTransactionsNotAllowed` is retried after 1, 2, 4, 8 and
  15 s (30 s in all). Then it fails with `StorageBusy`: "another save holds
  the conversation dataset … these files go up with the next save". The
  store records none of the save's files, so the next save sends them all.
- **One listing per save, per restore, per list.** `files()` has no prefix
  filter. A save or a restore reads it once (the batch) and keeps the file
  objects for its downloads. The sidebar's list is cached per browser
  session until **Refresh** or **Sync now**. An index file would not help:
  no draw needs the listing more than once.
- **Restore through the path guard.** Every local path is checked before
  any download (`_local_path_inside`: no absolute path, no `..` segment, no
  path that resolves outside the conversation's folder). Then up to six
  `File.download()` calls run at once, and each copy goes to its checked
  path.

## What each operation costs

`Dataset.get` once per process, then:

| Operation | Calls | Time, from the FDE's figures |
|---|---|---|
| Save after a turn, nothing changed | none | 0 |
| Save after a turn | 1 `upload_directory` | about 5-7 s, in the background |
| First save of a conversation with a title | also 1 `files()` (the folder-name check); 1 download more if a folder of that name exists | about 9-11 s, in the background |
| Save while the outputs folder holds something else | 1 `upload_file` per changed file | about 5-6 s each |
| Past-conversations list | 1 `files()` per browser session | about 3-4 s |
| Restore of N files | 1 `files()`, 1 download (`meta.json`), then N downloads six at a time | about 3-4 s + 3-4 s + N/6 × 3-4 s |
| Diagnostics line | 1 `files()` | about 3-4 s |

## Does the chat wait for the save?

**Before this change, yes.** The turn worker (`webapp/turn_jobs.py`) ran
the mirror before it marked the turn done. The page follows the job until
it is done, so the "Done" status, the rerun and the next input all waited
for the save: on the dataset about 5-7 s a turn, and about 10 s for a
conversation's first save.

**Now, no.** Once the record is complete (transcript, messages, meta,
trace), the worker calls `sharepoint_store.save_after_change`. On the
dataset (`SharePointStore.background`) that queues the save and returns
at once (`save_soon`).

- One background worker per process runs the queue in order, under the
  store's lock and the dataset's process-wide lock.
- A conversation already waiting is not queued twice; that save will send
  the newer change too. One that is saving now is queued again.
- The worker does not belong to the turn, so ending the turn does not end
  the save. At a normal exit the queue is finished first (up to 60 s,
  `atexit`).
- The feedback box's save goes the same way.
- **Sync now** and **Restore** stay in the page, with a spinner, because
  the person pressed them and waits for the answer.

The test `test_a_turn_saves_and_a_fresh_container_restores_it` holds the
dataset write open. It checks that the turn ended and the page drew,
showing "saving the latest turn…", before the save went through. Then a
fresh container lists the conversation and restores it.

## What changed

| File | Change |
|---|---|
| `webapp/foundry_dataset_store.py` | Rewritten over `foundry.transforms.Dataset`: the alias setting, the resource-ID and missing-library reasons, `AliasNotDefined` said plainly, staging under the outputs folder, `upload_directory` under a process-wide lock, the foreign-file fallback with a one-time note, the busy retry with `StorageBusy`, one listing per batch, `download_files` (parallel), `not_kept_reason` (the one line diagnostics and the sidebar show when nothing keeps the conversations; an unknown alias learned by one `Dataset.get`, `check_alias`). The platform-SDK code is removed. |
| `webapp/sharepoint_store.py` | The batch now spans the whole save, so the folder-name check reuses its listing. The save's note goes into the summary. A restore downloads through the file manager's `download_files` when it has one, after the path guard. `_pin_folder` reads `meta.json` again just before writing, so a background save cannot undo a title the next turn set. New: `background`, `not_kept_reason()`, `save_soon`, `save_state`, `flush` and `save_after_change`. `sync_for_conversation` prefers the store's own record for the conversation, because a background save reaches no session. |
| `webapp/turn_jobs.py` | The end-of-turn mirror goes through `save_after_change`. |
| `webapp/app.py` | The save-state line, the one-time note, on Foundry a "Conversations are NOT kept across restarts" block with the diagnostics line's reason and fix whenever no dataset keeps them (drawn even when the store is not configured), and spinners on Sync now and Restore. Sync now drops the cached list. The feedback box saves through `save_after_change`. The storage warning names the import and the alias, not token scopes. |
| `webapp/diagnostics.py` | The *conversation storage* line reads the alias, and its SKIP and FAIL texts come from `not_kept_reason`. |
| `docs/FOUNDRY.md` §4c | Rewritten: import the dataset (Data panel, read and write), push, set the alias, nothing to install; a push republishes the apps, which the store makes harmless; how a save works; the NOT-kept block; one user (per-viewer folders wait on the app knowing the viewer). |
| `webapp/README.md` | The module row. |
| `webapp/tests/test_foundry_dataset_store.py` | Rewritten: 40 tests over a fake `foundry.transforms`, made-up values only. |

## Tests (exit codes)

Run from the worktree with the main checkout's venv, in pieces, with at
least 3 GB free checked before each run:

| Run | Result | Exit |
|---|---|---|
| `test_foundry_dataset_store.py`, `test_sharepoint_store.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_privacy.py`, `test_conversation_restore.py`, `report_ingest/tests/test_mirror.py`, `test_diagnostics.py`, `test_foundry_mode.py`, `test_foundry_route.py` | 238 passed | 0 |
| `test_app_smoke.py`, `test_sharepoint_tools.py`, `test_past_conversations_pages.py`, `test_wave2a_bookkeeping.py`, `test_wave2c_fixes.py`, `test_tinyapps_wiring.py`, `test_record_ev06.py`, `test_titles_cards_scratch.py`, `test_output_capture.py` | 140 passed | 0 |
| `test_foundry_dataset_store.py`, `test_public_text.py`, `test_docs_currency.py`, `test_page_fingerprints.py` (files staged) | 60 passed | 0 |
| After the last edits: `test_foundry_dataset_store.py`, `test_sharepoint_store.py`, `test_conversation_restore.py`, `test_public_text.py`, `test_docs_currency.py` (staged) | 107 passed | 0 |

Second round (one user; the NOT-kept block), same pieces:

| Run | Result | Exit |
|---|---|---|
| `test_foundry_dataset_store.py` (40 tests) | 40 passed | 0 |
| the first piece above without the new file | 203 passed | 0 |
| the second piece above | 140 passed | 0 |
| `test_public_text.py`, `test_docs_currency.py`, `test_page_fingerprints.py` (staged) | 25 passed | 0 |

The fake library models what the FDE measured:

- `Dataset.get` of an unknown alias gives `404 …AliasNotDefined`.
- `upload_directory` refuses a folder outside the outputs folder
  (`IllegalArgumentError`), uploads everything under it in one
  transaction, and returns `{local: dataset path}`.
- `upload_file` is a transaction of its own.
- A write while another transaction is open gives
  `400 …SimultaneousOpenTransactionsNotAllowed`.
- `files()` lists everything, and hidden files only when asked.
- `File.download()` returns a local path.
- A later write overwrites a path and keeps the dataset's other files.

The new tests cover:

- **Listing.** `ls` filters the whole listing and makes folders; hidden
  files are listed.
- **Saves.** A single upload with overwrite; a batch is one
  `upload_directory`, sends a removed temporary file and leaves the
  outputs folder empty; a folder outside the outputs folder is refused.
- **The outputs folder.** A foreign file is never uploaded or deleted,
  saves go file by file, and the note comes once. The app's own leftover
  is cleared and not sent.
- **Busy and alias.** The busy retry (1, 2 s, then through) and the 30 s
  give-up. An unknown alias says how to import it.
- **Restore.** Parallel downloads: peak above 1 and at most 6, one
  listing, a missing file reported. Through the store: one save per
  mirror, nothing sent when nothing changed, only the changed file
  overwritten and the rest kept, and the round trip (13 files,
  byte-identical, one listing, in parallel, the next save sends nothing).
  The path guard, the record files and per-owner folders.
- **Background saves.** A busy save warns and goes next time; the
  foreign-file note reaches the summary once; saves run in order with no
  duplicate waits; a queued save outlives the turn's thread; the store's
  record wins over an older session record.
- **Configuration.** Foundry selects the dataset; a resource ID is
  refused; a missing library turns storage off; an unknown alias is
  learned with one `Dataset.get` and cleared by a call that works;
  Funhouse and Tiny Apps are unchanged.
- **Sidebar (AppTest).** The Foundry storage wording, the save-state line,
  the import warning on an unknown alias, and the filter's no-match line.
  Over the real store on Foundry, the "Conversations are NOT kept across
  restarts" block for each of the four cases (no alias, resource ID,
  library missing, alias not defined), with exactly the diagnostics
  line's text; no such block when the alias works, nor off Foundry.
- **End to end (AppTest).** A turn saves after the page has drawn, and a
  fresh container lists the conversation and restores it.
- **Also.** The diagnostics line (SKIP, PASS, resource ID, unknown alias,
  library missing) and report ingest's mirror (one `upload_file` per
  file).

## What the FDE must confirm in the real GSE app

1. **Alias import.** Before anything is set, the sidebar's **Permanent
   storage** block should say "Conversations are NOT kept across
   restarts", with the same reason as diagnostics. Then import the
   production conversation dataset (not the
   scratch one) into the GSE workspace: Data panel, read and write. Push,
   set `GEOTECH_FOUNDRY_CONVERSATION_DATASET` to its alias, and push again.
   Connection diagnostics, line *conversation storage*: PASS, with the
   alias. Send the line and how long the check took.
2. **Save.** Run one turn. The sidebar shows "saving the latest turn…",
   then, after the next click, "Foundry storage: N uploaded, … · Xs". **Xs
   is the save time**: send it for the first turn and for a later one. The
   dataset should hold `GeotechStaffEngineer/conversations/<title>_<date>/`
   with `meta.json`, the transcript, `activity.jsonl` and `files/`. Its
   history should show one transaction per turn.
3. **List.** **Find a past conversation (permanent storage)** lists that
   folder.
4. **Restore after a republish.** Push any change (the apps republish) and
   open the app: the conversation list is empty. Search **Find a past
   conversation**, press **Restore**, and check that it opens with its
   cards. Send the "Restored … N files in Xs" line.
5. **The outputs folder in the published container.** Is
   `/foundry/outputs` empty before the first save? If not, the sidebar
   says saves go file by file. Does `upload_directory` take a subfolder,
   such as `/foundry/outputs/GeotechStaffEngineer`, with dataset paths
   still relative to `/foundry/outputs`? If so, the app could upload only
   its own folder and drop the file-by-file fallback.
6. **Six downloads at once.** Does the sidecar take them without errors
   (the restore line's seconds show it)?

## Noted, not fixed

- **First page draw.** The past-conversations list reads the whole
  dataset (about 3-4 s) on the first page draw of each browser session.
  The expander's content runs even while it is closed, so on that first
  draw the chat input appears after the listing. Deferring the listing
  until it is asked for would change the sidebar on every host; left for
  the lead.
- **The save line is not live.** It updates on the next rerun (any click),
  not when the save ends.
- **Waiting behind a save.** **Sync now** waits behind a background save
  of any conversation (the store's lock), up to about 30 s more when the
  dataset is busy. A busy dataset holds the process-wide lock for that
  wait, so report ingest's mirror in the same process waits too.
- **Report ingest's mirror (RI-203)** fits with no change but uses no
  batch: one `upload_file` per file (5-6 s each). Its first print still
  says "SharePoint <folder>".
- **Downloaded copies.** The copies `File.download()` makes stay wherever
  the library puts them; the app does not remove them.
- **`meta.json` twice.** A restore downloads `meta.json` once to read the
  thread id, and again with the other files.
- **The upload's return map is not checked.** The map `upload_directory`
  returns is not compared with what was staged, because its exact path
  form is unconfirmed. An error raised by the library is the failure
  signal.
- **Copies newer than the manifest.** A background save copies files while
  the next turn may be writing. A copy newer than the manifest's stamp is
  sent again with the next save, which does no harm.
- **A moved folder's pointer.** When the store's root changes, the
  `MOVED.txt` pointer left in the old folder now goes up in the same save
  as the files. If that save fails, the pointer is not sent again, though
  the files are.
- **No folder links.** A dataset has no link per folder, so the sidebar
  shows the folder name but no "Open session folder" link.
- **One user, by decision.** The store is built for the owner alone
  (2026-11-03): one folder, `GeotechStaffEngineer/conversations/`, and
  **Find a past conversation** lists every conversation in it. Writes run
  as the publisher's account, as far as the FDE's test could tell. Whoever
  can open the dataset reads them all, so it belongs in a project only the
  right people can see (docs §4c). Per-viewer folders wait on the app
  knowing who is viewing; once it does (as with Tiny Apps' header), they
  follow with no store change.
- **Mixed hosts.** On a host where some viewers are identified and some
  are not, an unidentified viewer's list shows the owners' folders as
  entries that cannot be restored. Foundry identifies no one, so this
  does not arise there; SharePoint behaves the same today.
