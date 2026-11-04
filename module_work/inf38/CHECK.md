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
  M up-to-date · Xs" (since INF-65 "saved N file(s) just now, …", shown
  without a click). **Sync now** saves at once, with a spinner, and
  drops the cached past-conversations list.
- **Find a past conversation (permanent storage)** lists the dataset's
  conversation folders and restores one (record and files), downloading
  six files at a time. **The list is read when asked for**: **Show past
  conversations**, a search or **Refresh**, then cached for the browser
  session. No page's first draw waits on `files()`. This holds on every
  host, SharePoint too, since a listing there also takes seconds.
- **Connection diagnostics**, line *conversation storage*: PASS with the
  alias and how many files are the app's (one listing, no write); FAIL with
  a plain reason; SKIP when the variable is not set. A new dataset with
  nothing committed passes as "readable, 0 files" (INF-65, below).
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
  library case says to publish from a Code Workspace. Since INF-65 a
  fifth: any other refusal of a read (for example `FailedToResolveDataset`
  for an alias pointing at the wrong dataset), in Foundry's own words. An
  unknown alias or a refused read is found by one listing per process,
  started by the sidebar's first draw, in the background (INF-65, below).
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
  objects for its downloads. The sidebar's list is read only when asked
  for, then cached per browser session until **Refresh** or **Sync now**.
  An index file would not help: no draw needs the listing more than once.
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
| Opening the app (first draw) | none; the first draw in a process starts 1 `files()` in the background (INF-65) | 0 (the check about 3-4 s, not waited for) |
| Past-conversations list, when asked for | 1 `files()` per browser session | about 3-4 s |
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
fresh container draws with no `files()` call, lists the conversation once
when asked, and restores it.

## What changed

| File | Change |
|---|---|
| `webapp/foundry_dataset_store.py` | Rewritten over `foundry.transforms.Dataset`: the alias setting, the resource-ID and missing-library reasons, `AliasNotDefined` said plainly, staging under the outputs folder, `upload_directory` under a process-wide lock, the foreign-file fallback with a one-time note, the busy retry with `StorageBusy`, one listing per batch, `download_files` (parallel), `not_kept_reason` (the one line diagnostics and the sidebar show when nothing keeps the conversations; an unknown alias learned by one `Dataset.get`, `check_alias`). The platform-SDK code is removed. |
| `webapp/sharepoint_store.py` | The batch now spans the whole save, so the folder-name check reuses its listing. The save's note goes into the summary. A restore downloads through the file manager's `download_files` when it has one, after the path guard. `_pin_folder` reads `meta.json` again just before writing, so a background save cannot undo a title the next turn set. New: `background`, `not_kept_reason()`, `save_soon`, `save_state`, `flush` and `save_after_change`. `sync_for_conversation` prefers the store's own record for the conversation, because a background save reaches no session. |
| `webapp/turn_jobs.py` | The end-of-turn mirror goes through `save_after_change`. |
| `webapp/app.py` | The save-state line, the one-time note, on Foundry a "Conversations are NOT kept across restarts" block with the diagnostics line's reason and fix whenever no dataset keeps them (drawn even when the store is not configured), and spinners on Sync now and Restore. Sync now drops the cached list. The past-conversations list is read when asked for (Show past conversations, a search or Refresh), never on the first draw, on every host. The feedback box saves through `save_after_change`. The storage warning names the import and the alias, not token scopes. |
| `webapp/diagnostics.py` | The *conversation storage* line reads the alias, and its SKIP and FAIL texts come from `not_kept_reason`. |
| `docs/FOUNDRY.md` §4c | Rewritten: import the dataset (Data panel, read and write), push, set the alias, nothing to install; a push republishes the apps, which the store makes harmless; how a save works; the NOT-kept block; one user (per-viewer folders wait on the app knowing the viewer). |
| `webapp/README.md` | The module row. |
| `webapp/tests/test_foundry_dataset_store.py` | Rewritten: 40 tests over a fake `foundry.transforms`, made-up values only. |
| `webapp/tests/test_past_conversations_pages.py`, `test_app_smoke.py` | Press **Show past conversations** before a restore; a page's first draw lists nothing; a search reads the list (one new test). |

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

Third round (the list read when asked for):

| Run | Result | Exit |
|---|---|---|
| `test_past_conversations_pages.py`, `test_foundry_dataset_store.py` | 45 passed | 0 |
| `test_app_smoke.py`, `test_wave2c_fixes.py`, `test_sharepoint_store.py` | 57 passed | 0 |
| every other app test that drives the page (`test_working_files_note.py`, `test_ws_upload.py`, `test_record_ev06.py`, `test_orientation_overview.py`, `test_foundry_mode.py`, `test_foundry_model_family.py`, `test_feedback.py`, `test_download_mime.py`, `test_custom_model_rid.py`, `test_app_profiles.py`), with `test_turn_jobs.py` | 155 passed | 0 |
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
  fresh container draws with no `files()` call, lists the conversation
  once when asked (not again on the next rerun), and restores it.
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
   then, by itself since INF-65, "Foundry storage: saved N file(s) just
   now, … · Xs". **Xs is the save time**: send it for the first turn and
   for a later one. The
   dataset should hold `GeotechStaffEngineer/conversations/<title>_<date>/`
   with `meta.json`, the transcript, `activity.jsonl` and `files/`. Its
   history should show one transaction per turn.
3. **List.** Opening the app should not wait on the dataset. In **Find a
   past conversation (permanent storage)**, **Show past conversations**
   lists that folder after about 3-4 s.
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

- **One more press to restore.** The list now waits for **Show past
  conversations** (or a search), because an expander's content runs even
  while it is closed, and an expander that tells the app when it opens
  (`on_change`) exists only in recent Streamlit, while the app allows 1.39
  and up. After **Sync now** the list is read again only when next shown.
- ~~**The save line is not live.**~~ Fixed by INF-65 (below): the
  sidebar looks again every 2 s while a save runs.
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

## INF-65: a new, empty dataset; the sidebar and diagnostics agree; a live save line

Built 2026-11-03 on branch `wip/inf65`, from the FDE's build i hand-back
(report items 4 and 5, kept locally, never committed):

- A brand-new raw-files dataset with no committed transaction refused the
  first read with `400 FoundryContainerService:FailedToResolveDataset`,
  "The dataset does not have any committed transaction on the requested
  branch or across fallback branches". Connection diagnostics FAILED and
  the owner had to upload a placeholder file (Foundry then guessed a CSV
  schema for it, removed by hand).
- Meanwhile the sidebar already said "Foundry storage is on".
- After the turn the sidebar showed no save line; the save landed about
  7 s later, in one transaction.

### What changed

| File | Change |
|---|---|
| `webapp/foundry_dataset_store.py` | `_is_no_commit`: the error name `FailedToResolveDataset` AND the wording "no / not have any committed transaction", in the error or the errors it wraps. A listing that meets it is an empty store (`uncommitted`), so `ls` gives `[]`, the probe says "readable, 0 files: a new dataset with nothing committed yet; the first save makes its first commit", the folder-name check adds no shard, and the save path is unchanged (the first `upload_directory` makes the first commit). Any other `FailedToResolveDataset` still raises. Every listing records what it found in one record per alias, shared by every file manager in the process (`_READS`). `known_problem` (replacing `check_alias`) gives the sidebar the alias error or the recorded read failure; when nothing has read the alias yet it starts ONE listing in the background and the page does not wait (`checking`). |
| `webapp/diagnostics.py` | Every probe failure is shown as `not_kept_reason`, the sidebar's text; the probe records into the shared record, so a failed diagnostics read turns the sidebar to NOT kept, and a read that works clears it. |
| `webapp/sharepoint_store.py` | `check_state()` ("checking" while that first read runs). A save's summary carries `at`, when it ended. |
| `webapp/app.py` | The status lines of **Permanent storage** move into `_storage_status`. While the dataset's first check, or a save of this conversation, runs in the background (`_storage_busy`, Foundry only in practice), `_storage_panel` draws them as an `st.fragment(run_every=2 s)`. When that ends the fragment reruns the page once, which shows the result and defines no fragment, so the polling stops. A timed fragment run never interrupts a running turn: Streamlit queues it behind the full run. New lines: "Foundry storage: checking the conversation dataset…"; "Foundry storage: saved N file(s) just now, M up-to-date · Xs" (SharePoint keeps "N uploaded, …"). "Foundry storage is on" shows only after a read has worked. |
| `docs/FOUNDRY.md` §4c | A new dataset needs no placeholder file. Diagnostics passes on it as "readable, 0 files". The sidebar checks, then agrees with diagnostics. The save line updates without a click. |
| `webapp/tests/test_foundry_dataset_store.py` | The fake models the no-commit refusal (`files()` or `Dataset.get`), another `FailedToResolveDataset` cause, and a listing that hangs. 8 new tests, 48 in all. |

### What a fresh dataset now shows

- **Connection diagnostics:** PASS, "Foundry storage: workspace dataset
  alias …, saves staged in /foundry/outputs; readable, 0 files: a new
  dataset with nothing committed yet; the first save makes its first
  commit."
- **The sidebar:** "Foundry storage: checking the conversation dataset…"
  for the first 3-4 s after the container starts, then "Foundry storage is
  on — …".
- **After the first turn:** "saving the latest turn…", then, by itself
  within about 2 s of the save ending, "Foundry storage: saved 5 file(s)
  just now, 0 up-to-date · 6.8s".

### Tests (exit codes)

Run from the worktree with the main checkout's venv, in pieces, with at
least 3 GB free checked before each run:

| Run | Result | Exit |
|---|---|---|
| `test_foundry_dataset_store.py` | 48 passed | 0 |
| `test_sharepoint_store.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_privacy.py`, `test_conversation_restore.py`, `test_past_conversations_pages.py`, `test_diagnostics.py`, `test_foundry_mode.py` | 158 passed | 0 |
| `test_app_smoke.py` | 11 passed | 0 |
| `test_public_text.py`, `test_docs_currency.py`, `test_page_fingerprints.py`, `test_foundry_dataset_store.py` (files staged) | 73 passed | 0 |

The new tests:

- **Matching.** The no-commit error is matched by its name and its
  wording, and also when wrapped. The same name with other wording, or the
  wording under another name, is not.
- **A new dataset.** It lists as empty, the probe and diagnostics PASS
  with "readable, 0 files", and the past-conversations list is empty. The
  first save makes the first commit under the plain folder name, and the
  next probe counts its files.
- **`Dataset.get` refusing.** When `Dataset.get` itself refuses a new
  dataset, the listing is still empty, and a save fails plainly with
  Foundry's error (the save path is unchanged).
- **One record.** A failed diagnostics read after a good check turns the
  sidebar's reason to the diagnostics line's text. A good read clears it.
  One check, two probes, three listings.
- **The sidebar (AppTest).** It says "checking…", and not "on", while the
  first listing hangs, then "on" once it has worked: one listing per
  process. It shows NOT kept, in the diagnostics line's words, for a
  `FailedToResolveDataset` with another cause. It shows "on" with
  diagnostics PASS for a new dataset.
- **The live save line (AppTest).** A save that ends while the page
  draws: the status lines were a fragment with `run_every=2.0`. The page
  reran once by itself, shows "saved 5 file(s) just now", and defined no
  fragment on that run.

### What the FDE must confirm

7. **A new, empty dataset.** Create a fresh raw-files dataset with no
   file in it, import it, set the alias and push. Connection diagnostics
   should PASS with "readable, 0 files: a new dataset…". The sidebar should
   say "checking…", then "Foundry storage is on". Then run one turn. Does
   the first save make the first commit? The dataset should hold the
   conversation folder, its history one transaction, and the sidebar
   "saved N file(s)". If instead the save fails with
   `FailedToResolveDataset`, `Dataset.get` or `upload_directory` refuses a
   dataset with no commit. Send that error; the app then needs another way
   to make the first commit. (The app's own class has run this only
   against the fake.)
8. **The save line updates by itself.** After a turn, without clicking,
   the sidebar should go from "saving the latest turn…" to "saved N
   file(s) just now" within a few seconds of the save landing.

### Noted, not fixed

- **A transient read failure.** It turns the sidebar to NOT kept until a
  later read works: diagnostics, **Show past conversations**, **Refresh**,
  or a save's folder check.
- **Saves do not update the record.** A failed save shows in the save
  line, not in the NOT-kept block, and a save that works does not clear a
  read failure.
- **"just now" goes stale.** The time on the save line goes stale between
  reruns, as the conversation list's times do.
- **One extra rerun per save.** The rerun at the end of a background save
  redraws the whole page once, like a click.
