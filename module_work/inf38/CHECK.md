# INF-38: conversations that survive a rebuild on Foundry

Built 2026-11-03 on branch `wip/inf38-dataset` (the old `wip/inf38` is the
merged near-term half). The path is the AI FDE's answer of 2026-11-03: a
Foundry dataset of raw files, written through the Foundry platform SDK.
**No call below has been run against Foundry.** Every SDK call is in one
class, `webapp/foundry_dataset_store.py`, `DatasetFileManager._build_client`
and the `_sdk_*` methods, so a name the FDE's test corrects is a one-place
change.

## What a user gets

On a Foundry deployment with `GEOTECH_FOUNDRY_CONVERSATION_DATASET` set
(and the SDK installed):

- After every turn the conversation folder (`meta.json`, the transcript,
  `activity.jsonl`, `trace.jsonl`, `FEEDBACK.md` / `feedback.jsonl`,
  `files/`) goes to the dataset under
  `GeotechStaffEngineer/conversations/<title>_<date>/`, as it goes to
  SharePoint on Funhouse. The owner's full-trace reviews can read the
  records there (Foundry wiring audit row 8; below).
- When the conversation filter finds nothing, its line points at the block
  that is drawn: **Find a past conversation** under Permanent storage with
  the store configured, else **Restore a downloaded conversation** (the zip,
  INF-45) on the geotech page (audit, the `app.py` "try Permanent storage
  below" line).
- The sidebar shows **Permanent storage** with "Foundry storage: N
  uploaded, M up-to-date", **Sync now**, and **Find a past conversation
  (permanent storage)**, which lists the dataset's folders and restores one
  (record and files) after a rebuild.
- **Connection diagnostics** has a *conversation storage* line: PASS with
  the dataset, branch and where the host came from (one read, no write);
  FAIL with a plain reason (SDK missing, no token, no host, dataset
  unreadable); SKIP when the dataset is not set, saying conversations are
  lost on a rebuild.
- A failed save never breaks a turn: one warning in the sidebar, the files
  go up with the next save.

Funhouse and Tiny Apps are unchanged: the dataset setting is read only when
`GEOTECH_DEPLOYMENT=foundry`.

## What changed

| File | Change |
|---|---|
| `webapp/foundry_dataset_store.py` (new) | `DatasetFileManager`: `ls` (folders made from the file paths), `create_folder` (no-op), `upload_file`, `download_file`, `get_web_url` (none), `probe`; the batch hook `begin_batch` / `end_batch`; settings, host and token from `engine_config`; `foundry_sdk` imported lazily |
| `webapp/sharepoint_store.py` | the store builds the dataset file manager on Foundry with the dataset set; `backend` and `label` ("Foundry storage"); `root()` defaults to `GeotechStaffEngineer` there; the mirror sends one sync's files in one batch and records them only after the commit; a restore runs in a batch (one listing); module-level `configured()` (the gate of the agent's SharePoint tools and the reference-PDF fetch) is False when the dataset is the storage |
| `webapp/app.py` | sidebar wording names Foundry storage; the token warning gives the datasets scopes, not `stage_sharepoint`; `_sp_configured()` (the attach hint) is SharePoint only; the filter's no-match line (`_no_match_hint`) names the block drawn |
| `webapp/diagnostics.py` | the *conversation storage* line, on Foundry only (commit 1; not touched since, INF-62 works there) |
| `docs/FOUNDRY.md` | §4c only (setting it up, the files carried, who sees what); the Notes bullet points to it |
| `webapp/README.md` | the module row |
| `webapp/tests/test_foundry_dataset_store.py` (new) | 29 tests over a fake SDK client (made-up RIDs, hosts and tokens) |

## The record files the trace reviews read (audit row 8)

Confirmed carried. `activity.jsonl` (`activity_log.FILE_NAME`),
`trace.jsonl` (`core.trace_path`), `FEEDBACK.md` and `feedback.jsonl`
(`feedback.MD_NAME`, `JSONL_NAME`) are all top-level files of the
conversation folder; the mirror walks the whole folder and leaves out only
the in-flight files (`partial.json`, `partial_messages.json`, `*.part`,
`*.tmp`), the scratch and cache folders under `files/` and its own
manifest (`core.is_in_flight_file`, `core.mirror_skips_dir`). The turn
writes `trace.jsonl` before its end-of-turn save (`turn_jobs.py`), so a
turn's trace goes with that turn; the agent's `record_feedback` notes go
at the end of the turn; the sidebar's Feedback box saves at once. The
restore brings all of them back. `trace.jsonl` exists only while
`GEOTECH_TRACE` is on (it is on unless set to 0/false/no/off).
Test: `test_the_record_files_the_trace_reviews_read_are_carried` writes
each through the app's own writer, saves, checks the bytes in the dataset,
and restores.

How a save works: the store calls `begin_batch()`, uploads each changed
file (the first upload opens ONE `UPDATE` transaction; a sync with nothing
changed opens none), then `end_batch(commit=True)`. A failed commit is
aborted and none of its files is recorded, so all go again next sync.
Opening a transaction while another is open on the branch is retried 5
times with pauses of 0.5, 1, 2, 4 s (about 8 s), once per sync, not per
file. The batch state is per thread: one process serves every viewer, and
one session's restore cannot end another's save. Listings are cached for
one mirror or restore and dropped after an upload.

Host: `GEOTECH_FOUNDRY_HOST` when set; else, when the container sets
`FOUNDRY_SERVICE_DISCOVERY_V2`, the SDK finds the host itself (its own
container mechanism); else `FOUNDRY_HOSTNAME` / `FOUNDRY_URL` through
`engine_config.foundry_base_url()`. Token: `engine_config.foundry_token()`,
re-read per call (a rotated token rebuilds the client). CA file:
`engine_config.foundry_ca_bundle()`.

## Tests (exit codes)

Run from the worktree with the main checkout's venv, in pieces, free memory
checked first (3.0 to 4.1 GB):

| Run | Result | Exit |
|---|---|---|
| `test_foundry_dataset_store.py`, `test_sharepoint_store.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_privacy.py`, `test_diagnostics.py`, `test_foundry_mode.py`, `test_foundry_route.py`, `report_ingest/tests/test_mirror.py` | 196 passed | 0 |
| `test_sharepoint_tools.py`, `test_past_conversations_pages.py`, `test_wave2a_bookkeeping.py`, `test_wave2c_fixes.py`, `test_tinyapps_wiring.py`, `test_record_ev06.py`, `test_titles_cards_scratch.py`, `test_output_capture.py`, `test_app_smoke.py` | 140 passed | 0 |
| `test_public_text.py` (with these files staged, so the tracked-file walk reads them) | 4 passed | 0 |
| `test_page_fingerprints.py` | 15 passed | 0 |
| `test_docs_currency.py` | 6 passed | 0 |

Second round (the audit's additions: the filter line, the record files),
same pieces again:

| Run | Result | Exit |
|---|---|---|
| piece A above (now 29 tests in the new file) | 199 passed | 0 |
| piece B above | 140 passed | 0 |
| `test_public_text.py`, `test_page_fingerprints.py`, `test_docs_currency.py` (staged) | 25 passed | 0 |

Piece A of this round started with 2.4 GB free, under the 3 GB rule (the
check and the run were in one command); it passed, and the later runs
were checked first (3.1 to 3.8 GB).

The new tests cover: listing (folders from paths, the prefix asked for,
an SDK without `path_prefix`), a single upload, overwrite, download, one
transaction per batch and none when nothing changed, one listing per batch,
the batch per thread, a conflict retried with growing pauses, a conflict
that lasts (gives up after the last try, once per sync), a failed commit
(nothing recorded, sent next time), the mirror and the restore round trip
through the store (byte-identical; the next sync sends nothing), per-owner
folders, the client built from the app's token and host (service discovery,
an explicit `http://` host, a rotated token), the SDK missing, no token,
Funhouse and Tiny Apps unchanged, the sidebar wording (AppTest), the
filter's no-match line with and without the store (AppTest), the record
files (`activity.jsonl`, `FEEDBACK.md`, `feedback.jsonl`, `trace.jsonl`)
saved and restored, the diagnostics line, and report ingest's mirror.

## SDK calls for the FDE to confirm (paste into the round-trip test)

Package `foundry-platform-sdk` (read at 1.111.0), `import foundry_sdk`.
Review, 2026-11-03: every name, argument order, keyword and return type
below matches 1.111.0's source, and the app's class ran each call through
the real 1.111.0 package (not installed in the venv) against a local fake
of the datasets REST API (made-up values): the client with an `http://`
host, listing with `path_prefix` over three pages (`nextPageToken`), one
`UPDATE` transaction per batch, upload as bytes, overwrite, `content` as
bytes, the conflict (`OpenTransactionAlreadyExists`, five tries) and a
refused commit. Service discovery needs `FOUNDRY_HOSTNAME` set beside
`FOUNDRY_SERVICE_DISCOVERY_V2` (the SDK will not build its client without
it), so the app uses discovery only then.

```
client = foundry_sdk.FoundryClient(
    auth=foundry_sdk.UserTokenAuth(token=TOKEN),
    hostname=HOST,                      # omitted when the SDK's service discovery is used
                                        # (FOUNDRY_SERVICE_DISCOVERY_V2 and FOUNDRY_HOSTNAME both set)
    config=foundry_sdk.Config(timeout=120, verify=CA_FILE))   # verify only when a CA file is set;
                                                              # scheme="http" only for an http:// host
F = client.datasets.Dataset.File
T = client.datasets.Dataset.Transaction

F.list(RID, branch_name="master", page_size=1000, path_prefix="GeotechStaffEngineer/conversations")
    # iterate: items with .path, .size_bytes, .updated_time
txn = T.create(RID, transaction_type="UPDATE", branch_name="master")      # txn.rid
F.upload(RID, "GeotechStaffEngineer/conversations/<folder>/meta.json", b"...", transaction_rid=txn.rid)
T.commit(RID, txn.rid)
T.abort(RID, txn.rid)                   # on any error, and to test the abort path
F.content(RID, "GeotechStaffEngineer/conversations/<folder>/meta.json", branch_name="master")   # -> bytes

# only when a create is refused because a transaction is open (a leftover):
head = client.datasets.Dataset.Branch.get(RID, "master").transaction_rid   # newest open or committed
info = T.get(RID, head)                 # .status == "OPEN", .created_time (aware datetime)
```

A transaction a save left open would block every later save (one open per
branch), so at the first refusal the store aborts the app's own leftover
and tries again at once: first one named in its marker file
(`<data folder>/.foundry_open_transactions.json`, the transactions this
app opened and has not closed, with the process id; a restart on the same
disk), with no request; else the branch's open transaction when
`created_time` is more than 10 minutes old (`STALE_AFTER_S`). The SDK shows
a transaction's age but not who opened it (no creator field on
`Transaction`), so on a dedicated dataset age is the test; a transaction
a save in this process holds, or one another running process on the disk
named, is never aborted. Tests:
`test_a_leftover_named_in_the_marker_is_aborted_at_once`,
`test_a_stale_open_transaction_is_aborted_and_a_young_one_kept`,
`test_a_transaction_held_here_or_by_a_live_process_is_kept`; the smoke
through the real SDK ran both `get` calls and the stale abort.

The same round trip through the app's own class (exercises exactly these
calls; run in the published app's environment):

```python
import os
os.environ["GEOTECH_DEPLOYMENT"] = "foundry"
os.environ["GEOTECH_FOUNDRY_CONVERSATION_DATASET"] = "<dataset RID>"
from webapp import foundry_dataset_store as fds
print(fds.unconfigured_reason(), fds.sdk_target())
fm = fds.DatasetFileManager()
print(fm.describe(), fm.probe())
open("probe.txt", "w").write("hello")
fm.upload_file("probe.txt", "GeotechStaffEngineer/probe/probe.txt", overwrite=True)
print(fm.ls("GeotechStaffEngineer/probe"))
print(fm.download_file("GeotechStaffEngineer/probe/probe.txt"))
```

Questions for the FDE with it:

1. Do the names and keyword arguments above hold on the installed SDK
   version? Is `path_prefix` accepted (the code falls back to listing
   everything when it is refused)?
2. From a published app, which host works for these calls: the SDK's own
   discovery (`FOUNDRY_SERVICE_DISCOVERY_V2` set?), `FOUNDRY_HOSTNAME` (the
   local sidecar that refused the language-model paths), or another? Is the
   sidecar plain `http`? (The SDK puts `https` on a bare host.)
3. Is `FOUNDRY_TOKEN` in a published app the publisher's or the viewer's,
   and does it carry the datasets read and write scopes?
4. Does a published app's request carry the viewer's identity (a header)?
   Today the app cannot tell viewers apart on Foundry (below).
5. When a container is rebuilt in the middle of a save, does Foundry close
   the open transaction, or does it stay open and block every later save on
   that branch until someone aborts it? (The app now aborts its own
   leftover, above; the answer says whether that path is ever needed. Can
   the app's token read `Branch.get` and `Transaction.get` and abort a
   transaction the app opened before a rebuild?)

## Unknown viewer: whose folder

Foundry gives the app no viewer identity today (audit row 4). The app
keeps people apart only when a HOST names the viewer (the Tiny Apps proxy
header: `Identity.multi_user`). On Foundry `identity.current_identity()`
returns the anonymous identity; `DEV_IDENTITY` or `GEOTECH_USER_EMAIL`, if
someone set one in the app file, name one person for display and change no
folder.

So an unknown viewer's conversations go into **no one's folder: the one
shared folder of the app**, with no owner segment:

- geotech page: `GeotechStaffEngineer/conversations/<title>_<date>/`
  (the conversation's `meta.json` carries no `owner`; the app tags an
  owner only when `multi_user` is true, `app.py`, `core.tag_conversation`);
- Document Review page: `GeotechStaffEngineer/conversations/document_review/<title>_<date>/`.

It is not the publisher's folder, and not keyed by the token's account:
every viewer's conversations land side by side in that folder, and **Find a
past conversation** lists, and lets any viewer restore, all of them, as the
app's own conversation list in the container already shows every viewer's
conversations. Whoever can open the dataset in Foundry can read them all.
Two viewers on one day asking under the same title get two folders (the
second gets a six-character id suffix), never one shared folder.

When a host does name the viewer, each person's conversations go to
`conversations/<their key>/[document_review/]`, and the list and restore
look only there (`test_each_owner_lists_and_restores_only_their_folder`
and the SharePoint tests). Reading a viewer identity on Foundry is the
audit's U4 / FDE question 4; once read the way the IIS header is, the
dataset folders follow with no store change.

## Report ingest's durable mirror (RI-203)

Fits with no change. `Mirror(sharepoint=...)` takes the dataset file
manager directly, or the app's store (it unwraps `file_manager()` to the
same object); `mirror_dir` and `restore_dir` round-trip
(`test_report_ingests_mirror_fits_unchanged`). It has no batch hook, so
each file is its own transaction (create, upload, commit).

## Noted, not fixed

- Report ingest's mirror sends each file in its own transaction (three
  calls per file) and its first print says "SharePoint <folder>" on the
  dataset too; same files saved, so not changed.
- A listing of a person's conversations reads every file under the folder
  (the dataset lists files, not folders), so a long history makes the
  past-conversations list and the first save of a conversation slower;
  results are the same.
- A transaction left open by a crash would block saves on the branch
  (question 5): handled since the review, the store aborts the app's own
  leftover (above, under the SDK calls).
- The dataset has no link per folder, so the sidebar shows the folder name
  but no "Open session folder" link.
- On a mixed host (some viewers identified, some not), an unidentified
  viewer's list at the top level would show the owners' folders as entries
  that cannot be restored; Foundry identifies no one, so this does not
  arise there. Same on SharePoint today.

From the review (2026-11-03):

- After a rebuilt container (no marker), saves in the first 10 minutes
  after the leftover was opened each cost the five tries, about 8 s, and a
  warning; the first save after that aborts it and goes through.
- A host that takes connections but never answers costs one 120 s timeout
  per save (the SDK retries only 429 and 503; the batch's other files then
  fail at once).
- A refused transaction shows one line per changed file under "Sync
  errors", all the same cause.
- A dataset path with a `..` segment would have been restored outside the
  conversation's folder: fixed, a restore skips any path that is absolute,
  holds a `..` segment (either slash) or resolves outside the folder, with
  one warning line (`_local_path_inside`,
  `test_a_restore_never_writes_outside_the_conversation_folder`).
- A `KeyboardInterrupt` or `SystemExit` inside a commit would leave that
  transaction open (`end_batch` aborts on `Exception`).
