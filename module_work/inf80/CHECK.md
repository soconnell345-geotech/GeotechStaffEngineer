# INF-80: Document Review on Foundry, each turn bound to its own conversation

Branch `fix/inf-80-foundry-turn-binding`, from master 51e478f. Change commit
f820ac2; this note is the last commit. A repair: no setting.

## Confirmed on today's code (master 51e478f), before changing

Turning the binding on makes `aec_agent_core._fileio._bound_or_env` raise
`UnboundWorkingDir` in any context with no folder bound. Only `_fileio`'s
folder readers call it (`default_output_dir`, `host_output_dir`,
`resolve_output_path`, `find_in_working_folder`, `into_working_folder`,
`rescue_dir`, `conversation_name`, `hide_working_folder`). Each page path
below either binds the turn's folder or never reads a folder through
`_fileio`, so none can raise where it did not before:

| Path | What it does | Binds / needs no binding |
|---|---|---|
| Turn runner | `app.py:1805` takes `core.working_dir_for`, `app.py:1810` binds the script run, `turn_jobs.start_turn_job` (`app.py:1852`) starts the worker thread | `turn_jobs._run_turn_job` binds `ctx["working_dir"]` first thing in the worker (`turn_jobs.py:221-222`); the tools inherit it (LangGraph copies the context; the core's own pools pass `contextvars.copy_context()`: `scale_labels.py:244`, `vision_probe.py:400`, `vision_tools.py:3536`, `deep/vision_engine.py:727`, Document Review `core.py:1049`) |
| Automatic first turn after an upload | `app.py:1752-1763` turns `pending_orientation` into `prompt` | Goes through the same `if prompt:` block, so the same binding as above |
| Restore from permanent storage | `app.py:1417` `restore_conversation` (`sharepoint_store.py:1348`) | Writes to the conversation's explicit paths; `sharepoint_store.py`, `foundry_dataset_store.py` and `conversation_restore.py` import nothing from `_fileio`. It also runs after the page binds its run (`app.py:997`) |
| Restore from a zip | `app.py:1496-1498` `conversation_restore.restore` | As above |
| "Sync now" | `app.py:1328-1337`: `save_after_change`, `mirror_conversation` | As above (explicit paths, no `_fileio`) |
| Background save worker | `sharepoint_store.save_soon` (`:958`) starts `_bg_run` threads (`:980`), which call `mirror_conversation`; the Foundry store's own threads (`foundry_dataset_store.py:436`, `:743`) | Needs no binding: no `_fileio`. The new test saves and restores from unbound threads with binding on |

Other page code: a page run is a fresh thread, so nothing is bound until
`app.py:997`. Before that line the page calls no `_fileio` reader. The only
indirect reader, `error_text.scrub_paths`, wraps `hide_working_folder` in
try/except (`error_text.py:63-69`). `output_capture._real` does the same
(`output_capture.py:72-76`) and runs in the turn worker anyway. The
Foundry-only branches (`app.py:814`, `:1649`, `file_widgets.py`,
`diagnostics.py`) read no `_fileio`. Tiny Apps has run this same page code
with binding on since live smoke 2c.

What leaked on Foundry before this change: the turn runner already bound
each turn's folder on every host. So the cross-conversation saves that
section 6 of the guide describes came through these routes:
- a conversation's working folder named in its meta: anyone could type one
  into the box, or a restored record could carry one. Two conversations
  pointed at one folder each imported the other's new files after a turn
  (`core.import_external_artifacts`);
- the process-wide `GEOTECH_DEFAULT_OUTPUT_DIR` fallback in any context with
  no folder bound;
- a reported path in the shared temporary folder, which was copied in
  (`core.import_reported_outputs`);
- a restored transcript's absolute artifact reference
  (`core._resolve_artifact`).

All four are closed once `_shared_host()` is true.

## Changed

- `packages/aec-document-review/aec_document_review/identity.py`:
  `current_identity()` first calls `_bind_turns()` when `_foundry()` is
  true (`engine_config.is_foundry_deployment()`, which reads
  `GEOTECH_DEPLOYMENT=foundry`); the header case uses the same
  `_bind_turns()`. The identity is unchanged: still `ANONYMOUS`, so
  `multi_user` stays false, the conversation list stays shared
  (`profiles.session_root`), and nobody is refused (`_deployed()` is
  untouched).
- Why there and not in `foundry_entry`: `app.py:40` calls
  `current_identity()` on every page run, so a workspace that sets
  `GEOTECH_DEPLOYMENT=foundry` and starts the page some other way (for
  example `pages_entry.run()`) gets the binding too. The entry sets that
  variable, so the entry path is covered as well. A Foundry run without the
  variable is not a Foundry deployment to the page at all (no Foundry
  models, no dataset store).
- Docstrings: `foundry_entry.py` (a "Several people at once" paragraph) and
  `core._shared_host`.
- `docs/FOUNDRY.md` section 6: the Working folder sentence now says the box
  is not shown, and "Two people at the same time" says the protections are
  on for Document Review on Foundry and that the geotech page lacks them.
  The sentences on saves are kept.

Made-up example: two viewers' conversations, `made-up-a` and `made-up-b`,
both had `/home/made_up/project_out` typed in the box. They ask at the same
moment for a memo saved as `answer.txt`. A saves it, then B overwrites it,
and each turn's import copied the folder's new `answer.txt` in, so A's
conversation held B's answer. Now each turn's working folder is its own
conversation's `files/` folder, whatever the meta says. Each answer stays in
its own conversation, the project folder is untouched, and the box is gone.

## Tests

PowerShell, `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONPATH=<checkout>\packages\aec-agent-core;<checkout>\packages\aec-document-review`,
`C:\dev\GeotechStaffEngineer\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
Both packages resolved inside the worktree (`aec_document_review.__file__`
and `aec_agent_core.__file__` printed first).

- New file `packages/aec-document-review/tests/test_inf80_foundry_turn_binding.py`:
  10 passed, exit 0.
- The same file with `identity.py` put back to master: 8 failed and 2 passed
  (the two off-Foundry guards, which should pass). The two-turn test failed
  on its content check (`'made-up answer B' == 'made-up answer A'`). A probe
  copy of the box test without its condition line showed the published page
  drawing `Working folder (agent saves land here)`. Two tests fail without
  the change only on their binding check: "nobody is refused" (the refusal
  part held before) and the background save and restore test (a guard: its
  save and restore part held before).
- From `packages/aec-document-review`: the new file, `test_foundry_host.py`,
  `test_shared_host_files.py`, `test_review_binding_offline.py`,
  `test_conversation_restore.py` (names `_shared_host`; `conftest.py` names
  `require_turn_binding`), `test_review_fingerprints.py`, plus the files
  that set a Foundry deployment or call `current_identity`
  (`test_conversation_export.py`, `test_file_widgets.py`,
  `test_helper_models.py`, `test_tinyapps_wiring.py`): 149 passed,
  3 skipped, exit 0. The three skips are
  in `test_conversation_export.py` and are Windows-only: no symlink, a
  backslash is a separator here, no named pipes.
- From the checkout root, in another process: `webapp/tests/test_public_text.py`:
  5 passed, exit 0, run after this note was written.

## Fingerprints

`test_review_fingerprints.py` passed in the run above. No prompt or tool
text changed.

## Calls made

None: no model or network calls; local pytest only. Nothing pushed.

## Release note (next Document Review release)

On Foundry, files stay in their own conversation even when several people
use the app at the same moment. The **Working folder** box is gone from the
sidebar there: the agent always saves into the open conversation's own
folder. Everyone still sees the same list of conversations.

## Noted, not fixed

- GeotechStaffEngineer's geotech page has the same gap on Foundry, and a
  wider one. `webapp/identity.py:180-186` binds turns only for a header,
  `webapp/core.py:1733-1745` `working_dir_for` always honours the meta's
  folder, `webapp/app.py:924-926` always draws the box, and there are no
  `_shared_host` checks. Not changed, as the brief says.
- `tests/test_review_binding_offline.py::test_the_process_host_is_the_apps`
  fails when that file runs alone (`own_mark_words` lacks
  "Document Review"), on master 51e478f too. It passes in the combined run.
  This is an order dependence older than INF-80.
- A Foundry conversation whose meta already names a typed folder keeps that
  value, but the page ignores it there now. Files saved into such a folder
  earlier were copied into the conversation then, so their cards still work.
- The comments in `aec_agent_core/_fileio.py` still describe the shared-host
  flag as Tiny Apps only. They are left alone, because this item changes
  nothing in the core.
