# TA-35 check: SharePoint for testers

Branch `claude/ta-35-sharepoint-testers` (2026-11-06). BACKLOG, HANDOFF and
CLAUDE.md are not touched; the lead writes those notes at merge.

The owner's words (2026-11-05): "we can use secrets. They use some Azure gov
cloud, not clear if that matters. Using only one sharepoint site is good, but
might be helpful for a user to point at another one if possible? Any way you
can get those error messages to pass to Sharepoint? We'll be testing it out
anyway."

## What was built

- **The cloud is found by itself** (`webapp/graph_sharepoint.py`:
  `choose_cloud`, `cloud_of_site`, `SHAREPOINT_CLOUDS`). `cloud_urls` and
  `GraphFileManager` use it; the manager keeps `cloud` and `cloud_reason`.
- **A user can point at another site.** `GraphFileManager._route` now opens
  a site that is not in the settings directly (`_site_at`, `_from_address`,
  `_from_share`). It never reads the same path on the default site. A site
  that cannot be opened raises the new `SiteNotOpened`, a `GraphError` with
  `status=None`, so no caller takes it for an empty folder or a missing
  file. It keeps Graph's own status in `http_status` and its plain words in
  `user_message`. The manager also gained `get_file_details`, which the
  download tool already calls, so a sharing link is checked against the
  privacy rules on a multi-user host instead of never being followed.
- **Errors kept in SharePoint**: the new `webapp/sharepoint_log.py`. The
  callers:
  - the SharePoint tools' failures (`_failure`, `_not_answering`, the
    upload errors; `webapp/sharepoint_tools.py`);
  - a conversation save with errors (`SharePointStore.mirror_conversation`,
    SharePoint backend only, not the Foundry dataset);
  - every connection test (`check_connection(..., log=True)`, and the CLI
    waits up to 30 s for the write).

  The sidebar diagnostics offer the report as a download when the
  SharePoint test fails (`diagnostics.sharepoint_download`, a button in
  `webapp/app.py`), holding the test's report and only the downloader's
  own entries (see "Review and fixes", finding 1).
- **The guide** (`docs/SHAREPOINT_SETUP.md`) covers the cloud, sites not in
  the settings, and a new step 7 on where the log is.
- **The wrapper's `.env.example`** gives the cloud rule and the log switch.

## How a cloud is chosen

1. `GRAPH_CLOUD`, when set, wins ("GRAPH_CLOUD is set").
2. Else the sites' addresses decide:
   - `*.sharepoint.com` is commercial (also Microsoft 365 GCC);
   - `*.sharepoint.us` is GCC High;
   - `*.sharepoint-mil.us` is DoD.

   The reason reads "from the site address (contoso.sharepoint.us,
   *.sharepoint.us)".
3. Else commercial ("the default: no site address names a cloud").

`GRAPH_LOGIN_URL` / `GRAPH_API_URL` still replace either address; the
reason then adds "...; GRAPH_API_URL set". Sites in two clouds are refused
whatever `GRAPH_CLOUD` says ("The SharePoint sites are in more than one
Microsoft cloud (...). One app registration signs in to one cloud...").
The connection test then fails at its settings step with that message.
Where the app is hosted plays no part. The guide says so in two sentences.

The connection test's settings line now reads, for example:
`PASS  settings: 1 site(s); sign-in with a secret; cloud GCC High (from the
site address (contoso.sharepoint.us, *.sharepoint.us)); sign-in address
https://login.microsoftonline.us, Graph https://graph.microsoft.us`.

## How another site is opened

What a user can hand the tools, and what happens:

- **A plain address** (`https://<host>/sites/<name>/Shared Documents/...`,
  `…/Forms/AllItems.aspx?id=…`, the `/:f:/r/` links). On the default
  site's host it goes through the tools' `browser_url_to_path` and comes
  out as `/sites/<name>/…`. On any other host the tools pass it on whole
  (fixed after review, finding 4), and the manager reads it with its host:
  the site is matched by name AND host, so the default site's own name on
  another host is that host's site. In the manager, a server path whose
  site is not configured becomes a site of its own on the default site's
  host. Its id is read with `GET /sites/{host}:/sites/{name}`, the call
  configured sites already used.
- **The site's own address** (`https://<host>/sites/<name>`): that site's
  default library.
- **A token link** (`/:b:/s/...`, `/:f:/g/...`, anything else
  `browser_url_to_path` leaves alone):
  - read with `GET /shares/{u!base64url(link)}/driveItem` (`share_id`,
    encoded as Microsoft documents: base64url, padding dropped, `u!` in
    front);
  - when that answer has no `parentReference.path`, the item is read again
    with `GET /drives/{driveId}/items/{id}`; with no folder even then, the
    user is told plainly (never the library's top folder in its place);
  - then `GET /drives/{driveId}` for the library's address, which gives the
    site and the library;
  - the link becomes the path it points at; the link-to-path result is
    cached for the manager's life.
  - Microsoft does not list Sites.Selected for `GET /shares`. When that
    call is refused (401, 403, 404), the user is asked to paste the
    address from the browser bar or the path instead, and is not sent to
    ask for a site grant, which cannot fix it.

  A link into a configured site reads as that site's own path. For example,
  a link to the default site gives `Shared Documents/General/f`.
- **Paths handed back.** For a site not in the settings they keep the
  server-relative form `/sites/<name>/<library>/...`, or the whole address
  (`https://<host>/sites/<name>/<library>/...`) when the site is on another
  host. Passed back, they find that site again, since the tools pass
  `/sites/...` and another host's address through unchanged. Web links use
  that site's address. A configured site still uses `name:`; the default
  site uses no prefix. Nothing changes for them.
- **Another cloud.** A site whose address names another cloud (say a
  `.sharepoint.us` site from a commercial app) is refused with "that site is
  in the GCC High cloud and the app signs in to the commercial cloud",
  through the tools as well as in the manager (fixed after review).

**What the user sees when the site is not granted** (Graph answers 403, or
404). The tools return the message as it is, never followed by a name search
or a read on the default site:

> The app cannot open the site https://contoso.sharepoint.com/sites/Gamma:
> it is not one of the sites the app has been granted. The app opens only
> the SharePoint sites its administrator has granted to it. Ask the
> administrator to grant this site to the app (docs/SHAREPOINT_SETUP.md,
> step 3: grant the app the site), or copy the file to a site the app
> already uses. (Graph answered HTTP 403.)

On a 404 it adds "or the address is wrong". A sharing link the
sharing-link call refuses gets its own words instead (above): "The app
cannot open that sharing link: links of this kind cannot be opened with
the app's permission ... Paste the address from the browser bar ...".

**No new setting**, and **no tool description or prompt changed**. The tools
already tell the agent to pass a pasted address as it is, and the plain
message carries the rest. `test_page_fingerprints.py`: 15 passed, none
moved, with PyNiteFEA installed so no entry was skipped.

## The log: where and what

- **Where.** `<the store's root>/_app_log/sharepoint-YYYY-MM.log` (UTC
  month), on the default site, through the store's file manager. The
  store's root is `GEOTECH_SHAREPOINT_ROOT`, else the app's own folder
  `Shared Documents/GeotechStaffEngineer` (fixed after review: it was the
  library's top folder).
- **What.** One line an entry, tab-separated:
  `2026-11-06T02:10:06Z<TAB>sharepoint download of /sites/Gamma/...<TAB>The app cannot open the site ...<TAB>/sites/Gamma/Shared Documents/r.pdf`.
  Newlines become ` | `; a message is cut at 600 characters, a connection
  test's report at 3,000. A connection test is one entry, step
  `connection test`, whose message starts `PASS | ` or `FAIL | ` followed
  by the `format_check` lines.
- **Never in it.** These are replaced by `[removed]` (`scrub`):
  - bearer tokens and anything shaped like a JWT;
  - `client_secret=`, `client_assertion=`, `access_token`,
    `refresh_token`, `api_key` and `password` values;
  - PEM blocks;
  - the configured secret's own value, wherever it appears: from the
    environment, or from the settings reader's cache, which holds what the
    app read from Key Vault or `.env` (fixed after review: it read the
    environment only).

  The callers pass only the app's plain words, Graph's error text and
  paths, never conversation text. A failed save names its conversation by
  the thread id (`conversations/<thread id>`), never by its folder, whose
  name is the owner and a title made from the first question (fixed after
  review: the folder was logged).
- **Closed to the tools.** On a multi-user host the SharePoint tools refuse
  `_app_log` (list, search, download, upload) for every conversation, as
  they refuse other people's folders, and leave it out of listings and
  search results. The admin reads it in SharePoint.
- **Switch.** `GEOTECH_SHAREPOINT_LOG=0` (or `off` / `false` / `no`), read
  through `get_setting` like the other settings: the environment, else Key
  Vault or `.env`. Default on whenever SharePoint is configured
  (`sharepoint_store.configured()`, so never on the Foundry dataset).
- **Never raising, never looping.**
  - Entries are queued and written by one daemon thread: download the
    month's file, append, upload with overwrite. The turn never waits.
  - Any failure goes to the process log only, and `last_write` holds the
    reason.
  - A `record` made from inside a write is dropped.
  - A file that could not be read for any reason except "not found" is
    never overwritten, so a sign-in failure cannot wipe the log.
  - The last 200 entries stay in memory (`recent()`), each tagged there,
    never in the file, with the owner key of the conversation that met it.
- **Unreachable SharePoint.** When the diagnostics' SharePoint test fails,
  the sidebar shows the report as before and adds a **Download the
  SharePoint report** button: the test's report, whether it reached the log
  (and why not), and the in-memory entries of the downloader's own
  conversations only, all scrubbed (fixed after review: it held every
  user's entries).

## Tests

`webapp/tests/test_sharepoint_ta35.py`: 20 tests (24 after the review's
fixes, below) against a fake sign-in
endpoint and a fake Graph (made-up ids and secret, contoso hosts, no
network).

- **Cloud:**
  - inferred for each of `.sharepoint.com`, `.sharepoint.us` and
    `.sharepoint-mil.us`: sign-in address, scope, Graph host, and the test's
    "cloud … from the site address" line;
  - `GRAPH_CLOUD` wins over the address, and `GRAPH_API_URL` wins too;
  - an address naming no cloud is commercial;
  - sites in two clouds are refused, by the manager and in the test's
    settings step.
- **Another site:**
  - a plain address on a granted site lists and downloads there; the paths
    handed back find it again; the web link uses its address; the default
    site is never touched;
  - an unknown site not granted: `ls` and `download_file` raise
    `SiteNotOpened` naming step 3, with `http_status` 403, while the same
    path holds a file on the default site that is never read; a 404 adds
    "address is wrong";
  - a sharing link: the share id checked against an independent encoding,
    then resolved to `/sites/Gamma/Shared Documents/Reports/r.pdf` and
    downloaded;
  - a link into the default site reads as its own path;
  - a link not granted is said plainly;
  - a site in another cloud is refused;
  - through the tools (`_download`, `_list`, `_search_tool`): the plain
    message, no default-site read, each kept for the log.
- **Log:**
  - the line format and place (with and without `GEOTECH_SHAREPOINT_ROOT`),
    appending to what is there;
  - no secret, JWT, bearer or token value;
  - the off switch: nothing written, still in memory;
  - a failed write: tried once, `last_write` set, no second entry; a write
    attempted from inside a write is dropped; an unreadable (401) log is
    never overwritten; a store that cannot build a client does not raise;
  - a connection test's report logged;
  - a tool failure logged;
  - the diagnostics download on FAIL (no secret), none on PASS or with no
    test.

`webapp/tests/conftest.py` turns the log off for every webapp test (its own
tests turn it on and wait for the write). Left on, the write thread could
outlive a test's fake Graph, and a first run did exactly that.
`test_public_text.py` gains the placeholder host `contoso.sharepoint.us`.

Runs (Python 3.11, each judged by pytest's exit code):

| Command (`python -m pytest … -q -p no:cacheprovider`) | Result | Exit |
|---|---|---|
| `test_graph_sharepoint_setup.py test_sharepoint_store.py test_sharepoint_tools.py test_sharepoint_privacy.py test_sharepoint_links_and_failures.py test_sharepoint_ta35.py test_tinyapps_wiring.py test_diagnostics.py test_databricks_launcher.py test_page_fingerprints.py test_public_text.py` (under `webapp/tests/`) | 290 passed, none skipped | 0 |
| Other tests over the store and tools: `test_wave2a_bookkeeping.py test_wave2b_turns.py test_wave2c_fixes.py test_titles_cards_scratch.py test_output_capture.py test_app_smoke.py test_core.py` (under `webapp/tests/`) | 183 passed | 0 |

## What is left

- **Nothing has met a real tenant.** The site-by-path and `/shares` calls
  and the address endings are Microsoft's published ones.
- **A site not in the settings:**
  - Under Sites.Selected, whether Graph answers 403 or 404 for a site not
    granted varies; both give the plain message.
  - A server path with no host (`/sites/<name>/...`) is read on the
    default site's host; a full address keeps its own host. OneDrive
    (`-my`) addresses are not handled apart.
  - The sharing-link route has met only the fake Graph. Whether
    `GET /shares` answers an app with Sites.Selected is unverified
    (Microsoft does not list that permission for it). The testers' first
    session should paste a "Copy link" link to a file on the default site:
    it either opens, or the user is asked for the browser address.
  - The conversation mirror and the reference fetch still use the default
    site only, by design.
- **The log is one file a month, rewritten whole on each write.** Graph has
  no append. Two processes writing in the same second can lose one batch;
  the entries stay in each process's memory. A very long month makes each
  write longer; nothing rotates within a month.
- **The tools' `TOKEN_LINK_HINT`** (shown only when a token link opens
  nothing and no file is found by name) still says the folder cannot be
  read from such a link. It is left alone so no tool text moves.

## Review and fixes

An independent review of `git diff 05eab2d...HEAD` raised seven findings,
plus one claim the code did not keep (the log's place). Each was fixed on
this branch; the sections above now say what the code does.

1. **HIGH: the diagnostics download held every user's entries.** The
   in-memory entries are now `(owner key, line)`; the tag never reaches the
   file. The tools tag each call with their conversation's owner
   (`sharepoint_log.speaking_for(scope.owner)` around each bound tool), and
   a failed save tags itself with the owner in its meta.
   `diagnostics.sharepoint_download(checks, who)` (the app passes
   `_IDENT.key` on a multi-user host, else `""`) holds the test's report and
   `recent_text(who)` only. Entries no conversation made (a connection test
   run by anyone) are left out on a shared host. The download is also no
   longer waited for when the test passed.
2. **MEDIUM: conversation text in the log, and the log open to every
   user's agent.** `SharePointStore._log_save_errors` logs the path
   `conversations/<thread id>` and replaces the folder's owner/title part
   (as written and URL-quoted, in the error texts too) with the thread id.
   `_Scope.where` now answers "private" for any path in `_app_log`, and
   `_Scope.refusal` says it is the app's own log folder. So list, search,
   download and upload refuse it, and listings and search results leave it
   out. (The optional "log a token link as 'a sharing link'" was not done:
   not in the brief.)
3. **MEDIUM: sharing links under Sites.Selected.** A 401/403/404 from
   `GET /shares/...` itself now raises `SiteNotOpened` with
   `link_not_opened_message`: links of this kind cannot be opened with the
   app's permission; paste the browser-bar address or the path instead. It
   names no site grant. A refusal on the next call (`GET /drives/{id}`)
   keeps the grant-the-site words. With no `parentReference.path`, the item
   is read again by `GET /drives/{driveId}/items/{id}`; with no folder (or
   no `driveId`) even then, a plain `SiteNotOpened` is raised. It never
   falls back to the library's top folder, and never raises a bare
   `KeyError`.
4. **LOW-MEDIUM: the tools dropped a pasted address's host.**
   `sharepoint_tools._resolve` passes an address on another host than the
   default site's (`_default_host`) through whole. The manager's
   `_from_address` now returns `(site, path)`, always takes
   `browser_url_to_path(url, server_relative=True)`, and matches the site
   by name and host. Sites a user pointed at are keyed by host and path.
   Paths handed back for a site on another host carry its whole address.
   New tool-level test: a `.sharepoint.us` address through `_list` is
   refused for the GCC High cloud. An address on `host.sharepoint.com` with
   the default site's own name is refused for that host when not granted.
   When granted, it lists and downloads there (the right bytes), and the
   default site is never read.
5. **LOW: the worker race.** `_run` sets `_worker = None` under the lock
   in the same step that finds the queue empty, so `_queue` always starts a
   worker for a line queued as the old one ends. The test joins the worker,
   checks `_worker is None`, and checks that the next line is written.
6. **LOW: settings from Key Vault.** The switch is read through
   `get_setting`: the environment first, else Key Vault or `.env`, read
   once per process so a record never waits on Key Vault twice. The scrub
   takes the secrets' values from the environment and from
   `tinyapps_settings._cache` (every value read through `get_setting`,
   Key Vault included), with no Key Vault call of its own. The log's place
   follows the store's root, read as the store reads it, so the log always
   sits in the app's folder.
7. **LOW: the failed-save test did not run the save.** It is split. A tool
   failure keeps its own test. The new
   `test_a_failed_save_is_logged_by_its_thread_id` runs
   `mirror_conversation` with a file manager that refuses every upload with
   an error carrying the URL-quoted folder. It checks that the entry's path
   is `conversations/<thread id>`, that no title or owner word is in it,
   that `HTTP 503` is, and that in memory it is the owner's only.
8. **The log's place.** `log_path()` uses `get_store().root()`. With
   `GEOTECH_SHAREPOINT_ROOT` unset, that is
   `Shared Documents/GeotechStaffEngineer/_app_log/`.
   `docs/SHAREPOINT_SETUP.md` step 7 says so. It also covers the thread
   id, the folder closed to the tools, the download holding only the
   downloader's own entries, and the switch in Key Vault. The "Sites not in
   the settings" paragraph now covers another host, another cloud and the
   sharing-link call.

New tests in `test_sharepoint_privacy.py`: `_app_log` is refused for list,
search, download and upload, and is left out of a root listing and a name
search. Two users' tool failures each reach only their own download.

No tool description or prompt changed, so no page fingerprint moved.

Reruns after the fixes (Python 3.11, from the worktree, each file alone,
`python -m pytest -q -p no:cacheprovider webapp/tests/<file>`, logs under
`C:/dev/tmp/ta35_fix/`), each judged by pytest's exit code:

| File | Result | Exit |
|---|---|---|
| `test_sharepoint_ta35.py` | 24 passed | 0 |
| `test_graph_sharepoint_setup.py` | 38 passed | 0 |
| `test_sharepoint_store.py` | 31 passed | 0 |
| `test_sharepoint_tools.py` | 29 passed | 0 |
| `test_sharepoint_privacy.py` | 21 passed | 0 |
| `test_sharepoint_links_and_failures.py` | 33 passed | 0 |
| `test_diagnostics.py` | 34 passed | 0 |
| `test_tinyapps_wiring.py` | 27 passed | 0 |
| `test_page_fingerprints.py` | 15 passed, none moved | 0 |
| `test_public_text.py` | 4 passed | 0 |

## Release note

SharePoint for testers: with `GRAPH_CLOUD` not set, the app takes the
Microsoft cloud from the site's address (`.sharepoint.com`, `.sharepoint.us`,
`.sharepoint-mil.us`). A user can paste an address on another site, and the
app opens it there, on that address's own host, if that site was granted to
it, else says plainly to ask the administrator (it never reads the same path
on the default site). A sharing link is read through Microsoft's
sharing-link call; when the app's permission does not allow that call, the
user is asked for the browser address instead. Every SharePoint error and
each connection test's report is appended to `_app_log/sharepoint-YYYY-MM.log`
in the app's SharePoint folder (`GEOTECH_SHAREPOINT_LOG=0` turns it off; a
failed save is named by its thread id), closed to the agent on a shared
host, and offered as a download in the sidebar diagnostics when SharePoint
cannot be reached, with only the downloader's own entries.
