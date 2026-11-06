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
  `webapp/app.py`).
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
  `…/Forms/AllItems.aspx?id=…`, the `/:f:/r/` links). It goes through the
  tools' `browser_url_to_path` and comes out as `/sites/<name>/…`. In the
  manager, a server path whose site is not configured becomes a site of its
  own on the default site's host (a full address keeps its own host). Its id
  is read with `GET /sites/{host}:/sites/{name}`, the call configured sites
  already used.
- **The site's own address** (`https://<host>/sites/<name>`): that site's
  default library.
- **A token link** (`/:b:/s/...`, `/:f:/g/...`, anything else
  `browser_url_to_path` leaves alone):
  - read with `GET /shares/{u!base64url(link)}/driveItem` (`share_id`,
    encoded as Microsoft documents: base64url, padding dropped, `u!` in
    front);
  - then `GET /drives/{driveId}` for the library's address, which gives the
    site and the library;
  - the link becomes the path it points at; the link-to-path result is
    cached for the manager's life.

  A link into a configured site reads as that site's own path. For example,
  a link to the default site gives `Shared Documents/General/f`.
- **Paths handed back.** For a site not in the settings they keep the
  server-relative form `/sites/<name>/<library>/...`. Passed back, they find
  that site again, since the tools pass `/sites/...` through unchanged. Web
  links use that site's address. A configured site still uses `name:`; the
  default site uses no prefix. Nothing changes for them.
- **Another cloud.** A site whose address names another cloud (say a
  `.sharepoint.us` site from a commercial app) is refused with "that site is
  in the GCC High cloud and the app signs in to the commercial cloud".

**What the user sees when the site is not granted** (Graph answers 403, or
404). The tools return the message as it is, never followed by a name search
or a read on the default site:

> The app cannot open the site https://contoso.sharepoint.com/sites/Gamma:
> it is not one of the sites the app has been granted. The app opens only
> the SharePoint sites its administrator has granted to it. Ask the
> administrator to grant this site to the app (docs/SHAREPOINT_SETUP.md,
> step 3: grant the app the site), or copy the file to a site the app
> already uses. (Graph answered HTTP 403.)

On a 404 it adds "or the address is wrong". For a link it says "cannot open
that link" and adds "or the link has expired".

**No new setting**, and **no tool description or prompt changed**. The tools
already tell the agent to pass a pasted address as it is, and the plain
message carries the rest. `test_page_fingerprints.py`: 15 passed, none
moved, with PyNiteFEA installed so no entry was skipped.

## The log: where and what

- **Where.** `<GEOTECH_SHAREPOINT_ROOT, else "Shared Documents">/_app_log/sharepoint-YYYY-MM.log`
  (UTC month), on the default site, through the store's file manager.
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
  - the configured secret's own value, wherever it appears.

  The callers pass only the app's plain words, Graph's error text and
  paths, never conversation text.
- **Switch.** `GEOTECH_SHAREPOINT_LOG=0` (or `off` / `false` / `no`). Default
  on whenever SharePoint is configured (`sharepoint_store.configured()`, so
  never on the Foundry dataset).
- **Never raising, never looping.**
  - Entries are queued and written by one daemon thread: download the
    month's file, append, upload with overwrite. The turn never waits.
  - Any failure goes to the process log only, and `last_write` holds the
    reason.
  - A `record` made from inside a write is dropped.
  - A file that could not be read for any reason except "not found" is
    never overwritten, so a sign-in failure cannot wipe the log.
  - The last 200 entries stay in memory (`recent()`).
- **Unreachable SharePoint.** When the diagnostics' SharePoint test fails,
  the sidebar shows the report as before and adds a **Download the
  SharePoint report** button: the test's report, whether it reached the log
  (and why not), and the in-memory entries, all scrubbed.

## Tests

`webapp/tests/test_sharepoint_ta35.py`: 20 tests against a fake sign-in
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
  - Paths are matched on the default site's host, since one tenant has one
    SharePoint host. OneDrive (`-my`) addresses are not handled apart.
  - The conversation mirror and the reference fetch still use the default
    site only, by design.
- **The log is one file a month, rewritten whole on each write.** Graph has
  no append. Two processes writing in the same second can lose one batch;
  the entries stay in each process's memory. A very long month makes each
  write longer; nothing rotates within a month.
- **The tools' `TOKEN_LINK_HINT`** (shown only when a token link opens
  nothing and no file is found by name) still says the folder cannot be
  read from such a link. It is left alone so no tool text moves.

## Release note

SharePoint for testers: with `GRAPH_CLOUD` not set, the app takes the
Microsoft cloud from the site's address (`.sharepoint.com`, `.sharepoint.us`,
`.sharepoint-mil.us`). A user can paste a link or address on another site,
including a sharing link, and the app opens it there if that site was granted
to it, else says plainly to ask the administrator (it never reads the same
path on the default site). Every SharePoint error and each connection test's
report is appended to `_app_log/sharepoint-YYYY-MM.log` in the app's
SharePoint folder (`GEOTECH_SHAREPOINT_LOG=0` turns it off), and offered as a
download in the sidebar diagnostics when SharePoint cannot be reached.
