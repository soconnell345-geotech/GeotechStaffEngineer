# TA-34 check: SharePoint set up by the host team on any tenant

Branch `claude/ta-34-sharepoint-setup` (2026-11-06). BACKLOG, HANDOFF and
CLAUDE.md are not touched; the lead writes those notes at merge.

## What was built

- **The Microsoft cloud is a setting** (`webapp/graph_sharepoint.py`,
  `cloud_urls`, `CLOUDS`). Sign-in and Graph addresses come from
  `GRAPH_CLOUD`, and either can be overridden. The token scope is
  `<Graph address>/.default`, and every Graph call goes to that address.
- **A certificate as well as a client secret** (`CertificateCredential`).
  The client-credentials flow sends a signed client assertion
  (`client_assertion_type` = jwt-bearer). The assertion is PS256 (PSS
  padding) with header `alg`, `typ`, `x5t#S256` (base64url SHA-256 of the
  certificate's DER), as Microsoft's certificate-credentials page gives it;
  when only a thumbprint is known (the portal shows the SHA-1 one) it is
  RS256 with `x5t`. Claims: `aud` (the token URL), `iss` = `sub` (the client
  id), `jti`, `nbf`, `iat`, `exp` (10 minutes). The certificate and key come
  as PEM (text or a file) or as PKCS #12 (a `.pfx`/`.p12` file, or the base64
  text Key Vault returns for a certificate's secret); a value in neither form
  gets a plain error naming the forms. If a certificate setting is present it
  is used, and any secret is ignored. The secret path is unchanged, byte for
  byte. `cryptography` is not one of the app's required dependencies (it
  comes in only through pdfminer), so it is imported only when a certificate
  is set, and a missing package gives a plain message. The key is never
  logged, repr'd or returned.
- **More than one site** (`Site`, `parse_sites`, `GraphFileManager.sites`).
  Each site keeps its own site id and library cache. In the SharePoint tools
  (`webapp/sharepoint_tools.py`) a configured site's `name:` path is passed
  through unchanged. With two or more sites:
  - the list, download and search tool descriptions name the sites;
  - so does the prompt block;
  - listing the base folder starts with a line naming them.

  The upload tool's description is unchanged; its `dest_folder` takes the same
  prefix, and the prompt says so.
- **A connection test** (`check_connection`, `format_check`).
  - Runs as `python -m webapp.graph_sharepoint --check`: exit 0 when every
    step passes, 1 otherwise. It reads `.env` locally first. `--read-only`
    (`check_connection(write=False)`) skips the write probe; the output then
    shows `SKIP  write` and ends "set up for reading; saving was not
    tested".
  - Appears as a "SharePoint connection" line in the sidebar's Connection
    diagnostics (`webapp/diagnostics.py`), only when a registration setting is
    filled in. It runs the write probe too.
  - Steps: settings, sign-in, each site, the library, write.
  - The write step puts a small probe file, `gse-connection-check.txt`, in
    `GEOTECH_SHAREPOINT_ROOT` (created if missing, as the app's first save
    would) or, when that is not set, the library's root, then deletes it. The
    output says "Wrote and deleted a probe file, <path>". If the delete
    fails, the step still passes (saving works) and names the file to delete
    by hand.
  - Every Graph call of the check waits at most 30 s.
  - Messages are keyed on Entra's AADSTS codes (90002, 900023, 700016,
    7000215, 7000222, 700027, 700024, 7000112, 500011, 7000218, and 900382 /
    900432 for the wrong cloud) and on Graph's HTTP status and error code:
    - 401: no permission with admin consent;
    - 403 or `accessDenied` on the site: not granted under Sites.Selected,
      with the PnP and Graph PowerShell grant;
    - 403 or `accessDenied` on the write: the grant is read-only; grant
      write;
    - 404 or `itemNotFound`: site not found (with the GCC High and DoD
      SharePoint address endings);
    - library not found (the new `LibraryNotFound`, a `GraphError`): the
      libraries the site has;
    - an unreachable host.
  - Graph's or Entra's own text follows each message as "Server said: ...".
- **The setup guide**: `docs/SHAREPOINT_SETUP.md`.
- **The wrapper's `.env.example`** lists the optional settings, commented out.
  The `tinyapps_settings` docstring names them.

## The new settings and their defaults

| Setting | Default |
|---|---|
| `GRAPH_CLOUD` | `commercial` (`gcc` = commercial; `gcc_high`, `dod`; case, spaces and hyphens forgiven) |
| `GRAPH_LOGIN_URL` | the cloud's sign-in address (`https://login.microsoftonline.com`) |
| `GRAPH_API_URL` | the cloud's Graph address (`https://graph.microsoft.com`) |
| `GRAPH_CLIENT_CERT` | unset: the secret is used (PEM path or text, or PKCS #12 path or base64 text) |
| `GRAPH_CLIENT_KEY` | unset: the key is read from `GRAPH_CLIENT_CERT` |
| `GRAPH_CLIENT_KEY_PASSWORD` | unset: the key (or the PKCS #12) has no password |
| `GRAPH_CLIENT_CERT_THUMBPRINT` | unset: computed from the certificate (needed only when no certificate is given) |
| `SHAREPOINT_SITE_URLS` | unset: one site, `SHAREPOINT_SITE_URL` |

`configured()` asks for tenant, client id, a credential (secret, certificate or
key) and at least one site (`SHAREPOINT_SITE_URL` or `SHAREPOINT_SITE_URLS`).
It reads the settings in order and stops at the first present, so a deployment
with today's four settings reads only those four. This matters on Key Vault,
where a setting that is not there is looked up again on every read.

## How a path names its site

- **The default site.** It is the first site: `SHAREPOINT_SITE_URL` if set,
  then the entries of `SHAREPOINT_SITE_URLS` in order. A URL listed twice is
  kept once. A path with no prefix is on it, so every path that worked before
  still works.
- **Another site** is named by `name:Shared Documents/...`. The name is
  matched without regard to case. It comes from `name=URL`, or else from the
  URL's last segment (`/sites/Beta` gives `Beta`).
- **A server path**, `/sites/<site>/...` or `/teams/<site>/...`, of a
  configured site finds that site. An unconfigured one is read on the default
  site, as before.
- **Paths the manager hands back** carry the `name:` prefix when they are not
  on the default site, so they can be passed straight back in. Web links use
  that site's own address.
- **What is not a prefix:** text before a colon that contains `/` or is not a
  site name. It stays part of the path.

## Tests

`webapp/tests/test_graph_sharepoint_setup.py`: 38 tests. They run against a
fake sign-in endpoint and a fake Graph, with made-up ids, the `contoso` placeholder tenant and `example.*` hosts,
and a certificate generated in the test.

- **Clouds:** each cloud's addresses and scope (commercial by default and by
  name, `gcc_high`, "GCC High", `dod`); the explicit overrides; an unknown
  cloud refused.
- **No new settings:** the token request body is byte-for-byte master's
  (`grant_type`, `client_id`, `client_secret`, `scope`, in that order; the
  test compares the raw bytes, not a parsed dict); there is no site note.
  A probe outside pytest ran master's `_get_token` and the branch's side by
  side: method, URL, body and timeout identical.
- **Certificate:**
  - from a file: the assertion's header (PS256, `x5t#S256` = SHA-256 of the
    DER) and claims checked, its PSS signature verified with the test
    certificate's public key;
  - as PEM text with escaped newlines, with a separate encrypted key, on
    `gcc_high`: it wins over a secret;
  - a key with only a thumbprint (hex, as the portal shows it): RS256 with
    `x5t`, PKCS #1 v1.5 signature verified;
  - PKCS #12 as base64 text with no password (as Key Vault returns a
    certificate's secret): signs a valid PS256 assertion; form fields in
    the order grant type, client id, credential, scope;
  - a PKCS #12 file with a password: works; the wrong password or none
    gives a plain `ValueError` naming PKCS #12 and
    `GRAPH_CLIENT_KEY_PASSWORD`, without the password;
  - a binary file that is not PKCS #12, a missing path and junk text: a
    plain `ValueError` naming the accepted forms, never a
    `UnicodeDecodeError`; the connection test reports it in its settings
    step.
- **Sites:**
  - the site-list parsing;
  - path prefixes, server paths, results carrying the prefix, web links, an
    unknown prefix;
  - the site list alone is enough to be configured;
  - the tools and prompt name the sites only when there are several.
- **Connection test:**
  - missing settings named;
  - all steps pass, the write probe put in the library's root and then
    deleted, the output saying so, and every call of the check made with a
    30 s timeout;
  - the write probe in a configured `GEOTECH_SHAREPOINT_ROOT` (its folder
    made as the first save would);
  - a read-only grant (403 `accessDenied` on the probe): the write step
    fails with "can read this site but not write … Grant 'write'", and no
    "set up";
  - a probe the delete left behind is named, and the step still passes;
  - `write=False` and `--check --read-only`: no write, `SKIP  write`,
    "saving was not tested", exit 0;
  - seven sign-in refusals (including 900382 and 900432, the wrong cloud),
    each with the server's text kept and the secret absent;
  - site not found (with the GCC High and DoD address endings), site not
    granted (403, naming the PnP and Graph PowerShell grants and who makes
    them), no consent (401), a second site failing;
  - library not found;
  - the CLI's exit codes;
  - the diagnostics line appears only once a setting is filled in.

Run after the review fixes, from the worktree root (`webapp.__file__` in
the worktree), free RAM 3 GB or more before each run, one at a time, each
judged by pytest's own exit code:

| Command (`python -m pytest … -q -p no:cacheprovider`) | Result | Exit |
|---|---|---|
| `webapp/tests/test_graph_sharepoint_setup.py` | 38 passed | 0 |
| `test_sharepoint_store.py test_sharepoint_tools.py test_sharepoint_privacy.py test_sharepoint_links_and_failures.py test_tinyapps_wiring.py test_diagnostics.py test_databricks_launcher.py test_public_text.py test_page_fingerprints.py` (under `webapp/tests/`) | 232 passed, none skipped | 0 |
| The other tests that import graph_sharepoint, sharepoint_tools or sharepoint_store: `test_wave2a_bookkeeping.py test_wave2b_turns.py test_wave2c_fixes.py test_titles_cards_scratch.py test_record_ev06.py test_output_capture.py test_past_conversations_pages.py test_foundry_dataset_store.py test_app_smoke.py test_core.py` (under `webapp/tests/`) and `report_ingest/tests/test_mirror.py` | 288 passed | 0 |

**The page fingerprints did not move.** No tool input changed. With one site
(or none) the tool descriptions and prompt are the same strings as before. So
`test_top_level_params.py` and `test_defaults_used.py` were not needed.

## Review fixes (2026-11-05)

The review (verdict: merge after fixes) found the code sound and asked for
four fixes before merge and four recommended ones. Each, and its change:

1. **Guide step 3 could not work as written** (PnP with no `-ClientId`, no
   route on GCC High/DoD, who may grant and consent unsaid). Step 3 now
   gives Microsoft Graph PowerShell first (`Connect-MgGraph`, with
   `-Environment USGov` / `USGovDoD` on GCC High / DoD; `Get-MgSite`;
   `New-MgSitePermission` with roles `write`), since Graph Explorer is not
   available on GCC High and DoD; then PnP with `-ClientId` (your own PnP app
   with the delegated Graph `Sites.FullControl.All`, made by
   `Register-PnPEntraIDAppForInteractiveLogin -GraphDelegatePermissions`)
   and `-AzureEnvironment USGovernmentHigh` / `USGovernmentDoD`. It says the
   grant is made by a SharePoint administrator who is a member of the site,
   and step 2 says admin consent for a Graph application permission needs a
   Privileged Role Administrator (or Global Administrator).
   `New-MgSitePermission`'s body and `-BodyParameter` were checked against
   its learn.microsoft.com page, and `-GraphDelegatePermissions` /
   `-AzureEnvironment` against the PnP cmdlet page. `GRANT_ADVICE` names who
   grants and the Graph PowerShell cmdlet.
2. **The wrong-cloud sign-in error fell through to the generic message.**
   AADSTS900382 and AADSTS900432 ("Confidential Client isn't supported in
   Cross Cloud request") now map to "Wrong cloud: … Set GRAPH_CLOUD to the
   tenant's cloud"; both are in the sign-in test's parameters.
3. **DoD SharePoint addresses** end in `.sharepoint-mil.us`. The guide and
   the site-not-found message now say: GCC High `.sharepoint.us`, DoD
   `.sharepoint-mil.us`. Tested.
4. **A PKCS #12 certificate was refused with no hint** (a `.pfx` path even
   raised `UnicodeDecodeError`). `GRAPH_CLIENT_CERT` (and `GRAPH_CLIENT_KEY`)
   now take a `.pfx`/`.p12` file path or its base64 text (what Key Vault
   returns for a certificate's secret), opened with
   `GRAPH_CLIENT_KEY_PASSWORD` when it has a password, through
   `cryptography`'s `pkcs12.load_key_and_certificates`. Files are read as
   bytes, so a value in no known form gets a plain `ValueError` naming the
   forms. Guide: step 4 and the settings table. Tests with a test-made
   PKCS #12.
5. **PS256 with `x5t#S256`**, as Microsoft's certificate-credentials page
   gives it, whenever the certificate is known; RS256 with `x5t` is kept for
   the thumbprint-only setting. Tests verify the PSS signature with the test
   certificate.
6. **The connection test passed on a read-only grant.** It now has a write
   step: a probe file, `gse-connection-check.txt`, written and deleted;
   a 403 says the grant is read-only and to grant write. `--read-only`
   (`write=False`) skips it, and the output says so.
7. **Timeouts:** `check_connection` builds its manager with `timeout=30.0`,
   so every Graph call of the check waits at most 30 s, like the sign-in.
   Tested.
8. **The token form was not byte-for-byte today's.** The fields go out as
   grant type, client id, the credential, scope: with a secret, the same
   bytes as master. The test now compares the raw body with master's.

## What is left

- **Nothing here has met a real tenant.** The addresses are Microsoft's
  published ones; the AADSTS codes are the documented ones.
- **The first real setup** is the host team's run of `--check` on dosdev.
- **The reference-PDF fetch** (`reference_fetch`) and the conversation mirror
  use the default site only. That is by design: other sites are reached
  through the tools.
- **Delegated (per-user) sign-in** is out of scope, as TA-34 says.

## Release note

SharePoint can now be set up by the hosting team on any tenant: `GRAPH_CLOUD`
picks the commercial, GCC High or DoD cloud (or set `GRAPH_LOGIN_URL` /
`GRAPH_API_URL`); a certificate (`GRAPH_CLIENT_CERT`, PEM or PKCS #12, including
a Key Vault certificate) can stand in for the client secret; and
`SHAREPOINT_SITE_URLS` adds further sites, named in a path as `name:`.
`python -m webapp.graph_sharepoint --check` (and the sidebar's Connection
diagnostics) tests the connection step by step, ending with a probe file
written and deleted so a read-only grant shows (`--read-only` skips it); the
guide is `docs/SHAREPOINT_SETUP.md`. With none of the new settings, nothing
changes: the same requests go out, byte for byte.
