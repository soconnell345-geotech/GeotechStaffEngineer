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
  (`client_assertion_type` = jwt-bearer). The assertion is RS256 with header
  `alg`, `typ`, `x5t` (base64url SHA-1 of the certificate's DER) and claims
  `aud` (the token URL), `iss` = `sub` (the client id), `jti`, `nbf`, `iat`,
  `exp` (10 minutes). If a certificate setting is present it is used, and any
  secret is ignored. The secret path is unchanged. `cryptography` is not one
  of the app's required dependencies (it comes in only through pdfminer), so
  it is imported only when a certificate is set, and a missing package gives a
  plain message. The key is never logged, repr'd or returned.
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
    step passes, 1 otherwise. It reads `.env` locally first.
  - Appears as a "SharePoint connection" line in the sidebar's Connection
    diagnostics (`webapp/diagnostics.py`), only when a registration setting is
    filled in.
  - Steps: settings, sign-in, each site, the library.
  - Messages are keyed on Entra's AADSTS codes (90002, 900023, 700016,
    7000215, 7000222, 700027, 700024, 7000112, 500011, 7000218) and on Graph's
    HTTP status and error code:
    - 401: no permission with admin consent;
    - 403 or `accessDenied`: not granted under Sites.Selected, with the PnP
      and Graph grant;
    - 404 or `itemNotFound`: site not found;
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
| `GRAPH_CLIENT_CERT` | unset: the secret is used |
| `GRAPH_CLIENT_KEY` | unset: the key is read from `GRAPH_CLIENT_CERT` |
| `GRAPH_CLIENT_KEY_PASSWORD` | unset: the key is not encrypted |
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

`webapp/tests/test_graph_sharepoint_setup.py`: 29 tests. They run against a
fake sign-in endpoint and a fake Graph, with made-up ids, `example.*` hosts,
and a certificate generated in the test.

- **Clouds:** each cloud's addresses and scope (commercial by default and by
  name, `gcc_high`, "GCC High", `dod`); the explicit overrides; an unknown
  cloud refused.
- **No new settings:** the token request is byte-for-byte today's form; there
  is no site note.
- **Certificate:**
  - from a file: the assertion's header and claims checked, its signature
    verified with the test certificate's public key;
  - as PEM text with escaped newlines, with a separate encrypted key, on
    `gcc_high`: it wins over a secret;
  - a key with only a thumbprint (hex, as the portal shows it).
- **Sites:**
  - the site-list parsing;
  - path prefixes, server paths, results carrying the prefix, web links, an
    unknown prefix;
  - the site list alone is enough to be configured;
  - the tools and prompt name the sites only when there are several.
- **Connection test:**
  - missing settings named;
  - all steps pass;
  - five sign-in refusals, each with the server's text kept and the secret
    absent;
  - site not found, site not granted (403), no consent (401), a second site
    failing;
  - library not found;
  - the CLI's exit codes;
  - the diagnostics line appears only once a setting is filled in.

Run, each judged by pytest's exit code (all 0):

- the new file plus `test_tinyapps_wiring.py`, `test_sharepoint_tools.py`,
  `test_sharepoint_store.py`, `test_sharepoint_privacy.py`,
  `test_sharepoint_links_and_failures.py`, `test_databricks_launcher.py`,
  `test_page_fingerprints.py` and `test_public_text.py`: 227 passed;
- `test_diagnostics.py`: 34 passed;
- `test_page_fingerprints.py` again with PyNiteFEA installed: 15 passed, none
  skipped.

**The page fingerprints did not move.** No tool input changed. With one site
(or none) the tool descriptions and prompt are the same strings as before. So
`test_top_level_params.py` and `test_defaults_used.py` were not needed.

## What is left

- **Nothing here has met a real tenant.** The addresses are Microsoft's
  published ones; the AADSTS codes are the documented ones.
- **The first real setup** is the host team's run of `--check` on dosdev.
- **The connection test only reads.** A site granted `read` rather than
  `write` passes the test, but uploads then fail. The guide says to grant
  write.
- **The reference-PDF fetch** (`reference_fetch`) and the conversation mirror
  use the default site only. That is by design: other sites are reached
  through the tools.
- **Delegated (per-user) sign-in** is out of scope, as TA-34 says.

## Release note

SharePoint can now be set up by the hosting team on any tenant: `GRAPH_CLOUD`
picks the commercial, GCC High or DoD cloud (or set `GRAPH_LOGIN_URL` /
`GRAPH_API_URL`); a certificate (`GRAPH_CLIENT_CERT`) can stand in for the
client secret; and `SHAREPOINT_SITE_URLS` adds further sites, named in a path as `name:`.
`python -m webapp.graph_sharepoint --check` (and the sidebar's Connection
diagnostics) tests the connection step by step; the guide is
`docs/SHAREPOINT_SETUP.md`. With none of the new settings, nothing changes.
