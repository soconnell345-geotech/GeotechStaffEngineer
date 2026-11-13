# Setting up SharePoint for Document Review

This page is for the administrator of the team that hosts the app. With it you
connect the app to SharePoint yourselves, on your own tenant, without a code
change. The app keeps each conversation there and reads reference documents
from there. It signs in **as an app registration**, not as a person, and only
reaches the sites you grant it.

## 1. Register the app

In the Entra admin center (on GCC High or DoD, the government portal), go to
**App registrations → New registration**. Give it a name, for example
`Document Review`. Single tenant. No redirect URI.

From its **Overview** page, note:

- **Application (client) ID**: this goes in `GRAPH_CLIENT_ID`
- **Directory (tenant) ID**: this goes in `GRAPH_TENANT_ID`

## 2. Give it the `Sites.Selected` permission

**API permissions → Add a permission → Microsoft Graph → Application
permissions → `Sites.Selected`**, then **Grant admin consent**.

Granting admin consent for a Microsoft Graph application permission needs a
Privileged Role Administrator (or Global Administrator). An Application
Administrator or Cloud Application Administrator cannot consent to it.

`Sites.Selected` on its own reaches **no** site. You grant each site in the
next step.

## 3. Grant each site to the app

A SharePoint administrator who is also a member of the site does this once
per site: Microsoft requires both. Use either method.

**Microsoft Graph PowerShell** (works on every cloud, including GCC High and
DoD, where Graph Explorer is not available):

```powershell
# Commercial and GCC: as shown. GCC High: add -Environment USGov. DoD: add -Environment USGovDoD.
Connect-MgGraph -Scopes Sites.FullControl.All
$site = Get-MgSite -SiteId "<tenant>.sharepoint.com:/sites/<site>"
New-MgSitePermission -SiteId $site.Id -BodyParameter @{
    roles = @("write")
    grantedToIdentities = @(@{ application = @{ id = "<client id>"; displayName = "Document Review" } })
}
```

**PnP PowerShell** needs your own PnP app registration (since September 2024)
with the *delegated* Microsoft Graph permission `Sites.FullControl.All`.
`Register-PnPEntraIDAppForInteractiveLogin ... -GraphDelegatePermissions
"Sites.FullControl.All"` creates one; its default set does not include it.

```powershell
# Commercial and GCC: as shown. GCC High: add -AzureEnvironment USGovernmentHigh. DoD: add -AzureEnvironment USGovernmentDoD.
Connect-PnPOnline -Url https://<tenant>.sharepoint.com/sites/<site> -Interactive -ClientId <your PnP app id>
Grant-PnPAzureADAppSitePermission -AppId <client id> -DisplayName "Document Review" `
    -Site https://<tenant>.sharepoint.com/sites/<site> -Permissions Write
```

(Newer PnP versions call it `Grant-PnPEntraIDAppSitePermission`.)

On commercial and GCC, the raw Graph call also works from Graph Explorer:
`POST /v1.0/sites/<site id>/permissions` with the body above.

The app needs **write**, because it saves conversations. On GCC High,
SharePoint addresses end in `.sharepoint.us`; on DoD, in
`.sharepoint-mil.us`. Graph's address is the government one (see step 5).

## 4. Give it a credential: a secret or a certificate

Under **Certificates & secrets**, do one of these:

- **Client secret.** Click **New client secret** and copy its **Value** (it
  is shown once; the Secret ID is not what you need). It goes in
  `GRAPH_CLIENT_SECRET`. Note its expiry date: the app stops signing in when
  the secret expires.
- **Certificate** (use this where your tenant does not allow secrets).
  Upload the certificate's **public** part (`.cer`, `.crt` or `.pem`) under
  **Certificates → Upload certificate**. Give the app the certificate **with
  its private key**, in `GRAPH_CLIENT_CERT`, in either form:
  - one PEM file (certificate and key), or its text;
  - a PKCS #12 file (`.pfx` or `.p12`), or its base64 text. A certificate
    kept in Key Vault comes back this way by default: name it
    `GRAPH-CLIENT-CERT` and the app reads it by that name. A password, if
    the file has one, goes in `GRAPH_CLIENT_KEY_PASSWORD`.

  For a test certificate:

  ```sh
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=document-review" \
      -keyout app.key -out app.crt
  cat app.key app.crt > app.pem      # app.pem goes to the app; app.crt is uploaded
  ```

  If a certificate is set, the app uses it and ignores any secret.

## 5. Fill in the settings

The settings go in the App Service app settings, in Key Vault (spelled with
hyphens: `GRAPH-TENANT-ID`), or in `.env` for a local run.

| Setting | What to put in it |
|---|---|
| `GRAPH_TENANT_ID` | Directory (tenant) ID (required) |
| `GRAPH_CLIENT_ID` | Application (client) ID (required) |
| `GRAPH_CLIENT_SECRET` | the secret's value, **or** use the certificate settings |
| `GRAPH_CLIENT_CERT` | the certificate and private key: the path to the PEM file or the PEM text itself, or the path to a `.pfx`/`.p12` file or its base64 text |
| `GRAPH_CLIENT_KEY` | only if the private key is kept apart from the certificate: its path or PEM text |
| `GRAPH_CLIENT_KEY_PASSWORD` | only if the private key or the `.pfx`/`.p12` has a password |
| `GRAPH_CLIENT_CERT_THUMBPRINT` | only if `GRAPH_CLIENT_CERT` holds no certificate: the thumbprint the portal shows |
| `SHAREPOINT_SITE_URL` | the site's address, `https://<tenant>.sharepoint.com/sites/<site>` |
| `SHAREPOINT_SITE_URLS` | optional, for more sites: one per line or comma-separated, each `name=address` or just the address |
| `GRAPH_CLOUD` | optional: `commercial`, `gcc_high` or `dod`; when not set, the app takes it from the site's address (below) |
| `GRAPH_LOGIN_URL`, `GRAPH_API_URL` | only for a cloud not in that list: the sign-in and Graph addresses |
| `GEOTECH_SHAREPOINT_ROOT` | optional: the folder the app works in on the first site (default `Shared Documents/GeotechStaffEngineer`) |
| `GEOTECH_SHAREPOINT_LOG` | optional: `0` turns off the app's SharePoint log (step 7); on by default |
| `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB` | optional: the largest file the app sends to SharePoint, in MB (default 2048); see step 8 |

**Which cloud.** Only the cloud of your SharePoint tenant matters; where the
app itself is hosted (an Azure Government subscription or not) does not. With
`GRAPH_CLOUD` not set, the app reads it from the site's address:
`.sharepoint.com` is commercial (also Microsoft 365 GCC), `.sharepoint.us`
is GCC High and `.sharepoint-mil.us` is DoD. `GRAPH_CLOUD`, or
`GRAPH_LOGIN_URL` / `GRAPH_API_URL`, overrides that. The sites listed must
all be in one cloud; sites in two clouds are refused.

The cloud chooses these addresses:

| `GRAPH_CLOUD` | Sign-in | Graph |
|---|---|---|
| `commercial` (also GCC) | `https://login.microsoftonline.com` | `https://graph.microsoft.com` |
| `gcc_high` | `https://login.microsoftonline.us` | `https://graph.microsoft.us` |
| `dod` | `https://login.microsoftonline.us` | `https://dod-graph.microsoft.us` |

**More than one site.** The first site listed (`SHAREPOINT_SITE_URL`, then
`SHAREPOINT_SITE_URLS`) is the default: conversations are kept there, and a
path with no site name is on it. A path on another site starts with that
site's name and a colon, for example `projects:Shared Documents/Reports/a.pdf`.
The name is the one given as `name=address`, or else the last part of the
address (`/sites/Projects` → `Projects`). Grant each site to the app (step 3).

**Sites not in the settings.** A user can paste the address of a file or
folder on any other site of the tenant, and the app opens it there
directly, on that address's own host. This works only for sites granted to
the app (step 3); for any other site the user is told that the app cannot
open it and to ask you to grant it. A site in another Microsoft cloud is
refused. The app never reads the same path on the default site instead.

A sharing link with a token in it (what "Copy link" gives, `/:b:/s/...`) is
read through Microsoft's sharing-link call, which Microsoft does not list
for the site-only permission (`Sites.Selected`). If that call is refused,
the user is asked to paste the address from the browser bar, or the file's
path, instead; granting a site does not change it.

## 6. Test the connection

On the host, in the app's Python environment:

```sh
python -m aec_document_review.graph_sharepoint --check
```

(On a GeotechStaffEngineer install the same test is
`python -m webapp.graph_sharepoint --check`.)

In the app, open **Connection diagnostics** in the sidebar. The line
"SharePoint connection" runs the same test.

The test runs each step in order and stops at the first one that fails:

1. **settings**: names any that are missing;
2. **sign-in**: a wrong tenant ID, client ID, secret or certificate, an
   expired secret, or the wrong cloud;
3. **each site**: the address is wrong (not found), or the site was not
   granted to the app (step 3), or the permission has no admin consent
   (step 2);
4. **the library**: the first folder of `GEOTECH_SHAREPOINT_ROOT` is not a
   document library on the first site (the test lists the libraries there);
5. **write**: the test writes a small probe file, `gse-connection-check.txt`,
   in `GEOTECH_SHAREPOINT_ROOT` (or, if that is not set, the library's top
   folder), and deletes it straight away. If the site was granted only
   `read`, this step fails: grant `write` (step 3). Its output says that it
   wrote and deleted the file.

To test without writing anything, add `--read-only`; the write step is then
skipped, and a read-only grant is not caught.

Each failure says what to do, followed by Microsoft's own error text. When
every step passes, it ends with "SharePoint is set up." The settings line
also says which cloud was used and why (`GRAPH_CLOUD` set, or from the site
address).

## 7. Where the app's SharePoint errors are kept

The app appends every SharePoint error it meets (a tool that failed, a
conversation that could not be saved, a site a user pointed at that it
cannot open) and each connection test's report to a log file in its own
SharePoint folder, the folder it keeps conversations in:

```
<GEOTECH_SHAREPOINT_ROOT, else Shared Documents/GeotechStaffEngineer>/_app_log/sharepoint-YYYY-MM.log
```

One file a month, one line an entry: the time in UTC, the step or tool, the
message, and the path, separated by tabs. It never holds a secret, a token,
a key or conversation text: a conversation that could not be saved is
named by its thread id, never by its title. On a host with several users,
the app's SharePoint tools never open the `_app_log` folder for anyone's
conversation; you read it in SharePoint.

When the app cannot reach SharePoint at all (for example, sign-in is
refused), the sidebar's **Connection diagnostics** shows the report and
offers it as a download instead. That download holds the connection test's
report and the errors met in the downloader's own conversations, never
another user's. Set `GEOTECH_SHAREPOINT_LOG=0` to turn the log off; like
the other settings, it may go in Key Vault.

## 8. How conversations are saved

After each turn the app copies what changed in the conversation's folder to
SharePoint. Nothing here needs setting up; this is what to expect.

- **In the background.** The copy starts when the turn ends and does not
  hold the turn or the page. Up to four conversations are copied at the same
  time, each on its own; a fifth waits for a free place. The sidebar's
  **Permanent storage** block says "saving the latest turn…" until the copy
  is done, then how many files went up. If the app is stopped in that time,
  what was left is sent at the conversation's next save.
- **One request per file.** A file is sent by its full path and SharePoint
  makes the folders. A file above 4 MB goes in pieces, read from disk one
  piece at a time; a piece that fails is taken up again from where
  SharePoint says it stopped, up to five times.
- **A size limit.** A file above `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB` (default
  2048 MB) is not sent. The first save says so as an error that names the
  file and the limit; later saves leave the file alone until it changes, and
  keep naming it in the note under the save line.
- **File names SharePoint refuses.** A name with one of `" * : < > ? / \ |`,
  a space at either end, a dot at the end, or a reserved name (`CON`,
  `desktop.ini` and the like) is kept in SharePoint under an accepted name:
  each refused character becomes `_` and six characters are added before
  the extension (`plan: rev 2.pdf` becomes `plan_ rev 2_xxxxxx.pdf`, the
  same name every time). The conversation's folder then also holds
  `sp_names.json`, which lists those files' own names; restoring the
  conversation gives each file its own name back.
- **A busy service.** When SharePoint asks the app to wait (HTTP 429 or
  503), the conversation being saved waits, up to three times of at most
  20 seconds; the other conversations are not held by that wait. If
  SharePoint is still busy, the rest of that save fails, is reported in the
  sidebar and the log (step 7), and is sent at the next save.
