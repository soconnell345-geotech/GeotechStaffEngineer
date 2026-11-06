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

`Sites.Selected` on its own reaches **no** site. You grant each site in the
next step.

## 3. Grant each site to the app

Someone allowed to manage site permissions (a SharePoint administrator) does
this once per site. Use either method.

**PnP PowerShell:**

```powershell
Connect-PnPOnline -Url https://<tenant>.sharepoint.com/sites/<site> -Interactive
Grant-PnPAzureADAppSitePermission -AppId <client id> -DisplayName "Document Review" `
    -Site https://<tenant>.sharepoint.com/sites/<site> -Permissions Write
```

(Newer PnP versions call it `Grant-PnPEntraIDAppSitePermission`.)

**Microsoft Graph** (the caller needs `Sites.FullControl.All`). First find the
site's id with `GET /v1.0/sites/<tenant>.sharepoint.com:/sites/<site>`, then:

```http
POST /v1.0/sites/<site id>/permissions
Content-Type: application/json

{"roles": ["write"],
 "grantedToIdentities": [{"application": {"id": "<client id>",
                                          "displayName": "Document Review"}}]}
```

The app needs **write**, because it saves conversations. On GCC High and DoD,
SharePoint addresses end in `.sharepoint.us`, and Graph's address is the
government one (see step 5).

## 4. Give it a credential: a secret or a certificate

Under **Certificates & secrets**, do one of these:

- **Client secret.** Click **New client secret** and copy its **Value** (it
  is shown once; the Secret ID is not what you need). It goes in
  `GRAPH_CLIENT_SECRET`. Note its expiry date: the app stops signing in when
  the secret expires.
- **Certificate** (use this where your tenant does not allow secrets).
  Upload the certificate's **public** part (`.cer`, `.crt` or `.pem`) under
  **Certificates → Upload certificate**. Give the app the certificate **with
  its private key** as one PEM file, in `GRAPH_CLIENT_CERT`. For a test
  certificate:

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
| `GRAPH_CLIENT_CERT` | the path to the PEM file, or the PEM text itself (certificate and private key) |
| `GRAPH_CLIENT_KEY` | only if the private key is kept apart from the certificate: its path or PEM text |
| `GRAPH_CLIENT_KEY_PASSWORD` | only if the private key is encrypted |
| `GRAPH_CLIENT_CERT_THUMBPRINT` | only if `GRAPH_CLIENT_CERT` holds no certificate: the thumbprint the portal shows |
| `SHAREPOINT_SITE_URL` | the site's address, `https://<tenant>.sharepoint.com/sites/<site>` |
| `SHAREPOINT_SITE_URLS` | optional, for more sites: one per line or comma-separated, each `name=address` or just the address |
| `GRAPH_CLOUD` | `commercial` (the default), `gcc_high` or `dod` |
| `GRAPH_LOGIN_URL`, `GRAPH_API_URL` | only for a cloud not in that list: the sign-in and Graph addresses |
| `GEOTECH_SHAREPOINT_ROOT` | optional: the folder the app works in on the first site (default `Shared Documents/GeotechStaffEngineer`) |

The cloud settings choose these addresses:

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

## 6. Test the connection

On the host, from the app's folder:

```sh
python -m webapp.graph_sharepoint --check
```

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
   document library on the first site (the test lists the libraries there).

Each failure says what to do, followed by Microsoft's own error text. When
every step passes, it ends with "SharePoint is set up."
