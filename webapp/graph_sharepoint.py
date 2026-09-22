"""SharePoint files over Microsoft Graph with an app registration — no SDK.

The app has mirrored every conversation to SharePoint since 5.10.0 through
the Funhouse SDK's file manager, whose only login is an interactive device
code — fine in a notebook, impossible in a server process. Tiny Apps
provisions the alternative: an Entra APP REGISTRATION (client id + secret)
granted ``Sites.Selected`` on the team's site, the pattern in CfA's
``exampleCode/sharepoint.py`` (2026-08-28). Uploads then act as the app, not
as the person — SharePoint's audit trail shows the app's name.

:class:`GraphFileManager` speaks the SAME six methods the app already calls
on the SDK's manager (``ls``, ``upload_file``, ``download_file``,
``create_folder``, ``search_filenames``, ``get_web_url``), with the same
argument names and return shapes the callers read, so
:mod:`webapp.sharepoint_store`, :mod:`webapp.sharepoint_tools`,
:mod:`webapp.reference_fetch` and ``report_ingest.mirror`` need no change
beyond how the manager is built.

Paths are the UI paths the rest of the app uses — ``Shared Documents/
General/GSE_app/...`` — where the first segment names the document library
(Graph calls it a drive) and the rest is the path inside it.

The login and Graph hosts are the PUBLIC cloud ones on purpose: the
Department's SharePoint tenant is commercial-side even though the App Service
runs in Azure Government (their note). Only the standard library is used;
``urllib`` is what their connector uses and it keeps this file dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

LOGIN_URL = "https://login.microsoftonline.com"
GRAPH_URL = "https://graph.microsoft.com"

#: Graph's simple-upload limit; larger files go through an upload session.
SIMPLE_UPLOAD_MAX = 4 * 1024 * 1024
#: Upload-session chunk: a multiple of 320 KiB, as Graph requires.
CHUNK_SIZE = 10 * 320 * 1024

#: The setting names CfA's engineers provision (env or Key Vault spelling).
ENV_TENANT = "GRAPH_TENANT_ID"
ENV_CLIENT_ID = "GRAPH_CLIENT_ID"
ENV_CLIENT_SECRET = "GRAPH_CLIENT_SECRET"
ENV_SITE_URL = "SHAREPOINT_SITE_URL"


class GraphError(RuntimeError):
    """A Graph call failed; ``status`` is the HTTP status when there was one."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def configured() -> bool:
    """True when the four app-registration settings are all present."""
    from webapp.tinyapps_settings import get_setting
    return all(bool((get_setting(n) or "").strip())
               for n in (ENV_TENANT, ENV_CLIENT_ID, ENV_CLIENT_SECRET,
                         ENV_SITE_URL))


class GraphFileManager:
    """Files on one SharePoint site, as the app registration.

    Construct with explicit values or with none — then the four settings are
    read through :func:`webapp.tinyapps_settings.get_setting` (env var, then
    Key Vault on a deployment, then ``.env`` locally). Nothing is contacted
    until the first call; a bad secret therefore shows up as a request error
    in the log, not a process that will not start.
    """

    def __init__(self, site_url: Optional[str] = None,
                 tenant_id: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 *, timeout: float = 300.0):
        from webapp.tinyapps_settings import get_setting
        self.site_url = (site_url or get_setting(ENV_SITE_URL, required=True)
                         ).strip().rstrip("/")
        self._tenant = tenant_id or get_setting(ENV_TENANT, required=True)
        self._client_id = client_id or get_setting(ENV_CLIENT_ID, required=True)
        self._secret = client_secret or get_setting(ENV_CLIENT_SECRET,
                                                    required=True)
        self._timeout = timeout
        self._token: Optional[str] = None
        self._token_expires = 0.0
        self._site_id: Optional[str] = None
        self._drive_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------ auth
    def _get_token(self) -> str:
        """Client-credentials token for Graph (~1 h), renewed a minute early."""
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._secret,
            "scope": f"{GRAPH_URL}/.default",
        }).encode()
        req = urllib.request.Request(
            f"{LOGIN_URL}/{self._tenant}/oauth2/v2.0/token", data=body,
            method="POST")
        payload = self._send(req, timeout=30.0)
        data = json.loads(payload or b"{}")
        if "access_token" not in data:
            raise GraphError("Token endpoint returned no access_token: "
                             f"{data.get('error_description') or data}")
        self._token = data["access_token"]
        self._token_expires = time.time() + int(data.get("expires_in", 3600))
        return self._token

    # -------------------------------------------------------------- plumbing
    def _send(self, req: urllib.request.Request, timeout: Optional[float] = None
              ) -> bytes:
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise GraphError(f"{req.get_method()} {req.full_url} -> "
                             f"HTTP {exc.code}: {detail}", status=exc.code)
        except urllib.error.URLError as exc:
            raise GraphError(f"{req.get_method()} {req.full_url} failed: "
                             f"{exc.reason}")

    def _request(self, method: str, url: str, data: Optional[bytes] = None,
                 content_type: str = "application/json",
                 extra_headers: Optional[Dict[str, str]] = None,
                 raw: bool = False) -> Any:
        """One Graph call; parsed JSON, or bytes when ``raw``."""
        if not url.startswith("http"):
            url = f"{GRAPH_URL}/v1.0/{url.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        if data is not None:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        body = self._send(urllib.request.Request(url, data=data, headers=headers,
                                                 method=method))
        if raw:
            return body
        return json.loads(body) if body else {}

    @property
    def site_id(self) -> str:
        if self._site_id is None:
            parts = urllib.parse.urlparse(
                self.site_url if "://" in self.site_url
                else f"https://{self.site_url}")
            self._site_id = self._request(
                "GET", f"sites/{parts.netloc}:{parts.path}")["id"]
        return self._site_id

    @staticmethod
    def _clean(path: Optional[str]) -> str:
        p = (path or "").strip().replace("\\", "/")
        # Callers sometimes hand a server-relative path; the site part is ours.
        for marker in ("/sites/", "/teams/"):
            if p.lower().startswith(marker):
                p = p.split("/", 3)[-1] if p.count("/") >= 3 else ""
        return p.strip("/")

    def _split_path(self, path: str):
        """``Shared Documents/a/b.csv`` -> (drive id, ``a/b.csv``).

        The library is matched by the tail of its web URL (``Shared
        Documents``) OR its Graph name (``Documents``) — the default library
        has both, and callers use the UI spelling.
        """
        p = self._clean(path)
        library = p.split("/")[0] if p else "Shared Documents"
        rest = p[len(library):].strip("/") if p else ""
        key = library.lower()
        if key not in self._drive_ids:
            drives = self._request("GET", f"sites/{self.site_id}/drives")["value"]
            for d in drives:
                tail = urllib.parse.unquote(d["webUrl"].rstrip("/").split("/")[-1])
                if key in (tail.lower(), str(d.get("name", "")).lower()):
                    self._drive_ids[key] = d["id"]
                    break
            else:
                names = sorted({urllib.parse.unquote(
                    d["webUrl"].rstrip("/").split("/")[-1]) for d in drives})
                raise GraphError(f"No document library called {library!r} on "
                                 f"{self.site_url}. Libraries here: {names}")
        return self._drive_ids[key], rest

    def _item_url(self, drive_id: str, rest: str) -> str:
        return (f"drives/{drive_id}/root:/{urllib.parse.quote(rest)}"
                if rest else f"drives/{drive_id}/root")

    def _item(self, path: str) -> Dict[str, Any]:
        drive_id, rest = self._split_path(path)
        return self._request("GET", self._item_url(drive_id, rest))

    def _children(self, drive_id: str, item_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        url: Optional[str] = f"drives/{drive_id}/items/{item_id}/children"
        while url:
            page = self._request("GET", url)
            out.extend(page.get("value", []))
            url = page.get("@odata.nextLink")
        return out

    def _entry(self, item: Dict[str, Any], parent_path: str) -> Dict[str, Any]:
        """The dict shape the app's callers read (``name``/``type``/``path``/
        ``size``/``web_url``), plus Graph's own keys for anyone who wants them."""
        name = item.get("name", "")
        is_folder = "folder" in item
        return {
            "name": name,
            "type": "folder" if is_folder else "file",
            "is_folder": is_folder,
            "path": f"{parent_path}/{name}" if parent_path else name,
            "size": item.get("size", 0),
            "web_url": item.get("webUrl", ""),
            "webUrl": item.get("webUrl", ""),
            "lastModifiedDateTime": item.get("fileSystemInfo", {}).get(
                "lastModifiedDateTime", item.get("lastModifiedDateTime", "")),
            "id": item.get("id"),
        }

    # ----------------------------------------------------- the six methods
    def ls(self, path: Optional[str] = None, recursive: bool = False,
           type: str = "both", **_ignored) -> List[Dict[str, Any]]:
        """List a folder (``[]`` when it does not exist). ``type`` is
        ``files`` / ``folders`` / ``both``; ``recursive`` walks subfolders."""
        base = self._clean(path)
        try:
            drive_id, rest = self._split_path(base)
            item = self._request("GET", self._item_url(drive_id, rest))
        except GraphError as exc:
            if exc.status == 404:
                return []
            raise
        if "file" in item:
            return [self._entry(item, base.rsplit("/", 1)[0] if "/" in base else "")]
        out: List[Dict[str, Any]] = []
        for child in self._children(drive_id, item["id"]):
            entry = self._entry(child, base)
            if type == "both" or (type == "files") == (not entry["is_folder"]):
                out.append(entry)
            if recursive and entry["is_folder"]:
                out.extend(self.ls(entry["path"], recursive=True, type=type))
        return out

    def create_folder(self, folder_path: str) -> Dict[str, Any]:
        """Create the folder (and any missing parents); returns its details."""
        drive_id, rest = self._split_path(folder_path)
        parent = self._request("GET", f"drives/{drive_id}/root")
        made = parent
        for name in [p for p in rest.split("/") if p]:
            match = next((c for c in self._children(drive_id, parent["id"])
                          if "folder" in c and c["name"].lower() == name.lower()),
                         None)
            if match is None:
                match = self._request(
                    "POST", f"drives/{drive_id}/items/{parent['id']}/children",
                    data=json.dumps({"name": name, "folder": {},
                                     "@microsoft.graph.conflictBehavior": "fail"}
                                    ).encode())
            parent = made = match
        details = dict(made)
        details["web_url"] = self.get_web_url(folder_path)
        return details

    def upload_file(self, local_path: str, path: str, overwrite: bool = False,
                    **_ignored) -> bool:
        """Upload ``local_path`` to ``path``. False when the target exists and
        ``overwrite`` is False (the caller's retry-with-a-new-name relies on
        that), or when the upload fails; a missing local file raises."""
        with open(local_path, "rb") as fh:
            content = fh.read()
        drive_id, rest = self._split_path(path)
        folder, _, filename = rest.rpartition("/")
        behaviour = "replace" if overwrite else "fail"
        try:
            parent_id = self._ensure_folder(drive_id, folder)
            if len(content) <= SIMPLE_UPLOAD_MAX:
                url = (f"drives/{drive_id}/items/{parent_id}:/"
                       f"{urllib.parse.quote(filename)}:/content"
                       f"?@microsoft.graph.conflictBehavior={behaviour}")
                self._request("PUT", url, data=content,
                              content_type="application/octet-stream")
            else:
                self._upload_large(drive_id, parent_id, filename, content,
                                   behaviour)
            return True
        except GraphError as exc:
            if exc.status == 409 and not overwrite:
                return False
            log.error("SharePoint upload of %s -> %s failed: %s",
                      local_path, path, exc)
            return False

    def _ensure_folder(self, drive_id: str, folder: str) -> str:
        parent = self._request("GET", f"drives/{drive_id}/root")["id"]
        for name in [p for p in folder.split("/") if p]:
            match = next((c for c in self._children(drive_id, parent)
                          if "folder" in c and c["name"].lower() == name.lower()),
                         None)
            if match is None:
                match = self._request(
                    "POST", f"drives/{drive_id}/items/{parent}/children",
                    data=json.dumps({"name": name, "folder": {},
                                     "@microsoft.graph.conflictBehavior": "fail"}
                                    ).encode())
            parent = match["id"]
        return parent

    def _upload_large(self, drive_id: str, parent_id: str, filename: str,
                      content: bytes, behaviour: str) -> None:
        session = self._request(
            "POST", f"drives/{drive_id}/items/{parent_id}:/"
                    f"{urllib.parse.quote(filename)}:/createUploadSession",
            data=json.dumps({"item": {
                "@microsoft.graph.conflictBehavior": behaviour}}).encode())
        url, total = session["uploadUrl"], len(content)
        for start in range(0, total, CHUNK_SIZE):
            chunk = content[start:start + CHUNK_SIZE]
            end = start + len(chunk) - 1
            req = urllib.request.Request(
                url, data=chunk, method="PUT",
                headers={"Content-Range": f"bytes {start}-{end}/{total}",
                         "Content-Length": str(len(chunk))})
            self._send(req)

    def download_file(self, path: str, local_path: Optional[str] = None,
                      return_bytes: bool = True, overwrite: bool = False,
                      **_ignored) -> Union[bytes, bool]:
        """Bytes of ``path``; written to ``local_path`` when given (``True``
        back unless ``return_bytes``). An existing local file is kept unless
        ``overwrite``."""
        drive_id, rest = self._split_path(path)
        content = self._request(
            "GET", f"{self._item_url(drive_id, rest)}:/content"
            if rest else f"drives/{drive_id}/root/content", raw=True)
        if local_path:
            if os.path.exists(local_path) and not overwrite:
                return content if return_bytes else True
            os.makedirs(os.path.dirname(os.path.abspath(local_path)),
                        exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(content)
            if not return_bytes:
                return True
        return content

    def search_filenames(self, filename_filter: Union[str, List[str]],
                         folder_path: str = "/", **_ignored
                         ) -> List[Dict[str, Any]]:
        """Files whose names match ANY filter string, under ``folder_path``."""
        filters = ([filename_filter] if isinstance(filename_filter, str)
                   else list(filename_filter))
        base = self._clean(folder_path)
        drive_id, rest = self._split_path(base or "Shared Documents")
        library = base.split("/")[0] if base else "Shared Documents"
        scope = rest.lower()
        out: List[Dict[str, Any]] = []
        seen = set()
        for f in filters:
            q = urllib.parse.quote(f.replace("'", "''"))
            page = self._request(
                "GET", f"drives/{drive_id}/root/search(q='{q}')")
            for item in page.get("value", []):
                if "folder" in item or item.get("id") in seen:
                    continue
                parent = item.get("parentReference", {}).get("path", "")
                inside = urllib.parse.unquote(parent.split("/root:", 1)[-1]
                                              ).strip("/")
                if scope and not inside.lower().startswith(scope):
                    continue
                seen.add(item.get("id"))
                parent_path = "/".join(p for p in (library, inside) if p)
                out.append(self._entry(item, parent_path))
        return out

    def get_web_url(self, path: str) -> str:
        """The browser URL of a file or folder: the site URL plus the UI path,
        ``?web=1`` so Office files open in the browser rather than download."""
        p = self._clean(path)
        return f"{self.site_url}/{urllib.parse.quote(p)}?web=1" if p \
            else f"{self.site_url}?web=1"


__all__ = ["GraphFileManager", "GraphError", "configured",
           "ENV_TENANT", "ENV_CLIENT_ID", "ENV_CLIENT_SECRET", "ENV_SITE_URL",
           "LOGIN_URL", "GRAPH_URL"]
