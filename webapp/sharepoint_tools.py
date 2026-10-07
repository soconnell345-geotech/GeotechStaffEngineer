"""Agent-facing SharePoint tools for the web app (Funhouse SDK backed).

Four LangChain tools that let the AGENT work with the team SharePoint during a
conversation — fetch a project file into the session, push a deliverable out,
browse and search folders:

    sharepoint_list_files(path)            list a folder
    sharepoint_download_file(path, ...)    SharePoint -> working folder
    sharepoint_upload_file(local_path, ..) working folder -> SharePoint
    sharepoint_search_files(query, path)   filename search

They are injected into the deep agent by ``webapp.core.build_agent`` via
``build_deep_agent(extra_tools=...)`` ONLY when SharePoint is configured
(``sharepoint_store.configured()``), so unconfigured deployments carry zero
extra tool surface. The client/auth comes from :mod:`webapp.sharepoint_store`
(same delegated-OAuth token file or client credentials as the mirror).

Path convention (kept simple for the model): a RELATIVE path ("borings/",
"reports/site_A.pdf") resolves under the configured base folder
(``GEOTECH_SHAREPOINT_ROOT``); an absolute form — "Shared Documents/...",
"/sites/<site>/...", or a full "https://..." URL — is passed through
(a SharePoint address copied from a browser is first turned into the path it
names, :func:`browser_url_to_path`).

Every tool returns a plain string and NEVER raises: errors come back as
readable text so the agent can report/retry rather than crash the turn.

FAILURES ARE NEVER ANSWERS (field session 2026-10-06). The SDK's file manager
turns every refused request into an empty answer: ``ls`` returns ``[]`` on a
401, ``search_filenames`` returns ``[]``, and ``download_file`` raises
``FileNotFoundError("... Download failed: 401 ...")``. Rendered as "(empty or
missing folder)", "No files matching" and "SharePoint file not found", they
told the agent -- for eight seconds of a lapsed sign-in -- that a report it
had downloaded twice that afternoon did not exist, and the agent then told
the user its earlier work had never been done. Now: a download error is
classified by its HTTP status; a "not found", an empty listing or an empty
search is checked against the base folder before it is reported, and when
the base folder cannot be read either the tool says SharePoint is not
answering -- an access problem, from which nothing about the file follows.

A LINK THAT DOES NOT OPEN IS NOT THE END (owner, same session: "It can
essentially discern the path from the link, so even if it doesn't directly
work, it should be able to find it using the other sharepoint tools"). When
a download's path comes back not found, the tool tries the address itself
(the SDK resolves a sharing link through Graph's shares API, which also
follows a renamed or moved file), then looks for the file BY ITS NAME -- in
the folder the address named, through the search index, then by walking the
base folder -- and says plainly which file it found and downloaded.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from langchain_core.tools import tool

from webapp import sharepoint_store

#: Cap on listed/search entries so a huge folder cannot blow up the context.
MAX_ENTRIES = 60

#: How many folders a name search may list when the search index finds
#: nothing (one Graph call each, ~0.2 s).
WALK_BUDGET = 40

_NOT_CONFIGURED = ("SharePoint is not configured for this session — no "
                   "site/credentials were provided at launch.")


def _fm():
    return sharepoint_store.get_store().file_manager()


def _root() -> str:
    return sharepoint_store.get_store().root()


def _site_name(url: str) -> str:
    m = re.search(r"/(?:sites|teams)/([^/?#]+)", url or "", re.IGNORECASE)
    return unquote(m.group(1)).lower() if m else ""


def _configured_site() -> str:
    return _site_name(os.environ.get(sharepoint_store.ENV_SITE, "")
                      or os.environ.get("SHAREPOINT_SITE_URL", ""))


# ---------------------------------------------------------------------------
# Links and paths
# ---------------------------------------------------------------------------

#: A sharing link that carries a token instead of a path (``/:f:/s/...``,
#: ``/:b:/g/...``) — nothing in it says which folder it is.
TOKEN_LINK_HINT = (
    "That is a SharePoint sharing link with a token in it, not a path, so "
    "the folder cannot be read from it. Ask for the address in the browser "
    "bar while the folder is open (…/Forms/AllItems.aspx?id=…), or the "
    "folder's path (Shared Documents/…).")

_SITE_PATH = re.compile(r"^/(sites|teams)/([^/]+)/(.*)$", re.IGNORECASE)
_VIEW_SUFFIX = re.compile(r"/forms/[^/]+\.aspx$", re.IGNORECASE)


def browser_url_to_path(url: str) -> Optional[str]:
    """The library path inside a SharePoint address copied from a browser,
    or ``None`` when ``url`` is not one or carries no path.

    The forms that carry a path, all converted:

    * the address bar while a folder (or a file preview) is open,
      ``…/Forms/AllItems.aspx?id=/sites/<site>/Shared Documents/…`` (or
      ``RootFolder=``) -- field session 2026-10-01;
    * the "copy link" form ``/:f:/r/sites/<site>/…`` and its file twins
      ``/:b:/r/`` (PDF), ``/:w:/r/``, ``/:x:/r/``, ``/:p:/r/`` -- the ``r``
      link carries the path, whatever its query (``?d=…&csf=1&web=1&e=…``);
    * a plain address, ``https://<tenant>/sites/<site>/Shared Documents/…``
      (``/teams/`` as well), with any ``?web=1``.

    A link on the configured site becomes the library-relative
    ``Shared Documents/…`` form; another site's keeps ``/sites/<site>/…``.
    These return ``None`` and are left for the SDK, which resolves them
    through Graph's sharing API: a token link (``/:f:/s/…``, ``/:b:/g/…``),
    an Office viewer address (``…/_layouts/15/Doc.aspx?sourcedoc=…``), a
    OneDrive ``/personal/`` address, and any other page (``….aspx``).
    """
    parts = urlsplit((url or "").strip())
    if not parts.scheme.lower().startswith("http"):
        return None
    path = unquote(parts.path)
    query = parse_qs(parts.query)
    for key in ("id", "ID", "Id", "RootFolder", "rootfolder"):
        value = (query.get(key) or [""])[0]
        if value.startswith("/"):
            path = value                      # parse_qs has decoded it
            break
    else:
        m = re.match(r"^/:[a-z]+:/r(/.*)$", path, re.IGNORECASE)
        if m:
            path = m.group(1)
        elif re.match(r"^/:[a-z]+:/", path, re.IGNORECASE):
            return None                       # a token link: no path in it
    if "/_layouts/" in path.lower():
        return None                           # an Office viewer, not a path
    m = _SITE_PATH.match(path)
    if not m:
        return None
    kind, site, rest = m.group(1).lower(), m.group(2), m.group(3).strip("/")
    rest = _VIEW_SUFFIX.sub("", "/" + rest).strip("/")
    if not rest or rest.lower().endswith(".aspx"):
        return None
    segments = rest.split("/")
    if len(segments) == 1 and "." in segments[0]:
        return None                           # a file at the site root, no library
    if (kind == "sites" and site.lower() == _configured_site()
            and segments[0].lower() == "shared documents"):
        return rest
    # Another site, a team, or another library of this site: the
    # server-relative form, which both file clients read as absolute.
    return f"/{kind}/{site}/{rest}"


def link_target_name(path_or_url: str) -> str:
    """The decoded name of the file or folder an address points at, when it
    says: the ``file=`` of an Office viewer link, else the last segment of
    the path it carries. ``""`` for a token link, which names nothing."""
    s = (path_or_url or "").strip()
    if s.lower().startswith("http"):
        q = parse_qs(urlsplit(s).query)
        named = (q.get("file") or [""])[0]
        if named:
            return os.path.basename(unquote(named).replace("\\", "/"))
        p = browser_url_to_path(s)
        return p.rstrip("/").rsplit("/", 1)[-1] if p else ""
    return s.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _resolve(path: Optional[str]) -> str:
    """Resolve a tool-supplied path against the configured base folder.

    A SharePoint address copied from a browser is first turned into the path
    it names (:func:`browser_url_to_path`). Absolute forms (full URL,
    "/sites/...", "Shared Documents/...") pass through UNCHANGED (a leading
    slash is meaningful to the SDK); anything else is joined under the
    configured root.
    """
    p = (path or "").strip().rstrip("/")
    if p.lower().startswith("http"):
        p = (browser_url_to_path(p) or p).rstrip("/")
    if not p or p == "/":
        return _root()
    low = p.lstrip("/").lower()
    if (low.startswith("http") or low.startswith("sites/")
            or low.startswith("teams/")
            or low.startswith("shared documents")):
        return p
    rel = p.lstrip("/")
    root_parts = [s for s in _root().split("/") if s]
    parts = rel.split("/")
    # The library as the SharePoint page shows it -- the breadcrumb reads
    # "Documents > General > GSE_app", while the path says "Shared
    # Documents/General/GSE_app". A path that starts "Documents/" and then
    # continues the root's own folders is that breadcrumb, not a folder
    # called Documents under the root.
    if (len(parts) > 1 and len(root_parts) > 1
            and parts[0].lower() == "documents"
            and root_parts[0].lower() == "shared documents"
            and parts[1].lower() == root_parts[1].lower()):
        return "/".join([root_parts[0]] + parts[1:])
    # Naming the folder you can SEE is the obvious thing to do, and it used to
    # double: the root already ENDS in the base folder, so "GSE_app/uploaded
    # references/x.pdf" resolved to ".../GSE_app/GSE_app/uploaded references/
    # x.pdf" and 404'd (live 2026-09-09/10, 4-5 wasted round trips per turn).
    # On 2026-09-15 the agent named TWO of the root's segments
    # ("General/GSE_app/conversations/...") and an upload landed under
    # ".../General/GSE_app/General/GSE_app/..." (field feedback N5). Drop the
    # longest leading run that repeats the end of the root, as long as
    # something is left after it.
    for k in range(min(len(root_parts), len(parts) - 1), 0, -1):
        if ([s.lower() for s in parts[:k]]
                == [s.lower() for s in root_parts[-k:]]):
            parts = parts[k:]
            break
    return f"{_root()}/{'/'.join(parts)}"


def _parent(remote: str) -> str:
    return remote.rstrip("/").rsplit("/", 1)[0] if "/" in remote else ""


def _working_dir() -> str:
    """Where downloads land: the conversation's working folder when set."""
    try:
        from funhouse_agent._fileio import default_output_dir
        d = default_output_dir()
        if d:
            return str(d)
    except Exception:
        pass
    return tempfile.gettempdir()


def _fmt_entry(e: dict) -> str:
    name = e.get("name") or "?"
    kind = e.get("type") or ("folder" if e.get("folder") else "file")
    size = e.get("size")
    size_s = f", {int(size):,} B" if isinstance(size, (int, float)) else ""
    path = e.get("path") or e.get("url") or ""
    return f"- [{kind}] {name}{size_s}" + (f"  ({path})" if path else "")


def _is_folder(e: dict) -> bool:
    kind = str(e.get("type") or "").lower()
    if kind:
        return kind.startswith("folder")
    return bool(e.get("is_folder") or e.get("folder"))


# ---------------------------------------------------------------------------
# Telling a failure from an answer
# ---------------------------------------------------------------------------

_STATUS_IN_TEXT = re.compile(
    r"(?:Download failed:|HTTP|status(?: code)?[:= ]|->)\s*(\d{3})\b",
    re.IGNORECASE)


def _status(exc: BaseException) -> Optional[int]:
    """The HTTP status a SharePoint error carries, when it carries one."""
    for attr in ("status", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(value, int):
        return value
    m = _STATUS_IN_TEXT.search(str(exc))
    return int(m.group(1)) if m else None


def _classify(exc: BaseException) -> str:
    """``missing`` | ``refused`` | ``throttled`` | ``error``."""
    status = _status(exc)
    if status == 404:
        return "missing"
    if status in (401, 403):
        return "refused"
    if status == 429:
        return "throttled"
    if status is not None:
        return "error"
    text = str(exc).lower()
    if any(k in text for k in ("unauthor", "forbidden", "accessdenied",
                               "access denied", "invalidauthenticationtoken")):
        return "refused"
    if isinstance(exc, FileNotFoundError) or "not found" in text \
            or "itemnotfound" in text:
        return "missing"
    return "error"


def _probe(fm) -> Tuple[bool, str]:
    """Can SharePoint be read at all right now? Reads the base folder, which
    always exists and is never empty. ``(ok, detail)``."""
    root = _root()
    try:
        details = getattr(fm, "get_folder_details", None)
        if callable(details):
            try:
                if details(root):
                    return True, ""
            except Exception:                              # noqa: BLE001
                pass
        if fm.ls(root):
            return True, ""
        return False, "the base folder came back empty"
    except Exception as exc:                               # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"


def _not_answering(what: str, detail: str = "") -> str:
    """The message for a negative answer the base folder cannot vouch for."""
    return (f"SharePoint is not answering: {what} came back empty, and the "
            f"app's base folder ({_root()}) could not be read either"
            + (f" ({detail})" if detail else "")
            + ". That is an ACCESS problem -- usually an expired sign-in, "
            "sometimes a short outage -- not a missing file or folder, so "
            "nothing can be concluded about whether it exists. Files already "
            "downloaded in this conversation are still in the working folder "
            "(list_files). Tell the user SharePoint is not answering right "
            "now and try again in a few minutes.")


def _failure(what: str, exc: BaseException) -> str:
    """A download/list/search failure, said for what it is."""
    kind = _classify(exc)
    status = _status(exc)
    code = f"HTTP {status}: " if status else ""
    head = f"SharePoint {what} failed ({code}{type(exc).__name__}: " \
           f"{str(exc)[:300]})"
    if kind == "refused":
        return (head + ". SharePoint REFUSED the request: an expired sign-in "
                "(401) or no permission (403). This is not a missing file; "
                "nothing can be concluded about whether it exists. Tell the "
                "user, and try again in a few minutes.")
    if kind == "throttled":
        return (head + ". SharePoint is throttling requests (429); wait a "
                "minute and try again. This says nothing about the file.")
    return (head + ". The request failed; this says nothing about whether "
            "the file or folder exists.")


# ---------------------------------------------------------------------------
# Finding a file by its name
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """A name for comparing: lower case, every run of non-letters and
    non-digits one space ("Site _Volume I_09.pdf" -> "site volume i 09
    pdf")."""
    return " ".join(re.split(r"[^0-9a-z]+", unquote(str(name or "")).lower())
                    ).strip()


def _name_matches(query: str, name: str) -> bool:
    """A name a person would accept for ``query``: the query inside the name,
    or every word of the query among the name's words, in any order."""
    q, n = _norm(query), _norm(name)
    if not q or not n:
        return False
    if q in n:
        return True
    words = set(n.split())
    return all(w in words for w in q.split())


def _queries(text: str) -> List[str]:
    """What to ask the search index: the text as given, without its file
    extension, and then its longest word alone. The index can miss a long
    name with underscores in it (field session 2026-10-06: the exact name of
    a file listed in the folder found nothing, three times); one word finds
    the candidates, and the caller keeps only the ones whose NAME matches."""
    out: List[str] = []
    stem = os.path.splitext(text)[0] if re.search(r"\.[A-Za-z0-9]{2,5}$",
                                                  text or "") else text
    for q in (text, stem):
        q = (q or "").strip()
        if q and q not in out:
            out.append(q)
    words = sorted({w for w in re.split(r"[^0-9A-Za-z]+", stem or "")
                    if len(w) >= 4 and not w.isdigit()},
                   key=lambda w: (-len(w), w))
    if words and words[0] not in out:
        out.append(words[0])
    return out[:3]


def _search(fm, query: str, scope: str) -> List[dict]:
    try:
        hits = fm.search_filenames(query, scope)
    except TypeError:                    # backend without a path arg
        hits = fm.search_filenames(query)
    return [h for h in (hits or []) if isinstance(h, dict)]


def _walk(fm, base: str, budget: int = WALK_BUDGET) -> Tuple[List[dict], int]:
    """Files under ``base``, breadth first, listing at most ``budget``
    folders. The app's own mirror of conversations is walked last: the
    files people mean are almost never in it. ``(files, folders_listed)``."""
    files: List[dict] = []
    queue = deque([base])
    later: List[str] = []
    seen = set()
    listed = 0
    own_mirror = f"{_root()}/conversations".lower()
    while (queue or later) and listed < budget:
        folder = queue.popleft() if queue else later.pop(0)
        key = folder.lower().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        try:
            entries = fm.ls(folder) or []
        except Exception:                                  # noqa: BLE001
            entries = []
        listed += 1
        for e in entries:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "")
            path = str(e.get("path") or f"{folder}/{name}")
            if _is_folder(e):
                (later if path.lower().rstrip("/").endswith(own_mirror)
                 or own_mirror in path.lower() else queue).append(path)
            else:
                files.append({**e, "path": path})
    return files, listed


def _find_by_name(fm, name: str, near: str = "") -> Tuple[List[dict], str]:
    """Files whose name IS ``name`` (compared by :func:`_norm`): in the
    folder ``near`` first, then through the search index, then by a bounded
    walk of the base folder. ``(hits, how)``."""
    target = _norm(name)
    if not target:
        return [], ""

    def keep(entries) -> List[dict]:
        out, seen = [], set()
        for e in entries:
            if _is_folder(e) or _norm(e.get("name")) != target:
                continue
            path = str(e.get("path") or "")
            if path.lower() in seen:
                continue
            seen.add(path.lower())
            out.append(e)
        return out

    if near:
        try:
            hits = keep({**e, "path": e.get("path") or f"{near}/{e.get('name')}"}
                        for e in (fm.ls(near) or []) if isinstance(e, dict))
        except Exception:                                  # noqa: BLE001
            hits = []
        if hits:
            return hits, f"in the folder the address named ({near})"
    for q in _queries(name):
        try:
            hits = keep(_search(fm, q, _root()))
        except Exception:                                  # noqa: BLE001
            hits = []
        if hits:
            return hits, "through the SharePoint search"
    files, listed = _walk(fm, _root())
    hits = keep(files)
    return hits, (f"by looking through {listed} folders under {_root()}"
                  if hits else f"(looked through {listed} folders)")


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

@tool
def sharepoint_list_files(path: str = "") -> str:
    """List files and folders in a SharePoint folder.

    path: folder to list — relative to the app's base SharePoint folder
    (default "" = the base folder itself), or an absolute
    "Shared Documents/..." / "/sites/..." path, or a SharePoint address
    copied from the browser.
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    try:
        fm = _fm()
        remote = _resolve(path)
        entries = fm.ls(remote) or []
    except Exception as exc:
        return _failure("list", exc).replace("SharePoint list failed",
                                             "SharePoint list error", 1)
    if entries:
        lines = [f"Contents of {remote} ({len(entries)} items"
                 + (f", first {MAX_ENTRIES} shown" if len(entries) > MAX_ENTRIES
                    else "") + "):"]
        lines += [_fmt_entry(e) for e in entries[:MAX_ENTRIES]]
        return "\n".join(lines)
    # Nothing came back: an empty folder, no folder, or a refused request
    # the SDK turned into [] -- three different answers.
    details = getattr(fm, "get_folder_details", None)
    if callable(details):
        try:
            got = details(remote) or {}
        except Exception:                                  # noqa: BLE001
            got = {}
        if got and (got.get("folder") is not None or got.get("id")):
            return f"(empty folder: {remote})"
    ok, detail = _probe(fm)
    if not ok:
        return _not_answering(f"listing {remote}", detail)
    msg = f"(empty or missing folder: {remote})"
    if remote.lower().startswith("http") and browser_url_to_path(remote) is None:
        return f"{msg}\n{TOKEN_LINK_HINT}"
    return msg


#: (working folder, remote path) -> local copy: a file already fetched is
#: reused instead of downloaded again (field feedback 2026-09-15, N8: the same
#: 23 MB submittal was downloaded four times under two names).
_DOWNLOADS: dict = {}


def _prior_copy(dest_dir: str, remotes, record_dir: Optional[str]
                ) -> Optional[str]:
    """The local copy of any of ``remotes`` this conversation already has."""
    keys = {str(r).lower() for r in remotes if r}
    for r in keys:
        hit = _DOWNLOADS.get((os.path.abspath(dest_dir), r))
        if hit and os.path.isfile(hit):
            return hit
    if record_dir:
        from webapp import core
        for d in reversed(core.load_downloads(record_dir)):
            if str(d.get("remote", "")).lower() in keys \
                    and os.path.isfile(str(d.get("local"))):
                return str(d["local"])
    return None


def _safe_name(name: str) -> str:
    name = os.path.basename(str(name or "").replace("\\", "/")).strip()
    return re.sub(r'[<>:"|?*\x00-\x1f]', "_", name) or "download"


def _place(part: str, dest_dir: str, name: str, keep: Optional[str]) -> str:
    """Move the fetched ``part`` file to its home and return the path.

    ``keep`` is this SharePoint file's existing copy (a refresh): replaced in
    place, so a file keeps ONE name in the conversation. Otherwise ``name``
    in ``dest_dir`` -- unless a different file already has that name, which
    is never overwritten (a same-named upload, or a file from another
    folder); identical bytes are simply the same file."""
    if keep:
        os.replace(part, keep)
        return keep
    target = os.path.join(dest_dir, _safe_name(name))
    if os.path.exists(target):
        from webapp import core
        if core._same_bytes(part, target):
            os.remove(part)
            return target
        target = core._unique_dest(target)
    os.replace(part, target)
    return target


def _fetch(fm, target: str, dest_dir: str) -> str:
    """Download ``target`` to a hidden part file in ``dest_dir``; its path.
    A download that reports success but writes nothing is a failure."""
    os.makedirs(dest_dir, exist_ok=True)
    fd, part = tempfile.mkstemp(prefix=".sp_download_", suffix=".part",
                                dir=dest_dir)
    os.close(fd)
    try:
        fm.download_file(target, local_path=part, return_bytes=False,
                         overwrite=True)
        if not os.path.isfile(part):
            raise RuntimeError(f"the download of {target} wrote no file")
        return part
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


def _download(path: str, save_as: str, refresh: bool,
              record_dir: Optional[str]) -> str:
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    original = (path or "").strip()
    try:
        fm = _fm()
        remote = _resolve(original)
    except Exception as exc:
        return _failure("download", exc).replace(
            "SharePoint download failed", "SharePoint download error", 1)
    dest_dir = _working_dir()
    prior = _prior_copy(dest_dir, [remote], record_dir)
    if prior and not refresh:
        return (f"Downloaded {remote} -> {prior} "
                f"({os.path.getsize(prior):,} bytes) earlier in this "
                "conversation; reusing that copy (refresh=true fetches it "
                "again into the same file). It is an input to read, not a "
                "deliverable.")
    def local_name(found_remote: str) -> str:
        if save_as:
            return save_as
        if not found_remote.lower().startswith("http"):
            return os.path.basename(found_remote.rstrip("/"))
        named = link_target_name(found_remote)
        if named:
            return named
        try:                            # a sharing link names nothing
            details = fm.get_file_details(found_remote) or {}
            if details.get("name"):
                return str(details["name"])
        except Exception:                                  # noqa: BLE001
            pass
        return "sharepoint_download"

    def finish(found_remote: str, part: str, preface: str = "") -> str:
        keep = _prior_copy(dest_dir, [remote, found_remote], record_dir)
        local = _place(part, dest_dir, local_name(found_remote), keep)
        size = os.path.getsize(local)
        for r in {remote, found_remote}:
            _DOWNLOADS[(os.path.abspath(dest_dir), r.lower())] = local
            if record_dir:
                from webapp import core
                core.record_download(record_dir, r, local, size)
        kept = (f" One copy per SharePoint file: refreshed '{os.path.basename(local)}' in place."
                if keep else "")
        return (preface
                + f"Downloaded {found_remote} -> {local} ({size:,} bytes). "
                "The file is now in the working folder and available to the "
                "file tools by this path or by its name. It is an input to "
                "read, not a deliverable." + kept
                + (" (The file is empty.)" if size == 0 else ""))

    attempts = [remote]
    if original.lower().startswith("http") and original != remote:
        attempts.append(original)       # the SDK's sharing-link route
    for target in attempts:
        try:
            part = _fetch(fm, target, dest_dir)
        except Exception as exc:                           # noqa: BLE001
            if _classify(exc) != "missing":
                return _failure(f"download of {target}", exc)
            continue
        # A sharing link that resolved: file it under the path it was given
        # as when that path is a real one, else under the link itself.
        return finish(remote if (target == original
                                 and not remote.lower().startswith("http"))
                      else target, part)

    # Every direct attempt said "not found". Before believing it, make sure
    # SharePoint is answering at all -- the SDK reports a refused request as
    # FileNotFoundError too.
    ok, detail = _probe(fm)
    if not ok:
        return _not_answering(f"downloading {remote}", detail)
    wanted = link_target_name(original) or os.path.basename(remote)
    hits, how = _find_by_name(fm, wanted, near=_parent(remote))
    if len(hits) == 1:
        found = str(hits[0]["path"])
        try:
            part = _fetch(fm, found, dest_dir)
        except Exception as exc:                           # noqa: BLE001
            return _failure(f"download of {found}", exc)
        return finish(found, part, preface=(
            f"The address given did not open directly ('{remote}' was not "
            f"found there). A file with the same name, "
            f"'{hits[0].get('name')}', was found {how} at '{found}', and "
            "that file was downloaded. Tell the user which file you used.\n"))
    if len(hits) > 1:
        rows = "\n".join(_fmt_entry(h) for h in hits[:MAX_ENTRIES])
        return (f"SharePoint file not found at '{remote}'. {len(hits)} files "
                f"named '{wanted}' were found {how}; nothing was downloaded. "
                "Download the one meant by its path (ask the user if it is "
                f"not clear):\n{rows}")
    tail = f" {TOKEN_LINK_HINT}" if (original.lower().startswith("http")
                                    and not wanted) else ""
    return (f"SharePoint file not found: {remote} — and no file named "
            f"'{wanted}' was found by name either {how}. SharePoint is "
            "answering (the base folder reads normally). Check the path with "
            "sharepoint_list_files on its folder, or sharepoint_search_files "
            "with a word from its name." + tail)


_DOWNLOAD_DOC = """Download a file from SharePoint into the session working folder, so it
can be READ with the file tools (read_pdf_text, open_document, subsurface
parsers, ...). It is an input, not a deliverable. A file already downloaded
in this conversation is reused unless ``refresh`` is true, and a refresh
replaces that same copy, so each SharePoint file has ONE local copy.

path: the SharePoint file — relative to the base folder, absolute
("Shared Documents/...", "/sites/..."), or ANY SharePoint address the user
pasted (a "copy link" address, the browser's address bar, an Office viewer
link). If it does not open directly, the file is looked for by its name and
the result says which file was downloaded.
save_as: optional local filename for a NEW download (defaults to the
SharePoint name); ignored when this file already has a local copy.
refresh: download again (into the same local file) even if already fetched.
"""


def make_download_tool(record_dir: Optional[str] = None):
    """``sharepoint_download_file`` bound to one conversation's record folder,
    where it keeps the downloads ledger (``core.record_download``): which
    SharePoint file became which local file -- one copy per file, kept
    across a restart of the app, and named to the agent on every turn
    (``core.working_files_note``)."""

    def sharepoint_download_file(path: str, save_as: str = "",
                                 refresh: bool = False) -> str:
        return _download(path, save_as, refresh, record_dir)

    sharepoint_download_file.__doc__ = _DOWNLOAD_DOC
    return tool(sharepoint_download_file)


#: The download tool with no conversation ledger (in-process reuse only).
sharepoint_download_file = make_download_tool()


def _upload(local_path: str, dest_folder: str, default_folder=None,
            default_label: str = "") -> str:
    """Shared body of both upload tools; ``default_folder`` (a callable
    returning the remote folder) applies when ``dest_folder`` is empty."""
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    if not os.path.isfile(local_path):
        return f"Local file not found: {local_path}"
    try:
        fm = _fm()
        if (dest_folder or "").strip() or default_folder is None:
            folder, label = _resolve(dest_folder), ""
        else:
            folder, label = default_folder(), default_label
        try:
            fm.create_folder(folder)
        except Exception:
            pass                                    # may already exist
        name = os.path.basename(local_path)
        remote = f"{folder}/{name}"
        ok = fm.upload_file(local_path, remote, overwrite=False)
        if not ok:                                  # name taken -> unique name
            stem, ext = os.path.splitext(name)
            remote = f"{folder}/{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            ok = fm.upload_file(local_path, remote, overwrite=False)
        if not ok:
            return f"SharePoint upload failed for {remote} (upload rejected)."
        try:
            from webapp.sharepoint_store import fix_web_url
            url = fix_web_url(fm.get_web_url(remote))
        except Exception:
            url = ""
        return (f"Uploaded {local_path} -> {remote}{label}."
                + (f" Link: {url}" if url else ""))
    except Exception as exc:
        return f"SharePoint upload error: {type(exc).__name__}: {exc}"


@tool
def sharepoint_upload_file(local_path: str, dest_folder: str = "") -> str:
    """Upload a local file (e.g. a calc package or plot from the working
    folder) to SharePoint.

    local_path: the local file to upload (as returned by save_file /
    calc-package tools).
    dest_folder: SharePoint folder — relative to the base folder (default ""
    = the base folder), or absolute. Created if missing. An existing file of
    the same name is NOT overwritten — a timestamped name is used instead.
    """
    return _upload(local_path, dest_folder)


def make_conversation_upload_tool(thread_id: str):
    """``sharepoint_upload_file`` bound to one conversation: with no
    ``dest_folder`` the file goes to that conversation's SharePoint folder
    (``<root>/conversations/<title>_<date>/files``, beside everything the
    mirror keeps there).

    Field feedback 2026-09-15 (N5): asked to save a report to SharePoint, the
    agent chose the users' "uploaded references" folder, then guessed a
    conversation path by thread id -- but the mirror names the folder by title
    and date, which the agent had no way to know.
    """
    def _folder():
        return f"{sharepoint_store.get_store().session_folder(thread_id)}/files"

    @tool
    def sharepoint_upload_file(local_path: str, dest_folder: str = "") -> str:
        """Upload a local file (a report, figure or calc package) to SharePoint.

        local_path: the local file (as returned by save_file or the tool that
        built it).
        dest_folder: leave EMPTY to put the file in this conversation's
        SharePoint folder (the usual choice). Otherwise a folder relative to
        the base folder, or absolute. Created if missing. An existing file of
        the same name is not overwritten; a timestamped name is used instead.
        """
        return _upload(local_path, dest_folder, default_folder=_folder,
                       default_label=" (this conversation's SharePoint folder)")

    return sharepoint_upload_file


@tool
def sharepoint_search_files(query: str, path: str = "") -> str:
    """Search SharePoint for files by name.

    query: filename text to search for (e.g. "boring log", "Kinshasa"), or a
    whole file name.
    path: optional folder to scope the search — relative to the base folder,
    or absolute. Default searches from the base folder.

    NOTE: search uses an index that lags NEW uploads by several minutes and
    can miss long names with underscores; when the index finds nothing the
    tool also tries shorter queries and looks through the folders by name.
    """
    if not sharepoint_store.configured():
        return _NOT_CONFIGURED
    try:
        fm = _fm()
        remote = _resolve(path)
        hits = _search(fm, query, remote)
    except Exception as exc:
        return _failure("search", exc).replace("SharePoint search failed",
                                               "SharePoint search error", 1)
    how = ""
    if not hits:
        # The index found nothing for the text as given. Shorter queries
        # first, then the folders themselves -- matched by name.
        seen = set()
        for q in _queries(query)[1:]:
            try:
                more = _search(fm, q, remote)
            except Exception:                              # noqa: BLE001
                more = []
            for h in more:
                key = str(h.get("path") or h.get("name")).lower()
                if key not in seen and _name_matches(query, h.get("name")):
                    seen.add(key)
                    hits.append(h)
        if hits:
            how = " (found by a shorter search; the full text found nothing)"
        else:
            files, listed = _walk(fm, remote)
            hits = [f for f in files if _name_matches(query, f.get("name"))]
            if hits:
                how = (f" (found by looking through {listed} folders; the "
                       "search index found nothing)")
            else:
                ok, detail = _probe(fm)
                if not ok:
                    return _not_answering(f"the search for '{query}'", detail)
                return (f"No files matching '{query}' under {remote} (the "
                        f"search index found none, and {listed} folders were "
                        "looked through by name).")
    lines = [f"Matches for '{query}' ({len(hits)}"
             + (f", first {MAX_ENTRIES} shown" if len(hits) > MAX_ENTRIES
                else "") + f"){how}:"]
    for h in hits[:MAX_ENTRIES]:
        lines.append(_fmt_entry(h) if isinstance(h, dict) else f"- {h}")
    return "\n".join(lines)


#: Prompt block injected alongside the tools (build_agent), so the agent knows
#: the capability exists and the path convention.
SHAREPOINT_PROMPT = (
    "SHAREPOINT: This deployment is connected to the team SharePoint. You "
    "have sharepoint_list_files / sharepoint_search_files (browse + find), "
    "sharepoint_download_file (fetch a project file into the working folder "
    "to READ -- an input, not a deliverable), and sharepoint_upload_file. "
    "Paths are relative to the app's base SharePoint folder unless given as "
    "'Shared Documents/...', '/sites/...', or a full URL; a SharePoint "
    "address the user pastes can be passed as it is, and when it does not "
    "open directly the download looks for the file by its name and says "
    "which file it used. When the user "
    "references project files 'on SharePoint', use these tools rather than "
    "asking for an upload. For files uploaded RECENTLY (within the last ~15 "
    "minutes), prefer sharepoint_list_files over sharepoint_search_files: "
    "search rides an index that lags new uploads by several minutes, while "
    "listing a folder sees them immediately. A result saying SharePoint is "
    "not answering or refused the request is an access problem, not a "
    "missing file: say so, and use the copies already in the working folder. "
    "Files you produce (reports, "
    "figures, files written to /tmp) are copied into this conversation and "
    "mirrored to its SharePoint folder after every turn, so a deliverable "
    "needs no upload; if the user asks for one anyway, call "
    "sharepoint_upload_file WITHOUT dest_folder and it goes to this "
    "conversation's folder. The 'uploaded references' folder holds the users' "
    "input documents: never upload there.")


def tools_if_configured(thread_id: Optional[str] = None,
                        record_dir: Optional[str] = None) -> tuple:
    """``(tools, prompt)`` when SharePoint is configured, else ``([], "")``.
    With ``thread_id`` the upload tool defaults to that conversation's
    SharePoint folder; with ``record_dir`` (the conversation's record folder)
    the download tool keeps that conversation's downloads ledger."""
    if not sharepoint_store.configured():
        return [], ""
    upload = (make_conversation_upload_tool(thread_id) if thread_id
              else sharepoint_upload_file)
    download = (make_download_tool(record_dir) if record_dir
                else sharepoint_download_file)
    return ([sharepoint_list_files, download, upload,
             sharepoint_search_files], SHAREPOINT_PROMPT)


__all__ = ["tools_if_configured", "SHAREPOINT_PROMPT", "MAX_ENTRIES",
           "sharepoint_list_files", "sharepoint_download_file",
           "sharepoint_upload_file", "sharepoint_search_files",
           "make_download_tool", "browser_url_to_path", "link_target_name",
           "TOKEN_LINK_HINT"]
