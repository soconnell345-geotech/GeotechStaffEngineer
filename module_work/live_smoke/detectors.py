"""Deterministic plumbing detectors, run after every turn.

Each detector takes the turn context (:func:`turn_context` builds it from a
:class:`~live_smoke.session.TurnRecord`) and returns findings::

    {"detector": "a_files_outside", "code": "file_in_temp",
     "severity": "high"|"medium"|"low"|"info", "message": str,
     "evidence": {...}}

(a) files outside the conversation folder, (b) answer links, (c) download
cards, (d) the SharePoint mirror and the agent's SharePoint calls,
(e) health (tool errors, refusals, tracebacks, step caps, empty answers,
"unavailable" answers, success claimed after an error), (f) cost. A model's
answer QUALITY is never judged here -- only whether the app's plumbing did
what it says.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from live_smoke import watch

SEVERITIES = ("high", "medium", "low", "info")

#: Keys in a tool's JSON result that name a file the tool WROTE.
OUTPUT_KEYS = {"output_path", "saved", "saved_path", "saved_to",
               "plotly_json_path", "docx_path", "pdf_path", "html_path",
               "png_path", "out_path", "output_file", "marked_path",
               "written", "outfile", "dxf_path", "diggs_path", "csv_path",
               "path", "file", "output", "filename", "files", "outputs",
               "figures", "figure_paths"}
#: Of those, the ones that only ever mean "written" (a missing file there is
#: a broken promise; ``path``/``file`` are also used for inputs).
STRICT_OUTPUT_KEYS = {"output_path", "saved", "saved_path", "saved_to",
                      "plotly_json_path", "docx_path", "pdf_path",
                      "html_path", "png_path", "out_path", "output_file",
                      "marked_path", "dxf_path", "diggs_path", "csv_path"}

#: Files the app keeps in a conversation's record folder.
RECORD_FILES = {"meta.json", "transcript.jsonl", "messages.json",
                "attachments.json", "activity.jsonl", "trace.jsonl",
                "partial.json", "sp_manifest.json", "feedback.jsonl",
                "FEEDBACK.md", "downloads.json", "coverage.json"}

_WIN_PATH = re.compile(r"(?<![\w/])[A-Za-z]:[\\/][^\s`'\"<>|*?\]\)]+")
_POSIX_PATH = re.compile(
    r"(?<![\w/:.])/(?:tmp|mnt|home|root|Users|var|Workspace|Volumes|dbfs|"
    r"opt|srv|data)/[^\s`'\"<>\])]+")
_URL = re.compile(r"\bhttps?:/{1,2}[^\s<>\"'`\])]+", re.IGNORECASE)
_MD_LINK = re.compile(r"(?<!!)\[([^\]\n]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_IMAGE = re.compile(r"!\[([^\]\n]*)\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_SANDBOX = re.compile(r"sandbox:[^\s)\]>\"'`]+", re.IGNORECASE)
_FILE_NAME = re.compile(
    r"(?<![\w./\\-])([\w][\w\-.()]*\.(?:pdf|docx|xlsx|xls|csv|png|jpe?g|svg|"
    r"dxf|xml|diggs|json|md|txt|html|zip|gef|ags|pptx))\b", re.IGNORECASE)

_UNAVAILABLE = re.compile(
    r"(?:tool|function|capability) (?:is|was|isn'?t|wasn'?t) "
    r"(?:not )?(?:currently )?(?:un)?available"
    r"|(?:don'?t|do not|doesn'?t|does not) have (?:access|a tool|the (?:ability"
    r"|tool|capability))"
    r"|(?:can'?t|cannot|unable to|not able to) (?:directly )?(?:attach|save|"
    r"create|generate|produce|access|open|upload|download|write|export|send)"
    r"|not (?:currently )?(?:available|supported) (?:in|here|to me|in this)"
    r"|no (?:such )?tool (?:for|to|that)"
    r"|isn'?t available|is not available|not available to me",
    re.IGNORECASE)
_SUCCESS_CLAIM = re.compile(
    r"\b(?:successfully|has been (?:saved|created|uploaded|written|generated|"
    r"attached|exported)|i(?:'ve| have) (?:saved|created|uploaded|attached|"
    r"written|generated|exported|marked)|is (?:now )?(?:attached|saved|"
    r"uploaded|ready)|here is (?:the|your) (?:link|file|memo|pdf|report))",
    re.IGNORECASE)
_UNKNOWN_ARG = re.compile(
    r"unexpected keyword argument|validation error|field required|"
    r"input should be|extra (?:inputs|fields) (?:are )?not permitted|"
    r"unknown (?:argument|parameter|param)|got an unexpected|"
    r"missing \d+ required positional argument|"
    r"is not a valid tool|not a valid tool, try one of",
    re.IGNORECASE)
_SP_OK = re.compile(r"^(?:Uploaded |Downloaded |Contents of |Matches for |"
                    r"\(empty folder)", re.MULTILINE)
_SP_HONEST = re.compile(
    r"not found|not answering|REFUSED|failed|error|Local file not found|"
    r"not configured|No files matching|empty or missing folder|throttling|"
    r"sharing link with a token", re.IGNORECASE)


def finding(detector: str, code: str, severity: str, message: str,
            **evidence) -> dict:
    assert severity in SEVERITIES, severity
    return {"detector": detector, "code": code, "severity": severity,
            "message": message, "evidence": evidence}


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _under(path: str, base: str) -> bool:
    if not base:
        return False
    p, b = _norm(path), _norm(base)
    return p == b or p.startswith(b + os.sep)


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return str(x)


# ---------------------------------------------------------------------------
# The turn context
# ---------------------------------------------------------------------------

def load_activity(conv_dir: str) -> List[dict]:
    out = []
    try:
        with open(os.path.join(conv_dir, "activity.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        out.append({"event": "corrupt_line",
                                    "raw": line[:300]})
    except OSError:
        pass
    return out


def turn_context(rec: dict, *, fm=None, remote_folder: Optional[str] = None,
                 meter_calls: Optional[list] = None,
                 fm_calls: Optional[list] = None,
                 outside: Optional[list] = None,
                 input_roots: Iterable[str] = (),
                 prompts: Iterable[str] = ()) -> dict:
    """Everything the detectors read for one turn."""
    conv = rec.get("conv_dir") or ""
    acts = load_activity(conv) if conv else []
    lo = int(rec.get("activity_offset") or 0)
    hi = rec.get("activity_end")
    turn_acts = acts[lo:hi] if hi is not None else acts[lo:]
    return {"rec": rec, "activity": turn_acts, "activity_all": acts,
            "fm": fm, "remote_folder": remote_folder,
            "meter_calls": list(meter_calls or []),
            "fm_calls": list(fm_calls or []),
            "outside": list(outside or []),
            "input_roots": [r for r in input_roots if r],
            "prompts": list(prompts or [])}


def _tool_ends(acts) -> List[dict]:
    return [a for a in acts if a.get("event") == "tool_end"]


def _tool_errors(acts) -> List[dict]:
    return [a for a in acts if a.get("event") == "tool_error"]


def _turn_blob(ctx) -> str:
    """Every word of the turn the user or a tool saw (lower case)."""
    parts = [ctx["rec"].get("final") or ""]
    for a in ctx["activity"]:
        parts.append(_text(a.get("args") or ""))
        parts.append(_text(a.get("result") or ""))
        parts.append(_text(a.get("error") or ""))
    for e in ctx["rec"].get("events") or []:
        parts.append(e.get("text") or "")
    return "\n".join(parts).lower().replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# (a) files outside the conversation folder
# ---------------------------------------------------------------------------

def _paths_in_json(obj, key=None, out=None) -> List[Tuple[str, str]]:
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _paths_in_json(v, str(k).lower(), out)
    elif isinstance(obj, list):
        for v in obj:
            _paths_in_json(v, key, out)
    elif isinstance(obj, str) and key in OUTPUT_KEYS:
        s = obj.strip()
        if s and (os.path.isabs(s) or _WIN_PATH.fullmatch(s)
                  or s.startswith(("/tmp/", "/mnt/", "/home/"))):
            out.append((key, s))
    return out


def tool_result_paths(result: Any) -> List[Tuple[str, str]]:
    """``(key, path)`` for every absolute path a tool result names: JSON
    output keys first, then any absolute path in the text."""
    text = _text(result)
    found: List[Tuple[str, str]] = []
    body = text
    # deepagents/LangChain may prefix a JSON payload with a line of text.
    for cand in (body, body[body.find("{"):] if "{" in body else ""):
        try:
            data = json.loads(cand)
        except (ValueError, TypeError):
            continue
        _paths_in_json(data, None, found)
        break
    seen = {_norm(p) for _k, p in found}
    # JSON text doubles every backslash; read the paths as written.
    plain = text.replace("\\\\", "\\")
    for m in list(_WIN_PATH.findall(plain)) + list(_POSIX_PATH.findall(plain)):
        p = m.rstrip(".,;:'\")")
        if _norm(p) not in seen:
            seen.add(_norm(p))
            found.append(("text", p))
    return found


def detect_files_outside(ctx) -> List[dict]:
    D = "a_files_outside"
    rec = ctx["rec"]
    out: List[dict] = []
    conv = rec.get("conv_dir") or ""
    files_dir = rec.get("files_dir") or ""
    t0 = float(rec.get("t_start") or 0)
    blob = _turn_blob(ctx)
    # 1) new files in the watched places
    for d in ctx["outside"]:
        p, root = d["path"], d["root"]
        name = os.path.basename(p)
        deliverable = p.lower().endswith(watch.DELIVERABLE_EXT)
        named = (name.lower() in blob) or (p.lower() in blob)
        copied = bool(files_dir) and os.path.isfile(os.path.join(files_dir,
                                                                 name))
        ev = {"path": p, "root": root, "bytes": d.get("bytes"),
              "change": d.get("change"), "named_by_turn": named,
              "copied_into_conversation": copied}
        if root == "home_webapp":
            out.append(finding(D, "default_data_root_write", "high",
                               f"{name} written under ~/.geotech_webapp -- a "
                               "code path ignored GEOTECH_WEBAPP_DATA", **ev))
        elif root == "cwd":
            out.append(finding(D, "file_in_process_cwd",
                               "high" if deliverable else "medium",
                               f"{name} written in the app's working "
                               "directory (a relative-path save)", **ev))
        elif named:
            out.append(finding(D, f"file_in_{root}",
                               "medium" if copied else "high",
                               f"{name} written to {root} ({p})"
                               + (" -- the app copied it into the "
                                  "conversation" if copied else
                                  " -- NOT in the conversation folder"),
                               **ev))
        elif deliverable and root != "temp":
            out.append(finding(D, f"unattributed_file_in_{root}", "low",
                               f"{name} appeared in {root} during the turn; "
                               "no tool named it (possibly another process)",
                               **ev))
    # 2) paths the tools reported
    inputs = [r for r in ctx["input_roots"]]
    seen = set()
    for a in _tool_ends(ctx["activity"]):
        for key, p in tool_result_paths(a.get("result")):
            if (key, p) in seen:
                continue
            seen.add((key, p))
            if conv and _under(p, conv):
                continue
            if any(_under(p, r) for r in inputs):
                continue
            ev = {"tool": a.get("name"), "agent": a.get("agent"),
                  "key": key, "path": p}
            exists = os.path.exists(p)
            if not exists:
                if key in STRICT_OUTPUT_KEYS:
                    out.append(finding(D, "reported_output_missing", "high",
                                       f"{a.get('name')} reported writing "
                                       f"{p} ({key}) but no such file exists",
                                       **ev))
                continue
            try:
                fresh = os.path.getmtime(p) >= t0 - 1.0
            except OSError:
                fresh = False
            if os.path.isdir(p):
                continue
            if fresh:
                name = os.path.basename(p)
                copied = bool(files_dir) and os.path.isfile(
                    os.path.join(files_dir, name))
                ev["copied_into_conversation"] = copied
                out.append(finding(D, "tool_wrote_outside_conversation",
                                   "medium" if copied else "high",
                                   f"{a.get('name')} wrote {p} outside the "
                                   "conversation folder"
                                   + (" (the app copied it in)" if copied
                                      else " and it was NOT copied in"),
                                   **ev))
    return out


# ---------------------------------------------------------------------------
# (b) answer links
# ---------------------------------------------------------------------------

def _conv_names(conv_dir: str) -> Dict[str, str]:
    """lower-case basename -> path, every file in the conversation."""
    names: Dict[str, str] = {}
    for root, _dirs, files in os.walk(conv_dir or "."):
        for n in files:
            names.setdefault(n.lower(), os.path.join(root, n))
    return names if conv_dir else {}


def _tool_urls(ctx) -> set:
    urls = set()
    blobs = [_text(a.get("result") or "") for a in ctx["activity_all"]]
    blobs += [_text(a.get("args") or "") for a in ctx["activity_all"]]
    blobs += list(ctx["prompts"])
    side = (ctx["rec"].get("sidebar_after") or {}).get("storage") or {}
    if side.get("web_url"):
        blobs.append(str(side["web_url"]))
    sync = ctx["rec"].get("sp_sync") or {}
    if sync.get("web_url"):
        blobs.append(str(sync["web_url"]))
    for b in blobs:
        for u in _URL.findall(b):
            urls.add(_url_key(u))
    return urls


def _url_key(u: str) -> str:
    u = u.rstrip(".,;:!?'\")]")
    return unquote(u).rstrip("/").lower()


def _is_sharepointish(u: str) -> bool:
    low = u.lower()
    return "sharepoint.com" in low or "/sites/" in low or "/teams/" in low


def url_well_formed(u: str) -> Tuple[bool, str]:
    """``(ok, why)`` for a link a user would click."""
    if not u:
        return False, "empty"
    if re.match(r"^https?:/[^/]", u, re.IGNORECASE):
        return False, "single slash after the scheme (resolves relative)"
    try:
        parts = urlsplit(u)
    except ValueError as exc:
        return False, f"unparseable: {exc}"
    if parts.scheme.lower() not in ("http", "https"):
        return False, f"scheme {parts.scheme!r}"
    if not parts.netloc:
        return False, "no host"
    if any(ch.isspace() for ch in u):
        return False, "unencoded whitespace"
    return True, ""


def detect_answer_links(ctx) -> List[dict]:
    D = "b_answer_links"
    rec = ctx["rec"]
    out: List[dict] = []
    answer = rec.get("final") or ""
    if not answer.strip():
        return out
    cards = list((rec.get("entry") or {}).get("artifacts") or [])
    try:
        from webapp import core
        shown = core.displayable_markdown(answer, cards)
    except Exception:  # noqa: BLE001
        shown = answer
    conv = rec.get("conv_dir") or ""
    names = _conv_names(conv)
    card_names = {os.path.basename(str(c)).lower() for c in cards}
    blob_tools = "\n".join(_text(a.get("result") or "") + _text(
        a.get("args") or "") for a in ctx["activity_all"]).lower()
    blob_prompts = "\n".join(ctx["prompts"]).lower()

    # sandbox: links never open
    for m in _SANDBOX.findall(shown):
        out.append(finding(D, "sandbox_link", "high",
                           f"answer links {m} -- a sandbox: link opens "
                           "nothing", link=m))
    # images the chat cannot show (displayable_markdown replaced them)
    for m in re.findall(r"\*\(([^)]*) — local file `([^`]+)`, not viewable "
                        r"in chat\)\*", shown):
        out.append(finding(D, "image_not_viewable", "medium",
                           f"answer embeds local image {m[1]} that has no "
                           "card -- the reader sees a placeholder",
                           alt=m[0], target=m[1]))
    # markdown links
    for text, target in _MD_LINK.findall(shown):
        t = target.strip()
        if t.lower().startswith(("http://", "https://", "http:/",
                                 "https:/")):
            continue                              # checked with the URLs
        if t.lower().startswith(("sandbox:", "mailto:", "#")):
            continue
        base = os.path.basename(unquote(t).replace("\\", "/")).lower()
        has_card = base in card_names
        exists = base in names
        out.append(finding(
            D, "local_link_in_chat", "medium" if has_card else "high",
            f"answer links [{text}]({t}); a local path in a chat link does "
            "not open in the browser" + (" (a download card for it IS shown)"
                                         if has_card else
                                         " and no download card exists"
                                         + (" (the file is in the folder)"
                                            if exists else "")),
            text=text, target=t, card=has_card, file_exists=exists))
    # absolute local paths shown to the user
    for p in set(_WIN_PATH.findall(shown)) | set(_POSIX_PATH.findall(shown)):
        p = p.rstrip(".,;:'\")`")
        exists = os.path.exists(p)
        inside = bool(conv) and _under(p, conv)
        sev = "low" if (exists and inside) else ("medium" if exists
                                                 else "high")
        out.append(finding(D, "server_path_in_answer", sev,
                           f"answer shows a server path {p}"
                           + ("" if exists else " that does not exist")
                           + ("" if inside else
                              " (outside the conversation folder)"),
                           path=p, exists=exists, inside_conversation=inside))
    # file names that resolve to nothing
    seen = set()
    for name in _FILE_NAME.findall(shown):
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        if low in names or low in card_names:
            continue
        if low in blob_tools or low in blob_prompts:
            continue                       # a file the tools or user named
        out.append(finding(D, "file_name_resolves_to_nothing", "medium",
                           f"answer names {name}, which is not in the "
                           "conversation folder, not a card, and no tool "
                           "or prompt mentioned it", name=name))
    # URLs nobody returned
    known = _tool_urls(ctx)
    for u in set(_URL.findall(shown)):
        u = u.rstrip(".,;:!?'\")]")
        ok, why = url_well_formed(u)
        if not ok:
            out.append(finding(D, "malformed_url", "high",
                               f"answer link {u} is malformed: {why}", url=u))
            continue
        if _url_key(u) in known:
            continue
        if _is_sharepointish(u):
            out.append(finding(D, "sharepoint_url_not_from_a_tool", "high",
                               f"answer gives SharePoint link {u}, which no "
                               "tool returned (invented?)", url=u))
        else:
            out.append(finding(D, "external_url_not_from_a_tool", "low",
                               f"answer gives {u}, which no tool returned",
                               url=u))
    return out


# ---------------------------------------------------------------------------
# (c) download cards
# ---------------------------------------------------------------------------

def detect_download_cards(ctx) -> List[dict]:
    D = "c_download_cards"
    rec = ctx["rec"]
    out: List[dict] = []
    if not rec.get("conv_dir"):
        return out
    try:
        from webapp import core
    except Exception:  # noqa: BLE001
        return out
    entry = rec.get("entry") or {}
    cards = [str(c) for c in entry.get("artifacts") or []]
    files_dir = rec.get("files_dir") or ""
    before = set(rec.get("before") or [])
    after = set(rec.get("after") or [])
    staged = set(rec.get("staged_inputs") or [])
    fetched = {os.path.join(files_dir, n) for n in entry.get("inputs") or []}
    new = sorted(p for p in after - before - staged - fetched
                 if not os.path.basename(p).startswith(".sp_download_"))
    stems = core._plotly_sidecar_stems(cards)
    card_norm = {_norm(c) for c in cards}
    for p in new:
        if _norm(p) in card_norm or core._superseded_by_plotly(p, stems):
            continue
        out.append(finding(D, "file_without_card", "medium",
                           f"{os.path.basename(p)} appeared in the "
                           "conversation folder this turn but got no "
                           "download card", path=p))
    staged_norm = {_norm(p) for p in staged}
    for p in rec.get("files_modified") or []:
        if _norm(p) in card_norm or _norm(p) in staged_norm:
            continue
        if os.path.basename(p).startswith(".sp_download_"):
            continue
        try:
            top = os.path.relpath(p, files_dir).split(os.sep)[0]
        except ValueError:
            top = ""
        if top in getattr(core, "CACHE_DIRS", ()):
            continue
        out.append(finding(D, "updated_file_without_new_card", "low",
                           f"{os.path.basename(p)} was rewritten this turn "
                           "but got no card here; the card from the turn "
                           "that first made it now shows the new content",
                           path=p))
    conv = rec.get("conv_dir")
    for c in cards:
        if not os.path.isfile(c):
            out.append(finding(D, "card_missing_file", "high",
                               f"download card {os.path.basename(c)} points "
                               f"at a missing file ({c})", path=c))
        elif not _under(c, conv):
            out.append(finding(D, "card_outside_conversation", "high",
                               f"download card {os.path.basename(c)} points "
                               f"outside the conversation folder ({c}): not "
                               "durable, not mirrored", path=c))
    for d in (rec.get("sidebar_after") or {}).get("downloads") or []:
        if not d.get("ok"):
            out.append(finding(D, "sidebar_download_unreadable", "medium",
                               f"sidebar download {os.path.basename(d['path'])}"
                               " cannot be read (the app hides it silently)",
                               path=d["path"]))
    if rec.get("synced_in_place") is False and not rec.get("error"):
        out.append(finding(D, "session_resynced_from_disk", "info",
                           "the session's transcript did not match the "
                           "worker's answer and was reloaded from disk"))
    return out


# ---------------------------------------------------------------------------
# (d) the SharePoint mirror and the agent's SharePoint calls
# ---------------------------------------------------------------------------

def detect_sharepoint(ctx) -> List[dict]:
    D = "d_sharepoint"
    rec = ctx["rec"]
    fm = ctx["fm"]
    out: List[dict] = []
    if fm is None or not rec.get("conv_dir"):
        return out
    sync = rec.get("sp_sync")
    if sync is None:
        out.append(finding(D, "mirror_did_not_run", "high",
                           "SharePoint is configured but the turn reported "
                           "no mirror result"))
    else:
        for e in sync.get("errors") or []:
            out.append(finding(D, "mirror_error", "high",
                               f"mirror error: {str(e)[:300]}", error=str(e)))
    remote = ctx["remote_folder"]
    conv = rec["conv_dir"]
    if remote:
        missing, stale = [], []
        try:
            from webapp.core import mirror_skips_dir
        except Exception:  # noqa: BLE001 - an app without the skip list
            mirror_skips_dir = None
        for root, _dirs, files in os.walk(conv):
            rel_dir = os.path.relpath(root, conv).replace(os.sep, "/")
            if mirror_skips_dir is not None and rel_dir != "." and \
                    mirror_skips_dir(rel_dir):
                continue          # the mirror leaves scratch/caches local
            for n in files:
                if n == "sp_manifest.json":
                    continue
                local = os.path.join(root, n)
                rel = os.path.relpath(local, conv).replace(os.sep, "/")
                size = fm.size(f"{remote}/{rel}")
                if size is None:
                    missing.append(rel)
                elif size != os.path.getsize(local):
                    stale.append(rel)
        for rel in missing:
            out.append(finding(D, "file_not_mirrored", "high",
                               f"{rel} is in the conversation folder but not "
                               f"in SharePoint ({remote})", rel=rel,
                               remote=remote))
        for rel in stale:
            out.append(finding(D, "mirror_stale", "high",
                               f"{rel} in SharePoint differs in size from the "
                               "local file", rel=rel, remote=remote))
        if rec.get("multi_user"):
            try:
                from webapp.sharepoint_store import sanitize_folder_name
                # the owner folder is the person's unique key (corp__jdoe)
                who = sanitize_folder_name(str(
                    rec.get("owner_key")
                    or str(rec.get("user") or "").split("\\")[-1]))
            except Exception:  # noqa: BLE001
                who = ""
            segs = remote.split("/")
            if who and who not in segs:
                out.append(finding(D, "mirror_not_per_user", "high",
                                   f"multi-user conversation mirrored to "
                                   f"{remote}, which has no folder for "
                                   f"{who}", remote=remote))
    # the sidebar's folder link
    storage = (rec.get("sidebar_after") or {}).get("storage")
    if storage is not None:
        url = storage.get("web_url")
        if not url:
            out.append(finding(D, "no_folder_link", "medium",
                               "the sidebar shows no 'Open session folder' "
                               "link after the mirror ran"))
        else:
            ok, why = url_well_formed(url)
            if not ok:
                out.append(finding(D, "folder_link_malformed", "high",
                                   f"sidebar folder link {url} is malformed: "
                                   f"{why}", url=url))
            elif remote and fm.resolve_url(url) != fm.norm(remote):
                out.append(finding(D, "folder_link_wrong_folder", "high",
                                   f"sidebar folder link {url} does not point "
                                   f"at the conversation's folder {remote}",
                                   url=url, remote=remote))
    # the agent's own SharePoint calls
    for a in _tool_errors(ctx["activity"]):
        if str(a.get("name", "")).startswith("sharepoint_"):
            out.append(finding(D, "sharepoint_tool_raised", "high",
                               f"{a.get('name')} raised: "
                               f"{str(a.get('error'))[:300]}",
                               tool=a.get("name")))
    for a in _tool_ends(ctx["activity"]):
        name = str(a.get("name", ""))
        if not name.startswith("sharepoint_"):
            continue
        res = _text(a.get("result") or "")
        if _SP_OK.search(res):
            m = re.search(r"Uploaded (.+?) -> (.+?)(?: \(this conversation"
                          r"'s SharePoint folder\))?\.(?: Link: (\S+))?$",
                          res.strip(), re.DOTALL)
            if m:
                remote_path = m.group(2).strip()
                if not fm.exists(remote_path):
                    out.append(finding(D, "upload_not_in_library", "high",
                                       f"{name} said it uploaded to "
                                       f"{remote_path}, which is not there",
                                       remote=remote_path))
                link = (m.group(3) or "").rstrip(".")
                if link:
                    ok, why = url_well_formed(link)
                    target = fm.resolve_url(link)
                    if not ok or not target or not fm.exists(target):
                        out.append(finding(D, "upload_link_dead", "high",
                                           f"upload link {link} does not "
                                           "resolve to the uploaded file"
                                           + (f" ({why})" if why else ""),
                                           link=link))
                else:
                    out.append(finding(D, "upload_without_link", "low",
                                       f"{name} uploaded {remote_path} but "
                                       "returned no link", remote=remote_path))
                if "uploaded references" in remote_path.lower():
                    out.append(finding(D, "upload_into_users_inputs", "medium",
                                       f"deliverable uploaded into the users' "
                                       f"'uploaded references' folder "
                                       f"({remote_path})", remote=remote_path))
                # The upload tool's default folder IS the mirror's folder:
                # an upload of a file the mirror already put there lands as
                # a timestamped second copy, and the link given is to that
                # copy, which later turns' mirror updates never touch.
                parent, base = (remote_path.rsplit("/", 1)
                                if "/" in remote_path else ("", remote_path))
                ts = re.match(r"^(.*)_\d{8}_\d{6}(\.[^.]+)$", base)
                if ts and fm.exists(f"{parent}/{ts.group(1)}{ts.group(2)}"):
                    out.append(finding(
                        D, "upload_duplicates_mirrored_file", "medium",
                        f"{name} put a second copy, {base}, beside "
                        f"{ts.group(1)}{ts.group(2)} that the mirror already "
                        "keeps in the same folder; the link the user gets is "
                        "to the copy, which later mirror syncs never update",
                        remote=remote_path))
            continue
        if _SP_HONEST.search(res):
            out.append(finding(D, "sharepoint_tool_failed_honestly", "info",
                               f"{name}: {res[:200]}", tool=name))
        else:
            out.append(finding(D, "sharepoint_result_unclassified", "medium",
                               f"{name} returned something neither a success "
                               f"nor a stated failure: {res[:200]}",
                               tool=name))
    return out


# ---------------------------------------------------------------------------
# (j) the conversation record (meta)
# ---------------------------------------------------------------------------

#: How the automatic orientation request begins (webapp.profiles).
_ORIENTATION_TITLE = re.compile(r"^I just attached\b", re.IGNORECASE)


def detect_conversation_record(ctx) -> List[dict]:
    D = "j_conversation"
    rec = ctx["rec"]
    out: List[dict] = []
    conv = rec.get("conv_dir")
    if not conv or rec.get("skipped"):
        return out
    try:
        with open(os.path.join(conv, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        out.append(finding(D, "meta_missing", "high",
                           "the conversation has no readable meta.json after "
                           "the turn (it will not be listed or restorable)"))
        return out
    title = str(meta.get("title") or "")
    if _ORIENTATION_TITLE.search(title) and rec.get("turn_index") == 1:
        out.append(finding(D, "title_from_orientation_request", "low",
                           f"the conversation is titled '{title}' -- the "
                           "automatic orientation request, not anything the "
                           "user typed; every upload-started conversation "
                           "looks alike in the sidebar and in the SharePoint "
                           "folder name", title=title))
    if rec.get("multi_user"):
        # meta.owner is the person's unique key (corp__jdoe), meta.owner_name
        # the display name (app.py, 2026-10-09).
        who = str(rec.get("owner_key")
                  or str(rec.get("user") or "").split("\\")[-1])
        if who and str(meta.get("owner") or "") != who:
            out.append(finding(D, "owner_not_recorded", "high",
                               f"multi-user conversation's meta owner is "
                               f"{meta.get('owner')!r}, not {who!r}"))
        name = str(rec.get("owner_name") or "")
        if name and str(meta.get("owner_name") or "") != name:
            out.append(finding(D, "owner_name_not_recorded", "low",
                               f"multi-user conversation's meta owner_name "
                               f"is {meta.get('owner_name')!r}, not "
                               f"{name!r}"))
    if rec.get("page") and rec.get("page") != "geotech" and             meta.get("page") != rec.get("page"):
        out.append(finding(D, "page_not_recorded", "high",
                           f"meta page is {meta.get('page')!r}, not "
                           f"{rec.get('page')!r}"))
    return out


# ---------------------------------------------------------------------------
# (e) health
# ---------------------------------------------------------------------------

def _result_is_error(text: str) -> Tuple[bool, str]:
    try:
        from funhouse_agent.deep.eval_harness import _result_is_error as f
        is_err, note = f(text)
    except Exception:  # noqa: BLE001
        is_err, note = False, ""
    if not is_err and "Traceback (most recent call last)" in (text or ""):
        return True, text[-300:]
    return is_err, note


def _acknowledges_failure(answer: str) -> bool:
    try:
        from funhouse_agent.deep.eval_harness import answer_acknowledges_failure
        return answer_acknowledges_failure(answer)
    except Exception:  # noqa: BLE001
        return bool(re.search(r"error|fail|could not|couldn't|unable",
                              answer or "", re.IGNORECASE))


def detect_health(ctx) -> List[dict]:
    D = "e_health"
    rec = ctx["rec"]
    out: List[dict] = []
    answer = rec.get("final") or ""
    acts = ctx["activity"]
    err = rec.get("error")
    if rec.get("skipped"):
        out.append(finding(D, "turn_not_sent", "high",
                           f"the turn could not be sent: {err}"))
        return out
    if rec.get("timeout"):
        out.append(finding(D, "turn_timeout", "high",
                           "the turn did not finish within the harness's "
                           "time limit"))
    if err:
        low = str(err).lower()
        if "wavecapreached" in low.replace(" ", ""):
            out.append(finding(D, "spend_cap_stop", "info",
                               f"the wave cap stopped this turn: "
                               f"{str(err)[:200]}"))
        elif "recursion" in low and "limit" in low:
            out.append(finding(D, "step_limit", "high",
                               f"the turn hit the step cap: {str(err)[:300]}",
                               error=str(err)[:2000]))
        elif "ratelimit" in low or "overloaded" in low or " 429" in low \
                or " 529" in low:
            out.append(finding(D, "rate_limited", "medium",
                               f"the turn ended on a rate limit after "
                               f"retries: {str(err)[:300]}"))
        else:
            out.append(finding(D, "turn_exception", "high",
                               f"the turn ended with an exception: "
                               f"{str(err)[:300]}", error=str(err)[:2000]))
    if rec.get("save_error"):
        out.append(finding(D, "persistence_failed", "high",
                           f"the conversation could not be saved: "
                           f"{rec['save_error']}"))
    stripped = answer.strip()
    if not stripped or stripped == "(no answer text)":
        if not err:
            out.append(finding(D, "empty_answer", "high",
                               "the turn ended with no answer text"))
    # tool failures
    errored: Dict[str, int] = {}
    order: List[Tuple[str, bool]] = []
    soft: List[dict] = []
    for a in acts:
        ev = a.get("event")
        if ev == "tool_error":
            name = a.get("name") or "?"
            text = str(a.get("error") or "")
            code = ("unknown_argument_refusal" if _UNKNOWN_ARG.search(text)
                    else "tool_raised")
            out.append(finding(D, code, "high",
                               f"{name} raised: {text[:300]}", tool=name,
                               agent=a.get("agent"), error=text[:2000]))
            errored[name] = errored.get(name, 0) + 1
            order.append((name, True))
        elif ev == "tool_end":
            name = a.get("name") or "?"
            text = _text(a.get("result") or "")
            if name.startswith("sharepoint_"):
                order.append((name, False))
                continue                      # judged by detector (d)
            is_err, note = _result_is_error(text)
            if _UNKNOWN_ARG.search(text[:2000]) and (is_err or text.lower()
                                                     .startswith("error")):
                out.append(finding(D, "unknown_argument_refusal", "high",
                                   f"{name} refused its arguments: "
                                   f"{text[:300]}", tool=name,
                                   agent=a.get("agent")))
                errored[name] = errored.get(name, 0) + 1
                order.append((name, True))
            elif "Traceback (most recent call last)" in text:
                out.append(finding(D, "traceback_in_result", "high",
                                   f"{name} returned a traceback", tool=name,
                                   agent=a.get("agent"), tail=text[-600:]))
                errored[name] = errored.get(name, 0) + 1
                order.append((name, True))
            elif is_err:
                soft.append(finding(D, "tool_returned_error", "medium",
                                    f"{name} returned an error: "
                                    f"{str(note)[:300]}", tool=name,
                                    agent=a.get("agent")))
                out.append(soft[-1])
                errored[name] = errored.get(name, 0) + 1
                order.append((name, True))
            else:
                order.append((name, False))
        elif ev == "model_error":
            text = str(a.get("error") or "")
            out.append(finding(D, "model_call_failed",
                               "medium" if re.search(r"429|529|overload|rate",
                                                     text, re.IGNORECASE)
                               else "high",
                               f"a model call failed: {text[:300]}",
                               agent=a.get("agent")))
        elif ev == "corrupt_line":
            out.append(finding(D, "activity_log_corrupt", "medium",
                               "activity.jsonl has an unreadable line",
                               raw=a.get("raw")))
    # tool calls the graph refused before any tool ran (no activity event)
    for e in rec.get("events") or []:
        if e.get("kind") == "tool_result" and re.search(
                r"is not a valid tool", e.get("text") or "", re.IGNORECASE):
            out.append(finding(D, "unknown_tool", "high",
                               f"the model called a tool that does not "
                               f"exist: {(e.get('text') or '')[:200]}"))
    # scratch filesystem writes the user can never download
    for a in acts:
        if a.get("event") == "tool_start" and a.get("name") in ("write_file",
                                                                "edit_file"):
            args = a.get("args") or {}
            path = args.get("file_path") if isinstance(args, dict) else None
            if path and not str(path).startswith(("/large_tool_results",
                                                  "/memories",
                                                  "/conversation_history")):
                out.append(finding(D, "scratch_filesystem_write", "medium",
                                   f"{a.get('name')} wrote {path} to the "
                                   "agent's in-memory scratch space -- not a "
                                   "file the user can download",
                                   path=path, agent=a.get("agent")))
    # what the answer says
    if stripped and _UNAVAILABLE.search(answer):
        m = _UNAVAILABLE.search(answer)
        lo = max(0, m.start() - 80)
        out.append(finding(D, "answer_says_unavailable", "medium",
                           "the answer says a tool, file or capability is "
                           f"unavailable: ...{answer[lo:m.end() + 80]}...",
                           phrase=m.group(0)))
    unrecovered = []
    for name in errored:
        last_err = max(i for i, (n, e) in enumerate(order) if n == name and e)
        if not any(n == name and not e for n, e in order[last_err + 1:]):
            unrecovered.append(name)
    # An error result the model recovered from (a later call of the same
    # tool succeeded) is still worth a look -- usually a parameter the tool
    # description did not make clear -- but it did not reach the user.
    for f in soft:
        if f["evidence"].get("tool") not in unrecovered:
            f["severity"] = "low"
            f["message"] += (" (recovered: a later call of the same tool "
                             "succeeded)")
            f["evidence"]["recovered"] = True
    if unrecovered and len(stripped) >= 40 and _SUCCESS_CLAIM.search(answer) \
            and not _acknowledges_failure(answer):
        out.append(finding(D, "success_claimed_after_error", "high",
                           f"the answer claims success "
                           f"('{_SUCCESS_CLAIM.search(answer).group(0)}') "
                           f"while {', '.join(unrecovered)} failed and was "
                           "never retried successfully, and the answer does "
                           "not mention the failure", tools=unrecovered))
    return out


# ---------------------------------------------------------------------------
# (f) cost
# ---------------------------------------------------------------------------

def turn_metrics(ctx) -> dict:
    rec = ctx["rec"]
    calls = ctx["meter_calls"]
    tot = {"usd": 0.0, "model_calls": 0, "errors": 0, "unmetered": 0,
           "prompt": 0, "uncached": 0, "cache_read": 0, "cache_write_5m": 0,
           "cache_write_1h": 0, "output": 0}
    for c in calls:
        if c.get("error"):
            tot["errors"] += 1
            continue
        tot["model_calls"] += 1
        tot["usd"] += float(c.get("usd") or 0)
        tot["unmetered"] += 1 if c.get("unmetered") else 0
        for k, v in (c.get("tokens") or {}).items():
            if k in tot and not isinstance(v, bool):
                tot[k] += int(v)
    tot["usd"] = round(tot["usd"], 6)
    t0, t1 = rec.get("t_start"), rec.get("t_end")
    tot["seconds"] = round(float(t1) - float(t0), 1) if t0 and t1 else None
    tot["tool_calls"] = sum(1 for a in ctx["activity"]
                            if a.get("event") == "tool_start")
    tot["app_turn_tokens"] = (rec.get("result") or {}).get("turn_tokens")
    return tot


def detect_cost(ctx) -> List[dict]:
    D = "f_cost"
    m = turn_metrics(ctx)
    out = [finding(D, "turn_cost", "info",
                   f"${m['usd']:.4f}, {m['model_calls']} model calls, "
                   f"{m['prompt']:,} in / {m['output']:,} out tokens "
                   f"(cache read {m['cache_read']:,}, write "
                   f"{m['cache_write_5m'] + m['cache_write_1h']:,}), "
                   f"{m['seconds']} s", **m)]
    if m["unmetered"]:
        out.append(finding(D, "unmetered_calls", "medium",
                           f"{m['unmetered']} model call(s) carried no usage "
                           "and were not priced"))
    if m["errors"]:
        out.append(finding(D, "model_call_errors", "low",
                           f"{m['errors']} model call(s) failed (after the "
                           "SDK's retries)"))
    # Every model call the app's activity log saw must have been metered
    # (and the reverse): a gap is a model built somewhere the meter is not
    # attached (unpriced spend), or a call made outside the run's callbacks
    # (missing from the archive the owner reads).
    n_act = sum(1 for a in ctx["activity"]
                if a.get("event") in ("model_end", "model_error"))
    n_meter = len(ctx["meter_calls"])
    if (n_act or n_meter) and n_act != n_meter:
        if ctx["rec"].get("together_with"):
            # Overlapping turns (session.say_together): the meter cannot
            # tell their calls apart, so a gap here is the harness's.
            out.append(finding(D, "meter_overlapping_turns", "info",
                               f"the activity log recorded {n_act} model "
                               f"calls and the spend meter {n_meter}; this "
                               "turn overlapped "
                               f"{', '.join(ctx['rec']['together_with'])}'s",
                               activity=n_act, meter=n_meter))
        else:
            out.append(finding(D, "meter_activity_mismatch", "medium",
                               f"the activity log recorded {n_act} model "
                               f"calls and the spend meter {n_meter}",
                               activity=n_act, meter=n_meter))
    return out


DETECTORS = (detect_files_outside, detect_answer_links, detect_download_cards,
             detect_sharepoint, detect_health, detect_cost,
             detect_conversation_record)


def run_turn_detectors(ctx) -> dict:
    """All detectors on one turn: ``{"findings": [...], "metrics": {...}}``."""
    findings: List[dict] = []
    for det in DETECTORS:
        try:
            findings += det(ctx)
        except Exception as exc:  # noqa: BLE001 - a detector bug is a finding
            import traceback
            findings.append(finding("harness", "detector_crashed", "medium",
                                    f"{det.__name__} crashed: "
                                    f"{type(exc).__name__}: {exc}",
                                    tb=traceback.format_exc()[-1500:]))
    return {"findings": findings, "metrics": turn_metrics(ctx)}


# ---------------------------------------------------------------------------
# Scenario-level checks
# ---------------------------------------------------------------------------

def detect_isolation(sessions) -> List[dict]:
    """Two people on one host: nothing of one may show in the other's
    folders, lists, tool results or SharePoint folders."""
    D = "g_isolation"
    out: List[dict] = []
    try:
        from webapp import core, sharepoint_store
    except Exception:  # noqa: BLE001
        return out
    people = [s for s in sessions if s.turns]
    for a in people:
        for b in people:
            if a is b or a.ident.key == b.ident.key:
                continue
            if _norm(a.root) == _norm(b.root):
                out.append(finding(D, "shared_root", "high",
                                   f"{a.label} and {b.label} share one "
                                   f"conversation root {a.root}"))
                continue
            listed_b = {m.get("thread_id")
                        for m in core.list_conversations(b.root)}
            for tid in a.threads:
                if tid in listed_b:
                    out.append(finding(D, "conversation_listed_for_other",
                                       "high", f"{a.label}'s conversation "
                                       f"{tid[:8]} is listed for {b.label}"))
                if _under(core.conversation_dir(tid), b.root):
                    out.append(finding(D, "conversation_in_other_root",
                                       "high", f"{a.label}'s conversation "
                                       f"{tid[:8]} lives under {b.label}'s "
                                       "root"))
            a_names = set()
            for tid in a.threads:
                fd = os.path.join(core.conversation_dir(tid), "files")
                if os.path.isdir(fd):
                    a_names |= {n.lower() for n in os.listdir(fd)}
            b_names = set()
            for tid in b.threads:
                fd = os.path.join(core.conversation_dir(tid), "files")
                if os.path.isdir(fd):
                    b_names |= {n.lower() for n in os.listdir(fd)}
            own = a_names - b_names
            for tid in b.threads:
                text = "\n".join(_text(r.get("result") or "") + _text(
                    r.get("args") or "") + _text(r.get("context_note") or "")
                    for r in load_activity(core.conversation_dir(tid)))
                low = text.lower()
                for tid_a in a.threads:
                    if _norm(core.conversation_dir(tid_a)).lower() in \
                            os.path.normcase(low):
                        out.append(finding(
                            D, "path_leak", "high",
                            f"{b.label}'s turn saw {a.label}'s conversation "
                            f"folder ({tid_a[:8]})"))
                for n in sorted(own):
                    if len(n) > 6 and n in low:
                        out.append(finding(
                            D, "file_name_leak", "medium",
                            f"{b.label}'s turn mentions {n}, a file only "
                            f"{a.label}'s conversation holds"))
            if a.ident.multi_user and b.ident.multi_user:
                sp = sharepoint_store.get_store()
                if sp.configured:
                    names_b = {r["name"] for r in b.list_remote()}
                    for tid in a.threads:
                        try:
                            fold = sp.folder_name(tid, a.root)
                        except Exception:  # noqa: BLE001
                            continue
                        if fold in names_b and fold not in {
                                sp.folder_name(t, b.root) for t in b.threads}:
                            out.append(finding(
                                D, "remote_listed_for_other", "high",
                                f"{a.label}'s mirrored conversation {fold} is "
                                f"in {b.label}'s restore list"))
    return out


def detect_restore(session, mirrored_before: Optional[List[dict]] = None
                   ) -> List[dict]:
    """A restore from SharePoint brought the conversation back whole."""
    D = "h_restore"
    out: List[dict] = []
    res = session.last_restore
    if res is None:
        return out
    if res.get("status") not in ("restored", "exists"):
        out.append(finding(D, "restore_failed", "high",
                           f"restore failed: {res.get('status')} "
                           f"{(res.get('errors') or [''])[:3]}",
                           listed=res.get("listed")))
        return out
    for e in res.get("errors") or []:
        out.append(finding(D, "restore_file_error", "high",
                           f"restore error: {str(e)[:300]}"))
    from webapp import core
    tid = res.get("thread_id")
    if tid:
        conv = core.conversation_dir(tid, session.root)
        tr = core.load_transcript(tid, session.root)
        for entry in tr:
            for p in entry.get("artifacts") or []:
                if not os.path.isfile(p):
                    out.append(finding(D, "restored_card_missing", "high",
                                       f"restored transcript card "
                                       f"{os.path.basename(p)} has no file",
                                       path=p))
        if not core.load_messages(tid, session.root):
            out.append(finding(D, "restored_without_history", "high",
                               "the restored conversation has no message "
                               "history to replay"))
        if not os.path.isfile(os.path.join(conv, "meta.json")):
            out.append(finding(D, "restored_without_meta", "high",
                               "the restored conversation has no meta.json"))
    return out


def detect_expectations(flow: dict, sessions) -> List[dict]:
    """The flow's soft expectations -- whether the scenario exercised the
    plumbing it was written for (a model choice, so 'info')."""
    D = "i_expectations"
    exp = flow.get("expect") or {}
    out: List[dict] = []
    if not exp:
        return out
    tools, exts = set(), set()
    for s in sessions:
        for t in s.turns:
            for a in load_activity(t.get("conv_dir") or "")[
                    int(t.get("activity_offset") or 0):t.get("activity_end")]:
                if a.get("event") == "tool_start":
                    tools.add(a.get("name"))
                    if a.get("name") == "call_agent":
                        args = a.get("args") or {}
                        if isinstance(args, dict):
                            tools.add(f"{args.get('agent_name')}."
                                      f"{args.get('method')}")
            for c in (t.get("entry") or {}).get("artifacts") or []:
                exts.add(os.path.splitext(str(c))[1].lower())
    want_any = exp.get("tools_any") or []
    if want_any and not any(any(w == t or (t and w in t) for t in tools)
                            for w in want_any):
        out.append(finding(D, "expected_tool_not_called", "info",
                           f"none of {want_any} was called (called: "
                           f"{sorted(t for t in tools if t)[:20]})"))
    for ext in exp.get("artifact_ext") or []:
        if ext.lower() not in exts:
            out.append(finding(D, "expected_artifact_missing", "info",
                               f"no {ext} download card was produced "
                               f"(cards: {sorted(exts)})"))
    return out


def severity_counts(findings: Iterable[dict]) -> Dict[str, int]:
    out = {s: 0 for s in SEVERITIES}
    for f in findings:
        out[f["severity"]] = out.get(f["severity"], 0) + 1
    return out


__all__ = ["finding", "turn_context", "run_turn_detectors", "DETECTORS",
           "detect_files_outside", "detect_answer_links",
           "detect_download_cards", "detect_sharepoint", "detect_health",
           "detect_cost", "detect_isolation", "detect_restore",
           "detect_expectations", "tool_result_paths", "url_well_formed",
           "severity_counts", "load_activity", "turn_metrics"]
