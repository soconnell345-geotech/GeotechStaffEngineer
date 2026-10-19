"""Deterministic plumbing detectors, run after every turn.

Each detector takes the turn context (:func:`turn_context` builds it from a
:class:`~live_smoke.session.TurnRecord`) and returns findings::

    {"detector": "a_files_outside", "code": "file_in_temp",
     "severity": "high"|"medium"|"low"|"info", "message": str,
     "evidence": {...}}

(a) files outside the conversation folder, (b) answer links, (c) download
cards, (d) the SharePoint mirror and the agent's SharePoint calls,
(e) health (tool errors, refusals, tracebacks, step caps, empty answers,
"unavailable" answers, success claimed after an error), (f) cost,
(k) figure cards (a card showing an older draw than a figure written
later, and, where the flow asks, a card no look read), (l) the app's
judgment-call check (an answer sent back for calls with no source, and
calls still without one when it went out). A model's answer
QUALITY is never judged here -- only whether the app's plumbing did what it
says.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

from funhouse_agent import answer_sections as _answer_sections
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
                 prompts: Iterable[str] = (),
                 expect: Optional[dict] = None) -> dict:
    """Everything the detectors read for one turn. ``expect`` is the flow's
    ``expect`` block (detector (k) reads its ``cards``)."""
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
            "prompts": list(prompts or []),
            "expect": dict(expect or {})}


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
    # scratch filesystem writes the user can never download; a write that
    # failed wrote nothing (w12 F55j: an edit_file "not found" counted
    # twice, as its error and as a write; EV-28)
    failed = {a.get("run_id") for a in acts if a.get("run_id") and (
        a.get("event") == "tool_error" or (
            a.get("event") == "tool_end" and _result_is_error(
                _text(a.get("result") or ""))[0]))}
    for a in acts:
        if a.get("run_id") in failed:
            continue
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
    # the answers the judgment-call check read, and those it sent back
    # (GM-174)
    checks = judgment_checks(ctx["activity"])
    tot["judgment_checks"] = len(checks)
    tot["judgment_retries"] = sum(1 for a in checks
                                  if a.get("action") == "fed_back")
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


# ---------------------------------------------------------------------------
# (k) figure cards: stale cards, and cards no look read
# ---------------------------------------------------------------------------

#: A figure card is a picture or the interactive chart (``*.plotly.json``).
FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
CHART_SUFFIX = ".plotly.json"
#: Seconds by which a file must be newer than a card to count as drawn
#: after it. ``plot_data`` writes a figure's picture and chart in one call,
#: well inside this; a redraw comes at least one model call later (wave w9:
#: 49-77 s).
NEWER_S = 2.0
#: Slack between a file's time on disk and the activity log's clock when
#: asking whether a look read a file after it was written (a look in the
#: same step as the draw waits for it and reads ~0.1 s after; w9 F56 [37]).
CLOCK_SLACK_S = 0.25
#: How a later version of a figure is named: the draft's name and one of
#: these (wave w9 F57: ``plasticity_chart`` -> ``plasticity_chart_report``).
_VERSION_SUFFIX = re.compile(
    r"^[ _.\-]?(?:v\d+|r\d+|rev\w*|final\w*|report|revised|updated|new|"
    r"fixed|clean\w*|corrected|redraw\w*|redrawn|draft\d*|\d+)$",
    re.IGNORECASE)
#: Tools that read an image the agent names by ``attachment_key``.
LOOK_TOOLS = ("analyze_image", "render_region")


def is_figure(path: str) -> bool:
    low = str(path).lower()
    return low.endswith(CHART_SUFFIX) or low.endswith(FIGURE_EXTS)


def figure_stem(path: str) -> str:
    """A figure's base name without its extension (or ``.plotly.json``),
    lower case: the picture and the chart of one draw share it."""
    base = os.path.basename(str(path)).lower()
    if base.endswith(CHART_SUFFIX):
        return base[:-len(CHART_SUFFIX)]
    return os.path.splitext(base)[0]


def card_kinds(path: str) -> set:
    """What a card is, for ``expect.cards``: ``figure`` (a picture or a
    chart), ``image`` or ``chart``, and its extension with and without the
    dot (``pdf``, ``.pdf``)."""
    low = os.path.basename(str(path)).lower()
    if low.endswith(CHART_SUFFIX):
        return {"figure", "chart", CHART_SUFFIX, CHART_SUFFIX[1:]}
    ext = os.path.splitext(low)[1]
    kinds = {ext, ext.lstrip(".")}
    if ext in FIGURE_EXTS:
        kinds |= {"figure", "image"}
    return kinds


def card_matches(spec: dict, path: str) -> bool:
    """Whether a card fits an ``expect.cards`` entry (``kind`` and/or a
    ``pattern`` searched in its base name)."""
    kind = spec.get("kind")
    if kind and str(kind).lower() not in card_kinds(path):
        return False
    pat = spec.get("pattern")
    if pat and not _search(str(pat), os.path.basename(str(path))):
        return False
    return True


def _mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _base(name: Any) -> str:
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()


def _json_of(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    text = _text(result)
    for cand in (text, text[text.find("{"):] if "{" in text else ""):
        try:
            data = json.loads(cand)
        except (ValueError, TypeError):
            continue
        return data if isinstance(data, dict) else None
    return None


def _named_outputs(obj, key=None, out=None) -> set:
    """Base names (lower case) of every file a tool result names under an
    output key (``output_path``, ``file``, ...), relative or absolute."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _named_outputs(v, str(k).lower(), out)
    elif isinstance(obj, list):
        for v in obj:
            _named_outputs(v, key, out)
    elif isinstance(obj, str) and key in OUTPUT_KEYS and obj.strip():
        out.add(_base(obj.strip()))
    return out


def figure_looks(acts: List[dict]) -> List[dict]:
    """Every look at an image in ``acts``, oldest first::

        {"target": base name of the figure it read (a look copy is resolved
                   to its figure through plot_code's ``look_file``),
         "key": the name the look was given, "tool", "t": when it read
         (its first vision call, else its end), "bytes": [image sizes]}

    A look that raised or returned an error, and the vision probe's calls,
    read nothing and are left out."""
    copies: Dict[str, str] = {}
    starts: Dict[str, dict] = {}
    vision: Dict[str, List[dict]] = {}
    ends: Dict[str, dict] = {}
    failed = set()
    for a in acts:
        ev = a.get("event")
        rid = a.get("run_id")
        if ev == "tool_start" and rid:
            starts[rid] = a
        elif ev == "tool_end":
            if rid:
                ends[rid] = a
            if a.get("name") == "plot_code":
                for e in (_json_of(a.get("result")) or {}).get("images") or []:
                    if isinstance(e, dict) and e.get("look_file") \
                            and e.get("file"):
                        copies[_base(e["look_file"])] = _base(e["file"])
            res = _json_of(a.get("result"))
            if rid and res is not None and res.get("error"):
                failed.add(rid)
        elif ev == "tool_error" and rid:
            failed.add(rid)
        elif ev == "model_start" and isinstance(a.get("vision"), dict) \
                and a["vision"].get("kind") != "probe" and a.get("tool_run_id"):
            vision.setdefault(a["tool_run_id"], []).append(a)
    out = []
    for rid, s in starts.items():
        if rid in failed:
            continue
        if s.get("name") not in LOOK_TOOLS and rid not in vision:
            continue
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        key = next((args[k] for k in ("attachment_key", "image", "path",
                                      "file") if args.get(k)), None)
        if not key:
            continue
        vs = vision.get(rid) or []
        t = (min(float(v.get("ts") or 0) for v in vs) if vs else
             float((ends.get(rid) or s).get("ts") or 0))
        out.append({"target": copies.get(_base(key), _base(key)),
                    "key": str(key), "tool": s.get("name"), "t": t,
                    "bytes": [v["vision"].get("image_bytes") for v in vs
                              if v["vision"].get("image_bytes")]})
    return sorted(out, key=lambda lk: lk["t"])


def _rewritten_after(acts: List[dict], names: set, t0: float,
                     fallback: Optional[float]) -> Optional[float]:
    """When a file named in ``names`` was first written again after ``t0``:
    the earliest drawing tool's end that names it, else ``fallback`` (the
    file's time on disk, which is its LAST write)."""
    times = [float(a.get("ts") or 0) for a in acts
             if a.get("event") == "tool_end"
             and a.get("name") not in LOOK_TOOLS + ("analyze_pdf_page",)
             and float(a.get("ts") or 0) > t0 + NEWER_S
             and names & _named_outputs(_json_of(a.get("result")) or {})]
    return min(times) if times else fallback


def _chart_writer(acts: List[dict], chart: str, twins: set,
                  t_chart: float) -> Optional[str]:
    """The tool that wrote a chart last: the first drawing tool to end after
    the chart's time that names it, else one that names its picture twin
    within :data:`NEWER_S` of the chart's time."""
    named_twin = None
    for a in acts:
        if a.get("event") != "tool_end" \
                or a.get("name") in LOOK_TOOLS + ("analyze_pdf_page",):
            continue
        ts = float(a.get("ts") or 0)
        outs = _named_outputs(_json_of(a.get("result")) or {})
        if chart in outs and ts >= t_chart - CLOCK_SLACK_S:
            return a.get("name")
        if named_twin is None and outs & twins \
                and abs(ts - t_chart) <= NEWER_S:
            named_twin = a.get("name")
    return named_twin


def card_looks(card: str, looks: List[dict], acts: List[dict]) -> List[dict]:
    """The looks that read what ``card`` shows now.

    A picture card: a look at that file (or its look copy) after it was last
    written, or one that read exactly its bytes. A chart card: a look at its
    picture twin, after the chart was written and before the twin was drawn
    again (wave w9 F56: the twin was redrawn in code and later looks read
    the redraw, not the chart); and only when ``plot_data`` wrote the chart,
    since it draws both from one spec. A chart written by other code (w8
    F54: ``plot_code`` wrote a Plotly chart beside its own matplotlib PNG)
    is a different drawing from its PNG, and no look reads a chart."""
    t_card = _mtime(card)
    if t_card is None:
        return []
    base = os.path.basename(card).lower()
    if base.endswith(CHART_SUFFIX):
        stem = base[:-len(CHART_SUFFIX)]
        names = {stem + ext for ext in (".png", ".jpg", ".jpeg")}
        if _chart_writer(acts, base, names, t_card) != "plot_data":
            return []
        twin = next((os.path.join(os.path.dirname(card), n)
                     for n in os.listdir(os.path.dirname(card) or ".")
                     if n.lower() in names), None)
        t_twin = _mtime(twin) if twin else None
        until = (_rewritten_after(acts, names, t_card, t_twin)
                 if t_twin is not None and t_twin > t_card + NEWER_S
                 else float("inf"))
        return [lk for lk in looks if lk["target"] in names
                and t_card - CLOCK_SLACK_S <= lk["t"] < until]
    try:
        size = os.path.getsize(card)
    except OSError:
        size = None
    return [lk for lk in looks if lk["target"] == base and (
        lk["t"] >= t_card - CLOCK_SLACK_S or (size and size in lk["bytes"]))]


def detect_figure_cards(ctx) -> List[dict]:
    """(k) The turn's figure cards against the figure files and the looks.

    * ``stale_figure_card`` (high): a newer figure under the card's own name
      (its picture or chart) was written later, so the card shows an older
      draw (wave w9 F56: the chat showed the rejected ``plot_data`` draft's
      chart; the PNG redrawn 77 s later had no card).
    * ``superseded_figure_card``: a later version under the card's name plus
      a version word (``_v2``, ``_final``, ``_report`` ...) exists: medium
      when that version is shown too (w9 F57: draft and final side by side),
      high when only the draft is.
    * ``figure_card_not_looked_at`` (medium), only where the flow's
      ``expect.cards`` asks for looks (``"looked_at": true``): no look read
      what the card shows.
    """
    D = "k_figure_cards"
    rec = ctx["rec"]
    cards = [str(c) for c in (rec.get("entry") or {}).get("artifacts") or []]
    figs = [c for c in cards if is_figure(c) and os.path.isfile(c)]
    out: List[dict] = []
    if not figs:
        return out
    shown = {_norm(c) for c in cards}
    for c in figs:
        folder = os.path.dirname(c) or "."
        stem, t_card = figure_stem(c), _mtime(c)
        later_draws: Dict[str, list] = {}
        for name in sorted(os.listdir(folder)):
            p = os.path.join(folder, name)
            if _norm(p) == _norm(c) or not is_figure(p) \
                    or not os.path.isfile(p):
                continue
            later = (_mtime(p) or 0) - (t_card or 0)
            other = figure_stem(p)
            # A numbered sibling is another figure, not a later version
            # (sheet_1 and sheet_10, B1 and B12).
            if later > NEWER_S and (other == stem or (
                    other.startswith(stem)
                    and _VERSION_SUFFIX.match(other[len(stem):])
                    and not (stem[-1:].isdigit()
                             and other[len(stem):][:1].isdigit()))):
                later_draws.setdefault(other, []).append((later, name, p))
        # One finding per later draw: its picture and its chart are one
        # figure (the shown one named if either is shown).
        for other, files in sorted(later_draws.items()):
            later, name, p = max(files, key=lambda f: (_norm(f[2]) in shown,
                                                       f[0]))
            ev = {"card": c, "newer": p, "newer_by_s": round(later, 1),
                  "newer_shown": _norm(p) in shown,
                  "also": [f[1] for f in files if f[2] != p]}
            if other == stem:
                out.append(finding(
                    D, "stale_figure_card", "high",
                    f"the card {os.path.basename(c)} shows an older draw: "
                    f"{name} was written {later:.0f} s after it under the "
                    "same name" + ("" if ev["newer_shown"] else
                                   " and has no card"), **ev))
            else:
                out.append(finding(
                    D, "superseded_figure_card",
                    "medium" if ev["newer_shown"] else "high",
                    f"the card {os.path.basename(c)} is a draft: {name}, "
                    f"written {later:.0f} s later, is a later version of it"
                    + (" and is shown beside it" if ev["newer_shown"] else
                       ", and has no card"), **ev))
    wants = [s for s in (ctx.get("expect") or {}).get("cards") or []
             if isinstance(s, dict) and s.get("looked_at")]
    if wants:
        looks = figure_looks(ctx["activity"])
        for c in figs:
            if not any(card_matches(s, c) for s in wants):
                continue
            if not card_looks(c, looks, ctx["activity"]):
                out.append(finding(
                    D, "figure_card_not_looked_at", "medium",
                    f"no look read the figure the card "
                    f"{os.path.basename(c)} shows (the flow expects every "
                    "figure card checked)", card=c,
                    looks=[{k: lk[k] for k in ("key", "t")}
                           for lk in looks][:20]))
    return out


# ---------------------------------------------------------------------------
# (l) the judgment-call check (GM-174)
# ---------------------------------------------------------------------------

#: The activity-log event of the app's judgment-call check
#: (``funhouse_agent.judgment_check.RECORD_EVENT``).
JUDGMENT_EVENT = "judgment_check"


def judgment_checks(acts: Iterable[dict]) -> List[dict]:
    """The turn's judgment-call check records, in order."""
    return [a for a in acts if a.get("event") == JUDGMENT_EVENT]


def detect_judgment_check(ctx) -> List[dict]:
    """What the app's turn-end judgment-call check did (both 'info': the
    check's work and the model's choices, never a plumbing fault): the
    answer sent back for calls that named no source with no consult, and
    an answer that went out with such calls and no consult."""
    D = "l_judgment_check"
    out: List[dict] = []
    for ev in judgment_checks(ctx["activity"]):
        calls = [str(c) for c in ev.get("unsourced") or []]
        listed = "; ".join(c[:80] for c in calls[:5])
        if ev.get("action") == "fed_back":
            out.append(finding(
                D, "judgment_check_retry", "info",
                f"the answer was sent back once: {len(calls)} of "
                f"{ev.get('n_calls')} judgment calls named no source and no "
                f"references task had run ({listed})",
                n_calls=ev.get("n_calls"), unsourced=calls))
        elif calls and not ev.get("references_ran"):
            out.append(finding(
                D, "judgment_calls_unsourced", "info",
                f"the answer went out with {len(calls)} of "
                f"{ev.get('n_calls')} judgment calls naming no source and no "
                f"references task in the turn (mode {ev.get('mode')}: "
                f"{listed})", n_calls=ev.get("n_calls"), unsourced=calls,
                mode=ev.get("mode")))
    return out


DETECTORS = (detect_files_outside, detect_answer_links, detect_download_cards,
             detect_sharepoint, detect_health, detect_cost,
             detect_conversation_record, detect_figure_cards,
             detect_judgment_check)


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


# ---------------------------------------------------------------------------
# What a deliverable holds as given, and the consult's citations in it
# (expect keys ``given_inputs`` and ``pdf_cites_consult``; GM-155, GM-167,
# GM-181)
# ---------------------------------------------------------------------------

def _args_of(act: dict) -> dict:
    """A tool call's arguments as a dict: the activity log keeps a dict,
    JSON, or (``task``) a Python repr."""
    a = act.get("args")
    if isinstance(a, dict):
        return a
    text = str(a or "")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        try:
            import ast
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            data = None
    return data if isinstance(data, dict) else {}


def _failed_runs(acts) -> set:
    """The run ids of tool calls that raised or returned an error."""
    return {a.get("run_id") for a in acts if a.get("run_id") and (
        a.get("event") == "tool_error" or (
            a.get("event") == "tool_end" and _result_is_error(
                _text(a.get("result") or ""))[0]))}


def package_calls(acts) -> List[Tuple[dict, str, dict]]:
    """``(tool_start, method, parameters)`` of each calc package call
    (``call_agent`` on ``calc_package``, a ``*_package`` method) that did not
    fail."""
    failed = _failed_runs(acts)
    out = []
    for a in acts:
        if a.get("event") != "tool_start" or a.get("name") != "call_agent" \
                or a.get("run_id") in failed:
            continue
        args = _args_of(a)
        method = str(args.get("method") or "")
        if str(args.get("agent_name") or "") == "calc_package" \
                and method.endswith("_package"):
            params = args.get("parameters") or args.get("params") or {}
            out.append((a, method, params if isinstance(params, dict)
                        else {}))
    return out


def _read_text(path: str, limit: int = 4 * 1024 * 1024) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def bespoke_reports(acts, rec: Optional[dict] = None
                    ) -> List[Tuple[dict, str, str]]:
    """``(tool_start, file, html)`` of each report composed with
    ``calc_package`` ``html_to_pdf`` that did not fail: the HTML it was
    given, or the file its ``html_path`` names (read from the turn's
    folder; a file rewritten later reads as its last version)."""
    rec = rec or {}
    failed = _failed_runs(acts)
    out = []
    for a in acts:
        if a.get("event") != "tool_start" or a.get("name") != "call_agent" \
                or a.get("run_id") in failed:
            continue
        args = _args_of(a)
        if str(args.get("agent_name") or "") != "calc_package" \
                or args.get("method") != "html_to_pdf":
            continue
        params = args.get("parameters") or args.get("params") or {}
        params = params if isinstance(params, dict) else {}
        html = str(params.get("html") or "")
        src = str(params.get("html_path") or "")
        if not html and src:
            for cand in (src, os.path.join(rec.get("files_dir") or "",
                                           os.path.basename(src)),
                         os.path.join(rec.get("conv_dir") or "", src)):
                if cand and os.path.isfile(cand):
                    html = _read_text(cand)
                    break
        name = os.path.basename(str(params.get("output_path") or src
                                    or "html_to_pdf"))
        out.append((a, name, html))
    return out


def helper_tasks(acts, helper: str, event: str = "tool_start"
                 ) -> List[Tuple[dict, str]]:
    """``(record, text)`` of each ``task`` to ``helper``: its brief for
    ``tool_start``, its hand-back for ``tool_end``."""
    started = {a.get("run_id"): str(a.get("subagent") or _args_of(a).get(
        "subagent_type") or "") for a in acts
        if a.get("event") == "tool_start" and a.get("name") == "task"}
    out = []
    for a in acts:
        if a.get("event") != event or a.get("name") != "task":
            continue
        who = str(a.get("subagent") or started.get(a.get("run_id")) or "")
        if event == "tool_start":
            args = _args_of(a)
            who = who or str(args.get("subagent_type") or "")
            text = str(args.get("description") or "")
        else:
            text = _text(a.get("result") or "")
        if who == helper:
            out.append((a, text))
    return out


_HEADING_TAG = re.compile(r"(?is)<h([1-6])\b[^>]*>(.*?)</h\1\s*>")
_ROW_TAG = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>")
_CELL_TAG = re.compile(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]\s*>")
_ITEM_TAG = re.compile(r"(?is)<li\b[^>]*>(.*?)</li\s*>")
_GIVEN_WORD = re.compile(r"(?i)\bgiven\b")
_NOT_GIVEN = re.compile(r"(?i)\b(?:not|un)[ -]?given\b|\bassum")


def _plain(html: str) -> str:
    from funhouse_agent.cited_documents import plain_text
    return " ".join(plain_text(html).split())


def given_rows(html: str) -> List[str]:
    """What a composed report prints as given, as plain text: each table
    row, list item or (with neither) the text of a part headed "Given..."
    (not "not given" or "assumed"), to the next heading at its level or
    above; and each table row whose status cell says "Given"."""
    text = str(html or "")
    rows: List[str] = []
    heads = list(_HEADING_TAG.finditer(text))
    for i, h in enumerate(heads):
        title = _plain(h.group(2))
        if not _GIVEN_WORD.search(title) or _NOT_GIVEN.search(title):
            continue
        end = next((n.start() for n in heads[i + 1:]
                    if int(n.group(1)) <= int(h.group(1))), len(text))
        part = text[h.end():end]
        items = ([m.group(1) for m in _ROW_TAG.finditer(part)]
                 + [m.group(1) for m in _ITEM_TAG.finditer(part)]) or [part]
        rows += [" | ".join(_plain(c) for c in _CELL_TAG.findall(x))
                 or _plain(x) for x in items]
    for m in _ROW_TAG.finditer(text):
        cells = [_plain(c) for c in _CELL_TAG.findall(m.group(1))]
        if any(re.match(r"(?i)given\b", c) and not _NOT_GIVEN.search(c)
               for c in cells):
            rows.append(" | ".join(cells))
    out: List[str] = []
    for r in rows:
        if r and r not in out:
            out.append(r)
    return out


#: Where a brief's GIVEN block ends: the next block a brief opens with a
#: capitalised word (ASSUMPTIONS, METHOD, ...), or a line opening with
#: "Assum...".
_BRIEF_BLOCK_END = re.compile(
    r"(?<![A-Za-z])(?:ASSUM[A-Z]*|METHODS?|TASKS?|DELIVERABLES?|OUTPUTS?|"
    r"RETURN|REPORT|FIGURES?|CITATIONS?|REFERENCES?|CHECKS?|JUDGE?MENT)"
    r"(?![a-z])|(?m:^[ \t>*#_-]*Assum)")


def brief_given_block(text: str) -> Optional[str]:
    """A brief's GIVEN block: from "GIVEN" (or a line opening with
    "Given"), not "not given", to the next block; ``None`` without one."""
    t = str(text or "")
    for m in re.finditer(r"(?<![A-Za-z])(GIVEN|Given)(?![A-Za-z])", t):
        if re.search(r"(?i)(?:not|un)[\s-]*\W?$", t[max(0, m.start() - 6):
                                                     m.start()]):
            continue
        if m.group(1) == "Given":
            line = t[t.rfind("\n", 0, m.start()) + 1:m.start()]
            if line.strip(" \t>*#_-"):
                continue            # "Given" in running text is no block
        rest = t[m.end():]
        end = _BRIEF_BLOCK_END.search(rest)
        return rest[:end.start()] if end else rest
    return None


def _given_names(given: Any) -> List[str]:
    if isinstance(given, dict):
        return [str(k) for k in given]
    if isinstance(given, (list, tuple)):
        return [str(x) for x in given]
    return [str(given)] if isinstance(given, str) and given else []


def _flow_turns(sessions) -> List[Tuple[dict, List[dict]]]:
    """Every turn of a flow with its activity records."""
    out = []
    for s in sessions:
        for t in s.turns:
            acts = load_activity(t.get("conv_dir") or "")[
                int(t.get("activity_offset") or 0):t.get("activity_end")]
            out.append((t, acts))
    return out


def _expect_given_inputs(spec: dict, turns) -> List[dict]:
    """``expect.given_inputs`` (GM-155, GM-167, GM-181): ``package_given``,
    every calc package call names the user's inputs in ``given``;
    ``brief_given``, every brief to ``calc`` has a GIVEN block;
    ``not_given``, entries ``{"name", "inputs": [package input names],
    "any": [regexes]}`` the user did not state, found in a package call's
    ``given`` (by input name), in what a composed report prints as given
    or in a brief's GIVEN block (by regex)."""
    D = "i_expectations"
    out: List[dict] = []
    calls = [(c, t) for t, acts in turns for c in package_calls(acts)]
    reports = [(r, t) for t, acts in turns
               for r in bespoke_reports(acts, t)]
    briefs = [b for _t, acts in turns for b in helper_tasks(acts, "calc")]
    if spec.get("package_given"):
        if not calls:
            out.append(finding(D, "expected_package_not_called", "info",
                               "no calc package (*_package) was built, so "
                               "`given` was not exercised"))
        for (_a, method, params), _t in calls:
            if not _given_names(params.get("given")):
                out.append(finding(
                    D, "expected_package_given_missing", "info",
                    f"the {method} call passed no `given`, so its Given "
                    "Inputs name no input as the user's",
                    package=method, passed=sorted(params)))
    if spec.get("brief_given"):
        if not briefs:
            out.append(finding(D, "expected_calc_brief_missing", "info",
                               "no brief went to `calc`, so a GIVEN block "
                               "was not exercised"))
        missing = [text for _a, text in briefs
                   if brief_given_block(text) is None]
        if missing:
            out.append(finding(
                D, "expected_brief_given_missing", "info",
                f"{len(missing)} of {len(briefs)} brief(s) to `calc` had no "
                f"GIVEN block (first: {missing[0][:160]!r})",
                n_briefs=len(briefs), n_missing=len(missing)))
    items = spec.get("not_given") or []
    given_parts = [(name, given_rows(html)) for (_a, name, html), _t
                   in reports]
    for name, rows in given_parts if items else ():
        if not rows:
            out.append(finding(
                D, "report_given_part_missing", "info",
                f"{name} prints no Given part (no heading or status cell "
                "saying Given), so not_given was not checked in it",
                where=name))
    for item in items:
        label, patterns = _answer_name(item)
        inputs = {str(i).lower() for i in (item.get("inputs") or [])}
        for (_a, method, params), _t in calls:
            hit = [n for n in _given_names(params.get("given"))
                   if n.lower() in inputs]
            if hit:
                out.append(finding(
                    D, "assumed_input_given", "info",
                    f"{label} is under Given: the {method} call's `given` "
                    f"names {', '.join(hit)}", item=label, where=method,
                    given=params.get("given")))
        for name, rows in given_parts:
            hit = [r for r in rows if any(_search(p, r) for p in patterns)]
            if hit:
                out.append(finding(
                    D, "assumed_input_given", "info",
                    f"{label} is under Given in {name}: {hit[0][:160]!r}",
                    item=label, where=name, rows=hit[:5]))
        for _a, text in briefs:
            block = brief_given_block(text)
            if block and any(_search(p, block) for p in patterns):
                out.append(finding(
                    D, "assumed_input_given", "info",
                    f"{label} is in the GIVEN block of a brief to `calc`: "
                    f"{' '.join(block.split())[:160]!r}", item=label,
                    where="brief"))
    return _dedupe(out)


def _tool_named_documents(turns) -> set:
    """The documents and works the turns' tool results name, outside the
    references helper (its tools and its hand-back), the helpers'
    hand-backs and the calc package writers (which echo what they were
    given): a tool's notes or `choices` (w13b F55j: the PDF cited the wall
    tool's FHWA-NHI-06-089 and GEC-11, which the consult had named only to
    say it found nothing in one and the other did not apply)."""
    from funhouse_agent.cited_documents import find_citations
    named: set = set()
    for _t, acts in turns:
        writer = {a.get("run_id") for a in acts
                  if a.get("event") == "tool_start"
                  and str(_args_of(a).get("agent_name") or "")
                  == "calc_package"}
        for a in acts:
            if a.get("event") != "tool_end" or a.get("name") == "task" \
                    or a.get("agent") == "references" \
                    or a.get("run_id") in writer:
                continue
            named |= {c.key for c in find_citations(
                _text(a.get("result") or ""), internal=True)}
    return named


_SECTION_NO = re.compile(r"\d+(?:[.\-]\d+)+")


def _dedupe(found: List[dict]) -> List[dict]:
    seen, out = set(), []
    for f in found:
        key = (f["code"], f["message"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _expect_pdf_cites_consult(turns) -> List[dict]:
    """``expect.pdf_cites_consult`` (GM-155 w13b fix 10): each report
    composed with ``html_to_pdf``, and each calc package's
    ``assumptions``, cites a document or work the ``references`` helper's
    hand-back named, the consult having finished before the deliverable
    was written. A document a tool's result also named counts only when
    the deliverable cites it with a section number the consult's hand-back
    gives (else it may be the tool's citation copied)."""
    from funhouse_agent.cited_documents import (covered, find_citations,
                                                plain_text)
    D = "i_expectations"
    out: List[dict] = []
    consults = []
    for _t, acts in turns:
        for a, text in helper_tasks(acts, "references", "tool_end"):
            consults.append((float(a.get("ts") or 0),
                             find_citations(text, internal=True),
                             plain_text(text)))
    tools_named = _tool_named_documents(turns) if consults else set()
    deliverables = []
    for t, acts in turns:
        for a, name, html in bespoke_reports(acts, t):
            deliverables.append((a, f"the composed report {name}", html))
        for a, method, params in package_calls(acts):
            deliverables.append((a, f"the {method} call's assumptions",
                                 json.dumps(params.get("assumptions") or {})))
    if not deliverables:
        out.append(finding(D, "expected_deliverable_missing", "info",
                           "no report was composed with html_to_pdf and no "
                           "calc package built, so pdf_cites_consult was "
                           "not exercised"))
    for a, what, text in deliverables:
        t0 = float(a.get("ts") or 0)
        done = [(cs, words) for ts, cs, words in consults if ts <= t0]
        if not done:
            later = sum(1 for ts, _cs, _w in consults if ts > t0)
            out.append(finding(
                D, "expected_consult_before_deliverable", "info",
                f"{what} was written with no references consult finished "
                "before it" + (f" ({later} finished after it)" if later
                               else " (none ran)"), where=what))
            continue
        before = [c for cs, _w in done for c in cs]
        words = "\n".join(w for _cs, w in done)
        docs = {c.key for c in before if c.kind == "document"}
        works = {c.key for c in before if c.kind == "work"}
        cited = find_citations(text)
        shared, tools_too = [], []
        for c in cited:
            if not ((c.kind == "document" and covered(c.key, docs))
                    or (c.kind == "work" and c.key in works)):
                continue
            if not (c.key in tools_named or covered(c.key, tools_named)) \
                    or any(n in words
                           for n in _SECTION_NO.findall(c.section or "")):
                shared.append(c.shown)
            else:
                tools_too.append(c.shown)
        if not shared:
            out.append(finding(
                D, "expected_consult_citation_missing", "info",
                f"{what} cites none of the {len(docs) + len(works)} "
                "documents and works the consult named (consult: "
                f"{', '.join(c.shown for c in before[:8]) or 'none'}; "
                f"cited: {', '.join(c.shown for c in cited[:8]) or 'none'}"
                + (f"; {', '.join(tools_too)} also named by a tool, cited "
                   "with no section the consult gave" if tools_too else "")
                + ")", where=what, consult=[c.shown for c in before][:20],
                cited=[c.shown for c in cited][:20], tools_too=tools_too))
    return _dedupe(out)


def detect_expectations(flow: dict, sessions) -> List[dict]:
    """The flow's soft expectations -- whether the scenario exercised the
    plumbing it was written for (a model choice, so 'info'): ``tools_any``,
    ``artifact_ext``, ``answer_names``, the phrases or regexes the
    flow's final answer must contain (BACKLOG EV-26: does the answer name
    its judgment calls?; an entry with ``"within": "judgment_calls"`` is
    looked for in the answer's "Judgment calls" list only, or, where the
    answer has none, in the whole answer, which the finding says), and
    ``cards``, the download cards the flow should
    end with (BACKLOG EV-24): entries ``{"kind": "figure" | "image" |
    "chart" | an extension, "pattern": a regex on the base name, "count": n
    | "min": n | "max": n, "looked_at": true}``, counted over the distinct
    cards of every turn (``min`` 1 when no number is given). ``looked_at``
    is checked per turn by :func:`detect_figure_cards`. ``given_inputs``
    (:func:`_expect_given_inputs`) and ``pdf_cites_consult``
    (:func:`_expect_pdf_cites_consult`) look at what the flow's calc
    packages, composed reports and briefs to ``calc`` hold as given and
    cite (GM-155, GM-167, GM-181)."""
    D = "i_expectations"
    exp = flow.get("expect") or {}
    out: List[dict] = []
    if not exp:
        return out
    tools, exts = set(), set()
    cards: Dict[str, str] = {}
    turns = _flow_turns(sessions)
    for t, acts in turns:
        for a in acts:
            if a.get("event") == "tool_start":
                tools.add(a.get("name"))
                if a.get("name") == "call_agent":
                    args = a.get("args") or {}
                    if isinstance(args, dict):
                        tools.add(f"{args.get('agent_name')}."
                                  f"{args.get('method')}")
        for c in (t.get("entry") or {}).get("artifacts") or []:
            exts.add(os.path.splitext(str(c))[1].lower())
            cards.setdefault(_norm(c), str(c))
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
    names_shown = sorted(os.path.basename(c) for c in cards.values())
    for spec in exp.get("cards") or []:
        spec = spec if isinstance(spec, dict) else {"kind": str(spec)}
        label, lo, hi = card_spec_bounds(spec)
        hits = sorted(os.path.basename(c) for c in cards.values()
                      if card_matches(spec, c))
        want = (f"{lo}" if lo == hi else f"at least {lo}" if hi is None
                else f"{lo} to {hi}")
        if len(hits) < lo:
            out.append(finding(D, "expected_cards_missing", "info",
                               f"{len(hits)} {label} card(s); the flow "
                               f"expects {want} (cards: {names_shown})",
                               spec=spec, matched=hits))
        elif hi is not None and len(hits) > hi:
            out.append(finding(D, "unexpected_card_count", "info",
                               f"{len(hits)} {label} cards ({', '.join(hits)}"
                               f"); the flow expects {want}",
                               spec=spec, matched=hits))
    names = exp.get("answer_names") or []
    if names:
        final = _final_answer(sessions)
        for item in names:
            label, patterns = _answer_name(item)
            scope = item.get("within") if isinstance(item, dict) else None
            text, where, whole = final, "", False
            if scope:
                part = answer_section(final, scope)
                heading = _SCOPES[scope][0]
                if part is None:
                    whole = True
                    where = (f" (it has no {heading} list, so the whole "
                             "answer was searched)")
                else:
                    text, where = part, f" in its {heading} list"
            if not any(_search(p, text) for p in patterns):
                out.append(finding(
                    D, "expected_answer_name_missing", "info",
                    f"the final answer does not name {label}{where}"
                    + ("" if final.strip() else " (the answer is empty)"),
                    looked_for=patterns, answer_chars=len(final),
                    within=scope, searched_chars=len(text)))
            elif whole and final.strip():
                # found, but only in an answer with no list to hold the
                # key to (w13 F18j: a citation outside the calls)
                out.append(finding(
                    D, "expected_answer_name_unscoped", "info",
                    f"the final answer names {label}, but it has no "
                    f"{_SCOPES[scope][0]} list, so the whole answer was "
                    "searched", looked_for=patterns, within=scope))
    if exp.get("given_inputs"):
        out += _expect_given_inputs(exp["given_inputs"], turns)
    if exp.get("pdf_cites_consult"):
        out += _expect_pdf_cites_consult(turns)
    return out


#: The parts of an answer an ``answer_names`` entry can be held to
#: (``"within"``): name -> (the heading as written, its pattern). w13 F18j:
#: "Vesic (1973)" in a results heading hit the "a reference cited for a
#: judgment call" key though every call said "no reference basis". The app's
#: own module, so the harness reads the list the app's judgment-call check
#: (GM-174) acted on.
_SCOPES = _answer_sections.SCOPES


def answer_section(text: str, scope: str) -> Optional[str]:
    """The ``scope`` part of ``text`` (see ``_SCOPES``): from the line that
    heads it -- a Markdown heading, a bold line or a line that opens with
    its name -- to the next Markdown heading at its level or above, the
    next line that is only bold text, or the end. ``None`` when no line
    heads it (:func:`funhouse_agent.answer_sections.section`)."""
    return _answer_sections.section(text, scope)


def card_spec_bounds(spec: dict) -> Tuple[str, int, Optional[int]]:
    """``(label, min, max)`` for an ``expect.cards`` entry: ``count`` is
    exact, ``min``/``max`` bound it, and with no number at least one."""
    label = " ".join(x for x in (
        str(spec.get("kind") or ""),
        f"/{spec['pattern']}/" if spec.get("pattern") else "") if x) or "any"
    if spec.get("count") is not None:
        n = int(spec["count"])
        return label, n, n
    lo = int(spec.get("min", 1 if spec.get("max") is None else 0))
    hi = spec.get("max")
    return label, lo, (int(hi) if hi is not None else None)


def _final_answer(sessions) -> str:
    """The flow's final answer: the text of the last turn to finish, with
    typographic dashes, primes and spaces made plain."""
    turns = [t for s in sessions for t in s.turns]
    if not turns:
        return ""
    last = max(turns, key=lambda t: t.get("t_end") or t.get("t_start") or 0)
    text = str(last.get("final") or "")
    for a, b in (("‐", "-"), ("‑", "-"), ("‒", "-"),
                 ("–", "-"), ("—", "-"), ("′", "'"),
                 ("’", "'"), (" ", " "), (" ", " ")):
        text = text.replace(a, b)
    return text


def _answer_name(item) -> Tuple[str, List[str]]:
    """An ``expect.answer_names`` entry: a phrase or regex, or
    ``{"name": what it is, "any": [phrases or regexes]}``."""
    if isinstance(item, dict):
        pats = [str(p) for p in item.get("any") or [] if str(p)]
        return str(item.get("name") or " or ".join(pats)), pats
    return repr(str(item)), [str(item)]


def _search(pattern: str, text: str) -> bool:
    """Case-insensitive regex search; a pattern that is not a valid regex
    is matched as plain text."""
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


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
           "severity_counts", "load_activity", "turn_metrics",
           "detect_figure_cards", "figure_looks", "card_looks",
           "card_matches", "card_kinds", "card_spec_bounds", "is_figure",
           "detect_judgment_check", "judgment_checks", "answer_section",
           "package_calls", "bespoke_reports", "helper_tasks", "given_rows",
           "brief_given_block"]
