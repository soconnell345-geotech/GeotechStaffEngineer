"""A figure is one thing across a turn, whoever draws it (BACKLOG GM-151).

Live wave w12 (``module_work/live_smoke/runs/w12-figures-judgment/
TRACE_REVIEW.md`` §1.1, §2.7, §3.3): GM-107's one-card rule held inside one
helper run, but twice the main agent sent a SECOND helper to fix one flaw in
a figure and save it as ``<name>_v2``. The first drawing kept its card, the
second helper's draw cap started again at three, and with no plotting code
to edit it rebuilt the figure from a look at the old PNG (F55j's v2 is worse
than v1).

So every draw by ``plot_code`` or ``plot_data`` (and, on the calc helper, a
figure a module method drew) is booked here:

- **The turn's book of figures** (:func:`record_draw`): each figure's draws
  this turn, by any helper or tool, under its own name, a version of it
  (``plan_v2``) or a name whose draw ``replaces`` it. The draw cap counts
  there. The book lives on the turn
  (:class:`funhouse_agent.deep.keep_work.TurnScope`), else on the running
  helper (a host that opens no turn), else nowhere (a library call: nothing
  is counted).
- **The code beside each figure** (:func:`write_code`), in the working
  folder's hidden ``.figure_code/``: ``<name>.plot.py`` and
  ``<name>.data.json`` for plot_code, ``<name>.plot_data.json`` for
  plot_data's arguments. A dot-folder gets no card, is not among the files
  the model is told about and is not mirrored (``webapp.core``), like
  ``.scratch``.
- **Redraws** (:func:`redraw_note`): a helper whose brief names a figure
  with kept code is handed that code (``deep.figure_helper.FigureTasks``)
  and asked to draw under the figure's own name. A file is removed only by
  a draw under its own name or one that names it in ``replaces``: a name
  that reads as a version (``plan_v2``) only counts on the draws and says
  the earlier file still shows, since "keep both" is a brief too.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

#: The hidden folder, inside the working folder, that keeps each figure's
#: code.
CODE_DIR = ".figure_code"
CODE_SUFFIX = ".plot.py"
DATA_SUFFIX = ".data.json"
ARGS_SUFFIX = ".plot_data.json"
CHART_SUFFIX = ".plotly.json"
PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")
#: In a kept ``.plot.py``: the line between the header and the code.
CODE_MARK = "# --- the code plot_code ran ---"
#: How a redraw block opens; also how a brief that has one is known.
REDRAW_HEADING = "FIGURE TO REDRAW"
#: The longest code, data and plot_data arguments a redraw brief carries
#: inline; longer ones are named by their kept file.
MAX_BRIEF_CODE = 20_000
MAX_BRIEF_DATA = 8_000
MAX_BRIEF_ARGS = 12_000

#: A name that marks a version of another: ``plan_v2``, ``plan-rev3``,
#: ``plan version 2`` (not ``boring_V-1``, a boring's own name).
_VERSION_TAIL = r"[ _\-]+(?:v|rev|version[ _\-]?)\d+"
_VERSION = re.compile(r"^(?P<base>.+?)" + _VERSION_TAIL + "$", re.IGNORECASE)

_LOCK = threading.Lock()
_ORDER = itertools.count(1)

#: The names a renamed redraw replaces, while the tool that takes them runs
#: (plot_data's ``replaces`` is handled by a wrapper around the draw).
_REPLACES: "contextvars.ContextVar[tuple]" = contextvars.ContextVar(
    "geotech_figure_replaces", default=())


@contextmanager
def replacing(names):
    """``with replacing(names):`` -- the draw inside replaces ``names``."""
    if isinstance(names, str):
        names = [names]
    token = _REPLACES.set(tuple(n for n in (names or ()) if n))
    try:
        yield
    finally:
        _REPLACES.reset(token)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

def _relative(path: str, root: Optional[str]) -> str:
    """``path`` relative to ``root`` with '/' separators; its base name when
    it lies outside ``root`` (or there is none)."""
    p = str(path or "").strip()
    if os.path.isabs(p):
        rel = None
        for a, b in ((p, root), (os.path.realpath(p),
                                 os.path.realpath(root) if root else None)):
            if not b:
                continue
            try:
                rel = os.path.relpath(a, b)
            except ValueError:                     # another drive (Windows)
                rel = None
            if rel and not rel.startswith(".."):
                break
            rel = None
        p = rel if rel else os.path.basename(p)
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def is_figure_file(name) -> bool:
    low = str(name or "").lower()
    return low.endswith(CHART_SUFFIX) or os.path.splitext(low)[1] in \
        PICTURE_EXTS


def figure_stem(name, root: Optional[str] = None) -> str:
    """The figure a picture or chart file is, by its name without the
    extension, relative to the working folder ``root``; ``''`` for a file
    that is no figure."""
    if not is_figure_file(name):
        return ""
    p = str(name).strip()
    p = p[:-len(CHART_SUFFIX)] if p.lower().endswith(CHART_SUFFIX) \
        else os.path.splitext(p)[0]
    return _relative(p, root)


def _named_key(name, root: Optional[str]) -> str:
    """The book's key for a name a call gave: a figure file, or a figure's
    name without its extension."""
    stem = figure_stem(name, root) or _relative(name, root)
    return stem.lower()


def version_base(stem: str) -> Optional[str]:
    """The name ``stem`` is a version of (``plan`` for ``plan_v2``), else
    ``None``."""
    m = _VERSION.match(str(stem or ""))
    return m.group("base") if m else None


# ---------------------------------------------------------------------------
# the book
# ---------------------------------------------------------------------------

def _book() -> Optional[dict]:
    """The figures booked this turn (else in this helper run), or ``None``
    outside both."""
    try:
        from funhouse_agent.deep.keep_work import current_helper, current_turn
    except Exception:                                      # noqa: BLE001
        return None
    scope = current_turn()
    if scope is not None:
        return getattr(scope, "figures", None)
    rec = current_helper()
    return getattr(rec, "figures", None) if rec is not None else None


def draws_of(name, root: Optional[str] = None) -> int:
    """How many times the figure ``name`` was drawn this turn (0 when it was
    not, or nothing is booked)."""
    book = _book()
    if not book:
        return 0
    with _LOCK:
        entry = book.get(_named_key(name, root))
        return int(entry["draws"]) if entry else 0


def record_draw(files: Iterable, tool: str, root: Optional[str],
                replaces=None, code: Optional[str] = None,
                data_json: Optional[str] = None,
                args: Optional[dict] = None) -> Dict[str, Any]:
    """Book one successful draw of the figures ``files`` (pictures and
    charts; a figure's picture and chart are one figure) and keep the code
    that drew them (:func:`write_code`).

    A figure continues another booked this turn when this draw ``replaces``
    it (the argument, else :func:`replacing`) or when its name is a version
    of the other's (``plan_v2`` of ``plan``): its draws count on from the
    other's. Nothing is removed here: the earlier file of a version that is
    still there (and not in ``replaces``) is named in ``standing``.

    Returns ``{"figures": [{"file", "draw", "code_file", "continues"}],
    "draw": the most draws of any of them (or None when nothing is booked),
    "standing": [...]}``, a key of a figure left out when it has no value.
    """
    names: Dict[str, str] = {}
    stems: Dict[str, str] = {}
    for f in files or ():
        stem = figure_stem(f, root)
        if not stem:
            continue
        k = stem.lower()
        shown = _relative(f, root)
        if k not in names or (names[k].lower().endswith(CHART_SUFFIX)
                              and not shown.lower().endswith(CHART_SUFFIX)):
            names[k], stems[k] = shown, stem
    out: Dict[str, Any] = {"figures": [], "draw": None, "standing": []}
    if not names:
        return out
    kept = {k: write_code(root, stems[k], names[k], tool, code=code,
                          data_json=data_json, args=args)
            for k in names}
    book = _book()
    if replaces is None:
        replaces = _REPLACES.get()
    if isinstance(replaces, str):
        replaces = [replaces]
    replaced = {_named_key(r, root) for r in (replaces or ()) if r}
    with _LOCK:
        for k in names:
            info: Dict[str, Any] = {"file": names[k]}
            if kept.get(k):
                info["code_file"] = kept[k]
            if book is None:
                out["figures"].append(info)
                continue
            entry = book.get(k)
            prior = int(entry["draws"]) if entry else 0
            base = version_base(stems[k])
            base_k = base.lower() if base else None
            linked = []
            for other, e in list(book.items()):
                if other == k:
                    continue
                if other in replaced:
                    linked.append((other, "replaces"))
                elif base_k and (other == base_k or (
                        version_base(e["stem"]) or "").lower() == base_k):
                    linked.append((other, "version"))
            for other, how in linked:
                prior = max(prior, int(book[other]["draws"]))
                if how != "version":
                    continue
                left = book[other]["file"]
                if root and os.path.exists(os.path.join(root, left)) \
                        and left not in out["standing"]:
                    out["standing"].append(left)
            if linked:
                last = max(linked, key=lambda o: book[o[0]]["order"])[0]
                info["continues"] = book[last]["file"]
            book[k] = {"stem": stems[k], "file": names[k],
                       "draws": prior + 1, "order": next(_ORDER),
                       "tool": tool,
                       "code_file": kept.get(k) or (entry or {}).get(
                           "code_file")}
            info["draw"] = prior + 1
            out["figures"].append(info)
    draws = [i["draw"] for i in out["figures"] if i.get("draw")]
    out["draw"] = max(draws) if draws else None
    return out


# ---------------------------------------------------------------------------
# the code beside each figure
# ---------------------------------------------------------------------------

def _code_base(root: str, stem: str) -> Optional[str]:
    """``<root>/.figure_code/<stem>``, or ``None`` when it would leave the
    folder."""
    folder = os.path.realpath(os.path.join(root, CODE_DIR))
    base = os.path.realpath(os.path.join(folder, *stem.split("/")))
    if not os.path.normcase(base).startswith(os.path.normcase(folder)
                                             + os.sep):
        return None
    return base


def _write(path: str, text: str) -> None:
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def write_code(root: Optional[str], stem: str, figure: str, tool: str,
               code: Optional[str] = None, data_json: Optional[str] = None,
               args: Optional[dict] = None) -> Optional[str]:
    """Keep the code that drew ``figure`` beside it, in ``.figure_code/``:
    plot_code's ``code`` (``<stem>.plot.py``) and the data it was given
    (``<stem>.data.json``), or plot_data's arguments ``args``
    (``<stem>.plot_data.json``); the other tool's record of the same figure
    is removed. The kept file's name relative to ``root``, or ``None`` when
    nothing was kept: a figure drawn by other means (a module method) has
    no code to keep, and code kept from an earlier draw of it is removed,
    since it no longer drew what the file shows."""
    if not root or not stem:
        return None
    if code is None and args is None:
        forget_code(root, stem)
        return None
    base = _code_base(root, stem)
    if base is None:
        return None
    when = time.strftime("%Y-%m-%d %H:%M")
    name = stem.rsplit("/", 1)[-1]
    try:
        os.makedirs(os.path.dirname(base), exist_ok=True)
        if code is not None:
            has_data = str(data_json or "").strip() not in ("", "null", "{}",
                                                             "[]")
            head = [
                f"# figure: {figure}",
                f"# The code plot_code ran to draw {figure} ({when}).",
                (f"# Its data (the `data` object) is in {name}{DATA_SUFFIX} "
                 "beside this file." if has_data
                 else "# It was given no data."),
                "# To change the figure, change this code and run it again "
                f"with plot_code, saving under '{name}', so the redraw takes "
                "the figure's place.",
                CODE_MARK]
            _write(base + CODE_SUFFIX, "\n".join(head) + "\n" + str(code))
            if has_data:
                _write(base + DATA_SUFFIX, str(data_json))
            else:
                _remove(base + DATA_SUFFIX)
            _remove(base + ARGS_SUFFIX)
            kept = base + CODE_SUFFIX
        else:
            _write(base + ARGS_SUFFIX, json.dumps(
                {"figure": figure, "tool": tool, "drawn": when,
                 "arguments": args}, ensure_ascii=False, indent=1,
                default=str))
            _remove(base + CODE_SUFFIX)
            _remove(base + DATA_SUFFIX)
            kept = base + ARGS_SUFFIX
    except OSError:
        return None
    return _relative(kept, os.path.realpath(root))


def forget_code(root: Optional[str], name) -> None:
    """Remove the kept code of a figure that was retired."""
    stem = figure_stem(name, root) or _relative(name, root)
    if not root or not stem:
        return
    base = _code_base(root, stem)
    if base is None:
        return
    for suffix in (CODE_SUFFIX, DATA_SUFFIX, ARGS_SUFFIX):
        _remove(base + suffix)


def kept_code(name, root: Optional[str]) -> Optional[Dict[str, Any]]:
    """What is kept of the code that drew the figure ``name``: ``{"figure",
    "tool", "code_file", "code"[, "data", "data_file"]}`` for plot_code,
    ``{"figure", "tool", "code_file", "arguments"}`` for plot_data; ``None``
    when nothing is kept."""
    stem = figure_stem(name, root) or _relative(name, root)
    if not root or not stem:
        return None
    base = _code_base(root, stem)
    if base is None:
        return None
    real_root = os.path.realpath(root)
    try:
        if os.path.isfile(base + CODE_SUFFIX):
            with open(base + CODE_SUFFIX, encoding="utf-8") as fh:
                text = fh.read()
            figure = ""
            for line in text.splitlines()[:3]:
                if line.startswith("# figure: "):
                    figure = line[len("# figure: "):].strip()
            code = text.split(CODE_MARK + "\n", 1)[-1]
            out = {"figure": figure or stem + ".png", "tool": "plot_code",
                   "code_file": _relative(base + CODE_SUFFIX, real_root),
                   "code": code}
            if os.path.isfile(base + DATA_SUFFIX):
                with open(base + DATA_SUFFIX, encoding="utf-8") as fh:
                    out["data"] = fh.read()
                out["data_file"] = _relative(base + DATA_SUFFIX, real_root)
            return out
        if os.path.isfile(base + ARGS_SUFFIX):
            with open(base + ARGS_SUFFIX, encoding="utf-8") as fh:
                rec = json.load(fh)
            return {"figure": rec.get("figure") or stem + ".png",
                    "tool": "plot_data",
                    "code_file": _relative(base + ARGS_SUFFIX, real_root),
                    "arguments": rec.get("arguments") or {}}
    except (OSError, ValueError):
        return None
    return None


def kept_figures(root: Optional[str]) -> List[str]:
    """The figure stems (relative, '/' separators) whose code is kept."""
    if not root:
        return []
    folder = os.path.join(root, CODE_DIR)
    out: List[str] = []
    for base, _dirs, files in os.walk(folder):
        for f in files:
            for suffix in (CODE_SUFFIX, ARGS_SUFFIX):
                if f.endswith(suffix):
                    rel = _relative(os.path.join(base, f[:-len(suffix)]),
                                    folder)
                    if rel not in out:
                        out.append(rel)
        if len(out) > 500:
            break
    return out


def named_in(text, root: Optional[str]) -> List[str]:
    """The figures with kept code that ``text`` names: by file name (with a
    picture or chart extension), or by the name alone when it holds a
    character no plain word does (``_``, ``-`` or a digit), or by a version
    of either (``plan_v2.png`` names ``plan``). Their stems."""
    text = str(text or "")
    if not text or not root:
        return []
    found, versions = [], {}
    for stem in kept_figures(root):
        for cand in dict.fromkeys((stem, stem.rsplit("/", 1)[-1])):
            pat = (r"(?<![\w\-.])" + re.escape(cand)
                   + r"(?P<ver>" + _VERSION_TAIL + r")?"
                   + r"(?P<ext>\.(?:png|jpe?g|svg|gif|webp)|\.plotly\.json)?"
                   + r"(?![\w\-])")
            hit = None
            for m in re.finditer(pat, text, re.IGNORECASE):
                if m.group("ext") or re.search(r"[_\-\d]", cand) or (
                        m.group("ver") and not m.group("ver")[0].isspace()):
                    hit = m
                    if not m.group("ver"):
                        break
            if hit:
                found.append(stem)
                if hit.group("ver"):
                    versions[stem] = (stem + hit.group("ver")).lower()
                break
    # a version named in its own right is the figure meant, not its base
    named = {s.lower() for s in found}
    return [s for s in found if versions.get(s) not in named]


def _fence(lang: str, text: str) -> List[str]:
    return ["```" + lang, str(text).rstrip(), "```"]


def redraw_note(stems: Iterable[str], root: Optional[str]) -> str:
    """The block a brief that names figures with kept code gets: for each,
    the code it was drawn from (or plot_data's arguments) and what a redraw
    of it does."""
    blocks = []
    for stem in stems:
        rec = kept_code(stem, root)
        if rec is None:
            continue
        fig = rec["figure"]
        name = stem.rsplit("/", 1)[-1]
        n = draws_of(fig, root)
        drawn = (f"drawn {n} time(s) this turn" if n
                 else "drawn in an earlier turn")
        lines = [
            f"{REDRAW_HEADING}: {fig}, named in this brief, was {drawn} with "
            f"{rec['tool']}; what drew it is kept in {rec['code_file']} and "
            "is below. If the brief asks for a change to this figure, make "
            "the change in that code rather than drawing the figure anew "
            f"from a look at it, and draw it under the same name, {fig}, "
            "whatever name the brief gives: the redraw takes its place in "
            "the chat (one figure, one card). Only a second figure beside it "
            "(another unit, another case) takes a name of its own. If the "
            "brief only uses the figure, none of this applies."]
        if n:
            lines.append(
                f"Its draws count on from {n} this turn, whoever draws it, "
                f"under another name ({name}_v2) too: the draw's result says "
                f"when to stop. A drawing under another name leaves {fig} "
                "showing beside it unless the draw names it in replaces=.")
        if rec["tool"] == "plot_code":
            code = rec.get("code") or ""
            if len(code) <= MAX_BRIEF_CODE:
                lines.append("The code (pass it to plot_code with the "
                             "change):")
                lines += _fence("python", code)
            else:
                lines.append(f"The code is {len(code):,} characters: read it "
                             f"with read_text_file('{rec['code_file']}').")
            data = rec.get("data")
            if data is not None:
                if len(data) <= MAX_BRIEF_DATA:
                    lines.append("The data it was given (pass it as data):")
                    lines += _fence("json", data)
                else:
                    lines.append(
                        f"Its data ({len(data):,} characters) is in "
                        f"{rec['data_file']}: read it in the code (json or "
                        "pandas) rather than passing it as data.")
        else:
            text = json.dumps(rec.get("arguments") or {}, ensure_ascii=False,
                              default=str)
            if len(text) <= MAX_BRIEF_ARGS:
                lines.append("The plot_data arguments it was drawn with "
                             "(call plot_data with the change):")
                lines += _fence("json", text)
            else:
                lines.append(f"Its plot_data arguments ({len(text):,} "
                             "characters) are in "
                             f"{rec['code_file']}: read them with "
                             "read_text_file.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["CODE_DIR", "REDRAW_HEADING", "record_draw", "write_code",
           "forget_code", "kept_code", "kept_figures", "named_in",
           "redraw_note", "replacing", "draws_of", "figure_stem",
           "version_base", "is_figure_file"]
