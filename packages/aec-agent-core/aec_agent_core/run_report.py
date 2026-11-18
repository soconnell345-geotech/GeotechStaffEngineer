"""An offline report of one saved run: what the agents took in and where they
looked (BACKLOG EV-17).

WHY. The owner's rule is that every model run is read in full: every tool
call with its arguments and its result, the model's own words, the pages it
looked at. Until now that reading was done by an AI session from the raw
record files. This module turns a run's record into ONE self-contained HTML
file that an engineer opens in a browser and reads step by step, without a
developer and without a model.

USE::

    python -m aec_agent_core.run_report <run folder> [-o report.html]
                                        [--documents <folder>]

``<run folder>`` is a page conversation's folder (``activity.jsonl``,
``meta.json``, ``transcript.jsonl``, ``messages.json``, ``reads.json``,
``page_readings.json``, ``files/``) or one task run of the review suite
(``runs/<arm>/<task>/``: ``run.json``, ``activity.jsonl``, ``files/`` with the
vision records in ``files/.scratch/``). The report is written beside the
record as ``run_report.html`` unless ``-o`` names another file. The pages are
drawn behind the boxes when the document's PDF is in the run folder or under
``--documents`` (the suite deletes its uploads after scoring: point
``--documents`` at the suite's documents folder).

WHAT IT SHOWS

1. The run: what was asked, the model, when; totals (model calls, tokens in
   and out where recorded, vision calls, helpers started, wall time, and
   the turns' own count of image calls where their records keep it); the
   files the run delivered; the notes the record itself carries (an error, a
   step cap reached, a turn's image calls against its ceiling, a
   page-citation check, a helper's work kept, waits).
2. Step by step: every step in order, who took it (the main agent, or which
   helper, under the brief it was given), the model's own words, and each
   tool call with its arguments and its result (a long result folded, with
   its size), its vision calls, and the tokens of each step.
3. What was taken in: per document, a grid of its pages saying, for each,
   whether its text was read, it was looked at whole, in tiles, zoomed, made
   into a contact sheet, or never opened, with the counts.
4. Where they looked: per page looked at, the page's outline at its
   proportions with a box for every look, tile and zoom (from the view boxes
   the tools returned), numbered as the tool calls of part 2; the page itself
   drawn behind when its PDF is found.
5. What the record holds: each file found or missing and what that leaves
   out, and the older shapes met.

THE PAGE VIEWER. Choosing a square of the page grid (or "Open in the page
viewer" under a drawing) opens that page over the report: its image with a
box for every look, tile and zoom, each box telling its call (number, tool,
who made it and in which step); choosing a box shows what was asked and the
start of what came back. Previous / next page and previous / next page
looked at (also the arrow keys, with Shift for the pages looked at); Esc or
the browser's Back closes it. It is the report's own inline script reading
a JSON block of the same file: no network. Page images are bounded (see
the constants below): a page with no image still shows its outline where
the record gives one, and lists its looks.

Only what the record says is shown: nothing is inferred from a model's
account. A missing or older-shaped file is named in part 5 (and, for the
activity log, at the top), never a traceback. Nothing is sent anywhere; the
file holds the conversation's text and page images and is written only where
the caller says.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from aec_agent_core.activity_log import mask_payload, masked_record
from aec_agent_core.coverage import (GRID_TOOLS, LOOK_TOOLS, SWEEP_TOOLS,
                                     parse_pages, pages_from_call)

#: The report's file name beside the record.
REPORT_FILE = "run_report.html"

ACTIVITY_FILE = "activity.jsonl"
RUN_FILE = "run.json"
META_FILE = "meta.json"
TRANSCRIPT_FILE = "transcript.jsonl"
MESSAGES_FILE = "messages.json"
READS_FILE = "reads.json"
READINGS_FILE = "page_readings.json"
#: Where the activity log keeps a payload it cut short, whole.
SIDECAR_DIR = "activity_full"
#: Where the vision tools keep their records for a bare working folder.
SCRATCH_DIR = ".scratch"

#: The report's first line: it holds what the documents hold.
CONTENT_NOTE = ("Holds this run's text and page images; treat it as you "
                "treat the documents.")

#: A result, an argument list or a text up to this many characters is shown
#: open; a longer one is folded with its size.
OPEN_CHARS = 600
#: Most characters of one payload shown (a whole copy beside the log can be
#: megabytes); the rest is named, not shown.
MAX_SHOWN_CHARS = 200_000
#: A whole copy beside the log is read up to this size.
MAX_SIDECAR_BYTES = 20 * 1024 * 1024
#: Page images, all JPEG of quality :data:`JPEG_QUALITY`, bounded so that a
#: long document still gives a file a browser opens. Measured on made-up
#: 200-page PDFs (module_work/ev17/CHECK.md): at 420 pixels on the long
#: side a page takes about 20 KB, at 900 about 75 to 100 KB.
#: - Each page looked at is drawn small (:data:`PAGE_IMAGE_PX`) behind its
#:   boxes in part 4, at most :data:`MAX_PAGE_IMAGES` of them;
#: - and large (:data:`VIEW_IMAGE_PX`) for the page viewer while the large
#:   images stay under :data:`MAX_VIEW_IMAGE_BYTES` (past it the viewer
#:   shows the small one);
#: - the other pages of a document whose PDF is found are drawn small for
#:   the viewer, the pages taken in some other way first, while they stay
#:   under :data:`MAX_OTHER_IMAGE_BYTES`;
#: - and no more than :data:`MAX_PAGES_DRAWN` pages are drawn in all (a page
#:   looked at, drawn at both sizes, counts once).
#: The two byte caps count JPEG bytes; the file holds each image as base64,
#: a third more. The small images of part 4 have no byte cap: at most
#: :data:`MAX_PAGE_IMAGES` pages.
MAX_PAGE_IMAGES = 300
PAGE_IMAGE_PX = 420
VIEW_IMAGE_PX = 900
JPEG_QUALITY = 60
MAX_VIEW_IMAGE_BYTES = 8 * 1024 * 1024
MAX_OTHER_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PAGES_DRAWN = 800
#: In the page viewer, what a call was asked and what came back, each up to
#: this many characters (the whole is in part 2).
VIEW_ASKED_CHARS = 1500
VIEW_BACK_CHARS = 1500
#: A page outline's long side on screen, in CSS pixels.
FIGURE_PX = 300

TASK_TOOL = "task"
OPEN_TOOL = "open_document"
PAGE_TOOL = "analyze_pdf_page"
ZOOM_TOOL = "render_region"
THUMB_TOOL = "render_page_thumbnails"
TEXT_OPENING_TOOLS = ("read_document", "read_pdf_text") + GRID_TOOLS
OPENING_TOOLS = TEXT_OPENING_TOOLS + LOOK_TOOLS + SWEEP_TOOLS

#: How a page was reached, in the order the grid shows them: (mark, letter,
#: the count's label, what it means).
MARKS = (
    ("text", "T", "text read",
     "its text was read (read_document, read_pdf_text, log_grid)"),
    ("whole", "W", "looked at whole", "looked at whole, in one image"),
    ("tiles", "G", "tiled",
     "looked at whole and in tiles (a grid of images)"),
    ("zoom", "Z", "zoomed", "a zoom on part of it"),
    ("contact", "C", "on a contact sheet",
     "shown as a thumbnail on a contact sheet (render_page_thumbnails)"),
    ("text_empty", "E", "no text layer",
     "a text tool reached it and found no text layer"),
    ("sweep", "S", "swept", "asked about by a sweep (records before DR-39)"),
)
#: The counts every document shows, even at zero.
ALWAYS_COUNTED = ("text", "whole", "tiles", "zoom", "contact")
#: Said of a page count the record does not give.
N_FROM_LAST = "the last page reached (the record does not give the count)"

#: App events that come many to a run: shown as one line with a count.
COUNTED_EVENTS = ("model_busy", "pacer_wait", "engine_retry",
                  "helper_queued", "side_call_timeout", "side_call_late_end",
                  "vision_slot_timeout", "process_start", "image_read")
#: App events of a call that are part of its ordinary record: shown with
#: the call as a note, where the others (a timeout, a wait given up) are
#: shown as a warning.
PLAIN_EVENTS = ("image_read",)
#: What an app event is, in words, where its name does not say.
EVENT_WORDS = {
    "cited_pages": "the answer's page citations checked against the pages "
                   "opened",
    "cited_documents": "the documents cited checked against what was looked "
                       "up",
    "judgment_check": "the answer's judgment calls checked for a source",
    "coverage_gate": "a note the coverage gate gave the model (older records)",
    "helper_kept": "a helper's work was kept in a file",
    "helper_records_kept": "a helper's records were kept",
    "model_busy": "a model call refused as busy and asked again",
    "pacer_wait": "a call waited under the tokens-a-minute pacer",
    "engine_retry": "a model call retried by its own client",
    "helper_queued": "a helper waited for a place under the cap of helpers",
    "side_call_timeout": "a vision call given up at its time limit",
    "side_call_late_end": "a vision call given up that ended after all",
    "vision_slot_timeout": "a look that waited too long for a vision slot",
    "image_read": "the image a look read (a file's sha is the one the "
                  "turn's files give its card)",
    "process_start": "a process began writing this record (more than one "
                     "means the app restarted)",
    "restored": "the conversation was brought back from a copy",
    "page_error": "the page itself raised an error",
    "model_error": "a model call failed",
}
#: Fields of a record that say where it sits, not what happened.
_PLACE_KEYS = frozenset({"turn", "ts", "t", "agent", "event", "task_id",
                         "helper", "run_id", "parent_run_id"})
#: Fields whose seconds a counted event's line adds up.
_WAIT_KEYS = ("wait_s", "waited_s", "queued_s", "held_s")
#: Under the totals, where a turn's record keeps its own count of image
#: calls (``turn_end``'s ``image_calls``, from 1.2.0): what each count is.
IMAGE_COUNTS_NOTE = (
    "Vision calls counts the vision model's answers that the activity log "
    "holds, the vision probe's included; image calls counts the images the "
    "turn itself counted as sent for a separate reading (each image once "
    "with its retries; a page looked at twice counts twice), not those its "
    "ceiling refused; so the two can differ when a call failed or was given "
    "up, when an ask given up at its time limit answered after all, or when "
    "the probe ran.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _b64_len(n_bytes: int) -> int:
    """How many characters ``n_bytes`` of data take as base64."""
    return 4 * ((int(n_bytes) + 2) // 3)


def _size(n_bytes: int) -> str:
    """``"1.2 MB"``, or ``"35 kB"`` below a megabyte."""
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f} MB"
    return f"{max(1, round(n_bytes / 1024)) if n_bytes else 0} kB"


def _secs(value: Any) -> str:
    try:
        s = float(value)
    except (TypeError, ValueError):
        return ""
    if s < 60:
        return f"{s:.1f} s"
    m, s = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m} min {s} s"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def _when(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _folder_name(folder: Any) -> str:
    """A folder's own name: the page names no absolute path (a report is
    passed to other people, and a local path carries the user's name)."""
    full = os.path.abspath(str(folder))
    return os.path.basename(full.rstrip("\\/")) or "the run folder"


def _when_any(value: Any) -> str:
    """A time a record holds: a number of seconds (``meta.json``'s
    ``created``) as a date and time; a text (the suite's stamp) as it is."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _when(value)
    return "" if value is None else str(value)


def _json_obj(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _args(value: Any) -> Any:
    """A tool call's arguments as logged: a dict, or the string it came as."""
    if isinstance(value, str):
        data = _json_obj(value)
        return data if data is not None else value
    return value


def _doc_key(name: Any) -> str:
    """One key per document: its file name, case folded."""
    text = str(name or "").strip().replace("\\", "/").rstrip("/")
    return os.path.basename(text).lower() if text else ""


def _one_line(text: Any, room: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= room else s[:room - 3].rstrip() + "..."


_SCALAR = r'(?:-?\d[\d.eE+-]*|"(?:[^"\\\n]|\\.){0,40}"|true|false|null)'
#: A list of a few short scalars (a view box, a pixel size, page numbers),
#: as ``json.dumps(indent=1)`` spreads it over lines.
_FLAT_LIST = re.compile(r"\[\s*(" + _SCALAR + r"(?:,\s*" + _SCALAR
                        + r"){0,11})\s*\]")


def _pretty(text: str) -> str:
    """A JSON payload laid out one field a line, short lists of numbers or
    words kept on one; anything else as it is."""
    s = text.strip()
    if s[:1] in "{[" and len(s) < MAX_SHOWN_CHARS:
        try:
            laid = json.dumps(json.loads(s), indent=1, ensure_ascii=False)
        except ValueError:
            return text
        return _FLAT_LIST.sub(
            lambda m: "[" + ", ".join(
                p.strip() for p in re.split(r",\s*\n\s*", m.group(1))) + "]",
            laid)
    return text


def _box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    """``[x0, y0, x1, y1]`` as four floats with some area, else ``None``."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _usage(rec: dict) -> Dict[str, Optional[int]]:
    """``in``, ``out`` and ``cache`` tokens of one model call, ``None`` where
    the record does not say (records before a field existed)."""
    u = rec.get("usage") if isinstance(rec.get("usage"), dict) else {}
    first = (lambda *keys: next((_int(u.get(k)) for k in keys
                                 if _int(u.get(k)) is not None), None))
    return {"in": first("input_tokens", "prompt_tokens"),
            "out": first("output_tokens", "completion_tokens"),
            "cache": first("cache_read")}


def _usage_text(u: Dict[str, Optional[int]]) -> str:
    if u.get("in") is None and u.get("out") is None:
        return ""
    parts = []
    if u.get("in") is not None:
        cache = f" ({_n(u['cache'])} from cache)" if u.get("cache") else ""
        parts.append(f"{_n(u['in'])} tokens in{cache}")
    if u.get("out") is not None:
        parts.append(f"{_n(u['out'])} out")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass
class Part:
    """One file of the record: whether it was found, and what it gives."""
    name: str
    status: str
    meaning: str


def _read_json(path: str) -> Tuple[Any, Optional[str]]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, "not found"
    except (OSError, ValueError) as exc:
        return None, f"unreadable ({type(exc).__name__})"


def _read_jsonl(path: str) -> Tuple[Optional[list], int, Optional[str]]:
    out: list = []
    bad = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
                else:
                    bad += 1
    except FileNotFoundError:
        return None, 0, "not found"
    except OSError as exc:
        return None, 0, f"unreadable ({type(exc).__name__})"
    return out, bad, None


class RunRecord:
    """What one run folder holds, read once; each file optional."""

    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.parts: List[Part] = []
        self.older: List[str] = []
        acts, self.bad_lines, err = _read_jsonl(
            os.path.join(self.folder, ACTIVITY_FILE))
        # The text of an error is shown masked as the log now keeps it,
        # including in a record written before the log masked it.
        self.activity: List[dict] = [masked_record(r) for r in acts or []]
        self.has_activity = acts is not None
        status = err or f"found: {_n(len(self.activity))} records" + (
            f", {_n(self.bad_lines)} unreadable lines skipped"
            if self.bad_lines else "")
        self.parts.append(Part(
            ACTIVITY_FILE, status,
            "every tool call, model call and helper of each turn: the step "
            "list, the totals and most of the page grid come from it"))
        run, err = _read_json(os.path.join(self.folder, RUN_FILE))
        self.run: Optional[dict] = run if isinstance(run, dict) else None
        self.parts.append(Part(
            RUN_FILE, err or ("found" if self.run else "not an object"),
            "the review suite's summary of one task run: the question, the "
            "delivered files, the time (a page conversation has none)"))
        meta, err = _read_json(os.path.join(self.folder, META_FILE))
        self.meta: Optional[dict] = meta if isinstance(meta, dict) else None
        self.parts.append(Part(
            META_FILE, err or "found",
            "a page conversation's title and dates (a suite run has none)"))
        trans, _bad, err = _read_jsonl(
            os.path.join(self.folder, TRANSCRIPT_FILE))
        self.transcript: List[dict] = trans or []
        self.parts.append(Part(
            TRANSCRIPT_FILE, err or f"found: {_n(len(self.transcript))} "
                                    f"entries",
            "the messages as the page showed them: the questions in full, "
            "the answers and their files (a suite run has none)"))
        msgs, err = _read_json(os.path.join(self.folder, MESSAGES_FILE))
        self.messages: list = msgs if isinstance(msgs, list) else []
        self.parts.append(Part(
            MESSAGES_FILE, err or "found",
            "the conversation as the model was given it; used here only for "
            "the question when nothing else holds it"))
        self.reads: List[dict] = []
        self.readings: List[dict] = []
        for fname, key, attr, meaning in (
                (READS_FILE, "reads", "reads",
                 "the vision tools' list of pages and regions looked at"),
                (READINGS_FILE, "readings", "readings",
                 "each page reading kept whole, with its view boxes")):
            path = self._vision_file(fname)
            data, err = _read_json(path) if path else (None, "not found")
            rows = data.get(key) if isinstance(data, dict) else data
            rows = [r for r in rows or [] if isinstance(r, dict)] \
                if isinstance(rows, list) else []
            setattr(self, attr, rows)
            where = os.path.relpath(path, self.folder) if path else fname
            self.parts.append(Part(
                where, err or f"found: {_n(len(rows))} entries",
                meaning + "; fills the page grid where the activity log "
                          "is missing or older"))
        side = os.path.join(self.folder, SIDECAR_DIR)
        n_side = len(os.listdir(side)) if os.path.isdir(side) else 0
        self.parts.append(Part(
            SIDECAR_DIR + "/", f"found: {_n(n_side)} files" if n_side
            else "not found (nothing was cut short, or an older record)",
            "whole copies of payloads the activity log cut short"))

    def _vision_file(self, name: str) -> Optional[str]:
        for sub in ("", SCRATCH_DIR, os.path.join("files", SCRATCH_DIR)):
            path = os.path.join(self.folder, sub, name)
            if os.path.isfile(path):
                return path
        return None

    def kind(self) -> str:
        if self.run is not None:
            bits = [f"task {self.run.get('task')}" if self.run.get("task")
                    else "", f"arm {self.run.get('arm')}"
                    if self.run.get("arm") else ""]
            bits = ", ".join(b for b in bits if b)
            return "a review suite run" + (f" ({bits})" if bits else "")
        if self.meta is not None or self.transcript:
            title = (self.meta or {}).get("title")
            return "a page conversation" + (f": {title}" if title else "")
        return "a run record"

    def sidecar(self, rel: Any) -> Optional[str]:
        """The whole payload kept beside the log under ``rel``, if readable."""
        if not rel:
            return None
        # only a file inside the run folder itself: resolved paths, compared
        # whole (a sibling folder whose name starts with this one's is out)
        base = os.path.realpath(self.folder)
        path = os.path.realpath(os.path.join(base, str(rel)))
        try:
            if path == base or os.path.commonpath([base, path]) != base:
                return None
        except ValueError:                 # another drive
            return None
        try:
            if os.path.getsize(path) > MAX_SIDECAR_BYTES:
                return None
            with open(path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None

    def questions(self) -> Dict[int, str]:
        """Each turn's question in full, where a file other than the activity
        log holds it (that log keeps the first 500 characters)."""
        out: Dict[int, str] = {}
        for i, t in enumerate((self.run or {}).get("turns") or [], 1):
            if isinstance(t, dict) and t.get("question"):
                out[i] = str(t["question"])
        if not out and (self.run or {}).get("question"):
            out[1] = str(self.run["question"])
        if not out:
            # the pages' writer (append_transcript) keeps a message's words
            # under "text"
            users = [e for e in self.transcript if e.get("role") == "user"]
            for i, e in enumerate(users, 1):
                out[i] = str(e.get("text") or "")
        if not out:
            for m in self.messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("type")
                body = m.get("data") if isinstance(m.get("data"), dict) else m
                if role in ("user", "human"):
                    out[1] = _content_text(body.get("content"))
                    break
        return out

    def answers(self) -> Dict[int, str]:
        """Each turn's answer as the user received it, where held."""
        out: Dict[int, str] = {}
        for i, t in enumerate((self.run or {}).get("turns") or [], 1):
            if isinstance(t, dict) and t.get("answer"):
                out[i] = str(t["answer"])
        if not out:
            turn = 0
            for e in self.transcript:
                if e.get("role") == "user":
                    turn += 1
                elif e.get("role") == "assistant" and turn and e.get("text"):
                    out[turn] = str(e["text"])
        return out

    def answer_files(self) -> Dict[int, List[str]]:
        """Each turn's files shown with its answer, from the transcript."""
        out: Dict[int, List[str]] = {}
        turn = 0
        for e in self.transcript:
            if e.get("role") == "user":
                turn += 1
            elif e.get("role") == "assistant" and turn:
                files = [str(f) for f in e.get("artifacts") or []
                         if isinstance(f, (str, int, float))]
                if files:
                    out.setdefault(turn, []).extend(files)
        return out

    def delivered(self) -> List[dict]:
        """Files the run delivered, from the run summary and the transcript
        (the activity log's own ``turn_files`` are read with the turns)."""
        out: List[dict] = []
        for f in (self.run or {}).get("files") or []:
            out.append({"path": str(f), "how": "in run.json"})
        for e in self.transcript:
            for f in e.get("artifacts") or []:
                out.append({"path": str(f), "how": "shown with the answer"})
        return out


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text") if isinstance(b, dict) else b)
                         for b in content
                         if not isinstance(b, dict) or "text" in b)
    return "" if content is None else str(content)


# ---------------------------------------------------------------------------
# Turns, actors, steps and calls
# ---------------------------------------------------------------------------

@dataclass
class Call:
    """One tool call, with what came back."""
    name: str
    args: Any
    start: Optional[dict]
    end: Optional[dict] = None
    delivered: List[dict] = field(default_factory=list)
    side: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    helper: Optional["Actor"] = None
    number: int = 0
    #: ``(document key, page or None)`` for each page drawn or document
    #: grid this call reached (part 2 links to them).
    where: List[Tuple[str, Optional[int]]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.end is not None and self.end.get("event") == "tool_error"

    @property
    def refused(self) -> bool:
        return self.start is None and self.end is None and bool(
            self.delivered)


@dataclass
class Step:
    """One model call of an agent and the tool calls it asked for."""
    model: Optional[dict] = None
    calls: List[Call] = field(default_factory=list)


@dataclass
class Actor:
    """The main agent, or one helper (one ``task`` call)."""
    key: str
    agent: str
    brief: str = ""
    label: str = ""
    steps: List[Step] = field(default_factory=list)
    task_call: Optional[Call] = None
    numbered: bool = False

    def step_for_call(self) -> Step:
        if not self.steps:
            self.steps.append(Step())
        return self.steps[-1]

    def calls(self) -> Iterator[Call]:
        for s in self.steps:
            yield from s.calls


@dataclass
class Turn:
    number: Any
    start: Optional[dict] = None
    end: Optional[dict] = None
    files: List[dict] = field(default_factory=list)
    primary: Actor = field(default_factory=lambda: Actor("primary",
                                                         "primary",
                                                         label="Main agent"))
    helpers: List[Actor] = field(default_factory=list)
    notes: List[dict] = field(default_factory=list)
    side: Dict[str, dict] = field(default_factory=dict)
    loose_side: List[dict] = field(default_factory=list)
    agent_starts: set = field(default_factory=set)
    agent_ends: set = field(default_factory=set)
    models: Dict[str, set] = field(default_factory=dict)


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


class _TurnBuilder:
    """Sorts one turn's records into the main agent's and each helper's
    steps. A helper is told apart by the ``task_id`` its records carry
    (since 2026-10-14), tied to the ``task`` call that started it by that id,
    else by its brief; an older record without it goes to the open ``task``
    call of its agent's type."""

    def __init__(self, number: Any):
        self.t = Turn(number)
        self.by_key: Dict[str, Actor] = {}
        self.calls: Dict[str, Call] = {}
        self.task_calls: List[Call] = []
        self.task_by_id: Dict[str, Call] = {}

    # -- who --------------------------------------------------------------
    def _helper(self, key: str, agent: str, brief: str) -> Actor:
        a = self.by_key.get(key)
        if a is None:
            a = Actor(key, agent, brief=brief)
            self.by_key[key] = a
            self.t.helpers.append(a)
        return a

    def _tie(self, actor: Actor, call: Call) -> None:
        if actor.task_call is not None or call.helper is not None:
            return
        actor.task_call, call.helper = call, actor
        desc = call.args.get("description") if isinstance(call.args,
                                                          dict) else None
        if desc:
            actor.brief = str(desc)

    def _match_task(self, tid: str, agent: str, brief: str) -> Optional[Call]:
        if tid in self.task_by_id:
            return self.task_by_id[tid]
        free = [c for c in self.task_calls if c.helper is None]
        head = _norm(brief)
        if head.endswith(" ..."):
            head = head[:-4]
        if head:
            for c in free:
                desc = c.args.get("description") if isinstance(
                    c.args, dict) else ""
                if _norm(desc).startswith(head):
                    return c
        for c in free:
            if c.end is None and (c.start or {}).get("subagent") == agent:
                return c
        return None

    def actor(self, rec: dict) -> Actor:
        agent = str(rec.get("agent") or "primary")
        tid = str(rec.get("task_id") or "")
        if tid:
            a = self._helper("t:" + tid, agent, str(rec.get("helper") or ""))
            if a.task_call is None:
                call = self._match_task(tid, agent, a.brief)
                if call is not None:
                    self._tie(a, call)
            return a
        if agent in ("primary", "app"):
            return self.t.primary
        open_calls = [c for c in self.task_calls if c.end is None
                      and (c.start or {}).get("subagent") == agent]
        if open_calls:
            call = open_calls[-1]
            a = call.helper or self._helper("c:" + str(
                (call.start or {}).get("run_id")), agent, "")
            self._tie(a, call)
            return a
        return self._helper("a:" + agent, agent, "")

    # -- what -------------------------------------------------------------
    def add(self, rec: dict) -> None:
        ev = rec.get("event")
        t = self.t
        rid = str(rec.get("run_id") or "")
        if ev == "turn_start":
            t.start = t.start or rec
        elif ev == "turn_end":
            t.end = rec
        elif ev == "turn_files":
            t.files.extend(f for f in rec.get("files") or []
                           if isinstance(f, dict))
        elif ev in ("model_start", "model_end", "model_error"):
            self._model(ev, rid, rec)
        elif ev == "tool_start":
            self._tool_start(rid, rec)
        elif ev in ("tool_end", "tool_error"):
            call = self.calls.get(rid)
            if call is None:              # its start was not recorded
                call = Call(str(rec.get("name") or "tool"), None, None)
                self.actor(rec).step_for_call().calls.append(call)
                if rid:
                    self.calls[rid] = call
            call.end = rec
        elif ev in ("tool_delivered", "tool_refused"):
            self._delivered(ev, rec)
        else:
            tool_rid = str(rec.get("tool_run_id") or "")
            if tool_rid in self.calls:
                self.calls[tool_rid].events.append(rec)
            t.notes.append(rec)

    def _model(self, ev: str, rid: str, rec: dict) -> None:
        t = self.t
        if rec.get("vision") is not None or (rid and rid in t.side):
            side = t.side.setdefault(rid or f"_{len(t.side)}", {})
            side["start" if ev == "model_start" else "end"] = rec
            if ev == "model_start":
                if rec.get("model"):
                    t.models.setdefault("vision calls", set()).add(
                        str(rec["model"]))
                call = self.calls.get(str(rec.get("tool_run_id") or ""))
                (call.side if call is not None else t.loose_side).append(side)
            elif "start" not in side:
                t.loose_side.append(side)
            if ev == "model_error":
                t.notes.append(rec)
            return
        a = self.actor(rec)
        if ev == "model_start":
            if rid:
                t.agent_starts.add(rid)
            if rec.get("model"):
                who = "main agent" if a is t.primary else "helpers"
                t.models.setdefault(who, set()).add(str(rec["model"]))
            return
        if rid:
            t.agent_ends.add(rid)
        a.steps.append(Step(model=rec))
        if ev == "model_error":
            t.notes.append(rec)

    def _tool_start(self, rid: str, rec: dict) -> None:
        name = str(rec.get("name") or "tool")
        call = Call(name, _args(rec.get("args")), rec)
        if name == TASK_TOOL and str(rec.get("agent") or "primary") \
                in ("primary", "app"):
            actor = self.t.primary   # its own record may carry the helper's id
        else:
            actor = self.actor(rec)
        actor.step_for_call().calls.append(call)
        if rid:
            self.calls[rid] = call
        if name == TASK_TOOL:
            self.task_calls.append(call)
            tid = str(rec.get("task_id") or "")
            if tid:
                self.task_by_id[tid] = call
                waiting = self.by_key.get("t:" + tid)
                if waiting is not None:
                    self._tie(waiting, call)

    def _delivered(self, ev: str, rec: dict) -> None:
        name = str(rec.get("name") or "tool")
        if name == TASK_TOOL and str(rec.get("agent") or "primary") \
                in ("primary", "app"):
            actor = self.t.primary
        else:
            actor = self.actor(rec)
        want_failed = ev == "tool_refused" and bool(rec.get("raised"))
        for call in reversed(list(actor.calls())):
            if call.name != name or call.end is None:
                continue
            if want_failed and not call.failed:
                continue
            if any(d.get("event") == ev for d in call.delivered):
                continue
            call.delivered.append(rec)
            return
        if ev == "tool_refused":
            # no tool ran: an unknown tool name, an interception
            actor.step_for_call().calls.append(
                Call(name, None, None, delivered=[rec]))
        else:
            self.t.notes.append(rec)


def build_turns(records: List[dict]) -> List[Turn]:
    """The records of a run, as turns of steps and calls, numbered in the
    order the report shows them."""
    builders: Dict[Any, _TurnBuilder] = {}
    order: List[Any] = []
    current: Any = None
    early: List[dict] = []         # records with no turn before any turn
    for rec in records:
        turn = rec.get("turn")
        if turn is None:
            turn = current
        if turn is None:
            early.append(rec)
            continue
        current = turn
        if turn not in builders:
            builders[turn] = _TurnBuilder(turn)
            order.append(turn)
            for held in early:
                builders[turn].add(held)
            early = []
        builders[turn].add(rec)
    if early:                      # no record named a turn at all
        builders[None] = _TurnBuilder("?")
        order.append(None)
        for held in early:
            builders[None].add(held)
    turns = [builders[k].t for k in order]
    n = 0
    h = 0

    def walk(actor: Actor) -> None:
        nonlocal n, h
        for step in actor.steps:
            for call in step.calls:
                n += 1
                call.number = n
                helper = call.helper
                if helper is not None and not helper.numbered:
                    h += 1
                    helper.numbered = True
                    helper.label = f"Helper {h}"
                    walk(helper)

    for t in turns:
        walk(t.primary)
        for a in t.helpers:
            if not a.numbered:
                h += 1
                a.numbered = True
                a.label = f"Helper {h}"
                walk(a)
    return turns


def _all_calls(turns: List[Turn]) -> List[Call]:
    out: List[Call] = []

    def walk(actor: Actor) -> None:
        for call in actor.calls():
            out.append(call)
            if call.helper is not None and call.helper.task_call is call:
                walk(call.helper)

    for t in turns:
        walk(t.primary)
        for a in t.helpers:
            if a.task_call is None:
                walk(a)
    return sorted(out, key=lambda c: c.number)


def _places(turns: List[Turn]) -> Dict[int, dict]:
    """Call number -> where part 2 shows it: ``turn``, the ``step`` of its
    agent (from 1) and ``who`` made it (the main agent, or a helper with its
    type)."""
    out: Dict[int, dict] = {}

    def walk(actor: Actor, t: Turn) -> None:
        who = actor.label if actor.key == "primary" else (
            f"{actor.label} ({actor.agent})" if actor.agent else actor.label)
        for i, step in enumerate(actor.steps, 1):
            for call in step.calls:
                out[call.number] = {"turn": t.number, "step": i, "who": who}
                if call.helper is not None and call.helper.task_call is call:
                    walk(call.helper, t)

    for t in turns:
        walk(t.primary, t)
        for a in t.helpers:
            if a.task_call is None:
                walk(a, t)
    return out


def _owner(turns: List[Turn]) -> Dict[int, str]:
    """Call number -> the label of the agent that made it."""
    out: Dict[int, str] = {}

    def walk(actor: Actor) -> None:
        for call in actor.calls():
            out[call.number] = actor.label
            if call.helper is not None and call.helper.task_call is call:
                walk(call.helper)

    for t in turns:
        walk(t.primary)
        for a in t.helpers:
            if a.task_call is None:
                walk(a)
    return out


# ---------------------------------------------------------------------------
# Pages: what was taken in, and where
# ---------------------------------------------------------------------------

@dataclass
class Look:
    number: int
    kind: str                 # "page", "tile" or "zoom"
    box: Optional[Tuple[float, float, float, float]]
    label: str = ""
    tool: str = ""
    who: str = ""
    source: str = ""          # "" (the activity log) or the file it came from


@dataclass
class PageUse:
    marks: Dict[str, List[int]] = field(default_factory=dict)
    looks: List[Look] = field(default_factory=list)

    def mark(self, kind: str, number: int) -> None:
        nums = self.marks.setdefault(kind, [])
        if number and number not in nums:
            nums.append(number)


@dataclass
class Document:
    key: str
    name: str
    n_pages: Optional[int] = None
    n_from: str = ""
    pages: Dict[int, PageUse] = field(default_factory=dict)
    pdf: Optional[str] = None
    index: int = 0

    def use(self, page: int) -> PageUse:
        return self.pages.setdefault(page, PageUse())


_VIEW_RE = re.compile(r'"view"\s*:\s*\[([^\]]*)\]')
_TILE_RE = re.compile(r'"tile"\s*:\s*"([^"]+)"\s*,\s*"view"\s*:\s*\[([^\]]*)\]')


def _nums(text: str) -> Optional[Tuple[float, float, float, float]]:
    return _box([v for v in text.split(",")])


def _looks_of(name: str, args: Any, data: Optional[dict], text: str
              ) -> List[Tuple[str, Optional[tuple], str]]:
    """``[(kind, box, label)]`` one look call made: a whole page (with its
    tiles) for ``analyze_pdf_page``, a zoom for ``render_region``. The boxes
    are the ``view`` rects the tool returned (PDF points, as the page
    displays); from a result cut short with no whole copy, what can be found
    in its text."""
    args = args if isinstance(args, dict) else {}
    out: List[Tuple[str, Optional[tuple], str]] = []
    if data is not None:
        view = _box(data.get("view"))
        tiles = [t for t in data.get("tiles") or [] if isinstance(t, dict)]
    else:
        tiles_found = _TILE_RE.findall(text or "")
        tiles = [{"tile": lab, "view": list(_nums(v) or ())}
                 for lab, v in tiles_found]
        spans = [m.span() for m in _TILE_RE.finditer(text or "")]
        view = None
        for m in _VIEW_RE.finditer(text or ""):
            if not any(a <= m.start() < b for a, b in spans):
                view = _nums(m.group(1))
                break
    if name == ZOOM_TOOL:
        out.append(("zoom", view or _box(args.get("bbox")), ""))
        return out
    out.append(("page", view, ""))
    for t in tiles:
        out.append(("tile", _box(t.get("view")), str(t.get("tile") or "")))
    return out


class PageMap:
    """Every document the run reached and, per page, how (part 3) and every
    look with its box (part 4), from the record only."""

    def __init__(self, record: RunRecord, turns: List[Turn]):
        self.record = record
        self.docs: Dict[str, Document] = {}
        self.handles: Dict[str, str] = {}
        self.unplaced = 0
        owner = _owner(turns)
        for call in _all_calls(turns):
            try:
                self._call(call, owner.get(call.number, ""))
            except Exception:  # noqa: BLE001 - one odd result, not the report
                self.unplaced += 1
        self._vision_records()
        for i, doc in enumerate(sorted(self.docs.values(),
                                       key=lambda d: d.name.lower()), 1):
            doc.index = i

    def doc(self, key: str, name: str) -> Document:
        d = self.docs.get(key)
        if d is None:
            d = self.docs[key] = Document(key, name or key)
        return d

    def _ref_doc(self, ref: Any, args: dict) -> Optional[Document]:
        kind, value = ref if isinstance(ref, tuple) else ("", "")
        if not value:
            handle = str(args.get("handle") or "")
            kind, value = ("handle", handle) if handle else ("", "")
        if not value:
            return None
        if kind == "handle":
            shown = self.handles.get(value) or value
            return self.doc(_doc_key(shown), os.path.basename(
                shown.replace("\\", "/")))
        return self.doc(_doc_key(value), os.path.basename(
            str(value).replace("\\", "/")))

    def _call(self, call: Call, who: str) -> None:
        if call.end is None or call.failed:
            return
        text = _whole_text(self.record, call.end, "result")[0]
        data = _json_obj(text)
        if data is not None and data.get("error"):
            return
        args = call.args if isinstance(call.args, dict) else {}
        if call.name == OPEN_TOOL and data is not None:
            handle = str(data.get("handle") or "")
            shown = str(data.get("name") or data.get("source")
                        or args.get("source") or "")
            if handle and shown:
                self.handles[handle] = shown
            if shown and isinstance(data.get("n_pages"), int):
                d = self.doc(_doc_key(shown), os.path.basename(
                    shown.replace("\\", "/")))
                d.n_pages, d.n_from = data["n_pages"], "open_document"
            return
        if call.name == THUMB_TOOL and data is not None:
            doc = self._ref_doc(None, args)
            if doc is None:
                return
            for sheet in data.get("sheets") or []:
                if isinstance(sheet, dict):
                    for p in parse_pages(sheet.get("pages")):
                        doc.use(p).mark("contact", call.number)
            if (doc.key, None) not in call.where:
                call.where.append((doc.key, None))
            return
        if call.name not in OPENING_TOOLS:
            return
        ref, pages = pages_from_call(call.name, args, text)
        doc = self._ref_doc(ref, args)
        if doc is None or not pages:
            return
        for page, how in pages:
            if call.name in LOOK_TOOLS:
                continue
            doc.use(page).mark(how, call.number)
        if call.name in LOOK_TOOLS:
            page = pages[0][0]
            use = doc.use(page)
            looks = _looks_of(call.name, args, data, text)
            tiled = any(k == "tile" for k, _b, _l in looks)
            use.mark("zoom" if call.name == ZOOM_TOOL
                     else "tiles" if tiled else "whole", call.number)
            for kind, box, label in looks:
                use.looks.append(Look(call.number, kind, box, label,
                                      call.name, who))
            call.where.append((doc.key, page))

    def _vision_records(self) -> None:
        """Looks the vision tools' own records hold that the activity log
        does not (a missing or older log): unnumbered, named by file."""
        logged = {(d.key, p) for d in self.docs.values()
                  for p, u in d.pages.items() if u.looks}
        seen: set = set()
        rows = [(READINGS_FILE, r) for r in self.record.readings] + \
               [(READS_FILE, r) for r in self.record.reads]
        for fname, r in rows:
            page = r.get("page")
            name = str(r.get("document") or "")
            if not name or not isinstance(page, int) or isinstance(page, bool):
                continue
            key = _doc_key(name)
            if (key, page) in logged:      # the activity log has its looks
                continue
            view = r.get("view")
            region = _box(r.get("region")) or _box(view)
            zoom = view == "zoom" or (region is not None
                                      and fname == READS_FILE)
            what = (key, page, "zoom" if zoom else "page",
                    region if zoom else None)
            if what in seen:               # the same look in both files
                continue
            seen.add(what)
            use = self.doc(key, os.path.basename(name)).use(page)
            data = _json_obj(r.get("result"))
            if zoom:
                use.mark("zoom", 0)
                use.looks.append(Look(0, "zoom", region, "", str(
                    r.get("tool") or ZOOM_TOOL), "", fname))
                continue
            tiled = "tiles" in str(view or "")
            use.mark("tiles" if tiled else "whole", 0)
            looks = _looks_of(PAGE_TOOL, {}, data, "") if data else \
                [("page", None, "")]
            for kind, box, label in looks:
                use.looks.append(Look(0, kind, box, label, str(
                    r.get("tool") or PAGE_TOOL), "", fname))

    def anchor(self, call: Call) -> List[Tuple[str, str]]:
        """``(href, words)`` for the pages a call reached."""
        out = []
        for key, page in call.where:
            d = self.docs.get(key)
            if d is None:
                continue
            if page is None:
                out.append((f"#doc-{d.index}", f"{d.name}: its pages"))
            else:
                out.append((f"#p-{d.index}-{page}",
                            f"page {page + 1} drawn"))
        return out


def _whole_text(record: RunRecord, rec: dict, key: str) -> Tuple[str, str]:
    """``(text, note)``: a payload whole, from its copy beside the log when
    the record cut it short; the note says what the reader is seeing."""
    text = rec.get(key)
    text = "" if text is None else (text if isinstance(text, str)
                                    else json.dumps(text, ensure_ascii=False,
                                                    default=str))
    if not rec.get("truncated") and not rec.get("text_truncated"):
        return text, ""
    full = rec.get("full") or rec.get("text_full")
    whole = mask_payload(rec.get("event"), key, record.sidecar(full))
    if whole is not None:
        return whole, f"whole copy from {full}"
    total = rec.get("chars")
    return text, (f"cut in the record at {_n(len(text))} of {_n(total)} "
                  f"characters" if total else "cut short in the record") + (
        f"; its whole copy {full} is missing" if full else "")


# ---------------------------------------------------------------------------
# PDF pages behind the boxes
# ---------------------------------------------------------------------------

def _fitz():
    try:
        import pymupdf as fitz  # type: ignore
        return fitz
    except Exception:  # noqa: BLE001
        try:
            import fitz  # type: ignore
            return fitz
        except Exception:  # noqa: BLE001
            return None


def find_pdfs(folders: List[str]) -> Dict[str, str]:
    """File name (case folded) -> path of every PDF under ``folders``."""
    out: Dict[str, str] = {}
    for top in folders:
        if not top or not os.path.isdir(top):
            continue
        for root, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d != SIDECAR_DIR]
            for f in files:
                if f.lower().endswith(".pdf"):
                    out.setdefault(f.lower(), os.path.join(root, f))
    return out


class PageImages:
    """Small JPEGs of PDF pages and each page's rect, made with PyMuPDF."""

    def __init__(self):
        self.fitz = _fitz()
        self.drawn = 0
        self.problems: List[str] = []
        self._open: Dict[str, Any] = {}

    def _doc(self, path: str):
        if path not in self._open:
            try:
                self._open[path] = self.fitz.open(path)
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"{os.path.basename(path)} could not be "
                                     f"opened ({type(exc).__name__})")
                self._open[path] = None
        return self._open[path]

    def page_count(self, path: str) -> Optional[int]:
        if self.fitz is None:
            return None
        doc = self._doc(path)
        return int(doc.page_count) if doc is not None else None

    def rect(self, path: str, page: int) -> Optional[tuple]:
        """``(x0, y0, width, height)`` of one page, in PDF points."""
        if self.fitz is None:
            return None
        doc = self._doc(path)
        if doc is None or not 0 <= page < doc.page_count:
            return None
        try:
            r = doc.load_page(page).rect
            return (float(r.x0), float(r.y0), float(r.width),
                    float(r.height))
        except Exception:  # noqa: BLE001
            return None

    def _jpeg(self, pix) -> bytes:
        try:
            return pix.tobytes("jpg", jpg_quality=JPEG_QUALITY)
        except Exception:  # noqa: BLE001 - an older PyMuPDF
            import io
            from PIL import Image
            img = Image.frombytes("RGB", (pix.width, pix.height),
                                  pix.samples)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=JPEG_QUALITY)
            return buf.getvalue()

    def draw(self, path: str, page: int, sizes: Tuple[int, ...]
             ) -> Dict[int, bytes]:
        """JPEGs of one page, one per long-side size in ``sizes`` (the
        largest rendered, the others scaled from it where PyMuPDF can);
        ``{}`` when the page cannot be drawn (said once in ``problems``)."""
        if self.fitz is None or not sizes:
            return {}
        doc = self._doc(path)
        if doc is None or not 0 <= page < doc.page_count:
            return {}
        out: Dict[int, bytes] = {}
        try:
            pg = doc.load_page(page)
            r = pg.rect
            top = max(sizes)
            scale = top / max(r.width, r.height)
            pix = pg.get_pixmap(matrix=self.fitz.Matrix(scale, scale),
                                alpha=False)
            out[top] = self._jpeg(pix)
            for px in sorted(set(sizes) - {top}, reverse=True):
                f = px / top
                w, h = max(1, round(pix.width * f)), max(1, round(
                    pix.height * f))
                try:
                    small = self.fitz.Pixmap(pix, w, h, None)
                except Exception:  # noqa: BLE001 - render it instead
                    s = px / max(r.width, r.height)
                    small = pg.get_pixmap(matrix=self.fitz.Matrix(s, s),
                                          alpha=False)
                out[px] = self._jpeg(small)
            self.drawn += 1
            return out
        except Exception as exc:  # noqa: BLE001
            self.problems.append(f"page {page + 1} of "
                                 f"{os.path.basename(path)} could not be "
                                 f"drawn ({type(exc).__name__})")
            return {}

    def close(self) -> None:
        for doc in self._open.values():
            try:
                if doc is not None:
                    doc.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fff;--fg:#1c1c1c;--muted:#5e5e5e;--line:#d5d5d2;--soft:#f3f3f0;
--paper:#fff;--zoom:#c0262d;--tile:#1f5fa8;--whole:#555;--warn:#8a5300}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e4e4e4;
--muted:#a3a3a3;--line:#3b3b3b;--soft:#222;--paper:#2a2a2a;--zoom:#ff6b6b;
--tile:#7fb0ea;--whole:#bdbdbd;--warn:#e3b45a}}
body{background:var(--bg);color:var(--fg);margin:0 auto;max-width:64rem;
padding:1rem 1rem 4rem;font:15px/1.45 system-ui,-apple-system,"Segoe UI",
sans-serif}
h1{font-size:1.45rem;margin:.2rem 0}h2{font-size:1.2rem;margin:2.2rem 0 .6rem;
border-bottom:1px solid var(--line);padding-bottom:.2rem}
h3{font-size:1.05rem;margin:1.4rem 0 .4rem}a{color:var(--tile)}
.muted,.note{color:var(--muted)}.warn{color:var(--warn)}
.holds{border:1px solid var(--line);padding:.4rem .6rem;margin:.6rem 0}
pre{white-space:pre-wrap;word-break:break-word;margin:.25rem 0;padding:.45rem;
background:var(--soft);font:12.5px/1.4 ui-monospace,Consolas,monospace}
table{border-collapse:collapse;margin:.3rem 0}
td,th{border-bottom:1px solid var(--line);padding:.2rem .6rem .2rem 0;
text-align:left;vertical-align:top}th{font-weight:600}
td.r{text-align:right;font-variant-numeric:tabular-nums}
details>summary{cursor:pointer;color:var(--muted)}
.toc ol{margin:.2rem 0;padding-left:1.4rem}
.step{border-left:3px solid var(--line);padding:.1rem 0 .1rem .7rem;
margin:.9rem 0}.sh{font-size:.92rem}.sh b{font-size:1rem}
.said{white-space:pre-wrap;margin:.3rem 0}
.call{margin:.5rem 0 .5rem .3rem;padding-left:.6rem;
border-left:1px dashed var(--line)}.no{font-weight:700}
.helper{margin:.5rem 0;padding:.2rem .6rem;border:1px solid var(--line)}
.helper>summary{color:var(--fg);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(3.6rem,1fr));
gap:3px;margin:.4rem 0}
.pg{display:block;border:1px solid var(--line);padding:.1rem .25rem;
text-decoration:none;color:var(--fg);min-height:2.4rem;font-size:.8rem}
.pg b{display:block;font-size:.85rem}.pg i{font-style:normal;
letter-spacing:.05em;font-family:ui-monospace,Consolas,monospace}
.pg.never{color:var(--muted);border-style:dotted}
.k-zoom{background:rgba(192,38,45,.15)}.k-tiles{background:rgba(31,95,168,.15)}
.k-whole{background:rgba(128,128,128,.2)}
.k-text,.k-contact,.k-text_empty,.k-sweep{background:rgba(128,128,128,.07)}
.counts span{margin-right:1rem;white-space:nowrap;display:inline-block}
.figs{display:flex;flex-wrap:wrap;gap:1.2rem}
.fig{break-inside:avoid;max-width:100%}.fig ol{font-size:.85rem;
padding-left:1.1rem;margin:.3rem 0;max-width:22rem}
svg .sheet{fill:var(--paper);stroke:var(--fg);stroke-width:1px}
svg rect.bz{fill:var(--zoom);fill-opacity:.1;stroke:var(--zoom);
stroke-width:2px}svg rect.bt{fill:none;stroke:var(--tile);stroke-width:1.5px}
svg rect.bw{fill:none;stroke:var(--whole);stroke-width:2px;
stroke-dasharray:6 3}svg text{paint-order:stroke;stroke:var(--bg);
stroke-linejoin:round}svg text.bz{fill:var(--zoom)}
svg text.bt{fill:var(--tile)}svg text.bw{fill:var(--whole)}
#viewer{position:fixed;top:0;left:0;right:0;bottom:0;z-index:10;
background:var(--bg);display:flex;flex-direction:column;
padding:.5rem 1rem;box-sizing:border-box}#viewer[hidden]{display:none}
body.vw-open{overflow:hidden}
.vw-bar{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;
border-bottom:1px solid var(--line);padding-bottom:.4rem}
.vw-bar b{margin-right:auto}.vw-keys{margin:.2rem 0;font-size:.8rem}
.vw-body{flex:1;display:flex;gap:1rem;min-height:0}
#vw-page{flex:1 1 62%;min-width:0;min-height:0;display:flex}
#vw-page svg{width:100%;height:100%}
#vw-side{flex:1 1 38%;overflow:auto;font-size:.9rem;min-height:0}
#vw-side pre{max-height:40vh;overflow:auto}#vw-side h4{margin:.8rem 0 .3rem}
.vb{cursor:pointer}.vb rect{pointer-events:all}
.vb.sel rect{stroke-width:4px}.vw-on .vb:not(.sel){opacity:.35}
.vw-looks li{cursor:pointer}.vw-looks li.sel{font-weight:700}
.vw-ref{background:none;border:0;padding:0;color:var(--tile);
text-decoration:underline;cursor:pointer;font:inherit}
.has-viewer .pg{cursor:pointer}
@media (max-width:50rem){.vw-body{flex-direction:column}
#vw-page{flex:0 0 55vh}}
@media print{body{max-width:none;font-size:11px}#tools{display:none}
#viewer{display:none!important}
.step,.call,.fig{break-inside:avoid-page}a{color:inherit}}
"""

_VIEWER_HTML = (
    '<div id="viewer" hidden tabindex="-1" role="dialog" aria-modal="true" '
    'aria-labelledby="vw-title"><div class="vw-bar"><b id="vw-title"></b>'
    '<span id="vw-where" class="muted"></span>'
    '<button type="button" data-go="pl">Previous page looked at</button>'
    '<button type="button" data-go="prev">Previous page</button>'
    '<button type="button" data-go="next">Next page</button>'
    '<button type="button" data-go="nl">Next page looked at</button>'
    '<button type="button" data-go="close">Close</button></div>'
    '<p class="vw-keys muted">The page viewer. Left and right arrow keys: '
    'the previous and next page; with Shift, the previous and next page '
    'looked at; Esc or the browser\'s Back closes it.</p>'
    '<div class="vw-body"><div id="vw-page"></div><div id="vw-side"></div>'
    '</div></div>')

#: The page viewer: reads the JSON block ``viewer-data`` of the same file and
#: draws one page at a time. No network, no library; the address's
#: ``#view-<document>-<PDF page>`` names the page open, so Back closes it.
_VIEWER_SCRIPT = r"""
(function(){
var data=document.getElementById('viewer-data'),V=document.getElementById('viewer');
if(!data||!V)return;
var D;try{D=JSON.parse(data.textContent);}catch(e){return;}
var pg=document.getElementById('vw-page'),side=document.getElementById('vw-side'),
title=document.getElementById('vw-title'),where=document.getElementById('vw-where');
var docs={},looked=[],cur=null,mine=false,before='',last=location.hash,savedY=0;
var ORD={page:0,tile:1,zoom:2},CLS={page:'bw',tile:'bt',zoom:'bz'},B={};
D.docs.forEach(function(d){docs[d.i]=d;
 for(var p=0;p<d.n;p++){var e=d.pages[p];if(e&&e.l)looked.push([d.i,p]);}});
['pl','prev','next','nl','close'].forEach(function(k){
 B[k]=V.querySelector('[data-go="'+k+'"]');});
function h(s){return String(s==null?'':s).replace(/&/g,'&amp;')
 .replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmt(x){return Number(x).toLocaleString('en');}
function ent(d,p){var x=docs[d];return (x&&x.pages[p])||{};}
function img(d,p,e){if(e.i)return 'data:image/jpeg;base64,'+e.i;
 var f=document.querySelector('#p-'+d+'-'+p+' image');
 return f?f.getAttribute('href'):null;}
function who(n){var c=D.calls[n];if(!c)return '';
 return c.who+(c.step?', step '+c.step:'')+
  (D.multi&&c.turn!=null?', turn '+c.turn:'');}
function kw(k,lab,b){if(k==='page')return 'whole page';
 if(k==='tile')return 'tile '+lab;
 return 'zoom'+(b?' on x '+Math.round(b[0])+'-'+Math.round(b[2])+', y '+
  Math.round(b[1])+'-'+Math.round(b[3])+' pt':'');}
function tip(l){var s=(l[0]?'#'+l[0]+' ':'')+kw(l[1],l[3],l[2]);
 if(l[4])s+=', '+l[4];if(l[0]&&D.calls[l[0]])s+=', '+who(l[0]);
 if(l[5])s+=', from '+l[5];return s;}
function svg(e,r,src){var x=r[0],y=r[1],w=r[2],hh=r[3],fs=Math.max(w,hh)/55;
 var s='<svg viewBox="'+[x,y,w,hh].join(' ')+'" role="img" '+
  'aria-label="the page with a box for every look">'+
  '<rect class="sheet" x="'+x+'" y="'+y+'" width="'+w+'" height="'+hh+
  '" vector-effect="non-scaling-stroke"/>';
 if(src)s+='<image href="'+src+'" x="'+x+'" y="'+y+'" width="'+w+
  '" height="'+hh+'" preserveAspectRatio="none"/>';
 var seen={},ls=(e.l||[]).slice().sort(function(a,b){
  return ORD[a[1]]-ORD[b[1]]||a[0]-b[0];});
 ls.forEach(function(l){var b=l[2];if(!b)return;var c=CLS[l[1]],
  k=Math.round(b[0]/fs)+','+Math.round(b[1]/fs),i=seen[k]||0;seen[k]=i+1;
  var lab=(l[0]?'#'+l[0]:'')+(l[1]==='tile'&&l[3]?' '+l[3]:'');
  s+='<g class="vb"'+(l[0]?' data-n="'+l[0]+'" tabindex="0"':'')+
   '><title>'+h(tip(l))+'</title><rect class="'+c+'" x="'+b[0]+'" y="'+b[1]+
   '" width="'+(b[2]-b[0])+'" height="'+(b[3]-b[1])+
   '" vector-effect="non-scaling-stroke"/>';
  if(lab)s+='<text class="'+c+'" x="'+(b[0]+fs*0.3)+'" y="'+
   (b[1]+fs*(1.1+1.2*i))+'" font-size="'+fs+'" stroke-width="'+fs*0.3+
   '">'+h(lab)+'</text>';
  s+='</g>';});
 return s+'</svg>';}
function ref(n){return '<button type="button" class="vw-ref" data-n="'+n+
 '">#'+n+'</button>';}
function sideHtml(e,src){var o=[],m=e.m||{},g=[],by={};
 if(!src)o.push('<p class="note">'+h(D.why[e.x]||D.why.none)+'</p>');
 if(e.rf)o.push('<p class="note">'+h(D.rf[e.rf])+'</p>');
 var ks=D.marks.filter(function(k){return m.hasOwnProperty(k[0]);});
 if(ks.length){o.push('<h4>How it was taken in</h4><ul>');
  ks.forEach(function(k){var ns=m[k[0]];o.push('<li>'+h(k[1])+(ns.length?
   ': '+ns.map(ref).join(', '):" (from the vision tools' own records)")+
   '</li>');});
  o.push('</ul>');}
 else o.push('<p class="muted">'+h(D.empty)+'</p>');
 (e.l||[]).forEach(function(l){var key=l[0]+'|'+l[5];
  if(!by[key]){by[key]={n:l[0],tool:l[4],src:l[5],parts:[]};g.push(by[key]);}
  by[key].parts.push(l);});
 if(g.length){o.push('<h4>Looks on this page</h4><ol class="vw-looks">');
  g.forEach(function(x){
   var tiles=x.parts.filter(function(l){return l[1]==='tile';})
    .map(function(l){return l[3];}),
   main=x.parts.filter(function(l){return l[1]!=='tile';}).map(function(l){
    return kw(l[1],l[3],l[2])+(l[2]?'':' (no view box in the record)');}),
   w=main.join(', ')+(tiles.length?(main.length?', and ':'')+tiles.length+
    ' tiles ('+tiles.join(', ')+')':'');
   o.push('<li'+(x.n?' data-n="'+x.n+'" tabindex="0"':'')+'><b>'+
    (x.n?'#'+x.n:'from '+h(x.src))+'</b> '+h(w)+(x.tool?', '+h(x.tool):'')+
    (x.n&&D.calls[x.n]?', '+h(who(x.n)):'')+'</li>');});
  o.push('</ol>');}
 o.push('<div id="vw-call">'+(g.length||ks.length?'<p class="muted">Choose '+
  'a box or a numbered call to see what was asked and what came back.</p>'
  :'')+'</div>');
 return o.join('');}
function near(d,p,dir){var best=null;
 for(var i=0;i<looked.length;i++){var q=looked[i],c=(q[0]-d)||(q[1]-p);
  if(dir>0&&c>0)return q;if(dir<0&&c<0)best=q;}
 return dir<0?best:null;}
function show(d,p){var doc=docs[d];if(!doc)return;
 p=Math.max(0,Math.min(p,doc.n-1));cur=[d,p];
 var e=ent(d,p),r=e.r||doc.r||null,src=img(d,p,e);
 title.textContent=doc.name+', PDF page '+(p+1)+' of '+doc.n;
 var k=-1;for(var i=0;i<looked.length;i++){
  if(looked[i][0]===d&&looked[i][1]===p){k=i;break;}}
 where.textContent=k<0?'not looked at':'page looked at '+(k+1)+' of '+
  looked.length;
 pg.classList.remove('vw-on');
 pg.innerHTML=r?svg(e,r,src):'<p class="note">No page size in the record: '+
  'nothing to draw.</p>';
 side.innerHTML=sideHtml(e,src);
 B.prev.disabled=p<=0;B.next.disabled=p>=doc.n-1;
 B.pl.disabled=!near(d,p,-1);B.nl.disabled=!near(d,p,1);}
function select(n){var c=D.calls[n],box=document.getElementById('vw-call');
 if(!box)return;var all=V.querySelectorAll('[data-n]');
 for(var i=0;i<all.length;i++)all[i].classList.toggle('sel',
  all[i].getAttribute('data-n')===String(n));
 pg.classList.add('vw-on');
 if(!c){box.innerHTML='<p class="note">Call #'+n+' is not in the record.</p>';
  return;}
 var o=['<h4>#'+n+' '+h(c.tool)+'</h4><p>'+h(who(n))+'</p>'];
 if(c.status)o.push('<p class="warn">'+h(c.status)+'</p>');
 o.push('<b>What was asked</b>'+(c.asked?(c.asked_n>c.asked.length?
  ' <span class="muted">its first '+fmt(c.asked.length)+' characters of '+
  fmt(c.asked_n)+'</span>':'')+'<pre>'+h(c.asked)+'</pre>':
  '<p class="note">No arguments in the record.</p>'));
 o.push('<b>What came back</b>'+(c.back?' <span class="muted">'+
  (c.back_n>c.back.length?'its first '+fmt(c.back.length)+
  ' characters of '+fmt(c.back_n):fmt(c.back_n)+' characters')+
  '</span><pre>'+h(c.back)+'</pre>':'<p class="note">Nothing in the '+
  'record.</p>'));
 if(c.differs)o.push('<p class="note">The model received something else '+
  'than this result: see part 2.</p>');
 if(D.steps)o.push('<p><a href="#c'+n+'">This call in part 2</a></p>');
 box.innerHTML=o.join('');}
function parse(s){var m=String(s||'').split('-');
 if(m.length!==3||m[0]!=='#view')return null;
 var d=parseInt(m[1],10),p=parseInt(m[2],10);
 return isNaN(d)||isNaN(p)?null:[d,p-1];}
function route(){var t=parse(location.hash);
 if(t&&docs[t[0]]){if(V.hidden){before=last;savedY=window.pageYOffset;
   V.hidden=false;document.body.classList.add('vw-open');}
  show(t[0],t[1]);V.focus();}
 else if(!V.hidden){V.hidden=true;mine=false;
  document.body.classList.remove('vw-open');
  if(location.hash===before)window.scrollTo(0,savedY);
  else{var a=document.getElementById(location.hash.slice(1));
   if(a)a.scrollIntoView();}}
 last=location.hash;}
function go(q){if(q)location.replace('#view-'+q[0]+'-'+(q[1]+1));}
function open(d,p){mine=true;location.hash='view-'+d+'-'+(p+1);}
function close(){if(mine){history.back();return;}
 location.replace(before&&!parse(before)?before:'#doc-'+(cur?cur[0]:''));}
function act(k){if(k==='close'){close();return;}if(!cur)return;
 var d=cur[0],p=cur[1];
 if(k==='prev'&&p>0)go([d,p-1]);
 if(k==='next'&&p<docs[d].n-1)go([d,p+1]);
 if(k==='pl')go(near(d,p,-1));if(k==='nl')go(near(d,p,1));}
V.addEventListener('click',function(ev){var t=ev.target;if(!t.closest)return;
 var b=t.closest('[data-go]');if(b){act(b.getAttribute('data-go'));return;}
 var n=t.closest('[data-n]');if(n)select(+n.getAttribute('data-n'));});
document.addEventListener('keydown',function(ev){var t=ev.target;
 if(V.hidden){if((ev.key==='Enter'||ev.key===' ')&&t.classList&&
   t.classList.contains('pg')&&t.tagName!=='A'){ev.preventDefault();
   t.click();}return;}
 if(ev.altKey||ev.ctrlKey||ev.metaKey)return;
 if(ev.key==='Escape'){ev.preventDefault();act('close');return;}
 if((ev.key==='Enter'||ev.key===' ')&&t.getAttribute&&
   t.getAttribute('data-n')&&t.tagName!=='BUTTON'){ev.preventDefault();
  select(+t.getAttribute('data-n'));return;}
 if((ev.key==='ArrowLeft'||ev.key==='ArrowRight')&&t.tagName!=='PRE'){
  ev.preventDefault();var f=ev.key==='ArrowRight';
  act(ev.shiftKey?(f?'nl':'pl'):(f?'next':'prev'));}});
document.addEventListener('click',function(ev){
 if(!V.hidden||!ev.target.closest)return;var t=ev.target,a=t.closest('.pg');
 if(a){var g=a.closest('[data-i]');if(!g)return;ev.preventDefault();
  open(+g.getAttribute('data-i'),+a.getAttribute('data-page')-1);return;}
 var o=t.closest('a.vw-open');
 if(o){ev.preventDefault();var q=parse(o.getAttribute('href'));
  if(q)open(q[0],q[1]);}});
var sp=document.querySelectorAll('span.pg');
for(var i=0;i<sp.length;i++){sp[i].setAttribute('tabindex','0');
 sp[i].setAttribute('role','button');}
var vo=document.querySelectorAll('a.vw-open');
for(i=0;i<vo.length;i++)vo[i].hidden=false;
document.documentElement.className+=' has-viewer';
window.addEventListener('hashchange',route);route();})();
"""


def _script_json(obj: Any) -> str:
    """``obj`` as JSON that is safe inside a ``<script>`` element: ``<``,
    ``>`` and ``&`` written as JSON escapes (a record's text can hold
    ``</script>``)."""
    esc = chr(92) + "u00"
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                       default=str)
            .replace("<", esc + "3c").replace(">", esc + "3e")
            .replace("&", esc + "26"))

_SCRIPT = """
(function(){var t=document.getElementById('tools');if(!t)return;
t.innerHTML='<button type="button" id="xa">Expand all</button> '+
'<button type="button" id="ca">Collapse all</button>';
function s(o){var d=document.getElementsByTagName('details');
for(var i=0;i<d.length;i++){d[i].open=o;}}
document.getElementById('xa').onclick=function(){s(true);};
document.getElementById('ca').onclick=function(){s(false);};
window.addEventListener('beforeprint',function(){s(true);});})();
"""


def _fold(title: str, body: str, open_: bool = False, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    o = " open" if open_ else ""
    return f"<details{c}{o}><summary>{title}</summary>{body}</details>"


def _payload(label: str, text: str, note: str = "") -> str:
    """A payload in a ``<pre>``, open when short, folded with its size when
    long; cut at :data:`MAX_SHOWN_CHARS` with the cut said."""
    shown = _pretty(text)
    cut = ""
    if len(shown) > MAX_SHOWN_CHARS:
        shown = shown[:MAX_SHOWN_CHARS]
        cut = (f'<p class="note">Shown up to {_n(MAX_SHOWN_CHARS)} '
               f'characters of {_n(len(text))}.</p>')
    extra = f"; {_e(note)}" if note else ""
    body = f"<pre>{_e(shown)}</pre>{cut}"
    if len(text) <= OPEN_CHARS and text.count("\n") < 12:
        return f'<div class="note">{label}{extra}</div>{body}'
    return _fold(f"{label}: {_n(len(text))} characters{extra}", body)


class Report:
    """Builds the HTML of one run's report."""

    def __init__(self, folder: str, documents: Optional[str] = None):
        self.record = RunRecord(folder)
        self.documents = documents
        self.turns = build_turns(self.record.activity)
        self.pages = PageMap(self.record, self.turns)
        self.images = PageImages()
        self.pdfs = find_pdfs([self.record.folder] + (
            [documents] if documents else []))
        for doc in self.pages.docs.values():
            path = self.pdfs.get(doc.key)
            if path is None and not doc.key.endswith(".pdf"):
                path = self.pdfs.get(os.path.splitext(doc.key)[0] + ".pdf")
            doc.pdf = path
            if doc.n_pages is None and path:
                n = self.images.page_count(path)
                if n:
                    doc.n_pages, doc.n_from = n, "the PDF"
            if doc.n_pages is None and doc.pages:
                doc.n_pages, doc.n_from = max(doc.pages) + 1, N_FROM_LAST
        self._drawn: Optional[Dict[Tuple[int, int], Dict[int, str]]] = None
        self._failed: set = set()
        self._rects: Dict[Tuple[int, int], Optional[tuple]] = {}
        self.image_stats: Dict[str, int] = {}
        self.viewer_problem = ""

    # -- page images, drawn once within the caps --------------------------
    @staticmethod
    def _n_pages(doc: Document) -> int:
        return max(doc.n_pages or 0, (max(doc.pages) + 1) if doc.pages
                   else 0)

    def _plan_images(self) -> Dict[Tuple[int, int], Dict[int, str]]:
        """``(document index, page) -> {pixels: JPEG as base64}``: each page
        looked at small and, while the large ones stay under their cap,
        large; then the other pages small, those taken in some other way
        first, under their cap (the constants above say why)."""
        if self._drawn is not None:
            return self._drawn
        drawn: Dict[Tuple[int, int], Dict[int, str]] = {}
        self._drawn = drawn
        st = self.image_stats = dict.fromkeys(
            ("looked", "looked_large", "looked_small_only", "looked_none",
             "looked_over", "other", "other_drawn", "other_none",
             "other_failed", "bytes"), 0)
        if self.images.fitz is None:
            return drawn
        docs = [d for d in self._docs() if d.pdf]
        large = other = 0

        def keep(d: Document, p: int, got: Dict[int, bytes]) -> None:
            drawn[(d.index, p)] = {
                px: base64.b64encode(b).decode("ascii")
                for px, b in got.items()}
            # as the images sit in the file: base64
            st["bytes"] += sum(len(s) for s in drawn[(d.index, p)].values())

        for d in docs:
            for p, use in sorted(d.pages.items()):
                if not use.looks:
                    continue
                st["looked"] += 1
                if st["looked"] > MAX_PAGE_IMAGES \
                        or self.images.drawn >= MAX_PAGES_DRAWN:
                    st["looked_none"] += 1
                    st["looked_over"] += 1
                    continue
                sizes = (PAGE_IMAGE_PX, VIEW_IMAGE_PX) \
                    if large < MAX_VIEW_IMAGE_BYTES else (PAGE_IMAGE_PX,)
                got = self.images.draw(d.pdf, p, sizes)
                if not got:
                    st["looked_none"] += 1
                    self._failed.add((d.index, p))
                    continue
                if VIEW_IMAGE_PX in got:
                    large += len(got[VIEW_IMAGE_PX])
                    st["looked_large"] += 1
                else:
                    st["looked_small_only"] += 1
                keep(d, p, got)
        rest = []
        for d in docs:
            for p in range(self._n_pages(d)):
                use = d.pages.get(p)
                if use is not None and use.looks:
                    continue
                rest.append((0 if use is not None and use.marks else 1,
                             d.index, p, d))
        rest.sort(key=lambda r: r[:3])
        for _first, _i, p, d in rest:
            st["other"] += 1
            if other >= MAX_OTHER_IMAGE_BYTES \
                    or self.images.drawn >= MAX_PAGES_DRAWN:
                st["other_none"] += 1
                continue
            got = self.images.draw(d.pdf, p, (PAGE_IMAGE_PX,))
            if not got:
                st["other_none"] += 1
                st["other_failed"] += 1
                self._failed.add((d.index, p))
                continue
            other += len(got[PAGE_IMAGE_PX])
            st["other_drawn"] += 1
            keep(d, p, got)
        return drawn

    def _image_note(self) -> str:
        """What the page images hold and what the caps left out."""
        st = self.image_stats
        if not st or not (st["looked"] or st["other"]):
            return ""
        bits = []
        if st["looked"]:
            s = (f"pages looked at: {_n(st['looked'])}; drawn at "
                 f"{VIEW_IMAGE_PX} pixels on the long side for the page "
                 f"viewer and at {PAGE_IMAGE_PX} in part 4: "
                 f"{_n(st['looked_large'])}")
            if st["looked_small_only"]:
                s += (f"; only at {PAGE_IMAGE_PX}: "
                      f"{_n(st['looked_small_only'])} (the large images stop "
                      f"at about {_size(_b64_len(MAX_VIEW_IMAGE_BYTES))} in "
                      "this file)")
            if st["looked_over"]:
                s += (f"; not drawn: {_n(st['looked_over'])} (at most "
                      f"{_n(MAX_PAGE_IMAGES)} pages looked at are drawn, "
                      f"and {_n(MAX_PAGES_DRAWN)} pages in all)")
            bits.append(s)
        if st["other"]:
            s = (f"other pages: {_n(st['other'])}; drawn at {PAGE_IMAGE_PX} "
                 f"pixels for the page viewer: {_n(st['other_drawn'])} "
                 "(pages taken in some other way, such as read as text or "
                 "shown on a contact sheet, are drawn first)")
            capped = st["other_none"] - st["other_failed"]
            if capped > 0:
                s += (f"; not drawn: {_n(capped)} (their images stop at "
                      f"about {_size(_b64_len(MAX_OTHER_IMAGE_BYTES))} in "
                      f"this file, and {_n(MAX_PAGES_DRAWN)} pages are drawn "
                      "in all)")
            bits.append(s)
        return ("Page images. " + ". ".join(b[0].upper() + b[1:]
                                            for b in bits)
                + f". Images in all, as they sit in this file: "
                f"{_size(st['bytes'])}.")

    def _page_rect(self, doc: Document, page: int, use: Optional[PageUse]
                   ) -> Tuple[Optional[tuple], str]:
        """``((x0, y0, width, height), from)``: the PDF page's own size
        (``"pdf"``), else the whole-page view a look returned
        (``"view"``), else a rect round the page's boxes (``"boxes"``)."""
        key = (doc.index, page)
        if doc.pdf:
            if key not in self._rects:
                self._rects[key] = self.images.rect(doc.pdf, page)
            if self._rects[key]:
                return self._rects[key], "pdf"
        looks = use.looks if use is not None else []
        whole = [lk.box for lk in looks if lk.kind == "page" and lk.box]
        boxes = [lk.box for lk in looks if lk.box]
        if whole:
            b = max(whole, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            return (b[0], b[1], b[2] - b[0], b[3] - b[1]), "view"
        if boxes:
            return (0.0, 0.0, max(b[2] for b in boxes),
                    max(b[3] for b in boxes)), "boxes"
        return None, ""

    # -- the page viewer's data -------------------------------------------
    def _why_no_image(self, doc: Document, page: int) -> str:
        if self.images.fitz is None:
            return "nofitz"
        if not doc.pdf:
            return "nopdf"
        if (doc.index, page) in self._failed:
            return "fail"
        return "cap"

    def _view_calls(self, numbers: set) -> Dict[str, dict]:
        """What the viewer shows of each call a page names: who made it and
        in which step, what was asked and the start of what came back."""
        places = _places(self.turns)
        by_n = {c.number: c for c in _all_calls(self.turns)}
        out: Dict[str, dict] = {}
        for n in sorted(numbers):
            call = by_n.get(n)
            if call is None:
                continue
            place = places.get(n, {})
            asked = "" if call.args in (None, "", {}) else (
                call.args if isinstance(call.args, str) else json.dumps(
                    call.args, ensure_ascii=False, default=str))
            status = ""
            if call.refused:
                back = _whole_text(self.record, call.delivered[0], "text")[0]
                status = "refused: no tool ran; this is what the model was told"
            elif call.end is None:
                back, status = "", "no result in the record"
            elif call.failed:
                back = _whole_text(self.record, call.end, "error")[0]
                status = "the tool raised an error"
            else:
                back = _whole_text(self.record, call.end, "result")[0]
            asked, back = _pretty(asked), _pretty(back)
            out[str(n)] = {
                "tool": call.name, "who": place.get("who", ""),
                "step": place.get("step"), "turn": place.get("turn"),
                "asked": asked[:VIEW_ASKED_CHARS], "asked_n": len(asked),
                "back": back[:VIEW_BACK_CHARS], "back_n": len(back),
                "status": status,
                "differs": bool(call.delivered) and not call.refused}
        return out

    def _viewer_data(self) -> dict:
        """Everything the page viewer shows, for its JSON block: per
        document and page, the page's size, how it was taken in, its looks
        and its image (a page looked at keeps its small image in part 4,
        which the viewer takes when it has no large one)."""
        drawn = self._plan_images()
        docs, wanted = [], set()
        for d in self._docs():
            n = self._n_pages(d)
            if not n:
                continue
            rects: Dict[tuple, int] = {}
            pages: Dict[str, dict] = {}
            for p in range(n):
                use = d.pages.get(p)
                e: Dict[str, Any] = {}
                rect, how = self._page_rect(d, p, use)
                if rect:
                    e["r"] = [round(v, 2) for v in rect]
                    if how == "pdf":
                        rects[tuple(e["r"])] = rects.get(tuple(e["r"]), 0) + 1
                    else:
                        e["rf"] = how
                if use is not None and use.marks:
                    e["m"] = {k: list(v) for k, v in use.marks.items()}
                    for v in use.marks.values():
                        wanted.update(v)
                if use is not None and use.looks:
                    e["l"] = [[lk.number, lk.kind,
                               [round(v, 2) for v in lk.box] if lk.box
                               else None, lk.label, lk.tool, lk.source]
                              for lk in use.looks]
                    wanted.update(lk.number for lk in use.looks if lk.number)
                imgs = drawn.get((d.index, p)) or {}
                looked = use is not None and bool(use.looks)
                img = imgs.get(VIEW_IMAGE_PX) if looked else \
                    imgs.get(PAGE_IMAGE_PX)
                if img:
                    e["i"] = img
                elif not imgs:
                    e["x"] = self._why_no_image(d, p)
                pages[str(p)] = e
            common = max(rects, key=rects.get) if rects else None
            if common:
                for e in pages.values():
                    if e.get("r") == list(common) and "rf" not in e:
                        del e["r"]
            docs.append({"i": d.index, "name": d.name, "n": n,
                         "r": list(common) if common else None,
                         "pages": pages})
        where = ("the run folder" + (" or under --documents" if
                                     self.documents else
                                     "; name its folder with --documents"))
        return {
            "docs": docs, "calls": self._view_calls(wanted),
            "marks": [[m, words] for m, _l, _c, words in MARKS],
            "multi": len(self.turns) > 1,
            "steps": self.record.has_activity,
            "empty": ("Not taken in: no tool read or looked at this page."
                      if self.record.has_activity else
                      f"No look recorded (there is no {ACTIVITY_FILE})."),
            "why": {
                "nofitz": "No image: PyMuPDF is not installed where the "
                          "report was made.",
                "nopdf": f"No image: the document's PDF was not found in "
                         f"{where}.",
                "fail": "No image: this page could not be drawn from its "
                        "PDF.",
                "cap": "No image: the report's cap on page images was "
                       "reached. " + self._image_note(),
                "none": "No image of this page in the report."},
            "rf": {"view": "The outline is the whole-page view a look "
                           "returned; the page itself is not drawn.",
                   "boxes": "The page size is not in the record: the "
                            "outline is drawn round its boxes."},
        }

    # -- the page ---------------------------------------------------------
    def html(self) -> str:
        try:
            try:
                viewer = (_VIEWER_HTML + '<script type="application/json" '
                          'id="viewer-data">'
                          + _script_json(self._viewer_data()) + "</script>")
            except Exception as exc:  # noqa: BLE001 - the report without it
                viewer = ""
                self.viewer_problem = (f"{type(exc).__name__}: "
                                       f"{self._plain(str(exc))}")
            sections = [
                ("run", "1. The run", self._section(self._run)),
                ("steps", "2. Step by step", self._section(self._steps)),
                ("taken", "3. What was taken in",
                 self._section(self._taken_in)),
                ("where", "4. Where they looked",
                 self._section(self._where)),
                ("record", "5. What the record holds",
                 self._section(self._holds)),
            ]
        finally:
            self.images.close()
        name = _folder_name(self.record.folder)
        toc = ["<nav class='toc'><b>Contents</b><ol>"]
        for anchor, title, _body in sections:
            sub = ""
            if anchor == "steps" and len(self.turns) > 1:
                sub = "<ol>" + "".join(
                    f'<li><a href="#turn-{_e(t.number)}">Turn '
                    f'{_e(t.number)}</a></li>' for t in self.turns) + "</ol>"
            if anchor == "taken" and self.pages.docs:
                sub = "<ol>" + "".join(
                    f'<li><a href="#doc-{d.index}">{_e(d.name)}</a></li>'
                    for d in self._docs()) + "</ol>"
            toc.append(f'<li><a href="#{anchor}">{_e(title[3:])}</a>{sub}'
                       f'</li>')
        toc.append("</ol></nav>")
        banner = ""
        if not self.record.has_activity:
            banner = ('<p class="warn"><b>This record has no '
                      f'{ACTIVITY_FILE}</b>: the step list, the totals and '
                      'most of the page grid cannot be made. See part 5.</p>')
        body = "".join(f'<section id="{a}"><h2>{_e(t)}</h2>{b}</section>'
                       for a, t, b in sections)
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,"
            "initial-scale=1'>"
            "<meta name='color-scheme' content='light dark'>"
            f"<title>Run report: {_e(name)}</title><style>{_CSS}</style>"
            "</head><body>"
            f"<h1>Run report: {_e(name)}</h1>"
            f"<p class='holds'><b>{_e(CONTENT_NOTE)}</b></p>"
            f"<p class='muted'>Made {_e(_when(time.time()))} from the record "
            f"in the folder {_e(name)}, {_e(self.record.kind())}. Pages "
            "are numbered as a PDF viewer shows them, from 1.</p>"
            f"{banner}<div id='tools'></div>{''.join(toc)}{body}{viewer}"
            f"<script>{_SCRIPT}{_VIEWER_SCRIPT if viewer else ''}</script>"
            "</body></html>")

    def _section(self, fn) -> str:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - say it, never a traceback
            return (f'<p class="warn">This part could not be made from the '
                    f'record: {_e(type(exc).__name__)}: '
                    f'{_e(self._plain(str(exc)))}</p>')

    def _plain(self, text: str) -> str:
        """``text`` with the run folder's and the documents folder's paths
        cut to their own names (an error line can carry a path)."""
        for top in (self.record.folder, self.documents):
            if not top:
                continue
            full = os.path.abspath(top)
            for form in sorted({full, os.path.realpath(full),
                                full.replace("\\", "/"),
                                os.path.realpath(full).replace("\\", "/")},
                               key=len, reverse=True):
                text = text.replace(form, _folder_name(full))
        return text

    def _docs(self) -> List[Document]:
        return sorted(self.pages.docs.values(), key=lambda d: d.index)

    # -- 1. the run -------------------------------------------------------
    def _run(self) -> str:
        rec = self.record
        out = ['<table class="kv">',
               f"<tr><th>Record</th><td>{_e(rec.kind())}</td></tr>"]
        questions = rec.questions()
        asked = []
        for t in self.turns:
            q = questions.get(t.number) if isinstance(t.number, int) else None
            asked.append((t.number, q or _start_prompt(t)))
        if not asked:
            asked = sorted(questions.items())
        if asked:
            cells = []
            for number, q in asked:
                lab = f"Turn {_e(number)}: " if len(asked) > 1 else ""
                cells.append(f"<div>{lab}{_short_or_fold(q)}</div>")
            out.append(f"<tr><th>Asked</th><td>{''.join(cells)}</td></tr>")
        models: Dict[str, set] = {}
        for t in self.turns:
            if t.start and t.start.get("model"):
                models.setdefault("main agent", set()).add(
                    str(t.start["model"]))
            for who, names in t.models.items():
                models.setdefault(who, set()).update(names)
        if models:
            rank = {"main agent": 0, "helpers": 1, "vision calls": 2}
            text = "; ".join(f"{who}: {', '.join(sorted(n))}"
                             for who, n in sorted(
                                 models.items(),
                                 key=lambda kv: rank.get(kv[0], 3)))
            out.append(f"<tr><th>Model</th><td>{_e(text)}</td></tr>")
        when = next((t.start.get("ts") for t in self.turns
                     if t.start and t.start.get("ts")), None)
        when_text = _when(when) if when else _when_any(
            (rec.run or {}).get("started") or (rec.meta or {}).get(
                "created"))
        if when_text:
            out.append(f"<tr><th>When</th><td>{_e(when_text)}</td></tr>")
        out.append("</table>")
        out.append("<h3>Totals</h3>" + self._totals())
        out.append("<h3>Files the run delivered</h3>" + self._files())
        out.append("<h3>Notes the record carries</h3>" + self._notes())
        return "".join(out)

    def _totals(self) -> str:
        if not self.record.has_activity:
            run = self.record.run or {}
            if not run:
                return "<p class='note'>No activity log: no totals.</p>"
            rows = [("Model calls", run.get("model_calls")),
                    ("Tokens in", run.get("tokens_in")),
                    ("Tokens out", run.get("tokens_out")),
                    ("Wall time", _secs(run.get("seconds")))]
            return ("<p class='note'>From run.json only (no activity "
                    "log).</p><table>" + "".join(
                        f"<tr><th>{_e(k)}</th><td class='r'>{_e(_n(v))}"
                        f"</td></tr>" for k, v in rows if v not in (None, ""))
                    + "</table>")
        agent_calls = vision = helpers = 0
        tin = tout = cache = vin = vout = 0
        known_in = known_out = False
        tools = failed = refused = 0
        unended_agent = unended_vision = 0
        wall = 0.0
        wall_known = False
        for t in self.turns:
            for actor in [t.primary] + t.helpers:
                for step in actor.steps:
                    if step.model is not None and \
                            step.model.get("event") == "model_end":
                        agent_calls += 1
                        u = _usage(step.model)
                        if u["in"] is not None:
                            tin += u["in"]
                            known_in = True
                        if u["out"] is not None:
                            tout += u["out"]
                            known_out = True
                        cache += u["cache"] or 0
                    for call in step.calls:
                        if call.refused:
                            refused += 1
                            continue
                        tools += 1
                        failed += call.failed
                        helpers += call.name == TASK_TOOL
            for side in t.side.values():
                end = side.get("end")
                if end is None or end.get("event") != "model_end":
                    unended_vision += end is None
                    continue
                vision += 1
                u = _usage(end)
                vin += u["in"] or 0
                vout += u["out"] or 0
                if u["in"] is not None:
                    known_in = True
                if u["out"] is not None:
                    known_out = True
            unended_agent += len(t.agent_starts - t.agent_ends)
            if t.end and t.end.get("duration_s") is not None:
                try:
                    wall += float(t.end["duration_s"])
                    wall_known = True
                except (TypeError, ValueError):
                    pass
        if not wall_known:
            if (self.record.run or {}).get("seconds") is not None:
                wall, wall_known = self.record.run["seconds"], True
            else:
                stamps = [r.get("ts") for r in self.record.activity
                          if isinstance(r.get("ts"), (int, float))]
                if len(stamps) > 1:
                    wall, wall_known = max(stamps) - min(stamps), True
        rows = [
            ("turns", "Turns", _n(len(self.turns)), ""),
            ("model_calls", "Model calls by the agents", _n(agent_calls),
             "the main agent's and its helpers' own steps"),
            ("vision_calls", "Vision calls", _n(vision),
             "one image each, made inside a look tool"),
            ("tokens_in", "Tokens in", _n(tin + vin) if known_in
             else "not recorded",
             (f"agents {_n(tin)} ({_n(cache)} read from cache), vision "
              f"{_n(vin)}") if known_in else ""),
            ("tokens_out", "Tokens out", _n(tout + vout) if known_out
             else "not recorded",
             f"agents {_n(tout)}, vision {_n(vout)}" if known_out else ""),
            ("tool_calls", "Tool calls", _n(tools),
             f"{_n(failed)} ended in an error" + (
                 f"; {_n(refused)} more refused with no tool run"
                 if refused else "")),
            ("helpers", "Helpers started", _n(helpers), "task calls"),
            ("wall", "Wall time", _secs(wall) if wall_known
             else "not recorded",
             "the turns' own durations" if any(
                 t.end and t.end.get("duration_s") is not None
                 for t in self.turns) else ""),
        ]
        if unended_agent or unended_vision:
            rows.append(("unended", "Model calls with no end recorded",
                         _n(unended_agent + unended_vision),
                         f"agents {_n(unended_agent)}, vision "
                         f"{_n(unended_vision)}"))
        images = self._image_total()
        if images is not None:
            rows.insert(3, images)
        return "<table>" + "".join(
            f'<tr data-total="{k}"><th>{_e(lab)}</th><td class="r">'
            f'{_e(v)}</td><td class="muted">{_e(more)}</td></tr>'
            for k, lab, v, more in rows) + "</table>" + (
            f"<p class='note'>{_e(IMAGE_COUNTS_NOTE)}</p>"
            if images is not None else "")

    def _image_total(self) -> Optional[tuple]:
        """The totals row of the turns' own image calls, or ``None`` when no
        turn's record keeps them (a record written before 1.2.0)."""
        kept = [f for f in (_image_facts(t.end) for t in self.turns)
                if f is not None]
        if not kept:
            return None
        more = ["images sent to the vision model for a separate reading, as "
                "the turns counted them"]
        limits = sorted({f[1] for f in kept if f[1]})
        if limits:
            more.append("ceiling " + ", ".join(_n(x) for x in limits)
                        + " a turn")
        refused = sum(f[2] or 0 for f in kept)
        if refused:
            more.append(f"{_n(refused)} more refused at the ceiling")
        if len(kept) < len(self.turns):
            more.append(f"counted in {len(kept)} of {len(self.turns)} "
                        "turns' records")
        return ("image_calls", "Image calls", _n(sum(f[0] for f in kept)),
                "; ".join(more))

    def _files(self) -> str:
        rows: List[dict] = []
        for t in self.turns:
            for f in t.files:
                rows.append({"path": f.get("path"), "bytes": f.get("bytes"),
                             "how": f.get("how"), "turn": t.number})
        seen = {str(r["path"]) for r in rows}
        for f in self.record.delivered():
            if f["path"] not in seen:
                seen.add(f["path"])
                rows.append(f)
        if not rows:
            return "<p class='note'>None recorded.</p>"
        return ("<table><tr><th>File</th><th>Size</th><th>How</th>"
                "<th>Turn</th></tr>" + "".join(
                    f"<tr><td>{_e(r.get('path'))}</td><td class='r'>"
                    f"{_e(_n(r['bytes']) + ' bytes') if r.get('bytes') is not None else ''}"
                    f"</td><td>{_e(r.get('how') or '')}</td><td>"
                    f"{_e(r.get('turn') if r.get('turn') is not None else '')}"
                    f"</td></tr>" for r in rows) + "</table>")

    def _notes(self) -> str:
        lines: List[str] = []
        for t in self.turns:
            pre = f"Turn {t.number}: " if len(self.turns) > 1 else ""
            end = t.end or {}
            if end.get("error"):
                lines.append(f"{pre}the turn ended in an error: "
                             f"{_one_line(end['error'], 400)}")
            if end.get("error_detail"):
                lines.append(f"{pre}the provider said: "
                             f"{_one_line(end['error_detail'], 400)}")
            cap, steps = _int(end.get("step_cap")), _int(end.get("graph_steps"))
            if cap and steps is not None and steps >= cap:
                lines.append(f"{pre}the main agent reached its step cap "
                             f"({steps} of {cap})")
            images = _image_facts(end)
            if images is not None:
                calls, limit, refused = images
                lines.append(f"{pre}image calls: {_n(calls)}" + (
                    f" of a ceiling of {_n(limit)}" if limit
                    else ", no ceiling set") + (
                    f"; {_n(refused)} more refused at the ceiling"
                    if refused else ""))
            if self.record.has_activity and t.end is None:
                lines.append(f"{pre}no turn_end in the record: the turn did "
                             f"not finish, or its end was not written")
            counted: Dict[str, List[dict]] = {}
            for rec in t.notes:
                ev = str(rec.get("event") or "")
                if ev in COUNTED_EVENTS:
                    counted.setdefault(ev, []).append(rec)
                    continue
                who = self._who(t, rec)
                words = EVENT_WORDS.get(ev, ev.replace("_", " "))
                detail = _fields_line(rec)
                lines.append(f"{pre}{who}{words}" + (
                    f" ({ev})" if ev in EVENT_WORDS else "") + (
                    f": {detail}" if detail else ""))
            for ev, recs in counted.items():
                if ev == "process_start" and len(recs) < 2:
                    continue
                wait = sum(float(r.get(k) or 0) for r in recs
                           for k in _WAIT_KEYS
                           if isinstance(r.get(k), (int, float)))
                lines.append(f"{pre}{EVENT_WORDS.get(ev, ev)} ({ev}): "
                             f"{len(recs)} time" + ("s" if len(recs) > 1
                                                    else "") + (
                    f", {_secs(wait)} in all" if wait else ""))
        if not lines:
            return "<p class='note'>None.</p>"
        return "<ul>" + "".join(f"<li>{_e(x)}</li>" for x in lines) + "</ul>"

    @staticmethod
    def _who(t: Turn, rec: dict) -> str:
        tid = str(rec.get("task_id") or "")
        for a in t.helpers:
            if a.key == "t:" + tid and tid:
                return f"{a.label}: "
        agent = rec.get("agent")
        if agent and agent not in ("primary", "app"):
            return f"{agent}: "
        return ""

    # -- 2. step by step --------------------------------------------------
    def _steps(self) -> str:
        if not self.record.has_activity:
            return self._turns_without_log()
        if not self.turns:
            return "<p class='note'>The activity log holds no records.</p>"
        questions = self.record.questions()
        answers = self.record.answers()
        out = ["<p class='muted'>Every step in the order it was taken. A "
               "step is one call of an agent's model: its own words, then "
               "the tool calls it asked for, each with its arguments and "
               "what came back. A helper's steps sit under the task call "
               "that started it, after its brief. Calls are numbered "
               "through the whole run; part 4 uses the same numbers.</p>"]
        for t in self.turns:
            out.append(f'<section class="turn" id="turn-{_e(t.number)}">'
                       f"<h3>Turn {_e(t.number)}</h3>")
            q = questions.get(t.number) if isinstance(t.number, int) else None
            q = q or _start_prompt(t)
            if q:
                out.append(f"<div><b>Asked:</b> {_short_or_fold(q)}</div>")
            note = (t.start or {}).get("context_note")
            if note:
                out.append(_fold(f"What the page put before the message: "
                                 f"{_n(len(note))} characters",
                                 f"<pre>{_e(note)}</pre>"))
            out.append(self._actor_steps(t.primary))
            orphans = [a for a in t.helpers if a.task_call is None]
            if orphans:
                out.append("<h4>Helpers the record does not tie to a task "
                           "call</h4>")
                for a in orphans:
                    out.append(self._helper_block(a))
            if t.loose_side:
                out.append(f"<p class='note'>{len(t.loose_side)} vision "
                           "calls the record does not tie to a tool call "
                           "(records before EV-14).</p>")
            ans = answers.get(t.number) if isinstance(t.number, int) else None
            if ans:
                out.append(_fold(f"The answer as the user received it: "
                                 f"{_n(len(ans))} characters",
                                 f"<pre>{_e(ans)}</pre>"))
            else:
                out.append("<p class='note'>The record holds no answer as "
                           "the user received it for this turn.</p>")
            out.append("</section>")
        return "".join(out)

    def _turns_without_log(self) -> str:
        """No activity log: each turn's question and answer as the record's
        other files hold them (the transcript, or the suite's run.json)."""
        rec = self.record
        questions, answers = rec.questions(), rec.answers()
        files = rec.answer_files()
        numbers = sorted(set(questions) | set(answers) | set(files))
        if not numbers:
            return (f"<p class='warn'>No {ACTIVITY_FILE}: the steps cannot be "
                    "shown, and no other file of the record holds a question "
                    "or an answer.</p>")
        out = [f"<p class='warn'>No {ACTIVITY_FILE}: the steps cannot be "
               "shown. Each turn's question and answer as the record's other "
               "files hold them:</p>"]
        for i in numbers:
            out.append(f'<section class="turn" id="turn-{i}">'
                       f"<h3>Turn {i}</h3>")
            q = questions.get(i)
            out.append(f"<div><b>Asked:</b> {_short_or_fold(q)}</div>" if q
                       else "<p class='note'>The record holds no question "
                            "for this turn.</p>")
            ans = answers.get(i)
            if ans:
                out.append(_fold(f"The answer as the user received it: "
                                 f"{_n(len(ans))} characters",
                                 f"<pre>{_e(ans)}</pre>", open_=True))
            else:
                out.append("<p class='note'>The record holds no answer as "
                           "the user received it for this turn.</p>")
            if files.get(i):
                out.append("<div class='note'>Files shown with the answer: "
                           f"{_e(', '.join(files[i]))}</div>")
            out.append("</section>")
        return "".join(out)

    def _actor_steps(self, actor: Actor) -> str:
        out = []
        for i, step in enumerate(actor.steps, 1):
            out.append(self._step(actor, i, step))
        if not actor.steps:
            out.append("<p class='note'>No steps recorded.</p>")
        return "".join(out)

    def _step(self, actor: Actor, i: int, step: Step) -> str:
        head = [f"<b>Step {i}</b>", _e(actor.label)]
        said = ""
        m = step.model
        if m is not None:
            if m.get("source"):
                head.append(f"a call the agent framework made itself "
                            f"({_e(m['source'])})")
            if m.get("event") == "model_error":
                head.append("<span class='warn'>the model call failed</span>")
                said = (f"<div class='warn'>"
                        f"{_e(_one_line(m.get('error'), 600))}</div>")
            n_asked = _int(m.get("n_tool_calls"))
            if n_asked is not None:
                head.append(f"asked for {n_asked} tool call"
                            + ("" if n_asked == 1 else "s"))
            u = _usage_text(_usage(m))
            if u:
                head.append(_e(u))
            if m.get("duration_s") is not None:
                head.append(_e(_secs(m["duration_s"])))
            text, tnote = _whole_text(self.record, m, "text")
            if text:
                said += f"<div class='said'>{_e(text)}</div>"
                if tnote:
                    said += f"<div class='note'>{_e(tnote)}</div>"
            elif m.get("event") == "model_end" and not step.calls:
                said += "<div class='note'>(no words, no tool calls)</div>"
            reason = m.get("reasoning")
            if reason:
                full = self.record.sidecar(m.get("reasoning_full")) or reason
                said += _fold(f"Reasoning summary: {_n(len(full))} "
                              "characters", f"<pre>{_e(full)}</pre>")
        else:
            head.append("<span class='note'>tool calls with no model call "
                        "recorded before them</span>")
        calls = "".join(self._call(c) for c in step.calls)
        return (f'<div class="step"><div class="sh">{" · ".join(head)}</div>'
                f"{said}{calls}</div>")

    def _call(self, call: Call) -> str:
        head = [f'<span class="no">#{call.number}</span>',
                f"<b>{_e(call.name)}</b>"]
        body: List[str] = []
        if call.refused:
            d = call.delivered[0]
            head.append("<span class='warn'>refused: no tool ran</span>")
            text, note = _whole_text(self.record, d, "text")
            body.append(_payload("What the model was told", text, note))
            return (f'<div class="call" id="c{call.number}">'
                    f'<div class="ch">{" ".join(head)}</div>'
                    f"{''.join(body)}</div>")
        end = call.end or {}
        if end.get("duration_s") is not None:
            head.append("· " + _e(_secs(end["duration_s"])))
        if call.side:
            head.append(f"· {len(call.side)} vision call"
                        + ("" if len(call.side) == 1 else "s"))
        for href, words in self.pages.anchor(call):
            head.append(f'· <a href="{_e(href)}">{_e(words)}</a>')
        if call.start is None:
            head.append("<span class='note'>(its start, with the "
                        "arguments, is not in the record)</span>")
        tied = call.helper is not None and call.helper.task_call is call
        brief_only = tied and isinstance(call.args, dict) and set(
            call.args) <= {"description", "subagent_type"}
        if call.args not in (None, "", {}) and not brief_only:
            text = call.args if isinstance(call.args, str) else json.dumps(
                call.args, ensure_ascii=False, indent=1, default=str)
            body.append(_payload("Arguments", text))
        if call.helper is not None and call.helper.task_call is call:
            body.append(self._helper_block(call.helper))
        if call.end is None:
            body.append("<div class='warn'>No result in the record (the "
                        "call did not finish, or its end was not "
                        "written).</div>")
        elif call.failed:
            text, note = _whole_text(self.record, end, "error")
            body.append("<div class='warn'>The tool raised an error.</div>"
                        + _payload("Error", text, note))
        else:
            text, note = _whole_text(self.record, end, "result")
            label = ("What the helper handed back" if call.name == TASK_TOOL
                     else "Result")
            body.append(_payload(label, text, note))
        for d in call.delivered:
            text, note = _whole_text(self.record, d, "text")
            if d.get("event") == "tool_refused":
                lab = "The error message the model was given"
            elif d.get("stage") == "graph":
                lab = ("What the model received instead (the agent framework "
                       "moved the result" + (
                           f" to {d['moved_to']}" if d.get("moved_to")
                           else "") + ")")
            else:
                lab = "What the model received (it differs from the result)"
            body.append(_payload(_e(lab), text, note))
        if call.side:
            body.append(self._side_calls(call.side))
        for ev in call.events:
            # an ordinary record of the call is a note, not a warning
            kind = "note" if ev.get("event") in PLAIN_EVENTS else "warn"
            body.append(f"<div class='{kind}'>{_e(EVENT_WORDS.get(str(ev.get('event')), ev.get('event')))}: "
                        f"{_e(_fields_line(ev))}</div>")
        return (f'<div class="call" id="c{call.number}">'
                f'<div class="ch">{" ".join(head)}</div>'
                f"{''.join(body)}</div>")

    def _helper_block(self, actor: Actor) -> str:
        brief = actor.brief or "(the record holds no brief for this helper)"
        title = (f"{_e(actor.label)} ({_e(actor.agent)}): its brief and its "
                 f"{len(actor.steps)} step" + ("" if len(actor.steps) == 1
                                               else "s"))
        body = (f'<div class="brief"><b>Brief</b><pre>{_e(brief)}</pre></div>'
                + self._actor_steps(actor))
        return _fold(title, body, open_=True, cls="helper")

    def _side_calls(self, sides: List[dict]) -> str:
        rows = []
        tin = tout = 0
        for s in sides:
            start, end = s.get("start") or {}, s.get("end") or {}
            label = dict(start.get("vision") or {})
            label.update(end.get("vision") or {})
            u = _usage(end)
            tin += u["in"] or 0
            tout += u["out"] or 0
            page = label.get("page")
            bits = [str(label.get("kind") or "look")]
            if isinstance(page, int) and not isinstance(page, bool):
                bits.append(f"page {page + 1}")
            if label.get("tile"):
                bits.append(f"tile {label['tile']}")
            px = label.get("image_px")
            if isinstance(px, (list, tuple)) and len(px) == 2:
                bits.append(f"image {px[0]} x {px[1]} px")
            elif px:
                bits.append(f"image {px} px")
            if label.get("detail_sent") or label.get("detail"):
                bits.append(f"detail {label.get('detail_sent') or label.get('detail')}")
            ut = _usage_text(u)
            if ut:
                bits.append(ut)
            if end.get("duration_s") is not None:
                bits.append(_secs(end["duration_s"]))
            if not end:
                bits.append("no end recorded")
            elif end.get("event") == "model_error":
                bits.append("failed: " + _one_line(end.get("error"), 200))
            inner = ""
            if label.get("prompt"):
                inner += _payload("Sent with the image (as far as the record "
                                  "keeps it)", str(label["prompt"]),
                                  f"{_n(label.get('prompt_chars'))} characters "
                                  "in all" if label.get("prompt_chars") else "")
            text, note = _whole_text(self.record, end, "text") if end \
                else ("", "")
            if text:
                inner += _payload("What the vision call answered", text, note)
            rows.append(f"<li>{_e(', '.join(bits))}{inner}</li>")
        title = (f"{len(sides)} vision call" + ("" if len(sides) == 1
                                                else "s")
                 + (f": {_n(tin)} tokens in, {_n(tout)} out" if tin or tout
                    else ""))
        return _fold(_e(title), "<ol>" + "".join(rows) + "</ol>")

    # -- 3. what was taken in ---------------------------------------------
    def _taken_in(self) -> str:
        docs = self._docs()
        if not docs:
            return ("<p class='note'>The record names no document page that "
                    "a tool read or looked at.</p>")
        present = {m for d in docs for u in d.pages.values() for m in u.marks}
        legend = "".join(f"<li><b>{letter}</b> {_e(words)}</li>"
                         for m, letter, _c, words in MARKS
                         if m in ALWAYS_COUNTED or m in present)
        none = ("a page with none was never opened" if
                self.record.has_activity else
                f"a page with none has no look recorded (there is no "
                f"{ACTIVITY_FILE}, so only the vision tools' own records "
                f"count, and a page may have been opened without one)")
        out = ["<p class='muted'>Each square is one page. The letters say "
               f"how it was taken in; {_e(none)}. "
               "Only what the record says counts: a page named in a search "
               "hit is not counted as opened. Choose a square to open its "
               "page in the page viewer: the page with a box for every look, "
               "what each call asked and what came back, and the previous "
               "and next pages (with scripts off, a square of a page looked "
               "at links to its drawing in part 4).</p>",
               f"<ul class='legend'>{legend}</ul>"]
        if self.viewer_problem:
            out.append(f"<p class='warn'>The page viewer could not be made "
                       f"from the record: {_e(self.viewer_problem)}</p>")
        if self.pages.unplaced:
            out.append(f"<p class='note'>{self.pages.unplaced} tool results "
                       "could not be read for pages.</p>")
        for doc in docs:
            out.append(self._grid(doc))
        return "".join(out)

    def _grid(self, doc: Document) -> str:
        n = doc.n_pages or 0
        counts = {m[0]: 0 for m in MARKS}
        never = 0
        # with no activity log no look is recorded at all: an empty square
        # says only that
        empty = "never opened" if self.record.has_activity else \
            "no look recorded"
        cells = []
        for p in range(max(n, (max(doc.pages) + 1) if doc.pages else 0)):
            use = doc.pages.get(p)
            marks = [m[0] for m in MARKS if use and m[0] in use.marks]
            for m in marks:
                counts[m] += 1
            if not marks:
                never += 1
            deepest = next((m for m in ("zoom", "tiles", "whole", "text",
                                        "contact", "text_empty", "sweep")
                            if m in marks), "")
            letters = " ".join(m[1] for m in MARKS if m[0] in marks)
            words = "; ".join(m[3] for m in MARKS if m[0] in marks) or empty
            cls = f"pg k-{deepest}" if marks else "pg never"
            looked = use is not None and bool(use.looks)
            tag, href = ("a", f' href="#p-{doc.index}-{p}"') if looked \
                else ("span", "")
            cells.append(f'<{tag} class="{cls}"{href} data-page="{p + 1}" '
                         f'title="PDF page {p + 1}: {_e(words)}">'
                         f"<b>{p + 1}</b><i>{_e(letters)}</i></{tag}>")
        count_spans = "".join(
            f'<span data-kind="{m}">{_e(label)}: <b>{counts[m]}</b></span>'
            for m, _l, label, _w in MARKS
            if counts[m] or m in ALWAYS_COUNTED)
        count_spans += (f'<span data-kind="never">{empty}: '
                        f'<b>{never}</b></span>')
        about = f"{_n(n)} pages" if n else "page count unknown"
        if doc.n_from:
            about += f", from {doc.n_from}"
        return (f'<div class="doc" id="doc-{doc.index}" data-i="{doc.index}">'
                f'<h3>{_e(doc.name)}'
                f'</h3><p class="muted">{_e(about)}</p>'
                f'<p class="counts" data-doc="{_e(doc.name)}">{count_spans}'
                f'</p><div class="grid">{"".join(cells)}</div></div>')

    # -- 4. where they looked ---------------------------------------------
    def _where(self) -> str:
        looked = [(d, p, u) for d in self._docs()
                  for p, u in sorted(d.pages.items()) if u.looks]
        if not looked:
            return ("<p class='note'>No page was looked at (no "
                    f"{PAGE_TOOL} or {ZOOM_TOOL} call in the record).</p>")
        out = ["<p class='muted'>Each page looked at, at its proportions, "
               "with a box for every look: a dashed box for a whole-page "
               "view, thin boxes for its tiles, red boxes for zooms. Each "
               "box carries the number of its tool call in part 2.</p>"]
        missing = sorted({d.name for d, _p, _u in looked if not d.pdf})
        if missing:
            why = ("PyMuPDF is not installed here" if self.images.fitz is None
                   else "their PDF was not found in the record folder" + (
                       " or under --documents" if self.documents else
                       "; name the documents folder with --documents"))
            out.append(f"<p class='note'>Drawn as outlines only, because "
                       f"{_e(why)}: {_e(', '.join(missing))}.</p>")
        elif self.images.fitz is None:
            out.append("<p class='note'>Drawn as outlines only: PyMuPDF is "
                       "not installed here.</p>")
        last_doc = None
        figs: List[str] = []
        for doc, page, use in looked:
            if doc is not last_doc:
                if figs:
                    out.append(f'<div class="figs">{"".join(figs)}</div>')
                    figs = []
                out.append(f"<h3>{_e(doc.name)}</h3>")
                last_doc = doc
            figs.append(self._figure(doc, page, use))
        if figs:
            out.append(f'<div class="figs">{"".join(figs)}</div>')
        note = self._image_note()
        if note:
            out.append(f"<p class='note'>{_e(note)}</p>")
        for problem in self.images.problems:
            out.append(f"<p class='note'>{_e(problem)}</p>")
        return "".join(out)

    def _figure(self, doc: Document, page: int, use: PageUse) -> str:
        img = (self._plan_images().get((doc.index, page)) or {}).get(
            PAGE_IMAGE_PX)
        rect, how = self._page_rect(doc, page, use)
        note = {"pdf": "drawn from the PDF" if img
                else "outline at the PDF page's size",
                "view": "outline from the whole-page view",
                "boxes": "the page size is not in the record: the outline is "
                         "drawn round its boxes"}.get(how, "")
        if how != "pdf":
            img = None
        items = []
        tiles: Dict[Tuple[int, str], List[str]] = {}
        for lk in use.looks:
            if lk.kind == "tile":
                tiles.setdefault((lk.number, lk.source), []).append(lk.label)
        for lk in sorted(use.looks, key=lambda x: (x.number, x.kind != "page",
                                                    x.label)):
            if lk.kind == "tile":
                continue
            num = (f'<a href="#c{lk.number}">#{lk.number}</a>' if lk.number
                   else f"from {_e(lk.source)}")
            what = {"page": "whole page", "zoom": "zoom"}[lk.kind]
            if lk.box and lk.kind == "zoom":
                what += (f" on x {lk.box[0]:.0f}-{lk.box[2]:.0f}, "
                         f"y {lk.box[1]:.0f}-{lk.box[3]:.0f} pt")
            if lk.box is None:
                what += " (no view box in the record)"
            labels = tiles.get((lk.number, lk.source)) if lk.kind == "page" \
                else None
            if labels:
                what += f", and in {len(labels)} tiles ({', '.join(labels)})"
            who = f" by {_e(lk.who)}" if lk.who else ""
            tool = f" ({_e(lk.tool)})" if lk.tool else ""
            items.append(f"<li>{num} {_e(what)}{tool}{who}</li>")
        svg = self._svg(rect, img, use.looks) if rect else \
            "<p class='note'>No view box in the record: nothing to draw.</p>"
        return (f'<figure class="fig" id="p-{doc.index}-{page}"><figcaption>'
                f"<b>PDF page {page + 1}</b> <span class='muted'>"
                f"{_e(note)}</span> <a class='vw-open' hidden "
                f"href='#view-{doc.index}-{page + 1}'>Open in the page "
                f"viewer</a></figcaption>{svg}<ol>{''.join(items)}"
                f"</ol></figure>")

    @staticmethod
    def _svg(rect: tuple, img: Optional[str], looks: List[Look]) -> str:
        x0, y0, w, h = rect
        scale = FIGURE_PX / max(w, h)
        fs = max(w, h) / FIGURE_PX * 10.0
        parts = [f'<svg width="{round(w * scale)}" height="{round(h * scale)}"'
                 f' viewBox="{x0:g} {y0:g} {w:g} {h:g}" role="img" '
                 f'aria-label="page outline with the boxes looked at">',
                 f'<rect class="sheet" x="{x0:g}" y="{y0:g}" width="{w:g}" '
                 f'height="{h:g}" vector-effect="non-scaling-stroke"/>']
        if img:
            parts.append(f'<image href="data:image/jpeg;base64,{img}" '
                         f'x="{x0:g}" y="{y0:g}" width="{w:g}" '
                         f'height="{h:g}" preserveAspectRatio="none"/>')
        corners: Dict[Tuple[int, int], int] = {}
        order = {"page": 0, "tile": 1, "zoom": 2}
        for lk in sorted(looks, key=lambda x: (order[x.kind], x.number)):
            if lk.box is None:
                continue
            bx0, by0, bx1, by1 = lk.box
            cls = {"page": "bw", "tile": "bt", "zoom": "bz"}[lk.kind]
            tip = (f"#{lk.number} " if lk.number else "") + {
                "page": "whole page", "tile": f"tile {lk.label}",
                "zoom": "zoom"}[lk.kind]
            parts.append(f'<rect class="{cls}" x="{bx0:g}" y="{by0:g}" '
                         f'width="{bx1 - bx0:g}" height="{by1 - by0:g}" '
                         f'vector-effect="non-scaling-stroke"><title>'
                         f"{_e(tip)}</title></rect>")
            key = (round(bx0 / fs), round(by0 / fs))
            k = corners.get(key, 0)
            corners[key] = k + 1
            label = (f"#{lk.number}" if lk.number else "") + (
                f" {lk.label}" if lk.kind == "tile" and lk.label else "")
            if not label:
                continue
            tx, ty = bx0 + fs * 0.3, by0 + fs * (1.1 + 1.2 * k)
            text = (f'<text class="{cls}" x="{tx:g}" y="{ty:g}" '
                    f'font-size="{fs:g}" stroke-width="{fs * 0.3:g}">'
                    f"{_e(label)}</text>")
            parts.append(f'<a href="#c{lk.number}">{text}</a>' if lk.number
                         else text)
        parts.append("</svg>")
        return "".join(parts)

    # -- 5. what the record holds -----------------------------------------
    def _holds(self) -> str:
        def flagged(p: Part) -> bool:
            # a file one kind of folder never has is not a gap in it
            return p.status.startswith("unreadable") or (
                p.name == ACTIVITY_FILE and not p.status.startswith("found"))

        rows = "".join(
            f"<tr><td>{_e(p.name)}</td><td"
            f"{' class=warn' if flagged(p) else ''}>"
            f"{_e(p.status)}</td><td class='muted'>{_e(p.meaning)}</td></tr>"
            for p in self.record.parts)
        older = self._older()
        out = ["<table><tr><th>File</th><th>In this record</th>"
               f"<th>What it gives</th></tr>{rows}</table>"]
        if older:
            out.append("<h3>Older shapes met</h3><ul>" + "".join(
                f"<li>{_e(x)}</li>" for x in older) + "</ul>")
        return "".join(out)

    def _older(self) -> List[str]:
        acts = self.record.activity
        if not acts:
            return []
        out = []
        helper_recs = [r for r in acts if r.get("agent") not in
                       (None, "primary", "app")]
        if helper_recs and not any(r.get("task_id") for r in helper_recs):
            out.append("Helper records carry no task_id (written before "
                       "2026-10-14): helpers of one type that ran together "
                       "cannot be told apart, and each is tied to the open "
                       "task call of its type.")
        ends = [r for r in acts if r.get("event") == "model_end"]
        if ends and not any(isinstance(r.get("usage"), dict) for r in ends):
            out.append("Model calls carry no usage: tokens are not recorded.")
        if ends and not any("text" in r for r in ends):
            out.append("Model calls carry no text (written before "
                       "2026-10-07): the model's own words are not in the "
                       "record.")
        starts = [r for r in acts if r.get("event") == "model_start"]
        if any(r.get("vision") for r in starts) and not any(
                r.get("tool_run_id") for r in starts if r.get("vision")):
            out.append("Vision calls are not tied to their tool call "
                       "(written before EV-14).")
        if any(t.loose_side for t in self.turns) and not out:
            out.append("Some vision calls are not tied to a tool call.")
        return out


def _image_facts(end: Optional[dict]
                 ) -> Optional[Tuple[int, Optional[int], Optional[int]]]:
    """A turn's own count of its image calls, its ceiling and the calls the
    ceiling refused (``turn_end``'s ``image_calls``, ``image_limit`` and
    ``image_calls_refused``, written by
    :func:`aec_agent_core.deep.keep_work.image_call_facts`); ``None`` when
    the record keeps no count."""
    if not isinstance(end, dict):
        return None
    calls = _int(end.get("image_calls"))
    if calls is None:
        return None
    return calls, _int(end.get("image_limit")), \
        _int(end.get("image_calls_refused"))


def _start_prompt(t: Turn) -> str:
    p = str((t.start or {}).get("prompt") or "")
    if len(p) >= 500:
        p += " ... (the activity log keeps the first 500 characters)"
    return p


def _short_or_fold(text: str) -> str:
    if len(text) <= OPEN_CHARS:
        return f"<span class='said'>{_e(text)}</span>"
    return (f"<span class='said'>{_e(text[:300])} ...</span>" + _fold(
        f"the whole question: {_n(len(text))} characters",
        f"<pre>{_e(text)}</pre>"))


def _fields_line(rec: dict) -> str:
    bits = []
    for k, v in rec.items():
        if k in _PLACE_KEYS or v in (None, "", [], {}):
            continue
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False,
                                                       default=str)
        bits.append(f"{k}: {_one_line(text, 200)}")
    return _one_line("; ".join(bits), 700)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def build_report(folder: str, documents: Optional[str] = None) -> str:
    """The report of the run recorded in ``folder``, as one HTML string."""
    return Report(folder, documents).html()


def write_report(folder: str, out: Optional[str] = None,
                 documents: Optional[str] = None) -> str:
    """Write the report of ``folder``'s run to ``out`` (default: beside the
    record, :data:`REPORT_FILE`; a folder gets that name inside it) and
    return its path. Nothing else is written."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no such run folder: {folder}")
    target = out or os.path.join(folder, REPORT_FILE)
    if os.path.isdir(target):
        target = os.path.join(target, REPORT_FILE)
    text = build_report(folder, documents)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    return os.path.abspath(target)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aec_agent_core.run_report",
        description="Write an offline HTML report of one saved run: every "
                    "step, every tool call and its result, the pages read "
                    "and where the vision tools looked.")
    parser.add_argument("folder", help="the run's folder (a page "
                        "conversation, or one task run of the review suite)")
    parser.add_argument("-o", "--out", help="the report file to write "
                        f"(default: {REPORT_FILE} in the run folder)")
    parser.add_argument("--documents", help="a folder holding the run's "
                        "PDFs, to draw the pages behind the boxes")
    a = parser.parse_args(argv)
    if not os.path.isdir(a.folder):
        print(f"No such run folder: {a.folder}", file=sys.stderr)
        return 2
    if a.documents and not os.path.isdir(a.documents):
        print(f"No such documents folder: {a.documents}", file=sys.stderr)
        return 2
    try:
        path = write_report(a.folder, a.out, a.documents)
    except OSError as exc:
        print(f"The report could not be written: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - one plain line, never a traceback
        print(f"The report could not be made from the record in "
              f"{_folder_name(a.folder)}: {type(exc).__name__}: "
              f"{_one_line(exc, 300)}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    print(CONTENT_NOTE)
    return 0


__all__ = ["build_report", "write_report", "main", "RunRecord",
           "build_turns", "REPORT_FILE", "CONTENT_NOTE"]


if __name__ == "__main__":
    sys.exit(main())
