"""The locked box: plotting code the agent writes, run in its own process.

The owner, 2026-10-15 (the figures page, BACKLOG GM-98): "I've been
frustrated by the calc packages' lack of creative figures, and now I'm
guessing that the root cause is the lack of options the agent has. This
would solve it... Please try to build it as securely as possible, don't do
things that would trip the package up"; and "Don't limit the run time too
much, I want it to cook when we have some complicated things to visualize."

``plot_code`` hands matplotlib (or Plotly) code to a fresh Python process
(:mod:`funhouse_agent.plot_box.child`, where the layers are described) with:

- a scrubbed environment: no keys, tokens or settings of the app's;
- the conversation's folder as the only place it reads data from and the
  only place it writes, and only figure, chart and data files;
- no network, no processes, no native code (an audit hook that cannot be
  removed, over the code AND the libraries it calls);
- imports limited to the plotting and number libraries;
- a memory ceiling and a generous time limit the call can raise (the
  person sees the drawing is still running in the app's "Working..."
  label).

Standard library only: ``subprocess`` without a shell, ``resource`` limits
on Linux, a job object on Windows; nothing is installed.

ON by default since 2026-10-20 (the owner, GM-92: "on everywhere";
BACKLOG GM-227); ``GEOTECH_PLOT_CODE=0`` turns it off. Offered to the
figure helper and the ``calc`` helper. Settings:

``GEOTECH_PLOT_CODE_SECONDS``
    the time limit a call gets when it names none (600);
``GEOTECH_PLOT_CODE_MAX_SECONDS``
    the most a call may ask for (3600);
``GEOTECH_PLOT_CODE_MEMORY_MB``
    the memory ceiling (2048);
``GEOTECH_PLOT_CODE_AT_ONCE``
    how many boxes run at once in one app process (2).
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from funhouse_agent.plot_box import figures
from funhouse_agent.plot_box.child import (ALLOWED_IMPORTS, check_code,
                                           DEFAULT_PRINT_WIDTH_IN,
                                           MAX_CODE_CHARS)

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child.py")

SECONDS_ENV = "GEOTECH_PLOT_CODE_SECONDS"
MAX_SECONDS_ENV = "GEOTECH_PLOT_CODE_MAX_SECONDS"
MEMORY_ENV = "GEOTECH_PLOT_CODE_MEMORY_MB"
AT_ONCE_ENV = "GEOTECH_PLOT_CODE_AT_ONCE"
SETTINGS_ENVS = (SECONDS_ENV, MAX_SECONDS_ENV, MEMORY_ENV, AT_ONCE_ENV)

DEFAULT_SECONDS = 600
DEFAULT_MAX_SECONDS = 3600
DEFAULT_MEMORY_MB = 2048
DEFAULT_AT_ONCE = 2
#: The largest ``data`` the box takes (as JSON); bigger data goes in a file.
MAX_DATA_BYTES = 5_000_000
#: Files one run may write.
MAX_FILES = 40
#: How often the "Working..." note is refreshed while a box runs.
NOTE_EVERY_S = 10.0
#: The shortest time limit a call can ask for.
MIN_SECONDS = 10.0

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".webp")
#: An interactive chart beside a picture of the same name.
CHART_SUFFIX = ".plotly.json"

#: Tools whose calls draw a figure: drawn figures are counted across them,
#: and across file names, within one figure helper run (live wave w9: a
#: helper drew six times -- two with plot_data, four with plot_code under a
#: new name -- past a cap of three that nothing counted), and per figure
#: across the turn, whoever draws it (live wave w12: a second helper's cap
#: started again; :mod:`funhouse_agent.plot_box.figures`).
DRAW_TOOLS = ("plot_code", "plot_data")
#: The draw from which a result asks for the best version to be handed
#: back: a first draw and at most two redraws.
DRAW_CAP = 3

#: What ``replaces`` may remove: pictures and charts of a figure, never data
#: or documents.
RETIRE_EXTS = (CHART_SUFFIX, ".png", ".jpg", ".jpeg", ".svg", ".gif",
               ".webp")
#: The files of one figure that go with a named draft (its picture and its
#: interactive chart).
_FIGURE_TWINS = (".png", ".jpg", ".jpeg", CHART_SUFFIX)


def _int_env(name: str, default: int, low: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return max(low, int(float(raw))) if raw else default
    except ValueError:
        return default


def settings() -> Dict[str, int]:
    """The box's limits as this process's environment sets them."""
    max_s = _int_env(MAX_SECONDS_ENV, DEFAULT_MAX_SECONDS, 10)
    return {"seconds": min(_int_env(SECONDS_ENV, DEFAULT_SECONDS, 10), max_s),
            "max_seconds": max_s,
            "memory_mb": _int_env(MEMORY_ENV, DEFAULT_MEMORY_MB, 256),
            "at_once": _int_env(AT_ONCE_ENV, DEFAULT_AT_ONCE, 1)}


def available() -> bool:
    """Whether the box can run here (matplotlib installed, child present)."""
    import importlib.util
    try:
        return (os.path.isfile(CHILD)
                and importlib.util.find_spec("matplotlib") is not None)
    except (ImportError, ValueError):
        return False


_SLOTS_LOCK = threading.Lock()
_SLOTS: Dict[int, threading.BoundedSemaphore] = {}


def _slots(n: int) -> threading.BoundedSemaphore:
    with _SLOTS_LOCK:
        if n not in _SLOTS:
            _SLOTS[n] = threading.BoundedSemaphore(n)
        return _SLOTS[n]


def _root() -> str:
    """The conversation's folder; a fresh temporary folder for a library
    caller with none (never the current directory)."""
    from funhouse_agent._fileio import host_output_dir
    root = host_output_dir()
    if not root:
        root = tempfile.mkdtemp(prefix="plot_code_")
    os.makedirs(root, exist_ok=True)
    return os.path.realpath(root)


def _snapshot(root: str) -> Dict[str, tuple]:
    seen: Dict[str, tuple] = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            p = os.path.join(base, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            seen[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
            if len(seen) > 20_000:
                return seen
    return seen


def _child_env(tmp: str) -> Dict[str, str]:
    """What the child inherits: where home and fonts are, nothing secret."""
    keep = ("SYSTEMROOT", "SystemRoot", "WINDIR", "SystemDrive", "PATH",
            "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA",
            "APPDATA", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_DATA_DIRS", "MPLCONFIGDIR",
            "FONTCONFIG_PATH", "FONTCONFIG_FILE", "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE")
    env = {k: os.environ[k] for k in keep if os.environ.get(k)}
    env.update({"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp})
    return env


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def _png_size(path: str):
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return (int.from_bytes(head[16:20], "big"),
                    int.from_bytes(head[20:24], "big"))
    except OSError:
        pass
    return None


def _minutes(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec:02d}s"


# ---------------------------------------------------------------------------
# draws counted across one helper run
# ---------------------------------------------------------------------------

#: The helper whose draws are capped: the figure helper draws ONE figure a
#: run, while the calc helper draws a package's several figures and a
#: general helper whatever it is asked (``deep.figure_helper.
#: FIGURE_HELPER_NAME``; named here so this module imports nothing heavy).
CAPPED_HELPER = "figure"


def drew_figure(text) -> bool:
    """Whether a drawing tool's result says it drew."""
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        return '"status": "success"' in str(text or "")[:400]
    return isinstance(out, dict) and out.get("status") == "success"


def _current_helper():
    try:
        from funhouse_agent.deep.keep_work import current_helper
    except Exception:                                      # noqa: BLE001
        return None
    return current_helper()


def draws_so_far() -> Optional[int]:
    """Figures the running FIGURE helper has drawn so far, by any drawing
    tool (:data:`DRAW_TOOLS`) under any file name; ``None`` anywhere else
    (the main agent, another helper, a library call): only a run that draws
    one figure is capped."""
    rec = _current_helper()
    if rec is None or getattr(rec, "subagent", None) != CAPPED_HELPER:
        return None
    return sum(1 for s in list(rec.steps)
               if s.get("kind") == "result" and s.get("name") in DRAW_TOOLS
               and s.get("status") != "error"
               and drew_figure(s.get("text")))


def draw_cap_note(n: int) -> str:
    """What a draw at or past :data:`DRAW_CAP` says."""
    return (f"This is draw {n} of this figure this turn (plot_code and "
            "plot_data draws count together, by any helper, under its name, "
            "a version of it or a name that replaces it; in the figure "
            "helper every draw of its run counts): look at it once if no "
            "look has read it, then hand back the best version and name "
            "what is left. Do not draw again.")


def drawn_count(helper_draw: Optional[int], figure_draw: Optional[int]):
    """The draw number a result gives: the more of this figure helper run's
    draws (:func:`draws_so_far` + 1) and this figure's draws this turn;
    ``None`` when neither is counted."""
    known = [n for n in (helper_draw, figure_draw) if n]
    return max(known) if known else None


def booked_notes(booked: Dict[str, Any]) -> List[str]:
    """What a draw says about the figures it continues
    (:func:`funhouse_agent.plot_box.figures.record_draw`)."""
    notes = []
    if booked.get("standing"):
        notes.append(
            ", ".join(booked["standing"]) + ", an earlier version of this "
            "figure drawn this turn, still shows as its own card: if this "
            "drawing takes its place, draw under that name or name it in "
            "replaces=")
    return notes


# ---------------------------------------------------------------------------
# drafts retired by a renamed redraw
# ---------------------------------------------------------------------------

def _within(path: str, base: str) -> bool:
    p, b = os.path.normcase(path), os.path.normcase(base.rstrip(os.sep))
    return p.startswith(b + os.sep)


def _drafts_since() -> Optional[float]:
    """The time a file must be newer than to count as a draft of this
    work: the start of the turn, else of the running helper (a host that
    opens no turn scope, such as the eval harness); ``None`` when neither is
    known, and then nothing is removed."""
    try:
        from funhouse_agent.deep.keep_work import current_turn
    except Exception:                                      # noqa: BLE001
        return None
    scope = current_turn()
    if scope is not None:
        return scope.started
    rec = _current_helper()
    started = getattr(rec, "started", None)
    return float(started) if isinstance(started, (int, float)) else None


def retire_drafts(names, root: Optional[str] = None, keep=(),
                  since: Optional[float] = None) -> Dict[str, Any]:
    """Remove the drafts a renamed redraw replaces, so the chat shows only
    the final figure (live wave w9: a draft the helper called superseded
    kept its card beside the final figure).

    ``names`` are file names in the working folder ``root`` (default: the
    conversation's): a picture or a chart (with its twin of the other kind
    beside it), or a name without its extension (every picture and chart of
    that name). Refused, each with the reason: a path outside ``root``, a
    file that is not a picture or a chart, the figure just drawn (``keep``),
    and a file older than ``since`` (default: the start of this turn, else
    of the running helper; a draft an earlier turn drew stays with that
    turn's answer, and an upload is never a draft). With no such time at
    all, nothing is removed. Returns ``{"retired": [...], "not_retired":
    [{"name", "why"}]}``, either key left out when empty; ``{}`` when no
    names were given.
    """
    if names in (None, "", [], ()):
        return {}
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)):
        return {"not_retired": [{"name": str(names),
                                 "why": "replaces takes a list of file "
                                        "names"}]}
    if not root:
        try:
            from funhouse_agent._fileio import host_output_dir
            root = host_output_dir()
        except Exception:                                  # noqa: BLE001
            root = None
    if not root:
        return {"not_retired": [{"name": str(n), "why": "no working folder"}
                                for n in names]}
    base = os.path.realpath(root)
    if since is None:
        since = _drafts_since()
    if since is None:
        return {"not_retired": [{"name": str(n),
                                 "why": "no turn or helper run to tell this "
                                        "work's drafts from older files; "
                                        "nothing was removed"}
                                for n in names]}
    kept = {os.path.normcase(os.path.realpath(
        k if os.path.isabs(k) else os.path.join(base, k)))
        for k in keep or () if k}
    retired, refused, seen = [], [], set()

    def rel(p):
        return os.path.relpath(p, base).replace(os.sep, "/")

    for raw in list(names)[:20]:
        name = str(raw or "").strip()
        if not name:
            continue
        cand = os.path.realpath(name if os.path.isabs(name)
                                else os.path.join(base, name))
        if not _within(cand, base):
            refused.append({"name": name,
                            "why": "outside the working folder; nothing "
                                   "outside it is removed"})
            continue
        low = cand.lower()
        ext = next((e for e in RETIRE_EXTS if low.endswith(e)), None)
        if ext is None and os.path.splitext(cand)[1] \
                and not os.path.isfile(cand + ".png"):
            refused.append({"name": name,
                            "why": "not a picture or chart of a figure; "
                                   "only figure drafts are removed"})
            continue
        stem = cand[:-len(ext)] if ext else cand
        twins = _FIGURE_TWINS if ext else RETIRE_EXTS
        files = [cand] if ext and os.path.isfile(cand) else []
        files += [stem + e for e in twins
                  if os.path.isfile(stem + e) and stem + e != cand]
        if not files:
            refused.append({"name": name,
                            "why": "no such figure in the working folder"})
            continue
        for f in files:
            key = os.path.normcase(f)
            if key in seen:
                continue
            seen.add(key)
            if key in kept:
                refused.append({"name": rel(f),
                                "why": "the figure this call drew"})
                continue
            try:
                if os.path.getmtime(f) < since:
                    refused.append({"name": rel(f),
                                    "why": "drawn before this turn; it "
                                           "stays with that turn's answer"})
                    continue
                os.remove(f)
                retired.append(rel(f))
                # its kept code goes too: no later brief is handed the code
                # of a figure that is gone (GM-151)
                figures.forget_code(base, rel(f))
            except OSError as exc:
                refused.append({"name": rel(f),
                                "why": f"could not be removed: {exc}"})
    out: Dict[str, Any] = {}
    if retired:
        out["retired"] = retired
    if refused:
        out["not_retired"] = refused
    return out


def _settle_charts(images: list, made: set, root: str) -> list:
    """After a draw: a chart an EARLIER draw left under a picture's name is
    removed, so the chat shows the new picture (live wave w9: a plot_data
    draft's chart stood in for the redrawn PNG); a chart drawn in the same
    run as its picture is dated with it, so the chat shows the chart the
    code chose to save. A chart is removed only when it is newer than the
    start of the turn (else of the running helper); with neither known,
    none is. Returns the notes to give."""
    notes = []
    since = _drafts_since()
    for entry in images:
        rel = entry["file"]
        stem, ext = os.path.splitext(rel)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        chart_rel = stem + CHART_SUFFIX
        chart = os.path.join(root, chart_rel)
        if not os.path.isfile(chart):
            continue
        try:
            pic_ns = os.stat(entry["output_path"]).st_mtime_ns
            st = os.stat(chart)
            if chart_rel in made:
                if st.st_mtime_ns < pic_ns:
                    os.utime(chart, ns=(st.st_atime_ns, pic_ns))
                continue
            if since is None or st.st_mtime < since:
                continue        # an earlier turn's chart stays with it
            os.remove(chart)
        except OSError:
            continue
        notes.append(f"removed {chart_rel.replace(os.sep, '/')}, the "
                     "interactive chart an earlier draw left under this "
                     "name: the chat shows this PNG now")
    return notes


def run_plot_code(code: str, data: Any = None, name: str = "",
                  time_limit_s: Optional[float] = None,
                  root: Optional[str] = None, replaces=None,
                  print_width_in: Optional[float] = None,
                  draw: Optional[int] = None) -> Dict[str, Any]:
    """Run plotting ``code`` in the locked box; the files it made and how
    it went, as a JSON-ready dict (never raises for the code's own faults).

    ``replaces``: drafts a successful draw retires (:func:`retire_drafts`).
    ``print_width_in``: the width the figure is printed at, for the
    lettering's printed size (default 6.5 in). ``draw``: this draw's number
    in the running helper (:func:`draws_so_far` + 1), which decides whether
    the result invites a redraw.
    """
    problems = check_code(code)
    if problems:
        return {"status": "error",
                "error": "The code was not run:\n" + "\n".join(problems),
                "next": ("Rewrite the code within the box's rules (see the "
                         "tool description) and call plot_code again.")}
    cfg = settings()
    notes = []
    limit = cfg["seconds"]
    if time_limit_s not in (None, "", 0):
        try:
            asked = float(time_limit_s)
        except (TypeError, ValueError):
            asked = limit
        limit = max(MIN_SECONDS, min(asked, cfg["max_seconds"]))
        if asked > cfg["max_seconds"]:
            notes.append(f"time_limit_s {asked:g} is above this host's "
                         f"ceiling ({cfg['max_seconds']} s); ran with "
                         f"{cfg['max_seconds']} s")
    try:
        data_json = json.dumps(data, allow_nan=True)
    except (TypeError, ValueError) as exc:
        return {"status": "error",
                "error": f"data is not plain JSON ({exc}); pass numbers, "
                         "text, lists and objects"}
    if len(data_json) > MAX_DATA_BYTES:
        return {"status": "error",
                "error": f"data is {len(data_json):,} bytes as JSON; the box "
                         f"takes at most {MAX_DATA_BYTES:,} -- save it as a "
                         "CSV or JSON file in the conversation folder and "
                         "read the file in the code"}
    try:
        root = os.path.realpath(root) if root else _root()
    except Exception as exc:                               # noqa: BLE001
        return {"status": "error",
                "error": f"no conversation folder to draw in: {exc}"}
    try:
        width = float(print_width_in) if print_width_in not in (None, "",
                                                                0) else None
    except (TypeError, ValueError):
        width = None
    if width is not None and not 1.0 <= width <= 60.0:
        notes.append(f"print_width_in {print_width_in} is not a page width; "
                     f"measured at {DEFAULT_PRINT_WIDTH_IN:g} in")
        width = None
    extra = {"replaces": replaces, "draw": draw,
             "print_width_in": width or DEFAULT_PRINT_WIDTH_IN}

    from funhouse_agent.drawing_wait import drawing
    # A look at a figure in this folder waits until the box has written it
    # (live wave w8: a draw and a look in the same step read the old PNG).
    with drawing(root):
        return _run_in_box(code, data_json, name, root, limit, cfg, notes,
                           extra)


def next_step(draw: Optional[int], lettering: list = (),
              renamed_drafts: bool = True,
              checked_in: str = "layout_check") -> str:
    """What a successful draw asks for next: a LOOK that checks what the
    figure shows against the data before the lettering (live wave w9: 13 of
    13 look prompts asked about lettering only, and figures that misdrew
    passed), and a redraw only within the cap. ``plot_data`` gives it too
    (BACKLOG GM-152: live wave w12, its result asked for no look);
    ``renamed_drafts`` False where the tool takes no ``replaces``;
    ``checked_in`` the result key listing the layout check's findings."""
    look = (
        "LOOK at each figure with analyze_image (its look_file when it has "
        "one), in a step after this one. Ask first what it SHOWS, against the "
        "data you drew it from: have the look read two or three plotted "
        "values and the axis ranges back off the figure, and compare them "
        "here with the data; is anything drawn that the data do not give (a "
        "stair or a smoothed curve between readings, an extrapolation, a "
        "line, zone or label that is not in the brief or the files)? are the "
        "names the source's own words? does a legend repeat names already on "
        "the data? Then the lettering: overlaps, what " + checked_in
        + " lists (judged by the pixels drawn; the look decides), units, "
        "scale")
    small = [c for c in lettering if c.get("texts_under_7_pt")]
    if small:
        look += (", and lettering that prints under 7 pt (" + "; ".join(
            f"{c['file']}: {c['says']}" for c in small) + "): enlarge it or "
            "draw the figure at the width it is printed")
    look += ". "
    if draw is not None and draw >= DRAW_CAP:
        return look + draw_cap_note(draw)
    redraw = ("Redraw only to fix what the look found, under the same name "
              + ("(a draft under another name is retired with replaces=), "
                 if renamed_drafts else "(the same output_path), ")
              + "at most twice in all")
    if draw is not None:
        redraw += (f" (this was draw {draw} of at most {DRAW_CAP} of this "
                   "figure this turn)")
    return look + redraw + ". Paste html_img_tag into report HTML."


def _run_in_box(code, data_json, name, root, limit, cfg, notes, extra=None):
    extra = extra or {}
    from funhouse_agent import progress_notes
    key = progress_notes.current_key()
    slots = _slots(cfg["at_once"])
    if not slots.acquire(timeout=max(60.0, limit)):
        return {"status": "error",
                "error": (f"{cfg['at_once']} other drawings were running for "
                          "the whole wait; try again shortly")}
    tmp = tempfile.mkdtemp(prefix="plotbox_")
    t0 = time.monotonic()
    timed_out = False
    out = err = b""
    proc = None
    try:
        before = _snapshot(root)
        marker = f"@@PLOT_BOX_{secrets.token_hex(8)}@@"
        payload = json.dumps({
            "code": code, "data": json.loads(data_json),
            "name": name or "", "root": root, "tmp": tmp, "marker": marker,
            "print_width_in": extra.get("print_width_in"),
            "sys_path": [p for p in sys.path if p and os.path.isdir(p)],
            "limits": {"memory_mb": cfg["memory_mb"],
                       "cpu_s": int(limit) + 60, "fsize_mb": 200,
                       "max_files": MAX_FILES},
        }).encode("utf-8")
        kwargs: Dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = getattr(subprocess,
                                              "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            [sys.executable, "-I", "-B", CHILD], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root,
            env=_child_env(tmp), **kwargs)
        deadline = t0 + limit
        first = True
        while True:
            wait = max(0.1, min(NOTE_EVERY_S, deadline - time.monotonic()))
            try:
                out, err = proc.communicate(input=payload if first else None,
                                            timeout=wait)
                break
            except subprocess.TimeoutExpired:
                first = False
                elapsed = time.monotonic() - t0
                if elapsed >= limit:
                    timed_out = True
                    _kill(proc)
                    out, err = proc.communicate()
                    break
                progress_notes.set_note(
                    f"drawing with code: {_minutes(elapsed)} so far, "
                    f"stops at {_minutes(limit)}", key)
    except OSError as exc:
        return {"status": "error",
                "error": f"the box could not start: {exc}"}
    finally:
        if proc is not None and proc.poll() is None:
            _kill(proc)
        slots.release()
        progress_notes.clear_note(key)
        shutil.rmtree(tmp, ignore_errors=True)
    seconds = round(time.monotonic() - t0, 1)

    text = out.decode("utf-8", errors="replace")
    child: Dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith(marker):
            try:
                child = json.loads(line[len(marker):])
            except ValueError:
                child = {}
    after = _snapshot(root)
    made = sorted(rel for rel, sig in after.items()
                  if before.get(rel) != sig)

    res: Dict[str, Any] = {"seconds": seconds, "time_limit_s": int(limit),
                           "memory_mb": cfg["memory_mb"]}
    images, charts, others = [], [], []
    looks = child.get("look_copies") if isinstance(
        child.get("look_copies"), dict) else {}
    for rel in made:
        path = os.path.join(root, rel)
        low = rel.lower()
        if low.endswith(".plotly.json"):
            entry = {"file": rel, "output_path": path}
            try:
                with open(path, encoding="utf-8") as fh:
                    if "data" not in json.load(fh):
                        entry["warning"] = "not a Plotly figure (no 'data')"
            except (OSError, ValueError):
                entry["warning"] = "not valid JSON; the chat cannot show it"
            charts.append(entry)
        elif low.endswith(_IMAGE_EXTS):
            entry = {"file": rel, "output_path": path,
                     "html_img_tag": (f'<img src="{path}" alt="{rel}" '
                                      'style="width:100%;max-width:640px;">')}
            size = _png_size(path) if low.endswith(".png") else None
            if size:
                entry["width_px"], entry["height_px"] = size
            look = looks.get(rel.replace(os.sep, "/")) or looks.get(rel)
            if isinstance(look, str) and os.path.isfile(
                    os.path.join(root, look)):
                # the report copy is at the dpi asked; a look reads this
                # one, which one look takes whole (live wave w9)
                entry["look_file"] = look
                look_size = _png_size(os.path.join(root, look))
                if look_size:
                    entry["look_px"] = list(look_size)
            images.append(entry)
        else:
            others.append({"file": rel, "output_path": path})
    if images:
        res["images"] = images
    if charts:
        res["charts"] = charts
    if others:
        res["files"] = others
    for k in ("stdout", "stderr"):
        if child.get(k):
            res["printed" if k == "stdout" else "warnings"] = \
                child[k][-3000:]
    if child.get("refused"):
        res["refused"] = child["refused"]
    if child.get("layout"):
        # measured from the drawing itself, not from a look
        res["layout_check"] = child["layout"]
    lettering = [c for c in child.get("lettering") or []
                 if isinstance(c, dict) and c.get("says")]
    if lettering:
        # the lettering's size as printed (live wave w9: 4.6-5.6 pt)
        res["lettering"] = lettering
    if child.get("auto_saved"):
        notes.append("figures left open were saved as "
                     + ", ".join(child["auto_saved"]))
    stand_ins = "; ".join(
        f"{s.get('file')}: {s.get('why')}"
        for s in child.get("not_saved") or [] if isinstance(s, dict))
    if stand_ins and (images or charts or child.get("error")):
        # with nothing saved and no error the error itself says so below
        notes.append("Not saved, as a stand-in: " + stand_ins + ". "
                     + figures.STAND_IN_NOTE)
    if child:
        # kept for the record: whether the OS limits were in force, and how
        # the time split between starting the box and drawing
        res["box"] = {"limits_applied": child.get("limits_applied") or {},
                      "run_seconds": child.get("run_seconds"),
                      "start_seconds": (round(seconds - child["run_seconds"],
                                              1)
                                        if isinstance(child.get(
                                            "run_seconds"), (int, float))
                                        else None)}
        if child.get("preload_failed"):
            res["box"]["preload_failed"] = child["preload_failed"]
        if child.get("layout_seconds"):
            # the pixel layout check's time per saved figure (GM-108)
            res["box"]["layout_check_seconds"] = child["layout_seconds"]
        if not child.get("limits_applied"):
            notes.append("no OS memory or CPU limit could be set on this "
                         "host; only the app's time limit held")
    if timed_out:
        res["status"] = "error"
        res["error"] = (
            f"The code ran past its time limit ({_minutes(limit)}) and was "
            "stopped. If the figure is genuinely heavy, call again with a "
            f"larger time_limit_s (up to {cfg['max_seconds']} s here); "
            "otherwise look for a loop that never ends.")
    elif not child:
        res["status"] = "error"
        tail = err.decode("utf-8", errors="replace").strip()[-1500:]
        res["error"] = (
            "The box stopped without a report (exit code "
            f"{proc.returncode if proc else '?'}): most often the memory "
            f"ceiling ({cfg['memory_mb']} MB) -- draw from fewer points or a "
            "coarser grid." + (f"\n{tail}" if tail else ""))
    elif child.get("error"):
        res["status"] = "error"
        res["error"] = child["error"]
    else:
        res["status"] = "success" if (images or charts) else "error"
        if not (images or charts) and stand_ins:
            # every figure the code drew held a place (GM-190)
            res["error"] = ("No figure was saved, each a stand-in: "
                            + stand_ins + ". "
                            + figures.STAND_IN_NOTE)
        elif not (images or charts):
            res["error"] = ("The code ran but saved no figure: call "
                            "save_figure(fig, name) (or leave the figure "
                            "open and it is saved for you).")
    if res.get("status") == "error":
        # nothing to look at yet: fix the error first (live wave w8b)
        res.pop("layout_check", None)
        res.pop("lettering", None)
        if images:
            notes.append("the run failed: any image it saved before the "
                         "error is a partial drawing; fix the error and draw "
                         "again before looking")
        if extra.get("replaces"):
            notes.append("replaces: nothing was retired, because this draw "
                         "failed")
    else:
        notes.extend(_settle_charts(images, set(made), root))
        # the figure is one thing across the turn: its draws by any helper
        # are counted, and the code that drew it is kept beside it (GM-151)
        booked = figures.record_draw(
            [i["file"] for i in images] + [c["file"] for c in charts],
            "plot_code", root, replaces=extra.get("replaces"), code=code,
            data_json=data_json)
        kept = {figures.figure_stem(f["file"]).lower(): f
                for f in booked["figures"]}
        for entry in images + charts:
            fig = kept.get(figures.figure_stem(entry["file"]).lower()) or {}
            if fig.get("code_file"):
                entry["code_file"] = fig["code_file"]
        continued = sorted({f["continues"] for f in booked["figures"]
                            if f.get("continues")})
        if continued:
            res["continues"] = continued
        if extra.get("replaces"):
            gone = retire_drafts(extra["replaces"], root=root,
                                 keep=[i["output_path"] for i in images]
                                 + [c["output_path"] for c in charts])
            res.update(gone)
        notes.extend(booked_notes(booked))
        if images:
            draw = drawn_count(extra.get("draw"), booked["draw"])
            if draw is not None:
                res["draw"] = draw
            res["next"] = next_step(draw, lettering)
    if charts:
        # live wave w8: the chart the chat showed had lost lines the PNG had,
        # and no look had checked it
        notes.append(
            "The chat shows each chart in place of the PNG with the same "
            "name, and a look at the PNG does not check the chart. Save a "
            "chart only when an interactive chart is wanted, drawn from the "
            "same numbers; Plotly leaves out lines and shaded bands added to "
            "a subplot that has no data.")
    if notes:
        res["notes"] = notes
    return res


def _chart_lines_note() -> str:
    """What the description says of ``chart_lines`` in the box: only where
    the installed reference library has it (BACKLOG GM-153)."""
    try:
        from geotech_references.chart_lines import chart_names
        names = ", ".join(chart_names())
    except Exception:                                      # noqa: BLE001
        return ""
    return (
        "chart_lines(name, x_range, y_range) gives a standard chart's lines "
        f"as the reference prints them ({names}): each line's points over "
        "the range, its equation and dash, the zones with their names and "
        "where to write them (hatched: true for the CL-ML zone), the axis "
        "names, and "
        "the source and page for the caption; chart_line_value(name, line, "
        "x) reads one line at x (chart_line_value('uscs_plasticity', "
        "'A-line', 24) is 4.0). Draw a standard line from these, never from "
        "memory; ")


def plot_code_description(cfg: Optional[Dict[str, int]] = None) -> str:
    """The model-facing description, with this host's limits."""
    cfg = cfg or settings()
    libs = ", ".join(sorted(ALLOWED_IMPORTS))
    return (
        "Draw a figure by writing matplotlib (or Plotly) code, run in a "
        "LOCKED BOX: for what plot_data cannot draw -- a section with "
        "layers filled and a wall or pile drawn to scale, pressure diagrams "
        "beside a wall, side-by-side panels sharing one elevation axis, "
        "contours or a coloured grid, a plan with symbols and dimension "
        "lines. The code runs in its own process: it may import "
        "only "
        f"{libs}; it reads the conversation's files by name (CSV, JSON, "
        "Excel through pandas) and writes figure, chart and data files "
        "there, nothing else; no network. In the code: `data` is the JSON "
        "object you pass as data; save_figure(fig, name) saves a matplotlib "
        "figure as <name>.png (.svg or .pdf by the name's extension) at the "
        "dpi asked (200 unless save_figure(..., dpi=) says) and returns the "
        "file name; a PNG over 2,048 px on its long side also gets a copy "
        "that size for looks (look_file); "
        "only when an interactive chart is wanted, save_chart(fig, name) "
        "saves a Plotly figure as the chat's chart <name>.plotly.json, shown "
        "in place of the PNG with the same name and NOT checked by a look at "
        "the PNG (so draw both from the same numbers); a chart an earlier "
        "draw left under a PNG's name is removed when the PNG is redrawn; "
        "scale_bar(ax, x0, y, length, height, units) and north_arrow(ax, "
        "(x, y), north_deg) draw the app's plan furniture; "
        f"{_chart_lines_note()}"
        "a figure left "
        "open is saved as <name>.png. Draw as for a report: every number "
        "from the data or the files (never made up), labels on the data, "
        "units on every axis (stresses in kPa, forces in kN), sections to "
        "scale (ax.set_aspect('equal')), a profile against elevation rising "
        "upward (depth downward only when no elevations are given, and say "
        "so; never both on one profile), a wall or pile drawn to scale with "
        "its diagrams beside it (the diagrams alone only when it is not yet "
        "known, and say so), test results as points and an assumed design "
        "profile as a line named as one, a plan as a drawing (no axes or "
        "grid; a scale bar and a north arrow), no decorative ink; arrows "
        "point the way the force or load acts; dimensions carry their "
        "units; readings joined by straight lines or shown as points, never "
        "a stair unless the quantity itself steps; nothing on the figure "
        "that the data or the files do not give (no test standard, "
        "classification or citation added from memory); lettering "
        "at least 7 pt at the size the figure is printed; to clear a label, "
        "move the label, never the drawing. Look after each draw, and redraw "
        "a figure at most twice (its drafts by plot_code and plot_data, "
        "under any name, count together, and so do its redraws by another "
        "helper this turn): then hand back the best and say what is left. "
        "The code that drew each figure is kept beside it (code_file, in "
        "the hidden .figure_code folder): to change a figure drawn earlier, "
        "change that code and draw under the same name, so the redraw takes "
        "its place. "
        "replaces: file names of drafts this turn drew under another name, "
        "by any helper (a picture or chart, or the name without its "
        "extension), removed once this draw succeeds, so the chat shows "
        "only the final figure. "
        "print_width_in: the width the figure is printed at (6.5 in unless "
        "given), for the lettering's printed size. time_limit_s: "
        f"{cfg['seconds']} s unless you ask for more, up to "
        f"{cfg['max_seconds']} s for a heavy figure; memory up to "
        f"{cfg['memory_mb']} MB. Returns the saved files (each PNG with its "
        "html_img_tag for report HTML), layout_check (judged by the pixels "
        "drawn: lettering that overlaps or runs into other lettering, that a "
        "line, an edge or markers show through, or that an edge cuts off), "
        "lettering (its smallest size as printed, e.g. '5.6 pt "
        "at 6.5 in'), anything the code printed, and on a failure the error "
        "with your code's line; next says what to look at.")


def make_plot_code_tool():
    """``plot_code`` as a LangChain tool (refusing unknown arguments)."""
    from langchain_core.tools import StructuredTool
    from funhouse_agent.deep.strict_args import strict_tool

    def plot_code(code: str, data: Optional[dict] = None, name: str = "",
                  time_limit_s: Optional[int] = None,
                  replaces: Optional[List[str]] = None,
                  print_width_in: Optional[float] = None) -> str:
        """Run plotting code in the locked box (see the description)."""
        before = draws_so_far()
        return json.dumps(run_plot_code(
            code, data=data, name=name, time_limit_s=time_limit_s,
            replaces=replaces, print_width_in=print_width_in,
            draw=None if before is None else before + 1), default=str)

    return strict_tool(StructuredTool.from_function(
        plot_code, name="plot_code", description=plot_code_description()))


__all__ = ["next_step", "run_plot_code", "make_plot_code_tool", "plot_code_description",
           "available", "settings", "check_code", "ALLOWED_IMPORTS",
           "MAX_CODE_CHARS", "SETTINGS_ENVS", "retire_drafts",
           "draws_so_far", "draw_cap_note", "drawn_count", "booked_notes",
           "drew_figure", "DRAW_TOOLS", "DRAW_CAP", "CAPPED_HELPER",
           "figures"]
