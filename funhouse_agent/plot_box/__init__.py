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

OFF until switched on (``GEOTECH_PLOT_CODE=1``; owner rule: new agent
behaviour waits behind a switch until measured). Offered to the figure
helper and the ``calc`` helper. Settings:

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
from typing import Any, Dict, Optional

from funhouse_agent.plot_box.child import (ALLOWED_IMPORTS, check_code,
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


def run_plot_code(code: str, data: Any = None, name: str = "",
                  time_limit_s: Optional[float] = None,
                  root: Optional[str] = None) -> Dict[str, Any]:
    """Run plotting ``code`` in the locked box; the files it made and how
    it went, as a JSON-ready dict (never raises for the code's own faults).
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

    from funhouse_agent.drawing_wait import drawing
    # A look at a figure in this folder waits until the box has written it
    # (live wave w8: a draw and a look in the same step read the old PNG).
    with drawing(root):
        return _run_in_box(code, data_json, name, root, limit, cfg, notes)


def _run_in_box(code, data_json, name, root, limit, cfg, notes):
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
    if child.get("auto_saved"):
        notes.append("figures left open were saved as "
                     + ", ".join(child["auto_saved"]))
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
        if not (images or charts):
            res["error"] = ("The code ran but saved no figure: call "
                            "save_figure(fig, name) (or leave the figure "
                            "open and it is saved for you).")
    if res.get("status") == "error":
        # nothing to look at yet: fix the error first (live wave w8b)
        res.pop("layout_check", None)
        if images:
            notes.append("the run failed: any image it saved before the "
                         "error is a partial drawing; fix the error and draw "
                         "again before looking")
    elif images:
        res["next"] = (
            "LOOK at each PNG with analyze_image, in a step after this one, "
            "and check it against what the figure must show (labels on the "
            "data, units, scale, nothing overlapping), and at what "
            "layout_check lists (boxes around lettering, measured: the look "
            "decides); then call plot_code again with the same name to "
            "redraw. Paste html_img_tag into report HTML.")
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


def plot_code_description(cfg: Optional[Dict[str, int]] = None) -> str:
    """The model-facing description, with this host's limits."""
    cfg = cfg or settings()
    libs = ", ".join(sorted(ALLOWED_IMPORTS))
    return (
        "Draw a figure by writing matplotlib (or Plotly) code, run in a "
        "LOCKED BOX: for what plot_data cannot draw -- a section with "
        "layers filled and a wall or pile drawn to scale, pressure diagrams "
        "beside a wall, side-by-side panels sharing one depth axis, contours "
        "or a coloured grid, a plan with symbols, leaders and dimension "
        "lines. The code runs in its own process: it may import only "
        f"{libs}; it reads the conversation's files by name (CSV, JSON, "
        "Excel through pandas) and writes figure, chart and data files "
        "there, nothing else; no network. In the code: `data` is the JSON "
        "object you pass as data; save_figure(fig, name) saves a matplotlib "
        "figure as <name>.png (.svg or .pdf by the name's extension) and "
        "returns the file name (a PNG at most 2,048 px on its long side); "
        "only when an interactive chart is wanted, save_chart(fig, name) "
        "saves a Plotly figure as the chat's chart <name>.plotly.json, shown "
        "in place of the PNG with the same name and NOT checked by a look at "
        "the PNG (so draw both from the same numbers); "
        "scale_bar(ax, x0, y, length, height, units) and north_arrow(ax, "
        "(x, y), north_deg) draw the app's plan furniture; a figure left "
        "open is saved as <name>.png. Draw as for a report: every number "
        "from the data or the files (never made up), labels on the data, "
        "units on every axis, sections to scale (ax.set_aspect('equal')), "
        "depth increasing downward, a plan as a drawing (no axes or grid; a "
        "scale bar and a north arrow), no decorative ink; arrows point the "
        "way the force or load acts; dimensions carry their units; lettering "
        "at least 7 pt at the size the figure is printed; to clear a label, "
        "move the label, never the drawing. Look after each draw, and redraw "
        "at most twice: then hand back the best and say what is left. "
        "time_limit_s: "
        f"{cfg['seconds']} s unless you ask for more, up to "
        f"{cfg['max_seconds']} s for a heavy figure; memory up to "
        f"{cfg['memory_mb']} MB. Returns the saved files (each PNG with its "
        "html_img_tag for report HTML), layout_check (lettering that overlaps "
        "other lettering, or that a plotted line runs through, measured from "
        "the drawing), anything the code printed, and on a "
        "failure the error with your code's line. Then look at the PNG with "
        "analyze_image and redraw until it is right.")


def make_plot_code_tool():
    """``plot_code`` as a LangChain tool (refusing unknown arguments)."""
    from langchain_core.tools import StructuredTool
    from funhouse_agent.deep.strict_args import strict_tool

    def plot_code(code: str, data: Optional[dict] = None, name: str = "",
                  time_limit_s: Optional[int] = None) -> str:
        """Run plotting code in the locked box (see the description)."""
        return json.dumps(run_plot_code(code, data=data, name=name,
                                        time_limit_s=time_limit_s),
                          default=str)

    return strict_tool(StructuredTool.from_function(
        plot_code, name="plot_code", description=plot_code_description()))


__all__ = ["run_plot_code", "make_plot_code_tool", "plot_code_description",
           "available", "settings", "check_code", "ALLOWED_IMPORTS",
           "MAX_CODE_CHARS", "SETTINGS_ENVS"]
