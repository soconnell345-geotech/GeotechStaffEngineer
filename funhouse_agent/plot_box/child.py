"""The locked box's child process: runs one piece of plotting code.

Started by :mod:`funhouse_agent.plot_box` as ``python -I -B child.py`` with
the payload as JSON on stdin, a scrubbed environment (no keys, no tokens) and
the conversation's folder as its working directory. It reports back ONE line
on stdout: a marker the parent chose, then JSON.

The layers, in the order they go up:

1. a STATIC CHECK of the code (:func:`check_code`): imports only from
   :data:`ALLOWED_IMPORTS`; no ``eval`` / ``exec`` / ``getattr`` / dunder
   attributes (the classic ways out of a Python namespace). A speed bump,
   not the wall: the parent runs it too, so a refused script costs no
   process.
2. OS LIMITS on this process: memory, CPU time, file size, open files, no
   child processes (``resource`` on Linux; a job object on Windows).
3. The libraries the code names are imported and drawn with once, so their
   lazy loading is done before the guard.
4. An AUDIT HOOK (:pep:`578`), which cannot be removed once added and sees
   every file open, directory listing, process start, socket and ``ctypes``
   call made by the code AND by the libraries it calls: reads are allowed in
   the conversation folder and the installed libraries (plus font and time
   zone files), writes only in the conversation folder and only figure,
   chart and data files, never a process, a socket or ``ctypes``.
5. The code runs with its own builtins: ``import`` refuses anything outside
   :data:`ALLOWED_IMPORTS` (and a library's own imports handed on by name);
   ``eval``, ``exec``, ``getattr`` and the like are not there.

Stdlib only at the top: the heavy imports happen after the limits are set.
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import time
import types

#: The box's own clock, held before the code runs: ``time`` is one of the
#: code's modules, and the code may replace what it holds.
_monotonic = time.monotonic

#: Top-level modules plotting code may import: the plotting and number
#: libraries the owner named (matplotlib, Plotly, numpy, pandas), SciPy for
#: interpolation and statistics, and the standard library's maths and text
#: modules. Nothing that reaches the network, processes or the system, and
#: nothing that turns a string into code or into an attribute (``typing``'s
#: type hints, ``string.Formatter``).
#:
#: ``time`` (BACKLOG GM-152; live wave w12 had a script refused for it): it
#: reads clocks and formats dates, which ``datetime`` already offers, and
#: waits (``sleep``), which costs no more than a loop: the parent stops the
#: box at its time limit either way. The one thing in it that reaches the
#: system, setting the clock (``clock_settime``, which a box run as root
#: could do and which raises no audit event), is refused by name
#: (:data:`BLOCKED_ATTRIBUTES`) and taken out of the module before the code
#: runs.
ALLOWED_IMPORTS = frozenset({
    "numpy", "pandas", "matplotlib", "mpl_toolkits", "plotly", "scipy",
    "math", "cmath", "statistics", "random", "itertools", "functools",
    "collections", "json", "csv", "datetime", "decimal", "fractions", "re",
    "textwrap", "enum", "copy", "bisect", "heapq", "colorsys", "numbers",
    "io", "time",
})

#: Names the code may not use: the ways to run new code, to reach a module's
#: or a frame's insides by a name in a string, or to wait for a keyboard.
BLOCKED_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "help", "exit",
    "quit", "__builtins__", "__loader__", "__spec__",
})

#: Attribute names the code may not use (besides every dunder but the three
#: in :data:`ALLOWED_DUNDERS`): the attributes through which a library hands
#: out the system, a frame or a code object, and the evaluators that read a
#: string as code (``DataFrame.eval`` / ``query``). The audit hook stands
#: behind these whatever the code reaches.
BLOCKED_ATTRIBUTES = frozenset({
    "os", "sys", "subprocess", "system", "popen", "environ", "socket",
    "ctypes", "ctypeslib", "importlib", "builtins", "shutil", "pickle",
    "marshal", "gc", "inspect", "f_globals", "f_locals", "f_builtins",
    "f_back", "f_code", "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code", "tb_frame", "tb_next", "co_code", "eval", "query",
    "attrgetter", "methodcaller", "format_map", "read_pickle", "to_pickle",
    "load_library", "Formatter", "get_field", "vformat", "get_type_hints",
    "clock_settime", "clock_settime_ns",
})
ALLOWED_DUNDERS = frozenset({"__init__", "__name__", "__doc__"})

#: Files the code may write (by extension) in the conversation folder, and
#: the ones it may also overwrite when they were there before the run (a
#: redrawn figure). ``.json`` covers a Plotly chart (``.plotly.json``).
WRITE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".webp", ".json",
    ".csv", ".txt"})
OVERWRITE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"})
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".afm", ".pfb", ".pfa", ".woff",
                   ".woff2")

MAX_CODE_CHARS = 100_000


# ---------------------------------------------------------------------------
# 1. the static check (the parent imports this too)
# ---------------------------------------------------------------------------

def check_code(code: str) -> list:
    """The reasons the code is refused before it runs; empty when none."""
    if not isinstance(code, str) or not code.strip():
        return ["no code given"]
    if len(code) > MAX_CODE_CHARS:
        return [f"the code is {len(code):,} characters; the box takes at "
                f"most {MAX_CODE_CHARS:,} (put large data in `data` or a "
                "file in the conversation folder)"]
    try:
        tree = ast.parse(code, filename="<plot_code>")
    except SyntaxError as exc:
        return [f"line {exc.lineno}: syntax error: {exc.msg}"]
    problems = []
    allowed = ", ".join(sorted(ALLOWED_IMPORTS))

    def bad(node, what):
        problems.append(f"line {getattr(node, 'lineno', '?')}: {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    bad(node, f"imports '{alias.name}'; plotting code may "
                              f"import only: {allowed}")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if node.level or top not in ALLOWED_IMPORTS:
                bad(node, f"imports from '{'.' * node.level}"
                          f"{node.module or ''}'; plotting code may import "
                          f"only: {allowed}")
            for alias in node.names:
                # ``from matplotlib.cbook import os``: a library's own
                # import handed on by name
                if (alias.name in BLOCKED_ATTRIBUTES
                        or alias.name.startswith("__")):
                    bad(node, f"imports '{alias.name}' from "
                              f"'{node.module}', which the box does not "
                              "allow")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            bad(node, f"uses '{node.id}', which the box does not offer")
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if (name.startswith("__") and name.endswith("__")
                    and name not in ALLOWED_DUNDERS):
                bad(node, f"uses the attribute '{name}', which the box does "
                          "not allow")
            elif name in BLOCKED_ATTRIBUTES:
                bad(node, f"uses the attribute '.{name}', which the box does "
                          "not allow")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ".__" in node.value or "{__" in node.value:
                bad(node, "a string names a dunder attribute, which the box "
                          "does not allow")
    return problems


def imported_modules(code: str) -> list:
    """Full names of the modules the code imports (allowed ones only)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module \
                and not node.level:
            names.append(node.module)
            # ``from scipy import interpolate``: a submodule by name
            names.extend(f"{node.module}.{a.name}" for a in node.names
                         if a.name != "*")
    return [n for n in dict.fromkeys(names)
            if n.split(".")[0] in ALLOWED_IMPORTS]


# ---------------------------------------------------------------------------
# 2. OS limits
# ---------------------------------------------------------------------------

def _posix_limits(memory_mb: int, cpu_s: int, fsize_mb: int) -> dict:
    import resource
    applied = {}

    def lim(name, value):
        res = getattr(resource, name, None)
        if res is None:
            return
        try:
            _soft, hard = resource.getrlimit(res)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            # soft and hard both: the code cannot raise it again
            resource.setrlimit(res, (value, value))
            applied[name] = value
        except (ValueError, OSError):
            pass

    lim("RLIMIT_AS", int(memory_mb) * 1024 * 1024)
    lim("RLIMIT_CPU", int(cpu_s))
    lim("RLIMIT_FSIZE", int(fsize_mb) * 1024 * 1024)
    lim("RLIMIT_CORE", 0)
    lim("RLIMIT_NOFILE", 256)
    return applied


def _posix_no_children() -> dict:
    """After the libraries are loaded: no new processes (or threads)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        return {"RLIMIT_NPROC": 0}
    except (ImportError, AttributeError, ValueError, OSError):
        return {}


def _windows_job(memory_mb: int, cpu_s: int) -> dict:
    """A job object holding only this process: a memory ceiling, a CPU-time
    ceiling and no child processes. Best effort; nothing else runs before
    it."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,
                                                 wintypes.HANDLE]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount",
                "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic),
                        ("IoInfo", Counters),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return {}
        info = Extended()
        # process time | process memory | active process count | die on
        # unhandled exception | kill on job close
        info.BasicLimitInformation.LimitFlags = (0x2 | 0x100 | 0x8 | 0x400
                                                 | 0x2000)
        # 100-ns ticks
        info.BasicLimitInformation.PerProcessUserTimeLimit = (
            int(cpu_s) * 10_000_000)
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = int(memory_mb) * 1024 * 1024
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                           ctypes.sizeof(info)):
            return {}
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return {}
        # The handle stays open for the life of the process (kill on close).
        globals()["_JOB_HANDLE"] = job
        return {"job_memory_bytes": info.ProcessMemoryLimit,
                "job_cpu_seconds": int(cpu_s), "job_active_processes": 1}
    except Exception:                                      # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# 3-4. the libraries, then the guard
# ---------------------------------------------------------------------------

def _preload(modules: list) -> list:
    """Import what the code names, and draw once, before the guard."""
    import importlib
    failed = []
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:                                  # noqa: BLE001
            # ``from matplotlib.patches import Polygon`` names a class, not a
            # module: no failure. Anything else, the code's own import says.
            parent, _, last = name.rpartition(".")
            if not (parent and hasattr(sys.modules.get(parent), last)):
                failed.append(name)
    # Read once now, not under the guard: the system's MIME table (in /etc
    # on Linux; openpyxl asks for it when pandas reads an Excel file) ...
    import mimetypes
    mimetypes.init()
    # ... and the Excel reader itself when the code reads a workbook.
    if any(m.startswith("pandas") for m in modules):
        for extra in ("openpyxl",):
            try:
                importlib.import_module(extra)
            except Exception:                              # noqa: BLE001
                pass
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1], label="a")
    ax.annotate("b", (0.5, 0.5), xytext=(4, 4), textcoords="offset points",
                arrowprops={"arrowstyle": "->"})
    ax.fill_between([0, 1], [0, 0], [1, 1], hatch="//", alpha=0.2)
    ax.legend()
    ax.set_title("c")
    fig.savefig(io.BytesIO(), format="png")
    plt.close(fig)
    return failed


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _inside(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def _read_roots(root: str) -> list:
    """Folders the libraries read from (never the conversation folder's
    parents, a drive's root or the home folder itself)."""
    import sysconfig
    cands = {sys.prefix, sys.base_prefix, sys.exec_prefix,
             sys.base_exec_prefix}
    try:
        cands.update(v for k, v in sysconfig.get_paths().items()
                     if k in ("stdlib", "platstdlib", "purelib", "platlib"))
    except Exception:                                      # noqa: BLE001
        pass
    for name in ("numpy", "matplotlib", "pandas", "plotly", "scipy",
                 "mpl_toolkits", "PIL", "kiwisolver", "contourpy",
                 "dateutil", "pytz", "tzdata", "pyparsing", "cycler",
                 "fontTools", "narwhals", "openpyxl", "et_xmlfile",
                 "packaging", "six"):
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None)
        if f:
            cands.add(os.path.dirname(os.path.dirname(os.path.abspath(f))))
    mpl = sys.modules.get("matplotlib")
    if mpl is not None:
        for fn in ("get_configdir", "get_cachedir", "get_data_path"):
            try:
                cands.add(getattr(mpl, fn)())
            except Exception:                              # noqa: BLE001
                pass
    home = _norm(os.path.expanduser("~"))
    roots = []
    for c in cands:
        if not c:
            continue
        n = _norm(c)
        if (_inside(root, n) or n == home
                or os.path.dirname(n.rstrip(os.sep)) == n.rstrip(os.sep)
                or len(n.rstrip(os.sep)) <= 3):
            continue
        roots.append(n)
    for z in ("/usr/share/zoneinfo", "/usr/lib/zoneinfo",
              "/usr/share/lib/zoneinfo"):
        roots.append(_norm(z))
    return sorted(set(roots))


#: Events refused outright (a prefix ending in '.' refuses the family).
_DENY_EVENTS = (
    "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "os.startfile", "os.kill", "os.killpg",
    "os.add_dll_directory", "pty.", "socket.", "ctypes.", "_winapi.",
    "winreg.", "http.client.", "urllib.", "ftplib.", "smtplib.", "imaplib.",
    "poplib.", "nntplib.", "telnetlib.", "webbrowser.", "sqlite3.",
    "sys._current_frames", "sys.remote_exec", "cpython.remote_debugger",
    "gc.get_objects", "gc.get_referrers", "gc.get_referents",
    "code.__new__", "pickle.find_class", "builtins.input",
    "builtins.breakpoint", "resource.setrlimit", "resource.prlimit",
    "os.remove", "os.rmdir", "os.rename", "shutil.rmtree", "shutil.move",
    "os.link", "os.symlink", "os.chmod", "os.chown", "os.chflags",
    "os.lchmod", "os.lchown", "os.lchflags", "os.truncate", "os.utime",
)
#: Modules a library may not newly import once the guard is up.
_DENY_IMPORTS = frozenset({"ctypes", "_ctypes", "multiprocessing",
                           "_multiprocessing", "pty", "winreg", "_winreg"})


def make_guard(root: str, tmp: str, read_roots: list, max_files: int):
    """The audit hook and the record it keeps (``written``, ``refused``).

    Everything the hook decides by is bound into it as keyword defaults
    when it is made: strings, tuples and a frozenset, and the functions it
    calls. Nothing the code can reach repoints them, and the hook has no
    off switch: once added it stays for the life of the process.
    """
    root = _norm(root)
    tmp = _norm(tmp) if tmp else ""
    existing = set()
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            existing.add(os.path.normcase(os.path.join(base, f)))
    record = {"written": [], "refused": []}

    def hook(event, args, *, _root=root, _tmp=tmp,
             _reads=tuple(read_roots), _existing=frozenset(existing),
             _max=int(max_files), _written=record["written"],
             _refused=record["refused"], _sep=os.sep,
             _realpath=os.path.realpath, _normcase=os.path.normcase,
             _join=os.path.join, _fspath=os.fspath, _fsdecode=os.fsdecode,
             _splitext=os.path.splitext, _dirname=os.path.dirname,
             _pathlike=os.PathLike, _isinstance=isinstance, _str=str,
             _bytes=bytes, _int=int, _len=len, _any=any, _all=all,
             _next=next, _sorted=sorted,
             _error=PermissionError, _missing=ModuleNotFoundError,
             _wflags=(os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT
                      | os.O_TRUNC),
             _deny=_DENY_EVENTS, _deny_imports=_DENY_IMPORTS,
             _write_exts=WRITE_EXTENSIONS,
             _overwrite_exts=OVERWRITE_EXTENSIONS,
             _fonts=FONT_EXTENSIONS,
             _special=(os.path.normcase(os.path.realpath(os.devnull)),
                       os.path.normcase("/dev/urandom"),
                       os.path.normcase("/etc/localtime")),
             _words=tuple(_EVENT_WORDS.items())):

        def inside(p, base):
            return p == base or p.startswith(base.rstrip(_sep) + _sep)

        def resolve(p):
            if _isinstance(p, _int):
                return None              # an fd already opened under guard
            try:
                p = _fspath(p)
            except TypeError:
                return ""
            if _isinstance(p, _bytes):
                p = _fsdecode(p)
            if not _isinstance(p, _str):
                return ""
            return _normcase(_realpath(_join(_root, p)))

        def readable(p):
            return (inside(p, _root) or (_tmp and inside(p, _tmp))
                    or p.endswith(_fonts) or p in _special
                    or _any(inside(p, r) for r in _reads))

        def refuse(what):
            if _len(_refused) < 20:
                _refused.append(what)
            raise _error("refused in the plotting box: " + what)

        if event == "open":
            path = args[0]
            mode = args[1] if _len(args) > 1 else None
            flags = args[2] if _len(args) > 2 else None
            p = resolve(path)
            if p is None:
                return
            if _isinstance(mode, _str):
                writing = _any(c in mode for c in "wax+")
            else:
                writing = _isinstance(flags, _int) and bool(flags & _wflags)
            if not p:
                refuse("opening a file by an unusual path")
            if not writing:
                if not readable(p):
                    refuse(f"reading '{path}' (outside the conversation "
                           "folder)")
                return
            if _tmp and inside(p, _tmp):
                return
            if not inside(p, _root):
                refuse(f"writing '{path}' (outside the conversation folder)")
            ext = _splitext(p)[1].lower()
            if ext not in _write_exts:
                refuse(f"writing '{path}': the box writes only figure, "
                       "chart and data files ("
                       + ", ".join(_sorted(_write_exts)) + ")")
            if (p in _existing and ext not in _overwrite_exts
                    and not p.endswith(".plotly.json")
                    and p not in _written):
                refuse(f"overwriting '{path}', which was there before this "
                       "run (write a new name)")
            if p not in _written:
                if _len(_written) >= _max:
                    refuse(f"writing more than {_max} files")
                _written.append(p)
            return

        if event in ("os.listdir", "os.scandir", "glob.glob", "glob.glob/2",
                     "os.mkdir", "os.chdir"):
            path = args[0] if args else None
            if event.startswith("glob."):
                pat = _str(path or "")
                cuts = [i for i in (pat.find("*"), pat.find("?"),
                                    pat.find("[")) if i >= 0]
                path = _dirname(pat[:cuts[0] if cuts else _len(pat)]) or "."
            if path is None:
                path = "."
            p = resolve(path)
            if p is None:
                return
            if event == "os.chdir":
                if p != _root:
                    refuse("changing folder")
                return
            if event == "os.mkdir":
                if not (inside(p, _root) or (_tmp and inside(p, _tmp))):
                    refuse(f"making the folder '{path}' (outside the "
                           "conversation folder)")
                return
            if not p or not readable(p):
                refuse(f"listing '{path}' (outside the conversation folder)")
            return

        if event == "import":
            if args and _str(args[0]).split(".")[0] in _deny_imports:
                if _len(_refused) < 20:
                    _refused.append(f"importing '{args[0]}'")
                # as if it were not installed: libraries guard optional
                # imports with ``except ImportError`` (mimetypes and winreg)
                raise _missing(f"No module named '{args[0]}' (refused in "
                               "the plotting box)")
            return

        if _tmp and event in ("os.remove", "os.rmdir", "os.rename"):
            paths = [resolve(a) for a in args[:2]
                     if _isinstance(a, (_str, _bytes, _pathlike))]
            if paths and _all(q and inside(q, _tmp) for q in paths):
                return
        for deny in _deny:
            if event == deny or (deny.endswith(".")
                                 and event.startswith(deny)):
                if event.startswith(("os.remove", "os.rmdir", "os.rename",
                                     "shutil.")):
                    words = "deleting, moving or renaming files"
                else:
                    head = event.split(".")[0]
                    words = _next((w for k, w in _words if k == head),
                                  "the operation")
                refuse(f"{words} ({event})")

    return hook, record


_EVENT_WORDS = {
    "subprocess": "starting a process", "os": "a system operation",
    "socket": "a network connection", "ctypes": "calling native code",
    "http": "a network connection", "urllib": "a network request",
    "ftplib": "a network connection", "smtplib": "sending mail",
    "webbrowser": "opening a browser", "sqlite3": "opening a database",
    "pickle": "unpickling objects", "gc": "inspecting memory",
    "code": "making code objects", "builtins": "waiting for input",
    "resource": "changing the box's limits", "shutil": "moving or "
    "deleting files", "_winapi": "a system operation", "winreg":
    "reading the registry", "pty": "starting a process",
}


# ---------------------------------------------------------------------------
# 5. the run
# ---------------------------------------------------------------------------

class _Capped(io.TextIOBase):
    """A text sink keeping the first ``limit`` characters."""

    def __init__(self, limit: int):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.cut = False

    def writable(self):
        return True

    def write(self, s):
        s = str(s)
        room = self.limit - self.size
        if room > 0:
            self.parts.append(s[:room])
            self.size += min(len(s), room)
        if len(s) > max(room, 0):
            self.cut = True
        return len(s)

    def text(self) -> str:
        out = "".join(self.parts)
        return out + ("\n[... cut]" if self.cut else "")


def _safe_stem(name, default: str) -> str:
    import re
    stem = os.path.basename(str(name or "").strip()) or default
    for ext in (".plotly.json", ".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .") or default
    return stem[:80]


#: The longest side of the copy a look reads: what one look at an image
#: takes whole (live wave w8: 200 dpi saved 2538 x 1546 px, and every look
#: was tiled).
MAX_PX = 2048
#: The longest side, and the highest dpi, of the copy for a report.
MAX_REPORT_PX = 6000
MAX_REPORT_DPI = 600
#: Where the look copy of a figure larger than :data:`MAX_PX` goes: the
#: conversation's tool scratch folder (no download card, not mirrored).
LOOK_DIR = ".scratch"
#: The width a figure is printed at when the call names none: a report
#: page's text width (8.5 in less two 1 in margins).
DEFAULT_PRINT_WIDTH_IN = 6.5
#: The smallest lettering a printed figure should carry (points).
MIN_PRINT_PT = 7.0


def _dpi_for(fig, dpi) -> int:
    """``dpi``, lowered so the figure's longest side stays within
    :data:`MAX_PX` pixels."""
    w, h = fig.get_size_inches()
    return max(40, min(int(dpi), int(MAX_PX / max(w, h, 1e-6))))


def _report_dpi(fig, dpi) -> int:
    """``dpi`` for the report copy (live wave w9: the look's 2,048 px cap
    also capped the deliverable), within :data:`MAX_REPORT_PX` and
    :data:`MAX_REPORT_DPI`."""
    w, h = fig.get_size_inches()
    return max(40, min(int(dpi), MAX_REPORT_DPI,
                       int(MAX_REPORT_PX / max(w, h, 1e-6))))


def _lettering(fig, fname: str, print_width_in: float,
               measured=None) -> dict:
    """The lettering's size as printed ``print_width_in`` wide (live wave
    w9: figures drawn 9.5-11 in wide printed at 4.6-5.6 pt on a report
    page, and nothing measured it): the smallest size, the width the figure
    was saved at, and how many texts print under :data:`MIN_PRINT_PT`.
    ``measured``: ``layout_ink.lettering_sizes(fig)``, when taken already."""
    sizes, saved_w = measured or _layout_ink.lettering_sizes(fig)
    if not sizes or saved_w <= 0:
        return {}
    scale = float(print_width_in) / max(saved_w, 1e-6)
    sizes = [s * scale for s in sizes]
    smallest = min(sizes)
    under = sum(1 for s in sizes if s < MIN_PRINT_PT - 0.05)
    return {"file": fname, "smallest_pt": round(smallest, 1),
            "print_width_in": float(print_width_in),
            "saved_width_in": round(saved_w, 2),
            "texts_under_7_pt": under,
            "says": (f"{smallest:.1f} pt at {float(print_width_in):g} in "
                     f"wide (saved {saved_w:.1f} in wide; {under} of "
                     f"{len(sizes)} texts under {MIN_PRINT_PT:g} pt)")}


def _load_layout_ink():
    """``geotech_common.layout_ink``, the layout check this box shares with
    ``plot_data``: imported as usual where this file is a module of the
    app; in the box (``python -I child.py``) loaded from its file, beside
    this package, so the rest of ``geotech_common`` stays out of the box."""
    if __name__ != "__main__":
        from geotech_common import layout_ink
        return layout_ink
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(os.path.dirname(here)),
                        "geotech_common", "layout_ink.py")
    if not os.path.isfile(path):
        spec = importlib.util.find_spec("geotech_common")
        path = os.path.join(list(spec.submodule_search_locations)[0],
                            "layout_ink.py")
    fspec = importlib.util.spec_from_file_location("_box_layout_ink", path)
    mod = importlib.util.module_from_spec(fspec)
    fspec.loader.exec_module(mod)
    return mod


#: The layout check judges lettering by the pixels drawn (GM-108; the
#: measures and their reasons are in ``geotech_common/layout_ink.py``).
_layout_ink = _load_layout_ink()
LINE_MIN_PX = _layout_ink.LINE_MIN_PX
VISIBLE_DIFF = _layout_ink.VISIBLE_DIFF
INK_ALPHA = _layout_ink.INK_ALPHA
TOUCH_EM = _layout_ink.TOUCH_EM
TOUCH_PT = _layout_ink.TOUCH_PT
BOX_OPAQUE = _layout_ink.BOX_OPAQUE
#: The box's budgets for the check: past either, "unchecked".
LAYOUT_SECONDS = _layout_ink.LAYOUT_SECONDS
LAYOUT_MB = _layout_ink.LAYOUT_MB
_Unchecked = _layout_ink._Unchecked
_Ink = _layout_ink._Ink
_ink_check = _layout_ink._ink_check
_leaf_kinds = _layout_ink._leaf_kinds
_grow = _layout_ink._grow
_meet = _layout_ink._meet
_ink_rows = _layout_ink._ink_rows
_figure_roles = _layout_ink._figure_roles


def _layout_check(fig, fname: str, dpi=200) -> dict:
    """The layout check (``layout_ink.layout_check``) on a figure as it is
    saved for a look (within :data:`MAX_PX`), on the box's budgets."""
    return _layout_ink.layout_check(fig, fname, _dpi_for(fig, dpi),
                                    seconds=LAYOUT_SECONDS, mb=LAYOUT_MB)


def _load_figure_rules():
    """``funhouse_agent.plot_box.figures``, whose stand-in rule the box
    applies before it saves (BACKLOG GM-190): imported as usual where this
    file is a module of the app; in the box loaded from its file beside
    this one, as ``layout_ink`` is. ``None`` when it cannot be loaded (the
    box then saves as before)."""
    try:
        if __name__ != "__main__":
            from funhouse_agent.plot_box import figures
            return figures
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "figures.py")
        fspec = importlib.util.spec_from_file_location("_box_figures", path)
        mod = importlib.util.module_from_spec(fspec)
        fspec.loader.exec_module(mod)
        return mod
    except Exception:                                      # noqa: BLE001
        return None


_figure_rules = _load_figure_rules()


def _visible_text(t) -> str:
    try:
        return t.get_text() if t is not None and t.get_visible() else ""
    except Exception:                                      # noqa: BLE001
        return ""


def _line_has_points(line) -> bool:
    try:
        import numpy as np
        xy = np.asarray(line.get_xydata(), dtype=float)
        return bool(xy.size) and bool(np.isfinite(xy).all(axis=-1).any())
    except Exception:                                      # noqa: BLE001
        return True                  # in doubt, drawn: never refuse on it


def _collection_has_marks(coll) -> bool:
    try:
        from matplotlib.collections import PathCollection
        from matplotlib.quiver import Quiver
        # scatter: its points; quiver: its arrows' tails (its paths are
        # made only when it is drawn)
        if isinstance(coll, (PathCollection, Quiver)):
            return len(coll.get_offsets()) > 0
        return len(coll.get_paths()) > 0
    except Exception:                                      # noqa: BLE001
        return True


def _figure_words(fig) -> dict:
    """What a matplotlib figure draws and the words on it, for the
    stand-in rule (``figures.stand_in``): titles, axis names, legend
    entries, other texts; ``drawn`` (a line with a point, a collection
    with marks, a patch, an artist, an image or a table)."""
    words = {"titles": [], "axes": [], "series": [], "notes": [],
             "drawn": False}
    sup = getattr(fig, "_suptitle", None)
    supxy = (getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel",
                                                       None))
    for t in getattr(fig, "texts", ()):
        key = ("titles" if t is sup else "axes" if t in supxy
               else "notes")
        words[key].append(_visible_text(t))
    for leg in getattr(fig, "legends", ()):
        words["series"] += [_visible_text(t) for t in leg.get_texts()]
    if (getattr(fig, "lines", None) or getattr(fig, "patches", None)
            or getattr(fig, "images", None)):
        words["drawn"] = True
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        words["titles"] += [ax.get_title(loc=loc)
                            for loc in ("left", "center", "right")]
        words["axes"] += [ax.get_xlabel(), ax.get_ylabel()]
        if hasattr(ax, "get_zlabel"):
            words["axes"].append(ax.get_zlabel())
        leg = ax.get_legend()
        if leg is not None:
            words["series"] += [_visible_text(t) for t in leg.get_texts()]
        words["notes"] += [_visible_text(t) for t in ax.texts]
        marks = (any(l.get_visible() and _line_has_points(l)
                     for l in ax.lines)
                 or any(c.get_visible() and _collection_has_marks(c)
                        for c in ax.collections)
                 or bool(ax.patches) or bool(ax.artists)
                 or bool(ax.images) or bool(ax.tables))
        if marks:
            words["drawn"] = True
    return words


#: A Plotly trace's keys that hold its data.
_TRACE_DATA = ("x", "y", "z", "values", "lat", "lon", "r", "theta", "a",
               "b", "c", "open", "close", "cells", "source")


def _has_values(value) -> bool:
    if isinstance(value, (list, tuple, str)):
        return len(value) > 0
    if isinstance(value, dict):       # a typed array {"dtype", "bdata"};
        return bool(value.get("bdata") or value.get("values"))  # cells
    return value is not None


def _chart_words(fig: dict) -> dict:
    """What a Plotly figure (as its JSON object) draws and names, for the
    stand-in rule."""
    def title(v):
        return v.get("text", "") if isinstance(v, dict) else (v or "")

    layout = fig.get("layout") if isinstance(fig.get("layout"), dict) else {}
    words = {"titles": [title(layout.get("title"))], "axes": [],
             "series": [], "notes": [],
             # a drawing made of layout shapes or images draws too
             "drawn": bool(layout.get("shapes") or layout.get("images"))}
    for key, value in layout.items():
        if not isinstance(value, dict):
            continue
        if key.startswith(("xaxis", "yaxis")):
            words["axes"].append(title(value.get("title")))
        elif key.startswith("scene"):
            words["axes"] += [title((value.get(a) or {}).get("title"))
                              for a in ("xaxis", "yaxis", "zaxis")
                              if isinstance(value.get(a), dict)]
    words["notes"] += [a.get("text") for a in layout.get("annotations") or []
                       if isinstance(a, dict)]
    for trace in fig.get("data") or []:
        if not isinstance(trace, dict):
            continue
        if any(_has_values(trace.get(k)) for k in _TRACE_DATA):
            words["drawn"] = True
        if trace.get("name") and trace.get("showlegend") is not False:
            words["series"].append(trace["name"])
    return words


def _stand_in(words) -> str:
    """Why a figure (its :func:`_figure_words` or :func:`_chart_words`)
    is a stand-in, else ``''``; ``''`` too when the rule cannot be read."""
    try:
        return (_figure_rules.stand_in(**words) or "") if _figure_rules \
            else ""
    except Exception:                                      # noqa: BLE001
        return ""


def _format_error(exc: BaseException) -> str:
    import traceback
    here = os.path.normcase(os.path.abspath(__file__))
    # Lines are looked up only for the frames shown: the code's own (held
    # in linecache) and the library frame that raised (inside the read
    # roots); the box's own frames are left out.
    frames = [f for f in traceback.StackSummary.extract(
        traceback.walk_tb(exc.__traceback__), lookup_lines=False)
        if os.path.normcase(os.path.abspath(f.filename)) != here]
    keep = [f for f in frames if f.filename == "<plot_code>"]
    if frames and frames[-1].filename != "<plot_code>":
        keep.append(frames[-1])          # where a library raised it
    lines = ["Traceback (the plotting code's lines):"]
    for f in keep:
        where = ("your code" if f.filename == "<plot_code>"
                 else os.path.basename(f.filename))
        lines.append(f"  {where}, line {f.lineno}, in {f.name}")
        if f.line:
            lines.append(f"    {f.line.strip()}")
    lines.extend(traceback.format_exception_only(type(exc), exc))
    text = "\n".join(s.rstrip("\n") for s in lines)
    return text[-3000:]


def main() -> int:
    raw = sys.stdin.buffer.read()
    payload = json.loads(raw.decode("utf-8"))
    marker = str(payload.get("marker") or "")
    t0 = _monotonic()
    result = {"ok": False}

    def report(res):
        sys.__stdout__.write(marker + json.dumps(res, ensure_ascii=True,
                                                 default=str) + "\n")
        sys.__stdout__.flush()

    # The folders the app itself imports from (a user site, PYTHONPATH),
    # after the box's own: -I leaves them out.
    for extra in payload.get("sys_path") or []:
        if (isinstance(extra, str) and extra and extra not in sys.path
                and os.path.isdir(extra)):
            sys.path.append(extra)

    code = payload.get("code") or ""
    problems = check_code(code)
    if problems:
        result["error"] = "The code was not run:\n" + "\n".join(problems)
        result["refused_code"] = problems
        report(result)
        return 0

    root = os.path.realpath(payload["root"])
    os.chdir(root)
    limits = payload.get("limits") or {}
    memory_mb = int(limits.get("memory_mb", 2048))
    applied = {}
    if os.name == "posix":
        applied.update(_posix_limits(
            memory_mb, int(limits.get("cpu_s", 660)),
            int(limits.get("fsize_mb", 200))))
    elif os.name == "nt":
        applied.update(_windows_job(memory_mb,
                                    int(limits.get("cpu_s", 660))))

    try:
        failed = _preload(imported_modules(code))
    except Exception as exc:                               # noqa: BLE001
        result["error"] = ("The plotting libraries could not be loaded in "
                           f"the box: {type(exc).__name__}: {exc}")
        report(result)
        return 0
    stem = _safe_stem(payload.get("name"), "figure")
    furniture = {}
    try:
        # The plan furniture the app's own plans use (scale bar, north
        # arrow), loaded from its file so geotech_common's __init__ (and the
        # rest of the package) stays out of the box.
        import importlib.util
        spec = importlib.util.find_spec("geotech_common")
        if spec and spec.submodule_search_locations:
            path = os.path.join(list(spec.submodule_search_locations)[0],
                                "plan_furniture.py")
            fspec = importlib.util.spec_from_file_location(
                "_box_plan_furniture", path)
            mod = importlib.util.module_from_spec(fspec)
            fspec.loader.exec_module(mod)
            furniture = {"scale_bar": mod.mpl_scale_bar,
                         "north_arrow": mod.mpl_north_arrow,
                         "nice_length": mod.nice_length}
    except Exception:                                      # noqa: BLE001
        furniture = {}
    try:
        # Standard chart lines as the references print them (the A-line,
        # the U-line, the sieve sizes; BACKLOG GM-153), loaded from their
        # file the same way, so the reference library stays out of the box.
        import importlib.util
        spec = importlib.util.find_spec("geotech_references")
        if spec and spec.submodule_search_locations:
            path = os.path.join(list(spec.submodule_search_locations)[0],
                                "chart_lines.py")
            if os.path.isfile(path):
                fspec = importlib.util.spec_from_file_location(
                    "_box_chart_lines", path)
                mod = importlib.util.module_from_spec(fspec)
                fspec.loader.exec_module(mod)
                furniture.update(chart_lines=mod.chart_lines,
                                 chart_line_value=mod.line_value)
    except Exception:                                      # noqa: BLE001
        pass

    import builtins
    import linecache
    import matplotlib.pyplot as plt
    linecache.cache["<plot_code>"] = (len(code), None,
                                      code.splitlines(True), "<plot_code>")
    compiled = compile(code, "<plot_code>", "exec")
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(),
                       level=0):
        if level or str(name).split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(
                f"'{name}' cannot be imported in the plotting box; it "
                f"offers {', '.join(sorted(ALLOWED_IMPORTS))}")
        for item in fromlist or ():
            if str(item) in BLOCKED_ATTRIBUTES or str(item).startswith("__"):
                raise ImportError(f"'{item}' cannot be imported from "
                                  f"'{name}' in the plotting box")
        mod = real_import(name, globals, locals, fromlist, level)
        if fromlist:
            # ``from pandas.io.common import os``: refuse a module the
            # library imported for itself unless it is one of the box's
            for item in fromlist:
                got = getattr(mod, str(item), None)
                if (isinstance(got, types.ModuleType) and got.__name__
                        .split(".")[0] not in ALLOWED_IMPORTS):
                    raise ImportError(
                        f"'{item}' (the module {got.__name__}) cannot be "
                        f"imported from '{name}' in the plotting box")
        return mod

    # setting the system clock is not the code's to do (ALLOWED_IMPORTS)
    for name in ("clock_settime", "clock_settime_ns"):
        if hasattr(time, name):
            delattr(time, name)

    safe_builtins = {k: v for k, v in vars(builtins).items()
                     if k not in BLOCKED_NAMES}
    safe_builtins["__import__"] = guarded_import

    hook, record = make_guard(root, payload.get("tmp") or "",
                              _read_roots(_norm(root)),
                              int(limits.get("max_files", 40)))
    saved_names = []
    not_saved = []
    checks = []
    check_seconds = []
    letters = []
    look_copies = {}
    try:
        print_w = float(payload.get("print_width_in")
                        or DEFAULT_PRINT_WIDTH_IN)
    except (TypeError, ValueError):
        print_w = DEFAULT_PRINT_WIDTH_IN

    def _save(fig, fname, dpi):
        """``fname`` for the report at ``dpi``; for a picture larger than a
        look takes whole, a copy that size in :data:`LOOK_DIR` for looks."""
        look_copies.pop(fname, None)
        # the lettering, measured before the save so a PNG carries it: a
        # PDF built from the file can then tell how large it prints (GM-168)
        measured = _layout_ink.lettering_sizes(fig)
        ext = os.path.splitext(fname)[1].lower()
        meta = None
        if ext == ".png" and measured[0] and measured[1] > 0:
            meta = {_layout_ink.LETTERING_KEY: _layout_ink.lettering_text(
                min(measured[0]), measured[1], print_w)}
        if ext in (".png", ".jpg", ".jpeg"):
            fig.savefig(fname, dpi=_report_dpi(fig, dpi), bbox_inches="tight",
                        metadata=meta)
            look_dpi = _dpi_for(fig, dpi)
            if look_dpi < _report_dpi(fig, dpi):
                os.makedirs(LOOK_DIR, exist_ok=True)
                look = LOOK_DIR + "/" + os.path.splitext(fname)[0] \
                    + "_look.png"
                fig.savefig(look, dpi=look_dpi, bbox_inches="tight")
                look_copies[fname] = look
        else:
            fig.savefig(fname, dpi=_dpi_for(fig, dpi), bbox_inches="tight")
        t_check = _monotonic()
        checks.append(_layout_check(fig, fname, dpi))
        check_seconds.append(round(_monotonic() - t_check, 2))
        letters.append(_lettering(fig, fname, print_w, measured))

    def refused(fname, words) -> bool:
        """Whether ``fname`` is a stand-in, which is not saved: every figure
        saved reaches the user (BACKLOG GM-190)."""
        why = _stand_in(words)
        if why:
            not_saved.append({"file": fname, "why": why})
        return bool(why)

    def save_figure(fig=None, name=None, dpi=200):
        """Save a matplotlib figure in the conversation folder; returns its
        file name (``None`` when it holds a place and is not saved). ``name``
        without an extension saves a PNG."""
        fig = fig if fig is not None else plt.gcf()
        base = _safe_stem(name, stem)
        ext = os.path.splitext(str(name or ""))[1].lower()
        fname = base + (ext if ext in (".png", ".svg", ".pdf", ".jpg",
                                       ".jpeg") else ".png")
        if refused(fname, _figure_words(fig)):
            return None
        _save(fig, fname, dpi)
        saved_names.append(fname)
        return fname

    def save_chart(fig, name=None):
        """Save a Plotly figure as the app's interactive chart
        (``<name>.plotly.json``); returns its file name (``None`` when it
        holds a place and is not saved)."""
        text = fig if isinstance(fig, str) else (
            json.dumps(fig) if isinstance(fig, dict) else fig.to_json())
        fname = _safe_stem(name, stem) + ".plotly.json"
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and refused(fname, _chart_words(parsed)):
            return None
        with open(fname, "w", encoding="utf-8") as fh:
            fh.write(text)
        saved_names.append(fname)
        return fname

    namespace = {"__name__": "__plot_code__", "__builtins__": safe_builtins,
                 "data": payload.get("data"), "NAME": stem,
                 "save_figure": save_figure, "save_chart": save_chart}
    namespace.update(furniture)

    out, err = _Capped(8000), _Capped(4000)
    sys.stdout, sys.stderr = out, err
    if os.name == "posix":
        applied.update(_posix_no_children())
    t_run = _monotonic()
    # From here to the end of the process every file, folder, process,
    # network and native-code operation goes past the hook, including the
    # box's own saving and reporting below (the code may have replaced any
    # function they call).
    sys.addaudithook(hook)
    del hook
    try:
        exec(compiled, namespace)
        result["ok"] = True
    except BaseException as exc:                           # noqa: BLE001
        result["error"] = _format_error(exc)
    run_s = _monotonic() - t_run

    # Figures the code drew but did not save: saved for it, when it saved
    # no image itself (a script that saved its own, or whose save was
    # refused, chose what to keep) and ran to the end (live wave w8b: a
    # half-drawn figure from a failed run was saved over the name a good
    # one would carry). A stand-in is not saved here either (GM-190).
    auto = []
    wrote_image = any(os.path.splitext(p)[1].lower()
                      in (".png", ".jpg", ".jpeg", ".svg", ".pdf")
                      for p in record["written"])
    try:
        nums = list(plt.get_fignums())
        if nums and not wrote_image and not not_saved and result["ok"]:
            for k, num in enumerate(nums, 1):
                fname = stem + (".png" if len(nums) == 1 else f"_{k}.png")
                fig = plt.figure(num)
                if refused(fname, _figure_words(fig)):
                    continue
                _save(fig, fname, 200)
                auto.append(fname)
        plt.close("all")
    except BaseException as exc:                           # noqa: BLE001
        result.setdefault("error", f"saving the open figures failed: "
                                   f"{type(exc).__name__}: {exc}")
        result["ok"] = False
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    result.update({
        "written": [os.path.relpath(p, _norm(root))
                    for p in record["written"]],
        "auto_saved": auto,
        "not_saved": not_saved,
        "stdout": out.text(),
        "stderr": err.text(),
        "refused": record["refused"],
        "run_seconds": round(run_s, 2),
        "seconds": round(_monotonic() - t0, 2),
        "limits_applied": applied,
        "preload_failed": failed,
        "layout": [c for c in checks if len(c) > 1],
        "layout_seconds": check_seconds,
        "lettering": [c for c in letters if c],
        "look_copies": look_copies,
    })
    report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
