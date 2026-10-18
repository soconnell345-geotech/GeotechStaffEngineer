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

#: Top-level modules plotting code may import: the plotting and number
#: libraries the owner named (matplotlib, Plotly, numpy, pandas), SciPy for
#: interpolation and statistics, and the standard library's maths and text
#: modules. Nothing that reaches the network, processes or the system, and
#: nothing that turns a string into code or into an attribute (``typing``'s
#: type hints, ``string.Formatter``).
ALLOWED_IMPORTS = frozenset({
    "numpy", "pandas", "matplotlib", "mpl_toolkits", "plotly", "scipy",
    "math", "cmath", "statistics", "random", "itertools", "functools",
    "collections", "json", "csv", "datetime", "decimal", "fractions", "re",
    "textwrap", "enum", "copy", "bisect", "heapq", "colorsys", "numbers",
    "io",
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


def _hidden_tick_labels(fig) -> set:
    """Tick labels matplotlib keeps but never draws: those of ticks outside
    the axis's view limits (live wave w8b: a phantom "100" beyond a panel's
    right edge was reported against the next panel's "0")."""
    hidden = set()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            tol = 1e-9 * max(abs(hi - lo), 1e-12)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - tol <= loc <= hi + tol):
                    hidden.add(id(tick.label1))
                    hidden.add(id(tick.label2))
    return hidden


def _drawn_texts(fig, renderer) -> list:
    """``[(text, window box, artist)]`` for the lettering that is drawn:
    visible, not empty, not a tick label outside the view; an annotation
    measured by its text, not its leader."""
    from matplotlib.text import Annotation, Text
    hidden = _hidden_tick_labels(fig)
    texts = []
    for t in fig.findobj(Text):
        s = t.get_text().strip() if t.get_visible() else ""
        if not s or id(t) in hidden:
            continue
        try:
            # an Annotation's own extent takes in its leader line
            bb = (Text.get_window_extent(t, renderer)
                  if isinstance(t, Annotation)
                  else t.get_window_extent(renderer))
        except Exception:                                  # noqa: BLE001
            continue
        if bb.width > 1 and bb.height > 1:
            texts.append((" ".join(s.split())[:40], bb, t))
    return texts


def _lettering(fig, fname: str, print_width_in: float) -> dict:
    """The lettering's size as printed ``print_width_in`` wide (live wave
    w9: figures drawn 9.5-11 in wide printed at 4.6-5.6 pt on a report
    page, and nothing measured it): the smallest size, the width the figure
    was saved at, and how many texts print under :data:`MIN_PRINT_PT`."""
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        texts = _drawn_texts(fig, r)
        if not texts:
            return {}
        import matplotlib
        pad = float(matplotlib.rcParams.get("savefig.pad_inches", 0.1))
        # the width the figure is saved at (bbox_inches="tight"), inches
        saved_w = fig.get_tightbbox(r).width + 2 * pad
        scale = float(print_width_in) / max(saved_w, 1e-6)
        sizes = [float(t.get_fontsize()) * scale for _s, _bb, t in texts]
    except Exception:                                      # noqa: BLE001
        return {}
    smallest = min(sizes)
    under = sum(1 for s in sizes if s < MIN_PRINT_PT - 0.05)
    return {"file": fname, "smallest_pt": round(smallest, 1),
            "print_width_in": float(print_width_in),
            "saved_width_in": round(saved_w, 2),
            "texts_under_7_pt": under,
            "says": (f"{smallest:.1f} pt at {float(print_width_in):g} in "
                     f"wide (saved {saved_w:.1f} in wide; {under} of "
                     f"{len(sizes)} texts under {MIN_PRINT_PT:g} pt)")}


#: The layout check judges lettering by the pixels drawn (live wave w9: 22
#: of its 36 reports were false while it judged by boxes -- rotated text by
#: its upright box, legend text through the legend's opaque frame, text
#: through its own white box -- and lettering outside the axes was never
#: tested). The measures are the w9 trace review's own (§9): a line runs
#: through lettering when at least LINE_MIN_PX of its pixels show on the
#: lettering's glyphs, grown by one pixel, in the finished picture (at 200
#: dpi; scaled with the dpi); a pixel shows when the picture with the line
#: differs from the picture without it by more than VISIBLE_DIFF over the
#: three colour channels (of 255 each).
LINE_MIN_PX = 3
VISIBLE_DIFF = 60
#: A glyph pixel: drawn at an alpha above this (of 255).
INK_ALPHA = 40
#: Lettering closer than this on one line reads as one word ("200" against
#: "0" reads "2000"): a quarter of the smaller lettering's size (letters in
#: a word sit about 0.13 em apart, words about 0.45 em), and never under
#: TOUCH_PT (2 px at 100 dpi).
TOUCH_EM = 0.25
TOUCH_PT = 1.44
#: A box under lettering (its own bbox, a legend's frame) at least this
#: opaque is the lettering's footprint.
BOX_OPAQUE = 0.5
#: The check gives up ("unchecked") rather than slow the save past this
#: many seconds per figure, or hold more than this many MB of drawing kept
#: on the lettering (the box has 2 GB): thousands of texts, or a thousand
#: lines under a large opaque legend.
LAYOUT_SECONDS = 3.0
LAYOUT_MB = 256


class _Unchecked(Exception):
    """The layout is too large to check; the message says why."""


def _layout_check(fig, fname: str, dpi=200) -> dict:
    """Lettering the reader cannot read cleanly, judged by the pixels drawn
    in the figure as it is saved (not from a look):

    - ``text_overlaps``: lettering whose ink overlaps other lettering, or
      runs into it on one line (closer than :data:`TOUCH_PT`: "200" against
      "0" reads "2000"). Lettering on its own opaque box (a bbox, a legend's
      frame) has the box as its footprint; an opaque legend is one footprint,
      "the legend".
    - ``text_on_a_line``: lettering that a line, a shape's edge, markers, a
      tick mark or an axes frame shows through, or that a filled shape is
      drawn over. A line hidden by a box drawn over it (the lettering's own
      box, a legend's frame) is not through the lettering.
    - ``text_cut_off``: lettering the figure edge or its axes' edge cuts.

    Every drawn text counts, inside the axes or out (titles, axis and tick
    labels, legend entries, figure text); tick labels outside the view are
    never drawn, and an annotation is its text, not its leader. Up to eight
    of each; a look decides. A figure too large to check within
    :data:`LAYOUT_SECONDS` or :data:`LAYOUT_MB` is ``unchecked``."""
    out = {"file": fname}
    try:
        found = _ink_check(fig, _dpi_for(fig, dpi))
    except _Unchecked as exc:
        out["unchecked"] = str(exc)
        return out
    except Exception as exc:                               # noqa: BLE001
        out["unchecked"] = (f"the layout could not be measured "
                            f"({type(exc).__name__}: {exc})")[:200]
        return out
    for key in ("text_overlaps", "text_on_a_line", "text_cut_off"):
        if found.get(key):
            out[key] = found[key][:8]
    return out


def _leaf_kinds() -> list:
    """The classes whose ``draw`` draws every leaf of a figure: lettering,
    lines, shapes, collections (meshes draw on their own) and images."""
    import matplotlib.collections as mcoll
    import matplotlib.image as mimage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    kinds = [Text, Line2D, Patch, mcoll.Collection]
    # pcolormesh, a colorbar's fill, tripcolor: their draw is their own
    for name in ("QuadMesh", "TriMesh"):
        k = getattr(mcoll, name, None)
        if k is not None and "draw" in vars(k):
            kinds.append(k)
    base = getattr(mimage, "_ImageBase", None)
    if base is not None and "draw" in vars(base):
        kinds.append(base)
    return kinds


def _ink_check(fig, dpi: int) -> dict:
    """Save ``fig`` into memory as :func:`save_figure` saves it
    (``bbox_inches="tight"``, so lettering outside the axes and the canvas
    is in the frame), at ``dpi``, noting each leaf artist as it is drawn;
    after the last one, measure them in that frame (:class:`_Ink`)."""
    from matplotlib.artist import Artist
    from matplotlib.text import Text
    kinds = _leaf_kinds()
    originals = {k: vars(k)["draw"] for k in kinds}
    state = {"drawn": [], "placed": {}, "off": None, "depth": 0,
             "found": None, "deadline": time.monotonic() + LAYOUT_SECONDS}

    def wrap(orig):
        def draw(self, renderer, *args, **kwargs):
            # a leaf's own parts (a text's box) are the leaf; the second
            # buffer's draws are the check's own
            if state["depth"] or renderer is state["off"]:
                return orig(self, renderer, *args, **kwargs)
            state["drawn"].append((self, orig))
            if isinstance(self, Text):
                # where it is drawn: 3D text is placed only while it draws
                state["placed"][id(self)] = (self, self._x, self._y,
                                             self._rotation)
            if "draw_path" not in vars(renderer):
                # the saving pass: the leaf is drawn alone later, and this
                # picture is thrown away (the containers still lay out)
                return None
            state["depth"] += 1
            try:
                return orig(self, renderer, *args, **kwargs)
            finally:
                state["depth"] -= 1
        draw._supports_rasterization = getattr(
            orig, "_supports_rasterization", False)
        return draw

    class _Measure(Artist):
        """Drawn last (the highest z-order), when the frame is the saved
        one and every leaf has been drawn in place."""

        def draw(self, renderer):
            drawn, state["drawn"] = state["drawn"], []
            placed, state["placed"] = state["placed"], {}
            # matplotlib's sizing pass for bbox_inches="tight" draws nothing
            if "draw_path" in vars(renderer):
                return
            moved = []
            try:
                for a, x, y, rot in placed.values():
                    now = (a._x, a._y, a._rotation)
                    if any(u is not v for u, v in zip(now, (x, y, rot))):
                        moved.append((a,) + now)
                        a._x, a._y, a._rotation = x, y, rot
                state["found"] = _Ink(fig, renderer, drawn, state).found()
            finally:
                for a, x, y, rot in moved:
                    a._x, a._y, a._rotation = x, y, rot

    probe = _Measure()
    probe.set_zorder(1e12)
    probe.set_in_layout(False)
    for k, orig in originals.items():
        setattr(k, "draw", wrap(orig))
    try:
        fig.add_artist(probe)
        fig.savefig(io.BytesIO(), format="rgba", dpi=dpi,
                    bbox_inches="tight")
    finally:
        for k, orig in originals.items():
            setattr(k, "draw", orig)
        try:
            probe.remove()
        except Exception:                                  # noqa: BLE001
            pass
    return state["found"] or {}


def _grow(mask, k: int):
    """``mask`` grown by ``k`` pixels each way (a square)."""
    if k <= 0:
        return mask
    rows = mask.copy()
    for d in range(1, k + 1):
        rows[d:] |= mask[:-d]
        rows[:-d] |= mask[d:]
    out = rows.copy()
    for d in range(1, k + 1):
        out[:, d:] |= rows[:, :-d]
        out[:, :-d] |= rows[:, d:]
    return out


def _meet(a, b, grow: int = 0) -> bool:
    """Whether two masks placed in the frame (``(row, col, mask)``) share a
    pixel, the first grown by ``grow`` pixels."""
    import numpy as np
    ra, ca, ma = a
    rb, cb, mb = b
    if grow:
        ma = _grow(np.pad(ma, grow), grow)
        ra, ca = ra - grow, ca - grow
    r0, r1 = max(ra, rb), min(ra + ma.shape[0], rb + mb.shape[0])
    c0, c1 = max(ca, cb), min(ca + ma.shape[1], cb + mb.shape[1])
    if r1 <= r0 or c1 <= c0:
        return False
    return bool((ma[r0 - ra:r1 - ra, c0 - ca:c1 - ca]
                 & mb[r0 - rb:r1 - rb, c0 - cb:c1 - cb]).any())


def _ink_rows(mask, row0: int):
    """The first and last frame rows of a mask's ink, or None."""
    import numpy as np
    rows = np.nonzero(mask.any(axis=1))[0]
    return (row0 + int(rows[0]), row0 + int(rows[-1])) if len(rows) else None


def _figure_roles(fig) -> dict:
    """What some leaves are, by ``id``: grid lines, tick marks, the panels'
    and the figure's own backgrounds, the axes' frame lines (by side), and
    each legend's frame, entries and symbols."""
    from matplotlib.figure import FigureBase
    from matplotlib.legend import Legend
    roles = {"grid": set(), "ticks": set(), "panels": set(), "spines": {},
             "frames": {}, "in_legend": {}, "symbols": set()}
    for f in fig.findobj(lambda a: isinstance(a, FigureBase)):
        roles["panels"].add(id(f.patch))
    for ax in fig.get_axes():
        roles["panels"].add(id(ax.patch))
        for side, sp in ax.spines.items():
            roles["spines"][id(sp)] = side
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None),
                     getattr(ax, "zaxis", None)):
            if axis is None:
                continue
            if getattr(axis, "gridlines", None) is not None:
                roles["grid"].add(id(axis.gridlines))     # a 3D axis's grid
            for tick in axis.majorTicks + axis.minorTicks:
                roles["grid"].add(id(tick.gridline))
                roles["ticks"].update((id(tick.tick1line),
                                       id(tick.tick2line)))
    for leg in fig.findobj(lambda a: isinstance(a, Legend)):
        frame = leg.get_frame()
        roles["frames"][id(frame)] = leg
        for a in leg.findobj():
            if a is leg or a is frame:
                continue
            roles["in_legend"][id(a)] = leg
            if not hasattr(a, "get_text"):
                roles["symbols"].add(id(a))
    return roles


class _Ink:
    """The lettering of one saved figure, measured by what each artist
    draws.

    Each leaf artist is drawn again ALONE into a second buffer of the saved
    frame, in the order the figure drew it (a shape's fill and its edge
    apart; a text's glyphs apart from its box), and kept where it falls on
    a piece of lettering. Per piece of lettering the leaves are then laid
    over one another in that order, as the picture was made, so what shows
    is known: a line under an opaque box shows nowhere; a line under a box
    85 % white still shows."""

    def __init__(self, fig, renderer, drawn, state):
        import numpy as np
        if time.monotonic() > state["deadline"]:
            raise _Unchecked(f"too many texts to check in time "
                             f"({LAYOUT_SECONDS:g} s per figure); a look "
                             f"decides")
        from matplotlib.backends.backend_agg import RendererAgg
        self.np = np
        self.W, self.H = int(renderer.width), int(renderer.height)
        self.dpi = float(renderer.dpi)
        self.off = RendererAgg(self.W, self.H, self.dpi)
        state["off"] = self.off
        self.buf = np.asarray(self.off.buffer_rgba())
        self.gap = max(1, int(round(TOUCH_PT * self.dpi / 72.0)))
        self.min_px = max(2, int(round(LINE_MIN_PX * self.dpi / 200.0)))
        self.deadline = state["deadline"]
        self.kept = 0           # bytes of drawing kept on the lettering
        self.roles = _figure_roles(fig)
        self.leaves = [(a, o) for a, o in drawn if a.get_visible()]
        self._find_lettering(renderer)
        self.boxes = np.array(self.regions, dtype=float).reshape(-1, 4)

    def _in_time(self):
        if time.monotonic() > self.deadline:
            raise _Unchecked(
                f"too many texts to check in time ({len(self.names)} "
                f"texts, {LAYOUT_SECONDS:g} s per figure); a look decides")

    # -- where the lettering is -------------------------------------------

    def _box(self, bb, pad: int):
        """A frame box ``(r0, r1, c0, c1)`` round a display bbox, or None
        when it falls outside the frame."""
        np = self.np
        c0 = max(0, int(np.floor(bb.x0)) - pad)
        c1 = min(self.W, int(np.ceil(bb.x1)) + pad)
        r0 = max(0, self.H - int(np.ceil(bb.y1)) - pad)
        r1 = min(self.H, self.H - int(np.floor(bb.y0)) + pad)
        return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None

    def _find_lettering(self, renderer):
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        self.names = {}         # leaf index -> its words
        self.extent = {}        # leaf index -> its display bbox
        self.region = {}        # leaf index -> region index
        self.regions = []       # (r0, r1, c0, c1)
        self.reach = {}         # leaf index -> display bbox it can draw in
        pt = self.dpi / 72.0
        for i, (a, _orig) in enumerate(self.leaves):
            box = None
            if isinstance(a, Text):
                words = " ".join(a.get_text().split())
                if not words:
                    continue
                bb = Text.get_window_extent(a, renderer)
                self.names[i] = words[:40]
                self.extent[i] = bb
                bp = a.get_bbox_patch()
                if bp is not None and bp.get_visible():
                    # placed in this frame (the saving pass drew no text)
                    a.update_bbox_position_size(renderer)
                    bb = Bbox.union([bb, bp.get_window_extent(renderer)])
                halo = 3 * pt if a.get_path_effects() else 0
                self.reach[i] = bb.padded(halo + 2 * pt + 4)
                box = self._box(bb, self.gap + 2 + int(halo))
            elif (id(a) in self.roles["frames"]
                  and self._opaque_frame(self.roles["frames"][id(a)])):
                self.names[i] = "the legend"
                box = self._box(a.get_window_extent(renderer), 1)
            if box is not None:
                self.region[i] = len(self.regions)
                self.regions.append(box)

    def _opaque_frame(self, leg) -> bool:
        frame = leg.get_frame()
        return bool(frame.get_visible() and frame.get_fill()
                    and frame.get_facecolor()[3] >= BOX_OPAQUE)

    def _boxed(self, text) -> bool:
        """Whether a text is drawn on its own opaque box (or a halo)."""
        bp = text.get_bbox_patch()
        return bool(text.get_path_effects()) or bool(
            bp is not None and bp.get_visible() and bp.get_fill()
            and bp.get_facecolor()[3] >= BOX_OPAQUE)

    # -- drawing each leaf alone ------------------------------------------

    def _parts(self, i, a):
        """``(part, kind)`` for each separate drawing of leaf ``a``: kind
        ``glyph`` (a text's letters alone: its mask), ``text`` (a text as
        drawn, with its box), ``stroke`` (a line, an edge, markers: what can
        run through lettering), ``face`` (a fill, an image: what can hide
        it), ``grid`` (drawn, never reported) and ``unclipped`` (a text's
        letters without its clip)."""
        from matplotlib.collections import Collection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.text import Text
        roles = self.roles
        if isinstance(a, Text):
            if i not in self.names:
                return []
            parts = []
            bp = a.get_bbox_patch()
            if a.get_path_effects() or (bp is not None
                                        and bp.get_visible()):
                parts.append(("glyph", "glyph"))
            parts.append(("full", "text"))
            if a.get_clip_on() and (a.get_clip_box() is not None
                                    or a.get_clip_path() is not None):
                parts.append(("unclipped", "unclipped"))
            return parts
        if isinstance(a, Line2D):
            return [("full", "grid" if id(a) in roles["grid"] else "stroke")]
        if isinstance(a, Patch):
            if (id(a) in roles["panels"] or id(a) in roles["frames"]
                    or not hasattr(a, "_facecolor")
                    or not hasattr(a, "_edgecolor")):
                return [("full", "face")]
            if id(a) in roles["spines"]:
                return [("full", "stroke")]
            parts = []
            if (a.get_fill() and a.get_facecolor()[3] > 0) or a.get_hatch():
                parts.append(("face", "face"))
            if (a.get_linewidth() > 0 and a.get_edgecolor()[3] > 0
                    and a.get_linestyle() not in ("None", "none", "", " ")):
                parts.append(("edge", "stroke"))
            return parts
        if isinstance(a, Collection):
            if id(a) in roles["grid"]:
                return [("full", "grid")]
            if self._lines(a):
                return [("full", "stroke")]
            # fill_between, stackplot, a mesh: the fill hides, the edges
            # run through (as a shape's)
            np = self.np
            try:
                a.update_scalarmappable()
                faces = np.asarray(a.get_facecolor(), float).reshape(-1, 4)
                edges = np.asarray(a.get_edgecolor(), float).reshape(-1, 4)
                widths = np.atleast_1d(np.asarray(a.get_linewidth(), float))
            except Exception:                              # noqa: BLE001
                return [("full", "face")]
            parts = []
            if (faces[:, 3] > 0).any() or a.get_hatch():
                parts.append(("cface", "face"))
            if (edges[:, 3] > 0).any() and (widths > 0).any():
                parts.append(("cedge", "stroke"))
            return parts
        return [("full", "face")]

    def _lines(self, a) -> bool:
        """A collection that is lines or markers, drawn whole as a line."""
        from matplotlib.collections import LineCollection, PathCollection
        return isinstance(a, (LineCollection, PathCollection)) or (
            hasattr(a, "filled") and not a.filled)

    def _draw(self, a, orig, part):
        """Draw one part of leaf ``a`` alone into the second buffer."""
        off = self.off
        if part == "full":
            orig(a, off)
        elif part in ("glyph", "unclipped"):
            bp = a.get_bbox_patch()
            effects = a.get_path_effects()
            shown = bp.get_visible() if bp is not None else None
            try:
                if bp is not None:
                    bp.set_visible(False)
                if effects:
                    a.set_path_effects([])
                if part == "unclipped":
                    a.set_clip_on(False)
                orig(a, off)
            finally:
                if bp is not None:
                    bp.set_visible(shown)
                if effects:
                    a.set_path_effects(effects)
                if part == "unclipped":
                    a.set_clip_on(True)
        elif part in ("cface", "cedge"):        # a collection's fill or edges
            np = self.np
            # its colours as drawn, held while it draws (a mapped
            # collection would map them again)
            edges = np.asarray(a.get_edgecolor(), float).reshape(-1, 4)
            kept = (a._facecolors, a._edgecolors, a._hatch)
            a.update_scalarmappable = lambda: None
            try:
                if part == "cface":
                    a._edgecolors = np.zeros((1, 4))
                else:
                    a._edgecolors = edges
                    a._facecolors = np.zeros((1, 4))
                    a._hatch = None
                orig(a, off)
            finally:
                a._facecolors, a._edgecolors, a._hatch = kept
                del a.update_scalarmappable
        else:                                   # a shape's fill or edge
            kept = (a._facecolor, a._edgecolor, a._hatch)
            try:
                if part == "face":
                    a._edgecolor = (0.0, 0.0, 0.0, 0.0)
                else:
                    a._facecolor = (0.0, 0.0, 0.0, 0.0)
                    a._hatch = None
                orig(a, off)
            finally:
                a._facecolor, a._edgecolor, a._hatch = kept

    def _near(self, i, a):
        """The regions leaf ``a`` can draw in: a clipped leaf draws inside
        its clip box, a text round itself; anything else anywhere."""
        if i in self.reach:
            bb = self.reach[i]
        elif a.get_clip_on() and a.get_clip_box() is not None:
            bb = a.get_clip_box().padded(2)
        else:
            return range(len(self.regions))
        r0, r1, c0, c1 = self.boxes.T
        return self.np.nonzero((c0 <= bb.x1) & (bb.x0 <= c1)
                               & (r0 <= self.H - bb.y0)
                               & (self.H - bb.y1 <= r1))[0].tolist()

    def _collect(self):
        """Draw every leaf part alone, keeping what falls on each region.
        A region that is not lettering (a legend's frame) keeps only its
        own leaf: nothing else is read there."""
        from matplotlib.text import Text
        self.layers = [[] for _ in self.regions]
        owner = {k: i for i, k in self.region.items()}
        buf = self.buf
        for i, (a, orig) in enumerate(self.leaves):
            self._in_time()
            parts = self._parts(i, a)
            if not parts or not self.regions:
                continue
            near = self._near(i, a)
            for n, (part, kind) in enumerate(parts):
                if part == "unclipped" and i not in self.region:
                    continue
                self._draw(a, orig, part)
                inked = []
                look = ([self.region[i]] if part == "unclipped" else near)
                for k in look:
                    r0, r1, c0, c1 = self.regions[k]
                    crop = buf[r0:r1, c0:c1]
                    if crop[..., 3].any():
                        keep = owner[k] == i or isinstance(
                            self.leaves[owner[k]][0], Text)
                        inked.append((k, crop.copy() if keep else None))
                for k, crop in inked:
                    if crop is not None:
                        self.layers[k].append(((i, n), kind, i, crop))
                        self.kept += crop.nbytes
                    r0, r1, c0, c1 = self.regions[k]
                    buf[r0:r1, c0:c1] = 0
                if self.kept > LAYOUT_MB * 2 ** 20:
                    raise _Unchecked(
                        f"too much drawn on the lettering to check in "
                        f"memory (over {LAYOUT_MB} MB); a look decides")
                if part == "unclipped":
                    # it may draw outside its own region: clear it all
                    self.off.clear()

    # -- what shows ---------------------------------------------------------

    def _mask(self, i, kinds):
        """Leaf ``i``'s ink in its own region from the first of ``kinds``
        drawn, as ``(row, col, bool mask)``; None when it drew nothing."""
        k = self.region.get(i)
        if k is None:
            return None
        r0, _r1, c0, _c1 = self.regions[k]
        for kind in kinds:
            for _order, kd, leaf, crop in self.layers[k]:
                if leaf == i and kd == kind:
                    return r0, c0, crop[..., 3] > INK_ALPHA
        return None

    def _what(self, a, kind) -> str:
        from matplotlib.collections import Collection, PathCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        roles = self.roles
        if kind == "face":
            if id(a) in roles["panels"]:
                return "an axes panel drawn over it"
            if isinstance(a, Patch):
                return "a filled shape drawn over it"
            return "an image or filled area drawn over it"
        if id(a) in roles["spines"]:
            return f"the {roles['spines'][id(a)]} frame line"
        if id(a) in roles["ticks"]:
            return "a tick mark"
        if id(a) in roles["symbols"]:
            return "a legend symbol"
        if isinstance(a, Patch) or (isinstance(a, Collection)
                                    and not self._lines(a)):
            return "an outline"
        label = str(a.get_label() or "")
        if label and not label.startswith("_"):
            return label[:40]
        if isinstance(a, PathCollection) or (
                isinstance(a, Line2D)
                and a.get_linestyle() in ("None", "none", "", " ")):
            return "markers"
        return "a line"

    def _through(self, i):
        """The leaf showing most on text ``i``'s glyphs (grown one pixel):
        a line, an edge or markers anywhere in the order, or a fill drawn
        after the text. ``(pixels, what)`` or None."""
        np = self.np
        k = self.region[i]
        mine = self._mask(i, ("glyph", "text"))
        if mine is None or not mine[2].any():
            return None
        grown = _grow(mine[2], 1)
        layers = sorted((e for e in self.layers[k]
                         if e[1] not in ("glyph", "unclipped")),
                        key=lambda e: e[0])
        at = next((e[0] for e in layers if e[2] == i), None)

        def tested(e):
            order, kind, leaf, _crop = e
            if leaf == i:
                return False
            if kind == "stroke":
                return True
            # a fill drawn over the text; a legend's frame over lettering is
            # the legend's footprint (text_overlaps)
            return (kind == "face" and at is not None and order > at
                    and id(self.leaves[leaf][0]) not in self.roles["frames"])
        if not any(tested(e) for e in layers):
            return None
        # only the glyph pixels count: lay the leaves over one another
        # there, leaving out any that draw nothing on them
        layers = [e for e in layers if e[3][..., 3][grown].any()]
        if not any(tested(e) for e in layers):
            return None
        alphas = [e[3][..., 3][grown].astype(np.float32) / 255.0
                  for e in layers]
        above = [None] * len(layers)
        clear = np.ones(alphas[0].shape, np.float32)
        for n in range(len(layers) - 1, -1, -1):
            above[n] = clear
            clear = clear * (1.0 - alphas[n])
        under = np.full(alphas[0].shape + (3,), 255.0, np.float32)
        best = None
        for n, e in enumerate(layers):
            a = alphas[n][..., None]
            colour = e[3][..., :3][grown].astype(np.float32)
            if tested(e):
                shows = (above[n][..., None] * a
                         * np.abs(colour - under)).sum(axis=1)
                px = int((shows > VISIBLE_DIFF).sum())
                if px >= self.min_px and (best is None or px > best[0]):
                    best = (px, self._what(self.leaves[e[2]][0], e[1]))
            under = a * colour + (1.0 - a) * under
        return best

    def found(self) -> dict:
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        self._collect()
        texts = [i for i in self.region if isinstance(self.leaves[i][0],
                                                       Text)]
        texts.sort()
        on_a_line, cut = [], []
        for i in texts:
            self._in_time()
            hit = self._through(i)
            if hit:
                on_a_line.append([self.names[i], hit[1]])
        for i in sorted(self.names):
            a = self.leaves[i][0]
            if not isinstance(a, Text):
                continue
            seen = self._mask(i, ("glyph", "text"))
            whole = self._mask(i, ("unclipped",))
            if seen is not None and whole is not None:
                shown, all_ink = int(seen[2].sum()), int(whole[2].sum())
                if shown and all_ink - shown >= max(self.min_px,
                                                    0.02 * all_ink):
                    cut.append([self.names[i], "the edge of its axes"])
                    continue
            bb = self.extent[i]
            if a.get_clip_on() and a.get_clip_box() is not None:
                # a clipped text reaches only as far as its clip
                bb = Bbox.intersection(bb, a.get_clip_box())
                if bb is None:
                    continue
            if (bb.x0 < -1 or bb.y0 < -1 or bb.x1 > self.W + 1
                    or bb.y1 > self.H + 1):
                cut.append([self.names[i], "the figure edge"])
        return {"text_overlaps": self._overlaps(texts),
                "text_on_a_line": on_a_line, "text_cut_off": cut}

    def _overlaps(self, texts) -> list:
        """Pairs of lettering whose footprints share ink, or whose letters
        come within :attr:`gap` pixels on one line."""
        roles = self.roles
        items = []
        for i in texts:
            mask = self._mask(i, ("glyph", "text"))
            if mask is None or not mask[2].any():
                continue
            a = self.leaves[i][0]
            foot = mask
            drawn = self._alpha(i, "text")
            if self._boxed(a) and drawn is not None:
                # the letters and the opaque box (or halo) they sit on
                foot = (mask[0], mask[1],
                        mask[2] | (drawn >= 255 * BOX_OPAQUE))
            items.append((i, mask, foot, roles["in_legend"].get(id(a)),
                          float(a.get_fontsize())))
        for i in sorted(self.region):
            a = self.leaves[i][0]
            alpha = self._alpha(i, "face")
            if id(a) in roles["frames"] and i in self.names \
                    and alpha is not None:
                r0, _r1, c0, _c1 = self.regions[self.region[i]]
                items.append((i, None, (r0, c0, alpha >= 255 * BOX_OPAQUE),
                              roles["frames"][id(a)], None))
        np = self.np
        # each footprint's frame box, to pass over pairs far apart at once
        top = np.array([e[2][0] for e in items], float)
        left = np.array([e[2][1] for e in items], float)
        bottom = top + [e[2][2].shape[0] for e in items]
        right = left + [e[2][2].shape[1] for e in items]
        reach = max([self._gap(e[4]) for e in items if e[4]] + [self.gap])
        pairs = []
        for x in range(len(items)):
            self._in_time()
            near = np.nonzero((top - reach <= bottom[x])
                              & (top[x] - reach <= bottom)
                              & (left - reach <= right[x])
                              & (left[x] - reach <= right))[0]
            for y in near[near > x].tolist():
                i, gi, fi, li, si = items[x]
                j, gj, fj, lj, sj = items[y]
                frame_i, frame_j = gi is None, gj is None
                size = min(s for s in (si, sj, 1e9) if s is not None)
                g = self._gap(size)
                if (frame_i and frame_j) or self._apart(fi, fj, g):
                    continue
                if li is not None and li is lj:
                    if frame_i or frame_j:
                        continue                  # an entry on its own frame
                    hit = self._close(gi, gj, g)
                else:
                    # an entry of an opaque legend meets others as "the
                    # legend"
                    if ((li is not None and not frame_i
                         and self._opaque_frame(li))
                            or (lj is not None and not frame_j
                                and self._opaque_frame(lj))):
                        continue
                    # a box is its text's footprint: a label's box against
                    # another's letters reads as one word too (live wave
                    # w9: "24" on its white box against "Medium")
                    hit = self._close(fi, fj, g)
                if hit:
                    if frame_i:
                        pairs.append([self.names[j], "the legend"])
                    elif frame_j:
                        pairs.append([self.names[i], "the legend"])
                    else:
                        pairs.append([self.names[i], self.names[j]])
        return pairs

    def _alpha(self, i, kind="text"):
        k = self.region[i]
        for _order, kd, leaf, crop in self.layers[k]:
            if leaf == i and kd == kind:
                return crop[..., 3]
        return None

    def _gap(self, size_pt: float) -> int:
        """The one-word gap in pixels for lettering ``size_pt`` tall."""
        pt = max(TOUCH_PT, TOUCH_EM * min(size_pt, 1e3))
        return max(1, int(round(pt * self.dpi / 72.0)))

    @staticmethod
    def _apart(f, h, g: int) -> bool:
        """Two placed masks whose frames are more than ``g`` apart."""
        return (f[0] - g > h[0] + h[2].shape[0]
                or h[0] - g > f[0] + f[2].shape[0]
                or f[1] - g > h[1] + h[2].shape[1]
                or h[1] - g > f[1] + f[2].shape[1])

    def _close(self, a, b, g: int) -> bool:
        """Two footprints (letters, or letters on their box) overlap, or
        come within ``g`` pixels with their ink on the same rows."""
        if _meet(a, b):
            return True
        ra, rb = _ink_rows(a[2], a[0]), _ink_rows(b[2], b[0])
        if ra is None or rb is None or min(ra[1], rb[1]) < max(ra[0],
                                                                 rb[0]):
            return False
        return _meet(a, b, grow=g)


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
    t0 = time.monotonic()
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

    safe_builtins = {k: v for k, v in vars(builtins).items()
                     if k not in BLOCKED_NAMES}
    safe_builtins["__import__"] = guarded_import

    hook, record = make_guard(root, payload.get("tmp") or "",
                              _read_roots(_norm(root)),
                              int(limits.get("max_files", 40)))
    saved_names = []
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
        if os.path.splitext(fname)[1].lower() in (".png", ".jpg", ".jpeg"):
            fig.savefig(fname, dpi=_report_dpi(fig, dpi), bbox_inches="tight")
            look_dpi = _dpi_for(fig, dpi)
            if look_dpi < _report_dpi(fig, dpi):
                os.makedirs(LOOK_DIR, exist_ok=True)
                look = LOOK_DIR + "/" + os.path.splitext(fname)[0] \
                    + "_look.png"
                fig.savefig(look, dpi=look_dpi, bbox_inches="tight")
                look_copies[fname] = look
        else:
            fig.savefig(fname, dpi=_dpi_for(fig, dpi), bbox_inches="tight")
        t_check = time.monotonic()
        checks.append(_layout_check(fig, fname, dpi))
        check_seconds.append(round(time.monotonic() - t_check, 2))
        letters.append(_lettering(fig, fname, print_w))

    def save_figure(fig=None, name=None, dpi=200):
        """Save a matplotlib figure in the conversation folder; returns its
        file name. ``name`` without an extension saves a PNG."""
        fig = fig if fig is not None else plt.gcf()
        base = _safe_stem(name, stem)
        ext = os.path.splitext(str(name or ""))[1].lower()
        fname = base + (ext if ext in (".png", ".svg", ".pdf", ".jpg",
                                       ".jpeg") else ".png")
        _save(fig, fname, dpi)
        saved_names.append(fname)
        return fname

    def save_chart(fig, name=None):
        """Save a Plotly figure as the app's interactive chart
        (``<name>.plotly.json``); returns its file name."""
        text = fig if isinstance(fig, str) else (
            json.dumps(fig) if isinstance(fig, dict) else fig.to_json())
        fname = _safe_stem(name, stem) + ".plotly.json"
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
    t_run = time.monotonic()
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
    run_s = time.monotonic() - t_run

    # Figures the code drew but did not save: saved for it, when it saved
    # no image itself (a script that saved its own chose what to keep) and
    # ran to the end (live wave w8b: a half-drawn figure from a failed run
    # was saved over the name a good one would carry).
    auto = []
    wrote_image = any(os.path.splitext(p)[1].lower()
                      in (".png", ".jpg", ".jpeg", ".svg", ".pdf")
                      for p in record["written"])
    try:
        nums = list(plt.get_fignums())
        if nums and not wrote_image and result["ok"]:
            for k, num in enumerate(nums, 1):
                fname = stem + (".png" if len(nums) == 1 else f"_{k}.png")
                _save(plt.figure(num), fname, 200)
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
        "stdout": out.text(),
        "stderr": err.text(),
        "refused": record["refused"],
        "run_seconds": round(run_s, 2),
        "seconds": round(time.monotonic() - t0, 2),
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
