"""A kernel layer under the plotting box, on Linux (BACKLOG INF-16).

The box's static check, audit hook and builtins are Python-level: they hold
while the interpreter itself is sound. This file adds restrictions the
kernel keeps, which the code in the box cannot reach whatever it does to the
interpreter. The box's CHILD process puts them on itself
(:func:`put_up`, called from ``child.py`` before any library is loaded);
nothing here is ever applied to the process of the page.

OFF unless the host asks: ``GEOTECH_PLOT_CODE_KERNEL_LAYER`` (read by
:func:`aec_agent_core.plot_box.kernel_layer_mode` in the page's process
each time the box is run):

unset, empty, ``0`` / ``false`` / ``no`` / ``off``
    the box as it was: this file is not loaded, no new system call is made
    and the result has no new key;
``1`` / ``true`` / ``yes`` / ``on``
    the child puts up what this kernel offers; the box's result says under
    ``box.kernel_layer`` which parts went up and which did not, and why. On
    Windows and macOS: "not available on this system", and the box runs as
    before;
``require``
    as ``1``, and where the FILE layer (Landlock) cannot go up the box
    refuses to run the code, with a plain error: before a process is
    started where the kernel has no Landlock, in the child before any
    library is loaded where putting it up fails;
anything else
    refused with a plain error (a mistyped ``require`` must not run as
    "off").

The parts, in the order they go up (all are restrictions a process may put
on itself: no namespace, no mount, no privilege):

1. ``prctl(PR_SET_NO_NEW_PRIVS)``: nothing this process starts can gain a
   privilege (the kernel also asks for it before parts 2 and 3).
2. LANDLOCK (Linux 5.13 and later, where enabled):

   - files: every file right the kernel's Landlock version knows is denied
     everywhere, except reading under the paths :func:`box_paths` lists
     (the installed libraries, the interpreter's own files, fonts, time
     zones) and reading and writing in the conversation folder and the
     box's own scratch folder. No path may be executed. Landlock cannot
     tell a ``.png`` from a ``.py``: which KINDS of file are written stays
     the audit hook's rule;
   - TCP (version 4 and later): no ``bind`` and no ``connect``;
   - scopes (version 6 and later): no signal to a process outside the box,
     no connection to an abstract Unix socket outside it.

   A Landlock domain also may not trace (``ptrace``) a process outside it,
   and may not mount.
3. A SECCOMP filter (x86_64 and aarch64, little-endian): :data:`REFUSED`
   are refused with ``EPERM``: making a socket of any family (so no UDP and
   no Unix socket either, which Landlock does not cover), starting a
   process (``fork``, ``vfork``, ``execve``, ``execveat``, and ``clone``
   unless it makes a thread in this process), ``io_uring`` (which can make
   sockets without ``socket``), ``ptrace`` and reading or writing another
   process's memory, and joining or making a namespace. ``clone3`` answers
   "not implemented", so the C library falls back to ``clone``, whose flags
   the filter can read. A system call made the 32-bit way (which has other
   numbers) is refused whole.

ORDER. A Landlock domain covers the calling thread and what it starts
later, not threads that already exist. The child puts the layer up as its
first act after reading its payload, when only the standard library has
been imported and one thread exists; the number of threads counted just
before is in the report (``threads_before``).

Standard library only (``ctypes``). ``child.py`` loads this file by its
path, as it loads its other helpers, so it imports nothing of the package.
The administrator's probe is :mod:`aec_agent_core.plot_box.kernel_probe`.
"""

from __future__ import annotations

import errno
import os
import stat
import sys

ENV = "GEOTECH_PLOT_CODE_KERNEL_LAYER"
OFF_WORDS = ("", "0", "false", "no", "off")
ON_WORDS = ("1", "true", "yes", "on")
REQUIRE = "require"

# -- Landlock (include/uapi/linux/landlock.h) -------------------------------
_NR_CREATE_RULESET, _NR_ADD_RULE, _NR_RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1

FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1, 2, 4, 8
FS_REMOVE_DIR, FS_REMOVE_FILE, FS_MAKE_DIR, FS_MAKE_REG = 16, 32, 128, 256
FS_REFER, FS_TRUNCATE, FS_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
#: Every file right a Landlock version knows (all are handled, so all are
#: denied where no rule allows them).
_FS_BY_ABI = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1}
_FS_ALL = (1 << 16) - 1                                    # version 5 on
#: Rights that can be granted on a path that is a file, not a folder.
_FS_FILE = (FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE
            | FS_IOCTL_DEV)
NET_BIND_TCP, NET_CONNECT_TCP = 1, 2
SCOPE_ABSTRACT_UNIX_SOCKET, SCOPE_SIGNAL = 1, 2

READ = FS_READ_FILE | FS_READ_DIR
#: In the conversation folder: read, write, new files and folders. No
#: delete and no rename (the box does neither there).
WRITE = READ | FS_WRITE_FILE | FS_MAKE_REG | FS_MAKE_DIR | FS_TRUNCATE
#: In the box's own scratch folder, which the page removes after the run.
SCRATCH = WRITE | FS_REMOVE_FILE | FS_REMOVE_DIR

# -- prctl and seccomp (linux/prctl.h, linux/seccomp.h, linux/filter.h) -----
_PR_SET_NO_NEW_PRIVS, _PR_GET_NO_NEW_PRIVS, _PR_SET_SECCOMP = 38, 39, 22
_SECCOMP_MODE_FILTER = 2
_RET_ALLOW, _RET_ERRNO = 0x7FFF0000, 0x00050000
_LD_W_ABS, _JEQ_K, _JGE_K, _JSET_K, _RET_K = 0x20, 0x15, 0x35, 0x45, 0x06
_CLONE_THREAD = 0x00010000
#: clone flags that make a namespace (CLONE_NEWTIME, NEWNS, NEWCGROUP,
#: NEWUTS, NEWIPC, NEWUSER, NEWPID, NEWNET).
_CLONE_NEW = (0x80 | 0x00020000 | 0x02000000 | 0x04000000 | 0x08000000
              | 0x10000000 | 0x20000000 | 0x40000000)

#: Per machine: the audit architecture word and the numbers of the system
#: calls the filter refuses (asm/unistd_64.h; asm-generic/unistd.h).
_ARCH = {
    "x86_64": {
        "audit": 0xC000003E, "clone": 56, "clone3": 435,
        "refused": {"socket": 41, "fork": 57, "vfork": 58, "execve": 59,
                    "ptrace": 101, "unshare": 272, "setns": 308,
                    "process_vm_readv": 310, "process_vm_writev": 311,
                    "execveat": 322, "io_uring_setup": 425}},
    "aarch64": {
        "audit": 0xC00000B7, "clone": 220, "clone3": 435,
        "refused": {"socket": 198, "execve": 221, "ptrace": 117,
                    "unshare": 97, "setns": 268, "process_vm_readv": 270,
                    "process_vm_writev": 271, "execveat": 281,
                    "io_uring_setup": 425}},
}
#: What the filter refuses, by name (``fork`` and ``vfork`` exist on x86_64
#: only; aarch64 starts a process with ``clone``).
REFUSED = ("socket", "fork", "vfork", "execve", "execveat",
           "clone (unless it makes a thread)", "clone3 (not implemented)",
           "io_uring_setup", "ptrace", "process_vm_readv",
           "process_vm_writev", "unshare", "setns")

#: The plotting libraries and what they load: the folder each is installed
#: in becomes readable.
LIBRARIES = ("numpy", "matplotlib", "pandas", "plotly", "scipy",
             "mpl_toolkits", "PIL", "kiwisolver", "contourpy", "dateutil",
             "pytz", "tzdata", "pyparsing", "cycler", "fontTools",
             "narwhals", "openpyxl", "et_xmlfile", "packaging", "six")


def mode(raw) -> str:
    """The setting's value as ``''`` (off), ``'1'``, ``'require'``, or
    ``'?'`` for a value that is none of these."""
    word = str(raw or "").strip().lower()
    if word in OFF_WORDS:
        return ""
    if word in ON_WORDS:
        return "1"
    return REQUIRE if word == REQUIRE else "?"


def _machine() -> str:
    try:
        m = os.uname().machine
    except (AttributeError, OSError):
        return ""
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(m.lower(), m)


def _libc():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.prctl.restype = ctypes.c_int
    return ctypes, libc


def _why(err: int) -> str:
    return f"{errno.errorcode.get(err, err)}: {os.strerror(err)}"


def landlock_abi() -> tuple:
    """``(version, '')`` where this kernel has Landlock, else ``(0, why)``.
    It only asks the kernel for a number: nothing is restricted."""
    if not sys.platform.startswith("linux"):
        return 0, f"not available on this system ({sys.platform})"
    try:
        ctypes, libc = _libc()
        n = libc.syscall(ctypes.c_long(_NR_CREATE_RULESET), None,
                         ctypes.c_size_t(0),
                         ctypes.c_uint(_CREATE_RULESET_VERSION))
        err = ctypes.get_errno()
    except Exception as exc:                               # noqa: BLE001
        return 0, f"the kernel could not be asked ({type(exc).__name__})"
    if n >= 1:
        return int(n), ""
    if err == errno.ENOSYS:
        return 0, ("Landlock is not built into this kernel (or the kernel "
                   "is older than 5.13)")
    if err == errno.EOPNOTSUPP:
        return 0, ("Landlock is built into this kernel but not enabled at "
                   "boot (the kernel's lsm= list leaves it out)")
    return 0, f"the kernel answered {_why(err)}"


def file_layer_missing() -> str:
    """Why the file layer cannot go up here; ``''`` where the kernel offers
    it. What ``require`` asks before a process is started."""
    return landlock_abi()[1]


# ---------------------------------------------------------------------------
# the paths
# ---------------------------------------------------------------------------

def _real(path) -> str:
    try:
        return os.path.realpath(os.fspath(path))
    except (TypeError, ValueError, OSError):
        return ""


def _inside(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def box_paths(root: str, tmp: str = "", files=(), libraries=LIBRARIES,
              box_dir: str = "") -> list:
    """What the file layer allows, as ``[path, 'read' | 'write' | 'scratch',
    why]`` rows; only paths that exist. Everything else is denied.

    A candidate for READING is left out when it is the file system's root,
    the home folder itself, or a folder that holds the conversation folder:
    the layer would then open what it is there to close.
    """
    import sysconfig
    root = _real(root)
    home = _real(os.path.expanduser("~"))
    rows, seen = [], set()

    def add(path, kind, why):
        p = _real(path) if path else ""
        if not p or p in seen or not os.path.exists(p):
            return
        if kind == "read" and (p == os.sep or p == home
                               or (root and _inside(root, p))):
            return
        seen.add(p)
        rows.append([p, kind, why])

    add(root, "write", "the conversation folder: the data the code reads "
                       "and the figures it writes")
    add(tmp, "scratch", "the box's own scratch folder for this run, removed "
                        "by the page afterwards")
    try:
        paths = sysconfig.get_paths()
    except Exception:                                      # noqa: BLE001
        paths = {}
    for key, why in (("stdlib", "the interpreter's standard library"),
                     ("platstdlib", "the standard library's compiled "
                                    "modules"),
                     ("purelib", "the installed libraries"),
                     ("platlib", "the installed libraries (compiled)")):
        add(paths.get(key), "read", why)
    import importlib.util
    for name in libraries:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:                                  # noqa: BLE001
            spec = None
        origin = getattr(spec, "origin", None)
        if origin and os.path.isabs(origin):
            add(os.path.dirname(os.path.dirname(origin)), "read",
                f"the folder {name} is installed in")
    box_dir = box_dir or os.path.dirname(os.path.abspath(__file__))
    add(box_dir, "read", "the box's own code")
    add(os.path.join(os.path.dirname(box_dir), "charts"), "read",
        "the box's layout check and plan furniture")
    for f in files:
        add(f, "read", "a file the page handed the box (standard chart "
                       "lines)")
    for d in ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/local/lib",
              os.path.join(sys.base_prefix, "lib")):
        add(d, "read", "shared libraries the interpreter and the compiled "
                       "modules load")
    add("/etc/ld.so.cache", "read", "the loader's index of shared libraries")
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(
        home, ".local", "share")
    for d in ("/usr/share/fonts", "/usr/local/share/fonts",
              "/usr/X11R6/lib/X11/fonts", "/usr/lib/X11/fonts",
              os.path.join(data_home, "fonts"), os.path.join(home, ".fonts")):
        add(d, "read", "fonts")
    for d in ("/usr/share/zoneinfo", "/usr/lib/zoneinfo",
              "/usr/share/lib/zoneinfo", "/etc/localtime"):
        add(d, "read", "time zones")
    # matplotlib's settings and its font list, by its own rule for where
    # they are; read only (it rebuilds the list in the scratch folder when
    # there is none)
    config = os.environ.get("MPLCONFIGDIR")
    if config:
        add(config, "read", "matplotlib's settings and font list")
    else:
        add(os.path.join(os.environ.get("XDG_CONFIG_HOME")
                         or os.path.join(home, ".config"), "matplotlib"),
            "read", "matplotlib's settings")
        add(os.path.join(os.environ.get("XDG_CACHE_HOME")
                         or os.path.join(home, ".cache"), "matplotlib"),
            "read", "matplotlib's font list")
    try:
        import mimetypes
        known = list(mimetypes.knownfiles)
    except Exception:                                      # noqa: BLE001
        known = []
    for f in known:
        add(f, "read", "the system's table of file types (read when a "
                       "workbook is opened)")
    add("/dev/urandom", "read", "random numbers")
    add("/dev/null", "write", "the null device")
    return rows


# ---------------------------------------------------------------------------
# the three parts
# ---------------------------------------------------------------------------

def _threads() -> int:
    """Threads in this process now (0 when it cannot be read)."""
    try:
        with open("/proc/self/status", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _no_new_privs(ctypes, libc) -> str:
    """``''`` when set, else why not."""
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, ctypes.c_ulong(1), ctypes.c_ulong(0),
                  ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        return _why(ctypes.get_errno())
    return ""


def _landlock(ctypes, libc, rows) -> dict:
    """Put the Landlock domain on this thread. ``rows`` as
    :func:`box_paths` gives them."""
    abi, why = landlock_abi()
    out = {"version": abi, "files": False, "tcp": False, "scopes": False}
    if not abi:
        out["why_not"] = why
        return out
    handled_fs = _FS_BY_ABI.get(abi, _FS_ALL)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64),
                    ("handled_access_net", ctypes.c_uint64),
                    ("scoped", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("parent_fd", ctypes.c_int32)]

    attr = Ruleset(handled_fs,
                   (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0,
                   (SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL)
                   if abi >= 6 else 0)
    # an older kernel takes only the fields it knows
    size = 8 if abi < 4 else (16 if abi < 6 else 24)
    fd = libc.syscall(ctypes.c_long(_NR_CREATE_RULESET), ctypes.byref(attr),
                      ctypes.c_size_t(size), ctypes.c_uint(0))
    if fd < 0:
        out["why_not"] = "no rule set: " + _why(ctypes.get_errno())
        return out
    allowed, skipped = [], []
    try:
        for path, kind, _reason in rows:
            access = {"read": READ, "write": WRITE,
                      "scratch": SCRATCH}[kind] & handled_fs
            try:
                pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError as exc:
                skipped.append([path, _why(exc.errno or 0)])
                continue
            try:
                if not stat.S_ISDIR(os.fstat(pfd).st_mode):
                    access &= _FS_FILE
                rule = PathBeneath(access, pfd)
                if libc.syscall(ctypes.c_long(_NR_ADD_RULE), ctypes.c_int(fd),
                                ctypes.c_int(_RULE_PATH_BENEATH),
                                ctypes.byref(rule), ctypes.c_uint(0)) != 0:
                    skipped.append([path, _why(ctypes.get_errno())])
                else:
                    allowed.append([path, kind])
            finally:
                os.close(pfd)
        if not any(kind == "write" for _p, kind in allowed):
            # without its folder the box could draw nothing: say so, and
            # leave the process as it was
            out["why_not"] = "no rule could be made for the conversation " \
                             "folder"
            out["skipped"] = skipped
            return out
        if libc.syscall(ctypes.c_long(_NR_RESTRICT_SELF), ctypes.c_int(fd),
                        ctypes.c_uint(0)) != 0:
            out["why_not"] = "not put on: " + _why(ctypes.get_errno())
            return out
    finally:
        os.close(fd)
    out.update(files=True, tcp=abi >= 4, scopes=abi >= 6,
               truncate=abi >= 3, allowed=allowed)
    if skipped:
        out["skipped"] = skipped
    return out


def _filter_program(arch: dict) -> list:
    """The seccomp filter as ``(code, jt, jf, k)`` rows."""
    rows, labels = [], {}

    def put(code, k, jt=0, jf=0):
        rows.append([code, jt, jf, k])

    def mark(name):
        labels[name] = len(rows)

    put(_LD_W_ABS, 4)                                      # the architecture
    put(_JEQ_K, arch["audit"], 0, "refuse")
    put(_LD_W_ABS, 0)                                      # the call's number
    if arch["audit"] == 0xC000003E:
        put(_JGE_K, 0x40000000, "refuse", 0)               # the x32 numbers
    for number in sorted(arch["refused"].values()):
        put(_JEQ_K, number, "refuse", 0)
    put(_JEQ_K, arch["clone3"], "not_implemented", 0)
    put(_JEQ_K, arch["clone"], "clone", 0)
    put(_RET_K, _RET_ALLOW)
    mark("clone")
    put(_LD_W_ABS, 16)                 # clone's flags: argument 0, low word
    put(_JSET_K, _CLONE_THREAD, 0, "refuse")
    put(_JSET_K, _CLONE_NEW, "refuse", 0)
    put(_RET_K, _RET_ALLOW)
    mark("refuse")
    put(_RET_K, _RET_ERRNO | errno.EPERM)
    mark("not_implemented")
    put(_RET_K, _RET_ERRNO | errno.ENOSYS)
    for i, row in enumerate(rows):
        for j in (1, 2):
            if isinstance(row[j], str):
                row[j] = labels[row[j]] - i - 1
    return rows


def _seccomp(ctypes, libc) -> str:
    """Install the filter; ``''`` when installed, else why not."""
    arch = _ARCH.get(_machine())
    if arch is None or sys.byteorder != "little":
        return (f"no filter is written for this machine ({_machine()}, "
                f"{sys.byteorder}-endian)")

    class Row(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                    ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort),
                    ("filter", ctypes.POINTER(Row))]

    rows = _filter_program(arch)
    table = (Row * len(rows))(*[Row(*r) for r in rows])
    program = Program(len(rows), table)
    if libc.prctl(_PR_SET_SECCOMP, ctypes.c_ulong(_SECCOMP_MODE_FILTER),
                  ctypes.byref(program), ctypes.c_ulong(0),
                  ctypes.c_ulong(0)) != 0:
        return _why(ctypes.get_errno())
    return ""


def put_up(asked: str, root: str, tmp: str = "", files=(),
           libraries=LIBRARIES, parts=("landlock", "seccomp"),
           rows=None) -> dict:
    """Put the kernel layer on THIS process, for good, and say what went up.

    For the box's child only: there is no taking it off, and a page that
    called it could no longer read its own files. ``parts`` lets a test put
    up one part alone; ``rows`` replaces :func:`box_paths`.

    The report: ``asked``; ``up`` and ``not_up`` (part: why) over
    ``no_new_privs``, ``files``, ``tcp``, ``scopes`` and ``seccomp``;
    ``files`` (whether the file layer is up, which ``require`` reads);
    ``landlock_version``; ``allowed`` (how many paths; the list itself is
    what :func:`box_paths` gives); ``threads_before``.
    """
    report = {"asked": asked, "system": sys.platform, "files": False,
              "up": [], "not_up": {}}
    names = ("no_new_privs", "files", "tcp", "scopes", "seccomp")
    if not sys.platform.startswith("linux"):
        why = f"not available on this system ({sys.platform})"
        report["not_up"] = {n: why for n in names}
        return report
    report["threads_before"] = _threads()
    try:
        ctypes, libc = _libc()
    except Exception as exc:                               # noqa: BLE001
        why = f"the C library could not be loaded ({type(exc).__name__})"
        report["not_up"] = {n: why for n in names}
        return report
    why = _no_new_privs(ctypes, libc)
    if why:
        report["not_up"]["no_new_privs"] = why
    else:
        report["up"].append("no_new_privs")
    if "landlock" in parts:
        if rows is None:
            rows = box_paths(root, tmp, files=files, libraries=libraries)
        land = _landlock(ctypes, libc, rows)
        report["landlock_version"] = land["version"]
        report["files"] = bool(land["files"])
        for part, needs in (("files", 1), ("tcp", 4), ("scopes", 6)):
            if land[part]:
                report["up"].append(part)
            else:
                report["not_up"][part] = land.get("why_not") or (
                    f"this kernel's Landlock is version {land['version']}; "
                    f"this part came with version {needs}")
        if land["files"]:
            report["allowed"] = len(land["allowed"])
            if not land.get("truncate"):
                report["not_up"]["truncate"] = (
                    "this kernel's Landlock (before version 3) cannot "
                    "refuse cutting a file short")
        if land.get("skipped"):
            report["skipped"] = land["skipped"][:10]
    if "seccomp" in parts:
        why = _seccomp(ctypes, libc)
        if why:
            report["not_up"]["seccomp"] = why
        else:
            report["up"].append("seccomp")
    return report


# ---------------------------------------------------------------------------
# the probe (a command an administrator runs; see kernel_probe.py)
# ---------------------------------------------------------------------------

def probe() -> dict:
    """What this host's kernel offers the layer. Changes nothing: the
    Landlock version is a number the kernel is asked for, and the three
    restrictions are TRIED in a forked child that exits at once, so the
    caller is never restricted.

    It forks: run it as a command (``python -m
    aec_agent_core.plot_box.kernel_probe``), not inside a running page.
    """
    out = {"system": sys.platform}
    try:
        uname = os.uname()
        out.update(kernel=uname.release, machine=uname.machine)
    except (AttributeError, OSError):
        pass
    if not sys.platform.startswith("linux"):
        out.update(available=False,
                   why=f"not available on this system ({sys.platform})")
        return out
    before = _caller_flags()
    abi, why = landlock_abi()
    out["landlock_version"] = abi
    if why:
        out["landlock_why_not"] = why
    import json
    import tempfile
    scratch = tempfile.mkdtemp(prefix="kernel_probe_")
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(r)
            got = put_up("probe", scratch, rows=[[scratch, "write", ""]])
            os.write(w, json.dumps(got).encode("ascii"))
            code = 0
        finally:
            os._exit(code)
    os.close(w)
    data = b""
    while True:
        chunk = os.read(r, 65536)
        if not chunk:
            break
        data += chunk
    os.close(r)
    os.waitpid(pid, 0)
    try:
        os.rmdir(scratch)
    except OSError:
        pass
    try:
        got = json.loads(data.decode("ascii"))
    except ValueError:
        got = {"not_up": {"all": "the throwaway child gave no answer"}}
    up, not_up = got.get("up") or [], got.get("not_up") or {}
    out.update(
        no_new_privs="no_new_privs" in up, landlock_files="files" in up,
        landlock_tcp="tcp" in up, landlock_scopes="scopes" in up,
        seccomp_filter="seccomp" in up, not_available=not_up,
        available="files" in up,
        # the caller's own kernel flags, read before and after: the probe
        # must leave them as they were (a container's own seccomp filter
        # shows here as seccomp 2 both times)
        caller_before=before, caller_after=_caller_flags(),
        uid_is_root=(os.geteuid() == 0))
    out["caller_changed"] = out["caller_before"] != out["caller_after"]
    return out


def _caller_flags() -> dict:
    """THIS process's ``NoNewPrivs``, ``Seccomp`` and ``Seccomp_filters``
    lines of ``/proc/self/status`` (empty when it cannot be read)."""
    flags = {}
    try:
        with open("/proc/self/status", encoding="ascii",
                  errors="replace") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key in ("NoNewPrivs", "Seccomp", "Seccomp_filters"):
                    flags[key] = value.strip()
    except OSError:
        pass
    return flags


__all__ = ["ENV", "REQUIRE", "REFUSED", "LIBRARIES", "mode", "landlock_abi",
           "file_layer_missing", "box_paths", "put_up", "probe"]
