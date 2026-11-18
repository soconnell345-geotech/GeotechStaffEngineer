"""The kernel layer under the plotting box (BACKLOG INF-16).

``GEOTECH_PLOT_CODE_KERNEL_LAYER`` is off unless set. With it unset the box
is as it was: nothing new is loaded and no new system call is made. With it
at ``1`` the box's CHILD process puts on itself what the kernel offers (no
new privileges, Landlock file and TCP rules, a seccomp filter); ``require``
refuses to run code where the file rules cannot go up.

The layer is exercised here DIRECTLY, in a throwaway process, without the
box's Python-level guard: a plain ``open``, a plain socket, a plain
``fork``. Every value is made up. The tests that need Landlock or seccomp
skip, with the reason, where the kernel has none.
"""

import json
import os
import socket
import subprocess
import sys
import types

import pytest

from aec_agent_core.plot_box import kernel_layer

LAYER = os.path.abspath(kernel_layer.__file__)
CHILD = os.path.join(os.path.dirname(LAYER), "child.py")
ENV = "GEOTECH_PLOT_CODE_KERNEL_LAYER"
LINUX = sys.platform.startswith("linux")
ABI, WHY_NOT = kernel_layer.landlock_abi()

needs_linux = pytest.mark.skipif(
    not LINUX, reason="the kernel layer is Linux only")
needs_landlock = pytest.mark.skipif(
    not ABI, reason=f"this kernel offers no Landlock: {WHY_NOT}")
needs_filter = pytest.mark.skipif(
    not LINUX or kernel_layer._machine() not in kernel_layer._ARCH,
    reason="the seccomp filter is written for x86_64 and aarch64 Linux")


@pytest.fixture(autouse=True)
def _setting_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def _python(script, arg, **kw):
    """Run ``script`` in a fresh interpreter; its last line of JSON."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in sys.path if p and os.path.isdir(p)))
    env.pop(ENV, None)
    env.update(kw.pop("env", {}))
    done = subprocess.run([sys.executable, "-B", "-c", script,
                           json.dumps(arg)], capture_output=True, text=True,
                          timeout=300, env=env, **kw)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"exit {done.returncode}\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# the setting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, mode", [
    (None, ""), ("", ""), ("  ", ""), ("0", ""), ("off", ""), ("No", ""),
    ("false", ""), ("1", "1"), ("on", "1"), ("TRUE", "1"), (" yes ", "1"),
    ("require", "require"), (" Require ", "require"),
    ("required", "?"), ("2", "?"), ("landlock", "?"),
])
def test_the_settings_values(raw, mode):
    assert kernel_layer.mode(raw) == mode


def test_the_setting_is_not_one_of_the_boxs_numbers():
    """The review suite clears ``SETTINGS_ENVS`` between arms; a host's
    kernel layer is not the suite's to clear, and the quiet application's
    tests pin that tuple."""
    from aec_agent_core import plot_box, review_flags
    assert plot_box.KERNEL_LAYER_ENV == ENV == kernel_layer.ENV
    assert ENV not in plot_box.SETTINGS_ENVS
    assert ENV not in review_flags.SETTINGS_ENVS


# ---------------------------------------------------------------------------
# the filter, checked without a kernel (both machines)
# ---------------------------------------------------------------------------

def _filter_verdict(rows, arch_word, number, arg0=0):
    """What the filter returns for one system call: a small reading of the
    classic BPF the filter uses."""
    data = {0: number, 4: arch_word, 16: arg0 & 0xFFFFFFFF}
    acc, i = 0, 0
    for _ in range(len(rows) + 1):
        code, jt, jf, k = rows[i]
        assert 0 <= jt < 256 and 0 <= jf < 256
        if code == kernel_layer._LD_W_ABS:
            acc, i = data[k], i + 1
        elif code == kernel_layer._JEQ_K:
            i += 1 + (jt if acc == k else jf)
        elif code == kernel_layer._JGE_K:
            i += 1 + (jt if acc >= k else jf)
        elif code == kernel_layer._JSET_K:
            i += 1 + (jt if acc & k else jf)
        else:
            assert code == kernel_layer._RET_K
            return k
    raise AssertionError("the filter does not end")


@pytest.mark.parametrize("machine, write, thread_flags", [
    ("x86_64", 1, 0x003D0F00), ("aarch64", 64, 0x003D0F00)])
def test_the_filter_refuses_sockets_and_processes_and_lets_threads(
        machine, write, thread_flags):
    import errno
    arch = kernel_layer._ARCH[machine]
    rows = kernel_layer._filter_program(arch)
    allow = kernel_layer._RET_ALLOW
    eperm = kernel_layer._RET_ERRNO | errno.EPERM
    enosys = kernel_layer._RET_ERRNO | errno.ENOSYS

    def verdict(number, arg0=0, word=arch["audit"]):
        return _filter_verdict(rows, word, number, arg0)

    assert verdict(write) == allow
    for name, number in arch["refused"].items():
        assert verdict(number) == eperm, name
    assert verdict(arch["clone3"]) == enosys
    assert verdict(arch["clone"], thread_flags) == allow       # a thread
    assert verdict(arch["clone"], 0x11) == eperm               # a process
    assert verdict(arch["clone"], thread_flags | 0x40000000) == eperm
    assert verdict(arch["clone"], thread_flags | 0x10000000) == eperm
    # a call made by another machine's numbers is refused whole
    other = 0x40000003 if machine == "x86_64" else 0xC000003E
    assert verdict(write, word=other) == eperm
    if machine == "x86_64":
        assert verdict(0x40000000 + 41) == eperm               # x32 socket


# ---------------------------------------------------------------------------
# the paths
# ---------------------------------------------------------------------------

def test_the_paths_allow_writing_only_in_the_conversation_folder(tmp_path):
    conv = tmp_path / "conversations" / "c-0001"
    conv.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    lines = tmp_path / "lines.py"
    lines.write_text("# made up\n")
    rows = kernel_layer.box_paths(str(conv), str(scratch),
                                  files=[str(lines)])
    kinds = {p: kind for p, kind, _why in rows}
    assert kinds[os.path.realpath(conv)] == "write"
    assert kinds[os.path.realpath(scratch)] == "scratch"
    assert kinds[os.path.realpath(lines)] == "read"
    writable = sorted(p for p, kind in kinds.items() if kind != "read")
    assert set(writable) - {os.path.realpath(conv),
                            os.path.realpath(scratch)} <= {"/dev/null"}
    home = os.path.realpath(os.path.expanduser("~"))
    for path, kind, why in rows:
        assert why and os.path.exists(path)
        if kind == "read":
            # never the root, the home folder, or a folder that holds the
            # conversation folder (its neighbours would be readable)
            assert path not in (os.sep, home)
            assert not str(conv).startswith(path.rstrip(os.sep) + os.sep)


def test_a_read_root_that_holds_the_conversation_folder_is_left_out(
        tmp_path, monkeypatch):
    """Libraries installed in a folder above the conversations (a made-up
    layout): that folder is not opened for reading."""
    site = tmp_path / "app"
    conv = site / "conversations" / "c-0002"
    conv.mkdir(parents=True)
    pkg = site / "madeuplib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(site))
    rows = kernel_layer.box_paths(str(conv), libraries=("madeuplib",))
    assert os.path.realpath(site) not in [p for p, _k, _w in rows]


# ---------------------------------------------------------------------------
# the kernel layer ALONE (no audit hook, no static check, plain calls)
# ---------------------------------------------------------------------------

ALONE = r"""
import importlib.util, json, os, socket, subprocess, sys, threading
a = json.loads(sys.argv[1])
spec = importlib.util.spec_from_file_location("_layer", a["layer"])
layer = importlib.util.module_from_spec(spec); spec.loader.exec_module(layer)
status = os.open("/proc/self/status", os.O_RDONLY)
report = layer.put_up("1", a["conv"], a["scratch"], parts=tuple(a["parts"]))
out = {}

def attempt(name, fn):
    try:
        got = fn()
        out[name] = "ok" if got is None else got
    except OSError as exc:
        out[name] = type(exc).__name__

def read(path):
    with open(path) as fh:
        return fh.read()

def write(path):
    with open(path, "w") as fh:
        fh.write("made up")

def connect():
    socket.create_connection(("127.0.0.1", a["port"]), timeout=5).close()

def udp():
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()

def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)

def thread():
    t = threading.Thread(target=lambda: None); t.start(); t.join()

attempt("read_inside", lambda: read(os.path.join(a["conv"], "data.csv")))
attempt("write_inside", lambda: write(os.path.join(a["conv"], "new.png")))
attempt("mkdir_inside", lambda: os.mkdir(os.path.join(a["conv"], "sub")))
attempt("write_scratch", lambda: write(os.path.join(a["scratch"], "t.tmp")))
attempt("read_outside", lambda: read(a["secret"]))
attempt("list_outside", lambda: os.listdir(os.path.dirname(a["secret"])))
attempt("write_outside", lambda: write(a["secret"] + ".new"))
attempt("overwrite_outside", lambda: write(a["secret"]))
attempt("delete_inside", lambda: os.remove(
    os.path.join(a["conv"], "data.csv")))
attempt("read_proc", lambda: read("/proc/self/environ") and None)
attempt("tcp_connect", connect)
attempt("udp_socket", udp)
attempt("fork", fork)
attempt("start_program", lambda: subprocess.run([sys.executable, "-c",
                                                 "pass"]) and None)
attempt("thread", thread)
flags = dict(line.split(":", 1) for line in
             os.read(status, 65536).decode().splitlines() if ":" in line)
print(json.dumps({"report": report, "out": out,
                  "no_new_privs": flags["NoNewPrivs"].strip(),
                  "seccomp": flags["Seccomp"].strip()}))
"""


@pytest.fixture
def places(tmp_path):
    """A conversation folder, the box's scratch folder, a neighbour's
    folder with a file in it, and a TCP listener on this machine."""
    conv = tmp_path / "conv"
    scratch = tmp_path / "scratch"
    other = tmp_path / "neighbour"
    for d in (conv, scratch, other):
        d.mkdir()
    (conv / "data.csv").write_text("depth_m,value\n1.0,12\n2.0,15\n")
    (other / "notes.txt").write_text("made-up private notes")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    yield {"layer": LAYER, "conv": str(conv), "scratch": str(scratch),
           "secret": str(other / "notes.txt"),
           "port": server.getsockname()[1]}
    server.close()


def _alone(places, parts):
    return _python(ALONE, dict(places, parts=parts))


@needs_landlock
def test_landlock_alone_keeps_files_to_the_conversation_folder(places):
    got = _alone(places, ["landlock"])
    report, out = got["report"], got["out"]
    assert report["files"] and "files" in report["up"], report
    assert report["threads_before"] == 1
    assert out["read_inside"].startswith("depth_m")
    assert out["write_inside"] == out["mkdir_inside"] == "ok"
    assert out["write_scratch"] == "ok"
    for name in ("read_outside", "list_outside", "write_outside",
                 "overwrite_outside", "read_proc", "delete_inside"):
        assert out[name] == "PermissionError", (name, out)
    # nothing may be executed: no path carries that right
    assert out["start_program"] == "PermissionError", out
    # the neighbour's file is as it was
    with open(places["secret"]) as fh:
        assert fh.read() == "made-up private notes"
    assert not os.path.exists(places["secret"] + ".new")
    assert os.path.isfile(os.path.join(places["conv"], "new.png"))
    assert got["seccomp"] == "0"                 # this part alone
    if ABI >= 4:
        assert "tcp" in report["up"]
        assert out["tcp_connect"] == "PermissionError", out
    else:
        assert "tcp" in report["not_up"]
    assert out["thread"] == "ok"


@needs_filter
def test_the_filter_alone_refuses_sockets_and_new_processes(places):
    got = _alone(places, ["seccomp"])
    report, out = got["report"], got["out"]
    if "seccomp" not in report["up"]:
        pytest.skip("this host lets no seccomp filter be set: "
                    + str(report["not_up"].get("seccomp")))
    assert got["no_new_privs"] == "1" and got["seccomp"] == "2"
    for name in ("tcp_connect", "udp_socket", "fork", "start_program"):
        assert out[name] == "PermissionError", (name, out)
    assert out["thread"] == "ok"                 # a thread is not a process
    assert out["read_outside"] == "made-up private notes"   # no file rules


@needs_landlock
def test_the_whole_layer_alone(places):
    got = _alone(places, ["landlock", "seccomp"])
    report, out = got["report"], got["out"]
    assert {"no_new_privs", "files"} <= set(report["up"]), report
    assert got["no_new_privs"] == "1"
    assert out["read_inside"].startswith("depth_m")
    assert out["write_inside"] == "ok"
    for name in ("read_outside", "write_outside", "list_outside"):
        assert out[name] == "PermissionError", (name, out)
    if ABI >= 4 or "seccomp" in report["up"]:
        assert out["tcp_connect"] == "PermissionError", out
    if "seccomp" in report["up"]:
        assert out["udp_socket"] == out["fork"] == "PermissionError", out


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

def test_the_probe_is_one_line_of_json_and_restricts_nothing():
    done = subprocess.run(
        [sys.executable, "-B", "-m", "aec_agent_core.plot_box.kernel_probe"],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
            p for p in sys.path if p and os.path.isdir(p))))
    assert done.returncode == 0, done.stderr
    (line,) = done.stdout.splitlines()
    got = json.loads(line)
    assert got["system"] == sys.platform
    if not LINUX:
        assert got["available"] is False and "not available" in got["why"]
        return
    assert got["kernel"] == os.uname().release
    assert got["landlock_version"] == ABI
    assert got["caller_changed"] is False
    assert got["caller_before"] == got["caller_after"]
    assert set(got["caller_after"]) >= {"NoNewPrivs", "Seccomp"}
    assert got["available"] is bool(ABI) or got["not_available"]
    if not ABI:
        assert got["landlock_why_not"] == WHY_NOT


def test_the_reason_landlock_is_missing_is_told_apart(monkeypatch):
    """Not built in, and built in but not enabled at boot, are different
    things for an administrator to do."""
    import errno

    def libc_answering(err):
        class Libc:
            @staticmethod
            def syscall(*_a):
                return -1
        ctypes = types.SimpleNamespace(
            c_long=int, c_size_t=int, c_uint=int, get_errno=lambda: err)
        return lambda: (ctypes, Libc)

    monkeypatch.setattr(kernel_layer, "sys",
                        types.SimpleNamespace(platform="linux"))
    monkeypatch.setattr(kernel_layer, "_libc", libc_answering(errno.ENOSYS))
    assert "not built into" in kernel_layer.landlock_abi()[1]
    monkeypatch.setattr(kernel_layer, "_libc",
                        libc_answering(errno.EOPNOTSUPP))
    assert "not enabled at boot" in kernel_layer.landlock_abi()[1]


# ---------------------------------------------------------------------------
# another system: "1" says so, "require" refuses
# ---------------------------------------------------------------------------

def test_on_another_system_nothing_is_tried(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_layer, "sys", types.SimpleNamespace(
        platform="win32", byteorder="little"))
    monkeypatch.setattr(kernel_layer, "_libc", lambda: 1 / 0)
    assert kernel_layer.landlock_abi() == (
        0, "not available on this system (win32)")
    report = kernel_layer.put_up("1", str(tmp_path))
    assert report["up"] == [] and report["files"] is False
    assert set(report["not_up"].values()) == {
        "not available on this system (win32)"}


# ---------------------------------------------------------------------------
# the box, through its normal entry
# ---------------------------------------------------------------------------

MATPLOTLIB = (
    "import csv\nimport matplotlib.pyplot as plt\n"
    "with open('data.csv', newline='') as fh:\n"
    "    rows = list(csv.DictReader(fh))\n"
    "fig, ax = plt.subplots(figsize=(5, 3))\n"
    "ax.plot([float(r['value']) for r in rows],\n"
    "        [float(r['depth_m']) for r in rows], marker='o')\n"
    "ax.plot(data['x'], data['y'])\n"
    "ax.set_xlabel('Value (kPa)')\nax.set_ylabel('Depth (m)')\n"
    "save_figure(fig, 'made_up_profile')\n")
PLOTLY = (
    "import plotly.graph_objects as go\n"
    "fig = go.Figure(go.Scatter(x=data['x'], y=data['y'], name='made up'))\n"
    "save_chart(fig, 'made_up_chart')\n")
DATA = {"x": [10.0, 14.0, 19.0], "y": [1.0, 2.0, 3.0]}
#: The payload's keys and the result's ``box`` keys as they are with the
#: setting unset (and were before the kernel layer existed).
PAYLOAD_KEYS = {"code", "data", "name", "root", "tmp", "marker",
                "print_width_in", "chart_lines_path", "sys_path", "limits"}
BOX_KEYS = {"limits_applied", "run_seconds", "start_seconds",
            "layout_check_seconds"}


@pytest.fixture
def box(tmp_path, monkeypatch):
    """The box on a made-up conversation folder; ``payloads`` is what each
    child process was handed, ``started`` how many were started."""
    pytest.importorskip("matplotlib")
    from aec_agent_core import plot_box
    conv = tmp_path / "conv"
    conv.mkdir()
    (conv / "data.csv").write_text("depth_m,value\n1.0,12\n2.0,15\n")
    seen = types.SimpleNamespace(payloads=[], started=0, conv=conv,
                                 run=None, module=plot_box)
    real = subprocess.Popen

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.started += 1
            super().__init__(*a, **kw)

        def communicate(self, input=None, timeout=None):
            if input is not None:
                seen.payloads.append(json.loads(input))
            return super().communicate(input=input, timeout=timeout)

    monkeypatch.setattr(plot_box.subprocess, "Popen", Spy)
    seen.run = lambda code, **kw: plot_box.run_plot_code(
        code, data=DATA, root=str(conv), **kw)
    return seen


@pytest.mark.parametrize("value", [None, "0", "off", ""])
def test_unset_the_box_is_as_it_was(box, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    r = box.run(MATPLOTLIB)
    assert r["status"] == "success", r
    (payload,) = box.payloads
    assert set(payload) == PAYLOAD_KEYS
    assert set(r["box"]) == BOX_KEYS
    assert not any("kernel layer" in n for n in r.get("notes", []))


UNSET_CHILD = r"""
import io, json, os, runpy, sys
a = json.loads(sys.argv[1])
loaded = []
sys.addaudithook(lambda event, args: loaded.append(str(args[0]))
                 if event == "open" and "kernel_layer." in
                 os.path.basename(str(args[0])) else None)
status = os.open("/proc/self/status", os.O_RDONLY) if a["linux"] else None
sys.stdin = io.TextIOWrapper(io.BytesIO(json.dumps(a["payload"]).encode()))
try:
    runpy.run_path(a["child"], run_name="__main__")
except SystemExit:
    pass
flags = {}
if status is not None:
    flags = dict(line.split(":", 1) for line in
                 os.read(status, 65536).decode().splitlines() if ":" in line)
sys.__stdout__.write("\n" + json.dumps({
    "layer_file_opened": len(loaded),
    "no_new_privs": flags.get("NoNewPrivs", "").strip(),
    "seccomp": flags.get("Seccomp", "").strip()}) + "\n")
"""


def _child_run(tmp_path, asked):
    """The box's child run as the box runs it, in a process that reports
    its own kernel flags afterwards."""
    pytest.importorskip("matplotlib")
    conv = tmp_path / "conv"
    scratch = tmp_path / "scratch"
    conv.mkdir()
    scratch.mkdir()
    payload = {"code": "import matplotlib.pyplot as plt\n"
                       "plt.plot(data['x'], data['y'])\n",
               "data": DATA, "name": "made_up", "root": str(conv),
               "tmp": str(scratch), "marker": "@@T@@",
               "sys_path": [p for p in sys.path if p and os.path.isdir(p)],
               "limits": {"memory_mb": 2048, "cpu_s": 120, "fsize_mb": 50,
                          "max_files": 40}}
    if asked:
        payload["kernel_layer"] = asked
    # the environment the box gives its child (scrubbed), and -I as there
    from aec_agent_core import plot_box
    done = subprocess.run(
        [sys.executable, "-I", "-B", "-c", UNSET_CHILD,
         json.dumps({"payload": payload, "child": CHILD, "linux": LINUX})],
        capture_output=True, text=True, timeout=300,
        env=plot_box._child_env(str(scratch)))
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"exit {done.returncode}\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1]), os.path.isfile(conv / "made_up.png")


def test_unset_the_child_makes_none_of_the_new_system_calls(tmp_path):
    """No payload key: the layer's file is never opened, so none of its
    calls can be made; on Linux the process ends with no-new-privileges
    unset and no seccomp filter."""
    flags, drew = _child_run(tmp_path, "")
    assert drew
    assert flags["layer_file_opened"] == 0
    if LINUX:
        assert flags["no_new_privs"] == "0" and flags["seccomp"] == "0"


@needs_landlock
def test_asked_the_child_puts_the_layer_on_itself(tmp_path):
    flags, drew = _child_run(tmp_path, "1")
    assert drew
    assert flags["layer_file_opened"] >= 1
    assert flags["no_new_privs"] == "1"


@needs_landlock
def test_at_1_the_box_still_draws_with_matplotlib(box, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    r = box.run(MATPLOTLIB)
    assert r["status"] == "success", r
    assert [i["file"] for i in r["images"]] == ["made_up_profile.png"]
    assert (box.conv / "made_up_profile.png").stat().st_size > 1000
    (payload,) = box.payloads
    assert payload["kernel_layer"] == "1"
    layer = r["box"]["kernel_layer"]
    assert layer["asked"] == "1" and layer["files"] is True
    assert {"no_new_privs", "files"} <= set(layer["up"])
    assert layer["threads_before"] == 1
    assert layer["landlock_version"] == ABI
    for part, why in layer["not_up"].items():
        # every part that is not up is said, with its reason
        assert why and any(part in n for n in r["notes"])
    assert not r.get("refused")


@needs_landlock
def test_at_1_the_box_still_draws_with_plotly(box, monkeypatch):
    pytest.importorskip("plotly")
    monkeypatch.setenv(ENV, "1")
    r = box.run(PLOTLY)
    assert r["status"] == "success", r
    assert [c["file"] for c in r["charts"]] == ["made_up_chart.plotly.json"]
    with open(box.conv / "made_up_chart.plotly.json") as fh:
        assert json.load(fh)["data"][0]["x"] == DATA["x"]
    assert r["box"]["kernel_layer"]["files"] is True


@needs_landlock
def test_require_draws_where_the_layer_is_there(box, monkeypatch):
    monkeypatch.setenv(ENV, "require")
    r = box.run(MATPLOTLIB)
    assert r["status"] == "success", r
    assert r["box"]["kernel_layer"]["asked"] == "require"


@pytest.mark.parametrize("why", [
    "Landlock is not built into this kernel (or the kernel is older than "
    "5.13)",
    "not available on this system (win32)"])
def test_require_refuses_without_the_layer_and_starts_no_process(
        box, monkeypatch, why):
    monkeypatch.setenv(ENV, "require")
    monkeypatch.setattr(kernel_layer, "landlock_abi", lambda: (0, why))
    r = box.run(MATPLOTLIB)
    assert r["status"] == "error"
    assert "require" in r["error"] and why in r["error"]
    assert box.started == 0
    assert not (box.conv / "made_up_profile.png").exists()


def test_a_value_the_box_does_not_know_is_refused_not_run_as_off(
        box, monkeypatch):
    monkeypatch.setenv(ENV, "required")
    r = box.run(MATPLOTLIB)
    assert r["status"] == "error" and ENV in r["error"]
    assert box.started == 0


def test_at_1_without_the_layer_the_box_runs_and_says_so(box, monkeypatch):
    """What ``1`` gives on Windows, macOS or a kernel without the layer:
    the box as before, with the missing parts said. The child's report is
    made up here, as such a system would give it."""
    monkeypatch.setenv(ENV, "1")
    why = "not available on this system (win32)"
    made_up = {"asked": "1", "system": "win32", "files": False, "up": [],
               "not_up": {n: why for n in ("no_new_privs", "files", "tcp",
                                           "scopes", "seccomp")}}
    real = box.module.subprocess.Popen

    class Told(real):
        def communicate(self, input=None, timeout=None):
            if input is not None:
                payload = json.loads(input)
                self.marker = payload.pop("kernel_layer") and \
                    payload["marker"]
                input = json.dumps(payload).encode()
            out, err = super().communicate(input=input, timeout=timeout)
            lines = []
            for line in out.decode().splitlines():
                if line.startswith(self.marker):
                    res = json.loads(line[len(self.marker):])
                    res["kernel_layer"] = made_up
                    line = self.marker + json.dumps(res)
                lines.append(line)
            return "\n".join(lines).encode(), err

    monkeypatch.setattr(box.module.subprocess, "Popen", Told)
    r = box.run(MATPLOTLIB)
    assert r["status"] == "success", r
    assert r["box"]["kernel_layer"]["up"] == []
    (note,) = [n for n in r["notes"] if n.startswith("kernel layer")]
    assert "up: nothing" in note and why in note


# ---------------------------------------------------------------------------
# the page's own process is never restricted
# ---------------------------------------------------------------------------

PAGE = r"""
import json, os, socket, sys
a = json.loads(sys.argv[1])
from aec_agent_core import plot_box
r = plot_box.run_plot_code(a["code"], data=a["data"], root=a["conv"])
with open("/proc/self/status") as fh:
    flags = dict(line.split(":", 1) for line in fh if ":" in line)
with open(a["secret"]) as fh:                # a file outside every rule
    secret = fh.read()
socket.socket(socket.AF_INET, socket.SOCK_DGRAM).close()
pid = os.fork()
if pid == 0:
    os._exit(0)
os.waitpid(pid, 0)
print(json.dumps({
    "status": r["status"], "layer": r.get("box", {}).get("kernel_layer"),
    "no_new_privs": flags["NoNewPrivs"].strip(),
    "seccomp": flags["Seccomp"].strip(), "secret": secret,
    "layer_imported":
        "aec_agent_core.plot_box.kernel_layer" in sys.modules}))
"""


def _page(tmp_path, env):
    pytest.importorskip("matplotlib")
    conv = tmp_path / "conv"
    other = tmp_path / "neighbour"
    conv.mkdir()
    other.mkdir()
    (conv / "data.csv").write_text("depth_m,value\n1.0,12\n2.0,15\n")
    (other / "notes.txt").write_text("made-up private notes")
    return _python(PAGE, {"code": MATPLOTLIB, "data": DATA,
                          "conv": str(conv),
                          "secret": str(other / "notes.txt")}, env=env)


@needs_landlock
def test_the_pages_process_is_not_restricted(tmp_path):
    got = _page(tmp_path, {ENV: "1"})
    assert got["status"] == "success"
    assert got["layer"]["files"] is True       # the child was restricted
    # ... and the process that called the box was not: it read a file no
    # rule allows, made a socket and started a process after the call
    assert got["no_new_privs"] == "0" and got["seccomp"] == "0"
    assert got["secret"] == "made-up private notes"


@needs_linux
def test_unset_the_pages_process_does_not_load_the_layer(tmp_path):
    got = _page(tmp_path, {})
    assert got["status"] == "success" and got["layer"] is None
    assert got["layer_imported"] is False
    assert got["no_new_privs"] == "0" and got["seccomp"] == "0"
