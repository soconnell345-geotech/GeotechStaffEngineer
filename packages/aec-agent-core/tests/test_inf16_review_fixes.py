"""Fixes from the review of the kernel layer (BACKLOG INF-16;
``module_work/inf16/REVIEW.md``). Every value is made up. The layer is
called directly, in a throwaway process, as in
``test_inf16_kernel_layer.py``.
"""

import errno
import json
import os
import socket
import subprocess
import sys

import pytest

from aec_agent_core.plot_box import kernel_layer

LAYER = os.path.abspath(kernel_layer.__file__)
LINUX = sys.platform.startswith("linux")

#: The file rights each Landlock version knows, as a count of bits from bit
#: 0 (the kernel's include/uapi/linux/landlock.h: 13 rights in version 1,
#: REFER with 2, TRUNCATE with 3, IOCTL_DEV with 5; version 4 added the TCP
#: rules and no file right).
FS_RIGHTS = {1: 13, 2: 14, 3: 15, 4: 15, 5: 16, 6: 16, 7: 16}


class _Recorder:
    """A stand-in C library: it keeps the rule set the layer asks the
    kernel for and answers "no" to everything, so nothing is put up."""

    def __init__(self):
        self.asked = None

    def syscall(self, number, *args):
        import ctypes
        if number.value == kernel_layer._NR_CREATE_RULESET:
            attr, size = args[0]._obj, args[1].value
            self.asked = (attr.handled_access_fs, attr.handled_access_net,
                          attr.scoped, size)
        ctypes.set_errno(errno.EINVAL)
        return -1


@pytest.mark.parametrize("version", sorted(FS_RIGHTS))
def test_each_landlock_version_is_asked_only_for_rights_it_knows(
        monkeypatch, version):
    """A kernel refuses (EINVAL) a rule set that names a right it does not
    know, and then no file rule goes up at all. Version 4 (Linux 6.7 to
    6.9) knows the TCP rules but not the right for device ioctl."""
    import ctypes
    monkeypatch.setattr(kernel_layer, "landlock_abi", lambda: (version, ""))
    libc = _Recorder()
    out = kernel_layer._landlock(ctypes, libc, [])
    assert not out["files"]                       # the stand-in said no
    fs, net, scoped, size = libc.asked
    assert fs == (1 << FS_RIGHTS[version]) - 1, hex(fs)
    assert net == (3 if version >= 4 else 0)
    assert scoped == (3 if version >= 6 else 0)
    assert size == (8 if version < 4 else 16 if version < 6 else 24)


# ---------------------------------------------------------------------------
# the filter: a pair of sockets is a socket too
# ---------------------------------------------------------------------------

def _verdict(rows, arch_word, number, arg0=0):
    """Read the filter as the kernel would, for one call."""
    data = {0: number, 4: arch_word, 16: arg0 & 0xFFFFFFFF}
    acc, i = 0, 0
    while True:
        code, jt, jf, k = rows[i]
        i += 1
        if code == kernel_layer._LD_W_ABS:
            acc = data[k]
        elif code == kernel_layer._RET_K:
            return k
        else:
            hit = {kernel_layer._JEQ_K: acc == k,
                   kernel_layer._JGE_K: acc >= k,
                   kernel_layer._JSET_K: bool(acc & k)}[code]
            i += jt if hit else jf


#: ``socketpair`` in asm/unistd_64.h and asm-generic/unistd.h.
SOCKETPAIR = {"x86_64": 53, "aarch64": 199}


@pytest.mark.parametrize("machine", sorted(SOCKETPAIR))
def test_the_filter_refuses_a_pair_of_sockets(machine):
    arch = kernel_layer._ARCH[machine]
    rows = kernel_layer._filter_program(arch)
    refuse = kernel_layer._RET_ERRNO | errno.EPERM
    assert _verdict(rows, arch["audit"], SOCKETPAIR[machine]) == refuse
    assert _verdict(rows, arch["audit"], arch["refused"]["socket"]) == refuse
    # and what the libraries do all day is still let through
    for name in ("read", "write", "mmap"):
        number = {"x86_64": {"read": 0, "write": 1, "mmap": 9},
                  "aarch64": {"read": 63, "write": 64, "mmap": 222}}
        assert _verdict(rows, arch["audit"],
                        number[machine][name]) == kernel_layer._RET_ALLOW
    assert any("socketpair" in name for name in kernel_layer.REFUSED)


PAIR = r"""
import importlib.util, json, socket, sys
a = json.loads(sys.argv[1])
spec = importlib.util.spec_from_file_location("_layer", a["layer"])
layer = importlib.util.module_from_spec(spec); spec.loader.exec_module(layer)
report = layer.put_up("1", a["conv"], parts=("seccomp",))
out = {"up": report["up"], "not_up": report["not_up"]}
try:
    one, two = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    out["pair"] = "ok"
    one.sendto(b"made up", a["listener"])
    out["sent"] = "ok"
except OSError as exc:
    out.setdefault("pair", type(exc).__name__)
    out.setdefault("sent", type(exc).__name__)
print(json.dumps(out))
"""


@pytest.mark.skipif(
    not LINUX or kernel_layer._machine() not in kernel_layer._ARCH,
    reason="the seccomp filter is written for x86_64 and aarch64 Linux")
def test_no_datagram_reaches_a_unix_socket_outside_the_box(tmp_path):
    """The filter refused ``socket`` and let ``socketpair`` through; a
    datagram pair can send to any Unix socket by its path, and Landlock has
    no rule for that. With the pair refused, a listener that stands for
    some service of the host hears nothing."""
    conv = tmp_path / "conv"
    conv.mkdir()
    path = str(tmp_path / "service.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        listener.bind(path)
        listener.settimeout(0.5)
        env = dict(os.environ)
        env.pop(kernel_layer.ENV, None)
        done = subprocess.run(
            [sys.executable, "-B", "-c", PAIR,
             json.dumps({"layer": LAYER, "conv": str(conv),
                         "listener": path})],
            capture_output=True, text=True, timeout=120, env=env)
        got = json.loads(done.stdout.strip().splitlines()[-1])
        if "seccomp" not in got["up"]:
            pytest.skip("this host lets no seccomp filter be set: "
                        + str(got["not_up"].get("seccomp")))
        assert got["pair"] == "PermissionError", got
        with pytest.raises(socket.timeout):
            listener.recv(64)
    finally:
        listener.close()


# ---------------------------------------------------------------------------
# the paths: a library that is one file
# ---------------------------------------------------------------------------

def test_a_one_file_library_opens_its_own_folder_not_the_one_above(
        tmp_path, monkeypatch):
    """A package's folder sits in the folder it is installed in; a library
    that is ONE file sits there itself. Taken two folders up, an app that
    keeps its libraries beside its settings (``pip install --target``)
    would have had the app's whole folder made readable."""
    app = tmp_path / "app"
    libs = app / "libs"
    conv = tmp_path / "conversations" / "c1"
    for d in (libs / "madeuppkg", conv):
        d.mkdir(parents=True)
    (app / "settings.toml").write_text("made_up_key = 'made up'\n")
    (libs / "madeuplib.py").write_text("VALUE = 1\n")
    (libs / "madeuppkg" / "__init__.py").write_text("VALUE = 2\n")
    monkeypatch.syspath_prepend(str(libs))
    import importlib
    importlib.invalidate_caches()
    for name in ("madeuplib", "madeuppkg"):
        rows = kernel_layer.box_paths(str(conv), libraries=(name,))
        readable = [p for p, kind, _why in rows if kind == "read"]
        assert os.path.realpath(libs) in readable, (name, readable)
        assert os.path.realpath(app) not in readable, (name, readable)
