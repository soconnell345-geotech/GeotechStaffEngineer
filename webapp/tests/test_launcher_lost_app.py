"""BACKLOG TA-27: the Funhouse launcher with a Prompter key, and an app the
notebook has lost.

Offline only: a fake ``/proc`` tree in a tmp dir (link targets stored as the
files' text), a fake ``Popen``, a fake ``kill``. No real process is started,
signalled or read. Covers

* the Prompter-key route of ``run_on_databricks`` (the short cell, the older
  four-line cell, the key never printed or written to the script);
* ``running_apps`` / ``stop_running`` / ``diagnose`` against the fake tree,
  for x86_64 and aarch64 system-call numbers, and their Linux-only answer
  elsewhere;
* the port clash: an explicit port held by a running copy is refused, the
  default port moves on and names the older copy, ``replace=True`` stops it.
"""

import os
import shutil
import signal

import pytest

from webapp import databricks_launcher as dl

SECRET = "sk-test-NEVER-PRINT-1234567890"
PROMPTER_URL = "https://prompter.example/api/v1/chat/completions"
PROMPTER_IP = "10.1.2.3"
OTHER_IP = "10.9.9.9"
BTIME = 1_760_000_000
TERM = getattr(signal, "SIGTERM", 15)
KILL = getattr(signal, "SIGKILL", 9)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    dl._LAUNCHED.clear()
    for name in dl.PROMPTER_ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)
    for name in ("GEOTECH_PROMPTER_MODELS", "GEOTECH_WEBAPP_MODEL"):
        monkeypatch.delenv(name, raising=False)
    yield
    dl._LAUNCHED.clear()


# ---------------------------------------------------------------------------
# A fake /proc tree
# ---------------------------------------------------------------------------

def _addr(ip, port):
    return bytes(int(x) for x in ip.split("."))[::-1].hex().upper() + \
        f":{port:04X}"


def _tcp_line(i, local, remote, state, inode, tx=0, rx=0):
    return (f"   {i}: {_addr(*local)} {_addr(*remote)} {state} "
            f"{tx:08X}:{rx:08X} 00:00000000 00000000     0        0 "
            f"{inode} 1 0000000000000000 100 0 0 10 0")


X86 = {  # tid offset -> (comm, /proc/<pid>/task/<tid>/syscall)
    0: ("python3", "232 0x3 0x7f00 0x400 0xffffffff 0x0 0x0 0x7ffd 0x7f01"),
    1: ("python3", "0 0x7 0x7f10 0x4000 0x0 0x0 0x0 0x7ffd 0x7f02"),
    2: ("python3", "202 0x7f20 0x80 0x0 0x0 0x0 0x0 0x7ffd 0x7f03"),
    3: ("python3", "45 0x8 0x7f30 0x4000 0x0 0x0 0x0 0x7ffd 0x7f04"),
    4: ("python3", "running"),
}
ARM = {
    0: ("python3", "22 0x3 0x7f00 0x400 0xffffffff 0x0 0x0 0x7ffd 0x7f01"),
    1: ("python3", "63 0x7 0x7f10 0x4000 0x0 0x0 0x0 0x7ffd 0x7f02"),
    2: ("python3", "98 0x7f20 0x80 0x0 0x0 0x0 0x0 0x7ffd 0x7f03"),
}


class FakeProc:
    def __init__(self, tmp_path):
        self.root = tmp_path / "proc"
        self.tmp = tmp_path / "tmp"
        self.root.mkdir()
        self.tmp.mkdir()
        (self.root / "stat").write_text(f"cpu  1 2 3 4\nbtime {BTIME}\n")
        (self.root / "self").mkdir()             # a non-numeric entry

    def stat(self, pid, state="S", start_s=3600):
        # field 2 holds a space and a bracket, as a real comm can
        rest = ["S", "1", str(pid), str(pid), "0", "-1", "4194560"] + \
            ["0"] * 12 + [str(int(start_s * dl._clk_tck())), "0", "0"]
        rest[0] = state
        (self.root / str(pid) / "stat").write_text(
            f"{pid} (python3 (x)) " + " ".join(rest) + "\n")

    def add_process(self, pid, argv, state="S"):
        d = self.root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(
            b"\0".join(a.encode() for a in argv) + b"\0")
        self.stat(pid, state)
        return d

    def add_app(self, pid, port=8501, *, threads=X86, script=True,
                environ=None, python="/databricks/python3/bin/python3",
                log_lines=40, start_s=3600):
        script_path = self.tmp / f"geotech_streamlit_boot_{pid}.py"
        if script:
            script_path.write_text(f"PORT = {port}\nMODEL = 'm'\n")
        log = self.tmp / dl.LOG_NAME.format(port=port)
        body = [f"log line {i}" for i in range(1, log_lines + 1)]
        body[-3] = f"Authorization: Bearer {SECRET}"
        body[-2] = f"PROMPTER_API_KEY={SECRET}"
        log.write_text("\n".join(body) + "\n")

        d = self.root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(f"{python}\0{script_path}\0".encode())
        self.stat(pid, start_s=start_s)
        env = environ if environ is not None else {
            "PATH": "/usr/bin", "PROMPTER_URL": PROMPTER_URL,
            "PROMPTER_API_KEY": SECRET}
        (d / "environ").write_bytes(
            b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0")
        fd = d / "fd"
        fd.mkdir()
        targets = {0: "/dev/null", 1: str(log), 2: str(log),
                   5: "socket:[1001]", 7: "socket:[1002]",
                   8: "socket:[1003]", 9: "socket:[1004]"}
        for n, target in targets.items():
            (fd / str(n)).write_text(target)
        (d / "net").mkdir()
        (d / "net" / "tcp").write_text("\n".join([
            "  sl  local_address rem_address   st tx_queue rx_queue tr "
            "tm->when retrnsmt   uid  timeout inode",
            _tcp_line(0, ("0.0.0.0", port), ("0.0.0.0", 0), "0A", 1001),
            _tcp_line(1, ("10.0.0.2", 40000), (PROMPTER_IP, 443), "01", 1002),
            _tcp_line(2, ("10.0.0.2", 40001), (OTHER_IP, 443), "01", 1003,
                      rx=12),
            _tcp_line(3, ("10.0.0.2", port), ("10.0.0.50", 51000), "01",
                      1004),
            _tcp_line(4, ("10.0.0.2", 40002), (PROMPTER_IP, 443), "01",
                      7777),                       # another process's
        ]) + "\n")
        (d / "net" / "tcp6").write_text(
            "  sl  local_address remote_address st tx_queue rx_queue\n")
        task = d / "task"
        task.mkdir()
        for off, (comm, line) in threads.items():
            t = task / str(pid + off)
            t.mkdir()
            (t / "comm").write_text(comm + "\n")
            (t / "syscall").write_text(line + "\n")
        return script_path, log

    def apps(self):
        return dl.running_apps(proc_root=str(self.root), tmp_dir=str(self.tmp))


@pytest.fixture
def fake(tmp_path):
    return FakeProc(tmp_path)


# ---------------------------------------------------------------------------
# running_apps
# ---------------------------------------------------------------------------

def test_running_apps_finds_the_app_and_nothing_else(fake):
    script, log = fake.add_app(4242, 8501)
    fake.add_process(5000, ["/bin/bash", "-c", "sleep 100"])
    fake.add_process(5001, ["vim", str(script)])          # not python
    zombie = fake.tmp / "geotech_streamlit_boot_5002.py"
    fake.add_process(5002, ["/usr/bin/python3", str(zombie)], state="Z")

    apps = fake.apps()
    assert [a.pid for a in apps] == [4242]
    app = apps[0]
    assert app.port == 8501
    assert app.log_path == str(log)                  # its stdout (fd 1)
    assert app.script_path == str(script) and app.script_exists
    assert app.started == pytest.approx(BTIME + 3600)
    assert app.python == "/databricks/python3/bin/python3"
    assert app.mine is False and app.same_python is False

    text = repr(apps)
    assert "1 copy of the app" in text and "pid 4242" in text
    assert "port 8501" in text
    assert "not started by this notebook's Python" in text
    assert "not this notebook's Python" in text
    assert "stop_running(pid)" in text and "diagnose(pid)" in text
    assert SECRET not in text


def test_port_comes_from_the_listening_socket_when_the_script_is_gone(fake):
    fake.add_app(4243, 8600, script=False)
    (app,) = fake.apps()
    assert app.port == 8600 and app.script_exists is False
    assert "(already removed)" in repr(fake.apps())


def test_nothing_running_says_so(fake):
    apps = fake.apps()
    assert list(apps) == []
    assert "No copy of the app" in repr(apps)


def test_this_notebooks_own_app_is_marked(fake):
    fake.add_app(4242, 8501)

    class _Proc:
        pid = 4242

        def __init__(self, *a, **k):
            pass

        def terminate(self):
            pass

        def poll(self):
            return None

    handle = dl.run_on_databricks(port=8501, spark=_Spark(), quiet=True,
                                  _popen=lambda *a, **k: _Proc(),
                                  _running_apps=lambda: [])
    try:
        assert handle.pid == 4242 and dl._LAUNCHED[4242] is handle
        (app,) = fake.apps()
        assert app.mine is True
        assert "[this notebook's app]" in repr(fake.apps())
    finally:
        handle.stop()
    assert 4242 not in dl._LAUNCHED


def test_linux_only_elsewhere(monkeypatch, capsys):
    with monkeypatch.context() as m:        # undone before pytest's teardown
        m.setattr(dl.sys, "platform", "win32")
        apps = dl.running_apps()
        stopped = dl.stop_running()
        dl.diagnose()
    assert list(apps) == [] and "Linux only" in repr(apps)
    assert stopped == []
    out = capsys.readouterr().out
    assert out.count("Linux only") == 2


# ---------------------------------------------------------------------------
# stop_running
# ---------------------------------------------------------------------------

def _killer(fake, on_term="exit", on_kill="zombie"):
    calls = []

    def kill(pid, sig):
        calls.append((pid, sig))
        action = on_term if sig == TERM else on_kill
        if action == "exit":
            shutil.rmtree(fake.root / str(pid))
        elif action == "zombie":
            fake.stat(pid, state="Z")
        elif action == "denied":
            raise PermissionError(1, "Operation not permitted")
    return kill, calls


def test_stop_running_asks_politely_and_cleans_up(fake, capsys):
    script, log = fake.add_app(4242, 8501)
    kill, calls = _killer(fake)
    stopped = dl.stop_running(4242, proc_root=str(fake.root),
                              tmp_dir=str(fake.tmp), _kill=kill,
                              _sleep=lambda s: None)
    assert stopped == [4242]
    assert calls == [(4242, TERM)]
    assert not script.exists()                      # bootstrap removed
    assert log.exists()                             # the log is kept
    out = capsys.readouterr().out
    assert "Stopped pid 4242 (port 8501)" in out and str(log) in out


def test_stop_running_forces_a_hung_app(fake, capsys):
    fake.add_app(4242, 8501)
    kill, calls = _killer(fake, on_term="ignore", on_kill="zombie")
    sleeps = []
    stopped = dl.stop_running(4242, grace_s=1.0, proc_root=str(fake.root),
                              tmp_dir=str(fake.tmp), _kill=kill,
                              _sleep=sleeps.append)
    assert stopped == [4242]
    assert calls == [(4242, TERM), (4242, KILL)]
    assert sum(sleeps) >= 1.0                       # waited the grace period
    assert "forced it to stop" in capsys.readouterr().out


def test_stop_running_never_signals_a_process_that_is_not_the_app(fake,
                                                                  capsys):
    fake.add_app(4242, 8501)
    fake.add_process(5000, ["/bin/bash", "-c", "sleep 100"])
    kill, calls = _killer(fake)
    assert dl.stop_running(5000, proc_root=str(fake.root),
                           tmp_dir=str(fake.tmp), _kill=kill) == []
    assert calls == []
    assert "No copy of the app with pid 5000" in capsys.readouterr().out


def test_stop_running_with_no_pid_stops_every_copy(fake):
    fake.add_app(4242, 8501)
    fake.add_app(4300, 8502, start_s=7200)
    kill, calls = _killer(fake)
    stopped = dl.stop_running(proc_root=str(fake.root), tmp_dir=str(fake.tmp),
                              _kill=kill, _sleep=lambda s: None, quiet=True)
    assert sorted(stopped) == [4242, 4300]
    assert sorted(c[0] for c in calls) == [4242, 4300]
    assert list(fake.apps()) == []


def test_stop_running_reports_a_refusal(fake, capsys):
    fake.add_app(4242, 8501)
    kill, _ = _killer(fake, on_term="denied")
    assert dl.stop_running(4242, proc_root=str(fake.root),
                           tmp_dir=str(fake.tmp), _kill=kill,
                           _sleep=lambda s: None) == []
    assert "Not allowed to stop pid 4242" in capsys.readouterr().out


def test_stop_running_with_nothing_running(fake, capsys):
    assert dl.stop_running(proc_root=str(fake.root)) == []
    assert "nothing to stop" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------

SS = "\n".join([
    "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process",
    "ESTAB 0 0 10.0.0.2:40000 10.1.2.3:443 "
    "users:((\"python3\",pid=4242,fd=7)) timer:(keepalive,1min,0)",
    "ESTAB 0 0 10.0.0.2:40002 10.1.2.3:443 "
    "users:((\"python3\",pid=42420,fd=7))",
    "LISTEN 0 128 0.0.0.0:8501 0.0.0.0:* users:((\"python3\",pid=4242,fd=5))",
])


def _diag(fake, **kw):
    kw.setdefault("arch", "x86_64")
    kw.setdefault("_resolve", lambda host: {PROMPTER_IP})
    return dl._diagnosis_text(proc_root=str(fake.root), tmp_dir=str(fake.tmp),
                              **kw)


def test_diagnose_marks_the_thread_waiting_on_the_prompter(fake):
    fake.add_app(4242, 8501)
    text = _diag(fake, _run_ss=lambda: SS)
    assert "=== pid 4242, port 8501" in text
    assert "Prompter host: prompter.example -> 10.1.2.3" in text
    # connections: only this process's sockets, the Prompter one tagged
    assert "listening on 0.0.0.0:8501" in text
    assert "10.0.0.2:40000 -> 10.1.2.3:443 ESTABLISHED  [Prompter host]" in text
    assert "10.0.0.2:40002" not in text.split("ss -tnpo")[0]   # not ours
    assert "unread 12 B" in text
    # threads, named for x86_64
    assert "tid 4242 python3: epoll_wait" in text
    assert ("tid 4243 python3: read(fd 7 -> socket 10.0.0.2:40000 -> "
            "10.1.2.3:443 ESTABLISHED)  <-- WAITING FOR THE PROMPTER HOST "
            "TO ANSWER") in text
    assert "tid 4244 python3: futex  - waiting on another thread" in text
    assert "waiting for an answer from 10.9.9.9:443" in text
    assert "tid 4246 python3: running (on a CPU)" in text
    assert "FOUND: 1 thread(s) waiting for the Prompter host" in text
    assert "stop_running(4242)" in text
    # ss lines for this pid only
    assert "ss -tnpo (2 lines for this pid)" in text
    assert "pid=42420" not in text
    # the last 30 lines of the log, keys hidden
    assert "log line 11" in text and "log line 40" in text
    assert "log line 10\n" not in text
    assert "Authorization: Bearer <hidden>" in text
    assert "PROMPTER_API_KEY=<hidden>" in text
    assert SECRET not in text


def test_diagnose_names_aarch64_system_calls(fake):
    fake.add_app(4242, 8501, threads=ARM)
    text = _diag(fake, arch="aarch64")
    assert "(aarch64)" in text
    assert "tid 4242 python3: epoll_pwait" in text
    assert "tid 4243 python3: read(fd 7 -> socket" in text
    assert "WAITING FOR THE PROMPTER HOST TO ANSWER" in text
    assert "tid 4244 python3: futex" in text
    assert SECRET not in text


def test_diagnose_without_a_prompter_setting(fake):
    fake.add_app(4242, 8501, environ={"PATH": "/usr/bin",
                                      "PROMPTER_API_KEY": SECRET})
    text = _diag(fake)
    assert "Prompter host: none set" in text
    assert "WAITING FOR THE PROMPTER HOST" not in text
    assert "waiting for an answer from 10.1.2.3:443" in text
    assert SECRET not in text


def test_diagnose_prints_and_handles_nothing_running(fake, capsys):
    dl.diagnose(proc_root=str(fake.root), tmp_dir=str(fake.tmp))
    assert "nothing to diagnose" in capsys.readouterr().out
    fake.add_app(4242, 8501)
    dl.diagnose(4242, proc_root=str(fake.root), tmp_dir=str(fake.tmp),
                arch="x86_64", _resolve=lambda h: {PROMPTER_IP})
    out = capsys.readouterr().out
    assert "FOUND: 1 thread(s)" in out and SECRET not in out
    dl.diagnose(31337, proc_root=str(fake.root))
    assert "No process 31337" in capsys.readouterr().out


def test_redact_hides_keys_and_passwords():
    for line in (f"api-key: {SECRET}", f"'api_key': '{SECRET}'",
                 f"Authorization: Bearer {SECRET}", f"password={SECRET}",
                 f"token was {SECRET} here"):
        assert SECRET not in dl._redact(line), line
    assert dl._redact("plain line 42") == "plain line 42"


# ---------------------------------------------------------------------------
# The Prompter-key route of run_on_databricks
# ---------------------------------------------------------------------------

class _Spark:
    class conf:
        @staticmethod
        def get(key):
            return {dl._ORG_ID_KEY: "1", dl._CLUSTER_ID_KEY: "c",
                    "spark.databricks.workspaceUrl": "h.example"}[key]


class _Proc:
    def __init__(self, argv, env=None, **kwargs):
        self.argv, self.env = argv, env

    def terminate(self):
        pass

    def poll(self):
        return None


class _FhPrompter:
    username = "svc\\account"
    password = "ntlm-password"
    base_url = "https://funhouse.example/api/v1"
    verify = True


def _launch(**kw):
    kw.setdefault("spark", _Spark())
    kw.setdefault("quiet", True)
    kw.setdefault("_popen", _Proc)
    kw.setdefault("_running_apps", lambda: [])
    return dl.run_on_databricks(**kw)


def _script(handle):
    with open(handle.script_path, encoding="utf-8") as fh:
        return fh.read()


def test_the_short_key_cell(capsys):
    handle = _launch(prompter_key=dict(url=PROMPTER_URL, model="dep-model-1",
                                       api_key=SECRET),
                     prompter=_FhPrompter(), quiet=False)
    try:
        env = handle.process.env
        assert env["PROMPTER_URL"] == PROMPTER_URL
        assert env["PROMPTER_MODEL"] == "dep-model-1"
        assert env["PROMPTER_API_KEY"] == SECRET
        assert "PROMPTER_CA_BUNDLE" not in env
        # the one model in the picker, and as the default
        assert env["GEOTECH_PROMPTER_MODELS"] == "dep-model-1"
        assert env["GEOTECH_WEBAPP_MODEL"] == "dep-model-1"
        # the NTLM credentials are not needed on this route, so not handed on
        assert not [k for k in env if k.startswith("GEOTECH_FH_")]
        assert handle.model == "dep-model-1"
        assert handle.route == dl.ROUTE_KEY
        src = _script(handle)
        assert "tinyapps_entry.py" in src and "KEY_ROUTE = True" in src
        assert "MODEL = 'dep-model-1'" in src
        compile(src, handle.script_path, "exec")
        assert SECRET not in src
        out = capsys.readouterr().out
        assert "model=dep-model-1; engine: your Prompter API key" in out
        assert "funhouse-gpt-high" not in out
        assert "running_apps()" in out
        assert SECRET not in out and SECRET not in repr(handle)
        # the notebook's own environment is left alone
        assert "PROMPTER_API_KEY" not in os.environ
    finally:
        handle.stop()


def test_key_cell_options(monkeypatch):
    handle = _launch(prompter_key={"url": PROMPTER_URL, "model": "m2",
                                   "key": SECRET,
                                   "ca_bundle": "-----BEGIN CERTIFICATE-----"},
                     models=["Fast=m2", "m3"], app_path=None)
    try:
        env = handle.process.env
        assert env["PROMPTER_API_KEY"] == SECRET
        assert env["PROMPTER_CA_BUNDLE"].startswith("-----BEGIN")
        assert env["GEOTECH_PROMPTER_MODELS"] == "Fast=m2,m3"
    finally:
        handle.stop()
    # True (and a dict's missing pieces) read the environment
    monkeypatch.setenv("PROMPTER_URL", PROMPTER_URL)
    monkeypatch.setenv("PROMPTER_MODEL", "env-model")
    monkeypatch.setenv("PROMPTER_API_KEY", SECRET)
    for given in (True, {"model": "dict-model"}):
        handle = _launch(prompter_key=given)
        try:
            want = "env-model" if given is True else "dict-model"
            assert handle.model == want
            assert handle.process.env["PROMPTER_MODEL"] == want
        finally:
            handle.stop()


@pytest.mark.parametrize("given,error,words", [
    ({"url": PROMPTER_URL, "model": "m"}, ValueError, "api_key"),
    ({"url": PROMPTER_URL, "api_key": SECRET}, ValueError, "model"),
    ({"model": "m", "api_key": SECRET, "url": "prompter.example/x"},
     ValueError, "full address"),
    ({"url": PROMPTER_URL, "model": "m", "api_key": SECRET, "modle": "x"},
     TypeError, "modle"),
    ("not a dict", TypeError, "prompter_key takes"),
])
def test_a_bad_key_is_refused_before_anything_starts(given, error, words):
    def _never(*a, **k):
        raise AssertionError("nothing may start")
    with pytest.raises(error) as info:
        _launch(prompter_key=given, _popen=_never,
                _running_apps=_never)
    assert words in str(info.value)
    assert SECRET not in str(info.value)


def test_prompter_key_repr_hides_the_key():
    key = dl.PrompterKey(url=PROMPTER_URL, model="m", api_key=SECRET,
                         ca_bundle="-----BEGIN CERTIFICATE-----")
    assert SECRET not in repr(key) and "BEGIN" not in repr(key)


def test_the_older_four_line_cell_names_the_right_model(monkeypatch, capsys):
    monkeypatch.setenv("PROMPTER_URL", PROMPTER_URL)
    monkeypatch.setenv("PROMPTER_MODEL", "env-model")
    monkeypatch.setenv("PROMPTER_API_KEY", SECRET)
    app_path = os.path.join(os.path.dirname(dl.__file__), "tinyapps_entry.py")
    handle = _launch(prompter=_FhPrompter(), app_path=app_path,
                     models=["env-model"], quiet=False)
    try:
        assert handle.model == "env-model" and handle.route == dl.ROUTE_KEY
        assert "KEY_ROUTE = True" in _script(handle)
        assert not [k for k in handle.process.env
                    if k.startswith("GEOTECH_FH_")]
        out = capsys.readouterr().out
        assert "model=env-model" in out and "funhouse-gpt-high" not in out
        assert SECRET not in out
    finally:
        handle.stop()
    # without models= the picker still gets the key's one model
    handle = _launch(app_path=app_path)
    try:
        assert handle.process.env["GEOTECH_PROMPTER_MODELS"] == "env-model"
    finally:
        handle.stop()


def test_the_funhouse_route_is_unchanged(monkeypatch):
    # PROMPTER_* in the notebook do not switch the default two-page app
    monkeypatch.setenv("PROMPTER_URL", PROMPTER_URL)
    monkeypatch.setenv("PROMPTER_MODEL", "env-model")
    monkeypatch.setenv("PROMPTER_API_KEY", SECRET)
    handle = _launch(prompter=_FhPrompter())
    try:
        assert handle.route == dl.ROUTE_SDK
        assert handle.model == dl.DEFAULT_MODEL
        src = _script(handle)
        assert "KEY_ROUTE = False" in src and "pages_entry.py" in src
        assert handle.process.env["GEOTECH_FH_USERNAME"] == "svc\\account"
        assert handle.process.env["GEOTECH_PROMPTER_MODELS"] == \
            "funhouse-gpt-high,funhouse-gpt-medium"
    finally:
        handle.stop()


# ---------------------------------------------------------------------------
# The port clash
# ---------------------------------------------------------------------------

def _lost(pid=4242, port=8501):
    return dl.RunningApp(pid=pid, port=port, started=BTIME, script_path=(
        "/tmp/geotech_streamlit_boot_x.py"), script_exists=True,
        log_path=f"/tmp/geotech_webapp_{port}.log",
        python="/databricks/python3/bin/python3")


def test_an_explicit_port_held_by_a_lost_copy_is_refused():
    def _never(*a, **k):
        raise AssertionError("nothing may start")
    with pytest.raises(dl.PortInUse) as info:
        _launch(port=8501, _popen=_never, _running_apps=lambda: [_lost()])
    msg = str(info.value)
    assert "Port 8501 is already held" in msg and "pid 4242" in msg
    assert "stop_running(4242)" in msg and "replace=True" in msg
    assert "leave out port=" in msg


def test_the_default_port_moves_on_and_names_the_older_copy(monkeypatch,
                                                            capsys):
    monkeypatch.setattr(dl, "find_available_port", lambda *a, **k: 8502)
    handle = _launch(_running_apps=lambda: [_lost()], quiet=False)
    try:
        assert handle.port == 8502 and handle.base_path.endswith("/8502")
        out = capsys.readouterr().out
        assert "pid 4242 on port 8501" in out
        assert "stop_running(4242)" in out and "replace=True" in out
    finally:
        handle.stop()


def test_another_explicit_port_starts_and_mentions_the_copy(capsys):
    handle = _launch(port=8502, _running_apps=lambda: [_lost()], quiet=False)
    try:
        assert handle.port == 8502
        assert "pid 4242 on port 8501" in capsys.readouterr().out
    finally:
        handle.stop()


def test_replace_stops_the_copy_then_starts(capsys):
    running = [_lost()]
    stopped = []

    def _stop(pid=None, **kw):
        stopped.append(pid)
        running.clear()

    handle = _launch(port=8501, replace=True, quiet=False,
                     _running_apps=lambda: list(running), _stop_running=_stop)
    try:
        assert stopped == [4242] and handle.port == 8501
        assert "replace=True: stopping the copy on port 8501 (pid 4242)" in \
            capsys.readouterr().out
    finally:
        handle.stop()


def test_replace_that_cannot_stop_the_copy_still_refuses():
    with pytest.raises(dl.PortInUse):
        _launch(port=8501, replace=True, _running_apps=lambda: [_lost()],
                _stop_running=lambda **kw: None)
