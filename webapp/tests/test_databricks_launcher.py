"""Offline tests for webapp.databricks_launcher — no Databricks, no streamlit run.

Covers the pure parts (base-path construction, cluster-id resolution, workspace
host, URL join, bootstrap-script generation + validity, env passthrough) and the
``run_on_databricks`` orchestration with an injected fake ``Popen`` + fake spark.
"""

import os
import sys

import pytest

from webapp import databricks_launcher as dl


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeConf:
    def __init__(self, d):
        self._d = d

    def get(self, key):
        if key in self._d:
            return self._d[key]
        raise KeyError(key)          # a real spark.conf.get raises on a missing key


class _FakeSpark:
    def __init__(self, d):
        self.conf = _FakeConf(d)


class _FakeProc:
    def __init__(self, argv, env=None, **kwargs):
        self.argv = argv
        self.env = env
        self.popen_kwargs = kwargs
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def poll(self):
        return None


def _fake_popen(argv, env=None, **kwargs):
    return _FakeProc(argv, env, **kwargs)


_CLUSTER_CONF = {
    dl._ORG_ID_KEY: "1234567890",
    dl._CLUSTER_ID_KEY: "0710-abc-cluster",
    "spark.databricks.workspaceUrl": "dbc-deadbeef.cloud.databricks.com",
}


# ---------------------------------------------------------------------------
# driver_proxy_base_path
# ---------------------------------------------------------------------------

def test_driver_proxy_base_path_format():
    assert (dl.driver_proxy_base_path("ORG", "CL", 8501)
            == "/driver-proxy/o/ORG/CL/8501")


def test_driver_proxy_base_path_strips_and_coerces_port():
    assert (dl.driver_proxy_base_path("  ORG ", " CL ", "8502")
            == "/driver-proxy/o/ORG/CL/8502")


@pytest.mark.parametrize("org,cluster", [("", "CL"), ("ORG", ""), ("  ", "CL")])
def test_driver_proxy_base_path_rejects_empty(org, cluster):
    with pytest.raises(ValueError):
        dl.driver_proxy_base_path(org, cluster, 8501)


# ---------------------------------------------------------------------------
# resolve_cluster_ids
# ---------------------------------------------------------------------------

def test_resolve_cluster_ids_explicit_overrides_skip_spark():
    # No spark provided/needed when both ids are explicit.
    assert dl.resolve_cluster_ids(org_id="O", cluster_id="C") == ("O", "C")


def test_resolve_cluster_ids_from_spark():
    spark = _FakeSpark(_CLUSTER_CONF)
    assert dl.resolve_cluster_ids(spark=spark) == ("1234567890", "0710-abc-cluster")


def test_resolve_cluster_ids_partial_override_reads_rest_from_spark():
    spark = _FakeSpark(_CLUSTER_CONF)
    assert dl.resolve_cluster_ids(spark=spark, org_id="OVERRIDE") == (
        "OVERRIDE", "0710-abc-cluster")


def test_resolve_cluster_ids_missing_raises(monkeypatch):
    monkeypatch.setattr(dl, "_active_spark", lambda: None)
    with pytest.raises(ValueError):
        dl.resolve_cluster_ids()


# ---------------------------------------------------------------------------
# workspace_host_from_spark / proxy_url
# ---------------------------------------------------------------------------

def test_workspace_host_from_workspace_url_adds_scheme():
    spark = _FakeSpark({"spark.databricks.workspaceUrl": "dbc-x.cloud.databricks.com"})
    assert dl.workspace_host_from_spark(spark) == "https://dbc-x.cloud.databricks.com"


def test_workspace_host_falls_back_to_browser_host_tag():
    spark = _FakeSpark(
        {"spark.databricks.clusterUsageTags.browserHostName": "adb-99.11.azuredatabricks.net"})
    assert dl.workspace_host_from_spark(spark) == (
        "https://adb-99.11.azuredatabricks.net")


def test_workspace_host_none_when_absent():
    assert dl.workspace_host_from_spark(_FakeSpark({})) is None
    assert dl.workspace_host_from_spark(None) is None


def test_proxy_url_join_and_trailing_slash():
    url = dl.proxy_url("https://dbc-x.cloud.databricks.com",
                       "/driver-proxy/o/O/C/8501")
    assert url == "https://dbc-x.cloud.databricks.com/driver-proxy/o/O/C/8501/"


def test_proxy_url_none_host_returns_none():
    assert dl.proxy_url(None, "/driver-proxy/o/O/C/8501") is None
    assert dl.proxy_url("", "/driver-proxy/o/O/C/8501") is None


# ---------------------------------------------------------------------------
# render_bootstrap_script
# ---------------------------------------------------------------------------

def test_render_bootstrap_script_is_valid_python():
    src = dl.render_bootstrap_script(
        app_path=r"C:\repo\webapp\app.py", repo_root=r"C:\repo",
        base="/driver-proxy/o/O/C/8501", port=8501, model="funhouse-gpt-high")
    # Compiles cleanly (catches template/quoting regressions).
    compile(src, "<bootstrap>", "exec")


def test_render_bootstrap_script_registers_prompter_with_fallback():
    src = dl.render_bootstrap_script(
        app_path="/repo/webapp/app.py", repo_root="/repo",
        base="/driver-proxy/o/O/C/8080", port=8080, model="funhouse-gpt-high")
    # The reconstruct-on-driver branch we ship.
    assert "PrompterAPI(chat_model=MODEL)" in src
    assert "register_model_builder(" in src
    # Model-aware builder (in-app Prompter picker; None -> launch default).
    assert "PrompterChatModel(prompter=prompter, model=model_id or MODEL)" in src
    # The automatic ANTHROPIC_API_KEY fallback.
    assert "ANTHROPIC_API_KEY" in src
    # In-process streamlit start with the proxy flags. load_config_options
    # MUST precede run(): run() only installs config watchers — it never
    # applies flag_options (the 5.11.1 60s-socket-death root cause).
    assert "bootstrap.load_config_options(flag_options=_FLAGS)" in src
    assert src.index("bootstrap.load_config_options") < \
        src.index("bootstrap.run(APP_PATH")
    assert "bootstrap.run(APP_PATH, False, [], _FLAGS)" in src
    assert '"server_baseUrlPath": BASE_PATH' in src
    # Production posture: no file watcher (no rerun prompt) + viewer toolbar.
    assert '"server_fileWatcherType": "none"' in src
    assert '"client_toolbarMode": "viewer"' in src
    # Driver-proxy websocket fix: the proxy swallows ping/pong control
    # frames, so the default 30s ping + 30s timeout made the SERVER hang up
    # every socket ~60s after open ("Connecting" flaps). Ping horizon = 1 hr.
    assert '"server_websocketPingInterval": 3600' in src
    # Injected values present as literals.
    assert "'/repo'" in src and "8080" in src and "'funhouse-gpt-high'" in src


# ---------------------------------------------------------------------------
# build_launch_env
# ---------------------------------------------------------------------------

def test_build_launch_env_injects_key_and_pythonpath():
    env = dl.build_launch_env(
        {"PATH": "/usr/bin"}, anthropic_key="sk-test", repo_root="/repo")
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == "/repo"
    assert env["PATH"] == "/usr/bin"                       # base env preserved


def test_build_launch_env_no_key_leaves_env_clean():
    env = dl.build_launch_env({"X": "1"})
    assert "ANTHROPIC_API_KEY" not in env
    assert env["X"] == "1"


def test_build_launch_env_does_not_duplicate_repo_root():
    env = dl.build_launch_env({"PYTHONPATH": "/repo"}, repo_root="/repo")
    assert env["PYTHONPATH"].split(os.pathsep).count("/repo") == 1


class _FakePrompter:
    """Live-fh_prompter stand-in with the NTLM credential attributes."""

    def __init__(self, verify=False):
        self.username = "appservices.state.sbu\\svcAccount"
        self.password = "s3cret"
        self.base_url = "https://dsaaiapi.example/api/v1"
        self.verify = verify
        self.chat_model = "funhouse-gpt-high"


def test_build_launch_env_threads_prompter_credentials():
    env = dl.build_launch_env({}, prompter=_FakePrompter(verify=False))
    assert env["GEOTECH_FH_USERNAME"] == "appservices.state.sbu\\svcAccount"
    assert env["GEOTECH_FH_PASSWORD"] == "s3cret"
    assert env["GEOTECH_FH_BASE_URL"] == "https://dsaaiapi.example/api/v1"
    assert env["GEOTECH_FH_VERIFY"] == "0"
    assert dl.build_launch_env(
        {}, prompter=_FakePrompter(verify=True))["GEOTECH_FH_VERIFY"] == "1"


def test_build_launch_env_incomplete_prompter_sets_nothing():
    class _NoCreds:
        username = password = base_url = None
    env = dl.build_launch_env({}, prompter=_NoCreds())
    assert not [k for k in env if k.startswith("GEOTECH_FH_")]


def test_stage_sharepoint_writes_token_and_envs(tmp_path):
    tok_path = str(tmp_path / "tok.txt")
    env = {}
    calls = []

    def getter():
        calls.append(1)
        return f"token-{len(calls)}"

    out = dl.stage_sharepoint(
        "https://usdos.sharepoint.com/sites/CSEGeotechGroup/",
        root="/Shared Documents/General/GSE_app/",
        token_getter=getter, token_path=tok_path,
        start_refresher=False, env=env)
    assert open(tok_path).read() == "token-1"
    assert env["GEOTECH_SHAREPOINT_SITE_URL"] == \
        "https://usdos.sharepoint.com/sites/CSEGeotechGroup"
    assert env["GEOTECH_SHAREPOINT_TOKEN_FILE"] == tok_path
    assert env["GEOTECH_SHAREPOINT_ROOT"] == "Shared Documents/General/GSE_app"
    assert out["refresher"] is False


def test_stage_sharepoint_no_token_raises(tmp_path):
    with pytest.raises(RuntimeError):
        dl.stage_sharepoint("https://x.sharepoint.com/sites/y",
                            token_getter=lambda: None,
                            token_path=str(tmp_path / "t.txt"),
                            start_refresher=False, env={})


def test_bootstrap_script_reconstructs_from_env_creds():
    src = dl.render_bootstrap_script(
        app_path="/x/app.py", repo_root="/repo", base="/b", port=8080,
        model="funhouse-gpt-high")
    # Preferred path: explicit-credential PrompterAPI from GEOTECH_FH_* envs,
    # with the bare self-config retained as fallback.
    assert 'GEOTECH_FH_USERNAME' in src and 'GEOTECH_FH_PASSWORD' in src
    assert 'backend="prompter"' in src
    assert "PrompterAPI(chat_model=MODEL)" in src


# ---------------------------------------------------------------------------
# run_on_databricks (orchestration, fake Popen + fake spark)
# ---------------------------------------------------------------------------

def test_run_on_databricks_end_to_end_offline(tmp_path):
    spark = _FakeSpark(_CLUSTER_CONF)
    handle = dl.run_on_databricks(
        port=8501, model="funhouse-gpt-high", spark=spark,
        anthropic_key="sk-fallback", quiet=True, _popen=_fake_popen)
    try:
        # Base path + URL derived from spark conf.
        assert handle.base_path == "/driver-proxy/o/1234567890/0710-abc-cluster/8501"
        assert handle.url == (
            "https://dbc-deadbeef.cloud.databricks.com"
            "/driver-proxy/o/1234567890/0710-abc-cluster/8501/")
        assert handle.model == "funhouse-gpt-high"
        # The subprocess was launched as [python, script_path].
        assert handle.process.argv == [sys.executable, handle.script_path]
        # The temp bootstrap script exists and is valid Python.
        assert os.path.isfile(handle.script_path)
        with open(handle.script_path, encoding="utf-8") as fh:
            src = fh.read()
        compile(src, handle.script_path, "exec")
        assert "register_model_builder(" in src
        # The fallback key + repo root reached the subprocess env.
        assert handle.process.env["ANTHROPIC_API_KEY"] == "sk-fallback"
        expected_root = os.path.dirname(os.path.dirname(dl._default_app_path()))
        assert handle.process.env["PYTHONPATH"].split(os.pathsep)[0] == expected_root
    finally:
        handle.stop()
    # stop() terminated the process and removed the temp script.
    assert handle.process.terminated is True
    assert not os.path.isfile(handle.script_path)


def test_run_on_databricks_unknown_host_leaves_url_none():
    spark = _FakeSpark({dl._ORG_ID_KEY: "O", dl._CLUSTER_ID_KEY: "C"})  # no host
    handle = dl.run_on_databricks(
        spark=spark, quiet=True, _popen=_fake_popen)
    try:
        assert handle.base_path == "/driver-proxy/o/O/C/8501"
        assert handle.url is None                          # host unknown → None
    finally:
        handle.stop()


# ============================================================================
# SDK-example alignment (2026-09): adb-dp host, free-port scan, auto-shutdown
# ============================================================================

def test_proxy_url_gov_cloud_uses_dp_host():
    """Azure Gov driver proxy lives on the adb-dp- host (SDK examples force
    this rewrite); other clouds untouched."""
    url = dl.proxy_url("https://adb-429679569790375.3.databricks.azure.us",
                       "/driver-proxy/o/O/C/8501")
    assert url.startswith("https://adb-dp-429679569790375.3.databricks.azure.us/")
    # Already-dp stays; commercial Azure and AWS untouched.
    assert dl.proxy_url("https://adb-dp-9.3.databricks.azure.us", "/p"
                        ).count("adb-dp-") == 1
    assert dl.proxy_url("adb-99.11.azuredatabricks.net", "/p"
                        ).startswith("https://adb-99.11.azuredatabricks.net")
    assert dl.proxy_url("dbc-x.cloud.databricks.com", "/p"
                        ).startswith("https://dbc-x.cloud.databricks.com")


def test_find_available_port_returns_bindable():
    import socket
    port = dl.find_available_port(8501, 8899)
    assert 8501 <= port <= 8899
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", port))               # genuinely free


def test_run_on_databricks_default_port_auto_picks(monkeypatch):
    monkeypatch.setattr(dl, "find_available_port", lambda *a, **k: 8517)
    spark = _FakeSpark(_CLUSTER_CONF)
    handle = dl.run_on_databricks(spark=spark, quiet=True, _popen=_fake_popen)
    try:
        assert handle.port == 8517
        assert handle.base_path.endswith("/8517")
    finally:
        handle.stop()


def test_bootstrap_auto_shutdown_rendering():
    on = dl.render_bootstrap_script(
        app_path="/x/app.py", repo_root="/r", base="/b", port=8501,
        model="m", auto_shutdown_min=120)
    assert "AUTO_SHUTDOWN_MIN = 120" in on and "_auto_shutdown" in on
    compile(on, "<boot>", "exec")
    off = dl.render_bootstrap_script(
        app_path="/x/app.py", repo_root="/r", base="/b", port=8501, model="m")
    assert "AUTO_SHUTDOWN_MIN = None" in off
    compile(off, "<boot>", "exec")
