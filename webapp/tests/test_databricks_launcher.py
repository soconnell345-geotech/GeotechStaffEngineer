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
    def __init__(self, argv, env=None):
        self.argv = argv
        self.env = env
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def poll(self):
        return None


def _fake_popen(argv, env=None):
    return _FakeProc(argv, env)


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
    assert "PrompterChatModel(prompter=prompter, model=MODEL)" in src
    # The automatic ANTHROPIC_API_KEY fallback.
    assert "ANTHROPIC_API_KEY" in src
    # In-process streamlit start with the proxy flags.
    assert "bootstrap.run(APP_PATH, False, [], _FLAGS)" in src
    assert '"server_baseUrlPath": BASE_PATH' in src
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
