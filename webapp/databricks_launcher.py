"""Launch the Streamlit web app on a Databricks cluster with the Funhouse
Prompter engine — the TinyApp dress rehearsal, no Anthropic key required.

Why this exists
---------------
The README §3 "driver-proxy" recipe launches streamlit with a plain
``subprocess.Popen([... "streamlit", "run", ...])``. That subprocess is a fresh
Python process: it never runs the notebook's
``register_model_builder(lambda: PrompterChatModel(fh_prompter))`` call, and the
notebook's *live* ``fh_prompter`` object cannot cross the process boundary
anyway. So the app boots into the "no engine configured" banner unless an
``ANTHROPIC_API_KEY`` is set.

The fix (the branch we ship)
----------------------------
:func:`run_on_databricks` writes a tiny **bootstrap script** to a temp file and
launches it as the streamlit process. The bootstrap, running IN the streamlit
process (so its ``register_model_builder`` is visible when the app calls
``resolve_engine``):

1. **RECONSTRUCTS a ``PrompterAPI`` on the driver** and registers a
   ``PrompterChatModel`` built on it, then
2. starts streamlit **in-process** via ``streamlit.web.bootstrap.run`` with the
   driver-proxy server flags (``baseUrlPath`` etc.).

Reconstruction is the real path because ``PrompterAPI`` **self-configures**: its
``__init__`` loads a ``FunhouseConfig`` singleton from the driver's config file /
environment (the SDK's own code and examples construct ``PrompterAPI()`` bare —
e.g. ``funhouse/utils/ai_text_utils.py``). The credentials live in that on-disk
config, not only in the notebook kernel's memory, so a driver subprocess can
rebuild an equivalent client. If reconstruction fails (no Funhouse config on the
driver), the bootstrap **falls back automatically to the ``ANTHROPIC_API_KEY``
path** — the launcher threads that key (secrets → env → subprocess env) through
robustly.

Everything except the actual ``Popen`` / spark reads is a pure function, unit
tested offline. The live driver run is owner-verified (NEEDS-LIVE-VERIFICATION,
like the README recipe).

Owner usage (one notebook cell)::

    from webapp.databricks_launcher import run_on_databricks
    handle = run_on_databricks(port=8501, model="funhouse-gpt-high",
                               prompter=fh_prompter)   # pass the live object!
    print(handle.url)      # open this in your browser
    # ... later ...
    handle.stop()

``prompter=fh_prompter`` is the reliable engine path (live-verified findings
2026-07-24): Prompter auth is **NTLM with a domain service account** — the
username/password/base_url are plain strings on the live object, so the
launcher threads them to the app process, which reconstructs a fully working
client with no dbutils/Py4J. Without ``prompter=``, the bootstrap's bare
``PrompterAPI()`` self-config is attempted; on workspaces where FunhouseConfig
needs the notebook's live session that DIES with Py4J "Object ID unknown".

If the workspace host cannot be read from spark, pass ``workspace_host=...`` or
build the URL yourself from the printed ``base_path``.
"""

from __future__ import annotations

import os
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "run_on_databricks",
    "stage_sharepoint",
    "LaunchHandle",
    "driver_proxy_base_path",
    "resolve_cluster_ids",
    "workspace_host_from_spark",
    "proxy_url",
    "render_bootstrap_script",
    "build_launch_env",
]

#: Spark conf keys for the driver-proxy identifiers.
_ORG_ID_KEY = "spark.databricks.clusterUsageTags.clusterOwnerOrgId"
_CLUSTER_ID_KEY = "spark.databricks.clusterUsageTags.clusterId"
#: Spark conf keys we try (in order) for the workspace host.
_WORKSPACE_HOST_KEYS = (
    "spark.databricks.workspaceUrl",
    "spark.databricks.clusterUsageTags.browserHostName",
)

#: Default Prompter model id for the reasoning tier the owner runs.
DEFAULT_MODEL = "funhouse-gpt-high"
DEFAULT_PORT = 8501


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------

def driver_proxy_base_path(org_id: str, cluster_id: str, port: int) -> str:
    """Return the Databricks driver-proxy base path for ``port``.

    ``/driver-proxy/o/{org_id}/{cluster_id}/{port}`` — the path streamlit must be
    told to serve under (``server.baseUrlPath``) and the path appended to the
    workspace host to open the app.
    """
    org = str(org_id).strip()
    cluster = str(cluster_id).strip()
    if not org or not cluster:
        raise ValueError("org_id and cluster_id must both be non-empty")
    return f"/driver-proxy/o/{org}/{cluster}/{int(port)}"


def _active_spark():
    """Best-effort fetch of the active SparkSession, or ``None`` off-Databricks."""
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return None
    try:
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _spark_conf_get(spark, key: str) -> Optional[str]:
    """``spark.conf.get(key)`` that returns ``None`` instead of raising."""
    try:
        val = spark.conf.get(key)
    except Exception:
        return None
    return val or None


def resolve_cluster_ids(
    spark: Any = None,
    org_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve ``(org_id, cluster_id)`` from explicit overrides then spark conf.

    Explicit ``org_id`` / ``cluster_id`` win; any not supplied are read from the
    (active or passed) SparkSession. Raises ``ValueError`` with an actionable
    message if a value is still missing.
    """
    if not (org_id and cluster_id):
        if spark is None:
            spark = _active_spark()
        if spark is not None:
            org_id = org_id or _spark_conf_get(spark, _ORG_ID_KEY)
            cluster_id = cluster_id or _spark_conf_get(spark, _CLUSTER_ID_KEY)
    if not org_id or not cluster_id:
        raise ValueError(
            "Could not determine the Databricks org id / cluster id. Run inside a "
            "Databricks notebook (with an active Spark session), or pass "
            "spark=spark, or pass org_id=... and cluster_id=... explicitly.")
    return str(org_id).strip(), str(cluster_id).strip()


def workspace_host_from_spark(spark: Any) -> Optional[str]:
    """Return ``https://<host>`` for the workspace from spark conf, or ``None``.

    Tries ``spark.databricks.workspaceUrl`` then the ``browserHostName`` cluster
    tag. The scheme is added if the conf value omits it.
    """
    if spark is None:
        return None
    for key in _WORKSPACE_HOST_KEYS:
        host = _spark_conf_get(spark, key)
        if host:
            host = host.strip().rstrip("/")
            return host if host.startswith("http") else f"https://{host}"
    return None


def proxy_url(workspace_host: Optional[str], base_path: str) -> Optional[str]:
    """Join a workspace host and driver-proxy base path into the openable URL.

    Returns ``None`` when the host is unknown (the caller then prints the
    ``base_path`` and instructs the owner to prepend their workspace host).

    On Azure **Government** workspaces (``*.databricks.azure.us``) the driver
    proxy is served from a dedicated host with an ``adb-dp-`` prefix, NOT the
    workspace host — the Funhouse SDK's own web-app examples force this
    rewrite ("Driver proxy requires 'adb-dp-' prefix"). Applied here so the
    printed URL is the sanctioned direct one instead of relying on a
    redirect. Other clouds are left untouched.
    """
    host = (workspace_host or "").strip().rstrip("/")
    if not host:
        return None
    if not host.startswith("http"):
        host = f"https://{host}"
    if ".databricks.azure.us" in host and "https://adb-dp-" not in host:
        host = host.replace("https://adb-", "https://adb-dp-")
    return f"{host}{base_path}/"


def find_available_port(start_port: int = 8501, end_port: int = 8899) -> int:
    """First locally-bindable port in ``[start_port, end_port]`` (SDK-example
    pattern) — sidesteps collisions with orphaned app processes on a shared
    driver. Raises ``RuntimeError`` when the whole range is taken."""
    import socket
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No available ports between {start_port} and {end_port}")


# The bootstrap script run AS the streamlit process. ``$name`` placeholders are
# substituted with repr()'d Python literals (safe against path/quote escaping).
# No literal ``$`` appears elsewhere in the body, so string.Template is clean.
_BOOTSTRAP_TEMPLATE = string.Template(
    '''# AUTO-GENERATED by webapp.databricks_launcher — do not edit by hand.
"""In-process Streamlit launcher: register the Funhouse Prompter engine, then
run the app in THIS process so the registration is visible to resolve_engine."""
import os
import sys

REPO_ROOT = $repo_root
APP_PATH = $app_path
BASE_PATH = $base
PORT = $port
MODEL = $model
AUTO_SHUTDOWN_MIN = $auto_shutdown_min

if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if AUTO_SHUTDOWN_MIN:
    # Shared-cluster etiquette (mirrors the Funhouse SDK examples' 10-minute
    # auto-kill): terminate this app process after the configured runtime.
    import threading as _threading
    import time as _time

    def _auto_shutdown():
        _time.sleep(AUTO_SHUTDOWN_MIN * 60)
        print("[databricks_launcher] auto-shutdown after "
              f"{AUTO_SHUTDOWN_MIN} min (auto_shutdown_min).", flush=True)
        os._exit(0)

    _threading.Thread(target=_auto_shutdown, daemon=True).start()


def _register_prompter():
    """Reconstruct a PrompterAPI ON THE DRIVER (the notebook's live fh_prompter
    cannot cross the process boundary) and register it as the webapp engine.

    Preferred path: explicit credentials threaded from the notebook's live
    ``fh_prompter`` via env vars (``GEOTECH_FH_*``) — Prompter auth is NTLM
    with a domain service account, and username/password/base_url are plain
    strings, so they reconstruct a fully working client with NO dbutils/Py4J
    involvement. Fallback: bare ``PrompterAPI(chat_model=...)`` self-config."""
    import os as _os
    from webapp.engine_config import register_model_builder
    from funhouse.services.prompter.prompter_api import PrompterAPI
    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    _user = _os.environ.get("GEOTECH_FH_USERNAME")
    _pw = _os.environ.get("GEOTECH_FH_PASSWORD")
    _burl = _os.environ.get("GEOTECH_FH_BASE_URL")
    if _user and _pw and _burl:
        prompter = PrompterAPI(
            backend="prompter", username=_user, password=_pw, base_url=_burl,
            verify=_os.environ.get("GEOTECH_FH_VERIFY", "1") == "1",
            chat_model=MODEL)
    else:
        prompter = PrompterAPI(chat_model=MODEL)

    def _build(model_id=None):
        # The in-app picker selection (GEOTECH_PROMPTER_MODELS entries)
        # arrives as model_id; None -> the launch-time default.
        return PrompterChatModel(prompter=prompter, model=model_id or MODEL)

    register_model_builder(_build)


try:
    _register_prompter()
    print("[databricks_launcher] Prompter engine registered (model=%s)." % MODEL,
          flush=True)
except Exception as exc:  # fall back to the ANTHROPIC_API_KEY path
    print("[databricks_launcher] Could not construct a PrompterAPI on the driver "
          "(%s: %s)." % (type(exc).__name__, exc), flush=True)
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[databricks_launcher] Falling back to ANTHROPIC_API_KEY.", flush=True)
    else:
        print("[databricks_launcher] No ANTHROPIC_API_KEY either — the app will "
              "show the 'no engine configured' banner.", flush=True)

from streamlit.web import bootstrap

# Belt-and-braces: ALSO set the streamlit config env vars. flag_options
# naming has churned across streamlit majors (tornado -> uvicorn server);
# env vars are honored by every version, and XSRF being accidentally ON
# behind the driver proxy 403s every file-upload PUT.
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")

# flag_options keys are the CLI-flag form; streamlit maps "_" -> "." internally
# (server_port -> server.port). CORS + XSRF are disabled so the driver proxy can
# frame/serve the app; headless suppresses the browser-open attempt. The watcher
# is off (no "Rerun / Always rerun" prompt) and the toolbar is viewer-mode — the
# same production posture as .streamlit/config.toml, set here too because the
# subprocess CWD may not be the repo root where that file is read from.
_FLAGS = {
    "server_port": PORT,
    "server_address": "0.0.0.0",
    "server_baseUrlPath": BASE_PATH,
    "server_enableCORS": False,
    "server_enableXsrfProtection": False,
    "server_headless": True,
    "server_fileWatcherType": "none",
    "client_toolbarMode": "viewer",
    "browser_gatherUsageStats": False,
    # THE "Connecting"-flap root cause (2026-09-02, confirmed by the funhouse
    # dev's WS probe): the driver proxy swallows WebSocket control frames, so
    # the server's protocol pings (default 30s interval + 30s timeout) never
    # get a pong back and the SERVER hangs up every socket ~60s after open.
    # Push the ping horizon out to an hour; liveness during turns comes from
    # the app-level heartbeat (core.with_heartbeat) whose DATA frames do
    # traverse the proxy.
    "server_websocketPingInterval": 3600,
}
bootstrap.run(APP_PATH, False, [], _FLAGS)
''')


def render_bootstrap_script(
    *, app_path: str, repo_root: str, base: str, port: int, model: str,
    auto_shutdown_min: Optional[int] = None,
) -> str:
    """Render the standalone bootstrap-script source (pure; unit-tested).

    All values are injected as ``repr()`` literals so Windows paths, spaces, and
    quotes survive verbatim.
    """
    return _BOOTSTRAP_TEMPLATE.substitute(
        repo_root=repr(repo_root),
        app_path=repr(app_path),
        base=repr(base),
        port=repr(int(port)),
        model=repr(model),
        auto_shutdown_min=repr(
            int(auto_shutdown_min) if auto_shutdown_min else None),
    )


def build_launch_env(
    base_env: dict,
    anthropic_key: Optional[str] = None,
    repo_root: Optional[str] = None,
    prompter: Any = None,
) -> dict:
    """Build the subprocess environment: inherit ``base_env`` (so the driver's
    Funhouse config env reaches the bootstrap), optionally inject
    ``ANTHROPIC_API_KEY``, and prepend ``repo_root`` to ``PYTHONPATH``.

    When the notebook's live ``prompter`` (``fh_prompter``) is given, its
    Prompter credentials — plain strings: NTLM ``username``/``password`` plus
    ``base_url``/``verify`` — are threaded through ``GEOTECH_FH_*`` env vars so
    the bootstrap reconstructs a working client with no dbutils/Py4J."""
    env = dict(base_env)
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key
    # The app user's own email, for the agent's "email it to me" (the
    # sanctioned resolution — fh_config session.user_name, per the SDK email
    # example — only works notebook-side, so capture it here). Best-effort.
    if not env.get("GEOTECH_USER_EMAIL"):
        try:
            from funhouse.config.funhouse_config import FunhouseConfig
            _ue = str(FunhouseConfig.get_instance().get(
                "session.user_name", default="") or "").strip()
            if "@" in _ue:
                env["GEOTECH_USER_EMAIL"] = _ue
        except Exception:
            pass
    if prompter is not None:
        user = getattr(prompter, "username", None)
        pw = getattr(prompter, "password", None)
        burl = getattr(prompter, "base_url", None)
        if user and pw and burl:
            env["GEOTECH_FH_USERNAME"] = str(user)
            env["GEOTECH_FH_PASSWORD"] = str(pw)
            env["GEOTECH_FH_BASE_URL"] = str(burl)
            env["GEOTECH_FH_VERIFY"] = (
                "1" if getattr(prompter, "verify", True) else "0")
    if repo_root:
        parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if repo_root not in parts:
            parts.insert(0, repo_root)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _default_app_path() -> str:
    """Absolute path to ``webapp/app.py`` shipped alongside this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


# ---------------------------------------------------------------------------
# SharePoint staging (delegated OAuth -> token file + refresher)
# ---------------------------------------------------------------------------

def _default_sp_token_getter() -> Callable[[], Optional[str]]:
    """Token getter for the State/Funhouse delegated-OAuth setup.

    ``MSALAuth(cache_storage="secret_manager")`` holds the user's years-long
    refresh token (established by ``%run setup_sharepoint``), so
    ``get_access_token_silent()`` mints a fresh ~90-min Graph token with no
    browser prompt. Falls back to a full ``authenticate()`` (which may prompt
    device-code — fine in a notebook) if the silent path returns nothing.
    """
    from funhouse.services.auth.msal_auth import MSALAuth
    ma = MSALAuth(cache_storage="secret_manager")

    def _get() -> Optional[str]:
        token = ma.get_access_token_silent()
        if token:
            return token
        result = ma.authenticate() or {}
        auth = (result.get("headers") or {}).get("Authorization", "")
        return auth.split()[-1] if auth else None

    return _get


def stage_sharepoint(
    site_url: str,
    root: str = "Shared Documents/GeotechStaffEngineer",
    *,
    token_getter: Optional[Callable[[], Optional[str]]] = None,
    token_path: str = "/tmp/geotech_sp_token.txt",
    refresh_interval_s: int = 1800,
    start_refresher: bool = True,
    env: Any = None,
) -> dict:
    """Stage SharePoint permanent storage for the app subprocess (notebook-side).

    Run this in the notebook BEFORE launching the app. It writes the current
    Graph token to a driver-local file, points the ``GEOTECH_SHAREPOINT_*``
    env vars at it (inherited by the launched app), and starts a daemon thread
    that re-mints a fresh token every ``refresh_interval_s`` seconds — so the
    app's token never goes stale while the notebook kernel lives (the app's
    token provider re-reads the file on every request/401-retry).

    Typical use::

        from webapp.databricks_launcher import stage_sharepoint, run_on_databricks
        stage_sharepoint("https://usdos.sharepoint.com/sites/CSEGeotechGroup",
                         root="Shared Documents/General/GSE_app")
        handle = run_on_databricks(prompter=fh_prompter)

    ``token_getter`` defaults to the Funhouse ``MSALAuth`` silent flow (the
    ``%run setup_sharepoint`` refresh token); pass a callable to override.
    """
    import threading

    if env is None:
        env = os.environ
    if token_getter is None:
        token_getter = _default_sp_token_getter()

    token = token_getter()
    if not token:
        raise RuntimeError(
            "Could not obtain a SharePoint Graph token — run "
            "'%run setup_sharepoint' first, then retry.")
    with open(token_path, "w", encoding="utf-8") as fh:
        fh.write(token)

    env["GEOTECH_SHAREPOINT_SITE_URL"] = site_url.strip().rstrip("/")
    env["GEOTECH_SHAREPOINT_TOKEN_FILE"] = token_path
    if root:
        env["GEOTECH_SHAREPOINT_ROOT"] = root.strip().strip("/")

    if start_refresher:
        def _refresh_loop() -> None:
            while True:
                time.sleep(refresh_interval_s)
                try:
                    fresh = token_getter()
                    if fresh:
                        with open(token_path, "w", encoding="utf-8") as fh:
                            fh.write(fresh)
                except Exception:
                    pass                     # keep last token; retry next tick

        threading.Thread(target=_refresh_loop, daemon=True,
                         name="geotech-sp-token-refresher").start()

    return {"site_url": env["GEOTECH_SHAREPOINT_SITE_URL"],
            "root": env.get("GEOTECH_SHAREPOINT_ROOT", ""),
            "token_path": token_path,
            "refresher": bool(start_refresher)}


# ---------------------------------------------------------------------------
# The launch handle + orchestrator
# ---------------------------------------------------------------------------

@dataclass
class LaunchHandle:
    """Handle for a launched streamlit process.

    Attributes
    ----------
    process : subprocess.Popen
        The streamlit bootstrap process.
    base_path : str
        The driver-proxy base path streamlit is serving under.
    port, model : int, str
        The port and Prompter model id used.
    script_path : str
        The temp bootstrap script (removed by :meth:`stop`).
    url : str | None
        The full openable URL, or ``None`` if the workspace host was unknown.
    workspace_host : str | None
        The resolved workspace host (``https://…``), if any.
    log_path : str | None
        Driver-local file receiving the app's stdout/stderr (bootstrap
        messages, streamlit banner, tracebacks). ``print(open(h.log_path
        ).read())`` to inspect.
    """

    process: Any
    base_path: str
    port: int
    model: str
    script_path: str
    url: Optional[str] = None
    workspace_host: Optional[str] = None
    log_path: Optional[str] = None

    def poll(self) -> Optional[int]:
        """Return the process exit code, or ``None`` while it is still running."""
        poll = getattr(self.process, "poll", None)
        return poll() if callable(poll) else None

    def stop(self) -> None:
        """Terminate the streamlit process and delete the temp bootstrap script."""
        try:
            self.process.terminate()
        except Exception:
            pass
        try:
            os.remove(self.script_path)
        except OSError:
            pass


def run_on_databricks(
    port: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    *,
    models: Optional[list] = None,
    auto_shutdown_min: Optional[int] = None,
    prompter: Any = None,
    spark: Any = None,
    org_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    workspace_host: Optional[str] = None,
    app_path: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    python_executable: Optional[str] = None,
    quiet: bool = False,
    _popen: Callable[..., Any] = subprocess.Popen,
) -> LaunchHandle:
    """Launch the Streamlit app on the Databricks driver under the driver proxy.

    Writes a bootstrap script that registers a reconstructed Funhouse Prompter
    engine (falling back to ``ANTHROPIC_API_KEY``) and runs streamlit in-process,
    then starts it as a background subprocess and returns a :class:`LaunchHandle`.

    Parameters
    ----------
    port : int, optional
        Driver port to serve on. Default ``None`` = the first FREE port from
        8501 upward (SDK-example pattern — avoids collisions with orphaned
        app processes on a shared driver). Pass an explicit port to pin it.
    model : str
        Prompter chat-model id the app STARTS on (default
        ``"funhouse-gpt-high"``).
    models : list of str, optional
        Entries for the in-app model picker (``"Label=id"`` or bare ids),
        published via ``GEOTECH_PROMPTER_MODELS``. Default: the launch
        ``model`` plus ``funhouse-gpt-medium`` (GPT 5.1 — the cheaper tier),
        deduplicated. Pass ``[]`` to disable the picker (model fixed).
    auto_shutdown_min : int, optional
        Auto-terminate the app process after this many minutes (the Funhouse
        SDK examples enforce 10 on their apps as shared-cluster etiquette).
        Default ``None`` = run until ``handle.stop()`` / cluster shutdown.
    prompter : PrompterAPI, optional
        **Pass the notebook's live ``fh_prompter`` — the reliable path.** Its
        NTLM credentials (plain strings) are threaded to the app process,
        which reconstructs a working client with no dbutils/Py4J. Without it,
        the bootstrap falls back to bare ``PrompterAPI()`` self-configuration,
        which fails on workspaces whose Funhouse config needs the notebook's
        live session (observed live 2026-07-24: Py4J "Object ID unknown").
    spark : SparkSession, optional
        The notebook's spark session; auto-detected if omitted.
    org_id, cluster_id : str, optional
        Override the driver-proxy identifiers instead of reading spark conf.
    workspace_host : str, optional
        Override the workspace host for the printed URL (e.g.
        ``"https://dbc-….cloud.databricks.com"``).
    app_path : str, optional
        Path to ``app.py`` (defaults to the one shipped in ``webapp/``).
    anthropic_key : str, optional
        If given (e.g. ``dbutils.secrets.get(...)``), threaded into the subprocess
        env as the Prompter fallback.
    python_executable : str, optional
        Python used to run the bootstrap (defaults to ``sys.executable``).
    quiet : bool
        Suppress the printed banner.
    _popen : callable
        Injection seam for testing (defaults to ``subprocess.Popen``).
    """
    if port is None:
        try:
            port = find_available_port(DEFAULT_PORT)
        except RuntimeError:
            port = DEFAULT_PORT
    org_id, cluster_id = resolve_cluster_ids(
        spark=spark, org_id=org_id, cluster_id=cluster_id)
    base = driver_proxy_base_path(org_id, cluster_id, port)

    app_path = os.path.abspath(app_path or _default_app_path())
    repo_root = os.path.dirname(os.path.dirname(app_path))

    script = render_bootstrap_script(
        app_path=app_path, repo_root=repo_root, base=base, port=port, model=model,
        auto_shutdown_min=auto_shutdown_min)
    fd, script_path = tempfile.mkstemp(
        prefix="geotech_streamlit_boot_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(script)

    env = build_launch_env(
        os.environ, anthropic_key=anthropic_key, repo_root=repo_root,
        prompter=prompter)
    # In-app Prompter model picker: publish the choices (None -> launch model
    # + the cheaper funhouse-gpt-medium; [] -> no picker, model fixed).
    if models is None:
        models = [model] + (["funhouse-gpt-medium"]
                            if model != "funhouse-gpt-medium" else [])
    if models:
        env["GEOTECH_PROMPTER_MODELS"] = ",".join(
            str(m).strip() for m in models if str(m).strip())
    python_exe = python_executable or sys.executable

    # Detach the app process from the notebook session: its own log file
    # (instead of inheriting the kernel's console pipes — which entangles the
    # child with the notebook's session plumbing and was correlated with
    # spurious Py4J errors on subsequent cells) and, on POSIX, its own
    # session/process group so kernel restarts can't signal it.
    log_path = os.path.join(tempfile.gettempdir(),
                            f"geotech_webapp_{int(port)}.log")
    popen_kwargs: dict = {"env": env}
    try:
        log_fh = open(log_path, "ab")
        popen_kwargs["stdout"] = log_fh
        popen_kwargs["stderr"] = subprocess.STDOUT
    except OSError:
        log_fh, log_path = None, None
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    try:
        process = _popen([python_exe, script_path], **popen_kwargs)
    except TypeError:      # an injected _popen with a narrower signature
        process = _popen([python_exe, script_path], env=env)
    if log_fh is not None:
        log_fh.close()                    # the child holds its own copy

    if workspace_host is None:
        workspace_host = workspace_host_from_spark(spark or _active_spark())
    url = proxy_url(workspace_host, base)

    handle = LaunchHandle(
        process=process, base_path=base, port=port, model=model,
        script_path=script_path, url=url, workspace_host=workspace_host,
        log_path=log_path)

    if not quiet:
        print(f"[databricks_launcher] streamlit starting on port {port} "
              f"(model={model}).")
        print("[databricks_launcher] First click may show '502 Bad Gateway' "
              "while the app boots (~20-30 s) — just reload.")
        if log_path:
            print(f"[databricks_launcher] App log: {log_path}")
        if url:
            print(f"[databricks_launcher] Open: {url}")
        else:
            print(f"[databricks_launcher] base path: {base}")
            print("[databricks_launcher] Workspace host unknown — open "
                  f"https://<your-workspace-host>{base}/ (pass workspace_host=... "
                  "to have it printed for you).")
        print("[databricks_launcher] Call handle.stop() to shut it down.")

    return handle
