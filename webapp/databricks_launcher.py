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

With a personal Prompter API key instead of ``fh_prompter`` (BACKLOG TA-27)::

    import webapp.databricks_launcher as dl
    handle = dl.run_on_databricks(prompter_key=dict(
        url="https://<host>/api/v1/chat/completions",
        model="<deployment name>",
        api_key=fh_secrets.get_secret("<secret name>")))

The launcher puts those values in the APP's environment only (never the
notebook's, never printed), boots ``webapp/tinyapps_entry.py`` (whose engine
reads them), offers that one model in the picker and prints it in the banner.

An app this notebook has lost
-----------------------------
The app runs in its own session, so it outlives a restart of the notebook's
Python, which loses ``handle``. :func:`running_apps` lists the copies running
on this driver (found by the bootstrap script in their command line),
:func:`stop_running` stops one or all of them, and :func:`diagnose` prints what
a copy is doing: its network connections, the system call each thread waits
in, and the end of its log. All three read ``/proc`` (Linux only) and accept a
``proc_root`` so they can be tested on a fake tree. :func:`run_on_databricks`
refuses to start a second copy on a port a running copy holds.
"""

from __future__ import annotations

import os
import re
import signal
import string
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    # the Prompter-key route (TA-27)
    "PrompterKey",
    "resolve_prompter_key",
    # an app the notebook has lost (TA-27)
    "running_apps",
    "stop_running",
    "diagnose",
    "RunningApp",
    "PortInUse",
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

#: The bootstrap script's file-name prefix (``tempfile.mkstemp``) and the
#: app log's name: how :func:`running_apps` recognises a copy of the app.
BOOT_PREFIX = "geotech_streamlit_boot_"
LOG_NAME = "geotech_webapp_{port}.log"

#: How the launched app gets its model: the notebook's Funhouse SDK
#: (``fh_prompter`` / driver config) or a personal Prompter API key.
ROUTE_SDK = "funhouse-sdk"
ROUTE_KEY = "prompter-key"


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
KEY_ROUTE = $key_route
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


if KEY_ROUTE:
    # The Prompter-key route (TA-27): webapp/tinyapps_entry.py registers the
    # engine itself from PROMPTER_URL / PROMPTER_MODEL / PROMPTER_API_KEY,
    # which the launcher put in THIS process's environment. The Funhouse SDK
    # is not used, and its bare self-configuration fails noisily on some
    # workspaces, so it is not attempted.
    print("[databricks_launcher] Prompter key route: the app's engine uses "
          "the Prompter API key it was given (model=%s)." % MODEL, flush=True)
else:
    try:
        _register_prompter()
        print("[databricks_launcher] Prompter engine registered (model=%s)."
              % MODEL, flush=True)
    except Exception as exc:  # fall back to the ANTHROPIC_API_KEY path
        print("[databricks_launcher] Could not construct a PrompterAPI on the "
              "driver (%s: %s)." % (type(exc).__name__, exc), flush=True)
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("[databricks_launcher] Falling back to ANTHROPIC_API_KEY.",
                  flush=True)
        else:
            print("[databricks_launcher] No ANTHROPIC_API_KEY either — the app "
                  "will show the 'no engine configured' banner.", flush=True)

from streamlit.web import bootstrap

# Belt-and-braces: ALSO set the streamlit config env vars. flag_options
# naming has churned across streamlit majors (tornado -> uvicorn server);
# env vars are honored by every version, and XSRF being accidentally ON
# behind the driver proxy 403s every file-upload PUT.
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
# Attach files over the websocket instead of HTTP PUT — the driver proxy
# 403s the native uploader's PUT (POSTs pass; suspected method filter).
# Override with GEOTECH_UPLOAD_MODE=http to A/B test the native path.
os.environ.setdefault("GEOTECH_UPLOAD_MODE", "ws")

# flag_options keys are the CLI-flag form; streamlit maps "_" -> "." internally
# (server_port -> server.port). CORS + XSRF are disabled so the driver proxy can
# frame/serve the app; headless suppresses the browser-open attempt. The watcher
# is off (no "Rerun / Always rerun" prompt) and the toolbar is viewer-mode — the
# same production posture as .streamlit/config.toml, set here too because the
# subprocess CWD may not be the repo root where that file is read from.
_FLAGS = {
    "server_port": PORT,
    "server_address": "0.0.0.0",
    # server_baseUrlPath is deliberately ABSENT. The driver proxy STRIPS the
    # /driver-proxy/o/{org}/{cluster}/{port} prefix before forwarding, so the
    # app must serve at root. Setting baseUrlPath here 404s every request
    # ("Not Found") — proven live 2026-09-04 on 5.11.2, the first release
    # where load_config_options() made these flags real (5.10.x-5.11.1
    # worked only because the flag was silently ignored). BASE_PATH is still
    # used for the browser-facing URL.
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
# THE missing call (found 2026-09-03 via a lab ws probe replicating the live
# 30s-ping/60s-1011 signature exactly): bootstrap.run() only installs config
# WATCHERS for flag_options — it never applies them; the streamlit CLI does
# that via load_config_options() before calling run(). Without this line
# every _FLAGS entry was silently ignored (masked by env-var backups and by
# server_port matching the 8501 default) — including websocketPingInterval,
# which is why sockets kept dying at exactly 60s on 5.11.1.
bootstrap.load_config_options(flag_options=_FLAGS)
bootstrap.run(APP_PATH, False, [], _FLAGS)
''')


def render_bootstrap_script(
    *, app_path: str, repo_root: str, base: str, port: int, model: str,
    auto_shutdown_min: Optional[int] = None, key_route: bool = False,
) -> str:
    """Render the standalone bootstrap-script source (pure; unit-tested).

    All values are injected as ``repr()`` literals so Windows paths, spaces, and
    quotes survive verbatim. ``key_route`` (the Prompter-key route) skips the
    Funhouse SDK registration: the app's own entry registers the engine from
    the ``PROMPTER_*`` settings. No key is ever written into the script.
    """
    return _BOOTSTRAP_TEMPLATE.substitute(
        repo_root=repr(repo_root),
        app_path=repr(app_path),
        base=repr(base),
        port=repr(int(port)),
        model=repr(model),
        key_route=repr(bool(key_route)),
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
    # METERING OBLIGATION. Funhouse meters Prompter usage per user by writing
    # to a LOCAL SQLite file whose path lives only in the notebook kernel's
    # in-memory FunhouseConfig singleton. This app is Popen'd as a fresh
    # process, so it sees config DEFAULTS, `should_write_sqlite_for_user`
    # returns False, and meter_log silently drops EVERY record — the one
    # diagnostic it emits is INFO, below the default WARNING stdout level, so
    # nothing ever says so. Diagnosed 2026-09-10 with executed proof; the
    # notebook kernel's own usage still meters, which is why the counter looked
    # low rather than empty. See module_work/prompter_metering/DIAGNOSIS.md.
    #
    # FunhouseConfig._load_env_vars maps FUNHOUSE_<A>__<B> -> a.b, so handing
    # the child these env vars restores the meter with no SDK change. The
    # per-user sqlite DIRECTORY carries the user's identity in its path, which
    # is what the admin cross-user report keys on — so attribution survives
    # even if the user_name column reads unknown_user.
    try:
        from funhouse.config.funhouse_config import FunhouseConfig
        _cfg = FunhouseConfig.get_instance()
        for _key, _env in (
            ("budget.storage_backend", "FUNHOUSE_BUDGET__STORAGE_BACKEND"),
            ("budget.sqlite_directory", "FUNHOUSE_BUDGET__SQLITE_DIRECTORY"),
            ("budget.sqlite_path", "FUNHOUSE_BUDGET__SQLITE_PATH"),
            ("budget.sqlite_filename", "FUNHOUSE_BUDGET__SQLITE_FILENAME"),
            ("budget.monthly_budget", "FUNHOUSE_BUDGET__MONTHLY_BUDGET"),
            ("session.user_name", "FUNHOUSE_SESSION__USER_NAME"),
        ):
            _val = _cfg.get(_key, default=None)
            if _val not in (None, "") and not env.get(_env):
                env[_env] = str(_val)
        _who = str(_cfg.get("session.user_name", default="") or "").strip()
        if _who and not env.get("CURRENT_USER_NAME"):
            env["CURRENT_USER_NAME"] = _who
    except Exception:
        pass          # a launch must never fail over metering setup

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


def _default_app_path(pages: bool = True) -> str:
    """Absolute path to the Streamlit script shipped alongside this module:
    the two-page ``webapp/pages_entry.py`` (Document Review + the geotech
    page, 5.26), or the single-page ``webapp/app.py`` when ``pages`` is
    False."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pages_entry.py" if pages else "app.py")


def _tinyapps_entry_path() -> str:
    """``webapp/tinyapps_entry.py``: the entry whose engine reads the
    ``PROMPTER_*`` settings (the Prompter-key route)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tinyapps_entry.py")


# ---------------------------------------------------------------------------
# The Prompter-key route (BACKLOG TA-27)
# ---------------------------------------------------------------------------

#: What the key route hands the app, in the names ``webapp.tinyapps_engine``
#: reads.
PROMPTER_ENV_NAMES = {
    "url": "PROMPTER_URL",
    "model": "PROMPTER_MODEL",
    "api_key": "PROMPTER_API_KEY",
    "ca_bundle": "PROMPTER_CA_BUNDLE",
}
_PROMPTER_KEY_ALIASES = {"key": "api_key", "apikey": "api_key",
                         "api-key": "api_key", "ca": "ca_bundle"}
_PROMPTER_KEY_HELP = (
    "prompter_key takes dict(url=..., model=..., api_key=..., ca_bundle=None) "
    "(url is the full https://<host>/api/v1/chat/completions address, model "
    "the deployment name), or True to read PROMPTER_URL, PROMPTER_MODEL and "
    "PROMPTER_API_KEY from the environment.")


@dataclass(frozen=True)
class PrompterKey:
    """A personal Prompter API key and where it points.

    The key and the CA bundle are left out of ``repr``, so printing this
    object never shows them."""

    url: str
    model: str
    api_key: str = field(repr=False)
    ca_bundle: Optional[str] = field(default=None, repr=False)

    def env(self) -> dict:
        """The app process's settings (the only place the key travels)."""
        out = {PROMPTER_ENV_NAMES["url"]: self.url,
               PROMPTER_ENV_NAMES["model"]: self.model,
               PROMPTER_ENV_NAMES["api_key"]: self.api_key}
        if self.ca_bundle:
            out[PROMPTER_ENV_NAMES["ca_bundle"]] = self.ca_bundle
        return out


def resolve_prompter_key(prompter_key: Any,
                         environ: Optional[Mapping] = None
                         ) -> Optional[PrompterKey]:
    """Read ``run_on_databricks(prompter_key=...)`` into a :class:`PrompterKey`.

    ``None`` / ``False`` -> ``None`` (the Funhouse SDK route). A dict gives
    ``url``, ``model``, ``api_key`` and optionally ``ca_bundle`` (PEM text or
    a path; ``key`` is accepted for ``api_key``); a value it leaves out is
    read from the environment (``PROMPTER_URL`` …). ``True`` reads all four
    from the environment. Raises ``ValueError`` naming what is missing and
    ``TypeError`` for a shape it does not know; neither message ever
    contains a value."""
    if prompter_key is None or prompter_key is False:
        return None
    environ = os.environ if environ is None else environ
    given: dict = {}
    if prompter_key is True:
        pass
    elif isinstance(prompter_key, Mapping):
        for raw_name, value in prompter_key.items():
            name = str(raw_name).strip().lower()
            name = _PROMPTER_KEY_ALIASES.get(name, name)
            if name not in PROMPTER_ENV_NAMES:
                raise TypeError(
                    f"prompter_key has an entry the launcher does not know: "
                    f"{str(raw_name)!r}. " + _PROMPTER_KEY_HELP)
            given[name] = value
    else:
        raise TypeError(_PROMPTER_KEY_HELP)

    values: dict = {}
    for name, env_name in PROMPTER_ENV_NAMES.items():
        value = given.get(name)
        text = "" if value is None else str(value).strip()
        if not text:
            text = str(environ.get(env_name) or "").strip()
        values[name] = text or None
    missing = [n for n in ("url", "model", "api_key") if not values[n]]
    if missing:
        names = ", ".join(f"{n} (or the {PROMPTER_ENV_NAMES[n]} setting)"
                          for n in missing)
        raise ValueError(f"The Prompter key route is missing {names}. "
                         + _PROMPTER_KEY_HELP)
    if not values["url"].lower().startswith(("https://", "http://")):
        raise ValueError(
            "prompter_key url must be the full address, like "
            "https://<host>/api/v1/chat/completions.")
    return PrompterKey(**values)


def _env_key_model(app_path: str, environ: Mapping) -> Optional[str]:
    """The older key-route cell (PROMPTER_* set in ``os.environ`` by hand and
    ``app_path=<webapp>/tinyapps_entry.py``): the model that app will really
    use, or ``None`` when this is not that cell. The banner printed the
    launcher's default model there (field session 2026-10-14)."""
    if os.path.basename(app_path) != "tinyapps_entry.py":
        return None
    if all(str(environ.get(PROMPTER_ENV_NAMES[n]) or "").strip()
           for n in ("url", "model", "api_key")):
        return str(environ[PROMPTER_ENV_NAMES["model"]]).strip()
    return None


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


#: How often the refresher looks at the staged token, and how close to its
#: expiry it asks for a new one.
TOKEN_CHECK_S = 60
TOKEN_EXPIRY_MARGIN_S = 600


def token_expiry(token: Optional[str]) -> Optional[float]:
    """The ``exp`` (epoch seconds) of a JWT access token, read without
    verifying it; ``None`` for anything that is not a readable JWT."""
    import base64
    import json as _json
    try:
        payload = str(token or "").split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = _json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return float(exp) if exp is not None else None
    except Exception:                                      # noqa: BLE001
        return None


def token_refresh_due(token: Optional[str], last_refresh: float, now: float,
                      interval_s: float,
                      margin_s: float = TOKEN_EXPIRY_MARGIN_S) -> bool:
    """Whether the staged token should be re-minted now: the fixed interval
    has passed, or the token itself expires within ``margin_s``."""
    if now - last_refresh >= interval_s:
        return True
    exp = token_expiry(token)
    return exp is not None and exp - now <= margin_s


def run_token_refresher(token_getter: Callable[[], Optional[str]],
                        token_path: str, interval_s: float, *,
                        check_s: float = TOKEN_CHECK_S,
                        margin_s: float = TOKEN_EXPIRY_MARGIN_S,
                        sleep: Callable[[float], None] = time.sleep,
                        clock: Callable[[], float] = time.time,
                        stop: Optional[Callable[[], bool]] = None) -> None:
    """Keep the staged token fresh: look every ``check_s`` seconds and
    re-mint when the interval has passed OR the token is within ``margin_s``
    of its expiry.

    Why both (field session 2026-10-06): the loop used to sleep the whole
    interval (30 min) between re-mints, and the silent MSAL flow hands back
    the SAME cached access token until about five minutes before it expires.
    A token with ten minutes left at one tick therefore lapsed before the
    next, and the app ran on an expired token for up to twenty minutes -- the
    SDK's 401 retry re-reads the file and finds the same token. A SharePoint
    turn that afternoon failed every call within eight seconds while the
    same paths worked before and sixteen minutes after (the record kept no
    status code; it does now). Looking every minute and re-minting near
    expiry keeps the file holding a live token. Never raises."""
    last = clock()
    while not (stop and stop()):
        sleep(check_s)
        try:
            with open(token_path, "r", encoding="utf-8") as fh:
                current = fh.read().strip()
        except OSError:
            current = ""
        if not token_refresh_due(current, last, clock(), interval_s,
                                 margin_s):
            continue
        try:
            fresh = token_getter()
            if fresh:
                with open(token_path, "w", encoding="utf-8") as fh:
                    fh.write(fresh)
                last = clock()
        except Exception:                                  # noqa: BLE001
            pass                     # keep last token; retry next tick


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
    that re-mints a fresh token every ``refresh_interval_s`` seconds AND
    whenever the staged token is within ten minutes of expiring
    (:func:`run_token_refresher`) — so the app's token never goes stale while
    the notebook kernel lives (the app's token provider re-reads the file on
    every request/401-retry).

    Typical use::

        from webapp.databricks_launcher import stage_sharepoint, run_on_databricks
        stage_sharepoint("https://contoso.sharepoint.com/sites/GeotechTeam",
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
        threading.Thread(target=run_token_refresher,
                         args=(token_getter, token_path, refresh_interval_s),
                         daemon=True,
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
    route : str
        ``"funhouse-sdk"`` (the notebook's Prompter) or ``"prompter-key"``
        (a personal Prompter API key; ``model`` is then the key's model).
    """

    process: Any
    base_path: str
    port: int
    model: str
    script_path: str
    url: Optional[str] = None
    workspace_host: Optional[str] = None
    log_path: Optional[str] = None
    route: str = ROUTE_SDK

    @property
    def pid(self) -> Optional[int]:
        """The app's process id (what :func:`running_apps` lists)."""
        pid = getattr(self.process, "pid", None)
        return pid if isinstance(pid, int) else None

    def poll(self) -> Optional[int]:
        """Return the process exit code, or ``None`` while it is still running."""
        poll = getattr(self.process, "poll", None)
        return poll() if callable(poll) else None

    def stop(self, grace_s: float = 10.0) -> None:
        """Terminate the streamlit process and delete the temp bootstrap script.

        Asks politely (SIGTERM) and, if the process is still there after
        ``grace_s`` seconds, forces it (SIGKILL) — a hung app may not answer
        the first."""
        try:
            self.process.terminate()
        except Exception:
            pass
        wait = getattr(self.process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    wait(timeout=5)
                except Exception:                          # noqa: BLE001
                    pass
            except Exception:                              # noqa: BLE001
                pass
        try:
            os.remove(self.script_path)
        except OSError:
            pass
        if self.pid is not None and _LAUNCHED.get(self.pid) is self:
            _LAUNCHED.pop(self.pid, None)

    def diagnose(self, **kwargs) -> None:
        """:func:`diagnose` for this app."""
        if self.pid is None:
            print("[databricks_launcher] This handle has no process id "
                  "to look at.")
            return
        diagnose(self.pid, **kwargs)


def run_on_databricks(
    port: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    *,
    models: Optional[list] = None,
    auto_shutdown_min: Optional[int] = None,
    prompter: Any = None,
    prompter_key: Any = None,
    replace: bool = False,
    spark: Any = None,
    org_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    workspace_host: Optional[str] = None,
    app_path: Optional[str] = None,
    pages: bool = True,
    anthropic_key: Optional[str] = None,
    python_executable: Optional[str] = None,
    quiet: bool = False,
    _popen: Callable[..., Any] = subprocess.Popen,
    _running_apps: Optional[Callable[[], list]] = None,
    _stop_running: Optional[Callable[..., Any]] = None,
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
        A copy of the app already running here (one this notebook may have
        lost in a restart) is never doubled up on: with an explicit port it
        holds, nothing is started and :class:`PortInUse` says which process
        holds it and how to stop it; with the default, the next free port is
        used and the older copy is named. ``replace=True`` stops the copy on
        the port first.
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
    prompter_key : dict or True, optional
        **A personal Prompter API key instead of the Funhouse SDK**:
        ``dict(url="https://<host>/api/v1/chat/completions",
        model="<deployment name>", api_key=..., ca_bundle=None)``, or
        ``True`` to read ``PROMPTER_URL`` / ``PROMPTER_MODEL`` /
        ``PROMPTER_API_KEY`` (/ ``PROMPTER_CA_BUNDLE``) from the environment.
        The values go into the APP's environment only — not the notebook's,
        never printed or logged; the app boots ``webapp/tinyapps_entry.py``
        (unless ``app_path`` says otherwise), the picker offers that one model
        (unless ``models`` says otherwise), the banner and ``handle.model``
        name it, and ``prompter``/``model`` are not used.
    replace : bool
        Stop a copy of the app that holds the port (``port``, or 8501 by
        default) before starting this one. Default ``False``.
    spark : SparkSession, optional
        The notebook's spark session; auto-detected if omitted.
    org_id, cluster_id : str, optional
        Override the driver-proxy identifiers instead of reading spark conf.
    workspace_host : str, optional
        Override the workspace host for the printed URL (e.g.
        ``"https://dbc-….cloud.databricks.com"``).
    app_path : str, optional
        Path to the Streamlit script to boot. Defaults to the shipped
        two-page ``webapp/pages_entry.py`` (Document Review at the root,
        GeotechStaffEngineer at ``/geotech`` — the same two pages as the Tiny
        Apps deployment), or ``webapp/app.py`` when ``pages`` is False.
    pages : bool
        ``True`` (default, 5.26) boots the two-page app; ``False`` boots the
        single geotech page exactly as before 5.26 — the fallback if page
        switching misbehaves behind the driver proxy (the proxy strips the
        ``/driver-proxy/...`` prefix and ``server_baseUrlPath`` cannot be
        set, so the browser's page links are the thing to check live).
    anthropic_key : str, optional
        If given (e.g. ``dbutils.secrets.get(...)``), threaded into the subprocess
        env as the Prompter fallback.
    python_executable : str, optional
        Python used to run the bootstrap (defaults to ``sys.executable``).
    quiet : bool
        Suppress the printed banner.
    _popen : callable
        Injection seam for testing (defaults to ``subprocess.Popen``).
    _running_apps, _stop_running : callable
        Injection seams for testing (default :func:`running_apps` /
        :func:`stop_running`).
    """
    # The Prompter-key route (TA-27), resolved first so that a missing piece
    # fails before anything is written or started.
    key = resolve_prompter_key(prompter_key)
    if key is not None:
        app_path = app_path or _tinyapps_entry_path()
        model = key.model
        if models is None:
            models = [key.model]

    # A copy this notebook has lost may still hold the port: it runs in its
    # own session and outlives a restart of the notebook's Python.
    port = _claim_port(port, replace=replace, quiet=quiet,
                       list_apps=_running_apps or running_apps,
                       stop=_stop_running or stop_running)

    org_id, cluster_id = resolve_cluster_ids(
        spark=spark, org_id=org_id, cluster_id=cluster_id)
    base = driver_proxy_base_path(org_id, cluster_id, port)

    app_path = os.path.abspath(app_path or _default_app_path(pages))
    repo_root = os.path.dirname(os.path.dirname(app_path))

    route = ROUTE_SDK
    if key is not None:
        route = ROUTE_KEY
    else:
        # The older key-route cell: PROMPTER_* set by hand in os.environ and
        # app_path=<webapp>/tinyapps_entry.py. Name the model it really uses.
        env_model = _env_key_model(app_path, os.environ)
        if env_model:
            route, model = ROUTE_KEY, env_model
            if models is None:
                models = [env_model]

    script = render_bootstrap_script(
        app_path=app_path, repo_root=repo_root, base=base, port=port, model=model,
        auto_shutdown_min=auto_shutdown_min, key_route=(route == ROUTE_KEY))
    fd, script_path = tempfile.mkstemp(prefix=BOOT_PREFIX, suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(script)

    # On the key route the notebook's NTLM credentials are not needed by the
    # app, so they are not handed to it.
    env = build_launch_env(
        os.environ, anthropic_key=anthropic_key, repo_root=repo_root,
        prompter=None if route == ROUTE_KEY else prompter)
    if key is not None:
        env.update(key.env())
    if route == ROUTE_KEY:
        env["GEOTECH_WEBAPP_MODEL"] = model
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
                            LOG_NAME.format(port=int(port)))
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
        log_path=log_path, route=route)
    if handle.pid is not None:
        _LAUNCHED[handle.pid] = handle

    if not quiet:
        engine = ("your Prompter API key" if route == ROUTE_KEY
                  else "the Funhouse Prompter")
        print(f"[databricks_launcher] streamlit starting on port {port} "
              f"(model={model}; engine: {engine}).")
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
        print("[databricks_launcher] Call handle.stop() to shut it down. If "
              "this notebook's Python restarts, handle is lost but the app "
              "keeps running: running_apps() lists it, stop_running(pid) "
              "stops it, diagnose(pid) shows what it is doing.")

    return handle


# ---------------------------------------------------------------------------
# An app the notebook has lost (BACKLOG TA-27): find it, stop it, and see
# what it is waiting on — with /proc and `ss` only, no extra packages.
# ---------------------------------------------------------------------------

#: The apps THIS Python process launched (pid -> LaunchHandle), so a listing
#: can say which copy is this notebook's own. Empty after a restart, which is
#: exactly when the others are "lost".
_LAUNCHED: dict = {}


class PortInUse(RuntimeError):
    """A running copy of the app already holds the port asked for."""


@dataclass
class RunningApp:
    """One copy of the app running on this machine (:func:`running_apps`)."""

    pid: int
    port: Optional[int]
    started: Optional[float]          # epoch seconds
    script_path: str
    script_exists: bool
    log_path: Optional[str]
    python: str
    mine: bool = False

    @property
    def same_python(self) -> bool:
        """Whether it runs on this notebook's Python (a lost app can be on an
        older notebook environment than the kernel's)."""
        return self.python == sys.executable

    def describe(self, now: Optional[float] = None) -> str:
        whose = ("  [this notebook's app]" if self.mine
                 else "  [not started by this notebook's Python]")
        lines = [f"pid {self.pid}   port {self.port or 'unknown'}   "
                 f"started {_when(self.started, now)}{whose}"]
        log_note = ("" if self.log_path and os.path.isfile(self.log_path)
                    else "  (not found)")
        lines.append(f"    log:    {self.log_path or 'unknown'}{log_note}")
        lines.append(f"    script: {self.script_path}"
                     + ("" if self.script_exists else "  (already removed)"))
        py_note = ("" if self.same_python
                   else "  (not this notebook's Python: an older environment?)")
        lines.append(f"    python: {self.python}{py_note}")
        return "\n".join(lines)


class AppList(list):
    """The list :func:`running_apps` returns; it shows as a readable table."""

    def __init__(self, apps=(), note: str = ""):
        super().__init__(apps)
        self.note = note

    def __repr__(self) -> str:
        if self.note:
            return self.note
        if not self:
            return ("No copy of the app (GeotechStaffEngineer) is running on "
                    "this machine.")
        head = (f"{len(self)} cop{'y' if len(self) == 1 else 'ies'} of the app "
                "running on this machine:")
        tail = ("Stop one with stop_running(pid), all with stop_running(); "
                "see what one is doing with diagnose(pid).")
        return "\n".join([head] + [a.describe() for a in self] + [tail])

    __str__ = __repr__


def _when(started: Optional[float], now: Optional[float] = None) -> str:
    """``2026-10-14 10:42 (3 h 05 min ago)``."""
    if not started:
        return "unknown"
    now = time.time() if now is None else now
    mins = max(0, int((now - started) // 60))
    if mins >= 48 * 60:
        ago = f"{mins // 1440} d {(mins % 1440) // 60} h"
    elif mins >= 60:
        ago = f"{mins // 60} h {mins % 60:02d} min"
    else:
        ago = f"{mins} min"
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
    return f"{stamp} ({ago} ago)"


def _say(quiet: bool, text: str) -> None:
    if not quiet:
        print(f"[databricks_launcher] {text}")


def _linux_root(proc_root: Optional[str]) -> Optional[str]:
    """The /proc to read: the one given (tests), else the real one on Linux,
    else ``None`` (this machine has none)."""
    if proc_root is not None:
        return str(proc_root)
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        return "/proc"
    return None


def _not_linux_note() -> str:
    return (f"This works on Linux only (the Databricks driver): it reads "
            f"/proc. This machine is {sys.platform}, so there is nothing to "
            f"look at here.")


def _read_bytes(path: str, limit: int = 1 << 20) -> Optional[bytes]:
    try:
        with open(path, "rb") as fh:
            return fh.read(limit)
    except OSError:
        return None


def _read_text(path: str) -> Optional[str]:
    data = _read_bytes(path)
    return None if data is None else data.decode("utf-8", "replace")


def _link(path: str) -> Optional[str]:
    """Where a /proc symlink points (``socket:[123]``, a file path). A fake
    tree in the tests holds the target as the file's text instead."""
    try:
        return os.readlink(path)
    except OSError:
        pass
    if os.path.isfile(path) and not os.path.islink(path):
        text = _read_text(path)
        return text.strip() if text else None
    return None


def _cmdline(root: str, pid: int) -> list:
    data = _read_bytes(os.path.join(root, str(pid), "cmdline")) or b""
    return [p.decode("utf-8", "replace") for p in data.split(b"\0") if p]


def _app_script(argv: list) -> Optional[str]:
    """The bootstrap script in a process's command line, when it is a copy
    of the app (``<python> /tmp/geotech_streamlit_boot_*.py``)."""
    if not argv or "python" not in os.path.basename(argv[0]).lower():
        return None
    for arg in argv[1:]:
        name = os.path.basename(arg)
        if name.startswith(BOOT_PREFIX) and name.endswith(".py"):
            return arg
    return None


def _stat_fields(root: str, pid) -> Optional[list]:
    """``/proc/<pid>/stat`` from field 3 (state) on; the command name in
    field 2 may hold spaces and brackets, so split after its last ``)``."""
    text = _read_text(os.path.join(root, str(pid), "stat"))
    if not text or ")" not in text:
        return None
    return text.rsplit(")", 1)[1].split()


def _alive(root: str, pid: int) -> bool:
    fields = _stat_fields(root, pid)
    return bool(fields) and fields[0] not in ("Z", "X", "x")


def _clk_tck() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        return 100


def _start_time(root: str, pid: int) -> Optional[float]:
    """Process start, epoch seconds: boot time (``/proc/stat`` btime) plus
    the start tick in field 22 of ``/proc/<pid>/stat``."""
    fields = _stat_fields(root, pid)
    stat = _read_text(os.path.join(root, "stat")) or ""
    match = re.search(r"^btime\s+(\d+)", stat, re.M)
    try:
        return int(match.group(1)) + int(fields[19]) / _clk_tck()
    except (AttributeError, TypeError, IndexError, ValueError):
        return None


def _script_port(script_path: str) -> Optional[int]:
    text = _read_text(script_path) or ""
    match = re.search(r"^PORT = (\d+)\s*$", text, re.M)
    return int(match.group(1)) if match else None


def _fd_targets(root: str, pid: int) -> dict:
    """``{fd: target}`` for a process's open files (sockets read
    ``socket:[inode]``)."""
    fd_dir = os.path.join(root, str(pid), "fd")
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return {}
    out = {}
    for name in names:
        if name.isdigit():
            target = _link(os.path.join(fd_dir, name))
            if target:
                out[int(name)] = target
    return out


def _socket_inode(target: Optional[str]) -> Optional[int]:
    match = re.match(r"socket:\[(\d+)\]", target or "")
    return int(match.group(1)) if match else None


_TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV",
    "04": "FIN_WAIT1", "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE",
    "08": "CLOSE_WAIT", "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
}


def _decode_addr(text: str) -> tuple:
    """``0100007F:1F95`` -> ``("127.0.0.1", 8085)``; IPv6 too (each 32-bit
    word of /proc's hex is in host byte order, little-endian here)."""
    import ipaddress
    host, port = text.split(":")
    raw = bytes.fromhex(host)
    if len(raw) == 4:
        ip = str(ipaddress.IPv4Address(raw[::-1]))
    else:
        raw = b"".join(raw[i:i + 4][::-1] for i in range(0, len(raw), 4))
        addr = ipaddress.IPv6Address(raw)
        ip = str(addr.ipv4_mapped or addr)
    return ip, int(port, 16)


@dataclass
class _Sock:
    inode: int
    local_ip: str
    local_port: int
    remote_ip: str
    remote_port: int
    state: str
    tx_queue: int
    rx_queue: int


def _net_sockets(root: str, pid: int) -> dict:
    """``{inode: _Sock}`` from ``/proc/<pid>/net/tcp`` and ``tcp6`` (the
    process's own network namespace)."""
    out = {}
    for table in ("tcp", "tcp6"):
        text = _read_text(os.path.join(root, str(pid), "net", table)) or ""
        for line in text.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                lip, lport = _decode_addr(parts[1])
                rip, rport = _decode_addr(parts[2])
                tx, rx = (int(x, 16) for x in parts[4].split(":"))
                inode = int(parts[9])
            except (ValueError, TypeError):
                continue
            if inode:
                out[inode] = _Sock(inode, lip, lport, rip, rport,
                                   _TCP_STATES.get(parts[3].upper(), parts[3]),
                                   tx, rx)
    return out


def _listening_port(root: str, pid: int, fds: dict) -> Optional[int]:
    socks = _net_sockets(root, pid)
    for target in fds.values():
        sock = socks.get(_socket_inode(target))
        if sock is not None and sock.state == "LISTEN":
            return sock.local_port
    return None


def _log_from_fds(fds: dict) -> Optional[str]:
    """The app's log is its stdout (fd 1)."""
    target = (fds.get(1) or "").replace(" (deleted)", "")
    return target if target.endswith(".log") else None


def running_apps(*, proc_root: Optional[str] = None,
                 tmp_dir: Optional[str] = None) -> AppList:
    """The copies of the app running on this machine.

    A copy is a ``python`` process whose command line holds a
    ``geotech_streamlit_boot_*.py`` bootstrap script. For each: ``pid``,
    ``port`` (the script's ``PORT =`` line, else its listening socket),
    ``started``, ``script_path``, ``log_path`` (its stdout, else
    ``geotech_webapp_<port>.log`` in the temp folder), the ``python`` it runs
    on, and ``mine`` — whether this notebook's Python launched it (a copy
    from before a restart is not). Returns a list that shows itself as a
    table; on Windows / macOS it is empty and says this is Linux-only.
    ``proc_root`` / ``tmp_dir`` point it at a fake tree (tests)."""
    root = _linux_root(proc_root)
    if root is None:
        return AppList([], note=_not_linux_note())
    tmp = tmp_dir or tempfile.gettempdir()
    try:
        names = os.listdir(root)
    except OSError as exc:
        return AppList([], note=f"Could not read {root}: {exc}")
    me = os.getpid()
    apps = []
    for name in names:
        if not name.isdigit() or int(name) == me:
            continue
        pid = int(name)
        argv = _cmdline(root, pid)
        script = _app_script(argv)
        if not script or not _alive(root, pid):
            continue
        fds = _fd_targets(root, pid)
        port = _script_port(script) or _listening_port(root, pid, fds)
        log = _log_from_fds(fds) or (
            os.path.join(tmp, LOG_NAME.format(port=port)) if port else None)
        apps.append(RunningApp(
            pid=pid, port=port, started=_start_time(root, pid),
            script_path=script, script_exists=os.path.isfile(script),
            log_path=log, python=argv[0], mine=pid in _LAUNCHED))
    apps.sort(key=lambda a: (a.started or 0.0, a.pid))
    return AppList(apps)


def _wait_gone(root: str, pids: list, grace_s: float,
               sleep: Callable[[float], None], step: float = 0.25) -> list:
    """Poll until every pid has exited or ``grace_s`` passes; the ones
    still alive."""
    left = [p for p in pids if _alive(root, p)]
    for _ in range(max(1, int(grace_s / step))):
        if not left:
            break
        sleep(step)
        for p in list(left):
            handle = _LAUNCHED.get(p)
            if handle is not None:
                handle.poll()          # reap our own child, or it lingers
            if not _alive(root, p):
                left.remove(p)
    return left


def stop_running(pid: Optional[int] = None, *, grace_s: float = 10.0,
                 proc_root: Optional[str] = None,
                 tmp_dir: Optional[str] = None, quiet: bool = False,
                 _kill: Optional[Callable[[int, int], None]] = None,
                 _sleep: Callable[[float], None] = time.sleep) -> list:
    """Stop one copy of the app (``pid``) or every copy (no ``pid``).

    Only processes :func:`running_apps` lists are ever signalled — a pid
    that is not a copy of the app is refused. Each is asked to stop
    (SIGTERM); one still there after ``grace_s`` seconds is forced
    (SIGKILL). Its bootstrap script is removed; its log is kept. Returns the
    pids that stopped. Linux only. On a cluster other people use, "every
    copy" means every copy on this driver that you are allowed to stop."""
    root = _linux_root(proc_root)
    if root is None:
        _say(quiet, _not_linux_note())
        return []
    apps = running_apps(proc_root=root, tmp_dir=tmp_dir)
    if pid is not None:
        targets = [a for a in apps if a.pid == int(pid)]
        if not targets:
            _say(quiet, f"No copy of the app with pid {pid} is running here, "
                        "so nothing was stopped. running_apps() lists the "
                        "ones that are.")
            return []
    else:
        targets = list(apps)
        if not targets:
            _say(quiet, "No copy of the app is running here; nothing to stop.")
            return []

    kill = _kill or os.kill
    term = getattr(signal, "SIGTERM", 15)
    force = getattr(signal, "SIGKILL", 9)
    asked = []
    for app in targets:
        try:
            kill(app.pid, term)
            asked.append(app)
            _say(quiet, f"Asked pid {app.pid} (port {app.port or 'unknown'}) "
                        "to stop.")
        except ProcessLookupError:
            asked.append(app)                  # already gone
        except PermissionError:
            _say(quiet, f"Not allowed to stop pid {app.pid} (it may belong "
                        "to another user on this cluster).")
    left = _wait_gone(root, [a.pid for a in asked], grace_s, _sleep)
    for p in left:
        try:
            kill(p, force)
            _say(quiet, f"pid {p} did not stop within {grace_s:g} s; "
                        "forced it to stop.")
        except ProcessLookupError:
            pass
        except PermissionError:
            _say(quiet, f"Not allowed to force pid {p} to stop.")
    if left:
        left = _wait_gone(root, left, 2.0, _sleep)

    stopped = []
    for app in asked:
        if app.pid in left:
            _say(quiet, f"pid {app.pid} is still running.")
            continue
        stopped.append(app.pid)
        if os.path.basename(app.script_path).startswith(BOOT_PREFIX):
            try:
                os.remove(app.script_path)
            except OSError:
                pass
        handle = _LAUNCHED.pop(app.pid, None)
        if handle is not None:
            handle.poll()
        _say(quiet, f"Stopped pid {app.pid} (port {app.port or 'unknown'}). "
                    f"Its log stays at {app.log_path or 'an unknown path'}.")
    return stopped


def port_in_use_message(app: RunningApp, port: int) -> str:
    """What :class:`PortInUse` says: who holds the port and how to free it."""
    return (
        f"Port {port} is already held by a copy of the app started "
        f"{_when(app.started)} (pid {app.pid}). Not starting a second copy "
        f"on the same port.\n"
        f"  To stop it:  from webapp.databricks_launcher import stop_running;"
        f" stop_running({app.pid})   then run this cell again.\n"
        f"  To stop it and start fresh in one go: add replace=True to this "
        f"call.\n"
        f"  To keep it and start another copy: leave out port= (the next "
        f"free port is used).")


def _claim_port(port: Optional[int], *, replace: bool, quiet: bool,
                list_apps: Callable[[], list],
                stop: Callable[..., Any]) -> int:
    """The port this launch uses, after checking for a copy already running.

    An explicit port a running copy holds -> :class:`PortInUse` (nothing is
    started), unless ``replace`` stops that copy first. The default (``None``)
    takes the next free port from 8501 and names any older copy left running,
    with how to stop it — so a lost app is never silently doubled up on."""
    def _apps() -> list:
        try:
            return list(list_apps())
        except Exception:                                  # noqa: BLE001
            return []

    apps = _apps()
    target = DEFAULT_PORT if port is None else int(port)
    holders = [a for a in apps if a.port == target]
    if holders and replace:
        for app in holders:
            _say(quiet, f"replace=True: stopping the copy on port {target} "
                        f"(pid {app.pid}) first.")
            stop(pid=app.pid, quiet=quiet)
        apps = _apps()
        holders = [a for a in apps if a.port == target]
    if holders and port is not None:
        raise PortInUse(port_in_use_message(holders[0], target))
    if port is None:
        try:
            port = find_available_port(DEFAULT_PORT)
        except RuntimeError:
            port = DEFAULT_PORT
    for app in apps:
        if app.port != port:
            _say(quiet, f"Note: another copy of the app is still running here: "
                        f"pid {app.pid} on port {app.port or 'unknown'}, "
                        f"started {_when(app.started)}. This launch uses port "
                        f"{port}. To stop the older copy: stop_running("
                        f"{app.pid}) (or relaunch with replace=True to take "
                        f"over port {app.port or DEFAULT_PORT}).")
    return int(port)


# --- diagnose ---------------------------------------------------------------

#: System-call numbers worth naming, per architecture (Linux syscall tables:
#: arch/x86/entry/syscalls/syscall_64.tbl and asm-generic/unistd.h).
SYSCALLS = {
    "x86_64": {
        0: "read", 1: "write", 2: "open", 3: "close", 7: "poll", 16: "ioctl",
        17: "pread64", 19: "readv", 20: "writev", 23: "select",
        24: "sched_yield", 34: "pause", 35: "nanosleep", 41: "socket",
        42: "connect", 43: "accept", 44: "sendto", 45: "recvfrom",
        46: "sendmsg", 47: "recvmsg", 48: "shutdown", 56: "clone",
        61: "wait4", 72: "fcntl", 128: "rt_sigtimedwait",
        130: "rt_sigsuspend", 202: "futex", 230: "clock_nanosleep",
        232: "epoll_wait", 247: "waitid", 257: "openat", 270: "pselect6",
        271: "ppoll", 281: "epoll_pwait", 288: "accept4", 299: "recvmmsg",
        435: "clone3", 441: "epoll_pwait2",
    },
    "aarch64": {
        22: "epoll_pwait", 25: "fcntl", 29: "ioctl", 56: "openat",
        57: "close", 63: "read", 64: "write", 65: "readv", 66: "writev",
        67: "pread64", 72: "pselect6", 73: "ppoll", 95: "waitid",
        98: "futex", 101: "nanosleep", 115: "clock_nanosleep",
        124: "sched_yield", 133: "rt_sigsuspend", 137: "rt_sigtimedwait",
        198: "socket", 202: "accept", 203: "connect", 206: "sendto",
        207: "recvfrom", 210: "shutdown", 211: "sendmsg", 212: "recvmsg",
        220: "clone", 242: "accept4", 243: "recvmmsg", 260: "wait4",
        435: "clone3", 441: "epoll_pwait2",
    },
}

#: Calls whose first argument is a file descriptor (so its target is shown).
_FD_CALLS = {"read", "write", "readv", "writev", "pread64", "recvfrom",
             "recvmsg", "recvmmsg", "sendto", "sendmsg", "connect", "accept",
             "accept4", "shutdown", "fcntl", "ioctl", "close"}
#: Calls that wait for data to ARRIVE on that descriptor.
_READ_CALLS = {"read", "readv", "recvfrom", "recvmsg", "recvmmsg"}

_GLOSS = {
    "futex": "waiting on another thread (normal when idle)",
    "epoll_wait": "waiting for network events (normal: the web server)",
    "epoll_pwait": "waiting for network events (normal: the web server)",
    "epoll_pwait2": "waiting for network events (normal: the web server)",
    "poll": "waiting for network or file events",
    "ppoll": "waiting for network or file events",
    "select": "waiting for network or file events",
    "pselect6": "waiting for network or file events",
    "nanosleep": "sleeping (a timer)",
    "clock_nanosleep": "sleeping (a timer)",
    "wait4": "waiting for a child process",
    "waitid": "waiting for a child process",
    "accept": "waiting for a new connection",
    "accept4": "waiting for a new connection",
    "connect": "opening a connection",
}

#: Settings in the app's environment that name the Prompter's address. Only
#: these are read out of /proc/<pid>/environ; nothing else is kept or shown.
_HOST_SETTINGS = (b"PROMPTER_URL", b"GEOTECH_FH_BASE_URL")

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[-_]?key[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]+)"),
    re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?"
               r"(?:bearer\s+|basic\s+|ntlm\s+)?)([^\s\"',;]+)"),
    re.compile(r"(?i)(password[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]+)"),
    re.compile(r"(?i)(secret[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]+)"),
    re.compile(r"()\b(sk-[A-Za-z0-9_-]{8,})"),
)


def _redact(line: str) -> str:
    """Hide anything in a log line that looks like a key or a password."""
    for pattern in _SECRET_PATTERNS:
        line = pattern.sub(lambda m: m.group(1) + "<hidden>", line)
    return line


def _arch(arch: Optional[str]) -> str:
    import platform
    name = (arch or platform.machine() or "").lower()
    if name in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if name in ("aarch64", "arm64"):
        return "aarch64"
    return name or "unknown"


def _prompter_hosts(root: str, pid: int) -> Optional[list]:
    """Host names of the Prompter address(es) in the app's environment, or
    ``None`` when the environment cannot be read. Only the two address
    settings are looked at; no other value (a key) is kept."""
    from urllib.parse import urlparse
    data = _read_bytes(os.path.join(root, str(pid), "environ"))
    if data is None:
        return None
    hosts = []
    for entry in data.split(b"\0"):
        name, sep, value = entry.partition(b"=")
        if sep and name in _HOST_SETTINGS:
            host = urlparse(value.decode("utf-8", "replace").strip()).hostname
            if host and host not in hosts:
                hosts.append(host)
    return hosts


def _default_resolve(host: str) -> set:
    import socket
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        return set()


def _default_ss() -> Optional[str]:
    import shutil
    exe = shutil.which("ss")
    if not exe:
        return None
    try:
        done = subprocess.run([exe, "-tnpo"], capture_output=True, text=True,
                              timeout=10)
        return done.stdout
    except Exception:                                      # noqa: BLE001
        return None


def _tail(path: Optional[str], lines: int) -> Optional[list]:
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            data = fh.read()
    except OSError:
        return None
    return [_redact(x) for x in
            data.decode("utf-8", "replace").splitlines()[-lines:]]


def _sock_text(sock) -> str:
    if sock.state == "LISTEN":
        return f"listening on {sock.local_ip}:{sock.local_port}"
    return (f"{sock.local_ip}:{sock.local_port} -> "
            f"{sock.remote_ip}:{sock.remote_port} {sock.state}")


def _diagnosis_text(pid: Optional[int] = None, *, lines: int = 30,
                    proc_root: Optional[str] = None,
                    tmp_dir: Optional[str] = None, arch: Optional[str] = None,
                    _run_ss: Optional[Callable[[], Optional[str]]] = None,
                    _resolve: Optional[Callable[[str], set]] = None,
                    now: Optional[float] = None) -> str:
    """The text :func:`diagnose` prints (separate so it can be tested)."""
    root = _linux_root(proc_root)
    if root is None:
        return _not_linux_note()
    apps = running_apps(proc_root=root, tmp_dir=tmp_dir)
    if pid is None:
        if not apps:
            return ("No copy of the app (GeotechStaffEngineer) is running on "
                    "this machine, so there is nothing to diagnose.")
        targets = list(apps)
    else:
        targets = [a for a in apps if a.pid == int(pid)]
        if not targets:
            if not os.path.isdir(os.path.join(root, str(pid))):
                return f"No process {pid} is running on this machine."
            argv = _cmdline(root, int(pid))
            targets = [RunningApp(pid=int(pid), port=None,
                                  started=_start_time(root, int(pid)),
                                  script_path="(not a copy of the app)",
                                  script_exists=False, log_path=None,
                                  python=argv[0] if argv else "?")]
    if _run_ss is None and root == "/proc":
        _run_ss = _default_ss
    resolve = _resolve or _default_resolve
    arch_name = _arch(arch)
    table = SYSCALLS.get(arch_name, {})
    ss_out = _run_ss() if _run_ss else None

    out = []
    for app in targets:
        p = app.pid
        out.append(f"=== pid {p}, port {app.port or 'unknown'}, started "
                   f"{_when(app.started, now)} ===")
        if app.script_path == "(not a copy of the app)":
            out.append("(This process is not a copy of the app; showing it "
                       "anyway.)")
        hosts = _prompter_hosts(root, p)
        prompter_ips: set = set()
        if hosts is None:
            out.append("Prompter host: the app's environment could not be "
                       "read.")
        elif not hosts:
            out.append("Prompter host: none set in the app's environment.")
        else:
            for host in hosts:
                ips = set(resolve(host) or ())
                prompter_ips |= ips
                out.append(f"Prompter host: {host} -> "
                           f"{', '.join(sorted(ips)) or 'could not resolve'}")

        fds = _fd_targets(root, p)
        socks = _net_sockets(root, p)
        conns = {}
        for fd, target in sorted(fds.items()):
            inode = _socket_inode(target)
            if inode in socks:
                conns[fd] = socks[inode]
        out.append(f"Network connections ({len(conns)}):")
        if not conns:
            out.append("  none found (or not readable)")
        for fd, sock in conns.items():
            tag = ("  [Prompter host]" if sock.remote_ip in prompter_ips
                   and sock.state != "LISTEN" else "")
            queues = (f"  (waiting to send {sock.tx_queue} B, "
                      f"unread {sock.rx_queue} B)"
                      if sock.tx_queue or sock.rx_queue else "")
            out.append(f"  fd {fd}: {_sock_text(sock)}{tag}{queues}")
        if ss_out:
            hits = [x.strip() for x in ss_out.splitlines()
                    if f"pid={p}," in x]
            out.append(f"ss -tnpo ({len(hits)} lines for this pid):")
            out.extend(f"  {x}" for x in hits)

        task_dir = os.path.join(root, str(p), "task")
        try:
            tids = sorted(int(t) for t in os.listdir(task_dir) if t.isdigit())
        except OSError:
            tids = []
        out.append(f"Threads ({len(tids)}) and the system call each waits in "
                   f"({arch_name}):")
        waiting_prompter, waiting_remote = [], []
        for tid in tids:
            tdir = os.path.join(task_dir, str(tid))
            comm = (_read_text(os.path.join(tdir, "comm")) or "?").strip()
            raw = _read_text(os.path.join(tdir, "syscall"))
            if raw is None:
                out.append(f"  tid {tid} {comm}: (not readable)")
                continue
            parts = raw.split()
            if not parts:
                out.append(f"  tid {tid} {comm}: (empty)")
                continue
            if parts[0] == "running":
                out.append(f"  tid {tid} {comm}: running (on a CPU)")
                continue
            try:
                nr = int(parts[0])
            except ValueError:
                out.append(f"  tid {tid} {comm}: {raw.strip()}")
                continue
            if nr < 0:
                out.append(f"  tid {tid} {comm}: not in a system call")
                continue
            name = table.get(nr, f"syscall {nr}")
            text = name
            mark = ""
            if name in _FD_CALLS and len(parts) > 1:
                try:
                    fd = int(parts[1], 16)
                except ValueError:
                    fd = None
                target = fds.get(fd) if fd is not None else None
                sock = socks.get(_socket_inode(target))
                if sock is not None:
                    text = f"{name}(fd {fd} -> socket {_sock_text(sock)})"
                    if name in _READ_CALLS and sock.state != "LISTEN":
                        if sock.remote_ip in prompter_ips:
                            mark = ("  <-- WAITING FOR THE PROMPTER HOST "
                                    "TO ANSWER")
                            waiting_prompter.append(tid)
                        else:
                            mark = ("  <-- waiting for an answer from "
                                    f"{sock.remote_ip}:{sock.remote_port}")
                            waiting_remote.append(tid)
                elif fd is not None:
                    text = f"{name}(fd {fd} -> {target or 'unknown'})"
            gloss = _GLOSS.get(name)
            out.append(f"  tid {tid} {comm}: {text}"
                       + (f"  - {gloss}" if gloss and not mark else "")
                       + mark)
        if waiting_prompter:
            out.append(
                f"FOUND: {len(waiting_prompter)} thread(s) waiting for the "
                f"Prompter host to answer (tid "
                f"{', '.join(map(str, waiting_prompter))}): a model request "
                f"is in flight. Run diagnose({p}) again in a minute; if the "
                f"same thread still waits on the same connection, that "
                f"request is stuck — stop_running({p}) and relaunch.")
        elif waiting_remote:
            out.append(f"{len(waiting_remote)} thread(s) waiting for an answer "
                       f"from another server (tid "
                       f"{', '.join(map(str, waiting_remote))}).")
        else:
            out.append("No thread is waiting on a model request.")

        tail = _tail(app.log_path, lines)
        if tail is None:
            out.append(f"Log: {app.log_path or 'unknown'} (not found)")
        else:
            out.append(f"Last {len(tail)} lines of {app.log_path}:")
            out.extend(f"  | {x}" for x in tail)
        out.append("")
    return "\n".join(out).rstrip()


def diagnose(pid: Optional[int] = None, *, lines: int = 30,
             proc_root: Optional[str] = None, tmp_dir: Optional[str] = None,
             arch: Optional[str] = None,
             _run_ss: Optional[Callable[[], Optional[str]]] = None,
             _resolve: Optional[Callable[[str], set]] = None) -> None:
    """Print what a copy of the app is doing (every copy when no ``pid``).

    For each: the Prompter host its settings name (read from its
    environment — only ``PROMPTER_URL`` / ``GEOTECH_FH_BASE_URL``, never a
    key), its network connections (from ``/proc/<pid>/net/tcp``, plus
    ``ss -tnpo`` where ``ss`` exists), every thread with the system call it
    waits in (``/proc/<pid>/task/*/syscall``, named for x86_64 and aarch64)
    — a thread in ``read``/``recvfrom`` on a connection to the Prompter host
    is marked, the sign of a model request in flight or stuck — and the last
    ``lines`` lines of its log, with anything that looks like a key hidden.
    Linux only; ``proc_root`` / ``tmp_dir`` / ``arch`` point it at a fake
    tree (tests)."""
    print(_diagnosis_text(pid, lines=lines, proc_root=proc_root,
                          tmp_dir=tmp_dir, arch=arch, _run_ss=_run_ss,
                          _resolve=_resolve))
