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
    handle = run_on_databricks(port=8501, model="funhouse-gpt-high")
    print(handle.url)      # open this in your browser
    # ... later ...
    handle.stop()

If the workspace host cannot be read from spark, pass ``workspace_host=...`` or
build the URL yourself from the printed ``base_path``.
"""

from __future__ import annotations

import os
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "run_on_databricks",
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
    """
    host = (workspace_host or "").strip().rstrip("/")
    if not host:
        return None
    if not host.startswith("http"):
        host = f"https://{host}"
    return f"{host}{base_path}/"


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

if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _register_prompter():
    """Reconstruct a PrompterAPI ON THE DRIVER (the notebook's live fh_prompter
    cannot cross the process boundary) and register it as the webapp engine.
    PrompterAPI self-configures from the driver's FunhouseConfig / environment."""
    from webapp.engine_config import register_model_builder
    from funhouse.services.prompter.prompter_api import PrompterAPI
    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    prompter = PrompterAPI(chat_model=MODEL)
    register_model_builder(
        lambda: PrompterChatModel(prompter=prompter, model=MODEL))


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

# flag_options keys are the CLI-flag form; streamlit maps "_" -> "." internally
# (server_port -> server.port). CORS + XSRF are disabled so the driver proxy can
# frame/serve the app; headless suppresses the browser-open attempt.
_FLAGS = {
    "server_port": PORT,
    "server_address": "0.0.0.0",
    "server_baseUrlPath": BASE_PATH,
    "server_enableCORS": False,
    "server_enableXsrfProtection": False,
    "server_headless": True,
    "browser_gatherUsageStats": False,
}
bootstrap.run(APP_PATH, False, [], _FLAGS)
''')


def render_bootstrap_script(
    *, app_path: str, repo_root: str, base: str, port: int, model: str,
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
    )


def build_launch_env(
    base_env: dict,
    anthropic_key: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> dict:
    """Build the subprocess environment: inherit ``base_env`` (so the driver's
    Funhouse config env reaches the bootstrap), optionally inject
    ``ANTHROPIC_API_KEY``, and prepend ``repo_root`` to ``PYTHONPATH``."""
    env = dict(base_env)
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key
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
    """

    process: Any
    base_path: str
    port: int
    model: str
    script_path: str
    url: Optional[str] = None
    workspace_host: Optional[str] = None

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
    port: int = DEFAULT_PORT,
    model: str = DEFAULT_MODEL,
    *,
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
    port : int
        Driver port to serve on (default ``8501``).
    model : str
        Prompter chat-model id (default ``"funhouse-gpt-high"``).
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
    org_id, cluster_id = resolve_cluster_ids(
        spark=spark, org_id=org_id, cluster_id=cluster_id)
    base = driver_proxy_base_path(org_id, cluster_id, port)

    app_path = os.path.abspath(app_path or _default_app_path())
    repo_root = os.path.dirname(os.path.dirname(app_path))

    script = render_bootstrap_script(
        app_path=app_path, repo_root=repo_root, base=base, port=port, model=model)
    fd, script_path = tempfile.mkstemp(
        prefix="geotech_streamlit_boot_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(script)

    env = build_launch_env(
        os.environ, anthropic_key=anthropic_key, repo_root=repo_root)
    python_exe = python_executable or sys.executable
    process = _popen([python_exe, script_path], env=env)

    if workspace_host is None:
        workspace_host = workspace_host_from_spark(spark or _active_spark())
    url = proxy_url(workspace_host, base)

    handle = LaunchHandle(
        process=process, base_path=base, port=port, model=model,
        script_path=script_path, url=url, workspace_host=workspace_host)

    if not quiet:
        print(f"[databricks_launcher] streamlit starting on port {port} "
              f"(model={model}).")
        if url:
            print(f"[databricks_launcher] Open: {url}")
        else:
            print(f"[databricks_launcher] base path: {base}")
            print("[databricks_launcher] Workspace host unknown — open "
                  f"https://<your-workspace-host>{base}/ (pass workspace_host=... "
                  "to have it printed for you).")
        print("[databricks_launcher] Call handle.stop() to shut it down.")

    return handle
