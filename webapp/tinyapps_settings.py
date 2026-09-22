"""Where a Tiny Apps deployment's configuration and secrets come from.

This is the CfA ``settings.py`` pattern (their ``exampleCode`` repo, version
2026-08-28), reproduced here so the packaged app can read the SAME names their
engineers provision, from the SAME three places, in the SAME order:

1. a real environment variable — always wins (that is how App Service app
   settings arrive, and how the Databricks launcher hands values down);
2. on a Tiny Apps deployment (``APP_ENV=azure``, auto-detected from the
   ``IDENTITY_ENDPOINT`` variable App Service injects), the Key Vault named by
   the ``KV_NAME`` app setting, read with the app's managed identity;
3. locally (``APP_ENV=local``), a ``.env`` file — next to the wrapper repo's
   ``app.py`` when it names one via ``GEOTECH_DOTENV``, else the current
   working directory.

Names: Key Vault secrets use hyphens (``PROMPTER-API-KEY``); env vars and
``.env`` lines use underscores (``PROMPTER_API_KEY``). :func:`get_setting`
accepts either spelling and checks both, exactly as theirs does.

The names the engineers fill in for us (their ``.env.example``)::

    PROMPTER_URL   PROMPTER_MODEL   PROMPTER_API_KEY   PROMPTER_CA_BUNDLE
    GRAPH_TENANT_ID   GRAPH_CLIENT_ID   GRAPH_CLIENT_SECRET   SHAREPOINT_SITE_URL

Nothing here imports Azure libraries until a Key Vault read is actually
needed, so the module is importable in every environment the app runs in.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

#: The switch. ``azure`` = Key Vault via managed identity; ``local`` = ``.env``.
APP_ENV_VAR = "APP_ENV"
#: App Service injects this; its presence is how ``APP_ENV`` auto-detects.
IDENTITY_ENDPOINT_VAR = "IDENTITY_ENDPOINT"
#: App setting naming the Key Vault (the name, not the URL).
KV_NAME_VAR = "KV_NAME"
#: Optional path to the ``.env`` file for local runs.
DOTENV_PATH_VAR = "GEOTECH_DOTENV"
#: Azure Government Key Vault DNS suffix (CfA fixes the cloud to US Gov).
KEY_VAULT_SUFFIX = "vault.usgovcloudapi.net"

_cache: Dict[str, Optional[str]] = {}
_dotenv: Optional[Dict[str, str]] = None


def app_env() -> str:
    """``azure`` or ``local`` — explicit ``APP_ENV`` first, else auto-detect."""
    explicit = os.environ.get(APP_ENV_VAR, "").strip().lower()
    if explicit:
        return explicit
    return "azure" if os.environ.get(IDENTITY_ENDPOINT_VAR) else "local"


def dotenv_path() -> str:
    """The ``.env`` file consulted in local mode."""
    return os.environ.get(DOTENV_PATH_VAR, "").strip() or os.path.join(
        os.getcwd(), ".env")


def _load_dotenv() -> Dict[str, str]:
    global _dotenv
    if _dotenv is None:
        _dotenv = {}
        path = dotenv_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    _dotenv[key.strip().upper()] = (
                        value.strip().strip('"').strip("'"))
    return _dotenv


def _from_key_vault(secret_name: str) -> Optional[str]:
    """One secret from the vault named by ``KV_NAME``; None when it is not there.

    Managed identity inside App Service; ``DefaultAzureCredential`` elsewhere so
    an engineer's ``az login`` also works. Imports the Azure SDK lazily — it is
    a wrapper-repo dependency (``packages.txt``), not one of this package's.
    """
    vault = os.environ.get(KV_NAME_VAR, "").strip()
    if not vault:
        raise RuntimeError(
            f"{APP_ENV_VAR}=azure but the {KV_NAME_VAR} app setting is missing")
    try:
        from azure.core.exceptions import ResourceNotFoundError
        from azure.identity import (DefaultAzureCredential,
                                    ManagedIdentityCredential)
        from azure.keyvault.secrets import SecretClient
    except ImportError as exc:
        raise RuntimeError(
            "Key Vault reads need azure-identity and azure-keyvault-secrets "
            "(the wrapper repo's packages.txt carries them): "
            f"{exc}") from exc
    credential = (ManagedIdentityCredential()
                  if os.environ.get(IDENTITY_ENDPOINT_VAR)
                  else DefaultAzureCredential())
    client = SecretClient(vault_url=f"https://{vault}.{KEY_VAULT_SUFFIX}/",
                          credential=credential)
    try:
        return client.get_secret(secret_name).value
    except ResourceNotFoundError:
        return None


def get_setting(name: str, default: Optional[str] = None,
                required: bool = False) -> Optional[str]:
    """The one function everything uses.

    ``get_setting("PROMPTER-API-KEY")`` and ``get_setting("PROMPTER_API_KEY")``
    are the same lookup. Values are cached after the first read; call
    :func:`reset` to forget them (tests, or after a secret rotation).
    """
    env_name = name.replace("-", "_").upper()
    if env_name in _cache:
        value = _cache[env_name]
    else:
        value = os.environ.get(env_name)
        if value is None:
            if app_env() == "azure":
                value = _from_key_vault(env_name.replace("_", "-"))
            else:
                value = _load_dotenv().get(env_name)
        if value is not None:
            _cache[env_name] = value
    if value is None:
        value = default
    if value is None and required:
        where = ("Key Vault or the App Service app settings"
                 if app_env() == "azure" else f".env file ({dotenv_path()})")
        raise RuntimeError(
            f"Required setting {env_name} is not set — add it to your {where}.")
    return value


def reset() -> None:
    """Forget cached values and the parsed ``.env`` (tests, secret rotation)."""
    global _dotenv
    _cache.clear()
    _dotenv = None


__all__ = ["get_setting", "app_env", "dotenv_path", "reset",
           "APP_ENV_VAR", "KV_NAME_VAR", "IDENTITY_ENDPOINT_VAR",
           "DOTENV_PATH_VAR", "KEY_VAULT_SUFFIX"]
