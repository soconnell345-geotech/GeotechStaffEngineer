"""The Prompter engine for a Tiny Apps deployment — a key, not the SDK.

On Databricks the app reaches Prompter through the Funhouse SDK, which
authenticates a Windows identity with NTLM. Tiny Apps hands each app its OWN
model deployment and an API KEY for it (CfA ``exampleCode/prompter.py``,
2026-08-28), and the SDK is not installable there. Under the SDK Prompter was
always a plain OpenAI-compatible endpoint (``OpenAI(base_url=..., api_key=...)``
in its source), so the same endpoint is driven here with ``langchain-openai``
— which the Foundry route in :mod:`webapp.engine_config` already uses — and
nothing else new.

The four settings, in CfA's names, read through
:mod:`webapp.tinyapps_settings` (env var, else Key Vault on the deployment,
else a local ``.env``)::

    PROMPTER_URL         the endpoint their engineers give, e.g.
                         https://prompter.<host>/api/v1/chat/completions
                         (the OpenAI client wants the part BEFORE
                         /chat/completions; :func:`base_url_from` strips it)
    PROMPTER_MODEL       the model deployment's name — the only model the
                         pilot key serves, and the default picker entry
    PROMPTER_API_KEY     that deployment's key. Their reference sends it as
                         BOTH ``api-key`` and ``Authorization: Bearer``; so
                         does this module (the OpenAI client sends Bearer,
                         ``default_headers`` adds ``api-key``)
    PROMPTER_CA_BUNDLE   optional: the internal CA certificate, as the PEM
                         text itself (how the Key Vault secret holds it) or a
                         path to a ``.pem`` — Prompter's TLS certificate is
                         signed by the Department's own authority, so
                         without this the connection is refused

Behaviour knobs (env): ``GEOTECH_PROMPTER_DISABLE_STREAMING=1`` makes
``.stream()`` fall back to one non-streaming request, for a gateway that
rejects streaming; ``GEOTECH_WEBAPP_MAX_TOKENS`` is honoured as everywhere
else. :func:`register` installs the model builder the app already looks for
(:func:`webapp.engine_config.register_model_builder`) and publishes the
deployment's model to the sidebar picker, so nothing downstream knows which
host it is on.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from typing import Optional

from webapp.tinyapps_settings import get_setting

ENV_URL = "PROMPTER_URL"
ENV_MODEL = "PROMPTER_MODEL"
ENV_KEY = "PROMPTER_API_KEY"
ENV_CA_BUNDLE = "PROMPTER_CA_BUNDLE"
DISABLE_STREAMING_ENV = "GEOTECH_PROMPTER_DISABLE_STREAMING"

#: CfA's module allows 180 s per request; the proxy in front of the app times
#: HTTP out near 120 s, but the app's turns run detached from the request.
REQUEST_TIMEOUT_S = 180.0
_CHAT_SUFFIX = "/chat/completions"


@dataclass(frozen=True)
class PrompterSettings:
    url: str
    model: str
    api_key: str
    ca_bundle: Optional[str] = None

    @property
    def base_url(self) -> str:
        return base_url_from(self.url)


def base_url_from(url: str) -> str:
    """``https://h/api/v1/chat/completions?x=y`` -> ``https://h/api/v1``.

    Their engineers give the full chat-completions URL; the OpenAI client
    appends ``/chat/completions`` itself. A URL that already stops at the
    base is returned unchanged (less a trailing slash).
    """
    u = (url or "").strip().split("?", 1)[0].rstrip("/")
    if u.lower().endswith(_CHAT_SUFFIX):
        u = u[: -len(_CHAT_SUFFIX)]
    return u.rstrip("/")


def ssl_context(bundle: Optional[str]) -> Optional[ssl.SSLContext]:
    """Trust the internal CA named by the bundle — PEM text or a file path.

    None means "default trust" (no bundle given). The partial-chain flag lets
    the internal CA act as a trust anchor although it is not a self-signed
    root, which is what browsers do too (their ``_ssl_context``, kept).
    """
    b = (bundle or "").strip()
    if not b:
        return None
    if b.startswith("-----BEGIN"):
        ctx = ssl.create_default_context(cadata=b)
    elif os.path.exists(b):
        ctx = ssl.create_default_context(cafile=b)
    else:
        raise ValueError(
            f"{ENV_CA_BUNDLE} is neither PEM text nor a path that exists")
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


def settings() -> Optional[PrompterSettings]:
    """The four values, or None when the deployment has not been given them.

    Missing values are not an error here: the app must boot and show its
    "no engine configured" banner rather than crash on a reviewer's first
    click. A CA bundle that is set but unusable IS raised, because a wrong
    bundle is a misconfiguration to fix, not an absence to tolerate.
    """
    url = (get_setting(ENV_URL) or "").strip()
    model = (get_setting(ENV_MODEL) or "").strip()
    key = (get_setting(ENV_KEY) or "").strip()
    if not (url and model and key):
        return None
    return PrompterSettings(url=url, model=model, api_key=key,
                            ca_bundle=(get_setting(ENV_CA_BUNDLE) or "").strip()
                            or None)


def configured() -> bool:
    return settings() is not None


def _streaming_disabled() -> bool:
    return str(os.environ.get(DISABLE_STREAMING_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


def build_chat_model(model_id: Optional[str] = None, *,
                     prompter: Optional[PrompterSettings] = None):
    """A ``ChatOpenAI`` against the deployment's Prompter endpoint.

    ``model_id`` is the picker's choice; on the pilot's one-model key it can
    only ever be the deployment's own name, but the parameter keeps the
    builder shape the picker expects. Sends ``max_completion_tokens`` rather
    than ``max_tokens`` for the same reason the Foundry route does: the
    reasoning-model tiers reject the old name and a deployment name gives
    langchain-openai no hint.
    """
    import httpx
    from langchain_openai import ChatOpenAI
    from webapp.engine_config import _default_max_tokens

    ps = prompter or settings()
    if ps is None:
        raise RuntimeError(
            f"Prompter is not configured: set {ENV_URL}, {ENV_MODEL} and "
            f"{ENV_KEY} (Key Vault secrets / app settings on the deployment, "
            "a .env file locally).")
    ctx = ssl_context(ps.ca_bundle)
    client = httpx.Client(verify=ctx if ctx is not None else True,
                          timeout=REQUEST_TIMEOUT_S)
    return ChatOpenAI(
        model=model_id or ps.model,
        api_key=ps.api_key,
        base_url=ps.base_url,
        default_headers={"api-key": ps.api_key},
        http_client=client,
        max_completion_tokens=_default_max_tokens(),
        disable_streaming=_streaming_disabled(),
    )


def register() -> bool:
    """Install the Prompter builder when the settings are present.

    Returns True when registered. Also publishes the deployment's model to the
    sidebar picker (``GEOTECH_PROMPTER_MODELS``) unless the deployment set its
    own list, and records the choice as the default model.
    """
    ps = settings()
    if ps is None:
        return False
    from webapp.engine_config import register_model_builder

    def _build(model_id: Optional[str] = None):
        return build_chat_model(model_id, prompter=ps)

    register_model_builder(_build)
    os.environ.setdefault("GEOTECH_PROMPTER_MODELS", f"{ps.model}={ps.model}")
    os.environ.setdefault("GEOTECH_WEBAPP_MODEL", ps.model)
    return True


__all__ = ["PrompterSettings", "settings", "configured", "build_chat_model",
           "register", "base_url_from", "ssl_context",
           "ENV_URL", "ENV_MODEL", "ENV_KEY", "ENV_CA_BUNDLE",
           "DISABLE_STREAMING_ENV", "REQUEST_TIMEOUT_S"]
