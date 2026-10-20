"""Which model each helper (sub-agent) runs on (BACKLOG INF-41).

The deep agent's helpers (the references librarian, the calculation,
figure and reviewer helpers, deepagents' general-purpose one) ran on the
main agent's model. On Foundry the owner wants the references helper on
Claude Sonnet 5.5 and the main agent and every other helper on Opus 5.5
(2026-10-20). So:

* ``build_deep_agent(subagent_models={name: chat model})`` gives a named
  helper a model of its own; a helper not named runs on the main model, as
  before (:mod:`funhouse_agent.deep.agent`).
* This module decides the map for the app. Each host has defaults
  (:data:`HOST_DEFAULTS`; only Foundry has any), ``GEOTECH_HELPER_MODELS``
  (``references=Sonnet 5.5,calc=main``) overrides them, and the sidebar
  overrides both for the session.
* A model is named by its label in the model list (``GEOTECH_FOUNDRY_MODELS``,
  e.g. ``Sonnet 5.5``) or by its id (a RID, with its family if the RID does
  not say it, or an API name); ``main`` is the main agent's model. A host
  default whose label is not in the list leaves the helper on the main model
  and says so.
* Each helper model is built by ``engine_config.resolve_engine``, exactly
  as the main model is: on Foundry the same proxy route, TLS and request-field
  sanitizer.

The main agent's model is unchanged: the sidebar's Model picker, whose
default is the first entry of the model list (put Opus 5.5 first).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from webapp import engine_config

#: ``helper=model`` pairs, comma-separated; a model is a label from the model
#: list, a model id, or ``main``.
ENV = "GEOTECH_HELPER_MODELS"

#: The setting that keeps a helper on the main agent's model.
MAIN = "main"

#: Each host's helper defaults, by deployment (``GEOTECH_DEPLOYMENT``).
#: Foundry: the owner's choice of 2026-10-20 (INF-41).
HOST_DEFAULTS: Dict[str, Dict[str, str]] = {
    "foundry": {"references": "Sonnet 5.5"},
}

#: Where a setting came from, as the sidebar and diagnostics say it.
SOURCE_HOST = "this host's default"
SOURCE_ENV = ENV
SOURCE_SIDEBAR = "the sidebar"

#: What the owner calls each helper (``build_deep_agent``'s helper names).
HELPER_LABELS = {
    "references": "References helper",
    "calc": "Calculation helper",
    "figure": "Figure helper",
    "reviewer": "Reviewer helper",
    "general-purpose": "General helper",
}


def helper_label(name: str) -> str:
    return HELPER_LABELS.get(name, f"{name} helper")


def deployment() -> str:
    """This host's name in :data:`HOST_DEFAULTS` (``""`` for none)."""
    if engine_config.is_foundry_deployment():
        return "foundry"
    if engine_config.is_tinyapps_deployment():
        return "tinyapps"
    return ""


def parse(text: Optional[str]) -> Dict[str, str]:
    """``"references=Sonnet 5.5, calc=main"`` -> ``{"references": "Sonnet
    5.5", "calc": "main"}``; an item without ``=`` is ignored."""
    out: Dict[str, str] = {}
    for item in (text or "").split(","):
        name, sep, model = item.partition("=")
        if sep and name.strip():
            out[name.strip()] = model.strip()
    return out


def settings(overrides: Optional[Dict[str, str]] = None
             ) -> Dict[str, Tuple[str, str]]:
    """``{helper: (setting, source)}``: this host's defaults, then the env,
    then ``overrides`` (the sidebar's picks)."""
    out: Dict[str, Tuple[str, str]] = {
        name: (model, SOURCE_HOST)
        for name, model in HOST_DEFAULTS.get(deployment(), {}).items()}
    for name, model in parse(os.environ.get(ENV)).items():
        out[name] = (model, SOURCE_ENV)
    for name, model in (overrides or {}).items():
        out[name] = (model, SOURCE_SIDEBAR)
    return out


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def lookup(setting: str) -> Tuple[Optional[str], str]:
    """``(model id, label)`` for a label or id in the model list, else
    ``(None, setting)``."""
    from webapp import core
    choices = core.model_choices()
    for c in choices:
        if c["id"] == setting:
            return c["id"], c["label"]
    for c in choices:
        if _key(c["label"]) == _key(setting):
            return c["id"], c["label"]
    return None, setting


@dataclass
class HelperModel:
    """One helper's model: ``model`` is ``None`` when it runs on the main
    agent's model (``note`` says why when that was not asked for)."""

    name: str
    setting: str
    source: str
    model_id: Optional[str] = None
    label: str = ""
    model: Any = None
    note: str = ""
    route: str = ""

    @property
    def own_model(self) -> bool:
        return self.model is not None

    def describe(self) -> str:
        head = f"{self.name}: "
        if self.own_model:
            body = f"{self.label} ({self.model_id}; {self.source})"
        elif self.label and self.setting.lower() != MAIN:
            body = f"{self.label}, the main agent's model ({self.source})"
        else:
            body = f"the main agent's model ({self.source})"
        return head + body + (f" -- {self.note}" if self.note else "")


def _resolve_kwargs(model_id: str, identity: Any) -> dict:
    kw: Dict[str, Any] = {"model_id": model_id}
    if identity is not None:
        kw["identity"] = identity
    return kw


def resolve(main_model_id: Optional[str],
            overrides: Optional[Dict[str, str]] = None,
            identity: Any = None) -> List[HelperModel]:
    """Each named helper's model for this host, built like the main one.
    Never raises; a helper whose model cannot be built runs on the main
    model with a note."""
    out: List[HelperModel] = []
    for name, (setting, source) in settings(overrides).items():
        h = HelperModel(name=name, setting=setting, source=source)
        out.append(h)
        if not setting or setting.lower() == MAIN:
            continue
        mid, label = lookup(setting)
        if mid is None:
            if source == SOURCE_HOST:
                h.note = (f"'{setting}' is not in the model list "
                          f"(GEOTECH_FOUNDRY_MODELS)")
                continue
            mid = setting
        h.model_id, h.label = mid, label
        if mid == main_model_id:
            continue
        try:
            res = engine_config.resolve_engine(**_resolve_kwargs(mid, identity))
        except Exception as exc:                       # noqa: BLE001
            h.note = f"{type(exc).__name__}: {exc}"
            continue
        if not res.ok:
            h.note = f"its model did not resolve: {res.message}"
            continue
        h.model = res.model
        h.route = getattr(res, "route", "") or ""
    return out


def build_kwargs(helpers: List[HelperModel]) -> dict:
    """``build_deep_agent`` arguments for these helpers (``{}`` when every
    helper runs on the main model, so the build is unchanged)."""
    own = {h.name: h.model for h in helpers if h.own_model}
    return {"subagent_models": own} if own else {}


def summary(main_label: str, helpers: List[HelperModel]) -> str:
    """One line naming the model each agent runs on."""
    parts = [f"main agent: {main_label}"]
    parts += [h.describe() for h in helpers]
    parts.append("every other helper: the main agent's model")
    return "; ".join(parts)


__all__ = ["ENV", "MAIN", "HOST_DEFAULTS", "HELPER_LABELS", "HelperModel",
           "helper_label", "deployment", "parse", "settings", "lookup",
           "resolve", "build_kwargs", "summary"]
