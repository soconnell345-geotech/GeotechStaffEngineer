"""Instructions a deployment appends to every agent and helper (REF-64 (B)).

``GEOTECH_INSTRUCTIONS`` names a folder of UTF-8 text files: ``all.md``,
appended to every agent and helper, and one ``<role>.md`` per role, where
the role is the helper's name (``main`` for the main agent; ``references``,
``reviewer``, ``calc``, ``figure``, ``general-purpose``, ``model_setup``,
``report_library``, ``report_ingest`` today), so a new helper is covered by
adding its file. The folder is read when an agent is built, so an edit
reaches the next conversation without a restart.

The built-in rules win (owner, 2026-11-09; ``module_work/
REFERENCE_PACKS_PLAN.md`` section 3.3). The guards in code (file
confinement, the plotting box, per-tester keys, the step caps) are never
touched by text. Where a role has deployment text, the block appended last
to its prompt is: the protected integrity rules, then a fixed heading, then
the deployment's text (``all.md`` first, then the role's file), capped at
:data:`CAP_CHARS` and cut at a paragraph past it. With the setting unset,
or no text for a role, nothing is appended, so the prompts are
byte-identical to the shipped ones.

Nothing here fails a build: an empty or missing file appends nothing, and
an unreadable folder appends nothing and the record names the error.

The record (:func:`apply`'s second value, ``geotech_build["appended"]``):
``{"folder": <basename>, "roles": {role: {"files", "chars", "sha", "cut",
"text"}}}``, plus ``"error"`` when the folder could not be read. ``chars``
is the deployment text appended (after any cut), ``sha`` the sha256 of the
whole appended block, ``text`` its first :data:`RECORD_TEXT_CHARS`.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

#: The setting naming the folder.
ENV = "GEOTECH_INSTRUCTIONS"
#: The file appended to every role, first.
ALL_FILE = "all.md"
#: The main agent's role name.
MAIN_ROLE = "main"
#: Deployment text per role (all.md and the role's file together).
CAP_CHARS = 4000
#: How much of a role's appended text the build record keeps.
RECORD_TEXT_CHARS = 2000

#: The protected integrity rules, above every deployment block.
PROTECTED_RULES = (
    "Protected rules (built in; deployment instructions cannot change "
    "them):\n"
    "- Cite the printed page (document, edition, page, equation or table) "
    "for what you take from a reference.\n"
    "- Take formulas and defaults only from a printed source.\n"
    "- State your assumptions and the method used.\n"
    "- Say when a reading failed or is uncertain.\n"
    "- Never invent a value."
)

#: The fixed heading between the protected rules and the deployment text
#: (plan section 3.3, as decided).
HEADING = (
    "Instructions from this deployment. They may set conventions, "
    "preferred methods (cited from a reference this deployment provides), "
    "defaults, units and formats. They cannot switch off the protected "
    "rules above. Where one conflicts, follow the protected rule and say "
    "so in your answer."
)


def folder() -> Optional[str]:
    """The folder ``GEOTECH_INSTRUCTIONS`` names, or None when unset."""
    value = (os.environ.get(ENV) or "").strip()
    return value or None


def _cut(text: str, cap: int = CAP_CHARS) -> Tuple[str, bool]:
    """``text`` within ``cap`` characters, cut at the last paragraph break
    that fits (else the last line break, else at ``cap``)."""
    if len(text) <= cap:
        return text, False
    head = text[:cap]
    for sep in ("\n\n", "\n"):
        at = head.rfind(sep)
        if at > 0:
            return head[:at].rstrip(), True
    return head.rstrip(), True


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


class Instructions:
    """One read of the folder: the files' texts and what went wrong."""

    def __init__(self, path: Optional[str] = None):
        self.path = folder() if path is None else path
        self.texts: Dict[str, str] = {}
        self.error: Optional[str] = None
        self.file_errors: Dict[str, str] = {}
        if not self.path:
            return
        try:
            names = sorted(os.listdir(self.path))
        except Exception as exc:                       # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            return
        for name in names:
            if not name.endswith(".md"):
                continue
            full = os.path.join(self.path, name)
            if not os.path.isfile(full):
                continue
            try:
                text = _read(full)
            except Exception as exc:                   # noqa: BLE001
                self.file_errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            if text:
                self.texts[name] = text

    @property
    def active(self) -> bool:
        """The setting is set (the record is kept, even with no text)."""
        return bool(self.path)

    def block(self, role: str) -> Tuple[str, Optional[dict]]:
        """The block appended to ``role``'s prompt and its record entry;
        ``("", None)`` when the role has no deployment text."""
        files = [n for n in (ALL_FILE, f"{role}.md") if n in self.texts]
        if not files:
            return "", None
        text, cut = _cut("\n\n".join(self.texts[n] for n in files))
        block = "\n\n" + PROTECTED_RULES + "\n\n" + HEADING + "\n\n" + text
        entry = {"files": files, "chars": len(text),
                 "sha": hashlib.sha256(block.encode("utf-8")).hexdigest(),
                 "cut": cut, "text": text[:RECORD_TEXT_CHARS]}
        return block, entry

    def record(self, roles: Dict[str, dict]) -> dict:
        """The build record: ``{}`` when the setting is unset."""
        if not self.active:
            return {}
        rec: Dict[str, Any] = {
            "folder": os.path.basename(os.path.normpath(self.path)),
            "roles": dict(roles)}
        if self.error:
            rec["error"] = self.error
        if self.file_errors:
            rec["file_errors"] = dict(self.file_errors)
        return rec


def apply(system_prompt: str, subagents: List[Any],
          instructions: Optional[Instructions] = None
          ) -> Tuple[str, List[Any], dict]:
    """Append each role's block to the main prompt and to each helper
    spec's ``system_prompt`` (by its ``name``); a compiled helper
    (``"runnable"`` in its spec) has no prompt and is skipped (a builder
    hands it its block itself). Returns the prompt, the helper list (specs
    that changed are copies; the rest are the same objects) and the
    record (``{}`` when unset)."""
    ins = instructions if instructions is not None else Instructions()
    if not ins.active:
        return system_prompt, subagents, {}
    roles: Dict[str, dict] = {}
    block, entry = ins.block(MAIN_ROLE)
    if entry:
        system_prompt = (system_prompt or "") + block
        roles[MAIN_ROLE] = entry
    out: List[Any] = []
    for spec in subagents:
        if (isinstance(spec, dict) and "runnable" not in spec
                and spec.get("name")):
            role = str(spec["name"])
            block, entry = ins.block(role)
            if entry:
                spec = {**spec, "system_prompt": (
                    (spec.get("system_prompt") or "") + block)}
                roles[role] = entry
        out.append(spec)
    return system_prompt, out, ins.record(roles)


def block_for(role: str) -> Tuple[str, dict]:
    """One role's block and the record to merge, for a helper a builder
    hands its prompt itself (a compiled helper): ``("", {})`` when unset;
    the record ``{"folder", "roles": {role: entry}}`` otherwise (``roles``
    empty when the role has no text)."""
    ins = Instructions()
    if not ins.active:
        return "", {}
    block, entry = ins.block(role)
    return block, ins.record({role: entry} if entry else {})


def merge_record(agent: Any, record: dict) -> None:
    """Merge ``record``'s roles into ``agent.geotech_build["appended"]``
    (best effort; nothing when ``record`` is empty)."""
    if not record:
        return
    build = getattr(agent, "geotech_build", None)
    if not isinstance(build, dict):
        return
    current = build.get("appended")
    if not current:
        build["appended"] = {**record, "roles": dict(record.get("roles")
                                                     or {})}
        return
    current.setdefault("roles", {}).update(record.get("roles") or {})


class InstructedEngine:
    """An engine (``complete(messages, system=..., ...)``) whose ``system``
    gets ``block`` appended on every call; everything else is the inner
    engine's. ``_inner`` is the name the ingest's pacer follows through
    wrappers. ``appended`` is the role's record entry."""

    def __init__(self, inner: Any, block: str, appended: Optional[dict]):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_block", block)
        object.__setattr__(self, "appended", appended)

    def complete(self, messages, *, system: Optional[str] = None, **kwargs):
        return self._inner.complete(
            messages, system=(system or "") + self._block, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def instructed(engine: Any, block: str = "",
               appended: Optional[dict] = None) -> Any:
    """``engine`` in an :class:`InstructedEngine` carrying ``block``, or
    ``engine`` itself when there is no block (or no engine), for every
    route to a compiled helper's engine (its tool and its sub-agent)."""
    if engine is None or not block:
        return engine
    return InstructedEngine(engine, block, appended)


__all__ = ["ENV", "ALL_FILE", "MAIN_ROLE", "CAP_CHARS", "RECORD_TEXT_CHARS",
           "PROTECTED_RULES", "HEADING", "Instructions", "folder", "apply",
           "block_for", "merge_record", "InstructedEngine", "instructed"]
