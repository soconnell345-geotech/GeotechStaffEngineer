"""What one page of the app is: its title, its agent and its hooks.

The shell (``aec_document_review/app.py``, ``aec_document_review/core.py``, ``aec_document_review/diagnostics.py``)
asks the page, not the geotechnical toolkit (INF-31, the app split's step 7):
the disclaimer it shows, the agent it builds and what is wired in when it is
built, the check run at each turn's end, the page's own diagnostics and the
distributions named under the sidebar. Each page fills these in its own
file (``aec_document_review/geotech_page.py``, ``aec_document_review/review_page.py``), the seam the
copy of the page follows at step 13.

A hook is a callable or ``"module:name"``, imported the first time it is
used, so a page's profile can name its own file's functions without the
shell (or the other page) importing that file when it loads.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

#: A hook: a callable, or "module:name" imported on first use.
Hook = Union[None, str, Callable[..., Any]]


def resolve(hook: Hook) -> Optional[Callable[..., Any]]:
    """The callable ``hook`` names (None for no hook)."""
    if hook is None or callable(hook):
        return hook
    module, _, name = str(hook).partition(":")
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class AppProfile:
    """One page of the app.

    ``build_kwargs`` are handed to :func:`aec_document_review.core.build_agent` and on to
    the page's agent builder: the geotech profile passes none and gets the
    agent it always had; the document-review profile passes an empty module
    scope, no reference sub-agent and its own prompt.

    The hooks (each a callable or ``"module:name"``, see :func:`resolve`):

    * ``disclaimer``: ``() -> str``, the professional-use notice the page
      shows (:func:`aec_document_review.core.disclaimer_text`);
    * ``placeholder``: the chat box's prompt (plain text, not a hook);
    * ``sidebar_extras``: ``(session_state, rebuild) -> None``, the page's
      own sidebar controls drawn under the version line (the geotech page's
      Agent picker); ``rebuild(model)`` rebuilds the agent;
    * ``behavior_kwargs``: ``(behavior) -> dict``, the Behavior settings as
      build arguments (default :func:`aec_document_review.core.behavior_build_kwargs`);
    * ``agent_builder``: ``(model, *, attachments, save_fn, temp_dir, **kw)
      -> agent``, the page's agent, after the shell's shared wiring;
    * ``specialist_builder``: ``(kind, model, attachments, temp_dir,
      artifacts, **kw) -> agent``, a narrow specialist from the Agent picker
      (the geotech page's);
    * ``on_build``: ``() -> None``, run before each build (the geotech
      page's reference-PDF fetcher);
    * ``turn_checks``: ``(agent, callbacks) -> check or None``, the check of
      the answer at each turn's end (the judgment-call check);
    * ``diagnostics_checks``: ``() -> [check]``, the page's own lines in the
      connection diagnostics;
    * ``version_dists``: the distributions whose versions the sidebar names.
    """
    name: str
    title: str
    icon: str
    caption: str
    #: The path segment in the browser (``/geotech``). Empty for the DEFAULT
    #: page, which Streamlit serves at the app root and nowhere else.
    url_path: str
    #: Whether the sidebar "Agent" picker (reviewer specialists) applies.
    specialists: bool = True
    #: Whether a fresh upload triggers an automatic orientation turn.
    orientation: bool = False
    orientation_request: str = ""
    #: Whether the sidebar offers "Download this conversation" (a zip of
    #: its folder; ``aec_document_review.conversation_export``, INF-38).
    conversation_download: bool = False
    _build_kwargs: Dict[str, object] = field(default_factory=dict)
    disclaimer: Hook = None
    placeholder: str = ""
    sidebar_extras: Hook = None
    behavior_kwargs: Hook = None
    agent_builder: Hook = None
    specialist_builder: Hook = None
    on_build: Hook = None
    turn_checks: Hook = None
    diagnostics_checks: Hook = None
    version_dists: Tuple[str, ...] = ()
    #: ``(kwargs) -> kwargs``: the page's own finishing of its build
    #: arguments (the review page builds its prompt here, so the memory flag
    #: is honoured).
    finish_build_kwargs: Hook = None

    def build_kwargs(self) -> Dict[str, object]:
        """The agent-build overrides for this page (a fresh dict)."""
        kw = dict(self._build_kwargs)
        finish = resolve(self.finish_build_kwargs)
        return finish(kw) if finish is not None else kw

    def hook(self, name: str) -> Optional[Callable[..., Any]]:
        """The page's ``name`` hook as a callable, or None."""
        return resolve(getattr(self, name))

    def behavior_build_kwargs(self, behavior: Optional[dict]) -> dict:
        """The Behavior settings as build arguments, the page's way."""
        fn = self.hook("behavior_kwargs")
        if fn is None:
            from aec_document_review import core
            fn = core.behavior_build_kwargs
        return fn(behavior)


__all__ = ["AppProfile", "Hook", "resolve"]
