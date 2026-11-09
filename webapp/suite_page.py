"""This app's two pages, as the review suite's page host (BACKLOG INF-32,
the app split's step 11 (c); plan ``module_work/APP_SPLIT_PLAN.md`` 2.7 B).

The suite's runner moved into the shared core
(``aec_agent_core.review_eval.runner``), which imports neither app. It asks
for a page host through the hook ``review_eval.page_host``;
``funhouse_agent/review_eval/__init__.py`` fills it with :func:`host`, which
returns this module. Each function below is what the runner did itself
before the move, through ``webapp.core``, ``webapp.profiles`` and
``webapp.output_capture`` -- looked up on those modules at call time, so a
test that patches ``webapp.core.stream_turn`` still reaches the run.

``page`` is a task's ``Task.page``: ``"review"`` (the Document Review page)
or ``"geotech"`` (the GeotechStaffEngineer page).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Optional

#: What run.json names as the app (the runner's ``versions`` and ``commits``).
APP_DIST = "geotech-staff-engineer"
APP_PACKAGE = "funhouse_agent"
APP_COMMIT_ENV = "GEOTECH_APP_COMMIT"


def host():
    """The page host: this module (the hook's target)."""
    return sys.modules[__name__]


def _profile(page: str = "review"):
    from webapp.profiles import DOCUMENT_REVIEW, GEOTECH
    return GEOTECH if page == "geotech" else DOCUMENT_REVIEW


def build_agent(model, attachments: Dict[str, bytes], files_dir: str,
                artifacts: List[str], *, page: str = "review",
                markup_author: Optional[str] = None):
    """A page's agent, built the way ``webapp/app.py`` builds it (behaviour
    defaults, then the page profile's overrides)."""
    from webapp import core
    profile = _profile(page)
    kw = profile.behavior_build_kwargs(None)
    kw.update(profile.build_kwargs())
    if markup_author:
        kw.setdefault("markup_author", markup_author)
    return core.build_agent(model, attachments, files_dir, artifacts,
                            profile=profile, **kw)


def stage_upload(attachments: dict, files_dir: str, name, data: bytes):
    from webapp import core
    return core.stage_upload(attachments, files_dir, name, data)


def attachment_note(atts) -> str:
    from webapp import core
    return core.attachment_note(atts)


def assemble_user_message(notes: Iterable[str], text: str) -> str:
    from webapp import core
    return core.assemble_user_message(notes, text)


def orientation_request(atts, page: str = "review") -> Optional[str]:
    """The Document Review page's automatic orientation turn; the geotech
    page sends none."""
    if page != "review":
        return None
    from webapp.profiles import DOCUMENT_REVIEW, orientation_request_for
    return orientation_request_for(DOCUMENT_REVIEW, atts)


def stream_turn(agent, history: list, thread: str, *,
                recursion_limit: Optional[int] = None,
                callbacks: Optional[list] = None):
    from webapp import core
    return core.stream_turn(agent, history, thread,
                            recursion_limit=recursion_limit,
                            callbacks=callbacks)


def default_recursion_limit() -> int:
    from webapp import core
    return core.DEFAULT_BEHAVIOR["recursion_limit"]


def engine_record(agent) -> dict:
    from webapp import core
    return core.engine_record(agent)


def tools_fingerprint(agent) -> dict:
    from webapp import core
    return core.tools_fingerprint(agent)


def output_collector():
    from webapp.output_capture import OutputCollector
    return OutputCollector()


def import_reported_outputs(outputs, files_dir: str,
                            exclude: Iterable[str] = ()) -> Dict[str, Any]:
    from webapp import core
    return core.import_reported_outputs(outputs, files_dir,
                                        exclude=list(exclude))
