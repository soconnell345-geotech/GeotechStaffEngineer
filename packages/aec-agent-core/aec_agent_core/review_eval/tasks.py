"""The review suite's task type and task selection (BACKLOG INF-32, the app
split's step 11 (c); plan ``module_work/APP_SPLIT_PLAN.md`` section 2.3: "the
``Task`` type and task selection from ``tasks.py``").

A task is a question as a reviewer would type it, the document(s) it is asked
of, and deterministic CHECKS on the answer (:mod:`.checks`). Each app brings
its own tasks: it fills the hook :data:`OPEN_TASKS_HOOK` with a callable
returning its shipped (open) tasks, and :func:`load_tasks` adds any private
task files (a blind set) to them. GeotechStaffEngineer's tasks, and where
their truth came from, are in ``funhouse_agent/review_eval/tasks.py``.

Checks every task gets unless it says otherwise (:data:`AUTO_CHECKS`): the
answer never gives up with "too blurry" / "send a higher-resolution file",
and never cites "page 0" (a tool's 0-based index leaking into a citation).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: The hook an app fills with its shipped tasks (``aec_agent_core.hooks``):
#: a callable with no arguments returning a list of :class:`Task`.
#: GeotechStaffEngineer fills it with
#: ``funhouse_agent.review_eval.tasks:open_tasks``. Unfilled, a run has only
#: the tasks it is given (``extra``).
OPEN_TASKS_HOOK = "review_eval.open_tasks"


#: Giving up on legibility: calling the file too blurry, or asking for a
#: better one. ("I zoomed in at higher resolution" is not giving up.)
GIVE_UP_PHRASES = (
    "too blurry",
    {"re": r"(?:send|provide|upload|share|supply|need|request|obtain|get)"
           r"\w*\b[^.\n]{0,60}\b(?:higher[- ]resolution|better(?:[- ]quality)?"
           r"|clearer|sharper|higher[- ]quality|more legible)\s+"
           r"(?:copy|scan|version|file|pdf|image|print)"},
)

#: Added to every task unless ``auto_checks=False``.
AUTO_CHECKS: List[Dict[str, Any]] = [
    {"type": "not_contains", "terms": list(GIVE_UP_PHRASES),
     "label": "does not give up on legibility"},
    {"type": "not_contains", "terms": [{"re": r"\b(?:page|p\.|pg\.?)\s*0\b"}],
     "label": "no 0-based page citation"},
]


@dataclass
class Task:
    """One question, the document(s) it is asked of, and its checks."""
    id: str
    question: str
    documents: List[str]
    category: str
    doc_type: str
    checks: List[Dict[str, Any]]
    truth: str = ""
    split: str = "open"
    followups: List[str] = field(default_factory=list)
    auto_checks: bool = True
    #: The app page the task is asked on (:data:`PAGES`).
    page: str = "review"

    def all_checks(self) -> List[Dict[str, Any]]:
        return list(self.checks) + (list(AUTO_CHECKS) if self.auto_checks
                                    else [])

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "question": self.question,
                "documents": list(self.documents), "category": self.category,
                "doc_type": self.doc_type, "checks": list(self.checks),
                "truth": self.truth, "split": self.split,
                "followups": list(self.followups),
                "auto_checks": self.auto_checks, "page": self.page}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(id=str(d["id"]), question=str(d["question"]),
                   documents=list(d.get("documents") or []),
                   category=str(d.get("category", "locate")),
                   doc_type=str(d.get("doc_type", "other")),
                   checks=list(d.get("checks") or []),
                   truth=str(d.get("truth", "")),
                   split=str(d.get("split", "open")),
                   followups=list(d.get("followups") or []),
                   auto_checks=bool(d.get("auto_checks", True)),
                   page=str(d.get("page", "review")))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def open_tasks() -> List[Task]:
    """The app's shipped tasks (:data:`OPEN_TASKS_HOOK`); none when no app
    hands them in."""
    from aec_agent_core.hooks import get_hook
    hook = get_hook(OPEN_TASKS_HOOK)
    return list(hook()) if hook is not None else []


def load_tasks(extra: Optional[Iterable[Any]] = None,
               include_open: bool = True,
               open: Optional[Sequence[Task]] = None) -> List[Task]:
    """The shipped open tasks plus any from ``extra``.

    The open tasks are ``open`` when given, else the app's
    (:func:`open_tasks`).

    ``extra`` is a list of paths to JSON files (each a list of task dicts, or
    ``{"tasks": [...]}``) and/or task dicts — how a private blind set on the
    owner's SharePoint is added. A later task with an id already seen
    replaces the earlier one.
    """
    out: Dict[str, Task] = {}
    if include_open:
        for t in (open_tasks() if open is None else open):
            out[t.id] = t
    for item in extra or ():
        for d in _read_extra(item):
            t = Task.from_dict(d)
            out[t.id] = t
    return list(out.values())


def _read_extra(item: Any) -> List[Dict[str, Any]]:
    if isinstance(item, dict):
        return [item]
    if isinstance(item, Task):
        return [item.to_dict()]
    path = os.fspath(item)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    return list(data)


def select(tasks: Sequence[Task], ids: Optional[Iterable[str]] = None,
           categories: Optional[Iterable[str]] = None,
           split: Optional[str] = None) -> List[Task]:
    """Filter by id prefix, category and split (``None`` keeps all)."""
    ids = [str(i) for i in (ids or [])]
    cats = set(categories or [])
    out = []
    for t in tasks:
        if ids and not any(t.id == i or t.id.startswith(i) for i in ids):
            continue
        if cats and t.category not in cats:
            continue
        if split and t.split != split:
            continue
        out.append(t)
    return out


__all__ = ["Task", "AUTO_CHECKS", "GIVE_UP_PHRASES", "OPEN_TASKS_HOOK",
           "open_tasks", "load_tasks", "select"]
