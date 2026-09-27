"""The Document Review suite: a fixed, varied set of review tasks with
deterministic checks, run the way the Document Review page runs.

Every change to the Document Review harness is scored on the WHOLE suite, not
on the example that prompted it — the fix for the overfitting found in the
2026-09-26 review. See :mod:`funhouse_agent.review_eval.runner` for the
notebook cell, :mod:`~funhouse_agent.review_eval.tasks` for the tasks and
where their truth came from, and :mod:`funhouse_agent.review_flags` for the
arms being compared.
"""

from funhouse_agent.review_eval.documents import (
    DOCUMENTS, MissingDocument, collect_public_docs, resolve,
)
from funhouse_agent.review_eval.runner import (
    build_page_agent, run_task, score_review_suite, summarize,
)
from funhouse_agent.review_eval.tasks import (
    OPEN_TASKS, Task, load_tasks, select,
)

__all__ = ["score_review_suite", "run_task", "build_page_agent", "summarize",
           "OPEN_TASKS", "Task", "load_tasks", "select", "DOCUMENTS",
           "MissingDocument", "collect_public_docs", "resolve"]
