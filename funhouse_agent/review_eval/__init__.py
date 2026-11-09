"""The Document Review suite: a fixed, varied set of review tasks with
deterministic checks, run the way the Document Review page runs.

Every change to the Document Review harness is scored on the WHOLE suite, not
on the example that prompted it — the fix for the overfitting found in the
2026-09-26 review. See :mod:`funhouse_agent.review_eval.runner` for the
notebook cell, :mod:`~funhouse_agent.review_eval.tasks` for the tasks and
where their truth came from, and :mod:`funhouse_agent.review_flags` for the
arms being compared.
"""

# The framework (runner, checks, documents, the Task type) is the shared
# core's since BACKLOG INF-32 step 11 (c) (aec_agent_core.review_eval); its
# old names under this package are aliases (funhouse_agent.CORE_ALIASES).
# This package keeps GSE's tasks, its report fixture and its tests, and hands
# the framework what is GSE's through hooks: the open tasks, the page the
# turns run on (webapp.suite_page), and where documents are found.
from aec_agent_core.hooks import set_hook as _set_hook

_set_hook("review_eval.open_tasks",
          "funhouse_agent.review_eval.tasks:open_tasks")
_set_hook("review_eval.page_host", "webapp.suite_page:host")
_set_hook("review_eval.document_dirs",
          "funhouse_agent.review_eval.host:document_dirs")
_set_hook("review_eval.fetch_document",
          "funhouse_agent.review_eval.host:fetch_document")
_set_hook("review_eval.fixture", "funhouse_agent.review_eval.host:fixture")

from funhouse_agent.review_eval.documents import (  # noqa: E402
    DOCUMENTS, MissingDocument, collect_public_docs, resolve,
)
from funhouse_agent.review_eval.runner import (  # noqa: E402
    build_page_agent, run_task, score_review_suite, summarize,
)
from funhouse_agent.review_eval.tasks import (  # noqa: E402
    OPEN_TASKS, Task, load_tasks, select,
)

__all__ = ["score_review_suite", "run_task", "build_page_agent", "summarize",
           "OPEN_TASKS", "Task", "load_tasks", "select", "DOCUMENTS",
           "MissingDocument", "collect_public_docs", "resolve"]
