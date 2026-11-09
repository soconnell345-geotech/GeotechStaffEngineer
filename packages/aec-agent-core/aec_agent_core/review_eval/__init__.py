"""The review suite's framework: run an app page's turn on fixed tasks and
score the answers with deterministic checks (BACKLOG INF-32, the app split's
step 11 (c); plan ``module_work/APP_SPLIT_PLAN.md`` sections 2.3 and 2.7 B).

- :mod:`~aec_agent_core.review_eval.runner`: ``score_review_suite``,
  ``run_task``, ``summarize``; it drives a page's own turn path, which the
  app hands in (the ``review_eval.page_host`` hook);
- :mod:`~aec_agent_core.review_eval.checks`: the checks;
- :mod:`~aec_agent_core.review_eval.tasks`: the ``Task`` type and task
  selection (each app brings its own tasks);
- :mod:`~aec_agent_core.review_eval.documents`: where a task's documents are
  found, with the app's own sources handed in through hooks.

Nothing is imported here: the names load on first use.
"""

import importlib as _importlib

#: Each public name -> the module that defines it.
_LAZY = {
    "score_review_suite": "aec_agent_core.review_eval.runner",
    "run_task": "aec_agent_core.review_eval.runner",
    "build_page_agent": "aec_agent_core.review_eval.runner",
    "summarize": "aec_agent_core.review_eval.runner",
    "Task": "aec_agent_core.review_eval.tasks",
    "load_tasks": "aec_agent_core.review_eval.tasks",
    "select": "aec_agent_core.review_eval.tasks",
    "MissingDocument": "aec_agent_core.review_eval.documents",
    "resolve": "aec_agent_core.review_eval.documents",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module), name)
    globals()[name] = value
    return value


__all__ = sorted(_LAZY)
