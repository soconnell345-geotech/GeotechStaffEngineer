"""Document Review's review suite: its 52 tasks, run through the shared
core's framework (``aec_agent_core.review_eval``) on this package's page.

Loading this package hands the framework what is this application's
(``aec_agent_core.hooks``):

- ``review_eval.open_tasks``: the tasks (:mod:`.tasks`);
- ``review_eval.page_host``: the page whose turn a task runs on
  (:mod:`aec_document_review.suite_page`);
- ``review_eval.fixture``: the coordination tasks' documents, one made-up
  building's drawings, manual, memorandum and submittal
  (:mod:`.coordination_fixture`).

``review_eval.document_dirs`` and ``review_eval.fetch_document`` stay
unfilled: the public documents are found in the suite's ``docs_dir`` or
``$GEOTECH_REVIEW_EVAL_DOCS``, and the other synthetic ones are built by
``planlens.testing`` (the core does both). The notebook cell::

    from aec_document_review.review_eval import score_review_suite
    score_review_suite(model, out_dir="/tmp/review_eval", docs_dir=...)

A dry run (``dry_run=True``) lists the tasks and checks their documents
without calling a model.

:func:`method_arm` is the review method (``aec_document_review.
review_method``, BACKLOG DR-06) as an arm, beside the core's named arms::

    score_review_suite(model, arms=["baseline", "show_me", method_arm()],
                       ids=["coord-"], out_dir=...)
"""

from aec_agent_core.hooks import set_hook as _set_hook

_set_hook("review_eval.open_tasks",
          "aec_document_review.review_eval.tasks:open_tasks")
_set_hook("review_eval.page_host", "aec_document_review.suite_page:host")
_set_hook("review_eval.fixture",
          "aec_document_review.review_eval.coordination_fixture:fixture")

from aec_agent_core.review_eval.documents import (  # noqa: E402
    DOCUMENTS, MissingDocument, collect_public_docs, resolve,
)
from aec_agent_core.review_eval.runner import (  # noqa: E402
    build_page_agent, run_task, score_review_suite, summarize,
)
from aec_document_review.review_eval.tasks import (  # noqa: E402
    OPEN_TASKS, Task, load_tasks, select,
)



def method_arm(name: str = "method", **more: str):
    """The review method as a suite arm: ``(name, environment)`` for
    ``score_review_suite(arms=[...])``. The arm names the method's folder in
    ``GEOTECH_INSTRUCTIONS`` for its runs only; the page is otherwise the
    default build, so the arm measures the method and nothing else.
    ``more`` adds settings to the same arm (``method_arm("method_show_me",
    GEOTECH_VISION_SHOW_ME="1")``). Do not set ``GEOTECH_INSTRUCTIONS`` in
    the notebook's own environment: no arm clears it, so every arm would
    then carry the method."""
    from aec_agent_core.deep.instructions import ENV
    from aec_document_review import review_method
    return name, {ENV: review_method.FOLDER, **more}


__all__ = ["score_review_suite", "run_task", "build_page_agent", "summarize",
           "OPEN_TASKS", "Task", "load_tasks", "select", "DOCUMENTS",
           "MissingDocument", "collect_public_docs", "resolve",
           "method_arm"]
