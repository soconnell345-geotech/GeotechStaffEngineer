"""The summary guard's ``install`` (BACKLOG TA-24): two first builds at the
same moment wrap the summarizer class once. Made-up classes only."""

import threading

from aec_agent_core.deep import summary_guard


def _racing_class(first_patch_started, let_first_go):
    """A summarizer-shaped class whose FIRST method patch stops half way,
    as a thread does when another one is scheduled: it says so and waits
    (a short while) for the other ``install`` to finish."""

    class Slow(type):
        def __setattr__(cls, name, value):
            if name == "wrap_model_call" \
                    and not first_patch_started.is_set():
                first_patch_started.set()
                let_first_go.wait(1.0)
            super().__setattr__(name, value)

    class Shaped(metaclass=Slow):
        def _should_summarize(self, messages, total_tokens):
            return False

        def _create_summary(self, messages):
            return "made-up summary"

        async def _acreate_summary(self, messages):
            return "made-up summary"

        def wrap_model_call(self, request, handler):
            return handler(request)

        async def awrap_model_call(self, request, handler):
            return await handler(request)

    return Shaped


def test_two_installs_at_one_moment_wrap_once():
    first_patch_started, let_first_go = threading.Event(), threading.Event()
    cls = _racing_class(first_patch_started, let_first_go)
    results = []

    def first():
        results.append(summary_guard.install(cls))

    def second():
        assert first_patch_started.wait(10)
        results.append(summary_guard.install(cls))
        let_first_go.set()

    threads = [threading.Thread(target=first),
               threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert results == [True, True]
    for name in ("wrap_model_call", "awrap_model_call", "_create_summary",
                 "_acreate_summary", "_should_summarize"):
        wrapped = cls.__dict__[name].__wrapped__     # guarded...
        assert not hasattr(wrapped, "__wrapped__")   # ...once
