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


# ---------------------------------------------------------------------------
# A library whose summarizer is not the shape the guard knows
# ---------------------------------------------------------------------------

import logging                                         # noqa: E402
import sys                                             # noqa: E402
import types                                           # noqa: E402

import pytest                                          # noqa: E402


@pytest.fixture
def warnings_of(monkeypatch, caplog):
    """The guard's warnings, in a process that has not warned yet."""
    monkeypatch.setattr(summary_guard, "_warned", False, raising=False)
    caplog.set_level(logging.WARNING, logger=summary_guard.__name__)
    return lambda: [r.getMessage() for r in caplog.records
                    if r.name == summary_guard.__name__
                    and r.levelno == logging.WARNING]


def _without(method):
    cls = _racing_class(threading.Event(), threading.Event())
    plain = type("MadeUpSummarizer", (), {
        name: value for name, value in cls.__dict__.items()
        if callable(value) and name != method})
    return plain


def test_a_missing_method_is_said_once(warnings_of):
    cls = _without("_should_summarize")
    assert summary_guard.install(cls) is False
    assert "__wrapped__" not in cls.__dict__["wrap_model_call"].__dict__
    (said,) = warnings_of()
    assert "_should_summarize" in said and "MadeUpSummarizer" in said
    assert "not installed" in said
    # once per process: a second build, or another class, adds nothing
    assert summary_guard.install(cls) is False
    assert summary_guard.install(_without("_create_summary")) is False
    assert len(warnings_of()) == 1


def test_a_moved_class_is_said_and_the_app_still_builds(monkeypatch,
                                                        warnings_of):
    """deepagents imports, its summarizer class is not where the guard
    knows it: one warning names it, and ``compile_agent`` builds."""
    from aec_agent_core.deep import assembly
    pytest.importorskip("deepagents")
    monkeypatch.setitem(sys.modules, "deepagents.middleware.summarization",
                        types.ModuleType("made_up_empty_module"))
    assert summary_guard.install() is False
    (said,) = warnings_of()
    assert "SummarizationMiddleware" in said
    built = assembly.compile_agent(
        lambda **kw: types.SimpleNamespace(nodes={}), model=None, tools=[],
        system_prompt="p", subagents=[], create_kwargs={})
    assert built is not None
    assert len(warnings_of()) == 1


def test_no_library_is_not_a_warning(monkeypatch, warnings_of):
    monkeypatch.setitem(sys.modules, "deepagents", None)   # import fails
    assert summary_guard.install() is False
    assert warnings_of() == []


def test_a_known_class_says_nothing(warnings_of):
    cls = _racing_class(threading.Event(), threading.Event())
    cls.__class__.__setattr__ = type.__setattr__
    assert summary_guard.install(cls) is True
    assert warnings_of() == []


def test_the_docstring_names_langchains_real_boundary():
    """langchain 1.3.14 and older hands back the error line; 1.3.15 and
    newer raises (the review read the wheels 1.3.4 to 1.3.18). The boundary
    is not "1.4"."""
    doc = " ".join(summary_guard.__doc__.split())
    assert "langchain 1.3.14 and older" in doc
    assert "langchain 1.3.15 and newer" in doc
    assert "before 1.4" not in doc
