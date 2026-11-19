"""Review of B15 (a failing summary leaves nothing in the history files).

Three things the builder's file does not hold:

* the summary remembered within one pass of the library's wrap belongs to
  the messages it was made from: a list changed in place between the two
  requests is summarized again (the fix of this review), and a second list
  is never answered with the first one's summary;
* the agent library's private parts the repair leans on are still there
  and still shaped as it needs: this fails by name when an update renames
  ``_offload_to_backend`` / ``_aoffload_to_backend`` or stops handing both
  steps the same messages;
* on a summarizer class without the two history writes the guard is still
  installed and a failing summary still lets the turn go on.

Made-up messages only; no model is called.
"""

import asyncio
import inspect

import pytest

from aec_agent_core.deep import summary_guard as sg


class _Stub:
    """A summarizer of the library's shape WITHOUT the two history writes;
    its summary names how many messages it was made from, or fails."""

    fail = False

    def __init__(self):
        self.calls = 0

    def _should_summarize(self, messages):
        return len(messages) > 2

    def _create_summary(self, messages):
        self.calls += 1
        if self.fail:
            raise RuntimeError("made-up refusal")
        return "summary of %d: %s" % (len(messages), messages[-1])

    async def _acreate_summary(self, messages):
        return self._create_summary(messages)

    def wrap_model_call(self, request, handler):
        if self._should_summarize(request):
            return handler([self._create_summary(request)])
        return handler(request)

    async def awrap_model_call(self, request, handler):
        if self._should_summarize(request):
            return await handler([await self._acreate_summary(request)])
        return await handler(request)


def _stub(fail=False):
    cls = type("Stub", (_Stub,), dict(vars(_Stub), fail=fail))
    assert sg.install(cls)
    return cls()


def _ask(stub, messages, run):
    if run == "sync":
        return stub._create_summary(messages)
    return stub._acreate_summary(messages)


async def _pass(stub, steps, run):
    """One guarded pass; ``steps`` yields lists to ask a summary of."""
    tokens, out = sg._on(stub), []
    try:
        for messages in steps:
            got = _ask(stub, messages, run)
            out.append(await got if run == "async" else got)
    finally:
        sg._off(tokens)
    return out


_RUNS = pytest.mark.parametrize("run", ["sync", "async"])


@_RUNS
def test_the_same_list_is_summarized_once_in_a_pass(run):
    stub, messages = _stub(), ["a", "b", "c"]
    out = asyncio.run(_pass(stub, [messages, messages], run))
    assert out == ["summary of 3: c"] * 2 and stub.calls == 1


@_RUNS
def test_a_list_changed_in_place_is_summarized_again(run):
    stub, messages = _stub(), ["a", "b", "c"]

    def steps():
        yield messages
        messages.append("d")                 # the same list, other messages
        yield messages
        messages[0] = "z"                    # the same length, another one
        yield messages

    out = asyncio.run(_pass(stub, steps(), run))
    assert out == ["summary of 3: c", "summary of 4: d", "summary of 4: d"]
    assert stub.calls == 3


@_RUNS
def test_a_second_list_is_never_handed_the_first_ones_summary(run):
    """The first list is held for the pass, so a list made after the first
    is dropped cannot take its id."""
    stub = _stub()

    def steps():
        for last in "cdefgh":
            yield ["a", "b", last]           # each freed by the caller

    out = asyncio.run(_pass(stub, steps(), run))
    assert out == ["summary of 3: %s" % last for last in "cdefgh"]
    assert stub.calls == 6


def test_the_librarys_history_writes_are_where_the_repair_needs_them():
    cls = sg._library_class()
    if cls is None:
        pytest.skip("deepagents is not installed")
    assert sg.install()
    pairs = (("wrap_model_call", "_offload_to_backend", "_create_summary"),
             ("awrap_model_call", "_aoffload_to_backend", "_acreate_summary"))
    for wrap, write, summary in pairs:
        assert callable(getattr(cls, write, None)), (
            f"deepagents has no {write}: a failed summary leaves its copy "
            "of the history again (summary_guard._OPTIONAL)")
        assert write in dict(sg._OPTIONAL)
        params = list(inspect.signature(
            inspect.unwrap(getattr(cls, write))).parameters)
        assert params[:3] == ["self", "backend", "messages"], params
        source = inspect.getsource(inspect.unwrap(getattr(cls, wrap)))
        (handed,) = {line.split(f"self.{write}(backend, ")[1].split(",")[0]
                     .rstrip(")") for line in source.splitlines()
                     if f"self.{write}(backend, " in line}
        assert f"self.{summary}({handed})" in source, (
            f"{wrap} no longer hands {write} and {summary} the same list")


@_RUNS
def test_without_the_history_writes_a_failed_summary_still_goes_on(
        run, monkeypatch):
    events = []
    monkeypatch.setattr(sg, "_event",
                        lambda name, data: events.append(name))
    stub, messages = _stub(fail=True), ["a", "b", "c"]
    assert not hasattr(stub, "_offload_to_backend")
    if run == "sync":
        out = stub.wrap_model_call(messages, lambda request: request)
    else:
        async def handler(request):
            return request
        out = asyncio.run(stub.awrap_model_call(messages, handler))
    assert out == messages                   # the history as it was
    assert events == [sg.EVENT] and stub.calls == 1
