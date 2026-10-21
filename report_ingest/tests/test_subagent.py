"""The ingest sub-agent's own ceiling (BACKLOG RI-02), offline.

deepagents runs a compiled sub-agent as provided, so a model-call budget put
on its spec ran nowhere. The ceiling is :class:`CallCeiling` now, counted
inside the graph; the wiring through the app's build is tested in
``funhouse_agent/deep/tests/test_report_ingest_offline.py``.
"""

from __future__ import annotations

import pytest

from report_ingest.subagent import (
    CallCeiling, CallCeilingReached, build_report_ingest_subagent,
)
from report_ingest.tests.fake_engine import FakeEngine


def _ask(engine):
    return engine.complete([{"role": "user", "content": "hello"}])


class TestTheCeiling:

    def test_two_engines_share_one_ceiling(self):
        reader = FakeEngine([{"text": "a"}, {"text": "b"}], name="reader")
        vision = FakeEngine([{"text": "c"}], name="vision")
        ceiling = CallCeiling(2)
        counted_reader, counted_vision = (ceiling.wrap(reader),
                                          ceiling.wrap(vision))

        assert _ask(counted_reader).text == "a"
        assert _ask(counted_vision).text == "c"
        with pytest.raises(CallCeilingReached, match="ceiling of 2"):
            _ask(counted_reader)
        # Refused before anything was sent.
        assert reader.n_calls == 1 and vision.n_calls == 1
        assert ceiling.used == 2 and ceiling.reached

    def test_the_wrapped_engine_is_the_engine_otherwise(self):
        engine = FakeEngine([{"text": "a"}], name="fake-model")
        counted = CallCeiling(5).wrap(engine)

        assert counted.name == "fake-model"
        assert counted.calls is engine.calls

    def test_no_engine_stays_no_engine(self):
        assert CallCeiling(1).wrap(None) is None

    def test_a_ceiling_below_one_is_refused(self):
        with pytest.raises(ValueError):
            CallCeiling(0)


class TestTheSpec:

    def test_it_carries_no_middleware(self):
        pytest.importorskip("langgraph")
        spec = build_report_ingest_subagent(lambda: None,
                                            max_model_calls=10)

        assert set(spec) == {"name", "description", "runnable"}

    def test_middleware_is_refused_not_ignored(self):
        with pytest.raises(ValueError, match="runs no middleware"):
            build_report_ingest_subagent(lambda: None,
                                         middleware=[object()])
