"""DR-36: a probe call the model answers busy is asked again, and a probe
that still fails is never cached, never read as a refusal, and said.

Chart suite EV-35 (2026-11-08, TRACE_REVIEW section 4.1): one 409 on a
probe call set the run's vision calls to detail ``high`` instead of
``original``, silently; in the app that profile would have been cached for
every conversation on the model until a restart. A fake chat model charges
image tokens like a patch model (32 px patches, 2,500 at ``high``, 10,000
at ``original``) and fails the calls it is told to.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")

from funhouse_agent import vision_probe, vision_view  # noqa: E402
from funhouse_agent.deep import vision_engine  # noqa: E402
from funhouse_agent.deep.vision_engine import (  # noqa: E402
    LangChainVisionEngine, current_side_call, side_call)


class Busy409(Exception):
    """What an SDK raises for a 409 Conflict."""

    status_code = 409


class Refused400(Exception):
    """A real refusal of the ``detail`` value: a 400 naming it."""

    status_code = 400


def _png_size(data_url: str):
    raw = base64.b64decode(data_url.split(",", 1)[1][:64])
    return struct.unpack(">II", raw[16:24])


def _which(content) -> str:
    """Which probe call this is: text, small, large, original or wide."""
    if isinstance(content, str):
        return "text"
    for block in content:
        if block.get("type") == "image_url":
            w, h = _png_size(block["image_url"]["url"])
            detail = block["image_url"].get("detail")
            if w != h:
                return "wide"
            if detail == "original":
                return "original"
            return "small" if w < 2048 else "large"
    return "text"


class FlakyModel:
    """A patch model that answers ``Busy409`` to a named call ``times``
    times (``times=None``: always), or refuses ``original`` outright."""

    def __init__(self, fail_call=None, times=1, error=Busy409,
                 answered="gpt-9-omni"):
        self.fail_call, self.times, self.error = fail_call, times, error
        self.answered = answered
        self.model_name = "made-up-vision-alias"
        self.calls = []
        self.labels = []
        self._lock = threading.Lock()
        self._failed = 0

    def _tokens(self, w, h, detail):
        cap = 10000 if detail == "original" else 2500
        patches = math.ceil(w / 32) * math.ceil(h / 32)
        if patches > cap:
            s = math.sqrt(cap / patches)
            patches = min(cap, math.ceil(w * s / 32) * math.ceil(h * s / 32))
        return int(patches * 1.2)

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        content = messages[0].content
        call = _which(content)
        with self._lock:
            self.calls.append(call)
            self.labels.append(current_side_call())
            fail = (call == self.fail_call
                    and (self.times is None or self._failed < self.times))
            if fail:
                self._failed += 1
        if fail:
            raise self.error(f"made-up error on the {call} call "
                             "(Invalid value for 'detail')"
                             if self.error is Refused400
                             else f"409 Conflict on the {call} call")
        tokens = 12
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    w, h = _png_size(block["image_url"]["url"])
                    tokens += self._tokens(
                        w, h, block["image_url"].get("detail", "auto"))
        msg = AIMessage(content="OK", response_metadata={
            "model_name": self.answered})
        msg.usage_metadata = {"input_tokens": tokens, "output_tokens": 1,
                              "total_tokens": tokens + 1}
        return msg


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV,
                vision_engine.BUSY_TRIES_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "1")
    waits = []
    monkeypatch.setattr(vision_engine, "_sleep", waits.append)
    vision_probe.clear_cache()
    yield waits
    vision_probe.clear_cache()


def _clean_profile():
    return vision_probe.probe(FlakyModel()).to_dict()


@pytest.mark.parametrize("call", ["text", "small", "large", "original"])
def test_a_409_on_each_probe_call_then_success_gives_the_full_profile(
        call, clean):
    model = FlakyModel(fail_call=call, times=1)
    prof = vision_probe.profile_for(model)
    want = _clean_profile()
    assert prof.failed is False and prof.error is None
    assert (prof.general, prof.detailed) == (want["general"],
                                             want["detailed"])
    assert prof.detailed == "openai-original"      # nothing lowered
    assert "original_refused" not in prof.ratios
    assert prof.image_tokens == want["image_tokens"]
    assert prof.retries == {call: 1}
    assert model.calls.count(call) == 2
    assert len(clean) == 1 and clean[0] > 0        # waited before asking again
    # A finished probe is cached for the process, as before.
    assert vision_probe._CACHE and not vision_probe._FAILED
    n = len(model.calls)
    assert vision_probe.profile_for(model) is prof
    assert len(model.calls) == n
    assert vision_view.probe_fallback_note(prof) is None


def test_the_retry_is_labelled_in_the_record():
    model = FlakyModel(fail_call="small", times=1)
    vision_probe.probe(model)
    labels = [lab for c, lab in zip(model.calls, model.labels)
              if c == "small"]
    assert labels[0] == {"kind": "probe", "probe_call": "small"}
    assert labels[1] == {"kind": "probe", "probe_call": "small",
                         "probe_try": 2}


def test_a_busy_host_edge_call_is_retried_too():
    # Call 5 (the wide image) runs where "original" is honoured past 2,048.
    model = FlakyModel(fail_call="wide", times=1)
    prof = vision_probe.probe(model)
    if "wide" not in model.calls:
        pytest.skip("this planlens gives no budget past 2,048 px")
    assert prof.failed is False and prof.retries == {"wide": 1}
    assert "wide_original" in prof.image_tokens


def test_a_persistent_409_on_original_is_a_failed_probe_not_a_refusal(clean):
    model = FlakyModel(fail_call="original", times=None)
    prof = vision_probe.profile_for(model)
    assert prof.failed is True
    assert "original_refused" not in prof.ratios   # never read as a refusal
    assert "409" in prof.error and "original" in prof.error
    assert model.calls.count("original") == vision_engine.busy_tries()
    assert prof.source in ("model name", "none")   # the fallback
    # Never cached for the process ...
    assert not vision_probe._CACHE
    assert "FAILED" in prof.summary()
    assert prof.to_dict()["failed"] is True
    # ... and stated.
    note = prof.fallback_note()
    assert note and "could not be measured" in note


@pytest.mark.parametrize("call", ["text", "small", "large"])
def test_a_persistent_failure_on_the_first_calls_gives_a_stated_fallback(
        call):
    model = FlakyModel(fail_call=call, times=None)
    prof = vision_probe.profile_for(model)
    assert prof.failed is True and prof.error
    assert not vision_probe._CACHE
    assert vision_probe.cached_profile(model) is prof   # for the run record
    engine = LangChainVisionEngine(model)
    assert vision_view.budget_name(engine) == vision_view.DEFAULT_BUDGET
    info = {"clip": [0, 0, 100, 100], "width_px": 800, "height_px": 800}
    payload = vision_view.view_payload(info, engine)
    assert "could not be measured" in payload["vision_probe"]
    json.dumps(payload)                                 # the record keeps it


def test_a_failed_probe_is_held_briefly_then_measured_again(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vision_probe, "_now", lambda: clock[0])
    model = FlakyModel(fail_call="large", times=vision_engine.busy_tries())
    first = vision_probe.profile_for(model)
    assert first.failed is True
    n = len(model.calls)
    # Within the hold: the same fallback, no new probe.
    clock[0] += vision_probe.FAILED_HOLD_S / 2
    assert vision_probe.profile_for(model) is first
    assert len(model.calls) == n
    # After it: measured again, and this time it finishes and is cached.
    clock[0] += vision_probe.FAILED_HOLD_S
    again = vision_probe.profile_for(model)
    assert again is not first and again.failed is False
    assert again.detailed == "openai-original"
    assert vision_probe._CACHE and not vision_probe._FAILED


def test_a_vision_call_on_a_failed_probe_says_so_in_its_record():
    model = FlakyModel(fail_call="text", times=None)
    engine = LangChainVisionEngine(model)
    png = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), 0)
    png.clear_with(255)
    with side_call(kind="image") as label:
        engine.analyze_image(png.tobytes("png"), "What is drawn here?")
    assert "could not be measured" in label["probe_fallback"]


def test_a_real_refusal_of_original_is_still_a_refusal_and_cached(clean):
    model = FlakyModel(fail_call="original", times=None, error=Refused400)
    prof = vision_probe.profile_for(model)
    assert prof.failed is False
    assert prof.ratios.get("original_refused") == 1.0
    assert model.calls.count("original") == 1       # not retried
    assert clean == []
    assert vision_probe._CACHE and not vision_probe._FAILED
