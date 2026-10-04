"""The process-wide cap on vision calls in flight (``GEOTECH_VISION_MAX_INFLIGHT``)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage

from funhouse_agent.deep import vision_engine
from funhouse_agent.deep.vision_engine import LangChainVisionEngine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _CountingModel:
    """Records how many ``invoke`` calls overlap."""

    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def invoke(self, messages):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.03)
        with self.lock:
            self.now -= 1
        return AIMessage(content="ok")


def _fan_out(engine, n=24):
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(lambda _: engine.analyze_image(PNG, "x"), range(n)))


def test_cap_bounds_calls_in_flight(monkeypatch):
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "3")
    model = _CountingModel()
    out = _fan_out(LangChainVisionEngine(model, detail=""))
    assert out == ["ok"] * 24
    assert model.peak <= 3


def test_default_cap_is_eight(monkeypatch):
    monkeypatch.delenv(vision_engine.INFLIGHT_ENV, raising=False)
    model = _CountingModel()
    _fan_out(LangChainVisionEngine(model, detail=""))
    assert 1 <= model.peak <= vision_engine.DEFAULT_MAX_INFLIGHT == 8


def test_zero_means_no_cap(monkeypatch):
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "0")
    model = _CountingModel()
    _fan_out(LangChainVisionEngine(model, detail=""))
    assert model.peak > 8


def test_retry_without_detail_also_holds_a_slot(monkeypatch):
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "2")

    class _RefusesDetail(_CountingModel):
        def invoke(self, messages):
            block = messages[0].content[0]["image_url"]
            if "detail" in block:
                raise ValueError("unsupported detail")
            return super().invoke(messages)

    model = _RefusesDetail()
    out = _fan_out(LangChainVisionEngine(model, detail="original"), n=10)
    assert out == ["ok"] * 10
    assert model.peak <= 2
