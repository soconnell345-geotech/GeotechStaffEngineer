"""TA-39: a probe that failed on its caller's key leaves no verdict.

Deployment review of Document Review, 2026-11-10 (section 2.2): a failed
probe's fallback is held for ``FAILED_HOLD_S`` under the model's class,
name and address, not its key. On a shared host every tester's model object
carries a key of their own, so one tester's spent or refused key lowered
every tester's image size on that model for two minutes.

A fake patch model charges image tokens (32 px patches, 2,500 at ``high``,
10,000 at ``original``); each instance stands for one made-up tester's
model object on the same deployment, and refuses the way it is told to.
"""

from __future__ import annotations

import base64
import math
import struct
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import vision_probe, vision_view  # noqa: E402
from aec_agent_core.deep import vision_engine  # noqa: E402


class Unauthorized401(Exception):
    status_code = 401


class Forbidden403(Exception):
    status_code = 403


class BudgetSpent429(Exception):
    """An OpenAI-style gateway's spent budget: a 429 coded
    ``insufficient_quota``."""

    status_code = 429
    body = {"error": {"code": "insufficient_quota",
                      "message": "You exceeded your current quota."}}


class Busy409(Exception):
    status_code = 409


class Refused400(Exception):
    """The model refusing a ``detail`` value."""

    status_code = 400


def _png_size(data_url: str):
    raw = base64.b64decode(data_url.split(",", 1)[1][:64])
    return struct.unpack(">II", raw[16:24])


def _which(content) -> str:
    if isinstance(content, str):
        return "text"
    for block in content:
        if block.get("type") == "image_url":
            w, h = _png_size(block["image_url"]["url"])
            if w != h:
                return "wide"
            if block["image_url"].get("detail") == "original":
                return "original"
            return "small" if w < 2048 else "large"
    return "text"


class OneTestersModel:
    """One tester's model object on the shared deployment. ``error`` is
    raised on every call (``only`` narrows it to one probe call);
    ``gate``, when given, holds the calls until it is set."""

    def __init__(self, error=None, only=None, gate=None):
        self.error, self.only, self.gate = error, only, gate
        self.model_name = "made-up-vision-alias"
        self.openai_api_base = "https://models.example/v1"
        self.calls = []
        self.started = threading.Event()
        self._lock = threading.Lock()

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
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None and self.only in (None, call):
            raise self.error(f"made-up refusal on the {call} call")
        tokens = 12
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    w, h = _png_size(block["image_url"]["url"])
                    tokens += self._tokens(
                        w, h, block["image_url"].get("detail", "auto"))
        msg = AIMessage(content="OK", response_metadata={
            "model_name": "gpt-9-omni"})
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
    monkeypatch.setattr(vision_engine, "_sleep", lambda s: None)
    vision_probe.clear_cache()
    yield
    vision_probe.clear_cache()


REFUSALS = [(Unauthorized401, "sign-in refused"),
            (Forbidden403, "no access"),
            (BudgetSpent429, "budget used up")]


@pytest.mark.parametrize("error, why", REFUSALS)
def test_a_refused_key_leaves_no_verdict_for_the_next_caller(error, why):
    spent, fine = OneTestersModel(error=error), OneTestersModel()
    assert vision_probe._cache_key(spent) == vision_probe._cache_key(fine)
    mine = vision_probe.profile_for(spent)
    # the refused caller's own look is on the fallback, and says so
    assert mine.failed is True and mine.key_refused == why
    assert "could not be measured" in mine.fallback_note()
    assert mine.to_dict()["key_refused"] == why
    # nothing is held or cached for the model
    assert not vision_probe._FAILED and not vision_probe._CACHE
    assert vision_probe.cached_profile(fine) is None
    assert vision_probe.cached_profiles() == []
    # the next caller, with a key that works, measures the model at once
    theirs = vision_probe.profile_for(fine)
    assert fine.calls, "the next caller was handed the refused caller's verdict"
    assert theirs.failed is False and theirs.key_refused is None
    assert theirs.detailed == "openai-original"
    assert vision_probe._CACHE and not vision_probe._FAILED


def test_a_refused_key_does_not_displace_the_shared_measurement():
    fine, spent = OneTestersModel(), OneTestersModel(error=BudgetSpent429)
    measured = vision_probe.profile_for(fine)
    assert measured.failed is False
    # a successful probe stays shared: the refused caller is not even asked
    assert vision_probe.profile_for(spent) is measured
    assert spent.calls == []


def test_a_key_refused_on_the_original_call_is_not_the_model_refusing_it():
    """A budget that runs out between the probe's calls: before TA-39 the
    refusal of call 4 was read as the model refusing ``original`` and
    cached for every tester until a restart."""
    spent = OneTestersModel(error=BudgetSpent429, only="original")
    mine = vision_probe.profile_for(spent)
    assert mine.failed is True and mine.key_refused == "budget used up"
    assert "original_refused" not in mine.ratios
    assert not vision_probe._CACHE and not vision_probe._FAILED
    assert vision_probe.profile_for(OneTestersModel()).detailed == \
        "openai-original"


def test_a_failure_about_the_model_is_still_held(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vision_probe, "_now", lambda: clock[0])
    down, other = OneTestersModel(error=Busy409), OneTestersModel()
    first = vision_probe.profile_for(down)
    assert first.failed is True and first.key_refused is None
    assert first.to_dict()["key_refused"] is None
    assert vision_probe._FAILED
    # another tester on the same model gets the held fallback, unprobed
    clock[0] += vision_probe.FAILED_HOLD_S / 2
    assert vision_probe.profile_for(other) is first
    assert other.calls == []
    assert vision_probe.cached_profile(other) is first


def test_a_detail_the_model_refuses_is_still_a_shared_verdict():
    picky, other = OneTestersModel(error=Refused400, only="original"), \
        OneTestersModel()
    prof = vision_probe.profile_for(picky)
    assert prof.failed is False and prof.key_refused is None
    assert prof.ratios.get("original_refused") == 1.0
    assert vision_probe.profile_for(other) is prof and other.calls == []


def test_a_caller_waiting_on_a_refused_probe_measures_for_itself():
    """Two testers look at once; the first to probe has a spent key. The
    second waited for that probe: it must not go ahead unmeasured, nor on
    the first one's fallback."""
    gate = threading.Event()
    spent, fine = OneTestersModel(error=Unauthorized401, gate=gate), OneTestersModel()
    got = {}
    first = threading.Thread(
        target=lambda: got.update(spent=vision_probe.profile_for(spent)))
    first.start()
    assert spent.started.wait(10)                 # the probe is in flight
    second = threading.Thread(
        target=lambda: got.update(fine=vision_probe.profile_for(fine)))
    second.start()
    second.join(0.3)
    assert second.is_alive() and fine.calls == []      # waiting, not probing
    gate.set()
    first.join(20)
    second.join(20)
    assert not first.is_alive() and not second.is_alive()
    assert got["spent"].key_refused == "sign-in refused"
    assert got["fine"] is not None and got["fine"].failed is False
    assert got["fine"].detailed == "openai-original" and fine.calls
    assert vision_probe._CACHE and not vision_probe._FAILED


def test_key_refusal_reads_the_error_and_what_it_was_raised_from():
    assert vision_probe.key_refusal(Busy409("409 Conflict")) is None
    assert vision_probe.key_refusal(Refused400("Invalid 'detail'")) is None
    assert vision_probe.key_refusal(ValueError("no route to host")) is None
    try:
        try:
            raise Forbidden403("made-up: this key has no access")
        except Forbidden403 as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert vision_probe.key_refusal(outer) == "no access"


# ------------------------------------------------- review fixes (2026-11-13)

class _Engine:
    """What the vision tools hold: the caller's model, asked for its
    profile at every lookup of a look."""

    def __init__(self, model):
        self._model = model

    def vision_profile(self):
        return vision_probe.profile_for(self._model)


_VIEW = {"clip": [0, 0, 612, 792], "width_px": 1000, "height_px": 1294}


@pytest.mark.parametrize("error, why", REFUSALS)
def test_a_key_refused_on_the_host_edge_call_is_not_a_measured_profile(
        error, why):
    """Calls 1 to 4 pass and the caller's key is refused on call 5 (a budget
    that ran out after call 4). Before the review fix the profile was cached
    for the process as measured, with no host edge, and never measured
    again."""
    spent, fine = OneTestersModel(error=error, only="wide"), OneTestersModel()
    mine = vision_probe.profile_for(spent)
    assert "wide" in spent.calls
    assert mine.failed is True and mine.key_refused == why
    assert not vision_probe._CACHE and not vision_probe._FAILED
    theirs = vision_probe.profile_for(fine)
    assert theirs is not mine and "wide" in fine.calls
    assert theirs.failed is False and theirs.key_refused is None
    assert vision_probe._CACHE


def test_a_waiter_on_a_refused_probe_is_not_handed_an_old_model_failure(
        monkeypatch):
    """An earlier failure about the model is still in the table, its hold
    run out. A caller that waited on a refused probe must measure, not take
    that old fallback."""
    clock = [1000.0]
    monkeypatch.setattr(vision_probe, "_now", lambda: clock[0])
    old = vision_probe.profile_for(OneTestersModel(error=Busy409))
    assert old.failed is True and vision_probe._FAILED
    clock[0] += vision_probe.FAILED_HOLD_S + 1          # its hold has run out
    gate = threading.Event()
    spent, fine = OneTestersModel(error=Unauthorized401, gate=gate), \
        OneTestersModel()
    got = {}
    first = threading.Thread(
        target=lambda: got.update(spent=vision_probe.profile_for(spent)))
    first.start()
    assert spent.started.wait(10)
    second = threading.Thread(
        target=lambda: got.update(fine=vision_probe.profile_for(fine)))
    second.start()
    second.join(0.3)
    assert second.is_alive() and fine.calls == []      # waiting, not probing
    gate.set()
    first.join(20)
    second.join(20)
    assert not first.is_alive() and not second.is_alive()
    assert got["fine"] is not old and fine.calls
    assert got["fine"].failed is False
    assert got["fine"].detailed == "openai-original"


def test_one_refused_look_makes_one_probe_and_holds_it_for_itself_alone():
    """A look asks for the profile at several lookups. Before the review
    fix each lookup by a refused caller was a whole probe (7 probes, 28
    refused calls, for the three lookups below)."""
    spent, other_spent, fine = (OneTestersModel(error=Unauthorized401),
                                OneTestersModel(error=BudgetSpent429),
                                OneTestersModel())
    mine = vision_probe.profile_for(spent)
    assert mine.key_refused == "sign-in refused"
    eng = _Engine(spent)
    vision_view.budget(eng)
    vision_view.detail(eng)
    assert "could not be measured" in \
        vision_view.view_payload(dict(_VIEW), eng)["vision_probe"]
    assert vision_probe.profile_for(spent) is mine
    assert len(spent.calls) == 4                       # one probe in all
    # the verdict is held for that model object, never for the model
    assert not vision_probe._CACHE and not vision_probe._FAILED
    assert vision_probe.cached_profile(fine) is None
    # another refused tester is answered about their own key
    theirs = vision_probe.profile_for(other_spent)
    assert theirs is not mine and theirs.key_refused == "budget used up"
    assert len(other_spent.calls) == 4
    # and a tester whose key works measures the model at once
    good = vision_probe.profile_for(fine)
    assert fine.calls and good.failed is False
    assert good.detailed == "openai-original"


def test_a_refused_callers_own_waiting_calls_do_not_probe_again():
    """Tiles of one look in flight at once, on one model object."""
    gate = threading.Event()
    spent = OneTestersModel(error=Forbidden403, gate=gate)
    got = []
    first = threading.Thread(
        target=lambda: got.append(vision_probe.profile_for(spent)))
    first.start()
    assert spent.started.wait(10)
    others = [threading.Thread(
        target=lambda: got.append(vision_probe.profile_for(spent)))
        for _ in range(3)]
    for t in others:
        t.start()
    others[-1].join(0.3)
    gate.set()
    for t in [first] + others:
        t.join(20)
        assert not t.is_alive()
    assert len(got) == 4 and all(p is got[0] for p in got)
    assert got[0].key_refused == "no access"
    assert len(spent.calls) == 4


def test_a_refused_verdict_runs_out_and_the_key_is_tried_again(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vision_probe, "_now", lambda: clock[0])
    spent = OneTestersModel(error=BudgetSpent429)
    mine = vision_probe.profile_for(spent)
    clock[0] += vision_probe.REFUSED_HOLD_S / 2
    assert vision_probe.profile_for(spent) is mine and len(spent.calls) == 4
    # the budget is renewed; once the hold has run out the model is measured
    spent.error = None
    clock[0] += vision_probe.REFUSED_HOLD_S / 2 + 1
    again = vision_probe.profile_for(spent)
    assert again is not mine and again.failed is False
    assert len(spent.calls) > 4
    assert vision_probe._CACHE and not vision_probe._REFUSED


def test_refused_verdicts_do_not_pile_up(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(vision_probe, "_now", lambda: clock[0])
    monkeypatch.setattr(vision_probe, "REFUSED_MAX", 3)
    models = [OneTestersModel(error=Unauthorized401) for _ in range(5)]
    for m in models:
        vision_probe.profile_for(m)
        clock[0] += 1
    assert len(vision_probe._REFUSED) == 3
    assert id(models[-1]) in vision_probe._REFUSED
    assert id(models[0]) not in vision_probe._REFUSED
    # and every verdict leaves once its hold has run out
    clock[0] += vision_probe.REFUSED_HOLD_S
    assert vision_probe.profile_for(OneTestersModel()).failed is False
    assert not vision_probe._REFUSED
