"""Offline tests of ``run_range_labels`` (BACKLOG RI-73): the guards, at $0.

What the paid run relies on before any call goes out: a blind report is
refused before anything about it is read, the estimate counts a picture as
a picture and not as its base64 text, and the reserve refuses a call whose
estimate -- with the calls still in flight -- would pass the wave's cap.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_run_range_labels.py -q
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage  # noqa: E402

from module_work.live_smoke.spend import WaveCapReached  # noqa: E402
from module_work.report_ingest_harness import run_range_labels as rrl  # noqa: E402


class _Ledger:
    def __init__(self, spent):
        self.spent = spent

    def wave_spent(self, wave):
        return self.spent


def test_a_blind_report_is_refused_before_anything_is_read():
    with pytest.raises(SystemExit, match="blind set"):
        rrl._report_ids("R18,R17")


def test_no_report_named_is_the_thirteen_stage_b_reports():
    assert rrl._report_ids("") == list(rrl.STAGE_B_REPORTS)
    assert len(rrl.STAGE_B_REPORTS) == 13


def test_a_picture_is_estimated_as_a_picture_not_as_its_base64():
    picture = {"type": "image_url",
               "image_url": {"url": "data:image/png;base64," + "A" * 300000}}
    words = {"type": "text", "text": "x" * 3000}
    tokens = rrl._estimate_tokens([[HumanMessage(content=[words, picture])]])
    assert tokens == 1000 + rrl.PICTURE_TOKENS


def _reserve(spent, cap, pending=None):
    return rrl._reserve_class()(_Ledger(spent), "made-up-wave", cap,
                                "claude-opus-5-5", 16000,
                                pending if pending is not None else {},
                                threading.Lock())


def test_the_reserve_refuses_a_call_that_would_pass_the_cap():
    # 30,000 tokens in at $4 a million and 16,000 out at $20: $0.44.
    messages = [[HumanMessage(content="x" * 90000)]]
    with pytest.raises(WaveCapReached, match="would pass"):
        _reserve(spent=5.80, cap=6.0).on_chat_model_start(
            {}, messages, run_id="a")
    _reserve(spent=5.50, cap=6.0).on_chat_model_start(
        {}, messages, run_id="b")


def test_the_reserve_counts_the_calls_still_in_flight():
    pending = {"other": 0.30}
    messages = [[HumanMessage(content="x" * 90000)]]
    with pytest.raises(WaveCapReached):
        _reserve(spent=5.50, cap=6.0, pending=pending).on_chat_model_start(
            {}, messages, run_id="c")
    reserve = _reserve(spent=5.00, cap=6.0, pending=pending)
    reserve.on_chat_model_start({}, messages, run_id="d")
    assert set(pending) == {"other", "d"}
    reserve.on_llm_end(None, run_id="d")
    assert set(pending) == {"other"}


def test_a_shared_ledger_written_from_many_threads_loses_no_call(tmp_path):
    from module_work.report_ingest_harness.run_claude_local import (
        shared_ledger)

    path = tmp_path / "ledger.json"
    ledger = shared_ledger(path)
    ledger.begin_wave("made-up-wave", 100.0, model="made-up", mode="test")

    def describe_pages():
        for _ in range(20):
            ledger.wave_spent("made-up-wave")       # what a cap check reads
            ledger.add("made-up-wave", 0.01, {"prompt": 10}, "made-up")
            assert ledger.cumulative >= 0.0

    threads = [threading.Thread(target=describe_pages) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wave = shared_ledger(path).wave("made-up-wave")
    assert wave["calls"] == 120
    assert wave["spent_usd"] == pytest.approx(1.20)


def test_the_shared_ledger_refuses_a_change_it_cannot_read_first(tmp_path):
    from module_work.report_ingest_harness.run_claude_local import \
        shared_ledger

    path = tmp_path / "ledger.json"
    ledger = shared_ledger(path)
    ledger.begin_wave("made-up-wave", 1.0, model="made-up", mode="test")
    before = path.read_text(encoding="utf-8")
    path.write_text("{ torn write", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not be read"):
        ledger.begin_wave("made-up-other", 1.0, model="made-up",
                          mode="test")
    assert path.read_text(encoding="utf-8") == "{ torn write"
    path.write_text(before, encoding="utf-8")
    ledger.begin_wave("made-up-other", 1.0, model="made-up", mode="test")


def test_the_table_has_the_columns_of_both_conventions_and_every_arm(tmp_path):
    import json

    from report_ingest.range_labels import (DESCRIBE_PROMPT_VERSION,
                                            conventions_named,
                                            labeller_fingerprint)

    def save(folder, arm, conventions):
        blob = {"id": "R01", "arm": arm, "describe_model": "made-up-eyes",
                "range_model": "made-up-reader",
                "describe_prompt": DESCRIBE_PROMPT_VERSION,
                "labeller_prompt": labeller_fingerprint(
                    arm == "hinted", *conventions_named(conventions)),
                "describe": {}}
        (tmp_path / "ranges" / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / "ranges" / folder / "R01.json").write_text(
            json.dumps(blob), encoding="utf-8")

    save("plain", "plain", "A")             # an earlier call, conventions A
    got = rrl._every_finished(tmp_path, ["R01"], "made-up-eyes",
                              "made-up-reader")
    assert got["arms"] == ["plain"]
    save("hinted_B", "hinted", "B")         # a later call, conventions B
    got = rrl._every_finished(tmp_path, ["R01"], "made-up-eyes",
                              "made-up-reader")
    assert got["arms"] == ["plain", "hinted_B"]
    assert got["runs"]["plain"] and got["runs"]["hinted_B"]
