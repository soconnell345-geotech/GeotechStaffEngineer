"""A spend line without the Funhouse SDK (BACKLOG TA-05), behind
GEOTECH_SPEND_ESTIMATE.

``budget_panel`` read only the Funhouse SDK's meter, so a tester on a host
without it (a key with a monthly budget) saw nothing of what their turns
cost. With the setting on, the sidebar shows the caller's own spend this
calendar month, estimated from the token counts in their own
conversations' records and priced by settings; nothing from another
tester's folder is read.
"""

from __future__ import annotations

import calendar
import json
import os

import pytest

from aec_document_review import budget_panel as bp

NOW = float(calendar.timegm((2026, 11, 13, 12, 0, 0)))
OCT = float(calendar.timegm((2026, 10, 30, 12, 0, 0)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (bp.SPEND_ESTIMATE_ENV, bp.SPEND_INPUT_PRICE_ENV,
                 bp.SPEND_OUTPUT_PRICE_ENV, bp.SPEND_BUDGET_ENV):
        monkeypatch.delenv(name, raising=False)


def _log(root, tid, *records):
    d = os.path.join(root, "conversations", tid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "activity.jsonl"), "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _turn_end(ts, inp=None, out=None, total=None):
    rec = {"turn": 1, "ts": ts, "agent": "primary", "event": "turn_end",
           "turn_tokens": total if total is not None else (inp or 0)
           + (out or 0)}
    if inp is not None:
        rec["usage"] = {"input_tokens": inp, "output_tokens": out or 0}
    return rec


def test_the_estimate_counts_this_months_turns_in_the_callers_own_folder(
        tmp_path, monkeypatch):
    me = str(tmp_path / "users" / "corp-j.doe" / "document_review")
    other = str(tmp_path / "users" / "corp-a.smith" / "document_review")
    _log(me, "T1", _turn_end(NOW - 3600, 800_000, 20_000),
         {"event": "model_end", "ts": NOW, "usage": {"input_tokens": 9}},
         _turn_end(OCT, 5_000_000, 5_000_000))          # last month
    _log(me, "T2", _turn_end(NOW - 60, total=200_000))  # no split kept
    _log(other, "T9", _turn_end(NOW, 9_000_000, 9_000_000))
    toks = bp.month_tokens(me, now=NOW)
    assert toks == {"input_tokens": 1_000_000, "output_tokens": 20_000,
                    "turns": 2}

    monkeypatch.setenv(bp.SPEND_INPUT_PRICE_ENV, "2.50")
    monkeypatch.setenv(bp.SPEND_OUTPUT_PRICE_ENV, "15")
    monkeypatch.setenv(bp.SPEND_BUDGET_ENV, "50")
    st = bp.estimate_status(me, now=NOW)
    assert st["spent"] == pytest.approx(2.50 + 0.30)
    assert st["cap"] == 50.0 and st["estimate"] is True
    line = bp.format_estimate_line(st)
    assert line.startswith("AI spend this month (estimate): $2.80 of $50.00")
    assert "1.0M input and 20k output tokens" in line


def test_without_prices_the_line_gives_tokens_and_no_money(tmp_path):
    root = str(tmp_path)
    _log(root, "T1", _turn_end(NOW, 1_500, 300))
    st = bp.estimate_status(root, now=NOW)
    assert st["spent"] is None and st["cap"] is None
    assert bp.format_estimate_line(st) == (
        "AI use this month (estimate): 2k input and 300 output tokens in "
        "your conversations; no token prices set")


def test_the_setting_is_off_unless_set(monkeypatch):
    assert bp.estimate_on() is False
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv(bp.SPEND_ESTIMATE_ENV, v)
        assert bp.estimate_on() is True
    monkeypatch.setenv(bp.SPEND_ESTIMATE_ENV, "0")
    assert bp.estimate_on() is False


def test_no_folder_or_a_bad_record_never_raises(tmp_path):
    assert bp.month_tokens(None)["turns"] == 0
    assert bp.month_tokens(str(tmp_path / "nowhere"))["turns"] == 0
    d = tmp_path / "conversations" / "T1"
    d.mkdir(parents=True)
    (d / "activity.jsonl").write_text('{"event": "turn_end", broken\n'
                                      + json.dumps(_turn_end(NOW, 10, 5))
                                      + "\n", encoding="utf-8")
    assert bp.month_tokens(str(tmp_path), now=NOW)["turns"] == 1


def test_the_sdk_line_is_unchanged():
    assert bp.format_line({"spent": 12.345, "cap": 50.0}) == (
        "AI budget: $12.35 of $50.00 used this month")
