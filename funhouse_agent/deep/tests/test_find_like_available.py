"""_find_like_available: find_like is offered wherever planlens can search.

On a FIPS host (Foundry 2026-10-03, Funhouse 2026-10-07) loading OpenCV
aborts the interpreter. planlens after 0.11.0 answers for itself
(``planlens.document.findlike.available``: OpenCV where it loads, a numpy
matcher anywhere else), so find_like stays on those hosts; planlens 0.11 has
only OpenCV, so there the answer is whether OpenCV loads; before 0.11 there
is nothing to ask.
"""

import sys

import pytest

from funhouse_agent.deep.tools import _find_like_available

findlike = pytest.importorskip("planlens.document.findlike")
opencv = pytest.importorskip("planlens.opencv")


def _needs_planlens_answer():
    if not hasattr(findlike, "available"):
        pytest.skip("the installed planlens predates findlike.available")


def test_planlens_answers_for_itself(monkeypatch):
    _needs_planlens_answer()
    # Whether OpenCV loads is planlens' business now, not the app's.
    monkeypatch.setattr(opencv, "available", lambda: (False, "FIPS abort"))
    monkeypatch.setattr(findlike, "available", lambda: (True, ""))
    assert _find_like_available() is True
    monkeypatch.setattr(findlike, "available",
                        lambda: (False, "PLANLENS_FINDLIKE_BACKEND=opencv"))
    assert _find_like_available() is False


def test_a_host_where_opencv_cannot_load_keeps_find_like(monkeypatch):
    _needs_planlens_answer()
    monkeypatch.setattr(opencv, "available",
                        lambda: (False, "OpenCV cannot load on this host"))
    monkeypatch.delenv("PLANLENS_FINDLIKE_BACKEND", raising=False)
    assert _find_like_available() is True


def test_planlens_0_11_asks_whether_opencv_loads(monkeypatch):
    monkeypatch.delattr(findlike, "available", raising=False)
    monkeypatch.setattr(opencv, "available", lambda: (False, "FIPS abort"))
    assert _find_like_available() is False
    monkeypatch.setattr(opencv, "available", lambda: (True, ""))
    assert _find_like_available() is True


def test_planlens_before_0_11_has_nothing_to_ask(monkeypatch):
    monkeypatch.delattr(findlike, "available", raising=False)
    monkeypatch.setitem(sys.modules, "planlens.opencv", None)
    assert _find_like_available() is True
