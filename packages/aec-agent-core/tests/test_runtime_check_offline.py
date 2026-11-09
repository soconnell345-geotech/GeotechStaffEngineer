"""Offline tests for funhouse_agent.runtime_check.ensure_typing_extensions.

These simulate the Databricks "stale pre-imported typing_extensions" situation by
swapping a FAKE module into ``sys.modules`` and a FAKE installed-dist lookup —
the real ``typing_extensions`` is never reloaded or mutated (``monkeypatch``
restores every ``sys.modules`` entry on teardown), so the rest of the suite is
unaffected.
"""

import types

import pytest

from funhouse_agent import runtime_check
from funhouse_agent.runtime_check import (
    TypingExtensionsCheck,
    TypingExtensionsError,
    ensure_typing_extensions,
)

_TE = "typing_extensions"


def _stale_te(version="4.9.0"):
    """A fake pre-4.13 typing_extensions: its TypedDict rejects PEP 728 kwargs."""
    mod = types.ModuleType(_TE)
    mod.__version__ = version

    def _old_typeddict(typename, fields=None, **kwargs):
        if kwargs:
            bad = next(iter(kwargs))
            raise TypeError(
                f"TypedDict() got an unexpected keyword argument {bad!r}"
            )
        return dict

    mod.TypedDict = _old_typeddict
    return mod


def _modern_te(version=None):
    """A fake >=4.13 typing_extensions whose TypedDict accepts ``closed=``."""
    mod = types.ModuleType(_TE)
    if version is not None:
        mod.__version__ = version

    def _new_typeddict(typename, fields=None, *, closed=False, extra_items=None, total=True):
        return dict

    mod.TypedDict = _new_typeddict
    return mod


# --------------------------------------------------------------------------- #
# no-op paths
# --------------------------------------------------------------------------- #

def test_not_loaded_is_noop(monkeypatch):
    """typing_extensions absent from sys.modules -> nothing to reload."""
    monkeypatch.delitem(runtime_check.sys.modules, _TE, raising=False)
    # A reload here would be a bug: make it explode if called.
    monkeypatch.setattr(
        runtime_check.importlib, "reload",
        lambda m: pytest.fail("reload must not run on the not-loaded path"),
    )
    result = ensure_typing_extensions()
    assert isinstance(result, TypingExtensionsCheck)
    assert result.action == "not-loaded"


def test_already_ok_is_noop(monkeypatch):
    """A loaded, PEP 728-capable module needs no reload."""
    monkeypatch.setitem(runtime_check.sys.modules, _TE, _modern_te())
    monkeypatch.setattr(
        runtime_check.importlib, "reload",
        lambda m: pytest.fail("reload must not run when the module is already OK"),
    )
    result = ensure_typing_extensions()
    assert result.action == "already-ok"


# --------------------------------------------------------------------------- #
# reload path (the main fix)
# --------------------------------------------------------------------------- #

def test_stale_module_reloaded_in_place(monkeypatch):
    """Stale loaded module + adequate disk version -> reload in place, PEP 728 OK."""
    stale = _stale_te("4.9.0")
    monkeypatch.setitem(runtime_check.sys.modules, _TE, stale)
    monkeypatch.setattr(runtime_check, "_installed_dist_version", lambda name=_TE: "4.15.0")

    calls = []

    def fake_reload(module):
        calls.append(module)
        # Simulate the on-disk 4.15 re-executing into the same object.
        module.__version__ = "4.15.0"
        module.TypedDict = _modern_te().TypedDict
        return module

    monkeypatch.setattr(runtime_check.importlib, "reload", fake_reload)

    result = ensure_typing_extensions()
    assert result.action == "reloaded"
    assert calls == [stale]  # reload actually ran, on the stale object
    assert result.installed_version == "4.15.0"


# --------------------------------------------------------------------------- #
# too-old-disk error path
# --------------------------------------------------------------------------- #

def test_disk_too_old_raises_pip_instruction(monkeypatch):
    stale = _stale_te("4.5.0")
    monkeypatch.setitem(runtime_check.sys.modules, _TE, stale)
    monkeypatch.setattr(runtime_check, "_installed_dist_version", lambda name=_TE: "4.5.0")
    monkeypatch.setattr(
        runtime_check.importlib, "reload",
        lambda m: pytest.fail("reload must not run when the disk copy is too old"),
    )

    with pytest.raises(TypingExtensionsError) as excinfo:
        ensure_typing_extensions()
    msg = str(excinfo.value)
    assert "pip install" in msg
    assert ">=4.13" in msg


# --------------------------------------------------------------------------- #
# reload-failure fallbacks -> restart instruction
# --------------------------------------------------------------------------- #

def test_reload_raises_falls_back_to_restart(monkeypatch):
    stale = _stale_te("4.9.0")
    monkeypatch.setitem(runtime_check.sys.modules, _TE, stale)
    monkeypatch.setattr(runtime_check, "_installed_dist_version", lambda name=_TE: "4.15.0")

    def boom(module):
        raise RuntimeError("cannot reload built-in-ish module")

    monkeypatch.setattr(runtime_check.importlib, "reload", boom)

    with pytest.raises(TypingExtensionsError) as excinfo:
        ensure_typing_extensions()
    assert "restartPython" in str(excinfo.value)


def test_reload_that_does_not_take_falls_back_to_restart(monkeypatch):
    """Reload 'succeeds' but the object is still stale (older copy wins)."""
    stale = _stale_te("4.9.0")
    monkeypatch.setitem(runtime_check.sys.modules, _TE, stale)
    monkeypatch.setattr(runtime_check, "_installed_dist_version", lambda name=_TE: "4.15.0")
    # reload returns the module unchanged -> still no PEP 728.
    monkeypatch.setattr(runtime_check.importlib, "reload", lambda m: m)

    with pytest.raises(TypingExtensionsError) as excinfo:
        ensure_typing_extensions()
    assert "restartPython" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# unit helpers
# --------------------------------------------------------------------------- #

def test_supports_pep728_probe():
    assert runtime_check._supports_pep728(_modern_te()) is True
    assert runtime_check._supports_pep728(_stale_te()) is False
    assert runtime_check._supports_pep728(types.ModuleType("x")) is False  # no TypedDict


def test_version_lt():
    assert runtime_check._version_lt("4.9.0", "4.13") is True
    assert runtime_check._version_lt("4.15.0", "4.13") is False
    assert runtime_check._version_lt("4.13.0", "4.13") is False  # 4.13.0 satisfies >=4.13


def test_real_typing_extensions_untouched():
    """The real module is still importable and PEP 728-capable after the fakes."""
    import typing_extensions as te
    assert runtime_check._supports_pep728(te) is True
