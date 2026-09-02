"""Offline tests for the sidebar budget panel (fake Funhouse SDK modules)."""

import sys
import types

import pytest

import webapp.budget_panel as bp
from webapp import core


# ---------------------------------------------------------------------------
# Fake SDK (funhouse.admin.budget + funhouse.config in sys.modules)
# ---------------------------------------------------------------------------

def _install_fake_sdk(monkeypatch, spend=12.34, cap=50.0, spend_exc=None,
                      cap_exc=None):
    class FakeConfig:
        instances = []

        @classmethod
        def get_instance(cls):
            inst = cls()
            cls.instances.append(inst)
            return inst

        def get(self, key, default=None):
            if key == "budget.monthly_budget":
                if cap_exc is not None:
                    raise cap_exc
                return cap
            return default

    class FakeBudget:
        built_with = []

        def __init__(self, config=None):
            FakeBudget.built_with.append(config)

        def get_current_spend(self):
            if spend_exc is not None:
                raise spend_exc
            return spend

    fh = types.ModuleType("funhouse")
    admin = types.ModuleType("funhouse.admin")
    budget_mod = types.ModuleType("funhouse.admin.budget")
    config_mod = types.ModuleType("funhouse.config")
    budget_mod.FunhouseBudget = FakeBudget
    config_mod.FunhouseConfig = FakeConfig
    fh.admin, admin.budget, fh.config = admin, budget_mod, config_mod
    for name, mod in (("funhouse", fh), ("funhouse.admin", admin),
                      ("funhouse.admin.budget", budget_mod),
                      ("funhouse.config", config_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return FakeBudget, FakeConfig


def _uninstall_sdk(monkeypatch):
    """Make ``import funhouse...`` fail even if something is installed."""
    for name in ("funhouse", "funhouse.admin", "funhouse.admin.budget",
                 "funhouse.config", "funhouse.services",
                 "funhouse.services.email"):
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------

def test_status_from_fake_sdk(monkeypatch):
    FakeBudget, FakeConfig = _install_fake_sdk(monkeypatch, spend=12.34,
                                               cap=75.0)
    status = bp.get_budget_status()
    assert status == {"spent": 12.34, "cap": 75.0}
    # the budget object was built with the config instance (per SDK pattern)
    assert FakeBudget.built_with[-1] is FakeConfig.instances[-1]
    assert bp.available() is True


def test_status_none_when_sdk_absent(monkeypatch):
    _uninstall_sdk(monkeypatch)
    assert bp.get_budget_status() is None
    assert bp.available() is False


def test_status_none_on_spend_error(monkeypatch):
    _install_fake_sdk(monkeypatch, spend_exc=RuntimeError("sqlite locked"))
    assert bp.get_budget_status() is None


def test_status_none_on_config_error(monkeypatch):
    _install_fake_sdk(monkeypatch, cap_exc=KeyError("budget.monthly_budget"))
    assert bp.get_budget_status() is None


def test_cap_defaults_to_50(monkeypatch):
    """The SDK read pattern: config.get('budget.monthly_budget', default=50.0)."""
    class NoCapConfig:
        @classmethod
        def get_instance(cls):
            return cls()

        def get(self, key, default=None):
            return default                          # no key configured

    _install_fake_sdk(monkeypatch, spend=1.0)
    sys.modules["funhouse.config"].FunhouseConfig = NoCapConfig
    assert bp.get_budget_status() == {"spent": 1.0, "cap": 50.0}


def test_format_line():
    line = bp.format_line({"spent": 12.34, "cap": 50.0})
    assert line == "AI budget: $12.34 of $50.00 used this month"
    assert bp.format_line({"spent": 0.0, "cap": 1250.0}) == \
        "AI budget: $0.00 of $1,250.00 used this month"


def test_friendly_turn_error_budget_exceeded():
    class BudgetExceededError(RuntimeError):
        pass
    err = core.friendly_turn_error(
        BudgetExceededError("Monthly AI budget exceeded: $50.12 of $50.00"))
    assert "Monthly AI budget exceeded" in err       # raw error preserved
    assert "resets next month" in err and "Funhouse admins" in err
    # unrelated errors stay untouched
    assert core.friendly_turn_error(ValueError("boom")) == "ValueError: boom"
