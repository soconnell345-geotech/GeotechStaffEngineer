"""Shared fixtures for the webapp tests.

Booting the app (``AppTest``) runs ``core.apply_default_output_dir``, which
writes ``GEOTECH_DEFAULT_OUTPUT_DIR`` straight into ``os.environ`` — correct in
the app, but monkeypatch never sees it, so the var (pointing at a tmp_path that
is gone by then) outlives the test and every later test in the session inherits
it. That made the gate order-dependent: funhouse_agent's
``test_default_output_path_html`` fails only when a webapp test ran first.
"""

import os

import pytest

_LEAKY_ENV = ("GEOTECH_DEFAULT_OUTPUT_DIR",)


@pytest.fixture(autouse=True)
def _restore_process_env():
    """Restore the env vars the app sets directly, after every webapp test."""
    before = {k: os.environ.get(k) for k in _LEAKY_ENV}
    yield
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
