"""Shared fixtures for Document Review's tests (from GeotechStaffEngineer's
root ``conftest.py`` and ``webapp/tests/conftest.py``, copied at the app
split's step 13, BACKLOG INF-33)."""

import os

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

os.environ.setdefault("GEOTECH_NO_DISCLAIMER", "1")

_LEAKY_ENV = ("GEOTECH_DEFAULT_OUTPUT_DIR",)


@pytest.fixture(autouse=True, scope="session")
def _app_data_in_a_temp_folder(tmp_path_factory):
    """The page's data folder is a temporary one for the whole run, so no
    test writes into the user's own ``~/.geotech_webapp`` (every writer
    finds it through ``core.data_root``, which reads ``GEOTECH_WEBAPP_DATA``
    at each call). It stays set to the end: a worker thread a test started
    may still be writing."""
    os.environ["GEOTECH_WEBAPP_DATA"] = str(
        tmp_path_factory.mktemp("document_review_data"))
    yield


@pytest.fixture(autouse=True)
def _restore_process_env():
    """Restore the env vars the app sets directly, after every test (booting
    the app writes ``GEOTECH_DEFAULT_OUTPUT_DIR`` into ``os.environ``)."""
    before = {k: os.environ.get(k) for k in _LEAKY_ENV}
    yield
    for key, value in before.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _no_sharepoint_log(monkeypatch):
    """The SharePoint log writes on a thread of its own; off for every test
    (the log's own tests turn it on and wait for the write)."""
    monkeypatch.setenv("GEOTECH_SHAREPOINT_LOG", "0")
    yield


@pytest.fixture(autouse=True)
def _shared_host_flag_reset():
    """``aec_agent_core._fileio.require_turn_binding`` is process-wide (a
    multi-user identity sets it); keep one test's shared-host mode from
    leaking into the next."""
    from aec_agent_core import _fileio
    saved = _fileio._REQUIRE_BINDING
    yield
    _fileio._REQUIRE_BINDING = saved
