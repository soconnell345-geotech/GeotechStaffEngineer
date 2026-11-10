"""The Tiny Apps entry: two pages over one shell, and the profiles behind them."""

import os

import pytest

from aec_document_review import profiles, tinyapps_entry
from aec_agent_core import engine_config
from aec_document_review.identity import ANONYMOUS, parse_principal


#: Env the entry WRITES straight into ``os.environ`` (``mark_deployment``'s
#: ``GEOTECH_DEPLOYMENT``, ``register_engine``'s model list and default), on
#: top of what the tests set. monkeypatch records no undo for a variable it
#: deleted while absent, so these are restored by hand: left behind,
#: ``GEOTECH_DEPLOYMENT=tinyapps`` turned four engine tests in
#: ``test_core.py`` into "No model configured" when they ran after this file.
_ENV = ("GEOTECH_DEPLOYMENT", "GEOTECH_APP_PROFILE", "PROMPTER_URL",
        "PROMPTER_MODEL", "PROMPTER_API_KEY", "GEOTECH_PROMPTER_MODELS",
        "GEOTECH_WEBAPP_MODEL")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    before = {e: os.environ.get(e) for e in _ENV}
    for e in _ENV:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(tinyapps_entry, "_ENGINE_REGISTERED", False)
    engine_config.register_model_builder(None)
    yield
    engine_config.register_model_builder(None)
    for e, value in before.items():
        if value is None:
            os.environ.pop(e, None)
        else:
            os.environ[e] = value


def test_app_path_points_at_packaged_app():
    p = tinyapps_entry.app_path()
    assert os.path.basename(p) == "app.py"
    assert os.path.exists(p)


def test_deployment_marker_and_engine_registration(monkeypatch):
    tinyapps_entry.mark_deployment()
    assert os.environ["GEOTECH_DEPLOYMENT"] == "tinyapps"
    assert engine_config.is_tinyapps_deployment()
    # no Prompter settings: nothing registered, the app will show its banner
    assert tinyapps_entry.register_engine() is False
    assert not engine_config.has_model_builder()




# ------------------------------------------------------------------ profiles









#: Core dependencies known to cap pandas below 3.0 - CfA's fleet pins
#: pandas==3.0.2, so one of these in core makes the wrapper's packages.txt
#: unresolvable (BACKLOG TA-02, 2026-10-12: every python-ags4 release).
PANDAS_LT3 = ("python-ags4",)




