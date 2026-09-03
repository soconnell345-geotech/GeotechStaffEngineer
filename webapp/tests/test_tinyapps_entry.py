"""Smoke tests for the TinyApps entry stub."""

import os

from webapp import tinyapps_entry


def test_app_path_points_at_packaged_app():
    p = tinyapps_entry.app_path()
    assert os.path.basename(p) == "app.py"
    assert os.path.exists(p)


def test_deployment_marker(monkeypatch):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    monkeypatch.setattr("runpy.run_path", lambda *a, **k: None)
    tinyapps_entry.main()
    assert os.environ["GEOTECH_DEPLOYMENT"] == "tinyapps"
