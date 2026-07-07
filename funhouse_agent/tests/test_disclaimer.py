"""Tests for the professional-use disclaimer: one-time first-import banner
(marker / env / pytest suppression), disclaimer() content, and console wiring.

The banner is normally suppressed under pytest (``pytest`` is in ``sys.modules``),
so the "it should show" tests monkeypatch ``_running_under_pytest`` to False and
redirect the marker file into a tmp dir — the real ``~/.geotech_staff_engineer``
is never touched.
"""

import io
from pathlib import Path

import pytest

import funhouse_agent
from funhouse_agent import _disclaimer


@pytest.fixture
def tmp_ack(tmp_path, monkeypatch):
    """Redirect the acknowledgement marker into a throwaway dir."""
    monkeypatch.setattr(_disclaimer, "_ACK_DIR", str(tmp_path / ".geotech_staff_engineer"))
    return tmp_path


# --------------------------------------------------------------------------- #
# first-import banner: suppression paths
# --------------------------------------------------------------------------- #

def test_banner_suppressed_under_pytest(tmp_ack, monkeypatch, capsys):
    """Real pytest detection active -> never shows, never writes the marker."""
    monkeypatch.delenv(_disclaimer.SUPPRESS_ENV, raising=False)
    assert _disclaimer._running_under_pytest() is True  # sanity
    assert _disclaimer.maybe_show_first_import_notice() is False
    assert capsys.readouterr().err == ""
    assert not Path(_disclaimer._ack_path()).exists()


def test_banner_suppressed_by_env(tmp_ack, monkeypatch, capsys):
    monkeypatch.setattr(_disclaimer, "_running_under_pytest", lambda: False)
    monkeypatch.setenv(_disclaimer.SUPPRESS_ENV, "1")
    assert _disclaimer.maybe_show_first_import_notice() is False
    assert capsys.readouterr().err == ""
    assert not Path(_disclaimer._ack_path()).exists()


def test_banner_skipped_when_already_acknowledged(tmp_ack, monkeypatch, capsys):
    monkeypatch.setattr(_disclaimer, "_running_under_pytest", lambda: False)
    monkeypatch.delenv(_disclaimer.SUPPRESS_ENV, raising=False)
    # Pre-create the marker.
    Path(_disclaimer._ack_path()).parent.mkdir(parents=True, exist_ok=True)
    Path(_disclaimer._ack_path()).write_text("shown\n", encoding="utf-8")
    assert _disclaimer.maybe_show_first_import_notice() is False
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# first-import banner: the show-once path
# --------------------------------------------------------------------------- #

def test_banner_shows_once_then_writes_marker(tmp_ack, monkeypatch, capsys):
    monkeypatch.setattr(_disclaimer, "_running_under_pytest", lambda: False)
    monkeypatch.delenv(_disclaimer.SUPPRESS_ENV, raising=False)

    # First call: prints to stderr (not stdout) and drops the marker.
    assert _disclaimer.maybe_show_first_import_notice() is True
    out = capsys.readouterr()
    assert out.out == ""  # stderr only, never stdout
    assert "ANALYSIS/RESEARCH AID" in out.err
    assert "GEOTECH_NO_DISCLAIMER" in out.err
    assert Path(_disclaimer._ack_path()).exists()

    # Second call: marker present -> silent.
    assert _disclaimer.maybe_show_first_import_notice() is False
    assert capsys.readouterr().err == ""


def test_banner_never_raises_on_broken_stderr(tmp_ack, monkeypatch):
    monkeypatch.setattr(_disclaimer, "_running_under_pytest", lambda: False)
    monkeypatch.delenv(_disclaimer.SUPPRESS_ENV, raising=False)

    class _Broken:
        def write(self, *a, **k):
            raise OSError("no stderr")

    monkeypatch.setattr(_disclaimer.sys, "stderr", _Broken())
    # Must swallow the error and report False, not propagate.
    assert _disclaimer.maybe_show_first_import_notice() is False


# --------------------------------------------------------------------------- #
# disclaimer() content + on-demand printing
# --------------------------------------------------------------------------- #

def test_disclaimer_content_to_custom_stream():
    buf = io.StringIO()
    _disclaimer.disclaimer(file=buf)
    text = buf.getvalue()
    assert "Professional-use disclaimer" in text
    assert "engineer-of-record" in text
    assert "WITHOUT WARRANTY OF ANY KIND" in text
    assert "SI (m, kPa, kN, degrees)" in text
    assert "validation_examples/RESULTS.md" in text


def test_disclaimer_defaults_to_stdout(capsys):
    _disclaimer.disclaimer()
    out = capsys.readouterr()
    assert "Professional-use disclaimer" in out.out
    assert out.err == ""


def test_package_exposes_disclaimer():
    assert callable(funhouse_agent.disclaimer)
    assert "disclaimer" in funhouse_agent.__all__


# --------------------------------------------------------------------------- #
# console-script wiring
# --------------------------------------------------------------------------- #

def test_console_main_prints_and_returns_zero(capsys):
    rc = _disclaimer.main()
    assert rc == 0
    assert "Professional-use disclaimer" in capsys.readouterr().out


def test_console_script_declared_in_pyproject():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():  # not shipped in the wheel; source-tree only
        pytest.skip("pyproject.toml not present (installed wheel)")
    text = pyproject.read_text(encoding="utf-8")
    assert 'geotech-disclaimer = "funhouse_agent._disclaimer:main"' in text
