"""The application's details (INF-33 step 13): the core names no
application; what the page hands in is what the run record, the Word and
markup authors and the runtime messages name."""

import pytest

from aec_agent_core import app_details
from aec_agent_core.hooks import clear_hook, get_hook, set_hook


@pytest.fixture
def saved_hook():
    from aec_agent_core import hooks
    before = hooks.hooks().get(app_details.HOOK)
    yield
    if before is None:
        clear_hook(app_details.HOOK)
    else:
        set_hook(app_details.HOOK, before)


def test_nothing_handed_in_names_no_application(saved_hook):
    clear_hook(app_details.HOOK)
    assert app_details.details() == app_details.DEFAULTS
    assert app_details.name() == "AEC agent"
    assert app_details.dist() is None
    from aec_agent_core import activity_log, docx_renderer, document_tools
    from aec_agent_core.review_eval import runner
    assert activity_log._app_version() is None
    assert docx_renderer.DEFAULT_AUTHOR == "AEC agent"
    assert document_tools.DEFAULT_MARKUP_AUTHOR == "AEC agent (AI draft)"
    assert document_tools.OWN_MARK_WORDS == ("AEC agent", "(AI draft)")
    assert runner.SUITE_AUTHOR == "Review suite via AEC agent (AI draft)"


def test_what_the_page_hands_in_is_named(saved_hook, monkeypatch):
    app_details.hand_in(name="Made-up App", dist="made-up-app",
                        package="made_up_app")
    assert get_hook(app_details.HOOK) is not None
    assert app_details.name() == "Made-up App"
    assert app_details.package() == "made_up_app"
    from aec_agent_core import activity_log, docx_renderer, document_tools
    from aec_agent_core import runtime_check
    from aec_agent_core.review_eval import runner
    import importlib.metadata as md
    monkeypatch.setattr(md, "version",
                        lambda d: "9.9" if d == "made-up-app" else None)
    assert activity_log._app_version() == "9.9"
    assert docx_renderer.DEFAULT_AUTHOR == "Made-up App"
    assert document_tools.markup_author() == "Made-up App (AI draft)"
    assert runner.SUITE_AUTHOR == "Review suite via Made-up App (AI draft)"
    assert runner._suite_folder() == "Made-up App/review_eval"
    assert runtime_check._who() == "made_up_app"


def test_a_broken_hook_falls_back(saved_hook):
    def _broken():
        raise RuntimeError("no")
    set_hook(app_details.HOOK, _broken)
    assert app_details.details() == app_details.DEFAULTS
