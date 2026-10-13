"""A model call refused at sign-in (HTTP 401) - BACKLOG TA-01.

Field session 2026-09-09 (Nairobi SoE): the agent's own model call to the
Funhouse proxy came back as an IIS "401 - Unauthorized" HTML page; the turn
ended on it, the raw page was shown, and it read as a SharePoint permissions
failure. The next turn worked seconds later on the same credentials. So the
call is asked once more, and a refusal that stands is said plainly.
"""

import pytest

pytest.importorskip("langchain.agents.middleware")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import error_text  # noqa: E402
from funhouse_agent.deep import tool_guards  # noqa: E402
from webapp import core  # noqa: E402

IIS_401 = (
    'Error code: 401 - <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"'
    ' "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n<html '
    'xmlns="http://www.w3.org/1999/xhtml">\n<head>\n<meta http-equiv='
    '"Content-Type" content="text/html; charset=iso-8859-1"/>\n<title>401 - '
    'Unauthorized: Access is denied due to invalid credentials.</title>\n'
    '<style type="text/css">body{margin:0;font-size:.7em;}</style>\n</head>'
    '\n<body><div id="header"><h1>Server Error</h1></div><div id="content">'
    '<h2>401 - Unauthorized: Access is denied due to invalid credentials.'
    '</h2></div></body>\n</html>')


class AuthenticationError(Exception):
    """Stands in for openai's: same name, a 401."""
    status_code = 401


class Status:
    def __init__(self, code):
        self.status_code = code


class HttpError(Exception):
    def __init__(self, text, status):
        super().__init__(text)
        self.response = Status(status)


def test_a_401_is_a_refusal_and_nothing_else_is():
    assert error_text.auth_refused(AuthenticationError(IIS_401))
    assert error_text.auth_refused(HttpError("denied", 401))
    assert error_text.auth_refused(RuntimeError(IIS_401))
    wrapped = RuntimeError("model call failed")
    wrapped.__cause__ = AuthenticationError("no")
    assert error_text.auth_refused(wrapped)
    assert not error_text.auth_refused(HttpError("busy", 429))
    assert not error_text.auth_refused(HttpError("server", 500))
    assert not error_text.auth_refused(HttpError("forbidden", 403))
    # a spent budget is said as one, never as a sign-in refusal
    assert not error_text.auth_refused(HttpError("payment required", 402))


def test_the_html_page_is_cut_to_its_title_and_keys_never_show():
    assert error_text.html_gist(IIS_401) == (
        "401 - Unauthorized: Access is denied due to invalid credentials.")
    said = error_text.html_gist(
        "Error code: 401 - Incorrect API key provided: sk-proj-abcd1234wxyz.")
    assert "sk-proj" not in said and "abcd1234" not in said
    assert error_text.html_gist("  plain\n  text ") == "plain text"


def test_the_turn_error_names_the_service_and_hides_the_page():
    text = core.friendly_turn_error(AuthenticationError(IIS_401))
    assert "AI model service refused this app's sign-in" in text
    assert "Access is denied due to invalid credentials" in text
    assert "<html" not in text and "<title" not in text
    assert "SharePoint" not in text
    assert "tell the app owner" in text and "Ask again" in text


class Refuses(FakeMessagesListChatModel):
    """Refuses the first ``refusals`` calls with a 401, then answers."""

    refusals: int = 1
    calls: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls.append(1)
        if len(self.calls) <= self.refusals:
            raise AuthenticationError(IIS_401)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="the answer"))])


def _review_agent(model):
    from funhouse_agent.deep.agent import build_deep_agent
    from webapp.profiles import DOCUMENT_REVIEW
    return build_deep_agent(model, **DOCUMENT_REVIEW.build_kwargs())


def test_one_refusal_is_asked_again_and_the_turn_answers(monkeypatch):
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.setattr(tool_guards, "AUTH_RETRY_DELAY_S", 0.0)
    model = Refuses(responses=[AIMessage(content="x")], refusals=1, calls=[])
    out = list(core.stream_turn(_review_agent(model),
                                [{"role": "user", "content": "go"}], "t"))
    done = [e for e in out if e["kind"] == "turn_done"][0]
    assert "the answer" in done["answer"]
    assert len(model.calls) == 2


def test_a_refusal_that_stands_ends_the_turn_said_plainly(monkeypatch):
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.setattr(tool_guards, "AUTH_RETRY_DELAY_S", 0.0)
    model = Refuses(responses=[AIMessage(content="x")], refusals=9, calls=[])
    with pytest.raises(Exception) as caught:
        list(core.stream_turn(_review_agent(model),
                              [{"role": "user", "content": "go"}], "t"))
    assert len(model.calls) == 2                      # asked once more only
    assert "refused this app's sign-in" in core.friendly_turn_error(
        caught.value)


def test_nothing_but_a_refusal_is_retried(monkeypatch):
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.setattr(tool_guards, "AUTH_RETRY_DELAY_S", 0.0)

    class Breaks(Refuses):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(1)
            raise ValueError("not a sign-in problem")

    model = Breaks(responses=[AIMessage(content="x")], calls=[])
    with pytest.raises(ValueError):
        list(core.stream_turn(_review_agent(model),
                              [{"role": "user", "content": "go"}], "t"))
    assert len(model.calls) == 1
