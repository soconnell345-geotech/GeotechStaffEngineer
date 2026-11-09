"""The core's hooks (BACKLOG INF-32, step 11 (a)): the registry, and what the
two modules that ask for hooks do with and without an app filling them.

Run with only the core and planlens installed as well as in the app's
tests, so every test here starts from an EMPTY registry (an app imported
by another test in the same process fills it from its package door) and
puts it back after.
"""

from __future__ import annotations

import json
import os

import pytest

from aec_agent_core import hooks


@pytest.fixture(autouse=True)
def empty_hooks():
    with hooks._LOCK:
        saved = (dict(hooks._HOOKS), dict(hooks._RESOLVED))
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
    yield
    with hooks._LOCK:
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
        hooks._HOOKS.update(saved[0])
        hooks._RESOLVED.update(saved[1])


# -- the registry -----------------------------------------------------------

def test_an_unfilled_hook_is_none():
    assert hooks.get_hook("nobody.filled.this") is None
    assert hooks.hooks() == {}


def test_a_callable_fills_a_hook_and_clear_empties_it():
    hooks.set_hook("t.double", lambda x: 2 * x)
    assert hooks.get_hook("t.double")(4) == 8
    hooks.clear_hook("t.double")
    assert hooks.get_hook("t.double") is None
    hooks.clear_hook("t.double")          # no error when already empty


def test_a_string_is_imported_on_first_use_and_refilling_drops_it():
    hooks.set_hook("t.join", "os.path:join")
    assert "t.join" in hooks.hooks()
    assert hooks.get_hook("t.join") is os.path.join
    hooks.set_hook("t.join", "json:dumps")
    assert hooks.get_hook("t.join") is json.dumps


def test_a_string_names_a_dotted_attribute():
    hooks.set_hook("t.sep", "os:path.sep")
    assert hooks.get_hook("t.sep") == os.path.sep


def test_a_missing_module_is_raised_to_the_caller():
    hooks.set_hook("t.gone", "no_such_package_here_xyz:f")
    with pytest.raises(ImportError):
        hooks.get_hook("t.gone")


@pytest.mark.parametrize("name,target", [
    ("", "os:getcwd"), (None, "os:getcwd"), ("t.x", "os.getcwd"),
    ("t.x", "os:"), ("t.x", ":getcwd"), ("t.x", "os:get cwd"), ("t.x", 3),
])
def test_bad_names_and_targets_are_refused(name, target):
    with pytest.raises(ValueError):
        hooks.set_hook(name, target)


# -- the Prompter connection without and with the page -------------------------

def _prompter_env(monkeypatch, key="k-1"):
    monkeypatch.setenv("PROMPTER_URL",
                       "https://prompter.example/api/v1/chat/completions")
    monkeypatch.setenv("PROMPTER_MODEL", "model-a")
    if key is None:
        monkeypatch.delenv("PROMPTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PROMPTER_API_KEY", key)
    monkeypatch.delenv("PROMPTER_CA_BUNDLE", raising=False)


def test_without_a_page_settings_come_from_the_environment(monkeypatch):
    from aec_agent_core import tinyapps_engine as te
    _prompter_env(monkeypatch)
    ps = te.settings()
    assert ps.base_url == "https://prompter.example/api/v1"
    assert ps.model == "model-a" and ps.api_key == "k-1"
    rp = te.resolve_for(object())
    assert rp.source == te.SOURCE_SHARED and rp.settings == ps
    assert te.tester_keys_possible() is False


def test_without_a_page_and_no_key_nobody_is_served(monkeypatch):
    from aec_agent_core import tinyapps_engine as te
    _prompter_env(monkeypatch, key=None)
    assert te.settings() is None
    with pytest.raises(te.EngineUnavailable) as e:
        te.resolve_for(None)
    assert str(e.value) == te.NOT_CONFIGURED_MESSAGE
    assert te.register() is False


def test_a_page_reads_the_settings_and_decides_whose_key(monkeypatch):
    from aec_agent_core import tinyapps_engine as te
    for name in ("PROMPTER_URL", "PROMPTER_MODEL", "PROMPTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    store = {"PROMPTER_URL": "https://p.example/api/v1",
             "PROMPTER_MODEL": "model-b", "PROMPTER_API_KEY": "shared"}
    hooks.set_hook("prompter.get_setting", store.get)
    assert te.settings().model == "model-b"

    def resolve(ident):
        if ident == "nobody":
            raise te.NoPrompterKey("no key for nobody")
        return te.ResolvedPrompter(te.settings(), te.SOURCE_TESTER)

    hooks.set_hook("prompter.resolve_for", resolve)
    hooks.set_hook("prompter.tester_keys_possible", lambda: True)
    assert te.resolve_for("someone").source == te.SOURCE_TESTER
    assert te.resolve_for("someone").source_text == "per-tester secret"
    with pytest.raises(te.NoPrompterKey):
        te.resolve_for("nobody")
    assert te.tester_keys_possible() is True


# -- the vision tools without and with an app's reference library ------------

def test_without_an_app_the_reference_readers_are_unknown_tools():
    from aec_agent_core import vision_tools as vt
    for name in vt.APP_READER_TOOLS:
        out = json.loads(vt.dispatch_extended_tool(name, {}, None, {}))
        assert out == {"error": f"Unknown extended tool: {name}"}


def test_an_app_serves_the_reference_readers():
    from aec_agent_core import vision_tools as vt
    seen = []
    hooks.set_hook("vision_tools.read_reference_figure",
                   lambda args, engine: (seen.append((args, engine))
                                         or json.dumps({"read": True})))
    out = vt.dispatch_extended_tool("read_reference_figure",
                                    {"reference": "r", "figure_number": "1"},
                                    "an engine", {})
    assert json.loads(out) == {"read": True}
    assert seen == [({"reference": "r", "figure_number": "1"}, "an engine")]


def test_reference_read_roots_env_first_then_the_app(monkeypatch, tmp_path):
    from aec_agent_core import vision_tools as vt
    env_dir, app_dir = tmp_path / "env", tmp_path / "lib"
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(env_dir))
    assert vt.reference_read_roots() == [str(env_dir)]
    hooks.set_hook("vision_tools.reference_read_roots",
                   lambda: [str(app_dir), str(env_dir)])
    assert vt.reference_read_roots() == [str(env_dir), str(app_dir)]


def test_a_broken_app_library_never_breaks_the_read_tools(monkeypatch):
    from aec_agent_core import vision_tools as vt
    monkeypatch.delenv("GEOTECH_REFERENCES_DOCS", raising=False)

    def broken():
        raise RuntimeError("the library is not installed")

    hooks.set_hook("vision_tools.reference_read_roots", broken)
    assert vt.reference_read_roots() == []
