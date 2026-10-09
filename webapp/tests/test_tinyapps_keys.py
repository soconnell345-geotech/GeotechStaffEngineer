"""Per-tester Prompter keys on Tiny Apps, offline (owner, 2026-10-09).

Each tester on the published deployment gets an AI key, and so a budget, of
their own. The key is found by the signed-in identity: a per-tester secret
``PROMPTER-API-KEY--<SUFFIX>``, else the ``PROMPTER-KEYS`` JSON mapping,
else the shared key only when ``PROMPTER_SHARED_KEY_FALLBACK`` is on; a
tester with none is refused politely. Single-user hosts keep the shared key.

Nothing here touches a network or a real Key Vault: values come from the
environment, or from a fake vault standing in for
``tinyapps_settings._from_key_vault``.
"""

from __future__ import annotations

import json
import logging
import os
import re

import pytest

from webapp import diagnostics, engine_config, identity
from webapp import tinyapps_engine as te
from webapp import tinyapps_settings as ts

URL = "https://prompter.example/api/v1/chat/completions"
ALICE = identity.Identity("CORP", "alice", "header")
BOB = identity.Identity("CORP", "bob", "header")
CAROL = identity.Identity("CORP", "carol", "header")
DEV = identity.Identity("CORP", "alice", "dev")      # DEV_IDENTITY on dosdev

_ALL_ENVS = (
    "APP_ENV", "IDENTITY_ENDPOINT", "KV_NAME", "GEOTECH_DOTENV",
    "PROMPTER_URL", "PROMPTER_MODEL", "PROMPTER_API_KEY", "PROMPTER_CA_BUNDLE",
    "PROMPTER_KEYS", "PROMPTER_SHARED_KEY_FALLBACK",
    "GEOTECH_DEPLOYMENT", "GEOTECH_PROMPTER_MODELS", "GEOTECH_WEBAPP_MODEL",
    "ANTHROPIC_API_KEY", "DEV_IDENTITY", "GEOTECH_USER_EMAIL",
    "WINDOWS_AUTH_HEADER", "GEOTECH_PROMPTER_DISABLE_STREAMING",
)
_TESTER_ENV = re.compile(r"^PROMPTER_(API_KEY|MODEL|URL)__")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    # register() writes GEOTECH_PROMPTER_MODELS / GEOTECH_WEBAPP_MODEL into
    # os.environ directly: restore by hand (see test_tinyapps_wiring.py).
    before = {e: os.environ.get(e) for e in _ALL_ENVS}
    for e in _ALL_ENVS:
        monkeypatch.delenv(e, raising=False)
    for e in [e for e in os.environ if _TESTER_ENV.match(e)]:
        monkeypatch.delenv(e, raising=False)
    # no stray .env from the working directory
    monkeypatch.setenv("GEOTECH_DOTENV", str(tmp_path / "absent.env"))
    ts.reset()
    engine_config.register_model_builder(None)
    yield
    ts.reset()
    engine_config.register_model_builder(None)
    for e, value in before.items():
        if value is None:
            os.environ.pop(e, None)
        else:
            os.environ[e] = value


def _shared(monkeypatch, key="shared-key", model="gpt-shared"):
    monkeypatch.setenv("PROMPTER_URL", URL)
    monkeypatch.setenv("PROMPTER_MODEL", model)
    if key:
        monkeypatch.setenv("PROMPTER_API_KEY", key)


@pytest.fixture
def vault(monkeypatch):
    """A fake Key Vault: ``{secret name: value}``; records every name asked."""
    store, asked = {}, []

    def _fake(name):
        asked.append(name)
        return store.get(name)

    monkeypatch.setenv("APP_ENV", "azure")
    monkeypatch.setenv("KV_NAME", "kv-test")
    monkeypatch.setattr(ts, "_from_key_vault", _fake)
    store["asked"] = asked          # handy for assertions; never a real name
    return store


def _key_of(model) -> str:
    return model.openai_api_key.get_secret_value()


# ------------------------------------------------------- Key Vault naming

def test_suffix_is_the_identity_key_in_key_vault_alphabet():
    assert ALICE.key == "corp__alice"
    assert te.tester_suffix(ALICE) == "CORP-ALICE"
    assert te.tester_suffix(identity.Identity("Corp", "J.Doe", "header")) \
        == "CORP-J-DOE"
    upn = identity.parse_principal("jdoe@state.gov", "header")
    assert te.tester_suffix(upn) == "STATE-GOV-JDOE"
    name = te.tester_secret_name("PROMPTER_API_KEY", ALICE)
    assert name == "PROMPTER-API-KEY--CORP-ALICE"
    assert te.tester_secret_name("PROMPTER-MODEL", ALICE) == \
        "PROMPTER-MODEL--CORP-ALICE"
    # Key Vault: letters, digits, hyphens; starts with a letter; <= 127
    for who in (ALICE, upn, identity.Identity("D_1", "a b.c-d", "header")):
        n = te.tester_secret_name("PROMPTER_API_KEY", who)
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,126}", n), n
        # the separator is the only double hyphen: the split is unambiguous
        assert n.count("--") == 1
    # a name past the vault's limit is never asked for
    long_one = identity.Identity("CORP", "x" * 200, "header")
    assert te.tester_secret_name("PROMPTER_API_KEY", long_one) is None


def test_the_vault_is_asked_for_exactly_the_documented_name(vault,
                                                           monkeypatch):
    pytest.importorskip("langchain_openai")
    vault.update({"PROMPTER-URL": URL, "PROMPTER-MODEL": "gpt-shared",
                  "PROMPTER-API-KEY--CORP-ALICE": "alice-key"})
    rp = te.resolve_for(ALICE)
    assert rp.source == te.SOURCE_TESTER and rp.settings.api_key == "alice-key"
    assert "PROMPTER-API-KEY--CORP-ALICE" in vault["asked"]
    # every per-tester name asked is valid for Key Vault
    for n in vault["asked"]:
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,126}", n), n


def test_an_app_setting_spelling_also_works(monkeypatch):
    # App Service app settings / .env lines use underscores
    _shared(monkeypatch, key=None)
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    assert te.resolve_for(ALICE).settings.api_key == "alice-key"


# ---------------------------------------------------- two testers, two keys

def test_two_testers_resolve_two_keys(vault):
    pytest.importorskip("langchain_openai")
    vault.update({"PROMPTER-URL": URL, "PROMPTER-MODEL": "gpt-shared",
                  "PROMPTER-API-KEY": "shared-key",
                  "PROMPTER-API-KEY--CORP-ALICE": "alice-key",
                  "PROMPTER-API-KEY--CORP-BOB": "bob-key"})
    assert te.register() is True
    a = engine_config.resolve_engine("gpt-shared", identity=ALICE)
    b = engine_config.resolve_engine("gpt-shared", identity=BOB)
    assert a.ok and b.ok and a.source == b.source == "prompter"
    assert _key_of(a.model) == "alice-key"
    assert a.model.default_headers == {"api-key": "alice-key"}
    assert _key_of(b.model) == "bob-key"
    assert b.model.default_headers == {"api-key": "bob-key"}


def test_a_testers_own_model_and_url_win(monkeypatch):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch)
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    monkeypatch.setenv("PROMPTER_MODEL__CORP_ALICE", "gpt-alice")
    monkeypatch.setenv("PROMPTER_URL__CORP_ALICE",
                       "https://other.example/api/v1/chat/completions")
    model = te.build_for(ALICE, "gpt-shared")      # the picker's choice
    assert model.model_name == "gpt-alice"         # a key serves ONE model
    assert str(model.openai_api_base).rstrip("/") == \
        "https://other.example/api/v1"
    # without an own model the shared one (or the picker's) serves
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_BOB", "bob-key")
    assert te.build_for(BOB, None).model_name == "gpt-shared"


# ------------------------------------------------------ the polite refusal

def test_a_tester_with_no_key_is_refused_politely(monkeypatch):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch, key="shared-key")         # present, but no fallback
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    assert te.register() is True
    res = engine_config.resolve_engine("gpt-shared", identity=CAROL)
    assert not res.ok and res.model is None
    assert res.source == "unavailable"
    assert res.message.startswith("No AI key is set up for **carol** yet")
    assert "ask the app owner" in res.message
    assert "PROMPTER-API-KEY--CORP-CAROL" in res.message   # for the owner
    for leak in ("shared-key", "alice-key", "Traceback", "Error",
                 "NoPrompterKey"):
        assert leak not in res.message
    with pytest.raises(te.NoPrompterKey):
        te.resolve_for(CAROL)


def test_a_key_without_an_endpoint_is_a_polite_refusal_too(monkeypatch):
    monkeypatch.setenv("PROMPTER_KEYS", json.dumps(
        {"CORP\\alice": {"key": "alice-key"}}))      # no URL / model anywhere
    with pytest.raises(engine_config.EngineUnavailable) as caught:
        te.resolve_for(ALICE)
    msg = str(caught.value)
    assert "the endpoint URL and the model name are missing" in msg
    assert "alice-key" not in msg


def test_the_shared_key_serves_testers_only_when_allowed(monkeypatch):
    _shared(monkeypatch, key="shared-key")
    with pytest.raises(te.NoPrompterKey):           # default: no fallback
        te.resolve_for(CAROL)
    monkeypatch.setenv("PROMPTER_SHARED_KEY_FALLBACK", "1")
    rp = te.resolve_for(CAROL)
    assert rp.source == te.SOURCE_SHARED_FALLBACK
    assert rp.settings.api_key == "shared-key"
    # a tester with their own key still gets THEIRS
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    assert te.resolve_for(ALICE).settings.api_key == "alice-key"


# ------------------------------------------------------------ single-user

@pytest.mark.parametrize("who", [DEV, identity.ANONYMOUS,
                                 identity.Identity("example.com", "jdoe",
                                                   "email")])
def test_single_user_hosts_use_the_shared_key(monkeypatch, who):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch, key="shared-key")
    # even a per-tester secret for the same name is not consulted
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    rp = te.resolve_for(who)
    assert rp.source == te.SOURCE_SHARED and rp.settings.api_key == "shared-key"
    assert te.register() is True
    res = engine_config.resolve_engine("gpt-shared", identity=who)
    assert res.ok and _key_of(res.model) == "shared-key"


def test_single_user_with_no_settings_gets_the_not_configured_banner(
        monkeypatch):
    monkeypatch.setenv("PROMPTER_URL", URL)         # registers, but no key
    assert te.register() is True
    res = engine_config.resolve_engine(None, identity=DEV)
    assert res.source == "unavailable"
    assert res.message == te.NOT_CONFIGURED_MESSAGE


def test_no_identity_resolves_the_current_caller(monkeypatch):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch, key="shared-key")
    monkeypatch.setenv("DEV_IDENTITY", "CORP\\alice")
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    te.register()
    # no Streamlit header outside a run: DEV_IDENTITY, single-user -> shared
    res = engine_config.resolve_engine("gpt-shared")
    assert res.ok and _key_of(res.model) == "shared-key"


# ---------------------------------------------------- no cross-user reuse

def test_no_client_is_reused_across_users(monkeypatch):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch, key=None)                  # no shared key at all
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_BOB", "bob-key")
    assert te.register() is True                    # URL alone registers
    a1 = engine_config.resolve_engine("gpt-shared", identity=ALICE).model
    b1 = engine_config.resolve_engine("gpt-shared", identity=BOB).model
    a2 = engine_config.resolve_engine("gpt-shared", identity=ALICE).model
    assert len({id(a1), id(b1), id(a2)}) == 3       # built afresh each time
    assert a1.http_client is not b1.http_client
    assert [_key_of(m) for m in (a1, b1, a2)] == \
        ["alice-key", "bob-key", "alice-key"]
    # the registered builder holds no key of its own
    builder = engine_config._MODEL_BUILDER
    cells = [c.cell_contents for c in (builder.__closure__ or ())]
    assert not any("key" in repr(c) for c in cells)


def test_identity_reaches_only_builders_that_ask_for_it():
    seen = []
    engine_config.register_model_builder(lambda mid=None: seen.append(mid)
                                         or object())
    res = engine_config.resolve_engine("m", identity=ALICE)
    assert res.ok and seen == ["m"]                 # a legacy builder is fine

    def _with_identity(model_id=None, identity=None):
        seen.append(identity)
        return object()
    engine_config.register_model_builder(_with_identity)
    assert engine_config.resolve_engine("m", identity=BOB).ok
    assert seen[-1] is BOB


# ------------------------------------------------------- the JSON mapping

def test_the_mapping_serves_testers_case_insensitively(monkeypatch):
    pytest.importorskip("langchain_openai")
    _shared(monkeypatch, key=None)
    monkeypatch.setenv("PROMPTER_KEYS", json.dumps({
        "corp\\ALICE": {"key": "alice-key", "model": "gpt-alice"},
        "bob@corp": "bob-key",                       # bare string, UPN form
        "CORP\\dave": {"model": "no-key-here"},       # ignored: no key
    }))
    a = te.resolve_for(ALICE)
    assert a.source == te.SOURCE_MAPPING and a.settings.api_key == "alice-key"
    assert a.own_model and a.settings.model == "gpt-alice"
    assert a.settings.url == URL                     # shared URL fills in
    b = te.resolve_for(BOB)
    assert b.settings.api_key == "bob-key" and b.settings.model == "gpt-shared"
    with pytest.raises(te.NoPrompterKey):
        te.resolve_for(identity.Identity("CORP", "dave", "header"))
    assert te.build_for(ALICE, "gpt-shared").model_name == "gpt-alice"


def test_the_per_tester_secret_beats_the_mapping(monkeypatch):
    _shared(monkeypatch, key="shared-key")
    monkeypatch.setenv("PROMPTER_SHARED_KEY_FALLBACK", "1")
    monkeypatch.setenv("PROMPTER_KEYS", json.dumps(
        {"CORP\\alice": "mapped-key"}))
    assert te.resolve_for(ALICE).source == te.SOURCE_MAPPING
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "secret-key")
    rp = te.resolve_for(ALICE)
    assert rp.source == te.SOURCE_TESTER and rp.settings.api_key == "secret-key"


def test_a_broken_mapping_is_ignored_without_quoting_it(monkeypatch, caplog):
    _shared(monkeypatch, key=None)
    monkeypatch.setenv("PROMPTER_KEYS",
                       '{"CORP\\\\alice": {"key": "leaky-key"')   # cut short
    with caplog.at_level(logging.DEBUG):
        assert te.key_mapping() == {}
        with pytest.raises(te.NoPrompterKey):
            te.resolve_for(ALICE)
    assert "PROMPTER_KEYS is not valid JSON" in caplog.text
    assert "leaky-key" not in caplog.text


# --------------------------------------------- never shown, never logged

def test_diagnostics_show_the_source_never_the_key(monkeypatch, caplog):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "tinyapps")
    _shared(monkeypatch, key="shared-key")
    monkeypatch.setenv("PROMPTER_API_KEY__CORP_ALICE", "alice-key")
    monkeypatch.setenv("PROMPTER_KEYS", json.dumps({"CORP\\bob": "bob-key"}))
    with caplog.at_level(logging.DEBUG):
        reports = {
            "alice": diagnostics._prompter_key_check(ALICE),
            "bob": diagnostics._prompter_key_check(BOB),
            "carol": diagnostics._prompter_key_check(CAROL),
            "dev": diagnostics._prompter_key_check(DEV),
        }
        te.register()
        engine_config.resolve_engine("gpt-shared", identity=ALICE)
        env = diagnostics._env_check()
    assert reports["alice"]["status"] == diagnostics.PASS
    assert "key source = per-tester secret" in reports["alice"]["detail"]
    assert "PROMPTER-API-KEY--CORP-ALICE" in reports["alice"]["detail"]
    assert "key source = PROMPTER-KEYS mapping" in reports["bob"]["detail"]
    assert reports["carol"]["status"] == diagnostics.WARN
    assert "No AI key is set up for **carol**" in reports["carol"]["detail"]
    assert "shared key (single-user host)" in reports["dev"]["detail"]
    text = json.dumps(reports) + env["detail"] + caplog.text
    for secret in ("alice-key", "bob-key", "shared-key"):
        assert secret not in text
    assert "PROMPTER_KEYS=set(" in env["detail"]


def test_register_needs_something_to_serve_anyone(monkeypatch):
    assert te.register() is False                    # nothing at all
    monkeypatch.setenv("PROMPTER_KEYS", json.dumps({"CORP\\a": "k"}))
    assert te.register() is True                     # the mapping alone
