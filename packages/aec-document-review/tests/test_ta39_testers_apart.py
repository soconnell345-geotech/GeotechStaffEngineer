"""TA-39: two different sign-in names never share an AI key or a folder.

Deployment review of Document Review, 2026-11-10 (section 2.2): the
per-tester secret's name turned every run of punctuation into one hyphen,
so ``corp\\j.doe``, ``corp\\j-doe`` and ``corp\\j_doe`` all asked Key Vault
for ``PROMPTER-API-KEY--CORP-J-DOE``; the folder key turned every character
outside its alphabet into ``_``, so ``o'brien`` and ``o_brien``, or a name
and the same name with an accent, shared one folder and one SharePoint
mirror.

Every name here is made up. Nothing touches a network or a real Key Vault:
a fake vault stands in for ``tinyapps_settings._from_key_vault``.
"""

from __future__ import annotations

import itertools
import json
import os
import re

import pytest

from aec_agent_core import engine_config
from aec_agent_core import tinyapps_engine as te
from aec_document_review import identity, profiles
from aec_document_review import tinyapps_keys as tk
from aec_document_review import tinyapps_settings as ts

URL = "https://prompter.example/api/v1/chat/completions"
E_ACUTE = chr(233)

#: Groups of sign-in names that shared a key, a folder or both.
COLLISIONS = {
    "punctuation": ["corp\\j.doe", "corp\\j-doe", "corp\\j_doe"],
    "apostrophe": ["corp\\o'brien", "corp\\o_brien", "corp\\o brien"],
    "accent": ["corp\\jos" + E_ACUTE, "corp\\jos", "corp\\jose"],
    "two accents": ["corp\\ren" + E_ACUTE, "corp\\ren" + chr(232)],
    "trimmed underscore": ["corp\\_svc", "corp\\svc", "corp\\svc_"],
    "domain or user": ["a-b\\c", "a\\b-c", "a\\b.c"],
    "the hash mark": ["corp\\o'brien", "corp\\o_brien__3210ff1719"],
}

#: Names the folder alphabet spells: ``(sign-in name, folder key before
#: TA-39)``. None of these folders may move.
ORDINARY = [
    ("CORP\\alice", "corp__alice"),
    ("CORP\\JDoe", "corp__jdoe"),
    ("corp\\j.doe", "corp__j.doe"),
    ("corp\\j-doe", "corp__j-doe"),
    ("corp\\j_doe", "corp__j_doe"),
    ("corp\\o_brien", "corp__o_brien"),
    ("Agency-1\\first.last-2", "agency-1__first.last-2"),
    ("jane.doe@agency.example", "agency.example__jane.doe"),
]

_TESTER_ENV = re.compile(r"^PROMPTER_(API_KEY|MODEL|URL)__")
_ENVS = ("APP_ENV", "IDENTITY_ENDPOINT", "KV_NAME", "PROMPTER_URL",
         "PROMPTER_MODEL", "PROMPTER_API_KEY", "PROMPTER_CA_BUNDLE",
         "PROMPTER_KEYS", "PROMPTER_SHARED_KEY_FALLBACK", "DEV_IDENTITY",
         "GEOTECH_USER_EMAIL", "WINDOWS_AUTH_HEADER")


def who(name: str, source: str = "header") -> identity.Identity:
    ident = identity.parse_principal(name, source)
    assert ident is not None, name
    return ident


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    for e in [e for e in os.environ if _TESTER_ENV.match(e)]:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_DOTENV", str(tmp_path / "absent.env"))
    ts.reset()
    engine_config.register_model_builder(None)
    yield
    ts.reset()
    engine_config.register_model_builder(None)


@pytest.fixture
def vault(monkeypatch):
    """A fake Key Vault ``{secret name: value}`` holding the shared URL and
    model; ``asked`` records every name looked up."""
    store, asked = {"PROMPTER-URL": URL, "PROMPTER-MODEL": "model-shared"}, []

    def _fake(name):
        asked.append(name)
        return store.get(name)

    monkeypatch.setenv("APP_ENV", "azure")
    monkeypatch.setenv("KV_NAME", "kv-test")
    monkeypatch.setattr(ts, "_from_key_vault", _fake)
    store["asked"] = asked
    return store


def _pairs():
    for kind, names in COLLISIONS.items():
        for a, b in itertools.combinations(names, 2):
            yield pytest.param(a, b, id=f"{kind}: {a} / {b}")


# ------------------------------------------------------------------ folders

@pytest.mark.parametrize("a, b", list(_pairs()))
def test_two_sign_in_names_never_share_a_folder(a, b, tmp_path):
    ia, ib = who(a), who(b)
    assert ia.key != ib.key
    ra = profiles.session_root(profiles.DOCUMENT_REVIEW, ia, str(tmp_path))
    rb = profiles.session_root(profiles.DOCUMENT_REVIEW, ib, str(tmp_path))
    assert ra != rb
    # the folder key is also the owner segment of the SharePoint folders
    for ident in (ia, ib):
        assert re.fullmatch(r"[a-z0-9._-]+", ident.key), ident.key


@pytest.mark.parametrize("name, key", ORDINARY)
def test_an_ordinary_names_folder_has_not_moved(name, key, tmp_path):
    ident = who(name)
    assert ident.key == key == ident.legacy_key
    assert profiles.session_root(
        profiles.DOCUMENT_REVIEW, ident, str(tmp_path)) == os.path.join(
            str(tmp_path), "users", key, "document_review")


def test_one_person_spelt_two_ways_keeps_one_folder():
    """Case, and an accent sent as one character or as a letter and a
    combining accent, are one sign-in name."""
    assert who("CORP\\O'Brien").key == who("corp\\o'brien").key
    assert who("CORP\\JOS" + chr(201)).key == who("corp\\jos" + E_ACUTE).key
    assert who("corp\\jose" + chr(0x301)).key == who("corp\\jos" + E_ACUTE).key
    assert who("jdoe@Agency.Example").key == who("AGENCY.EXAMPLE\\JDoe").key


def test_a_name_outside_the_folder_alphabet_carries_its_hash():
    ident = who("corp\\o'brien")
    assert ident.legacy_key == "corp__o_brien"          # the folder before
    assert ident.key == f"corp__o_brien__{ident.name_hash}"
    assert re.fullmatch(r"[0-9a-f]{%d}" % identity.NAME_HASH_LEN,
                        ident.name_hash)
    import hashlib
    assert ident.name_hash == hashlib.sha256(
        "corp\\o'brien".encode("utf-8")).hexdigest()[:identity.NAME_HASH_LEN]


def test_a_single_user_host_keeps_its_folder(tmp_path):
    """``DEV_IDENTITY`` and the launcher's email name one person in a
    process of their own: the folder never held their name."""
    for name, source in (("corp\\o'brien", "dev"),
                         ("jane.doe@agency.example", "email")):
        ident = who(name, source)
        assert not ident.multi_user
        assert profiles.session_root(
            profiles.DOCUMENT_REVIEW, ident, str(tmp_path)) == os.path.join(
                str(tmp_path), "pages", "document_review")
    assert identity.ANONYMOUS.key == "anonymous"
    assert identity.UNIDENTIFIED.key == "unidentified"


# --------------------------------------------------------------------- keys

@pytest.mark.parametrize("a, b", list(_pairs()))
def test_two_sign_in_names_can_be_given_two_keys(a, b, vault):
    """Each has a secret name of its own unless letters and digits spell
    it; with each provisioned, each is served its own key."""
    ia, ib = who(a), who(b)
    na, nb = tk.advertised_secret_name(ia), tk.advertised_secret_name(ib)
    assert na and nb and na != nb
    for n in (na, nb):
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,126}", n), n
    vault[na], vault[nb] = "key-a", "key-b"
    ra, rb = tk.resolve_for(ia), tk.resolve_for(ib)
    assert (ra.settings.api_key, rb.settings.api_key) == ("key-a", "key-b")
    assert ra.source == rb.source == te.SOURCE_TESTER
    assert (ra.secret_name, rb.secret_name) == (na, nb)


def test_the_exact_secret_is_asked_for_before_the_shared_name(vault):
    dot, under = who("corp\\j.doe"), who("corp\\j_doe")
    shared_name = "PROMPTER-API-KEY--CORP-J-DOE"
    assert tk.tester_secret_name("PROMPTER_API_KEY", dot) == shared_name \
        == tk.tester_secret_name("PROMPTER_API_KEY", under)
    exact = tk.tester_exact_secret_name("PROMPTER_API_KEY", under)
    assert exact == f"{shared_name}--{under.name_hash.upper()}"
    vault[shared_name] = "key-dot"            # provisioned before TA-39
    vault[exact] = "key-under"
    assert tk.resolve_for(under).settings.api_key == "key-under"
    assert tk.resolve_for(dot).settings.api_key == "key-dot"
    asked = vault["asked"]
    assert asked.index(exact) < asked.index(shared_name)
    # a model and URL of the tester's own sit beside the exact key
    vault[tk.tester_exact_secret_name("PROMPTER_MODEL", under)] = "model-u"
    vault[tk.tester_secret_name("PROMPTER_MODEL", dot)] = "model-dot"
    assert tk.resolve_for(under).settings.model == "model-u"
    assert tk.resolve_for(dot).settings.model == "model-dot"


def test_a_tester_with_a_key_today_still_has_it(vault):
    """Secrets named before TA-39 keep resolving, for every kind of name."""
    before = {"CORP\\alice": "PROMPTER-API-KEY--CORP-ALICE",
              "corp\\j.doe": "PROMPTER-API-KEY--CORP-J-DOE",
              "corp\\o'brien": "PROMPTER-API-KEY--CORP-O-BRIEN",
              "jdoe@agency.example": "PROMPTER-API-KEY--AGENCY-EXAMPLE-JDOE"}
    for i, (name, secret) in enumerate(before.items()):
        ident = who(name)
        assert tk.tester_secret_name("PROMPTER_API_KEY", ident) == secret
        vault[secret] = f"key-{i}"
        rp = tk.resolve_for(ident)
        assert rp.settings.api_key == f"key-{i}"
        assert rp.source == te.SOURCE_TESTER and rp.secret_name == secret


def test_a_name_of_letters_and_digits_has_one_secret_name(vault):
    alice = who("CORP\\alice")
    assert tk.tester_exact_secret_name("PROMPTER_API_KEY", alice) is None
    assert tk.tester_secret_names("PROMPTER_API_KEY", alice) == [
        "PROMPTER-API-KEY--CORP-ALICE"]
    vault["PROMPTER-API-KEY--CORP-ALICE"] = "alice-key"
    assert tk.resolve_for(alice).settings.api_key == "alice-key"
    # the vault is asked for nothing it was not asked for before
    assert [n for n in vault["asked"] if n.startswith("PROMPTER-API-KEY--")] \
        == ["PROMPTER-API-KEY--CORP-ALICE"]


def test_the_exact_mapping_entry_comes_before_the_shared_secret(vault):
    """``PROMPTER-KEYS`` is matched on the exact name, so an entry there
    also tells two such testers apart."""
    vault["PROMPTER-API-KEY--CORP-J-DOE"] = "key-dot"
    vault["PROMPTER-KEYS"] = json.dumps({"CORP\\j_doe": "key-under"})
    under = tk.resolve_for(who("corp\\j_doe"))
    assert under.settings.api_key == "key-under"
    assert under.source == te.SOURCE_MAPPING
    dot = tk.resolve_for(who("corp\\j.doe"))
    assert dot.settings.api_key == "key-dot" and dot.source == te.SOURCE_TESTER


def test_the_refusal_names_the_exact_secret(vault):
    under = who("corp\\j_doe")
    with pytest.raises(te.NoPrompterKey) as caught:
        tk.resolve_for(under)
    exact = tk.tester_exact_secret_name("PROMPTER_API_KEY", under)
    assert f"`{exact}`" in str(caught.value)
    ok, text = tk.key_source_report(under)
    assert not ok and exact in text
    vault[exact] = "key-under"
    ok, text = tk.key_source_report(under)
    assert ok and exact in text and "key-under" not in text


def test_a_long_name_still_has_an_exact_secret_name():
    long_one = identity.Identity("CORP", "x." + "y" * 200, "header")
    assert tk.tester_secret_name("PROMPTER_API_KEY", long_one) is None
    exact = tk.tester_exact_secret_name("PROMPTER_API_KEY", long_one)
    assert exact and len(exact) <= tk.KEY_VAULT_NAME_MAX
    assert exact.endswith("--" + long_one.name_hash.upper())
    other = identity.Identity("CORP", "x." + "y" * 201, "header")
    assert tk.tester_exact_secret_name("PROMPTER_API_KEY", other) != exact


def test_a_single_user_host_keeps_the_shared_key(monkeypatch):
    monkeypatch.setenv("PROMPTER_URL", URL)
    monkeypatch.setenv("PROMPTER_MODEL", "model-shared")
    monkeypatch.setenv("PROMPTER_API_KEY", "shared-key")
    rp = tk.resolve_for(who("corp\\o'brien", "dev"))
    assert rp.source == te.SOURCE_SHARED
    assert rp.settings.api_key == "shared-key"


def test_the_name_printer_shows_names_and_no_key(capsys, monkeypatch):
    monkeypatch.setenv("PROMPTER_API_KEY", "shared-key")
    assert tk._main(["CORP\\j_doe", "not a name"]) == 1
    out = capsys.readouterr().out
    under = who("corp\\j_doe")
    assert tk.tester_exact_secret_name("PROMPTER_API_KEY", under) in out
    assert "users/corp__j_doe/" in out and "shared-key" not in out
    assert "not a name: not a sign-in name" in out
