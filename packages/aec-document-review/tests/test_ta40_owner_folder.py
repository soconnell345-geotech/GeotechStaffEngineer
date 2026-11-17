"""TA-40, from the review of TA-39: one person, one SharePoint owner folder.

A person's conversations are mirrored under
``<root>/conversations/<owner folder>/<page>/``. The owner folder was
``sanitize_folder_name(key)`` of the person's folder key, and that function
loses what tells some keys apart (``__`` becomes ``_``, dots at the ends are
dropped, the name is cut at 64 characters), so some pairs of sign-in names
shared one folder: one listing, one restore. ``owner_folder`` keeps a plain
key's folder as it was and gives every other key a folder of its own.

The same fake Graph as the other TA-40 tests. Every sign-in name, key and
file here is made up. Each test fails before the fix (its docstring says
how).
"""

import itertools
import os

import aec_document_review.core as core
import aec_document_review.sharepoint_store as sp
import aec_document_review.sharepoint_tools as tools
from tests.test_ta40_sharepoint_off_turn import (  # noqa: F401 (fixtures)
    _clean_env, _conversation, _http, _store, graph)

PAGE = "document_review"

# The keys of two sign-in names each, as ``Identity.key`` makes them
# (``<domain>__<user>``), that shared one owner folder.
PAIRS = [
    ("corp_a__b", "corp__a_b"),            # corp_a\b and corp\a_b
    ("corp__jdoe.", "corp__jdoe"),         # corp\jdoe. and corp\jdoe
    ("corp__" + "a" * 70 + "x",            # two long names that differ
     "corp__" + "a" * 70 + "y"),           # after the 64th character
    ("corp__o_brien__3f9c1a2b7d",          # a key that carries a hash, and
     "corp__o_brien_3f9c1a2b7d"),          # a name made up to spell it
    ("corp__a__b", "corp__a_b"),           # a doubled underscore in a name
    ("Corp__JDoe", "corp__jdoe"),          # SharePoint ignores case
]


def _folder(store, key):
    return store.conversations_base(key, PAGE).split("/conversations/")[1] \
        .split("/")[0]


def test_keys_that_shared_an_owner_folder_get_two(graph):
    """Each pair gets two folders (SharePoint compares names without
    case). Before the fix each pair got one."""
    store = _store()
    for one, two in PAIRS:
        assert _folder(store, one).lower() != _folder(store, two).lower(), \
            (one, two)
    # a name made up to spell another key's own folder gets another one
    for key, _other in PAIRS:
        made = _folder(store, key)
        for spelt in (made, made.replace("__", "_"),
                      made.replace("_", "__", 1), made.upper()):
            if spelt != key:
                assert _folder(store, spelt).lower() != made.lower(), \
                    (key, spelt)


def test_no_two_made_up_keys_share_an_owner_folder(graph):
    """A few thousand made-up keys, plain and odd: as many folders as keys,
    each a name SharePoint accepts, none over the length the other folder
    names keep to. Before the fix hundreds of them shared a folder."""
    domains = ["corp", "corp_a", "a.b", "a-b", "_a", "a_", "CORP", ".corp"]
    atoms = ["a", "b", ".", "-", "_", "__", "é", " ", "'", "1", "A"]
    keys = {"anonymous", "corp__a_vti_b", "corp__" + "k" * 300}
    for domain in domains:
        for n in (1, 2, 3):
            for combo in itertools.product(atoms, repeat=n):
                keys.add(f"{domain}__{''.join(combo)}")
    store = _store()
    folders = {}
    for key in sorted(keys):
        folder = _folder(store, key)
        assert folder and sp.remote_name(folder) == folder, (key, folder)
        assert len(folder) <= sp.MAX_NAME_CHARS, (key, folder)
        folders.setdefault(folder.lower(), []).append(key)
    shared = [ks for ks in folders.values() if len(ks) > 1]
    assert not shared, (f"{len(shared)} folders are shared, e.g. "
                        f"{[ks[:3] for ks in shared[:3]]}")
    assert len(keys) > 9000


def test_a_plain_key_keeps_the_folder_it_had(graph):
    """An ordinary tester's mirror does not move: a key in the plain form
    (no underscore in the domain, single ones inside the name) has the
    folder ``sanitize_folder_name`` gave it, and no owner still means no
    owner folder. (A guard: before the fix there is no ``owner_folder``.)"""
    store = _store()
    root = store.root()
    for key in ("corp__j.doe", "corp__jdoe", "corp__john_smith",
                "a-b.c__x.y-z", "corp__a.b_c-d_e9", "anonymous"):
        old = sp.sanitize_folder_name(key)
        assert sp.owner_folder(key) == old
        assert store.conversations_base(key, PAGE) == \
            f"{root}/conversations/{old}/{PAGE}"
        assert sp.listing_cache_key(key, PAGE) == \
            f"sp_remote_list:{old}:{PAGE}"
    assert sp.owner_folder(None) == "" == sp.owner_folder("  ")
    assert store.conversations_base(None, PAGE) == \
        f"{root}/conversations/{PAGE}"
    assert store.conversations_base() == f"{root}/conversations"


def _tester(tmp_path, key, tid, title):
    root = str(tmp_path / tid)
    _conversation(root, tid, title=title, owner=key)
    meta = core.load_meta(tid, root)
    meta["page"] = PAGE
    core.save_meta(tid, meta, root)
    return root


def test_one_of_a_pair_never_lists_or_restores_the_others(tmp_path, graph):
    """Two testers whose keys shared a folder each save a conversation.
    Each one's listing holds their own only; restoring the other's by its
    name finds nothing; the SharePoint tools treat the other's folder as
    private; a listing that fails for one is not the other's failure.
    Before the fix both conversations were in one folder and one list."""
    one, two = "corp_a__b", "corp__a_b"
    store = _store()
    root_1 = _tester(tmp_path, one, "AAAA1111", "Tower set review")
    root_2 = _tester(tmp_path, two, "BBBB2222", "Garage slab check")
    saved_1 = store.mirror_conversation("AAAA1111", root_1)
    saved_2 = store.mirror_conversation("BBBB2222", root_2)
    assert saved_1["errors"] == [] == saved_2["errors"]
    name_1 = os.path.basename(saved_1["folder"])
    name_2 = os.path.basename(saved_2["folder"])

    assert [e["name"] for e in store.list_remote_conversations(
        owner=one, page=PAGE)] == [name_1]
    assert [e["name"] for e in store.list_remote_conversations(
        owner=two, page=PAGE)] == [name_2]

    elsewhere = str(tmp_path / "new-host")
    os.makedirs(elsewhere)
    other = store.restore_conversation(name_2, root=elsewhere, owner=one,
                                       page=PAGE)
    assert other["status"] == "error" and other["thread_id"] is None
    assert not os.path.exists(core.conversation_dir("BBBB2222", elsewhere))
    own = store.restore_conversation(name_1, root=elsewhere, owner=one,
                                     page=PAGE)
    assert own["status"] == "restored" and own["thread_id"] == "AAAA1111"

    scope = tools._Scope([store.conversations_base(one)], one)
    assert scope.where(saved_1["folder"] + "/files/calc.pdf") == "own"
    assert scope.where(saved_2["folder"] + "/files/calc.pdf") == "private"

    def before(method, path, req):
        if method == "GET" and path.endswith(
                f"/{sp.owner_folder(one)}/{PAGE}"):
            raise _http(req.full_url, 500)

    graph.before = before
    assert store.list_remote_conversations(owner=one, page=PAGE).error
    assert store.list_remote_conversations(owner=two, page=PAGE).error is None
    assert store.listing_error(one, PAGE)
    assert store.listing_error(two, PAGE) is None
    assert sp.listing_cache_key(one, PAGE) != sp.listing_cache_key(two, PAGE)


def test_a_folder_that_moves_is_sent_again_with_a_pointer(tmp_path, graph,
                                                          monkeypatch):
    """A key that is not in the plain form had its mirror in the shared
    folder. Its next save goes to its own folder, whole, and leaves
    MOVED.txt in the old one (what the store does whenever a conversation's
    folder changes); the save after that sends nothing. The plain key of
    the pair does not move. Before the fix nothing moved."""
    mover, stays = "corp_a__b", "corp__a_b"
    store = _store()
    root_1 = _tester(tmp_path, mover, "AAAA1111", "Tower set review")
    root_2 = _tester(tmp_path, stays, "BBBB2222", "Garage slab check")
    with monkeypatch.context() as old_rule:            # as saved before
        old_rule.setattr(sp, "owner_folder",
                         lambda key: sp.sanitize_folder_name(str(key or "")))
        before_1 = store.mirror_conversation("AAAA1111", root_1)
        before_2 = store.mirror_conversation("BBBB2222", root_2)
    assert "/conversations/corp_a_b/" in before_1["folder"]
    assert "/conversations/corp_a_b/" in before_2["folder"]

    after_1 = store.mirror_conversation("AAAA1111", root_1)
    assert after_1["errors"] == []
    assert after_1["renamed_from"] == before_1["folder"]
    assert f"/conversations/{sp.owner_folder(mover)}/{PAGE}/" \
        in after_1["folder"]
    assert after_1["uploaded"] == before_1["uploaded"]
    in_drive = after_1["folder"].split("/", 1)[1]      # without the library
    assert f"{in_drive}/meta.json" in graph.files
    assert f"{in_drive}/files/calc.pdf" in graph.files
    pointer = graph.files[
        before_1["folder"].split("/", 1)[1] + "/" + sp.MOVED_NAME]
    assert after_1["folder"] in pointer.decode("utf-8")
    again = store.mirror_conversation("AAAA1111", root_1)
    assert "renamed_from" not in again and again["uploaded"] == 0

    after_2 = store.mirror_conversation("BBBB2222", root_2)
    assert "renamed_from" not in after_2 and after_2["uploaded"] == 0
    assert after_2["folder"] == before_2["folder"]
