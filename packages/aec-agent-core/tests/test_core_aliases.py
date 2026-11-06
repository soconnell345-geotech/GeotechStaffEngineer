"""The alias helper step 11 uses to keep old import paths working
(``aec_agent_core.install_aliases``; plan ``module_work/APP_SPLIT_PLAN.md``,
section 4, "Moves that git can follow").

Each test builds a throwaway pair of packages on disk: ``old_<n>`` (the
package a module moved out of) and ``new_<n>`` (the one it moved into).
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import pickle
import sys
import textwrap

import pytest

import aec_agent_core as core

_COUNTER = itertools.count()


@pytest.fixture
def pair(tmp_path, monkeypatch):
    """(old, new) package names on sys.path; cleaned up afterwards."""
    n = next(_COUNTER)
    old, new = f"old_pkg_{n}", f"new_pkg_{n}"

    def write(rel, text=""):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")

    write(f"{old}/__init__.py")
    write(f"{new}/__init__.py")
    write(f"{new}/tool.py", """
        VALUE = 1

        def answer():
            return VALUE

        def made_here():
            return 42
        """)
    write(f"{new}/box/__init__.py", "NAME = 'box'\n")
    write(f"{new}/box/child.py", "WHO = 'child'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield old, new, write
    core.remove_aliases()
    for name in list(sys.modules):
        if name.split(".")[0] in (old, new):
            del sys.modules[name]


def test_the_old_name_is_the_same_module_object(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    moved = importlib.import_module(f"{old}.tool")
    real = importlib.import_module(f"{new}.tool")
    assert moved is real
    assert sys.modules[f"{old}.tool"] is real


def test_from_import_and_the_parent_attribute(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    exec(f"from {old} import tool as t1\nimport {old}.tool as t2\n"
         f"assert t1 is t2", {})
    parent = sys.modules[old]
    assert parent.tool is sys.modules[f"{new}.tool"]


def test_the_module_keeps_its_own_name_spec_and_file(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    mod = importlib.import_module(f"{old}.tool")
    assert mod.__name__ == f"{new}.tool"
    assert mod.__spec__.name == f"{new}.tool"
    assert mod.__file__.endswith("tool.py") and new in mod.__file__
    # so reload still works on the real module
    assert importlib.reload(mod) is mod


def test_a_patch_by_the_old_name_reaches_the_real_module(pair, monkeypatch):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    monkeypatch.setattr(f"{old}.tool.VALUE", 7)
    real = importlib.import_module(f"{new}.tool")
    assert real.answer() == 7


def test_install_is_lazy(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    assert f"{new}.tool" not in sys.modules
    assert f"{old}.tool" not in sys.modules


def test_a_package_alias_covers_what_lies_under_it(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.box": f"{new}.box"})
    child = importlib.import_module(f"{old}.box.child")
    assert child is importlib.import_module(f"{new}.box.child")
    assert importlib.import_module(f"{old}.box").NAME == "box"


def test_the_longest_alias_wins(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.box": f"{new}.box",
                          f"{old}.box.other": f"{new}.tool"})
    assert core.resolve_alias(f"{old}.box.other") == f"{new}.tool"
    assert core.resolve_alias(f"{old}.box.child") == f"{new}.box.child"
    assert core.resolve_alias(f"{old}.nothing") is None


def test_find_spec_by_the_old_name(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    spec = importlib.util.find_spec(f"{old}.tool")
    assert spec is not None and spec.name == f"{old}.tool"


def test_a_stale_file_at_the_old_path_does_not_shadow_the_move(pair):
    old, new, write = pair
    write(f"{old}/tool.py", "STALE = True\n")
    importlib.invalidate_caches()
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    mod = importlib.import_module(f"{old}.tool")
    assert not hasattr(mod, "STALE")
    assert mod is importlib.import_module(f"{new}.tool")


def test_pickle_by_reference_uses_the_real_name(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    fn = importlib.import_module(f"{old}.tool").made_here
    assert pickle.loads(pickle.dumps(fn)) is fn
    assert new.encode() in pickle.dumps(fn)


def test_installing_the_same_pair_twice_is_harmless(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    assert core.aliases() == {f"{old}.tool": f"{new}.tool"}
    assert sys.meta_path.count(core._FINDER) == 1


@pytest.mark.parametrize("table", [
    {"a.b": "a.b"},
    {"a.b": "a.b.c"},
    {"a.b.c": "a.b"},
    {"a..b": "x"},
    {"a.b": ""},
    {"a-b.c": "x.y"},
])
def test_aliases_that_could_never_apply_are_refused(table):
    with pytest.raises(ValueError):
        core.install_aliases(table)
    assert core.aliases() == {}


def test_repointing_an_alias_is_refused(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    with pytest.raises(ValueError, match="already an alias"):
        core.install_aliases({f"{old}.tool": f"{new}.box"})


def test_an_old_name_already_imported_as_its_own_module_is_refused(pair):
    old, new, write = pair
    write(f"{old}/tool.py", "OWN = True\n")
    importlib.invalidate_caches()
    importlib.import_module(f"{old}.tool")
    with pytest.raises(ValueError, match="two copies"):
        core.install_aliases({f"{old}.tool": f"{new}.tool"})


def test_remove_aliases_stops_new_imports(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.tool": f"{new}.tool"})
    core.remove_aliases(f"{old}.tool")
    assert core.aliases() == {}
    assert core._FINDER not in sys.meta_path
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{old}.tool")


def test_a_missing_target_is_an_import_error(pair):
    old, new, _ = pair
    core.install_aliases({f"{old}.gone": f"{new}.gone"})
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{old}.gone")

