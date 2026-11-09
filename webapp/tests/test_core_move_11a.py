"""The app side of the move into the shared core (BACKLOG INF-32, the app
split's step 11 (a); ``module_work/inf32/CHECK_STEP11A.md``): every old
import path still imports and IS the core's module, and the GSE pages fill
the core's hooks as the code they replaced did.
"""

from __future__ import annotations

import importlib
import json

import pytest

import funhouse_agent
import webapp
from aec_agent_core import aliases, hooks

ALL_ALIASES = {**funhouse_agent.CORE_ALIASES, **webapp.CORE_ALIASES}


def test_the_alias_tables_cover_the_moved_modules():
    # 16 at step 11 (a), 7 more at 11 (b) (test_core_move_11b.py), 31 more
    # at 11 (c) (test_core_move_11c.py)
    assert len(ALL_ALIASES) == 54
    for old, new in ALL_ALIASES.items():
        # only the prefix changes (plan 2.3)
        assert new == "aec_agent_core." + old.split(".", 1)[1], old
        assert aliases().get(old) == new, old


@pytest.mark.parametrize("old,new", sorted(ALL_ALIASES.items()))
def test_every_old_name_is_the_core_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)
    parent, _, leaf = old.rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is \
        importlib.import_module(new)


def test_a_patch_by_the_old_name_reaches_the_core(monkeypatch):
    monkeypatch.setattr("funhouse_agent.vision_tools.SCRATCH_DIR", "here")
    from aec_agent_core import vision_tools
    assert vision_tools.SCRATCH_DIR == "here"


def test_the_package_doors_fill_the_core_hooks():
    filled = hooks.hooks()
    assert filled["prompter.get_setting"] == \
        "webapp.tinyapps_settings:get_setting"
    assert filled["prompter.resolve_for"] == \
        "webapp.tinyapps_keys:resolve_for"
    assert filled["prompter.tester_keys_possible"] == \
        "webapp.tinyapps_keys:tester_keys_possible"
    from funhouse_agent import reference_vision as rv
    from webapp import tinyapps_keys, tinyapps_settings
    assert hooks.get_hook("prompter.get_setting") is \
        tinyapps_settings.get_setting
    assert hooks.get_hook("prompter.resolve_for") is tinyapps_keys.resolve_for
    assert hooks.get_hook("vision_tools.reference_read_roots") is \
        rv.reference_read_roots
    assert hooks.get_hook("vision_tools.read_reference_figure") is \
        rv._dispatch_read_reference_figure
    assert hooks.get_hook("vision_tools.view_worked_example_source") is \
        rv._dispatch_view_worked_example


def test_gse_reference_readers_answer_through_the_old_dispatch(monkeypatch):
    """``funhouse_agent.vision_tools.dispatch_extended_tool`` still serves the
    two readers (the v1 agent and GSE's added tools call it by those names);
    a missing argument is the reader's own refusal, so the GSE code ran."""
    from funhouse_agent import vision_tools
    out = json.loads(vision_tools.dispatch_extended_tool(
        "read_reference_figure", {}, None, {}))
    assert "figure_number" in out["error"]
    out = json.loads(vision_tools.dispatch_extended_tool(
        "view_worked_example_source", {}, None, {}))
    assert "example_id" in out["error"]


def test_the_reference_folders_are_the_ones_before_the_move():
    """reference_read_roots: GEOTECH_REFERENCES_DOCS, then the fetch cache,
    the installed library's docs and this checkout's
    geotech-references/docs (the path computed from funhouse_agent/, as
    vision_tools.py computed it before the move)."""
    import os
    from funhouse_agent import reference_docs, vision_tools
    roots = vision_tools.reference_read_roots()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(
        funhouse_agent.__file__)))
    assert os.path.abspath(reference_docs.cache_dir()) in roots
    assert os.path.join(repo, "geotech-references", "docs") in roots
