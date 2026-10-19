"""The calc sub-agent's prompt carries the FIGURE rules (owner feedback
2026-09-11: calc packages lack figures).

Root cause, verified: the primary prompt held the figure doctrine, the web
app delegates every package build to the ``calc`` sub-agent, and deepagents
builds sub-agents standalone — so the agent that built the package had never
seen the rules. Worse, its prompt began with the references consultant's
LIBRARIAN preamble ("Do NOT perform engineering calculations"). These guards
pin the fix and the real tool names it points at.
"""

import pytest

pytest.importorskip("deepagents")

from funhouse_agent.deep.agent import (_CALC_DELEGATION_NUDGE, _CALC_FRAMING,
                                       _CALC_PREAMBLE, build_calc_subagent)
from funhouse_agent.deep.prompt import _PLANNING_AND_SCRATCH_SECTION


def test_calc_prompt_is_not_the_librarian():
    from funhouse_agent.reviewer import CONSULTANT_FRAMING
    spec = build_calc_subagent()
    prompt = spec["system_prompt"]
    assert not prompt.startswith(CONSULTANT_FRAMING)
    assert "REFERENCE LIBRARIAN" not in prompt
    assert "Do NOT perform engineering calculations" not in prompt
    assert not prompt.rstrip().endswith("Question:")
    assert prompt.startswith(_CALC_PREAMBLE)
    assert "CALCULATION ENGINE" in prompt


def test_calc_prompt_carries_the_figure_rules_and_real_tool_names():
    prompt = build_calc_subagent()["system_prompt"]
    assert "FIGURES ARE PART OF EVERY DELIVERABLE" in prompt
    for name in ("subsurface_profile", "render_figures", "plot_data",
                 "html_img_tag", "html_to_pdf"):
        assert name in prompt, name
    # the canned package already has figures — prefer it when one fits
    assert "PREFER the canned package" in prompt
    # the report skeleton puts the profile figure before the method
    assert prompt.index("Subsurface profile figure") < prompt.index("-> Method")
    # compactness and no-data-loss rules survived the rewrite
    assert "NO DATA LOSS" in prompt and "REAL DISK ONLY" in prompt
    assert "SOURCE DOCUMENTS" in prompt


def test_figure_tool_names_in_prompts_are_real():
    """Every method the prompts promise must be registered, so the model is
    never told to call a plotting tool that does not exist (the old primary
    bullet promised `output_path` on a 'plotting tool' for slope/FEM/p-y
    results that had no such tool)."""
    from funhouse_agent.adapters.calc_package import (
        METHOD_REGISTRY as CP)
    from funhouse_agent.adapters.profile_figure_adapter import (
        METHOD_REGISTRY as PF)
    assert {"render_figures", "html_to_pdf"} <= set(CP)
    assert {"subsurface_profile", "plot_data"} <= set(PF)
    text = _CALC_FRAMING + _PLANNING_AND_SCRATCH_SECTION
    assert "plotting tool" not in text          # the phantom tool is gone
    assert "cross_section.html" not in text


def test_primary_bullet_names_the_real_inventory():
    sec = _PLANNING_AND_SCRATCH_SECTION
    for name in ("profile_figure.subsurface_profile", "profile_figure.plot_data",
                 "calc_package.render_figures", "subsurface.plot_*"):
        assert name in sec, name


def test_primary_bullet_describes_what_the_app_really_shows():
    """The prompt promised a saved figure renders INLINE. It does not: it is a
    card UNDER the reply, interactive for plot_data / subsurface.plot_*, an
    image otherwise, and a saved Plotly HTML is a collapsed preview. Three
    places say this — here and the two figure tools' embed_notes — so they
    must agree."""
    sec = _PLANNING_AND_SCRATCH_SECTION
    assert "INLINE" not in sec
    assert "CARD UNDER" in sec
    assert "INTERACTIVE" in sec
    assert "collapsed" in sec                      # the saved-HTML caveat
    assert "markdown image link to" in sec         # still forbidden


def test_delegation_nudge_tells_primary_to_pass_figure_data():
    assert "PASS THE DATA THE FIGURES NEED" in _CALC_DELEGATION_NUDGE
    assert "layer stack" in _CALC_DELEGATION_NUDGE
    assert "draws only what it is handed" in _CALC_DELEGATION_NUDGE


def test_calc_subagent_has_the_figure_modules_in_scope():
    """The prompt points at profile_figure and calc_package; the sub-agent's
    call_agent scope must actually include them."""
    from funhouse_agent.dispatch import ANALYSIS_MODULES
    assert "profile_figure" in ANALYSIS_MODULES
    assert "calc_package" in ANALYSIS_MODULES


@pytest.mark.parametrize("env", ["GEOTECH_FIGURE_HELPER", "GEOTECH_PLOT_CODE"])
def test_a_fault_a_look_finds_is_fixed_before_handing_back(monkeypatch, env):
    """Live wave w14 (TRACE_REVIEW §8 fix 10, BACKLOG GM-190): the calc
    helper's look at a page of its own PDF found a figure it could not read
    and the PDF went back as it was. Behind the figure switches the helper
    is told to fix what a look at a figure or a page finds, then look
    again; with them off its prompt is unchanged."""
    from funhouse_agent import review_flags
    from funhouse_agent.deep.agent import _CALC_LOOK_FIXES
    for name in (review_flags.FIGURE_HELPER_ENV, review_flags.PLOT_CODE_ENV):
        monkeypatch.delenv(name, raising=False)
    assert _CALC_LOOK_FIXES not in build_calc_subagent()["system_prompt"]
    monkeypatch.setenv(env, "1")
    prompt = build_calc_subagent()["system_prompt"]
    assert _CALC_LOOK_FIXES in prompt
    assert "a page of a PDF you built" in _CALC_LOOK_FIXES
    assert "look again, at most twice" in _CALC_LOOK_FIXES
