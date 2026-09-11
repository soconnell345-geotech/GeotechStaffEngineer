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
    assert "HTML — for the chat, not for a PDF" in sec


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
