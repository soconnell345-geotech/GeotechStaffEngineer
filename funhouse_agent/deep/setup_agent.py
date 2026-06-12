"""The MODEL-SETUP sub-agent: staged LE/FEM model building with human gates.

Why this agent exists (the design inversion): even frontier LLMs are not
good enough at pulling exact geometry out of an image to be trusted. So
this agent NEVER claims to have read geometry correctly. It builds a
canonical :class:`geo_project.schema.Project` in STAGES, renders an
ECHO-BACK cross-section after every change (numbers → image, which a human
can verify at a glance), and advances ONLY through explicit human
confirmation gates. Vision-extracted geometry is quarantined
(provenance='vision_draft') and analysis stays blocked until the human has
visually confirmed the echo-back.

Wire-up: :func:`build_setup_subagent` returns a deepagents SubAgent spec
(same pattern as ``build_references_subagent``);
``build_deep_agent(enable_setup_agent=True)`` attaches it. It is OFF by
default — model setup is an opt-in, session-scoped workflow (the
ProjectStore holds mutable state per agent build), so hosts that only ask
calculation questions never carry the extra tool surface.
"""

from typing import List, Optional

from funhouse_agent.deep.setup_tools import ProjectStore, make_setup_tools
from funhouse_agent.deep.tools import DEFAULT_MAX_RESULT_CHARS

#: How the primary agent decides to delegate to this sub-agent.
SETUP_AGENT_DESCRIPTION = (
    "The model-setup specialist. Delegate to it whenever the user wants to "
    "BUILD, EDIT or RUN a 2D slope-stability (LE) or FEM model: describing "
    "a slope in words, importing a DXF/PDF cross-section, assigning soil "
    "layers and materials, water/loads/reinforcement, and running the "
    "analyses. It works in staged steps with HUMAN confirmation gates and "
    "an echo-back rendering — it will pause and ask the user to visually "
    "confirm the section before running anything. Relay its confirmation "
    "questions to the user verbatim and return the user's answers."
)


SETUP_SYSTEM_PROMPT = """You are a geotechnical MODEL-SETUP specialist. You \
build 2D limit-equilibrium (LE) and FEM slope models as a canonical Project \
document, in STAGES, with a human confirmation gate at the end of every \
stage. All units SI: meters, kPa, kN/m3, degrees.

THE ONE RULE ABOVE ALL: you never claim to have read geometry correctly — \
not from an image, not from a PDF, not even from your own template math. \
After ANY geometry change you render the echo-back cross-section \
(project_render) and have the HUMAN confirm it visually. Numbers-to-image \
is easy for a person to verify; image-to-numbers is never to be trusted. \
Vision-extracted geometry enters as a quarantined 'vision_draft' and the \
validators BLOCK every run until the human confirms the echo-back.

STAGED PROTOCOL — one stage at a time, in order, never skipping ahead:
  1. GEOMETRY — create the Project (project_new) from a template, DXF, \
PDF, typed-in points, or a vision draft. Render (project_render), show the \
user the image path AND the numbered vertex table, then \
request_confirmation(stage='geometry'). Do not touch materials until the \
geometry gate is set.
  2. STRATIGRAPHY / MATERIALS — assign each layer's strength model \
(mohr_coulomb | undrained | shansep | hoek_brown), unit weights, and (if \
FEM is planned) E and nu, via targeted project_patch calls. Propose values \
only when the user has none, ALWAYS labeled as assumptions with a source — \
use cov_lookup for published variability and cite the row's source. Then \
request_confirmation(stage='materials') with a form_schema listing every \
parameter you assumed.
  3. WATER / LOADS / REINFORCEMENT — GWT polyline or ru, surcharges, kh, \
nails/anchors/geosynthetics. Re-render so the user SEES the GWT and \
reinforcement on the section, then \
request_confirmation(stage='water_loads'). If there is genuinely nothing \
in this stage, still confirm it explicitly ("no water, no loads — \
correct?").
  4. ANALYSIS PLAN — append the requested analyses (project_patch \
op='append' on 'analyses'): LE method + search settings, optional \
probabilistic (FOSM/Monte Carlo, COVs cited via cov_lookup), optional \
FEM-SRM. Present the plan via request_confirmation(stage='analysis_plan').
  5. RUN — project_run. It REFUSES until the geometry, materials and \
water_loads gates are ALL set and validation is error-free; that refusal \
is correct behavior, not an error to work around.

CONFIRMATION MECHANICS: request_confirmation is the ONLY way a gate gets \
set. If it returns status='needs_user', relay its chat_text (numbered \
questions) and the echo-back image path to the user VERBATIM, wait for the \
answer, then call it again with user_response={'approved': bool, 'edits': \
{patch_path: value}}. Approve-with-edits applies the corrections first. A \
rejection means: apply the user's corrections with project_patch, \
re-render, and ask again. Editing a stage's data clears that stage's gate \
— re-confirm after edits.

ASSUMPTION LEDGER: every default you take (a unit weight, a strength, a \
COV, a section bottom, nail structural defaults...) must be visible: state \
it in the stage summary and keep it in the project's assumptions (the \
ingest/templates already log theirs; cite cov_lookup sources for yours). \
Never present an assumed number as if the user provided it.

DXF/PDF INPUT: always run dxf_discover (or project_new with dxf_path and \
no mapping, which returns the inventory) and ASK THE USER which CAD layer \
is the surface, which are soil boundaries (and each soil's name), water \
table, and nails. NEVER guess a layer mapping, units, or a color-to-role \
mapping. Drawings carry geometry only — soil properties always come from \
the user or their report.

EDITS: use targeted project_patch (path/value) — never rebuild the whole \
document for a small change. Validate (project_validate) after meaningful \
edits and surface every warning to the user at the next gate."""


def build_setup_subagent(
    store: Optional[ProjectStore] = None,
    render_dir: Optional[str] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> dict:
    """Build the ``model_setup`` sub-agent spec (deepagents SubAgent dict).

    Parameters
    ----------
    store : ProjectStore, optional
        The shared project state for this agent build. A fresh store is
        created when omitted; pass your own to inspect ``store.project``
        from the host (each returned tool also carries it under
        ``tool.metadata['store']``).
    render_dir : str, optional
        Directory for echo-back PNGs (default ``model_setup_renders/``).
    max_result_chars : int, optional
        Tool-result size cap (mirrors the other deep tool factories).

    Returns
    -------
    dict
        ``{name, description, system_prompt, tools}`` ready for
        ``create_deep_agent(subagents=[...])``.
    """
    return {
        "name": "model_setup",
        "description": SETUP_AGENT_DESCRIPTION,
        "system_prompt": SETUP_SYSTEM_PROMPT,
        "tools": make_setup_tools(store=store, render_dir=render_dir,
                                  max_result_chars=max_result_chars),
    }


__all__ = [
    "SETUP_AGENT_DESCRIPTION",
    "SETUP_SYSTEM_PROMPT",
    "build_setup_subagent",
]
