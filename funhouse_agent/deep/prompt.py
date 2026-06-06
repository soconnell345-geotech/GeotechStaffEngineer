"""System prompt for the deepagents (v5.0) port.

Reuses :func:`funhouse_agent.system_prompt.build_system_prompt` but strips the
text-based ReAct / ``<tool_call>`` protocol sections, exactly like
``agent._build_native_system_prompt`` does for the native-tool-calling path.
deepagents binds tools through LangChain's native tool-calling, so the
``## ReAct Protocol`` / ``## Available Tools`` / ``## Rules`` blocks (which
document the ``<tool_call>`` XML format) are noise that can mislead the model.

The domain guidance, DIGGS workflow, tool discipline, and the module catalog
are all preserved.
"""

import re

from funhouse_agent.system_prompt import build_system_prompt


def build_domain_prompt(allowed_agents=None) -> str:
    """Return the domain system prompt with the ReAct XML sections stripped.

    Parameters
    ----------
    allowed_agents : iterable of str, optional
        If provided, only these modules appear in the catalog (same scoping as
        ``build_system_prompt``). Defaults to the full registry.

    Returns
    -------
    str
        The system prompt for ``create_deep_agent``: domain guidance + DIGGS
        workflow + tool discipline + module catalog, with the
        ``## ReAct Protocol`` through ``## Rules`` sections removed.
    """
    base = build_system_prompt(allowed_agents)
    # Remove "## ReAct Protocol" through the start of "## Available Modules"
    # (drops Protocol + Available Tools + Rules). Mirrors
    # agent._build_native_system_prompt's regex.
    base = re.sub(
        r"## ReAct Protocol.*?(?=## Available Modules|\Z)",
        "",
        base,
        flags=re.DOTALL,
    )
    return base.strip()


__all__ = ["build_domain_prompt"]
