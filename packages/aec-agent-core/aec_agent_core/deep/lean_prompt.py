"""The agent library's generic prompt off a page, behind a switch (BACKLOG
DR-29, ``GEOTECH_REVIEW_LEAN_PROMPT``; :mod:`aec_agent_core.review_flags`).

WHY. deepagents 0.6 appends its generic base prompt to the system prompt of
every main agent it builds ("You are a deep agent ...", with coding-agent
advice such as "read relevant files, check existing patterns"), and its
to-do and helper middleware each add a long section about their tool
(``write_todos``; ``task``, the helpers), with tool descriptions to match
whose examples are about code, research and ordering food. A page that hands
in a prompt of its own (Document Review's) carries all of it: on the review
page's default build 5,312 of the system prompt's characters, and 9,968 in
the two tools' descriptions (deepagents 0.6.8, langchain 1.3.4; the lean
build removed in DR-39 had short texts in their place). deepagents 0.7
writes no base prompt and no helper section (its ``BASE_AGENT_PROMPT`` is a
deprecated stand-in that no prompt uses), but langchain's to-do section and
both long tool descriptions are still handed to the model.

WHAT. :class:`LeanFrameworkPrompt` is a middleware for a page's main agent.
At each model call it

- removes the agent library's base prompt from the system prompt, word for
  word, with the blank line before it;
- swaps the to-do section and the helper section of the system prompt for
  the page's short texts (the list of helpers deepagents writes after the
  helper section stays: it is the page's own helper descriptions);
- swaps the ``write_todos`` and ``task`` tool descriptions for the page's
  short texts; the ``task`` description keeps deepagents' list of helpers
  (its ``{available_agents}``).

Only the texts the INSTALLED library defines (:data:`LIBRARY_TEXTS`, read
by name: :func:`library_texts`). A text it does not define (the module
does not hold the name, as deepagents 0.7 does not hold the base prompt or
the helper section) is not required and is not cut. Every text it does
define must be found word for word where the library puts it, or nothing
is cut: an upgrade that moved or reworded one, or a model profile of the
library that replaced one, sends the call to the model exactly as it came,
and the turn's record says which text was not found (``lean_prompt`` on
``turn_end``: :func:`aec_agent_core.deep.keep_work.note_lean_prompt`).
There is never a partial cut. A tool the call does not offer has no
description to swap and is not counted missing.

It must be the main agent's LAST middleware, so that it sees the prompt and
the tools as the model gets them: on deepagents 0.7 the to-do section is
written by the to-do middleware the core attaches itself, which
:func:`aec_agent_core.deep.assembly.compile_agent` puts just ahead of a
middleware marked ``keep_last``.

The page's texts are handed in; this module holds none of a page's wording.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Mapping, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)

BASE = "base prompt"
TODO_SECTION = "to-do section"
TASK_SECTION = "task section"
TODO_TOOL = "write_todos description"
TASK_TOOL = "task description"

#: The agent library's texts this middleware cuts or swaps, as the record
#: names them: ``(label, module, name)``.
LIBRARY_TEXTS: Tuple[Tuple[str, str, str], ...] = (
    (BASE, "deepagents.graph", "BASE_AGENT_PROMPT"),
    (TODO_SECTION, "langchain.agents.middleware.todo",
     "WRITE_TODOS_SYSTEM_PROMPT"),
    (TASK_SECTION, "deepagents.middleware.subagents", "TASK_SYSTEM_PROMPT"),
    (TODO_TOOL, "langchain.agents.middleware.todo",
     "WRITE_TODOS_TOOL_DESCRIPTION"),
    (TASK_TOOL, "deepagents.middleware.subagents", "TASK_TOOL_DESCRIPTION"),
)

#: Where deepagents puts its list of helpers in the ``task`` description.
AGENTS_SLOT = "{available_agents}"


def library_texts(table: Tuple[Tuple[str, str, str], ...] = LIBRARY_TEXTS
                  ) -> Dict[str, Optional[str]]:
    """The installed library's texts by label; ``None`` for one it does not
    define: the module is missing or does not hold the name itself. A name
    a module only serves through its ``__getattr__`` counts as not defined:
    deepagents 0.7 serves ``BASE_AGENT_PROMPT`` that way, deprecated, and
    puts it in no prompt."""
    out: Dict[str, Optional[str]] = {}
    for label, module, name in table:
        try:
            value = vars(importlib.import_module(module)).get(name)
        except Exception:                              # noqa: BLE001
            value = None
        out[label] = value if isinstance(value, str) and value.strip() \
            else None
    return out


def _unbrace(text: str) -> str:
    """A ``str.format`` template's literal text (``{{`` and ``}}`` halved)."""
    return text.replace("{{", "{").replace("}}", "}")


def _task_frame(template: Optional[str]) -> Optional[Tuple[str, str]]:
    """The ``task`` description's fixed text before and after the list of
    helpers, or ``None`` when the template has no place for the list."""
    if not template or AGENTS_SLOT not in template:
        return None
    before, _, after = template.partition(AGENTS_SLOT)
    return _unbrace(before), _unbrace(after)


def _tool_name(tool) -> Optional[str]:
    return tool.get("name") if isinstance(tool, dict) \
        else getattr(tool, "name", None)


def _tool_description(tool) -> Optional[str]:
    value = tool.get("description") if isinstance(tool, dict) \
        else getattr(tool, "description", None)
    return value if isinstance(value, str) else None


def _with_description(tool, text: str):
    if isinstance(tool, dict):
        return {**tool, "description": text}
    return tool.model_copy(update={"description": text})


def _blocks(content) -> Optional[List]:
    """The system message's content as a list to edit, or ``None`` when it
    is neither text nor a list of blocks."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return list(content)
    return None


def _block_text(block) -> Optional[str]:
    if isinstance(block, str):
        return block
    if isinstance(block, dict) and isinstance(block.get("text"), str):
        return block["text"]
    return None


def _set_block_text(block, text: str):
    return text if isinstance(block, str) else {**block, "text": text}


def _find(blocks: List, text: str) -> Optional[int]:
    for i, block in enumerate(blocks):
        body = _block_text(block)
        if body is not None and text in body:
            return i
    return None


class LeanFrameworkPrompt(AgentMiddleware):
    """The agent library's base prompt out of a main agent's system prompt,
    and its to-do and helper sections and tool descriptions swapped for the
    page's short texts, all or nothing (see the module docstring).

    ``task_tool_description`` holds ``{available_agents}`` where deepagents'
    list of helpers goes. ``texts`` replaces :func:`library_texts` (tests).
    ``last`` is what the latest call did: ``{"applied": True, "cut":
    [labels], ...}`` or ``{"applied": False, "not_found": [labels]}``."""

    #: kept last by :func:`aec_agent_core.deep.assembly.compile_agent` when
    #: it attaches the to-do middleware (deepagents 0.7)
    keep_last = True

    def __init__(self, *, todo_prompt: str, todo_tool_description: str,
                 task_prompt: str, task_tool_description: str,
                 texts: Optional[Mapping[str, Optional[str]]] = None):
        super().__init__()
        self.todo_prompt = str(todo_prompt)
        self.todo_tool_description = str(todo_tool_description)
        self.task_prompt = str(task_prompt)
        self.task_tool_description = str(task_tool_description)
        self.texts = dict(library_texts() if texts is None else texts)
        self.last: Optional[dict] = None
        self._warned = False

    # -- the swap -----------------------------------------------------------
    def lean(self, system_message, tools) -> Tuple[object, list, dict]:
        """``(system_message, tools, outcome)``: the lean pair with
        ``outcome["cut"]`` the labels of the texts cut or swapped, or the
        very same pair with ``outcome["applied"]`` False and the labels of
        the texts the library defines that were not found."""
        texts = self.texts
        order = [label for label, _, _ in LIBRARY_TEXTS]
        defined = [label for label in order if texts.get(label)]
        missing: List[str] = []
        frame = _task_frame(texts.get(TASK_TOOL))
        if TASK_TOOL in defined and frame is None:
            missing.append(TASK_TOOL)
        content = getattr(system_message, "content", None)
        blocks = _blocks(content) if system_message is not None else None
        edits = [(label, new) for label, new in (
            (BASE, None), (TODO_SECTION, self.todo_prompt),
            (TASK_SECTION, self.task_prompt)) if label in defined]
        for label, _ in edits:
            if blocks is None or _find(blocks, texts[label]) is None:
                missing.append(label)
        tools = list(tools or [])
        swaps: Dict[str, str] = {}
        for tool in tools:
            name = _tool_name(tool)
            desc = _tool_description(tool)
            if name == "write_todos" and TODO_TOOL in defined:
                if desc != texts[TODO_TOOL]:
                    missing.append(TODO_TOOL)
                swaps[name] = TODO_TOOL
            if name == "task" and TASK_TOOL in defined \
                    and TASK_TOOL not in missing:
                if not (desc is not None
                        and len(desc) >= len(frame[0]) + len(frame[1])
                        and desc.startswith(frame[0])
                        and desc.endswith(frame[1])):
                    missing.append(TASK_TOOL)
                swaps[name] = TASK_TOOL
        if missing:
            return system_message, tools, {
                "applied": False,
                "not_found": sorted(set(missing), key=order.index)}

        new_content = content
        before = after = 0
        if edits:                  # every one found, so ``blocks`` is a list
            before = sum(len(_block_text(b) or "") for b in blocks)
            for label, new in edits:
                old = texts[label]
                i = _find(blocks, old)
                body = _block_text(blocks[i])
                if new is None:                # the base prompt: cut out
                    cut = ("\n\n" + old) if ("\n\n" + old) in body else old
                    body = body.replace(cut, "", 1)
                else:
                    body = body.replace(old, new, 1)
                blocks[i] = _set_block_text(blocks[i], body)
            if isinstance(content, str):
                new_content = blocks[0]
            else:
                new_content = [b for b in blocks if _block_text(b) is None
                               or _block_text(b).strip()]
            after = sum(len(_block_text(b) or "") for b in (
                [new_content] if isinstance(new_content, str)
                else new_content))
        tool_cut = 0
        out_tools = []
        for tool in tools:
            name = _tool_name(tool)
            if swaps.get(name) == TODO_TOOL:
                new = self.todo_tool_description
            elif swaps.get(name) == TASK_TOOL:
                desc = _tool_description(tool)
                helpers = desc[len(frame[0]):len(desc) - len(frame[1])]
                new = self.task_tool_description.replace(AGENTS_SLOT, helpers)
            else:
                out_tools.append(tool)
                continue
            tool_cut += len(_tool_description(tool)) - len(new)
            out_tools.append(_with_description(tool, new))
        cut = [label for label in order
               if label in dict(edits) or label in swaps.values()]
        lean_message = system_message if not edits \
            else system_message.model_copy(update={"content": new_content})
        return lean_message, out_tools, {
            "applied": True, "cut": cut, "system_chars_cut": before - after,
            "tool_chars_cut": tool_cut}

    def _lean_request(self, request):
        sm = getattr(request, "system_message", None)
        message, tools, outcome = self.lean(sm, getattr(request, "tools",
                                                        None))
        self._note(outcome)
        if not outcome["applied"]:
            return request
        return request.override(system_message=message, tools=tools)

    def _note(self, outcome: dict) -> None:
        self.last = outcome
        try:
            from aec_agent_core.deep.keep_work import note_lean_prompt
            note_lean_prompt(outcome)
        except Exception:                              # noqa: BLE001
            pass
        if not outcome["applied"] and not self._warned:
            self._warned = True
            logger.warning(
                "GEOTECH_REVIEW_LEAN_PROMPT is set but the prompt was left "
                "as it is: not found word for word in the installed agent "
                "library or the prompt: %s", ", ".join(outcome["not_found"]))

    # -- the middleware -----------------------------------------------------
    def wrap_model_call(self, request, handler):
        return handler(self._lean_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._lean_request(request))


__all__ = ["LeanFrameworkPrompt", "library_texts", "LIBRARY_TEXTS",
           "AGENTS_SLOT", "BASE", "TODO_SECTION", "TASK_SECTION",
           "TODO_TOOL", "TASK_TOOL"]
