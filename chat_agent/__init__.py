"""
Chat Agent — ReAct agent for text-only LLM chat functions.

Gives any chat function with signature (prompt, system_prompt, temp) -> str
access to all 44 geotechnical Foundry agents via <tool_call> tag parsing.

Usage:
    from chat_agent import GeotechChatAgent

    agent = GeotechChatAgent(chat_fn=fh_prompter.chat, verbose=True)
    result = agent.ask("Calculate bearing capacity of a 2m footing, phi=30")
    print(result.answer)
"""

from chat_agent.agent import GeotechChatAgent, AgentResult
from chat_agent.parser import parse_response, ToolCall, ParseResult
from chat_agent.react_prompt import build_system_prompt, build_system_prompt_compact

__all__ = [
    "GeotechChatAgent",
    "AgentResult",
    "parse_response",
    "ToolCall",
    "ParseResult",
    "build_system_prompt",
    "build_system_prompt_compact",
]
