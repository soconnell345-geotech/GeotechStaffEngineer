"""INF-67: what the document and drawing tools add to every model call's
prefix on a geotech turn with no document.

Builds the geotech page's agent offline as ``webapp/tests/page_builds.py``
does (the page as shipped, every GEOTECH_* setting unset, a fake chat model),
runs one turn, and records what the MAIN agent's first model call is sent:
the system message and the tool list, each tool in the OpenAI function form
(``convert_to_openai_tool``) as the GPT routes send it. Counted with
tiktoken's ``o200k_base`` (the GPT-4o / GPT-5 family's encoding; a stand-in
for the Anthropic routes, whose tokenizer is not public) and in characters.

Run from the repository root:  python -m module_work.inf67.measure_prefix\n(add ``--rows`` for every tool's size).
"""

from __future__ import annotations

import json
import sys

from webapp.tests import page_builds

#: The document and drawing tools the trace review counted (§2.2): the
#: planlens bridge, the app's vision tools and the drawing-IR tools.
DOC_PREFIXES = ("document_", "render_", "analyze_")
DOC_NAMES = {"open_document", "read_document", "search_document",
             "find_like", "find_quantities", "log_grid", "measure",
             "kept_readings", "annotate_document", "annotate_dxf",
             "read_pdf_text"}


def is_doc_tool(name: str) -> bool:
    return name in DOC_NAMES or name.startswith(DOC_PREFIXES)


def capture(name="geotech_default", settings=None, bind=True,
            attach=False):
    """The main agent's first model call on a one-line geotech question.
    ``bind``: the conversation's working folder bound for the turn, as the
    web app does (the gate hides nothing without one); ``attach``: a made-up
    PDF in the conversation's attachments."""
    from langchain_core.messages import AIMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool

    seen = []

    base = page_builds.fake_model()
    cls = type(base)

    class _Capture(cls):
        sent: list = []

        def bind_tools(self, tools, **kw):
            specs = []
            for t in tools:
                try:
                    specs.append(convert_to_openai_tool(t))
                except Exception:                        # noqa: BLE001
                    specs.append(t if isinstance(t, dict) else
                                 {"name": getattr(t, "name", str(t))})
            return self.model_copy(update={"sent": specs})

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.append({"messages": list(messages),
                         "tools": list(self.sent)})
            from langchain_core.outputs import ChatGeneration, ChatResult
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="done"))])

    import contextlib
    from funhouse_agent._fileio import working_dir_bound
    model = _Capture(responses=[AIMessage(content="x")])
    agent, _calls, files_dir = page_builds.build(name, model=model,
                                                 settings=settings)
    if attach:
        agent.geotech_attachments["made_up.pdf"] = b"%PDF-1.4 made up"
    with (working_dir_bound(files_dir) if bind
          else contextlib.nullcontext()):
        agent.invoke({"messages": [{"role": "user",
                                    "content": "Rankine Ka for phi = 30 deg?"}]},
                     config={"configurable": {"thread_id": "inf67"}})
    return seen[0]


def _tool_name(spec):
    fn = spec.get("function") if isinstance(spec, dict) else None
    return (fn or spec).get("name")


def report(first):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")

    def tok(s):
        return len(enc.encode(s))

    sys_msg = first["messages"][0]
    sys_text = sys_msg.content if isinstance(sys_msg.content, str) else \
        "".join(b.get("text", "") for b in sys_msg.content
                if isinstance(b, dict))
    rows = []
    for spec in first["tools"]:
        text = json.dumps(spec, separators=(",", ":"))
        rows.append((_tool_name(spec), len(text), tok(text)))
    doc = [r for r in rows if is_doc_tool(r[0])]
    tools_tok = sum(r[2] for r in rows)
    doc_tok = sum(r[2] for r in doc)
    prefix = tok(sys_text) + tools_tok
    out = {
        "tokenizer": "tiktoken o200k_base",
        "system_chars": len(sys_text), "system_tokens": tok(sys_text),
        "tools": len(rows), "tool_chars": sum(r[1] for r in rows),
        "tool_tokens": tools_tok,
        "doc_tools": len(doc), "doc_tool_chars": sum(r[1] for r in doc),
        "doc_tool_tokens": doc_tok,
        "prefix_tokens": prefix,
        "doc_share_of_prefix": round(doc_tok / prefix, 3),
        "doc_tool_rows": sorted(doc, key=lambda r: -r[2]),
        "other_tool_rows": sorted((r for r in rows if r not in doc),
                                  key=lambda r: -r[2]),
    }
    return out


if __name__ == "__main__":
    full = "--rows" in sys.argv
    for label, kw in (("no document (gate on)", {}),
                      ("document attached", {"attach": True})):
        res = report(capture(**kw))
        if not full:
            res = {k: v for k, v in res.items() if not k.endswith("rows")}
        print(f"== {label}")
        print(json.dumps(res, indent=1))
