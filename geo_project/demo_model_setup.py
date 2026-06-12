"""Generate the MODEL-SETUP staged-flow walkthrough (offline, scripted model).

Drives a compiled deep agent bound to the setup tools with a SCRIPTED fake
chat model (no API key, no network) through the full staged protocol:

  geometry (create -> premature run REFUSED -> echo-back -> human gate)
  -> materials (cov_lookup-cited values, approve WITH an edit)
  -> water/loads (GWT + surcharge, re-render, gate)
  -> analysis plan (Bishop search + FOSM, review)
  -> run (gates all set -> FOS).

Outputs:
  docs/examples/model_setup_walkthrough.md
  docs/examples/model_setup_echo_back_geometry.png
  docs/examples/model_setup_echo_back_full.png

Run from the repo root:
  python -m geo_project.demo_model_setup
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # tools run in worker threads; never open a GUI

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

OUT_DIR = os.path.join("docs", "examples")
MD_PATH = os.path.join(OUT_DIR, "model_setup_walkthrough.md")


class ScriptedToolModel(GenericFakeChatModel):
    """A scripted fake chat model that tolerates bind_tools."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tc(name, args, cid):
    return {"name": name, "args": args, "id": f"call_{cid}"}


def _script():
    """The two scripted turns (turn boundary = the agent asks the human)."""
    turn1 = [
        AIMessage(content="", tool_calls=[_tc("project_new", {
            "template": "simple_slope",
            "params": {"H": 8.0, "slope_ratio": 2.0,
                       "foundation_depth": 6.0},
            "name": "Demo slope (8 m, 2H:1V, soft foundation)"}, 1)]),
        # Deliberate premature run — the gate must refuse it.
        AIMessage(content="", tool_calls=[_tc("project_run", {}, 2)]),
        AIMessage(content="", tool_calls=[_tc(
            "project_render",
            {"filename": "model_setup_echo_back_geometry.png"}, 3)]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "geometry",
            "summary_markdown": (
                "8 m high slope at 2H:1V over a 6 m foundation layer "
                "(template geometry — crest z=8, toe z=0, section bottom "
                "z=-6). Please verify the echo-back PNG and vertex table."),
        }, 4)]),
        AIMessage(content=(
            "I have set up the section from the template and rendered the "
            "echo-back. Please open the PNG, check it against your "
            "intent, and confirm the geometry stage (see the numbered "
            "questions above).")),
    ]
    turn2 = [
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "geometry",
            "summary_markdown": "Geometry as rendered.",
            "user_response": {"approved": True, "edits": {}}}, 5)]),
        AIMessage(content="", tool_calls=[_tc("cov_lookup", {
            "property": "phi", "soil_type": "sand"}, 6)]),
        AIMessage(content="", tool_calls=[
            _tc("project_patch", {"path": "stratigraphy[0].material.phi",
                                  "value": 32.0}, 7),
            _tc("project_patch", {"path": "stratigraphy[0].material.c_prime",
                                  "value": 0.0}, 8),
            _tc("project_patch", {"path": "stratigraphy[0].material.gamma",
                                  "value": 19.0}, 9),
            _tc("project_patch", {
                "path": "stratigraphy[0].material.probabilistic.phi",
                "value": {"cov": 0.079, "dist": "lognormal",
                          "source": "ISSMGE-TC304 (2021) Table 1.3 "
                                    "(sand, site-specific), mean COV"}}, 10),
            _tc("project_patch", {
                "path": "stratigraphy[1].material.strength_model",
                "value": "undrained"}, 11),
            _tc("project_patch", {"path": "stratigraphy[1].material.cu",
                                  "value": 60.0}, 12),
            _tc("project_patch", {"path": "stratigraphy[1].material.gamma",
                                  "value": 17.5}, 13),
        ]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "materials",
            "summary_markdown": (
                "Slope soil: drained sand, phi=32 deg (ASSUMED — typical "
                "compacted granular fill; site-specific COV ~7.9% per "
                "ISSMGE-TC304 Table 1.3), c'=0, gamma=19. Foundation: "
                "undrained clay, cu=60 kPa (from your boring log note), "
                "gamma=17.5."),
            "form_schema": [
                {"name": "stratigraphy[0].material.phi",
                 "label": "Sand friction angle", "type": "number",
                 "unit": "deg", "default": 32.0},
                {"name": "stratigraphy[1].material.cu",
                 "label": "Clay undrained strength", "type": "number",
                 "unit": "kPa", "default": 60.0},
            ],
            "user_response": {"approved": True,
                              "edits": {"stratigraphy[1].material.cu":
                                        55.0}}}, 14)]),
        AIMessage(content="", tool_calls=[
            _tc("project_patch", {"path": "water.gwt_points",
                                  "value": [[0.0, 5.0], [40.0, -1.0]]}, 15),
            _tc("project_patch", {"path": "loads.surcharges", "op": "append",
                                  "value": {"q": 10.0, "x_start": 0.0,
                                            "x_end": 8.0,
                                            "label": "crest traffic"}}, 16),
        ]),
        AIMessage(content="", tool_calls=[_tc(
            "project_render",
            {"filename": "model_setup_echo_back_full.png"}, 17)]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "water_loads",
            "summary_markdown": (
                "GWT from z=5 at the crest to z=-1 beyond the toe (dashed "
                "blue on the render); 10 kPa traffic surcharge over the "
                "crest 8 m; kh=0 (static). No reinforcement."),
            "user_response": {"approved": True, "edits": {}}}, 18)]),
        AIMessage(content="", tool_calls=[_tc("project_patch", {
            "path": "analyses", "op": "append",
            "value": {"type": "le", "name": "LE-bishop", "method": "bishop",
                      "n_slices": 30, "search": {"nx": 6, "ny": 6},
                      "probabilistic": {"kind": "fosm"}}}, 19)]),
        AIMessage(content="", tool_calls=[_tc("request_confirmation", {
            "stage": "analysis_plan",
            "summary_markdown": (
                "Plan: Bishop circular search (6x6 center grid, 30 "
                "slices) + FOSM reliability using the cited phi COV "
                "(7.9%, lognormal)."),
            "user_response": {"approved": True, "edits": {}}}, 20)]),
        AIMessage(content="", tool_calls=[_tc("project_run", {}, 21)]),
        AIMessage(content=(
            "All three stage gates were human-confirmed, validation is "
            "clean, and the run is complete — the Bishop FOS and FOSM "
            "reliability are in the result above. The full assumption "
            "ledger (template defaults, assumed phi with its TC304 "
            "citation) is recorded on the project document.")),
    ]
    return turn1, turn2


def _fmt_args(args, limit=200):
    s = json.dumps(args, default=str)
    return s if len(s) <= limit else s[:limit] + "...}"


def _fmt_result(text, limit=900):
    try:
        obj = json.loads(text)
    except Exception:
        return text[:limit]
    s = json.dumps(obj, indent=2, default=str)
    if len(s) > limit:
        s = s[:limit] + "\n  ... (truncated)"
    return s


def _transcript_md(messages, skip=0):
    lines = []
    for m in messages[skip:]:
        kind = type(m).__name__
        if kind == "HumanMessage":
            lines.append(f"**User:** {m.content}\n")
        elif kind == "AIMessage":
            for tc in (getattr(m, "tool_calls", None) or []):
                lines.append(f"`agent -> {tc['name']}"
                             f"({_fmt_args(tc['args'])})`\n")
            content = m.content if isinstance(m.content, str) else ""
            if content.strip():
                lines.append(f"**Agent:** {content}\n")
        elif kind == "ToolMessage":
            name = getattr(m, "name", "tool")
            lines.append(f"<details><summary>{name} result</summary>\n\n"
                         f"```json\n{_fmt_result(str(m.content))}\n```\n"
                         f"</details>\n")
    return lines


def main():
    from deepagents import create_deep_agent

    from funhouse_agent.deep.setup_agent import SETUP_SYSTEM_PROMPT
    from funhouse_agent.deep.setup_tools import (
        ProjectStore,
        make_setup_tools,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    store = ProjectStore()
    tools = make_setup_tools(store, render_dir=OUT_DIR)
    turn1, turn2 = _script()
    model = ScriptedToolModel(messages=iter(turn1 + turn2))
    agent = create_deep_agent(model=model, tools=tools,
                              system_prompt=SETUP_SYSTEM_PROMPT)

    r1 = agent.invoke({"messages": [{
        "role": "user",
        "content": ("Set up an 8 m high 2H:1V sand slope on 6 m of soft "
                    "clay, water table partway up, traffic at the crest, "
                    "and run a Bishop search with reliability.")}]})
    n1 = len(r1["messages"])
    r2 = agent.invoke({"messages": r1["messages"] + [{
        "role": "user",
        "content": ("Geometry approved — looks exactly right. For the "
                    "materials: phi 32 is fine, but use cu = 55 kPa for "
                    "the clay (CIU triaxial average), not 60.")}]})

    runs = [json.loads(m.content) for m in r2["messages"]
            if type(m).__name__ == "ToolMessage"
            and getattr(m, "name", "") == "project_run"]
    final = runs[-1]
    assert final["status"] == "complete", final
    fos = final["results"]["LE-bishop"]["FOS"]
    prob = final["results"]["LE-bishop"].get("probabilistic", {})

    md = []
    md.append("# MODEL-SETUP walkthrough — staged, human-gated (scripted "
              "offline demo)\n")
    md.append(
        "Generated by `geo_project/demo_model_setup.py` with a SCRIPTED "
        "fake model (no API calls). It demonstrates the design inversion: "
        "the agent never claims to have read geometry correctly — it "
        "echoes back a rendered cross-section and the human confirms "
        "visually before anything advances. Note the deliberately "
        "premature `project_run` in turn 1: the gate refuses it.\n")
    md.append("## Echo-back artifacts\n")
    md.append("Geometry stage:\n\n"
              "![geometry echo-back](model_setup_echo_back_geometry.png)\n")
    md.append("After water/loads (GWT, ponded toe, surcharge):\n\n"
              "![full echo-back](model_setup_echo_back_full.png)\n")
    md.append("## Turn 1 — geometry stage (and a refused early run)\n")
    md.extend(_transcript_md(r1["messages"]))
    md.append("\n## Turn 2 — gates: geometry -> materials (edit applied) "
              "-> water/loads -> plan -> run\n")
    md.extend(_transcript_md(r2["messages"], skip=n1))
    md.append("\n## Outcome\n")
    md.append(f"- Bishop critical-circle FOS: **{fos:.3f}**")
    if prob:
        md.append(f"- FOSM: beta_LN = {prob.get('beta_lognormal', 0):.2f}, "
                  f"pf(lognormal) = {prob.get('pf_lognormal', 0):.2%} "
                  f"(phi COV 7.9%, ISSMGE-TC304 Table 1.3, cited in the "
                  f"assumption ledger)")
    md.append(f"- Confirmation gates at run time: "
              f"{ {s: getattr(store.project.confirmations, s) for s in ('geometry', 'materials', 'water_loads')} }")
    md.append(f"- Clay cu after the user's approve-with-edit: "
              f"{store.project.stratigraphy[1].material.cu} kPa")
    md.append("- Assumption ledger entries: "
              + "; ".join(f"`{a.field}` = {a.value}"
                          + (f" [{a.source}]" if a.source else "")
                          for a in store.project.assumptions))
    md.append("\nTool-call ordering enforced end-to-end (store history): "
              + " -> ".join(store.history))

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {MD_PATH}")
    print(f"FOS = {fos:.3f}; gates = {store.project.confirmations}")


if __name__ == "__main__":
    main()
