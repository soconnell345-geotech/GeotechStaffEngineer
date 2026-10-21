"""BACKLOG EV-33: the Foundry model split, measured live on the API credit.

On Foundry the main agent runs on Opus 5.5 and the references helper on
Sonnet 5.5 (INF-41, the owner's choice of 2026-10-20). This module runs the
same geotech-page questions under two ARMS, through the app's own wiring
(``webapp.helper_models`` -> ``build_deep_agent(subagent_models=...)``):

* ``opus_all``: the main agent and every helper on Opus 5.5;
* ``split``: the main agent on Opus 5.5, the references helper on Sonnet 5.5.

A flow runs once in each arm, the arms one after the other, in the opposite
order on alternate flows: both arms send the main agent the same prompt
prefix, so whichever runs second reads the other's prompt cache, and
alternating keeps that from always favouring one arm. Each arm's scenario is
``<flow id>@<arm>`` in the wave's folder, so a stopped wave resumes with the
same ``--wave`` (a finished scenario is never paid for twice).

Before any model call the scenario checks its wiring
(:func:`wiring_findings`): the main model, and each helper on the model the
arm names, read off the session's resolved helpers AND the compiled agent's
``geotech_build["helper_models"]``. A mismatch ends the scenario unpaid.

Each flow carries a ``truth`` block (what a right answer contains):

``helpers``
    the helpers the question should make work (``[]`` for a control);
``key``
    the answer's key number: ``name``, ``near`` (a regex for the words it
    follows), ``ranges`` (``[[lo, hi], ...]``, any one right, e.g. one per
    unit), ``units`` (optional regex the number must be followed by),
    ``value`` (the reference value in words) and ``source`` (how it was
    worked out, at $0);
``methods`` / ``methods_min``
    ``[{"name", "any": [regexes]}]``, how many the answer must name;
``cites`` / ``cites_needed``
    ``[{"ref": regex for the document, "any": [regexes for its pages,
    figures, tables, equations or sections]}]``; a cite is right when the
    answer names the document and one of its locators.

:func:`write_results` writes ``RESULTS.md`` and ``results.json`` beside the
wave: per flow and arm the key number and method, the cited pages and
whether they are right, the references helper's calls, tokens and cost per
agent and the wall time; then a summary per arm. One run per flow and arm
is a measurement, not a verdict: the full traces are read before any
conclusion (the owner's rule, CLAUDE.md "REVIEW EVERY MODEL RUN IN FULL").
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: What the model list calls each model (``GEOTECH_PROMPTER_MODELS`` in an
#: arms wave; the Foundry list uses the same labels).
MODEL_LABELS = {
    "claude-opus-5-5": "Opus 5.5",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-haiku-5-5": "Haiku 5.5",
}

#: The arms: the main agent's model and ``GEOTECH_HELPER_MODELS`` (what the
#: app reads after the host's defaults; a Tiny Apps host has none).
ARMS: Dict[str, dict] = {
    "opus_all": {
        "main": "claude-opus-5-5",
        "helpers": {"references": "main"},
        "about": "Opus 5.5 main agent; the references helper and every "
                 "other helper on the main agent's model",
    },
    "split": {
        "main": "claude-opus-5-5",
        "helpers": {"references": "Sonnet 5.5"},
        "about": "Opus 5.5 main agent; the references helper on Sonnet 5.5 "
                 "(Foundry's host default, INF-41); every other helper on "
                 "the main agent's model",
    },
}
DEFAULT_ARMS = ("opus_all", "split")

#: Where an arm's scenario id joins the flow id and the arm.
SEP = "@"

#: The helper the arms differ in.
REFERENCES = "references"


class ArmWiringError(RuntimeError):
    """An arm's agent was not built on the models the arm names."""


def label_id(label: str) -> Optional[str]:
    """The model id a label (or an id) names, else ``None``."""
    for mid, lab in MODEL_LABELS.items():
        if label in (mid, lab):
            return mid
    return None


def arm(name: str) -> dict:
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]


def arm_models(names: Iterable[str]) -> List[str]:
    """Every model id the arms use, the first arm's main model first."""
    out: List[str] = []
    for n in names:
        spec = arm(n)
        for mid in [spec["main"]] + [label_id(v) for v in
                                     spec["helpers"].values()]:
            if mid and mid not in out:
                out.append(mid)
    return out


def arm_env(name: str) -> Dict[str, str]:
    """The arm's ``GEOTECH_HELPER_MODELS``."""
    spec = arm(name)
    return {"GEOTECH_HELPER_MODELS": ",".join(
        f"{h}={m}" for h, m in spec["helpers"].items())}


def expected_helper_models(name: str) -> Dict[str, Optional[str]]:
    """``{helper: model id}`` the arm puts a helper on, ``None`` for the
    main agent's model."""
    spec = arm(name)
    out: Dict[str, Optional[str]] = {}
    for h, m in spec["helpers"].items():
        mid = None if str(m).lower() == "main" else label_id(m)
        out[h] = None if mid == spec["main"] else mid
    return out


def scenario_id(flow_id: str, arm_name: str) -> str:
    return f"{flow_id}{SEP}{arm_name}"


def split_scenario_id(sid: str) -> Tuple[str, Optional[str]]:
    head, sep, tail = str(sid).rpartition(SEP)
    return (head, tail) if sep else (str(sid), None)


def arm_flow(flow: dict, arm_name: str) -> dict:
    """The flow as one arm runs it: its id ``<flow>@<arm>``, the arm's env
    over the flow's own, and the arm named for the wiring check."""
    spec = arm(arm_name)
    out = json.loads(json.dumps(flow))
    out["id"] = scenario_id(flow["id"], arm_name)
    out["flow"] = flow["id"]
    out["arm"] = arm_name
    out["main_model"] = spec["main"]
    out["env"] = dict(flow.get("env") or {}, **arm_env(arm_name))
    return out


def order(flows: List[dict], arms: List[str]) -> List[Tuple[dict, str]]:
    """``(flow, arm)`` pairs: each flow's arms back to back, reversed on
    every other flow (the second arm reads the first's prompt cache)."""
    out = []
    for i, f in enumerate(flows):
        seq = list(arms) if i % 2 == 0 else list(reversed(arms))
        out += [(f, a) for a in seq]
    return out


def _model_name(model) -> Optional[str]:
    for attr in ("model", "model_name"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def wiring_findings(session, arm_name: str) -> List[dict]:
    """Whether the session's agent runs on the arm's models: the main
    model, and each helper the arm names, both as the app resolved it
    (``ss.helper_engines``) and as the compiled agent records it
    (``geotech_build["helper_models"]``). ``[]`` when all agree."""
    from live_smoke import detectors as det
    spec = arm(arm_name)
    ss = session.ss
    out: List[dict] = []

    def bad(msg, **kw):
        out.append(det.finding("arms", "arm_wiring", "high",
                               f"arm {arm_name}: {msg}", **kw))

    if ss.agent is None:
        bad(f"no agent was built ({ss.agent_error})")
        return out
    if ss.model != spec["main"]:
        bad(f"the main model is {ss.model!r}, not {spec['main']!r}")
    main_name = _model_name(getattr(ss.engine, "model", None))
    if main_name != spec["main"]:
        bad(f"the main engine's model is {main_name!r}")
    engines = {h.name: h for h in ss.helper_engines or []}
    built = (getattr(ss.agent, "geotech_build", None) or {}).get(
        "helper_models") or {}
    for helper, want in expected_helper_models(arm_name).items():
        h = engines.get(helper)
        if want is None:
            if h is not None and h.own_model:
                bad(f"{helper} resolved to its own model {h.model_id!r}; "
                    "the arm puts it on the main model")
            if helper in built:
                bad(f"the agent built {helper} on {built[helper]!r}; the "
                    "arm puts it on the main model")
            continue
        if h is None or not h.own_model or h.model_id != want:
            bad(f"{helper} resolved to "
                f"{(h.describe() if h else 'nothing')!r}, not {want!r}",
                note=getattr(h, "note", ""))
        if built.get(helper) != want:
            bad(f"the agent built {helper} on {built.get(helper)!r}, not "
                f"{want!r}")
    return out


def wiring_record(session) -> dict:
    """What the session's agent runs on, for the scenario's record."""
    ss = session.ss
    return {"main": ss.model,
            "main_engine": _model_name(getattr(ss.engine, "model", None)),
            "helpers": [h.describe() for h in ss.helper_engines or []],
            "built": (getattr(ss.agent, "geotech_build", None) or {}).get(
                "helper_models") or {}}


# ---------------------------------------------------------------------------
# Reading an answer against the truth
# ---------------------------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PAGE = re.compile(r"\b(?:pp?\.|pages?)\s*([A-Z]?\d+(?:[-–.]\d+)*"
                   r"(?:\s*(?:to|[-–])\s*[A-Z]?\d+(?:[-–.]\d+)*)?)",
                   re.IGNORECASE)
_LOCATOR = re.compile(r"\b(Fig(?:ure)?|Table|Eq(?:uation)?|Sec(?:tion)?|"
                      r"§)\.?\s*([A-Z]?\d+(?:[-.]\d+)*[a-z]?)",
                      re.IGNORECASE)


#: Sub- and superscript digits, signs and letters folded to plain ones, so
#: "K₀" reads "K0" and "m²" reads "m2" (EV-34: F70 scored K₀ as missing).
_SCRIPT_FOLD = {ord(c): p for c, p in (
    list(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    + list(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    + list(zip("₊₋⁺⁻", "+-+-"))
    + list(zip("ₐₑₒₓₕₖₗₘₙₚₛₜ",
               "aeoxhklmnpst"))
    + list(zip("ᵢⁱⁿ", "iin")))}


def _plain(text: str) -> str:
    for a, b in (("‐", "-"), ("‑", "-"), ("‒", "-"),
                 ("–", "-"), ("—", "-"), ("−", "-"),
                 ("′", "'"), ("’", "'"), (" ", " "),
                 (" ", " "), (" ", " "), ("**", "")):
        text = text.replace(a, b)
    return text.translate(_SCRIPT_FOLD)


def _search(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


def key_number(answer: str, key: dict, window: int = 200) -> dict:
    """The answer's key number: every number within ``window`` characters
    after a match of ``key["near"]`` (followed by ``key["units"]`` when
    given) is a candidate; ``value`` is the first candidate inside one of
    ``key["ranges"]``, else the first candidate."""
    text = _plain(answer or "")
    ranges = key.get("ranges") or ([key["range"]] if key.get("range")
                                   else [])
    cands: List[float] = []
    for m in re.finditer(key.get("near") or ".", text, re.IGNORECASE):
        seg = text[m.end():m.end() + window]
        for n in _NUM.finditer(seg):
            if key.get("units"):
                after = seg[n.end():n.end() + 12].lstrip()
                if not re.match(key["units"], after, re.IGNORECASE):
                    continue
            try:
                v = float(n.group(0).replace(",", "").replace("−", "-"))
            except ValueError:
                continue
            if v not in cands:
                cands.append(v)
    hit = next((v for v in cands for lo, hi in ranges if lo <= v <= hi),
               None)
    return {"name": key.get("name"), "value": hit if hit is not None
            else (cands[0] if cands else None),
            "in_range": hit is not None, "candidates": cands[:10],
            "ranges": ranges, "truth": key.get("value")}


def methods_named(answer: str, truth: dict) -> dict:
    text = _plain(answer or "")
    named = [m["name"] for m in truth.get("methods") or []
             if any(_search(p, text) for p in m.get("any") or [])]
    need = int(truth.get("methods_min", 1 if truth.get("methods") else 0))
    return {"named": named, "need": need, "ok": len(named) >= need}


def cited(answer: str) -> List[str]:
    """The page, figure, table, equation and section references an answer
    makes, as written (first 30)."""
    text = _plain(answer or "")
    out: List[str] = []
    for m in _PAGE.finditer(text):
        s = f"p. {m.group(1).strip()}"
        if s not in out:
            out.append(s)
    for m in _LOCATOR.finditer(text):
        s = f"{m.group(1)} {m.group(2)}"
        if s not in out:
            out.append(s)
    return out[:30]


def cites_check(answer: str, truth: dict) -> dict:
    text = _plain(answer or "")
    rows = []
    for c in truth.get("cites") or []:
        named = _search(c["ref"], text)
        hits = [p for p in c.get("any") or [] if _search(p, text)]
        rows.append({"ref": c["ref"], "named": named, "locators": hits,
                     "right": bool(named and hits)})
    need = int(truth.get("cites_needed", 1 if rows else 0))
    right = sum(1 for r in rows if r["right"])
    return {"rows": rows, "right": right, "need": need, "ok": right >= need}


# ---------------------------------------------------------------------------
# One scenario, read off disk
# ---------------------------------------------------------------------------

def _load(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _activity(scen_dir: str) -> List[dict]:
    out = []
    try:
        with open(os.path.join(scen_dir, "activity.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _brief(args: Any) -> str:
    if isinstance(args, dict):
        if "agent_name" in args or "method" in args:
            p = args.get("parameters") or args.get("params") or {}
            return (f"{args.get('agent_name')}.{args.get('method')} "
                    f"{json.dumps(p, ensure_ascii=False)[:140]}")
        return json.dumps(args, ensure_ascii=False)[:180]
    return str(args)[:180]


def helper_calls(acts: List[dict], helper: str = REFERENCES) -> dict:
    """The helper's work in a scenario's activity: the briefs the main
    agent sent it (``task`` calls), its tool calls and its model calls."""
    briefs = [str((a.get("args") or {}).get("description") or "")[:300]
              for a in acts if a.get("event") == "tool_start"
              and a.get("name") == "task"
              and (a.get("args") or {}).get("subagent_type") == helper]
    tools = [{"name": a.get("name"), "args": _brief(a.get("args"))}
             for a in acts if a.get("event") == "tool_start"
             and a.get("agent") == helper]
    models = sum(1 for a in acts if a.get("event") == "model_end"
                 and a.get("agent") == helper)
    return {"briefs": briefs, "tools": tools, "model_calls": models,
            "used": bool(briefs or tools or models)}


def _agent_costs(by_agent: dict) -> Dict[str, dict]:
    """``summary.json``'s ``by_agent`` folded to one row per agent (a
    vision side call under ``<agent> (vision)``), each with its model(s)."""
    out: Dict[str, dict] = {}
    for key, s in (by_agent or {}).items():
        agent = s.get("agent") or key.partition(" on ")[0]
        row = out.setdefault(agent, {"usd": 0.0, "calls": 0, "prompt": 0,
                                     "cache_read": 0, "cache_write_5m": 0,
                                     "output": 0, "models": []})
        row["usd"] += float(s.get("usd") or 0.0)
        for k in ("calls", "prompt", "cache_read", "cache_write_5m",
                  "output"):
            row[k] += int(s.get(k) or 0)
        if s.get("model") and s["model"] not in row["models"]:
            row["models"].append(s["model"])
    for row in out.values():
        row["usd"] = round(row["usd"], 6)
    return out


def analyse_scenario(scen_dir: str, flow: dict, arm_name: str) -> dict:
    """One ``<flow>@<arm>`` scenario against the flow's truth."""
    summary = _load(os.path.join(scen_dir, "summary.json"), {}) or {}
    dj = _load(os.path.join(scen_dir, "detectors.json"), {}) or {}
    turns = dj.get("turns") or []
    final = str((turns[-1] if turns else {}).get("final") or "")
    errors = [t.get("error") for t in turns if t.get("error")]
    truth = flow.get("truth") or {}
    acts = _activity(scen_dir)
    refs = helper_calls(acts)
    want_refs = REFERENCES in (truth.get("helpers") or [])
    row = {
        "flow": flow["id"], "arm": arm_name, "ran": bool(summary),
        "incomplete": bool(summary.get("incomplete")),
        "error": errors[-1] if errors else None,
        "key": key_number(final, truth["key"]) if truth.get("key") else None,
        "methods": methods_named(final, truth),
        "cited": cited(final),
        "cites": cites_check(final, truth),
        "references": refs,
        "references_expected": want_refs,
        "references_as_expected": refs["used"] == want_refs,
        "usd": float(summary.get("usd") or 0.0),
        "seconds": summary.get("seconds"),
        "model_calls": summary.get("model_calls"),
        "tokens": summary.get("tokens") or {},
        "by_agent": _agent_costs(summary.get("by_agent") or {}),
        "wiring": summary.get("wiring"),
        "answer_chars": len(final),
        "answer_head": final[:600],
    }
    return row


def analyse_wave(wave_dir: str, flows: List[dict],
                 arms: Iterable[str] = DEFAULT_ARMS) -> dict:
    """Every ``<flow>@<arm>`` scenario on disk for these flows and arms,
    and a summary per arm."""
    arms = list(arms)
    rows = []
    for f in flows:
        for a in arms:
            d = os.path.join(wave_dir, scenario_id(f["id"], a))
            if os.path.isfile(os.path.join(d, "summary.json")):
                rows.append(analyse_scenario(d, f, a))
            else:
                rows.append({"flow": f["id"], "arm": a, "ran": False})
    per_arm = {}
    for a in arms:
        mine = [r for r in rows if r["arm"] == a and r.get("ran")]
        agents: Dict[str, float] = {}
        tok = {"prompt": 0, "cache_read": 0, "cache_write_5m": 0,
               "output": 0}
        for r in mine:
            for ag, s in (r.get("by_agent") or {}).items():
                agents[ag] = round(agents.get(ag, 0.0) + s["usd"], 6)
            for k in tok:
                tok[k] += int((r.get("tokens") or {}).get(k) or 0)
        per_arm[a] = {
            "about": arm(a)["about"],
            "flows_run": len(mine),
            "incomplete": sum(1 for r in mine if r.get("incomplete")),
            "usd": round(sum(r["usd"] for r in mine), 6),
            "usd_by_agent": agents,
            "seconds": round(sum(float(r.get("seconds") or 0)
                                 for r in mine), 1),
            "model_calls": sum(int(r.get("model_calls") or 0)
                               for r in mine),
            "tokens": tok,
            "key_right": sum(1 for r in mine if (r.get("key") or {}).get(
                "in_range")),
            "key_total": sum(1 for r in mine if r.get("key")),
            "methods_ok": sum(1 for r in mine if r["methods"]["need"]
                              and r["methods"]["ok"]),
            "methods_total": sum(1 for r in mine if r["methods"]["need"]),
            "cites_ok": sum(1 for r in mine if r["cites"]["need"]
                            and r["cites"]["ok"]),
            "cites_total": sum(1 for r in mine if r["cites"]["need"]),
            "references_used": sum(1 for r in mine
                                   if r["references"]["used"]),
            "references_as_expected": sum(
                1 for r in mine if r["references_as_expected"]),
        }
    return {"arms": per_arm, "rows": rows,
            "flows": [f["id"] for f in flows]}


def _usd(v) -> str:
    return f"${float(v or 0):.3f}"


def _key_cell(r: dict) -> str:
    k = r.get("key")
    if not k:
        return "-"
    v = k.get("value")
    shown = "none found" if v is None else f"{v:g}"
    return f"{shown} ({'in range' if k['in_range'] else 'OUT of range'})"


def render_results(res: dict, flows: List[dict], wave: str,
                   meta: Optional[dict] = None) -> str:
    """``RESULTS.md`` for an arms wave."""
    meta = meta or {}
    fl = {f["id"]: f for f in flows}
    arms = list(res["arms"])
    L = [f"# EV-33 model split: wave {wave}", "",
         "One run per flow and arm. This is a measurement, not a verdict: "
         "read every run's full record (`<flow>@<arm>/activity.jsonl`, "
         "`transcript.md`) before concluding anything (CLAUDE.md, \"REVIEW "
         "EVERY MODEL RUN IN FULL\").", ""]
    code = (meta.get("code") or {}).get("app") or {}
    if code:
        L.append(f"Code: app {str(code.get('commit'))[:10]} on "
                 f"{code.get('branch')}"
                 + (" (DIRTY)" if code.get("dirty") else "") + ".")
    if meta.get("fake"):
        L.append("**Scripted fake models (dry run): every number below is "
                 "the harness checking itself, not a measurement.**")
    L += ["", "## Arms", ""]
    for a in arms:
        L.append(f"- `{a}`: {res['arms'][a]['about']}")
    L += ["", "Images (chart read-offs, worked-example pages) are read by "
          "the vision tools, which are built on the MAIN model: Opus 5.5 "
          "in both arms.", "", "## Arm summary", "",
          "| arm | flows | USD | USD by agent | seconds | model calls | "
          "key number right | methods named | cites right | references "
          "used (as expected) | cache read / write tokens |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = res["arms"][a]
        ag = ", ".join(f"{k} {_usd(v)}" for k, v in
                       sorted(s["usd_by_agent"].items())) or "-"
        L.append(f"| {a} | {s['flows_run']}"
                 + (f" ({s['incomplete']} incomplete)" if s["incomplete"]
                    else "")
                 + f" | {_usd(s['usd'])} | {ag} | {s['seconds']:.0f} | "
                 f"{s['model_calls']} | {s['key_right']}/{s['key_total']} | "
                 f"{s['methods_ok']}/{s['methods_total']} | {s['cites_ok']}/"
                 f"{s['cites_total']} | {s['references_used']} "
                 f"({s['references_as_expected']}/{s['flows_run']}) | "
                 f"{s['tokens']['cache_read']:,} / "
                 f"{s['tokens']['cache_write_5m']:,} |")
    if len(arms) == 2 and all(res["arms"][a]["flows_run"] for a in arms):
        a, b = arms
        ua, ub = res["arms"][a]["usd"], res["arms"][b]["usd"]
        if ua:
            L += ["", f"`{b}` cost {_usd(ub)} against `{a}`'s {_usd(ua)}: "
                  f"{(ub - ua) / ua * 100:+.0f} %."]
    L += ["", "## Per flow and arm", "",
          "| flow | arm | key number | methods | cites right | references "
          "helper (model calls / tools) | USD (primary / references / "
          "other) | seconds |", "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if not r.get("ran"):
            L.append(f"| {r['flow']} | {r['arm']} | not run | | | | | |")
            continue
        ag = r.get("by_agent") or {}
        prim = sum(v["usd"] for k, v in ag.items() if k == "primary")
        refs = sum(v["usd"] for k, v in ag.items()
                   if k == REFERENCES)
        other = sum(v["usd"] for k, v in ag.items()
                    if k not in ("primary", REFERENCES))
        rc = r["references"]
        L.append(f"| {r['flow']} | {r['arm']} | {_key_cell(r)} | "
                 f"{', '.join(r['methods']['named']) or '-'}"
                 f"{'' if r['methods']['ok'] else ' (too few)'} | "
                 f"{r['cites']['right']}/{len(r['cites']['rows'])} | "
                 f"{rc['model_calls']} / {len(rc['tools'])}"
                 f"{'' if r['references_as_expected'] else ' (NOT as expected)'}"
                 f" | {_usd(prim)} / {_usd(refs)} / {_usd(other)} | "
                 f"{r.get('seconds')} |")
    L += ["", "## Each flow", ""]
    for fid in res["flows"]:
        f = fl.get(fid) or {}
        t = f.get("truth") or {}
        L += [f"### {fid}", "", f"{f.get('title', '')}", ""]
        if t.get("key"):
            k = t["key"]
            L.append(f"- Truth, {k.get('name')}: {k.get('value')} "
                     f"(ranges {k.get('ranges')}); worked out by "
                     f"{k.get('source')}.")
        if t.get("method"):
            L.append(f"- Method: {t['method']}")
        if t.get("cites"):
            L.append("- Should cite: " + "; ".join(
                f"/{c['ref']}/ at one of {c.get('any')}"
                for c in t["cites"]))
        L.append(f"- Helpers it should use: {t.get('helpers') or 'none'}")
        for r in [x for x in res["rows"] if x["flow"] == fid]:
            if not r.get("ran"):
                L += ["", f"**{r['arm']}**: not run."]
                continue
            L += ["", f"**{r['arm']}** ({_usd(r['usd'])}, "
                  f"{r.get('seconds')} s"
                  + (", INCOMPLETE" if r.get("incomplete") else "")
                  + (f", error: {str(r['error'])[:200]}" if r.get("error")
                     else "") + ")", ""]
            if r.get("wiring"):
                w = r["wiring"]
                L.append(f"- Ran on: main {w.get('main_engine')}; "
                         f"helpers {w.get('helpers')}")
            if r.get("key"):
                k = r["key"]
                L.append(f"- Key number: {_key_cell(r)}; candidates "
                         f"{k['candidates']}")
            L.append(f"- Methods named: {r['methods']['named'] or 'none'} "
                     f"(needs {r['methods']['need']})")
            L.append(f"- Cited: {', '.join(r['cited']) or 'nothing'}")
            for c in r["cites"]["rows"]:
                L.append(f"  - /{c['ref']}/: named {c['named']}, locators "
                         f"{c['locators'] or 'none'} -> "
                         f"{'right' if c['right'] else 'not right'}")
            rc = r["references"]
            L.append(f"- References helper: {len(rc['briefs'])} brief(s), "
                     f"{rc['model_calls']} model calls, "
                     f"{len(rc['tools'])} tool calls")
            for b in rc["briefs"][:3]:
                L.append(f"  - brief: {b[:200]}")
            for tc in rc["tools"][:12]:
                L.append(f"  - {tc['name']}: {tc['args'][:160]}")
            for ag, s in sorted((r.get("by_agent") or {}).items()):
                L.append(f"- {ag} on {', '.join(s['models'])}: "
                         f"{_usd(s['usd'])}, {s['calls']} calls, "
                         f"{s['prompt']:,} prompt tokens "
                         f"({s['cache_read']:,} cache read, "
                         f"{s['cache_write_5m']:,} cache write), "
                         f"{s['output']:,} output")
            L += ["", "> " + r["answer_head"].replace("\n", "\n> ")[:900]]
        L.append("")
    return "\n".join(L) + "\n"


def write_results(wave_dir: str, flows: List[dict],
                  arms: Iterable[str] = DEFAULT_ARMS) -> str:
    """``RESULTS.md`` and ``results.json`` beside the wave; the path of
    ``RESULTS.md``."""
    res = analyse_wave(wave_dir, flows, arms)
    meta = _load(os.path.join(wave_dir, "wave.json"), {}) or {}
    res["wave"] = os.path.basename(os.path.normpath(wave_dir))
    res["code"] = meta.get("code")
    res["fake"] = bool(meta.get("fake"))
    with open(os.path.join(wave_dir, "results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False, default=str)
    path = os.path.join(wave_dir, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_results(res, flows, res["wave"], meta))
    return path


def wave_arms(wave_dir: str) -> Tuple[List[str], List[str]]:
    """``(flow ids, arms)`` of the ``<flow>@<arm>`` scenarios on disk."""
    flows, arms = [], []
    if os.path.isdir(wave_dir):
        for sid in sorted(os.listdir(wave_dir)):
            f, a = split_scenario_id(sid)
            if a in ARMS:
                if f not in flows:
                    flows.append(f)
                if a not in arms:
                    arms.append(a)
    return flows, [a for a in ARMS if a in arms]


# ---------------------------------------------------------------------------
# What a wave should cost, before it runs
# ---------------------------------------------------------------------------

#: The recorded waves the estimate reads: the geotech-page judgment waves
#: (w11 to w14, Sonnet 5.5), the only ones whose flows consulted the
#: references helper.
BASIS_WAVES = r"^w1[1-4]"
BASIS_MODEL = "claude-sonnet-5-5"


def recorded_agent_runs(runs_dir: str, model_id: str = BASIS_MODEL,
                        waves: str = BASIS_WAVES) -> List[dict]:
    """Every finished scenario on ``model_id`` in the waves matching
    ``waves``: its cost (the meter's), the agents its activity shows, and
    the share of its cost the references helper took. The activity keeps
    tokens per call but not their cache split, so the share weights output
    tokens by the model's output-to-input price ratio and every input token
    alike: an estimate."""
    from live_smoke import spend
    r = spend.rates_for(model_id)
    w_out = r["output"] / r["input"]
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for wave in sorted(os.listdir(runs_dir)):
        if not re.match(waves, wave):
            continue
        for sid in sorted(os.listdir(os.path.join(runs_dir, wave))):
            d = os.path.join(runs_dir, wave, sid)
            s = _load(os.path.join(d, "summary.json"))
            if not s or s.get("model") != model_id or s.get("incomplete") \
                    or s.get("unmetered"):
                continue
            agg: Dict[str, float] = {}
            for a in _activity(d):
                if a.get("event") != "model_end":
                    continue
                u = a.get("usage") or {}
                agg[a.get("agent") or "primary"] = agg.get(
                    a.get("agent") or "primary", 0.0) + int(
                    u.get("input_tokens") or 0) + w_out * int(
                    u.get("output_tokens") or 0)
            if not agg:
                continue
            tot = sum(agg.values()) or 1.0
            out.append({"wave": wave, "id": s.get("id") or sid,
                        "usd": float(s.get("usd") or 0.0),
                        "agents": sorted(agg),
                        "references_share": agg.get(REFERENCES, 0.0) / tot})
    return out


def estimate(flows: List[dict], arms: Iterable[str], runs_dir: str, *,
             cap: Optional[float] = None, margin: Optional[float] = None
             ) -> dict:
    """What a wave of ``flows`` x ``arms`` should cost.

    Per flow, a Sonnet 5.5 basis: the dearest recorded run of the flow
    itself on Sonnet; else, for a flow whose truth says it uses the
    references helper, the mean of the recorded runs that used it; else
    the mean of the recorded runs that used no helper. Each arm prices that
    basis at its models: the references helper's share (the mean share of
    the recorded runs that used it) at the helper's model, the rest at the
    main model, by the ratio of the models' input prices (Opus 5.5 is
    Sonnet 5.5 x 2 in every token class). Returns ``{"rows", "total_usd",
    "suggested_cap_usd", "fits", "cut", "basis"}``; ``cut`` names the
    flows to leave out, dearest first, when the total passes ``cap``."""
    from live_smoke import spend
    arms = list(arms)
    margin = spend.CAP_MARGIN if margin is None else margin
    runs = recorded_agent_runs(runs_dir)
    with_refs = [r for r in runs if REFERENCES in r["agents"]]
    no_helper = [r for r in runs if r["agents"] == ["primary"]]
    share = (sum(r["references_share"] for r in with_refs) / len(with_refs)
             if with_refs else 0.35)
    mean_refs = (sum(r["usd"] for r in with_refs) / len(with_refs)
                 if with_refs else spend.FALLBACK_USD["medium"])
    mean_none = (sum(r["usd"] for r in no_helper) / len(no_helper)
                 if no_helper else spend.FALLBACK_USD["light"])
    own = {}
    for r in runs:
        own[r["id"]] = max(own.get(r["id"], 0.0), r["usd"])
    base_in = spend.rates_for(BASIS_MODEL)["input"]

    def ratio(mid):
        return spend.rates_for(mid)["input"] / base_in

    rows = []
    for f in flows:
        t = f.get("truth") or {}
        uses = REFERENCES in (t.get("helpers") or [])
        if f["id"] in own:
            basis, why = own[f["id"]], "its own dearest recorded run"
        elif uses:
            basis, why = mean_refs, (f"mean of {len(with_refs)} recorded "
                                     "runs that used the references helper")
        else:
            basis, why = mean_none, (f"mean of {len(no_helper)} recorded "
                                     "runs that used no helper")
        s = share if uses else 0.0
        for a in arms:
            spec = arm(a)
            ref_model = expected_helper_models(a).get(REFERENCES) \
                or spec["main"]
            usd = basis * ((1 - s) * ratio(spec["main"])
                           + s * ratio(ref_model))
            rows.append({"flow": f["id"], "arm": a, "usd": round(usd, 4),
                         "sonnet_basis_usd": round(basis, 4),
                         "references_share": round(s, 3), "basis": why})
    total = round(sum(r["usd"] for r in rows), 4)
    out = {"rows": rows, "total_usd": total,
           "suggested_cap_usd": math.ceil(total * margin * 2) / 2.0,
           "basis": {"runs": len(runs), "with_references": len(with_refs),
                     "no_helper": len(no_helper),
                     "references_share": round(share, 3),
                     "mean_usd_with_references": round(mean_refs, 4),
                     "mean_usd_no_helper": round(mean_none, 4)},
           "cap_usd": cap, "fits": cap is None or total <= cap, "cut": []}
    if cap is not None and total > cap:
        per_flow: Dict[str, float] = {}
        for r in rows:
            per_flow[r["flow"]] = per_flow.get(r["flow"], 0.0) + r["usd"]
        left = total
        for fid, usd in sorted(per_flow.items(), key=lambda kv: -kv[1]):
            if left <= cap:
                break
            out["cut"].append(fid)
            left -= usd
    return out


__all__ = ["ARMS", "DEFAULT_ARMS", "MODEL_LABELS", "ArmWiringError",
           "arm_models", "arm_env", "arm_flow", "expected_helper_models",
           "order", "scenario_id", "split_scenario_id", "wiring_findings",
           "wiring_record", "key_number", "methods_named", "cited",
           "cites_check", "helper_calls", "analyse_scenario",
           "analyse_wave", "render_results", "write_results", "wave_arms",
           "recorded_agent_runs", "estimate"]
