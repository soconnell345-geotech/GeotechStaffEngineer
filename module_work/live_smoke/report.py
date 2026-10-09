"""REPORT.md for a wave, and transcript.md for a scenario.

Both are rendered from what is on disk (``summary.json`` and
``detectors.json`` per scenario, ``wave.json``, the spend ledger), so
``run.py report --wave <name>`` re-renders a wave without running anything.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}

DETECTOR_TITLES = {
    "a_files_outside": "(a) Files outside the conversation folder",
    "b_answer_links": "(b) Answer links",
    "c_download_cards": "(c) Download cards",
    "d_sharepoint": "(d) SharePoint mirror and SharePoint tools",
    "e_health": "(e) Health",
    "f_cost": "(f) Cost",
    "g_isolation": "(g) Per-user isolation",
    "h_restore": "(h) Restore from SharePoint",
    "i_expectations": "(i) Flow expectations (model choices, not bugs)",
    "j_conversation": "(j) Conversation record (meta, titles, owner)",
    "harness": "Harness problems",
}


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _scenarios(wave_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(wave_dir)):
        if name.startswith("_"):
            continue
        if os.path.isfile(os.path.join(wave_dir, name, "summary.json")):
            out.append(name)
    return out


def _md(text: str, n: int = 160) -> str:
    t = " ".join(str(text or "").split())
    t = t.replace("|", "\\|")
    return t if len(t) <= n else t[:n - 1] + "…"


def write_report(wave_dir: str, *, ledger=None, wave: Optional[str] = None
                 ) -> str:
    """Render ``<wave_dir>/REPORT.md``; returns its path."""
    wave = wave or os.path.basename(wave_dir.rstrip("/\\"))
    meta = _load(os.path.join(wave_dir, "wave.json")) or {}
    rows: List[dict] = []
    findings: List[dict] = []
    for sid in _scenarios(wave_dir):
        summ = _load(os.path.join(wave_dir, sid, "summary.json")) or {}
        rows.append(summ)
        dj = _load(os.path.join(wave_dir, sid, "detectors.json")) or {}
        for i, t in enumerate(dj.get("turns") or [], 1):
            for f in t.get("findings") or []:
                findings.append({**f, "scenario": sid, "turn": i})
        for f in dj.get("scenario") or []:
            findings.append({**f, "scenario": sid, "turn": None})
    L: List[str] = [f"# Live smoke wave `{wave}`", ""]
    L.append(f"Mode `{meta.get('mode')}` · model `{meta.get('model')}` · "
             f"cap ${float(meta.get('cap_usd') or 0):.2f} · updated "
             f"{meta.get('updated')}")
    if meta.get("stopped"):
        L.append("")
        L.append(f"**STOPPED:** {meta['stopped']}")
    L.append("")
    L.append("Claude stands in for the production GPT models: answer quality "
             "is not scored. Only plumbing findings count.")
    # -- spend
    L += ["", "## Spend", ""]
    usd = sum(float(r.get("usd") or 0) for r in rows)
    tok: Dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("tokens") or {}).items():
            tok[k] = tok.get(k, 0) + int(v or 0)
    calls = sum(int(r.get("model_calls") or 0) for r in rows)
    unmetered = sum(int(r.get("unmetered") or 0) for r in rows)
    L.append(f"- Scenarios in this report: {len(rows)}; model calls {calls}; "
             f"spend ${usd:.4f}")
    if ledger is not None:
        w = ledger.wave(wave)
        L.append(f"- Ledger: wave ${float(w.get('spent_usd') or 0):.4f} of "
                 f"${float(w.get('cap_usd') or 0):.2f}; cumulative "
                 f"${ledger.cumulative:.4f} of ${ledger.hard_total:.2f} "
                 f"(`{os.path.basename(ledger.path)}`)")
    prompt = tok.get("prompt", 0)
    cr = tok.get("cache_read", 0)
    cw = tok.get("cache_write_5m", 0) + tok.get("cache_write_1h", 0)
    L.append(f"- Tokens: {prompt:,} prompt ({tok.get('uncached', 0):,} "
             f"uncached, {cr:,} cache read, {cw:,} cache write), "
             f"{tok.get('output', 0):,} output"
             + (f"; cache read = {100.0 * cr / prompt:.0f} % of prompt"
                if prompt else ""))
    if unmetered:
        L.append(f"- **{unmetered} model call(s) carried no usage and were "
                 "not priced.**")
    # -- findings by detector
    L += ["", "## Findings by detector", ""]
    real = [f for f in findings if f["severity"] != "info"
            or f["detector"] in ("i_expectations",)]
    if not real:
        L.append("No findings.")
    groups: Dict[str, Dict[tuple, List[dict]]] = {}
    for f in real:
        groups.setdefault(f["detector"], {}).setdefault(
            (f["code"], f["severity"]), []).append(f)
    for detector in sorted(groups, key=lambda d: (d not in DETECTOR_TITLES,
                                                  d)):
        L += [f"### {DETECTOR_TITLES.get(detector, detector)}", "",
              "| code | severity | count | scenarios | example |",
              "|---|---|---|---|---|"]
        items = sorted(groups[detector].items(),
                       key=lambda kv: (SEV_ORDER[kv[0][1]], -len(kv[1])))
        for (code, sev), fs in items:
            scen = sorted({f["scenario"] for f in fs})
            links = ", ".join(f"[{s}]({s}/transcript.md)" for s in scen[:8])
            if len(scen) > 8:
                links += f" +{len(scen) - 8}"
            L.append(f"| `{code}` | {sev} | {len(fs)} | {links} | "
                     f"{_md(fs[0]['message'])} |")
        L.append("")
    # -- scenarios
    L += ["## Scenarios", "",
          "| scenario | turns | USD | model calls | seconds | high | medium "
          "| low | note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sev = r.get("severity") or {}
        L.append(f"| [{r['id']}]({r['id']}/transcript.md) | "
                 f"{r.get('turns', 0)} | {float(r.get('usd') or 0):.4f} | "
                 f"{r.get('model_calls', 0)} | {r.get('seconds')} | "
                 f"{sev.get('high', 0)} | {sev.get('medium', 0)} | "
                 f"{sev.get('low', 0)} | {_md(r.get('note') or '', 80)} |")
    # -- every high / medium finding
    L += ["", "## High and medium findings", ""]
    hm = sorted((f for f in findings if f["severity"] in ("high", "medium")),
                key=lambda f: (SEV_ORDER[f["severity"]], f["scenario"],
                               f["turn"] or 0))
    if not hm:
        L.append("None.")
    for f in hm:
        where = f"[{f['scenario']}]({f['scenario']}/transcript.md)" + (
            f" turn {f['turn']}" if f["turn"] else "")
        L.append(f"- **{f['severity']}** `{f['detector']}/{f['code']}` "
                 f"{where}: {_md(f['message'], 400)}")
    path = os.path.join(wave_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def write_transcript(path: str, flow: dict, turns: List[dict],
                     scen_findings: List[dict]) -> None:
    """A readable record of one scenario."""
    L = [f"# {flow.get('id')} — {flow.get('title', '')}", ""]
    if flow.get("steps"):
        L.append("Steps: " + "; ".join(
            ", ".join(f"{k}={str(v)[:60]}" for k, v in s.items())
            for s in flow["steps"]))
        L.append("")
    for i, t in enumerate(turns, 1):
        m = t.get("metrics") or {}
        L.append(f"## Turn {i} · {t.get('session')} · {t.get('kind')} · "
                 f"{t.get('page') or ''} · ${float(m.get('usd') or 0):.4f} · "
                 f"{m.get('model_calls', 0)} model calls · "
                 f"{m.get('seconds')} s")
        L.append("")
        L.append("**User:** " + (t.get("prompt") or "").strip()[:3000])
        L.append("")
        tools = t.get("tools") or []
        if tools:
            L.append(f"**Tool calls ({len(tools)}):**")
            for c in tools[:80]:
                L.append(f"- `{c.get('agent')}` {c.get('name')} "
                         f"{_md(c.get('args'), 200)}")
            if len(tools) > 80:
                L.append(f"- … {len(tools) - 80} more")
            L.append("")
        L.append("**Answer:**")
        L.append("")
        L.append((t.get("final") or "(none)").strip())
        L.append("")
        if t.get("error"):
            L.append(f"**Turn error:** {t['error']}")
            L.append("")
        if t.get("cards"):
            L.append("**Download cards:** " + ", ".join(t["cards"]))
            L.append("")
        if t.get("suite_score"):
            sc = t["suite_score"]
            L.append(f"**Suite score:** {sc.get('checks_passed')}/"
                     f"{sc.get('checks_total')}")
            L.append("")
        if t.get("sp_web_url"):
            L.append(f"**Sidebar folder link:** {t['sp_web_url']}")
            L.append("")
        fs = [f for f in t.get("findings") or [] if f["severity"] != "info"]
        if fs:
            L.append("**Findings:**")
            for f in sorted(fs, key=lambda f: SEV_ORDER[f["severity"]]):
                L.append(f"- {f['severity']} `{f['detector']}/{f['code']}`: "
                         f"{_md(f['message'], 500)}")
            L.append("")
    if scen_findings:
        L.append("## Scenario findings")
        L.append("")
        for f in sorted(scen_findings, key=lambda f: SEV_ORDER[f["severity"]]):
            L.append(f"- {f['severity']} `{f['detector']}/{f['code']}`: "
                     f"{_md(f['message'], 500)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


__all__ = ["write_report", "write_transcript"]
