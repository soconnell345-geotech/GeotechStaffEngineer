"""The figure set's report: RESULTS.md, results.json and the side-by-side
page for the owner's look (GM-314).

``side_by_side.html`` shows each request's figures, OFF left and ON right,
the vision marks under each and a blank box for the owner's own mark; its
button saves the owner's marks as a JSON file. It links the PNGs where the
run folders keep them, so keep the output folder whole.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from funhouse_agent.figure_eval.checklist import ITEM_IDS, items_for

#: The headline's rows for requests kept out of better / same / worse
#: (``compare.APART``).
APART_TITLES = {"run_failed": "run failed, not compared",
                "not_scored": "vision reply unreadable, not scored",
                "neither_drew": "neither arm drew a figure, not compared"}

ARM_TITLES = {"off": "Plot code OFF (baseline)", "on": "Plot code ON"}


def _ids(xs: Sequence[str]) -> str:
    return ", ".join(x.split("-")[0] for x in xs) or "-"


def _marks_count(check: Optional[Dict[str, Any]]) -> str:
    if not check:
        return "not checked"
    if check.get("run_failed"):
        return "run failed"
    if check.get("not_scored"):
        return "not scored"
    c = {"yes": 0, "no": 0, "na": 0, "unmarked": 0}
    for m in check["marks"].values():
        c[m["mark"]] = c.get(m["mark"], 0) + 1
    s = f"{c['yes']}/{c['no']}/{c['na']}"
    if c["unmarked"]:
        s += f" + {c['unmarked']} unmarked"
    return s + (" (no figure)" if check.get("no_figure") else "")


def _run_cell(run: Optional[Dict[str, Any]]) -> str:
    if not run:
        return "not run"
    if run.get("error"):
        return "failed"
    return f"{len(run.get('figures') or [])} fig"


def _apart_ids(rows: Dict[str, List[str]]) -> str:
    return ", ".join(f"{rid.split('-')[0]} ("
                     + ", ".join(a.upper() for a in arms) + ")"
                     for rid, arms in rows.items()) or "-"


def _apart_of(cmp_: Dict[str, Any], rid: str) -> Optional[str]:
    """Why ``rid`` is kept apart, in the headline's words, if it is."""
    for key, rows in (cmp_.get("apart") or {}).items():
        if rid in rows:
            return APART_TITLES.get(key, key).split(",")[0]
    return None


def render_markdown(res: Dict[str, Any]) -> str:
    cmp_ = res["comparison"]
    base, other = cmp_["base"], cmp_["other"]
    reqs = res["requests"]
    lines = [
        "# Figure test set (GM-314): plot code OFF against ON", "",
        f"Written {res['written']}. Agent model: {res.get('model') or '-'}; "
        f"vision model: {res.get('vision_model') or '-'} (image detail "
        f"{res.get('detail') or 'none'}).", "",
        "Arms: " + "; ".join(f"`{a}` = {ARM_TITLES.get(a, a)}, env "
                             f"`{json.dumps(res['arm_env'][a])}`"
                             for a in res["arms"]) + ".", "",
        "Marks are the vision model's, one run per request and arm: a "
        "measurement, not a verdict. Read the side-by-side page "
        "(`side_by_side.html`) and the full traces before any conclusion.",
        "", "## Overall", ""]
    r = cmp_["requests"]
    n = sum(len(r[o]) for o in ("better", "same", "worse"))
    lines += [
        f"Requests compared: {n} of {len(reqs)}"
        + (f" (not yet both checked: {_ids(cmp_['unchecked'])})"
           if cmp_["unchecked"] else "") + ".", "",
        "| ON against OFF | requests | which |", "|---|---|---|"]
    for o in ("better", "same", "worse"):
        lines.append(f"| {o} | {len(r[o])} | {_ids(r[o])} |")
    apart = cmp_.get("apart") or {}
    for key, title in APART_TITLES.items():
        rows = apart.get(key) or {}
        lines.append(f"| {title} | {len(rows)} | {_apart_ids(rows)} |")
    drew = cmp_.get("drew_none") or {}
    if drew:
        lines += ["", "Decided against an arm that finished and drew no "
                  "figure (every item counted no): " + ", ".join(
                      f"{rid.split('-')[0]} ({arm.upper()} drew none)"
                      for rid, arm in drew.items()) + "."]
    if apart.get("run_failed"):
        lines += ["", "A failed run is asked again by the next call of "
                  "`score_figure_set` with the same folders. Name each run "
                  "that still fails in the hand-back."]
    on_runs = {rid: arms[other] for rid, arms in res["runs"].items()
               if arms.get(other) and not arms[other].get("error")}
    coded = [rid for rid, run in on_runs.items()
             if (run.get("tool_counts") or {}).get("plot_code")]
    lines += ["", f"ON runs that called plot_code: {len(coded)} of "
              f"{len(on_runs)}"
              + (f" (none in {_ids([r for r in on_runs if r not in coded])})"
                 if len(coded) < len(on_runs) else "")
              + ". An ON run without a plot_code call drew as OFF does: "
              "'same' there is no use, not no gain."]
    it = cmp_["items"]
    lines += ["", "Items over all compared requests: "
              f"better {it['better']}, same {it['same']}, worse "
              f"{it['worse']}, one arm n/a {it['one_na']}, unmarked "
              f"{it.get('unmarked', 0)}.", "",
              "| Arm | yes | no | n/a | unmarked |", "|---|---|---|---|---|"]
    for a in (base, other):
        m = cmp_["marks"][a]
        lines.append(f"| {a} | {m['yes']} | {m['no']} | {m['na']} | "
                     f"{m.get('unmarked', 0)} |")
    lines += ["", "## By item", "",
              "| Item | better | same | worse | one arm n/a | unmarked |",
              "|---|---|---|---|---|---|"]
    order = list(ITEM_IDS) + ["request's own items"]
    for iid in order:
        row = cmp_["per_item"].get(iid)
        if not row:
            continue
        lines.append(
            f"| {iid} | {len(row['better'])} ({_ids(row['better'])}) | "
            f"{len(row['same'])} | {len(row['worse'])} "
            f"({_ids(row['worse'])}) | {len(row['one_na'])} | "
            f"{len(row.get('unmarked') or [])} |")
    lines += ["", "## By request", "",
              f"Marks are yes/no/n/a counts. Tokens are the agent's input + "
              "output.", "",
              "| Request | kind | OFF run | ON run | OFF marks | ON marks | "
              "better/same/worse/one n/a/unmarked | verdict | "
              "tokens OFF / ON | "
              "seconds OFF / ON |", "|---|---|---|---|---|---|---|---|---|---|"]
    for req in reqs:
        rid = req["id"]
        runs = res["runs"].get(rid, {})
        checks = res["checks"].get(rid, {})
        pr = cmp_["per_request"].get(rid)
        c = pr["counts"] if pr else None
        cnt = (f"{c['better']}/{c['same']}/{c['worse']}/{c['one_na']}/"
               f"{c.get('unmarked', 0)}" if c else "-")

        def tok(a):
            run = runs.get(a) or {}
            return str((run.get("tokens_in") or 0) +
                       (run.get("tokens_out") or 0)) if run else "-"

        def sec(a):
            run = runs.get(a) or {}
            return str(run.get("seconds", "-")) if run else "-"
        lines.append(
            f"| {rid}{' (numpy)' if req.get('numpy_note') else ''} | "
            f"{req['kind']}{'' if req.get('named') else ', not named'} | "
            f"{_run_cell(runs.get(base))} | {_run_cell(runs.get(other))} | "
            f"{_marks_count(checks.get(base))} | "
            f"{_marks_count(checks.get(other))} | {cnt} | "
            f"{pr['verdict'] if pr else (_apart_of(cmp_, rid) or '-')} | "
            f"{tok(base)} / {tok(other)} | "
            f"{sec(base)} / {sec(other)} |")
    notes = []
    for req in reqs:
        for a in res["arms"]:
            run = res["runs"].get(req["id"], {}).get(a)
            chk = res["checks"].get(req["id"], {}).get(a)
            if run and run.get("error"):
                notes.append(f"- {req['id']} {a}: the run failed "
                             f"({run['error'][:200]})"
                             + ("; the request is marked for the host's "
                                "numpy" if req.get("numpy_note") else "")
                             + ".")
            elif run and not run.get("figures"):
                notes.append(f"- {req['id']} {a}: no figure appeared "
                             "(every item counted no).")
            if chk and chk.get("not_scored"):
                notes.append(f"- {req['id']} {a}: the vision reply could "
                             f"not be read, asked {chk.get('tries', 1)} "
                             "time(s); not scored (reply: "
                             f"{str(chk.get('reply', ''))[:200]!r}).")
            elif chk and chk.get("unmarked"):
                notes.append(f"- {req['id']} {a}: the checker left "
                             f"{', '.join(chk['unmarked'])} unmarked "
                             "(counted apart, never same).")
            if chk and chk.get("dropped"):
                notes.append(f"- {req['id']} {a}: {chk['dropped']} older "
                             "figure(s) not shown to the checker.")
    lines += ["", "## Notes", ""] + (notes or ["None."])
    tot = res["totals"]
    lines += ["", "## Tokens and time", "",
              "| Arm | runs | agent tokens in | out | vision tokens in | "
              "out | seconds |", "|---|---|---|---|---|---|---|"]
    for a in res["arms"]:
        t = tot[a]
        lines.append(f"| {a} | {t['runs']} | {t['tokens_in']} | "
                     f"{t['tokens_out']} | {t['vision_in']} | "
                     f"{t['vision_out']} | {t['seconds']} |")
    return "\n".join(lines) + "\n"


_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mute:#666;--line:#ddd;--yes:#1b7f3b;--no:#b3261e;--na:#777;--card:#fafafa}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mute:#aaa;--line:#333;--yes:#5fd38a;--no:#ff8a80;--na:#999;--card:#1f1f1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}
h1{font-size:22px}h2{font-size:18px;margin:0 0 4px}
.req{border-top:2px solid var(--line);padding:16px 0}
.ask{color:var(--mute);margin:0 0 10px;white-space:pre-wrap}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:800px){.pair{grid-template-columns:1fr}}
.arm{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;min-width:0}
.arm img{max-width:100%;height:auto;display:block;margin:6px 0;border:1px solid var(--line);background:#fff}
.none{color:var(--no);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}
td{border-top:1px solid var(--line);padding:3px 4px;vertical-align:top}
.yes{color:var(--yes);font-weight:600}.no{color:var(--no);font-weight:600}.na{color:var(--na)}.unmarked{color:var(--na);font-style:italic}
.owner{margin-top:10px}.owner textarea{width:100%;min-height:54px;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--line)}
button{font:inherit;padding:6px 12px}
"""


def _arm_html(req: Dict[str, Any], arm: str, run: Optional[Dict[str, Any]],
              check: Optional[Dict[str, Any]]) -> str:
    e = html.escape
    out = [f'<div class="arm"><h3>{e(ARM_TITLES.get(arm, arm))}</h3>']
    if not run:
        out.append('<p class="none">Not run.</p>')
    elif run.get("error"):
        out.append(f'<p class="none">The run failed: '
                   f'{e(str(run["error"])[:300])}</p>')
    elif not run.get("figures"):
        out.append('<p class="none">No figure appeared.</p>')
    for rel in (run or {}).get("figures") or []:
        src = rel.replace(os.sep, "/")
        out.append(f'<a href="{e(src)}"><img src="{e(src)}" '
                   f'alt="{e(os.path.basename(rel))}" loading="lazy"></a>')
    if check:
        out.append("<table>")
        texts = {it["id"]: it["text"] for it in items_for(req)}
        for iid, m in check["marks"].items():
            out.append(f'<tr><td title="{e(texts.get(iid, ""))}">{e(iid)}'
                       f'</td><td class="{m["mark"]}">{m["mark"]}</td>'
                       f'<td>{e(m["reason"])}</td></tr>')
        out.append("</table>")
        if check.get("note"):
            out.append(f"<p>{e(check['note'])}</p>")
    else:
        out.append("<p>Not checked yet.</p>")
    out.append("</div>")
    return "".join(out)


def render_html(res: Dict[str, Any]) -> str:
    e = html.escape
    cmp_ = res["comparison"]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Figure set side by side</title>", f"<style>{_CSS}</style>",
        "</head><body><h1>Figure test set (GM-314): OFF left, ON right</h1>",
        f"<p>Written {e(res['written'])}. Agent model "
        f"{e(str(res.get('model') or '-'))}; vision model "
        f"{e(str(res.get('vision_model') or '-'))}. Under each figure, the "
        "vision model's marks; under each pair, a box for your own mark. "
        "'Save my marks' writes them to a file.</p>"]
    for req in res["requests"]:
        rid = req["id"]
        runs = res["runs"].get(rid, {})
        checks = res["checks"].get(rid, {})
        pr = cmp_["per_request"].get(rid)
        verdict = (f"vision: ON {pr['verdict']} "
                   f"({pr['counts']['better']} better, "
                   f"{pr['counts']['same']} same, {pr['counts']['worse']} "
                   f"worse, {pr['counts']['one_na']} one arm n/a, "
                   f"{pr['counts'].get('unmarked', 0)} unmarked)"
                   if pr else "vision: not compared ("
                   + (_apart_of(cmp_, rid) or "not yet both checked") + ")")
        parts.append(
            f"<section class='req' data-id='{e(rid)}'><h2>{e(rid)}</h2>"
            f"<p class='ask'>{e(req['ask'])}</p><p>{e(verdict)}</p>"
            "<div class='pair'>"
            + "".join(_arm_html(req, a, runs.get(a), checks.get(a))
                      for a in res["arms"])
            + "</div><div class='owner'><span>Your mark: "
            + "".join(f"<label><input type='radio' name='v-{e(rid)}' "
                      f"value='{v}'> {t}</label> "
                      for v, t in (("off", "OFF better"), ("same", "same"),
                                   ("on", "ON better")))
            + f"</span><textarea name='n-{e(rid)}' placeholder='Your "
            "notes on these two figures'></textarea></div></section>")
    parts.append(
        "<p><button id='save'>Save my marks</button></p><script>"
        "document.getElementById('save').onclick=function(){var o={};"
        "document.querySelectorAll('section.req').forEach(function(s){"
        "var id=s.dataset.id,r=s.querySelector('input:checked'),"
        "t=s.querySelector('textarea');o[id]={mark:r?r.value:null,"
        "note:t.value};});var b=new Blob([JSON.stringify(o,null,1)],"
        "{type:'application/json'});var a=document.createElement('a');"
        "a.href=URL.createObjectURL(b);a.download='owner_marks.json';"
        "a.click();};</script></body></html>")
    return "".join(parts)


def write_report(out_dir: str, res: Dict[str, Any]) -> Dict[str, str]:
    """Write the three files; their paths."""
    res = dict(res, written=res.get("written") or
               datetime.now().isoformat(timespec="seconds"))
    paths = {"json": os.path.join(out_dir, "results.json"),
             "md": os.path.join(out_dir, "RESULTS.md"),
             "html": os.path.join(out_dir, "side_by_side.html")}
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write(render_markdown(res))
    with open(paths["html"], "w", encoding="utf-8") as fh:
        fh.write(render_html(res))
    return paths
