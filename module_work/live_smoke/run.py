"""Live smoke-test harness for the GeotechStaffEngineer app -- the command line.

Run with the project venv from the repo root::

    .venv\\Scripts\\python.exe module_work\\live_smoke\\run.py <mode> [options]

Modes
-----
``flow``     multi-turn Tiny Apps tester scenarios from ``flows.json``
             through the app's turn path (both pages, uploads, orientation,
             pastes, Word/PDF/plot/CSV/DXF/DIGGS outputs, SharePoint up and
             down, two users, restart + restore). ``--ids F01,F02`` picks
             flows; ``--max-cost light|medium|heavy`` skips costlier ones.
``geotech``  geotech eval questions (``funhouse_agent/geotech_test_suite.json``)
             through the app's turn path on the GeotechStaffEngineer page,
             one fresh conversation each. ``--ids BC-1,SE-1`` (prefixes ok).
``suite``    Document Review suite tasks through ``score_review_suite``
             (its own scoring kept), plus detectors (a)(b)(e)(f).
             ``--ids fixture-markups,meck-revision-block``.
``dry-run``  the $0 scripted-fake scenario that does the bad things on
             purpose (saves to temp, links a missing file, fails a tool) and
             prints which detectors fired. Never touches the spend ledger.
             With ``--arms``: the $0 arms wave (EV-33) on scripted fakes
             named after the arms' models, end to end to ``RESULTS.md``.
``report``   re-render ``runs/<wave>/REPORT.md`` from disk.
``recheck``  detectors (e) health, (k) figure cards, (l) judgment-call
             check and (m) answer numbers, and the flow expectations,
             again over a recorded wave, with today's ``flows.json`` and
             detectors ($0, no model). ``--looks`` asks for a look at
             every figure card.
``arms``     BACKLOG EV-33, the Foundry model split: each flow in
             ``--ids`` once per arm in ``--arms`` (default
             ``opus_all,split``; ``live_smoke/arms.py``), as the scenario
             ``<flow>@<arm>``, the arms in turn and reversed on every other
             flow. Each scenario checks the arm's models before any model
             call. Writes ``runs/<wave>/RESULTS.md`` and ``results.json``.
             Resume with the same ``--wave``.
``estimate`` what a flow wave should cost on ``--model``, from the meter's
             record of earlier runs (``runs/*/*/summary.json``), and a
             suggested cap. ``--ids`` picks as ``flow`` does (by prefix),
             so the same ids give the same flows; an id matching no flow
             (one not written yet) is estimated from its other arm. With
             ``--arms``: what an arms wave of those flows should cost, from
             the w11-w14 Sonnet runs, and the flows to cut to fit ``--cap``.
``ledger``   print the spend ledger.
``list``     list flows, geotech question ids and suite task ids.

Spend: every live wave needs ``--wave NAME --cap USD``. The ledger
(``spend_ledger.json``) refuses a wave whose remaining cap would carry the
cumulative spend past $80; a wave stops cleanly at the next model call once
its cap is spent. ``--fake`` runs flows/geotech with the scripted model ($0,
its own ledger) to check the scenarios themselves.

Examples::

    run.py flow --wave w1-haiku --cap 2.00 --model claude-haiku-5-5 --max-cost medium
    run.py flow --wave w2-sonnet --cap 8.00 --model claude-sonnet-5-5 --ids F02,F04,F11
    run.py geotech --wave g1 --cap 3.00 --model claude-haiku-5-5 --ids BC-1,SE-1,RW-1
    run.py suite --wave s1 --cap 4.00 --model claude-haiku-5-5 --ids fixture-markups,produce-memo
    run.py flow --wave fake-all --cap 0 --fake
    run.py recheck --wave w9-figures
    run.py estimate --model claude-sonnet-5-5 --ids F56,F57,F60,F18
    run.py estimate --arms opus_all,split --cap 30 --ids F67,F68,F69,F70,F71,F72,F73,F74
    run.py arms --wave w15-model-split --cap 30 --ids F67,F68,F69,F70,F71,F72,F73,F74
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.dirname(HERE), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

COST_ORDER = {"light": 0, "medium": 1, "heavy": 2}


def _ids(text):
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def _pick(items, ids, key="id"):
    if not ids:
        return list(items)
    out = []
    for it in items:
        if any(it[key] == i or it[key].startswith(i) for i in ids):
            out.append(it)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("flow", "geotech", "suite", "dry-run",
                                     "report", "recheck", "estimate",
                                     "ledger", "list", "arms"))
    ap.add_argument("--wave", help="wave name (its folder under runs/)")
    ap.add_argument("--cap", type=float, help="wave cap in USD")
    ap.add_argument("--model", default="claude-haiku-5-5",
                    help="claude-haiku-5-5 | claude-sonnet-5-5 | "
                         "claude-opus-5-5")
    ap.add_argument("--ids", default="", help="comma-separated ids/prefixes")
    ap.add_argument("--max-cost", default="heavy", choices=tuple(COST_ORDER),
                    help="flows: skip flows costlier than this tier")
    ap.add_argument("--fake", action="store_true",
                    help="flow/geotech with the scripted fake model ($0)")
    ap.add_argument("--redo", action="store_true",
                    help="re-run scenarios already finished in this wave")
    ap.add_argument("--keep-sandbox", action="store_true",
                    help="keep runs/<wave>/_sandbox (data root, cwd, library)")
    ap.add_argument("--arm", default="baseline", help="suite: arm name")
    ap.add_argument("--arms", default=None,
                    help="arms / estimate: comma-separated arms of "
                         "live_smoke/arms.py (default opus_all,split)")
    ap.add_argument("--orientation", action="store_true",
                    help="suite: send the orientation turn first")
    ap.add_argument("--docs-dir", default=None, help="suite: docs folder")
    ap.add_argument("--looks", action="store_true",
                    help="recheck: ask for a look at every figure card")
    ap.add_argument("--ledger", default=None, help="ledger path (default "
                    "module_work/live_smoke/spend_ledger.json)")
    args = ap.parse_args(argv)

    from live_smoke import runner, spend

    if args.mode == "ledger":
        led = spend.SpendLedger(args.ledger or spend.DEFAULT_LEDGER)
        print(json.dumps(led.data, indent=1))
        return 0
    if args.mode == "list":
        for f in runner.load_flows():
            print(f"{f['id']:<34} {f.get('cost', 'medium'):<6} "
                  f"{f.get('page', 'document_review'):<16} {f.get('title')}")
        from funhouse_agent.deep.eval_harness import load_suite
        print("geotech:", ", ".join(q["id"] for q in load_suite()))
        from funhouse_agent.review_eval.tasks import load_tasks
        print("suite:", ", ".join(t.id for t in load_tasks()))
        return 0
    if args.mode == "report":
        if not args.wave:
            ap.error("report needs --wave")
        from live_smoke import report
        led = spend.SpendLedger(args.ledger or spend.DEFAULT_LEDGER)
        wdir = os.path.join(runner.RUNS, args.wave)
        print(report.write_report(wdir, ledger=led, wave=args.wave))
        from live_smoke import arms
        fids, arm_names = arms.wave_arms(wdir)
        if fids:
            flows = {f["id"]: f for f in runner.load_flows()}
            print(arms.write_results(wdir, [flows.get(i) or {"id": i}
                                            for i in fids], arm_names))
        return 0
    if args.mode == "dry-run" and args.arms is not None:
        from live_smoke import dryrun
        res = dryrun.run_arms(wave=args.wave or "dry-arms", verbose=True,
                              arms_names=_ids(args.arms) or None)
        print(json.dumps({k: res[k] for k in ("ok", "problems", "results",
                                              "arms")}, indent=1))
        return 0 if res["ok"] else 1
    if args.mode == "dry-run":
        from live_smoke import dryrun
        res = dryrun.run(wave=args.wave or "dry-run", verbose=True)
        print(json.dumps(res["fired"], indent=1))
        return 0 if res["ok"] else 1
    if args.mode == "recheck":
        if not args.wave:
            ap.error("recheck needs --wave")
        from live_smoke import dryrun
        res = dryrun.recheck(os.path.join(runner.RUNS, args.wave),
                             _ids(args.ids), looks_everywhere=args.looks)
        for sid, r in res.items():
            print(f"[{sid}]")
            for i, fs in enumerate(r["turns"], 1):
                for f in fs:
                    print(f"  turn {i}: {f['severity']} {f['detector']}/"
                          f"{f['code']}: {f['message']}")
            for f in r["scenario"]:
                print(f"  flow: {f['severity']} {f['detector']}/"
                      f"{f['code']}: {f['message']}")
        return 0
    if args.mode == "estimate" and args.arms is not None:
        from live_smoke import arms
        flows = _pick(runner.load_flows(), _ids(args.ids))
        est = arms.estimate(flows, _ids(args.arms) or arms.DEFAULT_ARMS,
                            runner.RUNS, cap=args.cap)
        for r in est["rows"]:
            print(f"{r['flow'] + '@' + r['arm']:<48} ${r['usd']:.4f}  "
                  f"(Sonnet basis ${r['sonnet_basis_usd']:.4f}, references "
                  f"share {r['references_share']:.2f}: {r['basis']})")
        b = est["basis"]
        print(f"basis: {b['runs']} recorded w11-w14 Sonnet runs; "
              f"{b['with_references']} used the references helper (mean "
              f"${b['mean_usd_with_references']:.4f}, mean helper share "
              f"{b['references_share']:.2f}); {b['no_helper']} used no "
              f"helper (mean ${b['mean_usd_no_helper']:.4f})")
        print(f"{'total':<48} ${est['total_usd']:.4f}; suggested cap "
              f"${est['suggested_cap_usd']:.2f}"
              + ("" if args.cap is None else
                 f"; fits the ${args.cap:.2f} cap" if est["fits"] else
                 f"; OVER the ${args.cap:.2f} cap: cut {est['cut']}"))
        return 0
    if args.mode == "estimate":
        flows = runner.load_flows()
        picked = []
        # the same prefix rule as ``flow --ids`` (F18 picks F18 and F18j)
        for i in _ids(args.ids) or [f["id"] for f in flows]:
            hits = _pick(flows, [i])
            picked += [f for f in hits if f not in picked] or [{"id": i}]
        est = spend.estimate_flows(picked, args.model, runner.RUNS,
                                   pool=flows)
        for r in est["rows"]:
            print(f"{r['id']:<44} ${r['usd']:.4f}  {r['basis']}")
        print(f"{'total':<44} ${est['total_usd']:.4f} on {est['model']}; "
              f"suggested cap ${est['suggested_cap_usd']:.2f}")
        return 0

    if not args.wave or args.cap is None:
        ap.error(f"{args.mode} needs --wave and --cap")
    arm_names, models = [], [args.model]
    if args.mode == "arms":
        from live_smoke import arms
        arm_names = _ids(args.arms) or list(arms.DEFAULT_ARMS)
        for a in arm_names:
            arms.arm(a)
        models = arms.arm_models(arm_names)
        if not _ids(args.ids):
            ap.error("arms needs --ids (the flows to run in each arm)")
    wave = runner.Wave(args.wave, mode=args.mode, model_id=models[0],
                       cap=args.cap, ledger_path=args.ledger, fake=args.fake,
                       keep_sandbox=args.keep_sandbox, redo=args.redo,
                       extra_models=models[1:])
    try:
        wave.start()
    except spend.RefuseToStart as exc:
        print(f"[wave] REFUSED: {exc}")
        return 2
    ids = _ids(args.ids)
    try:
        if args.mode == "arms":
            from live_smoke import arms
            picked = _pick(runner.load_flows(), ids)
            for f, a in arms.order(picked, arm_names):
                if not wave.can_continue():
                    break
                wave.run_arm(f, a)
        elif args.mode == "flow":
            flows = [f for f in _pick(runner.load_flows(), ids)
                     if COST_ORDER.get(f.get("cost", "medium"), 1)
                     <= COST_ORDER[args.max_cost]]
            for f in flows:
                if not wave.can_continue():
                    break
                wave.run_flow(f)
        elif args.mode == "geotech":
            from funhouse_agent.deep.eval_harness import load_suite
            for q in _pick(load_suite(), ids):
                if not wave.can_continue():
                    break
                wave.run_geotech(q)
        elif args.mode == "suite":
            from funhouse_agent.review_eval.tasks import load_tasks
            tasks = [t.id for t in load_tasks()]
            for tid in [t for t in tasks if not ids or any(
                    t == i or t.startswith(i) for i in ids)]:
                if not wave.can_continue():
                    break
                wave.run_suite_task(tid, arm=args.arm, docs_dir=args.docs_dir,
                                    orientation=args.orientation)
    except KeyboardInterrupt:
        print("[wave] interrupted")
    finally:
        wave.finish()
        if args.mode == "arms":
            from live_smoke import arms
            print("[wave] results: " + arms.write_results(
                wave.dir, _pick(runner.load_flows(), ids), arm_names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
