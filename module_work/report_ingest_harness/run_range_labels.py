"""BACKLOG RI-73: the range voter on Claude, locally, every call metered.

Every page of each report is described by one model (one vision call a
page) and each report is then labelled by sections by another, in the arms
asked for, through the ``range_labels`` stage's own functions in
:mod:`report_ingest.cluster_scoring` with a ``ChatAnthropic`` on each side
(the engine Foundry runs over its own Claude classes). Run it from the repo
root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.run_range_labels \\
        run --only R18 --describe-model claude-sonnet-5-5 \\
        --label-model claude-opus-5-5 --arms plain hinted \\
        --wave ri73-smoke --cap 6 --out-dir <a gitignored folder>

    ... run_range_labels probe --only R18 --describe-model claude-haiku-5-5 \\
        --pages 4 --wave ri73-smoke --cap 6 --out-dir <a gitignored folder>

``run`` with no ``--only`` takes the thirteen in-sample reports the stage-b
labels cover (:data:`STAGE_B_REPORTS`). ``probe`` describes the first
``--pages`` pages of one report and prints what each cost, to price a
describer before a whole run; it labels nothing.

THE SPEND. One wave on the ONE ledger (the main checkout's
``module_work/live_smoke/spend_ledger.json``), shared with every other local
run through the lock-file ledger of ``run_claude_local``; the describer's
``ChatAnthropic`` carries the wave's :class:`SpendMeter`, the labeller's a
:class:`ModelMeter` of it. On top of the meter's own check (the cap is not
yet spent), :class:`Reserve` refuses a call whose ESTIMATE -- its words at
three characters a token, each picture at a full-size picture's tokens, and
an output allowance -- added to what is spent and what is still in flight
would pass the cap, so a run stops BEFORE the cap rather than one call past
it.

WHERE IT WRITES. ``--out-dir`` must be a folder git ignores (default the
main checkout's ``module_work/field_feedback/2026-10-24_claude_local/raw/
ri73/``): the descriptions and the sections' reasons quote the pages.
``ranges/descriptions/<ID>/`` holds one file a page, ``ranges/<arm>/<ID>.json``
each arm's labels, ``RESULTS.md`` the scores of every report finished in
the folder (IDs and numbers only; written fresh by each call) and
``spend_calls.jsonl`` every metered call. The same command again reuses what
is on disk: a description from the same describer and prompt version, and a
report's labels drawn by the same labeller from the same instructions;
``--redo`` asks again.

THE SCORES. Against the spreadsheet's hand labels only: a report in the
blind set (:data:`report_ingest.scoring.OOS_BLIND`) is refused before
anything about it is read, and no out-of-sample labels file is opened. To
set the arms beside the stored stage-b label sets::

    ... measure_range_labels --stage-b <stage b ingest folder> \\
        --runs <out-dir>/ranges/plain <out-dir>/ranges/hinted --only R18

PRIVACY. What this prints carries report IDs, page counts, tokens, dollars
and scores. The API key is read by ``live_smoke.models.api_key`` and never
printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from module_work.report_ingest_harness.run_claude_local import (
    LEDGER_REL, OUT_REL, RAW_REL, REPO, RetryLog, _append_jsonl, _inside,
    claude_chat, git_ignored, main_checkout, price_the_scorecard,
    shared_ledger, usd,
)

#: The in-sample reports whose stage-b labels the 2026-10-24 analysis
#: scored (R21 left no usable labels file).
STAGE_B_REPORTS = ("R09", "R11", "R12", "R13", "R15", "R16", "R18", "R20",
                   "R23", "R24", "R28", "R29", "R30")

#: What a full-size picture costs on the hi-res Claude models (planlens'
#: ``claude-hires`` budget: 4,784 patches of 28 px), used only to ESTIMATE
#: a call before it is sent.
PICTURE_TOKENS = 4800
#: The output a call is assumed to spend, thinking included, for the same
#: estimate: a description is short, a report's sections longer, and the
#: labeller thinks first.
OUTPUT_ALLOWANCE = {"describe": 3000, "label": 16000}

OK, REFUSED, STOPPED_BY_CAP = 0, 2, 3


def _estimate_tokens(messages: Any) -> int:
    """Input tokens a LangChain message batch will cost, roughly: words at
    three characters a token, each picture at :data:`PICTURE_TOKENS`."""
    total = 0.0
    for batch in messages or []:
        for message in batch or []:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                total += len(content) / 3.0
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    total += len(str(block)) / 3.0
                elif block.get("type") in ("image", "image_url"):
                    total += PICTURE_TOKENS
                else:
                    total += len(str(block.get("text") or "")) / 3.0
    return int(total)


def _reserve_class():
    from langchain_core.callbacks import BaseCallbackHandler

    from module_work.live_smoke.spend import WaveCapReached, rates_for

    class Reserve(BaseCallbackHandler):
        """Refuses a call whose estimate would carry the wave past its cap,
        counting the calls still in flight (pages described side by side)."""

        raise_error = True
        run_inline = True

        def __init__(self, ledger: Any, wave: str, cap: float, model: str,
                     output_tokens: int, pending: Dict[str, float],
                     lock: threading.Lock) -> None:
            super().__init__()
            self.ledger, self.wave, self.cap = ledger, wave, float(cap)
            self.model, self.output_tokens = model, int(output_tokens)
            self.pending, self.lock = pending, lock
            self.refused: Optional[str] = None

        def on_chat_model_start(self, serialized, messages, *, run_id,
                                **kwargs):
            tokens = _estimate_tokens(messages)
            rates = rates_for(self.model, tokens)
            guess = (tokens * rates["input"]
                     + self.output_tokens * rates["output"]) / 1e6
            with self.lock:
                spent = self.ledger.wave_spent(self.wave)
                flying = sum(self.pending.values())
                if spent + flying + guess > self.cap:
                    self.refused = (
                        f"reserve: ${spent:.2f} spent + ${flying:.2f} in "
                        f"flight + this call's ${guess:.2f} would pass the "
                        f"${self.cap:.2f} cap of wave {self.wave!r}")
                    raise WaveCapReached(self.refused)
                self.pending[str(run_id)] = guess

        def _done(self, run_id) -> None:
            with self.lock:
                self.pending.pop(str(run_id), None)

        def on_llm_end(self, response, *, run_id, **kwargs):
            self._done(run_id)

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._done(run_id)

    return Reserve


def _paths(args: argparse.Namespace) -> Dict[str, Path]:
    main = main_checkout()
    raw = Path(args.raw_dir) if args.raw_dir else main / RAW_REL
    out = Path(args.out_dir) if args.out_dir else main / OUT_REL / "ri73"
    return {"main": main, "raw": raw, "out": out,
            "ledger": main / LEDGER_REL}


def _refuse_unignored(out: Path, main: Path) -> bool:
    for root in {main, REPO}:
        if _inside(out, root) and git_ignored(out, root) is False:
            print(f"refused: {out} is inside the repo and git does not "
                  f"ignore it; descriptions quote the pages")
            return True
    return False


def _report_ids(only: str) -> List[str]:
    from report_ingest.scoring import OOS_BLIND

    ids = [t.strip() for t in (only or "").split(",") if t.strip()]
    ids = ids or list(STAGE_B_REPORTS)
    blind = [rid for rid in ids if rid in OOS_BLIND]
    if blind:
        raise SystemExit(f"refused: {', '.join(blind)} in the blind set; "
                         f"this runner scores against the spreadsheet only")
    return ids


def _corpus(raw: Path) -> Any:
    from module_work.report_ingest_harness import corpus as harness
    from report_ingest.corpus import Corpus

    return Corpus(raw / "corpus", di_dir=raw / "di",
                  labels_xlsx=raw / "corpus" / harness.LABELS_XLSX.name)


class _Wave:
    """The wave, its meters and the two chat models, opened together."""

    def __init__(self, args: argparse.Namespace, paths: Dict[str, Path],
                 mode: str) -> None:
        from module_work.live_smoke.spend import SpendMeter, rates_for

        self.describe_model = args.describe_model
        self.label_model = getattr(args, "label_model", None) \
            or args.describe_model
        for name in {self.describe_model, self.label_model}:
            rates_for(name, 0)                 # an unpriced model stops here
        self.ledger = shared_ledger(paths["ledger"])
        self.wave, self.cap = args.wave, float(args.cap)
        self.ledger.begin_wave(self.wave, self.cap,
                               model=self.describe_model, mode=mode)
        self.meter = SpendMeter(self.ledger, self.wave, self.cap,
                                model_id=self.describe_model)
        self.meter.scenario = mode
        pending: Dict[str, float] = {}
        lock = threading.Lock()
        Reserve = _reserve_class()
        self.reserves = [
            Reserve(self.ledger, self.wave, self.cap, self.describe_model,
                    OUTPUT_ALLOWANCE["describe"], pending, lock),
            Reserve(self.ledger, self.wave, self.cap, self.label_model,
                    OUTPUT_ALLOWANCE["label"], pending, lock)]
        self.describer = claude_chat(self.describe_model,
                                     [self.meter, self.reserves[0]])
        self.labeller = claude_chat(
            self.label_model,
            [self.meter.for_model(self.label_model), self.reserves[1]])
        price_the_scorecard({self.describe_model, self.label_model})

    def close(self, out: Path) -> None:
        self.ledger.end_wave(self.wave)
        _append_jsonl(out / "spend_calls.jsonl", self.meter.calls)

    def spend_lines(self) -> List[str]:
        by_model: Dict[str, Dict[str, float]] = {}
        for call in self.meter.calls:
            row = by_model.setdefault(call.get("model") or "?",
                                      {"calls": 0, "usd": 0.0, "errors": 0})
            row["calls"] += 1
            row["usd"] += float(call.get("usd") or 0.0)
            row["errors"] += 1 if call.get("error") else 0
        lines = [f"- {model}: {int(r['calls'])} call(s)"
                 + (f" ({int(r['errors'])} error)" if r["errors"] else "")
                 + f", {usd(r['usd'])}" for model, r in by_model.items()]
        this_run = sum(r["usd"] for r in by_model.values())
        spent = self.ledger.wave_spent(self.wave)
        lines.append(f"- this run {usd(this_run)}; wave {self.wave} "
                     f"{usd(spent)} of {usd(self.cap)}")
        if self.stopped:
            lines.append(f"- STOPPED: {self.stopped}")
        return lines

    @property
    def stopped(self) -> Optional[str]:
        return self.meter.stopped or next(
            (r.refused for r in self.reserves if r.refused), None)


ALL_ARMS = ("plain", "hinted")


def _every_finished(out: Path, ids: Sequence[str], describe_model: str,
                    label_model: str) -> Dict[str, Any]:
    """Every finished range run in ``out``, every arm under both
    conventions (``ranges/<arm>/`` and ``ranges/<arm>_B/``), so a call for
    one arm or one convention does not drop the others' columns."""
    from report_ingest import cluster_scoring as cs

    merged: Optional[Dict[str, Any]] = None
    for conventions in ("A", "B"):
        got = cs._saved_ranges(out, ids, ALL_ARMS, describe_model,
                               label_model, conventions)
        if merged is None:
            merged = got
            continue
        merged["runs"].update(got["runs"])
        merged["arms"] = list(merged["arms"]) + list(got["arms"])
    runs = merged["runs"]
    merged["arms"] = [a for a in merged["arms"] if runs.get(a)]
    return merged


def run(args: argparse.Namespace) -> int:
    from report_ingest import cluster_scoring as cs

    paths = _paths(args)
    out = paths["out"]
    if _refuse_unignored(out, paths["main"]):
        return REFUSED
    if not (paths["raw"] / "corpus").is_dir():
        print(f"no private corpus at {paths['raw']} (pass --raw-dir)")
        return REFUSED
    ids = _report_ids(args.only)
    out.mkdir(parents=True, exist_ok=True)
    corpus = _corpus(paths["raw"])
    wave = _Wave(args, paths, f"report_ingest:range_labels:"
                              f"{'+'.join(args.arms)}:{args.conventions}")
    engines = cs._Engines(None, wave.labeller, wave.describer)
    print(f"range labels: {len(ids)} report(s) {', '.join(ids)}; describer "
          f"{wave.describe_model}, labeller {wave.label_model}, arms "
          f"{', '.join(args.arms)}, conventions {args.conventions}; wave "
          f"{wave.wave} cap {usd(wave.cap)}; "
          f"out_dir {out}")
    retries = RetryLog()
    try:
        with retries:
            ranged = cs._run_ranges(
                corpus, engines, wave.describe_model, wave.label_model, out,
                ids, arms=tuple(args.arms), workers=args.workers,
                redo=args.redo, sync=None,
                deadline=time.time() + 60.0 * float(args.max_minutes),
                conventions=args.conventions)
    finally:
        wave.close(out)
        _append_jsonl(out / "sdk_retries.jsonl", retries.rows)
    # The table is every report finished in this folder, not only this
    # call's, and written fresh: a run done a report at a time ends with one
    # whole table rather than a pile of partial ones.
    every = _every_finished(out, sorted(set(STAGE_B_REPORTS) | set(ids)),
                            wave.describe_model, wave.label_model)
    every["failures"] = dict(ranged.get("failures") or {})
    scored = cs._score_ranges(every, corpus, {}, corpus.mapped_ids(),
                              ("insample",))
    done = sorted(set().union(*(set(r) for r in every["runs"].values())))
    lines = cs._render_ranges(scored)
    lines[2:2] = [f"Every report finished in this folder ({len(done)}): "
                  f"{', '.join(done)}. Written "
                  f"{time.strftime('%Y-%m-%d %H:%M')} after a call for "
                  f"{', '.join(ids)}.", ""]
    lines += ["", "## Spend (this call)", ""] + wave.spend_lines()
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n",
                                    encoding="utf-8")
    (out / "results.json").write_text(json.dumps(cs._plain(scored),
                                                 indent=2), encoding="utf-8")
    print("\n".join(lines))
    if wave.stopped:
        print(f"\nSTOPPED BY THE CAP: {wave.stopped}. Raise --cap on wave "
              f"{wave.wave} and run the same command to go on.")
        return STOPPED_BY_CAP
    return OK


def probe(args: argparse.Namespace) -> int:
    """The first ``--pages`` pages of one report described, and what each
    cost; nothing is labelled."""
    from report_ingest.engine import CostMeter, engine_for
    from report_ingest.range_labels import describe_pages

    paths = _paths(args)
    out = paths["out"]
    if _refuse_unignored(out, paths["main"]):
        return REFUSED
    (rid,) = _report_ids(args.only)[:1]
    wave = _Wave(args, paths, f"report_ingest:range_labels:probe")
    meter = CostMeter()
    folder = out / "probe" / wave.describe_model / rid
    try:
        doc = _corpus(paths["raw"]).open_report(rid, di="auto", warn=False)
        try:
            found = describe_pages(
                doc, engine_for(wave.describer, meter=meter),
                pages=range(min(int(args.pages), doc.n_pages)),
                folder=str(folder), workers=args.workers, redo=args.redo)
        finally:
            doc.close()
    finally:
        wave.close(out)
    print(f"probe {rid} on {wave.describe_model}: {len(found.pages)} "
          f"page(s), {len(found.unresolved)} unresolved")
    for page in sorted(found.pages):
        cost = found.pages[page].get("cost") or {}
        render = found.pages[page].get("render") or {}
        print(f"  p{page}: {render.get('width_px')}x{render.get('height_px')}"
              f" {render.get('format')}, {cost.get('input_tokens')} in / "
              f"{cost.get('output_tokens')} out, "
              f"${float(cost.get('dollars') or 0):.4f}")
    print("\n".join(wave.spend_lines()))
    return STOPPED_BY_CAP if wave.stopped else OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--only", default="",
                       help="report IDs, comma-separated (run: default the "
                            "thirteen stage-b reports; probe: the first)")
        p.add_argument("--describe-model", default="claude-sonnet-5-5")
        p.add_argument("--wave", required=True)
        p.add_argument("--cap", type=float, required=True)
        p.add_argument("--workers", type=int, default=4,
                       help="pages described at once")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--raw-dir", default=None)
        p.add_argument("--redo", action="store_true")
        if name == "run":
            p.add_argument("--conventions", choices=("A", "B"),
                           default="A",
                           help="the labeller's conventions: A (OWN-92's "
                                "default) or B (what the hand labels mostly "
                                "do); B's runs go in ranges/<arm>_B/")
            p.add_argument("--max-minutes", type=float, default=7.0,
                           help="stop sending pages to the describer after "
                                "this long; the same command again goes on")
            p.add_argument("--label-model", default="claude-opus-5-5")
            p.add_argument("--arms", nargs="+", default=["plain", "hinted"],
                           choices=["plain", "hinted"])
        else:
            p.add_argument("--pages", type=int, default=4)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args) if args.command == "run" else probe(args)


if __name__ == "__main__":
    sys.exit(main())
