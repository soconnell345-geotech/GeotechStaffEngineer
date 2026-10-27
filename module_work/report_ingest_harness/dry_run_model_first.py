"""Both model-first arms end to end at $0 (BACKLOG RI-87 piece 9).

``score_on_cluster`` runs the four reader stages twice on the private
corpus, ``model_first="off"`` (today's readers) and ``"on"``, over the first
items of each stage's answer sheets, with a replaying engine in the
Prompter's place: each call is answered with the reading the plain baseline
saved for that item (``plain_<stage>_opus``, 2026-10-24), in the schema the
call asks for. No model is called and nothing is spent. What it checks is
the plumbing on real reports before a paid wave: the page cuts and routes,
placement, the builders, the scorers, both arms' run files and RESULTS.md,
and that every run file names its arm. The scores it prints measure the
saved readings, not either arm's reader.

Run it from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.dry_run_model_first --items 2

PRIVACY. The corpus, the truth and the saved readings are read in place;
both arms write under the gitignored ``raw/`` folder
(``<main checkout>/module_work/field_feedback/2026-10-24_claude_local/raw/
dry_run_model_first/``). What it prints is item ids and counts only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import run_claude_local as rcl

#: The reader stages, each with the plain baseline arm whose readings it
#: replays.
STAGES: Dict[str, str] = {"logs": "plain_logs_opus",
                          "lab": "plain_lab_opus",
                          "calc": "plain_calc_opus",
                          "narrative": "plain_narrative_opus"}
#: The answer schema each stage's calls ask for, by class name (today's
#: classes and the model-first ones share the name).
READING: Dict[str, str] = {"logs": "LogReading", "lab": "LabSheetReading",
                           "calc": "CalcReading",
                           "narrative": "NarrativeReading"}
#: The name every engine is built under: one replaying engine serves all.
REPLAY = "replay"


class ReplayEngine:
    """Answers every call of the item it is set to with that item's saved
    reading, validated into the schema the call asks for; any other call
    (the narrative's summary, an item with no saved reading) gets prose
    and no answer, as a model that did not use the answer tool."""

    accepts_pdf = True

    def __init__(self, name: str, readings: Dict[Tuple[str, str], dict]
                 ) -> None:
        from report_ingest.tests.fake_engine import FakeEngine

        self._fake = FakeEngine([], name=name)
        self.name = self.model = name
        self.served_by = "replay (no model)"
        self.readings = readings
        self.stage = ""
        self.item = ""

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._fake.calls

    def complete(self, messages: Sequence[Dict[str, Any]], **kwargs: Any
                 ) -> Any:
        schema = kwargs.get("output_format")
        saved = self.readings.get((self.stage, self.item))
        turn: Dict[str, Any] = {"text": "No saved reading for this call."}
        if (saved is not None and schema is not None
                and getattr(schema, "__name__", "") == READING[self.stage]):
            turn = {"final": schema.model_validate(saved)}
        self._fake.turns.append(turn)
        return self._fake.complete(messages, **kwargs)


def _saved_readings(arms: Path) -> Dict[Tuple[str, str], dict]:
    """``(stage, item) -> the plain baseline's saved reading``."""
    out: Dict[Tuple[str, str], dict] = {}
    for stage, arm in STAGES.items():
        for path in sorted((arms / arm / stage).glob("*.json")):
            reading = (rcl._load(path) or {}).get("reading")
            if isinstance(reading, dict):
                out[(stage, path.stem)] = reading
    return out


def _wrap(engines: List[ReplayEngine]) -> List[Tuple[Any, str, Any]]:
    """Point each item's engine at its item: wrap the per-item functions
    the stages call (looked up at call time), so the engine knows whose
    reading to replay. Returns what to put back."""
    import report_ingest.calc_scoring as calc
    import report_ingest.lab_scoring as lab
    import report_ingest.log_scoring as logs
    import report_ingest.model_first_scoring as first
    import report_ingest.narrative_scoring as narrative

    def pointed(stage: str, read: Any, key: str) -> Any:
        def call(truth: Dict[str, Any], doc: Any, engine: Any, **kw: Any):
            engine.stage = stage
            engine.item = str(kw.get(key) if key == "report"
                              else truth.get("id") or "")
            engines.append(engine)
            return read(truth, doc, engine, **kw)
        return call

    swaps = [(logs, "score_one_log", "logs", "id"),
             (lab, "score_one_sheet", "lab", "id"),
             (calc, "score_one_calc", "calc", "id"),
             (narrative, "score_one_report", "narrative", "report"),
             (first, "first_log", "logs", "id"),
             (first, "first_sheet", "lab", "id"),
             (first, "first_calc", "calc", "id"),
             (first, "first_narrative", "narrative", "report")]
    undo = []
    for module, name, stage, key in swaps:
        original = getattr(module, name)
        undo.append((module, name, original))
        setattr(module, name, pointed(stage, original, key))
    return undo


def _overall(part: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    got = (part or {}).get("overall") or {}
    return int(got.get("found") or 0), int(got.get("total") or 0)


def _summary(out: Path, stage: str) -> Dict[str, Any]:
    """Counts over one arm's run files of one stage."""
    rows = [rcl._load(p) or {} for p in sorted((out / stage).glob("*.json"))]
    before = [0, 0]
    after = [0, 0]
    errors = 0
    arms = set()
    routes = set()
    placed = {"at_a_spot": 0, "with_text": 0, "values": 0}
    for row in rows:
        arms.add(str(row.get("model_first") or "unnamed"))
        if row.get("route"):
            routes.add(str(row["route"]))
        for key in placed:
            placed[key] += int((row.get("placement") or {}).get(key) or 0)
        if stage == "narrative":
            score = row.get("score") or {}
            errors += 1 if score.get("error") else 0
            recall = score.get("recall") or {}
            after[0] += int(recall.get("found") or 0)
            after[1] += int(recall.get("total") or 0)
            continue
        errors += 1 if (row.get("after") or {}).get("error") else 0
        for acc, side in ((before, "before"), (after, "after")):
            found, total = _overall(row.get(side))
            acc[0] += found
            acc[1] += total
    return {"items": len(rows), "errors": errors, "arms": sorted(arms),
            "routes": sorted(routes), "before": before, "after": after,
            "placed": placed,
            "befores": {p.stem: ((rcl._load(p) or {}).get("before")
                                 or {}).get("overall")
                        for p in sorted((out / stage).glob("*.json"))}}


def run(args: argparse.Namespace) -> int:
    import report_ingest.engine as engine_module
    from report_ingest.cluster_scoring import score_on_cluster

    main = rcl.main_checkout()
    raw = Path(args.raw_dir) if args.raw_dir else main / rcl.RAW_REL
    arms = Path(args.saved_dir) if args.saved_dir else main / rcl.OUT_REL
    root = Path(args.out_dir) if args.out_dir else \
        main / rcl.OUT_REL / "dry_run_model_first"
    refusal = rcl.out_dir_refusal(root, {main, rcl.REPO})
    if refusal:
        print(refusal)
        return rcl.REFUSED
    readings = _saved_readings(arms)
    stages = tuple(s for s in STAGES if not args.stages
                   or s in args.stages.split(","))
    print(f"saved readings: {len(readings)} over "
          f"{len({s for s, _i in readings})} stage(s)")
    for key in [k for k in os.environ if k.startswith(rcl.SWITCH_PREFIX)]:
        os.environ.pop(key)
    def replay_engine(prompter: Any, model: str, meter: Any = None,
                      **kwargs: Any) -> ReplayEngine:
        return ReplayEngine(model, readings)

    pointed: List[ReplayEngine] = []
    saved_engine = engine_module.PrompterEngine
    engine_module.PrompterEngine = replay_engine
    undo = _wrap(pointed)
    by_arm: Dict[str, List[ReplayEngine]] = {}
    try:
        for arm in ("off", "on"):
            start = len(pointed)
            out = root / arm
            if args.redo and out.is_dir():
                import shutil
                shutil.rmtree(out)
            score_on_cluster(
                reports_dir=raw / "corpus", di_dir=raw / "di", out_dir=out,
                prompter=object(), model=REPLAY, stages=stages,
                truth_dir=raw / "truth", sets=(), redo=False,
                max_reports=args.items, model_first=arm,
                stage_models={s: REPLAY for s in stages} if arm == "on"
                else None)
            by_arm[arm] = pointed[start:]
    finally:
        engine_module.PrompterEngine = saved_engine
        for module, name, original in undo:
            setattr(module, name, original)

    print("\nBoth arms, counts only (the scores measure the saved readings, "
          "not a reader):")
    for stage in stages:
        off = _summary(root / "off", stage)
        on = _summary(root / "on", stage)
        calls = {arm: sum(len(e.calls) for e in by_arm.get(arm, [])
                          if e.stage == stage)
                 for arm in ("off", "on")}
        print(f"  {stage}: off {off['items']} item(s), arm {off['arms']}, "
              f"{calls['off']} call(s), "
              f"{off['errors']} error(s), before {off['before'][0]}/"
              f"{off['before'][1]}, after {off['after'][0]}/"
              f"{off['after'][1]} | on {on['items']} item(s), arm "
              f"{on['arms']}, {calls['on']} call(s), routes "
              f"{on['routes']}, {on['errors']} "
              f"error(s), before {on['before'][0]}/{on['before'][1]}, "
              f"after {on['after'][0]}/{on['after'][1]}, placed "
              f"{on['placed']['at_a_spot']} of {on['placed']['with_text']} "
              f"on pages with text ({on['placed']['values']} values)")
        if stage != "narrative":
            same = off["befores"] == on["befores"]
            print(f"    before column the same in both arms: {same}")
    print(f"  replayed calls: {sum(len(e.calls) for e in pointed)} "
          f"(no model called)")
    print(f"\noutputs (gitignored): {root}")
    return rcl.OK if pointed else rcl.REFUSED


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--items", type=int, default=2,
                    help="the first N answer sheets of each stage")
    ap.add_argument("--stages", default="",
                    help="comma-separated stages (default: all four)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--raw-dir", default=None,
                    help="the private folder holding corpus/, di/ and "
                         "truth/ (default: the main checkout's)")
    ap.add_argument("--saved-dir", default=None,
                    help="the folder holding the plain_<stage>_opus arms "
                         "(default: the 2026-10-24 claude_local raw folder)")
    ap.add_argument("--redo", action="store_true",
                    help="clear both arms' folders first")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
