"""The PLAIN baseline: Claude handed an item's pages and asked to extract.

The owner asked on 2026-10-24 whether all of report ingest's machinery
improves the outcome over "just telling the Opus 5.5 FDE to extract the
report data". This script measures that baseline on the same hand-truthed
items the readers are scored on, with none of the machinery:

* **the pages**: each item's own pages -- the pages its truth file names, as
  the stage picks them -- cut out of the report PDF into a small PDF and sent
  as a native PDF document block, so the Anthropic API reads both the text
  layer and the page images. Nothing else is sent: no page triage, no brief,
  no text-layer cells, no Document Intelligence text, no starting record;
* **the instruction**: one short instruction per stage in general terms
  (:data:`INSTRUCTIONS`), with the vocabulary the answer schema's own fields
  point at ("one of the kind vocabulary you were given") and nothing about
  depth gates, halts, links or provenance rules;
* **the answer**: the SAME answer schema the stage's reader takes
  (``LogReading``, ``LabSheetReading``, ``CalcReading``,
  ``NarrativeReading``), offered as the ``answer`` tool by the same engine
  (:class:`report_ingest.engine.ChatModelEngine`, with a PDF block allowed in
  the user turn). ONE call per item: no continuation, no follow-up, no re-ask;
* **the record**: built from that answer alone by the stage's own builder --
  the step the stage's ``model`` column goes through -- with no floor and no
  merge. The log builder's depth gate is opened wide (no grid, so no ruler
  window: every depth the model gives is kept; the stage's own gate would
  refuse them all); the calculation builder gets no pool of printed numbers
  (that only lowers a result's confidence, never its score); the narrative
  builder gets no caption count and no logs (the model's own counts stand);
* **the score**: the SAME scorer and tolerances as the stage
  (``score_record`` for logs, lab and calc, ``score_narrative``), so the
  numbers sit beside the pipeline runs' ``model`` and ``after`` columns.

Run it from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.run_plain_baseline \\
        run --stage calc --model claude-opus-5-5 \\
        --wave plain-calc-opus --cap 2.5

    ... run_plain_baseline run --stage logs --model claude-opus-5-5 \\
        --only R06_p51 --wave plain-smoke-opus --cap 0.6 --arm plain_smoke_opus

    ... run_plain_baseline report --stage calc       # $0: RESULTS.md again

METERED AND RESUMABLE exactly as :mod:`.run_claude_local`: the wave is
opened on the one shared ledger (the main checkout's), a ``SpendMeter`` sits
on the ``ChatAnthropic`` object, and the first call past the wave's cap or
the hard total is refused before it goes out; that item gets no run file, so
the same command (with a raised ``--cap``) reads it next time. Items that
already have a run file are skipped. ``--time-limit`` (seconds) stops
starting new items after that long, so one command stays inside a ten-minute
window and the next one goes on.

WHERE IT WRITES: ``<main checkout>/module_work/field_feedback/
2026-10-24_claude_local/raw/plain_<stage>_<model>/`` (gitignored; an
``out_dir`` git would track is refused): ``<stage>/<item>.json`` per item
(the answer, the record, the scores, the cost), ``spend_calls.jsonl`` and
``RESULTS.md``, which is written afresh from the run files every time and
sets each item beside the pipeline's arm (``--pipeline-arm``; by default the
Opus arms of 2026-10-24) RE-SCORED with the current scorers at $0.

PRIVACY. The corpus and the truth are read in place from the private
``raw/`` folder; the cut PDFs are never written to disk. What this script
prints carries item IDs, counts, metric names, tokens and dollars only; the
run files and RESULTS.md stay in the gitignored folder. The API key is read
by ``live_smoke.models`` and never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import run_claude_local as rcl

#: The stages this baseline runs.
STAGES: Tuple[str, ...] = ("logs", "lab", "calc", "narrative")

#: The pipeline arms each stage is set beside: the Opus runs of 2026-10-24.
PIPELINE_ARMS: Dict[str, str] = {"logs": "logs_opus",
                                 "lab": "lab_opus_linkon",
                                 "calc": "calc_opus",
                                 "narrative": "narrative_opus"}

#: The Anthropic API's limits on one PDF document (pages, bytes).
MAX_PDF_PAGES = 100
MAX_PDF_BYTES = 32 * 1024 * 1024

#: A cut larger than this is sanitized (:func:`cut_pdf`).
SANITIZE_ABOVE = 8 * 1024 * 1024

#: Seconds after which no new item is started (one command, < 10 minutes).
DEFAULT_TIME_LIMIT = 420.0


# ---------------------------------------------------------------------------
# the instruction: one short paragraph per stage, in general terms
# ---------------------------------------------------------------------------

INSTRUCTIONS: Dict[str, str] = {
    "logs": (
        "These pages are from a geotechnical report ({where}). Extract the "
        "exploration log on them (a boring, test pit or similar log): its "
        "header (the hole's identifier, location, elevation, total depth, "
        "dates, drilling method, hammer, who logged it), every layer with "
        "its depths, description and the USCS symbol where one is printed, "
        "every sample with its depths and any laboratory values printed "
        "beside it, every SPT or other blow-count record, and the water "
        "levels. Give depths in the log's own unit and say which unit. "
        "{pages_note} Return it with the answer tool."),
    "lab": (
        "These pages are from a geotechnical report ({where}). Extract the "
        "laboratory test results on them: for each test, its kind, the "
        "boring or test pit and the sample depth it belongs to, and every "
        "value it reports (index values, grading points, curve points, "
        "strengths and so on) with the units as printed. Give depths in the "
        "sheet's own unit and say which unit. {pages_note}\n\n"
        "Test kinds:\n{kinds}\n\nSeries names:\n{series}\n\n"
        "Return it with the answer tool."),
    "calc": (
        "These pages are from a geotechnical report ({where}). They hold one "
        "calculation. Extract it: what kind of calculation it is, the "
        "program that printed it if one is named, the method, what it is "
        "for, the labelled values it was given (inputs) and the labelled "
        "values it worked out (results), each with its label and unit as "
        "printed. {pages_note}\n\nCalculation kinds:\n{kinds}\n\n"
        "Return it with the answer tool."),
    "narrative": (
        "This is a geotechnical report ({where}). Answer each field of the "
        "answer tool from it, and leave a field null when the report does "
        "not say. {pages_note}\n\n{vocabulary}\n\n"
        "Return it with the answer tool."),
}


def _kinds_text(definitions: Dict[str, str]) -> str:
    return "\n".join(f"  {name}: {text}" for name, text in definitions.items())


def instruction(stage: str, report: str, pages: Sequence[int],
                whole: bool = False) -> str:
    """The plain instruction for one item."""
    pages = [int(p) for p in pages]
    if whole:
        where = f"report {report}, all {len(pages)} pages"
        note = ("Wherever a page is asked for, give the 0-based page index "
                "(the PDF's first page is 0).")
    else:
        where = (f"report {report}, page{'s' if len(pages) > 1 else ''} "
                 + ", ".join(str(p) for p in pages))
        mapping = ", ".join(f"PDF page {n + 1} is page {p}"
                            for n, p in enumerate(pages))
        note = (f"The attached PDF holds only these pages, in order; "
                f"wherever a page is asked for, give the report's 0-based "
                f"page index ({mapping}).")
    fill: Dict[str, str] = {"where": where, "pages_note": note}
    if stage == "lab":
        from report_ingest.lab_reader import KIND_DEFINITIONS, SERIES_NAMES
        fill.update(kinds=_kinds_text(KIND_DEFINITIONS),
                    series=_kinds_text(SERIES_NAMES))
    elif stage == "calc":
        from report_ingest.calc_reader import CALC_KIND_DEFINITIONS
        fill.update(kinds=_kinds_text(CALC_KIND_DEFINITIONS))
    elif stage == "narrative":
        from report_ingest.narrative_reader import _vocabulary_block
        fill.update(vocabulary=_vocabulary_block())
    return INSTRUCTIONS[stage].format(**fill)


def reading_class(stage: str) -> Any:
    """The answer schema the stage's reader takes."""
    if stage == "logs":
        from report_ingest.log_reader import LogReading
        return LogReading
    if stage == "lab":
        from report_ingest.lab_reader import LabSheetReading
        return LabSheetReading
    if stage == "calc":
        from report_ingest.calc_reader import CalcReading
        return CalcReading
    from report_ingest.narrative_reader import NarrativeReading
    return NarrativeReading


# ---------------------------------------------------------------------------
# the pages, as a PDF, and the engine that sends one
# ---------------------------------------------------------------------------

def cut_pdf(source: Path, pages: Optional[Sequence[int]]
            ) -> Tuple[bytes, int, bool]:
    """``(the PDF's bytes, its page count, sanitized)``: ``pages`` (0-based,
    in order) copied out of ``source`` with their text layer and images, or
    the whole file for ``None``. Nothing is written to disk.

    A copied page keeps its whole resource dictionary, and some reports hang
    every font and form of the file on every page: four corpus items cut to
    25-284 MB, past the API's 32 MB. A cut larger than
    :data:`SANITIZE_ABOVE` is therefore SANITIZED (its content streams
    rewritten and the resources they never use dropped: 69 MB to 74 KB);
    on those four the text and a 40 dpi rendering came out identical. Not
    every page does -- one other corpus page renders differently once
    sanitized -- so a cut that fits is sent as copied."""
    import pymupdf

    if pages is None:
        data = source.read_bytes()
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return data, doc.page_count, False
    with pymupdf.open(str(source)) as src:
        out = pymupdf.open()
        try:
            for page in pages:
                page = int(page)
                if not 0 <= page < src.page_count:
                    raise ValueError(f"page {page} is not in {source.name} "
                                     f"({src.page_count} pages)")
                out.insert_pdf(src, from_page=page, to_page=page)
            data = out.tobytes(garbage=3, deflate=True)
            if len(data) <= SANITIZE_ABOVE:
                return data, out.page_count, False
            for page in out:
                page.clean_contents(sanitize=True)
            return (out.tobytes(garbage=4, deflate=True, clean=True),
                    out.page_count, True)
        finally:
            out.close()


def pdf_block(data: bytes) -> Dict[str, Any]:
    """A PDF for the model to read: the engine's own block (RI-87)."""
    from report_ingest.engine import pdf_block as block

    return block(data)


def _engine_class() -> Any:
    """The stage's own engine with the PDF block on.

    The engine sends a :func:`pdf_block` as LangChain's ``file`` block,
    which ``langchain_anthropic`` sends as an Anthropic ``document`` block
    (text layer and page images both); :func:`read_item` puts it first in
    the turn, ahead of the words, as the API advises. This script reads
    over Anthropic's own API, which takes one, so the PDF is allowed
    outright rather than read off the model (the tests' stand-in is no
    ``ChatAnthropic``). Everything else -- the answer tool, its schema,
    how an answer is taken -- is the stage's own engine unchanged."""
    from functools import partial

    from report_ingest.engine import ChatModelEngine

    return partial(ChatModelEngine, accepts_pdf=True)


# ---------------------------------------------------------------------------
# the answer, as a record, scored the stage's way
# ---------------------------------------------------------------------------

#: The log builder's depth window with no ruler to bound it: every depth.
OPEN_WINDOW = (-math.inf, math.inf)


def item_pages(stage: str, truth: Dict[str, Any]) -> List[int]:
    """The item's own pages, as the stage picks them."""
    if stage == "logs":
        return [int(p) for p in truth.get("pages") or []]
    if stage == "lab":
        from report_ingest.lab_scoring import pages_of
        return pages_of(truth)
    from report_ingest.calc_scoring import pages_of
    return pages_of(truth)


def build_and_score(stage: str, truth: Dict[str, Any], reading: Any,
                    pages: Sequence[int], report: str
                    ) -> Dict[str, Any]:
    """The record built from the answer alone and its score: the run file's
    ``after`` (``score`` for the narrative)."""
    pages = [int(p) for p in pages]
    if stage == "logs":
        from report_ingest.log_reader import _Builder
        from report_ingest.log_scoring import (
            _scored, dump_investigations, score_record,
        )
        builder = _Builder(reading, OPEN_WINDOW, pages, report)
        investigation = builder.build()
        out = _scored(score_record(truth, [investigation]))
        out["record"] = dump_investigations([investigation])
    elif stage == "lab":
        from report_ingest.lab_reader import _Builder
        from report_ingest.lab_scoring import (
            _as_blob, _dump_tests, score_record, score_unlinked,
        )
        builder = _Builder(reading, pages, report, [])
        tests = builder.build()
        out = _as_blob(score_record(truth, tests))
        out["record"] = _dump_tests(tests)
        out["unlinked"] = _as_blob(score_unlinked(truth, tests))
    elif stage == "calc":
        from report_ingest.calc_reader import _Builder
        from report_ingest.calc_scoring import _dump, _scored, score_record
        builder = _Builder(reading, pages, report, [], [])
        calculation = builder.build()
        out = _scored(score_record(truth, calculation))
        out["record"] = _dump(calculation)
    else:
        from report_ingest.narrative_reader import _Builder
        from report_ingest.narrative_scoring import score_narrative
        builder = _Builder(pages)
        builder.add(reading)
        general, hazards, _facts = builder.build(pages, {}, None)
        score = score_narrative(truth, general, hazards, report=report)
        score.pages = pages
        score.unresolved = len(builder.unresolved)
        score.model_calls = 1
        return score.to_dict()
    out["unresolved"] = len(builder.unresolved)
    out["error"] = None
    return out


def rescore_pipeline(stage: str, truth: Dict[str, Any], blob: Dict[str, Any]
                     ) -> Dict[str, Any]:
    """A pipeline run file's ``after`` (or narrative ``score``) re-scored
    with the CURRENT scorer, at $0 -- the calculation merge redone with the
    current merge, as the pipeline would now produce it. The saved one when
    it cannot be re-scored."""
    try:
        if stage == "narrative":
            from report_ingest.narrative_scoring import rescore_saved
            got = rescore_saved(truth, blob)
            return got.to_dict() if got is not None else (blob.get("score")
                                                          or {})
        after = blob.get("after") or {}
        if stage == "logs":
            from report_ingest.log_scoring import rescore_saved
            got = rescore_saved(truth, after)
        elif stage == "lab":
            from report_ingest.lab_scoring import rescore_saved
            got = rescore_saved(truth, after)
        else:
            from report_ingest.calc_scoring import rescore_saved
            got = rescore_saved(truth, after, remerge=True)
        return got if got is not None else after
    except Exception:                                   # noqa: BLE001
        return blob.get("score" if stage == "narrative" else "after") or {}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _paths(args: argparse.Namespace) -> Dict[str, Path]:
    main = rcl.main_checkout()
    raw = Path(args.raw_dir) if args.raw_dir else main / rcl.RAW_REL
    return {"main": main, "raw": raw,
            "reports": Path(args.reports_dir) if args.reports_dir
            else raw / "corpus",
            "truth": Path(args.truth_dir) if args.truth_dir
            else raw / "truth",
            "ledger": Path(args.ledger) if getattr(args, "ledger", None)
            else main / rcl.LEDGER_REL,
            "pipeline": Path(args.pipeline_dir) if args.pipeline_dir
            else main / rcl.OUT_REL}


def _out_dir(args: argparse.Namespace, paths: Dict[str, Path]) -> Path:
    arm = args.arm or f"plain_{args.stage}_{rcl.short_model(args.model)}"
    return Path(args.out_dir) if args.out_dir else paths["main"] / \
        rcl.OUT_REL / arm


def _saved_ok(path: Path) -> bool:
    """A run file whose one call came back -- with an answer or without one,
    which is the plain arm's result either way -- or that was skipped. A
    call that raised (a network or API error) is read again."""
    blob = rcl._load(path)
    return bool(blob) and ("stop_reason" in blob or "skipped" in blob)


def _set_of(report: str, openset: Sequence[str]) -> str:
    return "open" if report in set(openset) else "blind"


def read_item(stage: str, truth_path: Path, reports: Path, engine: Any,
              openset: Sequence[str]) -> Dict[str, Any]:
    """One item: cut, ask once, build, score. Raises what the call raised."""
    from report_ingest.engine import text_block, user

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    item = truth_path.stem
    report = rcl.report_of(item)
    whole = stage == "narrative"
    source = reports / f"{report}.pdf"
    pages = None if whole else item_pages(stage, truth)
    data, n_pages, sanitized = cut_pdf(source, pages)
    if whole:
        pages = list(range(n_pages))
    blob: Dict[str, Any] = {
        "item": item, "report": report, "set": _set_of(report, openset),
        "stage": stage, "arm": "plain", "pages": list(pages),
        "pdf_pages": n_pages, "pdf_bytes": len(data),
        "pdf_sanitized": sanitized}
    if n_pages > MAX_PDF_PAGES or len(data) > MAX_PDF_BYTES:
        blob["skipped"] = (f"the PDF is {n_pages} pages and {len(data):,} "
                           f"bytes; the API takes at most {MAX_PDF_PAGES} "
                           f"pages and {MAX_PDF_BYTES:,} bytes in one "
                           f"document")
        return blob
    said = instruction(stage, report, pages, whole=whole)
    blob["instruction_chars"] = len(said)
    started = time.time()
    reply = engine.complete([user(pdf_block(data), text_block(said))],
                            output_format=reading_class(stage))
    blob["seconds"] = round(time.time() - started, 1)
    blob["served_by"] = engine.served_by
    blob["stop_reason"] = reply.stop_reason
    key = "score" if whole else "after"
    if reply.parsed is None:
        blob[key] = {"error": f"no structured answer (stop_reason "
                              f"{reply.stop_reason!r}, notes "
                              f"{'; '.join(reply.notes)[:300]!r})"}
    else:
        blob["reading"] = reply.parsed.model_dump(mode="json")
        try:
            blob[key] = build_and_score(stage, truth, reply.parsed, pages,
                                        report)
        except Exception as exc:                        # noqa: BLE001
            blob[key] = {"error": f"record not built: "
                                  f"{type(exc).__name__}: {exc}"}
    blob["cost"] = engine.meter.to_dict()
    return blob


def run(args: argparse.Namespace,
        chat_builder: Callable[[str, Sequence[Any]], Any] = rcl.claude_chat
        ) -> int:
    from module_work.live_smoke.spend import (RefuseToStart, SpendMeter,
                                              rates_for)
    from report_ingest.engine import CostMeter

    stage, model = args.stage, args.model
    rates_for(model, 0)                          # an unpriced model stops here
    paths = _paths(args)
    if not paths["reports"].is_dir():
        print(f"no reports at {paths['reports']}")
        return rcl.REFUSED
    out = _out_dir(args, paths)
    for root in {paths["main"], rcl.REPO}:
        if rcl._inside(out, root) and rcl.git_ignored(out, root) is False:
            print(f"refused: {out} is inside the repo and git does not "
                  f"ignore it; run files hold private content")
            return rcl.REFUSED
    (out / stage).mkdir(parents=True, exist_ok=True)
    settings = {"stage": stage, "model": model, "arm": "plain",
                "instruction": INSTRUCTIONS[stage]}
    saved = rcl._load(out / "arm.json")
    if saved and {k: saved.get(k) for k in settings} != settings:
        print(f"refused: {out} holds a run with other settings "
              f"({', '.join(k for k in settings if saved.get(k) != settings[k])}"
              f"); use another --arm or --out-dir")
        return rcl.REFUSED
    if not saved:
        (out / "arm.json").write_text(json.dumps(
            {**settings, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
            indent=2), encoding="utf-8")

    only = [t.strip() for t in args.only.split(",") if t.strip()]
    chosen = rcl.pick_items(stage, paths["truth"], args.sets, only)
    openset = rcl.open_reports(stage, paths["truth"] / stage)
    todo = [p for p in chosen
            if args.redo or not _saved_ok(out / stage / f"{p.stem}.json")]
    print(f"plain {stage} on {model}: {len(chosen)} item(s) chosen, "
          f"{len(chosen) - len(todo)} already read, {len(todo)} to read; "
          f"wave {args.wave} cap {rcl.usd(args.cap)}")
    if not todo:
        write_results(out, stage, paths, args.pipeline_arm)
        return rcl.OK

    ledger = rcl.shared_ledger(paths["ledger"])
    try:
        ledger.begin_wave(args.wave, args.cap, model=model,
                          mode=f"report_ingest:plain:{stage}")
    except RefuseToStart as exc:
        print(f"refused: {exc}")
        return rcl.REFUSED
    meter = SpendMeter(ledger, args.wave, args.cap, model_id=model)
    meter.scenario = f"plain_{stage}"
    reader = chat_builder(model, [meter])
    rcl.price_the_scorecard({model})
    Engine = _engine_class()
    retries = rcl.RetryLog()
    started = time.time()
    capped: Optional[str] = None
    try:
        with retries:
            for n, path in enumerate(todo, 1):
                if time.time() - started > args.time_limit:
                    print(f"  time limit ({args.time_limit:.0f} s) reached; "
                          f"{len(todo) - n + 1} item(s) left for the next "
                          f"run of the same command")
                    break
                first = len(meter.calls)
                engine = Engine(reader, meter=CostMeter())
                try:
                    blob = read_item(stage, path, paths["reports"], engine,
                                     openset)
                except Exception as exc:                # noqa: BLE001
                    said = f"{type(exc).__name__}: {exc}"
                    if rcl.CAP_TEXT.search(said):
                        capped = said[:200]
                        print(f"  [{n}/{len(todo)}] {path.stem}: refused by "
                              f"the cap; no run file, read again next time")
                        break
                    blob = {"item": path.stem,
                            "report": rcl.report_of(path.stem),
                            "set": _set_of(rcl.report_of(path.stem), openset),
                            "stage": stage, "arm": "plain",
                            "score" if stage == "narrative" else "after":
                                {"error": said[:500]},
                            "cost": engine.meter.to_dict()}
                for call in meter.calls[first:]:
                    call["item"] = path.stem
                    call["pdf_pages"] = blob.get("pdf_pages")
                (out / stage / f"{path.stem}.json").write_text(
                    json.dumps(blob, indent=1, default=str), encoding="utf-8")
                print(f"  [{n}/{len(todo)}] {path.stem}: "
                      f"{_score_text(stage, blob)}, "
                      f"{blob.get('pdf_pages', '?')} page(s), "
                      f"{(blob.get('cost') or {}).get('input_tokens', 0):,} in"
                      f" / {(blob.get('cost') or {}).get('output_tokens', 0):,}"
                      f" out, ${(blob.get('cost') or {}).get('dollars', 0):.4f},"
                      f" {blob.get('seconds', 0):.0f} s")
    finally:
        ledger.end_wave(args.wave)
        rcl._append_jsonl(out / "spend_calls.jsonl", (
            {**call, "arm": f"plain_{stage}"} for call in meter.calls))
        rcl._append_jsonl(out / "sdk_retries.jsonl", retries.rows)

    lines = write_results(out, stage, paths, args.pipeline_arm)
    w = ledger.wave(args.wave)
    print("\n".join(lines[:40]))
    print(f"\nwave {args.wave}: {rcl.usd(float(w.get('spent_usd') or 0))} of "
          f"{rcl.usd(args.cap)}; ledger ${ledger.cumulative:.2f} of "
          f"${ledger.hard_total:.2f}")
    if capped or meter.stopped:
        print(f"STOPPED BY THE CAP: {capped or meter.stopped}")
        return rcl.STOPPED_BY_CAP
    return rcl.OK


# ---------------------------------------------------------------------------
# RESULTS.md: the plain arm beside the pipeline's, at $0
# ---------------------------------------------------------------------------

def _overall(part: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not part or part.get("error"):
        return None
    got = part.get("overall") or {}
    return int(got.get("found") or 0), int(got.get("total") or 0)


def _recall(part: Optional[Dict[str, Any]], key: str = "recall"
            ) -> Optional[Tuple[int, int]]:
    if not part or part.get("error"):
        return None
    got = part.get(key) or {}
    return int(got.get("found") or 0), int(got.get("total") or 0)


def _score_text(stage: str, blob: Dict[str, Any]) -> str:
    if blob.get("skipped"):
        return "skipped"
    if stage == "narrative":
        part = blob.get("score") or {}
        if part.get("error"):
            return "ERROR"
        r, p = _recall(part), _recall(part, "precision")
        return f"recall {r[0]}/{r[1]}, precision {p[0]}/{p[1]}"
    got = _overall(blob.get("after"))
    return "ERROR" if got is None else f"{got[0]}/{got[1]}"


def _metric_counts(part: Optional[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for name, row in ((part or {}).get("scores") or {}).items():
        out[name] = (int(row.get("found") or 0), int(row.get("total") or 0))
    return out


def _frac(pair: Optional[Tuple[int, int]]) -> str:
    if pair is None:
        return "-"
    found, total = pair
    return f"{found}/{total}" + (f" ({100 * found / total:.0f}%)"
                                 if total else "")


def _add(acc: List[int], pair: Optional[Tuple[int, int]]) -> None:
    if pair is not None:
        acc[0] += pair[0]
        acc[1] += pair[1]


def _record_facts(stage: str, part: Dict[str, Any]) -> str:
    """Counts and units off a record (no values), for the causes column."""
    record = (part or {}).get("record")
    if stage == "logs" and isinstance(record, list) and record:
        inv = record[0]
        return (f"unit {inv.get('depth_unit') or '(none)'}; "
                f"{len(inv.get('layers') or [])} layers, "
                f"{len(inv.get('samples') or [])} samples, "
                f"{len(inv.get('spt') or [])} drives, "
                f"{len(inv.get('water') or [])} water")
    if stage == "lab" and isinstance(record, list):
        kinds = sorted({str(t.get("kind")) for t in record})
        linked = sum(1 for t in record if t.get("investigation_id"))
        return (f"{len(record)} tests ({', '.join(kinds) or '-'}), "
                f"{linked} with a hole")
    if stage == "calc" and isinstance(record, dict):
        return (f"kind {record.get('kind')}; "
                f"{len(record.get('inputs') or [])} inputs, "
                f"{len(record.get('results') or [])} results")
    return ""


def compare_rows(out: Path, stage: str, paths: Dict[str, Path],
                 pipeline_arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row per plain item: the plain score, the pipeline's ``model`` and
    ``after`` re-scored with the current scorer, and what differs."""
    arm_dir = paths["pipeline"] / (pipeline_arm or PIPELINE_ARMS[stage])
    truth_dir = paths["truth"] / stage
    openset = rcl.open_reports(stage, truth_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted((out / stage).glob("*.json")):
        blob = rcl._load(path) or {}
        item = path.stem
        truth_path = truth_dir / f"{item}.json"
        truth = rcl._load(truth_path) or {}
        theirs = rcl._load(arm_dir / stage / f"{item}.json")
        row: Dict[str, Any] = {
            "item": item, "set": _set_of(rcl.report_of(item), openset),
            "cost": float((blob.get("cost") or {}).get("dollars") or 0.0),
            "in": int((blob.get("cost") or {}).get("input_tokens") or 0),
            "out": int((blob.get("cost") or {}).get("output_tokens") or 0),
            "seconds": float(blob.get("seconds") or 0.0),
            "skipped": blob.get("skipped")}
        if stage == "narrative":
            mine = blob.get("score") or {}
            row["plain"] = _recall(mine)
            row["plain_p"] = _recall(mine, "precision")
            row["error"] = mine.get("error")
            if theirs:
                again = rescore_pipeline(stage, truth, theirs)
                row["after"] = _recall(again)
                row["after_p"] = _recall(again, "precision")
                row["model"] = None
                row["verdicts"] = {
                    "plain": _verdicts(mine), "after": _verdicts(again)}
            rows.append(row)
            continue
        mine = blob.get("after") or {}
        row["plain"] = _overall(mine)
        row["error"] = mine.get("error")
        row["metrics_plain"] = _metric_counts(mine)
        row["facts_plain"] = _record_facts(stage, mine)
        if stage == "lab":
            row["unlinked_plain"] = _overall(mine.get("unlinked"))
        if theirs:
            again = rescore_pipeline(stage, truth, theirs)
            row["after"] = _overall(again)
            row["model"] = _overall(again.get("model_alone"))
            row["metrics_after"] = _metric_counts(again)
            row["metrics_model"] = _metric_counts(again.get("model_alone"))
            row["facts_after"] = _record_facts(stage, again)
            row["page_sources"] = theirs.get("page_sources")
            if stage == "lab":
                row["unlinked_after"] = _overall(again.get("unlinked"))
        rows.append(row)
    return rows


def _verdicts(part: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for field in part.get("fields") or []:
        verdict = str(field.get("verdict") or "")
        out[verdict] = out.get(verdict, 0) + 1
    return out


def _diff_text(row: Dict[str, Any]) -> str:
    """The metrics where the plain arm and the pipeline's after differ."""
    mine, theirs = row.get("metrics_plain") or {}, row.get(
        "metrics_after") or {}
    parts = []
    for name in sorted(set(mine) | set(theirs)):
        a, b = mine.get(name, (0, 0)), theirs.get(name, (0, 0))
        if a[0] != b[0]:
            total = max(a[1], b[1])
            parts.append(f"{name} {a[0]} vs {b[0]} of {total}")
    return "; ".join(parts) or "same"


def write_results(out: Path, stage: str, paths: Dict[str, Path],
                  pipeline_arm: Optional[str] = None) -> List[str]:
    """RESULTS.md, written afresh from the run files: the plain arm beside
    the pipeline's arm, per set and per item; and the spend."""
    rows = compare_rows(out, stage, paths, pipeline_arm)
    arm = pipeline_arm or PIPELINE_ARMS[stage]
    lines = [f"# The plain baseline: {stage}", "",
             f"Each item's pages as one PDF document block, one short "
             f"instruction, the stage's own answer schema as the answer tool, "
             f"ONE call; the record built from the answer alone by the "
             f"stage's builder and scored by the stage's scorer. Beside it, "
             f"the pipeline's arm `{arm}`, re-scored with the current "
             f"scorers at $0"
             + (" (the calculation merge redone with the current merge)"
                if stage == "calc" else "")
             + ". `model` is the pipeline reader's answer alone (it had the "
               "brief and the starting record in view; the log depth gate "
               "applied); `after` is the pipeline's record. The plain arm is "
               "built the way `model` is.", ""]
    if stage == "narrative":
        lines += ["| set | items | plain recall | plain precision | "
                  "pipeline recall | pipeline precision |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in ("open", "blind", "all"):
            group = [r for r in rows if not r.get("skipped")
                     and (name == "all" or r["set"] == name)]
            if not group:
                continue
            sums = {k: [0, 0] for k in ("plain", "plain_p", "after",
                                        "after_p")}
            for r in group:
                for k in sums:
                    _add(sums[k], r.get(k))
            lines.append(f"| {name} | {len(group)} | "
                         + " | ".join(_frac(tuple(sums[k])) for k in sums)
                         + " |")
        lines += ["", "| item | set | plain recall | plain precision | "
                      "pipeline recall | pipeline precision | plain verdicts "
                      "| pipeline verdicts | USD |",
                  "|---|---|---:|---:|---:|---:|---|---|---:|"]
        for r in rows:
            v = r.get("verdicts") or {}
            lines.append(
                f"| {r['item']} | {r['set']} | {_frac(r.get('plain'))} | "
                f"{_frac(r.get('plain_p'))} | {_frac(r.get('after'))} | "
                f"{_frac(r.get('after_p'))} | "
                f"{json.dumps(v.get('plain') or {})} | "
                f"{json.dumps(v.get('after') or {})} | {r['cost']:.4f} |")
    else:
        lines += ["| set | items | plain | pipeline model | pipeline after |",
                  "|---|---:|---:|---:|---:|"]
        for name in ("open", "blind", "all"):
            group = [r for r in rows if not r.get("skipped")
                     and (name == "all" or r["set"] == name)]
            if not group:
                continue
            sums = {k: [0, 0] for k in ("plain", "model", "after")}
            for r in group:
                for k in sums:
                    _add(sums[k], r.get(k))
            lines.append(f"| {name} | {len(group)} | "
                         + " | ".join(_frac(tuple(sums[k])) for k in sums)
                         + " |")
        lines += ["", "Per item (counts only; `differs` names the metrics "
                      "where plain and after found a different number):", "",
                  "| item | set | plain | model | after | plain - after | "
                  "differs | plain record | pipeline record | page text | "
                  "USD |",
                  "|---|---|---:|---:|---:|---:|---|---|---|---|---:|"]
        for r in rows:
            gap = ""
            if r.get("plain") is not None and r.get("after") is not None:
                gap = f"{r['plain'][0] - r['after'][0]:+d}"
            facts = r.get("facts_plain", "")
            if stage == "lab" and r.get("unlinked_plain"):
                facts += f"; unlinked {_frac(r['unlinked_plain'])}"
            theirs = r.get("facts_after", "")
            if stage == "lab" and r.get("unlinked_after"):
                theirs += f"; unlinked {_frac(r['unlinked_after'])}"
            sources = r.get("page_sources") or {}
            lines.append(
                f"| {r['item']} | {r['set']} | "
                f"{'ERROR' if r.get('error') else _frac(r.get('plain'))} | "
                f"{_frac(r.get('model'))} | {_frac(r.get('after'))} | {gap} | "
                f"{_diff_text(r)} | {facts} | {theirs} | "
                f"{', '.join(f'{k} {v}' for k, v in sources.items()) or '-'} | "
                f"{r['cost']:.4f} |")
    calls = rcl.spend_log(out)
    tot = rcl._sum_calls(calls)
    items = [r for r in rows if r["cost"]]
    lines += ["", "## Spend", "",
              f"Metered calls (spend_calls.jsonl): {tot['calls']} replies, "
              f"{tot['errors']} refused or failed, {tot['uncached']:,} input "
              f"and {tot['output']:,} output tokens, ${tot['usd']:.4f}. "
              f"The run files' cost blocks: {len(items)} item(s), "
              f"${sum(r['cost'] for r in items):.4f}"
              + (f", ${sum(r['cost'] for r in items) / len(items):.4f} an "
                 f"item" if items else "") + "."]
    errors = [r["item"] for r in rows if r.get("error")]
    if errors:
        lines.append(f"Items with no reading: {', '.join(errors)}.")
    skipped = [r["item"] for r in rows if r.get("skipped")]
    if skipped:
        lines.append(f"Items skipped (too large for one PDF block): "
                     f"{', '.join(skipped)}.")
    lines += ["", "## The instruction", "", "```",
              INSTRUCTIONS[stage], "```", ""]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def report(args: argparse.Namespace) -> int:
    paths = _paths(args)
    out = _out_dir(args, paths)
    if not (out / args.stage).is_dir():
        print(f"no {args.stage} run files in {out}")
        return rcl.REFUSED
    lines = write_results(out, args.stage, paths, args.pipeline_arm)
    print("\n".join(lines))
    return rcl.OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helps in (("run", "read items (paid; metered)"),
                        ("report", "write RESULTS.md again ($0)")):
        p = sub.add_parser(name, help=helps)
        p.add_argument("--stage", required=True, choices=STAGES)
        p.add_argument("--model", default="claude-opus-5-5")
        p.add_argument("--arm", default=None,
                       help="the arm's folder name (default "
                            "plain_<stage>_<model>)")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--raw-dir", default=None)
        p.add_argument("--truth-dir", default=None)
        p.add_argument("--reports-dir", default=None)
        p.add_argument("--pipeline-dir", default=None,
                       help="the folder holding the pipeline arms (default: "
                            "the 2026-10-24 claude_local raw folder)")
        p.add_argument("--pipeline-arm", default=None,
                       help="the pipeline arm to set beside it (default: "
                            "the Opus arm of the stage)")
        if name == "run":
            p.add_argument("--sets", default="all",
                           choices=("all", "open", "blind"))
            p.add_argument("--only", default="",
                           help="comma-separated items or report IDs")
            p.add_argument("--wave", required=True)
            p.add_argument("--cap", type=float, required=True, help="USD")
            p.add_argument("--ledger", default=None)
            p.add_argument("--redo", action="store_true")
            p.add_argument("--time-limit", type=float,
                           default=DEFAULT_TIME_LIMIT,
                           help="seconds after which no new item starts")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    sys.exit(main())
