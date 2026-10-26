"""BACKLOG RI-90: the owner's report question list, run on saved records.

Fills the rows answered by code ("Code" and "Both, compared") from each
saved full record, and with ``--engine fake`` runs the whole flow of
:func:`report_ingest.question_list.run_questions` (the screening call, the
gate, the main call, code's reading beside each "Both" answer) on an engine
that calls no model. Run it from the repo root (or a worktree) with the app
venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.run_question_list \\
        --list <the owner's question list> [--map <its code map>] \\
        [--only R13,R15] [--engine none|fake] [--no-text] \\
        [--out-dir <a gitignored folder>]

``--list`` defaults to the ``GSE_REPORT_QUESTIONS`` variable, ``--map`` to
the code map beside the list (``<list name>.code.json``, as private as the
list: it says which of code's readings answers each Code or Both row and
which screening answers close rows). ``--records``
defaults to the 2026-10-04 Foundry full records in the main checkout (the
stage-b ingest folder: one ``<ID>/report.record.json`` a report, a bound
report's own record under ``bound/``). ``--engine none`` (the default)
answers code's rows only; ``fake`` runs both calls on :class:`FakeEngine`,
which answers every question "not said" and each of the list's screening
gates the way that closes nothing. No model is called either way. Unless
``--no-text``, each report's PDF is opened from the private corpus for the
text of the pages the calls would read and the printed page numbers.

WHERE IT WRITES. One answer sheet a report (``<ID>.xlsx``) and
``coverage.json`` in ``--out-dir``, which must be a folder git ignores
(default: the main checkout's ``module_work/field_feedback/
2026-09-16_report-ingest/raw/question_list_runs/<date>/``): the sheets carry
the list's prompts and the reports' own words.

WHAT IT PRINTS: for each row answered by Code or Both, on how many reports
code gave an answer, and why it gave none on the others. A report in the
blind set (:data:`report_ingest.scoring.OOS_BLIND`) is counted in those
totals only: no line names it.

PRIVACY. Keys, report IDs and counts only; never a prompt, an answer or a
file name. The list is read in place and never copied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from module_work.report_ingest_harness.run_claude_local import (
    RAW_REL, REPO, _inside, git_ignored, main_checkout,
)

#: The saved Foundry full records, relative to the main checkout.
RECORDS_REL = Path("module_work/field_feedback/2026-10-04_foundry_evidence/"
                   "raw/b2all/stage_b_ingest/ingest")
#: Where the sheets go, relative to the main checkout (a gitignored raw/).
OUT_REL = RAW_REL / "question_list_runs"


class FakeEngine:
    """An engine that calls no model: every question "not said", and each
    of the list's screening gates answered the way that closes nothing, so
    both calls run."""

    name = "fake (no model)"
    accepts_pdf = False

    def __init__(self, gates: Sequence[Any] = ()) -> None:
        self.gates = list(gates)

    def complete(self, messages: Sequence[Dict[str, Any]], *,
                 output_format: Any = None, **_: Any) -> Any:
        from report_ingest.engine import Reply, Usage

        opens = {g.key: not g.closes_when for g in self.gates}
        answer = {}
        for key in output_format.model_fields:
            if key in opens:
                answer[key] = {"value": opens[key], "status": "answered",
                               "pdf_pages": [], "note": "fake engine"}
            else:
                answer[key] = {"value": None, "status": "not said",
                               "pdf_pages": [], "note": "fake engine"}
        chars = sum(len(b.get("text", "")) for m in messages
                    for b in m["content"] if isinstance(b, dict))
        return Reply(parsed=output_format.model_validate(answer),
                     stop_reason="end_turn", model=self.name,
                     usage=Usage(input_tokens=chars // 4))


def _load(path: Path) -> Any:
    from report_ingest.model import ReportRecord

    return ReportRecord.model_validate_json(path.read_bytes())


def _bound_records(folder: Path, record: Any) -> Dict[str, Any]:
    out = {}
    for doc in record.bound_documents:
        path = folder / doc.record_path if doc.record_path else None
        if path is not None and path.is_file():
            out[doc.bound_id] = _load(path)
    return out


def _corpus(main: Path) -> Any:
    from report_ingest.corpus import Corpus

    raw = main / RAW_REL
    return Corpus(raw / "corpus", di_dir=raw / "di")


def _text_of(corpus: Any, rid: str, record: Any
             ) -> Dict[str, Any]:
    """The pages the calls read and the printed page numbers, from the
    report's PDF; empty when the PDF is not on this machine."""
    from report_ingest.question_list import page_texts, pages_to_read

    try:
        doc = corpus.open_report(rid, warn=False)
    except Exception as exc:                            # noqa: BLE001
        return {"texts": {}, "printed": {},
                "why": f"the PDF did not open ({type(exc).__name__})"}
    with doc:
        printed = {s.page: s.printed_page for s in doc.page_map()
                   if s.printed_page is not None}
        texts = page_texts(doc, pages_to_read(record))
    return {"texts": texts, "printed": printed, "why": ""}


def _answers(args: argparse.Namespace, qlist: Any, record: Any,
             bound: Dict[str, Any], text: Dict[str, Any]) -> Any:
    from report_ingest.question_list import (Answer, QuestionRun,
                                             code_answers, run_questions)

    rid = record.document.report_id
    if args.engine == "fake":
        return run_questions(record, text["texts"], qlist,
                             FakeEngine(qlist.gates), bound=bound,
                             printed_pages=text["printed"])
    code = code_answers(record, qlist, bound=bound,
                        printed_pages=text["printed"])
    answers = {q.key: code.get(q.key) or Answer(
        status="", note="not asked: code's rows only") for q in qlist}
    return QuestionRun(report_id=rid, answers=answers)


def run(args: argparse.Namespace) -> int:
    from report_ingest.question_list import (QuestionListError, answer_sheet,
                                             code_answers, load_questions)
    from report_ingest.scoring import OOS_BLIND

    main = main_checkout()
    records = Path(args.records) if args.records else main / RECORDS_REL
    out = (Path(args.out_dir) if args.out_dir
           else main / OUT_REL / time.strftime("%Y-%m-%d"))
    for root in {main, REPO}:
        if _inside(out, root) and git_ignored(out, root) is False:
            print(f"refused: the output folder is inside the repo and git "
                  f"does not ignore it; the sheets quote the list and the "
                  f"reports")
            return 2
    try:
        qlist = load_questions(args.list, code_map=args.map)
    except QuestionListError as exc:
        print(exc)
        return 2
    wanted = [t.strip() for t in (args.only or "").split(",") if t.strip()]
    folders = sorted(p for p in records.iterdir()
                     if p.is_dir() and (not wanted or p.name in wanted))
    corpus = None if args.no_text else _corpus(main)
    rows = [q for q in qlist if q.answered_by in ("code", "both")]
    print(f"{len(qlist)} rows; the code map "
          f"{qlist.map_name or '(none: code reads no row)'} gives "
          f"{len(qlist.readings)} of the {len(rows)} Code or Both rows a "
          f"reading")
    given: Counter = Counter()
    why: Dict[str, Counter] = {q.key: Counter() for q in rows}
    reports: List[Dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        rid = folder.name
        blind = rid in OOS_BLIND
        path = folder / "report.record.json"
        if not path.is_file():
            reports.append({"id": "(blind)" if blind else rid,
                            "record": False})
            if not blind:
                print(f"{rid}: no saved record")
            continue
        record = _load(path)
        bound = _bound_records(folder, record)
        text = ({"texts": {}, "printed": {}, "why": "--no-text"}
                if corpus is None else _text_of(corpus, rid, record))
        result = _answers(args, qlist, record, bound, text)
        answer_sheet(qlist, result.answers, out / f"{rid}.xlsx",
                     report_id=rid, calls=result.calls)
        code = code_answers(record, qlist, bound=bound,
                            printed_pages=text["printed"])
        answered = 0
        for q in rows:
            if code[q.key].status == "answered":
                given[q.key] += 1
                answered += 1
            else:
                why[q.key][code[q.key].note.split(";")[0].strip()] += 1
        line = {"id": rid, "record": True, "code_rows": answered,
                "bound_records": len(bound), "pages_read": len(text["texts"]),
                "printed_pages": len(text["printed"]),
                "calls": [c.get("call") for c in result.calls],
                "chars_sent": [c.get("chars_sent") for c in result.calls],
                "closed": len(result.closed), "warnings": len(result.warnings)}
        if blind:
            reports.append({"id": "(blind)", "record": True})
            continue
        reports.append(line)
        print(f"{rid}: code answered {answered} of {len(rows)} rows; "
              f"{len(bound)} bound record(s); {len(text['texts'])} page(s) "
              f"of text{(' (' + text['why'] + ')') if text['why'] else ''}; "
              f"calls {line['calls']} chars {line['chars_sent']}")
    n = sum(1 for r in reports if r["record"])
    print(f"\ncode's rows on {n} saved record(s) "
          f"({len(reports) - n} folder(s) without one):")
    print(f"{'key':36} {'by':5} {'code gave':>9}  why not, on the others")
    table = []
    for q in rows:
        reasons = "; ".join(f"{k} ({v})" for k, v in why[q.key].most_common())
        print(f"{q.key:36} {q.answered_by:5} {given[q.key]:>4} of {n:<2}  "
              f"{reasons}")
        table.append({"key": q.key, "answered_by": q.answered_by,
                      "code_gave": given[q.key], "of": n,
                      "why_not": dict(why[q.key])})
    (out / "coverage.json").write_text(json.dumps(
        {"engine": args.engine, "list_rows": len(qlist), "rows": table,
         "reports": reports}, indent=1), encoding="utf-8")
    print(f"\nwrote {n} answer sheet(s) and coverage.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", default=None,
                    help="the question list (default: GSE_REPORT_QUESTIONS)")
    ap.add_argument("--map", default=None,
                    help="the list's code map (default: beside the list)")
    ap.add_argument("--records", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--engine", choices=("none", "fake"), default="none")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--out-dir", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
