"""Probe of the Document Review suite's scorer with made-up answers.

The audit of 2026-11-18 (SCORER.md beside this file). Each case in
``scorer_probe_cases/*.json`` is a made-up answer to one suite task:

    {"task": "<task id>", "kind": "right" | "wrong",
     "say": "<what the phrasing is, a few words>",
     "answer": "<the chat answer>",
     "files": {"log.xlsx": [["Sheet", "Comment"], ...],   # rows -> a workbook
               "memo.docx": ["paragraph", ...],           # strings -> Word
               "notes.csv": "text"},                      # text -> as is
     "checks": ["<check key>", ...]}

A check's key is its label, or its type where it has none (``#2`` added to
a repeat inside one task). ``checks`` names the checks the answer speaks
to; left out, a "right" case is held to every scored check that reads the
answer or the delivered text. A right answer that fails is a FALSE FAIL, a
wrong answer that passes a FALSE PASS. Every value here is made up or is
the suite's own built fixture truth.

Run with one tree's packages first on the path, once per tree:

    PYTHONPATH=<tree>/packages/aec-agent-core:<tree>/packages/aec-document-review \
        python scorer_probe.py --tree master --out master.json
    python scorer_probe.py --table t120.json t130.json master.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
#: Checks a made-up answer cannot speak to without a real run's files or
#: activity; a case reaches one only by naming it in ``checks``.
NOT_BY_DEFAULT = ("tool_used", "pages_covered", "tools_called",
                  "file_produced", "pdf_markups", "markups_on_targets",
                  "markups_point_at", "docx_contains")


def keyed(checks):
    """``[(key, check)]``: label, or type, numbered where repeated."""
    seen, out = {}, []
    for c in checks:
        base = c.get("label") or c.get("type")
        seen[base] = seen.get(base, 0) + 1
        out.append((base if seen[base] == 1 else f"{base}#{seen[base]}", c))
    return out


def write_files(spec, folder):
    paths = []
    for name, content in (spec or {}).items():
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if name.lower().endswith(".xlsx") and isinstance(content, list):
            import openpyxl
            book = openpyxl.Workbook()
            for row in content:
                book.active.append(row)
            book.save(path)
        elif name.lower().endswith(".docx") and isinstance(content, list):
            import docx
            d = docx.Document()
            for p in content:
                d.add_paragraph(p)
            d.save(path)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content if isinstance(content, str)
                         else json.dumps(content))
        paths.append(path)
    return paths


def load_cases(only=None):
    cases = []
    for path in sorted(glob.glob(os.path.join(HERE, "scorer_probe_cases",
                                              "*.json"))):
        if only and os.path.basename(path) not in only:
            continue
        with open(path, encoding="utf-8") as fh:
            for n, case in enumerate(json.load(fh)):
                case["_from"] = f"{os.path.basename(path)}#{n}"
                cases.append(case)
    return cases


def probe(tree, only=None):
    import aec_document_review.review_eval  # noqa: F401  (fills the hooks)
    from aec_agent_core.review_eval.checks import run_check
    from aec_document_review.review_eval.tasks import OPEN_TASKS
    tasks = {t.id: t for t in OPEN_TASKS}
    rows = []
    for case in load_cases(only):
        task = tasks.get(case["task"])
        if task is None:
            rows.append({"tree": tree, "task": case["task"], "key": "",
                         "kind": case["kind"], "say": case.get("say", ""),
                         "answer": case.get("answer", ""), "from": case["_from"],
                         "verdict": "no such task"})
            continue
        checks = dict(keyed(task.all_checks()))
        wanted = case.get("checks")
        if not wanted:
            wanted = [k for k, c in checks.items()
                      if not c.get("info") and c["type"] not in NOT_BY_DEFAULT]
        with tempfile.TemporaryDirectory() as folder:
            files = write_files(case.get("files"), folder)
            for key in wanted:
                row = {"tree": tree, "task": task.id, "key": key,
                       "kind": case["kind"], "say": case.get("say", ""),
                       "answer": case.get("answer", ""),
                       "files": sorted(case.get("files") or {}),
                       "from": case["_from"]}
                if key not in checks:
                    row["verdict"] = "no such check"
                    rows.append(row)
                    continue
                got = run_check(checks[key], case.get("answer", ""),
                                files=files,
                                activity=case.get("activity") or [])
                row.update(passed=got["passed"], detail=got["detail"],
                           info=bool(checks[key].get("info")),
                           check=json.dumps(checks[key], sort_keys=True))
                if case["kind"] == "right":
                    row["verdict"] = "ok" if got["passed"] else "FALSE FAIL"
                else:
                    row["verdict"] = "FALSE PASS" if got["passed"] else "ok"
                rows.append(row)
    return rows


def table(paths):
    by = {}
    trees = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for r in json.load(fh):
                if r["tree"] not in trees:
                    trees.append(r["tree"])
                by.setdefault((r["from"], r["key"]), {})[r["tree"]] = r
    carried = ("does not give up on legibility", "no 0-based page citation")
    tasks, checks, ff, fp = set(), set(), set(), set()
    lines = []
    for (_frm, key), per in sorted(by.items(),
                                   key=lambda kv: (kv[1][next(iter(kv[1]))]["task"], kv[0])):
        any_row = next(iter(per.values()))
        tasks.add(any_row["task"])
        if any_row.get("verdict") in ("no such task", "no such check"):
            continue
        # a check every task carries is one check, whichever task tried it
        check = (("" if key in carried else any_row["task"]), key)
        checks.add(check)
        bad = [t for t in trees if per.get(t, {}).get("verdict", "").startswith("FALSE")]
        if not bad:
            continue
        kind = per[bad[0]]["verdict"]
        (ff if kind == "FALSE FAIL" else fp).add(check)
        absent = [t for t in trees if per.get(t, {}).get("verdict") in
                  (None, "no such task", "no such check")]
        where = ", ".join(bad) + (f" (not at {', '.join(absent)})" if absent else "")
        lines.append(f"| {any_row['task']} | {key} | {kind} | {any_row['say']}: "
                     f"{' '.join(any_row['answer'].split())[:110]} | {where} |")
    print(f"{len(tasks)} tasks and {len(checks)} checks tried; "
          f"{len(ff)} false fails, {len(fp)} false passes "
          f"(checks with at least one; {len(lines)} results listed)")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="tree")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--table", nargs="*")
    a = ap.parse_args(argv)
    if a.table:
        return table(a.table)
    rows = probe(a.tree, a.only)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    for r in rows:
        if r["verdict"] != "ok":
            print(f"{r['verdict']:<11} {r['task']} [{r['key']}] {r['kind']} "
                  f"({r['say']}) {r['from']}\n    {r.get('detail', '')[:300]}")
    n = sum(1 for r in rows if r["verdict"] == "ok")
    print(f"{len(rows)} check results, {n} as expected")


if __name__ == "__main__":
    sys.exit(main())
