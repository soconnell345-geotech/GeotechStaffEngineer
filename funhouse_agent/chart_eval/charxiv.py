"""Make the chart suite's data folder from CharXiv (reasoning questions).

CharXiv (Wang et al., 2024; github.com/princeton-nlp/CharXiv,
huggingface.co/datasets/princeton-nlp/CharXiv) is 2,323 real charts taken
from arXiv papers. Each chart has one hand-written reasoning question whose
answer is a term, a short phrase or a number read off the chart. The 1,000
charts of the validation split carry their answers; the test split's are
withheld. It stands in for Surge AI's Chartography, which was not open to the
owner on 2026-11-06 (Hugging Face answered "Repository not found").

What this writes, into ``--out`` (the layout of
:mod:`funhouse_agent.chart_eval.data`):

* ``tasks.jsonl``: one task per validation reasoning question. ``prompt`` is
  the question as CharXiv's own generation script asks it, with the
  response instruction for its answer type (``REASONING_RESP_INST``);
  ``judge_question`` is the bare question and ``judge_key`` the answer type
  (1 text in chart, 2 text in general, 3 number in chart, 4 number in
  general), which pick the grading prompt;
* ``judge_prompts.json``: CharXiv's grading prompt for each answer type
  (``REASONING_GRADING_PREFIX`` + ``REASONING_GRADING_INST``), as their
  evaluation sends it to its judge;
* ``charts/``: the validation charts (``<figure_id>.jpg``);
* ``sample_100.txt``: 100 task ids drawn at random (seed 0) in proportion
  to the four answer types, for a run smaller than all 1,000;
* ``PROVENANCE.json`` and ``ATTRIBUTION.md``.

The prompts and grading instructions are read out of CharXiv's
``src/constants.py`` with :mod:`ast` (nothing in it is run), at a pinned
commit. The number instruction for answer type 4 follows CharXiv's
``get_number_instruction`` (``src/reasoning_utils.py``). CharXiv's code is
licensed under the Apache License 2.0; its questions and answers under
CC BY-SA 4.0; the charts belong to the papers' authors. So the folder stays
private: it is never committed (``.gitignore``) and the suite's results name
tasks by id only.

Run, anywhere with Python and the network (nothing beyond the standard
library is needed):

    python -m funhouse_agent.chart_eval.charxiv --out charxiv_data --zip
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import random
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: CharXiv's GitHub repository, pinned (data files and the prompt constants).
GITHUB_COMMIT = "7ebe88f78dee387691551f071abcb2b9e1a8025b"
GITHUB_RAW = ("https://raw.githubusercontent.com/princeton-nlp/CharXiv/"
              f"{GITHUB_COMMIT}/")
#: The Hugging Face dataset revision holding ``images.zip``.
HF_REVISION = "f441eb632fc62f6f777830a0f47619e6e86459b0"
IMAGES_URL = ("https://huggingface.co/datasets/princeton-nlp/CharXiv/"
              f"resolve/{HF_REVISION}/images.zip")

ANSWER_TYPES = {1: "text in chart", 2: "text in general",
                3: "number in chart", 4: "number in general"}


def fetch(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:   # noqa: S310
        return resp.read()


def constants_from_source(source: str) -> Dict[str, Any]:
    """``REASONING_RESP_INST``, ``REASONING_GRADING_PREFIX`` and
    ``REASONING_GRADING_INST`` out of CharXiv's ``constants.py`` text,
    read as literals (nothing is run)."""
    want = {"REASONING_RESP_INST", "REASONING_GRADING_PREFIX",
            "REASONING_GRADING_INST"}
    got: Dict[str, Any] = {}
    import warnings
    with warnings.catch_warnings():
        # CharXiv's strings hold LaTeX ("\\lambda") in plain literals; read
        # as Python reads them, without the escape warnings
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    got[t.id] = ast.literal_eval(node.value)
    missing = want - set(got)
    if missing:
        raise ValueError(f"CharXiv's constants.py has no {sorted(missing)}")
    return got


def number_instruction(answer: str) -> str:
    """The instruction CharXiv adds to a number-in-general question: an
    exact integer, or a number with as many decimal places as the answer
    (CharXiv's ``get_number_instruction``)."""
    parts = str(answer).split(".")
    if len(parts) == 1:
        return "* Your final answer must be an exact integer."
    return (f"* Your final answer must be a number with {len(parts[1])} "
            "decimal places.")


def build_question(query: str, answer: str, inst_category: int,
                   resp_inst: Dict[int, str]) -> str:
    """The question as CharXiv's ``build_reasoning_queries`` asks it."""
    if inst_category in (1, 2, 3):
        return resp_inst[inst_category].format(query)
    if inst_category == 4:
        return resp_inst[4].format(query, number_instruction(answer))
    raise ValueError(f"unknown answer type {inst_category!r}")


def stratified_sample(rows: List[Dict[str, Any]], n: int,
                      seed: int = 0) -> List[str]:
    """``n`` task ids drawn at random, in proportion to the answer types
    (largest remainders), in task order."""
    by_type: Dict[int, List[str]] = {}
    for r in rows:
        by_type.setdefault(int(r["judge_key"]), []).append(r["task_id"])
    total = sum(len(v) for v in by_type.values())
    n = min(int(n), total)
    quotas = {k: n * len(v) / total for k, v in by_type.items()}
    take = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - take[k],
                    reverse=True)[:n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    chosen = set()
    for k in sorted(by_type):
        chosen.update(rng.sample(by_type[k], take[k]))
    return [r["task_id"] for r in rows if r["task_id"] in chosen]


def prepare_charxiv(out_dir: Any, sample: int = 100, seed: int = 0,
                    fetcher: Callable[[str], bytes] = fetch,
                    images_zip: Optional[str] = None) -> Dict[str, Any]:
    """Write the data folder (module docstring). ``images_zip``: a local
    copy of CharXiv's ``images.zip`` instead of downloading 141 MB."""
    out = Path(os.fspath(out_dir)).expanduser()
    (out / "charts").mkdir(parents=True, exist_ok=True)
    consts = constants_from_source(
        fetcher(GITHUB_RAW + "src/constants.py").decode("utf-8"))
    questions = json.loads(fetcher(GITHUB_RAW + "data/reasoning_val.json"))
    meta = json.loads(fetcher(GITHUB_RAW + "data/image_metadata_val.json"))
    resp_inst = {int(k): v for k, v in consts["REASONING_RESP_INST"].items()}
    prefix = consts["REASONING_GRADING_PREFIX"]
    grading = {str(int(k)): prefix + v
               for k, v in consts["REASONING_GRADING_INST"].items()}

    if images_zip:
        zdata = Path(images_zip).read_bytes()
    else:
        zdata = fetcher(IMAGES_URL)
    archive = zipfile.ZipFile(io.BytesIO(zdata))
    names = set(archive.namelist())

    rows: List[Dict[str, Any]] = []
    for key in sorted(questions, key=lambda k: int(k)):
        q = questions[key]
        fid = str(q["figure_id"])
        member = f"{fid}.jpg"
        if member not in names:
            raise ValueError(f"figure {fid} is not in images.zip")
        (out / "charts" / member).write_bytes(archive.read(member))
        m = meta.get(fid) or meta.get(key) or {}
        cat = int(q["inst_category"])
        rows.append({
            "task_id": fid,
            "prompt": build_question(q["query"], str(q["answer"]), cat,
                                     resp_inst),
            "golden_answer": str(q["answer"]),
            "chart_path": f"charts/{member}",
            "domain_combined": str(m.get("category") or ""),
            "source_type": f"charxiv reasoning: {ANSWER_TYPES.get(cat)}",
            "source_url": (f"https://arxiv.org/abs/{m['paper_id']}"
                           if m.get("paper_id") else ""),
            "judge_key": str(cat),
            "judge_question": q["query"],
        })
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "judge_prompts.json").write_text(
        json.dumps(grading, indent=1, ensure_ascii=False), encoding="utf-8")
    ids = stratified_sample(rows, sample, seed) if sample else []
    if ids:
        (out / f"sample_{len(ids)}.txt").write_text("\n".join(ids) + "\n",
                                                    encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["judge_key"]] = counts.get(r["judge_key"], 0) + 1
    prov = {"dataset": "CharXiv reasoning, validation split",
            "github_commit": GITHUB_COMMIT, "hf_revision": HF_REVISION,
            "tasks": len(rows), "by_answer_type": counts,
            "sample": {"file": f"sample_{len(ids)}.txt", "seed": seed,
                       "n": len(ids)} if ids else None,
            "judge_prompt": "per task (judge_prompts.json, CharXiv's own)",
            "made": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1),
                                         encoding="utf-8")
    return prov


ATTRIBUTION = """# Attribution

This folder is made from CharXiv: Zirui Wang et al., "CharXiv: Charting Gaps
in Realistic Chart Understanding in Multimodal LLMs" (2024),
https://github.com/princeton-nlp/CharXiv and
https://huggingface.co/datasets/princeton-nlp/CharXiv.

- The questions and answers (the reasoning questions of the validation split)
  are CharXiv's, licensed under CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). They are unchanged; each
  prompt adds CharXiv's own response instruction for its answer type.
- The response instructions and grading prompts come from CharXiv's code,
  licensed under the Apache License 2.0
  (https://www.apache.org/licenses/LICENSE-2.0).
- The chart images belong to the authors of the arXiv papers they come from
  (each task's source_url). They are used here for evaluation only.

Keep this folder private: do not commit it or publish it.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the data folder to write")
    ap.add_argument("--sample", type=int, default=100,
                    help="also write sample_<n>.txt (0 for none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images-zip", default=None,
                    help="a local copy of CharXiv's images.zip")
    ap.add_argument("--zip", action="store_true",
                    help="also write <out>.zip for upload")
    args = ap.parse_args(argv)
    prov = prepare_charxiv(args.out, sample=args.sample, seed=args.seed,
                           images_zip=args.images_zip)
    print(json.dumps(prov, indent=1))
    if args.zip:
        path = shutil.make_archive(str(Path(args.out)), "zip",
                                   root_dir=str(Path(args.out)))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
