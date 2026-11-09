"""Make the chart suite's data folder from the gated Hugging Face dataset.

Run it on a machine that can reach Hugging Face, with a token whose account
has accepted the dataset's terms (huggingface.co/datasets/surgeai/
chartography, "Agree and access"):

    pip install huggingface_hub pandas pyarrow
    export HF_TOKEN=...            # a read token; never typed into a chat
    python -m funhouse_agent.chart_eval.prepare --out chartography_data --zip

It writes, into ``--out``:

* ``tasks.jsonl``: one task per line (task_id, prompt, golden_answer,
  chart_path, domain_combined, source_type, source_url);
* ``charts/``: the chart images ``chart_path`` names;
* ``judge_prompt.txt``: the judge prompt of Surge's own harness, read out of
  ``src/chartography/scorer.py`` on GitHub without importing it (its
  repository carries no licence, so it is copied here, into the private
  folder, and never into this package);
* ``DATASET_CARD.md`` and ``PROVENANCE.json`` (the dataset revision, the
  date, the counts).

The folder is the dataset: gated and NOT FOR TRAINING. Keep it out of every
repository (``.gitignore`` holds ``chartography_data*``) and upload it only
where the run happens.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO = "surgeai/chartography"
SCORER_URL = ("https://raw.githubusercontent.com/surge-ai/chartography/main/"
              "src/chartography/scorer.py")
FIELDS = ("task_id", "prompt", "golden_answer", "chart_path",
          "domain_combined", "source_type", "source_url")


def judge_template_from_source(source: str) -> Optional[str]:
    """``JUDGE_PROMPT_TEMPLATE`` out of Surge's ``scorer.py`` text, read
    with :mod:`ast` (nothing in it is run)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "JUDGE_PROMPT_TEMPLATE" in names and isinstance(
                    node.value, ast.Constant) and isinstance(node.value.value,
                                                             str):
                return node.value.value
    return None


def fetch_judge_template(url: str = SCORER_URL,
                         timeout: float = 30.0) -> Optional[str]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:   # noqa: S310
        return judge_template_from_source(resp.read().decode("utf-8"))


def _rows(snapshot: Path, split: str):
    import pandas as pd
    files = sorted(snapshot.rglob("*.parquet"))
    pick = [f for f in files if split in f.name] or files
    if not pick:
        raise SystemExit(f"no parquet files in the snapshot {snapshot}")
    frames = [pd.read_parquet(f) for f in pick]
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def prepare_chartography(out_dir: Any, repo: str = REPO, split: str = "test",
                         token: Optional[str] = None, revision: Optional[str]
                         = None, judge_url: Optional[str] = SCORER_URL
                         ) -> Dict[str, Any]:
    """Download ``repo`` and write the data folder (module docstring)."""
    from huggingface_hub import HfApi, snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    out = Path(os.fspath(out_dir)).expanduser()
    (out / "charts").mkdir(parents=True, exist_ok=True)
    snap = Path(snapshot_download(repo_id=repo, repo_type="dataset",
                                  token=token, revision=revision))
    try:
        sha = HfApi().dataset_info(repo, token=token,
                                   revision=revision).sha
    except Exception:                                     # noqa: BLE001
        sha = None
    rows = _rows(snap, split)
    written = 0
    with open(out / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for n, row in enumerate(rows):
            rec = {k: (None if row.get(k) is None else str(row.get(k)))
                   for k in FIELDS}
            rec["task_id"] = rec["task_id"] or str(n)
            rel = rec.get("chart_path") or ""
            src = snap / rel if rel else None
            if src is None or not src.is_file():
                img = row.get("image") or row.get("chart")
                data = img.get("bytes") if isinstance(img, dict) else None
                if not data:
                    raise SystemExit(f"task {rec['task_id']}: chart "
                                     f"{rel!r} not in the snapshot")
                rel = f"charts/{rec['task_id']}.png"
                (out / rel).write_bytes(data)
            else:
                dest = out / "charts" / Path(rel).name
                shutil.copyfile(src, dest)
                rel = f"charts/{dest.name}"
            rec["chart_path"] = rel
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    card = snap / "README.md"
    if card.is_file():
        shutil.copyfile(card, out / "DATASET_CARD.md")
    judge = None
    if judge_url:
        try:
            judge = fetch_judge_template(judge_url)
        except Exception as exc:                          # noqa: BLE001
            print(f"the judge prompt was not fetched ({exc}); the suite's "
                  "own prompt will grade", file=sys.stderr)
    if judge:
        (out / "judge_prompt.txt").write_text(judge, encoding="utf-8")
    prov = {"repo": repo, "split": split, "revision": sha or revision,
            "tasks": written, "judge_prompt": bool(judge),
            "judge_source": judge_url if judge else None,
            "made": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1),
                                         encoding="utf-8")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the data folder to write")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--split", default="test")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--no-judge-prompt", action="store_true",
                    help="do not fetch Surge's judge prompt")
    ap.add_argument("--zip", action="store_true",
                    help="also write <out>.zip for upload")
    args = ap.parse_args(argv)
    prov = prepare_chartography(args.out, repo=args.repo, split=args.split,
                                revision=args.revision,
                                judge_url=None if args.no_judge_prompt
                                else SCORER_URL)
    print(json.dumps(prov, indent=1))
    if args.zip:
        path = shutil.make_archive(str(Path(args.out)), "zip",
                                   root_dir=str(Path(args.out)))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
