"""The chart tasks: Chartography (Surge AI) as a local folder.

Chartography (huggingface.co/datasets/surgeai/chartography) is 100 expert
chart-reading questions, each a real chart image, a question and a golden
answer that carries an acceptable range where the chart's precision calls for
one. The dataset is gated and tagged NOT FOR TRAINING, and this repository is
public, so nothing from it is ever committed here: the questions, the golden
answers and the charts live only in the data folder the run is pointed at
(made on a machine with access by :mod:`funhouse_agent.chart_eval.prepare`),
and the results the suite writes name tasks by id alone.

A data folder holds, in order of preference:

* ``tasks.jsonl``: one task per line with ``task_id``, ``prompt``,
  ``golden_answer``, ``chart_path`` (relative to the folder) and optionally
  ``domain_combined``, ``source_type``, ``source_url`` (what
  :func:`funhouse_agent.chart_eval.prepare.prepare_chartography` writes);
* a Hugging Face snapshot: ``*.parquet`` files with the same columns (needs
  pandas with pyarrow) and the chart files beside them;
* a Surge task pack: ``manifest.yaml`` listing task YAML files with the same
  fields (the two dummy tasks in Surge's own repository are one).

Optional: ``judge_prompt.txt``, the judge prompt of Surge's own harness
(``src/chartography/scorer.py``), which ``prepare`` copies in so the run is
graded as Surge grades it; without it the suite's own wording of the same
rules is used (:mod:`funhouse_agent.chart_eval.judge`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

JUDGE_PROMPT_FILE = "judge_prompt.txt"
TASKS_FILE = "tasks.jsonl"


class ChartDataError(ValueError):
    """The data folder cannot be read as chart tasks."""


@dataclass
class ChartTask:
    """One question about one chart."""
    id: str
    prompt: str
    golden_answer: str
    image_path: str
    domain: str = ""
    source_type: str = ""
    source_url: str = ""

    def public_record(self) -> Dict[str, Any]:
        """What may go into a results file: the id and the labels, never
        the question, the answer or the chart."""
        return {"id": self.id, "domain": self.domain,
                "source_type": self.source_type,
                "image": os.path.basename(self.image_path)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rows_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ChartDataError(f"{path}:{n}: not JSON ({exc})")
    return rows


def _rows_from_parquet(files: List[Path]) -> List[Dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:                            # pragma: no cover
        raise ChartDataError(f"reading parquet needs pandas: {exc}")
    rows: List[Dict[str, Any]] = []
    for f in files:
        try:
            frame = pd.read_parquet(f)
        except Exception as exc:                          # noqa: BLE001
            raise ChartDataError(
                f"{f}: cannot read parquet ({type(exc).__name__}: {exc}); "
                "install pyarrow, or make tasks.jsonl with "
                "funhouse_agent.chart_eval.prepare")
        rows.extend(frame.to_dict(orient="records"))
    return rows


def _rows_from_task_pack(root: Path) -> List[Dict[str, Any]]:
    import yaml
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(
        encoding="utf-8")) or {}
    rows = []
    for rel in manifest.get("tasks") or []:
        rows.append(yaml.safe_load((root / rel).read_text(encoding="utf-8")))
    return rows


def _find_image(root: Path, rel: str) -> Optional[Path]:
    rel = str(rel or "").strip()
    if not rel:
        return None
    direct = root / rel
    if direct.is_file():
        return direct
    base = os.path.basename(rel)
    for cand in root.rglob(base):
        if cand.is_file():
            return cand
    return None


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:                                     # noqa: BLE001
        pass
    return str(value).strip()


def _image_from_row(root: Path, row: Dict[str, Any], task_id: str
                    ) -> Optional[Path]:
    """The chart's file: ``chart_path`` relative to the folder; or image
    bytes carried in the row (a Hugging Face image column), written once to
    ``charts/``."""
    path = _find_image(root, _clean(row.get("chart_path")))
    if path is not None:
        return path
    img = row.get("image") or row.get("chart")
    data = None
    if isinstance(img, dict):
        data = img.get("bytes")
    elif isinstance(img, (bytes, bytearray)):
        data = bytes(img)
    if data:
        out = root / "charts" / f"{task_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.is_file():
            out.write_bytes(data)
        return out
    return None


def load_chart_tasks(data_dir: Any, ids: Optional[Iterable[str]] = None,
                     limit: Optional[int] = None) -> List[ChartTask]:
    """The chart tasks in ``data_dir`` (see the module docstring), in file
    order; ``ids`` keeps only those, ``limit`` the first so many. A task
    whose chart cannot be found raises: a run must not quietly score 98 of
    100."""
    if data_dir is None:
        raise ChartDataError("pass data_dir: the folder holding tasks.jsonl "
                             "and the charts")
    root = Path(os.fspath(data_dir)).expanduser()
    if not root.is_dir():
        raise ChartDataError(f"{root} is not a folder")
    if (root / TASKS_FILE).is_file():
        rows = _rows_from_jsonl(root / TASKS_FILE)
    elif sorted(root.rglob("*.parquet")):
        rows = _rows_from_parquet(sorted(root.rglob("*.parquet")))
    elif (root / "manifest.yaml").is_file():
        rows = _rows_from_task_pack(root)
    else:
        raise ChartDataError(
            f"{root} holds no {TASKS_FILE}, *.parquet or manifest.yaml")
    wanted = {str(i) for i in ids} if ids is not None else None
    tasks: List[ChartTask] = []
    missing: List[str] = []
    seen = set()
    for n, row in enumerate(rows):
        task_id = _clean(row.get("task_id")) or str(n)
        if task_id in seen:
            raise ChartDataError(f"task id {task_id!r} appears twice")
        seen.add(task_id)
        if wanted is not None and task_id not in wanted:
            continue
        prompt = _clean(row.get("prompt"))
        golden = _clean(row.get("golden_answer"))
        if not prompt or not golden:
            raise ChartDataError(f"task {task_id}: no prompt or no "
                                 "golden_answer")
        image = _image_from_row(root, row, task_id)
        if image is None:
            missing.append(f"{task_id} ({_clean(row.get('chart_path'))})")
            continue
        tasks.append(ChartTask(
            id=task_id, prompt=prompt, golden_answer=golden,
            image_path=str(image), domain=_clean(row.get("domain_combined")),
            source_type=_clean(row.get("source_type")),
            source_url=_clean(row.get("source_url"))))
    if missing:
        raise ChartDataError(f"{len(missing)} chart(s) not found under "
                             f"{root}: " + ", ".join(missing[:10]))
    if wanted is not None:
        unknown = wanted - seen
        if unknown:
            raise ChartDataError("unknown task ids: "
                                 + ", ".join(sorted(unknown)[:10]))
    if limit:
        tasks = tasks[:int(limit)]
    return tasks


def judge_prompt_template(data_dir: Any) -> Optional[str]:
    """The judge prompt kept in the data folder (``judge_prompt.txt``), or
    ``None``."""
    if data_dir is None:
        return None
    path = Path(os.fspath(data_dir)).expanduser() / JUDGE_PROMPT_FILE
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    return None


__all__ = ["ChartTask", "ChartDataError", "load_chart_tasks",
           "judge_prompt_template", "JUDGE_PROMPT_FILE", "TASKS_FILE"]
