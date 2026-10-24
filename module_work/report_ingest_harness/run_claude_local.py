"""The report-ingest readers on Claude, run here through the Anthropic API.

The owner decided on 2026-10-24 that report ingest runs in Foundry on Claude
Opus 5.5 and Sonnet 5.5. Every live reader measurement before that ran on
GPT-5.4. This script runs the same measurement on the same hand-truthed
items, locally, on Claude: it calls
:func:`report_ingest.cluster_scoring.score_on_cluster` with a
``ChatAnthropic`` as its ``chat_model`` (BACKLOG RI-67), so every call goes
through :class:`~report_ingest.engine.ChatModelEngine`, the engine the
Foundry run uses over Foundry's own Claude classes. Run it from the repo
root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.run_claude_local \\
        run --stage lab --model claude-sonnet-5-5 \\
        --env GEOTECH_INGEST_LAB_LINK_FROM_LABEL=1 \\
        --wave cl-lab-sonnet --cap 4 --arm lab_sonnet_linkon

    ... run_claude_local run --stage narrative --model claude-sonnet-5-5 \\
        --only R05 --wave cl-smoke --cap 3 --arm smoke_narrative_sonnet

    ... run_claude_local check <an out_dir>     # $0: no model call

THE STAGES are the four readers -- ``logs``, ``lab``, ``calc``,
``narrative`` -- and ``soundings``. The items are the hand-truthed ones in
the private truth folder, the same files the Foundry runs read. ``--sets``
picks the open set, the blind set or both (the split is the truth folder's
own ``OPEN.txt``, or the package's default for the stage, exactly as the
stage itself decides it); ``--only`` picks items by their file name
(``gradation__R06_p67``) or by report (``R05``). A pick is run off a copy of
the chosen truth files under ``<out_dir>/_truth/``, so the stage scores only
those; a later run of the whole stage into the same ``out_dir`` reuses them.

THE READER MODEL AND THE VISION MODEL. ``--model`` reads; every picture a
reader sends goes with its own calls, so for these five stages that is the
whole run. ``--vision-model`` is passed through as ``vision_chat_model`` and
matters only to the stages that make a separate vision pass (none of these).

EVERY CALL IS METERED by the live-smoke spend control
(``module_work/live_smoke/spend.py``): the wave is opened on the ONE ledger
(the main checkout's ``spend_ledger.json``, found from a worktree too) with
its cap, a :class:`SpendMeter` is attached to the ``ChatAnthropic`` object
itself (a second model gets a :class:`ModelMeter`), each call is priced at
its model's rates, and the first call after the wave's cap or the hard total
is spent is refused before it goes out. The reader then fails that item and
the next; nothing more is paid for. The ledger is re-read and written under a
lock file for every call (:class:`SharedLedger`), so arms run side by side
keep one total.

The run's own scorecard prices its calls too: the two models' rates from
``spend.PRICES`` are put into :data:`report_ingest.engine.MODEL_PRICES` in
this process (never in the package's file), so RESULTS.md's cost block and
the ledger are priced alike and can be set side by side -- the "Claude,
locally" block this script appends to RESULTS.md does exactly that.

WHERE IT WRITES. ``--out-dir`` defaults to the main checkout's
``module_work/field_feedback/2026-10-24_claude_local/raw/<arm>/``, which is
gitignored (``module_work/field_feedback/**/raw/``); an ``out_dir`` inside
the repo that git does NOT ignore is refused. Beside ``score_on_cluster``'s
own files it keeps ``arm.json`` (the arm's settings: a later run with other
settings into the same folder is refused), ``spend_calls.jsonl`` (every
metered call of every run of the arm, with the pictures it carried) and
``sdk_retries.jsonl`` (each retry the Anthropic SDK made inside a call,
which no meter sees otherwise).

THE PUBLIC BLIND SET (BACKLOG RI-79) is run the same way, pointed at its
own folders: ``--raw-dir`` at the set's ``raw/`` (its truth is
``raw/truth/``, written from AGS4 data by :mod:`.ags_truth`) and
``--reports-dir`` at ``raw/assembled/``, where its PDFs are ``P01.pdf``
... Its ``OPEN.txt`` names no report, so every item is scored as blind::

    ... run_claude_local run --stage logs --model claude-sonnet-5-5 \\
        --raw-dir <set>/raw --reports-dir <set>/raw/assembled \\
        --wave pb-logs-sonnet --cap 1 --arm public_logs_sonnet

A LOG THAT CAN KEEP NO DEPTH IS REFUSED BEFORE ANY CALL. The logs reader
keeps a depth only on a scale the grid fitted or where the page prints it as
text; a scan with no text layer and no DI text gives it neither, so on a
boring every depth is refused and only header values can score (the public
set's first smoke: 3/76 and 5/69). Before a logs run opens its wave,
:func:`logs_with_no_depths` lists such items at $0 and the run stops, unless
the visual-scales switch is on or ``--allow-no-depths`` is passed.

IT RESUMES. The same command again skips every item that already has a run
file, as ``score_on_cluster`` always has; an item whose reading was cut by
the cap is set aside under ``<stage>/_capped/`` so it is read again whole.
To go on past a cap, raise ``--cap`` on the same wave.

PRIVACY. The corpus, the DI results and the truth are read in place from the
private ``raw/`` folder (``--raw-dir``); nothing is copied out of a gitignored
folder. What this script prints and appends carries item IDs, counts, tokens
and dollars only. The API key is read by ``live_smoke.models.api_key`` and is
never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

#: This checkout (a worktree's own root when run from one).
REPO = Path(__file__).resolve().parents[2]

#: The private corpus folder, relative to the MAIN checkout.
RAW_REL = Path("module_work/field_feedback/2026-09-16_report-ingest/raw")
#: Where the arms write, relative to the main checkout (gitignored raw/).
OUT_REL = Path("module_work/field_feedback/2026-10-24_claude_local/raw")
#: The one spend ledger, relative to the main checkout.
LEDGER_REL = Path("module_work/live_smoke/spend_ledger.json")

#: The reader stages this script runs, each with the budget argument
#: ``score_on_cluster`` takes for it and that argument's default.
STAGES: Dict[str, tuple] = {
    "logs": ("log_budget", 6),
    "lab": ("lab_budget", 4),
    "calc": ("calc_budget", 2),
    "narrative": ("narrative_budget", 8),
    "soundings": ("sounding_budget", 2),
}
#: Every switch of this kind is cleared before an arm sets its own, so a
#: switch left in the shell never leaks into an arm (as the Foundry brief's
#: cells clear them).
SWITCH_PREFIX = "GEOTECH_INGEST_"
#: What a refused call's error says (``live_smoke.spend.SpendMeter.check``).
CAP_TEXT = re.compile(r"cap reached|hard total reached|WaveCapReached", re.I)
#: The report an item belongs to, out of its file name: ``R06`` in the
#: corpus, ``P01`` in the public blind set.
REPORT_ID = re.compile(r"[RP]\d{2}")

#: Exit codes.
OK, REFUSED, STOPPED_BY_CAP = 0, 2, 3


# ---------------------------------------------------------------------------
# where things are
# ---------------------------------------------------------------------------

def main_checkout(start: Path = REPO) -> Path:
    """The main checkout's root. A worktree shares the main checkout's
    ledger and private corpus, so both are looked up there."""
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, timeout=30)
        common = Path(out.stdout.strip())
        if out.returncode == 0 and common.name == ".git":
            if not common.is_absolute():
                common = (start / common).resolve()
            return common.parent
    except Exception:                                   # noqa: BLE001
        pass
    return start


def git_ignored(path: Path, root: Path) -> Optional[bool]:
    """Whether the checkout at ``root`` ignores ``path`` (which need not
    exist yet); None without git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q",
             str(path.resolve())],
            capture_output=True, text=True, timeout=30)
    except Exception:                                   # noqa: BLE001
        return None
    if out.returncode == 0:
        return True
    if out.returncode == 1:
        return False
    return None                                          # not a checkout


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# the items
# ---------------------------------------------------------------------------

def report_of(stem: str) -> str:
    """``R06`` out of ``gradation__R06_p67``, ``R06_p51`` or ``R06``."""
    match = REPORT_ID.search(stem)
    return match.group(0) if match else ""


def open_reports(stage: str, truth_dir: Path) -> tuple:
    """The stage's open set as the stage itself decides it: the names in
    ``OPEN.txt``, else the package's default for the stage."""
    from report_ingest import cluster_scoring as cs

    default = {"logs": cs.DEFAULT_OPEN_LOGS, "lab": cs.DEFAULT_OPEN_LAB,
               "calc": cs.DEFAULT_OPEN_CALC,
               "soundings": cs.DEFAULT_OPEN_SOUNDINGS,
               "narrative": cs.DEFAULT_OPEN_NARRATIVE}[stage]
    open_file = truth_dir / "OPEN.txt"
    if open_file.is_file():
        names = tuple(line.strip() for line in open_file.read_text(
            encoding="utf-8").splitlines() if line.strip())
        if names:
            return names
    return tuple(default)


def pick_items(stage: str, truth_root: Path, sets: str,
               only: Sequence[str]) -> List[Path]:
    """The truth files this run reads, in the stage's own order."""
    truth_dir = truth_root / stage
    if not truth_dir.is_dir():
        raise FileNotFoundError(f"no {stage!r} truth folder at {truth_dir}")
    openset = set(open_reports(stage, truth_dir))
    chosen = []
    for path in sorted(truth_dir.glob("*.json")):
        in_open = report_of(path.stem) in openset
        if sets != "all" and (sets == "open") != in_open:
            continue
        if only and not any(tok in (path.stem, report_of(path.stem))
                            for tok in only):
            continue
        chosen.append(path)
    unmatched = [tok for tok in only if not any(
        tok in (p.stem, report_of(p.stem))
        for p in truth_dir.glob("*.json"))]
    if unmatched:
        raise ValueError(f"--only names no {stage} item: "
                         f"{', '.join(unmatched)}")
    return chosen


def truth_for_run(stage: str, truth_root: Path, chosen: Sequence[Path],
                  out: Path, picked: bool) -> Path:
    """The truth root to hand ``score_on_cluster``: the real one for a whole
    stage, else a fresh copy holding the chosen items, the stage's
    ``OPEN.txt`` (so the split is unchanged) and the log templates."""
    if not picked:
        return truth_root
    root = out / "_truth"
    if root.exists():
        shutil.rmtree(root)
    (root / stage).mkdir(parents=True)
    for path in chosen:
        shutil.copy2(path, root / stage / path.name)
    if (truth_root / stage / "OPEN.txt").is_file():
        shutil.copy2(truth_root / stage / "OPEN.txt",
                     root / stage / "OPEN.txt")
    if (truth_root / "templates.json").is_file():
        shutil.copy2(truth_root / "templates.json", root / "templates.json")
    return root


def set_switches(pairs: Sequence[str]) -> Dict[str, str]:
    """Clear every ``GEOTECH_INGEST_*`` switch, then set the arm's own."""
    for key in [k for k in os.environ if k.startswith(SWITCH_PREFIX)]:
        os.environ.pop(key)
    switches: Dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"--env takes NAME=VALUE, not {pair!r}")
        switches[key] = value.strip()
        os.environ[key] = value.strip()
    return switches


def logs_with_no_depths(reports_dir: Path, di_dir: Path,
                        chosen: Sequence[Path]) -> List[Dict[str, Any]]:
    """$0, before any call: the logs items on which the reader can keep no
    depth at all.

    The reader keeps a depth only inside a depth scale the grid fitted, or
    where the page prints that depth as text the grid can read
    (:mod:`report_ingest.log_reader`'s gate; RI-18); a test pit is the one
    exception, its printed numbers read off the picture. A scan with no
    text layer and no Document Intelligence text gives the grid neither, so
    on a boring every depth the model reads off the picture is refused and
    only header values can score. The public blind set's first smoke paid
    for exactly that. One row per such item, with the counts that show it.
    """
    from planlens.document.loggrid import log_grid
    from planlens.document.model import SOURCE_AZURE_DI
    from report_ingest.corpus import Corpus
    from report_ingest.printed_depths import printed_depths

    corpus = Corpus(reports_dir, di_dir=di_dir)
    present = set(corpus.present_ids())
    docs: Dict[str, Any] = {}
    out: List[Dict[str, Any]] = []
    try:
        for path in chosen:
            truth = json.loads(path.read_text(encoding="utf-8"))
            blocks = list(truth.get("investigations") or []) or [truth]
            if any(b.get("kind") == "test_pit" for b in blocks):
                continue
            report = report_of(path.stem)
            pages = [int(p) for p in truth.get("pages") or []]
            if not pages or report not in present:
                continue                  # the stage itself reports these
            if report not in docs:
                docs[report] = corpus.open_report(report, di="auto",
                                                  warn=False)
            doc = docs[report]
            grid = log_grid(doc, pages)
            lines = {p: list(doc.page(p).lines) for p in pages}
            printed = printed_depths(grid, pages, lines=lines)
            if grid.rulers or printed.readings:
                continue
            di = sum(1 for p in pages if doc.summary(p).evidence.get(
                "text_source") == SOURCE_AZURE_DI)
            nv = getattr(grid, "needs_values", None) or {}
            out.append({"item": path.stem, "pages": len(pages),
                        "di_pages": di, "cells": len(grid.rows),
                        "text_lines": sum(len(v) for v in lines.values()),
                        "scale_labels": sum(len(v.get("labels") or [])
                                            for v in nv.values())})
    finally:
        for doc in docs.values():
            doc.close()
    return out


def no_depth_lines(rows: Sequence[Dict[str, Any]], n_items: int,
                   di_dir: Path) -> List[str]:
    """What :func:`logs_with_no_depths` found, as the run prints it."""
    out = [f"{len(rows)} of {n_items} logs item(s) can keep NO depth: no "
           f"depth scale was fitted and no printed depth was read on any of "
           f"their pages, so the reader refuses every depth it reads off the "
           f"picture and only header values can score."]
    for row in rows:
        out.append(f"  {row['item']}: {row['pages']} page(s), "
                   f"{row['di_pages']} with DI text, {row['text_lines']} "
                   f"text line(s), {row['cells']} placed cell(s), "
                   f"{row['scale_labels']} scale label(s) the "
                   f"visual-scales switch would read")
    out.append(f"  Give these pages text (DI results in {di_dir}), or run "
               f"with --env GEOTECH_INGEST_VISUAL_SCALES=1 (it reads the "
               f"scale labels above, where there are any), or pass "
               f"--allow-no-depths to pay for the header values alone.")
    return out


def usd(value: float) -> str:
    """Dollars to the cent, or to a hundredth of a cent below a dime."""
    return f"${value:.2f}" if abs(value) >= 0.1 else f"${value:.4f}"


def short_model(model_id: str) -> str:
    for family in ("opus", "sonnet", "haiku", "fable"):
        if family in model_id:
            return family
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def default_arm(stage: str, model: str, vision: str,
                switches: Dict[str, str]) -> str:
    name = f"{stage}_{short_model(model)}"
    if vision != model:
        name += f"_vision-{short_model(vision)}"
    for key, value in sorted(switches.items()):
        name += "_" + key.replace(SWITCH_PREFIX, "").lower()
        if value != "1":
            name += f"-{value}"
    return name


# ---------------------------------------------------------------------------
# the ledger, shared by arms run side by side
# ---------------------------------------------------------------------------

class _FileLock:
    """A lock FILE beside the ledger, so two processes never interleave a
    read-modify-write. A lock older than ``stale`` seconds is a crash's
    leftover and is taken over."""

    def __init__(self, path: str, wait: float = 60.0, stale: float = 120.0):
        self.path = path + ".lock"
        self.wait, self.stale = wait, stale

    def __enter__(self) -> "_FileLock":
        deadline = time.time() + self.wait
        while True:
            try:
                os.close(os.open(self.path,
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    raise TimeoutError(f"the ledger lock {self.path} is "
                                       f"held; remove it if no run is live")
                time.sleep(0.05)

    def __exit__(self, *exc: Any) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass


def _shared_ledger_class():
    from module_work.live_smoke.spend import SpendLedger

    class SharedLedger(SpendLedger):
        """:class:`~live_smoke.spend.SpendLedger` that re-reads the file
        before every change and every cap check, under a lock file.

        The plain ledger holds the file in memory for a whole process and
        writes all of it back on every call, so two arms running at once
        would each overwrite the other's spend. Here the totals are always
        the file's. A read that fails (OneDrive holding the file a moment)
        is retried and, failing still, keeps what was last read -- it never
        starts from an empty ledger.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._held = False

        def _refresh(self) -> None:
            for attempt in range(8):
                try:
                    with open(self.path, encoding="utf-8") as fh:
                        data = json.load(fh)
                except FileNotFoundError:
                    return
                except (OSError, ValueError):
                    time.sleep(0.1 * (attempt + 1))
                    continue
                if isinstance(data, dict) and "waves" in data:
                    data.setdefault("cumulative_usd", 0.0)
                    self.data = data
                return

        def _locked(self, method: Callable, *args: Any, **kwargs: Any) -> Any:
            with self._lock, _FileLock(self.path):
                self._held = True
                try:
                    self._refresh()
                    return method(*args, **kwargs)
                finally:
                    self._held = False

        @property
        def cumulative(self) -> float:
            if not self._held:
                self._refresh()
            return float(self.data.get("cumulative_usd") or 0.0)

        def wave_spent(self, wave: str) -> float:
            if not self._held:
                self._refresh()
            return super().wave_spent(wave)

        def begin_wave(self, *args: Any, **kwargs: Any) -> dict:
            return self._locked(super().begin_wave, *args, **kwargs)

        def end_wave(self, *args: Any, **kwargs: Any) -> None:
            return self._locked(super().end_wave, *args, **kwargs)

        def add(self, *args: Any, **kwargs: Any) -> None:
            return self._locked(super().add, *args, **kwargs)

        def add_unmetered(self, *args: Any, **kwargs: Any) -> None:
            return self._locked(super().add_unmetered, *args, **kwargs)

        def record_stop(self, *args: Any, **kwargs: Any) -> None:
            return self._locked(super().record_stop, *args, **kwargs)

    return SharedLedger


def shared_ledger(path: Path) -> Any:
    return _shared_ledger_class()(str(path))


# ---------------------------------------------------------------------------
# what each call carried, and what the SDK retried
# ---------------------------------------------------------------------------

def _count_blocks(node: Any, kinds: Iterable[str]) -> int:
    kinds = tuple(kinds)
    if isinstance(node, dict):
        own = 1 if node.get("type") in kinds else 0
        return own + sum(_count_blocks(v, kinds) for v in node.values())
    if isinstance(node, (list, tuple)):
        return sum(_count_blocks(v, kinds) for v in node)
    return 0


def _picture_probe_class():
    from langchain_core.callbacks import BaseCallbackHandler

    class PictureProbe(BaseCallbackHandler):
        """Counts the pictures in each call, twice: as LangChain blocks
        (what the engine handed the model) and as Anthropic ``image``
        blocks in the request ``langchain_anthropic`` builds from them (what
        reaches the API). Never raises."""

        run_inline = True

        def __init__(self) -> None:
            super().__init__()
            self.by_run: Dict[str, Dict[str, Any]] = {}

        def on_chat_model_start(self, serialized, messages, *, run_id,
                                **kwargs):
            try:
                batch = list(messages[0]) if messages else []
                row: Dict[str, Any] = {"lc_images": sum(
                    _count_blocks(getattr(m, "content", None),
                                  ("image_url", "image")) for m in batch)}
                try:
                    from langchain_anthropic.chat_models import (
                        _format_messages,
                    )
                    _system, formatted = _format_messages(batch)
                    row["api_images"] = _count_blocks(formatted, ("image",))
                except Exception:                       # noqa: BLE001
                    row["api_images"] = None
                self.by_run[str(run_id)] = row
            except Exception:                           # noqa: BLE001
                pass

    return PictureProbe


class RetryLog(logging.Handler):
    """Keeps every "Retrying request" line the Anthropic SDK logs: a retry
    inside the client is invisible to both meters otherwise."""

    LOGGER = "anthropic._base_client"

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.rows: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(self.LOGGER)
        self._level = self._logger.level

    def emit(self, record: logging.LogRecord) -> None:
        try:
            said = record.getMessage()
        except Exception:                               # noqa: BLE001
            return
        if said.startswith("Retrying request"):
            self.rows.append({"ts": time.time(), "said": said[:200]})

    def __enter__(self) -> "RetryLog":
        self._logger.addHandler(self)
        if self._logger.getEffectiveLevel() > logging.INFO:
            self._logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._logger.removeHandler(self)
        self._logger.setLevel(self._level)


def claude_chat(model_id: str, callbacks: Sequence[Any]) -> Any:
    """``ChatAnthropic`` exactly as the live-smoke harness builds it
    (``live_smoke.models.build_claude``: the app's output cap, the key
    passed in and never printed, the SDK's backoff), with every callback
    on the model object."""
    from module_work.live_smoke.models import build_claude

    chat = build_claude(model_id, callbacks[0])
    chat.callbacks = list(callbacks)
    return chat


def price_the_scorecard(models: Iterable[str]) -> None:
    """Put the ledger's rates for these models into the package's price
    table, in this process only, so the run files' cost blocks are priced
    as the ledger prices the calls."""
    from module_work.live_smoke.spend import rates_for
    from report_ingest import engine

    for model in models:
        rates = rates_for(model, 0)
        engine.MODEL_PRICES.setdefault(model, (rates["input"],
                                               rates["output"]))


# ---------------------------------------------------------------------------
# reading an out_dir back ($0)
# ---------------------------------------------------------------------------

def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_files(out: Path, stage: str) -> Dict[str, dict]:
    folder = out / stage
    return {p.stem: blob for p in sorted(folder.glob("*.json"))
            if (blob := _load(p)) is not None} if folder.is_dir() else {}


def _scored_part(blob: dict) -> dict:
    for key in ("after", "score"):
        if isinstance(blob.get(key), dict):
            return blob[key]
    return {}


#: The engine's notes on an answer (report_ingest.engine._answer_notes).
_DROPPED_NOTE = "unknown keys dropped from the answer: "
_EMPTIED_NOTE = "nothing was left once the unknown keys"
#: An unknown key beside the answer's fields that held an object naming them.
_STRAY_OBJECT = re.compile(
    r"the unknown key (['\"])(.*?)\1 held an object naming fields")
#: A key name the model chose goes into RESULTS.md only if it looks like one.
_KEY_NAME = re.compile(r"[A-Za-z_][\w.]{0,40}", re.ASCII)


def _key_name(key: str) -> str:
    key = key.strip()
    return key if _KEY_NAME.fullmatch(key) else "(other)"


def _emptied_keys(note: str) -> List[str]:
    """The keys an emptied answer's note names, dropped or passed over."""
    said = note.split(_EMPTIED_NOTE, 1)[1]
    for words in (" were dropped", " were passed over"):
        said = said.replace(words, "")
    return [k for k in said.replace(" and ", ", ").split(",") if k.strip()]


def call_health(blobs: Dict[str, dict]) -> Dict[str, Any]:
    """Stop reasons, failures, cut-offs, refusals and answers not taken, off
    the run files' per-call cost rows.

    An answer lost to its own keys is an ERROR, not a count:
    ``answers_emptied`` (nothing was left once its unknown keys were
    dropped, so the engine did not take it) and ``emptied_unverified`` (a
    row from an engine before 2026-10-24, which took what was left and kept
    no raw arguments: a top-level key dropped from a taken answer may have
    been the whole answer, wrapped). ``stray_objects`` is a WARNING: an
    unknown key beside the answer's fields held an object naming them. The
    engine sets it aside, never merges it -- most are a map of the model's
    own (evidence, units) -- but it may be the answer hidden beside an id.
    Key names are the model's choosing: only identifier-like ones are
    reported, the rest as "(other)"."""
    from report_ingest.engine import CUT_REASONS

    stops: Counter = Counter()
    errors: Counter = Counter()
    emptied_by: Counter = Counter()
    unverified_by: Counter = Counter()
    stray_by: Counter = Counter()
    out = {"replies": 0, "failed": 0, "cut_off": 0, "refusals": 0,
           "answers_not_taken": 0, "unknown_keys_dropped": 0,
           "answers_unwrapped": 0, "answers_emptied": 0,
           "stray_objects": 0, "emptied_unverified": 0,
           "items_with_error": 0, "cap_refused": 0}
    for blob in blobs.values():
        if _scored_part(blob).get("error"):
            out["items_with_error"] += 1
        for row in (blob.get("cost") or {}).get("per_call") or []:
            if row.get("outcome") == "error":
                out["failed"] += 1
                errors[str(row.get("error_kind") or "other")] += 1
                if CAP_TEXT.search(str(row.get("error") or "")):
                    out["cap_refused"] += 1
                continue
            out["replies"] += 1
            stop = str(row.get("stop") or "(none)")
            stops[stop] += 1
            if stop in CUT_REASONS:
                out["cut_off"] += 1
            if stop == "refusal":
                out["refusals"] += 1
            notes = [str(n) for n in row.get("notes") or []]
            refused = any("not taken: " in n for n in notes)
            for note in notes:
                if "not taken: " in note:
                    out["answers_not_taken"] += 1
                if "arrived wrapped in" in note:
                    out["answers_unwrapped"] += 1
                stray = _STRAY_OBJECT.search(note)
                if stray:
                    out["stray_objects"] += 1
                    stray_by[_key_name(stray.group(2))] += 1
                if _EMPTIED_NOTE in note:
                    out["answers_emptied"] += 1
                    emptied_by.update(_key_name(k)
                                      for k in _emptied_keys(note))
                if _DROPPED_NOTE not in note:
                    continue
                out["unknown_keys_dropped"] += 1
                # Before 2026-10-24 a note of keys dropped beside no note of
                # an answer not taken meant the answer was taken, whatever
                # was left of it, and no raw arguments were kept to tell.
                keys = note.split(_DROPPED_NOTE, 1)[1].split(", ")
                top = [k for k in keys if k and "." not in k]
                if top and not refused and "answer_args" not in row:
                    out["emptied_unverified"] += 1
                    unverified_by.update(_key_name(k) for k in top)
    out["emptied_keys"] = dict(emptied_by.most_common())
    out["unverified_keys"] = dict(unverified_by.most_common())
    out["stray_keys"] = dict(stray_by.most_common())
    out["stops"] = dict(stops.most_common())
    out["errors"] = dict(errors.most_common())
    return out


def _key_list(counts: Dict[str, int]) -> str:
    return ", ".join(f"'{k}' {n}" for k, n in counts.items()) or "none named"


def _provs(value: Any) -> List[dict]:
    """A ``prov`` field as a list: one provenance, or a laboratory test's
    list of them (one per page or box it was read from)."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [p for p in value if isinstance(p, dict)]
    return []


def _has_page(provs: List[dict]) -> bool:
    return any(isinstance(p.get("page"), int) for p in provs)


def _has_box(provs: List[dict]) -> bool:
    return any(isinstance(p.get("bbox"), (list, tuple))
               and len(p["bbox"]) == 4 for p in provs)


def provenance(blobs: Dict[str, dict]) -> Dict[str, Dict[str, Any]]:
    """For the reader's final record and the model's own: where each value
    says it came from.

    A RECORD is one calculation, one boring, one laboratory test; it may
    carry a ``prov`` of its own. A VALUE is an object inside it with a
    ``prov`` field, counted three ways: with a provenance of its own (and
    then whether that names a page and a box, and how it was read), taking
    its record's (its own is empty and the record's is not: a laboratory
    test's numbers sit under the test's one provenance), or with none at
    all. A quantity whose empty ``prov`` sits inside a value that has one --
    a calculation value's number -- is part of that value, not a second.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any, tally: Dict[str, Any], owner: str) -> None:
        """``owner``: ``value`` inside a value with a provenance, else
        ``record`` or ``bare`` for a record with one or without."""
        if isinstance(node, list):
            for value in node:
                walk(value, tally, owner)
            return
        if not isinstance(node, dict):
            return
        inner = owner
        if "prov" in node:
            provs = _provs(node.get("prov"))
            if provs:
                tally["values"] += 1
                tally["own"] += 1
                tally["own_page"] += _has_page(provs)
                tally["own_box"] += _has_box(provs)
                tally["methods"][str(provs[0].get("method"))] += 1
                inner = "value"
            elif owner == "record":
                tally["values"] += 1
                tally["from_record"] += 1
            elif owner == "bare":
                tally["values"] += 1
                tally["none"] += 1
        for key, value in node.items():
            if key != "prov":
                walk(value, tally, inner)

    for which in ("record", "model_record"):
        tally: Dict[str, Any] = {
            "items": 0, "records": 0, "records_prov": 0,
            "records_page": 0, "records_box": 0, "values": 0, "own": 0,
            "own_page": 0, "own_box": 0, "from_record": 0, "none": 0,
            "methods": Counter()}
        for blob in blobs.values():
            part = _scored_part(blob).get(which)
            # One record (a calculation) or a list of them (a log's borings,
            # a sheet's tests).
            records = [r for r in (part if isinstance(part, list) else [part])
                       if isinstance(r, dict) and r]
            tally["items"] += bool(records)
            for one in records:
                tally["records"] += 1
                provs = _provs(one.get("prov"))
                if provs:
                    tally["records_prov"] += 1
                    tally["records_page"] += _has_page(provs)
                    tally["records_box"] += _has_box(provs)
                for key, value in one.items():
                    if key != "prov":
                        walk(value, tally, "record" if provs else "bare")
        if tally["items"]:
            tally["methods"] = dict(tally["methods"].most_common())
            out[which] = tally
    return out


def provenance_line(which: str, tally: Dict[str, Any]) -> str:
    return (f"- `{which}` ({tally['items']} items, {tally['records']} "
            f"records, {tally['records_prov']} with a provenance of their "
            f"own: {tally['records_page']} with a page, "
            f"{tally['records_box']} with a box). {tally['values']} values "
            f"inside them: {tally['own']} with their own provenance "
            f"({tally['own_page']} with a page, {tally['own_box']} with a "
            f"box; read by "
            + (", ".join(f"{k} {v}" for k, v in tally["methods"].items())
               or "-")
            + f"), {tally['from_record']} taking their record's, "
              f"{tally['none']} with none.")


def item_rows(stage: str, blobs: Dict[str, dict]) -> List[Dict[str, Any]]:
    rows = []
    for item, blob in blobs.items():
        cost = blob.get("cost") or {}
        part = _scored_part(blob)
        if stage == "narrative":
            rec, prec = part.get("recall") or {}, part.get("precision") or {}
            score = (f"recall {rec.get('found', 0)}/{rec.get('total', 0)}, "
                     f"precision {prec.get('found', 0)}/"
                     f"{prec.get('total', 0)}")
        else:
            before = (blob.get("before") or {}).get("overall") or {}
            after = part.get("overall") or {}
            score = (f"{before.get('found', 0)}/{before.get('total', 0)} -> "
                     f"{after.get('found', 0)}/{after.get('total', 0)}")
        rows.append({"item": item, "set": blob.get("set", ""),
                     "calls": int(cost.get("calls") or 0),
                     "failed": int(cost.get("failed") or 0),
                     "input": int(cost.get("input_tokens") or 0),
                     "cache_read": int(cost.get("cache_read_tokens") or 0),
                     "output": int(cost.get("output_tokens") or 0),
                     "usd": float(cost.get("dollars") or 0.0),
                     "seconds": float(blob.get("seconds") or 0.0),
                     "score": score,
                     "error": bool(part.get("error"))})
    return rows


def spend_log(out: Path) -> List[dict]:
    path = out / "spend_calls.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _sum_calls(calls: Sequence[dict]) -> Dict[str, Any]:
    tot = {"calls": 0, "errors": 0, "uncached": 0, "cache_read": 0,
           "cache_write": 0, "output": 0, "usd": 0.0}
    for call in calls:
        if call.get("error"):
            tot["errors"] += 1
            continue
        tot["calls"] += 1
        tot["usd"] += float(call.get("usd") or 0.0)
        tok = call.get("tokens") or {}
        tot["uncached"] += int(tok.get("uncached") or 0)
        tot["cache_read"] += int(tok.get("cache_read") or 0)
        tot["cache_write"] += int(tok.get("cache_write_5m") or 0) + int(
            tok.get("cache_write_1h") or 0)
        tot["output"] += int(tok.get("output") or 0)
    return tot


def _sum_items(rows: Sequence[dict]) -> Dict[str, Any]:
    return {"calls": sum(r["calls"] for r in rows),
            "failed": sum(r["failed"] for r in rows),
            "uncached": sum(r["input"] for r in rows),
            "cache_read": sum(r["cache_read"] for r in rows),
            "output": sum(r["output"] for r in rows),
            "usd": sum(r["usd"] for r in rows)}


def estimate(stage: str, rows: Sequence[dict], n_items: Optional[int]
             ) -> List[str]:
    """Per item, and for the whole stage, at each priced model's rates,
    from the tokens these items took. Output tokens differ by model, so the
    other model's figure is a guide, not a forecast."""
    from module_work.live_smoke.spend import PRICES, rates_for

    done = [r for r in rows if r["calls"] and not r["error"]]
    if not done:
        return []
    n = len(done)
    tin = sum(r["input"] + r["cache_read"] for r in done) / n
    tout = sum(r["output"] for r in done) / n
    lines = [f"Per item ({n} read): {tin:,.0f} input and {tout:,.0f} output "
             f"tokens. Items differ in size, so a pick of a few small ones "
             f"understates the stage."]
    for model in ("claude-sonnet-5-5", "claude-opus-5-5"):
        if model not in PRICES:
            continue
        rates = rates_for(model, int(tin))
        each = (tin * rates["input"] + tout * rates["output"]) / 1e6
        line = f"- {model}: ${each:.3f} an item"
        if n_items:
            line += (f"; the whole {stage} stage ({n_items} items) about "
                     f"${each * n_items:.2f}")
        lines.append(line)
    return lines


def _table(rows: Sequence[dict]) -> List[str]:
    out = ["| item | set | calls | failed | input | output | USD | s | "
           "before -> after |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        out.append(f"| {r['item']} | {r['set']} | {r['calls']} | "
                   f"{r['failed']} | {r['input'] + r['cache_read']:,} | "
                   f"{r['output']:,} | {r['usd']:.4f} | {r['seconds']:.0f} | "
                   f"{r['score']}{' (error)' if r['error'] else ''} |")
    return out


def check_lines(out: Path, stage: str, truth_root: Optional[Path] = None
                ) -> List[str]:
    """The $0 half of the block: calls, values, items, estimate."""
    blobs = run_files(out, stage)
    rows = item_rows(stage, blobs)
    health = call_health(blobs)
    lines = [f"### Calls ({stage}, {len(blobs)} run file(s))", ""]
    lines.append(
        f"Replies {health['replies']}; stop reasons "
        + (", ".join(f"{k} {v}" for k, v in health["stops"].items())
           or "none")
        + f". Failed requests {health['failed']}"
        + (f" ({', '.join(f'{k} {v}' for k, v in health['errors'].items())})"
           if health["errors"] else "")
        + (f", {health['cap_refused']} of them refused by the wave cap"
           if health["cap_refused"] else "")
        + f". Cut off at the output cap: {health['cut_off']}. Refusals: "
          f"{health['refusals']}. Answers not taken: "
          f"{health['answers_not_taken']}; unknown keys dropped from an "
          f"answer: {health['unknown_keys_dropped']}; answers unwrapped: "
          f"{health['answers_unwrapped']}. Items carrying an "
          f"error: {health['items_with_error']}.")
    if health["answers_emptied"]:
        lines.append(
            f"ERROR: {health['answers_emptied']} answer(s) held nothing once "
            f"their unknown keys were dropped, and were not taken (keys: "
            f"{_key_list(health['emptied_keys'])}). Those calls' answers are "
            f"lost; each row's answer_args holds what the model sent.")
    if health["stray_objects"]:
        lines.append(
            f"WARNING: {health['stray_objects']} answer(s) carried an unknown "
            f"key beside the answer's fields holding an object that names "
            f"them (keys: {_key_list(health['stray_keys'])}). Each was set "
            f"aside, not merged: usually a map of the model's own, but it "
            f"may be the answer hidden beside an id; each row's answer_args "
            f"holds what the model sent.")
    if health["emptied_unverified"]:
        lines.append(
            f"ERROR (unverified): {health['emptied_unverified']} answer(s) "
            f"lost a top-level key and were taken, by an engine that kept no "
            f"raw arguments (keys: {_key_list(health['unverified_keys'])}). "
            f"If that key wrapped the whole answer, the answer taken was "
            f"empty (the wrapped-answer bug fixed 2026-10-24); check the "
            f"reading.")
    calls = spend_log(out)
    seen = [c for c in calls if c.get("lc_images") is not None]
    if seen:
        with_pics = [c for c in seen if c.get("lc_images")]
        api = [c.get("api_images") for c in seen
               if c.get("api_images") is not None]
        lines.append(
            f"Pictures: {len(with_pics)} of {len(seen)} metered calls "
            f"carried one or more; {sum(c['lc_images'] for c in seen)} "
            f"handed to the model, "
            + (f"{sum(api)} in the Anthropic request" if api
               else "the Anthropic request not inspected")
            + f", at most {max(c['lc_images'] for c in seen)} in one call.")
    retries = out / "sdk_retries.jsonl"
    if retries.is_file() or calls:
        n_retries = (len(retries.read_text(encoding="utf-8").splitlines())
                     if retries.is_file() else 0)
        lines.append(f"Retries inside the Anthropic SDK: {n_retries}.")
    else:
        lines.append("Retries inside the Anthropic SDK: not recorded (this "
                     "out_dir was not written by run_claude_local).")
    prov = provenance(blobs)
    lines += ["", "### Values and where they came from", ""]
    if not prov:
        lines.append(
            "The run files keep no record with provenance for this stage"
            + (" (the narrative reader's citations are a page and the "
               "page's words, and its run file keeps neither)."
               if stage == "narrative" else "."))
    for which, tally in prov.items():
        lines.append(provenance_line(which, tally))
    lines += ["", "### Per item", ""] + _table(rows)
    n_items = None
    if truth_root is not None and (truth_root / stage).is_dir():
        n_items = len(list((truth_root / stage).glob("*.json")))
    est = estimate(stage, rows, n_items)
    if est:
        lines += ["", "### What the whole stage would cost", ""] + est
    return lines


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def set_aside_capped(out: Path, stage: str) -> List[str]:
    """Run files whose reading hit the cap, moved to ``<stage>/_capped/`` so
    the next run reads them whole: a reader may keep what came before a
    refused call and save the item as if it were done."""
    moved = []
    for item, blob in run_files(out, stage).items():
        rows = (blob.get("cost") or {}).get("per_call") or []
        refused = any(row.get("outcome") == "error"
                      and CAP_TEXT.search(str(row.get("error") or ""))
                      for row in rows)
        if refused or CAP_TEXT.search(str(_scored_part(blob).get("error")
                                          or "")):
            (out / stage / "_capped").mkdir(exist_ok=True)
            os.replace(out / stage / f"{item}.json",
                       out / stage / "_capped" / f"{item}.json")
            moved.append(item)
    return moved


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _spend_row(label: str, tot: Dict[str, Any]) -> str:
    return (f"| {label} | {tot['calls']} | {tot['uncached']:,} | "
            f"{tot['cache_read']:,} | {tot['output']:,} | "
            f"{tot['usd']:.4f} |")


def spend_lines(out: Path, stage: str, meter: Any, ledger: Any, wave: str,
                cap: float, results: Optional[dict]) -> List[str]:
    this = _sum_calls(meter.calls)
    arm = _sum_calls(spend_log(out))
    items = _sum_items(item_rows(stage, run_files(out, stage)))
    ledger.wave_spent(wave)                     # re-reads the shared file
    w = ledger.wave(wave)
    tok = w.get("tokens") or {}
    wave_tot = {"calls": int(w.get("calls") or 0),
                "uncached": int(tok.get("uncached") or 0),
                "cache_read": int(tok.get("cache_read") or 0),
                "output": int(tok.get("output") or 0),
                "usd": float(w.get("spent_usd") or 0.0)}
    lines = ["### Spend", "",
             "| | calls | input | cache read | output | USD |",
             "|---|---:|---:|---:|---:|---:|",
             _spend_row("this run (the meter)", this),
             _spend_row("this arm, every run (spend_calls.jsonl)", arm),
             _spend_row("the run files (the stage's cost block)", items),
             _spend_row(f"wave `{wave}` on the ledger", wave_tot), ""]
    same = all(arm[k] == items[k] for k in ("calls", "uncached",
                                            "cache_read", "output")) \
        and abs(arm["usd"] - items["usd"]) < 5e-4
    if same:
        lines.append("The arm's metered calls and its run files agree: "
                     "calls, tokens and dollars.")
    else:
        lines.append(
            f"The arm's metered calls and its run files differ by "
            f"{arm['calls'] - items['calls']} call(s), "
            f"{arm['uncached'] - items['uncached']:,} input and "
            f"{arm['output'] - items['output']:,} output tokens, "
            f"${arm['usd'] - items['usd']:.4f}: calls made for an item whose "
            f"run file did not survive (an item that failed, was retried or "
            f"was set aside at the cap) are metered and in no run file.")
    if stage_cost := ((results or {}).get(stage) or {}).get("cost"):
        lines.append(f"The stage's cost block above: {stage_cost['calls']} "
                     f"calls, ${float(stage_cost.get('dollars') or 0):.4f}.")
    lines.append(f"Wave `{wave}`: {usd(wave_tot['usd'])} of its "
                 f"{usd(cap)} cap"
                 + (f", STOPPED: {w.get('stop_reason')}"
                    if w.get("stopped_by_cap") else "")
                 + f". Ledger: ${ledger.cumulative:.4f} of "
                   f"${ledger.hard_total:.2f} across every live run.")
    if meter.accounting_errors:
        lines.append(f"Accounting errors: {len(meter.accounting_errors)} "
                     f"(first: {meter.accounting_errors[0][:120]}).")
    return lines


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one arm (paid; metered)")
    run.add_argument("--stage", required=True, choices=tuple(STAGES))
    run.add_argument("--model", required=True,
                     help="the reader: claude-sonnet-5-5 | claude-opus-5-5")
    run.add_argument("--vision-model", default=None,
                     help="vision_chat_model (default: --model)")
    run.add_argument("--sets", default="all",
                     choices=("all", "open", "blind"))
    run.add_argument("--only", default="",
                     help="comma-separated items or report IDs")
    run.add_argument("--env", action="append", default=[],
                     metavar="NAME=VALUE", help="a switch to set (repeat)")
    run.add_argument("--budget", type=int, default=None,
                     help="the stage's per-item call budget "
                          "(default: the package's)")
    run.add_argument("--front-pages", type=int, default=50,
                     help="narrative_front_pages")
    run.add_argument("--wave", required=True)
    run.add_argument("--cap", type=float, required=True, help="USD")
    run.add_argument("--arm", default=None,
                     help="the arm's name (default: from the settings)")
    run.add_argument("--out-dir", default=None)
    run.add_argument("--raw-dir", default=None,
                     help="the private corpus folder (default: the main "
                          "checkout's)")
    run.add_argument("--truth-dir", default=None,
                     help="the truth root (default: <raw-dir>/truth)")
    run.add_argument("--reports-dir", default=None,
                     help="the folder holding the PDFs (default: "
                          "<raw-dir>/corpus)")
    run.add_argument("--ledger", default=None,
                     help="the spend ledger (default: the main checkout's)")
    run.add_argument("--redo", action="store_true",
                     help="read every chosen item again")
    run.add_argument("--allow-no-depths", action="store_true",
                     help="run logs items on which no depth can be kept "
                          "(refused by default; see logs_with_no_depths)")
    chk = sub.add_parser("check", help="read an out_dir back ($0)")
    chk.add_argument("out_dir")
    chk.add_argument("--stage", default=None, choices=tuple(STAGES))
    chk.add_argument("--truth-dir", default=None)
    return ap


def _paths(args: argparse.Namespace) -> Dict[str, Path]:
    main = main_checkout()
    raw = Path(args.raw_dir) if args.raw_dir else main / RAW_REL
    return {"main": main, "raw": raw,
            "reports": Path(args.reports_dir) if args.reports_dir
            else raw / "corpus",
            "truth": Path(args.truth_dir) if args.truth_dir
            else raw / "truth",
            "ledger": Path(args.ledger) if args.ledger
            else main / LEDGER_REL}


def run(args: argparse.Namespace,
        chat_builder: Callable[[str, Sequence[Any]], Any] = claude_chat
        ) -> int:
    from module_work.live_smoke.spend import (RefuseToStart, SpendMeter,
                                              rates_for)
    from report_ingest.cluster_scoring import score_on_cluster

    stage = args.stage
    model = args.model
    vision = args.vision_model or model
    for name in {model, vision}:
        rates_for(name, 0)                       # an unpriced model stops here
    paths = _paths(args)
    if not paths["reports"].is_dir():
        print(f"no reports at {paths['reports']} (pass --raw-dir or "
              f"--reports-dir)")
        return REFUSED
    budget_arg, budget_default = STAGES[stage]
    budget = args.budget or budget_default
    switches = set_switches(args.env)
    arm = args.arm or default_arm(stage, model, vision, switches)
    out = (Path(args.out_dir) if args.out_dir
           else paths["main"] / OUT_REL / arm)
    for root in {paths["main"], REPO}:
        if _inside(out, root) and git_ignored(out, root) is False:
            print(f"refused: {out} is inside the repo and git does not "
                  f"ignore it; run files hold private content")
            return REFUSED
    out.mkdir(parents=True, exist_ok=True)

    settings = {"stage": stage, "model": model, "vision_model": vision,
                "switches": switches, "budget": budget}
    if stage == "narrative":
        settings["front_pages"] = args.front_pages
    saved = _load(out / "arm.json")
    if saved and {k: saved.get(k) for k in settings} != settings:
        print(f"refused: {out} holds arm {saved.get('arm')!r} run with "
              f"other settings:")
        for key in settings:
            if saved.get(key) != settings[key]:
                print(f"  {key}: {saved.get(key)!r} there, "
                      f"{settings[key]!r} now")
        print("  use another --arm or --out-dir for a different arm")
        return REFUSED
    if not saved:
        (out / "arm.json").write_text(json.dumps(
            {"arm": arm, **settings, "created": time.strftime(
                "%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")

    only = [t.strip() for t in args.only.split(",") if t.strip()]
    chosen = pick_items(stage, paths["truth"], args.sets, only)
    if not chosen:
        print(f"no {stage} item is in set {args.sets!r}"
              + (f" among {', '.join(only)}" if only else ""))
        return REFUSED
    picked = bool(only) or args.sets != "all"
    if stage == "logs":
        from report_ingest import visual_scales

        no_depths = logs_with_no_depths(paths["reports"],
                                        paths["raw"] / "di", chosen)
        if no_depths:
            print("\n".join(no_depth_lines(no_depths, len(chosen),
                                           paths["raw"] / "di")))
            if not (args.allow_no_depths or visual_scales.enabled()):
                print("refused: nothing was spent")
                return REFUSED
    truth_root = truth_for_run(stage, paths["truth"], chosen, out, picked)

    ledger = shared_ledger(paths["ledger"])
    try:
        ledger.begin_wave(args.wave, args.cap, model=model,
                          mode=f"report_ingest:{stage}:{arm}")
    except RefuseToStart as exc:
        print(f"refused: {exc}")
        return REFUSED
    meter = SpendMeter(ledger, args.wave, args.cap, model_id=model)
    meter.scenario = arm
    probe = _picture_probe_class()()
    reader = chat_builder(model, [meter, probe])
    looker = (reader if vision == model
              else chat_builder(vision, [meter.for_model(vision), probe]))
    price_the_scorecard({model, vision})

    print(f"arm {arm}: {stage} on {model}"
          + (f" (vision {vision})" if vision != model else "")
          + f", {len(chosen)} item(s), switches "
          + (", ".join(f"{k}={v}" for k, v in switches.items()) or "none")
          + f"; wave {args.wave} cap {usd(args.cap)}; out_dir {out}")
    results: Optional[dict] = None
    retries = RetryLog()
    try:
        with retries:
            results = score_on_cluster(
                reports_dir=paths["reports"],
                di_dir=paths["raw"] / "di",
                out_dir=out, prompter=None, model=model,
                vision_model=vision, stages=(stage,), truth_dir=truth_root,
                chat_model=reader, vision_chat_model=looker,
                redo=args.redo, sharepoint=None,
                narrative_front_pages=args.front_pages,
                **{budget_arg: budget})
    finally:
        ledger.end_wave(args.wave)
        _append_jsonl(out / "spend_calls.jsonl", (
            {**call, **probe.by_run.get(call.get("run_id"), {}),
             "arm": arm} for call in meter.calls))
        _append_jsonl(out / "sdk_retries.jsonl", retries.rows)
        capped = set_aside_capped(out, stage)

    lines = ["", "## Claude, locally (run_claude_local)", "",
             f"Arm `{arm}`: stage `{stage}`, reader `{model}`"
             + (f", vision `{vision}`" if vision != model else "")
             + f", budget {budget}, switches "
             + (", ".join(f"`{k}={v}`" for k, v in switches.items())
                or "none")
             + f"; {len(chosen)} item(s) asked (sets {args.sets}"
             + (f", only {', '.join(only)}" if only else "") + ").", ""]
    lines += spend_lines(out, stage, meter, ledger, args.wave, args.cap,
                         results)
    if capped:
        lines += ["", f"SET ASIDE AT THE CAP and read again by the next "
                      f"run: {', '.join(capped)}. RESULTS.md above still "
                      f"counts them."]
    lines += [""] + check_lines(out, stage, paths["truth"])
    with (out / "RESULTS.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    stopped = meter.stopped or ledger.wave(args.wave).get("stopped_by_cap")
    if stopped:
        print(f"\nSTOPPED BY THE CAP: {stopped}. Raise --cap on wave "
              f"{args.wave} and run the same command to go on.")
        return STOPPED_BY_CAP
    return OK


def check(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    truth = (Path(args.truth_dir) if args.truth_dir
             else main_checkout() / RAW_REL / "truth")
    stages = [args.stage] if args.stage else [
        s for s in STAGES if (out / s).is_dir()]
    if not stages:
        print(f"no stage folder in {out}")
        return REFUSED
    for stage in stages:
        print("\n".join(check_lines(out, stage, truth)))
        print()
    return OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args) if args.cmd == "run" else check(args)


if __name__ == "__main__":
    sys.exit(main())
