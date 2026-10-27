"""Score the two model passes on the cluster, against the real model.

The app runs in Funhouse against OpenAI models through Prompter. A score
measured on any other model measures a model that will never do the work, so
the numbers that count come from here: one function the owner calls in a
notebook with the live ``fh_prompter`` object already in hand.

    from report_ingest.cluster_scoring import score_on_cluster

    score_on_cluster(
        reports_dir = "/Volumes/main/geotech/reports",
        manifest    = "/Volumes/main/geotech/wp1b/MANIFEST.md",
        labels_xlsx = "/Volumes/main/geotech/wp1b/trial_pages_working_r2.xlsx",
        oos_labels  = "/Volumes/main/geotech/wp1b/oos_labels.json",
        di_dir      = "/Volumes/main/geotech/report_di",
        out_dir     = "/tmp/report_ingest_wp1b",
        prompter    = fh_prompter,
        model       = "funhouse-gpt-high",   # the tier the app runs on
    )

THE REPORTS STAY WHERE THEY ARE. ``reports_dir`` is the folder the owner
already has on the cluster, under the file names their authors gave them;
the manifest's source-file column is what turns a file name into an ID, so
that one small file has to travel with the run. The DI results are read in
either form -- ``<ID>.json.gz`` or ``DI_data_<original stem>.json``.

NO CREDENTIAL IS READ OR STORED. Authentication is whatever the passed-in
``fh_prompter`` was built with. Nothing here touches an environment
variable, a key file or a secret scope.

OR A CHAT MODEL READS INSTEAD (BACKLOG RI-67). ``chat_model=`` -- a
LangChain chat model, or a callable returning one for a model name --
puts the app's own engine in the Prompter's place: on Foundry, Claude
through the workspace's ``palantir_models`` classes, built and
authenticated by the caller. See :func:`score_on_cluster`.

MODEL FIRST, AS A SECOND ARM (BACKLOG RI-87). ``model_first="on"`` reads
each hand-truthed item with the model-first reader of its stage
(:mod:`report_ingest.model_first_scoring`) and each whole report with
``graph.ingest_report(model_first="on")``, scored by the same scorers as
today's arm. Every run file and RESULTS block names the arm it ran in, and
under ``on`` the page route, the models and the instruction version. One
``out_dir`` holds one arm. See :func:`score_on_cluster`.

WHERE THE OUTPUT GOES. ``out_dir`` is the working folder and local disk is
the sensible place for it: a run writes one small file per report per stage.
No path is refused -- a workspace ``out_dir`` is allowed and gets a printed
note (:func:`out_dir_note`), because the workspace folder is one of the two
places that keep things on this cluster.

AND IT IS MIRRORED SOMEWHERE THAT SURVIVES. ``/tmp`` does not survive a
cluster restart: the first full run's 38 label reviews, about $17 of model
calls, were wiped by one. Pass ``sharepoint=fh_sp_client`` (or
``durable_dir`` pointing at a workspace folder) and every run file is copied
to
``<sharepoint_folder>/<the out_dir's own name>`` as soon as it is written,
and copied BACK into a wiped ``out_dir`` at the start of the next call. A
mirror failure prints a warning and never stops the run; see
:mod:`report_ingest.mirror`.

IT IS RESTARTABLE. Every report writes ``runs/<ID>.json`` as it finishes and
a later call skips any report that already has one. A detached notebook, an
expired token or a 429 storm costs the reports that had not finished, not
the ones that had. Delete a file to redo that report; pass ``redo=True`` to
redo all of them.

WHAT IT WRITES into ``out_dir``::

    runs/<ID>.json        rules, profile, review, cost -- one per report
    triage/<ID>.json      the document profile alone, for the owner's audit
    vision/<ID>.json      the vision experiment's labels, reasons and cost
    vote/<ID>.json        every page the voters split on, and who won it
    ingest/<ID>/          the whole pipeline's output for one report: the
                          record, the summary, the library page, the DIGGS
                          file, qa.json, and run.json with the counts
                          and the cost of every attempt at the report
    ingest/reports.db     the library index over every report ingested
    results.json          every summary table as data
    RESULTS.md            the same tables to read, and to bring back

``RESULTS.md`` is the file to bring home. It carries IDs, labels, counts and
rates and nothing else -- never a page heading, a change's reason, a triage
rationale or a vision reason, any of which can name a firm, a project or a
person. Those stay in ``runs/``, ``triage/`` and ``vision/``, which stay on
the cluster unless the owner moves them deliberately.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from report_ingest.corpus import Corpus
from report_ingest.engine import (
    CallCeilingReached, tokens_in, tokens_in_words,
)
from report_ingest.mirror import DEFAULT_FOLDER, Mirror
from report_ingest.scoring import (
    CHECKPOINT, GATE, KEY_CONTENT, OOS_BLIND, OOS_OPEN, Scores,
    columns_label_table, disputed_drop, gate_failures, label_table,
    verdict_for,
)
from report_ingest.visual_scales import enabled as visual_scales_enabled
from report_ingest.vote import (
    POLICIES, STRUCTURAL_RULES_WIN, PageVote, agreement, build_votes,
    policy_labels, required_review_accuracy, trust_table,
    vision_confidences,
)

__all__ = ["score_on_cluster", "SET_NAMES", "STAGE_NAMES",
           "out_dir_note"]


def _saved_failure(blob: Dict[str, Any]) -> Optional[str]:
    """The error a saved run file records, or ``None`` for a run that finished.

    A scorer catches a failed model call and stores it on the score -- the
    log and lab readers under ``after.error``, the narrative reader under
    ``score.error`` -- so the stage writes the blob like any other and a
    later call would skip it as done. A failed call is not a result: the
    first cluster run (2026-09-18) froze six parameter refusals this way and
    the next run carried them forward as finished. Such a run file is
    retried instead.
    """
    for key in ("after", "score"):
        part = blob.get(key)
        if isinstance(part, dict) and part.get("error"):
            return str(part["error"])
    if blob.get("error"):
        return str(blob["error"])
    return None

SET_NAMES: Tuple[str, ...] = ("insample", "oos_open", "oos_blind")

#: The measurements this run can make. ``labels`` is WP1b -- triage and the
#: label review over whole reports. ``logs`` is WP2b -- the log grid and then
#: the log reader over each hand-truthed log, scored before and after.
#: ``lab`` is WP3 -- the page's own tables and then the lab reader over each
#: hand-truthed laboratory sheet, scored the same two ways. ``calc`` is WP5
#: -- the floor of labelled values a pattern reads off a calculation
#: printout, and then the calculation reader over each hand-truthed run,
#: scored the same two ways. ``vision_labels``
#: is the WP5 experiment -- every page's PICTURE to the cheapest tier, scored
#: against the same hand labels with the same scorer as the rules, so the
#: three ways of labelling a page can be read side by side.
#: ``vote`` is WP6 -- no model at all: the rules recomputed with planlens,
#: the vision labels off a saved run, and the review's labels where a label
#: run sits beside them, set against each other and against the hand labels
#: to say what a disagreement is worth.
#: ``ingest`` (5.23.0) is the whole pipeline: ``graph.ingest_report`` end
#: to end on each report -- triage, labels (a saved review reused where one
#: sits in ``runs/``), the three readers with their floors, the reconciler,
#: the writers -- producing the record and its exports, DIGGS included, and
#: scoring the record against whatever hand truth exists for that report.
#: ``range_labels`` (BACKLOG RI-73) describes every page with a vision model
#: and labels each report by sections from the descriptions, scored by
#: accuracy and by the pages a wrong label sends to the wrong reader.
STAGE_NAMES: Tuple[str, ...] = ("labels", "logs", "lab", "calc",
                                "soundings", "narrative", "vision_labels",
                                "vote", "range_labels", "ingest")

#: The stages that make no model call, and so need no prompter: the vote
#: combines label and vision runs already on disk (BACKLOG RI-08).
NO_MODEL_STAGES: Tuple[str, ...] = ("vote",)

#: The reports the log rules were allowed to be tuned on, when no ``OPEN.txt``
#: sits beside the truth files. Everything else is scored as blind.
DEFAULT_OPEN_LOGS: Tuple[str, ...] = ("R36", "R37", "R06", "R07", "R15",
                                      "R28")
#: The same, for the laboratory sheets.
DEFAULT_OPEN_LAB: Tuple[str, ...] = ("R36", "R28", "R17", "R06")

#: Prefixes an ``out_dir`` gets a printed note about. NOT a refusal: see
#: :func:`out_dir_note`.
_WORKSPACE_PREFIXES = ("/Workspace", "/workspace")


def out_dir_note(out_dir: Any) -> Optional[str]:
    """A line to print about where this run is writing, or ``None``.

    NOTHING IS REFUSED HERE ANY MORE. Until this train a ``/Workspace``
    ``out_dir`` raised, on the strength of one observed write that appeared
    to succeed and left nothing behind. The owner's own experience
    supersedes that, 2026-09-20: the workspace folder under ``geotech_app/``
    and SharePoint are the two places that KEEP things here, so refusing to
    write to a workspace folder was refusing the durable option.

    What is left is a note. A run writes one small file per report per
    stage, and a workspace folder is a poor place for hundreds of them, so
    the shape that works is a working ``out_dir`` on local disk with a
    workspace folder or SharePoint as the ``durable_dir`` / ``sharepoint``
    mirror.
    """
    text = str(out_dir).replace("\\", "/")
    if text.startswith(_WORKSPACE_PREFIXES):
        return (f"  note: out_dir {out_dir} is a workspace folder, which is "
                f"durable here but takes one small file per report per "
                f"stage. A local out_dir with durable_dir= pointing at a "
                f"workspace folder writes the same files and syncs them in "
                f"one pass.")
    return None


class _MirrorSync:
    """Copies ``out_dir`` to the durable mirror, and says so once.

    A mirror is insurance, never the work. A failure prints ONE line the
    first time that message appears -- not one per report, which on a broken
    SharePoint would be 38 identical lines -- and the run carries on with its
    output in ``out_dir`` alone. What was sent, and what failed, is counted
    and printed at the end, which is where it gets read.
    """

    def __init__(self, mirror: Mirror, out: Path) -> None:
        self._mirror = mirror
        self._out = Path(out)
        self.remote = mirror.remote_for(self._out.name)
        self.uploaded = 0
        self.failed = 0
        self.restored = 0
        self._said: set = set()

    def __call__(self) -> None:
        if not self._mirror.active:
            return
        summary = self._mirror.mirror_dir(self._out, self.remote)
        self.uploaded += summary["uploaded"]
        for message in summary["errors"]:
            self.failed += 1
            self._warn(message)

    def restore(self) -> None:
        """Bring back anything the mirror holds that ``out_dir`` does not."""
        if not self._mirror.active:
            print("  no durable mirror: pass sharepoint=fh_sp_client or "
                  "durable_dir=... so a cluster restart cannot cost this run "
                  "twice")
            return
        print(f"  durable mirror: {self._mirror.describe(self.remote)}")
        summary = self._mirror.restore_dir(self.remote, self._out)
        self.restored = summary["downloaded"]
        for message in summary["errors"]:
            self._warn(message)
        print(f"  restored {self.restored} file(s) the mirror held and "
              f"{self._out} did not")

    def report(self) -> None:
        if not self._mirror.active:
            return
        print(f"mirrored {self.uploaded} file(s) to {self.remote}"
              + (f"; {self.failed} upload(s) FAILED and are in {self._out} "
                 f"only" if self.failed else ""))

    def _warn(self, message: str) -> None:
        key = message.split(": ", 1)[-1]
        if key in self._said:
            return
        self._said.add(key)
        print(f"  WARNING: the durable mirror failed -- {message}. The run "
              f"carries on; its output is in {self._out} only.")


def _sync(sync: Optional[Any]) -> None:
    """Call a mirror sync if there is one. A stage takes no view on it."""
    if sync is not None:
        sync()


#: The subfolder each scoring stage keeps its hand truth in, under one truth
#: root. The names are the stage names, so ``stages`` and the folder listing
#: read the same way.
TRUTH_SUBDIRS: Dict[str, str] = {"logs": "logs", "lab": "lab",
                                 "calc": "calc",
                                 "soundings": "soundings",
                                 "narrative": "narrative"}


def _truth_dirs(truth_dir: Any, lab_truth_dir: Any, narrative_truth_dir: Any,
                calc_truth_dir: Any = None, sounding_truth_dir: Any = None
                ) -> Tuple[Optional[Path], Optional[Path], Optional[Path],
                           Optional[Path], Optional[Path]]:
    """``(logs, lab, narrative, calc)`` folders, from a ROOT or from four.

    ONE FOLDER TRAVELS TO THE CLUSTER, not three. The hand truth is private,
    so it is uploaded by hand before every run; asking for three Volume paths
    that must each be right invites one of them to be stale while the run
    still starts and scores the wrong thing. So ``truth_dir`` may be a root
    holding ``logs/``, ``lab/`` and ``narrative/`` -- the stage names -- and
    each stage takes its own folder from it.

    The older three-argument form still works and always wins: an explicit
    ``lab_truth_dir`` or ``narrative_truth_dir`` overrides the root, and a
    ``truth_dir`` that holds the log truth files DIRECTLY, with no ``logs/``
    in it, is used as-is the way it always was. Nothing here touches the
    disk beyond asking whether a subfolder exists, and a missing folder is
    left as ``None`` so the caller raises the message that names the stage.
    """
    root = Path(truth_dir) if truth_dir is not None else None

    def under(name: str) -> Optional[Path]:
        if root is None:
            return None
        candidate = root / TRUTH_SUBDIRS[name]
        return candidate if candidate.is_dir() else None

    logs = under("logs") or root
    lab = Path(lab_truth_dir) if lab_truth_dir is not None else under("lab")
    calc = Path(calc_truth_dir) if calc_truth_dir is not None \
        else under("calc")
    soundings = Path(sounding_truth_dir) if sounding_truth_dir is not None \
        else under("soundings")
    narrative = (Path(narrative_truth_dir)
                 if narrative_truth_dir is not None else under("narrative"))
    return logs, lab, narrative, calc, soundings


def _set_ids(name: str, corpus: Corpus) -> Tuple[str, ...]:
    if name == "insample":
        return tuple(corpus.mapped_ids()) if corpus.labels_available else ()
    if name == "oos_open":
        return OOS_OPEN
    if name == "oos_blind":
        return OOS_BLIND
    raise ValueError(f"unknown set {name!r}; the sets are {SET_NAMES}")


def _truth_for(rid: str, corpus: Corpus, oos: Dict[str, Dict[int, dict]],
               mapped: Sequence[str]
               ) -> Tuple[Dict[int, str], Dict[int, Tuple[str, ...]], str]:
    """``(label per page, acceptable alternates, where it came from)``.

    The spreadsheet wins when a report has one: it labels every page, and the
    out-of-sample file samples five.
    """
    if rid in mapped:
        try:
            return ({pl.page0: pl.label for pl in corpus.labels_for(rid)}, {},
                    "spreadsheet")
        except (FileNotFoundError, KeyError, ImportError):
            pass
    rows = oos.get(rid)
    if rows:
        return ({p: v["label"] for p, v in rows.items()},
                {p: tuple(v.get("alternates") or ()) for p, v in rows.items()},
                "lead, 5 pages")
    return {}, {}, "none"


# ---------------------------------------------------------------------------
# the engine each call goes through (BACKLOG RI-67)
# ---------------------------------------------------------------------------

#: What a run file calls each engine class.
_ENGINE_KINDS = {"PrompterEngine": "prompter",
                 "ChatModelEngine": "chat_model",
                 "ClaudeEngine": "claude"}


class _Engines:
    """Builds the engine every stage reads with (BACKLOG RI-67).

    With no chat model this is what every stage always did: a
    :class:`~report_ingest.engine.PrompterEngine` on the live Prompter and
    the tier asked for, and the run files say exactly what they said before.

    Given a chat model -- a LangChain chat model, the page's wrapper round
    one, or a callable that returns one for a model name -- each engine is
    built through :func:`report_ingest.engine.engine_for`, which makes a
    :class:`~report_ingest.engine.ChatModelEngine` of it: the engine the
    app's pages read a report on. ``chat_model`` serves the readers, the
    triage and the label review; ``vision_chat_model`` the vision pass (the
    ``vision_labels`` stage and the vote's vision voter inside the ingest),
    and it defaults to ``chat_model``. A callable is asked once per model
    name and its model kept for the whole run (the key is the source and the
    name, so a defaulted ``vision_chat_model`` -- the same object -- asks
    once, and two different sources each get asked).

    A run on a chat model says so in every run file and table: ``engine``
    (``chat_model``, or ``prompter`` for a stage left on the Prompter),
    ``model`` (the name the model gives itself) and ``chat_model`` (its
    class), so a number read through Foundry's classes is never taken for a
    Prompter number.
    """

    def __init__(self, prompter: Any = None, chat_model: Any = None,
                 vision_chat_model: Any = None) -> None:
        self.prompter = prompter
        self.sources: Dict[bool, Any] = {
            False: chat_model,
            True: (vision_chat_model if vision_chat_model is not None
                   else chat_model)}
        self._chats: Dict[Tuple[int, str], Any] = {}

    @classmethod
    def of(cls, prompter: Any) -> "_Engines":
        """The run's engines, or a live Prompter's alone (every caller that
        passes a Prompter straight in, as before RI-67)."""
        return prompter if isinstance(prompter, cls) else cls(prompter)

    @property
    def plain(self) -> bool:
        """No chat model anywhere: the Prompter run this always was."""
        return all(source is None for source in self.sources.values())

    def on_prompter(self, vision: bool = False) -> bool:
        return self.sources[vision] is None

    def _chat(self, name: str, vision: bool) -> Any:
        """The chat model for this model name, made once per run."""
        from report_ingest.engine import _chat_model_in

        source = self.sources[vision]
        key = (id(source), name)
        if key not in self._chats:
            chat = _chat_model_in(source)
            if chat is None and callable(source):
                chat = _chat_model_in(source(name))
            if chat is None:
                which = "vision_chat_model" if vision else "chat_model"
                raise ValueError(
                    f"{which} must be a LangChain chat model, or a callable "
                    f"that returns one for a model name; for {name!r} it "
                    f"gave {type(source).__name__}")
            self._chats[key] = chat
        return self._chats[key]

    def build(self, name: str, meter: Any, *, vision: bool = False) -> Any:
        """The engine for one item, on its own meter."""
        from report_ingest.engine import PrompterEngine, engine_for

        if self.sources[vision] is None:
            return PrompterEngine(self.prompter, name, meter=meter)
        return engine_for(self._chat(name, vision), meter=meter)

    def record(self, name: str, engine: Any) -> Dict[str, Any]:
        """What a run file says about the engine that read it: the tier
        alone on a Prompter run, as always; the engine, the model's own name
        and the chat model's class once a chat model is in the run."""
        if self.plain:
            return {"model": name}
        out: Dict[str, Any] = {
            "model": str(getattr(engine, "model", "") or name),
            "engine": _ENGINE_KINDS.get(type(engine).__name__,
                                        type(engine).__name__)}
        chat = getattr(engine, "chat_model", None)
        if chat is not None:
            out["chat_model"] = type(chat).__name__
        return out

    def describe(self, name: str, vision: bool = False) -> Dict[str, Any]:
        """:meth:`record` for the engine this run builds for ``name``,
        without a model call. A factory is asked for its model here, so a
        bad one fails before anything is spent."""
        from report_ingest.engine import CostMeter

        if self.plain:
            return {"model": name}
        if self.sources[vision] is None:
            return {"model": name, "engine": "prompter"}
        return self.record(name, self.build(name, CostMeter(), vision=vision))

    def stage(self, name: str, vision: bool = False
              ) -> Tuple[str, Dict[str, Any]]:
        """``(model, the other engine fields)`` for a stage's results: the
        tier and nothing more on a Prompter run."""
        fields = dict(self.describe(name, vision=vision))
        return str(fields.pop("model")), fields


def _through(results: Dict[str, Any]) -> str:
    """How a stage's calls went out, for its heading."""
    if results.get("engine") in (None, "prompter"):
        return "Prompter"
    return "a chat model"


def _engine_lines(results: Dict[str, Any], prefix: str = "",
                  label: str = "Engine") -> List[str]:
    """The line saying which engine a stage ran on; none on the Prompter,
    which is every run before RI-67 and what the headings already say.
    ``prefix`` reads a second engine's fields (``vision_``), whose model
    the stage's own model line does not name."""
    engine = results.get(f"{prefix}engine")
    if engine in (None, "prompter"):
        return []
    chat = results.get(f"{prefix}chat_model")
    model = results.get(f"{prefix}model") if prefix else None
    return [f"{label}: `{engine}`" + (f" over `{chat}`" if chat else "")
            + (f", model `{model}`" if model else "")
            + " -- the caller's own chat model (BACKLOG RI-67), not the "
              "Funhouse Prompter; the model is named as it names itself."]


# ---------------------------------------------------------------------------
# model first, as a second arm (BACKLOG RI-87)
# ---------------------------------------------------------------------------

#: The reader stages model first reads, each on its own model under ``on``
#: (``stage_models``). The soundings read as today in either arm, as they
#: do in the graph.
FIRST_STAGES: Tuple[str, ...] = ("logs", "lab", "calc", "narrative")


class _FirstArm:
    """The run's model-first arm (BACKLOG RI-87). ``off`` reads as today.
    ``on`` reads the four reader stages' items and the whole reports model
    first, its pages sent by ``page_route``. In either arm each stage reads
    on its model in ``stage_models`` (``model`` for a stage not named), so
    the two arms are compared on the same models (plan section 8.8).

    The setting is resolved once, here, and handed on, so every run file
    names the arm it ran in and a variable changed during a run cannot put
    part of it in the other arm.
    """

    def __init__(self, model_first: Any = "off",
                 stage_models: Optional[Dict[str, str]] = None,
                 page_route: Optional[str] = None) -> None:
        from report_ingest.graph import model_first_setting

        self.setting = model_first_setting(model_first)
        unknown = sorted(set(stage_models or {}) - set(FIRST_STAGES))
        if unknown:
            raise ValueError(f"stage_models names {unknown}; the stages "
                             f"model first reads are {list(FIRST_STAGES)}")
        self.stage_models = {str(k): str(v)
                             for k, v in (stage_models or {}).items()}
        self.page_route = page_route
        if self.on:
            from report_ingest.model_first_pages import page_route as route
            route(page_route)               # a typo is refused before a call

    @property
    def on(self) -> bool:
        return self.setting == "on"

    def model(self, stage: str, model: str) -> str:
        """The model ``stage`` reads on, in either arm."""
        return self.stage_models.get(stage, model)

    def describe(self, model: str) -> str:
        """One line for the run's opening print."""
        models = "; " + ", ".join(f"{stage} on {self.model(stage, model)}"
                                  for stage in FIRST_STAGES)
        if not self.on:
            return ("model first: off (today's readers)"
                    + (models if self.stage_models else ""))
        return ("model first: on (BACKLOG RI-87); pages "
                + (self.page_route or "as GEOTECH_INGEST_MODEL_FIRST_PAGES "
                                      "says (auto when unset)")
                + models)


def _stamp_arm(folder: Path, first: _FirstArm) -> None:
    """Write ``arm.json`` into a report's output folder, once, when the
    folder is made (BACKLOG RI-120). It is what keeps an interrupted folder
    -- one with items but no ``run.json`` yet -- from being taken over by
    the other arm (:func:`_arm_clash`). A folder made before the stamp
    existed, with items but no ``run.json`` and no ``arm.json``, is
    stamped with the arm of the run that finds it; nothing says which arm
    began it."""
    stamp = folder / "arm.json"
    if stamp.is_file():
        return
    stamp.write_text(json.dumps({"model_first": first.setting}),
                     encoding="utf-8")


def _tokens_of(cost: Dict[str, Any]) -> int:
    """Tokens in (plain, cache write and cache read, BACKLOG RI-171) plus
    output tokens of a run file's ``cost``."""
    return tokens_in(cost) + _n(cost.get("output_tokens"))


#: What an earlier attempt's entry in a report's ``run.json`` keeps of the
#: cost block that attempt's run file held (BACKLOG RI-124): its totals. The
#: row per request (``per_call``) and the per-model split are not carried.
#: A key the block does not hold is left out, so ``paced_seconds`` (BACKLOG
#: RI-127, RI-139; only in a block a pacer held a request in) is carried
#: only where there was a hold.
_ATTEMPT_COST_KEYS: Tuple[str, ...] = (
    "calls", "failed", "retried", "failed_by_kind", "input_tokens",
    "output_tokens", "reasoning_tokens", "reasoning_reported",
    "cache_read_tokens", "cache_write_tokens", "seconds", "waited_seconds",
    "paced_seconds", "dollars", "unpriced_calls")

#: The counts an ``all_attempts`` block adds up over a report's attempts.
_ATTEMPT_SUMS: Tuple[str, ...] = (
    "calls", "failed", "retried", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
    "reasoning_reported", "unpriced_calls")


def _attempt_entry(blob: Dict[str, Any]) -> Dict[str, Any]:
    """One earlier attempt at a report as its ``run.json`` carries it, read
    off the run file that attempt wrote (BACKLOG RI-124): the date, how it
    ended (``error``, ``token_brake``, ``item_errors``, or ``finished``),
    its wall clock and its cost totals. A failed attempt's file keeps no
    wall clock, so its calls' own seconds stand in."""
    cost = blob.get("cost") if isinstance(blob.get("cost"), dict) else {}
    if blob.get("error"):
        ended = "error"
    elif blob.get("token_brake"):
        ended = "token_brake"
    elif blob.get("item_errors"):
        ended = "item_errors"
    else:
        ended = "finished"
    seconds = blob.get("seconds", cost.get("seconds"))
    try:
        seconds = round(float(seconds or 0.0), 1)
    except (TypeError, ValueError):
        seconds = 0.0
    return {"run_date": blob.get("run_date"), "ended": ended,
            "seconds": seconds,
            "cost": {key: cost[key] for key in _ATTEMPT_COST_KEYS
                     if key in cost}}


def _attempt_rows(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The ``earlier_attempts`` a run file carries; none for one written
    before BACKLOG RI-124 or by a report's first attempt."""
    rows = blob.get("earlier_attempts")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _earlier_attempts(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every attempt before the one about to replace ``blob``'s run file:
    the earlier attempts that file carries, then its own (BACKLOG RI-124)."""
    return _attempt_rows(blob) + [_attempt_entry(blob)]


def _attempts_in(earlier: Sequence[Dict[str, Any]]) -> int:
    """The tokens in (plain, cache write and cache read, BACKLOG RI-171)
    the earlier attempts at a report sent."""
    total = 0
    for row in earlier:
        cost = row.get("cost")
        if isinstance(cost, dict):
            total += tokens_in(cost)
    return total


def _float(value: Any) -> float:
    """A number of seconds or dollars read off a run file, 0.0 where it is
    not one."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _attempts_fields(earlier: Sequence[Dict[str, Any]],
                     cost: Dict[str, Any], seconds: Any) -> Dict[str, Any]:
    """A run file's ``earlier_attempts`` and ``all_attempts`` (this attempt's
    ``cost`` and ``seconds`` added to the earlier ones'); nothing when this
    is the report's first attempt, so that run file is written as before
    (BACKLOG RI-124)."""
    if not earlier:
        return {}
    costs = [row.get("cost") if isinstance(row.get("cost"), dict) else {}
             for row in earlier] + [cost if isinstance(cost, dict) else {}]
    total: Dict[str, Any] = {"attempts": len(costs)}
    for key in _ATTEMPT_SUMS:
        total[key] = sum(_n(c.get(key)) for c in costs)
    kinds: Dict[str, int] = {}
    for c in costs:
        by_kind = c.get("failed_by_kind")
        for kind, n in (by_kind.items() if isinstance(by_kind, dict) else ()):
            kinds[kind] = kinds.get(kind, 0) + _n(n)
    total["failed_by_kind"] = kinds
    total["dollars"] = round(sum(_float(c.get("dollars")) for c in costs), 4)
    total["waited_seconds"] = round(
        sum(_float(c.get("waited_seconds")) for c in costs), 1)
    # Only when a pacer held a request in some attempt (BACKLOG RI-139), as
    # the cost block itself carries it.
    paced = round(sum(_float(c.get("paced_seconds")) for c in costs), 1)
    if paced:
        total["paced_seconds"] = paced
    total["seconds"] = round(sum(_float(row.get("seconds"))
                                 for row in earlier) + _float(seconds), 1)
    return {"earlier_attempts": [dict(row) for row in earlier],
            "all_attempts": total}


class _TokenBrake:
    """A token limit on one report's model calls, for a route whose dollar
    meter reads 0 (BACKLOG RI-120).

    :meth:`wrap` puts an engine behind it: a call that would start once the
    report's meter shows ``report_in`` tokens in, or ``total_left`` tokens
    (in plus output) spent, is refused with
    :class:`~report_ingest.engine.CallCeilingReached` before anything is
    sent. The graph already treats that as a step cut short, not failed.
    Both None wraps nothing. Tokens in are plain input plus cache writes
    plus cache reads (BACKLOG RI-171): on Foundry's Claude route nearly all
    input arrives as the two cache counts.

    ``earlier_in`` is what the report's earlier attempts sent (BACKLOG
    RI-124): the report limit counts it with this attempt's meter, so a
    report retried after the brake, an error or a give-up does not start
    on a fresh allowance. ``total_left`` is the run's, and counts this
    call's spending only, as before."""

    def __init__(self, meter: Any, report_in: Optional[int],
                 total_left: Optional[int], earlier_in: int = 0) -> None:
        self.meter = meter
        self.report_in = report_in
        self.total_left = total_left
        self.earlier_in = max(0, int(earlier_in or 0))
        self.tripped: Optional[str] = None

    def wrap(self, engine: Any) -> Any:
        if engine is None or (self.report_in is None
                              and self.total_left is None):
            return engine
        return _BrakedEngine(engine, self)

    def wrap_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """``ingest_report``'s arm arguments with each engine behind the
        brake; the caller's copy is untouched."""
        out = dict(kwargs)
        for key in ("describe_engine", "range_engine"):
            if key in out:
                out[key] = self.wrap(out[key])
        if "stage_engines" in out:
            out["stage_engines"] = {k: self.wrap(v) for k, v
                                    in out["stage_engines"].items()}
        return out

    def check(self) -> None:
        used_in = tokens_in(self.meter)
        report_used = used_in + self.earlier_in
        if self.report_in is not None and report_used >= self.report_in:
            self.tripped = "max_report_tokens_in"
            raise CallCeilingReached(
                f"the report has sent {report_used:,} tokens in"
                + (" over this and its earlier attempts" if self.earlier_in
                   else "")
                + f", at its {self.report_in:,} ceiling")
        if self.total_left is not None and (
                used_in + int(self.meter.output_tokens) >= self.total_left):
            self.tripped = "max_total_tokens"
            raise CallCeilingReached(
                "the run's token ceiling (max_total_tokens) is reached")

    def fields(self) -> Dict[str, Any]:
        """The run file's entry; empty unless a call was refused. Its
        counts are this attempt's: ``tokens_in`` (what the limits count)
        and its three parts, and ``output_tokens`` (BACKLOG RI-171); on a
        report with earlier attempts ``earlier_tokens_in`` says what they
        sent, which the report limit counted too (BACKLOG RI-124)."""
        if self.tripped is None:
            return {}
        entry: Dict[str, Any] = {"limit": self.tripped,
                                 "tokens_in": tokens_in(self.meter)}
        for key in ("input_tokens", "cache_write_tokens",
                    "cache_read_tokens", "output_tokens"):
            entry[key] = _n(getattr(self.meter, key, 0))
        if self.earlier_in:
            entry["earlier_tokens_in"] = self.earlier_in
        return {"token_brake": entry}


class _BrakedEngine:
    """An engine whose calls the brake may refuse; everything else is the
    engine it wraps."""

    def __init__(self, inner: Any, brake: _TokenBrake) -> None:
        self._inner = inner
        self._brake = brake

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_inner", "_brake"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def complete(self, messages: Any, **kwargs: Any) -> Any:
        self._brake.check()
        return self._inner.complete(messages, **kwargs)


def _arm_clash(run_file: Path, first: _FirstArm) -> Optional[str]:
    """Why this run may neither use nor replace ``run_file``: it was read
    in the other model-first arm. None when there is no such file. A run
    file written before the arm was recorded was read with it off. The
    folder's ``arm.json`` (:func:`_stamp_arm`) is checked first, so a folder
    with no run file yet, one a run was interrupted in, is held to its arm
    too."""
    stamp = run_file.parent / "arm.json"
    if stamp.is_file():
        try:
            saved = str(json.loads(stamp.read_text(encoding="utf-8")
                                   ).get("model_first") or "off")
        except (json.JSONDecodeError, OSError, AttributeError):
            saved = first.setting
        if saved != first.setting:
            return (f"its folder was started with model first {saved} and "
                    f"this run is {first.setting}; one out_dir holds one "
                    f"arm, so start a fresh out_dir for the other")
    if not run_file.is_file():
        return None
    try:
        blob = json.loads(run_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    saved = str(blob.get("model_first") or "off")
    if saved == first.setting:
        return None
    return (f"its run file was read with model first {saved} and this run "
            f"is {first.setting}; one out_dir holds one arm, so start a "
            f"fresh out_dir for the other")


def _first_results(rows: Iterable[dict]) -> Dict[str, Any]:
    """What a stage's results say about the arm its run files ran in:
    ``model_first`` (the arms seen, ``off`` for a file that does not say)
    and, over the files read model first, the page routes, the instruction
    versions and the placement counts summed (plan section 8.4's gate)."""
    rows = list(rows)
    out: Dict[str, Any] = {"model_first": sorted(
        {str(r.get("model_first") or "off") for r in rows})}
    on = [r for r in rows if r.get("model_first") == "on"]
    if not on:
        return out
    routes: Set[str] = set()
    for row in on:
        if row.get("route"):
            routes.add(str(row["route"]))
        routes.update(str(v) for v in (row.get("routes") or {}).values())
    placed: Dict[str, Any] = {"values": 0, "with_text": 0, "at_a_spot": 0,
                              "page_level": 0, "counts": {}}
    for row in on:
        got = row.get("placement") or {}
        for key in ("values", "with_text", "at_a_spot", "page_level"):
            placed[key] += int(got.get(key) or 0)
        for status, n in (got.get("counts") or {}).items():
            placed["counts"][status] = (placed["counts"].get(status, 0)
                                        + int(n or 0))
    out.update(routes=sorted(routes), placement=placed,
               prompt_versions=sorted({str(r["prompt_version"]) for r in on
                                       if r.get("prompt_version")}))
    return out


def _first_lines(results: Dict[str, Any]) -> List[str]:
    """The lines naming a stage's model-first arm; none for a stage with
    no run files."""
    arms = list(results.get("model_first") or ())
    if not arms:
        return []
    line = ("Model first (RI-87, GEOTECH_INGEST_MODEL_FIRST): "
            + " and ".join(f"`{arm}`" for arm in arms))
    if len(arms) > 1:
        line += " -- MIXED: these run files were not all read in one arm"
    if "on" not in arms:
        return [line + "."]
    out = [line + f"; pages sent as "
           f"{', '.join(results.get('routes') or ['?'])}; instructions "
           f"{', '.join(results.get('prompt_versions') or ['?'])}. Under "
           f"`on` there is no floor and no merge: the record is the "
           f"model's answer, built with checks='flag', so `model` is "
           f"`after`; the lab and calculation switches named here act on "
           f"today's readers only."]
    placed = results.get("placement") or {}
    if placed.get("values"):
        counts = placed.get("counts") or {}
        share = (f" ({100.0 * placed['at_a_spot'] / placed['with_text']:.1f}"
                 f" %)" if placed.get("with_text") else "")
        out.append(f"Placed at a spot (plan section 8.4, gate 90 %): "
                   f"{placed['at_a_spot']} of {placed['with_text']} values "
                   f"on pages with text{share}; {placed['page_level']} on "
                   f"the page only; {counts.get('not_found', 0)} not found "
                   f"printed on the cited page; {counts.get('no_text', 0)} "
                   f"on pages with no text.")
    return out


def _run_one(rid: str, corpus: Corpus, prompter: Any, model: str,
             triage_model: str, out_dir: Path) -> dict:
    """Rules, triage and review on one report. Writes its own run file.

    ``prompter`` is the live Prompter, or the run's :class:`_Engines`.
    """
    from planlens.document.roles import document_outline, page_roles

    from report_ingest.engine import CostMeter
    from report_ingest.label_review import review_labels
    from report_ingest.triage import document_facts, triage

    engines = _Engines.of(prompter)
    started = time.time()
    meter = CostMeter()
    doc = corpus.open_report(rid, di="auto", warn=False)
    try:
        roles = page_roles(doc)
        outline = document_outline(doc)
        facts = document_facts(doc, roles)
        triage_engine = engines.build(triage_model, meter)
        profile = triage(doc, roles, outline, engine=triage_engine,
                         facts=facts)
        review_engine = engines.build(model, meter)
        review = review_labels(doc, roles, outline, profile,
                               engine=review_engine)
        rules = {r.page: r.role for r in roles}
        # The rule's own confidence, saved beside the label so the vote
        # stage can weigh a voter without reopening every PDF.
        rules_confidence = {r.page: round(float(r.confidence), 3)
                            for r in roles}
        n_pages = doc.n_pages
        served_by = sorted({x for x in (triage_engine.served_by,
                                        review_engine.served_by) if x})
    finally:
        doc.close()

    review_fields = engines.record(model, review_engine)
    blob = {
        "id": rid,
        "run_date": date.today().isoformat(),
        "n_pages": n_pages,
        "engine": review_fields.get("engine", "prompter"),
        "triage_model": engines.record(triage_model, triage_engine)["model"],
        "review_model": review_fields["model"],
        **({"chat_model": review_fields["chat_model"]}
           if "chat_model" in review_fields else {}),
        "served_by": served_by,
        "rules_labels": {str(k): v for k, v in sorted(rules.items())},
        "rules_confidence": {str(k): v
                             for k, v in sorted(rules_confidence.items())},
        "profile": profile.to_dict(),
        "review": review.to_dict(),
        "cost": meter.to_dict(),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "runs" / f"{rid}.json").write_text(
        _dumps(blob), encoding="utf-8")
    (out_dir / "triage" / f"{rid}.json").write_text(
        json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return blob


def score_on_cluster(reports_dir: Any = None, labels_xlsx: Any = None,
                     di_dir: Any = None, out_dir: Any = "/tmp/report_ingest",
                     prompter: Any = None, model: str = "funhouse-gpt-high",
                     sets: Sequence[str] = SET_NAMES, *,
                     manifest: Any = None,
                     triage_model: Optional[str] = None,
                     oos_labels: Any = None,
                     max_total_dollars: Optional[float] = None,
                     max_total_tokens: Optional[int] = None,
                     max_report_tokens_in: Optional[int] = None,
                     max_request_tokens_in: Optional[int] = None,
                     lab_sheets_per_call: Optional[int] = None,
                     tokens_per_minute: Optional[int] = None,
                     max_reports: Optional[int] = None,
                     redo: bool = False,
                     corpus_dir: Any = None,
                     stages: Sequence[str] = ("labels",),
                     truth_dir: Any = None,
                     lab_truth_dir: Any = None,
                     calc_truth_dir: Any = None,
                     sounding_truth_dir: Any = None,
                     narrative_truth_dir: Any = None,
                     log_budget: int = 6,
                     lab_budget: int = 4,
                     calc_budget: int = 2,
                     sounding_budget: int = 2,
                     narrative_budget: int = 8,
                     open_reports: Optional[Sequence[str]] = None,
                     open_lab_reports: Optional[Sequence[str]] = None,
                     open_calc_reports: Optional[Sequence[str]] = None,
                     open_sounding_reports: Optional[Sequence[str]] = None,
                     open_narrative_reports: Optional[Sequence[str]] = None,
                     vision_model: str = "funhouse-gpt-low",
                     vision_mode: str = "page",
                     vision_dpi: Optional[float] = None,
                     vision_outline_context: bool = False,
                     vision_detail: Optional[str] = None,
                     vision_window: Optional[int] = None,
                     vision_images_per_call: Optional[int] = None,
                     vision_fallback: bool = True,
                     sharepoint: Any = None,
                     sharepoint_folder: str = DEFAULT_FOLDER,
                     durable_dir: Any = None,
                     vision_dirs: Optional[Sequence[Any]] = None,
                     review_dir: Any = None,
                     templates_path: Any = None,
                     narrative_front_pages: int = 50,
                     label_policy: str = "structural",
                     review_mode: str = "disagreements",
                     ingest_vision_mode: str = "sheet",
                     ingest_bound: bool = True,
                     ingest_counts_from_logs: str = "off",
                     ingest_narrative_skips_bound: bool = False,
                     trust_table: Any = None,
                     chat_model: Any = None,
                     vision_chat_model: Any = None,
                     describe_model: Optional[str] = None,
                     range_model: Optional[str] = None,
                     range_arms: Sequence[str] = ("plain", "hinted"),
                     range_workers: int = 1,
                     model_first: Any = None,
                     stage_models: Optional[Dict[str, str]] = None,
                     page_route: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Run the rules, triage and the label review over the corpus, and score.

    Parameters
    ----------
    reports_dir
        The folder holding the report PDFs, named either ``R01.pdf`` ... or
        as their authors named them. In the second case the manifest is what
        says which file is which, so it is required. (``corpus_dir`` is the
        older name for this argument and still works.)
    manifest
        The corpus manifest (``MANIFEST.md``). Defaults to one sitting beside
        the reports; pass it when the reports folder is not yours to write
        to, which on the cluster it is not.
    labels_xlsx, di_dir
        The hand-label spreadsheet and the Azure Document Intelligence
        results (``<ID>.json.gz`` or ``DI_data_<original stem>.json``).
        Without the spreadsheet the run measures but cannot score the
        in-sample set; without the DI results the scanned pages are read
        from their own text layer alone. ``di_dir=None`` is NOT "no DI": it
        falls back to a ``di/`` folder beside or inside ``reports_dir``.
        Pass ``di_dir=False`` to withhold DI text altogether (no fallback);
        ``results.json`` then carries ``di_withheld: true`` and RESULTS
        says so.
    out_dir
        Where everything is written. ``/tmp/...`` or a Volume, never
        ``/Workspace``.
    prompter
        The live ``fh_prompter``. Required by every stage that calls a
        model, which is every stage but ``vote``: the vote combines label
        runs already on disk and makes no call, so it runs without one
        (BACKLOG RI-08). A stage that reads on ``chat_model`` needs none
        either (BACKLOG RI-67).
    model, triage_model
        Funhouse tiers. ``triage_model`` defaults to ``model``; triage is one
        call over a ledger and a cheaper tier is usually enough, the review
        is the one that needs the reasoning.
    sets
        Which of ``insample``, ``oos_open``, ``oos_blind`` to run. Together
        they are all 38 reports, so the default gives every report a triage
        profile.
    oos_labels
        The lead's out-of-sample labels (``labels.json``). Without it the two
        out-of-sample sets run and produce profiles but score nothing.
    max_total_dollars
        Stop before starting another report once the run has spent this
        much. Funhouse publishes no per-token price for a tier, so this only
        bites if a price is known; ``max_reports`` is the reliable cap.
    max_total_tokens
        The same stop in tokens (in plus output), for a route whose
        dollar meter reads 0 (Foundry, BACKLOG RI-120). Checked before each
        report, and inside the ``ingest`` stage before each model call.
        None, the default, sets no limit. Tokens IN, here and in every
        limit below, are plain input plus cache writes plus cache reads
        (BACKLOG RI-171): on Foundry's Claude route nearly all input
        arrives as the two cache counts.
    max_report_tokens_in
        The ``ingest`` stage's limit on one report's tokens in. A model
        call that would start past it is refused (``CallCeilingReached``,
        so the step it belonged to is cut short and not saved), and the
        run stops after that report. It counts EVERY attempt at the report
        (BACKLOG RI-124): one read again after the brake, an error or a
        give-up starts with its earlier attempts' input counted, and one
        whose earlier attempts already reached the limit is not read again
        until the limit is raised. None, the default, sets no limit.
    max_request_tokens_in
        A limit on ONE request's input tokens in the ``ingest`` stage's log
        reader (BACKLOG RI-110): before a request nears it, earlier page
        pictures are dropped from the conversation, then its middle turns,
        and a request carrying too many pages is split across calls. Sized
        by estimate (a picture counts 1,700 tokens). Model first (BACKLOG
        RI-121), each page group's request is estimated before it is sent
        (a picture by its pixels, a PDF page as its text and one picture of
        up to 4,784 tokens, the answer schema) and a group over the limit
        is cut into smaller groups. None, the default, changes nothing.
    lab_sheets_per_call
        The ``ingest`` stage's lab reader reads an item of more than this
        many pages in calls of at most this many (BACKLOG RI-116), each told
        the run it belongs to, as a long run's windows are; the item file
        records the number and the windows. None, the default, changes
        nothing.
    tokens_per_minute
        The ``ingest`` stage's per-minute input-token budget for each model
        (BACKLOG RI-127; ``Budgets.tokens_per_minute``). After a request
        estimated above half of it comes back, the next request to that
        model waits out the minute, across reports; a rate limit with no
        retry-after holds the next request a whole minute. Every request's
        row in the cost record says how long it was held (``paced``); the
        holds add up in each cost block's ``paced_seconds``, in a run file's
        ``earlier_attempts`` / ``all_attempts``, in ``results.json``'s cost
        totals and on RESULTS' failed-requests line (BACKLOG RI-139), each
        only where a request was held. None, the default, paces nothing.
    redo
        Re-run reports that already have a run file. Off by default, which is
        what makes a detached notebook cheap to resume.
    stages
        Which measurements to make. ``("labels",)`` is WP1b -- triage and the
        label review over whole reports. ``"logs"`` is WP2b -- ``log_grid``
        and then the log reader over each hand-truthed log, scored before
        (the grid alone) and after (the record the reader built). ``"lab"``
        is WP3 -- the page's own detected tables and then the lab reader over
        each hand-truthed laboratory sheet, scored the same two ways.
        ``"soundings"`` is WP6 -- the log grid on each hand-truthed TEST
        PIT and the sounding floor on each hand-truthed CONE SOUNDING and
        DYNAMIC PROBE, then the reader over the same pages, scored the same
        two ways. Its numbers split on one fact and the scorecard prints
        the split: a TABULATED sheet's floor already holds the series, and
        a PLOTTED one's holds nothing but the axis ranges, so the reader's
        job on the second is the whole of the digitising.
        ``"narrative"`` is WP4 -- the narrative reader over every report that
        has a hand answer, scored field by field on recall, precision and the
        flattering agreement. ``"vision_labels"`` is the WP5 experiment --
        every page's PICTURE to a cheap model, scored against the SAME hand
        labels with the SAME scorer as the rules, so RESULTS.md can put the
        rules, the rules plus the review and the picture side by side.
        ``"vote"`` is WP6, and it calls NO model: it recomputes the rules
        with planlens, reads the vision labels off the saved runs and the
        review's labels off the label runs, and asks what their disagreement
        is worth -- how often they agree, which of them to believe per label
        class, what three combining policies score, and how good a targeted
        review of the splits would have to be. Pass any combination; all six
        is ``stages=("labels", "logs", "lab", "narrative", "vision_labels",
        "vote")``, though the vote is usually run on its own over run files
        that already exist. ``"range_labels"`` (BACKLOG RI-73) describes
        every page of every report in the chosen sets with a vision model
        and labels each report by sections from the descriptions, in each
        of ``range_arms``, scored against the hand labels by strict accuracy
        and by HARMFUL pages (sent to the wrong reader or none); see
        ``describe_model`` below. ``"ingest"`` (5.23.0) is the whole pipeline on
        each report in the chosen sets: ``graph.ingest_report`` end to end,
        into ``out_dir/ingest/<ID>/`` -- the record, the summary, the
        library page, the DIGGS file with both gates, ``qa.json`` -- reusing
        the label review and the triage a ``labels`` run left in ``runs/``
        (or in ``review_dir``) rather than paying for them again, resumable
        per report and per work item, and scored against whatever hand
        truth ``truth_dir`` holds for that report, so the stage doubles as
        the whole-pipeline score. Its page labels are a VOTE: see
        ``label_policy`` and ``review_mode`` below, and the
        "The page labels" section of its RESULTS block, which prints how
        many pages the voters split on and what the final labels scored
        beside the rules alone.
    truth_dir
        ONE truth root for every scoring stage: a folder holding ``logs/``,
        ``lab/`` and ``narrative/``, named after the stages that read them.
        The hand truth is private, so it is uploaded by hand before a run and
        does not live in the wheel; one folder to upload is one thing to get
        right rather than three. A ``truth_dir`` holding the log truth files
        directly, with no ``logs/`` in it, still works as it always did.
    calc_truth_dir
        The folder of hand-truthed CALCULATIONS
        (``<kind>__<ID>_p<first page>.json``), if not the ``calc/`` folder
        inside ``truth_dir``.
    sounding_truth_dir
        The folder of hand-truthed TEST PITS, CONE SOUNDINGS and DYNAMIC
        PROBES (``<kind>__<ID>_p<page>.json``, the kind being ``test_pit``,
        ``cpt`` or ``dcp``), if not the ``soundings/`` folder inside
        ``truth_dir``.
    lab_truth_dir, narrative_truth_dir
        The older per-stage form, and it still overrides the root. Pass them
        when the three sets of truth are not in one place.

        In every case an ``OPEN.txt`` beside the truth files names the
        reports whose pages the prompts were allowed to be tuned against;
        everything else is scored as blind.
    log_budget, lab_budget, calc_budget, sounding_budget
        Model calls a reader may spend per log and per sheet. Each reader's
        own ceiling -- six, four and two -- holds whatever these say. The
        sounding readers' two are spent as one on a TABULATED sheet (the
        table is the record and the call goes on the header) and both on a
        PLOTTED one (one per plot panel group, one for the header and the
        unsettled list).
    open_reports, open_lab_reports
        Override an open set instead of reading its ``OPEN.txt``.
    vision_model
        The tier the ``vision_labels`` experiment runs on. It defaults to
        ``funhouse-gpt-low`` on purpose: the question is whether the CHEAPEST
        model, looking at a page, can do what the rules and the review do by
        reading it. Scoring it on the tier the readers use would answer a
        different question.
    vision_mode
        ``"page"`` for one call per page, ``"sheet"`` for one call per
        contact sheet of six pages, ``"document"`` for a window of full-size
        pages with the whole report in thumbnail beside them. The three are
        the trade being priced: page mode gives every page the model's full
        attention and nothing else, sheet mode a sixth of the calls and a
        thumbnail a page, document mode the whole report in view at about a
        thirtieth of page mode's calls.
    vision_dpi
        What a page is rendered at in page and document mode. ``None`` takes
        :data:`report_ingest.vision_labels.DEFAULT_DPI`, which is the largest
        render a 4.1-class vision stack keeps.
    vision_outline_context
        Give the model what the document prints about ITSELF -- the contents
        list, the lists of figures, tables and appendices, the dividers --
        on every call, so a run can put pure vision beside vision that knows
        which appendix it is standing in.
    vision_detail
        ``"low"``, ``"high"`` or ``"auto"`` for OpenAI's ``image_url.detail``
        on every picture; ``None`` leaves it unset and the provider's own
        default applies. ``"low"`` is about 85 tokens an image rather than a
        page's four tiles, which is what makes a whole-corpus document-mode
        run affordable.
    vision_window, vision_images_per_call
        Document mode only: full-size pages in one call, and the endpoint's
        ceiling on images in one request. ``None`` takes
        :data:`report_ingest.vision_labels.DOCUMENT_WINDOW` (36) and
        :data:`report_ingest.vision_labels.MAX_IMAGES_PER_CALL` (50, measured
        on the owner's cluster on 2026-09-18). Raise the second only for an
        endpoint that has been shown to accept more. A window the gateway
        refuses as too large a REQUEST BODY -- which a report of scanned
        pages reaches well under 50 images -- halves itself and narrows the
        rest of the report; the ``split`` column says how often.
    vision_fallback
        After a sheet or document pass, give every page still unresolved ONE
        page-mode call. On by default: a page goes unresolved because the
        reply left it out, and asking about that page alone is the mode that
        cannot skip it. The vision budget still applies.
    sharepoint, sharepoint_folder, durable_dir
        WHERE THE OUTPUT ALSO GOES, so a cluster restart does not charge the
        run twice. ``sharepoint`` is the live ``fh_sp_client`` (its
        ``.file_manager`` is taken), a file manager, or the app's
        ``SharePointStore``; ``durable_dir`` is any folder that survives the
        driver. Either, or both. The run's folder is
        ``<sharepoint_folder>/<the out_dir's own name>``, so an ``out_dir``
        of ``/tmp/521_sheet`` mirrors to
        ``GeotechStaffEngineer/report_ingest/521_sheet``. Every run file is
        copied as soon as it is written, and anything the mirror holds that
        ``out_dir`` does not is copied back at the START. ``out_dir`` still
        refuses ``/Workspace``; ``durable_dir`` may point anywhere, because
        whether a path is durable is the owner's finding, not this module's.
    vision_dirs, review_dir
        The ``vote`` stage's inputs. ``vision_dirs`` is one or more folders
        of saved vision runs (``<ID>.json``), defaulting to this run's own
        ``out_dir/vision``; pass the folder a previous mode's run left
        behind to vote on that one. ``review_dir`` is the folder of label
        runs (``<ID>.json`` holding ``review.final_labels``), defaulting to
        ``out_dir/runs``; the review is a third voter only where it is
        actually there.
    label_policy, review_mode, ingest_vision_mode, trust_table
        The ``ingest`` stage's own page vote, and these are the parameters a
        production run takes. ``label_policy`` combines the rules, the
        vision pass and the printed form -- ``structural`` by default,
        ``rules`` to reproduce what the pipeline did before the vote.
        ``review_mode`` decides which pages the label review is given:
        ``disagreements`` (the default: only the pages the voters split on,
        plus two either side), ``all``, or ``none``.
        ``ingest_vision_mode`` is the mode that pass runs in inside the
        ingest -- ``sheet``, which is the one the corpus was measured in and
        costs about $0.05 a report -- kept apart from ``vision_mode`` because
        the ``vision_labels`` stage is pricing the three modes against each
        other and the ingest is not. The tier is ``vision_model``.
        ``trust_table`` is the path the ``vote`` stage writes to
        (``<run>/vote/trust_table.json``), needed only by
        ``label_policy="trust"``; without it that policy falls back to
        ``structural`` and prints a note.
    ingest_bound
        Read a report bound INSIDE a report as its own record, under
        ``ingest/<ID>/bound/<bound id>/``, and list it on the parent's
        ``bound_documents``. On by default. What a bound document holds is
        then in none of its parent's counts, which is the point: ledger run
        7 has 19 of 38 corpus reports carrying one, and their borings belong
        to the earlier investigation. ``False`` lists the pages and does not
        read them, which is what every run before this did.
    ingest_counts_from_logs, ingest_narrative_skips_bound
        ``graph.ingest_report``'s ``counts_from_logs`` and
        ``narrative_skips_bound`` (BACKLOG RI-14), both off by default as
        there: this stage is where they are measured. Each report's
        ``run.json`` records them, and a report already done under other
        settings is NOT redone unless ``redo`` -- start a fresh ``out_dir``
        to compare two settings.
    describe_model, range_model, range_arms, range_workers
        The ``range_labels`` stage (BACKLOG RI-73). ``describe_model``
        describes the pages, one vision call a page, on the vision engine
        (``vision_chat_model`` when one is given); it defaults to
        ``vision_model``. ``range_model`` labels each report by sections
        from the descriptions, on the readers' engine; it defaults to
        ``model``. ``range_arms`` is which of ``plain`` (descriptions and
        the printed outline) and ``hinted`` (the rules' label on every
        line as well) to run; both by default, over the same descriptions.
        ``range_workers`` is how many pages are described at once.
        Descriptions are saved one file a page under
        ``out_dir/ranges/descriptions/<ID>/`` and each arm's labels to
        ``out_dir/ranges/<arm>/<ID>.json``.
    model_first, stage_models, page_route
        Model-first ingest as a second arm on the same items (BACKLOG
        RI-87; plan section 8.8). ``off`` reads as today. ``on`` reads each
        hand-truthed item of the ``logs``, ``lab``, ``calc`` and
        ``narrative`` stages on its own pages with the model-first reader
        of its stage (:mod:`report_ingest.model_first_scoring`), and each
        whole report of the ``ingest`` stage with
        ``graph.ingest_report(model_first="on")``: every page described on
        ``describe_model`` and labelled by sections on ``range_model``, in
        place of the vision voter and the label review. Both arms are
        scored by the same scorers. ``None`` follows
        ``GEOTECH_INGEST_MODEL_FIRST``; the setting is resolved once and
        handed on. ``stage_models`` maps each of those four stages to the
        model it reads on, in either arm, so the two are compared on the
        same models (the name the Prompter or ``chat_model``'s factory
        takes; a stage not named reads on ``model``). Off, a named stage is
        today's reader on that model; in the ``ingest`` stage the triage,
        the vision voter and the label review stay on their own models.
        ``page_route`` is ``auto``, ``pdf`` or ``pictures`` (None:
        ``GEOTECH_INGEST_MODEL_FIRST_PAGES``). The ``soundings`` stage
        reads as today in either arm, as the graph does. Every run file
        names its arm (``model_first``) and, under ``on``, the route, the
        instruction version and what placement found; RESULTS.md says the
        same per stage. ONE ``out_dir`` HOLDS ONE ARM: an item whose run
        file was read in the other arm is named among the failures and
        left as it is, with or without ``redo``.
    chat_model, vision_chat_model
        A LangChain chat model to read with INSTEAD OF the Prompter (BACKLOG
        RI-67), or a callable that returns one for a model name. Each
        engine is then built through
        :func:`report_ingest.engine.engine_for` -- a
        :class:`~report_ingest.engine.ChatModelEngine` over the model, as
        the app's pages build theirs. ``chat_model`` serves the readers, the
        triage and the label review; ``vision_chat_model`` the vision pass
        (the ``vision_labels`` stage and the vote's vision voter in the
        ingest), and it defaults to ``chat_model``. A callable is asked
        with ``model``, ``triage_model`` or ``vision_model`` and its model is
        kept for the run; a single chat model is used whatever those say.
        Neither is needed and nothing changes without them: every stage
        then runs on the Prompter exactly as before, and its run files and
        tables are what they were. With one, ``prompter`` is needed only by
        a stage left on it. Every run file and table of a chat-model run
        names its ``engine``, the ``model`` as the model names itself and
        the ``chat_model`` class. On Foundry, Claude through the
        workspace's own model classes::

            from webapp.palantir_sdk_engine import PalantirClaudeChatModel

            claude = PalantirClaudeChatModel(
                model_api_name="AnthropicClaude_5_5_Opus", max_tokens=32000)
            score_on_cluster(..., chat_model=claude,
                             stages=("labels", "vision_labels"))

        or a callable over the app's own resolver, so ``model`` and
        ``vision_model`` pick the model by its Foundry API name (with
        ``GEOTECH_FOUNDRY_CLAUDE_SDK=1`` a Claude name takes the
        ``palantir_models`` class, as on the page)::

            from webapp.engine_config import resolve_engine

            def foundry_model(name):
                found = resolve_engine(name)
                if found.model is None:
                    raise RuntimeError(found.message)
                return found.model

            score_on_cluster(..., chat_model=foundry_model,
                             model="AnthropicClaude_5_5_Opus",
                             vision_model="AnthropicClaude_5_5_Opus")
    """
    reports_dir = reports_dir if reports_dir is not None else corpus_dir
    if reports_dir is None:
        raise ValueError("pass reports_dir: the folder holding the PDFs")
    stages = tuple(stages)
    unknown = [s for s in stages if s not in STAGE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown stage(s) {unknown}; the stages are {list(STAGE_NAMES)}")
    if not stages:
        raise ValueError(f"pass at least one stage: {list(STAGE_NAMES)}")
    calling = [s for s in stages if s not in NO_MODEL_STAGES]
    engines = _Engines(prompter, chat_model, vision_chat_model)
    first = _FirstArm(model_first, stage_models, page_route)
    # A stage left on the Prompter needs it; one on a chat model does not.
    on_prompter = [s for s in calling
                   if (s != "vision_labels" and engines.on_prompter())
                   or (s in ("vision_labels", "ingest", "range_labels")
                       and engines.on_prompter(vision=True))]
    describe_model = describe_model or vision_model
    range_model = range_model or model
    if prompter is None and on_prompter:
        raise ValueError(
            f"pass the live fh_prompter: stage(s) {on_prompter} call the "
            f"model, and this module reads no credential of its own"
            + ("" if engines.plain else
               " (or pass a chat model for them as well)"))
    engine_notes: List[str] = []
    if not engines.plain:
        # A chat model, or a factory, that cannot give one fails here,
        # before anything is opened or spent.
        asked_for = []
        if any(s != "vision_labels" for s in calling):
            asked_for.append(("readers", model, False))
            if triage_model and ("labels" in stages or "ingest" in stages):
                asked_for.append(("triage", triage_model, False))
        if any(s in ("vision_labels", "ingest") for s in calling):
            asked_for.append(("vision", vision_model, True))
        if "range_labels" in stages:
            asked_for.append(("describer", describe_model, True))
            if range_model != model:
                asked_for.append(("range labeller", range_model, False))
        asked_for += [(f"{stage}, model first" if first.on
                       else f"{stage} reader", name, False)
                      for stage, name in first.stage_models.items()
                      if stage in stages or "ingest" in stages]
        if first.on:
            if "ingest" in stages:
                asked_for.append(("describer, model first", describe_model,
                                  True))
                if range_model != model:
                    asked_for.append(("range labeller, model first",
                                      range_model, False))
        for what, name, vision in asked_for:
            said = engines.describe(name, vision=vision)
            engine_notes.append(
                f"  {what}: engine {said.get('engine')}"
                + (f" over {said['chat_model']}" if said.get("chat_model")
                   else "")
                + f", model {said['model']} (BACKLOG RI-67)")
    (logs_truth, lab_truth, narrative_truth, calc_truth,
     sounding_truth) = _truth_dirs(
        truth_dir, lab_truth_dir, narrative_truth_dir, calc_truth_dir,
        sounding_truth_dir)
    if "logs" in stages and logs_truth is None:
        raise ValueError(
            "the 'logs' stage scores the reader against the hand-truthed "
            "logs, so it needs truth_dir -- either the folder of "
            "<ID>_p<page>.json files itself, or one truth root with a "
            "'logs/' folder in it. They are private and do not ship in the "
            "wheel.")
    if "lab" in stages and lab_truth is None:
        raise ValueError(
            "the 'lab' stage scores the reader against the hand-truthed "
            "laboratory sheets, so it needs them: either lab_truth_dir -- "
            "the folder of <kind>__<ID>_p<page>.json files -- or a 'lab/' "
            "folder inside truth_dir. They are private and do not ship in "
            "the wheel.")
    if "calc" in stages and calc_truth is None:
        raise ValueError(
            "the 'calc' stage scores the reader against the hand-truthed "
            "calculation printouts, so it needs them: either calc_truth_dir "
            "-- the folder of <kind>__<ID>_p<first page>.json files -- or a "
            "'calc/' folder inside truth_dir. They are private and do not "
            "ship in the wheel.")
    if "soundings" in stages and sounding_truth is None:
        raise ValueError(
            "the 'soundings' stage scores the readers against the "
            "hand-truthed test pits, cone soundings and dynamic probes, so "
            "it needs them: either sounding_truth_dir -- the folder of "
            "<kind>__<ID>_p<page>.json files -- or a 'soundings/' folder "
            "inside truth_dir. They are private and do not ship in the "
            "wheel.")
    if "narrative" in stages and narrative_truth is None:
        raise ValueError(
            "the 'narrative' stage scores the reader against the hand "
            "answers, so it needs them: either narrative_truth_dir -- the "
            "folder of <ID>.json files, each holding the owner's two schemas "
            "answered by hand with null for 'not stated' -- or a "
            "'narrative/' folder inside truth_dir. They are private and do "
            "not ship in the wheel.")
    out = Path(out_dir)
    note = out_dir_note(out)
    if note:
        print(note)
    for sub in ("runs", "triage"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if "logs" in stages:
        (out / "logs").mkdir(parents=True, exist_ok=True)
    if "lab" in stages:
        (out / "lab").mkdir(parents=True, exist_ok=True)
    if "calc" in stages:
        (out / "calc").mkdir(parents=True, exist_ok=True)
    if "soundings" in stages:
        (out / "soundings").mkdir(parents=True, exist_ok=True)
    if "narrative" in stages:
        (out / "narrative").mkdir(parents=True, exist_ok=True)
    if "vision_labels" in stages:
        (out / "vision").mkdir(parents=True, exist_ok=True)
    if "vote" in stages:
        (out / "vote").mkdir(parents=True, exist_ok=True)
    if "ingest" in stages:
        (out / "ingest").mkdir(parents=True, exist_ok=True)
    if "range_labels" in stages:
        (out / "ranges").mkdir(parents=True, exist_ok=True)

    # THE DURABLE COPY, before anything else: a wiped /tmp resumes from what
    # the mirror holds rather than paying the model for it a second time.
    mirror = Mirror(sharepoint=sharepoint, durable_dir=durable_dir,
                    base_folder=sharepoint_folder)
    sync = _MirrorSync(mirror, out)
    sync.restore()

    corpus = Corpus(reports_dir, manifest=manifest, di_dir=di_dir,
                    labels_xlsx=labels_xlsx, cache_dir=out)
    if not corpus.available:
        raise FileNotFoundError(
            f"no reports at {reports_dir}: expected R01.pdf ... R38.pdf "
            f"there, or the original file names plus a manifest "
            f"({corpus.manifest}) whose source-file column names them")
    triage_model = triage_model or model

    oos: Dict[str, Dict[int, dict]] = {}
    if oos_labels:
        blob = json.loads(Path(oos_labels).read_text(encoding="utf-8"))
        oos = {rid: {int(p): v for p, v in pages.items()}
               for rid, pages in blob.items()}
    mapped = corpus.mapped_ids() if corpus.labels_available else []

    # The two whole-report stages -- the label review and the vision
    # experiment -- run over the same reports, because they are two answers
    # to one question and are scored against one set of hand labels.
    by_report = (("labels" in stages) or ("vision_labels" in stages)
                 or ("ingest" in stages) or ("range_labels" in stages))
    asked: List[str] = []
    if by_report:
        for name in sets:
            for rid in _set_ids(name, corpus):
                if rid not in asked:
                    asked.append(rid)
    # A cluster folder may hold a subset of the 38 the manifest lists. Say
    # which are not there once, up front, rather than failing them one at a
    # time deep in the run; a report already scored keeps its run file.
    present = set(corpus.present_ids())
    wanted = [rid for rid in asked
              if rid in present
              or (out / "runs" / f"{rid}.json").is_file()
              or (out / "vision" / f"{rid}.json").is_file()
              or (out / "ingest" / rid / "run.json").is_file()
              or any((out / "ranges" / arm / f"{rid}.json").is_file()
                     for arm in range_arms)]
    absent = [rid for rid in asked if rid not in wanted]
    if max_reports:
        wanted = wanted[:int(max_reports)]
    label_ids = wanted if "labels" in stages else []

    print(f"report ingest on the cluster: stage(s) {', '.join(stages)}; "
          f"model {model}, triage {triage_model}; out_dir {out}")
    for note in engine_notes:
        print(note)
    if any(s in stages for s in FIRST_STAGES + ("ingest",)):
        print(f"  {first.describe(model)}")
    # Every stage takes this in the Prompter's place: the Prompter itself
    # on a Prompter run, as before, else the run's engines.
    source = prompter if engines.plain else engines
    if "labels" in stages:
        print(f"  labels: {len(label_ids)} report(s)")
    if absent:
        print(f"  not in {corpus.reports_dir} and skipped: "
              f"{', '.join(absent)}")
    spent = 0.0
    spent_tokens = 0
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, rid in enumerate(label_ids, 1):
        run_file = out / "runs" / f"{rid}.json"
        if run_file.is_file() and not redo:
            done[rid] = json.loads(run_file.read_text(encoding="utf-8"))
            print(f"  [{n}/{len(wanted)}] {rid}: already done, skipping")
            continue
        if max_total_dollars is not None and spent > max_total_dollars:
            print(f"  stopping before {rid}: spent ${spent:.2f}, past the "
                  f"${max_total_dollars:.2f} ceiling")
            break
        if max_total_tokens is not None and spent_tokens >= max_total_tokens:
            print(f"  stopping before {rid}: {spent_tokens:,} tokens spent, "
                  f"past the {max_total_tokens:,} ceiling")
            break
        try:
            blob = _run_one(rid, corpus, source, model, triage_model, out)
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(wanted)}] {rid}: FAILED -- {failures[rid]}")
            traceback.print_exc()
            continue
        done[rid] = blob
        sync()                       # the run file is on disk; make it durable
        spent += blob["cost"].get("dollars", 0.0)
        spent_tokens += _tokens_of(blob["cost"])
        review = blob["review"]
        print(f"  [{n}/{len(wanted)}] {rid}: {blob['n_pages']} pp, "
              f"{len(review['changes'])} change(s), "
              f"{review['tool_calls']}/{review['budget']} tool calls, "
              f"{blob['cost']['calls']} model calls, "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    results = _score(done, corpus, oos, mapped, sets, model, triage_model,
                     failures, absent)
    if "labels" in stages and not engines.plain:
        review_model, fields = engines.stage(model)
        results["engine"] = fields.get("engine", "prompter")
        results["review_model"] = review_model
        results["triage_model"] = engines.stage(triage_model)[0]
        if fields.get("chat_model"):
            results["chat_model"] = fields["chat_model"]
    results["stages"] = list(stages)
    if corpus.di_withheld:
        results["di_withheld"] = True
    lines = _render(results) if "labels" in stages else _header(results)

    if "logs" in stages:
        logs = _run_logs(corpus, source, first.model("logs", model), out,
                         logs_truth, budget=log_budget, redo=redo,
                         max_logs=max_reports, open_reports=open_reports,
                         sync=sync, templates_path=templates_path,
                         first=first)
        results["logs"] = logs
        lines += _render_logs(logs)

    if "lab" in stages:
        lab = _run_lab(corpus, source, first.model("lab", model), out,
                       lab_truth, budget=lab_budget, redo=redo,
                       max_sheets=max_reports,
                       open_reports=open_lab_reports, sync=sync, first=first)
        results["lab"] = lab
        lines += _render_lab(lab)

    if "calc" in stages:
        calc = _run_calc(corpus, source, first.model("calc", model), out,
                         calc_truth, budget=calc_budget, redo=redo,
                         max_calcs=max_reports,
                         open_reports=open_calc_reports, sync=sync,
                         first=first)
        results["calc"] = calc
        lines += _render_calc(calc)

    if "soundings" in stages:
        soundings = _run_soundings(
            corpus, source, model, out, sounding_truth,
            budget=sounding_budget, redo=redo, max_sheets=max_reports,
            open_reports=open_sounding_reports, sync=sync)
        results["soundings"] = soundings
        lines += _render_soundings(soundings)

    if "narrative" in stages:
        narrative = _run_narrative(
            corpus, source, first.model("narrative", model), out,
            narrative_truth, budget=narrative_budget, redo=redo,
            max_reports=max_reports, open_reports=open_narrative_reports,
            sync=sync, front_pages=narrative_front_pages, first=first)
        results["narrative"] = narrative
        lines += _render_narrative(narrative)

    if "vision_labels" in stages:
        vision = _run_vision(corpus, source, vision_model, out, wanted,
                             mode=vision_mode, dpi=vision_dpi,
                             outline_context=vision_outline_context,
                             detail=vision_detail, window=vision_window,
                             images_per_call=vision_images_per_call,
                             fallback=vision_fallback, redo=redo,
                             sync=sync)
        seen_model, seen_fields = engines.stage(vision_model, vision=True)
        scored_vision = _score_vision(vision["runs"], corpus, oos, mapped,
                                      sets, out, seen_model,
                                      vision["failures"], vision["settings"])
        scored_vision.update(seen_fields)
        results["vision_labels"] = scored_vision
        lines += _render_vision(scored_vision)

    if "vote" in stages:
        vote = _run_vote(corpus, oos, mapped, sets, out,
                         vision_dirs=vision_dirs, review_dir=review_dir,
                         sync=sync)
        results["vote"] = vote
        lines += _render_vote(vote)

    if "ingest" in stages:
        ingest = _run_ingest(corpus, source, model, out, wanted,
                             triage_model=triage_model,
                             truth=(logs_truth, lab_truth,
                                    narrative_truth, calc_truth,
                                    sounding_truth),
                             log_budget=log_budget, lab_budget=lab_budget,
                             narrative_budget=narrative_budget,
                             redo=redo, review_dir=review_dir,
                             max_total_dollars=max_total_dollars,
                             max_total_tokens=max_total_tokens,
                             max_report_tokens_in=max_report_tokens_in,
                             max_request_tokens_in=max_request_tokens_in,
                             lab_sheets_per_call=lab_sheets_per_call,
                             tokens_per_minute=tokens_per_minute,
                             label_policy=label_policy,
                             review_mode=review_mode,
                             vision_model=vision_model,
                             vision_mode=ingest_vision_mode,
                             vision_detail=vision_detail,
                             trust_table=trust_table,
                             templates_path=templates_path,
                             ingest_bound=ingest_bound,
                             counts_from_logs=ingest_counts_from_logs,
                             narrative_skips_bound=(
                                 ingest_narrative_skips_bound),
                             oos=oos, mapped=mapped, sync=sync,
                             first=first, describe_model=describe_model,
                             range_model=range_model)
        results["ingest"] = ingest
        lines += _render_ingest(ingest)

    if "range_labels" in stages:
        ranged = _run_ranges(corpus, source, describe_model, range_model,
                             out, wanted, arms=tuple(range_arms),
                             workers=range_workers, redo=redo, sync=sync)
        scored_ranges = _score_ranges(ranged, corpus, oos, mapped, sets)
        scored_ranges["describer"] = engines.describe(describe_model,
                                                      vision=True)
        scored_ranges["labeller"] = engines.describe(range_model)
        results["range_labels"] = scored_ranges
        lines += _render_ranges(scored_ranges)

    (out / "results.json").write_text(json.dumps(_plain(results), indent=2),
                                      encoding="utf-8")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sync()                          # RESULTS.md and results.json, durable too
    print("\n".join(lines))
    print(f"\nwrote {out / 'RESULTS.md'} -- that is the file to bring back")
    print(f"per-report runs in {out / 'runs'}, profiles in {out / 'triage'}")
    if "logs" in stages:
        print(f"per-log runs in {out / 'logs'}")
    if "lab" in stages:
        print(f"per-sheet runs in {out / 'lab'}")
    if "calc" in stages:
        print(f"per-calculation runs in {out / 'calc'}")
    if "soundings" in stages:
        print(f"per-sounding runs in {out / 'soundings'}")
    if "narrative" in stages:
        print(f"per-report narrative runs in {out / 'narrative'}")
    if "vision_labels" in stages:
        print(f"per-report vision runs in {out / 'vision'}")
    if "vote" in stages:
        print(f"per-report disagreement lists in {out / 'vote'}")
    if "ingest" in stages:
        print(f"per-report records and exports in {out / 'ingest'}")
    if "range_labels" in stages:
        print(f"per-report descriptions and section labels in "
              f"{out / 'ranges'}")
    sync.report()
    return results


def _score(done: Dict[str, dict], corpus: Corpus,
           oos: Dict[str, Dict[int, dict]], mapped: Sequence[str],
           sets: Sequence[str], model: str, triage_model: str,
           failures: Dict[str, str],
           absent: Sequence[str] = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "date": date.today().isoformat(),
        "engine": "prompter",
        "review_model": model,
        "triage_model": triage_model,
        "served_by": sorted({s for b in done.values()
                             for s in (b.get("served_by") or [])}),
        "failures": dict(failures),
        "absent": list(absent),
        "sets": {},
        "triage": [],
        "totals": _cost_totals(),
    }
    for rid in sorted(done):
        blob = done[rid]
        _add_cost(out["totals"], blob["cost"])
        out["totals"]["seconds"] += blob.get("seconds", 0.0)
        profile = blob["profile"]
        out["triage"].append({
            "id": rid,
            "document_type": profile["document_type"],
            "workflow": profile["workflow"],
            "bound_together": len(profile["bound_together"]),
            "toc_agreement": profile["toc_agreement"],
            "scan_fraction": profile["scan_fraction"],
        })

    for name in sets:
        ids = [rid for rid in _set_ids(name, corpus) if rid in done]
        before, after = Scores(), Scores()
        clean_before, clean_after = Scores(), Scores()
        verdicts: Dict[str, int] = {}
        per_report: List[dict] = []
        dropped = 0
        for rid in ids:
            blob = done[rid]
            rules = {int(k): v for k, v in blob["rules_labels"].items()}
            final = {int(k): v
                     for k, v in blob["review"]["final_labels"].items()}
            hand, alternates, source = _truth_for(rid, corpus, oos, mapped)
            never_seen = name == "oos_blind" and rid not in CHECKPOINT
            hits_before = hits_after = scored = 0
            for page, want in sorted(hand.items()):
                alts = alternates.get(page, ())
                was, now = rules.get(page, "other"), final.get(page, "other")
                if disputed_drop(rid, page, now):
                    dropped += 1
                    continue
                before.add(want, was, alts)
                after.add(want, now, alts)
                if never_seen:
                    clean_before.add(want, was, alts)
                    clean_after.add(want, now, alts)
                scored += 1
                hits_before += int(was == want)
                hits_after += int(now == want)
            for change in blob["review"]["changes"]:
                verdict = verdict_for(rid, int(change["page"]),
                                      change["from"], change["to"],
                                      hand.get(int(change["page"])))
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
            per_report.append({
                "id": rid, "pages": blob["n_pages"], "scored": scored,
                "truth": source,
                "before": (hits_before / scored) if scored else None,
                "after": (hits_after / scored) if scored else None,
                "changes": len(blob["review"]["changes"]),
                "rejected": len(blob["review"]["rejected_changes"]),
                "unresolved": len(blob["review"]["unresolved"]),
                "tool_calls": blob["review"]["tool_calls"],
                "budget": blob["review"]["budget"],
                "calls": blob["cost"]["calls"],
                "tokens_in": tokens_in(blob["cost"]),
                "output_tokens": blob["cost"]["output_tokens"],
                "seconds": blob["seconds"],
            })
        row: Dict[str, Any] = {
            "reports": ids,
            "n_reports": len(ids),
            "disputed_dropped": dropped,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "verdicts": verdicts,
            "gate_failures": gate_failures(after),
            "per_report": per_report,
            "_scores": (before, after),
        }
        if name == "oos_blind" and clean_after.n:
            row["never_seen"] = {
                "reports": [r for r in ids if r not in CHECKPOINT],
                "excluded": [r for r in ids if r in CHECKPOINT],
                "before": clean_before.to_dict(),
                "after": clean_after.to_dict(),
                "gate_failures": gate_failures(clean_after),
                "_scores": (clean_before, clean_after),
            }
        out["sets"][name] = row
    return out


def _plain(value: Any) -> Any:
    """The results with the live :class:`Scores` objects taken out.

    ``_scores`` is how the renderer gets at the rates without recomputing
    them; it cannot be serialized, and a key beginning with an underscore is
    the signal that it is working state rather than a result.

    A score's ``misses`` are left out too. They travel in the RUN FILES,
    which stay with the run, so a check can be diagnosed and re-scored; a
    miss quotes the truth's own value -- a project, a driller, a client --
    and ``results.json`` is a file that comes home.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()
                if not str(k).startswith("_") and k != "misses"}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# the logs stage
# ---------------------------------------------------------------------------

def _open_set(truth_dir: Path,
              override: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """The reports the log rules were allowed to be tuned on."""
    if override:
        return tuple(override)
    open_file = truth_dir / "OPEN.txt"
    if open_file.is_file():
        names = tuple(line.strip() for line
                      in open_file.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return DEFAULT_OPEN_LOGS


def _run_logs(corpus: Corpus, prompter: Any, model: str, out: Path,
              truth_dir: Path, *, budget: int, redo: bool,
              max_logs: Optional[int],
              open_reports: Optional[Sequence[str]],
              sync: Optional[Any] = None,
              templates_path: Any = None,
              first: Optional[_FirstArm] = None) -> Dict[str, Any]:
    """Grid and reader over every hand-truthed log, before and after.

    Restartable the same way the labels stage is: each log writes
    ``logs/<log id>.json`` as it finishes and a later call skips it.

    A private log-TEMPLATE fingerprint file travelling with the truth folder
    (``<truth_dir>/../templates.json``, or ``templates_path``) is put in
    force for the whole stage, so a log printed on a form the file describes
    is recognised and its columns are named by the form. With no such file
    nothing changes.

    ``prompter`` is the live Prompter, or the run's :class:`_Engines`; so
    for every stage below. ``first`` is the run's model-first arm
    (:class:`_FirstArm`, off when not given); so for the lab, calc and
    narrative stages, and the ingest.
    """
    from report_ingest.engine import CostMeter
    from report_ingest.log_scoring import METRICS, score_one_log
    from report_ingest.log_templates import (
        load_templates, templates_beside, use_templates,
    )

    found = (load_templates(templates_path) if templates_path is not None
             else templates_beside(truth_dir))
    use_templates(found)
    if found:
        print(f"  log templates: {len(found)} fingerprint(s) in force")

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs at {truth_dir}; the 'logs' stage scores "
            f"against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_logs:
        truths = truths[:int(max_logs)]
    openset = _open_set(truth_dir, open_reports)
    print(f"  logs: {len(truths)} hand-truthed log(s); open set "
          f"{', '.join(openset)}")

    engines = _Engines.of(prompter)
    first = first or _FirstArm()
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        log_id = str(truth.get("id") or f"log{n}")
        report = log_id.split("_")[0]
        run_file = out / "logs" / f"{log_id}.json"
        clash = _arm_clash(run_file, first)
        if clash:
            failures[log_id] = clash
            print(f"  [{n}/{len(truths)}] {log_id}: skipped -- {clash}")
            continue
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            failed = _saved_failure(blob)
            if failed:
                print(f"  [{n}/{len(truths)}] {log_id}: previous attempt "
                      f"failed ({failed[:90]}); retrying")
            else:
                # The open/blind split is decided at SCORING time, not
                # frozen into the run file. A log moved into the open set
                # after it ran must move in the scorecard too, or the blind
                # figure quietly keeps crediting a log somebody has since
                # looked at.
                blob["set"] = "open" if report in openset else "blind"
                done[log_id] = blob
                print(f"  [{n}/{len(truths)}] {log_id}: already done, "
                      "skipping")
                continue
        if report not in set(corpus.present_ids()):
            failures[log_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {log_id}: skipped -- "
                  f"{failures[log_id]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            if first.on:
                from report_ingest.model_first_scoring import first_log
                before, after, arm = first_log(
                    truth, doc, engine, report_id=report,
                    page_route=first.page_route)
            else:
                before, after = score_one_log(truth, doc, engine,
                                              budget=budget,
                                              report_id=report)
                arm = {"model_first": "off"}
            sources = _page_sources(doc, [int(p) for p in
                                          truth.get("pages") or []])
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[log_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {log_id}: FAILED -- "
                  f"{failures[log_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "log_id": log_id,
            "report": report,
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            **engines.record(model, engine),
            **arm,
            "served_by": engine.served_by,
            "pages": [int(p) for p in truth.get("pages") or []],
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
            # What the visual-scales voters could act on (BACKLOG RI-05).
            "visual_scales_on": visual_scales_enabled(),
            "page_sources": sources,
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[log_id] = blob
        gain = (after.total.found - before.total.found)
        print(f"  [{n}/{len(truths)}] {log_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    stage_model, fields = engines.stage(model)
    return {**_score_logs(done, failures, stage_model, openset),
            **fields, **_first_results(done.values())}


# ---------------------------------------------------------------------------
# the lab stage
# ---------------------------------------------------------------------------

def _run_lab(corpus: Corpus, prompter: Any, model: str, out: Path,
             truth_dir: Path, *, budget: int, redo: bool,
             max_sheets: Optional[int],
             open_reports: Optional[Sequence[str]],
             sync: Optional[Any] = None,
             first: Optional[_FirstArm] = None) -> Dict[str, Any]:
    """Tables and reader over every hand-truthed laboratory sheet.

    Restartable the same way the other stages are: each sheet writes
    ``lab/<sheet id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter
    from report_ingest.lab_reader import (
        floor_per_page_enabled, link_from_label_enabled,
    )
    from report_ingest.lab_scoring import pages_of, report_of, score_one_sheet

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed laboratory sheets at {truth_dir}; the 'lab' "
            f"stage scores against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_sheets:
        truths = truths[:int(max_sheets)]
    openset = _open_set(truth_dir, open_reports) if (
        open_reports or (truth_dir / "OPEN.txt").is_file()) \
        else DEFAULT_OPEN_LAB
    print(f"  lab: {len(truths)} hand-truthed sheet(s); open set "
          f"{', '.join(openset)}")

    engines = _Engines.of(prompter)
    first = first or _FirstArm()
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        sheet_id = str(truth.get("id") or f"sheet{n}")
        report = report_of(truth)
        run_file = out / "lab" / f"{sheet_id}.json"
        clash = _arm_clash(run_file, first)
        if clash:
            failures[sheet_id] = clash
            print(f"  [{n}/{len(truths)}] {sheet_id}: skipped -- {clash}")
            continue
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            failed = _saved_failure(blob)
            if failed:
                print(f"  [{n}/{len(truths)}] {sheet_id}: previous attempt "
                      f"failed ({failed[:90]}); retrying")
            else:
                # The open/blind split is decided at SCORING time, never
                # frozen into a run file: a sheet moved into the open set
                # after it ran has to move in the scorecard too, or the
                # blind figure quietly keeps crediting a page somebody has
                # since looked at.
                blob["set"] = "open" if report in openset else "blind"
                done[sheet_id] = blob
                print(f"  [{n}/{len(truths)}] {sheet_id}: already done, "
                      "skipping")
                continue
        if report not in set(corpus.present_ids()):
            failures[sheet_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {sheet_id}: skipped -- "
                  f"{failures[sheet_id]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            if first.on:
                from report_ingest.model_first_scoring import first_sheet
                before, after, arm = first_sheet(
                    truth, doc, engine, report_id=report,
                    page_route=first.page_route)
            else:
                before, after = score_one_sheet(truth, doc, engine,
                                                budget=budget,
                                                report_id=report)
                arm = {"model_first": "off"}
            sources = _page_sources(doc, pages_of(truth))
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[sheet_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {sheet_id}: FAILED -- "
                  f"{failures[sheet_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "sheet_id": sheet_id,
            "report": report,
            "kind": str(truth.get("kind") or ""),
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            **engines.record(model, engine),
            **arm,
            "served_by": engine.served_by,
            "pages": pages_of(truth),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
            # Which arm of the specimen-label switch this sheet ran in
            # (RI-07, off by default), as the reader reported it; a sheet
            # whose reading failed is put in the arm it was asked to run in.
            "link_from_label": (after.link_from_label
                                if after.link_from_label is not None
                                else link_from_label_enabled()),
            # ... and of the per-page floor switch (RI-56), the same way.
            "floor_per_page": (after.floor_per_page
                               if after.floor_per_page is not None
                               else floor_per_page_enabled()),
            # What the visual-scales voters could act on (BACKLOG RI-05).
            "visual_scales_on": visual_scales_enabled(),
            "page_sources": sources,
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[sheet_id] = blob
        gain = after.total.found - before.total.found
        print(f"  [{n}/{len(truths)}] {sheet_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    stage_model, fields = engines.stage(model)
    return {**_score_lab(done, failures, stage_model, openset),
            **fields, **_first_results(done.values())}


def _lab_totals(rows: Sequence[dict], stage: str
                ) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of sheets."""
    from report_ingest.lab_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _score_lab(done: Dict[str, dict], failures: Dict[str, str], model: str,
               openset: Sequence[str]) -> Dict[str, Any]:
    from report_ingest.lab_scoring import METRICS

    def beside(group):
        floor, n_floor = _alone_totals(group, "floor_alone", METRICS)
        alone, n_model = _alone_totals(group, "model_alone", METRICS)
        read, n_read = _alone_totals(group, "unlinked", METRICS)
        return {"floor_alone": floor, "n_floor": n_floor,
                "model_alone": alone, "n_model": n_model,
                "unlinked": read, "n_unlinked": n_read}

    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "sheets": [r["sheet_id"] for r in group],
            "n_sheets": len(group),
            "before": _lab_totals(group, "before"),
            "after": _lab_totals(group, "after"),
            **beside(group),
        }
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "?" for r in rows}):
        group = [r for r in rows if (r.get("kind") or "?") == kind]
        by_kind[kind] = {
            "n_sheets": len(group),
            "before": _lab_totals(group, "before"),
            "after": _lab_totals(group, "after"),
            **beside(group),
        }
    cost = _cost_totals()
    visual = _visual_counts()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
        _add_visual(visual, row)
    return {
        "date": date.today().isoformat(),
        "visual_scales": visual,
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_sheets": len(rows),
        "failures": dict(failures),
        # The specimen-label switch's arm(s) these sheets ran in; a run file
        # written before the switch existed ran with it off.
        "link_from_label": sorted({bool(r.get("link_from_label"))
                                   for r in rows}),
        # ... and the per-page floor switch's (RI-56), the same way.
        "floor_per_page": sorted({bool(r.get("floor_per_page"))
                                  for r in rows}),
        "sets": sets,
        "kinds": by_kind,
        "per_sheet": [{
            "sheet_id": r["sheet_id"], "set": r["set"],
            "kind": r.get("kind") or "?",
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "floor_alone": _overall_of(r, "floor_alone"),
            "model_alone": _overall_of(r, "model_alone"),
            "unlinked": _overall_of(r, "unlinked"),
            "model_calls": r["after"].get("model_calls", 0),
            "tool_calls": r["after"].get("tool_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "tokens_in": tokens_in(r["cost"]),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_lab(lab: Dict[str, Any]) -> List[str]:
    """The lab stage, as tables that carry IDs, kinds, counts and rates only."""
    from report_ingest.lab_scoring import (
        DEPTH_TOL_M, EXACT_TOL, METRICS, MODEL_ONLY, PASSING_TOL,
        UNLINKED_METRICS,
    )

    out: List[str] = [
        "", f"# WP3 on the cluster: the lab reader through "
            f"{_through(lab)}", "",
        f"Run {lab['date']}. Model `{lab['model']}`, "
        f"{lab['n_sheets']} hand-truthed sheet(s).",
        "",
        f"**before** is what the page's own detected TABLES hold; **after** "
        f"is what the reader's records hold. A table has no idea what test "
        f"it is on or which boring it belongs to, so "
        f"{' and '.join('`' + m + '`' for m in MODEL_ONLY)} have "
        f"no before column at all, and the others ask of the tables "
        f"only whether the number is on the page. Tolerances: a depth links "
        f"within {DEPTH_TOL_M} m, compared in metres whatever the sheet "
        f"prints; an index value is exact to {EXACT_TOL}; a grading within "
        f"{PASSING_TOL} percent; a curve within the tolerance its own truth "
        f"file states.",
        "",
        "Beside them: **floor** is the tables' floor RECORD (the typed tests "
        "built before any model call) scored alone, **model** the model's "
        "answer scored alone, before the merge; **read** asks only whether "
        "each printed VALUE is anywhere in the record, linked to the right "
        "boring and depth or not -- after asks for the link too, and a "
        "specimen whose link misses loses every value on it there. So a low "
        "after with a high read is a linking failure, not a reading one. "
        "read counts values only (index, series, curve), so its OVERALL "
        "carries no kind or link.",
    ]
    out += _engine_lines(lab)
    out += _first_lines(lab)
    if lab.get("served_by"):
        out.append(f"Served by: {', '.join(lab['served_by'])}.")
    arms = lab.get("link_from_label")
    if arms:
        out.append("Specimen-label paragraph (RI-07, "
                   "GEOTECH_INGEST_LAB_LINK_FROM_LABEL): "
                   + " and ".join("on" if a else "off" for a in arms)
                   + (" -- MIXED: these sheets did not all run in one arm"
                      if len(arms) > 1 else "") + ".")
    arms = lab.get("floor_per_page")
    if arms:
        out.append("Per-page floor in the brief (RI-56, "
                   "GEOTECH_INGEST_LAB_FLOOR_PER_PAGE): "
                   + " and ".join("on" if a else "off" for a in arms)
                   + (" -- MIXED: these sheets did not all run in one arm"
                      if len(arms) > 1 else "") + ".")
    if lab.get("failures"):
        out += ["", "**Sheets that failed and are NOT in any number below:**"]
        out += [f"- {sheet_id}: {why}"
                for sheet_id, why in lab["failures"].items()]
    out += ["", f"Open set (the reports whose lab pages were looked at): "
                f"{', '.join(lab['open_set'])}. Everything else is blind."]

    for name in ("open", "blind", "all"):
        row = lab["sets"].get(name)
        if not row:
            continue
        floor = row.get("floor_alone") if row.get("n_floor") else None
        alone = row.get("model_alone") if row.get("n_model") else None
        read = row.get("unlinked") if row.get("n_unlinked") else None
        out += ["", f"## {name} -- {row['n_sheets']} sheet(s)", "", "```",
                f"{'metric':<10}{'before':>14}{'floor':>14}{'model':>14}"
                f"{'after':>14}{'read':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not after or not (before["total"] or after["total"]):
                continue
            got = (read or {}).get(metric) if metric in UNLINKED_METRICS \
                else None
            out.append(f"{metric:<10}{_rate(before):>14}"
                       f"{_opt_rate((floor or {}).get(metric)):>14}"
                       f"{_opt_rate((alone or {}).get(metric)):>14}"
                       f"{_rate(after):>14}{_opt_rate(got):>14}")
        read_all = (_sum_of({m: v for m, v in read.items()
                             if m in UNLINKED_METRICS}) if read else None)
        out += [f"{'OVERALL':<10}{_rate(_sum_of(row['before'])):>14}"
                f"{_opt_rate(_sum_of(floor) if floor else None):>14}"
                f"{_opt_rate(_sum_of(alone) if alone else None):>14}"
                f"{_rate(_sum_of(row['after'])):>14}"
                f"{_opt_rate(read_all):>14}", "```"]
        for key, label in (("n_floor", "floor"), ("n_model", "model"),
                           ("n_unlinked", "read")):
            if row.get(key, 0) < row["n_sheets"]:
                out.append(f"`{label}` is over the {row.get(key, 0)} of "
                           f"{row['n_sheets']} sheet(s) whose run file keeps "
                           f"it.")

    out += ["", "## Per kind", "", "```",
            f"{'kind':<20}{'sheets':>7}{'before':>14}{'after':>14}"
            f"{'read':>14}"]
    for kind, row in lab.get("kinds", {}).items():
        read = row.get("unlinked") if row.get("n_unlinked") else None
        read_all = (_sum_of({m: v for m, v in read.items()
                             if m in UNLINKED_METRICS}) if read else None)
        out.append(f"{kind:<20}{row['n_sheets']:>7}"
                   f"{_rate(_sum_of(row['before'])):>14}"
                   f"{_rate(_sum_of(row['after'])):>14}"
                   f"{_opt_rate(read_all):>14}")
    out.append("```")

    out += ["", "## Per sheet", "", "```",
            f"{'sheet':<28}{'set':<7}{'before':>12}{'floor':>12}{'model':>12}"
            f"{'after':>12}{'read':>12}{'calls':>7}"
            f"{'zoom':>6}{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in lab["per_sheet"]:
        if r.get("error"):
            out.append(f"{r['sheet_id']:<28}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:50]}")
            continue
        out.append(
            f"{r['sheet_id']:<28}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_opt_rate(r.get('floor_alone')):>12}"
            f"{_opt_rate(r.get('model_alone')):>12}{_rate(r['after']):>12}"
            f"{_opt_rate(r.get('unlinked')):>12}"
            f"{r['model_calls']:>7}{r['tool_calls']:>6}{r['unresolved']:>7}"
            f"{r['changes']:>6}{r['tokens_in']:>10,}"
            f"{r['output_tokens']:>8}{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`zoom` is how many times the reader magnified a plot. "
                "`unres` is what it could not settle plus what Python "
                "refused -- an impossible depth, a percentage past 100, a "
                "liquid limit below the plastic limit, a grading running "
                "the wrong way. `look` is every value it took from the "
                "picture and every curve it digitised.", ""]
    out += _visual_lines(lab.get("visual_scales") or {})
    cost = lab["cost"]
    n = max(1, lab["n_sheets"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per sheet: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, lab)]
    return out




# ---------------------------------------------------------------------------
# the calc stage (WP5)
# ---------------------------------------------------------------------------

#: The reports whose calculation pages were open to the builder when the
#: truth folder carries no ``OPEN.txt``. There is no blind calculation set
#: yet -- every hand-truthed run was read while the reader's prompt was
#: written -- and the scorecard says so rather than reporting a blind figure
#: that does not exist.
DEFAULT_OPEN_CALC: Tuple[str, ...] = ()

#: How an on/off switch's arm reads in a table. ``None`` is a run file
#: written before the switch was recorded (BACKLOG RI-67).
_ARM_NAMES = {False: "off", True: "on", None: "not recorded"}


def _arms(values: Iterable[Any]) -> List[Optional[bool]]:
    """The arms of an on/off switch a set of run files ran in, off
    first; ``None`` for a file that does not say."""
    seen = {None if value is None else bool(value) for value in values}
    return [arm for arm in (False, True, None) if arm in seen]


def _arm_line(what: str, arms: Sequence[Optional[bool]],
              things: str) -> str:
    """One line saying which arm(s) of a switch a stage's files ran in,
    and flagging a mix, as the lab stage does its own switch."""
    line = f"{what}: " + " and ".join(_ARM_NAMES[arm] for arm in arms)
    if len(arms) > 1:
        line += (f" -- MIXED: these {things} were not all read in one "
                 f"recorded arm")
    if None in arms:
        line += ("; `not recorded` is a run file written before the "
                 "switch was recorded (RI-67), whose arm only its "
                 "out_dir's name can say")
    return line + "."


def _run_calc(corpus: Corpus, prompter: Any, model: str, out: Path,
              truth_dir: Path, *, budget: int, redo: bool,
              max_calcs: Optional[int],
              open_reports: Optional[Sequence[str]],
              sync: Optional[Any] = None,
              first: Optional[_FirstArm] = None) -> Dict[str, Any]:
    """The floor and then the calculation reader over every hand-truthed run.

    Restartable the same way the other stages are: each calculation writes
    ``calc/<calc id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.calc_reader import page_images_enabled
    from report_ingest.calc_scoring import (
        pages_of, report_of, score_one_calc,
    )
    from report_ingest.engine import CostMeter

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed calculations at {truth_dir}; the 'calc' stage "
            f"scores against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_calcs:
        truths = truths[:int(max_calcs)]
    openset = _open_set(truth_dir, open_reports) if (
        open_reports or (truth_dir / "OPEN.txt").is_file()) \
        else DEFAULT_OPEN_CALC
    print(f"  calc: {len(truths)} hand-truthed run(s); open set "
          f"{', '.join(openset) or '(none named)'}")

    engines = _Engines.of(prompter)
    first = first or _FirstArm()
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        calc_id = str(truth.get("id") or f"calc{n}")
        report = report_of(truth)
        run_file = out / "calc" / f"{calc_id}.json"
        clash = _arm_clash(run_file, first)
        if clash:
            failures[calc_id] = clash
            print(f"  [{n}/{len(truths)}] {calc_id}: skipped -- {clash}")
            continue
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            failed = _saved_failure(blob)
            if failed:
                print(f"  [{n}/{len(truths)}] {calc_id}: previous attempt "
                      f"failed ({failed[:90]}); retrying")
            else:
                # The open/blind split is decided at SCORING time, never
                # frozen into a run file: a run moved into the open set
                # after it ran has to move in the scorecard too.
                blob["set"] = "open" if report in openset else "blind"
                done[calc_id] = blob
                print(f"  [{n}/{len(truths)}] {calc_id}: already done, "
                      "skipping")
                continue
        if report not in set(corpus.present_ids()):
            failures[calc_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {calc_id}: skipped -- "
                  f"{failures[calc_id]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            if first.on:
                from report_ingest.model_first_scoring import first_calc
                before, after, arm = first_calc(
                    truth, doc, engine, report_id=report,
                    page_route=first.page_route)
            else:
                before, after = score_one_calc(truth, doc, engine,
                                               budget=budget,
                                               report_id=report)
                arm = {"model_first": "off"}
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[calc_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {calc_id}: FAILED -- "
                  f"{failures[calc_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        pages = pages_of(truth)
        blob = {
            "calc_id": calc_id,
            "report": report,
            "kind": str(truth.get("kind") or ""),
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            **engines.record(model, engine),
            **arm,
            "served_by": engine.served_by,
            "pages": pages,
            "n_pages": len(pages),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
            # Which arm of the every-page-pictures switch this run read
            # in (RI-17's GEOTECH_INGEST_CALC_PAGE_IMAGES, off by
            # default), as the reader reported it; a run whose reading
            # failed is put in the arm it was asked to run in (RI-67).
            "page_images": (after.page_images
                            if after.page_images is not None
                            else page_images_enabled()),
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[calc_id] = blob
        gain = after.total.found - before.total.found
        print(f"  [{n}/{len(truths)}] {calc_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    stage_model, fields = engines.stage(model)
    return {**_score_calc(done, failures, stage_model, openset),
            **fields, **_first_results(done.values())}


def _calc_totals(rows: Sequence[dict], stage: str
                 ) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of calculations."""
    from report_ingest.calc_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _score_calc(done: Dict[str, dict], failures: Dict[str, str], model: str,
                openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "calcs": [r["calc_id"] for r in group],
            "n_calcs": len(group),
            "before": _calc_totals(group, "before"),
            "after": _calc_totals(group, "after"),
        }
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "?" for r in rows}):
        group = [r for r in rows if (r.get("kind") or "?") == kind]
        by_kind[kind] = {
            "n_calcs": len(group),
            "n_pages": sum(int(r.get("n_pages") or 0) for r in group),
            "before": _calc_totals(group, "before"),
            "after": _calc_totals(group, "after"),
        }
    cost = _cost_totals()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_calcs": len(rows),
        "n_pages": sum(int(r.get("n_pages") or 0) for r in rows),
        "failures": dict(failures),
        # The every-page-pictures arm(s) these runs read in (RI-67).
        "page_images": _arms(r.get("page_images") for r in rows),
        "sets": sets,
        "kinds": by_kind,
        "per_calc": [{
            "calc_id": r["calc_id"], "set": r["set"],
            "kind": r.get("kind") or "?",
            "kind_read": r["after"].get("kind_read") or "",
            "n_pages": int(r.get("n_pages") or 0),
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "floor_values": r["after"].get("floor_values", 0),
            "model_calls": r["after"].get("model_calls", 0),
            "tool_calls": r["after"].get("tool_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "misplaced": r["after"].get("misplaced", 0),
            "disagreements": r["after"].get("disagreements", 0),
            "kept": r["after"].get("kept", 0),
            "added": r["after"].get("added", 0),
            "reconciled": r["after"].get("reconciled", 0),
            "page_images": r.get("page_images"),
            "error": r["after"].get("error"),
            "tokens_in": tokens_in(r["cost"]),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_calc(calc: Dict[str, Any]) -> List[str]:
    """The calc stage, as tables carrying IDs, kinds, counts and rates only."""
    from report_ingest.calc_scoring import (
        METRICS, MODEL_ONLY, NAME_RATIO, PROGRAM_RATIO, VALUE_TOL,
    )

    out: List[str] = [
        "", "# WP5 on the cluster: the calculation reader", "",
        f"Run {calc['date']}. Model `{calc['model']}`, "
        f"{calc['n_calcs']} hand-truthed run(s) over {calc['n_pages']} "
        f"page(s).",
        "",
        f"**before** is the FLOOR alone -- every (label, value) the pages' "
        f"tables and lines state, read by pattern with no model at all; "
        f"**after** is what the reader's record holds. A pattern cannot say "
        f"what a calculation works out, what method it used or what it is "
        f"for, so "
        f"{' and '.join('`' + m + '`' for m in MODEL_ONLY)} have no before "
        f"column; `program` does, because a banner is a pattern. A value "
        f"matches when its printed LABEL matches at partial ratio "
        f"{NAME_RATIO:g}, or every word of the shorter label is among the "
        f"other's in any order with the same numbers in both (Greek letters "
        f"spelled as their names), and its "
        f"VALUE within {VALUE_TOL:.0%} or the last printed digit, compared "
        f"in SI wherever both units convert to the same kind; a program "
        f"name matches at {PROGRAM_RATIO:g}.",
    ]
    out += _engine_lines(calc)
    out += _first_lines(calc)
    if calc.get("served_by"):
        out.append(f"Served by: {', '.join(calc['served_by'])}.")
    if calc.get("page_images"):
        out.append(_arm_line("Every page as a picture (RI-17, "
                             "GEOTECH_INGEST_CALC_PAGE_IMAGES)",
                             calc["page_images"], "runs"))
    if calc.get("failures"):
        out += ["", "**Runs that failed and are NOT in any number below:**"]
        out += [f"- {calc_id}: {why}"
                for calc_id, why in calc["failures"].items()]
    openset = calc.get("open_set") or []
    out += ["", (f"Open set (the reports whose calculation pages were looked "
                 f"at): {', '.join(openset)}. Everything else is blind."
                 if openset else
                 "**There is no blind set.** Every hand-truthed run was read "
                 "while this reader's prompt was written, so every number "
                 "below is an IN-SAMPLE number and none of them is evidence "
                 "about an unseen report. A blind set is owed.")]

    for name in ("open", "blind", "all"):
        row = calc["sets"].get(name)
        if not row:
            continue
        out += ["", f"## {name} -- {row['n_calcs']} run(s)", "", "```",
                f"{'metric':<10}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not after or not (before["total"] or after["total"]):
                continue
            shown = "-" if metric in MODEL_ONLY else _rate(before)
            out.append(f"{metric:<10}{shown:>14}{_rate(after):>14}")
        before_all = {"found": sum(v["found"] for k, v in row["before"].items()
                                   if k not in MODEL_ONLY),
                      "total": sum(v["total"] for k, v in row["before"].items()
                                   if k not in MODEL_ONLY)}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out += [f"{'OVERALL':<10}{_rate(before_all):>14}"
                f"{_rate(after_all):>14}", "```",
                "", "The before OVERALL leaves out the three metrics the "
                "floor is not asked, so the two columns count different "
                "things: read the metrics."]

    out += ["", "## Per kind", "", "```",
            f"{'kind':<28}{'runs':>6}{'pages':>7}{'before':>14}{'after':>14}"]
    for kind, row in calc.get("kinds", {}).items():
        before_all = {"found": sum(v["found"] for k, v in row["before"].items()
                                   if k not in MODEL_ONLY),
                      "total": sum(v["total"] for k, v in row["before"].items()
                                   if k not in MODEL_ONLY)}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out.append(f"{kind:<28}{row['n_calcs']:>6}{row['n_pages']:>7}"
                   f"{_rate(before_all):>14}{_rate(after_all):>14}")
    out.append("```")

    out += ["", "## Per run", "", "```",
            f"{'calculation':<40}{'set':<7}{'pp':>4}{'before':>12}"
            f"{'after':>12}{'read as':<28}{'floor':>6}{'calls':>6}"
            f"{'zoom':>5}{'unres':>6}{'misp':>5}{'dis':>4}{'kept':>5}"
            f"{'add':>4}{'rec':>4}{'in':>9}{'out':>7}{'s':>6}"]
    for r in calc["per_calc"]:
        if r.get("error"):
            out.append(f"{r['calc_id']:<40}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:60]}")
            continue
        read_as = r["kind_read"] if r["kind_read"] != r["kind"] else "="
        out.append(
            f"{r['calc_id']:<40}{r['set']:<7}{r['n_pages']:>4}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{read_as:<28}{r['floor_values']:>6}{r['model_calls']:>6}"
            f"{r['tool_calls']:>5}{r['unresolved']:>6}{r['misplaced']:>5}"
            f"{r['disagreements']:>4}{r['kept']:>5}{r['added']:>4}"
            f"{r['reconciled']:>4}{r['tokens_in']:>9,}"
            f"{r['output_tokens']:>7}{r['seconds']:>6.0f}")
    out.append("```")
    out += ["", "`read as` is the kind the reader chose where it differs "
                "from the hand's. `floor` is how many labelled values the "
                "pattern found before any call. `zoom` is how many times the "
                "reader magnified a plotted result. `unres` is what it could "
                "not settle plus what Python refused -- a result printed on "
                "no page, a kind outside the list, a unit with no "
                "conversion. `misp` is a truth value found in the OTHER "
                "list, an input filed as a result or the reverse; it counts "
                "as found. `dis`, `kept`, `add` and `rec` are the merge: "
                "where the floor and the reader split, what was kept from "
                "the floor because the reader did not return it, what the "
                "reader added, and what both gave.", ""]
    cost = calc["cost"]
    n = max(1, calc["n_calcs"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per calculation: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, calc)]
    return out


# ---------------------------------------------------------------------------
# the soundings stage (WP6): test pits, cone soundings and dynamic probes
# ---------------------------------------------------------------------------

#: Every hand-truthed sounding and pit was read while these readers were
#: being written, so with no ``OPEN.txt`` beside the truth files there is no
#: blind set at all and the scorecard says so rather than printing a blind
#: column of nothing.
DEFAULT_OPEN_SOUNDINGS: Tuple[str, ...] = ()


def _run_soundings(corpus: Corpus, prompter: Any, model: str, out: Path,
                   truth_dir: Path, *, budget: int, redo: bool,
                   max_sheets: Optional[int],
                   open_reports: Optional[Sequence[str]],
                   sync: Optional[Any] = None) -> Dict[str, Any]:
    """The floor and then the reader over every hand-truthed pit and sounding.

    Restartable the same way the other stages are: each sheet writes
    ``soundings/<id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter
    from report_ingest.sounding_scoring import (
        kind_of, pages_of, report_of, score_one_sounding,
    )

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed soundings at {truth_dir}; the 'soundings' "
            f"stage scores against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_sheets:
        truths = truths[:int(max_sheets)]
    openset = _open_set(truth_dir, open_reports) if (
        open_reports or (truth_dir / "OPEN.txt").is_file()) \
        else DEFAULT_OPEN_SOUNDINGS
    print(f"  soundings: {len(truths)} hand-truthed sheet(s); open set "
          f"{', '.join(openset) or '(none named)'}")

    engines = _Engines.of(prompter)
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        sheet_id = str(truth.get("id") or f"sounding{n}")
        report = report_of(truth)
        kind = kind_of(truth)
        run_file = out / "soundings" / f"{sheet_id}.json"
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            failed = _saved_failure(blob)
            if failed:
                print(f"  [{n}/{len(truths)}] {sheet_id}: previous attempt "
                      f"failed ({failed[:90]}); retrying")
            else:
                blob["set"] = "open" if report in openset else "blind"
                done[sheet_id] = blob
                print(f"  [{n}/{len(truths)}] {sheet_id}: already done, "
                      "skipping")
                continue
        if report not in set(corpus.present_ids()):
            failures[sheet_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {sheet_id}: skipped -- "
                  f"{failures[sheet_id]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            before, after = score_one_sounding(truth, doc, engine,
                                               budget=budget,
                                               report_id=report)
            sources = _page_sources(doc, pages_of(truth))
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[sheet_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {sheet_id}: FAILED -- "
                  f"{failures[sheet_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        pages = pages_of(truth)
        blob = {
            "sounding_id": sheet_id,
            "report": report,
            "kind": kind,
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            **engines.record(model, engine),
            "served_by": engine.served_by,
            "pages": pages,
            "n_pages": len(pages),
            "tabulated": bool(after.tabulated or before.tabulated),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
            # What the visual-scales voters could act on (BACKLOG RI-05).
            "visual_scales_on": visual_scales_enabled(),
            "page_sources": sources,
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[sheet_id] = blob
        gain = after.total.found - before.total.found
        print(f"  [{n}/{len(truths)}] {sheet_id} ({kind}, "
              f"{'tabulated' if blob['tabulated'] else 'plotted'}): "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} ({gain:+d}), "
              f"{after.model_calls} model call(s), "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    stage_model, fields = engines.stage(model)
    return {**_score_soundings(done, failures, stage_model, openset),
            **fields}


def _sounding_totals(rows: Sequence[dict], stage: str
                     ) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of sheets."""
    from report_ingest.sounding_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _score_soundings(done: Dict[str, dict], failures: Dict[str, str],
                     model: str, openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "soundings": [r["sounding_id"] for r in group],
            "n_sheets": len(group),
            "before": _sounding_totals(group, "before"),
            "after": _sounding_totals(group, "after"),
        }
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "?" for r in rows}):
        group = [r for r in rows if (r.get("kind") or "?") == kind]
        by_kind[kind] = {
            "n_sheets": len(group),
            "n_pages": sum(int(r.get("n_pages") or 0) for r in group),
            "before": _sounding_totals(group, "before"),
            "after": _sounding_totals(group, "after"),
        }
    # The one split that decides how to read every number above: a tabulated
    # sheet's floor already holds the series and a plotted one's holds none.
    by_shape: Dict[str, Any] = {}
    for shape, wanted in (("tabulated", True), ("plotted", False)):
        group = [r for r in rows if bool(r.get("tabulated")) is wanted]
        if not group:
            continue
        by_shape[shape] = {
            "n_sheets": len(group),
            "before": _sounding_totals(group, "before"),
            "after": _sounding_totals(group, "after"),
        }
    cost = _cost_totals()
    visual = _visual_counts()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
        _add_visual(visual, row)
    return {
        "date": date.today().isoformat(),
        "visual_scales": visual,
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_sheets": len(rows),
        "n_pages": sum(int(r.get("n_pages") or 0) for r in rows),
        "failures": dict(failures),
        "sets": sets,
        "kinds": by_kind,
        "shapes": by_shape,
        "per_sounding": [{
            "sounding_id": r["sounding_id"], "set": r["set"],
            "kind": r.get("kind") or "?",
            "shape": "tabulated" if r.get("tabulated") else "plotted",
            "n_pages": int(r.get("n_pages") or 0),
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "floor_points": r["after"].get("floor_points", 0),
            "model_calls": r["after"].get("model_calls", 0),
            "tool_calls": r["after"].get("tool_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "disagreements": r["after"].get("disagreements", 0),
            "kept": r["after"].get("kept", 0),
            "added": r["after"].get("added", 0),
            "reconciled": r["after"].get("reconciled", 0),
            "error": r["after"].get("error"),
            "tokens_in": tokens_in(r["cost"]),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_soundings(soundings: Dict[str, Any]) -> List[str]:
    """The soundings stage, as tables of IDs, kinds, counts and rates only."""
    from report_ingest.sounding_scoring import (
        CHANNEL_TOL, DEPTH_TOL, METRICS, U2_TOL,
    )

    out: List[str] = [
        "", "# WP6 on the cluster: test pits, cone soundings and dynamic "
            "probes", "",
        f"Run {soundings['date']}. Model `{soundings['model']}`, "
        f"{soundings['n_sheets']} hand-truthed sheet(s) over "
        f"{soundings['n_pages']} page(s).",
        "",
        f"**before** is the FLOOR alone -- the log grid on a pit, and on a "
        f"sounding either the sheet's own TABULATED table or, where it "
        f"plots its traces, nothing at all but the axis ranges. **after** is "
        f"what the reader's record holds. A truth point counts as found when "
        f"the reader has a reading within {DEPTH_TOL:g} m of that depth and "
        f"its value is within {CHANNEL_TOL:.0%} or one axis tick (tip "
        f"resistance, sleeve friction, the printed index), {U2_TOL:.0%} "
        f"(pore pressure), or EXACTLY (a blow count, which is a count).",
    ]
    out += _engine_lines(soundings)
    if soundings.get("served_by"):
        out.append(f"Served by: {', '.join(soundings['served_by'])}.")
    if soundings.get("failures"):
        out += ["", "**Sheets that failed and are NOT in any number below:**"]
        out += [f"- {sheet}: {why}"
                for sheet, why in soundings["failures"].items()]
    openset = soundings.get("open_set") or []
    out += ["", (f"Open set (the reports whose sounding pages were looked "
                 f"at): {', '.join(openset)}. Everything else is blind."
                 if openset else
                 "**There is no blind set.** Every hand-truthed sheet was "
                 "read while these readers were written, so every number "
                 "below is an IN-SAMPLE number and none of them is evidence "
                 "about an unseen report. A blind set is owed.")]

    for name in ("open", "blind", "all"):
        row = soundings["sets"].get(name)
        if not row:
            continue
        out += ["", f"## {name} -- {row['n_sheets']} sheet(s)", "", "```",
                f"{'metric':<12}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before = row["before"].get(metric)
            after = row["after"].get(metric)
            if not after or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<12}{_rate(before):>14}{_rate(after):>14}")
        out += [f"{'OVERALL':<12}"
                f"{_rate(_sum_of(row['before'])):>14}"
                f"{_rate(_sum_of(row['after'])):>14}", "```"]

    if soundings.get("shapes"):
        out += ["", "## Tabulated against plotted", "", "```",
                f"{'shape':<12}{'sheets':>8}{'before':>14}{'after':>14}"]
        for shape, row in soundings["shapes"].items():
            out.append(f"{shape:<12}{row['n_sheets']:>8}"
                       f"{_rate(_sum_of(row['before'])):>14}"
                       f"{_rate(_sum_of(row['after'])):>14}")
        out += ["```", "",
                "A TABULATED sheet's floor is the table, so its before "
                "column is a real baseline and the reader's job is the "
                "header. A PLOTTED sheet's floor holds no points at all, so "
                "its before column is near zero by construction and its "
                "after column IS the digitising. Averaging the two would "
                "hide the only number anyone wants."]

    out += ["", "## Per kind", "", "```",
            f"{'kind':<12}{'sheets':>8}{'pages':>7}{'before':>14}"
            f"{'after':>14}"]
    for kind, row in soundings.get("kinds", {}).items():
        out.append(f"{kind:<12}{row['n_sheets']:>8}{row['n_pages']:>7}"
                   f"{_rate(_sum_of(row['before'])):>14}"
                   f"{_rate(_sum_of(row['after'])):>14}")
    out.append("```")

    out += ["", "## Per sheet", "", "```",
            f"{'sheet':<34}{'set':<7}{'kind':<9}{'shape':<11}{'pp':>4}"
            f"{'before':>12}{'after':>12}{'floor':>7}{'calls':>6}{'zoom':>5}"
            f"{'unres':>6}{'dis':>4}{'kept':>5}{'add':>4}{'rec':>4}"
            f"{'in':>9}{'out':>7}{'s':>6}"]
    for r in soundings["per_sounding"]:
        if r.get("error"):
            out.append(f"{r['sounding_id']:<34}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:60]}")
            continue
        out.append(
            f"{r['sounding_id']:<34}{r['set']:<7}{r['kind']:<9}"
            f"{r['shape']:<11}{r['n_pages']:>4}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{r['floor_points']:>7}{r['model_calls']:>6}"
            f"{r['tool_calls']:>5}{r['unresolved']:>6}"
            f"{r['disagreements']:>4}{r['kept']:>5}{r['added']:>4}"
            f"{r['reconciled']:>4}{r['tokens_in']:>9,}"
            f"{r['output_tokens']:>7}{r['seconds']:>6.0f}")
    out.append("```")
    out += ["", "`floor` is how many points the deterministic pass placed "
                "before any call -- the tabulated rows, or zero on a plotted "
                "sheet. `zoom` is how many times the reader magnified a "
                "plotted panel. `unres` is what it could not settle plus "
                "what Python refused: a depth outside the sheet's own depth "
                "axis, a channel value outside its printed range, a "
                "negative tip resistance. `dis`, `kept`, `add` and `rec` are "
                "the merge.", ""]
    out += _visual_lines(soundings.get("visual_scales") or {})
    cost = soundings["cost"]
    n = max(1, soundings["n_sheets"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per sheet: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, soundings)]
    return out


def _sum_of(totals: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Every metric of one column added together."""
    return {"found": sum(v["found"] for v in totals.values()),
            "total": sum(v["total"] for v in totals.values())}


# ---------------------------------------------------------------------------
# the narrative stage (WP4)
# ---------------------------------------------------------------------------

#: The reports whose narrative answers were written with the reader's output
#: in view, when the truth folder carries no OPEN.txt. Everything else is
#: blind, and the blind figure is the one that means anything.
DEFAULT_OPEN_NARRATIVE: Tuple[str, ...] = ("R36", "R05")


def _run_narrative(corpus: Corpus, prompter: Any, model: str, out: Path,
                   truth_dir: Path, *, budget: int, redo: bool,
                   max_reports: Optional[int],
                   open_reports: Optional[Sequence[str]],
                   sync: Optional[Any] = None,
                   front_pages: int = 50,
                   first: Optional[_FirstArm] = None) -> Dict[str, Any]:
    """The narrative reader over every report that has a hand answer.

    The stage reads the truth files that are PRESENT and skips every report
    without one: the hand answers arrive a few reports at a time, and a stage
    that failed on the ones not yet written would be unusable until the last
    one was.

    Restartable the same way the other stages are: each report writes
    ``narrative/<ID>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter
    from report_ingest.narrative_scoring import score_one_report

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand answers at {truth_dir}; the 'narrative' stage scores "
            f"against them and cannot run without them")
    truths: List[Tuple[str, dict]] = []
    for path in sorted(truth_dir.glob("*.json")):
        rid = path.stem
        try:
            truths.append((rid, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_reports:
        truths = truths[:int(max_reports)]
    openset = (tuple(open_reports) if open_reports
               else _open_set_or(truth_dir, DEFAULT_OPEN_NARRATIVE))
    present = set(corpus.present_ids())
    print(f"  narrative: {len(truths)} hand-answered report(s); open set "
          f"{', '.join(openset)}")

    engines = _Engines.of(prompter)
    first = first or _FirstArm()
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, (rid, truth) in enumerate(truths, 1):
        run_file = out / "narrative" / f"{rid}.json"
        clash = _arm_clash(run_file, first)
        if clash:
            failures[rid] = clash
            print(f"  [{n}/{len(truths)}] {rid}: skipped -- {clash}")
            continue
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            failed = _saved_failure(blob)
            if failed:
                print(f"  [{n}/{len(truths)}] {rid}: previous attempt "
                      f"failed ({failed[:90]}); retrying")
            else:
                # The open/blind split is decided at SCORING time, never
                # frozen into a run file, for the same reason the lab stage
                # does it.
                blob["set"] = "open" if rid in openset else "blind"
                done[rid] = blob
                print(f"  [{n}/{len(truths)}] {rid}: already done, skipping")
                continue
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {rid}: skipped -- {failures[rid]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            if first.on:
                from report_ingest.model_first_scoring import first_narrative
                score, arm = first_narrative(
                    truth, doc, engine, report=rid,
                    page_route=first.page_route, front_pages=front_pages)
            else:
                score = score_one_report(truth, doc, engine, budget=budget,
                                         report=rid, front_pages=front_pages)
                arm = {"model_first": "off"}
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {rid}: FAILED -- {failures[rid]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "id": rid,
            "run_date": date.today().isoformat(),
            "set": "open" if rid in openset else "blind",
            **engines.record(model, engine),
            **arm,
            "served_by": engine.served_by,
            "score": score.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[rid] = blob
        recall = score.recall
        print(f"  [{n}/{len(truths)}] {rid}: recall "
              f"{recall.found}/{recall.total}, precision "
              f"{score.precision.found}/{score.precision.total}, "
              f"{score.model_calls} model call(s), "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    stage_model, fields = engines.stage(model)
    return {**_score_narrative(done, failures, stage_model, openset),
            **fields, **_first_results(done.values())}


def _open_set_or(truth_dir: Path,
                 fallback: Sequence[str]) -> Tuple[str, ...]:
    """The reports whose answers were written with the output in view."""
    open_file = truth_dir / "OPEN.txt"
    if open_file.is_file():
        names = tuple(line.strip() for line
                      in open_file.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return tuple(fallback)


def _narrative_totals(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """The three numbers, the kinds and the summaries, summed over reports."""
    from report_ingest.narrative_scoring import KINDS

    names = ["recall", "precision", "agreement"]
    out: Dict[str, Dict[str, int]] = {
        name: {"found": 0, "total": 0} for name in names}
    for kind in KINDS:
        out[f"kind.{kind}"] = {"found": 0, "total": 0}
    for name in ("list_items.precision", "list_items.recall",
                 "summaries.present", "summaries.within_limit"):
        out[name] = {"found": 0, "total": 0}
    for row in rows:
        score = row.get("score") or {}
        for name in names:
            part = score.get(name) or {}
            out[name]["found"] += int(part.get("found") or 0)
            out[name]["total"] += int(part.get("total") or 0)
        for kind, part in (score.get("by_kind") or {}).items():
            key = f"kind.{kind}"
            out.setdefault(key, {"found": 0, "total": 0})
            out[key]["found"] += int(part.get("found") or 0)
            out[key]["total"] += int(part.get("total") or 0)
        for group in ("list_items", "summaries"):
            for name, part in (score.get(group) or {}).items():
                key = f"{group}.{name}"
                out.setdefault(key, {"found": 0, "total": 0})
                out[key]["found"] += int(part.get("found") or 0)
                out[key]["total"] += int(part.get("total") or 0)
    return out


def _field_table(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """``field -> {right, asked, missed, invented, wrong}`` over the reports.

    The per-field table is what says which QUESTIONS are hard, which is the
    thing a prompt change is actually aimed at. A field nobody hand-answered
    shows an asked of 0 and is not a failure.
    """
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        for entry in ((row.get("score") or {}).get("fields") or []):
            name = entry.get("field")
            if not name:
                continue
            cell = out.setdefault(name, {"right": 0, "asked": 0, "missed": 0,
                                         "invented": 0, "wrong": 0})
            verdict = entry.get("verdict")
            if verdict in ("missed", "wrong", "right", "too_long"):
                cell["asked"] += 1
            if verdict == "right":
                cell["right"] += 1
            elif verdict in ("missed", "invented", "wrong"):
                cell[verdict] += 1
    return out


def _score_narrative(done: Dict[str, dict], failures: Dict[str, str],
                     model: str, openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {"reports": [r["id"] for r in group],
                      "n_reports": len(group),
                      "totals": _narrative_totals(group)}
    cost = _cost_totals()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_reports": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "fields": _field_table(rows),
        "per_report": [{
            "id": r["id"], "set": r["set"],
            "recall": (r["score"].get("recall") or {}),
            "precision": (r["score"].get("precision") or {}),
            "agreement": (r["score"].get("agreement") or {}),
            "model_calls": r["score"].get("model_calls", 0),
            "unresolved": r["score"].get("unresolved", 0),
            "error": r["score"].get("error"),
            "tokens_in": tokens_in(r["cost"]),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_narrative(narrative: Dict[str, Any]) -> List[str]:
    """The narrative stage, as tables carrying IDs, counts and rates only."""
    from report_ingest.narrative_scoring import (
        FUZZY_RATIO, KINDS, LIST_HIT_JACCARD,
    )

    out: List[str] = [
        "", f"# WP4 on the cluster: the narrative reader through "
            f"{_through(narrative)}", "",
        f"Run {narrative['date']}. Model `{narrative['model']}`, "
        f"{narrative['n_reports']} hand-answered report(s).",
        "",
        "**recall** is of the questions the report DOES answer, how many came "
        "back right; **precision** is of the answers the reader gave, how "
        "many were right; **agreement** counts every field including the ones "
        "both sides left null, and flatters. Enumerations and counts are "
        f"exact; a string matches at a partial ratio of {FUZZY_RATIO} or on a "
        f"shared proper noun; a list at a Jaccard overlap of "
        f"{LIST_HIT_JACCARD}; the four prose summaries are scored for "
        "presence and word limit only, because whether a summary is a good "
        "summary is a person's call.",
    ]
    out += _engine_lines(narrative)
    out += _first_lines(narrative)
    if narrative.get("served_by"):
        out.append(f"Served by: {', '.join(narrative['served_by'])}.")
    if narrative.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number "
                    "below:**"]
        out += [f"- {rid}: {why}"
                for rid, why in narrative["failures"].items()]
    out += ["", f"Open set (answers written with the output in view): "
                f"{', '.join(narrative['open_set'])}. Everything else is "
                f"blind."]

    for name in ("open", "blind", "all"):
        row = narrative["sets"].get(name)
        if not row:
            continue
        totals = row["totals"]
        out += ["", f"## {name} -- {row['n_reports']} report(s)", "", "```",
                f"{'metric':<26}{'rate':>16}"]
        for metric in ("recall", "precision", "agreement"):
            out.append(f"{metric:<26}{_rate(totals[metric]):>16}")
        out.append("")
        for kind in KINDS:
            cell = totals.get(f"kind.{kind}")
            if cell and cell["total"]:
                out.append(f"{'  recall, ' + kind:<26}{_rate(cell):>16}")
        for label, key in (("list items, precision", "list_items.precision"),
                           ("list items, recall", "list_items.recall"),
                           ("summaries written", "summaries.present"),
                           ("summaries in limit",
                            "summaries.within_limit")):
            cell = totals.get(key)
            if cell and cell["total"]:
                out.append(f"{'  ' + label:<26}{_rate(cell):>16}")
        out.append("```")

    fields = narrative.get("fields") or {}
    asked = {name: cell for name, cell in fields.items() if cell["asked"]}
    if asked:
        out += ["", "## Per question", "", "```",
                f"{'field':<30}{'asked':>7}{'right':>7}{'missed':>8}"
                f"{'wrong':>7}{'invented':>10}"]
        for name in sorted(asked, key=lambda k: (-asked[k]["asked"], k)):
            cell = asked[name]
            out.append(f"{name:<30}{cell['asked']:>7}{cell['right']:>7}"
                       f"{cell['missed']:>8}{cell['wrong']:>7}"
                       f"{cell['invented']:>10}")
        out.append("```")

    out += ["", "## Per report", "", "```",
            f"{'report':<8}{'set':<7}{'recall':>14}{'precision':>14}"
            f"{'calls':>7}{'unres':>7}{'in':>10}{'out':>8}{'s':>7}"]
    for r in narrative["per_report"]:
        if r.get("error"):
            out.append(f"{r['id']:<8}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:50]}")
            continue
        out.append(
            f"{r['id']:<8}{r['set']:<7}{_rate(r['recall']):>14}"
            f"{_rate(r['precision']):>14}{r['model_calls']:>7}"
            f"{r['unresolved']:>7}{r['tokens_in']:>10,}"
            f"{r['output_tokens']:>8,}{r['seconds']:>7.0f}")
    out.append("```")

    cost = narrative["cost"]
    n = max(1, narrative["n_reports"])
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, narrative)]
    return out


def _totals(rows: Sequence[dict], stage: str) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of logs."""
    from report_ingest.log_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _alone_totals(rows: Sequence[dict], key: str, metrics: Sequence[str]
                  ) -> Tuple[Dict[str, Dict[str, int]], int]:
    """``(metric -> {found, total}, how many rows carry it)`` for one of the
    scores kept beside ``after`` (``floor_alone``, ``model_alone``,
    ``unlinked``), summed over the rows whose run file has it — a run file
    written before a score was kept simply does not count towards it."""
    out = {m: {"found": 0, "total": 0} for m in metrics}
    have = [r for r in rows if isinstance((r.get("after") or {}).get(key),
                                          dict)]
    for row in have:
        for metric, score in (row["after"][key].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out, len(have)


def _overall_of(row: dict, key: str) -> Optional[Dict[str, Any]]:
    got = (row.get("after") or {}).get(key)
    return got.get("overall") if isinstance(got, dict) else None


def _opt_rate(row: Optional[Dict[str, int]]) -> str:
    return "     -" if not row else _rate(row)


def _score_logs(done: Dict[str, dict], failures: Dict[str, str], model: str,
                openset: Sequence[str]) -> Dict[str, Any]:
    from report_ingest.log_scoring import METRICS
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        floor, n_floor = _alone_totals(group, "floor_alone", METRICS)
        alone, n_model = _alone_totals(group, "model_alone", METRICS)
        sets[name] = {
            "logs": [r["log_id"] for r in group],
            "n_logs": len(group),
            "before": _totals(group, "before"),
            "after": _totals(group, "after"),
            "floor_alone": floor, "n_floor": n_floor,
            "model_alone": alone, "n_model": n_model,
        }
    cost = _cost_totals()
    visual = _visual_counts()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
        _add_visual(visual, row)
    return {
        "date": date.today().isoformat(),
        "visual_scales": visual,
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_logs": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "per_log": [{
            "log_id": r["log_id"], "set": r["set"],
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "floor_alone": _overall_of(r, "floor_alone"),
            "model_alone": _overall_of(r, "model_alone"),
            "model_calls": r["after"].get("model_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "tokens_in": tokens_in(r["cost"]),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _rate(row: Dict[str, int]) -> str:
    if not row["total"]:
        return "     -"
    return f"{row['found'] / row['total']:5.0%} {row['found']}/{row['total']}"


def _render_logs(logs: Dict[str, Any]) -> List[str]:
    """The logs stage, as tables that carry IDs, counts and rates only."""
    from report_ingest.log_scoring import (
        LAYER_TOL_M, METRICS, SAMPLE_TOL_M, WATER_TOL_M,
    )

    out: List[str] = [
        "", f"# WP2b on the cluster: the log reader through "
            f"{_through(logs)}", "",
        f"Run {logs['date']}. Model `{logs['model']}`, "
        f"{logs['n_logs']} hand-truthed log(s).",
        "",
        f"**before** is what `log_grid` alone recovered; **after** is what "
        f"the reader's record holds. The grid runs once and is handed to the "
        f"reader, so the difference between the two columns is the model and "
        f"nothing else. Beside them: **floor** is the floor's RECORD scored "
        f"alone -- what the merge started from (before scores the grid's "
        f"cells, which can credit a value the record never held) -- and "
        f"**model** the model's answer scored alone, before the merge. "
        f"Tolerances: samples, index values and water "
        f"{SAMPLE_TOL_M} m, layer tops {LAYER_TOL_M} m "
        f"(water {WATER_TOL_M} m); depths compared in metres whatever the "
        f"log prints. N values exact.",
    ]
    out += _engine_lines(logs)
    out += _first_lines(logs)
    if logs.get("served_by"):
        out.append(f"Served by: {', '.join(logs['served_by'])}.")
    if logs.get("failures"):
        out += ["", "**Logs that failed and are NOT in any number below:**"]
        out += [f"- {log_id}: {why}"
                for log_id, why in logs["failures"].items()]
    out += ["", f"Open set (the logs the rules were tuned on): "
                f"{', '.join(logs['open_set'])}. Everything else is blind."]

    for name in ("open", "blind", "all"):
        row = logs["sets"].get(name)
        if not row:
            continue
        floor = row.get("floor_alone") if row.get("n_floor") else None
        alone = row.get("model_alone") if row.get("n_model") else None
        out += ["", f"## {name} -- {row['n_logs']} log(s)", "", "```",
                f"{'metric':<16}{'before':>14}{'floor':>14}{'model':>14}"
                f"{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not before or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<16}{_rate(before):>14}"
                       f"{_opt_rate((floor or {}).get(metric)):>14}"
                       f"{_opt_rate((alone or {}).get(metric)):>14}"
                       f"{_rate(after):>14}")
        before_all = _sum_of(row["before"])
        after_all = _sum_of(row["after"])
        out += [f"{'OVERALL':<16}{_rate(before_all):>14}"
                f"{_opt_rate(_sum_of(floor) if floor else None):>14}"
                f"{_opt_rate(_sum_of(alone) if alone else None):>14}"
                f"{_rate(after_all):>14}", "```"]
        for key, label in (("n_floor", "floor"), ("n_model", "model")):
            if row.get(key, 0) < row["n_logs"]:
                out.append(f"`{label}` is over the {row.get(key, 0)} of "
                           f"{row['n_logs']} log(s) whose run file keeps it.")

    out += ["", "## Per log", "", "```",
            f"{'log':<14}{'set':<7}{'before':>12}{'floor':>12}{'model':>12}"
            f"{'after':>12}{'calls':>7}"
            f"{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in logs["per_log"]:
        if r.get("error"):
            out.append(f"{r['log_id']:<14}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:60]}")
            continue
        out.append(
            f"{r['log_id']:<14}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_opt_rate(r.get('floor_alone')):>12}"
            f"{_opt_rate(r.get('model_alone')):>12}{_rate(r['after']):>12}"
            f"{r['model_calls']:>7}{r['unresolved']:>7}{r['changes']:>6}"
            f"{r['tokens_in']:>10,}{r['output_tokens']:>8,}"
            f"{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`unres` is what the reader could not settle plus what "
                "Python refused (a depth off the page). `look` is how many "
                "values it took from the picture rather than the rows.", ""]
    out += _visual_lines(logs.get("visual_scales") or {})
    cost = logs["cost"]
    n = max(1, logs["n_logs"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per log: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, logs)]
    return out


# ---------------------------------------------------------------------------
# the vision stage (WP5): the same pages, labelled from their pictures
# ---------------------------------------------------------------------------

def _run_vision(corpus: Corpus, prompter: Any, model: str, out: Path,
                report_ids: Sequence[str], *, mode: str,
                dpi: Optional[float], outline_context: bool,
                redo: bool, detail: Optional[str] = None,
                window: Optional[int] = None,
                images_per_call: Optional[int] = None,
                fallback: bool = True,
                sync: Optional[Any] = None) -> Dict[str, Any]:
    """A cheap model's look at every page of every report in the set.

    The rules' labels are computed here too, deterministically and with no
    model, so the three-column table works even in a run where the label
    review never happened.

    Restartable the same way every other stage is: each report writes
    ``vision/<ID>.json`` as it finishes and a later call skips it.
    """
    from planlens.document.roles import document_outline, page_roles

    from report_ingest.engine import CostMeter
    from report_ingest.vision_labels import (
        DEFAULT_DPI, DETAIL_LEVELS, DOCUMENT_WINDOW, MAX_IMAGES_PER_CALL,
        MODES,
    )
    from report_ingest.vision_labels import (
        classify_pages_by_vision as classify,
    )

    if mode not in MODES:
        raise ValueError(
            f"unknown vision_mode {mode!r}; the modes are {list(MODES)}")
    if detail is not None and detail not in DETAIL_LEVELS:
        raise ValueError(
            f"unknown vision_detail {detail!r}; the levels are "
            f"{list(DETAIL_LEVELS)}")
    dpi = float(DEFAULT_DPI if dpi is None else dpi)
    window = int(DOCUMENT_WINDOW if window is None else window)
    cap = int(MAX_IMAGES_PER_CALL if images_per_call is None
              else images_per_call)
    # What the run USED, for the results header. A setting only a document
    # run has is recorded only for a document run, and an unset detail is
    # absent rather than written as a value the provider never saw.
    settings: Dict[str, Any] = {"mode": mode, "dpi": dpi,
                                "outline_context": bool(outline_context),
                                "fallback": bool(fallback)}
    if detail is not None:
        settings["detail"] = detail
    if mode == "document":
        settings["window"] = window
        settings["images_per_call"] = cap
    extra = "".join([
        f", detail {detail}" if detail else "",
        f", window {window} pages, at most {cap} images a call"
        if mode == "document" else "",
        ", page-mode fallback on" if fallback and mode != "page" else "",
    ])
    print(f"  vision_labels: {len(report_ids)} report(s); model {model}, "
          f"mode {mode}, {dpi:.0f} dpi, outline context "
          f"{'on' if outline_context else 'off'}{extra}")

    present = set(corpus.present_ids())
    engines = _Engines.of(prompter)
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, rid in enumerate(report_ids, 1):
        run_file = out / "vision" / f"{rid}.json"
        if run_file.is_file() and not redo:
            done[rid] = json.loads(run_file.read_text(encoding="utf-8"))
            print(f"  [{n}/{len(report_ids)}] {rid}: already done, skipping")
            continue
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(report_ids)}] {rid}: skipped -- "
                  f"{failures[rid]}")
            continue
        meter = CostMeter()
        engine = engines.build(model, meter, vision=True)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            roles = page_roles(doc)
            outline = document_outline(doc) if outline_context else None
            seen = classify(doc, engine, mode=mode, dpi=dpi,
                            outline_context=outline_context, outline=outline,
                            detail=detail, window=window,
                            images_per_call=cap, fallback=fallback)
            rules = {r.page: r.role for r in roles}
            rules_confidence = {r.page: round(float(r.confidence), 3)
                                for r in roles}
            n_pages = doc.n_pages
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(report_ids)}] {rid}: FAILED -- "
                  f"{failures[rid]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "id": rid,
            "run_date": date.today().isoformat(),
            "n_pages": n_pages,
            **engines.record(model, engine),
            "served_by": engine.served_by,
            "mode": mode,
            "dpi": dpi,
            "outline_context": bool(outline_context),
            "detail": detail,
            "fallback": bool(fallback),
            "rules_labels": {str(k): v for k, v in sorted(rules.items())},
            "rules_confidence": {str(k): v for k, v
                                 in sorted(rules_confidence.items())},
            "vision": seen.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        done[rid] = blob
        windows = seen.cost.get("windows") or 0
        splits = seen.cost.get("splits") or 0
        rescued = seen.cost.get("fallback_pages") or 0
        print(f"  [{n}/{len(report_ids)}] {rid}: {n_pages} pp, "
              f"{len(seen.labels)} labelled, {len(seen.unresolved)} "
              f"unresolved, {seen.model_calls} model calls"
              + (f" over {windows} window(s)" if windows else "")
              + (f", {splits} split(s) on request size" if splits else "")
              + (f", {rescued} page(s) rescued by the fallback"
                 if rescued else "") + ", "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return {"runs": done, "failures": failures, "settings": settings}


def _review_labels_on_disk(out: Path, rid: str) -> Dict[int, str]:
    """The reviewed labels a label run left behind, or an empty map.

    The review column exists only where the review was actually run -- in
    this call or in an earlier one that wrote the same ``out_dir``. An empty
    map means the report prints two columns rather than three, which is
    honest; inventing a third from the rules would print the rules twice.
    """
    return _review_labels_in(out / "runs", rid)


def _score_vision(done: Dict[str, dict], corpus: Corpus,
                  oos: Dict[str, Dict[int, dict]], mapped: Sequence[str],
                  sets: Sequence[str], out: Path, model: str,
                  failures: Dict[str, str],
                  settings: Dict[str, Any]) -> Dict[str, Any]:
    """The three ways of labelling a page, on one set of hand labels.

    The scorer is :class:`report_ingest.scoring.Scores` -- the same one the
    labels stage uses -- and the truth is the same spreadsheet and the same
    out-of-sample file. That is the whole point: a vision number measured by
    a scorer of its own would be a number nobody could set beside the rules.
    """
    rows: Dict[str, Any] = {}
    cost = _cost_totals()
    per_report: List[dict] = []
    counted: set = set()

    for name in sets:
        ids = [rid for rid in _set_ids(name, corpus) if rid in done]
        rules, review, vision = Scores(), Scores(), Scores()
        clean_rules, clean_review, clean_vision = Scores(), Scores(), Scores()
        with_review: List[str] = []
        dropped = 0
        for rid in ids:
            blob = done[rid]
            rule_labels = {int(k): v
                           for k, v in
                           (blob.get("rules_labels") or {}).items()}
            seen = blob.get("vision") or {}
            vision_labels = {int(k): v
                             for k, v in (seen.get("labels") or {}).items()}
            reviewed = _review_labels_on_disk(out, rid)
            if reviewed:
                with_review.append(rid)
            hand, alternates, source = _truth_for(rid, corpus, oos, mapped)
            never_seen = name == "oos_blind" and rid not in CHECKPOINT
            hits = {"rules": 0, "review": 0, "vision": 0}
            scored = 0
            for page, want in sorted(hand.items()):
                alts = alternates.get(page, ())
                was = rule_labels.get(page, "other")
                now = reviewed.get(page, "other")
                saw = vision_labels.get(page, "other")
                # The same dispute rule as the labels stage, applied to all
                # three columns at once: a page dropped from one and kept in
                # another would make the columns incomparable, which is the
                # only thing this table is for.
                if reviewed and disputed_drop(rid, page, now):
                    dropped += 1
                    continue
                rules.add(want, was, alts)
                vision.add(want, saw, alts)
                if reviewed:
                    review.add(want, now, alts)
                if never_seen:
                    clean_rules.add(want, was, alts)
                    clean_vision.add(want, saw, alts)
                    if reviewed:
                        clean_review.add(want, now, alts)
                scored += 1
                hits["rules"] += int(was == want)
                hits["review"] += int(now == want)
                hits["vision"] += int(saw == want)
            if rid not in counted:
                counted.add(rid)
                _add_cost(cost, blob["cost"])
                cost["seconds"] += blob.get("seconds", 0.0)
                per_report.append({
                    "id": rid, "set": name, "pages": blob.get("n_pages", 0),
                    "scored": scored, "truth": source,
                    "rules": (hits["rules"] / scored) if scored else None,
                    "review": ((hits["review"] / scored)
                               if scored and reviewed else None),
                    "vision": (hits["vision"] / scored) if scored else None,
                    "labelled": len(seen.get("labels") or {}),
                    "unresolved": len(seen.get("unresolved") or []),
                    "qa": len(seen.get("qa") or []),
                    "calls": blob["cost"].get("calls", 0),
                    # Windows are a document-mode count; page and sheet mode
                    # have none and print a dash rather than a 0 that would
                    # read as "it did no work".
                    "windows": (seen.get("cost") or {}).get("windows") or 0,
                    # How often the gateway refused a window's request body
                    # and the pass halved it, and how many pages one extra
                    # page-mode call rescued. Both print a dash at 0.
                    "splits": (seen.get("cost") or {}).get("splits") or 0,
                    "fallback_pages": ((seen.get("cost") or {})
                                       .get("fallback_pages") or 0),
                    "tokens_in": tokens_in(blob["cost"]),
                    "output_tokens": blob["cost"].get("output_tokens", 0),
                    "dollars": blob["cost"].get("dollars", 0.0),
                    "seconds": blob.get("seconds", 0.0),
                })
        row: Dict[str, Any] = {
            "reports": ids,
            "n_reports": len(ids),
            "n_with_review": len(with_review),
            "disputed_dropped": dropped,
            "rules": rules.to_dict(),
            "review": review.to_dict(),
            "vision": vision.to_dict(),
            "gate_failures": gate_failures(vision),
            "_scores": (rules, review, vision),
        }
        if name == "oos_blind" and clean_vision.n:
            row["never_seen"] = {
                "reports": [r for r in ids if r not in CHECKPOINT],
                "excluded": [r for r in ids if r in CHECKPOINT],
                "rules": clean_rules.to_dict(),
                "review": clean_review.to_dict(),
                "vision": clean_vision.to_dict(),
                "_scores": (clean_rules, clean_review, clean_vision),
            }
        rows[name] = row

    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in done.values()
                             if r.get("served_by")}),
        "settings": dict(settings),
        "n_reports": len(done),
        "failures": dict(failures),
        "sets": rows,
        "per_report": per_report,
        "cost": cost,
    }


def _columns(row: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """The columns this set actually has, in the order they are read."""
    rules, review, vision = row["_scores"]
    out = [("rules", rules)]
    if review.n:
        out.append(("+review", review))
    out.append(("vision", vision))
    return out


def _render_vision(vision: Dict[str, Any]) -> List[str]:
    """The WP5 table: three ways of labelling a page, one set of hand labels.

    IDs, labels, counts and rates only. A vision reason names what the model
    saw on a page and can therefore carry a firm or a project, so the reasons
    stay in ``vision/<ID>.json`` on the cluster with the review's reasons and
    the triage rationales.
    """
    settings = vision.get("settings") or {}
    mode = settings.get("mode", "page")
    extra = "".join([
        f", detail `{settings['detail']}`" if settings.get("detail") else "",
        (f", window {settings.get('window')} pages, at most "
         f"{settings.get('images_per_call')} images a call"
         if mode == "document" else ""),
    ])
    looking = ("a cheap model looking at the page and nothing else"
               if mode != "document" else
               "a cheap model looking at the page with the whole report in "
               "thumbnail beside it")
    extra += (", page-mode fallback on" if settings.get("fallback")
              else ", no fallback")
    out: List[str] = [
        "", "# WP5 on the cluster: labelling a page by looking at it", "",
        f"Run {vision['date']}. Vision model `{vision['model']}`, mode "
        f"`{mode}`, {float(settings.get('dpi') or 0):.0f} dpi, outline "
        f"context {'ON' if settings.get('outline_context') else 'off'}"
        f"{extra}. {vision['n_reports']} report(s).",
        "",
        "**rules** is what planlens' per-page rules said; **+review** is the "
        f"rules with the label review's accepted changes applied; **vision** "
        f"is {looking}. All three "
        "are scored against the SAME hand labels with the SAME scorer, and a "
        "page the vision pass left unresolved counts as `other` -- a "
        "non-answer is scored, not excused. The `+review` column appears "
        "only where a label run sits beside the vision run in this "
        "`out_dir`.",
        "",
        "Every page picture travels as **JPEG** at quality 80, at the "
        "render's own pixel size: the provider scales and tiles exactly the "
        "page it always did, so the token count is unchanged and only the "
        "bytes on the wire move. A gateway refuses a request whose BODY is "
        "too large long before the 50-image cap bites on scanned pages, and "
        "`split` is how many times a document window was halved because it "
        "did.",
    ]
    out += _engine_lines(vision)
    if vision.get("served_by"):
        out.append(f"Served by: {', '.join(vision['served_by'])}.")
    if vision.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in vision["failures"].items()]

    for name, row in vision["sets"].items():
        if not row["n_reports"]:
            continue
        columns = _columns(row)
        rules, review, vision_scores = row["_scores"]
        out += ["", f"## {name} -- {row['n_reports']} report(s), "
                    f"{vision_scores.n} scored pages", ""]
        if review.n and row["n_with_review"] < row["n_reports"]:
            out.append(f"The `+review` column covers "
                       f"{row['n_with_review']} of these reports, not all "
                       f"{row['n_reports']}; read it as that subset.")
            out.append("")
        if row["disputed_dropped"]:
            out.append(f"{row['disputed_dropped']} page(s) dropped from every "
                       f"column as confirmed disputed hand labels.")
            out.append("")
        header = f"{'':<24}" + "".join(f"{n:>10}" for n, _s in columns)
        out += ["```", header,
                f"{'strict accuracy':<24}"
                + "".join(f"{s.accuracy:>10.3f}" for _n, s in columns),
                f"{'accepting alternates':<24}"
                + "".join(f"{s.lenient_accuracy:>10.3f}"
                          for _n, s in columns),
                "",
                f"key content (the gate is {GATE:.2f} on both rates)"]
        out += columns_label_table(columns, KEY_CONTENT)
        out.append("vision below the gate: "
                   + (", ".join(row["gate_failures"]) or "none"))
        out += ["", "every label"]
        out += columns_label_table(columns)
        out.append("```")

        if name == "oos_blind":
            # The blind set is reported as a summary and nothing else: no
            # per-report line, no per-page list. A blind figure read report
            # by report stops being blind the moment somebody goes looking
            # for which report dragged it down.
            out += ["", "The blind set is reported as a summary only: no "
                        "per-report line and no page list, so it stays "
                        "blind."]
            clean = row.get("never_seen")
            if clean:
                clean_columns = [(n, s) for n, s in
                                 zip(("rules", "+review", "vision"),
                                     clean["_scores"]) if s.n]
                out += ["",
                        (f"### the same set minus "
                         f"{', '.join(clean['excluded'])}, which the cost "
                         f"checkpoint used and so are no longer blind"
                         if clean["excluded"] else
                         "### the same set, none of which any checkpoint "
                         "has used"), "",
                        f"{len(clean['reports'])} report(s) nobody has "
                        f"opened. **This is the honest blind figure.**", "",
                        "```",
                        f"{'':<24}"
                        + "".join(f"{n:>10}" for n, _s in clean_columns),
                        f"{'strict accuracy':<24}"
                        + "".join(f"{s.accuracy:>10.3f}"
                                  for _n, s in clean_columns), ""]
                out += columns_label_table(clean_columns, KEY_CONTENT)
                out.append("```")
            continue

        rows = [r for r in vision["per_report"] if r["set"] == name]
        if not rows:
            continue
        out += ["", "```",
                f"{'report':<8}{'pages':>7}{'scored':>8}{'rules':>8}"
                f"{'review':>8}{'vision':>8}{'unres':>7}{'qa':>5}"
                f"{'calls':>7}{'wins':>6}{'split':>7}{'in':>10}"
                f"{'out':>9}{'s':>7}"]
        for r in rows:
            def cell(value: Optional[float]) -> str:
                return "   --   " if value is None else f"{value:>8.3f}"
            windows = f"{r.get('windows') or 0:>6}" if r.get("windows") \
                else f"{'--':>6}"
            # A dash, not a 0: page and sheet mode have no window to split,
            # and a document run that never hit the body limit is the normal
            # case rather than a run that did no work.
            splits = f"{r.get('splits') or 0:>7}" if r.get("splits") \
                else f"{'--':>7}"
            out.append(f"{r['id']:<8}{r['pages']:>7}{r['scored']:>8}"
                       f"{cell(r['rules'])}{cell(r['review'])}"
                       f"{cell(r['vision'])}{r['unresolved']:>7}{r['qa']:>5}"
                       f"{r['calls']:>7}{windows}{splits}"
                       f"{r['tokens_in']:>10,}"
                       f"{r['output_tokens']:>9,}{r['seconds']:>7.0f}")
        out.append("```")
        out += ["", "`calls` is model calls and `wins` the windows they "
                    "covered: one window is one call in document mode and "
                    "there are none in the other two, where a call is a page "
                    "or a sheet. `split` is how many windows the gateway "
                    "refused as too large a request and the pass halved; "
                    "each split adds windows, and therefore calls, without "
                    "adding a page."]

    cost = vision["cost"]
    n = max(1, vision["n_reports"])
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls, {tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost, vision),
            "",
            "Page mode is one call per page, sheet mode one per six and document "
            "mode one per window of pages: that is the trade this stage exists "
            "to price."]
    return out


# ---------------------------------------------------------------------------
# the vote stage (WP6): the voters set against each other, with no model
# ---------------------------------------------------------------------------

#: The sets the vote reports on. ``honest_blind`` is the blind set minus the
#: reports a cost checkpoint has already read, and it is the figure that
#: means anything; it appears only when ``oos_blind`` was asked for.
VOTE_SETS: Tuple[str, ...] = ("insample", "oos_open", "oos_blind",
                              "honest_blind")


def _review_labels_in(runs_dir: Path, rid: str) -> Dict[int, str]:
    """The reviewed labels a label run left behind, or an empty map."""
    run_file = Path(runs_dir) / f"{rid}.json"
    if not run_file.is_file():
        return {}
    try:
        blob = json.loads(run_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    final = ((blob.get("review") or {}).get("final_labels")) or {}
    return {int(k): v for k, v in final.items()}


def _vote_vision_dirs(out: Path,
                      vision_dirs: Optional[Sequence[Any]]) -> List[Path]:
    """The folders of saved vision runs the vote reads, in order."""
    if vision_dirs:
        return [Path(d) for d in vision_dirs]
    return [out / "vision"]


def _vote_vision_runs(dirs: Sequence[Path]
                      ) -> Tuple[Dict[str, dict], Dict[str, Path]]:
    """``id -> blob`` and ``id -> the folder it came from``.

    The FIRST folder that holds a report wins, so passing this run's own
    ``vision`` folder ahead of an older one tops the old run up rather than
    mixing two answers for the same report.
    """
    runs: Dict[str, dict] = {}
    source: Dict[str, Path] = {}
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            rid = path.stem
            if rid in runs:
                continue
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not (blob.get("vision") or {}).get("labels"):
                continue
            runs[rid] = blob
            source[rid] = folder
    return runs, source


def _vote_rules(rid: str, corpus: Corpus, blob: dict, present: set
                ) -> Tuple[Dict[int, str], Dict[int, float], str]:
    """The RULE labels for one report, with their confidences and where from.

    THE SAVED RUN WINS WHERE IT IS COMPLETE. A run file written by 5.22.0 or
    later carries ``rules_confidence`` -- planlens' own per-page confidence --
    beside the labels, and re-deriving what is already on disk would cost
    every report a PDF open for nothing.

    A run file written BEFORE that has the labels and no confidence, and the
    ``confidence`` policy needs it, so planlens is asked again: the rules are
    deterministic and free, and the answer is the same one that produced the
    saved labels. Where the report is not on this machine, or will not open,
    the saved labels are used with no confidence at all -- the vote is meant
    to run over run files that already exist, and refusing to vote because a
    PDF moved would defeat it.
    """
    labels = {int(k): v for k, v in (blob.get("rules_labels") or {}).items()}
    confidence = {int(k): float(v) for k, v
                  in (blob.get("rules_confidence") or {}).items()}
    if labels and confidence:
        return labels, confidence, "saved run"
    if rid in present:
        doc = None
        try:
            from planlens.document.roles import page_roles

            doc = corpus.open_report(rid, di="auto", warn=False)
            roles = page_roles(doc)
            return ({r.page: r.role for r in roles},
                    {r.page: float(r.confidence) for r in roles},
                    "planlens")
        except Exception:
            pass                       # fall through to what was saved
        finally:
            if doc is not None:
                doc.close()
    return labels, confidence, "saved run, no confidence"


# ---------------------------------------------------------------------------
# range_labels: every page described, then labelled by sections (RI-73)
# ---------------------------------------------------------------------------

def _run_ranges(corpus: Corpus, prompter: Any, describe_model: str,
                range_model: str, out: Path, report_ids: Sequence[str], *,
                arms: Sequence[str], workers: int, redo: bool,
                sync: Optional[Any] = None,
                deadline: Optional[float] = None,
                conventions: str = "A") -> Dict[str, Any]:
    """The range voter on every report, each arm, written per report.

    One vision call a page on ``describe_model`` (the vision engine) makes
    the descriptions, saved one file a page under
    ``ranges/descriptions/<ID>/`` and shared by the arms; then one call a
    report (or a window) on ``range_model`` (the readers' engine) labels
    them by sections, per arm, into ``ranges/<arm>/<ID>.json`` -- a folder
    of label files ``module_work/report_ingest_harness/
    measure_range_labels.py --runs`` reads as a column. The rules' labels
    are computed here too, with no model.

    Restartable at two grains: a report whose every arm file is finished
    (:func:`_finished_range_run`: the same models, the same describer
    prompt, the same labeller instructions) is skipped, and a description
    already on disk is not asked for again. While a report is described,
    ``sync`` mirrors what is saved every :data:`RANGE_MIRROR_EVERY` pages,
    so a restart loses few paid descriptions.
    ``conventions`` names the labeller's conventions
    (:data:`report_ingest.range_labels.CONVENTIONS`): ``A``'s runs are in
    ``ranges/<arm>/``, any other's in ``ranges/<arm>_<name>/``, and their
    columns are named the same way, so two conventions never share a file.
    ``deadline`` (a ``time.time()`` value) stops the describer sending
    pages; a report it stops in is a failure that the same call again
    goes on with. Each describing pass's cost record -- with what the engine
    said about every answer it did not take -- is kept under
    ``ranges/describe_meters/``.
    """
    from planlens.document.roles import document_outline, page_roles

    from report_ingest.engine import CostMeter
    from report_ingest.range_labels import (
        ARMS, DESCRIBE_PROMPT_VERSION, conventions_named,
        conventions_version, describe_pages, label_by_ranges,
        labeller_fingerprint,
    )
    from report_ingest.triage import outline_text

    bad = [arm for arm in arms if arm not in ARMS]
    if bad or not arms:
        raise ValueError(f"range_arms must be some of {list(ARMS)}, not "
                         f"{list(arms)}")
    conventions = str(conventions).strip().upper()
    text, not_used = conventions_named(conventions)
    column = {arm: _range_column(arm, conventions) for arm in arms}
    print(f"  range_labels: {len(report_ids)} report(s); describer "
          f"{describe_model}, labeller {range_model}, arm(s) "
          f"{', '.join(arms)}, conventions {conventions}, {int(workers)} "
          f"page(s) at a time")
    for arm in arms:
        (out / "ranges" / column[arm]).mkdir(parents=True, exist_ok=True)
    present = set(corpus.present_ids())
    engines = _Engines.of(prompter)
    done: Dict[str, Dict[str, dict]] = {column[arm]: {} for arm in arms}
    failures: Dict[str, str] = {}
    for n, rid in enumerate(report_ids, 1):
        files = {arm: out / "ranges" / column[arm] / f"{rid}.json"
                 for arm in arms}
        saved = {arm: _finished_range_run(path, arm, describe_model,
                                          range_model, conventions)
                 for arm, path in files.items()}
        if not redo and all(saved.values()):
            for arm, blob in saved.items():
                done[column[arm]][rid] = blob
            print(f"  [{n}/{len(report_ids)}] {rid}: already done, skipping")
            continue
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(report_ids)}] {rid}: skipped -- "
                  f"{failures[rid]}")
            continue
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            rules = {r.page: r.role for r in page_roles(doc)}
            outline = outline_text(document_outline(doc))
            d_meter = CostMeter()
            d_engine = engines.build(describe_model, d_meter, vision=True)
            try:
                described = describe_pages(
                    doc, d_engine,
                    folder=str(out / "ranges" / "descriptions" / rid),
                    workers=workers, redo=redo, deadline=deadline,
                    progress=_mirror_every(RANGE_MIRROR_EVERY, sync))
            finally:
                _keep_meter(out / "ranges" / "describe_meters", rid, d_meter)
            n_pages = doc.n_pages
            if described.stopped:
                failures[rid] = (f"the describer stopped: "
                                 f"{described.stopped}; the same call "
                                 f"again goes on")
                print(f"  [{n}/{len(report_ids)}] {rid}: "
                      f"{len(described.pages)} of {n_pages} described "
                      f"({described.reused} from disk), {d_meter.calls} "
                      f"describer call(s); STOPPED -- {described.stopped}")
                continue
            notes = [{"call": e.get("call"), "notes": e["notes"]}
                     for e in d_meter.per_call if e.get("notes")]
            line = []
            for arm in arms:
                if saved[arm] and not redo and not described.unresolved:
                    done[column[arm]][rid] = saved[arm]
                    continue
                l_meter = CostMeter()
                l_engine = engines.build(range_model, l_meter)
                found = label_by_ranges(
                    described.pages, n_pages, l_engine, outline=outline,
                    hints=rules if arm == "hinted" else None,
                    conventions=text, not_used=not_used)
                blob = {
                    "id": rid,
                    "run_date": date.today().isoformat(),
                    "n_pages": n_pages,
                    "arm": arm,
                    "describe_model": describe_model,
                    "range_model": range_model,
                    "describe_prompt": DESCRIBE_PROMPT_VERSION,
                    "labeller_prompt": labeller_fingerprint(
                        arm == "hinted", text, not_used),
                    "conventions_name": conventions,
                    "conventions": conventions_version(text),
                    "describer": engines.record(describe_model, d_engine),
                    "labeller": engines.record(range_model, l_engine),
                    "served_by": sorted({x for x in (d_engine.served_by,
                                                     l_engine.served_by)
                                         if x}),
                    "rules_labels": {str(k): v
                                     for k, v in sorted(rules.items())},
                    "labels": {str(k): v
                               for k, v in sorted(found.labels.items())},
                    "ranges": found.to_dict(),
                    "describe": described.to_dict(),
                    "describe_cost": dict(described.saved_cost),
                    "describe_notes": notes,
                    "cost": l_meter.to_dict(),
                    "seconds": round(time.time() - started, 1),
                }
                files[arm].write_text(_dumps(blob), encoding="utf-8")
                _sync(sync)
                done[column[arm]][rid] = blob
                line.append(f"{arm} {len(found.sections)} section(s), "
                            f"{len(found.unresolved)} unresolved, "
                            f"{l_meter.calls} call(s)")
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(report_ids)}] {rid}: FAILED -- "
                  f"{failures[rid]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        print(f"  [{n}/{len(report_ids)}] {rid}: {n_pages} pp, "
              f"{len(described.pages)} described ({described.reused} from "
              f"disk, {len(described.unresolved)} not), {d_meter.calls} "
              f"describer call(s); " + "; ".join(line)
              + f"; {time.time() - started:.0f} s")
    return {"runs": done, "failures": failures,
            "arms": [column[arm] for arm in arms],
            "describe_model": describe_model, "range_model": range_model}


def _range_column(arm: str, conventions: str = "A") -> str:
    """The folder and column an arm's runs go in: the arm under conventions
    A, ``<arm>_<name>`` under any other."""
    name = str(conventions).strip().upper()
    return arm if name == "A" else f"{arm}_{name}"


def _keep_meter(folder: Path, rid: str, meter: Any) -> None:
    """One describing pass's cost record, beside the runs, if it made any
    call: each pass its own file, so a resumed report keeps every pass's."""
    if not (meter.calls or meter.failed):
        return
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (folder / f"{rid}-{stamp}.json").write_text(
        _dumps({"id": rid, "cost": meter.to_dict()}), encoding="utf-8")


#: Pages between two mirror syncs while a report is described.
RANGE_MIRROR_EVERY = 50


def _mirror_every(n: int, sync: Optional[Any]) -> Optional[Any]:
    """A ``describe_pages`` progress callback that syncs the mirror every
    ``n`` pages described, or None with no mirror. A sync still running
    when the next is due is not started twice."""
    if sync is None:
        return None
    busy = threading.Lock()

    def progress(done: int, _wanted: int) -> None:
        if done % int(n) or not busy.acquire(blocking=False):
            return
        try:
            _sync(sync)
        finally:
            busy.release()
    return progress


def _finished_range_run(path: Path, arm: Optional[str] = None,
                        describe_model: Optional[str] = None,
                        range_model: Optional[str] = None,
                        conventions: str = "A") -> Optional[dict]:
    """A saved range run that is finished, or None.

    Not finished: a file that cannot be read (it is made again, and the run
    goes on); one drawn over a gap -- a page that could not be described --
    (the next call describes the page and draws the sections again); and one
    made by other models, from another describer prompt, or with other
    labeller instructions than this call's (its labels are drawn again; its
    descriptions are reused where the describer and its prompt match)."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict):
        return None
    if (blob.get("describe") or {}).get("unresolved"):
        return None
    from report_ingest.range_labels import (DESCRIBE_PROMPT_VERSION,
                                            conventions_named,
                                            labeller_fingerprint)
    if blob.get("describe_prompt") != DESCRIBE_PROMPT_VERSION:
        return None
    if arm is not None and blob.get("labeller_prompt") != \
            labeller_fingerprint(arm == "hinted",
                                 *conventions_named(conventions)):
        return None
    if describe_model is not None and \
            blob.get("describe_model") != describe_model:
        return None
    if range_model is not None and blob.get("range_model") != range_model:
        return None
    return blob


def _saved_ranges(out: Path, report_ids: Sequence[str],
                  arms: Sequence[str], describe_model: str,
                  range_model: str, conventions: str = "A"
                  ) -> Dict[str, Any]:
    """Every finished range run of ``report_ids`` on disk, in the shape
    :func:`_run_ranges` returns, with no call made: what a results table
    over a run done a few reports at a time is drawn from."""
    column = {arm: _range_column(arm, conventions) for arm in arms}
    runs: Dict[str, Dict[str, dict]] = {column[arm]: {} for arm in arms}
    for rid in report_ids:
        for arm in arms:
            blob = _finished_range_run(
                out / "ranges" / column[arm] / f"{rid}.json", arm,
                describe_model, range_model, conventions)
            if blob is not None:
                runs[column[arm]][rid] = blob
    return {"runs": runs, "failures": {},
            "arms": [column[arm] for arm in arms],
            "describe_model": describe_model, "range_model": range_model}


def _score_ranges(ranges: Dict[str, Any], corpus: Corpus,
                  oos: Dict[str, Dict[int, dict]], mapped: Sequence[str],
                  sets: Sequence[str]) -> Dict[str, Any]:
    """The rules and each arm, on one set of hand labels, two ways.

    Strict accuracy (and with the out-of-sample file's alternates) through
    :class:`report_ingest.scoring.Scores`, the scorer every label stage
    uses; and the HARMFUL pages -- a page sent to another reader than its
    hand label sends it to, or to none, or to one where it should go to none
    (:func:`report_ingest.scoring.harm_score`) -- which is what a wrong label
    costs the record. No disputed page is dropped: the numbers are the
    strict ones the offline analysis of 2026-10-24 printed.
    """
    from report_ingest.scoring import harm_score, hand_sections

    arms = list(ranges["arms"])
    runs = ranges["runs"]
    columns = ["rules"] + arms
    sets_out: Dict[str, Any] = {}
    per_report: List[dict] = []
    seen: set = set()
    cost = {"describe": _cost_totals(), "label": _cost_totals()}
    for name in sets:
        ids = [rid for rid in _set_ids(name, corpus)
               if all(rid in runs[arm] for arm in arms)]
        scores = {c: Scores() for c in columns}
        harm = {c: 0 for c in columns}
        runs_drawn = {c: 0 for c in columns}
        hand_runs = 0
        for rid in ids:
            first = runs[arms[0]][rid]
            hand, alternates, source = _truth_for(rid, corpus, oos, mapped)
            given = {"rules": {int(k): v for k, v in
                               (first.get("rules_labels") or {}).items()}}
            for arm in arms:
                given[arm] = {int(k): v for k, v in
                              (runs[arm][rid].get("labels") or {}).items()}
            row: Dict[str, Any] = {"id": rid, "set": name,
                                   "pages": first.get("n_pages", 0),
                                   "scored": len(hand), "truth": source}
            pages = sorted(hand)
            hand_runs += len(hand_sections(hand)) if hand else 0
            for column in columns:
                for page, want in sorted(hand.items()):
                    scores[column].add(want, given[column].get(page, "other"),
                                       tuple(alternates.get(page) or ()))
                scored = harm_score(hand, given[column]) if hand else None
                harm[column] += scored["harmful"] if scored else 0
                drawn = len(hand_sections(
                    {p: given[column].get(p, "other") for p in pages}))
                runs_drawn[column] += drawn if hand else 0
                row[column] = ({"strict": scored["strict"],
                                "harmful": scored["harmful"],
                                "runs": drawn} if scored else None)
            if rid not in seen:
                seen.add(rid)
                _add_cost(cost["describe"], first.get("describe_cost"))
                for arm in arms:
                    _add_cost(cost["label"], runs[arm][rid].get("cost"))
                row["describe_dollars"] = float(
                    (first.get("describe_cost") or {}).get("dollars") or 0.0)
                row["label_dollars"] = {
                    arm: float((runs[arm][rid].get("cost") or {})
                               .get("dollars") or 0.0) for arm in arms}
                row["unresolved"] = {
                    arm: len((runs[arm][rid].get("ranges") or {})
                             .get("unresolved") or []) for arm in arms}
                per_report.append(row)
        sets_out[name] = {
            "reports": ids, "n_reports": len(ids),
            "columns": {c: {**scores[c].to_dict(), "harmful": harm[c],
                            "runs": runs_drawn[c]} for c in columns},
            "hand_runs": hand_runs,
            "_scores": scores,
        }
    return {"date": date.today().isoformat(), "arms": arms,
            "describe_model": ranges["describe_model"],
            "range_model": ranges["range_model"],
            "failures": dict(ranges["failures"]), "sets": sets_out,
            "per_report": per_report, "cost": cost}


def _render_ranges(scored: Dict[str, Any]) -> List[str]:
    """The range voter's table. IDs, labels, counts and rates only: the
    descriptions and the sections' reasons quote the pages, and stay in
    ``ranges/``."""
    columns = ["rules"] + list(scored["arms"])
    out: List[str] = [
        "", "# RI-73: every page described, then labelled by sections", "",
        f"Run {scored['date']}. Describer `{scored['describe_model']}` "
        f"(one call a page), labeller `{scored['range_model']}` (one call a "
        f"report, or a window of a long one). **plain** gives the labeller "
        f"the descriptions and the report's printed outline; **hinted** adds "
        f"the rules' label on every line. **HARMFUL** counts the pages a "
        f"label sends to another reader than the hand label does, or to "
        f"none, or to one where it should go to none. No disputed page is "
        f"dropped. A page no section settled counts as `other`.",
    ]
    if scored.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in scored["failures"].items()]
    for name, row in scored["sets"].items():
        if not row["n_reports"]:
            continue
        scores = row["_scores"]
        n = scores["rules"].n
        out += ["", f"## {name} -- {row['n_reports']} report(s), {n} scored "
                    f"pages", "", "```",
                f"{'':<24}" + "".join(f"{c:>10}" for c in columns),
                f"{'strict accuracy':<24}" + "".join(
                    f"{scores[c].accuracy:>10.3f}" for c in columns),
                f"{'HARMFUL pages':<24}" + "".join(
                    f"{row['columns'][c]['harmful']:>10}" for c in columns),
                f"{'runs drawn':<24}" + "".join(
                    f"{row['columns'][c]['runs']:>10}" for c in columns)
                + f"   (hand: {row['hand_runs']})", "```"]
        out += ["", "```"] + columns_label_table(
            [(c, scores[c]) for c in columns]) + ["```"]
    out += ["", "## Per report", "", "```",
            f"{'report':<8}{'pages':>6}"
            + "".join(f"{c + ' acc':>12}{c + ' harm':>12}" for c in columns)
            + f"{'describe $':>12}"
            + "".join(f"{a + ' $':>10}" for a in scored["arms"])]
    for row in scored["per_report"]:
        if "describe_dollars" not in row:
            continue
        line = f"{row['id']:<8}{row['pages']:>6}"
        for c in columns:
            cell = row.get(c)
            line += (f"{cell['strict']:>12.3f}{cell['harmful']:>12}"
                     if cell else f"{'--':>12}{'--':>12}")
        line += f"{row['describe_dollars']:>12.2f}"
        line += "".join(f"{row['label_dollars'][a]:>10.2f}"
                        for a in scored["arms"])
        out.append(line)
    out.append("```")
    cost = scored["cost"]
    out += ["", f"Describer: {cost['describe'].get('calls', 0)} call(s)"
                f"{_money(cost['describe'])}; labeller: "
                f"{cost['label'].get('calls', 0)} call(s)"
                f"{_money(cost['label'])}. The describer's cost counts each "
                f"description once, when it was made, though a resumed run "
                f"read it off disk."]
    return out


def _run_vote(corpus: Corpus, oos: Dict[str, Dict[int, dict]],
              mapped: Sequence[str], sets: Sequence[str], out: Path, *,
              vision_dirs: Optional[Sequence[Any]] = None,
              review_dir: Any = None,
              sync: Optional[Any] = None) -> Dict[str, Any]:
    """The voters set against each other and against the hand labels.

    NO model is called and nothing is rendered: every input is either a run
    file already on disk or planlens run again over the PDF. That is what
    makes this stage free to re-run after every change to the arithmetic.
    """
    dirs = _vote_vision_dirs(out, vision_dirs)
    runs, source = _vote_vision_runs(dirs)
    if not runs:
        looked = ", ".join(str(d) for d in dirs)
        raise FileNotFoundError(
            f"the 'vote' stage needs saved vision runs (<ID>.json holding "
            f"vision.labels) and found none. Looked in: {looked}. Run "
            f"stages=('vision_labels',) first, or pass vision_dirs=[...] "
            f"naming the folder an earlier run left behind.")
    runs_dir = Path(review_dir) if review_dir is not None else out / "runs"
    (out / "vote").mkdir(parents=True, exist_ok=True)
    present = set(corpus.present_ids())

    members: Dict[str, List[str]] = {}
    for name in sets:
        members[name] = [rid for rid in _set_ids(name, corpus)]
    if "oos_blind" in members:
        members["honest_blind"] = [rid for rid in members["oos_blind"]
                                   if rid not in CHECKPOINT]
    wanted: List[str] = []
    for name in VOTE_SETS:
        for rid in members.get(name, []):
            if rid not in wanted:
                wanted.append(rid)
    absent = [rid for rid in wanted if rid not in runs]
    wanted = [rid for rid in wanted if rid in runs]

    print(f"  vote: {len(wanted)} report(s) with a saved vision run, from "
          f"{len(dirs)} folder(s); no model is called")
    if absent:
        print(f"  no saved vision run and so not voted on: "
              f"{', '.join(absent)}")

    votes: Dict[str, List[PageVote]] = {}
    per_report: List[dict] = []
    dropped = 0
    rules_source: Dict[str, str] = {}
    truth_of: Dict[str, str] = {}
    for n, rid in enumerate(wanted, 1):
        blob = runs[rid]
        rules, rules_confidence, where = _vote_rules(rid, corpus, blob,
                                                     present)
        rules_source[rid] = where
        seen = blob.get("vision") or {}
        vision = {int(k): v for k, v in (seen.get("labels") or {}).items()}
        review = _review_labels_in(runs_dir, rid)
        hand, alternates, truth = _truth_for(rid, corpus, oos, mapped)
        rows = build_votes(rid, rules, vision,
                           rules_confidence=rules_confidence,
                           vision_confidence=vision_confidences(seen),
                           review=review, hand=hand, alternates=alternates)
        kept: List[PageVote] = []
        for vote in rows:
            # The same dispute rule the other stages use: a page the lead
            # has confirmed the HAND label doubtful on is dropped from every
            # column at once, or the columns stop being comparable.
            if vote.review and disputed_drop(rid, vote.page, vote.review):
                dropped += 1
                continue
            kept.append(vote)
        votes[rid] = kept
        truth_of[rid] = truth
        print(f"  [{n}/{len(wanted)}] {rid}: {len(kept)} page(s), rules from "
              f"{where}, "
              f"{sum(1 for v in kept if not v.agree)} disagreement(s)")

    # The trust table is learned on the IN-SAMPLE reports and nothing else.
    learned_on = [rid for rid in _set_ids("insample", corpus) if rid in votes]
    table = trust_table([v for rid in learned_on for v in votes[rid]])
    # And it is SAVED, because the production graph's ``trust`` policy takes
    # it by path: the stage is where the table can honestly be learned, and
    # the ingest is where it would be used. Without this file the graph's
    # ``trust`` falls back to ``structural`` and says so.
    table_file = out / "vote" / "trust_table.json"
    table_file.write_text(json.dumps({
        "date": date.today().isoformat(),
        "learned_on": list(learned_on),
        "policy": "trust",
        "table": table}, indent=2), encoding="utf-8")
    _sync(sync)

    for rid in wanted:
        rows = votes[rid]
        agree = agreement(rows)
        qa = {
            "id": rid,
            "run_date": date.today().isoformat(),
            "n_pages": runs[rid].get("n_pages", 0),
            "vision_mode": runs[rid].get("mode"),
            "vision_run": str(source[rid]),
            "rules_from": rules_source[rid],
            "review": bool(any(v.review is not None for v in rows)),
            "pages_compared": len(rows),
            "agreement": agree,
            "trust_table_learned_on": list(learned_on),
            "disagreements": [v.to_row(policy_labels(v, table))
                              for v in rows if not v.agree],
        }
        (out / "vote" / f"{rid}.json").write_text(
            json.dumps(qa, indent=2), encoding="utf-8")
        _sync(sync)
        hand_pages = [v for v in rows if v.scored]
        per_report.append({
            "id": rid,
            "pages": len(rows),
            "scored": len(hand_pages),
            "truth": truth_of[rid],
            "agreement": agree["agreement"],
            "disagreed": agree["disagreed"]["pages"],
            "rules": _accuracy(hand_pages, "rules"),
            "vision": _accuracy(hand_pages, "vision"),
            "review": _accuracy(hand_pages, "review"),
        })

    rows_by_set: Dict[str, Any] = {}
    for name in VOTE_SETS:
        ids = [rid for rid in members.get(name, []) if rid in votes]
        if not ids:
            continue
        rows_by_set[name] = _score_vote(
            [v for rid in ids for v in votes[rid]], ids, table)

    return {
        "date": date.today().isoformat(),
        "n_reports": len(wanted),
        "reports": list(wanted),
        "absent": absent,
        "vision_dirs": [str(d) for d in dirs],
        "vision_folders": sorted({Path(d).name for d in source.values()}),
        "review_dir": str(runs_dir),
        "n_with_review": sum(1 for rid in wanted
                             if any(v.review is not None for v in votes[rid])),
        "disputed_dropped": dropped,
        "trust": table,
        "trust_learned_on": list(learned_on),
        "trust_table_file": str(table_file),
        "policies": list(POLICIES),
        "structural_rules_win": list(STRUCTURAL_RULES_WIN),
        "sets": rows_by_set,
        "per_report": per_report,
    }


def _accuracy(rows: Sequence[PageVote], voter: str) -> Optional[float]:
    """One voter's strict accuracy over the hand-labelled pages, or None."""
    if voter == "review":
        rows = [v for v in rows if v.review is not None]
    if not rows:
        return None
    hits = sum(1 for v in rows if v.right(getattr(v, voter)))
    return round(hits / len(rows), 4)


def _score_vote(rows: Sequence[PageVote], ids: Sequence[str],
                table: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One set: the agreement, every voter's score, and the three policies."""
    agree = agreement(rows)
    scorers: Dict[str, Scores] = {"rules": Scores(), "vision": Scores(),
                                  "review": Scores()}
    for policy in POLICIES:
        scorers[policy] = Scores()
    for vote in rows:
        if vote.hand is None:
            continue
        alts = vote.alternates
        scorers["rules"].add(vote.hand, vote.rules, alts)
        scorers["vision"].add(vote.hand, vote.vision, alts)
        if vote.review is not None:
            scorers["review"].add(vote.hand, vote.review, alts)
        for policy, label in policy_labels(vote, table).items():
            scorers[policy].add(vote.hand, label, alts)
    need = required_review_accuracy(agree["agreed"]["pages"],
                                    agree["agreed"]["correct"],
                                    agree["disagreed"]["pages"], GATE)
    return {
        "reports": list(ids),
        "n_reports": len(ids),
        "agreement": agree,
        "scores": {name: s.to_dict() for name, s in scorers.items() if s.n},
        "required_review_accuracy": need,
        "_scores": scorers,
    }


#: What each policy is called in the per-label table, where a column name
#: has ten characters to fit into beside its ``P ``/``R `` prefix.
POLICY_COLUMN: Dict[str, str] = {"trust": "trust", "structural": "struct",
                                 "confidence": "conf"}


def _vote_columns(row: Dict[str, Any]) -> List[Tuple[str, Scores]]:
    """The columns this set has: the voters it has, then the three policies."""
    scorers = row["_scores"]
    out = [("rules", scorers["rules"])]
    if scorers["review"].n:
        out.append(("+review", scorers["review"]))
    out.append(("vision", scorers["vision"]))
    out += [(POLICY_COLUMN[name], scorers[name]) for name in POLICIES]
    return [(name, s) for name, s in out if s.n]


def _pct(value: Optional[float]) -> str:
    return "     -" if value is None else f"{value:6.3f}"


def _render_vote(vote: Dict[str, Any]) -> List[str]:
    """The vote, as tables of IDs, labels, counts and rates. Nothing else."""
    folders = ", ".join(vote.get("vision_folders") or []) or "none"
    out: List[str] = [
        "", "# Vote: rules, vision and the review as voters", "",
        f"Run {vote['date']}. **No model was called.** The rules are "
        f"planlens' own, recomputed here where the report is to hand and "
        f"read off the saved run where it is not; the vision labels come "
        f"off saved vision runs (folder(s): {folders}); the review's labels "
        f"come off the label runs, where a report has one -- "
        f"{vote['n_with_review']} of {vote['n_reports']} do.",
        "",
        "The question is not which voter is better. It is what a page they "
        "SPLIT on is worth looking at, and which of them to believe where "
        "they split.",
    ]
    if vote.get("absent"):
        out += ["", "**Reports in the sets asked for with no saved vision "
                    "run, and so in no number below:** "
                    + ", ".join(vote["absent"]) + "."]
    if vote.get("disputed_dropped"):
        out += ["", f"{vote['disputed_dropped']} page(s) dropped from every "
                    f"column as confirmed disputed hand labels."]

    out += _render_vote_agreement(vote)
    out += _render_vote_trust(vote)
    out += _render_vote_policies(vote)
    out += _render_vote_disagreement(vote)
    out += _render_vote_per_report(vote)
    out += ["",
            "Every page the two voters split on is listed per report in "
            "`vote/<ID>.json` -- the page number, what each voter said, its "
            "confidence, the hand label where there is one, and what each "
            "policy chose. Page numbers and labels only, so those files "
            "carry nothing a reason would.", ""]
    return out


def _render_vote_agreement(vote: Dict[str, Any]) -> List[str]:
    out: List[str] = [
        "", "## 1. Agreement, and what it is worth", "",
        "`agreement` is over EVERY page both voters covered, hand-labelled "
        "or not, because that is the number a production run can compute "
        "for itself. A page the vision pass left unresolved counts as "
        "`other` and therefore as a disagreement, which is the honest "
        "reading: a page nothing could answer for is exactly the page that "
        "wants a second look. Every accuracy beside it is over the "
        "hand-labelled pages alone.",
        "", "```",
        f"{'set':<14}{'pages':>8}{'agree':>8}{'rate':>8}{'scored':>8}"
        f"{'agreed':>8}{'right':>9}{'split':>7}{'rules':>8}{'vision':>8}"
        f"{'either':>8}",
    ]
    for name, row in vote["sets"].items():
        agree = row["agreement"]
        agreed, split = agree["agreed"], agree["disagreed"]
        out.append(
            f"{name:<14}{agree['pages']:>8}{agree['agree']:>8}"
            f"{_pct(agree['agreement']):>8}{agree['scored']:>8}"
            f"{agreed['pages']:>8}{_pct(agreed['accuracy']):>9}"
            f"{split['pages']:>7}{_pct(split['rules_accuracy']):>8}"
            f"{_pct(split['vision_accuracy']):>8}"
            f"{_pct(split['ceiling']):>8}")
    out += ["```", "",
            "`right` is how often the two, agreeing, were right -- the "
            "confidence claim. `rules` and `vision` are how often each was "
            "right on the pages they split on, and `either` is how often one "
            "of them had it: that last is the ceiling a perfect tie-breaker "
            "would reach without a third reading of the page."]
    return out


def _render_vote_trust(vote: Dict[str, Any]) -> List[str]:
    learned = vote.get("trust_learned_on") or []
    out: List[str] = [
        "", "## 2. Per-label trust, learned on the IN-SAMPLE reports only",
        "",
        "For each class the RULES put a page in, which voter was right more "
        "often on the pages they split on. Keyed by the rules' label because "
        "that is what a production run has before it knows the answer. "
        + (f"Learned on {len(learned)} in-sample report(s): "
           f"{', '.join(learned)}. "
           if learned else
           "**Nothing was learned**: no in-sample report was voted on in "
           "this run, so the `trust` policy sends every disagreement to "
           "vision. ")
        + "The out-of-sample sets are SCORED with this table and never "
          "learned on, and the blind set never is under any circumstance.",
    ]
    table = vote.get("trust") or {}
    if not table:
        return out
    out += ["", "```",
            f"{'rules label':<18}{'splits':>8}{'rules right':>13}"
            f"{'vision right':>14}{'believe':>10}"]
    for label in sorted(table, key=lambda k: (-table[k]["pages"], k)):
        cell = table[label]
        out.append(f"{label:<18}{cell['pages']:>8}{cell['rules']:>13}"
                   f"{cell['vision']:>14}{cell['winner']:>10}")
    out += ["```", "",
            "A tie goes to vision, and so does a class the in-sample pages "
            "never split on, so `believe rules` always means the rules were "
            "strictly better on pages somebody has checked."]
    return out


def _render_vote_policies(vote: Dict[str, Any]) -> List[str]:
    out: List[str] = [
        "", "## 3. The combined labels, scored beside the voters", "",
        "Three ways of settling a disagreement, each scored by the SAME "
        "scorer against the SAME hand labels as the voters themselves:",
        "",
        "- **trust** -- believe whoever the in-sample table above favours "
        "for the rules' label class; vision where it says nothing.",
        "- **structural** -- vision everywhere except "
        + ", ".join(f"`{x}`" for x in vote["structural_rules_win"])
        + ", which the rules own because they are decided by where a page "
          "SITS in the document rather than by what it looks like. No "
          "learning at all.",
        "- **confidence** -- whoever said it more confidently (planlens' own "
        "rule confidence against the vision pass's), ties to the rules.",
        "",
        "**Read the out-of-sample rows, not the in-sample one.** `trust` is "
        "scored in sample with a table learned on those very pages, so its "
        "in-sample figure is a ceiling rather than a result. `structural` "
        "and `confidence` learn nothing and are honest everywhere.",
    ]
    for name, row in vote["sets"].items():
        columns = _vote_columns(row)
        if not columns:
            continue
        scored = row["_scores"]["rules"].n
        out += ["", f"### {name} -- {row['n_reports']} report(s), "
                    f"{scored} scored pages", "", "```",
                f"{'':<24}" + "".join(f"{n:>12}" for n, _s in columns),
                f"{'strict accuracy':<24}"
                + "".join(f"{s.accuracy:>12.3f}" for _n, s in columns),
                f"{'accepting alternates':<24}"
                + "".join(f"{s.lenient_accuracy:>12.3f}"
                          for _n, s in columns),
                "",
                f"key content (the gate is {GATE:.2f} on both rates)"]
        out += columns_label_table(columns, KEY_CONTENT)
        out.append("```")
    return out


def _render_vote_disagreement(vote: Dict[str, Any]) -> List[str]:
    out: List[str] = [
        "", "## 4. The disagreement set: what a targeted review would cost",
        "",
        "The pages the two voters split on are the ones a third reading -- a "
        "better model, a document-mode pass, or a person -- would be spent "
        "on. `must reach` is the accuracy that reading would need ON THOSE "
        f"PAGES for the whole set to clear {GATE:.2f}, leaving every agreed "
        "page as it is.",
        "", "```",
        f"{'set':<14}{'scored':>8}{'split':>8}{'fraction':>10}"
        f"{'agreed right':>14}{'must reach':>12}",
    ]
    for name, row in vote["sets"].items():
        agree = row["agreement"]
        need = row["required_review_accuracy"]
        if need is None:
            text = "       -"
        elif need > 1.0:
            text = "  >1.000"
        elif need <= 0.0:
            text = "   0.000"
        else:
            text = f"{need:8.3f}"
        out.append(
            f"{name:<14}{agree['scored']:>8}"
            f"{agree['disagreed']['pages']:>8}"
            f"{_pct(agree['disagreed']['fraction']):>10}"
            f"{agree['agreed']['correct']:>14}{text:>12}")
    out += ["```", "",
            "`>1.000` means the gate is out of reach on this set even with a "
            "perfect review of every disagreement, because the pages the two "
            "voters AGREE on already carry more error than the gate allows. "
            "That is a result about the agreed pages, not about the review."]
    return out


def _render_vote_per_report(vote: Dict[str, Any]) -> List[str]:
    """Per report, for the sets that are not blind.

    The blind set gets no per-report line here for the same reason it gets
    none anywhere else: a blind figure read report by report stops being
    blind the moment somebody goes looking for which report dragged it down.
    """
    blind = set()
    for name in ("oos_blind", "honest_blind"):
        blind |= set((vote["sets"].get(name) or {}).get("reports") or [])
    rows = [r for r in vote["per_report"] if r["id"] not in blind]
    if not rows:
        return ["", "Per-report lines are left out: every report voted on is "
                    "in the blind set, and a blind figure read report by "
                    "report stops being blind."]
    out = ["", "## 5. Per report", "", "```",
           f"{'report':<8}{'pages':>7}{'scored':>8}{'rate':>8}{'split':>7}"
           f"{'rules':>8}{'vision':>8}{'review':>8}"]
    for r in rows:
        out.append(f"{r['id']:<8}{r['pages']:>7}{r['scored']:>8}"
                   f"{_pct(r['agreement']):>8}{r['disagreed']:>7}"
                   f"{_pct(r['rules']):>8}{_pct(r['vision']):>8}"
                   f"{_pct(r['review']):>8}")
    out.append("```")
    return out


# ---------------------------------------------------------------------------
# the ingest stage: the whole pipeline, and its exports, per report
# ---------------------------------------------------------------------------

def _run_blob_on_disk(runs_dir: Path, rid: str) -> Dict[str, Any]:
    run_file = Path(runs_dir) / f"{rid}.json"
    if not run_file.is_file():
        return {}
    try:
        return json.loads(run_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _reuse_label_run(folder: Path, runs_dir: Path, rid: str,
                     review: bool = True) -> Tuple[bool, bool]:
    """Seed the ingest folder with a saved triage and review, when present.

    A ``labels`` run leaves ``runs/<ID>.json`` holding the triage profile
    and the review's final labels; ``graph.ingest_report`` resumes from
    ``triage.json`` and ``review.json`` in its own folder. Copying the one
    into the other is what stops the ingest paying for ~$0.50 of review a
    second time. Nothing already in the folder is overwritten.
    ``review=False`` seeds the triage alone: model first runs no review.

    Each seeded file is stamped ``reused_from``, which is how a later
    attempt's run file still says the step came from a label run
    (:func:`_resumed_fields`), and the review's carries the label run's own
    record of what the review cost, as the seeded profile carries the
    triage's: a resumed record's whole-report cost counts both (BACKLOG
    RI-125). Returns what was seeded now; whether the graph used it is the
    record's to say.
    """
    blob = _run_blob_on_disk(runs_dir, rid)
    if not blob:
        return False, False
    triage_reused = review_reused = False
    profile = blob.get("profile") or {}
    triage_file = folder / "triage.json"
    if profile and not triage_file.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        stamped = dict(profile)
        stamped["reused_from"] = str(Path(runs_dir) / f"{rid}.json")
        triage_file.write_text(json.dumps(stamped, indent=2),
                               encoding="utf-8")
        triage_reused = True
    final = (blob.get("review") or {}).get("final_labels") or {}
    review_file = folder / "review.json"
    if review and final and not review_file.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        saved = {"final_labels": {str(k): v for k, v in final.items()},
                 "reused_from": str(Path(runs_dir) / f"{rid}.json")}
        earlier = (blob.get("review") or {}).get("changes")
        if earlier is not None:
            # the earlier run's changes with their evidence, so the RI-15
            # rule applies to them when this review is read back
            saved["changes"] = [dict(c) for c in earlier
                                if isinstance(c, dict)]
        spent = (blob.get("review") or {}).get("cost")
        if isinstance(spent, dict) and spent:
            # what the label run's review cost, as its run file says
            saved["cost"] = dict(spent)
        review_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        review_reused = True
    return triage_reused, review_reused


def _resumed_fields(record: Any, folder: Path) -> Dict[str, Any]:
    """A report's run-file account of the steps it took from its folder
    (BACKLOG RI-125), off the record's ``document.resumed``.

    ``triage_reused`` and ``review_reused`` are true when that step was
    taken from the folder and made no call, whatever put its file there: a
    saved label run seeding it (:func:`_reuse_label_run`), an earlier
    attempt, or a folder copied in. When the run took any step from the
    folder, ``resumed`` names them as the record does (``steps``), says how
    many of the record's item readings came off disk (``items`` of
    ``items_of``), and lists the steps whose file a label run seeded, now
    or on an earlier attempt (``from_label_run``, by the file's
    ``reused_from`` stamp; only when there are any). A run that took
    nothing from the folder writes both flags false and no ``resumed``, as
    a first attempt always did.
    """
    resumed = getattr(record.document, "resumed", None)
    steps = [str(step) for step in resumed.steps] if resumed else []
    out: Dict[str, Any] = {"triage_reused": "triage" in steps,
                           "review_reused": "review" in steps}
    if resumed is None:
        return out
    seeded = list(_seeded_from(folder, steps))
    out["resumed"] = {"steps": steps, "items": int(resumed.items),
                      "items_of": int(resumed.items_of),
                      **({"from_label_run": seeded} if seeded else {})}
    return out


def _seeded_from(folder: Path, steps: Sequence[str]) -> Dict[str, str]:
    """``{step: label run file}`` for the steps among ``steps`` whose file in
    ``folder`` a label run seeded, by the ``reused_from`` stamp
    :func:`_reuse_label_run` put on it (triage and review only)."""
    found: Dict[str, str] = {}
    for step in ("triage", "review"):
        if step not in steps:
            continue
        try:
            blob = json.loads((folder / f"{step}.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(blob, dict) and blob.get("reused_from"):
            found[step] = str(blob["reused_from"])
    return found


def _label_run_engine(folder: Path, steps: Sequence[str]) -> str:
    """The engine kind of the label run that seeded ``steps`` of this
    folder, read off the label run file the seeded files' ``reused_from``
    stamp names (``prompter`` for a run file from before the engine was
    recorded); "" when no stamped file is on disk.

    The stamp, not this call's ``runs/``: a folder copied in from another
    ``out_dir`` keeps its stamps, and the run file in this call's ``runs/``
    (or none) need not be the one that seeded it. An engine no file on disk
    records is not named (BACKLOG RI-125)."""
    for stamp in _seeded_from(folder, steps).values():
        blob = _run_blob_on_disk(Path(stamp).parent, Path(stamp).stem)
        if blob:
            return str(blob.get("engine") or "prompter")
    return ""


def _truth_files_for(rid: str, truth: Sequence[Optional[Path]]
                     ) -> Dict[str, List[Tuple[str, dict]]]:
    """The hand-truth files that concern one report, by stage."""
    logs_dir, lab_dir, narrative_dir = truth[0], truth[1], truth[2]
    calc_dir = truth[3] if len(truth) > 3 else None
    sounding_dir = truth[4] if len(truth) > 4 else None
    out: Dict[str, List[Tuple[str, dict]]] = {
        "logs": [], "lab": [], "narrative": [], "calc": [], "soundings": []}

    def load(path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    if logs_dir is not None and logs_dir.is_dir():
        for path in sorted(logs_dir.glob(f"{rid}_p*.json")):
            blob = load(path)
            if blob:
                out["logs"].append((path.stem, blob))
    if lab_dir is not None and lab_dir.is_dir():
        for path in sorted(lab_dir.glob(f"*__{rid}_p*.json")):
            blob = load(path)
            if blob:
                out["lab"].append((path.stem, blob))
    if calc_dir is not None and calc_dir.is_dir():
        for path in sorted(calc_dir.glob(f"*__{rid}_p*.json")):
            blob = load(path)
            if blob:
                out["calc"].append((path.stem, blob))
    if sounding_dir is not None and sounding_dir.is_dir():
        for path in sorted(sounding_dir.glob(f"*__{rid}_p*.json")):
            blob = load(path)
            if blob:
                out["soundings"].append((path.stem, blob))
    if narrative_dir is not None and narrative_dir.is_dir():
        path = narrative_dir / f"{rid}.json"
        if path.is_file():
            blob = load(path)
            if blob:
                out["narrative"].append((path.stem, blob))
    return out


def _bound_records(record: Any, folder: Path) -> List[Any]:
    """The records of the reports bound inside this one, off disk.

    The graph writes each beside its parent (``bound/<id>/``) before the
    parent, and their page numbers are the parent FILE's, so a hand truth on
    a bound document's pages can be scored against the record that read
    them. A child that was not read, or will not load, is left out.
    """
    from report_ingest.model import ReportRecord

    out: List[Any] = []
    for row in getattr(record, "bound_documents", ()) or ():
        if not row.read or not row.record_path:
            continue
        try:
            out.append(ReportRecord.model_validate_json(
                (folder / row.record_path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def _score_ingest_record(record: Any, rid: str,
                         truth: Sequence[Optional[Path]],
                         bound: Sequence[Any] = ()) -> Dict[str, Any]:
    """The ingest's record against every hand truth this report has.

    The SAME scorers the reader stages use, on the record the whole
    pipeline produced -- so the stage doubles as the whole-pipeline score.
    A log truth is scored against the investigations read off its pages,
    a lab truth against the tests read off its page and a calculation truth
    against the calculation read off its run -- not against the whole
    record, or a sample at the right depth in a different hole would count.

    ``bound`` are the records of the reports bound inside this one
    (:func:`_bound_records`). A truth that names its pages is scored against
    whatever was read off those pages, the parent's or a bound document's:
    a page is read on one side of the boundary only (BACKLOG RI-14), so a
    truth on a bound document's pages has nothing in the parent to meet. A
    truth that names no page is scored against the parent alone. Each row
    says which record it met (``records``: ``parent`` and/or the bound
    ids), because a truth on a range the page labels drew used to meet
    nothing, and its score can now rise with no reader reading better.
    """
    from report_ingest.calc_scoring import pages_of as calc_pages_of
    from report_ingest.calc_scoring import score_record as score_calc
    from report_ingest.lab_scoring import pages_of
    from report_ingest.lab_scoring import score_record as score_lab
    from report_ingest.log_scoring import score_record as score_log
    from report_ingest.narrative_scoring import score_narrative
    from report_ingest.sounding_scoring import kind_of as sounding_kind
    from report_ingest.sounding_scoring import pages_of as sounding_pages_of
    from report_ingest.sounding_scoring import score_record as score_sounding

    holders = [("parent", record)] + [
        (getattr(child.parent, "bound_id", "") or "bound", child)
        for child in bound]

    def pool(name: str, pages: Set[int]) -> List[Tuple[str, Any]]:
        """(the record's name, the entry) for each entry off ``pages``."""
        return [(who, x) for who, holder in (holders if pages
                                             else holders[:1])
                for x in getattr(holder, name)
                if not pages or set(x.pages) & pages]

    def met(rows: Sequence[Tuple[str, Any]]) -> List[str]:
        return sorted({who for who, _x in rows})

    files = _truth_files_for(rid, truth)
    out: Dict[str, Any] = {"logs": [], "lab": [], "narrative": None,
                           "calc": [], "soundings": []}
    for name, blob in files["logs"]:
        pages = {int(p) for p in blob.get("pages") or []}
        invs = pool("investigations", pages)
        score = score_log(blob, [x for _who, x in invs])
        out["logs"].append({"id": name, "overall": score.total.to_dict(),
                            "scores": {k: v.to_dict()
                                       for k, v in score.scores.items()},
                            "records": met(invs)})
    for name, blob in files["lab"]:
        pages = set(pages_of(blob))
        tests = pool("lab_tests", pages)
        score = score_lab(blob, [x for _who, x in tests])
        out["lab"].append({"id": name, "kind": blob.get("kind") or "",
                           "overall": score.total.to_dict(),
                           "scores": {k: v.to_dict()
                                      for k, v in score.scores.items()},
                           "records": met(tests)})
    for name, blob in files["calc"]:
        pages = set(calc_pages_of(blob))
        mine = pool("calculations", pages)
        if not mine:
            out["calc"].append({"id": name, "kind": blob.get("kind") or "",
                                "overall": {"found": 0, "total": 0,
                                            "rate": None},
                                "scores": {}, "read": False, "records": []})
            continue
        # A run the graph split into more than one item is scored against
        # the one that covers the most of it: the truth file describes ONE
        # calculation and scoring it against a fragment would report the
        # fragment's misses as the reader's.
        who, best = max(mine, key=lambda row: len(pages & set(row[1].pages)))
        score = score_calc(blob, best)
        out["calc"].append({"id": name, "kind": blob.get("kind") or "",
                            "overall": score.total.to_dict(),
                            "scores": {k: v.to_dict()
                                       for k, v in score.scores.items()},
                            "read": True, "records": [who]})
    for name, blob in files["soundings"]:
        pages = set(sounding_pages_of(blob))
        invs = pool("investigations", pages)
        score = score_sounding(blob, [x for _who, x in invs])
        out["soundings"].append({
            "id": name, "kind": sounding_kind(blob),
            "overall": score.total.to_dict(),
            "scores": {k: v.to_dict() for k, v in score.scores.items()},
            "records": met(invs)})
    for name, blob in files["narrative"]:
        score = score_narrative(blob, record.general, record.natural_hazards,
                                report=rid)
        out["narrative"] = {"recall": score.recall.to_dict(),
                            "precision": score.precision.to_dict(),
                            "agreement": score.agreement.to_dict(),
                            "records": ["parent"]}
    return out


def _ingest_labels(folder: Path, record: Any, rid: str, corpus: Corpus,
                   oos: Dict[str, Dict[int, dict]],
                   mapped: Sequence[str]) -> Dict[str, Any]:
    """What the vote did on one report, and what the labels scored.

    The counts come off the graph's own ``labels.json``; the accuracy comes
    from the SAME scorer the ``labels`` and ``vision_labels`` stages use, so
    ``final`` here and ``rules`` there are the same arithmetic over the same
    hand labels. A report with no hand labels has counts and no accuracy,
    which is an unmeasured run rather than a failing one.
    """
    blob: Dict[str, Any] = {}
    path = folder / "labels.json"
    if path.is_file():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            blob = {}
    out: Dict[str, Any] = {
        "policy": blob.get("policy") or record.document.label_policy,
        "review_mode": blob.get("review_mode") or record.document.review_mode,
        "vision_mode": blob.get("vision_mode") or "",
        "note": blob.get("note") or "",
        "pages_voted": int(blob.get("pages_voted") or 0),
        "split": int(blob.get("n_split")
                     or record.document.label_split_pages or 0),
        "review_changed": int(blob.get("n_review_changed")
                              or record.document.review_changed or 0),
        "review_changed_off_split": int(
            blob.get("n_review_changed_off_split") or 0),
        "disagreements_left": sum(1 for e in record.qa
                                  if e.kind == "label_disagreement"),
        "cost": dict(blob.get("cost") or {}),
        "accuracy": None,
    }
    hand, alternates, truth = _truth_for(rid, corpus, oos, mapped)
    if not hand:
        out["truth"] = truth
        return out
    final = {int(p.page): p.label for p in record.page_labels}
    rules: Dict[int, str] = {}
    for page_label in record.page_labels:
        for voter in page_label.voters:
            if voter.voter == "rules" and voter.label:
                rules[int(page_label.page)] = voter.label
    scored: Dict[str, Scores] = {"rules": Scores(), "final": Scores()}
    for page, label in sorted(hand.items()):
        alts = tuple(alternates.get(page) or ())
        scored["rules"].add(label, rules.get(page, "other"), alts)
        scored["final"].add(label, final.get(page, "other"), alts)
    out["truth"] = truth
    out["accuracy"] = {name: s.to_dict()["accuracy"]
                       for name, s in scored.items()}
    out["scored_pages"] = scored["final"].n
    # The pages a label sends to another reader than its hand label does,
    # or to none, or to one where it should go to none: what a wrong label
    # costs the record (plan section 8.8, wave MF-2).
    from report_ingest.scoring import harm_score

    out["harmful"] = {"rules": harm_score(hand, rules)["harmful"],
                      "final": harm_score(hand, final)["harmful"]}
    return out


def _ingest_bound_rows(record: Any) -> List[Dict[str, Any]]:
    """One row per report bound inside this one, for the RESULTS table.

    IDs, kinds, page ranges and counts. NOT the title or the firm: those are
    a private report's own words, and a scorecard is written up in a public
    repository. The child's own record, in the run folder, carries them.
    """
    out: List[Dict[str, Any]] = []
    for row in getattr(record, "bound_documents", ()) or ():
        counts = dict(row.counts or {})
        out.append({
            "bound_id": row.bound_id,
            "report_id": row.report_id,
            "kind": row.kind,
            "pages": row.pages,
            "n_pages": row.n_pages,
            "said_by": list(row.said_by),
            "read": bool(row.read),
            "investigations": int(counts.get("investigations") or 0),
            "lab_tests": int(counts.get("lab_tests") or 0),
            "samples": int(counts.get("samples") or 0),
            "qa": int(counts.get("qa") or 0),
            "folder": row.folder,
        })
    return out


def _diggs_verdicts(record: Any, outputs: Dict[str, str]) -> Dict[str, str]:
    """written / validated / read back, off the record's own QA entries."""
    written = "yes" if outputs.get("diggs") else "no"
    schema = "not written"
    roundtrip = "not written"
    for entry in record.qa:
        if entry.where == "diggs.schema":
            if entry.kind == "note":
                schema = "valid"
            elif entry.kind == "not_run" or any(
                    "not installed" in v for v in entry.values):
                # The second test reads a record written before RI-20, when
                # a check that could not run was filed as a failure.
                schema = "not run"
            else:
                schema = "INVALID"
        elif entry.where == "diggs.roundtrip":
            roundtrip = "equal" if entry.kind == "note" else "DIFFERS"
        elif entry.where == "diggs" and entry.kind == "skipped":
            written = "no"
    return {"written": written, "schema": schema, "roundtrip": roundtrip}


def _run_ingest(corpus: Corpus, prompter: Any, model: str, out: Path,
                report_ids: Sequence[str], *,
                truth: Sequence[Optional[Path]],
                triage_model: Optional[str] = None,
                log_budget: int, lab_budget: int, narrative_budget: int,
                redo: bool, review_dir: Any = None,
                max_total_dollars: Optional[float] = None,
                max_total_tokens: Optional[int] = None,
                max_report_tokens_in: Optional[int] = None,
                max_request_tokens_in: Optional[int] = None,
                lab_sheets_per_call: Optional[int] = None,
                tokens_per_minute: Optional[int] = None,
                label_policy: str = "structural",
                review_mode: str = "disagreements",
                vision_model: str = "funhouse-gpt-low",
                vision_mode: str = "sheet",
                vision_detail: Optional[str] = None,
                trust_table: Any = None,
                templates_path: Any = None,
                ingest_bound: bool = True,
                counts_from_logs: str = "off",
                narrative_skips_bound: bool = False,
                oos: Optional[Dict[str, Dict[int, dict]]] = None,
                mapped: Sequence[str] = (),
                sync: Optional[Any] = None,
                first: Optional[_FirstArm] = None,
                describe_model: Optional[str] = None,
                range_model: Optional[str] = None) -> Dict[str, Any]:
    """The whole pipeline on each report, into ``ingest/<ID>/``.

    Restartable twice over: the stage skips a report whose ``run.json`` is
    on disk, and inside a report ``graph.ingest_report`` skips every work
    item whose file is already under ``items/``. A saved label run is
    reused for the triage and the review, so a report that went through
    the ``labels`` stage costs only its readers here.

    THE PAGE LABELS ARE A VOTE HERE TOO, on exactly the parameters a
    production run takes: ``label_policy`` combines the rules, one vision
    pass on ``vision_model`` and the printed form, and ``review_mode``
    decides which pages the review is given. The reused review from a
    ``labels`` run is the WHOLE-REPORT one, so a report that has one keeps
    it and ``review_mode`` bites on the reports that do not -- which is what
    makes the comparison between the two worth reading.

    A REPORT BOUND INSIDE A REPORT becomes its own record under
    ``ingest/<ID>/bound/<bound id>/`` and its own row in the shared library,
    and what it holds is in none of its parent's counts (``ingest_bound``).
    Ledger run 7 has 19 of the 38 corpus reports carrying between one and
    four of them, so this moves numbers in the per-report table: a report
    whose investigation count falls is one whose appendix was somebody
    else's investigation all along.

    MODEL FIRST (``first`` on, BACKLOG RI-87) runs the same reports with
    ``ingest_report(model_first="on")``: every page described on
    ``describe_model`` (the vision engine; default ``vision_model``) and
    labelled by sections on ``range_model`` (default ``model``), each of
    the four readers on its model (``first.stage_models``), no vision
    voter and no review (a saved label run seeds the triage alone). Each
    ``run.json`` names the arm, the routes, the models, the instruction
    and describer versions, and what placement found. Off, a stage named
    in ``first.stage_models`` is today's reader on that model, and
    ``run.json`` names those models.

    TOKEN LIMITS (BACKLOG RI-120) work where the dollar meter reads 0.
    ``max_total_tokens`` stops the run before a report once the arm's
    folder has spent that many (tokens in plus output, every attempt of
    every report in it, as the run files record them; tokens in are plain
    input plus cache writes plus cache reads, BACKLOG RI-171) and refuses
    a model call
    inside a report that would start past what is left. The folder's spend
    before the call (BACKLOG RI-133) is counted with the call's own, so
    several builds on one folder share one ceiling; a report finished by an
    earlier build is counted once. The stop is listed among the failures
    under ``(token ceiling)`` with what the folder had spent when the call
    started;
    ``max_report_tokens_in`` refuses a call once the report has sent that
    many tokens in and stops the run after the report. A refused call
    is a ``CallCeilingReached``: the step is cut short and not saved, and
    ``run.json`` gains a ``token_brake`` entry saying which limit it was.

    EVERY ATTEMPT'S COST IS KEPT (BACKLOG RI-124). A report read again --
    after an error, a give-up (``item_errors``) or the token brake --
    replaces its ``run.json``; the new file carries ``earlier_attempts``
    (each earlier attempt's date, how it ended, its wall clock and its cost
    totals, read off the file it replaces) and ``all_attempts`` (their sum
    with this attempt), while ``cost`` and ``seconds`` stay this attempt's
    own. ``max_report_tokens_in`` counts every attempt's tokens in: a
    retried report starts with its earlier attempts' input counted, and one
    whose earlier attempts already reached the limit is not read again
    (it is listed among the failures). A report's first attempt, and any
    attempt under ``redo``, which starts the report afresh, writes its run
    file as before. A run file moved aside by hand is not read: only the
    ``run.json`` an attempt replaces is. A report still unfinished when the
    call ends (an error, the brake, an unread item, or skipped at the
    limit), by this call or an earlier one, is listed among the failures,
    and what it spent over every attempt is a Cost line of its own
    (``cost_unfinished``), beside the total for every report.

    A RESUMED REPORT SAYS WHAT IT TOOK FROM ITS FOLDER (BACKLOG RI-125).
    ``triage_reused`` and ``review_reused`` are true when the graph took
    that step's saved file and made no call for it, whatever left the file
    there; ``resumed`` names every such step, the item readings that came
    off disk of those in the record, and the steps a label run seeded
    (:func:`_resumed_fields`). ``label_run_engine`` is read off the label
    run file those steps' ``reused_from`` stamp names, and left out when
    that file is not on disk. The record's own cost is this run's, and
    its ``document.resumed`` adds the whole report's.

    How the counts agree. The record's ``model_calls`` and this file's
    ``cost`` count the same requests, this attempt's (the record also
    counts a call that ended in an error, which ``cost`` keeps under
    ``failed``). The two whole-report figures are different sums:
    ``all_attempts`` adds the meters of the attempts that wrote a run file;
    the record's ``resumed.report_calls`` adds this run's calls to what the
    kept files the record used say their steps cost. They agree when every
    call of every attempt left the file the record used, and they differ:

    - higher in ``all_attempts`` for calls whose file the record did not
      use: a reading that kept no file (a give-up, a step cut short or
      refused by the ceiling); a file superseded or re-keyed (a narrative
      read again under the other ``narrative_skips_bound``, a lab item
      read whole and in windows (``-perK``), an item read again on its own
      pages once a bound document took some (``-own``), log items grouped
      again, a model-first unit read again over its own file, a ranges
      file refused as another describer's or labeller's);
    - higher in ``report_calls`` for calls no run file metered: steps a
      label run seeded (``from_label_run``; paid in the ``labels``
      stage), an attempt killed or interrupted before it wrote its run
      file but after it kept files, a folder copied in or a run file moved
      aside (its earlier attempts go with it), and this attempt's failed
      sends;
    - lower in ``report_calls`` for kept files that record no cost (files
      older than the cost they would carry).
    """
    from report_ingest.calc_reader import page_images_enabled
    from report_ingest.engine import CostMeter
    from report_ingest.graph import Budgets, ingest_report, output_paths
    from report_ingest.lab_reader import (
        floor_per_page_enabled, link_from_label_enabled,
    )
    from report_ingest.log_templates import load_templates

    runs_dir = Path(review_dir) if review_dir is not None else out / "runs"
    if (runs_dir / "runs").is_dir():
        runs_dir = runs_dir / "runs"
    templates = load_templates(templates_path) if templates_path else None
    print(f"  ingest: {len(report_ids)} report(s); model {model}, vision "
          f"{vision_model} ({vision_mode}); label_policy {label_policy}, "
          f"review_mode {review_mode}; a saved label run in {runs_dir} is "
          f"reused where one exists")
    present = set(corpus.present_ids())
    engines = _Engines.of(prompter)
    first = first or _FirstArm()
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    spent = 0.0
    spent_tokens = 0
    # What this arm's folder had spent before the call started (BACKLOG
    # RI-133): the ceiling counts it with the call's own spending.
    folder_tokens = _folder_tokens(out, first) if (
        max_total_tokens is not None) else 0
    if max_total_tokens is not None:
        print(f"  token ceiling {max_total_tokens:,}: the folder had spent "
              f"{folder_tokens:,} tokens (in plus output, every attempt) "
              "before this call")
    for n, rid in enumerate(report_ids, 1):
        folder = out / "ingest" / rid
        run_file = folder / "run.json"
        # The attempts before this one, read off the run file it replaces
        # (BACKLOG RI-124); none on a first attempt or a redo, which starts
        # the report afresh.
        earlier: List[Dict[str, Any]] = []
        clash = _arm_clash(run_file, first)
        if clash:
            failures[rid] = clash
            print(f"  [{n}/{len(report_ids)}] {rid}: skipped -- {clash}")
            continue
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            # A report the token brake cut short is unfinished, like one
            # that failed: its unsaved steps are asked for again (the saved
            # items are kept).
            if (not blob.get("error") and not blob.get("token_brake")
                    and not blob.get("item_errors")):
                done[rid] = blob
                print(f"  [{n}/{len(report_ids)}] {rid}: already done, "
                      "skipping")
                continue
            earlier = _earlier_attempts(blob)
            sent = _attempts_in(earlier)
            if max_report_tokens_in is not None and (
                    sent >= max_report_tokens_in):
                # The per-report limit counts every attempt: a report that
                # has already sent its allowance is not read again, and
                # this call spends nothing on it.
                failures[rid] = (
                    f"its {len(earlier)} earlier attempt(s) sent {sent:,} "
                    f"tokens in, at the {max_report_tokens_in:,} "
                    f"per-report ceiling (max_report_tokens_in); raise it "
                    f"to read the rest")
                print(f"  [{n}/{len(report_ids)}] {rid}: not retried -- "
                      f"{failures[rid]}")
                continue
            if blob.get("item_errors") and not blob.get("token_brake"):
                print(f"  [{n}/{len(report_ids)}] {rid}: "
                      f"{len(blob['item_errors'])} item(s) could not be "
                      "read (a give-up); retrying them")
            elif blob.get("token_brake"):
                print(f"  [{n}/{len(report_ids)}] {rid}: cut short by the "
                      f"token brake ({blob['token_brake'].get('limit')}); "
                      "retrying")
            else:
                print(f"  [{n}/{len(report_ids)}] {rid}: previous attempt "
                      f"failed ({str(blob.get('error'))[:90]}); retrying")
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(report_ids)}] {rid}: skipped -- "
                  f"{failures[rid]}")
            continue
        if max_total_dollars is not None and spent > max_total_dollars:
            print(f"  stopping before {rid}: spent ${spent:.2f}, past the "
                  f"${max_total_dollars:.2f} ceiling")
            break
        if (max_total_tokens is not None
                and folder_tokens + spent_tokens >= max_total_tokens):
            print(f"  stopping before {rid}: {spent_tokens:,} tokens spent "
                  f"by this call and {folder_tokens:,} by the folder before "
                  f"it, past the {max_total_tokens:,} ceiling")
            failures["(token ceiling)"] = (
                f"max_total_tokens {max_total_tokens:,} reached before "
                f"{rid}: the folder had spent {folder_tokens:,} tokens "
                f"when this call started and this call {spent_tokens:,}")
            break
        folder.mkdir(parents=True, exist_ok=True)
        _stamp_arm(folder, first)
        # Seeds the folder; what the graph then took from it is the
        # record's to say (``_resumed_fields``, BACKLOG RI-125).
        _reuse_label_run(folder, runs_dir, rid, review=not first.on)
        meter = CostMeter()
        engine = engines.build(model, meter)
        vision_engine = engines.build(vision_model, meter, vision=True)
        triage_engine = (engines.build(triage_model, meter)
                         if triage_model and triage_model != model else None)
        arm_kwargs, arm_models = _ingest_arm(
            first, engines, meter, model, describe_model or vision_model,
            range_model or model)
        budgets = Budgets(narrative=narrative_budget, log=log_budget,
                          lab=lab_budget,
                          max_request_tokens_in=max_request_tokens_in,
                          lab_sheets_per_call=lab_sheets_per_call,
                          tokens_per_minute=tokens_per_minute)
        brake = _TokenBrake(meter, max_report_tokens_in,
                            None if max_total_tokens is None
                            else max_total_tokens - folder_tokens
                            - spent_tokens,
                            earlier_in=_attempts_in(earlier))
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            record = ingest_report(doc, brake.wrap(engine), out_dir=folder,
                                   budgets=budgets, report_id=rid,
                                   resume=not redo, write=True,
                                   db_path=out / "ingest" / "reports.db",
                                   label_policy=label_policy,
                                   review_mode=review_mode,
                                   vision_engine=brake.wrap(vision_engine),
                                   vision_mode=vision_mode,
                                   vision_detail=vision_detail,
                                   trust_table=trust_table,
                                   templates=templates,
                                   ingest_bound=ingest_bound,
                                   counts_from_logs=counts_from_logs,
                                   narrative_skips_bound=(
                                       narrative_skips_bound),
                                   triage_engine=brake.wrap(triage_engine),
                                   **brake.wrap_kwargs(arm_kwargs))
            n_pages = doc.n_pages
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(report_ids)}] {rid}: FAILED -- "
                  f"{failures[rid]}")
            traceback.print_exc()
            spent_cost = meter.to_dict()
            run_file.write_text(_dumps({
                "id": rid, "run_date": date.today().isoformat(),
                "error": failures[rid], "cost": spent_cost,
                "model_first": first.setting,
                # The earlier attempts ride on, so the next one keeps them
                # (BACKLOG RI-124); this file keeps no wall clock, so this
                # attempt counts its calls' own seconds.
                **_attempts_fields(earlier, spent_cost,
                                   spent_cost.get("seconds"))}),
                encoding="utf-8")
            # What a failed report spent still counts toward the ceiling.
            spent_tokens += _tokens_of(spent_cost)
            _sync(sync)
            continue
        finally:
            if doc is not None:
                doc.close()
        outputs = output_paths(folder)
        (folder / "qa.json").write_text(json.dumps(
            [e.model_dump(mode="json") for e in record.qa], indent=2),
            encoding="utf-8")
        by_kind: Dict[str, int] = {}
        for inv in record.investigations:
            by_kind[inv.kind] = by_kind.get(inv.kind, 0) + 1
        lab_by_kind: Dict[str, int] = {}
        for test in record.lab_tests:
            lab_by_kind[test.kind] = lab_by_kind.get(test.kind, 0) + 1
        qa_by_kind: Dict[str, int] = {}
        for entry in record.qa:
            qa_by_kind[entry.kind] = qa_by_kind.get(entry.kind, 0) + 1
        answered = (len(record.general.answered())
                    + len(record.natural_hazards.answered()))
        from report_ingest.model import GENERAL_FIELDS, NATURAL_HAZARD_FIELDS
        asked = len(GENERAL_FIELDS) + len(NATURAL_HAZARD_FIELDS)
        # The engine the readers ran on, as every stage records it; on a
        # chat-model run the vote's vision voter's and the triage's too
        # (BACKLOG RI-67). A Prompter run says its tier, as it always did.
        engine_fields = engines.record(model, engine)
        if not engines.plain:
            engine_fields["vision"] = engines.record(vision_model,
                                                     vision_engine)
            if triage_engine is not None:
                engine_fields["triage"] = engines.record(triage_model,
                                                         triage_engine)
        reuse_fields = _resumed_fields(record, folder)
        # A reused review made on the other engine is named in the run file
        # and RESULTS; the key exists only then, so a run whose label run
        # shares its engine writes what it always wrote. "Reused" is what
        # the record says this run took from the folder that a label run
        # seeded, on this attempt or an earlier one; the engine is the one
        # the seeding label run file records, and none is named when that
        # file is not on disk (a folder copied in, ``runs/`` moved aside).
        other_engine = ""
        seeded = (reuse_fields.get("resumed") or {}).get("from_label_run")
        if seeded:
            saved_engine = _label_run_engine(folder, seeded)
            now_engine = "prompter" if engines.on_prompter() else "chat_model"
            if saved_engine and saved_engine != now_engine:
                other_engine = saved_engine
        blob = {
            "id": rid,
            "run_date": date.today().isoformat(),
            **engine_fields,
            "served_by": engine.served_by,
            "n_pages": n_pages,
            "workflow": record.document.workflow,
            **reuse_fields,
            **({"label_run_engine": other_engine} if other_engine else {}),
            **({"di_withheld": True} if corpus.di_withheld else {}),
            "labels": _ingest_labels(folder, record, rid, corpus,
                                     oos or {}, mapped),
            "bound": _ingest_bound_rows(record),
            "switches": {"ingest_bound": ingest_bound,
                         "lab_sheets_per_call": lab_sheets_per_call,
                         "counts_from_logs": counts_from_logs,
                         "narrative_skips_bound": narrative_skips_bound},
            **_ingest_arm_fields(first, engines, arm_kwargs, arm_models,
                                 folder),
            "counts": record.counts(),
            "investigations_by_kind": by_kind,
            "lab_by_kind": lab_by_kind,
            "narrative": {"answered": answered, "null": asked - answered},
            "qa_by_kind": qa_by_kind,
            "diggs": _diggs_verdicts(record, outputs),
            "outputs": {k: str(v) for k, v in outputs.items()},
            "scores": _score_ingest_record(
                record, rid, truth, _bound_records(record, folder)),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
            # Calls by step for the report and each report bound inside it
            # (BACKLOG RI-21), and what the visual-scales voters did in its
            # item files (RI-05).
            "visual_scales_on": visual_scales_enabled(),
            # The readers' switches as this stage ran them (RI-67):
            # every calculation page as a picture (RI-17), the
            # specimen-label paragraph (RI-07) and the per-page lab floor
            # in the brief (RI-56). Each calc and lab item file under
            # items/ says the arm its own reader ran in.
            "calc_page_images": page_images_enabled(),
            "lab_link_from_label": link_from_label_enabled(),
            "lab_floor_per_page": floor_per_page_enabled(),
            **_ingest_steps(folder, narrative_skips_bound,
                            lab_sheets_per_call),
            **brake.fields(),
            **_item_error_fields(record),
        }
        # Every attempt's cost (BACKLOG RI-124): ``cost`` and ``seconds``
        # stay this attempt's own; a report read before carries its earlier
        # attempts and their sum with this one.
        blob.update(_attempts_fields(earlier, blob["cost"], blob["seconds"]))
        run_file.write_text(_dumps(blob), encoding="utf-8")
        _sync(sync)
        if not blob.get("item_errors") and not blob.get("token_brake"):
            done[rid] = blob          # unfinished reports are not results
        if blob.get("item_errors") or blob.get("token_brake"):
            # An item the model could not be got to read (a rate-limit
            # give-up, a refused request) is a failed item: the report is
            # saved so a resume reads it again, and the run's failed count
            # (the brief's stop rule) includes it. A report the token brake
            # cut short says so (BACKLOG RI-124).
            failures[rid] = _unfinished_reason(blob)
        spent += blob["cost"].get("dollars", 0.0)
        spent_tokens += _tokens_of(blob["cost"])
        counts = blob["counts"]
        diggs = blob["diggs"]
        every = blob.get("all_attempts")
        took = blob.get("resumed")
        if took:
            print(f"  [{n}/{len(report_ids)}] {rid}: taken from the folder: "
                  + ", ".join(f"items {took['items']} of {took['items_of']}"
                              if step == "items" else step
                              for step in took["steps"]))
        print(f"  [{n}/{len(report_ids)}] {rid}: {n_pages} pp, "
              f"{blob['workflow']}, {counts['investigations']} "
              f"investigation(s), {counts['samples']} sample(s), "
              f"{counts['spt']} drive(s), {counts['lab_tests']} lab test(s), "
              f"{answered}/{asked} narrative fields, {counts['qa']} QA; "
              f"DIGGS {diggs['written']}/{diggs['schema']}/"
              f"{diggs['roundtrip']}; {blob['cost']['calls']} calls, "
              f"{tokens_in(blob['cost']):,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s"
              + (f" (this attempt; over its {every['attempts']} attempts "
                 f"{every['calls']} calls, {tokens_in(every):,} in / "
                 f"{every['output_tokens']:,} out, {every['failed']} failed)"
                 if every else ""))
        # The per-report limit counts every attempt's tokens in (RI-124,
        # RI-171).
        report_in = tokens_in(blob["cost"]) + _attempts_in(earlier)
        if (max_report_tokens_in is not None
                and report_in >= max_report_tokens_in):
            print(f"  stopping after {rid}: {report_in:,}"
                  f" tokens in"
                  + (" over its attempts" if earlier else "")
                  + f", at the {max_report_tokens_in:,} "
                  f"per-report ceiling")
            break
    for rid, blob in _finished_elsewhere(out, first, done).items():
        done[rid] = blob
    # Reports left unfinished in the folder, by this call or an earlier one
    # (BACKLOG RI-124): their cost goes in the Cost block on a line of its
    # own, and each is listed among the failures.
    unfinished = _unfinished_in(out, first, done)
    for rid, blob in unfinished.items():
        failures.setdefault(rid, "left unfinished by an earlier call: "
                                 + _unfinished_reason(blob))
    stage_model, fields = engines.stage(model)
    if not engines.plain:
        seen = engines.describe(vision_model, vision=True)
        fields.update({f"vision_{k}": v for k, v in seen.items()})
    return {**_score_ingest(done, failures, stage_model, unfinished),
            **fields, **_first_results(done.values())}


def _folder_tokens(out: Path, first: "_FirstArm") -> int:
    """Tokens in plus output tokens every report in ``out/ingest/*/`` of this
    arm has spent, over every attempt (BACKLOG RI-133), read before a call
    starts so the call's token ceiling counts the arm folder's earlier
    builds. A run file counts once: its ``all_attempts`` where it has one,
    else its own ``cost`` plus its ``earlier_attempts`` (the helpers of
    RI-124). A report finished earlier and skipped as done is counted here
    and not again by the call."""
    root = out / "ingest"
    total = 0
    if not root.is_dir():
        return total
    for run_file in sorted(root.glob("*/run.json")):
        if _arm_clash(run_file, first):
            continue
        try:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(blob, dict) or blob.get("id") != run_file.parent.name:
            continue
        every = blob.get("all_attempts")
        if isinstance(every, dict):
            total += _tokens_of(every)
            continue
        cost = blob.get("cost") if isinstance(blob.get("cost"), dict) else {}
        total += _tokens_of(cost)
        for row in _attempt_rows(blob):
            if isinstance(row.get("cost"), dict):
                total += _tokens_of(row["cost"])
    return total


def _unfinished_reason(blob: Dict[str, Any]) -> str:
    """Why a report's run file is not a result: the token brake cut it
    short, items could not be read, or the attempt failed."""
    if blob.get("token_brake"):
        limit = (blob["token_brake"].get("limit")
                 if isinstance(blob["token_brake"], dict) else None)
        return (f"cut short by the token brake ({limit}); a resume reads "
                f"the rest")
    if blob.get("item_errors"):
        rows = blob["item_errors"]
        n = len(rows) if isinstance(rows, list) else 1
        return f"{n} item(s) could not be read; a resume retries them"
    return (f"the attempt failed ({str(blob.get('error'))[:90]}); a resume "
            f"retries it")


def _unfinished_in(out: Path, first: "_FirstArm",
                   done: Dict[str, dict]) -> Dict[str, dict]:
    """Run files in ``out/ingest/*/`` of this arm that are not results: an
    error, a token brake or an unread item, and not in ``done`` (BACKLOG
    RI-124). The twin of :func:`_finished_elsewhere`: what these reports
    spent, over every attempt, is in the Cost block too."""
    found: Dict[str, dict] = {}
    root = out / "ingest"
    if not root.is_dir():
        return found
    for run_file in sorted(root.glob("*/run.json")):
        rid = run_file.parent.name
        if rid in done or _arm_clash(run_file, first):
            continue
        try:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (isinstance(blob, dict) and blob.get("id") == rid
                and (blob.get("error") or blob.get("token_brake")
                     or blob.get("item_errors"))):
            found[rid] = blob
    return found


def _item_error_fields(record: Any) -> Dict[str, Any]:
    """``{"item_errors": [...]}`` for the work items (or bound documents)
    the graph could not read, from the record's ``skipped`` QA entries that
    say "could not be read" (BACKLOG RI-110); empty when there are none.

    These are the entries a rate-limit give-up, a refused request or a
    token-brake refusal leave behind: the exception is caught per item so
    the report goes on, and the item has no saved file, so a resume reads
    it again. Without this field the run file carried no sign of them."""
    rows = []
    for entry in getattr(record, "qa", None) or []:
        if entry.kind != "skipped" or "could not be read" not in (
                entry.detail or ""):
            continue
        rows.append({"where": entry.where,
                     "pages": [int(p) for p in (entry.pages or [])],
                     "detail": str(entry.detail)[:300]})
    return {"item_errors": rows} if rows else {}


def _finished_elsewhere(out: Path, first: "_FirstArm",
                        have: Dict[str, dict]) -> Dict[str, dict]:
    """Run files in ``out/ingest/*/`` of reports finished by an earlier
    call (no error, no brake, no unread item), of this arm, that this call
    did not run: RESULTS.md covers every finished report in the folder, not
    only the call's own (BACKLOG RI-110)."""
    found: Dict[str, dict] = {}
    root = out / "ingest"
    if not root.is_dir():
        return found
    for run_file in sorted(root.glob("*/run.json")):
        rid = run_file.parent.name
        if rid in have or _arm_clash(run_file, first):
            continue
        try:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (isinstance(blob, dict) and blob.get("id") == rid
                and not blob.get("error") and not blob.get("token_brake")
                and not blob.get("item_errors") and blob.get("counts")):
            found[rid] = blob
    return found


def _ingest_arm(first: _FirstArm, engines: "_Engines", meter: Any,
                model: str, describe_model: str, range_model: str
                ) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """``(ingest_report's arguments for the arm, the model each engine was
    built for)``. Off names the arm and builds an engine for each stage
    ``first.stage_models`` names (today's reader of that stage); on builds
    the describer, the range labeller and each reader's engine. All are on
    the report's meter."""
    if not first.on:
        if not first.stage_models:
            return {"model_first": "off"}, {}
        names = dict(first.stage_models)
        return ({"model_first": "off",
                 "stage_engines": {stage: engines.build(name, meter)
                                   for stage, name in names.items()}},
                names)
    names = {stage: first.model(stage, model) for stage in FIRST_STAGES}
    names.update(describer=describe_model, labeller=range_model)
    return ({"model_first": "on", "page_route": first.page_route,
             "stage_engines": {stage: engines.build(names[stage], meter)
                               for stage in FIRST_STAGES},
             "describe_engine": engines.build(describe_model, meter,
                                              vision=True),
             "range_engine": engines.build(range_model, meter)}, names)


def _ingest_arm_fields(first: _FirstArm, engines: "_Engines",
                       kwargs: Dict[str, Any], names: Dict[str, str],
                       folder: Path) -> Dict[str, Any]:
    """What one report's ``run.json`` says about its arm: ``model_first``;
    under on the page route asked and each reader's as resolved, the model
    of each reader, the describer and the labeller, the instruction
    version, the describer's and labeller's versions as the saved sections
    record them, and the placement counts over the report's model-first
    items (its bound reports' included). Off, the models of the stages
    named in ``stage_models``, when any are."""
    if not first.on:
        if not names:
            return {"model_first": "off"}
        built = kwargs["stage_engines"]
        return {"model_first": "off",
                "models": {stage: engines.record(names[stage],
                                                 built[stage])["model"]
                           for stage in built}}
    from report_ingest.model_first import READ_PROMPT_VERSION
    from report_ingest.model_first_pages import page_route, resolve_route
    from report_ingest.model_first_scoring import placement_counts

    built = dict(kwargs["stage_engines"],
                 describer=kwargs["describe_engine"],
                 labeller=kwargs["range_engine"])
    models = {what: engines.record(names[what], engine)["model"]
              for what, engine in built.items()}
    ranges: Dict[str, Any] = {}
    for path in sorted(folder.glob("ranges-*.json")):
        blob = _run_blob_on_disk(folder, path.stem)
        ranges = {"arm": path.stem[len("ranges-"):],
                  **{key: blob.get(key) for key in (
                      "describe_prompt", "conventions", "labeller_prompt")}}
        break
    # What the graph placed, from the file it writes beside the items
    # (graph._write_placement): not the item files on disk, which leave out
    # a group part of whose pages went unanswered and keep one a regroup
    # replaced.
    placements: List[Any] = []
    for path in sorted(folder.rglob("placement.json")):
        for found in _run_blob_on_disk(path.parent, "placement").values():
            placements.extend(found or [])
    return {"model_first": "on",
            "page_route": page_route(first.page_route),
            "routes": {stage: resolve_route(
                first.page_route,
                bool(getattr(kwargs["stage_engines"][stage], "accepts_pdf",
                             False))) for stage in FIRST_STAGES},
            "models": models,
            "prompt_version": READ_PROMPT_VERSION,
            "ranges": ranges,
            "placement": placement_counts(placements)}


def _score_ingest(done: Dict[str, dict], failures: Dict[str, str],
                  model: str,
                  unfinished: Optional[Dict[str, dict]] = None
                  ) -> Dict[str, Any]:
    """The ingest stage's results over its finished run files.

    ``cost`` counts EVERY attempt at each report (BACKLOG RI-124): a run
    file's own ``cost`` and ``seconds`` plus its ``earlier_attempts``. When
    any report was read in more than one attempt the results also carry
    ``cost_last_attempt`` (each run file's own ``cost`` alone, the figure
    ``cost`` was before), ``reports_with_earlier_attempts`` (report id to
    how many attempts came before its last) and ``cost_covers``; when none
    was, they carry neither and ``cost`` is what it always was.

    ``unfinished`` holds the run files that are not results
    (:func:`_unfinished_in`). When there are any, ``cost_unfinished`` is
    what they spent over every attempt (with ``reports``, their ids) and
    ``cost_every_report`` is that added to ``cost``; ``cost`` and the
    per-report averages stay over the finished reports."""
    rows = [done[k] for k in sorted(done)]
    cost = _cost_totals()
    own = _cost_totals()
    retried: Dict[str, int] = {}
    totals: Dict[str, Any] = {
        "pages": 0, "investigations": 0, "samples": 0, "spt": 0,
        "lab_tests": 0, "calculations": 0, "qa": 0,
        "bound": {"reports": 0, "documents": 0, "read": 0, "pages": 0,
                  "investigations": 0, "lab_tests": 0, "by_kind": {}},
        "investigations_by_kind": {},
        "lab_by_kind": {}, "qa_by_kind": {},
        "narrative_answered": 0, "narrative_null": 0,
        "labels": {"pages_voted": 0, "split": 0, "review_changed": 0,
                   "review_changed_off_split": 0, "disagreements_left": 0,
                   "scored_pages": 0, "harmful_rules": 0,
                   "harmful_final": 0, "modes": {}, "policies": {}},
        "diggs": {"written": 0, "valid": 0, "not_run": 0, "invalid": 0,
                  "equal": 0, "differs": 0},
        "scores": {"logs": {"found": 0, "total": 0, "n": 0},
                   "lab": {"found": 0, "total": 0, "n": 0},
                   "calc": {"found": 0, "total": 0, "n": 0},
                   "soundings": {"found": 0, "total": 0, "n": 0},
                   "narrative": {"recall_found": 0, "recall_total": 0,
                                 "precision_found": 0, "precision_total": 0,
                                 "n": 0}},
    }
    visual = _visual_counts()
    for row in rows:
        _add_cost(cost, row["cost"])
        cost["seconds"] += row.get("seconds", 0.0)
        _add_cost(own, row["cost"])
        own["seconds"] += row.get("seconds", 0.0)
        earlier = _attempt_rows(row)
        for entry in earlier:
            _add_cost(cost, entry.get("cost"))
            cost["seconds"] += _float(entry.get("seconds"))
        if earlier:
            retried[str(row.get("id"))] = len(earlier)
        _merge_visual(visual, row)
        totals["pages"] += int(row.get("n_pages") or 0)
        counts = row.get("counts") or {}
        for key in ("investigations", "samples", "spt", "lab_tests",
                    "calculations", "qa"):
            totals[key] += int(counts.get(key) or 0)
        for group in ("investigations_by_kind", "lab_by_kind",
                      "qa_by_kind"):
            for kind, n in (row.get(group) or {}).items():
                totals[group][kind] = totals[group].get(kind, 0) + int(n)
        narrative = row.get("narrative") or {}
        totals["narrative_answered"] += int(narrative.get("answered") or 0)
        totals["narrative_null"] += int(narrative.get("null") or 0)
        bound_rows = row.get("bound") or []
        if bound_rows:
            cell = totals["bound"]
            cell["reports"] += 1
            cell["documents"] += len(bound_rows)
            for entry in bound_rows:
                cell["pages"] += int(entry.get("n_pages") or 0)
                cell["read"] += 1 if entry.get("read") else 0
                cell["investigations"] += int(entry.get("investigations") or 0)
                cell["lab_tests"] += int(entry.get("lab_tests") or 0)
                kind = str(entry.get("kind") or "other")
                cell["by_kind"][kind] = cell["by_kind"].get(kind, 0) + 1
        labels = row.get("labels") or {}
        cell = totals["labels"]
        for key in ("pages_voted", "split", "review_changed",
                    "review_changed_off_split", "disagreements_left",
                    "scored_pages"):
            cell[key] += int(labels.get(key) or 0)
        for key in ("rules", "final"):
            cell[f"harmful_{key}"] += int(
                (labels.get("harmful") or {}).get(key) or 0)
        for key, where in (("review_mode", "modes"), ("policy", "policies")):
            name = str(labels.get(key) or "")
            if name:
                cell[where][name] = cell[where].get(name, 0) + 1
        diggs = row.get("diggs") or {}
        if diggs.get("written") == "yes":
            totals["diggs"]["written"] += 1
        if diggs.get("schema") == "valid":
            totals["diggs"]["valid"] += 1
        elif diggs.get("schema") == "not run":
            totals["diggs"]["not_run"] += 1
        elif diggs.get("schema") == "INVALID":
            totals["diggs"]["invalid"] += 1
        if diggs.get("roundtrip") == "equal":
            totals["diggs"]["equal"] += 1
        elif diggs.get("roundtrip") == "DIFFERS":
            totals["diggs"]["differs"] += 1
        scores = row.get("scores") or {}
        for entry in scores.get("logs") or []:
            totals["scores"]["logs"]["found"] += int(
                entry["overall"].get("found") or 0)
            totals["scores"]["logs"]["total"] += int(
                entry["overall"].get("total") or 0)
            totals["scores"]["logs"]["n"] += 1
        for entry in scores.get("lab") or []:
            totals["scores"]["lab"]["found"] += int(
                entry["overall"].get("found") or 0)
            totals["scores"]["lab"]["total"] += int(
                entry["overall"].get("total") or 0)
            totals["scores"]["lab"]["n"] += 1
        for entry in scores.get("calc") or []:
            totals["scores"]["calc"]["found"] += int(
                entry["overall"].get("found") or 0)
            totals["scores"]["calc"]["total"] += int(
                entry["overall"].get("total") or 0)
            totals["scores"]["calc"]["n"] += 1
        for entry in scores.get("soundings") or []:
            totals["scores"]["soundings"]["found"] += int(
                entry["overall"].get("found") or 0)
            totals["scores"]["soundings"]["total"] += int(
                entry["overall"].get("total") or 0)
            totals["scores"]["soundings"]["n"] += 1
        narrative_score = scores.get("narrative")
        if narrative_score:
            cell = totals["scores"]["narrative"]
            cell["recall_found"] += int(
                narrative_score["recall"].get("found") or 0)
            cell["recall_total"] += int(
                narrative_score["recall"].get("total") or 0)
            cell["precision_found"] += int(
                narrative_score["precision"].get("found") or 0)
            cell["precision_total"] += int(
                narrative_score["precision"].get("total") or 0)
            cell["n"] += 1
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "n_reports": len(rows),
        "failures": dict(failures),
        # Each report's run file whole, less its row per request: that list
        # stays in ingest/<ID>/run.json, and results.json keeps the totals.
        "per_report": [{**r, "cost": {k: v for k, v in (r.get("cost")
                                                         or {}).items()
                                      if k != "per_call"}}
                       for r in rows],
        "totals": totals,
        "visual_scales": visual,
        "cost": cost,
        **({"cost_covers": "every attempt at each report (run.json's own "
                           "cost plus its earlier_attempts); "
                           "cost_last_attempt is each run.json's own cost "
                           "alone",
            "cost_last_attempt": own,
            "reports_with_earlier_attempts": retried} if retried else {}),
        **_unfinished_cost(cost, unfinished or {}),
    }


def _unfinished_cost(cost: Dict[str, Any],
                     unfinished: Dict[str, dict]) -> Dict[str, Any]:
    """``cost_unfinished`` (what the reports that are not results spent,
    each run file's own attempt and its earlier ones) and
    ``cost_every_report`` (that added to the finished reports' ``cost``);
    nothing when every report finished (BACKLOG RI-124)."""
    if not unfinished:
        return {}
    spent = _cost_totals()
    for rid in sorted(unfinished):
        # The attempts it carries and its own: every attempt it has had.
        for entry in _earlier_attempts(unfinished[rid]):
            _add_cost(spent, entry.get("cost"))
            spent["seconds"] += _float(entry.get("seconds"))
    every = _cost_totals()
    for part in (cost, spent):
        for key in _COST_SUMS + ("files", "files_counting_failures"):
            every[key] += _n(part.get(key))
        for key in ("dollars", "waited_seconds", "seconds"):
            every[key] += _float(part.get(key))
        if _float(part.get("paced_seconds")):
            every["paced_seconds"] = (_float(every.get("paced_seconds"))
                                      + _float(part.get("paced_seconds")))
        for kind, n in (part.get("failed_by_kind") or {}).items():
            every["failed_by_kind"][kind] = (
                every["failed_by_kind"].get(kind, 0) + _n(n))
    every["waited_seconds"] = round(every["waited_seconds"], 1)
    if "paced_seconds" in every:
        every["paced_seconds"] = round(every["paced_seconds"], 1)
    spent["reports"] = sorted(unfinished)
    return {"cost_unfinished": spent, "cost_every_report": every}


def _kinds_cell(counts: Dict[str, int]) -> str:
    return ", ".join(f"{n} {kind}" for kind, n in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "-"


def _render_ingest_labels(ingest: Dict[str, Any]) -> List[str]:
    """What the page vote did inside the ingest, per report and in total.

    The two numbers worth the space are the SPLIT COUNT -- how many pages a
    production run would send to the expensive look, which it can compute
    with no hand labels at all -- and, where hand labels exist, the final
    labels beside the rules alone, which says whether the vote was worth
    taking.
    """
    rows = [r for r in ingest["per_report"] if r.get("labels")]
    if not rows:
        return []
    cell = (ingest["totals"].get("labels") or {})
    modes = _kinds_cell(cell.get("modes") or {})
    policies = _kinds_cell(cell.get("policies") or {})
    out: List[str] = [
        "", "## The page labels: the vote, and what went to the review", "",
        f"Policy: {policies}. Review mode: {modes}. A `split` page is one "
        f"the cheap voters -- planlens' rules, one vision pass over the "
        f"pages as pictures, and the printed form where a fingerprint file "
        f"is in force -- did not agree on, and in `disagreements` mode it "
        f"is the only kind of page the review is shown. `left` is the "
        f"splits the review did not settle; each is a "
        f"`label_disagreement` QA entry in that report's `qa.json`, "
        f"carrying both voters and both confidences.",
        "", "```",
        f"{'report':<8}{'pages':>7}{'split':>7}{'split %':>9}"
        f"{'reviewed':>10}{'off split':>11}{'left':>7}"
        f"{'rules':>9}{'final':>9}{'scored':>8}{'wrong rdr':>11}"]
    for r in rows:
        row = r["labels"]
        harm = row.get("harmful")
        harm_cell = (f"{harm.get('rules', 0)}/{harm.get('final', 0)}"
                     if harm else "-")
        voted = int(row.get("pages_voted") or 0)
        split = int(row.get("split") or 0)
        accuracy = row.get("accuracy") or {}
        out.append(
            f"{r['id']:<8}{voted:>7}{split:>7}"
            f"{(f'{100.0 * split / voted:.1f}' if voted else '-'):>9}"
            f"{int(row.get('review_changed') or 0):>10}"
            f"{int(row.get('review_changed_off_split') or 0):>11}"
            f"{int(row.get('disagreements_left') or 0):>7}"
            f"{_pct(accuracy.get('rules')):>9}"
            f"{_pct(accuracy.get('final')):>9}"
            f"{int(row.get('scored_pages') or 0):>8}{harm_cell:>11}")
    voted = int(cell.get("pages_voted") or 0)
    split = int(cell.get("split") or 0)
    harm_all = (f"{int(cell.get('harmful_rules') or 0)}/"
                f"{int(cell.get('harmful_final') or 0)}"
                if any(r["labels"].get("harmful") for r in rows) else "")
    out += [
        "",
        f"{'ALL':<8}{voted:>7}{split:>7}"
        f"{(f'{100.0 * split / voted:.1f}' if voted else '-'):>9}"
        f"{int(cell.get('review_changed') or 0):>10}"
        f"{int(cell.get('review_changed_off_split') or 0):>11}"
        f"{int(cell.get('disagreements_left') or 0):>7}"
        f"{'':>9}{'':>9}{int(cell.get('scored_pages') or 0):>8}"
        f"{harm_all:>11}",
        "```"]
    if any((r["labels"].get("accuracy") or {}) for r in rows):
        out += ["", "`rules` and `final` are strict accuracy over the pages "
                    "this report has hand labels for -- the rules alone "
                    "against what the record ended up with -- scored by the "
                    "same scorer as the `labels` and `vision_labels` "
                    "stages. A dash is a report with no hand labels, which "
                    "is unmeasured rather than failing. `wrong rdr` is the "
                    "rules alone / the final labels counted by what a wrong "
                    "label costs: the hand-labelled pages it sends to "
                    "another reader than the hand label does, or to none, "
                    "or to a reader where the hand label sends it to none "
                    "(`report_ingest.scoring.harm_score`)."]
    else:
        out += ["", "No hand labels for these reports, so the final labels "
                    "are unscored; `split` is still the number a production "
                    "run computes for itself."]
    notes = sorted({str(r["labels"].get("note") or "") for r in rows
                    if r["labels"].get("note")})
    if notes:
        out += [""] + [f"Note: {note}" for note in notes]
    return out


_SWITCH_NAMES = (
    ("ingest_bound", "Bound documents read as their own records "
                     "(`ingest_bound`)"),
    ("counts_from_logs", "Counts from the logs (RI-14, `counts_from_logs`)"),
    ("narrative_skips_bound", "Narrative skips bound pages (RI-14, "
                              "`narrative_skips_bound`)"),
    ("lab_sheets_per_call", "Lab sheets per call (RI-116, "
                            "`lab_sheets_per_call`)"),
)


def _switch_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per ingest option a run file records under ``switches``:
    how many reports ran under each value, and MIXED when they differ, as
    the lab stage's arm lines do (BACKLOG RI-128). ``not recorded`` is a run
    file written before the option was recorded."""
    out: List[str] = []
    for key, what in _SWITCH_NAMES:
        counts: Dict[str, int] = {}
        for r in rows:
            got = r.get("switches")
            if not isinstance(got, dict) or key not in got:
                value = "not recorded"
            else:
                value = got[key]
                value = ("on" if value is True else
                         "off" if value is False or value is None
                         else str(value))
            counts[value] = counts.get(value, 0) + 1
        order = sorted(counts, key=lambda v: (v == "not recorded", v))
        line = f"{what}: " + " and ".join(
            f"{v} for {counts[v]} report" + ("" if counts[v] == 1 else "s")
            for v in order)
        if len(order) > 1:
            line += (" -- MIXED: these reports were not all read in one "
                     "recorded arm")
        out.append(line + ".")
    return out


def _render_ingest(ingest: Dict[str, Any]) -> List[str]:
    """The ingest stage: what each record holds, what its exports passed,
    and the record scored against the hand truth where there is any.
    IDs, kinds, counts and rates only."""
    out: List[str] = [
        "", "# Ingest: the record and its exports", "",
        f"Run {ingest['date']}. Model `{ingest['model']}`, "
        f"{ingest['n_reports']} report(s) through the whole pipeline -- "
        f"triage, the label review (reused from a saved label run where one "
        f"exists), the three readers on their floors, the reconciler and "
        f"the writers -- into `ingest/<ID>/`: the record, the summary page, "
        f"the library page, the DIGGS file and `qa.json`.",
        "",
        "**DIGGS** reads `written / schema / read back`: whether a file was "
        "written, whether it validated against the bundled 2.6 schema "
        "(`not run` when no schema check is installed on the host: "
        "neither a pass nor a failure), and whether "
        "reading it back with the app's own parser gives the record's "
        "values. **qa** counts the record's QA entries: `disagreement` is "
        "the two voters inside a reader (the grid or the tables against the "
        "model) splitting on a value, both kept; `partial` is what a reader "
        "could not settle; `out_of_range` is what Python refused.",
    ]
    out += _engine_lines(ingest)
    out += _engine_lines(ingest, "vision_", "The vision voter")
    out += _first_lines(ingest)
    if ingest.get("served_by"):
        out.append(f"Served by: {', '.join(ingest['served_by'])}.")
    crossed = [f"{r['id']} ({r['label_run_engine']})"
               for r in ingest.get("per_report") or []
               if r.get("label_run_engine")]
    if crossed:
        out.append(f"Reused a saved label run made on the other engine "
                   f"than this run's: {', '.join(crossed)}.")
    rows = ingest.get("per_report") or []
    if rows:
        out.append(_arm_line(
            "Every calculation page as a picture (RI-17, "
            "GEOTECH_INGEST_CALC_PAGE_IMAGES)",
            _arms(r.get("calc_page_images") for r in rows), "reports"))
        out.append(_arm_line(
            "Specimen-label paragraph (RI-07, "
            "GEOTECH_INGEST_LAB_LINK_FROM_LABEL)",
            _arms(r.get("lab_link_from_label") for r in rows), "reports"))
        out.append(_arm_line(
            "Per-page lab floor in the brief (RI-56, "
            "GEOTECH_INGEST_LAB_FLOOR_PER_PAGE)",
            _arms(r.get("lab_floor_per_page") for r in rows), "reports"))
    out += _switch_lines(rows)
    if ingest.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in ingest["failures"].items()]

    out += ["", "## Per report", "", "```",
            f"{'report':<8}{'pages':>6}  {'workflow':<14}  "
            f"{'investigations':<28}{'samp':>6}{'spt':>5}  "
            f"{'lab tests':<32}{'calc':>6}{'narr':>7}{'bound':>7}  "
            f"{'qa dis/unres/ref':<17}"
            f"{'diggs written/schema/read back':<32}{'calls':>6}{'in':>10}"
            f"{'out':>8}{'$':>7}{'s':>6}"]
    for r in ingest["per_report"]:
        counts = r.get("counts") or {}
        qa = r.get("qa_by_kind") or {}
        diggs = r.get("diggs") or {}
        narrative = r.get("narrative") or {}
        bound_rows = r.get("bound") or []
        money = r["cost"].get("dollars", 0.0)
        qa_cell = (f"{qa.get('disagreement', 0)}/{qa.get('partial', 0)}/"
                   f"{qa.get('out_of_range', 0)}")
        diggs_cell = (f"{diggs.get('written', '-')}/{diggs.get('schema', '-')}"
                      f"/{diggs.get('roundtrip', '-')}")
        narr_cell = (f"{narrative.get('answered', 0)}/"
                     f"{narrative.get('null', 0)}")
        out.append(
            f"{r['id']:<8}{r.get('n_pages', 0):>6}  "
            f"{str(r.get('workflow') or '-')[:14]:<14}  "
            f"{_kinds_cell(r.get('investigations_by_kind') or {})[:27]:<28}"
            f"{counts.get('samples', 0):>6}{counts.get('spt', 0):>5}  "
            f"{_kinds_cell(r.get('lab_by_kind') or {})[:31]:<32}"
            f"{(counts.get('calculations', 0) or '-'):>6}"
            f"{narr_cell:>7}{(len(bound_rows) or '-'):>7}  "
            f"{qa_cell:<17}{diggs_cell:<32}"
            f"{r['cost'].get('calls', 0):>6}"
            f"{tokens_in(r['cost']):>10,}"
            f"{r['cost'].get('output_tokens', 0):>8,}"
            f"{(f'{money:.2f}' if money else '-'):>7}"
            f"{r.get('seconds', 0.0):>6.0f}")
        # One line per report bound inside that one, indented under it: its
        # own record, its own pages, its own counts. IDs and numbers only.
        for entry in bound_rows:
            held = (f"{entry.get('investigations', 0)} investigation(s), "
                    f"{entry.get('lab_tests', 0)} lab test(s), "
                    f"{entry.get('samples', 0)} sample(s)"
                    if entry.get("read") else "NOT READ")
            out.append(
                f"  +- {entry.get('bound_id', ''):<6}"
                f"{entry.get('n_pages', 0):>4} pp at "
                f"{str(entry.get('pages') or ''):<12}"
                f"{str(entry.get('kind') or ''):<24}"
                f"{held:<52}"
                f"said by {', '.join(entry.get('said_by') or []) or '-'}")
    out.append("```")
    out += ["", "`calc` is how many calculation printouts were read into "
                "the record. `narr` is the owner's schema fields answered / "
                "left null "
                "(37 asked). `qa dis/unres/ref` is disagreement / partial / "
                "out_of_range entries; every kind is in `results.json`. "
                "`bound` counts reports bound INSIDE that report; each gets "
                "the `+-` line under it and is a record of its own under "
                "`ingest/<ID>/bound/<bound id>/`, with its own summary, "
                "library page and DIGGS file. What a bound document holds is "
                "in NONE of the counts on its parent's line: an earlier "
                "firm's borings are that firm's."]

    out += _render_ingest_labels(ingest)

    totals = ingest["totals"]
    diggs = totals["diggs"]
    out += ["", "## Totals", "", "```",
            f"{totals['pages']} pages, {totals['investigations']} "
            f"investigations ({_kinds_cell(totals['investigations_by_kind'])}), "
            f"{totals['samples']} samples, {totals['spt']} driven records, "
            f"{totals['lab_tests']} lab tests "
            f"({_kinds_cell(totals['lab_by_kind'])}), "
            f"{totals['calculations']} calculations",
            f"narrative fields answered {totals['narrative_answered']}, "
            f"null {totals['narrative_null']}",
            f"QA entries {totals['qa']}: "
            f"{_kinds_cell(totals['qa_by_kind'])}",
            f"DIGGS: {diggs['written']} written, {diggs['valid']} valid, "
            f"{diggs['not_run']} not run, {diggs['invalid']} "
            f"invalid; {diggs['equal']} read back equal, {diggs['differs']} "
            f"differ",
            "```"]
    bound = totals.get("bound") or {}
    if bound.get("documents"):
        out += ["", "```",
                f"bound documents: {bound['documents']} inside "
                f"{bound['reports']} report(s) ({_kinds_cell(bound['by_kind'])}), "
                f"{bound['read']} read as their own record, "
                f"{bound['pages']} pages, holding {bound['investigations']} "
                f"investigations and {bound['lab_tests']} lab tests that are "
                f"NOT counted above",
                "```"]

    scores = totals["scores"]
    scored_rows = [r for r in ingest["per_report"]
                   if (r.get("scores") or {}).get("logs")
                   or (r.get("scores") or {}).get("lab")
                   or (r.get("scores") or {}).get("calc")
                   or (r.get("scores") or {}).get("soundings")
                   or (r.get("scores") or {}).get("narrative")]
    if scored_rows:
        out += ["", "## Scored against the hand truth", "",
                "The SAME scorers as the `logs`, `lab`, `calc`, "
                "`soundings` and `narrative` stages, on the record the "
                "whole pipeline produced: a log "
                "truth against the investigations read off its pages, a "
                "sheet truth against the tests read off its page, a "
                "calculation truth against the calculation read off its "
                "run, a sounding truth against the series read off its "
                "sheet, the hand answers against the record's two schemas. "
                "This is the whole-pipeline score.", "", "```",
                f"{'report':<8}{'logs':>18}{'lab sheets':>18}"
                f"{'calculations':>18}{'soundings':>18}"
                f"{'narrative recall':>18}{'precision':>12}"]
        for r in scored_rows:
            sc = r.get("scores") or {}
            logs = sc.get("logs") or []
            lab = sc.get("lab") or []
            calcs = sc.get("calc") or []
            sounds = sc.get("soundings") or []
            nar = sc.get("narrative")

            def agg(entries: List[dict]) -> str:
                if not entries:
                    return f"{'-':>18}"
                found = sum(int(e["overall"].get("found") or 0)
                            for e in entries)
                total = sum(int(e["overall"].get("total") or 0)
                            for e in entries)
                return f"{_rate({'found': found, 'total': total}):>14}"                        f" ({len(entries)})"

            out.append(
                f"{r['id']:<8}{agg(logs)}{agg(lab)}{agg(calcs)}"
                f"{agg(sounds)}"
                + (f"{_rate(nar['recall']):>18}{_rate(nar['precision']):>12}"
                   if nar else f"{'-':>18}{'-':>12}"))
        out += ["",
                f"{'ALL':<8}"
                f"{_rate(scores['logs']):>14} ({scores['logs']['n']})"
                f"{_rate(scores['lab']):>14} ({scores['lab']['n']})"
                f"{_rate(scores['calc']):>14} ({scores['calc']['n']})"
                f"{_rate(scores['soundings']):>14} "
                f"({scores['soundings']['n']})"
                f"{_rate({'found': scores['narrative']['recall_found'], 'total': scores['narrative']['recall_total']}):>18}"
                f"{_rate({'found': scores['narrative']['precision_found'], 'total': scores['narrative']['precision_total']}):>12}",
                "```"]
        # A truth on a bound document's pages meets that document's record
        # (BACKLOG RI-14): said, so a rise there is not read as a reader's.
        elsewhere = [
            f"{r['id']} {e['id']} ({', '.join(e['records'])})"
            for r in scored_rows
            for kind in ("logs", "lab", "calc", "soundings")
            for e in (r.get("scores") or {}).get(kind) or []
            if set(e.get("records") or ()) - {"parent"}]
        if elsewhere:
            out += ["", "Truths met in the record of a report bound inside "
                        "the one scored (a page is read on one side of the "
                        "boundary only): " + "; ".join(elsewhere) + "."]
    else:
        out += ["", "No hand truth for these reports under `truth_dir`, so "
                    "the records are not scored; the counts above are what "
                    "the pipeline produced."]

    extra = (_visual_lines(ingest.get("visual_scales") or {}, "reports")
             + _step_lines(ingest.get("per_report") or []))
    if extra:
        out += [""] + extra[:-1]
    cost = ingest["cost"]
    n = max(1, ingest["n_reports"])
    # BACKLOG RI-124: when a report was read in more than one attempt the
    # totals count every attempt, and say so; each report's last attempt
    # alone, the figure the block printed before, stands beside them.
    own = ingest.get("cost_last_attempt")
    retried = ingest.get("reports_with_earlier_attempts") or {}
    over = " over every attempt" if own else ""
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls{over}, "
            f"{tokens_in_words(cost)}, "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            _failure_line(cost),
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{tokens_in(cost) / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}"]
    if own:
        out += [f"each report's last attempt alone: {own['calls']} model "
                f"calls, {tokens_in_words(own)}, "
                f"{own['output_tokens']:,} output, {own['seconds']:.0f} s"
                f"{_money(own)}, {own['failed']} failed requests",
                "read in more than one attempt: " + ", ".join(
                    f"{rid} ({k + 1} attempts)"
                    for rid, k in sorted(retried.items()))]
    # The reports that are not results, over every attempt, and every
    # report's total (BACKLOG RI-124); nothing when all of them finished.
    left = ingest.get("cost_unfinished")
    every = ingest.get("cost_every_report")
    if left and every:
        kinds = _failed_kinds(left)
        out += [f"reports not finished (listed under failures): "
                f"{len(left.get('reports') or ())}, {left['calls']} calls, "
                f"{tokens_in(left):,} in / {left['output_tokens']:,} "
                f"out, {left['failed']} failed requests"
                + (f" ({kinds})" if kinds else "")
                + f", over every attempt{_money(left)}",
                f"every report, finished or not: {every['calls']} calls, "
                f"{tokens_in(every):,} in / {every['output_tokens']:,} "
                f"out, {every['failed']} failed requests, "
                f"{every['seconds']:.0f} s{_money(every)}"]
    out += ["```", "",
            _price_note(every or cost, ingest)]
    if own:
        out += ["", "The totals above count EVERY attempt at each report: a "
                    "report read again after an error, a give-up "
                    "(`item_errors`) or the token brake carries its earlier "
                    "attempts' cost in its `run.json` (`earlier_attempts`, "
                    "`all_attempts`), and that is added here (BACKLOG "
                    "RI-124). The per-report table shows each report's "
                    "last attempt; the calls-by-step table counts the step "
                    "files saved, from whichever attempt saved them."]
    return out


def _money(cost: Dict[str, Any], per: float = 1.0) -> str:
    """``", $12.34"`` for a priced cost block, or nothing at all.

    Nothing, not ``$0.00``: a run whose deployment has no rate on file cost
    real money that this package cannot name, and printing a zero would say
    it was free.
    """
    dollars = float(cost.get("dollars") or 0.0)
    if not dollars:
        return ""
    each = dollars / max(per, 1e-9)
    return f", ${each:,.2f}" if each >= 0.01 else f", ${each:.4f}"


def _price_note(cost: Dict[str, Any],
                results: Optional[Dict[str, Any]] = None) -> str:
    """The line under a cost block saying what its dollars are, or are not.
    ``results`` is the stage's results: on a chat-model run the note names
    no Funhouse tier or budget endpoint, which that run never used."""
    if results is not None and _through(results) == "a chat model":
        if float(cost.get("dollars") or 0.0):
            return ("Dollars are priced from the rates on file for the "
                    "model that answered. A call with no rate on file adds "
                    "its tokens and no dollars, so read a total as a floor "
                    "and check the provider's own usage report for the "
                    "same window.")
        return ("A chat-model call carries no per-token price here, so "
                "this reports TOKENS rather than dollars. Read the spend "
                "from the provider's own usage report for the same "
                "window.")
    if float(cost.get("dollars") or 0.0):
        return ("Dollars are the owner's own Funhouse rates, read from the "
                "budget page on 2026-09-18, applied to the DEPLOYMENT that "
                "answered rather than to the tier that was asked for -- a "
                "tier is an alias and the deployment behind it changes. A "
                "call served by a deployment with no rate on file adds its "
                "tokens and no dollars, so read a total as a floor and check "
                "Funhouse's own budget endpoint for the same window.")
    return ("Funhouse publishes no per-token price for a capability tier, and "
            "no deployment that answered this run has a rate on file either, "
            "so this reports TOKENS rather than dollars. Read the spend from "
            "Funhouse's own budget endpoint for the same window.")


#: What a stage's cost total adds up over its run files. ``failed``,
#: ``retried``, ``reasoning_tokens`` and ``reasoning_reported`` (the replies
#: that said how much of their output was reasoning) came with BACKLOG
#: RI-21: a run file written before then has none of them and adds nothing.
#: ``cache_write_tokens`` with RI-171, so a stage's tokens in count it.
_COST_SUMS: Tuple[str, ...] = ("calls", "failed", "retried", "input_tokens",
                               "output_tokens", "cache_read_tokens",
                               "cache_write_tokens", "reasoning_tokens",
                               "reasoning_reported")


def _dumps(blob: Dict[str, Any]) -> str:
    """A run file as JSON, indented like the others, with the cost record's
    rows one per line: an 897-call report's rows run to thousands of lines
    indented, and nobody reads them one field at a time (BACKLOG RI-21)."""
    cost = blob.get("cost")
    rows = cost.get("per_call") if isinstance(cost, dict) else None
    if not rows:
        return json.dumps(blob, indent=2)
    marker = "per_call rows written below"
    text = json.dumps({**blob, "cost": {**cost, "per_call": marker}},
                      indent=2)
    lines = ",\n".join("      " + json.dumps(row, separators=(",", ":"))
                       for row in rows)
    return text.replace(json.dumps(marker), "[\n" + lines + "\n    ]", 1)


def _cost_totals() -> Dict[str, Any]:
    """A stage's cost total before any run file is added."""
    out: Dict[str, Any] = {key: 0 for key in _COST_SUMS}
    out.update({"failed_by_kind": {}, "waited_seconds": 0.0, "dollars": 0.0,
                "seconds": 0.0, "files": 0, "files_counting_failures": 0})
    return out


def _add_cost(total: Dict[str, Any], cost: Optional[Dict[str, Any]]) -> None:
    """One run file's cost block, added into a stage's total. ``seconds`` is
    left to the caller, which adds the run file's own wall clock."""
    cost = cost if isinstance(cost, dict) else {}
    for key in _COST_SUMS:
        total[key] = total.get(key, 0) + _n(cost.get(key))
    total["dollars"] = (total.get("dollars", 0.0)
                        + float(cost.get("dollars") or 0.0))
    total["waited_seconds"] = round(total.get("waited_seconds", 0.0)
                                    + float(cost.get("waited_seconds") or 0.0),
                                    1)
    # Only once a run file says a pacer held a request (BACKLOG RI-127,
    # RI-139): with no pacer the total is what it always was.
    paced = _float(cost.get("paced_seconds"))
    if paced:
        total["paced_seconds"] = round(
            _float(total.get("paced_seconds")) + paced, 1)
    total["files"] = total.get("files", 0) + 1
    if "failed" in cost:
        total["files_counting_failures"] = (
            total.get("files_counting_failures", 0) + 1)
    kinds = total.setdefault("failed_by_kind", {})
    by_kind = cost.get("failed_by_kind")
    for kind, n in (by_kind.items() if isinstance(by_kind, dict) else ()):
        kinds[kind] = kinds.get(kind, 0) + _n(n)


def _n(value: Any) -> int:
    """A count read off a run file, 0 where it is not one: a count is never
    the reason a run's RESULTS.md is not written."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


#: How each kind of failed request reads in RESULTS.md
#: (:data:`report_ingest.engine.FAILURE_KINDS`).
_FAILURE_WORDS: Dict[str, str] = {
    "rate_limit": "rate limited", "parameter": "a parameter refused",
    "too_large": "request too large", "timeout": "timed out",
    "no_reply": "nothing back", "other": "other errors"}


def _failed_kinds(cost: Dict[str, Any]) -> str:
    """``"4 rate limited, 2 timed out"`` for a cost block's failed requests,
    most first; empty when none failed."""
    return ", ".join(
        f"{n} {_FAILURE_WORDS.get(kind, kind)}"
        for kind, n in sorted((cost.get("failed_by_kind") or {}).items(),
                              key=lambda kv: (-kv[1], kv[0])) if n)


def _failure_line(cost: Dict[str, Any]) -> str:
    """The cost block's line on the requests that came back with an error,
    how many of them the engine sent again, and the reasoning tokens
    (BACKLOG RI-21). ``calls`` above counts replies only."""
    files = int(cost.get("files") or 0)
    counted = int(cost.get("files_counting_failures") or 0)
    if files and not counted:
        return ("failed requests: not recorded (these run files were "
                "written before failed requests were counted)")
    failed = int(cost.get("failed") or 0)
    kinds = _failed_kinds(cost)
    line = (f"failed requests, not among the calls above: {failed}"
            + (f" ({kinds})" if kinds else "")
            + f", {int(cost.get('retried') or 0)} of them sent again by the "
              f"engine")
    waited = float(cost.get("waited_seconds") or 0.0)
    if waited:
        line += f", {waited:.0f} s waiting on rate limits"
    # The pacer's holds (BACKLOG RI-127, RI-139): only when it held any.
    paced = _float(cost.get("paced_seconds"))
    if paced:
        line += f", {paced:.0f} s held back by the pacer (tokens_per_minute)"
    calls = int(cost.get("calls") or 0)
    reported = int(cost.get("reasoning_reported") or 0)
    if not reported:
        # Not a reported 0: Claude, for one, gives no split.
        line += "; reasoning: not reported"
    else:
        line += (f"; reasoning: {int(cost.get('reasoning_tokens') or 0):,} "
                 f"of the output tokens"
                 + (f", in the {reported} of {calls} calls that reported it"
                    if reported < calls else ""))
    if files and counted < files:
        line += (f" (counted in {counted} of {files} run files; the rest "
                 f"were written before failed requests were counted)")
    return line


# ---------------------------------------------------------------------------
# what the visual-scales voters did (BACKLOG RI-05, G3)
# ---------------------------------------------------------------------------

def _page_sources(doc: Any, pages: Sequence[int]) -> Dict[str, int]:
    """How many of a run's pages are scans (mostly picture, few drawn
    lines) and how many read their text from Azure Document Intelligence:
    what the visual-scales voters had to work on. Never fails a run."""
    out = {"pages": len(pages), "raster": 0, "di_text": 0}
    try:
        from planlens.document.model import SOURCE_AZURE_DI
        from planlens.document.raster import is_raster_page
    except Exception:                       # noqa: BLE001 - a count only
        return out
    for page in pages:
        try:
            if is_raster_page(doc._doc[int(page)]):
                out["raster"] += 1
            if (doc.summary(int(page)).evidence.get("text_source")
                    == SOURCE_AZURE_DI):
                out["di_text"] += 1
        except Exception:                   # noqa: BLE001 - a count only
            continue
    return out


def _visual_counts() -> Dict[str, Any]:
    """A stage's visual-scales counts before any run file is added."""
    return {"files": 0, "files_on": 0, "files_not_recorded": 0,
            "pages": 0, "raster": 0, "di_text": 0,
            "label_calls": 0, "pixel_layers": 0,
            "layer_votes": {"agree": 0, "disagree": 0, "no_record_layer": 0},
            "plot_checks": {"checked": 0, "no_plot": 0, "unavailable": 0},
            "plot_points": {"table": 0, "agree": 0, "disagree": 0,
                            "no_marker_at_size": 0},
            "trace_checks": {"checked": 0, "no_plot": 0},
            "trace_points": {"compared": 0, "agree": 0, "disagree": 0,
                             "several": 0, "no_trace": 0},
            "trace_channels_no_panel": 0}


def _add_checks(total: Dict[str, Any], info: Any = None, plot: Any = None,
                trace: Any = None) -> None:
    """One reading's visual-scales results, added into ``total``: ``info``
    from :func:`report_ingest.visual_scales.log_grid_for` (with the layer
    votes), ``plot`` from ``plot_check``, ``trace`` from ``trace_check``.
    What is not in the shape those write is passed over."""
    def listed(value: Any) -> List[Any]:
        return value if isinstance(value, list) else []

    if isinstance(info, dict):
        total["label_calls"] += _n(info.get("label_calls"))
        total["pixel_layers"] += _n(info.get("pixel_layers"))
        for vote in listed(info.get("layer_votes")):
            said = str(vote.get("agreement") or "") \
                if isinstance(vote, dict) else ""
            if said in total["layer_votes"]:
                total["layer_votes"][said] += 1
    if isinstance(plot, dict):
        total["label_calls"] += _n(plot.get("label_calls"))
        status = str(plot.get("status") or "")
        if status in total["plot_checks"]:
            total["plot_checks"][status] += 1
        if status == "checked":
            points = total["plot_points"]
            points["table"] += _n(plot.get("table_points"))
            points["agree"] += _n(plot.get("agree"))
            points["disagree"] += len(listed(plot.get("disagree")))
            points["no_marker_at_size"] += _n(plot.get("no_marker_at_size"))
    if isinstance(trace, dict):
        total["label_calls"] += _n(trace.get("label_calls"))
        status = str(trace.get("status") or "")
        if status in total["trace_checks"]:
            total["trace_checks"][status] += 1
        channels = trace.get("channels")
        for row in (channels.values() if isinstance(channels, dict) else ()):
            if not isinstance(row, dict):
                continue
            if row.get("status") == "no_panel":
                total["trace_channels_no_panel"] += 1
                continue
            points = total["trace_points"]
            for key in ("compared", "agree", "several", "no_trace"):
                points[key] += _n(row.get(key))
            points["disagree"] += len(listed(row.get("disagree")))


def _add_visual(total: Dict[str, Any], blob: Dict[str, Any]) -> None:
    """One logs, lab or soundings run file, added into a stage's counts."""
    total["files"] += 1
    if "visual_scales_on" not in blob:
        total["files_not_recorded"] += 1
    elif blob["visual_scales_on"]:
        total["files_on"] += 1
    sources = blob.get("page_sources")
    for key in ("pages", "raster", "di_text"):
        total[key] += _n(sources.get(key)) if isinstance(sources, dict) else 0
    after = blob.get("after") if isinstance(blob.get("after"), dict) else {}
    _add_checks(total, after.get("visual_scales"), after.get("plot_check"),
                after.get("code_trace"))


def _merge_visual(total: Dict[str, Any], row: Dict[str, Any]) -> None:
    """One ingest run file, added into the stage's visual-scales counts:
    the counts :func:`_ingest_steps` summed over its item files."""
    total["files"] += 1
    part = row.get("visual_scales")
    if "visual_scales_on" not in row or not isinstance(part, dict):
        total["files_not_recorded"] += 1
        return
    if row["visual_scales_on"]:
        total["files_on"] += 1
    for key, value in part.items():
        if key.startswith("files"):
            continue
        if isinstance(value, dict):
            cell = total.setdefault(key, {})
            for sub, n in value.items():
                cell[sub] = cell.get(sub, 0) + _n(n)
        else:
            total[key] = total.get(key, 0) + _n(value)


def _visual_lines(counts: Dict[str, Any], what: str = "run files"
                  ) -> List[str]:
    """The ``## Visual scales`` block: whether the setting was on, what the
    pages were, and what each voter found. Nothing for run files written
    before any of this was recorded."""
    files = int(counts.get("files") or 0)
    if not files or counts.get("files_not_recorded", 0) >= files:
        return []
    on = int(counts.get("files_on") or 0)
    out = ["## Visual scales", "",
           f"Setting (GEOTECH_INGEST_VISUAL_SCALES) on for {on} of "
           f"{files} {what}"
           + (f"; {counts['files_not_recorded']} written before this was "
              f"recorded" if counts.get("files_not_recorded") else "")
           + ". With it off, every count below is zero by design.", "", "```"]
    if counts.get("pages"):
        out.append(f"pages read: {counts['pages']}, of them "
                   f"{counts['raster']} scans and {counts['di_text']} with "
                   f"their text from Azure Document Intelligence")
    out.append(f"label-reading calls: {counts['label_calls']} (counted in "
               f"the calls above)")
    votes = counts["layer_votes"]
    out.append(f"layer tops measured from a scan's stratum lines: "
               f"{counts['pixel_layers']}; votes against the record's tops: "
               f"{votes['agree']} agree, {votes['disagree']} disagree, "
               f"{votes['no_record_layer']} with no record layer to compare")
    checks, points = counts["plot_checks"], counts["plot_points"]
    if any(checks.values()):
        out.append(f"grading curves checked against the table: "
                   f"{checks['checked']} (no plot found on "
                   f"{checks['no_plot']}, unreadable on "
                   f"{checks['unavailable']}); table points "
                   f"{points['table']}: {points['agree']} agree, "
                   f"{points['disagree']} disagree, "
                   f"{points['no_marker_at_size']} with no marker at their "
                   f"size")
    traces, tpoints = counts["trace_checks"], counts["trace_points"]
    if any(traces.values()):
        out.append(f"plotted soundings traced by code: {traces['checked']} "
                   f"(no plot found on {traces['no_plot']}; "
                   f"{counts['trace_channels_no_panel']} channel(s) with no "
                   f"panel); points {tpoints['compared']} compared: "
                   f"{tpoints['agree']} agree, {tpoints['disagree']} "
                   f"disagree; {tpoints['several']} with several traces "
                   f"at the depth, {tpoints['no_trace']} with none")
    out += ["```", ""]
    return out


def _item_step(blob: Dict[str, Any]) -> str:
    """Which reader wrote an item file, by what it holds (an item file does
    not name its reader). The sounding reader's result says whether the
    sheet was tabulated and how many points the floor placed; the log
    reader's never does, whatever kind of investigation it read."""
    if "general" in blob:
        return "narrative"
    if "calculation" in blob:
        return "calc"
    if "tests" in blob:
        return "lab"
    if "tabulated" in blob or "floor_points" in blob:
        return "sounding"
    return "log"


def _ingest_steps(folder: Path,
                  narrative_skips_bound: Optional[bool] = None,
                  lab_sheets_per_call: Optional[int] = None
                  ) -> Dict[str, Any]:
    """Model calls by step for the report and for each report bound inside
    it, and what the visual-scales voters did, read off the step and item
    files under ``folder`` (BACKLOG RI-21, RI-05).

    A step file kept from an earlier attempt counts the calls it took
    then, so on a resumed run these can add up to more than the run
    file's own ``cost``, which counts this attempt only. A narrative item
    file left behind by a re-read (the newest one is the reading the record
    used) is not counted in the step cells; it is listed apart under
    ``superseded_item_files`` (BACKLOG RI-129), as are a lab item's other
    reading (RI-138) and a log item's file read whole where its parts, one
    per hole, are beside it (RI-117). Never fails a
    run: it is called after the run's model calls are paid for, and an
    item file it cannot read is counted as unreadable and passed over.
    """
    def load(path: Path) -> Dict[str, Any]:
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return blob if isinstance(blob, dict) else {}

    def calls(blob: Dict[str, Any]) -> int:
        cost = blob.get("cost")
        return _n(cost.get("calls")) if isinstance(cost, dict) else 0

    steps: Dict[str, Dict[str, int]] = {}
    visual = _visual_counts()
    unreadable = 0
    superseded: Dict[str, Dict[str, Dict[str, int]]] = {}
    try:
        places = [("report", folder)]
        if (folder / "bound").is_dir():
            places += [(p.name, p)
                       for p in sorted((folder / "bound").iterdir())
                       if p.is_dir()]
        for name, where in places:
            row: Dict[str, int] = steps.setdefault(name, {})
            for step in ("triage", "vision", "review", "identity"):
                n = calls(load(where / f"{step}.json"))
                if n:
                    row[step] = n
            paths = sorted((where / "items").glob("*.json"))
            # The narrative is read once per document, but a re-read under
            # another switch or item range keeps the earlier file beside
            # the new one (graph.py names it by that key). Only the newest
            # is the reading the record used; older ones are counted apart
            # (BACKLOG RI-129).
            narrative = [(path, blob) for path in paths
                         for blob in (load(path),) if "general" in blob]
            kept_old: set = set()
            if len(narrative) > 1:
                # Exact: graph.py puts ``narrative_key`` ("-skips-bound")
                # on the file name exactly when the run's
                # ``narrative_skips_bound`` was on. Only when the switch is
                # not recorded, or no file carries its key, is the newest
                # by mtime taken (a download or unzip resets mtimes).
                chosen_by = "mtime"
                pool = narrative
                if narrative_skips_bound is not None:
                    match = [pair for pair in narrative
                             if pair[0].stem.endswith("-skips-bound")
                             == bool(narrative_skips_bound)]
                    if match:
                        pool, chosen_by = match, "key"

                def age(pair: Any) -> Tuple[float, str]:
                    try:
                        return (pair[0].stat().st_mtime, pair[0].name)
                    except OSError:
                        return (0.0, pair[0].name)
                used = max(pool, key=age)[0]
                kept_old = {pair[0] for pair in narrative
                            if pair[0] != used}
                old = {"files": len(kept_old), "chosen_by": chosen_by,
                       "calls": sum(calls(blob) for path, blob in narrative
                                    if path in kept_old)}
                superseded.setdefault(name, {})["narrative"] = old
            # A lab item read whole is ``item_N.json`` and read in windows
            # ``item_N-perK.json`` (lab_reader.split_key); a reused folder
            # can hold both. The run's recorded ``lab_sheets_per_call`` K
            # says which it used: the ``-perK`` file when there is one,
            # else the whole one (an item of K pages or fewer is not split)
            # (BACKLOG RI-138).
            lab_groups: Dict[str, List[Path]] = {}
            for path in paths:
                if path in kept_old:
                    continue
                base = re.sub(r"-per\d+$", "", path.stem)
                if base != path.stem or "tests" in load(path):
                    lab_groups.setdefault(base, []).append(path)
            lab_old: set = set()
            for base, files in lab_groups.items():
                if len(files) < 2:
                    continue
                want = (f"{base}-per{int(lab_sheets_per_call)}"
                        if lab_sheets_per_call is not None else base)
                used = [f for f in files if f.stem == want] or \
                    [f for f in files if f.stem == base]
                if not used:
                    continue
                lab_old |= {f for f in files if f != used[0]}
            if lab_old:
                got = superseded.setdefault(name, {}).setdefault(
                    "lab", {"files": 0, "calls": 0,
                            "chosen_by": "lab_sheets_per_call"})
                got["files"] += len(lab_old)
                got["calls"] += sum(calls(load(f)) for f in lab_old)
                kept_old |= lab_old
            # A log item cut into one item per hole is read as ``item_Na``,
            # ``item_Nb`` ... (log_holes.split_log_items); a folder reused
            # from a run before the cut keeps ``item_N`` beside them. The
            # parts are the reading the record used (BACKLOG RI-117). An
            # item whose bound pages were handed over carries ``-own``
            # (graph.py) on the whole item and on its parts alike.
            stems = {path.stem: path for path in paths
                     if path not in kept_old}
            cut: Dict[str, List[Path]] = {}
            for stem, path in stems.items():
                own = "-own" if stem.endswith("-own") else ""
                base = stem[:len(stem) - len(own)]
                whole = base[:-1] + own
                if base[-1:].islower() and whole in stems:
                    cut.setdefault(whole, []).append(path)
            log_old = {stems[base] for base, parts in cut.items()
                       if _item_step(load(stems[base])) == "log"
                       and any(_item_step(load(p)) == "log" for p in parts)}
            if log_old:
                superseded.setdefault(name, {})["log"] = {
                    "files": len(log_old), "chosen_by": "split_parts",
                    "calls": sum(calls(load(f)) for f in log_old)}
                kept_old |= log_old
            for path in paths:
                if path in kept_old:
                    continue
                blob = load(path)
                if not blob:
                    unreadable += 1
                    continue
                try:
                    step = _item_step(blob)
                    row[step] = row.get(step, 0) + calls(blob)
                    row[f"{step} items"] = row.get(f"{step} items", 0) + 1
                    _add_checks(visual, blob.get("visual_scales"),
                                blob.get("plot_check"), blob.get("code_trace"))
                except Exception:           # noqa: BLE001 - a count only
                    unreadable += 1
    except Exception:                       # noqa: BLE001 - a count only
        unreadable += 1
    out: Dict[str, Any] = {"calls_by_step": steps, "visual_scales": visual}
    if unreadable:
        out["step_files_unreadable"] = unreadable
    if superseded:
        out["superseded_item_files"] = superseded
    return out


#: The steps of an ingest run, in the order they run: the page labels
#: (triage, the vision pass, the review), a bound document's identity call,
#: and the readers.
_STEPS: Tuple[str, ...] = ("triage", "vision", "review", "identity", "log",
                           "sounding", "lab", "calc", "narrative")


def _step_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The ``## Calls by step`` table: the model calls of each report and of
    each report bound inside it, step by step, and on the report's row what
    its cost counts that no step or item file does (BACKLOG RI-21). Nothing
    for run files written before this was recorded.

    The steps a label run seeded (``resumed.from_label_run`` in the run
    file) keep their calls in their cells but not in ``unattributed``: no
    ingest attempt made them, and each such report gets a line of its own
    saying so (BACKLOG RI-125)."""
    table = []
    seeded_lines: List[str] = []
    for r in rows:
        documents = r.get("calls_by_step")
        if not isinstance(documents, dict) or not documents:
            continue
        totals = {name: sum(_n(v) for k, v in steps.items()
                            if not k.endswith(" items"))
                  for name, steps in documents.items()}
        cost = r.get("cost") if isinstance(r.get("cost"), dict) else {}
        resumed = r.get("resumed") if isinstance(r.get("resumed"),
                                                 dict) else {}
        seeded = [str(step) for step in resumed.get("from_label_run") or ()]
        own = documents.get("report") if isinstance(
            documents.get("report"), dict) else {}
        from_labels = sum(_n(own.get(step)) for step in seeded)
        left = _n(cost.get("calls")) - (sum(totals.values()) - from_labels)
        if seeded:
            seeded_lines += [
                f"{r.get('id', '')}: {', '.join(seeded)} seeded from a label "
                f"run ({from_labels} calls), not this ingest's: in the cells "
                f"above, not in `unattributed`.", ""]
        for i, (name, steps) in enumerate(documents.items()):
            table.append((r.get("id", ""), name, steps, totals[name],
                          f"{left:+d}" if i == 0 else ""))
    if not table:
        return []
    out = ["## Calls by step", "",
           "Model calls each step took, read off the step and item files of "
           "the report and of each report bound inside it (`bound1`, ...). "
           "A reader's cell is calls/items. `unattributed`, on the report's "
           "row, is the report's calls in the cost below less every "
           "document's total here, the calls of a step a label run seeded "
           "left out (listed under the table): above zero, calls no file "
           "holds (a reading that failed, or was cut short by the ceiling, "
           "keeps no file); below zero, step files kept from an earlier "
           "attempt, whose calls this attempt did not make.", "", "```",
           f"{'report':<8}{'document':<10}"
           + "".join(f"{step:>10}" for step in _STEPS)
           + f"{'total':>8}{'unattributed':>14}"]
    for rid, name, steps, total, left in table:
        cells = ""
        for step in _STEPS:
            n, items = _n(steps.get(step)), _n(steps.get(f"{step} items"))
            cell = f"{n}/{items}" if items else (str(n) if n else "-")
            cells += f"{cell:>10}"
        out.append(f"{rid:<8}{name:<10}{cells}{total:>8}{left:>14}")
    out += ["```", ""] + seeded_lines
    # A narrative file left by a re-read is kept apart, not in the cells
    # above (BACKLOG RI-129).
    for r in rows:
        kept = r.get("superseded_item_files")
        for doc, steps in (kept.items() if isinstance(kept, dict) else ()):
            for step, got in (steps.items() if isinstance(steps, dict)
                              else ()):
                if not (isinstance(got, dict) and got.get("files")):
                    continue
                if step == "lab":
                    out += [f"{r.get('id', '')} {doc}: {got['files']} lab "
                            f"item file(s) of the other reading (whole or "
                            f"in windows) kept from an earlier run "
                            f"({_n(got.get('calls'))} calls), not counted "
                            f"above; the file used was the one matching "
                            f"the run's recorded lab_sheets_per_call.", ""]
                    continue
                if step == "log":
                    out += [f"{r.get('id', '')} {doc}: {got['files']} log "
                            f"item file(s) read whole before the item was "
                            f"cut into one item per hole, kept from an "
                            f"earlier run ({_n(got.get('calls'))} calls), "
                            f"not counted above; the parts' files are the "
                            f"ones used.", ""]
                    continue
                out += [f"{r.get('id', '')} {doc}: {got['files']} older "
                        f"narrative item file(s) kept from a re-read "
                        f"({_n(got.get('calls'))} calls), not counted "
                        f"above; the file used was "
                        + ("picked by the run's recorded "
                           "narrative_skips_bound switch."
                           if got.get("chosen_by") == "key" else
                           "taken as the newest by file time, the switch "
                           "not being recorded (or no file carrying its "
                           "key)."), ""]
    return out


def _header(results: Dict[str, Any]) -> List[str]:
    """The heading a run writes when the labels stage did not run."""
    head = ["# Report ingest on the cluster", "",
            f"Run {results['date']}. Stage(s): "
            f"{', '.join(results.get('stages') or ())}."]
    return head + _di_lines(results)


def _di_lines(results: Dict[str, Any]) -> List[str]:
    """One line when the run withheld Document Intelligence text; none
    otherwise, so a run that did not is written as it always was."""
    if results.get("di_withheld"):
        return ["Document Intelligence text: withheld (`di_dir=False`); "
                "every report was read on its own text layer."]
    return []


def _render(results: Dict[str, Any]) -> List[str]:
    """RESULTS.md: IDs, labels, counts and rates. Nothing that names anyone."""
    out: List[str] = [
        f"# WP1b on the cluster: triage and label review through "
        f"{_through(results)}",
        "",
        f"Run {results['date']}. Review model `{results['review_model']}`, "
        f"triage `{results['triage_model']}`.",
    ]
    out += _engine_lines(results)
    out += _di_lines(results)
    if results.get("served_by"):
        if _through(results) == "a chat model":
            out.append(f"Served by: {', '.join(results['served_by'])}.")
        else:
            out.append(f"Served by: {', '.join(results['served_by'])}. A "
                       f"Funhouse tier is an alias and the deployment "
                       f"behind it changes, so this is what actually "
                       f"answered.")
    if results.get("absent"):
        out += ["", f"Not in the reports folder, so not run: "
                    f"{', '.join(results['absent'])}."]
    if results.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in results["failures"].items()]

    for name, row in results["sets"].items():
        if not row["n_reports"]:
            continue
        before, after = row["_scores"]
        out += ["", f"## {name} -- {row['n_reports']} report(s), "
                    f"{after.n} scored pages", ""]
        if row["disputed_dropped"]:
            out.append(f"{row['disputed_dropped']} page(s) dropped from both "
                       f"scores as confirmed disputed hand labels.")
            out.append("")
        out += ["```",
                f"{'':<24}{'before':>10}{'after':>10}",
                f"{'strict accuracy':<24}{before.accuracy:>10.3f}"
                f"{after.accuracy:>10.3f}",
                f"{'accepting alternates':<24}"
                f"{before.lenient_accuracy:>10.3f}"
                f"{after.lenient_accuracy:>10.3f}",
                "",
                f"key content (the gate is {GATE:.2f} on both rates after "
                f"review)"]
        out += label_table(before, after, KEY_CONTENT)
        out.append("below the gate after review: "
                   + (", ".join(row["gate_failures"]) or "none"))
        out += ["", "every label"]
        out += label_table(before, after)
        out += ["", "what the review's changes did, against the hand labels"]
        for verdict in ("fixed", "broke", "still_wrong", "disputed",
                        "unscored"):
            out.append(f"  {verdict:<14}{row['verdicts'].get(verdict, 0):>5}")
        out += ["",
                f"{'report':<8}{'pages':>7}{'scored':>8}{'before':>9}"
                f"{'after':>8}{'chg':>5}{'tools':>8}{'calls':>7}"
                f"{'in':>10}{'out':>9}{'s':>7}"]
        for r in row["per_report"]:
            before_txt = ("   --  " if r["before"] is None
                          else f"{r['before']:>9.3f}")
            after_txt = ("   --  " if r["after"] is None
                         else f"{r['after']:>8.3f}")
            out.append(f"{r['id']:<8}{r['pages']:>7}{r['scored']:>8}"
                       f"{before_txt}{after_txt}{r['changes']:>5}"
                       f"{r['tool_calls']:>5}/{r['budget']:<2}"
                       f"{r['calls']:>7}{r['tokens_in']:>10,}"
                       f"{r['output_tokens']:>9,}{r['seconds']:>7.0f}")
        out.append("```")

        clean = row.get("never_seen")
        if clean:
            cb, ca = clean["_scores"]
            out += ["",
                    (f"### the same set minus {', '.join(clean['excluded'])}, "
                     f"which the cost checkpoint used and so are no longer "
                     f"blind" if clean["excluded"] else
                     "### the same set, none of which any checkpoint has "
                     "used"),
                    "",
                    f"{len(clean['reports'])} reports nobody has opened, "
                    f"{ca.n} pages. **This is the honest blind figure.**",
                    "", "```",
                    f"{'':<24}{'before':>10}{'after':>10}",
                    f"{'strict accuracy':<24}{cb.accuracy:>10.3f}"
                    f"{ca.accuracy:>10.3f}",
                    f"{'accepting alternates':<24}"
                    f"{cb.lenient_accuracy:>10.3f}"
                    f"{ca.lenient_accuracy:>10.3f}", ""]
            out += label_table(cb, ca, KEY_CONTENT)
            out.append("below the gate after review: "
                       + (", ".join(clean["gate_failures"]) or "none"))
            out.append("```")

    out += ["", "## What triage said", "",
            "Enumerated fields only. The rationale and the anomalies stay in "
            "`triage/<ID>.json` on the cluster.", "",
            "| report | document_type | workflow | bound | toc | scan |",
            "|---|---|---|---|---|---|"]
    for r in results["triage"]:
        out.append(f"| {r['id']} | {r['document_type']} | {r['workflow']} | "
                   f"{r['bound_together']} | {r['toc_agreement']} | "
                   f"{r['scan_fraction']:.2f} |")

    totals = results["totals"]
    out += ["", "## Cost", "", "```",
            f"{totals['calls']} model calls, "
            f"{tokens_in_words(totals)}, "
            f"{totals['output_tokens']:,} output, "
            f"{totals['seconds']:.0f} s"
            f"{_money(totals)}",
            _failure_line(totals),
            "```", "",
            _price_note(totals, results)]
    return out
