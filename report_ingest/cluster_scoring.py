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

WHERE THE OUTPUT GOES. ``out_dir`` must be ``/tmp`` or a Volume. Writes to
``/Workspace`` are non-durable and permission-blocked on this cluster
(``docs/DATABRICKS_INSTALL.md``), and a run that appears to succeed and
leaves nothing behind is the worst outcome, so a ``/Workspace`` path is
refused up front rather than discovered at the end.

AND IT IS MIRRORED SOMEWHERE THAT SURVIVES. ``/tmp`` does not survive a
cluster restart: the first full run's 38 label reviews, about $17 of model
calls, were wiped by one. Pass ``sharepoint=fh_sp_client`` (or
``durable_dir="/Volumes/..."``) and every run file is copied to
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
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.corpus import Corpus
from report_ingest.mirror import DEFAULT_FOLDER, Mirror
from report_ingest.scoring import (
    CHECKPOINT, GATE, KEY_CONTENT, OOS_BLIND, OOS_OPEN, Scores,
    columns_label_table, disputed_drop, gate_failures, label_table,
    verdict_for,
)
from report_ingest.vote import (
    POLICIES, STRUCTURAL_RULES_WIN, PageVote, agreement, build_votes,
    policy_labels, required_review_accuracy, trust_table,
    vision_confidences,
)

__all__ = ["score_on_cluster", "SET_NAMES", "STAGE_NAMES"]


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
#: hand-truthed laboratory sheet, scored the same two ways. ``vision_labels``
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
STAGE_NAMES: Tuple[str, ...] = ("labels", "logs", "lab", "narrative",
                                "vision_labels", "vote", "ingest")

#: The reports the log rules were allowed to be tuned on, when no ``OPEN.txt``
#: sits beside the truth files. Everything else is scored as blind.
DEFAULT_OPEN_LOGS: Tuple[str, ...] = ("R36", "R37", "R06", "R07", "R15",
                                      "R28")
#: The same, for the laboratory sheets.
DEFAULT_OPEN_LAB: Tuple[str, ...] = ("R36", "R28", "R17", "R06")

#: Paths a run must not write to. A Workspace write looks like it worked and
#: then is not there.
_REFUSED_PREFIXES = ("/Workspace", "/dbfs/Workspace", "dbfs:/Workspace")


def _check_out_dir(out_dir: Path) -> None:
    text = str(out_dir).replace("\\", "/")
    for bad in _REFUSED_PREFIXES:
        if text.startswith(bad):
            raise ValueError(
                f"out_dir {out_dir} is under {bad}, where writes are "
                f"non-durable and permission-blocked on this cluster. Use "
                f"/tmp/... or a Volume path (/Volumes/...).")


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
                                 "narrative": "narrative"}


def _truth_dirs(truth_dir: Any, lab_truth_dir: Any, narrative_truth_dir: Any
                ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """``(logs, lab, narrative)`` truth folders, from a ROOT or from three.

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
    narrative = (Path(narrative_truth_dir)
                 if narrative_truth_dir is not None else under("narrative"))
    return logs, lab, narrative


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


def _run_one(rid: str, corpus: Corpus, prompter: Any, model: str,
             triage_model: str, out_dir: Path) -> dict:
    """Rules, triage and review on one report. Writes its own run file."""
    from planlens.document.roles import document_outline, page_roles

    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.label_review import review_labels
    from report_ingest.triage import document_facts, triage

    started = time.time()
    meter = CostMeter()
    doc = corpus.open_report(rid, di="auto", warn=False)
    try:
        roles = page_roles(doc)
        outline = document_outline(doc)
        facts = document_facts(doc, roles)
        triage_engine = PrompterEngine(prompter, triage_model, meter=meter)
        profile = triage(doc, roles, outline, engine=triage_engine,
                         facts=facts)
        review_engine = PrompterEngine(prompter, model, meter=meter)
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

    blob = {
        "id": rid,
        "run_date": date.today().isoformat(),
        "n_pages": n_pages,
        "engine": "prompter",
        "triage_model": triage_model,
        "review_model": model,
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
        json.dumps(blob, indent=2), encoding="utf-8")
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
                     max_reports: Optional[int] = None,
                     redo: bool = False,
                     corpus_dir: Any = None,
                     stages: Sequence[str] = ("labels",),
                     truth_dir: Any = None,
                     lab_truth_dir: Any = None,
                     narrative_truth_dir: Any = None,
                     log_budget: int = 6,
                     lab_budget: int = 4,
                     narrative_budget: int = 8,
                     open_reports: Optional[Sequence[str]] = None,
                     open_lab_reports: Optional[Sequence[str]] = None,
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
                     narrative_front_pages: int = 50
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
        from their own text layer alone.
    out_dir
        Where everything is written. ``/tmp/...`` or a Volume, never
        ``/Workspace``.
    prompter
        The live ``fh_prompter``. Required: there is no other way in.
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
        that already exist. ``"ingest"`` (5.23.0) is the whole pipeline on
        each report in the chosen sets: ``graph.ingest_report`` end to end,
        into ``out_dir/ingest/<ID>/`` -- the record, the summary, the
        library page, the DIGGS file with both gates, ``qa.json`` -- reusing
        the label review and the triage a ``labels`` run left in ``runs/``
        (or in ``review_dir``) rather than paying for them again, resumable
        per report and per work item, and scored against whatever hand
        truth ``truth_dir`` holds for that report, so the stage doubles as
        the whole-pipeline score.
    truth_dir
        ONE truth root for every scoring stage: a folder holding ``logs/``,
        ``lab/`` and ``narrative/``, named after the stages that read them.
        The hand truth is private, so it is uploaded by hand before a run and
        does not live in the wheel; one folder to upload is one thing to get
        right rather than three. A ``truth_dir`` holding the log truth files
        directly, with no ``logs/`` in it, still works as it always did.
    lab_truth_dir, narrative_truth_dir
        The older per-stage form, and it still overrides the root. Pass them
        when the three sets of truth are not in one place.

        In every case an ``OPEN.txt`` beside the truth files names the
        reports whose pages the prompts were allowed to be tuned against;
        everything else is scored as blind.
    log_budget, lab_budget
        Model calls a reader may spend per log and per sheet. Each reader's
        own ceiling -- six and four -- holds whatever these say.
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
    """
    if prompter is None:
        raise ValueError(
            "pass the live fh_prompter; this module reads no credential of "
            "its own")
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
    logs_truth, lab_truth, narrative_truth = _truth_dirs(
        truth_dir, lab_truth_dir, narrative_truth_dir)
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
    if "narrative" in stages and narrative_truth is None:
        raise ValueError(
            "the 'narrative' stage scores the reader against the hand "
            "answers, so it needs them: either narrative_truth_dir -- the "
            "folder of <ID>.json files, each holding the owner's two schemas "
            "answered by hand with null for 'not stated' -- or a "
            "'narrative/' folder inside truth_dir. They are private and do "
            "not ship in the wheel.")
    out = Path(out_dir)
    _check_out_dir(out)
    for sub in ("runs", "triage"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if "logs" in stages:
        (out / "logs").mkdir(parents=True, exist_ok=True)
    if "lab" in stages:
        (out / "lab").mkdir(parents=True, exist_ok=True)
    if "narrative" in stages:
        (out / "narrative").mkdir(parents=True, exist_ok=True)
    if "vision_labels" in stages:
        (out / "vision").mkdir(parents=True, exist_ok=True)
    if "vote" in stages:
        (out / "vote").mkdir(parents=True, exist_ok=True)
    if "ingest" in stages:
        (out / "ingest").mkdir(parents=True, exist_ok=True)

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
                 or ("ingest" in stages))
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
              or (out / "ingest" / rid / "run.json").is_file()]
    absent = [rid for rid in asked if rid not in wanted]
    if max_reports:
        wanted = wanted[:int(max_reports)]
    label_ids = wanted if "labels" in stages else []

    print(f"report ingest on the cluster: stage(s) {', '.join(stages)}; "
          f"model {model}, triage {triage_model}; out_dir {out}")
    if "labels" in stages:
        print(f"  labels: {len(label_ids)} report(s)")
    if absent:
        print(f"  not in {corpus.reports_dir} and skipped: "
              f"{', '.join(absent)}")
    spent = 0.0
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
        try:
            blob = _run_one(rid, corpus, prompter, model, triage_model, out)
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
        review = blob["review"]
        print(f"  [{n}/{len(wanted)}] {rid}: {blob['n_pages']} pp, "
              f"{len(review['changes'])} change(s), "
              f"{review['tool_calls']}/{review['budget']} tool calls, "
              f"{blob['cost']['calls']} model calls, "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    results = _score(done, corpus, oos, mapped, sets, model, triage_model,
                     failures, absent)
    results["stages"] = list(stages)
    lines = _render(results) if "labels" in stages else _header(results)

    if "logs" in stages:
        logs = _run_logs(corpus, prompter, model, out, logs_truth,
                         budget=log_budget, redo=redo,
                         max_logs=max_reports, open_reports=open_reports,
                         sync=sync, templates_path=templates_path)
        results["logs"] = logs
        lines += _render_logs(logs)

    if "lab" in stages:
        lab = _run_lab(corpus, prompter, model, out, lab_truth,
                       budget=lab_budget, redo=redo, max_sheets=max_reports,
                       open_reports=open_lab_reports, sync=sync)
        results["lab"] = lab
        lines += _render_lab(lab)

    if "narrative" in stages:
        narrative = _run_narrative(
            corpus, prompter, model, out, narrative_truth,
            budget=narrative_budget, redo=redo, max_reports=max_reports,
            open_reports=open_narrative_reports, sync=sync,
            front_pages=narrative_front_pages)
        results["narrative"] = narrative
        lines += _render_narrative(narrative)

    if "vision_labels" in stages:
        vision = _run_vision(corpus, prompter, vision_model, out, wanted,
                             mode=vision_mode, dpi=vision_dpi,
                             outline_context=vision_outline_context,
                             detail=vision_detail, window=vision_window,
                             images_per_call=vision_images_per_call,
                             fallback=vision_fallback, redo=redo,
                             sync=sync)
        scored_vision = _score_vision(vision["runs"], corpus, oos, mapped,
                                      sets, out, vision_model,
                                      vision["failures"], vision["settings"])
        results["vision_labels"] = scored_vision
        lines += _render_vision(scored_vision)

    if "vote" in stages:
        vote = _run_vote(corpus, oos, mapped, sets, out,
                         vision_dirs=vision_dirs, review_dir=review_dir,
                         sync=sync)
        results["vote"] = vote
        lines += _render_vote(vote)

    if "ingest" in stages:
        ingest = _run_ingest(corpus, prompter, model, out, wanted,
                             truth=(logs_truth, lab_truth, narrative_truth),
                             log_budget=log_budget, lab_budget=lab_budget,
                             narrative_budget=narrative_budget,
                             redo=redo, review_dir=review_dir,
                             max_total_dollars=max_total_dollars, sync=sync)
        results["ingest"] = ingest
        lines += _render_ingest(ingest)

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
    if "narrative" in stages:
        print(f"per-report narrative runs in {out / 'narrative'}")
    if "vision_labels" in stages:
        print(f"per-report vision runs in {out / 'vision'}")
    if "vote" in stages:
        print(f"per-report disagreement lists in {out / 'vote'}")
    if "ingest" in stages:
        print(f"per-report records and exports in {out / 'ingest'}")
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
        "totals": {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0},
    }
    for rid in sorted(done):
        blob = done[rid]
        cost = blob["cost"]
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            out["totals"][key] += cost.get(key, 0)
        out["totals"]["dollars"] += cost.get("dollars", 0.0)
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
                "input_tokens": blob["cost"]["input_tokens"],
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
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()
                if not str(k).startswith("_")}
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
              templates_path: Any = None) -> Dict[str, Any]:
    """Grid and reader over every hand-truthed log, before and after.

    Restartable the same way the labels stage is: each log writes
    ``logs/<log id>.json`` as it finishes and a later call skips it.

    A private log-TEMPLATE fingerprint file travelling with the truth folder
    (``<truth_dir>/../templates.json``, or ``templates_path``) is put in
    force for the whole stage, so a log printed on a form the file describes
    is recognised and its columns are named by the form. With no such file
    nothing changes.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
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

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        log_id = str(truth.get("id") or f"log{n}")
        report = log_id.split("_")[0]
        run_file = out / "logs" / f"{log_id}.json"
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
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            before, after = score_one_log(truth, doc, engine, budget=budget,
                                          report_id=report)
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
            "model": model,
            "served_by": engine.served_by,
            "pages": [int(p) for p in truth.get("pages") or []],
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        _sync(sync)
        done[log_id] = blob
        gain = (after.total.found - before.total.found)
        print(f"  [{n}/{len(truths)}] {log_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    return _score_logs(done, failures, model, openset)


# ---------------------------------------------------------------------------
# the lab stage
# ---------------------------------------------------------------------------

def _run_lab(corpus: Corpus, prompter: Any, model: str, out: Path,
             truth_dir: Path, *, budget: int, redo: bool,
             max_sheets: Optional[int],
             open_reports: Optional[Sequence[str]],
             sync: Optional[Any] = None) -> Dict[str, Any]:
    """Tables and reader over every hand-truthed laboratory sheet.

    Restartable the same way the other stages are: each sheet writes
    ``lab/<sheet id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
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

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        sheet_id = str(truth.get("id") or f"sheet{n}")
        report = report_of(truth)
        run_file = out / "lab" / f"{sheet_id}.json"
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
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            before, after = score_one_sheet(truth, doc, engine,
                                            budget=budget, report_id=report)
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
            "model": model,
            "served_by": engine.served_by,
            "pages": pages_of(truth),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        _sync(sync)
        done[sheet_id] = blob
        gain = after.total.found - before.total.found
        print(f"  [{n}/{len(truths)}] {sheet_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return _score_lab(done, failures, model, openset)


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
        }
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "?" for r in rows}):
        group = [r for r in rows if (r.get("kind") or "?") == kind]
        by_kind[kind] = {
            "n_sheets": len(group),
            "before": _lab_totals(group, "before"),
            "after": _lab_totals(group, "after"),
        }
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_sheets": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "kinds": by_kind,
        "per_sheet": [{
            "sheet_id": r["sheet_id"], "set": r["set"],
            "kind": r.get("kind") or "?",
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "model_calls": r["after"].get("model_calls", 0),
            "tool_calls": r["after"].get("tool_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "input_tokens": r["cost"].get("input_tokens", 0),
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
    )

    out: List[str] = [
        "", "# WP3 on the cluster: the lab reader through Prompter", "",
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
    ]
    if lab.get("served_by"):
        out.append(f"Served by: {', '.join(lab['served_by'])}.")
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
        out += ["", f"## {name} -- {row['n_sheets']} sheet(s)", "", "```",
                f"{'metric':<10}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not after or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<10}{_rate(before):>14}{_rate(after):>14}")
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out += [f"{'OVERALL':<10}{_rate(before_all):>14}"
                f"{_rate(after_all):>14}", "```"]

    out += ["", "## Per kind", "", "```",
            f"{'kind':<20}{'sheets':>7}{'before':>14}{'after':>14}"]
    for kind, row in lab.get("kinds", {}).items():
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out.append(f"{kind:<20}{row['n_sheets']:>7}"
                   f"{_rate(before_all):>14}{_rate(after_all):>14}")
    out.append("```")

    out += ["", "## Per sheet", "", "```",
            f"{'sheet':<28}{'set':<7}{'before':>12}{'after':>12}{'calls':>7}"
            f"{'zoom':>6}{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in lab["per_sheet"]:
        if r.get("error"):
            out.append(f"{r['sheet_id']:<28}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:50]}")
            continue
        out.append(
            f"{r['sheet_id']:<28}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{r['model_calls']:>7}{r['tool_calls']:>6}{r['unresolved']:>7}"
            f"{r['changes']:>6}{r['input_tokens']:>10,}"
            f"{r['output_tokens']:>8}{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`zoom` is how many times the reader magnified a plot. "
                "`unres` is what it could not settle plus what Python "
                "refused -- an impossible depth, a percentage past 100, a "
                "liquid limit below the plastic limit, a grading running "
                "the wrong way. `look` is every value it took from the "
                "picture and every curve it digitised.", ""]
    cost = lab["cost"]
    n = max(1, lab["n_sheets"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            f"per sheet: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost)]
    return out


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
                   front_pages: int = 50) -> Dict[str, Any]:
    """The narrative reader over every report that has a hand answer.

    The stage reads the truth files that are PRESENT and skips every report
    without one: the hand answers arrive a few reports at a time, and a stage
    that failed on the ones not yet written would be unusable until the last
    one was.

    Restartable the same way the other stages are: each report writes
    ``narrative/<ID>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
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

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, (rid, truth) in enumerate(truths, 1):
        run_file = out / "narrative" / f"{rid}.json"
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
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            score = score_one_report(truth, doc, engine, budget=budget,
                                     report=rid, front_pages=front_pages)
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
            "model": model,
            "served_by": engine.served_by,
            "score": score.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        _sync(sync)
        done[rid] = blob
        recall = score.recall
        print(f"  [{n}/{len(truths)}] {rid}: recall "
              f"{recall.found}/{recall.total}, precision "
              f"{score.precision.found}/{score.precision.total}, "
              f"{score.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return _score_narrative(done, failures, model, openset)


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
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
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
            "input_tokens": r["cost"].get("input_tokens", 0),
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
        "", "# WP4 on the cluster: the narrative reader through Prompter", "",
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
            f"{r['unresolved']:>7}{r['input_tokens']:>10,}"
            f"{r['output_tokens']:>8,}{r['seconds']:>7.0f}")
    out.append("```")

    cost = narrative["cost"]
    n = max(1, narrative["n_reports"])
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost)]
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


def _score_logs(done: Dict[str, dict], failures: Dict[str, str], model: str,
                openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "logs": [r["log_id"] for r in group],
            "n_logs": len(group),
            "before": _totals(group, "before"),
            "after": _totals(group, "after"),
        }
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
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
            "model_calls": r["after"].get("model_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "input_tokens": r["cost"].get("input_tokens", 0),
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
        "", "# WP2b on the cluster: the log reader through Prompter", "",
        f"Run {logs['date']}. Model `{logs['model']}`, "
        f"{logs['n_logs']} hand-truthed log(s).",
        "",
        f"**before** is what `log_grid` alone recovered; **after** is what "
        f"the reader's record holds. The grid runs once and is handed to the "
        f"reader, so the difference between the two columns is the model and "
        f"nothing else. Tolerances: samples, index values and water "
        f"{SAMPLE_TOL_M} m, layer tops {LAYER_TOL_M} m "
        f"(water {WATER_TOL_M} m); depths compared in metres whatever the "
        f"log prints. N values exact.",
    ]
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
        out += ["", f"## {name} -- {row['n_logs']} log(s)", "", "```",
                f"{'metric':<16}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not before or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<16}{_rate(before):>14}{_rate(after):>14}")
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out += [f"{'OVERALL':<16}{_rate(before_all):>14}"
                f"{_rate(after_all):>14}", "```"]

    out += ["", "## Per log", "", "```",
            f"{'log':<14}{'set':<7}{'before':>12}{'after':>12}{'calls':>7}"
            f"{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in logs["per_log"]:
        if r.get("error"):
            out.append(f"{r['log_id']:<14}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:60]}")
            continue
        out.append(
            f"{r['log_id']:<14}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{r['model_calls']:>7}{r['unresolved']:>7}{r['changes']:>6}"
            f"{r['input_tokens']:>10,}{r['output_tokens']:>8,}"
            f"{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`unres` is what the reader could not settle plus what "
                "Python refused (a depth off the page). `look` is how many "
                "values it took from the picture rather than the rows.", ""]
    cost = logs["cost"]
    n = max(1, logs["n_logs"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            f"per log: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost)]
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

    from report_ingest.engine import CostMeter, PrompterEngine
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
        engine = PrompterEngine(prompter, model, meter=meter)
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
            "model": model,
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
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
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
              f"{blob['cost']['input_tokens']:,} in / "
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
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
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
                for key in ("calls", "input_tokens", "output_tokens",
                            "cache_read_tokens"):
                    cost[key] += blob["cost"].get(key, 0)
                cost["dollars"] += blob["cost"].get("dollars", 0.0)
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
                    "input_tokens": blob["cost"].get("input_tokens", 0),
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
                       f"{r['input_tokens']:>10,}"
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
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost),
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


def _reuse_label_run(folder: Path, runs_dir: Path, rid: str
                     ) -> Tuple[bool, bool]:
    """Seed the ingest folder with a saved triage and review, when present.

    A ``labels`` run leaves ``runs/<ID>.json`` holding the triage profile
    and the review's final labels; ``graph.ingest_report`` resumes from
    ``triage.json`` and ``review.json`` in its own folder. Copying the one
    into the other is what stops the ingest paying for ~$0.50 of review a
    second time. Nothing already in the folder is overwritten.
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
    if final and not review_file.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps({
            "final_labels": {str(k): v for k, v in final.items()},
            "reused_from": str(Path(runs_dir) / f"{rid}.json")}, indent=2),
            encoding="utf-8")
        review_reused = True
    return triage_reused, review_reused


def _truth_files_for(rid: str, truth: Tuple[Optional[Path], Optional[Path],
                                            Optional[Path]]
                     ) -> Dict[str, List[Tuple[str, dict]]]:
    """The hand-truth files that concern one report, by stage."""
    logs_dir, lab_dir, narrative_dir = truth
    out: Dict[str, List[Tuple[str, dict]]] = {"logs": [], "lab": [],
                                              "narrative": []}

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
    if narrative_dir is not None and narrative_dir.is_dir():
        path = narrative_dir / f"{rid}.json"
        if path.is_file():
            blob = load(path)
            if blob:
                out["narrative"].append((path.stem, blob))
    return out


def _score_ingest_record(record: Any, rid: str,
                         truth: Tuple[Optional[Path], Optional[Path],
                                      Optional[Path]]) -> Dict[str, Any]:
    """The ingest's record against every hand truth this report has.

    The SAME scorers the reader stages use, on the record the whole
    pipeline produced -- so the stage doubles as the whole-pipeline score.
    A log truth is scored against the investigations read off its pages
    and a lab truth against the tests read off its page, not against the
    whole record, or a sample at the right depth in a different hole would
    count.
    """
    from report_ingest.lab_scoring import pages_of
    from report_ingest.lab_scoring import score_record as score_lab
    from report_ingest.log_scoring import score_record as score_log
    from report_ingest.narrative_scoring import score_narrative

    files = _truth_files_for(rid, truth)
    out: Dict[str, Any] = {"logs": [], "lab": [], "narrative": None}
    for name, blob in files["logs"]:
        pages = {int(p) for p in blob.get("pages") or []}
        invs = [i for i in record.investigations
                if not pages or set(i.pages) & pages]
        score = score_log(blob, invs)
        out["logs"].append({"id": name, "overall": score.total.to_dict(),
                            "scores": {k: v.to_dict()
                                       for k, v in score.scores.items()}})
    for name, blob in files["lab"]:
        pages = set(pages_of(blob))
        tests = [t for t in record.lab_tests
                 if not pages or set(t.pages) & pages]
        score = score_lab(blob, tests)
        out["lab"].append({"id": name, "kind": blob.get("kind") or "",
                           "overall": score.total.to_dict(),
                           "scores": {k: v.to_dict()
                                      for k, v in score.scores.items()}})
    for name, blob in files["narrative"]:
        score = score_narrative(blob, record.general, record.natural_hazards,
                                report=rid)
        out["narrative"] = {"recall": score.recall.to_dict(),
                            "precision": score.precision.to_dict(),
                            "agreement": score.agreement.to_dict()}
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
            elif any("pydiggs is not installed" in v for v in entry.values):
                schema = "not checked here"
            else:
                schema = "INVALID"
        elif entry.where == "diggs.roundtrip":
            roundtrip = "equal" if entry.kind == "note" else "DIFFERS"
        elif entry.where == "diggs" and entry.kind == "skipped":
            written = "no"
    return {"written": written, "schema": schema, "roundtrip": roundtrip}


def _run_ingest(corpus: Corpus, prompter: Any, model: str, out: Path,
                report_ids: Sequence[str], *,
                truth: Tuple[Optional[Path], Optional[Path], Optional[Path]],
                log_budget: int, lab_budget: int, narrative_budget: int,
                redo: bool, review_dir: Any = None,
                max_total_dollars: Optional[float] = None,
                sync: Optional[Any] = None) -> Dict[str, Any]:
    """The whole pipeline on each report, into ``ingest/<ID>/``.

    Restartable twice over: the stage skips a report whose ``run.json`` is
    on disk, and inside a report ``graph.ingest_report`` skips every work
    item whose file is already under ``items/``. A saved label run is
    reused for the triage and the review, so a report that went through
    the ``labels`` stage costs only its readers here.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.graph import Budgets, ingest_report, output_paths

    runs_dir = Path(review_dir) if review_dir is not None else out / "runs"
    if (runs_dir / "runs").is_dir():
        runs_dir = runs_dir / "runs"
    print(f"  ingest: {len(report_ids)} report(s); model {model}; a saved "
          f"label run in {runs_dir} is reused where one exists")
    present = set(corpus.present_ids())
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    spent = 0.0
    for n, rid in enumerate(report_ids, 1):
        folder = out / "ingest" / rid
        run_file = folder / "run.json"
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            if not blob.get("error"):
                done[rid] = blob
                print(f"  [{n}/{len(report_ids)}] {rid}: already done, "
                      "skipping")
                continue
            print(f"  [{n}/{len(report_ids)}] {rid}: previous attempt failed "
                  f"({str(blob.get('error'))[:90]}); retrying")
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(report_ids)}] {rid}: skipped -- "
                  f"{failures[rid]}")
            continue
        if max_total_dollars is not None and spent > max_total_dollars:
            print(f"  stopping before {rid}: spent ${spent:.2f}, past the "
                  f"${max_total_dollars:.2f} ceiling")
            break
        folder.mkdir(parents=True, exist_ok=True)
        triage_reused, review_reused = _reuse_label_run(folder, runs_dir, rid)
        meter = CostMeter()
        engine = PrompterEngine(prompter, model, meter=meter)
        budgets = Budgets(narrative=narrative_budget, log=log_budget,
                          lab=lab_budget)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            record = ingest_report(doc, engine, out_dir=folder,
                                   budgets=budgets, report_id=rid,
                                   resume=not redo, write=True,
                                   db_path=out / "ingest" / "reports.db")
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
            run_file.write_text(json.dumps({
                "id": rid, "run_date": date.today().isoformat(),
                "error": failures[rid], "cost": meter.to_dict()}, indent=2),
                encoding="utf-8")
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
        blob = {
            "id": rid,
            "run_date": date.today().isoformat(),
            "model": model,
            "served_by": engine.served_by,
            "n_pages": n_pages,
            "workflow": record.document.workflow,
            "triage_reused": triage_reused,
            "review_reused": review_reused,
            "counts": record.counts(),
            "investigations_by_kind": by_kind,
            "lab_by_kind": lab_by_kind,
            "narrative": {"answered": answered, "null": asked - answered},
            "qa_by_kind": qa_by_kind,
            "diggs": _diggs_verdicts(record, outputs),
            "outputs": {k: str(v) for k, v in outputs.items()},
            "scores": _score_ingest_record(record, rid, truth),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        _sync(sync)
        done[rid] = blob
        spent += blob["cost"].get("dollars", 0.0)
        counts = blob["counts"]
        diggs = blob["diggs"]
        print(f"  [{n}/{len(report_ids)}] {rid}: {n_pages} pp, "
              f"{blob['workflow']}, {counts['investigations']} "
              f"investigation(s), {counts['samples']} sample(s), "
              f"{counts['spt']} drive(s), {counts['lab_tests']} lab test(s), "
              f"{answered}/{asked} narrative fields, {counts['qa']} QA; "
              f"DIGGS {diggs['written']}/{diggs['schema']}/"
              f"{diggs['roundtrip']}; {blob['cost']['calls']} calls, "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return _score_ingest(done, failures, model)


def _score_ingest(done: Dict[str, dict], failures: Dict[str, str],
                  model: str) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    totals: Dict[str, Any] = {
        "pages": 0, "investigations": 0, "samples": 0, "spt": 0,
        "lab_tests": 0, "qa": 0, "investigations_by_kind": {},
        "lab_by_kind": {}, "qa_by_kind": {},
        "narrative_answered": 0, "narrative_null": 0,
        "diggs": {"written": 0, "valid": 0, "not_checked": 0, "invalid": 0,
                  "equal": 0, "differs": 0},
        "scores": {"logs": {"found": 0, "total": 0, "n": 0},
                   "lab": {"found": 0, "total": 0, "n": 0},
                   "narrative": {"recall_found": 0, "recall_total": 0,
                                 "precision_found": 0, "precision_total": 0,
                                 "n": 0}},
    }
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
        totals["pages"] += int(row.get("n_pages") or 0)
        counts = row.get("counts") or {}
        for key in ("investigations", "samples", "spt", "lab_tests", "qa"):
            totals[key] += int(counts.get(key) or 0)
        for group in ("investigations_by_kind", "lab_by_kind",
                      "qa_by_kind"):
            for kind, n in (row.get(group) or {}).items():
                totals[group][kind] = totals[group].get(kind, 0) + int(n)
        narrative = row.get("narrative") or {}
        totals["narrative_answered"] += int(narrative.get("answered") or 0)
        totals["narrative_null"] += int(narrative.get("null") or 0)
        diggs = row.get("diggs") or {}
        if diggs.get("written") == "yes":
            totals["diggs"]["written"] += 1
        if diggs.get("schema") == "valid":
            totals["diggs"]["valid"] += 1
        elif diggs.get("schema") == "not checked here":
            totals["diggs"]["not_checked"] += 1
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
        "per_report": rows,
        "totals": totals,
        "cost": cost,
    }


def _kinds_cell(counts: Dict[str, int]) -> str:
    return ", ".join(f"{n} {kind}" for kind, n in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "-"


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
        "(`not checked here` when pydiggs is not installed), and whether "
        "reading it back with the app's own parser gives the record's "
        "values. **qa** counts the record's QA entries: `disagreement` is "
        "the two voters inside a reader (the grid or the tables against the "
        "model) splitting on a value, both kept; `partial` is what a reader "
        "could not settle; `out_of_range` is what Python refused.",
    ]
    if ingest.get("served_by"):
        out.append(f"Served by: {', '.join(ingest['served_by'])}.")
    if ingest.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in ingest["failures"].items()]

    out += ["", "## Per report", "", "```",
            f"{'report':<8}{'pages':>6}  {'workflow':<14}  "
            f"{'investigations':<28}{'samp':>6}{'spt':>5}  "
            f"{'lab tests':<32}{'narr':>7}  {'qa dis/unres/ref':<17}"
            f"{'diggs written/schema/read back':<32}{'calls':>6}{'in':>10}"
            f"{'out':>8}{'$':>7}{'s':>6}"]
    for r in ingest["per_report"]:
        counts = r.get("counts") or {}
        qa = r.get("qa_by_kind") or {}
        diggs = r.get("diggs") or {}
        narrative = r.get("narrative") or {}
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
            f"{narr_cell:>7}  {qa_cell:<17}{diggs_cell:<32}"
            f"{r['cost'].get('calls', 0):>6}"
            f"{r['cost'].get('input_tokens', 0):>10,}"
            f"{r['cost'].get('output_tokens', 0):>8,}"
            f"{(f'{money:.2f}' if money else '-'):>7}"
            f"{r.get('seconds', 0.0):>6.0f}")
    out.append("```")
    out += ["", "`narr` is the owner's schema fields answered / left null "
                "(37 asked). `qa dis/unres/ref` is disagreement / partial / "
                "out_of_range entries; every kind is in `results.json`."]

    totals = ingest["totals"]
    diggs = totals["diggs"]
    out += ["", "## Totals", "", "```",
            f"{totals['pages']} pages, {totals['investigations']} "
            f"investigations ({_kinds_cell(totals['investigations_by_kind'])}), "
            f"{totals['samples']} samples, {totals['spt']} driven records, "
            f"{totals['lab_tests']} lab tests "
            f"({_kinds_cell(totals['lab_by_kind'])})",
            f"narrative fields answered {totals['narrative_answered']}, "
            f"null {totals['narrative_null']}",
            f"QA entries {totals['qa']}: "
            f"{_kinds_cell(totals['qa_by_kind'])}",
            f"DIGGS: {diggs['written']} written, {diggs['valid']} valid, "
            f"{diggs['not_checked']} not checked here, {diggs['invalid']} "
            f"invalid; {diggs['equal']} read back equal, {diggs['differs']} "
            f"differ",
            "```"]

    scores = totals["scores"]
    scored_rows = [r for r in ingest["per_report"]
                   if (r.get("scores") or {}).get("logs")
                   or (r.get("scores") or {}).get("lab")
                   or (r.get("scores") or {}).get("narrative")]
    if scored_rows:
        out += ["", "## Scored against the hand truth", "",
                "The SAME scorers as the `logs`, `lab` and `narrative` "
                "stages, on the record the whole pipeline produced: a log "
                "truth against the investigations read off its pages, a "
                "sheet truth against the tests read off its page, the hand "
                "answers against the record's two schemas. This is the "
                "whole-pipeline score.", "", "```",
                f"{'report':<8}{'logs':>18}{'lab sheets':>18}"
                f"{'narrative recall':>18}{'precision':>12}"]
        for r in scored_rows:
            sc = r.get("scores") or {}
            logs = sc.get("logs") or []
            lab = sc.get("lab") or []
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
                f"{r['id']:<8}{agg(logs)}{agg(lab)}"
                + (f"{_rate(nar['recall']):>18}{_rate(nar['precision']):>12}"
                   if nar else f"{'-':>18}{'-':>12}"))
        out += ["",
                f"{'ALL':<8}"
                f"{_rate(scores['logs']):>14} ({scores['logs']['n']})"
                f"{_rate(scores['lab']):>14} ({scores['lab']['n']})"
                f"{_rate({'found': scores['narrative']['recall_found'], 'total': scores['narrative']['recall_total']}):>18}"
                f"{_rate({'found': scores['narrative']['precision_found'], 'total': scores['narrative']['precision_total']}):>12}",
                "```"]
    else:
        out += ["", "No hand truth for these reports under `truth_dir`, so "
                    "the records are not scored; the counts above are what "
                    "the pipeline produced."]

    cost = ingest["cost"]
    n = max(1, ingest["n_reports"])
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s"
            f"{_money(cost)}",
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s"
            f"{_money(cost, n)}",
            "```", "",
            _price_note(cost)]
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


def _price_note(cost: Dict[str, Any]) -> str:
    """The line under a cost block saying what its dollars are, or are not."""
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


def _header(results: Dict[str, Any]) -> List[str]:
    """The heading a run writes when the labels stage did not run."""
    return ["# Report ingest on the cluster", "",
            f"Run {results['date']}. Stage(s): "
            f"{', '.join(results.get('stages') or ())}."]


def _render(results: Dict[str, Any]) -> List[str]:
    """RESULTS.md: IDs, labels, counts and rates. Nothing that names anyone."""
    out: List[str] = [
        "# WP1b on the cluster: triage and label review through Prompter",
        "",
        f"Run {results['date']}. Review model `{results['review_model']}`, "
        f"triage `{results['triage_model']}`.",
    ]
    if results.get("served_by"):
        out.append(f"Served by: {', '.join(results['served_by'])}. A Funhouse "
                   f"tier is an alias and the deployment behind it changes, "
                   f"so this is what actually answered.")
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
                       f"{r['calls']:>7}{r['input_tokens']:>10,}"
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
            f"{totals['input_tokens']:,} input tokens "
            f"(+{totals['cache_read_tokens']:,} the provider cached), "
            f"{totals['output_tokens']:,} output, "
            f"{totals['seconds']:.0f} s"
            f"{_money(totals)}",
            "```", "",
            _price_note(totals)]
    return out
