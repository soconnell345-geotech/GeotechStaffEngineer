"""The ingest as a sub-agent of the app, and the compact answer it returns.

The primary agent must never try to read a 400-page geotechnical report. Its
document tools are built for looking things up in a document a person is
discussing; pointed at a whole report they cost a fortune, fill the
conversation with pages of text and still miss the appendix. So the whole
ingest is one delegation: the primary hands over a file, and what comes back
is a paragraph and five paths.

WHAT THE SUB-AGENT IS. A LangGraph with ONE node that runs
:func:`report_ingest.graph.ingest_report` and returns a
``structured_response``. deepagents 0.6.8 calls that a ``CompiledSubAgent``:
any runnable whose state has a ``messages`` key, whose ``structured_response``
becomes the tool result the primary sees. There is no model in the graph
itself -- the models are inside the ingest, one bounded call at a time -- so
the primary cannot talk it into reading something else, and the run is exactly
as budgetable as the ingest is.

WHAT COMES BACK. Counts, the paths, the first lines of the summary, the number
of QA entries and the workflow. Never the record: it runs to megabytes on a
real report, and the primary's next turn would re-send all of it.

THE CEILINGS ARE IN THE GRAPH, NOT ON THE SPEC (BACKLOG RI-02). deepagents
uses a compiled sub-agent's runnable AS PROVIDED: for a spec with a
``runnable`` it reads the name, the description and the runnable and nothing
else, so middleware put on the spec reaches nothing. Until RI-02 the spec
carried a model-call budget and a filesystem guard all the same, and neither
ran. Now the spec carries no middleware, and what the other helpers get from
theirs is enforced here:

* the model-call budget is :class:`CallCeiling`: every model call of one
  delegation (triage, the page voters, the label review, every reader, every
  report bound inside this one) counts against one ceiling, over the reading
  engine and the vision engine alike. A call past it is refused before it is
  made; the pass or item it would have served is a ``skipped`` QA entry, the
  record is written with what was read, and the answer says so. Delegating
  the same report again resumes: what was read is kept in its folder.
  :class:`~report_ingest.graph.Budgets` still caps each reader;
* the filesystem guard's point, that a helper's files are real files in the
  conversation's folder, is the ``out_dir_factory`` the app passes: the
  record and its exports are written there, never into the working
  directory or only a temp folder. No model in here has a filesystem tool to
  guard;
* the answer comes back as the primary's ``task`` result, which the
  primary's own tool middleware (the activity log, the tool-error guard, the
  large-result key map) already wraps.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

__all__ = [
    "build_report_ingest_subagent", "build_report_ingest_graph",
    "run_ingest", "diggs_verdict", "IngestSummary", "parse_request",
    "CallCeiling", "CallCeilingReached", "DEFAULT_MAX_MODEL_CALLS",
    "run_checklist_on", "CHECKLIST_NAME",
    "SUBAGENT_NAME",
    "SUBAGENT_DESCRIPTION", "SUMMARY_LINES",
]

#: The name the primary delegates to, and the tool's own name.
SUBAGENT_NAME = "report_ingest"

SUBAGENT_DESCRIPTION = (
    "Reads a WHOLE geotechnical report PDF -- narrative, boring logs, "
    "laboratory sheets -- into one organised, cited record, and writes the "
    "record, a summary page, a library page and a DIGGS 2.6 file. Delegate "
    "any whole-report ingestion to it and never try to read a long report "
    "page by page yourself. It returns counts, the file paths and the first "
    "lines of the summary; ask it for the report's own questions by passing "
    "them in. A report with ANOTHER report bound inside it -- an earlier "
    "firm's investigation reproduced as an appendix -- comes back as two "
    "records, and the answer names the second one: the borings in there are "
    "that earlier investigation's, not this report's."
)

#: How many lines of the summary page travel back with the answer.
SUMMARY_LINES = 12
#: And how many characters, whichever is hit first.
SUMMARY_CHARS = 1200

#: The model calls one delegation may spend, every pass and every bound
#: report included. A ceiling against a runaway, not a budget for a normal
#: run: the 13 whole reports of Foundry stage b took 60 to 897 calls each,
#: bound reports included (``module_work/harness_theory/report_ingest.md``
#: section 7), and :class:`~report_ingest.graph.Budgets` caps each reader.
DEFAULT_MAX_MODEL_CALLS = 1000

#: The checklist's result, written beside the record it was run on.
CHECKLIST_NAME = "report.checklist.json"
#: How many failed checks the answer names, and how long each line may be;
#: the file beside the record has them all.
CHECKLIST_LINES = 6
CHECKLIST_CHARS = 240


class CallCeilingReached(RuntimeError):
    """A model call refused because the delegation's ceiling was reached."""


class CallCeiling:
    """One delegation's model calls, counted across every engine it uses.

    :meth:`wrap` returns an engine whose calls count here; the reading
    engine and the vision engine wrapped by one ceiling share it. A call
    past ``limit`` is refused with :class:`CallCeilingReached` before
    anything is sent, which the ingest records as a skipped pass or item.
    """

    def __init__(self, limit: int) -> None:
        if int(limit) < 1:
            raise ValueError("a model-call ceiling must be at least 1; pass "
                             "None for none")
        self.limit = int(limit)
        self.used = 0
        self.refused = 0

    @property
    def reached(self) -> bool:
        return self.refused > 0

    def wrap(self, engine: Any) -> Any:
        return None if engine is None else _CountedEngine(engine, self)

    def take(self) -> None:
        """Count one call, or refuse it."""
        if self.used >= self.limit:
            self.refused += 1
            raise CallCeilingReached(
                f"the ceiling of {self.limit} model calls for one ingest "
                f"was reached")
        self.used += 1


class _CountedEngine:
    """An ingest engine whose calls count against a :class:`CallCeiling`;
    everything else is the engine it wraps."""

    def __init__(self, inner: Any, ceiling: CallCeiling) -> None:
        self._inner = inner
        self._ceiling = ceiling

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_inner", "_ceiling"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def complete(self, messages: Any, *, system: Any = None,
                 tools: Any = None, images: Any = None,
                 output_format: Any = None, max_tokens: Any = None) -> Any:
        self._ceiling.take()
        return self._inner.complete(messages, system=system, tools=tools,
                                    images=images,
                                    output_format=output_format,
                                    max_tokens=max_tokens)


class IngestSummary(BaseModel):
    """The compact tool result the primary agent sees."""

    report: str = Field(
        default="", description="the report's title, as the record has it")
    workflow: str = Field(
        default="",
        description="the workflow triage chose: standard, appendix_only, "
                    "partial, multi_document, scanned or needs_person")
    n_pages: int = Field(default=0)
    investigations: int = Field(default=0)
    lab_tests: int = Field(default=0)
    calculations: int = Field(
        default=0,
        description="calculation printouts read into the record: what the "
                    "report worked out, what it assumed and what came out")
    layers: int = Field(default=0)
    samples: int = Field(default=0)
    qa_entries: int = Field(
        default=0,
        description="how many things a reviewer is told about: skipped, "
                    "partial, conflicting or unreadable")
    label_split_pages: int = Field(
        default=0,
        description="pages the automatic page labellers did not agree "
                    "on; these are the ones the label review was shown")
    label_disagreements: int = Field(
        default=0,
        description="of those, how many are still unsettled in the "
                    "record, each a label_disagreement QA entry")
    answered: int = Field(
        default=0,
        description="how many of the standing questions this report answered")
    model_calls: int = Field(default=0)
    model_call_ceiling: Optional[int] = Field(
        default=None,
        description="the most model calls this delegation could spend; null "
                    "when no ceiling was set")
    ceiling_reached: bool = Field(
        default=False,
        description="whether that ceiling stopped the run: what it stopped is "
                    "a skipped QA entry in the record, and asking for the "
                    "same report again resumes from what was read")
    diggs_ok: Optional[bool] = Field(
        default=None,
        description="whether the DIGGS file passed both of its gates: "
                    "false when either failed; null when no DIGGS file was "
                    "written, or when a gate could not run here (no schema "
                    "check installed), which is neither a pass nor a "
                    "failure -- the record's QA says which")
    paths: Dict[str, str] = Field(
        default_factory=dict,
        description="where the record, the summary, the library page, the "
                    "DIGGS file and the library database were written")
    library_root: str = Field(
        default="",
        description="the folder this report's library database sits in, so "
                    "the report just read is queryable with the report "
                    "library in the same session; empty when no database "
                    "was written")
    bound_documents: List[Dict[str, str]] = Field(
        default_factory=list,
        description="reports bound INSIDE this one, each read into its own "
                    "record: its title, the pages of this file it occupied, "
                    "what it holds and where its record was written. "
                    "Nothing of theirs is in the counts above")
    summary: str = Field(
        default="", description="the first lines of the summary page")
    answers: List[Dict[str, str]] = Field(
        default_factory=list,
        description="the caller's own questions and what the narrative said")
    checklist: Optional[Dict[str, Any]] = Field(
        default=None,
        description="the report review checklist's code checks, run on this "
                    "record when the checklist is on: the checklist's status "
                    "(a DRAFT), how many checks passed, failed, were unsure "
                    "or could not run, the failed ones, and how many items "
                    "need your judgement (report_checklist on the record "
                    "lists them); every row is in paths.checklist. Null when "
                    "the checklist is off")
    error: str = Field(default="")


def _head(path: str, lines: int = SUMMARY_LINES) -> str:
    """The first meaningful lines of a written page."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read(8000)
    except OSError:
        return ""
    kept: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|") or line.strip() == "---":
            continue
        if line.strip():
            kept.append(line.strip())
        if len(kept) >= lines or sum(len(k) for k in kept) > SUMMARY_CHARS:
            break
    return "\n".join(kept)[:SUMMARY_CHARS]


def run_checklist_on(record: Any, out_dir: Any) -> Dict[str, Any]:
    """The report review checklist's code checks on a record just written.

    Runs :func:`report_ingest.checklist.run_checklist` (no model call),
    writes its whole result to :data:`CHECKLIST_NAME` beside the record, and
    returns what the answer carries: the checklist's status, the code
    checks counted by result, the failed ones (at most
    :data:`CHECKLIST_LINES`), how many items are left to judge, and the
    file's ``path``. A checklist that cannot run says so in ``error``; the
    ingest it follows has still written its record.
    """
    from report_ingest.checklist import run_checklist

    try:
        result = run_checklist(record)
        path = os.path.join(str(out_dir), CHECKLIST_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - the record stands without it
        return {"error": f"the checklist did not run: "
                         f"{type(exc).__name__}: {exc}"}
    rows = result.get("code_checks") or []
    counts = {status: sum(1 for row in rows if row.get("status") == status)
              for status in ("pass", "fail", "unsure", "not_run")}
    failed = [f"{row.get('id')}: {row.get('detail', '')}"[:CHECKLIST_CHARS]
              for row in rows if row.get("status") == "fail"]
    return {"status": result.get("status"), "counts": counts,
            "failed": failed[:CHECKLIST_LINES],
            "to_judge": len(result.get("for_you_to_judge") or []),
            "path": path}


def diggs_verdict(qa: Sequence[Any]) -> Optional[bool]:
    """The DIGGS file's two gates as one answer, off the record's QA.

    True only when both gates RAN and passed; False when either failed;
    None when no file was written, or when a gate could not run here (a
    ``not_run`` entry: no schema check installed) and none failed -- a check
    that did not run is neither a pass nor a failure (RI-20).
    """
    verdicts = [entry for entry in qa
                if entry.where in ("diggs.schema", "diggs.roundtrip")]
    if not verdicts:
        return None
    if any(entry.kind not in ("note", "not_run") for entry in verdicts):
        return False
    if any(entry.kind == "not_run" for entry in verdicts):
        return None
    return True


def run_ingest(source: str, questions: Sequence[str], *,
               engine: Any, out_dir: str, budgets: Any = None,
               report_id: str = "", db_path: Any = None,
               di_result: Any = None, label_policy: str = "structural",
               review_mode: str = "disagreements",
               vision_engine: Any = None, vision_mode: str = "sheet",
               trust_table: Any = None,
               templates: Any = None,
               ingest_bound: bool = True,
               max_model_calls: Optional[int] = None,
               checklist: bool = False) -> IngestSummary:
    """Read one report and return the compact answer, never the record.

    The label parameters are the graph's own and are documented on
    :func:`report_ingest.graph.ingest_report`; ``vision_engine`` should be
    an engine on a CHEAP tier, since the vision voter looks at every page.
    ``ingest_bound`` reads a report bound inside this one as its own record
    and is on by default; the summary then names each one and where it went,
    so the primary agent knows there is a second record to ask about.

    ``max_model_calls`` is the run's :class:`CallCeiling` over both engines
    (``None``: none).

    ``checklist`` runs the report review checklist's code checks on the
    record just written (:func:`run_checklist_on`; no model call) and puts
    the counts and the failures in the answer (BACKLOG RI-35).
    """
    from report_ingest.graph import ingest_report, output_paths

    ceiling = CallCeiling(max_model_calls) if max_model_calls else None
    if ceiling is not None:
        engine = ceiling.wrap(engine)
        vision_engine = ceiling.wrap(vision_engine)
    record = ingest_report(source, engine, out_dir=out_dir, budgets=budgets,
                           report_id=report_id, questions=list(questions),
                           db_path=db_path, di_result=di_result,
                           label_policy=label_policy,
                           review_mode=review_mode,
                           vision_engine=vision_engine,
                           vision_mode=vision_mode,
                           trust_table=trust_table, templates=templates,
                           ingest_bound=ingest_bound)
    paths = output_paths(out_dir)
    counts = record.counts()
    diggs_ok = diggs_verdict(record.qa)
    checked = None
    if checklist:
        checked = run_checklist_on(record, out_dir)
        if checked.get("path"):
            paths = dict(paths, checklist=checked.pop("path"))
    from report_ingest.writers import title_of
    return IngestSummary(
        report=title_of(record),
        workflow=record.document.workflow,
        n_pages=record.document.n_pages,
        investigations=counts["investigations"],
        lab_tests=counts["lab_tests"],
        calculations=counts["calculations"],
        layers=counts["layers"],
        samples=counts["samples"],
        qa_entries=counts["qa"],
        answered=len(record.general.answered()
                     + record.natural_hazards.answered()),
        model_calls=record.document.model_calls,
        model_call_ceiling=ceiling.limit if ceiling is not None else None,
        ceiling_reached=bool(ceiling is not None and ceiling.reached),
        label_split_pages=record.document.label_split_pages,
        label_disagreements=sum(1 for entry in record.qa
                                if entry.kind == "label_disagreement"),
        diggs_ok=diggs_ok,
        paths=paths,
        # The library the row went into is the folder the database sits in:
        # one report's own folder for a single ingest, the whole run's folder
        # where the caller passed a shared db_path. Either way the report is
        # askable with report_library without waiting for a restart.
        library_root=(os.path.dirname(str(db_path)) if db_path
                      else os.path.dirname(paths.get("db", "")) or ""),
        bound_documents=[
            {"title": row.title or row.report_id or row.bound_id,
             "pages": row.pages,
             "holds": f"{row.counts.get('investigations', 0)} exploration(s), "
                      f"{row.counts.get('lab_tests', 0)} laboratory test(s)"
                      if row.read else "not read; see the QA section",
             "record": os.path.join(str(out_dir), row.record_path)
                       if row.record_path else ""}
            for row in record.bound_documents],
        summary=_head(paths.get("summary", "")),
        checklist=checked,
        answers=[{"question": str(row.get("question", "")),
                  "answer": str(row.get("answer", ""))}
                 for row in record.narrative.extra_answers])


# ---------------------------------------------------------------------------
# reading the request out of a delegation
# ---------------------------------------------------------------------------

#: A path or an attachment key that names a document.
_RE_SOURCE = re.compile(
    r"""["'`]?((?:[A-Za-z]:[\\/]|/|\./|\.\./)?[^\s"'`]+\.(?:pdf|PDF))["'`]?""")


@dataclass
class Request:
    """What a delegation asked for."""

    source: str = ""
    questions: List[str] = None        # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.questions is None:
            self.questions = []


def parse_request(text: str) -> Request:
    """The file and the questions out of a delegation written in prose.

    deepagents hands a sub-agent a description, not arguments, so the file has
    to be found in the sentence. A path or an attachment key ending in .pdf is
    the file; every sentence with a question mark is a question. Nothing is
    guessed: with no .pdf in the text the source comes back empty and the
    caller says what it needed.
    """
    body = str(text or "")
    match = _RE_SOURCE.search(body)
    source = match.group(1) if match else ""
    questions = [" ".join(part.split()) + "?"
                 for part in re.split(r"\?", body)[:-1]
                 if len(part.split()) >= 3]
    # A question is a sentence, not the whole preamble before the first "?".
    cleaned: List[str] = []
    for question in questions:
        tail = re.split(r"(?<=[.\n])\s+", question)[-1].strip()
        if tail and source not in tail and len(tail.split()) >= 3:
            cleaned.append(tail)
    return Request(source=source, questions=cleaned)


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

def build_report_ingest_graph(engine_factory: Callable[[], Any], *,
                              out_dir_factory: Optional[Callable[[str], str]]
                              = None,
                              budgets: Any = None,
                              db_path: Any = None,
                              resolve_source: Optional[Callable[[str], str]]
                              = None,
                              label_policy: str = "structural",
                              review_mode: str = "disagreements",
                              vision_engine_factory: Optional[
                                  Callable[[], Any]] = None,
                              vision_mode: str = "sheet",
                              trust_table: Any = None,
                              templates: Any = None,
                              ingest_bound: bool = True,
                              max_model_calls: Optional[int] =
                              DEFAULT_MAX_MODEL_CALLS,
                              checklist: bool = False) -> Any:
    """A one-node LangGraph that runs the ingest and answers with a summary.

    ``engine_factory`` is called once per run and must return an ingest
    engine -- on the cluster a
    :class:`~report_ingest.engine.PrompterEngine` over the app's live
    Prompter. It is a factory rather than an engine so that each run gets its
    own cost meter and nothing is held across turns.

    ``out_dir_factory`` decides where a report's outputs go, given the file
    as it was named (a path or an attachment key); the default is a folder
    named after the file under ``report_ingest_out`` in the working
    directory, and the app passes the conversation's own folder.
    ``resolve_source`` turns whatever the primary said into a readable path
    or the document's bytes -- the app passes its attachment resolver.

    ``max_model_calls`` is each delegation's :class:`CallCeiling`, enforced
    here (``None``: none). ``checklist`` runs the report review checklist's
    code checks on each record written (:func:`run_checklist_on`).
    """
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from typing_extensions import Annotated, TypedDict

    # The functional TypedDict spelling on purpose: this module carries
    # ``from __future__ import annotations``, so a class body's annotations
    # would be strings that LangGraph resolves against the MODULE's globals,
    # where these names (imported inside this function to keep LangGraph off
    # the import path of the rest of the package) are not. The functional form
    # stores the objects themselves and there is nothing to resolve.
    State = TypedDict("State", {
        "messages": Annotated[list, add_messages],
        "source": str,
        "questions": list,
        "structured_response": Any,
    }, total=False)

    def node(state: State) -> Dict[str, Any]:
        from langchain_core.messages import AIMessage

        source = str(state.get("source") or "")
        questions = list(state.get("questions") or [])
        if not source:
            asked = parse_request(_last_text(state.get("messages") or []))
            source = asked.source
            questions = questions or asked.questions
        if not source:
            answer = IngestSummary(
                error="no PDF was named. Give the report's file path or the "
                      "attachment key of the uploaded PDF.")
            return {"messages": [AIMessage(content=answer.error)],
                    "structured_response": answer}

        # The folder is named after the file as it was asked for: an
        # attachment key resolves to the document's bytes, not a name.
        named = source
        if resolve_source is not None:
            try:
                source = resolve_source(source)
            except Exception as exc:             # a key that resolves to nothing
                answer = IngestSummary(
                    error=f"{source!r} could not be opened: "
                          f"{type(exc).__name__}: {exc}")
                return {"messages": [AIMessage(content=answer.error)],
                        "structured_response": answer}

        engine = engine_factory()
        if engine is None:
            answer = IngestSummary(
                error="no ingest engine is configured in this deployment, so "
                      "no report can be read here.")
            return {"messages": [AIMessage(content=answer.error)],
                    "structured_response": answer}

        out_dir = (out_dir_factory(named) if out_dir_factory
                   else _default_out_dir(named))
        try:
            answer = run_ingest(source, questions, engine=engine,
                                out_dir=out_dir, budgets=budgets,
                                report_id=_stem(named),
                                db_path=db_path,
                                label_policy=label_policy,
                                review_mode=review_mode,
                                vision_engine=(vision_engine_factory()
                                               if vision_engine_factory
                                               else None),
                                vision_mode=vision_mode,
                                trust_table=trust_table,
                                templates=templates,
                                ingest_bound=ingest_bound,
                                max_model_calls=max_model_calls,
                                checklist=checklist)
        except Exception as exc:                 # one report, not the session
            answer = IngestSummary(
                error=f"the ingest failed: {type(exc).__name__}: {exc}")
        text = answer.error or (
            f"{answer.report}: {answer.investigations} exploration(s), "
            f"{answer.lab_tests} laboratory test(s), {answer.qa_entries} "
            f"QA entr(ies). Written to {answer.paths.get('record', out_dir)}."
            + (f" {len(answer.bound_documents)} report(s) bound inside this "
               f"one were read into records of their own."
               if answer.bound_documents else "")
            + (f" The ceiling of {answer.model_call_ceiling} model calls for "
               f"one ingest was reached, so some passes or items were not "
               f"read (each is a skipped QA entry); asking for the same "
               f"report again resumes from what was read."
               if answer.ceiling_reached else "")
            + _checklist_text(answer))
        return {"messages": [AIMessage(content=text)],
                "structured_response": answer}

    graph = StateGraph(State)
    graph.add_node("ingest", node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", END)
    return graph.compile()


def _last_text(messages: Sequence[Any]) -> str:
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content
                     if isinstance(block, dict)]
            if any(parts):
                return "\n".join(parts)
    return ""


def _checklist_text(answer: IngestSummary) -> str:
    """The checklist's lines in the sub-agent's answer (empty when off)."""
    checked = answer.checklist
    if not checked:
        return ""
    if checked.get("error"):
        return f" The review checklist: {checked['error']}."
    counts = checked.get("counts") or {}
    text = (f" The review checklist ({checked.get('status') or 'DRAFT'}), "
            f"its code checks on this record: {counts.get('pass', 0)} "
            f"passed, {counts.get('fail', 0)} failed, "
            f"{counts.get('unsure', 0)} unsure, {counts.get('not_run', 0)} "
            f"could not run; every row is in "
            f"{answer.paths.get('checklist', CHECKLIST_NAME)}.")
    if checked.get("failed"):
        text += " Failed: " + "; ".join(checked["failed"]) + "."
    if checked.get("to_judge"):
        text += (f" {checked['to_judge']} item(s) need judgement: "
                 f"report_checklist on this record lists them.")
    return text


def _stem(source: str) -> str:
    """The file's name without its folder or extension."""
    return os.path.splitext(os.path.basename(str(source)))[0] or "report"


def _default_out_dir(source: str) -> str:
    return os.path.join("report_ingest_out", _stem(source))


def build_report_ingest_subagent(engine_factory: Callable[[], Any], *,
                                 out_dir_factory: Optional[
                                     Callable[[str], str]] = None,
                                 budgets: Any = None,
                                 db_path: Any = None,
                                 resolve_source: Optional[
                                     Callable[[str], str]] = None,
                                 max_model_calls: Optional[int] =
                                 DEFAULT_MAX_MODEL_CALLS,
                                 middleware: Optional[Sequence[Any]] = None,
                                 label_policy: str = "structural",
                                 review_mode: str = "disagreements",
                                 vision_engine_factory: Optional[
                                     Callable[[], Any]] = None,
                                 vision_mode: str = "sheet",
                                 trust_table: Any = None,
                                 templates: Any = None,
                                 ingest_bound: bool = True,
                                 checklist: bool = False
                                 ) -> Dict[str, Any]:
    """The deepagents ``CompiledSubAgent`` spec for the ingest.

    The spec carries no middleware: deepagents uses a compiled runnable as
    provided and would drop it (BACKLOG RI-02). ``max_model_calls`` is
    enforced inside the graph instead (:class:`CallCeiling`; ``None`` for
    none), and a non-empty ``middleware`` is refused rather than accepted
    and ignored.
    """
    if middleware:
        raise ValueError(
            "a compiled sub-agent runs no middleware (deepagents uses its "
            "runnable as provided); the ingest's ceiling is max_model_calls")
    return {
        "name": SUBAGENT_NAME,
        "description": SUBAGENT_DESCRIPTION,
        "runnable": build_report_ingest_graph(
            engine_factory, out_dir_factory=out_dir_factory,
            budgets=budgets, db_path=db_path,
            resolve_source=resolve_source, label_policy=label_policy,
            review_mode=review_mode,
            vision_engine_factory=vision_engine_factory,
            vision_mode=vision_mode, trust_table=trust_table,
            templates=templates, ingest_bound=ingest_bound,
            max_model_calls=max_model_calls, checklist=checklist),
    }
