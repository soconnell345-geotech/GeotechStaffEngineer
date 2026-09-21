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

THE MIDDLEWARE ON THE SPEC, SAID PLAINLY. deepagents uses a compiled
sub-agent's runnable AS PROVIDED -- for a spec with a ``runnable`` it reads
the name, the description and the runnable and nothing else, so the
``middleware`` list this builder puts on the spec reaches no model and
enforces nothing by itself. It is there because every other sub-agent in this
app carries the same two entries and a reader comparing the specs should not
have to wonder which ones were forgotten. What actually holds in this graph is
Python: the model-call ceilings are :class:`~report_ingest.graph.Budgets`,
applied per reader by the ingest itself, and there is no filesystem tool on
any model in here for a guard to intercept.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

__all__ = [
    "build_report_ingest_subagent", "build_report_ingest_graph",
    "run_ingest", "IngestSummary", "parse_request", "SUBAGENT_NAME",
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
    diggs_ok: Optional[bool] = Field(
        default=None,
        description="whether the DIGGS file passed both of its gates; null "
                    "when no DIGGS file was written")
    paths: Dict[str, str] = Field(
        default_factory=dict,
        description="where the record, the summary, the library page, the "
                    "DIGGS file and the library database were written")
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


def run_ingest(source: str, questions: Sequence[str], *,
               engine: Any, out_dir: str, budgets: Any = None,
               report_id: str = "", db_path: Any = None,
               di_result: Any = None, label_policy: str = "structural",
               review_mode: str = "disagreements",
               vision_engine: Any = None, vision_mode: str = "sheet",
               trust_table: Any = None,
               templates: Any = None,
               ingest_bound: bool = True) -> IngestSummary:
    """Read one report and return the compact answer, never the record.

    The label parameters are the graph's own and are documented on
    :func:`report_ingest.graph.ingest_report`; ``vision_engine`` should be
    an engine on a CHEAP tier, since the vision voter looks at every page.
    ``ingest_bound`` reads a report bound inside this one as its own record
    and is on by default; the summary then names each one and where it went,
    so the primary agent knows there is a second record to ask about.
    """
    from report_ingest.graph import ingest_report, output_paths

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
    verdicts = [entry for entry in record.qa
                if entry.where in ("diggs.schema", "diggs.roundtrip")]
    diggs_ok = (all(entry.kind == "note" for entry in verdicts)
                if verdicts else None)
    from report_ingest.writers import title_of
    return IngestSummary(
        report=title_of(record),
        workflow=record.document.workflow,
        n_pages=record.document.n_pages,
        investigations=counts["investigations"],
        lab_tests=counts["lab_tests"],
        layers=counts["layers"],
        samples=counts["samples"],
        qa_entries=counts["qa"],
        answered=len(record.general.answered()
                     + record.natural_hazards.answered()),
        model_calls=record.document.model_calls,
        label_split_pages=record.document.label_split_pages,
        label_disagreements=sum(1 for entry in record.qa
                                if entry.kind == "label_disagreement"),
        diggs_ok=diggs_ok,
        paths=paths,
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
                              ingest_bound: bool = True) -> Any:
    """A one-node LangGraph that runs the ingest and answers with a summary.

    ``engine_factory`` is called once per run and must return an ingest
    engine -- on the cluster a
    :class:`~report_ingest.engine.PrompterEngine` over the app's live
    Prompter. It is a factory rather than an engine so that each run gets its
    own cost meter and nothing is held across turns.

    ``out_dir_factory`` decides where a report's outputs go, given its source;
    the default is a folder named after the file beside the working directory.
    ``resolve_source`` turns whatever the primary said into a readable path --
    the app passes its attachment resolver.
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

        out_dir = (out_dir_factory(source) if out_dir_factory
                   else _default_out_dir(source))
        try:
            answer = run_ingest(source, questions, engine=engine,
                                out_dir=out_dir, budgets=budgets,
                                db_path=db_path,
                                label_policy=label_policy,
                                review_mode=review_mode,
                                vision_engine=(vision_engine_factory()
                                               if vision_engine_factory
                                               else None),
                                vision_mode=vision_mode,
                                trust_table=trust_table,
                                templates=templates,
                                ingest_bound=ingest_bound)
        except Exception as exc:                 # one report, not the session
            answer = IngestSummary(
                error=f"the ingest failed: {type(exc).__name__}: {exc}")
        text = answer.error or (
            f"{answer.report}: {answer.investigations} exploration(s), "
            f"{answer.lab_tests} laboratory test(s), {answer.qa_entries} "
            f"QA entr(ies). Written to {answer.paths.get('record', out_dir)}."
            + (f" {len(answer.bound_documents)} report(s) bound inside this "
               f"one were read into records of their own."
               if answer.bound_documents else ""))
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


def _default_out_dir(source: str) -> str:
    stem = os.path.splitext(os.path.basename(str(source)))[0] or "report"
    return os.path.join("report_ingest_out", stem)


def build_report_ingest_subagent(engine_factory: Callable[[], Any], *,
                                 out_dir_factory: Optional[
                                     Callable[[str], str]] = None,
                                 budgets: Any = None,
                                 db_path: Any = None,
                                 resolve_source: Optional[
                                     Callable[[str], str]] = None,
                                 max_model_calls: Optional[int] = None,
                                 middleware: Optional[Sequence[Any]] = None,
                                 label_policy: str = "structural",
                                 review_mode: str = "disagreements",
                                 vision_engine_factory: Optional[
                                     Callable[[], Any]] = None,
                                 vision_mode: str = "sheet",
                                 trust_table: Any = None,
                                 templates: Any = None,
                                 ingest_bound: bool = True
                                 ) -> Dict[str, Any]:
    """The deepagents ``CompiledSubAgent`` spec for the ingest.

    See this module's docstring for what the ``middleware`` entry does and
    does not do: deepagents uses the runnable as provided, so the list is a
    declaration rather than an enforcement, and the real ceilings are
    :class:`~report_ingest.graph.Budgets`.
    """
    spec: Dict[str, Any] = {
        "name": SUBAGENT_NAME,
        "description": SUBAGENT_DESCRIPTION,
        "runnable": build_report_ingest_graph(
            engine_factory, out_dir_factory=out_dir_factory,
            budgets=budgets, db_path=db_path,
            resolve_source=resolve_source, label_policy=label_policy,
            review_mode=review_mode,
            vision_engine_factory=vision_engine_factory,
            vision_mode=vision_mode, trust_table=trust_table,
            templates=templates, ingest_bound=ingest_bound),
    }
    carried = list(middleware or [])
    if max_model_calls:
        from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
        carried.append(ModelCallBudgetMiddleware(max_model_calls))
    if carried:
        spec["middleware"] = carried
    return spec
