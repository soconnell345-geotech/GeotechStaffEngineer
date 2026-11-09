"""The GeotechStaffEngineer page's own parts, in one file (INF-31, step 7).

What the shell (``webapp/app.py``, ``core.py``, ``diagnostics.py``) used to
do for the geotech page itself, behind the page's hooks
(:class:`webapp.page_profile.AppProfile`): the professional-use notice, the
agent and the narrow specialists of the Agent picker, the report library
and the reference-PDF fetcher wired in when it is built, the judgment-call
check at each turn's end, and the reference-PDF line of the connection
diagnostics. When Document Review gets its own copy of the page (the app
split's step 13), this file stays here and ``webapp/review_page.py`` goes.

The shell names these functions in :data:`webapp.profiles.GEOTECH` and
imports this file only when the geotech page uses one. The old names
(``core.build_reviewer_agent``, ``core._REVIEWER_BUILDERS``,
``diagnostics.reference_pdf_names`` and the rest) still answer, as these.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from webapp.diagnostics import FAIL, PASS, SKIP, WARN, _check


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------

def disclaimer_text() -> Optional[str]:
    """The package's full professional-use disclaimer as text, or None
    where it cannot be had (the shell then shows its short notice).

    ``funhouse_agent.disclaimer`` PRINTS to a stream, so it is captured
    into a buffer."""
    import io
    try:
        from funhouse_agent import disclaimer
        buf = io.StringIO()
        disclaimer(file=buf)
        return buf.getvalue().strip() or None
    except Exception:  # noqa: BLE001 - the shell has a fallback
        return None


# ---------------------------------------------------------------------------
# Building the agent
# ---------------------------------------------------------------------------

def register_reference_fetcher() -> None:
    """Reference PDFs from SharePoint ``primary_references/``, fetched the
    first time a chart needs one (owner decision 2026-09-15). Process-wide and
    harmless to repeat; a no-op without SharePoint."""
    try:
        from webapp import reference_fetch
        reference_fetch.register_if_configured()
    except Exception:                                  # noqa: BLE001
        pass


def add_report_library(kw: dict, temp_dir: str) -> None:
    """Turn the report-library sub-agent on for the geotech page (RI-95).

    The library is ``GEOTECH_REPORT_LIBRARY`` when the deployment sets one,
    else the ``report_ingest`` folder in this conversation's working folder,
    where the ingest writes without it. ``build_deep_agent`` advertises the
    tool only when a record is under that folder, so with no library read
    yet the agent carries neither the tool nor its prompt line. A caller's
    own ``enable_report_library`` / ``library_root`` wins. Never for the
    Document Review page (its builder does not call this).
    """
    if "enable_report_library" in kw or "library_root" in kw:
        return
    try:
        from funhouse_agent.deep.tools import shared_report_library
        from webapp import core
        root = shared_report_library()
        if not root:
            thread_id = os.path.basename(
                os.path.dirname(os.path.abspath(temp_dir)))
            root = os.path.join(core.working_dir_for(thread_id),
                                "report_ingest")
    except Exception:
        return
    kw["enable_report_library"] = True
    kw["library_root"] = root


def document_signal(temp_dir: str):
    """Whether this conversation holds a document its attachments and
    messages do not show, for the geotech page's document gate (BACKLOG
    INF-67): a SharePoint download on the conversation's ledger (it is
    written as the download finishes, so a fetch counts from the next model
    call of the same turn, and it survives a summarised history), or a
    working folder the user chose in place of the conversation's own
    ``files`` folder (it may already hold their documents), or a deployment
    whose read tools open folders beyond the conversation (Databricks'
    ``/Volumes`` and ``/Workspace``, ``GEOTECH_EXTRA_READ_ROOTS``): a report
    named there by its path is a document too."""
    files_dir = os.path.normcase(os.path.abspath(temp_dir))
    conv_dir = os.path.dirname(files_dir)

    def _holds() -> bool:
        from funhouse_agent._fileio import host_output_dir
        from funhouse_agent.vision_tools import _extra_read_roots
        from webapp import core
        if _extra_read_roots():
            return True
        if core.load_downloads(conv_dir):
            return True
        bound = host_output_dir()
        return bool(bound) and \
            os.path.normcase(os.path.abspath(bound)) != files_dir

    return _holds


def build_page_agent(model, *, attachments, save_fn, temp_dir, **kw):
    """The geotech page's agent, after the shell's shared wiring
    (:func:`webapp.core.build_agent`): the report library where there is
    one, then the geotech builder."""
    add_report_library(kw, temp_dir)
    kw.setdefault("document_signal", document_signal(temp_dir))
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(model, attachments=attachments, save_fn=save_fn,
                            **kw)


#: agent_type value -> deep reviewer builder name in funhouse_agent.reviewers.
_REVIEWER_BUILDERS = {
    "seismic": "make_seismic_reviewer_deep",
    "foundations": "make_foundations_reviewer_deep",
    "earth_retention": "make_earth_retention_reviewer_deep",
    "slope_fem": "make_slope_fem_reviewer_deep",
    "pavement": "make_pavement_specialist_deep",
    "structural": "make_structural_specialist_deep",
}


def build_reviewer_agent(kind, model, attachments: dict, temp_dir: str,
                         artifacts: List[str], **build_kwargs):
    """Build a NARROW reviewer deep-agent (A5e) for ``kind``, wired to the SAME
    shared attachments dict + session-dir save_fn as the full agent (the reviewer
    ``make_*_reviewer_deep`` builders forward these to ``build_deep_agent``).

    An unknown ``kind`` (including ``"full"``) falls back to the full agent build.
    A reviewer manages its own scope + review-mode prompt + ``reference_mode``, so
    the behavior reference/analysis-depth build-kwargs are deliberately NOT applied
    to it (``specialist_build_kwargs`` passes the calc helper alone); the
    recursion cap still applies at stream time.

    It gets the full agent's deliverables (BACKLOG GM-40): the report and
    drawing modules are in every specialist's scope
    (``funhouse_agent.reviewers``), and the SharePoint and email tools and
    the feedback tool are spliced in here as ``build_agent`` does.
    """
    from webapp import core
    name = _REVIEWER_BUILDERS.get(kind)
    if name is None:
        from webapp.profiles import GEOTECH
        return core.build_agent(model, attachments, temp_dir, artifacts,
                                profile=GEOTECH, **build_kwargs)
    from funhouse_agent import reviewers as _reviewers
    builder = getattr(_reviewers, name)
    kw = dict(build_kwargs)
    # The page's document gate needs the page's signal: a specialist's
    # replayed history no longer holds an earlier turn's download (INF-67).
    kw.setdefault("document_signal", document_signal(temp_dir))
    core._add_file_tools(kw, temp_dir)
    # Specialists never pass through build_agent, so they had no feedback
    # tool at all before 2026-09-15.
    core._add_feedback_tool(kw, temp_dir)
    register_reference_fetcher()
    return builder(model, attachments=attachments,
                   save_fn=core.make_save_fn(temp_dir, artifacts), **kw)


def turn_checks(agent, callbacks):
    """The turn-end check of the answer's "Judgment calls" list
    (:mod:`funhouse_agent.judgment_check`, BACKLOG GM-174): with
    ``GEOTECH_JUDGMENT_CALLS`` on and a build that asks for the list after a
    consult, an answer whose calls name no source, with no ``references``
    task in the turn, is sent back once. Its result is kept in the turn's
    activity log. ``None`` where it cannot be had."""
    try:
        from funhouse_agent.judgment_check import (RECORD_EVENT,
                                                   TurnJudgmentCheck,
                                                   mode_for)
        logger = next((cb for cb in callbacks or ()
                       if hasattr(cb, "conv_dir")
                       and hasattr(cb, "on_custom_event")), None)
        log = None
        if logger is not None:
            def log(data, _logger=logger):
                _logger.on_custom_event(RECORD_EVENT, data, run_id=None)
        return TurnJudgmentCheck(mode=mode_for(agent), log=log)
    except Exception:  # noqa: BLE001 - the turn runs without it
        return None


# ---------------------------------------------------------------------------
# The sidebar
# ---------------------------------------------------------------------------

def agent_picker(ss, rebuild) -> None:
    """The Agent picker (A5e): the full geotech agent, or a narrow domain
    reviewer. Per conversation, persisted in meta, shown on the conversation
    list line. ``rebuild(model)`` rebuilds the agent as the one picked."""
    import streamlit as st
    from webapp import core
    _atypes = list(core.AGENT_TYPES)
    _cur_at = (ss.behavior or {}).get("agent_type", "full")
    _cur_at = _cur_at if _cur_at in _atypes else "full"
    _picked_at = st.selectbox(
        "Agent", _atypes, index=_atypes.index(_cur_at),
        format_func=lambda k: core.AGENT_TYPES[k], key=f"agent_{ss.thread_id}",
        help="The full geotech agent, or a narrow domain agent scoped to one "
             "discipline's methods + references (the reviewers check work in "
             "review mode; the pavement specialist designs). Applies to this "
             "conversation going forward; kept when you resume it.")
    if _picked_at != _cur_at:
        ss.behavior = {**ss.behavior, "agent_type": _picked_at}
        if core.load_meta(ss.thread_id) is not None:
            core.set_behavior(ss.thread_id, ss.behavior)
        rebuild(ss.model)          # rebuild as the selected variant
        st.rerun()


# ---------------------------------------------------------------------------
# The connection diagnostics: can the agent open the reference PDFs?
# ---------------------------------------------------------------------------

REFERENCE_DOCS_ENV = "GEOTECH_REFERENCES_DOCS"


def _figure_catalogs() -> List[Path]:
    """Every figure catalog the library reads: the installed package's and,
    on refs 1.6.0+, each local reference pack's (REF-64; a pack's catalog
    sits in its folder, named by its pack id)."""
    try:
        from geotech_references import _figures_db
        if hasattr(_figures_db, "_iter_catalogs"):
            from funhouse_agent.reference_packs import _packs_module
            if _packs_module() is not None:
                return list(_figures_db._iter_catalogs())
    except Exception:                                   # noqa: BLE001
        pass
    import geotech_references
    pkg = Path(geotech_references.__file__).parent
    return sorted(pkg.glob("*/figures_catalog.json"))


def _pack_folder_files() -> set:
    """The file names in the local reference packs' folders, where a pack's
    PDFs are looked for first (REF-64); empty with no packs."""
    names: set = set()
    try:
        from funhouse_agent.reference_packs import _packs_module
        mod = _packs_module()
        for _pid, folder in (mod.packs() if mod is not None else []):
            try:
                names |= set(os.listdir(folder))
            except OSError:
                continue
    except Exception:                                   # noqa: BLE001
        pass
    return names


def reference_pdf_names() -> dict:
    """``{pdf filename: [what needs it]}`` for every source PDF the agent can
    open to look at a page — ``read_reference_figure`` (the installed figure
    catalogs) and ``view_worked_example_source`` (the worked-example corpus).
    Both resolve a PDF by its bare filename inside ``GEOTECH_REFERENCES_DOCS``.
    """
    import json
    names: dict = {}
    try:
        for cat in _figure_catalogs():
            try:
                data = json.loads(cat.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            figs = data if isinstance(data, list) else (data.get("figures") or [])
            top = None if isinstance(data, list) else data.get("pdf_path")
            for fig in figs:
                pdf = fig.get("pdf_path") or top
                if pdf:
                    names.setdefault(Path(pdf).name, set()).add(cat.parent.name)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        from funhouse_agent import worked_examples as _we
        for e in _we.load_examples():
            if e.get("source_doc") and e.get("source_pdf_pages"):
                names.setdefault(Path(e["source_doc"]).name,
                                 set()).add("worked examples")
    except Exception:                                   # noqa: BLE001
        pass
    return {n: sorted(v) for n, v in sorted(names.items(),
                                            key=lambda kv: kv[0].lower())}


_UNSET_MESSAGE = (
    f"{REFERENCE_DOCS_ENV} is not set and SharePoint is not connected, so the "
    "agent cannot open any reference PDF: reading a value off a design chart "
    "and viewing a worked example's source page will both fail. Keep the PDFs "
    "in SharePoint GSE_app/primary_references and connect SharePoint "
    "(stage_sharepoint) -- each PDF is then fetched the first time it is "
    "needed -- or point GEOTECH_REFERENCES_DOCS at a local folder before "
    "run_on_databricks and relaunch. webapp/README.md section 3, 'Reference "
    "PDFs'.")

#: The same on Foundry, which has no SharePoint (Funhouse-wiring audit,
#: 2026-11-03; BACKLOG INF-62): the PDFs sit in the workspace repo's
#: ``references/`` folder (OWN-80).
_UNSET_MESSAGE_FOUNDRY = (
    f"{REFERENCE_DOCS_ENV} is not set, so the agent cannot open any "
    "reference PDF: reading a value off a design chart and viewing a worked "
    "example's source page will both fail. Commit the PDFs to a references/ "
    "folder beside the app file, under the file names this check lists once "
    f"the folder is found, and point {REFERENCE_DOCS_ENV} at it in the app "
    "file, then restart the app. docs/FOUNDRY.md section 3d, 'Reference PDFs "
    "on Foundry'.")


def reference_docs_check() -> dict:
    """Can the agent open the reference PDFs it reads charts off?

    Looks in the local folder (``GEOTECH_REFERENCES_DOCS``, else a source
    checkout's docs/), the cache of PDFs already fetched this session and --
    when SharePoint is connected -- the SharePoint folder the app fetches from
    on first use (``primary_references`` under the base folder). Every PDF
    available nowhere is listed by name, so the report doubles as the upload
    list. At most one SharePoint folder listing; no model call.
    """
    name = "reference PDFs (chart read-off)"
    needed = reference_pdf_names()
    if not needed:
        return _check(name, SKIP, "no figure catalogs found "
                                  "(geotech-references not installed?)")

    # -- local folder -------------------------------------------------------
    folder = (os.environ.get(REFERENCE_DOCS_ENV) or "").strip()
    docs, where, local_problem = None, None, None
    if folder:
        where = f"{REFERENCE_DOCS_ENV}={folder}"
        if Path(folder).is_dir():
            docs = Path(folder)
        else:
            local_problem = (f"{where} — that folder does not exist or this "
                             "app process cannot read it.")
    else:
        try:
            from geotech_references import _figures_db
            candidate = _figures_db._REPO_ROOT / "docs"
            if candidate.is_dir():
                docs = candidate
                where = f"source checkout {docs} ({REFERENCE_DOCS_ENV} unset)"
        except Exception:                               # noqa: BLE001
            pass
    present = set()
    if docs is not None:
        try:
            present = set(os.listdir(docs))
        except OSError as exc:
            local_problem = f"{where} — cannot list the folder: {exc}"

    # -- the local reference packs' own folders (REF-64) -------------------
    pack_files = _pack_folder_files()
    present |= pack_files
    from funhouse_agent.reference_packs import pack_problems
    packs_skipped = [f"reference pack skipped: {p}" for p in pack_problems()]

    # -- PDFs already fetched this session ---------------------------------
    try:
        from funhouse_agent import reference_docs
        cache = reference_docs.cache_dir()
        if os.path.isdir(cache):
            present |= {n for n in os.listdir(cache) if not n.endswith(".part")}
    except Exception:                                   # noqa: BLE001
        pass

    # -- SharePoint (fetched on first use) ---------------------------------
    remote, sp_label, sp_problem = set(), None, None
    try:
        from webapp import reference_fetch, sharepoint_store
        if sharepoint_store.configured():
            sp_label = f"SharePoint {reference_fetch.folder()}/"
            listed = reference_fetch.available_names()
            if listed is None:
                sp_problem = f"could not list {sp_label} (check the folder name)"
            else:
                remote = listed
    except Exception as exc:                            # noqa: BLE001
        sp_problem = f"SharePoint check failed: {type(exc).__name__}: {exc}"

    if sp_label is None and not present:
        if local_problem:
            return _check(name, FAIL, local_problem)
        if not folder:
            from webapp import engine_config
            return _check(name, WARN,
                          _UNSET_MESSAGE_FOUNDRY
                          if engine_config.is_foundry_deployment()
                          else _UNSET_MESSAGE)

    sources = [s for s in (where,
                           sp_label and f"{sp_label} (fetched on first use)")
               if s]
    src = " + ".join(sources) or "the fetched-PDF cache"
    missing = [n for n in needed if n not in present and n not in remote]
    n_remote = sum(1 for n in needed if n not in present and n in remote)
    extra = [p for p in (local_problem, sp_problem) if p] + packs_skipped
    if not missing:
        detail = f"{src}: all {len(needed)} PDFs found"
        if n_remote:
            detail += (f" ({len(needed) - n_remote} local, {n_remote} from "
                       f"{sp_label})")
        return _check(name, PASS, "\n".join([detail] + extra))
    lower = {p.lower(): p for p in present | remote}
    lines = [f"{src}: {len(needed) - len(missing)} of {len(needed)} PDFs "
             "found. Charts from a missing PDF cannot be read. Missing:"]
    for n in missing:
        hint = (f"  <- found {lower[n.lower()]!r}: names must match capital "
                "letters exactly on the cluster"
                if n.lower() in lower else "")
        lines.append(f"- {n}  ({', '.join(needed[n])}){hint}")
    return _check(name, FAIL if len(missing) == len(needed) else WARN,
                  "\n".join(lines + extra))


def diagnostics_checks() -> List[dict]:
    """The page's own lines in the connection diagnostics."""
    return [reference_docs_check()]


__all__ = ["disclaimer_text", "register_reference_fetcher",
           "add_report_library", "build_page_agent", "build_reviewer_agent",
           "turn_checks", "agent_picker", "reference_pdf_names",
           "reference_docs_check", "diagnostics_checks",
           "REFERENCE_DOCS_ENV"]
