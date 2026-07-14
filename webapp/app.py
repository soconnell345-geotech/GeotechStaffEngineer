"""Streamlit shell for the geotech web chat app — a thin view over webapp.core.

Run locally:

    streamlit run webapp/app.py

Everything with logic (attachment staging, artifact capture, streaming, engine
resolution, the disclaimer text) lives in ``webapp.core`` / ``webapp.engine_config``
so it is testable without a streamlit runtime. This file only wires those to
streamlit widgets. One conversation per browser session (``st.session_state``),
with a per-session LangGraph ``thread_id``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

# streamlit runs this file with webapp/ as the script dir; the repo/site root
# that contains the `webapp` package is one level up and must be importable.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import streamlit as st

from webapp import core, engine_config

st.set_page_config(page_title="GeotechStaffEngineer", page_icon="⛰️",
                   layout="wide")


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

#: Bump when the session-state SHAPE changes. On a streamlit hot-reload the old
#: session_state survives into new code; if the schema differs we re-init cleanly
#: rather than run half-alive (the model-picker reload is what surfaced this).
_SCHEMA_VERSION = 2

#: App-owned session-state keys cleared on a schema-mismatch re-init (widget keys
#: like open_*/model_pick/uploader_* are left alone — they re-key per thread).
_APP_STATE_KEYS = (
    "initialized", "thread_id", "temp_dir", "attachments", "artifacts",
    "messages", "transcript", "pending_notes", "total_tokens",
    "last_turn_tokens", "agent", "agent_error", "engine", "model", "save_error",
    "behavior",
)


def _build_agent_for_session() -> None:
    """(Re)build the compiled agent wired to the current attachments dict +
    persistent files dir. Surfaces build errors without crashing the app."""
    ss = st.session_state
    ss.agent = None
    ss.agent_error = None
    if ss.engine.ok:
        try:
            _kind = (ss.get("behavior") or {}).get("agent_type", "full")
            if _kind == "full":
                ss.agent = core.build_agent(
                    ss.engine.model, ss.attachments, ss.temp_dir, ss.artifacts,
                    **core.behavior_build_kwargs(ss.get("behavior")))     # A5
            else:                                                         # A5e
                ss.agent = core.build_reviewer_agent(
                    _kind, ss.engine.model, ss.attachments, ss.temp_dir,
                    ss.artifacts)
        except Exception as exc:  # surface, don't crash the app
            ss.agent_error = f"{type(exc).__name__}: {exc}"


def _resolve_and_build(model_id: str) -> None:
    """Set the active model, (re)resolve the engine for it, and rebuild the agent
    on the CURRENT conversation — keeping thread/messages/attachments/artifacts.
    The persist+replay design makes a mid-conversation model switch safe: the next
    turn replays the existing history into the newly-built model. (The deployment
    Prompter engine is fixed and ignores the picked id — see engine_config.)"""
    ss = st.session_state
    ss.model = model_id
    ss.engine = engine_config.resolve_engine(model_id=model_id)
    _build_agent_for_session()


def _new_conversation() -> None:
    """Start a fresh conversation: new thread id + persistent files dir + agent.
    The conversation's meta is not written until its first turn, so an unused
    'New conversation' does not clutter the saved list."""
    ss = st.session_state
    ss.thread_id = core.new_thread_id()
    ss.temp_dir = core.conversation_files_dir(ss.thread_id)
    ss.attachments = {}                 # shared with the agent's vision tools
    ss.artifacts = []                   # agent-produced files (download list)
    ss.messages = []                    # agent-facing history (replayed)
    ss.transcript = []                  # display entries
    ss.pending_notes = []               # attachment notes for the next turn
    ss.total_tokens = 0
    ss.last_turn_tokens = 0
    ss.save_error = None
    ss.recovered_notice = False
    ss.behavior = core.default_behavior()       # A5: per-conversation pickers
    _resolve_and_build(core.default_model_id())


def _open_conversation(thread_id: str) -> None:
    """Resume a saved conversation: reload the display transcript, replay the
    agent-facing messages, re-register staged attachments, rebuild the download
    list, restore the conversation's model, and rebuild the agent on the SAME
    thread + persistent files dir."""
    ss = st.session_state
    ss.thread_id = thread_id
    ss.temp_dir = core.conversation_files_dir(thread_id)
    ss.attachments = {}
    ss.recovered_notice = False
    core.load_attachments(thread_id, ss.attachments)   # re-register upload bytes
    ss.transcript = core.load_transcript(thread_id)
    # A3 crash recovery: a turn interrupted mid-stream (kill/OOM/reload) left a
    # partial checkpoint — fold it in as a clearly-marked "recovered" entry.
    _rec = core.recover_partial(thread_id)
    if _rec is not None:
        ss.transcript.append(_rec)
        try:
            core.append_transcript(thread_id, _rec)
        except Exception:
            pass
        ss.recovered_notice = True
    ss.messages = core.load_messages(thread_id)         # replayed → agent memory
    ss.artifacts = core.artifacts_from_transcript(ss.transcript)
    ss.pending_notes = []
    ss.total_tokens = 0
    ss.last_turn_tokens = 0
    ss.save_error = None
    _meta = core.load_meta(thread_id) or {}
    ss.behavior = core.behavior_from_meta(_meta)     # A5: restore pickers
    _resolve_and_build(_meta.get("model") or core.default_model_id())


def _persist_turn(first_user_text: Optional[str]) -> None:
    """After a completed turn: rewrite the agent-facing messages, and update the
    conversation meta (title from the first user message, running turn count,
    the active model)."""
    ss = st.session_state
    user_turns = sum(1 for e in ss.transcript if e.get("role") == "user")
    title = core.auto_title(first_user_text) if (user_turns == 1 and
                                                 first_user_text) else None
    core.save_messages(ss.thread_id, ss.messages)
    core.touch_conversation(ss.thread_id, title=title, turn_count=user_turns,
                            model=ss.model)
    core.set_behavior(ss.thread_id, ss.get("behavior") or core.default_behavior())


def _init_session() -> None:
    ss = st.session_state
    if ss.get("initialized") and ss.get("_schema_version") == _SCHEMA_VERSION:
        return
    # first run OR a post-hot-reload schema mismatch: drop app state + re-init
    # cleanly (persisted conversations are unaffected and remain resumable).
    for k in _APP_STATE_KEYS:
        ss.pop(k, None)
    ss._schema_version = _SCHEMA_VERSION
    _new_conversation()
    ss.initialized = True


_init_session()
ss = st.session_state


# ---------------------------------------------------------------------------
# Header + disclaimer (prominent, at the very top)
# ---------------------------------------------------------------------------

st.title("⛰️ GeotechStaffEngineer")
st.caption("An LLM agent that drives industry-standard geotechnical analysis "
           "methods. Research/analysis aid — not a design deliverable.")

_disc = core.disclaimer_text()
st.warning(_disc.splitlines()[0] if _disc else "Professional-use disclaimer.")
with st.expander("Professional-use disclaimer — read before relying on any result",
                 expanded=False):
    st.text(_disc)

if ss.get("save_error"):
    st.warning(f"⚠️ This conversation could not be auto-saved: {ss.save_error}. "
               "Your current chat is intact on screen; the next successful turn "
               "will re-save it.")

if ss.get("recovered_notice"):
    st.info("♻️ A previous turn was interrupted before it finished saving. The "
            "partial response was recovered and added to this conversation "
            "(marked below).")


# ---------------------------------------------------------------------------
# Sidebar: engine status, attachments, artifacts, tokens, reset
# ---------------------------------------------------------------------------

def _relative_time(updated: float) -> str:
    import time as _t
    ago = max(0, int(_t.time() - (updated or 0)))
    if ago < 60:
        return "just now"
    if ago < 3600:
        return f"{ago // 60}m ago"
    if ago < 86400:
        return f"{ago // 3600}h ago"
    return f"{ago // 86400}d ago"


with st.sidebar:
    st.header("Conversations")
    if st.button("➕ New conversation", use_container_width=True,
                 key="new_conv"):
        _new_conversation()
        st.rerun()

    for _m in core.list_conversations()[:50]:
        _tid = _m["thread_id"]
        _current = (_tid == ss.thread_id)
        _title = _m.get("title") or "Untitled"
        _at = (_m.get("behavior") or {}).get("agent_type")
        _at_lbl = (f" · {core.agent_type_label(_at)}"
                   if _at and _at != "full" else "")
        _row = st.columns([0.72, 0.14, 0.14])
        with _row[0]:
            if st.button(("● " if _current else "") + _title, key=f"open_{_tid}",
                         help=f"{_relative_time(_m.get('updated'))} · "
                              f"{_m.get('turn_count', 0)} turns" +
                              (f" · {core.model_label(_m['model'])}"
                               if _m.get("model") else "") + _at_lbl,
                         use_container_width=True):
                if not _current:
                    _open_conversation(_tid)
                    st.rerun()
        with _row[1]:
            if st.button("✏️", key=f"rn_{_tid}", help="Rename"):
                ss[f"renaming_{_tid}"] = not ss.get(f"renaming_{_tid}", False)
        with _row[2]:
            if st.button("🗑️", key=f"del_{_tid}", help="Delete (to trash)"):
                core.delete_conversation(_tid)
                if _current:
                    _new_conversation()
                st.rerun()
        if ss.get(f"renaming_{_tid}"):
            _new_title = st.text_input("New title", value=_title,
                                       key=f"rntext_{_tid}")
            if st.button("Save title", key=f"rnsave_{_tid}"):
                core.rename_conversation(_tid, _new_title or _title)
                ss[f"renaming_{_tid}"] = False
                st.rerun()

    st.divider()
    st.header("Session")

    eng = ss.engine
    if ss.agent is not None:
        st.success(eng.message)
    elif ss.agent_error:
        st.error(f"Engine resolved ({eng.model_name}) but the agent failed to "
                 f"build: {ss.agent_error}")
    elif eng.source == "error":
        st.error(eng.message)
    else:
        st.warning(eng.message)

    # Model picker — applies to the CURRENT conversation going forward; the
    # persist+replay design keeps history/uploads/artifacts across the switch.
    _opts = core.model_choices()
    _ids = [c["id"] for c in _opts]
    _labels = {c["id"]: f"{c['label']} — {c['blurb']}" for c in _opts}
    _cur = ss.model if ss.model in _ids else _ids[0]
    _picked = st.selectbox(
        "Model", _ids, index=_ids.index(_cur),
        format_func=lambda i: _labels.get(i, i), key="model_pick",
        help="Switch the model for this conversation going forward (cheaper/"
             "faster models for quick questions). History, uploads and artifacts "
             "are kept; the next turn replays the conversation into the new model.")
    if _picked != ss.model:
        _resolve_and_build(_picked)
        if core.load_meta(ss.thread_id) is not None:
            core.touch_conversation(ss.thread_id, model=_picked)
        st.rerun()
    if eng.source == "prompter":
        st.caption("Model is fixed by the deployment; the picker doesn't apply.")

    # Agent picker (A5e) — the full geotech agent, or a narrow domain reviewer.
    # Per conversation, persisted in meta, shown on the conversation list line.
    _atypes = list(core.AGENT_TYPES)
    _cur_at = (ss.behavior or {}).get("agent_type", "full")
    _cur_at = _cur_at if _cur_at in _atypes else "full"
    _picked_at = st.selectbox(
        "Agent", _atypes, index=_atypes.index(_cur_at),
        format_func=lambda k: core.AGENT_TYPES[k], key=f"agent_{ss.thread_id}",
        help="The full geotech agent, or a narrow domain reviewer scoped to one "
             "discipline's methods + references and prompted in review mode. "
             "Applies to this conversation going forward; kept when you resume it.")
    if _picked_at != _cur_at:
        ss.behavior = {**ss.behavior, "agent_type": _picked_at}
        if core.load_meta(ss.thread_id) is not None:
            core.set_behavior(ss.thread_id, ss.behavior)
        _resolve_and_build(ss.model)          # rebuild as the selected variant
        st.rerun()

    # Working folder — where the agent's saves (calc packages, plots, files)
    # land. Default = this conversation's files/ dir (durable; shown as download
    # cards). Point it elsewhere (e.g. a project folder) and a durable copy is
    # still kept with the conversation. Persisted per conversation; applied to
    # the tool default-output-dir env each render so the next turn saves here.
    _wd = core.working_dir_for(ss.thread_id)
    _wd_in = st.text_input(
        "Working folder (agent saves land here)", value=_wd,
        key=f"workdir_{ss.thread_id}",
        help="Calc packages, plots and saved files default into this folder. "
             "Default is the conversation's files/ dir (kept with the chat, "
             "shown as download cards). Clear the box to reset to that default.")
    if (_wd_in or "").strip() != _wd:
        core.set_working_dir(ss.thread_id, _wd_in)
        st.rerun()
    core.apply_default_output_dir(_wd)

    # Behavior (A5): per-conversation pickers, persisted in meta. A change
    # rebuilds the agent for THIS conversation going forward (defaults reproduce
    # today's behavior exactly).
    _b = ss.behavior
    with st.expander("Behavior", expanded=False):
        _refs_on = st.checkbox(
            "Consult references", value=(_b["references"] != "off"),
            key=f"refs_{ss.thread_id}",
            help="When on, the agent can consult the reference library (DM7 / GEC "
                 "/ UFC …) through a scoped sub-agent. Turn off for pure-calc "
                 "sessions to save tokens.")
        _depth = st.select_slider(
            "Analysis depth", options=list(core.ANALYSIS_DEPTHS),
            value=_b["analysis_depth"], format_func=str.title,
            key=f"depth_{ss.thread_id}",
            help="How much analysis the agent does (a prompt preset, every "
                 "engine). Screening: single most appropriate method, concise. "
                 "Standard: default. Comprehensive: multiple methods compared + a "
                 "second-approach cross-check + a short sensitivity on governing "
                 "inputs + governing conditions & confidence + an offer to build a "
                 "calc package.")
        _route_calc = st.checkbox(
            "Route calculations through a calc agent (recommended)",
            value=bool(_b.get("route_calc", True)),
            key=f"routecalc_{ss.thread_id}",
            help="Runs tool-heavy calculations in a scoped sub-agent so the bulky "
                 "calc output stays out of this conversation — cheaper long chats "
                 "(A2). The key results (values + units + method) and the saved "
                 "calc-package path come back; the full detail is saved to a file "
                 "so nothing is lost.")
        with st.expander("Advanced caps", expanded=False):
            _refcalls = st.number_input(
                "Reference consult budget (model calls)", min_value=1,
                max_value=40, value=int(_b["ref_max_calls"]), step=1,
                key=f"refcalls_{ss.thread_id}",
                help="Max model calls the reference consultant spends per consult "
                     "before it must summarize and answer.")
            _rlim = st.number_input(
                "Primary step cap (recursion limit)", min_value=5, max_value=200,
                value=int(_b["recursion_limit"]), step=5,
                key=f"rlim_{ss.thread_id}",
                help="Max reasoning/tool steps the main agent may take in one turn "
                     "(LangGraph recursion limit).")
        _new_b = {**_b, "references": "anytime" if _refs_on else "off",
                  "analysis_depth": _depth, "ref_max_calls": int(_refcalls),
                  "recursion_limit": int(_rlim), "route_calc": bool(_route_calc)}
        if _new_b != _b:
            ss.behavior = _new_b
            if core.load_meta(ss.thread_id) is not None:
                core.set_behavior(ss.thread_id, _new_b)
            _resolve_and_build(ss.model)     # references/depth/caps -> rebuild
            st.rerun()

    st.divider()
    st.subheader("Attachments")
    uploaded = st.file_uploader(
        "Attach files (PDF, image, DXF, CSV, DIGGS…)",
        type=core.ACCEPTED_UPLOAD_TYPES, accept_multiple_files=True,
        key=f"uploader_{ss.thread_id}",
    )
    if uploaded:
        pairs = [(f.name, f.getvalue()) for f in uploaded]
        # Only stage names not already registered this session.
        fresh = [(n, d) for (n, d) in pairs
                 if core.sanitize_key(n) not in ss.attachments]
        if fresh:
            atts = core.stage_uploads(ss.attachments, ss.temp_dir, fresh)
            ss.pending_notes.append(core.attachment_note(atts))
            entries = [{"role": "attach", "text": f"{a.key} ({a.size:,} bytes)"}
                       for a in atts]
            ss.transcript.extend(entries)                # in-memory (always)
            try:                                          # persist — never crash
                for entry in entries:
                    core.append_transcript(ss.thread_id, entry)
                core.save_attachments_index(ss.thread_id, list(ss.attachments))
                ss.save_error = None
            except Exception as exc:
                ss.save_error = f"{type(exc).__name__}: {exc}"
            st.rerun()

    if ss.attachments:
        for key in ss.attachments:
            st.markdown(f"- `{key}`")
    else:
        st.caption("No attachments yet.")

    if ss.artifacts:
        st.divider()
        st.subheader("Downloads")
        for path in ss.artifacts:
            import os as _os
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                st.download_button(_os.path.basename(path), data=data,
                                   file_name=_os.path.basename(path),
                                   key=f"dl_{path}")
            except OSError:
                pass

    st.divider()
    st.caption(core.token_line(ss.last_turn_tokens, ss.total_tokens))
    _at_now = (ss.behavior or {}).get("agent_type", "full")
    st.caption(f"model: {core.model_label(ss.model)}"
               + (f" · {core.agent_type_label(_at_now)}" if _at_now != "full"
                  else "")
               + f" · thread `{ss.thread_id[:8]}` · {len(ss.messages)} msgs · "
               "auto-saved")


# ---------------------------------------------------------------------------
# Artifact card (inline, in the chat flow at the turn that produced the file)
# ---------------------------------------------------------------------------

_KIND_ICON = {"plotly": "📈", "html": "📄", "pdf": "📕", "png": "🖼️",
              "image": "🖼️", "svg": "🖼️", "dxf": "📐", "csv": "📊",
              "text": "📄", "other": "📎"}


def _render_artifact_card(path: str) -> None:
    """Render one artifact as a card: name/size/type + download + inline preview
    (HTML in a sandboxed iframe inside an expander, PDF via base64 data-URI
    iframe, images inline). Big files are download-only."""
    card = core.describe_artifact(path)
    if not card.exists:
        # Fail soft: the file is gone (e.g. produced in an external working
        # folder and not copied in, or deleted after the fact) — show a small
        # note instead of silently dropping the card or crashing on resume.
        st.caption(f"📎 {card.name} — file no longer available")
        return
    icon = _KIND_ICON.get(card.kind, "📎")
    with st.container(border=True):
        st.markdown(f"{icon} **{card.name}** · {card.size:,} bytes · "
                    f"{card.kind.upper()}")
        try:
            data = core.artifact_bytes(path)
        except OSError:
            st.caption("(file unavailable)")
            return
        st.download_button("Download", data=data, file_name=card.name,
                           key=f"dlcard_{path}")
        if card.kind == "plotly":
            # Native interactive chart from a *.plotly.json sidecar. plotly is
            # imported lazily/guarded so the app still runs without it (then the
            # figure is just download-only). Any parse error falls back to the
            # download button already rendered above.
            try:
                import plotly.io as pio
                fig = pio.from_json(core.read_text(path))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.caption("(could not render Plotly figure inline — "
                           "download above)")
        elif card.kind in ("png", "image", "svg"):
            try:
                st.image(path)
            except Exception:
                st.caption("(could not render image inline — download above)")
        elif card.kind == "html":
            with st.expander("Preview (HTML)", expanded=False):
                if card.size <= core.HTML_PREVIEW_MAX_BYTES:
                    import streamlit.components.v1 as components
                    components.html(core.read_text(path), height=600,
                                    scrolling=True)
                else:
                    st.caption(f"Too large to preview inline "
                               f"({card.size:,} bytes) — download above.")
        elif card.kind == "pdf":
            with st.expander("Preview (PDF)", expanded=False):
                uri = core.pdf_data_uri(path)
                if uri:
                    import streamlit.components.v1 as components
                    components.html(
                        f'<iframe src="{uri}" width="100%" height="600" '
                        f'style="border:none"></iframe>', height=620)
                else:
                    st.caption(
                        f"Too large to preview inline "
                        f"(> {core.PDF_PREVIEW_MAX_BYTES // (1024 * 1024)} MB) — "
                        f"download above.")


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

for entry in ss.transcript:
    role = entry.get("role")
    text = entry.get("text", "")
    if role == "attach":
        st.chat_message("user").caption(f"📎 attached {text}")
    else:
        with st.chat_message(role):
            st.markdown(text)
            for _path in entry.get("artifacts", []):
                _render_artifact_card(_path)


# Turn details (A7 local tracer) — the latest turn's trace, when GEOTECH_TRACE=1.
if core.tracing_enabled():
    _recent = core.load_recent_traces(ss.thread_id, n=1)
    if _recent:
        _tr = _recent[-1]
        _lbl = (f"turn details · {_tr.get('duration_s', '?')}s · "
                f"{_tr.get('turn_tokens', 0):,} tok · "
                f"{_tr.get('n_tool_calls', 0)} tool calls"
                + (" · error" if _tr.get("error") else ""))
        with st.expander(_lbl, expanded=False):
            st.json(_tr)


# ---------------------------------------------------------------------------
# Chat input + streaming
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask a geotechnical question…"
                       if ss.agent is not None else
                       "Configure an engine to start (see the sidebar)")

if prompt:
    if ss.agent is None:
        st.chat_message("assistant").error(
            "No engine is configured, so I can't answer yet. "
            + ss.engine.message)
    else:
        st.chat_message("user").markdown(prompt)
        user_entry = {"role": "user", "text": prompt}
        ss.transcript.append(user_entry)
        try:
            core.append_transcript(ss.thread_id, user_entry)     # persist
        except Exception as exc:
            ss.save_error = f"{type(exc).__name__}: {exc}"

        agent_content = core.assemble_user_message(ss.pending_notes, prompt)
        ss.pending_notes = []
        ss.messages.append({"role": "user", "content": agent_content})

        import os as _os
        before = core.snapshot_dir(ss.temp_dir)
        artifacts_before_len = len(ss.artifacts)     # save_fn appends here live
        staged_inputs = {  # staged upload paths are inputs, not artifacts
            _os.path.join(ss.temp_dir, k) for k in ss.attachments
        }

        # A6/A4: the working folder may point OUTSIDE the conversation files/ dir
        # (the agent's calc packages / plots default there via the output-dir
        # env). Snapshot it so those saves can be bridged back into files/ below
        # for durable, portable cards. None => same dir (the default), where the
        # files/ diff already covers everything and no bridging is needed.
        working_dir = core.working_dir_for(ss.thread_id)
        before_wd = (core.snapshot_dir(working_dir)
                     if os.path.abspath(working_dir) != os.path.abspath(ss.temp_dir)
                     else None)

        core.begin_partial(ss.thread_id, prompt)   # A3: mark in-progress turn
        turn_error = None
        _trace_on = core.tracing_enabled()          # A7: local per-turn tracer
        _trace_t0 = time.time()
        _trace_tools = []
        with st.chat_message("assistant"):
            answer_box = st.empty()
            status = st.status("Working…", expanded=False)
            answer = ""
            final = ""
            turn_tokens = 0
            _chunks = 0
            try:
                for item in core.stream_turn(
                        ss.agent, ss.messages, ss.thread_id,
                        recursion_limit=ss.behavior.get("recursion_limit")):
                    kind = item["kind"]
                    if kind == "token":
                        answer += item["text"]
                        answer_box.markdown(answer)
                        _chunks += 1
                        if _chunks % 8 == 0:        # A3: checkpoint partial text
                            core.checkpoint_partial(ss.thread_id, answer)
                    elif kind in ("tool_call", "todos", "tool_result"):
                        status.write(item["text"])
                        core.checkpoint_partial(ss.thread_id, answer)
                        if _trace_on and kind == "tool_call":   # A7 trace hop
                            _trace_tools.append(
                                {"t": round(time.time() - _trace_t0, 3),
                                 "call": (item.get("text") or "")[:80]})
                    elif kind == "turn_done":
                        final = item["answer"]
                        turn_tokens = item["turn_tokens"]
                status.update(label="Done", state="complete")
            except Exception as exc:
                turn_error = f"{type(exc).__name__}: {exc}"
                status.update(label="Error", state="error")
                st.error(turn_error)

            final = final or answer or "(no answer text)"
            answer_box.markdown(final)

        ss.messages.append({"role": "assistant", "content": final})
        ss.last_turn_tokens = turn_tokens
        ss.total_tokens += turn_tokens

        # Associate this turn's artifacts: save_fn appended save_file outputs to
        # ss.artifacts live during the turn; the directory diff catches anything
        # else written to the session dir (calc packages, DXFs, plots).
        save_new = ss.artifacts[artifacts_before_len:]
        dir_new = core.new_artifacts(ss.temp_dir, before, staged_inputs)
        if before_wd is not None:               # A6/A4: bridge saves made in an
            for p in core.import_external_artifacts(  # external working folder
                    working_dir, ss.temp_dir, before_wd, staged_inputs):
                if p not in dir_new:            # into files/ (durable + portable)
                    dir_new.append(p)
        for p in dir_new:                       # add dir-only files to the list
            if p not in ss.artifacts:
                ss.artifacts.append(p)
        turn_paths = core.collect_turn_artifacts(save_new, dir_new)
        assistant_entry = {"role": "assistant", "text": final,
                           "artifacts": turn_paths}
        if turn_error:                 # A3(c): the turn crashed mid-stream —
            assistant_entry["error"] = turn_error   # keep the partial, mark it
        ss.transcript.append(assistant_entry)
        # Persist — a save failure must NEVER lose a completed turn. The counters
        # and the in-memory transcript are already updated above; guard only the
        # disk writes and surface a small banner (rendered near the top) on error.
        try:
            core.append_transcript(ss.thread_id, assistant_entry)
            _persist_turn(prompt)      # rewrite messages.json + update meta
            core.clear_partial(ss.thread_id)   # A3: turn is durably saved now
            ss.save_error = None
        except Exception as exc:
            ss.save_error = f"{type(exc).__name__}: {exc}"
        if _trace_on:                       # A7: one compact JSONL line per turn
            core.write_turn_trace(ss.thread_id, {
                "ts": time.time(),
                "duration_s": round(time.time() - _trace_t0, 3),
                "turn_tokens": turn_tokens,
                "n_tool_calls": len(_trace_tools),
                "tools": _trace_tools,
                "error": turn_error,
            })
        st.rerun()
