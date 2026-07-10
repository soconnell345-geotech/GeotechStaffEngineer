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

def _init_session() -> None:
    ss = st.session_state
    if ss.get("initialized"):
        return
    ss.attachments = {}                 # shared with the agent's vision tools
    ss.temp_dir = core.new_session_dir()
    ss.artifacts = []                   # agent-produced files (download list)
    ss.messages = []                    # agent-facing history (replayed)
    ss.transcript = []                  # display entries
    ss.pending_notes = []               # attachment notes for the next turn
    ss.thread_id = core.new_thread_id()
    ss.total_tokens = 0
    ss.last_turn_tokens = 0
    ss.agent_error = None
    ss.engine = engine_config.resolve_engine()
    ss.agent = None
    if ss.engine.ok:
        try:
            ss.agent = core.build_agent(
                ss.engine.model, ss.attachments, ss.temp_dir, ss.artifacts)
        except Exception as exc:  # surface, don't crash the app
            ss.agent_error = f"{type(exc).__name__}: {exc}"
    ss.initialized = True


def _reset_session() -> None:
    """Clear the conversation and start a fresh thread + temp dir + agent."""
    for k in ("initialized", "attachments", "temp_dir", "artifacts", "messages",
              "transcript", "pending_notes", "thread_id", "total_tokens",
              "last_turn_tokens", "agent", "engine", "agent_error"):
        st.session_state.pop(k, None)
    _init_session()


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


# ---------------------------------------------------------------------------
# Sidebar: engine status, attachments, artifacts, tokens, reset
# ---------------------------------------------------------------------------

with st.sidebar:
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
            for a in atts:
                ss.transcript.append(
                    {"role": "attach", "text": f"{a.key} ({a.size:,} bytes)"})
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
    st.caption(f"thread `{ss.thread_id[:8]}` · {len(ss.messages)} msgs")
    if st.button("Reset conversation", type="secondary"):
        _reset_session()
        st.rerun()


# ---------------------------------------------------------------------------
# Artifact card (inline, in the chat flow at the turn that produced the file)
# ---------------------------------------------------------------------------

_KIND_ICON = {"html": "📄", "pdf": "📕", "png": "🖼️", "image": "🖼️",
              "svg": "🖼️", "dxf": "📐", "csv": "📊", "text": "📄", "other": "📎"}


def _render_artifact_card(path: str) -> None:
    """Render one artifact as a card: name/size/type + download + inline preview
    (HTML in a sandboxed iframe inside an expander, PDF via base64 data-URI
    iframe, images inline). Big files are download-only."""
    card = core.describe_artifact(path)
    if not card.exists:
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
        if card.kind in ("png", "image", "svg"):
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
        ss.transcript.append({"role": "user", "text": prompt})

        agent_content = core.assemble_user_message(ss.pending_notes, prompt)
        ss.pending_notes = []
        ss.messages.append({"role": "user", "content": agent_content})

        import os as _os
        before = core.snapshot_dir(ss.temp_dir)
        artifacts_before_len = len(ss.artifacts)     # save_fn appends here live
        staged_inputs = {  # staged upload paths are inputs, not artifacts
            _os.path.join(ss.temp_dir, k) for k in ss.attachments
        }

        with st.chat_message("assistant"):
            answer_box = st.empty()
            status = st.status("Working…", expanded=False)
            answer = ""
            final = ""
            turn_tokens = 0
            try:
                for item in core.stream_turn(ss.agent, ss.messages,
                                             ss.thread_id):
                    kind = item["kind"]
                    if kind == "token":
                        answer += item["text"]
                        answer_box.markdown(answer)
                    elif kind in ("tool_call", "todos", "tool_result"):
                        status.write(item["text"])
                    elif kind == "turn_done":
                        final = item["answer"]
                        turn_tokens = item["turn_tokens"]
                status.update(label="Done", state="complete")
            except Exception as exc:
                status.update(label="Error", state="error")
                st.error(f"{type(exc).__name__}: {exc}")

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
        for p in dir_new:                       # add dir-only files to the list
            if p not in ss.artifacts:
                ss.artifacts.append(p)
        turn_paths = core.collect_turn_artifacts(save_new, dir_new)
        ss.transcript.append({"role": "assistant", "text": final,
                              "artifacts": turn_paths})
        st.rerun()
