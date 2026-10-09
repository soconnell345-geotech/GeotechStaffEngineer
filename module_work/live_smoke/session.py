"""A headless browser session that runs the app's turn path.

:class:`Session` is ``webapp/app.py`` without Streamlit: the same
``webapp.core`` / ``webapp.profiles`` / ``webapp.turn_jobs`` /
``webapp.sharepoint_store`` calls, in the same order, on the same kind of
state (``self.ss`` stands for ``st.session_state``). Nothing the app does in
those functions is re-implemented here -- re-implementing them would hide
the bugs the harness is looking for. Where app.py has a block, the method
below names it (``# app.py: ...``). ``test_harness.py`` pins the order of
the turn block against app.py's source, so a change there that the session
does not follow fails a test instead of silently diverging.

:class:`AppEnv` sets up the process the way a Tiny Apps deployment would, as
far as that can be done locally: ``GEOTECH_DEPLOYMENT=tinyapps`` (the keyless
deployment: no personal key read, the model arrives through a registered
builder), the model injected with ``engine_config.register_model_builder``
(where ``webapp.tinyapps_engine.register`` puts the Prompter model), a
sandboxed ``GEOTECH_WEBAPP_DATA``, the HTTP upload mode Tiny Apps uses (so
the chat box takes pasted screenshots), and SharePoint configured so the
mirror, the restore and the agent's SharePoint tools all run against
:class:`~live_smoke.fake_sharepoint.LocalSharePointFM`. Identity: a Tiny Apps
user is the IIS header (``identity.from_header_values``, the function
``current_identity`` calls on that header -- ``multi_user`` and the
per-user roots follow); ``DEV_IDENTITY`` and anonymous are available too.
"""

from __future__ import annotations

import os
import shutil
import time
import types
from typing import Any, Dict, List, Optional, Tuple

#: Env vars the harness sets or clears for a scenario (restored afterwards).
_MANAGED_ENV = (
    "GEOTECH_WEBAPP_DATA", "GEOTECH_DEPLOYMENT", "GEOTECH_UPLOAD_MODE",
    "GEOTECH_PROMPTER_MODELS", "GEOTECH_WEBAPP_MODEL",
    "GEOTECH_SHAREPOINT_SITE_URL", "GEOTECH_SHAREPOINT_TOKEN",
    "GEOTECH_SHAREPOINT_ROOT", "GEOTECH_SHAREPOINT_TOKEN_FILE",
    "GEOTECH_SHAREPOINT_CLIENT_ID", "GEOTECH_SHAREPOINT_CLIENT_SECRET",
    "GEOTECH_SHAREPOINT_DRIVE_NAME", "GEOTECH_DEFAULT_OUTPUT_DIR",
    "DEV_IDENTITY", "GEOTECH_USER_EMAIL", "GEOTECH_APP_PROFILE",
    "GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET",
    "SHAREPOINT_SITE_URL", "KV_NAME", "GEOTECH_MARKUP_AUTHOR",
    "GEOTECH_VISION_PROBE", "GEOTECH_HEARTBEAT_S", "ANTHROPIC_API_KEY",
    "GEOTECH_REVIEW_AGENT",
)

#: Seconds a single turn may run before the harness gives up following it.
TURN_TIMEOUT_S = 30 * 60


class AppEnv:
    """Process setup for one scenario; ``with AppEnv(...) as env:``."""

    def __init__(self, sandbox: str, model, *, model_id: str,
                 sharepoint: bool = True, extra_env: Optional[dict] = None):
        self.sandbox = os.path.abspath(sandbox)
        self.model = model
        self.model_id = model_id
        self.sharepoint = sharepoint
        self.extra_env = dict(extra_env or {})
        self.data_root = os.path.join(self.sandbox, "data")
        self.cwd = os.path.join(self.sandbox, "cwd")
        self.store_dir = os.path.join(self.sandbox, "sharepoint")
        self.fm = None
        self._saved_env: Dict[str, Optional[str]] = {}
        self._saved_cwd: Optional[str] = None
        self.wipes = 0
        #: Called around every turn with the TurnRecord (the runner puts the
        #: outside-folder snapshots and the meter/SharePoint marks here).
        self.turn_hooks: List[Any] = []

    def turn_start(self, rec: "TurnRecord") -> None:
        for h in self.turn_hooks:
            h.turn_start(rec)

    def turn_end(self, rec: "TurnRecord") -> None:
        for h in self.turn_hooks:
            h.turn_end(rec)

    def __enter__(self) -> "AppEnv":
        from webapp import engine_config, sharepoint_store
        from live_smoke import fake_sharepoint as fsp
        for d in (self.data_root, self.cwd, self.store_dir):
            os.makedirs(d, exist_ok=True)
        self._saved_env = {k: os.environ.get(k) for k in _MANAGED_ENV}
        for k in _MANAGED_ENV:
            os.environ.pop(k, None)
        env = {
            "GEOTECH_WEBAPP_DATA": self.data_root,
            "GEOTECH_DEPLOYMENT": "tinyapps",
            "GEOTECH_UPLOAD_MODE": "http",
            # What webapp.tinyapps_engine.register publishes for its model.
            "GEOTECH_PROMPTER_MODELS": f"{self.model_id}={self.model_id}",
            "GEOTECH_WEBAPP_MODEL": self.model_id,
        }
        if self.sharepoint:
            env.update({
                "GEOTECH_SHAREPOINT_SITE_URL": fsp.SITE_URL,
                # Enough for sharepoint_store.configured(); the store is
                # handed the fake file manager, so no token is ever used.
                "GEOTECH_SHAREPOINT_TOKEN": "fake-token-live-smoke",
                "GEOTECH_SHAREPOINT_ROOT": fsp.ROOT,
            })
        env.update(self.extra_env)
        os.environ.update({k: str(v) for k, v in env.items()})
        model = self.model

        def _build(model_id=None):          # the Prompter builder's shape
            return model

        engine_config.register_model_builder(_build)
        if self.sharepoint:
            self.fm = fsp.LocalSharePointFM(self.store_dir)
        self.reset_process_state()
        self._saved_cwd = os.getcwd()
        os.chdir(self.cwd)
        return self

    def reset_process_state(self) -> None:
        """What a fresh app process starts with: a new SharePoint store (no
        cached folder URLs) over the same library, no download cache, no
        finished turn jobs."""
        from webapp import sharepoint_store, sharepoint_tools, turn_jobs
        sharepoint_store._STORE = (
            sharepoint_store.SharePointStore(file_manager=self.fm)
            if self.fm is not None else None)
        sharepoint_tools._DOWNLOADS.clear()
        with turn_jobs._JOBS_LOCK:
            for tid in [t for t, j in turn_jobs._JOBS.items() if j.done]:
                turn_jobs._JOBS.pop(tid, None)

    def restart(self) -> str:
        """Simulate the host losing its disk (cluster restart / redeploy):
        the local data root is moved aside, process state is reset. The
        SharePoint library survives. Returns where the old data went."""
        from webapp import core
        self.wipes += 1
        gone = os.path.join(self.sandbox, f"data_wiped_{self.wipes}")
        if os.path.isdir(self.data_root):
            shutil.move(self.data_root, gone)
        os.makedirs(self.data_root, exist_ok=True)
        with core._THREAD_ROOTS_LOCK:
            core._THREAD_ROOTS.clear()
        self.reset_process_state()
        return gone

    def __exit__(self, *exc) -> None:
        from webapp import engine_config, sharepoint_store
        try:
            if self._saved_cwd:
                os.chdir(self._saved_cwd)
        finally:
            engine_config.register_model_builder(None)
            sharepoint_store._STORE = None
            for k, v in self._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def make_identity(user: Any):
    """A :class:`webapp.identity.Identity` the way the app would get one.

    ``"DOMAIN\\\\user"`` -> the Tiny Apps IIS header
    (``identity.from_header_values``: multi-user); ``{"dev": "DOMAIN\\\\u"}``
    -> ``DEV_IDENTITY`` through ``current_identity``; ``None`` -> nobody
    (``current_identity`` with nothing set).
    """
    from webapp import identity
    if isinstance(user, str) and user:
        ident = identity.from_header_values([user])
        if ident is None:
            raise ValueError(f"unparseable principal {user!r}")
        return ident
    if isinstance(user, dict) and user.get("dev"):
        os.environ[identity.DEV_IDENTITY_ENV] = str(user["dev"])
        try:
            return identity.current_identity()
        finally:
            os.environ.pop(identity.DEV_IDENTITY_ENV, None)
    os.environ.pop(identity.DEV_IDENTITY_ENV, None)
    os.environ.pop(identity.USER_EMAIL_ENV, None)
    return identity.current_identity()


class TurnRecord(dict):
    """Everything one turn left behind, for the detectors (a plain dict)."""


class Session:
    """One browser session of one person on one page (``ss`` = session
    state). Mirrors ``webapp/app.py``."""

    def __init__(self, env: AppEnv, page: str, user: Any = None,
                 label: str = "main"):
        from webapp import core, profiles
        self.env = env
        self.label = label
        # app.py module top: _PROFILE, _IDENT, _ROOT, _THREAD_ROOT
        self.profile = profiles.get(page)
        self.ident = make_identity(user)
        self.root = profiles.session_root(self.profile, self.ident)
        self.thread_root = (self.root if os.path.abspath(self.root)
                            != os.path.abspath(core.data_root()) else None)
        self.ss = types.SimpleNamespace(
            thread_id=None, temp_dir=None, attachments={}, artifacts=[],
            messages=[], transcript=[], pending_notes=[], total_tokens=0,
            last_turn_tokens=0, save_error=None, behavior=None, agent=None,
            agent_error=None, engine=None, model=None, sp_sync=None,
            pending_orientation=None, recovered_notice=False)
        self.turns: List[TurnRecord] = []
        self.threads: List[str] = []
        self.sidebar: dict = {}
        self.last_restore: Optional[dict] = None
        # app.py: _init_session() -> _new_conversation()
        self.new_conversation()

    # -- app.py: _build_agent_for_session / _resolve_and_build ---------------
    def _build_agent_for_session(self) -> None:
        from webapp import core
        ss = self.ss
        ss.agent = None
        ss.agent_error = None
        if ss.engine.ok:
            try:
                _kind = (ss.behavior or {}).get("agent_type", "full")
                if not self.profile.specialists:
                    _kind = "full"
                if _kind == "full":
                    _kw = core.behavior_build_kwargs(ss.behavior)
                    _kw.update(self.profile.build_kwargs())
                    _kw.setdefault("markup_author", self.ident.markup_author)
                    ss.agent = core.build_agent(
                        ss.engine.model, ss.attachments, ss.temp_dir,
                        ss.artifacts, **_kw)
                else:
                    ss.agent = core.build_reviewer_agent(
                        _kind, ss.engine.model, ss.attachments, ss.temp_dir,
                        ss.artifacts)
            except Exception as exc:  # surfaced like the app does
                ss.agent_error = f"{type(exc).__name__}: {exc}"

    def _resolve_and_build(self, model_id: str) -> None:
        from webapp import engine_config
        self.ss.model = model_id
        self.ss.engine = engine_config.resolve_engine(model_id=model_id)
        self._build_agent_for_session()

    # -- app.py: _new_conversation / _open_conversation -----------------------
    def new_conversation(self) -> str:
        from webapp import core
        ss = self.ss
        ss.thread_id = core.new_thread_id()
        core.register_thread_root(ss.thread_id, self.thread_root)
        ss.temp_dir = core.conversation_files_dir(ss.thread_id)
        ss.attachments = {}
        ss.artifacts = []
        ss.messages = []
        ss.transcript = []
        ss.pending_notes = []
        ss.total_tokens = 0
        ss.last_turn_tokens = 0
        ss.save_error = None
        ss.recovered_notice = False
        ss.sp_sync = None
        ss.behavior = core.default_behavior()
        self._resolve_and_build(core.default_model_id())
        self.threads.append(ss.thread_id)
        return ss.thread_id

    def open_conversation(self, thread_id: str) -> None:
        from webapp import core, turn_jobs
        ss = self.ss
        ss.thread_id = thread_id
        core.register_thread_root(thread_id, self.thread_root)
        ss.temp_dir = core.conversation_files_dir(thread_id)
        ss.attachments = {}
        ss.recovered_notice = False
        core.load_attachments(thread_id, ss.attachments)
        ss.transcript = core.load_transcript(thread_id)
        _rec = (None if turn_jobs.get_turn_job(thread_id) is not None
                else core.recover_partial(thread_id))
        if _rec is not None:
            ss.transcript.append(_rec)
            try:
                core.append_transcript(thread_id, _rec)
            except Exception:
                pass
            ss.recovered_notice = True
        ss.messages = core.load_messages(thread_id)
        ss.artifacts = core.artifacts_from_transcript(ss.transcript)
        ss.pending_notes = []
        ss.total_tokens = 0
        ss.last_turn_tokens = 0
        ss.save_error = None
        ss.sp_sync = None
        _meta = core.load_meta(thread_id) or {}
        ss.behavior = core.behavior_from_meta(_meta)
        self._resolve_and_build(_meta.get("model") or core.default_model_id())
        if thread_id not in self.threads:
            self.threads.append(thread_id)

    # -- app.py: the sidebar, every render ------------------------------------
    def render_sidebar(self) -> dict:
        """What the sidebar computes on each run that matters downstream:
        the working folder (applied to the tools' output env), the
        conversation list, the download list and the Permanent storage
        block (the folder link)."""
        from webapp import core, sharepoint_store
        ss = self.ss
        convs = core.list_conversations(self.root)
        _wd = core.working_dir_for(ss.thread_id)
        core.apply_default_output_dir(_wd)
        downloads = []
        for path in ss.artifacts:
            try:
                with open(path, "rb") as fh:
                    fh.read(1)
                downloads.append({"path": path, "ok": True})
            except OSError as exc:
                downloads.append({"path": path, "ok": False,
                                  "error": f"{type(exc).__name__}"})
        _sp = sharepoint_store.get_store()
        storage = None
        if _sp.configured:
            _sync = ss.sp_sync or _sp.last_sync
            storage = {"sync": _sync,
                       "web_url": (_sync or {}).get("web_url"),
                       "folder": (_sync or {}).get("folder")}
        self.sidebar = {"conversations": [m.get("thread_id") for m in convs],
                        "working_dir": _wd, "downloads": downloads,
                        "storage": storage, "agent_error": ss.agent_error,
                        "engine": getattr(ss.engine, "message", None)}
        return self.sidebar

    # -- app.py: _stage_files / _queue_orientation ----------------------------
    def _stage_files(self, pairs) -> list:
        from webapp import core
        ss = self.ss
        atts = core.stage_uploads(ss.attachments, ss.temp_dir, pairs)
        ss.pending_notes.append(core.attachment_note(
            atts, review=not self.profile.specialists))
        entries = [{"role": "attach", "text": f"{a.key} ({a.size:,} bytes)"}
                   for a in atts]
        ss.transcript.extend(entries)
        try:
            for entry in entries:
                core.append_transcript(ss.thread_id, entry)
            core.save_attachments_index(ss.thread_id, list(ss.attachments))
            ss.save_error = None
        except Exception as exc:
            ss.save_error = f"{type(exc).__name__}: {exc}"
        return atts

    def _queue_orientation(self, atts) -> None:
        from webapp import profiles, turn_jobs
        ss = self.ss
        if self.profile.orientation and ss.agent is not None \
                and turn_jobs.get_turn_job(ss.thread_id) is None \
                and profiles.orient_on_upload([a.key for a in atts],
                                              ss.transcript):
            ss.pending_orientation = [a.key for a in atts]

    # -- the user's actions ---------------------------------------------------
    def upload(self, pairs: List[Tuple[str, bytes]]) -> List[TurnRecord]:
        """The sidebar uploader (http mode): stage the fresh files, queue the
        orientation, rerun -- which sends the orientation turn on the review
        page. Returns the turns that ran (none on the geotech page)."""
        from webapp import core
        self.render_sidebar()
        fresh = [(n, d) for (n, d) in pairs
                 if core.sanitize_key(n) not in self.ss.attachments]
        if not fresh:
            return []
        atts = self._stage_files(fresh)
        self._queue_orientation(atts)
        return self._rerun(prompt=None)        # st.rerun()

    def paste(self, files: List[Tuple[str, bytes]],
              text: str = "") -> List[TurnRecord]:
        """The chat box with files (``accept_file``): a pasted screenshot is
        renamed by ``core.pasted_upload_name`` and staged; with text the
        message goes now, without it the upload's orientation (if any) goes
        on the rerun."""
        from webapp import core
        self.render_sidebar()
        _pairs = [(core.pasted_upload_name(n, i), d)
                  for i, (n, d) in enumerate(files)]
        _fresh = [(n, d) for (n, d) in _pairs
                  if core.sanitize_key(n) not in self.ss.attachments]
        if _fresh:
            _atts = self._stage_files(_fresh)
            if not (text or "").strip():
                self._queue_orientation(_atts)
                return self._rerun(prompt=None)
        return self._rerun(prompt=(text or "").strip() or None,
                           rendered=True)

    def say(self, text: str) -> List[TurnRecord]:
        """Type a message and press enter."""
        return self._rerun(prompt=text)

    def _rerun(self, prompt: Optional[str],
               rendered: bool = False) -> List[TurnRecord]:
        """One script run after an action: sidebar, then the orientation the
        upload queued (when no message was typed), then the turn."""
        from webapp import core, profiles, turn_jobs
        ss = self.ss
        if not rendered:
            self.render_sidebar()
        _orient = ss.pending_orientation
        ss.pending_orientation = None
        kind = "user"
        if not prompt and _orient and self.profile.orientation \
                and ss.agent is not None \
                and turn_jobs.get_turn_job(ss.thread_id) is None:
            prompt = profiles.orientation_request_for(self.profile, [
                core.Attachment(key=n, path=os.path.join(ss.temp_dir, n),
                                size=0) for n in _orient])
            kind = "orientation"
        if not prompt:
            return []
        rec = self._send(prompt, kind)
        # the rerun that follows _follow_turn_job: the sidebar shows the sync
        self.render_sidebar()
        rec["sidebar_after"] = dict(self.sidebar)
        return [rec]

    # -- app.py: the "if prompt:" block ---------------------------------------
    def _send(self, prompt: str, kind: str) -> TurnRecord:
        from webapp import core, profiles, turn_jobs
        # The harness's own bookkeeping goes through this alias, so the
        # app.py-order test (test_harness.py) sees only the app's calls.
        import webapp.core as record_core
        ss = self.ss
        rec = TurnRecord(kind=kind, prompt=prompt, session=self.label,
                         page=self.profile.name, thread_id=ss.thread_id,
                         user=self.ident.qualified_name,
                         multi_user=self.ident.multi_user)
        if ss.agent is None:
            rec.update(error=("No engine/agent: " + str(ss.agent_error or
                                                         ss.engine.message)),
                       final="", skipped=True, t_start=time.time(),
                       t_end=time.time())
            self.turns.append(rec)
            return rec
        if turn_jobs.get_turn_job(ss.thread_id) is not None:
            rec.update(error="a turn is already running", skipped=True,
                       final="", t_start=time.time(), t_end=time.time())
            self.turns.append(rec)
            return rec
        user_entry = {"role": "user", "text": prompt}
        ss.transcript.append(user_entry)
        rec["turn_index"] = sum(1 for e in ss.transcript
                                if e.get("role") == "user")
        try:
            core.append_transcript(ss.thread_id, user_entry)
        except Exception as exc:
            ss.save_error = f"{type(exc).__name__}: {exc}"

        agent_content = core.assemble_user_message(ss.pending_notes, prompt)
        ss.pending_notes = []
        ss.messages.append({"role": "user", "content": agent_content})

        before = core.snapshot_dir(ss.temp_dir)
        artifacts_before_len = len(ss.artifacts)
        staged_inputs = {os.path.join(ss.temp_dir, k) for k in ss.attachments}

        working_dir = core.working_dir_for(ss.thread_id)
        core.apply_default_output_dir(working_dir)
        before_wd = (core.snapshot_dir(working_dir)
                     if os.path.abspath(working_dir)
                     != os.path.abspath(ss.temp_dir) else None)

        if self.ident.multi_user or self.profile is not profiles.DEFAULT:
            try:
                core.tag_conversation(
                    ss.thread_id, owner=(self.ident.display_name
                                         if self.ident.multi_user else None),
                    page=self.profile.name)
            except Exception:
                pass
        try:
            _turn_note = core.working_files_note(
                ss.temp_dir, ss.transcript, exclude_text=agent_content)
        except Exception:
            _turn_note = ""
        core.begin_partial(ss.thread_id, prompt)

        conv_dir = record_core.conversation_dir(ss.thread_id)
        act_path = os.path.join(conv_dir, "activity.jsonl")
        rec.update(conv_dir=conv_dir, files_dir=ss.temp_dir,
                   working_dir=working_dir, agent_content=agent_content,
                   turn_note=_turn_note, before=sorted(before),
                   staged_inputs=sorted(staged_inputs),
                   activity_offset=_line_count(act_path),
                   artifacts_before_len=artifacts_before_len)
        self.env.turn_start(rec)
        rec["t_start"] = time.time()
        job = turn_jobs.start_turn_job(
            ss.agent, ss.messages, ss.thread_id,
            ss.behavior.get("recursion_limit"),
            ctx={
                "prompt": prompt,
                "temp_dir": ss.temp_dir,
                "before": before,
                "staged_inputs": staged_inputs,
                "working_dir": working_dir,
                "before_wd": before_wd,
                "artifacts": ss.artifacts,
                "artifacts_before_len": artifacts_before_len,
                "transcript": ss.transcript,
                "trace_on": core.tracing_enabled(ss.behavior.get("trace")),
                "model": ss.model,
                "behavior": ss.behavior,
                "turn_note": _turn_note,
            })
        self._follow_turn_job(job, rec)
        rec["t_end"] = time.time()
        rec["activity_end"] = _line_count(act_path)
        rec["after"] = sorted(record_core.snapshot_dir(ss.temp_dir))
        self.env.turn_end(rec)
        self.turns.append(rec)
        return rec

    # -- app.py: _follow_turn_job ---------------------------------------------
    def _follow_turn_job(self, job, rec: TurnRecord) -> None:
        from webapp import core
        ss = self.ss
        events: List[dict] = []
        deadline = time.time() + TURN_TIMEOUT_S
        i = 0
        while True:
            with job._lock:
                chunk = job.events[i:]
                finished = job.done
            events.extend(chunk)
            i += len(chunk)
            if finished and i >= len(job.events):
                break
            if time.time() > deadline:
                rec["timeout"] = True
                break
            time.sleep(0.2)
        res = dict(job.result or {})
        rec["events"] = [{"kind": e.get("kind"),
                          "text": str(e.get("text") or "")[:2000]}
                         for e in events if e.get("kind") != "token"]
        rec["result"] = res
        rec["final"] = res.get("final") or ""
        rec["error"] = res.get("error")
        rec["save_error"] = res.get("save_error")
        synced = bool(ss.transcript) and \
            ss.transcript[-1].get("role") == "assistant" and \
            ss.transcript[-1].get("text") == res.get("final")
        if not synced:
            ss.transcript = core.load_transcript(ss.thread_id)
            ss.messages = core.load_messages(ss.thread_id)
            ss.artifacts = core.artifacts_from_transcript(ss.transcript)
        rec["synced_in_place"] = synced
        ss.last_turn_tokens = res.get("turn_tokens", 0)
        ss.total_tokens += res.get("turn_tokens", 0) or 0
        ss.save_error = res.get("save_error")
        if res.get("sp_sync") is not None:
            ss.sp_sync = res["sp_sync"]
        rec["sp_sync"] = res.get("sp_sync")
        entry = ss.transcript[-1] if ss.transcript else {}
        rec["entry"] = dict(entry) if entry.get("role") == "assistant" else {}
        rec["session_artifacts"] = list(ss.artifacts)
        if not rec.get("timeout"):
            job.consumed = True

    # -- app.py: Permanent storage > Find a past conversation -----------------
    def sp_where(self) -> dict:
        return {"owner": (self.ident.display_name if self.ident.multi_user
                          else None),
                "page": self.profile.name}

    def list_remote(self) -> List[dict]:
        from webapp import sharepoint_store
        _sp = sharepoint_store.get_store()
        if not _sp.configured:
            return []
        return _sp.list_remote_conversations(**self.sp_where())

    def restore(self, name: Optional[str] = None) -> dict:
        """List this page's mirrored conversations and Restore one (the
        newest, or the one called ``name``), then open it."""
        from webapp import sharepoint_store
        self.render_sidebar()
        remote = self.list_remote()
        res: Dict[str, Any] = {"listed": [r["name"] for r in remote]}
        if not remote:
            res.update(status="error", errors=["nothing listed"])
            self.last_restore = res
            return res
        pick = next((r for r in remote if r["name"] == name), None) \
            if name else remote[0]
        if pick is None:
            res.update(status="error", errors=[f"{name!r} not listed"])
            self.last_restore = res
            return res
        _sp = sharepoint_store.get_store()
        out = _sp.restore_conversation(pick["name"], root=self.root,
                                       **self.sp_where())
        res.update(out)
        res["picked"] = pick["name"]
        if out.get("thread_id") and out.get("status") in ("restored",
                                                          "exists"):
            self.open_conversation(out["thread_id"])
        self.render_sidebar()
        self.last_restore = res
        return res


def _line_count(path: str) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


__all__ = ["AppEnv", "Session", "TurnRecord", "make_identity",
           "TURN_TIMEOUT_S"]
