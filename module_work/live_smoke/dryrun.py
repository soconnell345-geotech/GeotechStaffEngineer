"""The $0 dry run: a scripted model does the bad things on purpose.

Through the app's real turn path (review page, a Tiny Apps user, SharePoint
on, an upload and its automatic orientation turn), the scripted model:

* writes a report straight into the system temp folder through a ROGUE
  harness-only tool (``harness_write_temp``, added to the agent through
  ``session.HARNESS_TOOLS``; it calls ``open()`` itself, so no app writer can
  redirect it) -- detector (a). Since 2026-10-09 the app's own writers move
  a save aimed outside the conversation into it, so an app tool can no
  longer be used to prove (a) fires;
* asks ``save_file`` for ``/tmp/<name>`` -- the CONTROL for that fix: the
  file must land in the conversation's folder and nowhere else
  (``run()``'s ``redirect`` result);
* calls ``read_document`` without its required argument and a tool that does
  not exist -- detector (e);
* asks SharePoint to upload a file that does not exist -- detector (d)
  (an honest failure);
* answers with a link to a missing file, a ``sandbox:`` link, a SharePoint
  URL no tool returned and a server path, and claims success -- detectors
  (b) and (e).

The second turn is a clean control ("Thanks" -> a plain answer): it must
raise no high or medium finding.

:func:`recheck` is the other $0 check: detectors (e), (k), (l) and (m) and
the flow expectations run again over a wave already recorded on disk.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List

#: What must fire (detector -> any of these codes).
EXPECTED = {
    "a_files_outside": {"file_in_temp", "tool_wrote_outside_conversation"},
    "b_answer_links": {"sandbox_link"},
    "b_answer_links#2": {"local_link_in_chat"},
    "b_answer_links#3": {"sharepoint_url_not_from_a_tool"},
    "b_answer_links#4": {"server_path_in_answer"},
    # read_document with no document: a missing argument (EV-30)
    "e_health": {"unknown_argument_refusal", "missing_argument"},
    "e_health#2": {"unknown_tool"},
    "e_health#3": {"success_claimed_after_error"},
    "d_sharepoint": {"sharepoint_tool_failed_honestly"},
    "f_cost": {"turn_cost"},
}

BAD_ANSWER = (
    "I've saved the report successfully. Download it here: "
    "[review report](report_final.pdf), or from sandbox:/mnt/data/out.pdf. "
    "The SharePoint copy is at https://contoso.sharepoint.com/sites/LiveSmoke"
    "/Shared%20Documents/General/GSE_app/nowhere/ghost.pdf and on disk at "
    "C:\\nowhere\\ghost.docx.")

FAKE_USAGE = {"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280,
              "input_token_details": {"cache_read": 1000,
                                      "cache_creation": 100}}


def temp_target() -> str:
    """Where the rogue tool writes (the system temp folder, directly)."""
    return os.path.join(tempfile.gettempdir(),
                        f"livesmoke_dry_{os.getpid()}_report.md")


def redirect_name() -> str:
    """The file name ``save_file`` is asked to put in ``/tmp``."""
    return f"livesmoke_dry_{os.getpid()}_saved.md"


def redirect_request() -> str:
    """The path ``save_file`` is asked for: ``/tmp/<name>``, the way a model
    writes it."""
    return "/tmp/" + redirect_name()


def _redirect_escapes() -> List[str]:
    """Where the redirected save must NOT be: the literal ``/tmp`` path as
    this OS resolves it, and the system temp folder."""
    return [os.path.abspath(redirect_request()),
            os.path.join(tempfile.gettempdir(), redirect_name())]


def rogue_tool():
    """A harness-only tool that writes straight into the system temp folder
    and reports the path -- what a careless tool would do. No app writer is
    involved, so nothing can redirect it."""
    import json
    from langchain_core.tools import tool

    @tool
    def harness_write_temp(content: str) -> str:
        """(Live-smoke dry run only.) Write a report into the system temp
        folder and return its path."""
        path = temp_target()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return json.dumps({"saved_path": path})

    return harness_write_temp


def bad_policy():
    """The scripted model for the dry-run flow."""
    from live_smoke.models import turn_script_policy
    turns = [
        [   # turn 1: the orientation the upload sends
            {"content": "", "tool_calls": [
                {"name": "harness_write_temp", "args": {
                    "content": "# Review report\n\nWritten by the dry run."}},
                {"name": "save_file", "args": {
                    "path": redirect_request(),
                    "content": "# Saved report\n\nAsked for /tmp."}},
                {"name": "read_document", "args": {"pages": "1"}},
                {"name": "make_excel_workbook", "args": {"rows": 3}},
                {"name": "sharepoint_upload_file", "args": {
                    "local_path": "C:/no/such/folder/review.pdf"}},
            ]},
            {"content": BAD_ANSWER},
        ],
        [   # turn 2: a clean control
            {"content": "You're welcome. Ask if you need anything else on "
                        "this sheet."},
        ],
    ]
    return turn_script_policy(turns)


FLOW = {
    "id": "DRY-bad-things",
    "title": "scripted model does the bad things on purpose",
    "page": "document_review",
    "user": "LIVESMOKE\\dryrun",
    "env": {"GEOTECH_VISION_PROBE": "0"},
    "steps": [{"upload": ["meck_10.17a"]}, {"say": "Thanks"}],
}


def fired(detectors_json: dict, turn: int = 0) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    turns = detectors_json.get("turns") or []
    if turn < len(turns):
        for f in turns[turn]["findings"]:
            out.setdefault(f["detector"], []).append(f["code"])
    return {k: sorted(set(v)) for k, v in out.items()}


def run(wave: str = "dry-run", runs_dir: str = None, verbose: bool = False
        ) -> dict:
    """Run the dry flow; ``{"ok", "fired", "control", "missing",
    "wave_dir"}``."""
    import glob
    import json
    from live_smoke import runner, session
    w = runner.Wave(wave, mode="dry-run", model_id="claude-haiku-5-5",
                    cap=0, fake=True, fake_policy=bad_policy(),
                    runs_dir=runs_dir or runner.RUNS, redo=True,
                    verbose=verbose)
    w.model.usage = dict(FAKE_USAGE)
    saved_tools = list(session.HARNESS_TOOLS)
    session.HARNESS_TOOLS[:] = [rogue_tool()]
    escaped: List[str] = []
    w.start()
    try:
        w.run_flow(FLOW)
    finally:
        session.HARNESS_TOOLS[:] = saved_tools
        w.finish()
        escaped = [p for p in _redirect_escapes() if os.path.isfile(p)]
        for p in [temp_target()] + _redirect_escapes():
            try:
                os.remove(p)
            except OSError:
                pass
    with open(os.path.join(w.dir, FLOW["id"], "detectors.json"),
              encoding="utf-8") as fh:
        dj = json.load(fh)
    # The control: save_file asked for /tmp/<name> put it in the
    # conversation's own folder, and nowhere outside it.
    landed = glob.glob(os.path.join(w.dir, FLOW["id"], "conversations", "*",
                                    "files", redirect_name()))
    flagged = [f for t in dj.get("turns") or [] for f in t["findings"]
               if f["detector"] == "a_files_outside"
               and redirect_name() in json.dumps(f)]
    redirect = {"requested": redirect_request(), "landed": landed,
                "escaped": escaped, "flagged_outside": flagged,
                "ok": bool(landed) and not escaped and not flagged}
    got = fired(dj, 0)
    missing = []
    for key, codes in EXPECTED.items():
        d = key.split("#")[0]
        if not set(got.get(d, [])) & codes:
            missing.append(f"{d}: one of {sorted(codes)}")
    control = [f for f in (dj["turns"][1]["findings"]
                           if len(dj.get("turns") or []) > 1 else [])
               if f["severity"] in ("high", "medium")]
    ok = not missing and not control and len(dj.get("turns") or []) == 2 \
        and redirect["ok"]
    return {"ok": ok, "fired": got, "missing": missing,
            "control": control, "redirect": redirect,
            "scenario": dj.get("scenario"),
            "wave_dir": w.dir, "turns": dj.get("turns")}


class _Turns:
    """A session for :func:`detectors.detect_expectations`: its turns."""

    def __init__(self, turns):
        self.turns = turns


def _find(conv: str, name: str) -> str:
    """Where a card named ``name`` sits in a copied conversation folder
    (``files/`` first)."""
    first = os.path.join(conv, "files", name)
    if os.path.isfile(first):
        return first
    for root, _dirs, files in os.walk(conv):
        if name in files:
            return os.path.join(root, name)
    return first


def recheck(wave_dir: str, ids=(), *, flows=None,
            looks_everywhere: bool = False) -> dict:
    """Detectors (e) health, (k) figure cards, (l) the judgment-call check
    and (m) answer numbers, and the flow expectations, again over a wave
    already on disk: no model, no API, $0 (BACKLOG EV-24, EV-30). (e)
    cannot see what the record does not keep (a tool the graph refused
    before it ran, the harness's own time limit).

    Each turn is rebuilt from the scenario's ``detectors.json`` and its
    copied conversation folder (the copy keeps the files' times, which (k)
    compares with the activity log's), and judged with the CURRENT
    ``flows.json`` expectations, so new ``expect`` keys can be tried on an
    old wave. ``looks_everywhere`` asks for a look at every figure card, as
    if every flow said ``"looked_at": true``. Returns ``{scenario:
    {"turns": [[finding, ...], ...], "scenario": [finding, ...]}}``."""
    import json
    from live_smoke import detectors as det
    from live_smoke import runner
    flows = {f["id"]: f for f in (flows if flows is not None
                                  else runner.load_flows())}
    out = {}
    for sid in sorted(os.listdir(wave_dir)):
        path = os.path.join(wave_dir, sid, "detectors.json")
        if sid.startswith("_") or not os.path.isfile(path):
            continue
        if ids and not any(sid == i or sid.startswith(i) for i in ids):
            continue
        with open(path, encoding="utf-8") as fh:
            dj = json.load(fh)
        flow = dict(flows.get(sid) or dj.get("flow") or {"id": sid})
        expect = dict(flow.get("expect") or {})
        if looks_everywhere:
            expect["cards"] = list(expect.get("cards") or []) + [
                {"kind": "figure", "min": 0, "looked_at": True}]
        flow["expect"] = expect
        recs, found, seen = [], [], {}
        prompts = [t.get("prompt") or "" for t in dj.get("turns") or []]
        for i, t in enumerate(dj.get("turns") or []):
            conv = os.path.join(wave_dir, sid, "conversations",
                                f"{t.get('session')}_"
                                f"{str(t.get('thread_id') or '')[:8]}")
            if not os.path.isdir(conv):
                found.append([])
                continue
            n = seen[conv] = seen.get(conv, 0) + 1
            acts = det.load_activity(conv)
            mine = [k for k, a in enumerate(acts) if a.get("turn") == n]
            lo, hi = (mine[0], mine[-1] + 1) if mine else (0, None)
            rec = {"conv_dir": conv, "files_dir": os.path.join(conv, "files"),
                   "final": t.get("final") or "", "t_end": float(i + 1),
                   "prompt": t.get("prompt") or "", "error": t.get("error"),
                   "activity_offset": lo, "activity_end": hi,
                   "entry": {"artifacts": [_find(conv, c)
                                           for c in t.get("cards") or []]}}
            recs.append(rec)
            ctx = det.turn_context(rec, expect=expect, prompts=prompts)
            found.append([f for d in (det.detect_health,
                                      det.detect_figure_cards,
                                      det.detect_judgment_check,
                                      det.detect_answer_numbers)
                          for f in d(ctx)])
        out[sid] = {"turns": found,
                    "scenario": det.detect_expectations(flow,
                                                        [_Turns(recs)])}
    return out


__all__ = ["run", "bad_policy", "FLOW", "EXPECTED", "fired", "rogue_tool",
           "redirect_request", "recheck"]
