"""The $0 dry run: a scripted model does the bad things on purpose.

Through the app's real turn path (review page, a Tiny Apps user, SharePoint
on, an upload and its automatic orientation turn), the scripted model:

* saves a report to the system temp folder (``save_file`` with an absolute
  path) -- detector (a);
* calls ``read_document`` without its required argument and a tool that does
  not exist -- detector (e);
* asks SharePoint to upload a file that does not exist -- detector (d)
  (an honest failure);
* answers with a link to a missing file, a ``sandbox:`` link, a SharePoint
  URL no tool returned and a server path, and claims success -- detectors
  (b) and (e).

The second turn is a clean control ("Thanks" -> a plain answer): it must
raise no high or medium finding.
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
    "e_health": {"unknown_argument_refusal"},
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
    return os.path.join(tempfile.gettempdir(),
                        f"livesmoke_dry_{os.getpid()}_report.md")


def bad_policy():
    """The scripted model for the dry-run flow."""
    from live_smoke.models import turn_script_policy
    target = temp_target()
    turns = [
        [   # turn 1: the orientation the upload sends
            {"content": "", "tool_calls": [
                {"name": "save_file", "args": {
                    "path": target,
                    "content": "# Review report\n\nWritten by the dry run."}},
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
    import json
    from live_smoke import runner
    w = runner.Wave(wave, mode="dry-run", model_id="claude-haiku-5-5",
                    cap=0, fake=True, fake_policy=bad_policy(),
                    runs_dir=runs_dir or runner.RUNS, redo=True,
                    verbose=verbose)
    w.model.usage = dict(FAKE_USAGE)
    w.start()
    try:
        w.run_flow(FLOW)
    finally:
        w.finish()
        try:
            os.remove(temp_target())
        except OSError:
            pass
    with open(os.path.join(w.dir, FLOW["id"], "detectors.json"),
              encoding="utf-8") as fh:
        dj = json.load(fh)
    got = fired(dj, 0)
    missing = []
    for key, codes in EXPECTED.items():
        d = key.split("#")[0]
        if not set(got.get(d, [])) & codes:
            missing.append(f"{d}: one of {sorted(codes)}")
    control = [f for f in (dj["turns"][1]["findings"]
                           if len(dj.get("turns") or []) > 1 else [])
               if f["severity"] in ("high", "medium")]
    ok = not missing and not control and len(dj.get("turns") or []) == 2
    return {"ok": ok, "fired": got, "missing": missing,
            "control": control, "scenario": dj.get("scenario"),
            "wave_dir": w.dir, "turns": dj.get("turns")}


__all__ = ["run", "bad_policy", "FLOW", "EXPECTED", "fired"]
