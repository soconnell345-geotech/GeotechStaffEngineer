"""Review of brief 15 (the run report on malformed records, BACKLOG EV-17):
a well-formed look is never left out. Two documents of one file name (two
folders) are kept as one document in the report, with the page count of the
one opened last; a look at a later page of the longer one is a page of it
and stays drawn, and a page beyond both is still left out and counted.
Made-up records, written line by line.
"""

from __future__ import annotations

import json
import re

from aec_agent_core import run_report

NAME = "made_up_set.pdf"
VIEWER_DATA = re.compile(r'<script type="application/json"[^>]*>(.*?)'
                         r'</script>', re.S)


def _run(folder, steps):
    """``steps``: ``("open", source, n_pages, handle)`` or
    ``("zoom", source, page, view)``, written as the activity log holds
    them."""
    lines = [{"turn": 1, "t": 0.0, "agent": "primary", "event": "turn_start",
              "prompt": "made-up question", "model": "made-up-model"}]
    for k, step in enumerate(steps):
        if step[0] == "open":
            name, args = "open_document", {"source": step[1]}
            result = {"handle": step[3], "name": step[1], "n_pages": step[2]}
        else:
            name = "render_region"
            args = {"attachment_key": step[1], "page": step[2],
                    "bbox": list(step[3])}
            result = {"page": step[2], "pdf_page": step[2] + 1,
                      "view": list(step[3]), "analysis": "made-up"}
        same = {"turn": 1, "t": 0.01 * (k + 1), "agent": "primary",
                "name": name, "run_id": f"made-up-{k}",
                "parent_run_id": "made-up-top"}
        text = json.dumps(result)
        lines.append({**same, "event": "tool_start", "args": args})
        lines.append({**same, "event": "tool_end", "result": text,
                      "truncated": False, "chars": len(text)})
    (folder / "activity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")


def _boxes(page):
    """Document page -> the boxes the viewer draws on it."""
    data = json.loads(VIEWER_DATA.search(page).group(1))
    assert len(data["docs"]) == 1
    return {int(p): [lk[2] for lk in e.get("l", [])]
            for p, e in data["docs"][0]["pages"].items() if e.get("l")}


TWO = [("open", "made_up_a/" + NAME, 6, "h1"),
       ("zoom", "made_up_a/" + NAME, 4, (50.0, 60.0, 200.0, 220.0)),
       ("open", "made_up_b/" + NAME, 3, "h2"),
       ("zoom", "made_up_b/" + NAME, 1, (50.0, 60.0, 200.0, 220.0))]


def test_a_page_of_the_longer_of_two_documents_of_one_name_is_drawn(
        tmp_path):
    _run(tmp_path, TWO)
    page = run_report.build_report(str(tmp_path))
    assert "id='left-out'" not in page          # nothing here is malformed
    assert _boxes(page) == {1: [[50.0, 60.0, 200.0, 220.0]],
                            4: [[50.0, 60.0, 200.0, 220.0]]}


def test_a_page_beyond_both_documents_of_one_name_is_still_left_out(
        tmp_path):
    _run(tmp_path, TWO + [
        ("zoom", "made_up_b/" + NAME, 40, (50.0, 60.0, 200.0, 220.0))])
    page = run_report.build_report(str(tmp_path))
    assert f"{run_report.LEFT_OUT['past_end']}: <b>1</b>" in page
    assert sorted(_boxes(page)) == [1, 4]
