"""Named parts of an answer, found the same way by the app and the harness.

The live smoke harness holds an answer's citation key to its "Judgment
calls" list (``module_work/live_smoke/detectors.py``, ``answer_names`` with
``"within"``), and the app's turn-end judgment-call check
(:mod:`funhouse_agent.judgment_check`, BACKLOG GM-174) reads the calls in
that list. Both find the list here, so a wave measures the list the app
acted on.

* :func:`section`: the part, from the line that heads it -- a Markdown
  heading, a bold line or a line that opens with its name -- to the next
  Markdown heading at its level or above, the next line that is only bold
  text, or the end; ``None`` when no line heads it.
* :func:`items`: the entries of that part: its top-level numbered or
  bulleted items, each with the lines under it; else the rows of a table
  under it; else its paragraphs. Text after the list (a paragraph that
  starts at the list's own indent after a blank line) is not an entry.

Pure Python; no third-party import.
"""

from __future__ import annotations

import re
from typing import List, Optional

#: The parts an answer is searched for: name -> (the heading as written,
#: its pattern). w13 F18j: "Vesic (1973)" in a results heading hit the "a
#: reference cited for a judgment call" key though every call said "no
#: reference basis", so the key looks in this part only.
SCOPES = {"judgment_calls": ('"Judgment calls"', r"judge?ment calls?\b")}

_MD_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]")
#: A line that is only bold text, such as a "**References**" block after
#: the list (a call that opens in bold carries text after it, or a marker).
_BOLD_LINE = re.compile(r"^[ \t]{0,3}(\*\*|__)[^*_\n]+(\*\*|__)[ \t]*:?[ \t]*$")
#: A numbered or bulleted list item: its indent, then its marker.
_ITEM = re.compile(r"^([ \t]*)(?:\d+[.)]|[-*+•])[ \t]+\S")
_TABLE_ROW = re.compile(r"^[ \t]*\|")
_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}")


def section(text: str, scope: str) -> Optional[str]:
    """The ``scope`` part of ``text`` (see :data:`SCOPES`): from the line
    that heads it -- a Markdown heading, a bold line or a line that opens
    with its name -- to the next Markdown heading at its level or above,
    the next line that is only bold text, or the end. ``None`` when no line
    heads it."""
    head = re.compile(r"^[ \t]{0,3}(#{1,6}[ \t]*|\*\*|__)?[ \t]*"
                      r"(\d+[.)][ \t]*)?(\*\*|__)?[ \t]*" + SCOPES[scope][1],
                      re.IGNORECASE)
    lines = str(text or "").splitlines()
    for i, line in enumerate(lines):
        if not head.match(line):
            continue
        level = _MD_HEADING.match(line)
        level = len(level.group(1)) if level else None
        out = [line]
        for nxt in lines[i + 1:]:
            m = _MD_HEADING.match(nxt)
            if m and (level is None or len(m.group(1)) <= level):
                break
            if _BOLD_LINE.match(nxt):
                break
            out.append(nxt)
        return "\n".join(out)
    return None


def _indent(prefix: str) -> int:
    return len(prefix.expandtabs(4))


def items(part: Optional[str]) -> List[str]:
    """The entries of a part :func:`section` found (its heading line left
    out): the top-level list items, each with the lines under it; with no
    list, the rows of a table under the heading (its header row and rule
    left out); with neither, the paragraphs. ``[]`` for ``None`` or a part
    with nothing under its heading."""
    lines = str(part or "").splitlines()[1:]
    marked = [_indent(m.group(1)) for m in map(_ITEM.match, lines) if m]
    if marked:
        top = min(marked)
        out: List[List[str]] = []
        cur: Optional[List[str]] = None
        blank = False
        for line in lines:
            m = _ITEM.match(line)
            if m and _indent(m.group(1)) == top:
                cur = [line]
                out.append(cur)
                blank = False
                continue
            if cur is None:
                continue                    # text before the first item
            if not line.strip():
                blank = True
                continue
            lead = _indent(line[:len(line) - len(line.lstrip())])
            if blank and lead <= top:
                break                       # a paragraph after the list
            cur.append(line)
            blank = False
        return ["\n".join(c).strip() for c in out]
    rows = [ln for ln in lines if _TABLE_ROW.match(ln)]
    rule = next((i for i, ln in enumerate(rows) if _TABLE_RULE.match(ln)),
                None)
    if rule is not None:
        return [r.strip() for r in rows[rule + 1:] if r.strip("|-: \t")]
    paras = re.split(r"\n[ \t]*\n", "\n".join(lines))
    return [p.strip() for p in paras if p.strip()]


__all__ = ["SCOPES", "section", "items"]
