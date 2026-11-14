"""Name a failed pytest run's tests as GitHub error annotations.

A job's log needs a signed-in reader. Its annotations do not: anyone can read
them at /repos/<owner>/<repo>/check-runs/<job id>/annotations (the job ids
are at /repos/<owner>/<repo>/actions/runs/<run id>/jobs). The test jobs keep
pytest's output in a file and, when a step has failed, call this script on
it: the FAILED and ERROR lines of pytest's short summary are printed as
`::error` workflow commands, eight lines to an annotation and nine
annotations at most (GitHub keeps ten error annotations a step). The first
annotation opens with pytest's totals line.

Usage: python .github/scripts/name_failed_tests.py <pytest log>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PER_ANNOTATION = 8
MOST_ANNOTATIONS = 9
LINE_LIMIT = 300

_ANSI = re.compile(chr(27) + r"\[[0-9;]*m")
_TOTALS = re.compile(r"^=+ .*\b(failed|errors?)\b.* in [0-9.]+s.*=+$")


def _lines(text: str) -> list[str]:
    return [_ANSI.sub("", line).strip() for line in text.splitlines()]


def failed_lines(text: str) -> list[str]:
    """The FAILED and ERROR lines of the short summary, in order, once each."""
    seen: dict[str, None] = {}
    for line in _lines(text):
        if line.startswith(("FAILED ", "ERROR ")):
            seen.setdefault(line[:LINE_LIMIT])
    return list(seen)


def totals_line(text: str) -> str:
    """pytest's last line (`2 failed, 9000 passed in 2400.00s`), if it has one."""
    for line in reversed(_lines(text)):
        if _TOTALS.match(line):
            return line.strip("= ")
    return ""


def _escape(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotations(text: str) -> list[str]:
    """The workflow commands to print for this log."""
    failed = failed_lines(text)
    if not failed:
        tail = [line[:LINE_LIMIT] for line in _lines(text) if line][-15:]
        body = "\n".join(tail) if tail else "(the log is empty)"
        return ["::error title=No FAILED or ERROR line in the pytest log (its last lines)::" + _escape(body)]
    shown = failed[: PER_ANNOTATION * MOST_ANNOTATIONS]
    totals = totals_line(text)
    commands = []
    for start in range(0, len(shown), PER_ANNOTATION):
        chunk = shown[start : start + PER_ANNOTATION]
        title = "Failed tests %d to %d of %d" % (start + 1, start + len(chunk), len(failed))
        body = "\n".join(chunk)
        if start == 0 and totals:
            body = totals + "\n" + body
        commands.append("::error title=%s::%s" % (title, _escape(body)))
    return commands


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        text = Path(argv[1]).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print("::error title=No pytest log (a step before the tests failed)::" + _escape(str(exc)))
        return 0
    for command in annotations(text):
        print(command)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
