"""Professional-use disclaimer notice — one-time (per user) first-import banner
plus an on-demand full-text printer and a ``geotech-disclaimer`` console script.

Why this exists / the honest limitation
----------------------------------------
pip runs NO code when installing a wheel — there is no supported post-install
hook, and sdist-only ``setup.py`` tricks are unreliable (pip hides their output
without ``-v``). So the package cannot truly "print on install". The closest
legitimate equivalents are shipped instead:

* the README's "Professional-use disclaimer" section — this is the PyPI project
  page, the nearest thing to an install-time notice a wheel has;
* ``DISCLAIMER.md`` carried in the distribution; and
* this ONE-TIME stderr banner on first ``import funhouse_agent`` (persisted with a
  marker file so it shows once per user, not once per process).

The banner NEVER raises, prints to stderr only (never stdout), is silenced by
``GEOTECH_NO_DISCLAIMER=1``, and is suppressed under pytest.
"""

from __future__ import annotations

import os
import sys

#: Set to a non-empty value to silence the one-time first-import banner.
SUPPRESS_ENV = "GEOTECH_NO_DISCLAIMER"

#: Per-user marker dir/file (stdlib-only; no platformdirs dependency).
_ACK_DIR = "~/.geotech_staff_engineer"
_ACK_FILENAME = "disclaimer_ack"

#: 4-line stderr banner shown once per user on first import.
SHORT_NOTICE = (
    "GeotechStaffEngineer is an ANALYSIS/RESEARCH AID, not a design deliverable. "
    "Every\n"
    "result must be independently reviewed by a licensed professional engineer "
    "familiar\n"
    "with the site; using it creates no engineer-of-record relationship, and it "
    "carries no warranty.\n"
    "Full terms: run `geotech-disclaimer`.  Silence this notice: set "
    "GEOTECH_NO_DISCLAIMER=1"
)

#: Full text printed on demand by :func:`disclaimer` / the console script.
FULL_NOTICE = """\
GeotechStaffEngineer -- Professional-use disclaimer
===================================================

This toolkit is an ANALYSIS and RESEARCH AID -- a multiplier for a qualified
engineer's judgment, not a replacement for it, and NOT a design deliverable. It
runs industry-standard geotechnical methods and an LLM agent that drives them,
but it does not know your site and cannot exercise engineering judgment.

What it is / is not
  - A library of deterministic methods, a probabilistic variability engine,
    digitized references, and an LLM agent that drives them.
  - NOT a stamped design, a professional opinion, or a substitute for
    site-specific investigation, characterization, and review.
  - Using this software creates NO engineer-of-record relationship and no
    professional engagement with the authors or contributors.

Your responsibilities
  - A qualified, licensed professional engineer familiar with the site must
    independently review every input, assumption, method selection, and result
    before it is relied upon. Outputs are candidate calculations to be checked.
  - Confirm the method applies to your problem (each module's DESIGN.md /
    VALIDATION.md documents its applicability limits and conventions).
  - Confirm inputs and units -- all quantities are SI (m, kPa, kN, degrees).
  - Treat LLM-agent output with particular care: it can pick the wrong method,
    mis-transcribe a value, or read a chart imperfectly. It is a starting point
    for review, never a final basis for design.

Validation scope
  - Selected modules are validated against published worked examples; the
    specific cases and verdicts are in validation_examples/RESULTS.md and the
    per-module VALIDATION.md files. Validation covers ONLY those documented cases.

No warranty
  - Provided under the MIT License "AS IS", WITHOUT WARRANTY OF ANY KIND. The
    authors are not liable for any claim, damages, or other liability arising
    from its use. Responsibility for the safety, adequacy, and code-compliance
    of any design remains entirely with the licensed professional engineer who
    adopts it. See LICENSE and DISCLAIMER.md for the full terms.
"""


def _ack_path() -> str:
    return os.path.join(os.path.expanduser(_ACK_DIR), _ACK_FILENAME)


def _already_acknowledged() -> bool:
    """True if the one-time banner has been shown for this user before. On any
    error, returns True (fail toward NOT nagging)."""
    try:
        return os.path.exists(_ack_path())
    except Exception:
        return True


def _write_ack() -> None:
    """Best-effort marker write; a failure just risks showing the banner again."""
    try:
        os.makedirs(os.path.expanduser(_ACK_DIR), exist_ok=True)
        with open(_ack_path(), "w", encoding="utf-8") as fh:
            fh.write("shown\n")
    except Exception:
        pass


def _running_under_pytest() -> bool:
    # PYTEST_CURRENT_TEST is set during test execution; "pytest" in sys.modules
    # also covers collection/import time (before any test sets the env var), so we
    # never print the banner or write the marker during a test run.
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or ("pytest" in sys.modules)


def maybe_show_first_import_notice() -> bool:
    """Show the one-time stderr banner unless suppressed, already acknowledged, or
    running under pytest. Returns True iff it printed. NEVER raises."""
    try:
        if os.environ.get(SUPPRESS_ENV):
            return False
        if _running_under_pytest():
            return False
        if _already_acknowledged():
            return False
        sys.stderr.write("\n" + SHORT_NOTICE + "\n\n")
        _write_ack()
        return True
    except Exception:
        return False


def disclaimer(file=None) -> None:
    """Print the full professional-use disclaimer on demand (stdout by default)."""
    stream = sys.stdout if file is None else file
    try:
        stream.write(FULL_NOTICE.rstrip() + "\n")
    except Exception:
        pass


def main() -> int:
    """Entry point for the ``geotech-disclaimer`` console script."""
    disclaimer()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
