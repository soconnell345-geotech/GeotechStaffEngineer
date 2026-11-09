"""The "core-only" CI job's page check (BACKLOG INF-32, the app split's
step 11 (c); plan ``module_work/APP_SPLIT_PLAN.md`` section 5 and Step 11's
Exit).

The job installs only ``aec-agent-core``, planlens and the hosting fleet's
pins: NOT geotech-staff-engineer, not the geotech-references submodule, not an
analysis back end. This repository's checkout is on ``PYTHONPATH`` only so
that the review page's own files (``funhouse_agent.review_build``, the
review prompt and helpers) and the chat shell (``webapp.core``,
``webapp.profiles``) can be loaded, as Document Review's own copy will be
at step 13; every GeotechStaffEngineer module on that path is refused (the
boundary test's blocker, ``page_builds._Blocker``), as an install without
GSE would refuse it. Then, for each of today's review builds:

1. it is built and runs one turn on a fake model, calling the tools whose
   code once reached into GSE (``plot_data``, ``calculate``,
   ``write_docx``) and one helper (``page_builds.run_page``);
2. its fingerprint equals the recorded one (``page_fingerprints.json``):
   the same prompts, tools, helpers, middleware and budgets as with GSE
   installed. The reference library's chart lines are absent here, so the
   two tools that read them may differ in that text alone
   (``page_builds.split_library_differences``; the one deliberate change
   the plan drops at step 13).

Run from the checkout's root::

    PYTHONPATH=. python -m webapp.tests.core_only

Exit 0 when every build runs and matches; 1 otherwise, with the reasons.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys

#: Distributions that must NOT be installed in the core-only job.
NOT_INSTALLED = ("geotech-staff-engineer", "geotech-references")


def _installed(dist: str) -> bool:
    """Installed in the environment (a record the checkout itself carries,
    a ``*.egg-info`` an editable install left in it, does not count: that
    is the checkout on ``PYTHONPATH``, not an install)."""
    try:
        found = importlib.metadata.distribution(dist)
    except importlib.metadata.PackageNotFoundError:
        return False
    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    where = os.path.abspath(str(found.locate_file("")))
    return not (where == here or where.startswith(here + os.sep))


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["GEOTECH_NO_DISCLAIMER"] = "1"
    problems = [f"{d} is installed" for d in NOT_INSTALLED if _installed(d)]
    if importlib.util.find_spec("geotech_references") is not None:
        problems.append("geotech_references is importable")

    from webapp.tests import page_builds as pb

    blocker = pb._Blocker()
    blocker.install()
    with open(pb.GOLDEN, encoding="utf-8") as fh:
        golden = json.load(fh)
    env = pb.app_environment()
    report = {"refused": blocker.refused, "builds": {}}
    for name in pb.REVIEW_BUILDS:
        entry = {}
        try:
            entry["turn"] = pb.run_page(name)["tool_results"]
        except BaseException as exc:                     # noqa: BLE001
            problems.append(f"{name}: the turn failed: "
                            f"{type(exc).__name__}: {exc}"[:600])
        try:
            current = pb.fingerprint(name)["app"]
            diff = pb.differences(golden["builds"][name]["app"], current)
            library = []
            if env != golden.get("app_environment"):
                library, diff = pb.split_library_differences(diff)
            entry["library_text_only"] = len(library)
            if diff:
                problems.append(f"{name}: the build changed "
                                f"({len(diff)} differences): "
                                + "; ".join(diff[:20]))
        except BaseException as exc:                     # noqa: BLE001
            problems.append(f"{name}: the fingerprint failed: "
                            f"{type(exc).__name__}: {exc}"[:600])
        report["builds"][name] = entry
    report["problems"] = problems
    print(json.dumps(report, indent=1, default=str))
    if problems:
        print("core-only page check FAILED:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        return 1
    print("core-only page check passed: "
          + ", ".join(pb.REVIEW_BUILDS) + " build, run and match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
