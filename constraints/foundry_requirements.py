"""Print the third-party requirements of the installed app, its core
(aec-agent-core, since the app split's step 10), planlens and
geotech-references, for an environment where numpy is held at 1.26
(BACKLOG INF-53).

Install the four packages with ``pip install --no-deps`` first, then:

    python constraints/foundry_requirements.py > requirements.txt
    pip install -r requirements.txt -c constraints/foundry-numpy126.txt

Left out: the four packages themselves, numpy (the environment's own
1.26.4 stays; planlens 0.13 declares numpy >= 2.0, its next release
>= 1.26) and PyNiteFEA (no release the app accepts installs on numpy 1.26;
see foundry-numpy126.txt).
"""

import importlib.metadata as md
import re

OWN = ("geotech-staff-engineer", "aec-agent-core", "planlens",
       "geotech-references")
LEFT_OUT = set(OWN) | {"numpy", "pynitefea"}


def _app_requires_core() -> bool:
    """Whether the installed app pins aec-agent-core (master since step 10)."""
    try:
        reqs = md.requires("geotech-staff-engineer") or ()
    except md.PackageNotFoundError:
        return False
    return any(re.split(r"[\s<>=!~;\[(]", r, maxsplit=1)[0].lower()
               == "aec-agent-core" for r in reqs)


def requirements() -> list:
    """The core requirement lines of the four installed packages."""
    lines = []
    for dist in OWN:
        try:
            reqs = md.requires(dist) or ()
        except md.PackageNotFoundError:
            # An app released before the core existed (5.34.0 and older)
            # installs without it; any other missing package is an error,
            # and so is the core for an app that requires it (built from
            # master since step 10; since step 11 (a) the app imports its
            # model connections and vision tools from the core, so a test
            # wheel set without the core's wheel fails at import).
            if dist != "aec-agent-core" or _app_requires_core():
                raise
            continue
        for req in reqs:
            if "extra ==" in req:
                continue
            name = re.split(r"[\s<>=!~;\[(]", req, maxsplit=1)[0]
            if name.lower().replace("_", "-") not in LEFT_OUT:
                lines.append(req)
    return sorted(set(lines), key=str.lower)


if __name__ == "__main__":
    print("\n".join(requirements()))
