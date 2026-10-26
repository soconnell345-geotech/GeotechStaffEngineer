"""Print the third-party requirements of the installed app, planlens and
geotech-references, for an environment where numpy is held at 1.26
(BACKLOG INF-53).

Install the three packages with ``pip install --no-deps`` first, then:

    python constraints/foundry_requirements.py > requirements.txt
    pip install -r requirements.txt -c constraints/foundry-numpy126.txt

Left out: the three packages themselves, numpy (the environment's own
1.26.4 stays; planlens 0.13 declares numpy >= 2.0, its next release
>= 1.26) and PyNiteFEA (no release the app accepts installs on numpy 1.26;
see foundry-numpy126.txt).
"""

import importlib.metadata as md
import re

OWN = ("geotech-staff-engineer", "planlens", "geotech-references")
LEFT_OUT = set(OWN) | {"numpy", "pynitefea"}


def requirements() -> list:
    """The core requirement lines of the three installed packages."""
    lines = []
    for dist in OWN:
        for req in md.requires(dist) or ():
            if "extra ==" in req:
                continue
            name = re.split(r"[\s<>=!~;\[(]", req, maxsplit=1)[0]
            if name.lower().replace("_", "-") not in LEFT_OUT:
                lines.append(req)
    return sorted(set(lines), key=str.lower)


if __name__ == "__main__":
    print("\n".join(requirements()))
