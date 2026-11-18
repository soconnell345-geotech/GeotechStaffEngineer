"""What this host's kernel offers the plotting box's kernel layer.

One command, for a host's administrator::

    python -m aec_agent_core.plot_box.kernel_probe

It prints ONE line of JSON: the kernel's release and machine, the Landlock
version (or why Landlock is not there: not built in, or not enabled at
boot), and whether no-new-privileges, the Landlock file, TCP and scope
rules and a seccomp filter can be put up.

It changes nothing. The Landlock version is a number the kernel is asked
for. The restrictions are tried in a forked child that does nothing else
and exits; the process that ran the command is not restricted
(``caller_restricted`` says so), and no file is read or written but an empty
temporary folder that is removed again.

``--paths`` prints instead, one JSON line each, the paths the file layer
would allow for this interpreter with a stand-in conversation folder (the
current directory): ``[path, "read" | "write" | "scratch", why]``. It
restricts nothing either.

The layer itself: :mod:`aec_agent_core.plot_box.kernel_layer`.
"""

from __future__ import annotations

import json
import os
import sys

from aec_agent_core.plot_box import kernel_layer


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--paths" in argv:
        for row in kernel_layer.box_paths(os.getcwd()):
            print(json.dumps(row))
        return 0
    print(json.dumps(kernel_layer.probe(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
