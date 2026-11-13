"""Draw the three plot_code examples through the plotting box, into this
folder as ``code_<name>.png`` (GM-315's pairs). Run from the repo root:
``python module_work/gm315/render_code_examples.py``."""
import os
import shutil
import sys
import tempfile

import funhouse_agent  # noqa: F401  (registers the app with the core)
from aec_agent_core import plot_box
from aec_agent_core.charts.figure_examples import (code_example_names,
                                                   example)

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    bad = 0
    for name in code_example_names():
        ex = example(name)
        work = tempfile.mkdtemp()
        out = plot_box.run_plot_code(ex["code"], data=ex["data"], name=name,
                                     root=work)
        figs = [f for f in os.listdir(work) if f.endswith(".png")]
        print(name, out.get("status"), figs, out.get("layout_check"))
        if not figs:
            bad += 1
            print(out)
            continue
        shutil.copy(os.path.join(work, figs[0]),
                    os.path.join(HERE, f"code_{name}.png"))
    return bad


if __name__ == "__main__":
    sys.exit(main())
