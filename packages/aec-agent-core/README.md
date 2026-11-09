# aec-agent-core

The shared core of LLM agents that review and produce architecture,
engineering and construction (AEC) documents. It holds what an agent runs
below its chat page: the model connections, the vision calls and the file
tools beside them, the figure helper with its plotting box, the agent's
guards, limits and kept work, Word and Excel in and out, the run record and
its checks, and the switches. It has no user interface of its own.

Two applications are built on it:

- **GeotechStaffEngineer** (`pip install geotech-staff-engineer`), the
  geotechnical engineering agent, which pins this package exactly;
- a document review application.

## Status

**1.0.0.dev1: under construction.** It holds the model connections
(`engine_config` and its chat models, `deep.vision_engine`) and the vision
calls (`vision_tools`, `vision_view`, `image_view`, `vision_probe`), with
`hooks`, where an application hands in what the core must not import (its
settings and key lookup, its own reference readers). It also holds the
figure helper (`deep.figure_helper`), the plotting box (`plot_box`, which
runs the model's plotting code in a locked child process) and the chart
stack (`charts`: `plot_data`, the layout check, the scale bar and north
arrow, the worked figure examples). An application with standard chart
lines hands them in with `charts.register_reference_charts`. The guards
and records, and Word and Excel, move in next. Until they do, drawing
(`plot_data`, the plotting box, the worked examples) works with this
package alone, but building the figure helper's tools needs
GeotechStaffEngineer installed.
`aec_agent_core.install_aliases` lets a moved module keep answering to its
old import path, as the same module object.

PyPI holds only 1.0.0.dev0, an empty package that reserved the name;
GeotechStaffEngineer installs this one from its own repository
(`pip install -e packages/aec-agent-core`) until it is published.

## Install

```
pip install aec-agent-core              # once published
pip install "aec-agent-core[anthropic]" # with Claude through LangChain
```

## Licence

MIT, as GeotechStaffEngineer. See `LICENSE`.
