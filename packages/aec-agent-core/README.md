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
settings and key lookup, its own reference readers). The guards and
records, Word and Excel, and the figure helper move in next.
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
