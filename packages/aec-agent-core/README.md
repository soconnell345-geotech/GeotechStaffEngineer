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

**1.0.0.dev0: an empty package.** It exists so that its build, its tests and
the applications' pins on it can be checked before any code moves in. The
only thing in it today is `aec_agent_core.install_aliases`, which lets a
module keep answering to its old import path after it moves here, as the
same module object (so an import or a test patch by either name reaches the
one copy of the code).

It is not yet published; GeotechStaffEngineer installs it from its own
repository (`pip install -e packages/aec-agent-core`) until it is.

## Install

```
pip install aec-agent-core              # once published
pip install "aec-agent-core[anthropic]" # with Claude through LangChain
```

## Licence

MIT, as GeotechStaffEngineer. See `LICENSE`.
