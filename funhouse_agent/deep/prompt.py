"""System prompt for the deepagents (v5.0) port.

Reuses :func:`funhouse_agent.system_prompt.build_system_prompt` but strips the
text-based ReAct / ``<tool_call>`` protocol sections, exactly like
``agent._build_native_system_prompt`` does for the native-tool-calling path.
deepagents binds tools through LangChain's native tool-calling, so the
``## ReAct Protocol`` / ``## Available Tools`` / ``## Rules`` blocks (which
document the ``<tool_call>`` XML format) are noise that can mislead the model.

The domain guidance, DIGGS workflow, tool discipline, and the module catalog
are all preserved.
"""

import re

from funhouse_agent.system_prompt import build_system_prompt


# Capability nudges for the deepagents-native planning + filesystem features.
# These are ON by default in deepagents (TodoListMiddleware + FilesystemMiddleware),
# so the model just needs to be told to use them well. Kept terse on purpose.
_PLANNING_AND_SCRATCH_SECTION = """\
## Working Style

- **Plan multi-step jobs with `write_todos`.** When a request needs several
  analyses or a chain of methods (e.g. classify -> SPT correction -> bearing ->
  settlement), or asks you to run several methods and compare them, open a todo
  list first, then keep it updated — mark each item done as you finish it. Skip
  it for a single one-shot calculation.
- **Use the scratch filesystem to stay organized.** You have `write_file` /
  `read_file` / `edit_file` / `ls`. Stash intermediate results, large tool
  outputs (e.g. a full method dump or a long reference excerpt), and tables you
  will reuse, instead of re-deriving or re-quoting them. These scratch files
  live only for the current session.
- **The scratch filesystem is NOT the real disk.** `ls` / `read_file` / `glob` /
  `grep` see only your own scratch files — they always return empty/not-found
  for real paths (e.g. /tmp, /Workspace), even for files that exist. Never use
  them to verify a file written by an analysis tool (calc packages, DXF
  exports, saved plots). Trust the tool's own response instead: if it reports
  `file_exists: true` with a size and `output_path`, the file IS on disk at
  that path. If a save genuinely failed, the tool call itself returns an error
  — report that error; do not silently rebuild or claim success.
- **Reading a REAL file the user gives you (PDF report, DXF, image): use the
  file-reading TOOLS, not the scratch filesystem.** `read_file` failing on a
  real path does NOT mean the file is unreachable — `analyze_pdf_page`
  (vision-reads one PDF page), `analyze_image`, the `pdf_import` /
  `dxf_import` / `drawing_ir` agent methods, and geo_project ingest all open
  REAL paths directly (`/tmp/...`, `/Volumes/...`; plain path, never a
  `file:` URI). For a geotech report PDF: `analyze_pdf_page` page-by-page,
  starting with the table of contents, then the boring logs / lab-summary /
  recommendations pages. If a real path errors there too, ask the user to
  copy the file to driver-local `/tmp` or a `/Volumes` path (`/Workspace`
  reads are unreliable) — do NOT conclude the file cannot be read. Plain file
  writes to `/Workspace/...` are often not durably stored (the workspace
  keeps a literal PLACEHOLDER file; binary files like PDFs come out corrupt).
  Prefer `/tmp/...` or a Unity Catalog `/Volumes/...` path for `output_path`,
  and tell the user to copy it out with
  `dbutils.fs.cp('file:/tmp/<name>', ...)` or download it. If a tool response
  reports a `rescue_path`, the requested location failed — give the user the
  rescue path.
- **Theory names and qualifiers are not method names.** Names like
  vesic/meyerhof/hansen and qualifiers like ultimate/net/effective-area are
  `factor_method`/parameter values or output labels, not methods. Each module
  typically exposes ONE main analysis method (e.g. `bearing_capacity_analysis`);
  if you guess a method name and it gets redirected (a `_note` in the result),
  use the real method it points you to."""

_MEMORY_SECTION = """\
- **`/memories/` persists across sessions.** Files you write under `/memories/`
  survive after this conversation ends (everything else is wiped). Use it for
  durable project context — the agreed soil profile, design groundwater table,
  governing load cases, and signed-off design parameters. Do not put transient
  scratch work there.
- **Your durable memory file is `/memories/AGENTS.md`.** Its contents are loaded
  for you automatically at the start of every session, shown in the
  `<agent_memory>` block. So: **save durable project context by writing or
  updating `/memories/AGENTS.md`** (create it if absent; append/merge rather than
  overwrite good notes). Other `/memories/*` files persist too but are NOT
  auto-loaded — if `<agent_memory>` is empty yet you expect prior project context,
  run `ls /memories/` and `read_file` the relevant ones before asking the user to
  repeat themselves."""


def build_domain_prompt(allowed_agents=None, *, memory_enabled: bool = False) -> str:
    """Return the domain system prompt with the ReAct XML sections stripped.

    Parameters
    ----------
    allowed_agents : iterable of str, optional
        If provided, only these modules appear in the catalog (same scoping as
        ``build_system_prompt``). Defaults to the full registry.
    memory_enabled : bool, optional
        When ``True``, append a short note telling the agent that ``/memories/``
        persists across sessions (only meaningful when the agent is built with a
        store / ``enable_memory``). Defaults to ``False``.

    Returns
    -------
    str
        The system prompt for ``create_deep_agent``: domain guidance + DIGGS
        workflow + tool discipline + module catalog, with the
        ``## ReAct Protocol`` through ``## Rules`` sections removed, plus a
        concise note nudging the deepagents-native planning + scratch filesystem
        (and persistent ``/memories/`` when enabled).
    """
    base = build_system_prompt(allowed_agents)
    # Remove "## ReAct Protocol" through the start of "## Available Modules"
    # (drops Protocol + Available Tools + Rules). Mirrors
    # agent._build_native_system_prompt's regex.
    base = re.sub(
        r"## ReAct Protocol.*?(?=## Available Modules|\Z)",
        "",
        base,
        flags=re.DOTALL,
    )
    base = base.strip()

    section = _PLANNING_AND_SCRATCH_SECTION
    if memory_enabled:
        section = section + "\n" + _MEMORY_SECTION
    return base + "\n\n" + section


__all__ = ["build_domain_prompt"]
