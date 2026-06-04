# module_work — geotech module-improvement team

Durable state for the standing module-improvement agent team. Claude Code
teammates are ephemeral (fresh context each spawn), so the team's **identity and
memory live here as files**, version-controlled with the repo.

## Files

- `BOARD.md` — top-level backlog and status across all domains. The team lead
  maintains this: triages feedback into tasks, tracks what's in flight, what's
  merged.
- `<domain>.md` — one **progress ledger** per domain specialist. Holds the
  domain's owned modules, its reference map (where to look), its backlog, and a
  running log of what's been done. A specialist **reads its ledger at the start
  of a task and updates it at the end** — this is its persistent memory.

## Roles

- **Team lead** = the main Claude Code session (playbook:
  `.claude/agents/geotech-team-lead.md`). Triages, dispatches specialists,
  reviews every diff, gates on tests, serializes `geotech_common` changes.
- **Domain specialists** = teammates spawned with `subagent_type:
  geotech-module-specialist`, named by domain (e.g. `seismic`), each pointed at
  its `module_work/<domain>.md` ledger.

## Loop (every specialist task)

1. Read your ledger `module_work/<domain>.md` (owned modules, references, backlog).
2. For each task: read the module's `DESIGN.md`, source, `tests/`, and
   `funhouse_agent/adapters/<module>.py`.
3. Make the change (SI units, existing style; add `allowed_values` to METHOD_INFO
   enums per the established convention).
4. Verify: `.venv/Scripts/python.exe -m pytest <module>/ -v` (plus
   `funhouse_agent/tests/` if you touched an adapter).
5. Update your ledger (what changed, test result, open items).
6. Return a summary + `git diff`. **Do not commit** — the lead reviews and merges.
