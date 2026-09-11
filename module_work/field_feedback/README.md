# Field feedback intake

Owner-supplied feedback from REAL app sessions (SharePoint conversation
exports + the owner's written notes). One directory per feedback drop:

    module_work/field_feedback/YYYY-MM-DD_<slug>_v<app-version>/
        FINDINGS.md   <- committed: the owner's feedback items, the lead's
                         diagnosis (with transcript evidence), and the
                         disposition/plan for each item
        raw/          <- NOT committed (gitignored): the original zip
                         contents — conversation transcript, project PDFs,
                         generated reports. Real project data stays out of
                         git; the lead reads it locally during triage.

Conventions:
- Date = the conversation date (from the export's meta.json), not the
  triage date. Version = the geotech-staff-engineer release the session
  ran (check meta + the session's install cell).
- Every feedback item gets a disposition in FINDINGS.md: FIXED (commit),
  PLANNED (HANDOFF/backlog pointer), or WON'T/BLOCKED (reason).
- During heavy development the owner expects PLANS, not immediate fixes
  (their words, 2026-09-04) — but trivial fixes may land immediately.
- Cross-reference: durable backlog items also go in HANDOFF.md §0a so
  they survive session handoffs.
- Since 5.15.0 a conversation export carries `FEEDBACK.md` / `feedback.jsonl`
  (`webapp/feedback.py`): the owner's sidebar notes (`source: user`) and the
  agent's own `record_feedback` calls (`source: agent` — capability gaps,
  tool errors, feedback given in chat). Read those FIRST when triaging a
  drop; they name the gap in the agent's words, with the turn index.
