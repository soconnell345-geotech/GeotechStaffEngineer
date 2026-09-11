# Field feedback — general app usage (2026-09-11, v5.14.0)

Owner's notes, given in chat (no export zip — no `raw/`). Six items from
recent real sessions on 5.13/5.14. Owner instructions for the train: app
only (planlens items noted, not acted on), take our time, sequential work in
the main session. Plan of record:
`~/.claude/plans/idempotent-spinning-firefly.md`. All work is on master,
UNRELEASED — candidate **5.15.0**, owner-gated.

## Item 1 — Feedback files from user input and from the agent's own capability gaps

> Give app instructions and tools to create feedback files based on user
> input or its own instances of lacking capabilities. Feedback files would
> get saved in the specific conversation folder. If possible, a button or
> text box in the left sidebar.

**Before:** nothing — no button, no writer, no tool (grep-confirmed).

**FIXED `0dbec49`.** `webapp/feedback.py` writes `feedback.jsonl` (machine)
+ `FEEDBACK.md` (readable) into the conversation directory, so the mirror
carries both to SharePoint with no registration. Sidebar **Feedback**
expander (form, clears on submit, mirrors immediately). Agent tool
`record_feedback(kind, summary, details)` — built per conversation with the
record dir closed over, injected via `extra_tools` for the primary AND the
new `calc_extra_tools` for the calc sub-agent (where "no tool draws this" is
discovered). Prompt: call it once for user comments on the app, capability
gaps, or unrecoverable tool errors; it never replaces the answer. Intake
README updated: read `FEEDBACK.md` first when triaging a drop.

## Item 2 — Tool calls for all agents and agent-to-agent conversations in the archive

> Ensure tool calls for all agents and agent-to-agent convos are tracked in
> the SharePoint archive of the conversations, regardless of whether the
> "show turn details" box is ticked.

**Before (verified):** it was NOT. `turn_jobs.py` discarded the
`tool_call`/`tool_result` stream items; with the box ticked, `trace.jsonl`
kept an 80-character one-liner per PRIMARY tool call; sub-agent (calc /
references) internals appeared nowhere because `core.stream_turn` streams
without `subgraphs=True`. `core.py`'s docstring claiming "incl. sub-agent
hops" was wrong.

**FIXED `af76521`.** `webapp/activity_log.py` — a LangChain callback handler
(callbacks propagate into sub-agents; that is already how `turn_tokens`
sums them) writing `<conversation>/activity.jsonl`, ALWAYS on: every tool
call with full args, every tool result (capped 32 KB, cut marked), every
model call's usage and tool-call count, a `turn_start`/`turn_end` envelope,
and `agent` attribution by `task` nesting (no deepagents internals read).
"Show turn details" stays a display toggle for the per-turn summary. A
REAL-stack test (`build_deep_agent` + scripted model delegating to `calc`)
proves the sub-agent's own `list_agents` call is logged and attributed to
`calc` on the installed deepagents/langgraph.

## Item 3 — Calc packages lack figures

> The app agent's calc packages tend to lack figures (sketches of designs,
> subsurface profiles, plots of data, etc.). How can we encourage it to
> create more figures? And is it limited by the available tools?

**Both**, and the first half was invisible from the prompt file that
shipped the profile-figure rule in 5.12:

* **Prompt routing (the bigger cause).** The figure doctrine lived only in
  the PRIMARY prompt; the web app defaults `route_calc: True`, the
  delegation nudge sends "building a calc package" to the `calc` sub-agent,
  and deepagents builds sub-agents standalone — so the agent that built
  every package had never seen the rules. Worse, its prompt began with
  `reviewer.CONSULTANT_FRAMING`, the references consultant's LIBRARIAN
  preamble ("Do NOT perform engineering calculations — reference lookup
  only … Question: "). **FIXED `6150236`**: own `_CALC_PREAMBLE`, a
  `_CALC_FIGURES` block (profile when layers exist → the analysis's own
  figures, prefer canned → data plots; figures-first skeleton; record the
  gap when a figure cannot be made), the primary bullet no longer promises
  a phantom "plotting tool", and the delegation nudge tells the primary to
  pass the layer stack / geometry / depth-wise data the figures need.
* **Tool inventory.** Only the 1-D profile column existed as a figure tool;
  ~25 ready matplotlib figures were reachable only inside a whole canned
  package; no code-execution tool exists to improvise. **FIXED `eaced20`**:
  `calc_package.render_figures(package=…)` (a package's figures as
  standalone PNGs, no package written; figures-only branch of
  `_build_response`, so all 15 generators get it) and
  `profile_figure.plot_data` (generic x/y or depth data plot → PNG). Both
  verified to embed via `html_to_pdf`. Catalog 7,965/8,000.

**Owner ask recorded as the NEXT train:** Ensoft-style tabular output of the
calculation data (per-depth / per-slice / per-layer tables, iteration
histories) saved with the report — `calc_package.export_tables` as a sibling
of `render_figures`, `get_tables()` per module. PLANNED (HANDOFF §0a).

Follow-ups PLANNED: specialist agents (foundations / earth-retention /
slope-fem / seismic) scope out `calc_package` AND `profile_figure` and so
cannot produce a report or figure at all; Plotly→PNG twin for
`subsurface.plot_*` (blocked: static export broken in the tenant).

## Item 4 — Two "Professional-use disclaimer" banners

**FIXED `8bde466`.** The solid banner was `st.warning(disclaimer_text()
.splitlines()[0])` — the notice's TITLE line, zero content. Removed; the
expander holds the text. AppTest guard verified to fail against the old
header.

## Item 5 — Larger uploads

> Where do we stand? Are we only limited by Streamlit's maximum? …
> a sharepoint file upload widget buried within but not relying on
> streamlit websocket stuff?

**Answer:** the 25 MB cap is OURS (`ws_upload.MAX_FILE_MB`), not
Streamlit's (200 MB defaults, untouched). The file rides ONE base64
websocket message that must cross the driver proxy inside a single socket
lifetime; the native uploader's PUT is 403'd by the proxy so
`server.maxUploadSize` never applies. Browser→SharePoint direct is WON'T:
the delegated Graph token would have to live in page JS. The route that
works is the inverse — upload with SharePoint's own uploader, the agent
fetches (`sharepoint_download_file`).

**Owner decision: document only. DONE `7681352`** — Attachments caption,
README "Large files", guide bullet, HANDOFF backlog order: (a) sidebar
"Attach from SharePoint" box (no LLM turn, no cap), then (b) chunked ws
upload. Also fixed: `test_ws_upload.py` sandboxed with a nonexistent env
var and wrote into the real `~/.geotech_webapp`.

## Item 6 — Old conversations gone after the notebook clears

**Cause:** conversations live on the driver's local disk
(`~/.geotech_webapp`; the launcher never redirects it); a cluster restart /
30-min idle termination wipes it; the mirror was upload-only.

**FIXED `7ff13cf`.** `sharepoint_store.list_remote_conversations()` +
`restore_conversation(folder)` (record AND `files/` — owner's choice;
manifest written so re-mirror uploads nothing; refuses to clobber; handles
`MOVED.txt`). Sidebar: "Find a past conversation" under Permanent storage
(search / Refresh / Restore → opens), a filter box over the local list
(searches past the newest-50 cap), and a token-expiry hint — the refresher
lives in the launching notebook and dies when it clears; re-run
`stage_sharepoint(...)`.

## Disposition summary

| # | Item | Disposition |
|---|------|-------------|
| 1 | Feedback files (user + agent) | FIXED `0dbec49` |
| 2 | Full activity archive, sub-agents included | FIXED `af76521` |
| 3 | Calc packages lack figures | FIXED `6150236` (prompt) + `eaced20` (tools); tables = next train |
| 4 | Duplicate disclaimer banner | FIXED `8bde466` |
| 5 | Larger uploads | DOCUMENTED `7681352` (owner decision); widget FUTURE, chunking BACKLOG, browser→SP WON'T |
| 6 | Old conversations after restart | FIXED `7ff13cf` |

Suites run: webapp 308 passed; deep 284 passed / 1 skipped; calc_package +
profile_figure + funhouse_agent/tests 1552 passed / 7 skipped. Live on the
cluster only after 5.15.0 (owner word). First live check: a calc-package
question — figures in the PDF, `activity.jsonl` and `FEEDBACK.md` in the
SharePoint folder, and a pre-restart conversation restored from the sidebar.

## planlens — NOTED, not acted on (owner: planlens burns usage)

- `planlens/pdf/vision.py:222 _render_pdf_page` — no output-size cap (Nairobi F5).
- `planlens/ocr.py augment_ir_with_ocr` — dpi=300 with no megapixel clamp (Nairobi F7).
- Bound the raster extra `opencv-python-headless>=4.8,<6` (HANDOFF 4b).
- planlens findings 5/6 (60°+ triangles, concave dart).
