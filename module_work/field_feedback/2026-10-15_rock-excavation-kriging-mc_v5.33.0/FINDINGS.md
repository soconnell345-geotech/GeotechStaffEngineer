# Field session 2026-10-15: kriging and Monte Carlo of rock under a footprint (app 5.33.0, Funhouse, Prompter key)

**The session.** The same conversation as
`../2026-10-14_report-to-diggs_v5.33.0/` (turns 1–3 built a DIGGS file from a
two-volume geotechnical report), continued on 2026-10-15 for turns 4–11 on the
geotech page: GPT-5.6 Sol through the Prompter key, references off, recursion
limit 200. The owner dropped the export (the report itself removed to fit the
zip; it is in the earlier folder's `raw/files/`).

**What was asked, in turn.**
1. Kriging of the top of rock under a building footprint, with the excavation
   bottom one metre below the floor level, and sensitivity runs on the kriging
   inputs; re-check one historic boring's location.
2. "You need to consider joint-field probability, not just point wise... a
   monte carlo analysis": was that done? (No: deterministic kriging only.)
3. Use the observations from the whole site.
4. What matters is the VOLUME of rock excavation: a probability of exceedance
   for excavation quantities.
5. The owner pasted an old notebook using GSTools' `CondSRF` ("Do you not have
   this capability?").
6. A zero nugget had been a deliberate low-ball (the contractor's view).
7. Set the work up as a flexible build (notebook, workbook or data files);
   take the bottom of borings that refused after decomposed rock as top of
   hard rock.
8. Which borings count; boring and rock depth in each point's hover; the
   building lines and title were hard to read (screenshot).

**What came back.** A deterministic kriging report (footprint fractions from
43 nodes); then, three times, "the wrapper cannot do a conditional
simulation" (recorded by the agent as capability gaps); after the notebook, a
"flexible build" (inputs workbook, a Python engine script, a notebook, a
README, a report) that the agent could not run, so no exceedance numbers; a
plan of the footprint and borings rebuilt twice. Then the app stopped with
`StreamlitDuplicateElementKey` and the conversation could not be answered
again.

**Raw material** (gitignored `raw/`, never committed; it carries the project's
names): `raw/conversation/` (the app's export: `activity.jsonl` 5,429
records, `messages.json`, `transcript.jsonl`, `trace.jsonl`,
`feedback.jsonl` / `FEEDBACK.md`, `page_readings.json`, `reads.json`,
`files/`), `raw/export.zip` as received, `raw/notes/` (a readable dump of the
messages and the trace review's private tables).

## The owner's notes and the agent's own feedback

- **Owner:** "Annoying how the plots it's given aren't 1:1 in the axes. Can we
  encourage it to do that with plan views?"
- **Owner:** "It really doesn't fare so well when it comes to good visuals. I
  wonder why? Claude Code tends to be so good with visuals, what's going wrong
  here? Should we have a visualization agent that gets 'The Visual Display of
  Quantitative Information' by Tufte as an input?"
- **Owner:** the crash: "This error killed the conversation; couldn't reply
  after."
- **Agent (record_feedback, 8 entries):** no way to mask or summarise a kriged
  grid (the 60 x 120 result was offloaded to scratch); no conditional
  simulation, inequality data or polygon integration in `gstools`; scalar-only
  Monte Carlo in `reliability`; then "Previous response incorrectly stated
  conditional spatial simulation was unavailable" (true of GSTools, not of our
  wrapper); `plot_data` without polygon fill, marker shapes, point labels or
  hover; no way to turn a hand-written Plotly file into a PNG.

## First read (lead, 2026-10-15) and dispositions

1. **The crash.** The README was written in one turn and rewritten in the
   next; both turns carry a card for it, and the card's download key was the
   path alone. FIXED, TA-29 (keys carry the transcript entry and position;
   tested against the old code).
2. **Chart text unreadable in the chat.** Streamlit's theme recoloured chart
   text for the dark page; on the chart's light paper the title and axis
   labels vanished. FIXED, TA-29 (charts drawn in their own colours).
3. **Plans not to scale; no fill, names, marker shapes or hover; cases drawn
   as a line.** FIXED, GM-90 (`plot_data`: `equal_aspect`, `polygon`,
   `labels`, `hover`, marker options, named cases, axis limits, legend
   position; the tool's description says when to use each).
4. **No joint-field Monte Carlo.** The agent was right about the wrapper:
   `gstools_agent` offered deterministic kriging and an unconditioned random
   field only, and kriging returned its whole grid as JSON. DONE, GM-91
   (`conditional_simulation` by conditioning by kriging, kriging region
   totals and grids to files; follow-ups GM-96). GSTools' own `CondSRF`,
   which the owner's notebook used, gets the mean right but overstates the
   spread of a total (checked against an exact case: 787 against 521).
5. **The owner's question on figures.** Answered on the before-and-after page
   (https://claude.ai/artifact/TcPujcvMF19HZQXtPVAmMj): the app's agent cannot
   run code and had one plotting tool that drew lines and dots; the chat's
   theme hid its text; nothing told it the conventions. The owner's choice of
   next step: OWN-61, GM-92.
6. **The full read** of every record: `TRACE_REVIEW.md` (Opus subagent; private
   tables and the script run in `raw/notes/`). In short: the wrapper really
   lacked a conditional simulation and the agent was right to say so, but its
   turn-8 "I was wrong" was itself wrong about this app; kriging's description
   hid the grid it returned, so the moved result went unread; the turn-4
   fractions were counted by eye (10 of 40 wrong); the engine script runs
   locally and its answer turns on one boring inside the footprint (counted as
   hard rock: about zero volume; left out: a mean of 217 m³, P90 603 m³), which
   turn 11 counted against its own earlier reading without saying so; the
   helper that drew the figures had no way to look at them, and the agent
   checked a PNG of a different figure from the chart the owner saw.
   Dispositions: GM-91 (done), GM-96, GM-94, GM-95 (done), TA-30–TA-32, VIS-27,
   EV-19; the crash, the chart theme and plot_data were already fixed
   (TA-29, GM-90).
