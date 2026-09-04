# Feedback: Praia_Downdrag session (2026-09-04, v5.11.2)

Session: "Praia_Downdrag", 7 turns, funhouse-gpt-high, full agent,
route_calc=true, recursion_limit=200, on the Funhouse cluster the same
night as the 5.11.2 shakedown. Owner's notes: "downdrag convo
feedback.docx" (in raw/, not committed). Task: independent downdrag
estimate for an MCAC micropile from an uploaded profile PDF, using
CGPR #56 as reference; then a multi-method calc package with a
subsurface-profile figure.

## Item 1 — "Do we have a module for all the CGPR #56 downdrag methods?"

**Answer: no.** `downdrag.py` exposes exactly ONE method
(`downdrag_analysis`, Fellenius neutral plane). The agent coped by
running sensitivity bounds on that one method — reasonable, but not
what was asked. **PLANNED (two parts):**
(a) Extend the downdrag module with the additional CGPR #56 method
    family (neutral-plane variants, drag-load vs downdrag distinction
    per Fellenius unified, the simplified/AASHTO-style checks the
    report catalogs). Implement METHODS from the equations with proper
    citations; validate against the report's worked examples.
(b) Reference-layer evaluation of CGPR #56 itself: it is a Virginia
    Tech CGPR report — CHECK DISTRIBUTION TERMS before any
    digitization (CGPR reports are typically member-distributed, NOT
    public domain; equations/procedures are fair to implement with
    citation, wholesale text/figure digitization likely is not).
The source PDF is in raw/ for the evaluation.

## Item 2 — Conversations tab lost rename/delete icons

Hover tooltips still show the right names, so the buttons render but
their icon glyphs don't. **Almost certainly the cluster's new
streamlit 1.63.0** (the resolver now picks it; uvicorn-based, much
newer than our tested line — flagged the same night for the launcher).
**PLANNED:** repro locally under streamlit 1.63, fix the icon spec
(material-icon syntax/emoji rendering changed across the major line),
and decide whether to cap `streamlit<` a known-good version in the
next release or adapt. Quick win once reproduced.

## Item 3 — No real subsurface-profile figure; "[image]" placeholder;
## table masquerading as a figure

Transcript evidence: asked to "add a figure showing the pile and a
subsurface profile", the model produced (a) a PDF with an "[image]"
placeholder, then (b) an HTML **table with colored rows** presented as
the "profile figure" (raw/files/mcac_micropile_profile_figure.html).
**Root cause: capability gap, not (only) model dodge — there is NO
agent-facing tool that draws a subsurface profile schematic.** Module
figures exist (slope interslice plots etc.) but nothing generic. The
html_to_pdf contract explicitly requires base64 PNG/JPEG and rejects
inline SVG, which the model also tripped over.
**PLANNED (the big one from this session):**
(a) New adapter tool: `profile_figure` (working name) — matplotlib
    schematic of layered stratigraphy + water table + optional
    pile/wall/footing overlay + annotations, returning a PNG (file +
    base64) sized for calc packages. Generic, parameter-driven, usable
    by every module family.
(b) Prompt guidance: calc packages SHOULD include a subsurface
    visualization by default when layer data exists ("I would always
    want simple subsurface visualizations in calc packages" — owner).
(c) html_to_pdf: fail LOUDLY (error listing offending elements) when
    the HTML references images/SVG it cannot embed, instead of
    silently rendering placeholders.

## Item 4 — "Doesn't realize it uploaded the report in the chat"

Turns 7/9/11: the model wrote files into the conversation files
directory (which IS the chat-attachment surface — they appeared in the
sidebar and in this very export) while simultaneously claiming "I
can't directly attach the binary PDF into the chat stream."
**Root cause: model awareness, not capability.** **PLANNED:** prompt
note + tool-result feedback: when a file lands in the conversation
files dir, the tool result should say "this file is now attached in
the chat sidebar" so the model stops disclaiming a capability it just
used. Cheap fix.

## Item 5 — Report claims "source PDFs weren't available"

The turn-3 analysis clearly extracted numbers from the MCAC PDF, yet
the built report carries a sources-unavailable caveat. **Leading
hypothesis: calc-package sub-agent isolation** (route_calc=true) — the
isolated calc sub-agent doesn't inherit the working folder / uploaded
files, so the REPORT builder genuinely couldn't see the PDFs even
though the main loop had them. **PLANNED:** verify against the trace,
then either pass a working-folder file inventory into the calc
sub-agent's context or have the main agent forward the extracted
source data explicitly. Until fixed, reports may carry false
provenance caveats — worse than missing data, it undermines trust.

## Item 6 — SharePoint mirror folder naming

Request: name the mirrored conversation folder after the custom
sidebar conversation name (+ date stamp for uniqueness) instead of the
thread-id hex. **PLANNED:** rename-on-custom-name in the SharePoint
mirror (sanitize for path safety, `<custom-name>_YYYY-MM-DD`, fall
back to thread id when unnamed; handle the rename event when a user
renames later). Straightforward webapp feature.

## Disposition summary

| # | Item | Disposition |
|---|------|-------------|
| 1 | CGPR #56 multi-method downdrag | PLANNED — module extension + license-gated reference eval |
| 2 | Lost rename/delete icons | PLANNED — streamlit 1.63 repro + fix (quick win) |
| 3 | Profile figure capability | PLANNED — new profile_figure tool + prompt default + loud html_to_pdf |
| 4 | Attachment awareness | PLANNED — prompt/tool-result note (cheap) |
| 5 | False "sources unavailable" | PLANNED — calc sub-agent working-folder visibility (trust issue, prioritize) |
| 6 | SharePoint folder naming | PLANNED — mirror rename feature |

All six queued behind the current planlens/Phase-3 push per the
owner's "heavy development period → make plans" instruction; items 2
and 4 are small enough to ride along with the next webapp touch.
