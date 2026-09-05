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

## Disposition summary — ALL SIX FIXED (owner ordered execution
## 2026-09-05; landed same night, all on master, UNRELEASED until 5.12)

| # | Item | Disposition |
|---|------|-------------|
| 1 | CGPR #56 multi-method downdrag | FIXED c5d6299+0f9f4ad — Endo/Poulos/Fellenius-CGPR/PILENEG + groups + method_comparison; §3.4 worked example fully reproduced; methods-only (no digitization) |
| 2 | Lost rename/delete icons | FIXED d886638 — streamlit 1.62+ truncates narrow column labels; glyphs moved to the icon= slot (screenshot-verified on 1.59 AND 1.63); floor bumped to >=1.39 (d833c7b) |
| 3 | Profile figure capability | FIXED 18a2300+d2aecc0+3de0965 — profile_figure module (33rd) + calc-package default nudge + html_to_pdf auto-embeds local images / refuses loudly; bonus: _fileio binary-verify CRLF bug fixed |
| 4 | Attachment awareness | FIXED d96e712 — save_fn saved_note hook; tool result says the file is attached in chat |
| 5 | False "sources unavailable" | FIXED a310d5a — calc sub-agent gains list_files+read_pdf_text + never-claim-unavailable-without-checking rule |
| 6 | SharePoint folder naming | FIXED 0548f70 — <sanitized title>_<YYYY-MM-DD>, thread-id fallback, re-mirror+MOVED.txt on rename, shard suffix on collisions |

All live on the cluster only after the next release (5.12.0, held for
drawing-vision Phase 3.1 per owner).
