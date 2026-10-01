# Field feedback — Document Review, penetration tags on a drawing set (2026-10-01, v5.31.0)

Session `ecbf949d61c84d81b98a2ee34d679038` on the Document Review page,
`funhouse-gpt-high` (GPT-5.4), 14 turns, all review switches OFF (so the
legacy agent: step cap 50, no sweep, no located items). It continued a
2-turn session from earlier the same day. The document is the private
85-sheet drawing set from the 5.29.0 report: lettering drawn as lines (no text
layer), small three-letter penetration tags in hexagons. The owner shared the
export and two screenshots, not the PDF. Raw export in `raw/` (gitignored —
the screenshots carry the document's own markings and stay there).

Read for this triage: `FEEDBACK.md` (3 owner notes, 7 agent notes), the
transcript, all 552 `activity.jsonl` records, both screenshots.

## The session

| Turn | Owner asked | What happened |
|---|---|---|
| 1–3 | Read the earlier conversation from SharePoint | The share link failed in `sharepoint_list_files`; the owner pasted the browser URL and the agent parsed the path itself |
| 4 | Find every penetration tag | Looked at 6 of 85 sheets (chosen from thumbnails and the earlier conversation); reported 3 tags, two as uncertain |
| 5 | Verify by zooming, then mark up | Zoomed narrow regions of the same sheets; "not confirmed"; a notes-only markup |
| 6–7 | (owner's screenshot of a sheet showing a tag) What went wrong? | The agent: "over-trusted text/search and under-used systematic visual enumeration" |
| 8 | Systematic vision review | Started with text searches for the tag codes (no text layer: no hits), then picked 17 candidate sheets from thumbnails and looked at each whole; found tags on 6 sheets |
| 9–10 | Zoomed confirmation, notes markup, a log | 6 zooms; notes placed — one box converted correctly, one ~50 pt off |
| 11–12 | Red circles with short labels | Boxes drawn at coordinates written from nothing: GCE at [430,250,500,320] where the zoom had put it at about [646,518,660,532] |
| 13–14 | (screenshot of the misplaced boxes) What went wrong? | The agent: "annotated from inferred locations, not freshly captured page coordinates" |

## F1 — Text tools ahead of looking; candidate sheets instead of every sheet **[HIGH]**

The owner (sidebar, turn 7): *"See its write up of over-trusting text/search.
This was my concern with promoting certain tools too high in the
instructions! Systematic visual enumeration is much more flexible; even if
it's more expensive, we don't care much here. The tools can be a fallback or
supporting evidence, but the LLM shouldn't use them in lieu of vision."*

Evidence: turn 4 looked at 6 of 85 sheets. Turn 8 opened with
`search_document` for the codes (exact and fuzzy) on sheets the page map
already called drawn-as-lines, then narrowed to 17 sheets from thumbnails;
the sheets it never opened were never looked at. The prompt's ordering is the
cause: "**Then text, then your eyes**" (`_REVIEW_HOW_YOU_READ`), with looking
framed as what to do when a text result carries a `! look:` cue. A search miss
became the candidate list.

Disposition: **5.32 (owner direction).** Rewrite "How you read" around
looking: a question about what pages SHOW is answered by looking at every page
in scope (whole page, then tiles and zooms); text tools are exact and cheap
supporting evidence and the way to read text pages, never a filter on which
pages get looked at; every answer states the pages examined and the pages
not. Same change in the reading helper. Measured on the suite (set-wide
count/locate tasks on the stroke-lettered set are the closest public proxy).
The coverage tool (`sweep_pages`, `GEOTECH_REVIEW_SWEEP`) and the lean agent's
40-call budget decide whether an 85-sheet sweep fits a turn — defaults come
from the Foundry M0 results.

## F2 — Markups drawn at invented coordinates **[HIGH]**

The owner (turn 14): *"Yeah definitely need to make sure it never draws
anything based on inferred locations; always use confirmed coordinates. Just
wayyyyy off with those boxes. Probably should also visually confirm its
markups."*

Evidence: the turn-9 zoom returned the tag's box on the 0-999 grid of the
render's `view` ([671,109,722,161] on view [467,487,733,773]) = page points
[645,518,659,533]. Turn 9's note used [646,518,660,532] (right); its second
note's x was ~50 pt off (the model's own arithmetic, mixing the requested
clip with the rendered view). Turn 12's circles used [430,250,500,320] and
[520,300,590,370] — neither from any tool result. Nothing checked them.

Disposition: **5.32.**
1. `annotate_document` takes a location the way `render_region` does — a
   vision result's `view` plus the item's `image_box` (0-999) — and converts it
   itself; and a located item's `page_bbox` as is. The model does no
   coordinate arithmetic.
2. `box` accepted for `bbox` (the agent sent `box` twice and lost a call each
   time).
3. Every box/circle/highlight placed is CHECKED: a crop around it is rendered
   from the marked copy and a vision call asked whether the mark encloses what
   its comment names; the result lists each markup as confirmed / misplaced,
   and the tool description says to remove or redo a misplaced one before
   handing the file back.
4. A `circle` kind (red, as a reviewer marks a drawing) with an optional short
   visible label — planlens `markup_writer` (planlens 0.11.0).
5. Prompt and tool description: never place a markup at a location that did
   not come from a tool result for that page.
6. Suite task: mark every instance of a labelled item on a public sheet; check
   each markup's box against the truth positions.

## F3 — Absence concluded from regions that were never checked **[MEDIUM]**

Turn 5 reported "could not confirm" for the whole set after zooming four
regions. The prompt already says never conclude absence from whole-sheet
views; it does not say the same for unchecked sheets or regions.

Disposition: **5.32**, part of F1's rewrite: a negative finding names exactly
what was examined.

## F4 — SharePoint links pasted from the browser **[MEDIUM]**

`sharepoint_list_files` was given a share link (`/:f:/r/sites/...`) and
returned "empty or missing folder"; the owner then pasted an `AllItems.aspx?id=`
URL and the agent decoded it by hand.

Disposition: **5.32.** The SharePoint tools turn both browser forms into a
library path before calling the file manager.

## F5 — Paste a screenshot into the chat box **[MEDIUM]** (owner, turn 5)

Disposition: **5.32.** `st.chat_input(accept_file=...)` takes pasted and
dropped images where browser uploads work (local, Tiny Apps), feature-detected
on the installed Streamlit; on Funhouse, where the driver proxy blocks the
upload request, the websocket uploader gains a paste target.

## F6 — The automatic orientation fired on screenshots mid-conversation **[LOW]**

Turns 11 and 26: the owner attached a screenshot as evidence and the app sent
the orientation request on their behalf ("Before I ask anything, give me a
short orientation…"); the agent described the screenshot instead of
answering the owner's point.

Disposition: **5.32.** No automatic orientation when everything uploaded is an
image and the conversation already has turns; the upload note still tells the
agent the file is there.

## Noted, no change

- **Page numbers.** One slip (turn 7 cited "PDF page 16" for tool page 16);
  later turns cited `pdf_page` correctly. The 5.30 fix holds; the suite's
  citation checks watch it.
- **The PDF re-downloaded each turn** (4 times). Harmless; the download cache
  is keyed by name and the agent varied `save_as`.
- **"…" in auto-titled folder names** made the folder awkward to name in a
  tool call. Changing the naming would rename every mirrored folder; not
  worth it.
