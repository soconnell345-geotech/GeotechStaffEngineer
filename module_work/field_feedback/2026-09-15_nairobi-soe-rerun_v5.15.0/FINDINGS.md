# Field feedback — Nairobi SOE review, re-run (2026-09-15, v5.15.0)

Session `b9efee4d3c4d4785a033fba4cdc08557`, model `funhouse-gpt-high`, 9 turns,
calc routing on, references `anytime` (ref budget 10), recursion limit 80.
Same 260-page submittal as the 2026-09-09 drop (temporary tangent-pile SOE with
walers and rakers; PYWall, LPILE and spColumn appendices). Raw export in `raw/`
(gitignored).

Read for this triage: `FEEDBACK.md` (7 owner notes, 6 agent notes), the
transcript, all 722 `activity.jsonl` records, the delivered PDF, and the owner's
screenshot. The agent's SOE calls were re-run locally on the current tree, and
the PYWall pages were checked in the source PDF.

## The session

| Turn | Owner asked | What happened | Time / tokens |
|---|---|---|---|
| 1 | Review the SOE package | A solid 12-comment review | 181 s / 582k |
| 2 | Re-run with Stratum C c = 0 | `soe.braced_excavation`: no change (sand apparent-pressure envelope) | 136 s / 407k |
| 3 | Compare to PYWall | Found PYWall's "USING PRESSURE DIAGRAM FOR SAND"; compared the trend | 135 s / 408k |
| 4 | c = 0 sensitivity, strength/surcharge sensitivity, calc package | Cases A–F, cantilever and basal-heave checks, PDF built with its figures missing | 225 s / 456k |
| 5 | Fix the PDF figures; plot PYWall pressures inline | PDF rebuilt: figures in, **numbers gone**; plot saved to `/tmp` and never shown | 186 s / 401k |
| 6–9 | Where is the PDF? → email it → save to SharePoint → no, the conversation folder | Emailed; uploaded to "uploaded references"; then to a doubled path | 7–14 s each |

**Headline.** Three wrong things reached the owner: the rebuilt calc package
lost every number (N1); the cantilever results come from a defective module
(N2); the anchor load and moment for cases A–F are partial quantities presented
as complete (N3). On top of that came the file-handling problems the owner hit
directly (N4–N9).

---

## N1 — The "fixed" PDF dropped all the numbers **[HIGH]**

`SOE_sensitivity_review_embedded.pdf` (5 pages): rows B–E of the sensitivity
table read "Sensitivity variation / Per prior analysis / See previously
developed table". Case F is labelled "c = 0 / conservative framing case", but
it was φ = 26°, q = 14.4 kPa. The report says "No new calculations were
introduced" and that "the detailed numeric sensitivity table remains as
previously developed in the conversation record". The turn-4 PDF had the real
table. The gutted version is the one that was emailed and uploaded (turns 7–9),
and the turn-5 answer said "Confidence: high".

How it happened:
1. The primary delegated "Rebuild the previously created SOE sensitivity review
   PDF ... Use the same analysis results already developed in this
   conversation" with no numbers and no path to the turn-4 HTML. A sub-agent has
   no memory of the conversation.
2. The rebuilding sub-agent called `read_file` on `/tmp/SOE_sensitivity_review.html`
   and `/tmp/SOE_sensitivity_review.pdf` and got "not found". Both files existed.
   `read_file` is deepagents' in-memory scratch tool and cannot see the real
   disk. The calc agent's real-disk tools are `list_files`, `read_pdf_text`
   and `save_file`, so it had no way to read an HTML or text file.
3. Instead of stopping, it wrote placeholder prose and rendered it.
4. The primary relayed "fixed, high confidence" without opening the output.

(`activity.jsonl` labels this work `general-purpose`; by the task descriptions
it was the `calc` sub-agent. See N15.)

**5.16:** not addressed.
**Disposition: PLANNED (app, first of the app items).**
- Calc prompt: never write placeholder text where results belong. If the
  numbers are missing, return and say exactly what is missing.
- Delegation nudge: a redo, rebuild or reformat must carry the numbers, or the
  real path of the earlier file.
- Give the calc sub-agent a real-disk text reader (HTML/TXT/CSV/JSON, working
  folder and `/tmp`).
- The primary re-reads a rebuilt deliverable (`read_pdf_text` on the output)
  before it says the rebuild worked.

## N2 — `soe.cantilever_excavation` is wrong, and unconservative **[HIGH, analysis module]**

Re-run locally with the session's inputs:
- **First layer only** (`soe/beam_analysis.py:298`, "simplified single-layer
  analysis"). The session profile's top layer is the φ = 0, c = 19 kPa clay, so
  Ka = Kp = 1.0 for the whole wall. Result: embedment 0.60 m, moment
  213.47 kN·m/m. Changing φ in layers 2–4 (33° → 29° → 26°) gives identical
  output, which the agent reported as "cantilever: not sensitive".
- **Passive moment arm.** Moments are taken about the wall base, but the
  passive resultant uses arm `H + D/3` (line 335) instead of `D/3`. Passive
  resistance is grossly overstated, so embedment comes out far too short.
- **Maximum moment** is taken at the dredge line with arm `H/3` (lines 349–350).
  A cantilever's maximum moment is below the dredge line.

Uniform sand, φ = 33°, H = 6.3 m, q = 7.2 kPa, FS on passive 1.5:

| | Embedment D | Max moment |
|---|---:|---:|
| `soe.cantilever_excavation` | 2.50 m (incl. +20 %) | 230.8 kN·m/m |
| Hand check, moments about the toe (dry) | 6.91 m (8.30 m with +20 %) | — |
| `sheet_pile.cantilever_wall` | 6.92 m | 602 kN·m/m at 10.1 m |

The module's tests only check signs and monotonicity
(`soe/tests/test_soe.py:441-490`); it was never pinned to a textbook value.

**Also suspect, same pattern:** `soe/embedment.py::compute_embedment`, which
feeds `braced_excavation`'s `required_embedment_m`. The active triangle below
the pivot uses arm d/3 (should be 2d/3) and ignores the overburden above the
pivot. The passive arm is h + D/3 (should be h + 2D/3). It uses one layer, the
one at excavation level. Not hand-checked yet; verify in the fix.

Reached the owner in the turn-4 table and calc package ("Cantilever embedment
0.60 m → 0.60 m, unchanged; max moment 213.47 kN·m/m").

**Disposition: PLANNED (first engineering fix).** Rebuild the cantilever on
layered net pressures (or route to `sheet_pile.cantilever_wall`, which is right).
Pin it to a published cantilever example and cross-check against `sheet_pile`.
Hand-check `compute_embedment` against a free-earth-support example and fix it
in the same pass.

## N3 — FHWA single-anchor results are partial but unlabelled **[HIGH]**

`soe/earth_pressure.py:521-531`: for ONE anchor level the function returns
only `TH_upper` (the load above the anchor), with `TH_kN_per_m` and
`design_load_kN` set to null. `max_moment` covers only the span above the
anchor (13/54·H1²), and `R` uses the multi-anchor formula. The docstring says
the single-anchor total needs the free-earth-support solve
(`sheet_pile.analyze_anchored`), but the result carries no note.

The calc agent labelled `TH_upper` "Anchor TH*". The primary plotted it as
"anchor tributary load", and the calc package named case F governing with
"TH = 64.0 kN/m, Mmax = 43.5 kN·m/m".

Scale: `braced_excavation` on the same wall gave a strut load of 169 kN/m and
a moment of 62 kN·m/m. PYWall's raker load of 353.2 kN (Table 5) comes to about
137 kN/m horizontal, if it is per raker at the 2.4 m spacing the agent modelled
(spacing and load factoring not checked). `TH_upper` is roughly a quarter to a
third of the real support load.

**Disposition: PLANNED.** For n = 1, either compute the total through the
free-earth-support solve, or return a `note` saying the result is partial and
drop `max_moment`/`R`. The adapter brief should say the same.

## N4 — Deliverables were written outside the conversation folder, so the owner never got them **[HIGH]**

Owner, turns 6–7: "Where is the PDF? you only sent me the html inline" /
"Saving things on the local session disk is basically useless ... we should make
it at least standard to save it to the sharepoint."

The agents wrote PNGs, PDFs and HTML to `/tmp`, and TXT/PNG to
`/Workspace/Users/<user>/geotech_app`. The app offers downloads, previews and
the SharePoint mirror only for files in the conversation's `files/` folder,
plus new files in the launch working directory (which is why the two
`/Workspace` TXTs showed up). `save_file` records its path whatever the
location, so the turn-5 **HTML source** got a card. `html_to_pdf` does not
go through it, so the **PDF** got none. Nothing in `/tmp` ever reached SharePoint.

**Disposition: PLANNED (app).** Copy any file a tool reports writing
(`output_path` / `saved`) into the conversation's `files/` folder and show it as
a card. It then downloads from the chat and mirrors to the conversation's
SharePoint folder automatically, which is the "save to SharePoint by default"
the owner asked for. Also point default output paths at `files/`, and say so
in the prompt. This fixes N6 as well.

## N5 — SharePoint upload went to the wrong folder, then to a doubled path **[MED]**

Turn 8's destination was `Shared Documents/General/GSE_app/uploaded references`.
Turn 9's was `General/GSE_app/conversations/b9efee4d…`, which resolved to
`Shared Documents/General/GSE_app/General/GSE_app/conversations/…`.
`sharepoint_tools._resolve` (lines 74–77) strips only ONE leading segment equal
to the root's last segment; `General/GSE_app/…` repeats two. The mirror also
names the folder `<title>_<date>` (`sharepoint_store.conversation_folder`),
which the agent cannot know, so it guessed the thread id.

**Disposition: PLANNED.** Default `sharepoint_upload_file` to this
conversation's mirror folder and name that folder in the result. Strip the
longest leading run of segments that repeats the root's tail. Prompt: "uploaded
references" is for user inputs only. Mostly moot once N4 lands.

## N6 — Inline plot did not render **[MED]** (owner screenshot)

The answer contained `![PYWall lateral earth pressures](/tmp/pywall_lateral_pressures.png)`.
The chat cannot display a local path; images render only from file cards
(`app.py::_render_artifact_card`). `profile_figure`'s result had told the agent
"Saved as a PNG: the chat UI renders it inline", which is true only inside the
conversation folder.

**Disposition: folded into N4.** Also correct that `embed_note` wording.

## N7 — `html_to_pdf` rejected `file:///tmp/…` images **[MED]** (agent note, turn 4) — code bug

`calc_package.py:1300-1301` strips `file:///` (8 characters), turning
`file:///tmp/x.png` into the RELATIVE `tmp/x.png`, which then resolves under the
conversation folder (`…/files/tmp/x.png`). That is correct for Windows
(`file:///C:/…`) and wrong on the Linux driver. It is why the first PDF shipped
without figures.

**Disposition: PLANNED, trivial.** Strip `file://` and keep the leading slash;
drop the slash only before a drive letter; percent-decode.

## N8 — "It gives the document it reviewed back to you" **[LOW]** (owner note, turn 2)

`sharepoint_download_file` saves into `files/`, and new files there become
cards, so the 23 MB submittal looked like something the agent produced. The
same file was downloaded four times under two names (turns 1, 2, 3, 5), about
92 MB of copies.

**Disposition: PLANNED.** Record downloads as inputs rather than outputs. The
download tool should reuse an existing copy.

## N9 — `analyze_pdf_page` would not take the downloaded file's name **[LOW]** (agent note, turn 1)

The download result says "the file is now in the working folder and available
to the file tools". `analyze_pdf_page` and `read_pdf_text`, however, accept
only an attachment key or a full path (`vision_tools.py:235-259`). Downloads
are not registered as keys. It happened again in turn 5 (4 calls).

**5.16:** not addressed. `document_tools._resolve` has the same rule.
**Disposition: PLANNED, trivial.** Resolve a bare filename against the working
folder in both resolvers, and have the download result print the full path.

## N10 — `soe` `soil_type: 'clay'` is documented but rejected **[LOW]** (agent note, turn 2)

The adapter brief (`adapters/soe.py:184`) says `'sand' or 'clay'`; the module
takes `sand` / `soft_clay` / `stiff_clay` (`geometry.py:111`). Four failed calls
across turns 2–4.

**Disposition: PLANNED, trivial.** Correct the brief and allowed values;
optionally map `clay` by stability number.

## N11 — Module name `SOE` rejected over capitalisation **[LOW]** (agent note, turn 4)

**Disposition: PLANNED, trivial.** Case-insensitive module lookup in `dispatch`.

## N12 — Scratch-filesystem tools used on real files about 35 times **[MED]**

`grep` ran 25+ times over the PDF folder and always returned "No matches found".
`glob` and `read_file` also ran on real files ("not found" for files that
exist). These are deepagents' in-memory scratch tools, and their answers look
real. This is the root of N1.

**5.16:** partly addressed. `search_document` gives the primary a real in-PDF
search; sub-agents still have only the scratch tools.
**Disposition: PLANNED.** Remove or re-describe the scratch read/search tools
on our agents, and give sub-agents real-disk read and search.

## N13 — Reviewer sub-agent cited a reference it never found **[MED]**

Turn 1: `gec7` `list_methods(category="earth retaining structures")` → `{}`;
`describe_method` → error; `search_sections` → `[]`; `dm7`
`list_methods(category="retaining walls")` → `{}`. The reviewer then cited
"FHWA GEC-12 (Design and Construction of Deep Excavations)" about ten times;
GEC-12 is *Driven Piles*. The primary's answer did not repeat it, so it did not
reach the owner. This is the standing hallucination-after-failed-lookup risk
(CLAUDE.md figure TODO P1). Separately, a category filter that matches nothing
returns `{}` without saying so.

**Disposition: PLANNED.** Reviewer prompt: a citation must come from a tool
result in this consult. `list_methods` with a category that matches nothing
returns the available categories. (Branch `feature/subagent-feedback-reference-pdfs`
gives the reviewer `record_feedback`.)

## N14 — Turn 3 changed the wrong layers, and the answer said otherwise **[MED, engineering]**

The primary asked to zero the cohesion of "the two Stratum C layers": the 2.5 m
and 2.0 m layers with φ = 33°, c = 50 kPa, identified in turn 2. The calc
sub-agent renamed the layers itself (the φ = 33° layers became "B1/B2", the
φ = 26° layers "C1/C2") and zeroed c in the φ = 26°, c = 9 kPa layers at 6.0–9.0 m.
The owner was told "the required embedment increased significantly when
cohesion was removed". That change (1.16 → 2.33 m) came from the layers below
the excavation, not Stratum C, and from the suspect embedment routine (N2).

**Disposition: PLANNED.** Calc prompt: keep layer names exactly as delegated,
and echo which layers were changed.

## N15 — `activity.jsonl` mislabels sub-agents that run in parallel **[LOW, triage tooling]**

Turn 5 ran `calc` and `general-purpose` at once. The logger attributes tool
calls through a stack of `task` calls (`activity_log.py` docstring), so
everything nested was labelled `general-purpose`, and the two task-end records
carry each other's names (the "calc" end record holds the PYWall table).

**Disposition: PLANNED, small.** Attribute through the `parent_run_id` chain.

## N16 — About 2.25 M tokens in five turns **[NOTE]**

Turns 1–5 used 581,679 / 406,767 / 407,982 / 456,264 / 401,230 tokens. The PDF
was re-read in roughly 14k-character slices on every turn and downloaded four
times. 5.16's document tools (handles that survive turns, a structure map,
search) target exactly this; measure it on the next live run.

## N17 — Engineering notes on the answers (no code change)

- **Good.** The turn-1 review raised the right themes: an apparent-pressure
  model for mixed residual soils, parameters that disagree between notes and
  model, groundwater and dewatering, bottom stability, the 15 mm criterion
  against LPILE's 16.35 mm, the raker-removal stage, surcharge control, and
  monitoring. Turn 3 found PYWall's own "USING PRESSURE DIAGRAM FOR SAND".
- **Missed a real match.** `soe.braced_excavation`'s apparent pressure of
  26.85 kPa reproduces PYWall's. PYWall Table 7 plateaus at 20.98 kN/m over a
  width for earth pressure WA = 0.78 m (PYWall wall data, printed page 29 of
  245), which is 26.9 kPa: within 0.2 %. The agent had both numbers and said
  the absolute values "are not expected to match".
- **Supports the review comment.** PYWall's own layer-by-layer active pressure
  for the two c = 50 kPa layers is 0.00 (printed page 30). With that cohesion a
  layer-by-layer analysis puts no active load on Stratum C; the sand envelope
  alone is what loads it. The c = 50 kPa matters only outside the envelope
  (below the dredge line, stability checks).
- **Wrong plot explanation.** The jump at 6.35 m is not "a deeper
  apparent-pressure block". The envelope covers only the excavated height; below
  the dredge line PYWall applies each layer's own active pressure plus water
  pressure (water table 4.0 m behind the wall, 7.5 m on the excavation side).
  A hand check at 6.35 m gives about 35 kN/m against the table's 36.7. The drop
  to 5.06 kN/m below 10 m is layer 7 (rock).
- **Basal heave check misapplied.** It used Stratum C's cu = 50 kPa, but
  Stratum C sits above the excavation base (1.5–6.0 m) and heave involves the
  soil below it. Forcing cu to 0.001 in an undrained formula is not a
  meaningful check. The agent's caveat was right; the check was not.

---

## What 5.16.0 already covers

- N12 and N16, partly: document tools for the primary agent.
- Nothing else in this list.

## Proposed order

1. **N2 + N3.** Module defects that produce wrong numbers.
2. **N1 + N12 + N14.** Delegation and read tools; silent content loss.
3. **N4 + N6 + N5 + N8.** The deliverables train (the owner's main complaint).
   The trivial fixes N7, N9, N10 and N11 ride along.
4. **N13, N15.**

## Side answer recorded here: SharePoint download speed

The owner asked whether fetching the reference PDFs from SharePoint would slow
the app. This session measured it: the 23 MB submittal downloaded in 1.2–1.3 s,
four times over. At that rate all 38 reference PDFs (~750 MB) would add roughly
40 s to every launch, so **not at launch**. Fetching a PDF the first time a
chart from it is needed, and keeping it for the rest of the cluster session,
costs 1–6 s once per reference and nothing at launch. Not built; owner to
decide.
