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
**Disposition: FIXED in `6640446`.** All four below, plus the guard on the
scratch tools (N12):
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

**Disposition: FIXED in `349191e`.**
- New `soe/free_earth.py`: layered effective-stress Rankine pressures with
  water on both sides.
- Cantilevers now use the Caltrans Simplified Method (moments about O,
  D = 1.2 D0).
- Braced embedment now uses free earth support about the lowest support
  (MR = FS x MD). The old `compute_embedment` was confirmed wrong and replaced.
- Pinned to Caltrans Example 8-1 (D 6.09 ft, D′ 4.89 ft, T 14,254 lb/ft,
  M 22,494 ft-lb/ft, all within 1 %), to the closed form for a uniform sand,
  and to `sheet_pile.analyze_cantilever` (D0 within 0.5 %).
- Two more defects of the same class were found in the pass and fixed: the
  braced span above the first support was treated as simply supported
  (p·d²/8 instead of the cantilever p·d²/2), and surcharge and water were
  never added to braced loads (GEC-4 5.2.4).

On the session's own inputs the cantilever now needs about 9.7 m of
embedment; it had said 0.60 m.

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

**Disposition: FIXED in `349191e`.** One anchor level is now solved by free
earth support about the anchor (a `Kp` override for a log-spiral passive
coefficient, `FOS_embedment` default 1.3). It returns the total TH, the design
load, D, D′ and the wall moment, and reproduces Caltrans Example 8-1, which
closes the V-013 residual. `subgrade_reaction_kN_per_m` is None for one
anchor.

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

**Disposition: FIXED in `68f668e`.** A turn callback (`webapp/output_capture.py`)
records every path a tool reports writing, for the primary and every
sub-agent. After the turn, files outside the conversation folder are copied
into `files/`. They get a download card and a preview, and mirror to the
conversation's SharePoint folder, which is the "save to SharePoint as
standard" the owner asked for.

## N5 — SharePoint upload went to the wrong folder, then to a doubled path **[MED]**

Turn 8's destination was `Shared Documents/General/GSE_app/uploaded references`.
Turn 9's was `General/GSE_app/conversations/b9efee4d…`, which resolved to
`Shared Documents/General/GSE_app/General/GSE_app/conversations/…`.
`sharepoint_tools._resolve` (lines 74–77) strips only ONE leading segment equal
to the root's last segment; `General/GSE_app/…` repeats two. The mirror also
names the folder `<title>_<date>` (`sharepoint_store.conversation_folder`),
which the agent cannot know, so it guessed the thread id.

**Disposition: FIXED in `de65407`.** The upload tool is now built per
conversation and, with no `dest_folder`, uploads to that conversation's
SharePoint folder (`.../conversations/<title>_<date>/files`). Path resolution
drops the longest leading run that repeats the base folder's segments. The
prompt says deliverables are already mirrored and that "uploaded references"
is for the users' inputs.

## N6 — Inline plot did not render **[MED]** (owner screenshot)

The answer contained `![PYWall lateral earth pressures](/tmp/pywall_lateral_pressures.png)`.
The chat cannot display a local path; images render only from file cards
(`app.py::_render_artifact_card`). `profile_figure`'s result had told the agent
"Saved as a PNG: the chat UI renders it inline", which is true only inside the
conversation folder.

**Disposition: DELIVERY FIXED in `68f668e`; the KIND of figure fixed on
`feature/plotly-plot-data` (2026-09-16).** `68f668e` fixed the delivery half:
the figure is copied in and shown as a card under the reply, a markdown image
pointing at a local path is replaced with "shown below", and the figure tools'
notes no longer promise that a local PNG renders by itself.

What the owner had actually asked for was an INTERACTIVE line graph, and the
agent had picked the right tool — `profile_figure.plot_data` was
matplotlib-only, so a static PNG was all it could produce (only the
`subsurface.plot_*` family, which needs a parsed site, emitted the
`.plotly.json` sidecar the app renders with `st.plotly_chart`). `plot_data`
now builds the Plotly twin of the same cleaned series by default and writes it
beside the PNG; the chat shows the interactive chart and suppresses the PNG
card for that same figure, while the PNG stays on disk for the SharePoint
mirror and for `html_to_pdf`. `interactive: false` opts out. The deep prompt's
stale "renders a saved PNG/HTML figure INLINE" claim was corrected in the same
change.

## N7 — `html_to_pdf` rejected `file:///tmp/…` images **[MED]** (agent note, turn 4) — code bug

`calc_package.py:1300-1301` strips `file:///` (8 characters), turning
`file:///tmp/x.png` into the RELATIVE `tmp/x.png`, which then resolves under the
conversation folder (`…/files/tmp/x.png`). That is correct for Windows
(`file:///C:/…`) and wrong on the Linux driver. It is why the first PDF shipped
without figures.

**Disposition: FIXED in `b06001f`** (the leading slash is kept; drive letters
and %-escapes are handled; an end-to-end embed test was added).

## N8 — "It gives the document it reviewed back to you" **[LOW]** (owner note, turn 2)

`sharepoint_download_file` saves into `files/`, and new files there become
cards, so the 23 MB submittal looked like something the agent produced. The
same file was downloaded four times under two names (turns 1, 2, 3, 5), about
92 MB of copies.

**Disposition: FIXED in `68f668e` and `de65407`.** Fetched files are kept off
the produced-files list and shown as "Read from SharePoint: …", and a file
already downloaded in the session is reused.

## N9 — `analyze_pdf_page` would not take the downloaded file's name **[LOW]** (agent note, turn 1)

The download result says "the file is now in the working folder and available
to the file tools". `analyze_pdf_page` and `read_pdf_text`, however, accept
only an attachment key or a full path (`vision_tools.py:235-259`). Downloads
are not registered as keys. It happened again in turn 5 (4 calls).

**5.16:** not addressed. `document_tools._resolve` has the same rule.
**Disposition: FIXED in `b06001f`.** A bare name now resolves in the working
folder for the vision tools, `read_pdf_text`, the document tools and
`read_text_file`.

## N10 — `soe` `soil_type: 'clay'` is documented but rejected **[LOW]** (agent note, turn 2)

The adapter brief (`adapters/soe.py:184`) says `'sand' or 'clay'`; the module
takes `sand` / `soft_clay` / `stiff_clay` (`geometry.py:111`). Four failed calls
across turns 2–4.

**Disposition: FIXED in `349191e`.** The brief now lists the real values, and
`clay` maps to soft or stiff by N = γH/cu.

## N11 — Module name `SOE` rejected over capitalisation **[LOW]** (agent note, turn 4)

**Disposition: FIXED in `b06001f`.** Module names now match regardless of
case, spaces or hyphens.

## N12 — Scratch-filesystem tools used on real files about 35 times **[MED]**

`grep` ran 25+ times over the PDF folder and always returned "No matches found".
`glob` and `read_file` also ran on real files ("not found" for files that
exist). These are deepagents' in-memory scratch tools, and their answers look
real. This is the root of N1.

**5.16:** partly addressed. `search_document` gives the primary a real in-PDF
search; sub-agents still have only the scratch tools.
**Disposition: FIXED in `6640446`.** A middleware on the primary and every
sub-agent (deepagents' general-purpose one re-declared to carry it) answers a
scratch-tool call on a real path with the tool to use instead.
`read_text_file` reads real text files on the primary and the calc sub-agent.
Still open: sub-agents have no in-PDF search of their own.

## N13 — Reviewer sub-agent cited a reference it never found **[MED]**

Turn 1: `gec7` `list_methods(category="earth retaining structures")` → `{}`;
`describe_method` → error; `search_sections` → `[]`; `dm7`
`list_methods(category="retaining walls")` → `{}`. The reviewer then cited
"FHWA GEC-12 (Design and Construction of Deep Excavations)" about ten times;
GEC-12 is *Driven Piles*. The primary's answer did not repeat it, so it did not
reach the owner. This is the standing hallucination-after-failed-lookup risk
(CLAUDE.md figure TODO P1). Separately, a category filter that matches nothing
returns `{}` without saying so.

**Disposition: FIXED in `5cd3e7f`.** The reviewer and references prompts
allow citations only from tool results in that consult, and a category filter
that matches nothing lists the module's real categories. It is a prompt rule,
so check it on the next live review. (`5c1058b` also gives the reviewer
`record_feedback`.)

## N14 — Turn 3 changed the wrong layers, and the answer said otherwise **[MED, engineering]**

The primary asked to zero the cohesion of "the two Stratum C layers": the 2.5 m
and 2.0 m layers with φ = 33°, c = 50 kPa, identified in turn 2. The calc
sub-agent renamed the layers itself (the φ = 33° layers became "B1/B2", the
φ = 26° layers "C1/C2") and zeroed c in the φ = 26°, c = 9 kPa layers at 6.0–9.0 m.
The owner was told "the required embedment increased significantly when
cohesion was removed". That change (1.16 → 2.33 m) came from the layers below
the excavation, not Stratum C, and from the suspect embedment routine (N2).

**Disposition: FIXED in `6640446`.** The calc prompt says to keep layer names
as delegated and to name the layers changed; the delegation nudge tells the
primary to pass the named layer list. Prompt-level, so check it on the next
live run.

## N15 — `activity.jsonl` mislabels sub-agents that run in parallel **[LOW, triage tooling]**

Turn 5 ran `calc` and `general-purpose` at once. The logger attributes tool
calls through a stack of `task` calls (`activity_log.py` docstring), so
everything nested was labelled `general-purpose`, and the two task-end records
carry each other's names (the "calc" end record holds the PYWall table).

**Disposition: FIXED in `5cd3e7f`.** Events are now attributed by following
each run's parents to the task call it ran inside.

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

## Train record (2026-09-15, branch `feature/nairobi-rerun-fixes`, unreleased)

Owner said "start on the fixes in that order". Every item except N16 (cost;
measure on the next live run) and N17 (notes on the answers) is fixed:

**Full gate on the branch tip (4782aea): 11,743 passed / 33 skipped / 0 failed** (2026-09-15, run in eight batches because one process ran out of memory).

| Commit | Items |
|---|---|
| `349191e` | N2, N3, N10: SOE free-earth-support solver |
| `6640446` | N1, N12, N14: scratch-tool guard, `read_text_file`, delegation rules |
| `b06001f` | N7, N9, N11: file:// paths, bare names, module-name case |
| `68f668e` | N4, N6, N8: deliverables copied in, fetched files not outputs |
| `de65407` | N5, N8: conversation upload folder, doubled paths, download reuse |
| `b3ab1d5` | Owner decision: reference PDFs from SharePoint `primary_references` on first use |
| `5cd3e7f` | N13, N15: citations from tool results only; parallel sub-agent attribution |

**Check on the next live run:**
- a calc package or plot written anywhere shows as a card and reaches the
  conversation's SharePoint folder;
- "save it to SharePoint" lands in the conversation folder;
- a chart question fetches its PDF from `primary_references`;
- a rebuild delegation carries the numbers;
- the reviewer cites nothing it did not look up.

**Noticed, not done:**
- Braced support loads still send the bottom span's load wholly to the lowest
  support (conservative); the free-earth load is reported alongside.
- The user manual's SOE braced example will change when the manual is
  regenerated.
- The Agent-picker specialists still have no SharePoint or email tools.
- Sub-agents have no in-PDF search.

## Side answer recorded here: SharePoint download speed

The owner asked whether fetching the reference PDFs from SharePoint would slow
the app. This session measured it: the 23 MB submittal downloaded in 1.2–1.3 s,
four times over. At that rate all 38 reference PDFs (~750 MB) would add roughly
40 s to every launch, so **not at launch**. Fetching a PDF the first time a
chart from it is needed, and keeping it for the rest of the cluster session,
costs 1–6 s once per reference and nothing at launch. **Owner chose on-demand,
folder `GSE_app/primary_references`; built in `b3ab1d5`.**
