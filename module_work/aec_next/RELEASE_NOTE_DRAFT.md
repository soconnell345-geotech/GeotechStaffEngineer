# aec-agent-core and aec-document-review, the next release: draft release note

**The version numbers are the owner's to decide: none is set here, and "the next release" stands wherever one would go.**
Drafted 2026-11-18 from the branch `claude/b13-integration` (the app's repository and planlens'), which holds what was merged after the tag `aec-v1.2.0` and after planlens `v0.16.0`. No version, pin or changelog heading has moved: both packages still say 1.2.0 and Document Review still pins `aec-agent-core==1.2.0` and `planlens==0.16.0`. The note is written for the two packages released together with the next planlens.

Each item was built, then read by a second reviewer, and what the review found was fixed before the merge (two follow-ups were read by the lead only; they are named in the last list). **None has been measured on a model.** So every change that alters what the agent is offered, shown or told is behind a setting that is off unless set, and what is on from the start is the repair of a defect, with one exception that is said plainly below. The geotech application is not released with this.

## For testers (Document Review on the testers' host)

**You get nothing from this release until the host's pins are moved and the app is synced.** With every setting unset you will notice:

- **A figure imported a second time keeps its chart.** A figure's interactive file copied into a folder that already held another file of that name was renamed so that it no longer read as a chart; the chat showed a plain file card beside a separate picture. It is now one card with its chart (TA-10).
- **A reading helper names a close-up it failed to read.** A close-up asked for by a view and a box inside it had no place of its own in the helper's list of what it left unread, so a later good close-up anywhere on that page struck an earlier failed one off the list. A region never read is now named (DR-50, part 3).
- **A photograph or screenshot uploaded as an image file is measured in its own pixels.** `measure` takes and returns its boxes in the image's pixels, so a place one tool gives can be handed to the next as it is; `log_grid` and the list of kept readings say which frame their positions are in. A PDF's results are unchanged (VIS-26).
- **A secret named in an error is masked.** The text of an error (a provider's message, a tool's exception) can carry a signed link, a token or a key. It was written whole to the conversation's record, which a tester may download and send on, and a tool's error went to the model the same way. Both are now masked, and so is the error text a tester sees in the chat, in a feedback or SharePoint failure line and in the diagnostics report. An ordinary error reads as before, with one exception: a value of 16 or more characters with a letter and a digit that follows the word `key`, `token`, `code`, `secret` or `sig` is masked even where it is no secret. A document's file name is never masked (TA-46).
- **With the next planlens** (see the host's part): a marked copy of an image file takes a small mark on a close-up that was sent at full size, and refuses one on a shrunk view in words about pixels; `measure` reads `side` as given ("Bottom" used to give the top edge); removing a mark by its stable id removes it; a block placed as an array in a DXF is read at every cell.

**The one change with no switch that is more than a repair.** When a long conversation is shortened by a summary and the summary call fails, the turn used to end on the provider's error (on the agent library versions installed today: langchain 1.3.15 and newer). Now the turn goes on: the model is handed the whole history, nothing is stored, and the summary is not tried again until the next turn. Where the whole history is too long for the model, the library summarizes regardless, and if that fails too the call ends on the error with the history kept. On langchain 1.3.14 and older this is a plain repair: there the error line took the place of the earlier work without a word (TA-45). A turn that would have stopped can therefore run on, and cost, where it did not before.

What does not change with the settings unset: the system prompt, the tools and their descriptions, how many images a page read takes.

## For the host's administrator

**The pins.** Three lines move together, each to its next release: `aec-document-review`, `aec-agent-core` (Document Review pins the core at exactly its own version) and `planlens` (Document Review pins it exactly; the core alone asks only `planlens>=0.15`). No new dependency in either package.

**What needs the next planlens, and what happens on 0.16.0.** The core reads what the installed planlens offers, never its version number (`aec_agent_core/document_tools.py`):

- sheet numbers as addresses (under the existing `GEOTECH_REVIEW_SHEET_TOOLS`): on 0.16.0 the switch gives what it gave in 1.2.0;
- `GEOTECH_REVIEW_DRAWING_TOOLS` and `GEOTECH_REVIEW_DXF_PLOTS`: on 0.16.0 each is ignored and the page is the default build, with no message;
- `GEOTECH_REVIEW_PLAN_TOOLS` works on 0.16.0 and on the next planlens;
- the marks on an image file (VIS-26's planlens part) and the planlens repairs listed under "The planlens half": on 0.16.0 the old behaviour.

**Optional packages.** A DXF plot needs ezdxf and Pillow. planlens requires ezdxf; the core requires Pillow (`Pillow>=8`), so a host that installs Document Review has both. planlens alone does not require Pillow: without it a plot is refused with "plotting a DXF needs the Pillow package ...", handed on as "could not plot ...", and the Document Review page does not say a DXF is a plot. The kernel layer needs Linux and nothing beyond the standard library; elsewhere its result says "not available on this system" and the box runs as before.

**New settings since 1.2.0: eight, all off unless set, none measured on a model.** Names and values are read from `review_flags.py` and `plot_box/kernel_layer.py`. For the first seven, on is `1`, `true`, `yes` or `on` (any case); unset or any other value is off.

| Setting | What it does when on | Caution | Measured on a model |
|---|---|---|---|
| `GEOTECH_REVIEW_PLAN_TOOLS` | The page and its reading helper get planlens' `plan_symbols` (the symbols of a vector plan sheet with their centres and labels) and `plan_grid` (page points turned into local easting and northing). | Two more tools in the model's list. | no |
| `GEOTECH_REVIEW_DRAWING_TOOLS` | Adds `drawing_geometry`: one page's dimensions, leaders, callout bubbles, revision clouds and title block read from its drawn lines, for a sheet whose lettering is drawn in strokes. | Needs the next planlens. A value or note is the nearest text, not a reading of it; "none found" does not show the sheet has none. No test yet that could show a gain. | no |
| `GEOTECH_REVIEW_DXF_PLOTS` | An uploaded DXF is also opened as a one-page plot of its model space, so the page tools (looks, page map, marks) take it. | Needs the next planlens, ezdxf and Pillow. Bounded at 16 MB, 50,000 entities and 30 s; drawn under the document lock; one DXF is plotted twice (by planlens and by the core). | no |
| `GEOTECH_REVIEW_FIGURE_EXAMPLES` | The figure helper is handed six worked examples of document-review figures (made-up values) in place of the core's geotechnical set. | None known. | no |
| `GEOTECH_REVIEW_LEAN_PROMPT` | Takes the agent library's general texts off the page's default build where they repeat or cut across the page's prompt. All or nothing: if one text is not found word for word, nothing is cut and the turn's record says which (`lean_prompt` on `turn_end`). | The one rule that then appears nowhere else: a helper's result is not shown to the user, so relay a summary. Watch for answers that only point at a helper's report. | no |
| `GEOTECH_VISION_SHOW_UPLOADS` | An uploaded image file is shown to the page's main model itself at its next step; no vision call reads it. One shown image is one image call on the turn's count and ceiling. | Set only where the main model takes images: the page assumes it does unless the model's profile says otherwise, with no live probe and no fallback on a refusal. The record says what was decided (`build.show_uploads`). | no |
| `GEOTECH_VISION_PIXEL_REQUESTS` | A vision side call on a PDF page that is asked for positions in page units is asked for pixel boxes, which the tool converts (`page_points`); page coordinates the answer states unconverted are named as the model's own estimate (`coordinates_note`). | Read by the vision tools as they look, so it also acts on the geotech page of a host that sets it. | no |
| `GEOTECH_PLOT_CODE_KERNEL_LAYER` | `1`, `true`, `yes`, `on`: on Linux the plotting box's child process puts up Landlock file rules and a seccomp filter before any plotting code runs, and the result says which parts went up (`box.kernel_layer`). `require`: the same, and the box refuses to run where the file layer cannot go up. Off: unset, empty, `0`, `false`, `no`, `off`. | Any other value makes the box refuse to run code. It turns nothing on by itself: whether a host runs model-written plot code stays the owner's decision. The filter is a deny list (OWN-157). Probe a host with `python -m aec_agent_core.plot_box.kernel_probe`. | no |

The first six are read when the Document Review page is built; the geotech page reads none of them, and the `minimal` build gets neither the drawing tool, the DXF plot nor the lean prompt. All eight are in `tinyapps/wrapper_repo/.env.example`, commented out, in the Document Review README's settings table and in the Tiny Apps notes.

**What the review suite clears between arms.** `review_flags.switches()` unsets the first seven (they sit in `LATER_ENVS`, now ten names) and, new in this release, `GEOTECH_TURN_MAX_IMAGE_CALLS` (`LATER_SETTINGS_ENVS`), unless the arm names them; on 1.2.0 a ceiling set in the environment applied to every arm unseen (EV-49). It does **not** clear `GEOTECH_PLOT_CODE_KERNEL_LAYER`: set or unset it for the whole run.

**Existing settings that do more.** `GEOTECH_REVIEW_SHEET_TOOLS` with the next planlens: every document tool that takes `pages` also takes `sheets` ("S-201, A-101"), and `page` also `sheet`; a number the sheet index does not hold, or holds on more than one page, is refused, never guessed. `GEOTECH_CITED_PAGES` and `GEOTECH_CITED_DOCUMENTS` (both still off by default) fire less often on what is not a citation; see the developers' part.

## For developers

**Core (`aec_agent_core`).**

- `deep/summary_guard` (new; installed when an agent is compiled, so on every page built on the core, the geotech page included): the failed-summary behaviour above. The record gains a `summary_failed` event and, only above zero, `summaries_skipped` on `turn_end`. Where the library's shape is unknown the guard is absent and one log warning per process says so.
- `error_text`: `mask_error_secrets` and `masked_error_text(text, limit)`. Masked: the record's error fields, error text one level down (the vision probe's error, the photograph-page check's `why`, the instructions folder's errors), the suite's result, and what `error_line` and the tool guard hand the model. `mask_secrets` is as it was.
- `cited_pages` (EV-18): a citation that used to be listed as `never_opened` can now go to `page_map` (the answer says only what a structure tool already said of the page), `named_unread` (the answer says it was not read) or `unverified` (a printed number no page is known to print); a page whose markups were listed or a reference page a reader rendered counts as opened. On 70 made-up answers 63 to 65 fire, against 66 on 1.2.0. With the switch off only the recorded event differs.
- `cited_documents` (EV-23, REF-114): a document the answer only points to, denies or names in a condition is kept apart (`mentioned_not_cited`; `n_cited` counts citations only); a reference-table read counts as a look at its document unless every page of it failed. An unfilled hook, `vision_tools.read_reference_table`; no page offers a new tool.
- The run record: `cache_reported` on a model call whose reply carried a cache count, zero included, and `cache_reported_calls` on `turn_end` (INF-66); `helper_tools` in the build record, an `image_read` event with the file's hash for every look at an image, and `calls_not_started` on `turn_end`, with a call that was answered told apart from one cut off (EV-20).
- `run_report`: a page viewer (every page taken in, with the boxes looked at; at most 800 pages drawn) and each turn's own image calls, ceiling and refused looks (EV-17 b).
- `deep/lean_prompt`, `deep/upload_view`, `plot_box/kernel_layer` and `kernel_probe` (new), `document_tools` (`plan_tools_available`, `drawing_tools_available`, `dxf_plots_available`, `sheet_address_args`; `host_for` hands planlens the scale an image view was sent at where planlens takes it).
- `deep/keep_work`: the open-item key carries a close-up's place however it was given.

**Document Review (`aec_document_review`).** `review_build` reads the page switches; `figure_examples` (new); the `minimal` build's helper hands back its open items, and its `page_count` on an image returns one page with its pixel size, not an error.

**The suite.** `aec_document_review/review_eval/tasks.py` holds 54 tasks with 147 checks of their own (counted from the code: `OPEN_TASKS` and each task's `checks`); the core adds its 2 automatic checks to each task (`AUTO_CHECKS`; all 54 take them), so 255 are scored. Two tasks are new, on the made-up building, with nine checks, one per planted problem: a window specification section with its submittal, and a reissue of the drawing set (DR-25, in part). The core's `stated_together` gains `scope="finding"`: the nine checks read one finding at a time. Known limits of those checks: a finding stated and then taken back in the same paragraph passes; a finding split over hard-wrapped lines can fail; two numbers side by side pass the two value checks whatever the verdict. Read every run's traces in full.

**Fingerprints moved since 1.2.0: one.** Document Review's `review_minimal` build (commit 1f10f8c4): its system prompt goes from 1,637 to 1,749 characters, because the sentence that said an uploaded picture "is not shown to you whole" now says what the tools do with it (VIS-26). The default build's fingerprint and the geotech pages' file (`webapp/tests/page_fingerprints.json`) did not move. The next planlens will need its own entry in both files when it reaches PyPI (`RELEASING.md` section 4).

## The planlens half (the release after 0.16.0)

**What a caller notices with every option unset** (repairs):

- `measure`, kind edge: `side` is read in any case and with spaces around it; a word that is not one of the four sides is refused. Kind curve: an `at` with two axis values is refused where only the first was read. A `min_score` of 0 is still read as 80, and the result now says so (PL-26).
- `annotate_document`'s `remove` takes a mark's stable `uid`; before, a uid was read as words and removed nothing (PL-27).
- `boxes_to_grid` leaves a box the vision answer wrote in page units ("[300, 200, 320, 220] pt") exactly as written and counts it, where it used to convert it as a pixel box (VIS-24).
- A DXF MINSERT is read at every cell of its array, in both DXF readers; before, the first cell only, with a warning (PL-18, in part).
- Marks on an image file, with append on: a marked copy that holds an image of another pixel size is passed over for a new copy under the next free name, with a note. With no `sent_scale` from the host the view rule is the old one, its reasons worded in pixels (VIS-26).
- `planlens.review`'s dispatcher refuses `plan_symbols` and `plan_grid` by name to a host that did not ask; they used to be served there, though no host was offered them. The toolkit itself serves them as before.
- `opencv-python-headless>=4.8,<6` (PL-16).

**New options, all off by default.**

| Option | Where | What it does |
|---|---|---|
| `plan_tools` | `ReviewHost(toolkit_options=...)` | The two plan tools specified and served through the host layer. |
| `drawing_tools` | `ReviewToolkit`, `ReviewHost` | Adds `drawing_geometry`, listed after every other tool. |
| `dxf_plots` | `ReviewToolkit`, `ReviewHost` | Opens a DXF as a plot (`planlens.dxf.plot.dxf_to_pdf`); a host that did not ask is never handed one. |
| `sent_scale(w, h)` | a host's `ImageSupport` | A mark on an image view sent at full scale or larger is placed; one on a shrunk view is refused. |
| `column_warning` | `Document` | A page warning where text set in columns is drawn row by row. Cannot be made certain. |
| `continued_tables` | `Document` | Notes that a table continues from, or onto, the next page. |
| `captions_on_every_page` | `document_outline` | Reads a caption on any page, not only where a list names it. |

No tool or host passes the last three yet: only a caller that builds a `Document` or an outline itself can set them.

**For a host that already asks for the sheet tools** (`sheet_tools`): `sheets` and `sheet` wherever a page is taken, with `sheet_pages` in the result (PL-37, in part); the page map's sheet number follows the sheet index once it is made (PL-41); the schedule-mark check says nothing definite its text cannot settle, and its tool description changed, so that host's spec text moves (PL-43).

## The reviews of the whole change (2026-11-18)

Three reviews read everything since the last release as one change, after the item reviews: two of the packages (`module_work/b13_whole/REVIEW_A.md`, `REVIEW_B.md`) and one of planlens. Each said to merge after its fixes, and the fixes are merged, each with a test that failed before it.

- **With every new setting unset the default page is what 1.2.0 was:** its prompt, tools, schemas, helpers and middleware (the fingerprint has not moved), apart from the repairs listed above. In planlens, 1,241 of 1,287 calls on made-up files answer as 0.16.0 did; the 46 that differ are the four stated repairs. No public name, argument or result key of 0.16.0 is gone.
- **Fixed in the packages:** the masking took a document's file name that starts `sk-`, `pk-` or `rk-` for a key; `measure` and `log_grid` ran outside the build's own planlens host, so with the DXF-plot setting on a plotted DXF was refused as a damaged file; a vision call that fails inside a tiled read left its error text unmasked one level down in the result; with uploads shown, an upload that could not be shown wrote its `image_read` record twice; one test left a failing stand-in in the agent library for the rest of the process; `GEOTECH_PLOT_CODE_KERNEL_LAYER` was in none of the settings notes.
- **Fixed in planlens** (both where a DXF opened as a plot meets another item, so neither exists in 0.16.0): `annotate_document` with an `output_path` naming the DXF replaced the drawing with a PDF, and is now refused in plain words; `forget(path)` did not close the plot of a DXF that was replaced. Four statements of the CHANGELOG and the review DESIGN were untrue or unsaid and are corrected.
- **planlens' CI now installs Pillow.** It is not a requirement of the package, so eleven tests had skipped on every run: every plot, the rule that a host which did not ask gets none, the marks on an image file, and the markup check.
- **Settings built in pairs** (the lean prompt with each tool setting; shown uploads with pixel requests, the image ceiling and `show_me`; the photograph-page modes with the cited-pages modes; the DXF plot with the sheet, plan and drawing tools; every setting at once): each builds, no tool is offered twice, no prompt part is dropped or doubled, the ceiling counts every image call, and each key of the turn's record is written once.
- **A host with several testers in one process:** every piece of state added since 1.2.0 is keyed to the turn or held in a context variable, but for the inline image store (next section).

## Known limits, noted and not fixed
- A failing summary still leaves one copy of the older history in the conversation's history files, once per failing turn.
- The masker does not catch a value split by a line break, Digest and `Pwd=` forms, or a URL encoded twice; outside a URL query a bare `key`, `token`, `secret`, `signature`, `sig` or `code` is masked only at 16 characters or more with a letter and a digit. The server's own console lines and the geotech page's own transcript are not masked.
- TA-10: where only one of a figure's two files collides, the copies get different names. The geotech page's copy is the maintainer's.
- `GEOTECH_VISION_SHOW_UPLOADS`: a second look at one upload stores a second copy.
- The DXF plot: `open_document`'s plot line names a tool the page does not have, and the plot's page is mapped as blank.
- The kernel layer: `kill` and `chmod` outside the folder pass; the site folders stay readable.
- The cited-pages check (off by default): a zoom on an uploaded picture is recorded as page 0 of a document, so "PDF page 1" of any document then counts as opened and the check goes quiet for the conversation. The same on 1.2.0 (EV-18).
- `GEOTECH_VISION_SHOW_UPLOADS`: the shown image waits in a process-wide store of 96 images; if 96 others are stored before that tester's next model call, it is not shown and nothing says so.
- planlens: `find_like` on a DXF answers with a library error for every host, as in 0.16.0; in a conversation whose toolkit two hosts share, the page map of the host that did not ask for the sheet tools follows the sheet index once the other host has made it.
- planlens: a mark drawn as lines on a plan whose other marks are lettered in a font is still `no_mark` (PL-44); a different image of the same pixel size under the same name is still added to.

## Not checked on a model or a live host

1. None of the eight settings, and none of the three planlens reading options, on any model. Brief 9 (EV-45) measures them.
2. The failed summary on a real provider and in a page turn (four library pairs with a fake model only). Brief 9 counts `summary_failed` and `summaries_skipped`; zero is expected.
3. VIS-26 on the testers' model: a photograph with an EXIF turn, and a pasted screenshot (`module_work/LIVE_TEST_QUEUE.md`, check 9's extension).
4. INF-66: whether the hosted platform's proxy passes the provider's cache counts (`module_work/inf66/CHECK.md`).
5. The kernel layer on a host: Landlock versions 1 to 6 and aarch64 were read, not run.
6. EV-18: the offline pass over saved runs with the review's fixes, and a real page map. EV-23: the hosted measurement.
7. The two new suite tasks: no model run. A real long set and the orientation turn are not built.
8. `GEOTECH_VISION_SHOW_UPLOADS`: whether a given main model takes images (no live probe).
9. planlens: the column warning on real pages; `drawing_geometry` on real stroke-lettered sheets; a MINSERT written by a CAD program; the schedule-mark check on a real set (made-up sets only).
10. The planlens wiring's CHECK.md was written against a preview of planlens; the review ran against its integration branch (`module_work/b14_wiring/REVIEW.md`, section 6).
11. Read by the lead with no second review when merged: the failed-summary follow-up (3403e27c) and the masking follow-up (74615f37). The reviews of the whole change read both files since.
12. The reviews of the whole change did not run planlens on Python 3.10 (CI does), the lean prompt on deepagents 0.6, or `run_report`.
13. Whether the whole suite is green on GitHub for this branch was not read while writing this note.

## Differences found while writing

Where a merge message and the code differ, the code is what this note says.

- **The lean prompt.** Merge 69d6c2cc says "a text is cut only where it is found word for word". The code is all or nothing: one text not found and nothing is cut (`aec_agent_core/deep/lean_prompt.py`, lines 31 to 40; `review_flags.py`, the `GEOTECH_REVIEW_LEAN_PROMPT` entry).
- **What `drawing_geometry` reads.** Merge 59d3cdb8 (and planlens' ea7fc9b) say dimensions and leaders. The tool also reads callout bubbles, revision clouds and the title block (planlens `planlens/tools/specs.py`, lines 964 to 968, the `find` values).
- **The kernel layer's values.** Merge 452b160a gives unset, `1` and `require`. The code also takes `true`, `yes`, `on` and `0`, `false`, `no`, `off`, and refuses to run the box on any other value (`aec_agent_core/plot_box/kernel_layer.py`, lines 83 to 85 and 157 to 165; `plot_box/__init__.py`, lines 168 to 172).
- **Pillow.** Merge 59d3cdb8 says the DXF plot "needs ezdxf and Pillow". True in the code (`document_tools.py`, lines 352 to 375), but neither is a new install for a Document Review host: the core requires `Pillow>=8` (`packages/aec-agent-core/pyproject.toml`, line 103) and planlens requires ezdxf (its `pyproject.toml`, line 35).
- **The first summarizer merge.** Merge 026d351e describes a summary tried again at every model call; 3403e27c changed that, and the code tries again only at the next turn (`deep/summary_guard.py`, lines 35 to 50).
