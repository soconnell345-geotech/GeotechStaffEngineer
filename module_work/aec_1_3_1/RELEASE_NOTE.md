# aec-agent-core and aec-document-review 1.3.1: release note

**Released 2026-11-19 as one unit by the tag `aec-v1.3.1`, with planlens 0.18.0 (planlens' tag `v0.18.0`, the same day).** Document Review 1.3.1 pins `aec-agent-core==1.3.1` and `planlens==0.18.0`; the core alone still asks only `planlens>=0.15`. The release holds everything merged after the tags `aec-v1.3.0` and `v0.17.0`: ten pieces built and reviewed on the night of 2026-11-18 (seven here, three in planlens), and 21 small repairs from that night's audits of older code (15 here, 6 in planlens). It adds no setting. The number is 1.3.1, not 1.4.0, on the owner's word: "1.4" is kept for the release that follows brief 9's measurements (BACKLOG EV-45).

**None of it has been measured on a model.** Each new behaviour testers would see sits behind a setting that already existed and is off unless set; the repairs have no switch.

## For testers (Document Review on the testers' host)

**You get nothing from this release until the host's pins are moved and the app is synced.** With every setting unset you will notice:

- **A chart keeps both halves.** A chart whose picture or interactive file meets a file of the same name already in the conversation keeps both halves under one name and stays one chart card (TA-10).
- **A second file under a name already held is kept.** A revised report uploaded under the same name as the first, or two names that read alike once made safe, used to be dropped without a word; the second is now staged beside the first as `<name>_2`. A file the agent saves under the name of one of your uploads is saved beside it, and no longer replaces your upload (DR-53).
- **Excel deliveries.** A line break written `<br>` in a table cell is a line break in the cell, not the text "<br>"; a cell longer than Excel's 32,767 characters is cut there and the warnings say so (DR-53).
- **A downloaded conversation with no question asked yet** (files uploaded, nothing asked) can be restored (DR-53).
- **A turn that ran out of its budget** on its last reply keeps what it read, so the next turn can pick up from those pages, as the closing line offers (DR-52).
- **Error lines.** A failed network connection is told as a connection failure, not as a server error; a refused sign-in says what the turn kept (DR-52).
- **SharePoint.** A background save that failed is tried again by itself, up to three times (after about 1, 2 and 4 minutes); an app registration that is half set up says on the page that conversations are not kept, and which settings are missing; a site or a library that is not found is a failed listing, not an empty folder (TA-47).
- **Reading documents (planlens 0.18.0).** Exact search finds `contractor's` on a page that prints a curly apostrophe (also curly quotes, the prime, the minus sign and the en dash); a scale written `1-1/2" = 1'-0"` is read; a line measured on a plan that has a scale is no longer told the page has none; a sticky note at a sheet's edge keeps its icon on the sheet; `printed_pages` no longer gets a viewer twin one higher than any number printed.

Under settings that already existed, all off unless set:

- **`GEOTECH_VISION_SHOW_UPLOADS` (shown images, DR-38).** Another conversation's images can no longer push yours out; the same upload looked at twice is carried once; an image the turn's own limit left out is said not to have been shown.
- **`GEOTECH_REVIEW_DXF_PLOTS` (DR-19).** `open_document` names only tools the page has, promises no plot it could not make, and lists the plot's page as a plot, not as blank.
- **The review method (DR-06), where the host's instructions name its folder.** Two issues of a set are compared sheet by sheet, a change the revision block does not announce is a finding, and with one issue the answer says what it cannot show.
- **`GEOTECH_REVIEW_SHEET_TOOLS` (planlens PL-37).** Each row of the sheet index shows the scale the sheet's title block prints, as printed, and where in the block it was read; a view's scale is never taken.

What does not change with the settings unset: the system prompt, the tools and their descriptions, how many images a page read takes. No recorded fingerprint moved.

## For the host's administrator

**The pins.** Three lines move together: `aec-document-review==1.3.1`, `aec-agent-core==1.3.1` and `planlens==0.18.0` (the template `tinyapps/wrapper_repo/packages.txt` already says them). Nothing new to set.

**One repair to the check that keeps each tester inside their own conversation folders** (`sharepoint_tools._conversations_needle`). It matters only where `GEOTECH_SHAREPOINT_ROOT` is a library alone (for example `Shared Documents`); with the default, `Shared Documents/GeotechStaffEngineer`, or any folder inside a library, the check already held. It is in every release up to 1.3.0, so a host whose base folder is a library alone should move to 1.3.1.

**New background work.** A failed SharePoint save is queued again on a timer (a daemon thread; at most three tries per failure, at 60, 120 and 240 seconds, never before a wait the service named is over). Nothing else runs in the background that did not before.

**The turn's image store** changes a default-page turn only under `GEOTECH_VISION_SHOW_UPLOADS` or `GEOTECH_VISION_SHOW_ME`; with neither set, no image is stored. A turn's images are freed when it ends.

**A known limit, restated.** On the `minimal` build a summary of a long conversation that fails still ends the turn; 1.3.0's "the turn goes on" holds for the default build only.

## For developers, and whoever reads suite results

- **The suite** is 55 tasks and 151 checks (1.3.0: 54 and 147); the new task is `coord-second-issue` (DR-25). Its fourth check passes when the review method's own clause is repeated, so on the method arm it shows that the method was followed, not what a reviewer finds unasked.
- **Reading a score.** The scorer gave the same verdict on every one of 560 made-up answers at `aec-v1.2.0`, `aec-v1.3.0` and this release, and the 52 tasks those tags share are identical: compare releases on those tasks, never on a total. Six rules for reading a score are in `module_work/audit_2026_11_18/SCORER.md` (BACKLOG EV-50); in short, read the sentence before counting a pass or a miss, and subtract the 2 checks (3 with `no-false-alarm`) an empty answer passes. `rescore=True` runs a saved run again when its `run.json` holds an error: re-score a copy. Repaired: a saved run whose files are named with the other system's path separator is re-scored with its files.
- **The record.** A model call whose reply stopped at the output limit is marked `output_cut` in `activity.jsonl` (`run_report.py` does not print it yet, DR-52); a cut tool result's whole copy is named by a hash of a non-UUID call id, so two results no longer share one file. The run report leaves out and counts a record it cannot draw (EV-17).
- **New public names.** In Document Review: `core.uploads_to_stage(pairs, attachments)` and `core.make_save_fn(temp_dir, artifacts, uploads=None)`; `sharepoint_store.SAVE_RETRIES` and `SAVE_RETRY_S`. In the core: `xlsx_writer.MAX_CELL_CHARS`. Nothing published in 1.3.0 was removed or renamed.
- **Fingerprints.** None moved. Both fingerprint tests were run against planlens 0.18.0 before the tag: the recorded builds do not include the sheet tools, so the `sheet_index` description's extra word changes no recorded hash.

## For the maintainer (the geotech page)

The core's changes reach the geotech page when its pin moves: the record's `output_cut` and the cut-result file names, the Excel writer's `<br>` and cut warning, and the suite runner's re-score repair. The geotech page's own copy of the page code (`webapp/core.py`) still has the figure pair's older defect (its `_unique_dest`, about line 1787, splits a name at its last dot, and its two imports, about lines 1815 and 1902, name file by file); both repairs from `packages/aec-document-review/` would port as they are. The upload repairs (a second file under a held name) were made in Document Review's page only.

## The planlens half (planlens 0.18.0)

Every new behaviour is for a host that asks and is off by default (planlens' `CHANGELOG.md`, 0.18.0):

- **The reading options (PL-09).** `column_warning`, `continued_tables` and `captions_on_every_page` are passed for a host that names them in `toolkit_options` (at 0.17.0 naming one raised `TypeError`). That host's document has its own handle and counts as a second open document of the file against the conversation's limit (eight); the added lines count against a result's size, so a long read pages sooner; `captions_on_every_page` changes nothing through `ReviewHost` (only `document_roles` on a `ReviewToolkit`).
- **The printed scale (PL-37).** With the sheet tools on, each sheet-index row shows `scale` and `scale_from`, and the result's note says what they are. A row of `Document.sheet_index()` holds `scale` and `scale_how` for every library caller, asked or not; a sheet read on its own has no scale; a long index takes more calls (a made-up set of 150 sheets: one call at 0.17.0, two now); the `sheet_index` description is 7 characters longer.
- **Region tiles (VIS-30, first slice).** New public names nothing calls yet: `region_tiles`, `page_marks`, `RegionTiles`, `PageMarks` and five constants in `planlens.review.frames`, and `planlens.testing.esize_fixtures`. `region_tiles` may answer with the uniform grid (`method="grid"`), which cuts text lines. No tool, specification or result moves.
- **The audit's repairs** listed under "For testers", each with a test that fails before it.

## The reviews and the gate

- The ten pieces were each built, read by an independent reviewer and fixed (check files and reviews under `module_work/b15*/`). Two more reviews then read them as one change: this repository's seven are releasable as they are (`module_work/b15_whole/REVIEW_APP.md`); planlens' three are releasable with one wording repair, which is in (`module_work/audit_2026_11_18/PLANLENS_WHOLE.md`).
- The 21 audit repairs were read by the lead twice, in two sessions, and by no independent reviewer. Each has a test that fails before it; the reports are in `module_work/audit_2026_11_18/`.
- The gate: the whole suite green on GitHub on the release commit, with planlens 0.18.0 installed from the package index; planlens' own suite green on its release commit.

## Known limits, noted and not fixed

What the audits found and did not repair is one BACKLOG item per audit: DR-52 (one turn: a refused request told as "ask again", a cut reply delivered with no word, an empty last reply, a 429 for size asked again five times, a key pasted into a question written to the record), DR-53 (a tester's files), TA-47 (SharePoint: waits of up to 300 s from the page, no size check before a download, a non-JSON answer, a 500 not retried), EV-50 (the scorer), PL-46 (reading a report: a page number printed alone in the footer is not read; a repair is held), PL-47 (drawing sets: a sheet with several views at several scales), DOC-25 (documents).

## Not checked on a model or a live host

1. No piece and no repair has been run on a model.
2. The SharePoint repairs were run against a stand-in for the service, not the testers' library.
3. The upload repairs were run through the page's functions, not through a browser on the testers' host.
