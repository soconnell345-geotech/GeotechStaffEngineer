# aec-agent-core and aec-document-review 1.1.0: release note (INF-76)

Prepared 2026-11-17 on `release/aec-1.1.0`; not tagged. The first release
of the shared core and Document Review as ONE unit: both at 1.1.0, one tag
`aec-v1.1.0`. Document Review 1.1.0 pins `aec-agent-core==1.1.0` and
`planlens==0.16.0`. It carries everything merged under `packages/` since
`core-v1.0.1` and `document-review-v0.1.1` (2026-11-10), from BACKLOG's
Done entries; items that change only GeotechStaffEngineer are left out.

## For testers (Document Review on Tiny Apps)

**You get nothing from this release until the app is updated.** The
testers' app changes only when the version numbers in the Tiny Apps
wrapper repository (its `packages.txt`) are moved to 1.1.0 and the App
Services team syncs the app. Until then you keep the app you have now.

**At that update, if your sign-in name is not plain**, you get a new,
empty folder of your own. Your earlier conversations stay in your old
folder: nothing moves them, so ask the app owner to copy them across.
Plain means letters, digits, dots and hyphens, with single underscores
only inside the user name; not plain is a name with an apostrophe, a
space, an accented letter, an underscore in the domain or at an end of
the user name, a doubled underscore, a dot at either end, or more than
about 60 characters. Whether any tester's name is not plain is still
open (BACKLOG OWN-147, the owner's, before that update).

What you will notice after the update:

- **More sheets are looked at on a drawing set.** Every page 11 x 17 in or
  larger is now on the list of pages to view, so titled drawing sheets the
  text classed as text are no longer skipped. Expect more page readings and
  more time on such sets (planlens 0.16.0, PL-39).
- **Searches say less that is wrong.** A search that finds nothing lists
  only the pages it could not read (scans, or lettering drawn as lines).
  Contact sheets label each thumbnail with both page numbers,
  "pg <page> | PDF <pdf_page>". A rebar callout such as "10-20M EACH WAY"
  is no longer read as a length (planlens 0.16.0, PL-39).
- **Word and Excel files are read more fully.** From a Word file, after
  the body: its comments (author, date and the text each is on),
  footnotes, endnotes, headers, footers and text boxes; charts and objects
  it cannot read are counted and named. From an Excel sheet, a label
  merged over several rows is given on every row, and a merged title stays
  above its table. Word files the agent writes keep superscripts,
  subscripts and struck-through text, and a tall picture no longer runs off
  the page (DR-41).
- **Large sets split among reading helpers are compared better.** Each
  helper now hands back what other pages must be checked against (tags,
  schedule rows, references to other sheets) and what it could not check,
  and the agent compares these itself. If a helper's page reading failed
  or came back incomplete, the agent is told even if the helper did not
  say (DR-46).
- **Page readings are told they see one image**, so a reading does not
  claim a comparison with something it was not shown; a page read in
  close-up tiles keeps its note on small lettering (VIS-31, VIS-30).
- **Waits have an end.** A stuck call to the model ends in about 6
  minutes instead of 15; a stalled page reading is given up after 90 s
  instead of 180 and asked once more. When the service is busy, a page
  reading is asked six times instead of fifteen, so more may come back
  "not read"; when every reading slot is taken for 10 minutes, a look
  comes back as not read instead of waiting without end (INF-75, VIS-31,
  TA-24).
- **One budget per turn.** The page's 40 model calls cover the whole turn:
  an automatic "continue" no longer starts a fresh budget (the most used in
  84 recorded runs was 17). A turn whose budget ran out says so and what it
  did, instead of "(no answer text)" (TA-38).
- **Saving to SharePoint no longer holds up a turn.** The turn ends when
  its answer is written; the sidebar shows "saving the latest turn…" and
  then the result. A large file is sent in pieces and a failed piece is
  taken up again; a file above 2048 MB is not sent and is named. A file
  whose name SharePoint refuses (a colon, a question mark) is saved under a
  nearby accepted name (TA-40).
- **Your own key and folder.** Two testers whose sign-in names differ only
  in punctuation or accents can now have separate AI keys, and never share
  a folder. One tester's spent or refused key no longer lowers the image
  size of other testers' looks (TA-39). A sign-in name that is not plain
  gets a new folder, as said above (TA-39, TA-40; OWN-147).
- **A saved file is named by its file name**, not a `/tmp` path or a
  `sandbox:` link (DR-43).
- **Figures**: the plotting tool can draw panels side by side on one
  vertical axis, a section to scale, and contours; no existing figure moves
  (GM-315). A figure option outside the values the tool lists is refused
  with a suggestion instead of drawing something else, and a warning raised
  while drawing reaches the agent (GM-317, GM-316).

Not on unless the app owner turns it on, so nothing changes until then: a
Stop button, a line estimating your own AI spend this month, the sheet
tools (sheet list, callout check, schedule check), to-do updates folded
into the next step, and the agent's own look at a page. When the sheet
tools are on: "When you open a drawing set, the review now starts with the
set's sheet list and a check of its schedules against the tags on the
plans. It reads the PDF's text, so it cannot check scanned sheets or tags
drawn only as lines: it lists scanned sheets instead" (PL-42's note).

## For the host's administrator

**Install.** `tinyapps/wrapper_repo/packages.txt` pins
`aec-document-review==1.1.0`, `aec-agent-core==1.1.0` and
`planlens==0.16.0`; Nexus must carry all three (route B's wheels:
`aec_agent_core-1.1.0-py3-none-any.whl` and
`aec_document_review-1.1.0-py3-none-any.whl`). From this release an update
moves the two `aec-` pins to the same number; planlens has its own.

**Settings.** All optional; app settings or `.env` lines, not Key Vault.
Details in `tinyapps/TINYAPPS.md` ("Long turns and rate limits") and
`tinyapps/wrapper_repo/.env.example`.

| Setting | Default | What it does |
|---|---|---|
| `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB` | 2048 | The largest file sent to SharePoint; a larger one is not sent and the save names it (TA-40) |
| `GEOTECH_VISION_SLOT_WAIT_S` | 600 | How long a look waits for a free reading slot before it comes back as not read; `0` = no bound (TA-24, INF-75) |
| `GEOTECH_VISION_CALL_TIMEOUT_S` | 90 (was 180) | When a stalled page reading is given up and asked once more (VIS-31) |
| `GEOTECH_PROMPTER_MAX_RETRIES` | 1 | The model client asks a request again once, at once; a request that timed out twice is not asked again (INF-75). A deployment that set this or the timeout keeps its value |
| `GEOTECH_STOP_BUTTON` | off | A Stop button while a tester's own turn runs; what the turn did is kept (TA-37) |
| `GEOTECH_SPEND_ESTIMATE`, with `GEOTECH_SPEND_INPUT_USD_PER_MTOK`, `GEOTECH_SPEND_OUTPUT_USD_PER_MTOK`, `GEOTECH_SPEND_MONTHLY_BUDGET_USD` | off | Each tester's estimated AI spend this month, from their own conversations' token counts at the two prices (US dollars per million tokens), against the budget if set; tokens only without the prices (TA-05) |
| `GEOTECH_REVIEW_SHEET_TOOLS` | off | planlens' sheet index, callout check and schedule-mark check on the page (PL-37, PL-42); not measured on a model yet |
| `GEOTECH_REVIEW_TODOS_WITH_CALLS` | off | Asks the agent to update its to-do list in the same step as its next tool call (DR-43); not measured yet |
| `GEOTECH_VISION_SHOW_ME` | off | Lets the agent have a page or region shown to its own model (DR-38); not measured on the review suite yet |

Turning any "off" setting on for a tester round is the owner's call. The
live checks of the keys, the folders, SharePoint, the Stop button and the
spend line on the real host are BACKLOG TA-42.

**Keys (TA-39).** A sign-in name holding anything other than letters and
digits also has an exact secret name, `PROMPTER-API-KEY--<TESTER>--<HASH>`,
looked up first; then that person's `PROMPTER-KEYS` entry; then the old
`PROMPTER-API-KEY--<TESTER>`, so a key set up earlier still works. The app
shows the exact name, never a key: in the refusal, in Connection
diagnostics and from `python -m aec_document_review.tinyapps_keys
"DOMAIN\user"`. To give two look-alike testers different keys, give each
the exact secret or a `PROMPTER-KEYS` entry, then delete the old shared
name. The full paragraph is in `tinyapps/TINYAPPS.md`, "Naming
`<TESTER>`". A tester with punctuation in the name who holds both a
`PROMPTER-KEYS` entry and an old secret is now served the entry's key.

**Folders: the rule for sign-in names.**
- On the host (TA-39): a name of letters, digits, `.`, `-` and `_` (no `_`
  at an end of the domain or of the user name, no `__`) keeps its folder.
  Any other name (an apostrophe, a space, an accented letter) gets a new,
  empty folder, `users/<old name>__<10 hex digits>/`, and a new SharePoint
  folder under `conversations/` (the same name with single underscores,
  e.g. `corp_o_brien_3210ff1719`). Earlier conversations stay in the old
  folder (`users/<old name>/document_review/` on the host,
  `conversations/<old name, single underscores>/document_review/` on
  SharePoint) and are not moved, because another person may have shared it.
- On SharePoint, as TA-40 states it: "Two sign-in names can no longer
  share one SharePoint folder of conversations: a name with an underscore
  in its domain, a doubled underscore, a dot at its end or more than 64
  characters gets a folder of its own (its conversations on the host are
  sent there at their next save, with MOVED.txt left in the old folder);
  every other tester's folder stays where it is."

**Still unanswered (OWN-147).** Whether every tester's sign-in name is
plain is not known yet. Until it is: **a tester whose sign-in name is not
plain gets a new, empty folder**, on the host and on SharePoint, and
nothing copies their earlier conversations; the administrator can copy
their old folders into the new ones. Also open: whether `PROMPTER-KEYS`
holds an entry for a tester with punctuation in the name who also has an
old secret (that tester is served the entry's key from this release, so
it should be the key meant for them).

**On Foundry** (a second host, `docs/FOUNDRY.md` section 6): Document
Review can be published there from 1.1.0, with its conversations kept in
a Foundry dataset (INF-74). Each turn's files stay in their own
conversation even when several people use the app at once, and the
Working folder box is gone there (INF-80). Plotting code the model writes
starts off there, as on Tiny Apps. Nothing keeps one viewer's
conversations apart from another's on Foundry: everyone who can open the
app sees the same list.

## For developers

**One unit, one tag.** `aec-agent-core` and `aec-document-review` carry
one version; the tag `aec-v<version>` starts both `publish-core.yml` and
`publish-document-review.yml`. Both refuse unless the two pyprojects carry
one version, the tag is `aec-v<that version>` and Document Review pins
exactly `aec-agent-core==<that version>`. Document Review's run waits (at
most 30 min) until the core of that version is on PyPI; planlens must
already be there. The old `core-v*` and `document-review-v*` triggers are
gone; their tags stay as the record. GSE's `publish.yml` refuses while the
pinned core is missing from PyPI or while `packages/aec-agent-core/` has
changed since that version's tag (`aec-v<pin>`, or `core-v<pin>` before
INF-76). No change on pypi.org: the trusted publishers name the workflow
files and the `pypi` environment, not the tag. Detail:
`module_work/inf76/CHECK.md`.

**The pins.**
- `aec-agent-core` 1.1.0: `planlens>=0.15` (kept: the sheet tools and
  0.16's schedule-mark check are offered only where the installed planlens
  publishes them, `document_tools.sheet_tool_names`).
- `aec-document-review` 1.1.0: `aec-agent-core==1.1.0`,
  `planlens==0.16.0`.
- GeotechStaffEngineer master: `aec-agent-core==1.1.0` (its planlens floor
  is the app release's to move).
- The Tiny Apps wrapper (`packages.txt`, its README, `TINYAPPS.md`): the
  three above.

**In the core since 1.0.1** (all apps on it get these):
`foundry_dataset_store` moved in from GSE's `webapp` (INF-74; the old name
is an alias); Python warnings raised inside a method reach the agent under
`warnings` (`call_warnings.py`, GM-316), and a value outside a parameter's
`allowed_values` is refused with a did-you-mean (`method_call.py`, GM-317);
`plot_data` panels, sections and contours (`charts/panel_plot.py`,
GM-315); every reading told it sees one image (`turn_question.ONE_IMAGE_NOTE`,
VIS-31); vision side calls given up at 90 s and asked once more, a bound on
the wait for a slot (`deep/vision_engine.py`, VIS-31, TA-24), the Tiny Apps
client's retries at 1 (`tinyapps_engine.py`, INF-75); the run record's new
events `side_call_timeout`, `side_call_late_end`, `engine_retry`,
`vision_slot_timeout` and `model_end`'s `tool_names` (INF-75, DR-43); a
refused key's probe verdict held for its own caller only
(`vision_probe.py`, TA-39); a helper's whole record kept with its
unfinished reads (`keep_work.KeepWork(readers=...)`, DR-46); the sheet
tools' binding, feature-detected (PL-37, PL-42); `show_me` looks (DR-38);
the review suite's framework: checks that read what a run delivered, the
coordination checks with a false-alarm check, three more RESULTS columns,
to-do-only steps counted (EV-41, EV-44, EV-46, EV-47, INF-75, DR-43).

**In Document Review since 0.1.1:** `identity.py` and `tinyapps_keys.py`
(TA-39); `sharepoint_store.py` and `graph_sharepoint.py` (TA-40);
`turn_jobs.py` and `budget_panel.py` (TA-37, TA-05); the whole-turn budget
(TA-38); the office readers (DR-41); `foundry_entry` (INF-74, INF-80); the
reader's return contract (`review_prompt`, `review_build`, DR-46); the
review suite at 52 tasks, fourteen of them coordination tasks (DR-25), the
review method as package data and a suite arm (DR-06), arms `show_me`,
`sheet_tools`, `todos_with_calls`.

**Fingerprints.** Document Review's planlens part is recorded under
planlens 0.16.0 (surface `3e415249e4551735`): only `search_document`'s
description moved (PL-39). The app part did not move. GSE's page
fingerprints (`webapp/tests/test_page_fingerprints.py`) also need an entry
for planlens 0.16.0, because GSE's CI installs planlens from PyPI.

**Each item's own record:** `module_work/ta39/CHECK.md`,
`module_work/ta40/CHECK.md`, `module_work/ta37/CHECK.md`,
`module_work/inf75/CHECK.md`, `module_work/vis31/CHECK.md`,
`module_work/dr41/CHECK.md`, `module_work/inf80/CHECK.md`,
`module_work/gm315/CHECK.md`, `module_work/gm316/CHECK.md`,
`module_work/pl39/CHECK.md`; BACKLOG Done for the rest.
