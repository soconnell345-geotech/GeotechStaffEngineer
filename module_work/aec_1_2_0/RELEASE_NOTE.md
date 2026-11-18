# aec-agent-core and aec-document-review 1.2.0: release note

Prepared 2026-11-18. Both packages at 1.2.0, one tag `aec-v1.2.0`.
Document Review 1.2.0 pins `aec-agent-core==1.2.0` and `planlens==0.16.0`
(planlens is unchanged). It carries the BACKLOG items merged under
`packages/` since `aec-v1.1.0` (2026-11-17): DR-34, DR-42, DR-45, DR-47,
DR-48, DR-50 (parts 1 and 2), EV-17 (a first version), EV-48, TA-41, TA-43,
VIS-32, VIS-33 and VIS-34.

Each item was built, then read by a second reviewer, and what the review
found was fixed before the merge. **None has been measured on a model
yet.** For that reason every change that alters how a page is read, what
the agent is told at the end of a turn or how many images a turn may send
is behind a setting that is off unless set; what is on from the start is
the repair of a defect. The geotech application is not released with this
version: its repository copy still pins `aec-agent-core` exactly, and the
pin moves to 1.2.0 here only so that the repository installs.

## For testers (Document Review on Tiny Apps)

**You get nothing from this release until the app is updated.** The
testers' app changes only when the version numbers in the Tiny Apps
wrapper repository (its `packages.txt`) are moved to 1.2.0 and the App
Services team syncs the app. Until then you keep the app you have now.
Nothing about your folders or saved conversations changes at this update.

What you will notice after the update:

- **Word reports with photographs are smaller.** A photograph placed in a
  Word file the agent writes is stored as a JPEG, at no more than 200
  pixels per printed inch, instead of as the full-size PNG it was cut as
  (about 1 MB each). A report of 20 photographs comes to about 4 to 13 MB
  where it passed 20 MB before, so the agent no longer has a reason to
  leave photographs out to get under a mail limit. Line work, lettering
  and most charts and scans are stored as before; a chart or crop whose
  pixels look like a photograph (a shaded heat map, a shaded-relief map,
  line work drawn over a photograph) becomes a full-colour JPEG (DR-47).
- **Word files are read more fully.** A character from the Symbol font (a
  Greek phi in a soils report, a degree sign) is read as that character.
  Equations in Word's own format are counted and said to be not read,
  instead of dropped without a word. A file that holds tracked changes says
  so, and the text that was deleted is given in a section of its own,
  after the body, with who deleted it and where (DR-45).
- **The agent is no longer pointed at tools this page does not have.** The
  note it gets with an attached file, the note about files a helper saved
  and the SharePoint tools' descriptions named tools and soil-test examples
  from another application (DR-42, DR-34).
- **A reading helper's hand-back names more of what it left unread:** a
  text read that was cut at its size limit and never finished, and a
  close-up that failed and was not done again (DR-50).
- **A page read in close-up tiles says when the two readings disagree.**
  When the whole-page reading and a close-up give different readings of
  one tag at one place, the result names both and says a zoom settles it,
  instead of listing them as one agreed item (VIS-32).
- **Connection diagnostics ends.** Each live check waits at most 30 s (60 s
  for the image check), the run stops at the first one that fails, and
  after 3 minutes in all the checks not yet started read "not run: the
  time allowed for the checks was used up" (TA-41).

What does not change: how many images a page read takes, how long a turn
takes and what it costs. The four new settings below are off.

## For the host's administrator

**The pins** (`packages.txt` in the wrapper repository; the template in
`tinyapps/wrapper_repo/` has them):

```
aec-document-review==1.2.0
aec-agent-core==1.2.0
planlens==0.16.0
```

No new dependency, no change to `requirements-thirdparty.txt`, `run.sh`,
the Key Vault secrets or the SharePoint folders.

**New settings, all optional and all off unless set** (app settings or
`.env` lines, not Key Vault; listed in `wrapper_repo/.env.example`).
Turning one on for a tester round is the owner's call.

| Setting | What it does when set | What to know first |
|---|---|---|
| `GEOTECH_TURN_MAX_IMAGE_CALLS` | A number: the most images one turn may send to the model for reading, helpers included. At the ceiling a further look is refused with a plain message and the agent finishes with what it has read; the running turn shows "image calls this turn: k of N", and one line under the answer says how many image calls were not made. Unset or `0`: no ceiling. | A page read in tiles is up to 17 images, so a 50-sheet set read sheet by sheet is about 850. The count itself is always kept in the turn's record (`image_calls`), set or not, so a sensible number can be chosen from real turns. It counts images sent for a separate reading; a page shown to the agent's own model (`GEOTECH_VISION_SHOW_ME`) is not counted (TA-43). |
| `GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK` | `1`: a page of photographs takes one look instead of a whole-page look and four to sixteen tiles. Only a page that shows it is one (few words, photographs over a fifth of it, little line work, nothing drawn over a picture); any doubt tiles the page as before. | Saves image calls on photograph pages. Not measured on a model yet (VIS-33). |
| `GEOTECH_VISION_TILE_ANSWERS` | `1`: each tile of a tiled read answers the request from what it holds, with one line where it holds nothing on it, and quotes the part of a table or note its edge cuts. | Saves output on large sheets. Not measured on a model yet (VIS-34). |
| `GEOTECH_PHOTO_PAGES` | `record`: the turn's record names the attached documents' photograph pages that nothing looked at. `retry` (or `1`): the same, and the agent is told once, before its answer goes out. | The check itself calls no model. It reads every page of each attached PDF once, at the first turn end after the PDF is attached: a few seconds for a document of a few hundred pages, before the answer is shown. With `retry`, a turn that left photograph pages unlooked-at takes one more pass of the agent: more model calls, and image calls if it looks; where the turn's image ceiling is already met the note waits for the next turn. Not measured on a model yet (DR-48). |

## For developers

**Core (`aec_agent_core`).**

- `docx_renderer`: `write_docx` stores photographs as JPEG
  (`PHOTO_JPEG_QUALITY` 85, 4:4:4, at most `PHOTO_MAX_DPI` 200 per printed
  inch; `_is_photograph` decides by pixels, and a grey, palette, CMYK,
  transparent or rotated picture is left as it is). `docx_to_markdown`
  returns a new key `tracked_changes` (empty with none; bounded: a deleted
  passage over 300 characters is given by its ends, the list stops at about
  4,000 characters and says how many more there are); `markdown` stays what
  it was, the file with every change accepted.
- `review_flags`: three new switches and their readers
  (`TILE_ANSWERS_ENV` / `tile_answers()`, `PHOTO_PAGE_ONE_LOOK_ENV` /
  `photo_page_one_look()`, `PHOTO_PAGES_ENV` / `photo_pages()` returning
  `off`, `record` or `retry`). They sit in a new tuple, `LATER_ENVS`, which
  `switches()` clears and `describe()` names; `ALL_ENVS` and `ARMS` are
  unchanged. The review suite takes each as an arm given as a pair, for
  example `("tile_answers", {"GEOTECH_VISION_TILE_ANSWERS": "1"})`.
- `cited_pages`: `photograph_pages`, `attached_documents`,
  `photo_coverage`, `photo_note_for_model`; `TurnPageCheck` takes
  `photo_mode`, `photo_source` and `confirms_notes`, and has
  `note_handed()`: a page that may drop a note it was given calls it when
  the note is really handed to the model, and only then is a photograph
  page recorded as told (a `photos_fed_back` event). With `retry` and the
  turn's image ceiling met, no note is handed and nothing is recorded as
  told; the next turn is told. With the switch off the class does what it
  did in 1.1.0.
- `vision_tools`, `deep/limits`, `deep/vision_engine`, `deep/vision_kit`:
  the turn counts the images sent for a separate reading (`image_calls` on
  the turn's scope and in the record's `turn_end`; images shown inline to
  the agent's own model, as the minimal build and `show_me` do, are not
  counted, so 0 there means "not counted"); `_place_disagreements` writes
  VIS-32's sentence into `found_note`, tested at each place, and the row
  that holds both readings of a named pair no longer lists `page` in its
  `seen_in`. A tiled read whose tiles were not all read (a tile failed, or
  was refused at the ceiling) no longer says the page answer and the
  tiles agree: `found_note` says the list is from the page answer alone,
  or how many tiles were not read. A look that meets the ceiling says in
  its `image_calls` line that any further look this turn is refused.
- `deep/keep_work`: a helper's open items include a cut text read not
  finished and a failed zoom not redone; `hooks` has a new place,
  `keep_work.saved_files_hint`, for the sentence after the names of the
  files a helper saved (unfilled: the 1.1.0 wording).
- `run_report` (new): `python -m aec_agent_core.run_report <run folder>
  [-o report.html] [--documents <folder>]` writes one HTML page from a
  saved conversation or suite run: what was asked and answered, what the
  main agent and each helper took in, the tool calls and their size, the
  tokens, and for each document a grid of its pages marked by how each was
  taken in, with the boxes of the looks. It reads the record only and
  changes nothing; the page names no absolute local path. Measured on a
  made-up run of 300 large sheets, the page is about 1.5 times the size
  of the activity log, with at most 150 page images drawn (EV-17, a first
  version; the page viewer of its part (b) is not built).
- `review_eval`: the "found only in the working notes" line of RESULTS.md
  reads only what a reading said and a helper's own words, not the notes'
  JSON whole. The checks for an absent mark no longer take a table column
  headed "NO." for the word "no", in that line and in the score: **a
  re-scored older run can lose a pass it should not have had** (EV-48).

**Document Review (`aec_document_review`).**

- `diagnostics`: per-check limits, `BUDGET_S` (180 s, bounds only the
  start of a check) and the "not run" result.
- `core`: fills `keep_work.saved_files_hint`; builds its page check with
  `confirms_notes=True`; shows the image count on the running turn when
  `GEOTECH_TURN_MAX_IMAGE_CALLS` is set. `turn_jobs` writes the ceiling's
  line under the answer.
- No fingerprint moved in either package's tests: what the page's model is
  handed at the start of a turn (system prompt, tool names, descriptions,
  schemas, helpers, budgets) is what 1.1.0 hands it, apart from the two
  SharePoint tool descriptions (DR-34), which the fingerprint builds do
  not cover (they run with SharePoint unset). With the new settings unset,
  a turn differs from 1.1.0 only by the repairs listed above.

**Known limits, noted and not fixed.**

- A grey photograph is never taken for one (it stays a PNG in a Word file,
  and its page still tiles with the one-look switch on).
- A Symbol font set through a paragraph style, the document defaults or a
  theme font is not followed; such text reads as it did before.
- DR-50's third part (open items for the minimal build's helper) is not
  built and stays open.
- VIS-32 stays silent where a misread changes a letter for a digit or adds
  a revision letter (S09 against 509, 509A against 509): such a pair still
  merges into one row.
- The run report does not show a turn's image calls, its ceiling or the
  looks refused there; its "Vision calls" counts the side calls that
  answered, not the images sent.
- The run report shows each turn's answer only where the folder holds its
  activity log; with a transcript alone it names the questions.

**To measure before a switch goes on** (the review suite, each arm against
the baseline build; the notes are in `module_work/<item>/CHECK.md`, with
VIS-34's in `vis33/`, TA-41's in `dr48/`, TA-43's in `vis32/`, DR-45's in
`dr47/` and DR-34's in `dr42/`): image calls per page, output tokens, seconds and the same
facts kept (VIS-33, with drawing sheets unchanged as the control); output
tokens per tile and recall on tiles that hold the asked items (VIS-34);
how often a turn leaves photograph pages unlooked at, and whether the
note leads the agent to look or to say it did not (DR-48).
