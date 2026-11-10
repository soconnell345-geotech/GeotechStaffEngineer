# Document Review: deployment review, 2026-11-10

A review of `aec-document-review` 0.1.0 and the code under it
(`aec-agent-core` 1.0.0, `planlens.review` in planlens 0.15.0, the Tiny Apps
wrapper), asked for by the owner before the testers' round: find and fix weak
points and likely errors on deployment, and recommend simplifications and
improvements.

Six read-only reviewers each took a slice (isolation on the shared host; the
agent loop; the model connection and vision; SharePoint; saved files, records
and the page shell; `planlens.review`). The lead checked each finding against
the code before acting on it. Open work from this review is in `BACKLOG.md`
under the IDs named here; this file holds the detail.

Two findings are described here by their kind only. How they were shown is
kept out of this public repository.

## 1. What the review changed

Everything in this section is fixed on master (commits b159fba and after) and
has a test with made-up values. Nothing is released: see §4.

### 1.1 One process, several testers

On Tiny Apps one Streamlit process serves every tester. The review found
five ways one tester's session could reach outside its own conversation.

| Was | Now |
| --- | --- |
| A request that reached the deployed app with no sign-in header was served as "anonymous": every such request shared one folder and the shared settings. `DEV_IDENTITY` was read on the deployed host too. | On the App Service the header is the only source of identity. Without it the page shows a plain message and stops. |
| The sidebar's Working folder box took any path on the server, for any tester. | The box is not drawn for a tester the proxy named, and a stored path is not used on a shared host. |
| A restored zip's `meta.json` could carry a working folder, and its transcript could name files anywhere on the server as downloads. | The zip's working folder is dropped. On a shared host a file reference never leaves the conversation folder. |
| A file path in the text of a tool result was copied into the conversation as a download. A tool result can quote a document, so a line in an uploaded file could pick a server file. | Only a file in the working folder or the temporary folder is copied. |
| `find_in_working_folder("../x")` found a file outside the folder. | It does not. |

**Model-written plotting code.** The figure helper can run matplotlib or
Plotly code the model writes, in a box: a separate process with a code check,
an audit hook and operating-system limits (GM-98). The box limits Python. It
is not an operating-system boundary, which BACKLOG INF-16 already said and
this review confirmed. On a host one person uses that is acceptable. On the
App Service the same process tree holds every tester's files and the app's
settings, and a document can ask the model for code. The wrapper's `app.py`
now sets `GEOTECH_PLOT_CODE=0` on the App Service unless the app setting says
`1`. The figure helper still draws, with the fixed plotting tool. This
reverses "on everywhere" (GM-92) for that one host: OWN-133.

### 1.2 SharePoint (the Graph route Tiny Apps uses)

- **A refused upload was counted as saved.** The Graph file manager answered
  False for any failure, and the conversation mirror did not read the answer.
  The file was stamped as sent and never sent again, with no message. After a
  redeploy a restore would lack it. A refused upload now raises, the mirror
  reports it, and the next sync sends it again.
- **No waiting on a busy service.** A throttled call (429) now waits as the
  service asks and is sent again, up to four sends; so is a read or an
  upload the service was too busy for (503, 504). A 401 on a token the app
  still held as good gets a new token and one more send.
- **An upload session's address went into the log** with its query, which
  carries a temporary token. Error texts now give the address without it.
- **"SharePoint refused the app"** was decided by finding "403" or "token"
  anywhere in the error text, so a file named `Spec_03403.pdf` raised it, and
  the message told a Tiny Apps tester to re-run a Databricks notebook. The
  check now looks for a refusal, and the message is worded for the host.

### 1.3 The turn

- **The model-call budget restarted on every pass.** A turn may run up to
  three passes (the automatic "continue"). A pass the budget ended wrote a
  reply that named a next step, which started a new pass with a new budget:
  up to 120 primary calls in one turn against a stated 40. A pass the budget
  ended is now the turn's answer.
- **Two error messages gave advice that does nothing on this page** (raise a
  step cap that is already 350; set a setting a tester cannot reach). They
  now say what to do.
- **Finished turns stayed in memory** for the life of the process. They are
  dropped once the page has shown them.

### 1.4 Saved files and settings

- **Word and Excel: a control character lost the whole file.** Text copied
  out of a PDF often carries a form feed. `write_docx` and `write_xlsx` now
  take such characters out.
- **Excel: text that opens with "="** (a note such as "= 2 x 150 mm") was
  written as a formula, which Excel shows as an error or offers to repair.
  It is written as text, and the tool's result says so.
- **A setting that is not a number** (`GEOTECH_DOCUMENT_TOOLKITS=eight`)
  stopped the app starting. It now falls back to the default.
- **Connection diagnostics on Tiny Apps:** the upload probe showed a standing
  false warning; the streaming advice named the Foundry setting; the report
  could carry a token an error text quoted. All three fixed.

### 1.5 The wrapper (`tinyapps/wrapper_repo/`)

- **A failed install** went on to start Streamlit, and the first error in
  the log was an import error far from the cause. `run.sh` now says the
  install failed and why that usually happens, and stops unless an earlier
  install left the packages in place.
- **The published address** was a line to edit in `run.sh`
  (`https://CHANGE-ME`), lost at every update of the file. It is now the app
  setting `APP_ORIGINS`; while neither is set the start log carries a
  warning that the page will not load through IIS.
- `/home/data/...` is used only where `/home` is the app's to write in.

### 1.6 planlens (`fix/review-findings-2026-11-10`, PL-36)

- A marked copy could replace the document that was opened, or a user's own
  file that happened to carry the default name.
- Exact search missed text a reader can see: a ligature, two spaces, a word
  broken at a line end, a no-break space. A reviewer told "the specification
  never mentions X" on that basis is the worst failure a review tool has.
- A search with many hits returned an error instead of a shortened list.
- A password-protected or empty PDF got no useful message.

## 2. Found, not fixed

Each is in `BACKLOG.md` under the ID given. "Needs a live check" means the
fix depends on how the rollout deployment behaves, which this review could
not measure.

### 2.1 The turn and the model connection

- **Retries inside retries (TA-24).** The model client retries a failed call
  four times, 180 s each, inside each try of the agent's own busy ladder,
  and inside the vision ladder. One stuck call can hold a turn for 15 to 20
  minutes before the tester sees an error, and hold one of the eight vision
  slots for about as long while other testers' looks queue with no message.
  Keep one ladder: client retries at 0 or 1, and a bound on the queue wait.
  Needs a live check.
- **No way to stop a running turn (TA-37).** A tester who sent the wrong
  request waits it out, and their monthly budget pays for it.
- **Ceilings for the whole turn (TA-38).** A pass that does not reach the
  budget can still continue with a new one; the continuation calls after a
  cut-off reply are not counted; the last forced reply can come back empty
  and reads "(no answer text)". A page look can be up to 17 image calls, so
  a 50-sheet set read sheet by sheet is about 850, with no count and no word
  to the tester first. With TA-05's spend line.
- **Reasoning effort (TA-03).** No `reasoning_effort` is sent, and the
  per-answer caps were sized on Sonnet. On a reasoning-class deployment a
  tile read can spend its cap on reasoning and return nothing. Needs the
  live deployment.
- **A fallback nobody would see.** If a private name in the agent library
  moves, the page silently gets the stock helper (all tools, 30 calls) in
  place of the read-only reader (14). Noted under INF-71.

### 2.2 Telling testers apart (TA-39)

- Sign-in names that differ only in punctuation map to one secret name and
  one folder (`j.doe`, `j-doe`, `j_doe`).
- A conversation's folder and its running turn are found by the
  conversation id alone, and a restore accepts the id inside the zip.
- The vision probe's verdict is kept per model and address, not per key:
  one tester's spent key lowers every tester's image size for two minutes.

### 2.3 SharePoint, the structural part (TA-40)

- Saves run inside the turn and under one process-wide lock. One tester's
  first mirror of a large drawing set holds every other tester's finished
  turn at "running", and "Sync now" can freeze the page.
- Each upload walks the folder chain again and lists every level: about 300
  calls for a 40-file first mirror, growing with the library.
- A large upload has no retry of a failed piece. Whole files are held in
  memory with no size limit.
- A listing that fails reads as "nothing mirrored yet".
- A name SharePoint refuses (`:` and the like) fails the same way at every
  sync.

### 2.4 Records and diagnostics on a slow day (TA-41)

- Connection diagnostics runs in the page with no time limit: about ten
  model calls in a row, each up to 180 s, exactly when the service is slow.
- The turn's crash record is rewritten whole after every message, and the
  activity log opens the share once per event and keeps error text as it
  came.

### 2.5 What the tools lose (DR-41, DR-42, VIS-30, PL-37, PL-38)

- **Word out:** `<sup>` and `<sub>` are printed as typed ("12 m<sup>2</sup>");
  strike-through stays as tildes; a tall image runs off the page.
- **Excel in:** a label merged over several rows reads as empty on all but
  the first, and nothing says so. A schedule grouped by level loses its
  groups.
- **Word in:** comments, footnotes, headers and text boxes are not read and
  nothing says so. A reviewer's comments on a Word file are invisible.
- **Text the model reads that names tools this page lacks:** the attachment
  note (`pdf_import`, `dxf_import`, `drawing_ir`, `call_agent`), a
  saved-file note (`write_diggs`), SharePoint tool text (DR-34).
- **E-size sheets (VIS-30):** on a 36 x 48 in sheet a 4 x 4 tile still shows
  4.3 pt lettering at about 9 px, under the app's own 12 px line, and the
  result's legibility note is removed once tiles were read. The reviewer
  treats dimension strings and keynotes as read.
- **Sheet numbers (PL-37):** the printed number is missed for "SHEET NO.
  S-201A" and for a label and value in separate cells, and "3 of 24" is
  returned as a sheet number. No tool takes a sheet number, so "check the
  callouts on S-201" leaves the model to guess the page.
- **planlens under load (PL-38):** each annotate call holds several whole
  copies of the file; two calls on one copy can lose marks; opening a
  document does work on every page.

### 2.6 Noted, not planned

Small, or seen only by reading: feedback files appended with no lock; the
`calculate` tool raises Python's recursion limit for the whole process per
call; a new HTTP client per engine build, never closed; the SharePoint error
log is read, added to and written whole; search reads one page of results;
with three of four SharePoint settings the storage block disappears instead
of naming the fourth; the mirror never removes a file; no size limit on an
image sent to the model; an open PDF keeps its file handle for 30 minutes
(Windows only); the upload box (Databricks route) has a limit per file but
none on the total.

## 3. Recommendations

### 3.1 Simplifications (INF-71)

The package was cut from a larger app and still carries what that app
needed. Each line below removes code this package never runs on its one
host, and several remove the root of a finding above.

1. **One folder per conversation.** Remove the Working folder box and
   `meta["working_dir"]` from Document Review. That deletes the env-var
   fallback and the import of outside files, and with them the root of two
   isolation findings.
2. **Find new files by looking at the folder, not by reading tool text.**
   The page already compares the folder before and after a turn. The second
   mechanism, which searches every tool result for phrases like `'saved':`,
   only adds the finding fixed in §1.1.
3. **One "is this path inside that folder" function** in place of five
   copies that already disagree.
4. **Pass the conversation's folder to the turn, and drop the table that
   looks it up by id.**
5. **One retry ladder** in place of three (§2.1).
6. **One pass per turn.** The automatic "continue" decides by matching the
   reply's last sentence against patterns; it caused the budget fault. A
   tester can type "continue".
7. **SharePoint: the Graph route only.** The Funhouse SDK builders, the
   Foundry dataset branches, error sniffing by text and the rename
   machinery serve hosts this package does not ship to. Upload by path lets
   Graph make the folders and removes three layers of folder creation.
8. **The page shell:** two-page machinery for one page; a budget panel that
   is never available here; formatting for tools the page never offers;
   diagnostics checks for other hosts (the source of two false readings).
9. **The measuring build** (`minimal`) and its shared image store leave the
   tester package.
10. **planlens binding:** feature detection for a planlens that is pinned
    exactly; two copies of the tool notes; two page arguments on every tool
    (give the model the viewer's 1-based number only).
11. **Plot box:** four in-process layers are a lot of code for something
    that is still not a boundary. One operating-system layer (INF-16) could
    replace most of it. Until then: off on the shared host, on for one
    person's install.

### 3.2 Review quality: what a stronger model changes

The page was built for models that needed a lot of help: a side call
describes each page in words, and the reviewer reasons from the words. The
recommendations below are in the order I would try them. Each is a
measured change, not a default flip.

1. **Raise the bar first (DR-25).** Every arm of the suite scores 33 to 34
   of 35. A suite at its ceiling cannot show that a change helped, or that
   a stronger model reads better. The tasks that separate reviewers are the
   ones that need two places at once: a callout that points at a detail
   that is not there; a schedule that disagrees with the plan's tags; a
   level that differs between plan and section; a note that cites a
   specification section the specification does not have; a revision that
   changed one sheet and not its twin. A long synthetic set with such
   problems planted, a specification with a submittal, and two revisions of
   one set would give every other item here something to be measured on.
2. **Let the reviewer see the pages (DR-38).** Today the reviewer never
   sees a drawing, only a description of it written by a side call. On the
   chart suite the same model scores 84 of 99 through side calls and 95
   with the image in front of it. Coordination checks are where this
   matters most: whether a callout matches a detail cannot be judged from
   two separate descriptions. Keep side calls for bulk transcription.
3. **A sheet index, and sheet numbers as addresses (PL-37).** Built once
   per document from the title blocks: sheet number, title, revision, date.
   Returned at open, accepted wherever a page is accepted. This is the base
   for the next item and for "say what you covered" in the reader's own
   terms.
4. **Cross-reference check and two-revision comparison (DR-07).** Parked
   until a suite task needs them; item 1 supplies the task. Both are
   mechanical once the sheet index exists, and both are what a human
   checker spends hours on.
5. **Tile by the sheet's own regions (VIS-30).** Title block, general
   notes, schedules, legend and each view, with a second level where
   lettering is still under 12 px. A uniform grid cuts dimension strings
   and revision clouds at tile edges.
6. **A review method (DR-06).** The prompt is thorough on how to read and
   says nothing on how to review. A draft is beside this file
   (`review_method_draft/`): the set before the sheets, checks that need
   two places, the basis of each comment, and a comment log an engineer can
   act on. It is written as deployment instructions (`GEOTECH_INSTRUCTIONS`),
   so it can be tried on the test machine without a release and measured
   before any of it enters the prompt.
7. **Standards (DR-13).** The page has no reference library, and a strong
   model knows many codes well enough to quote a clause number that is
   close but wrong. The draft method's rule: a comment rests on the
   document, on a document the user supplied, or on general practice named
   as such and to be verified. Never a clause number from memory.
8. **Specifications (DR-25).** Section numbers as addresses, the folded
   search of §1.6, and a suite task.

### 3.3 Order of work

1. Release the fixes (§4), then the owner's round on the test machine.
2. During the round: the live checks (retries, reasoning effort, the spend
   of a full-set review).
3. The harder suite tasks, then DR-38 and the sheet index measured on them.
4. The simplifications ride with INF-35 (GSE lets go of the review files),
   when the package's code is being touched anyway.

## 4. Releases

The fixes are on master and unreleased. To reach Tiny Apps they need, in
order: planlens 0.15.1; `aec-agent-core` 1.0.1; `aec-document-review` 0.1.1
pinning both; then the three pins in the wrapper's `packages.txt`. The
geotechnical app can take the new core at its next release. The wrapper's
own changes (`run.sh`, `app.py`) need no release: they arrive when the
wrapper repository is updated from the template. OWN-134.
