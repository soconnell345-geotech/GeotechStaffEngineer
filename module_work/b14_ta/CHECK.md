# Tiny Apps repairs, third wave (BACKLOG TA-18, TA-06, TA-24, TA-13, TA-23, TA-10)

Six Tiny Apps items were read against the code. One is built; five are
left, each with the reason and what a live check would need to capture.

## TA-10, built: a re-imported figure keeps its chart

**What the item asked.** When a figure's interactive sidecar
(`plot.plotly.json`) is copied into a conversation folder that already holds
a different file of that name, the copy was named `plot.plotly_1.json`. That
name no longer ends in `.plotly.json`, so it showed as a plain file card,
and it no longer paired with its picture (`plot_1.png`): the chat showed two
cards where one chart belongs. The item names the app's copy of the page
(`webapp/core.py`); Document Review has its own copy of the same helper.

**What was built.** `aec_document_review/core.py`, `_unique_dest`: a name
ending in `.plotly.json` (any case) keeps that two-part suffix and takes the
number before it (`plot_1.plotly.json`, then `plot_2.plotly.json`). Every
other name is split at its last dot as before. Its three callers in Document
Review gain the repair: `import_external_artifacts`,
`import_reported_outputs` (the same `core.py`) and the SharePoint download in
`sharepoint_tools.py`.

**Judgement calls.**
- Only the one two-part suffix the page reads by name (`PLOTLY_SIDECAR_SUFFIX`)
  is kept whole; no general list of compound suffixes, since no other one
  changes how a file is shown.
- The app's own copy (`webapp/core.py`) is left as it is: the app is quiet in
  this repository and its keeper is the maintainer.

**Settings.** None: the repair of a defect. A free name, and every name not
ending in `.plotly.json`, come out exactly as before.

**Tests** (`packages/aec-document-review/tests/test_ta10_plotly_sidecar_name.py`,
made-up names and contents):
- a taken sidecar name becomes `plot_1.plotly.json`, then `plot_2.plotly.json`,
  and classifies as a chart;
- the suffix is matched in any case;
- free names, pictures, plain `.json` files, names with no suffix and a file
  named `plotly.json` come out as before;
- a figure re-made under names the folder already holds is copied as
  `plot_1.plotly.json` and `plot_1.png`, makes ONE chart card, and the older
  files are not overwritten.

**Noted, not fixed.** When only one of a figure's two files collides (the
picture is taken but the sidecar's name is free), the two copies get
different stems and do not pair; it needs a figure-level rename, not a
file-level one.

## Left for a live check or a decision

- **TA-18** (rebuild the client when a 401 survives the retry). Parked, and
  the reason holds: no refusal has been seen to survive the one retry. The
  live check must capture a 401 that comes back on the retry, with the time
  between the two tries and whether a new client then succeeds.
- **TA-06** (pin the budget-spent response in a test). It waits on TA-42's
  check (2): the generic budget wordings are already tested, but no answer
  of the testers' gateway to a spent budget has been read. The live check
  must capture the HTTP status, the error body (code, type, message) and the
  headers of that answer; the test then pins that body to the budget line
  with no retry.
- **TA-24** (the busy ladder's leftovers). The waits, tries and the tokens-
  per-minute setting need the live check the item names. The summarizer part
  is not a small repair: the summarizer that runs is the agent library's
  built-in one, built inside its agent constructor with no hook, and (in the
  library version installed now) it runs inside the model call rather than
  before it. Its own model call sits outside the ladder, and the library
  catches every error of that call and uses the error text as the summary,
  so a busy refusal there does not end the turn; it replaces the older
  history with an error line (the history itself is written to a file the
  summary names). Putting the ladder around it means replacing the built-in
  summarizer in its place in the stack: a design step for the shared core,
  not an afternoon repair.
- **TA-13** (metering on the backup host). Parked, and the reason holds:
  log labels on the backup host; the budget decorator belongs to the backup
  host's own client library, and the user attribution lives in the app's
  launcher (`webapp/`), which is quiet here.
- **TA-23** (the primary waits for its slowest helper). Parked as a design
  question; no small repair is plain.

## For the lead: what a measurement on the hosted platform should compare

TA-10 needs none: the tests show the names and the one card offline. For
TA-06 and TA-24, the captures listed above.
