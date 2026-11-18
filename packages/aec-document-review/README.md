# aec-document-review

**Document Review**: an LLM agent that reviews architecture, engineering and
construction (AEC) documents, as a Streamlit page. Upload drawings,
specifications, submittals, reports or calculation packages; it reads the
text, looks at the pages where the text is not the page (drawings, scans,
tables drawn as lines), answers with the sheet or page it found each thing
on, and hands back Word memos and marked-up copies of the PDF. A review aid:
the judgement stays with the reviewer, and every markup it writes is signed
as an AI draft.

It runs on two packages of the same project:

- [`aec-agent-core`](https://pypi.org/project/aec-agent-core/), the shared
  core (model connections, vision calls, guards and records, Word and
  Excel), pinned exactly;
- [`planlens`](https://pypi.org/project/planlens/), the document layer
  (page maps, located text, tables, review markups), pinned exactly.

It imports nothing geotechnical. The geotechnical agent it was split from,
GeotechStaffEngineer, is a separate package.

**Releases.** This package and `aec-agent-core` release as one unit: they
carry one version number, and one tag in GeotechStaffEngineer's repository,
`aec-v<version>`, publishes both, the core first. A release of this package
always pins the core of its own version (`aec-document-review==X` requires
`aec-agent-core==X`); planlens keeps its own versions. Until 0.1.1 and 1.0.1
the two had separate versions and tags.

## Install

```
pip install aec-document-review
pip install "aec-document-review[azure]"   # Key Vault and certificate sign-in
```

Python 3.11 or later; it is tested on 3.11 and 3.12. A host behind a package
mirror installs it from the mirror like any other package; a wheel installed
by path works the same way.

## Run

```
aec-document-review                       # streamlit run of the page
aec-document-review --server.port 8502    # further arguments go to Streamlit
```

or, in a host that starts Streamlit on a script of its own (an `app.py` of a
few lines):

```python
from aec_document_review.tinyapps_entry import main
main()
```

With no model configured the page starts and says so in a banner.

### On Palantir Foundry

Published from a Foundry Code Workspace, the app file is these two lines:

```python
from aec_document_review.foundry_entry import main
main()
```

`main()` marks the process as a Foundry deployment and runs the page. The
models are then Foundry model RIDs only, listed in `GEOTECH_FOUNDRY_MODELS`
(for Claude models also install `langchain-anthropic`), and no personal API
key is read or mentioned. With `GEOTECH_FOUNDRY_CONVERSATION_DATASET` set to
a Foundry dataset's workspace alias, each conversation is saved to that
dataset after every turn and can be brought back from the sidebar; without
it, conversations are lost when the app's container is rebuilt.

A Foundry app is not told who is viewing it, so everyone who can open the
published app shares one list of conversations and their files: nothing
keeps one viewer's work apart from another's there. Share the app, and the
dataset, only with people who may see each other's documents. The steps are
in the project's
[`docs/FOUNDRY.md`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/blob/master/docs/FOUNDRY.md),
section 6.

## Settings

Read from the environment, then from an Azure Key Vault (when `KV_NAME`
names one and the host has a managed identity), then from a `.env` file in
the working folder.

| Setting | What it is |
|---|---|
| `PROMPTER_URL`, `PROMPTER_MODEL`, `PROMPTER_API_KEY`, `PROMPTER_CA_BUNDLE` | The OpenAI-compatible model endpoint (the "Prompter"), its model (it must accept images), the key, and a CA bundle for the endpoint's certificate |
| `PROMPTER-API-KEY--<NAME>`, `PROMPTER_KEYS`, `PROMPTER_SHARED_KEY_FALLBACK` | One key per signed-in user (a Key Vault secret per user, or a JSON map), and whether a user without one may use the shared key. `<NAME>` is the sign-in `DOMAIN\user` in upper case, every run of other characters one hyphen (`CORP\jdoe` gives `PROMPTER-API-KEY--CORP-JDOE`). Sign-in names that differ only in punctuation or accents (`CORP\j.doe`, `CORP\j-doe`, `CORP\j_doe`; `o'brien` and `o_brien`) give the same `<NAME>`. So a name holding anything other than letters and digits also has an exact secret name, `PROMPTER-API-KEY--<NAME>--<HASH>`, which the app looks up first; then that person's `PROMPTER-KEYS` entry; then `PROMPTER-API-KEY--<NAME>`, so a key set up earlier still works. To give two such testers different keys, give each the exact secret (or a `PROMPTER-KEYS` entry) and delete the plain `PROMPTER-API-KEY--<NAME>` once both are served: while it exists, whichever of them has neither is served by it. A name of letters and digits has only the plain name, and a look-alike with punctuation or an accent at an end of a part (`CORP\jdoe_` beside `CORP\jdoe`) that has no key of its own is served by it too: the key that is such a tester's alone is a `PROMPTER-KEYS` entry. The app shows the exact name, never a key: in the refusal a tester without a key sees, in the sidebar's Connection diagnostics, and from `python -m aec_document_review.tinyapps_keys "CORP\j.doe"`. As an app setting or `.env` line the name is spelt with underscores (`PROMPTER_API_KEY__CORP_J_DOE__<HASH>`). A tester's own model and URL sit beside the key under the same ending. A name of letters and digits whose secret name would pass Key Vault's 127 characters has no secret name (give it a `PROMPTER-KEYS` entry); any other name that long is looked up by its exact secret name, whose readable part is cut to fit. |
| `KV_NAME` | The Key Vault the settings above may be read from |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` (or `GRAPH_CLIENT_CERT`, `GRAPH_CLIENT_KEY`, `GRAPH_CLIENT_KEY_PASSWORD`, `GRAPH_CLIENT_CERT_THUMBPRINT`) | An app registration that may read and write the SharePoint site, by secret or certificate |
| `SHAREPOINT_SITE_URL` (or `SHAREPOINT_SITE_URLS`), `GEOTECH_SHAREPOINT_ROOT` | The site, or several, and the folder conversations are kept in; the cloud is read off the site's address (`GRAPH_CLOUD` to set it) |
| `GEOTECH_FOUNDRY_MODELS`, `GEOTECH_FOUNDRY_CONVERSATION_DATASET` | On Palantir Foundry (above): the model RIDs (`Label=RID`, comma-separated, the first is the default), and the workspace alias of the dataset conversations are saved to |
| `X-Windows-Auth-Header` (a request header, `WINDOWS_AUTH_HEADER` to rename it) | The signed-in user, as a reverse proxy doing Windows sign-in passes it; each user's conversations are kept apart |
| `DEV_IDENTITY` | A stand-in for that header on a laptop (`DOMAIN\user`) |
| `GEOTECH_WEBAPP_DATA` | The folder conversations are kept in locally |
| `GEOTECH_MARKUP_AUTHOR` | Who markups are signed by (default: `<user> via Document Review (AI draft)`) |
| `GEOTECH_INSTRUCTIONS` | A folder of the deployment's own instructions for the agents: `all.md` for every agent and helper, then one `<role>.md` per role (`main`, `general-purpose` for the reading helper, `figure`). Each role's text goes last in its prompt, under protected rules the text cannot switch off, at most 4,000 characters a role; the connection diagnostics say what each role gets. Details: GeotechStaffEngineer's [`webapp/README.md`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/blob/master/webapp/README.md), "Your deployment's own instructions" |
| `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB` | The largest file sent to SharePoint, in MB (default 2048); a larger one is not sent and the save names it |
| `GEOTECH_STOP_BUTTON` | `1`: a Stop button while the user's own turn runs; what the turn did is kept (off by default) |
| `GEOTECH_SPEND_ESTIMATE`, `GEOTECH_SPEND_INPUT_USD_PER_MTOK`, `GEOTECH_SPEND_OUTPUT_USD_PER_MTOK`, `GEOTECH_SPEND_MONTHLY_BUDGET_USD` | `1`: on a host without the Funhouse SDK, each user's estimated AI spend this month, from their own conversations' token counts at the two prices (US dollars per million tokens), against the budget if set; without the prices it counts tokens only (off by default) |
| `GEOTECH_REVIEW_SHEET_TOOLS` | `1`: planlens' sheet tools on the page (the sheet index, the callout check and, from planlens 0.16, the schedule-mark check), with `open_document` bringing the sheet index and the schedule check; with a planlens after 0.16.0 each document tool that takes `pages` also takes `sheets` (printed sheet numbers, "S-201, A-101") and each that takes `page` also takes `sheet` (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_PLAN_TOOLS` | `1`: planlens' plan tools on the page and its reading helper: `plan_symbols` (every small symbol on a vector plan sheet with its exact centre and its label) and `plan_grid` (page points turned into local easting / northing, right-handed) (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_DRAWING_TOOLS` | `1`: planlens' drawing tool on the page and its reading helper, with a planlens after 0.16.0: `drawing_geometry` (one page's dimensions, leaders, callout bubbles, revision clouds and title block read from its drawn geometry; a fallback for a CAD sheet the text and the look have not settled) (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_DXF_PLOTS` | `1`: with a planlens after 0.16.0, an uploaded DXF is also opened as a one-page PDF plot of its model space, so the page can look at it (`analyze_pdf_page`, `render_region`), map it, and write a marked PDF copy of the plot; reading and searching still read the DXF's own text (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_TODOS_WITH_CALLS` | `1`: one sentence asking the agent to update its to-do list in the same step as its next tool call (off by default; not measured yet) |
| `GEOTECH_VISION_SHOW_ME` | `1`: the page's looks take `show_me`, so the agent's own model sees the page or region at its next step instead of a separate reading's description (off by default; not measured on the review suite yet) |
| `GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK` | `1`: a page of photographs takes one look instead of a whole-page look and tiles, only where the page itself shows it is one; any doubt tiles the page as before (off by default; not measured on a model yet) |
| `GEOTECH_VISION_TILE_ANSWERS` | `1`: each tile of a tiled read answers the request from what it holds, with one line where it holds nothing on it (off by default; not measured on a model yet) |
| `GEOTECH_PHOTO_PAGES` | `record`: the turn's record names the attached documents' photograph pages that nothing looked at; `retry` (or `1`): the same, and the agent is told once before its answer goes out. Every page of each attached PDF is read once at the first turn end after it is attached; with `retry`, a turn that left photograph pages unlooked-at takes one more pass of the agent (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_FIGURE_EXAMPLES` | `1`: the figure helper works from this page's own worked examples (a plan redrawn to scale from a sheet's dimensions, a schedule against the count on the sheets, comments by sheet, values against a stated limit, a level as each sheet gives it, a submittal timeline) in place of the geotechnical ones (off by default; not measured on a model yet) |
| `GEOTECH_REVIEW_LEAN_PROMPT` | `1`: takes the agent library's generic prompt texts off the page's build (its base prompt, its to-do and helper sections, the `write_todos` and `task` descriptions) and puts the page's short ones in their place; all or nothing: where a text the library defines is not found word for word, the prompt goes out as it was and the turn's record says so (`lean_prompt` on `turn_end`) (off by default; not measured on a model yet) |
| `GEOTECH_VISION_SHOW_UPLOADS` | `1`: an image file a user uploads is shown to the page's main model itself at its next step, with no vision call reading it for the model; one shown image is one image call on the turn's count. **Set it only where the main model takes images**: a text-only model's turn then fails at its first call after a look. The turn's record says whether the page shows uploads (`build.show_uploads`) (off by default; not measured on a model yet) |
| `GEOTECH_VISION_PIXEL_REQUESTS` | `1`: a request to `analyze_pdf_page` or `render_region` for positions in the page's own units is asked of the vision model as pixel boxes on the image it sees, and the tool converts them to page points itself; page coordinates a reading writes into its prose are named as the model's own estimate (off by default; not measured on a model yet) |
| `GEOTECH_TURN_MAX_IMAGE_CALLS` | A number: the most images one turn, helpers included, may send to the model for a reading; past it a look is refused and the turn finishes with what it read (unset or `0`: no ceiling; the count is kept in the turn's record either way) |
| `GEOTECH_PLOT_CODE_KERNEL_LAYER` | Only where plotting code is on (`GEOTECH_PLOT_CODE`), on Linux. `1`: the process the plotting box runs code in also puts on what the host's kernel offers (no new privileges, Landlock file and TCP rules, a seccomp filter), and the result says which parts went up (`box.kernel_layer`). `require`: the same, and where the file rules cannot go up the box runs no code and says why. Unset or `0`: the box as it was; any other value: the box runs no code. A host's own setting: the review suite does not clear it between arms. What a host offers: `python -m aec_agent_core.plot_box.kernel_probe` (off by default; not checked on a tester host yet) |
| `GEOTECH_PROMPTER_MAX_RETRIES`, `GEOTECH_VISION_CALL_TIMEOUT_S`, `GEOTECH_VISION_SLOT_WAIT_S` | The model client's quick retries (default 1), the seconds before a page reading is given up and asked once more (default 90), and the seconds a look waits for a free reading slot (default 600; 0 = no bound) |

The settings in the last eleven rows are read from the environment only (on a local run a `.env` line is put there at start-up), not from the Key Vault.

The settings keep the `GEOTECH_` names the page had before the split, so a
deployment carries on unchanged.

The project's deployment notes are in its repository:
[`tinyapps/TINYAPPS.md`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/blob/master/tinyapps/TINYAPPS.md)
and the wrapper template
[`tinyapps/wrapper_repo/`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/tree/master/tinyapps/wrapper_repo).

## The review suite

The 54 review tasks the page is measured on, with deterministic checks, run
through the core's suite framework on this page:

```python
from aec_document_review.review_eval import score_review_suite
score_review_suite(model, out_dir="review_eval_run", docs_dir="docs",
                   dry_run=True)   # lists the tasks and checks the documents
```

## Licence

MIT. See `LICENSE`.
