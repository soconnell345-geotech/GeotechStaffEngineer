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
| `PROMPTER-API-KEY--<NAME>`, `PROMPTER_KEYS`, `PROMPTER_SHARED_KEY_FALLBACK` | One key per signed-in user (a Key Vault secret per user, or a JSON map), and whether a user without one may use the shared key |
| `KV_NAME` | The Key Vault the settings above may be read from |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` (or `GRAPH_CLIENT_CERT`, `GRAPH_CLIENT_KEY`, `GRAPH_CLIENT_KEY_PASSWORD`, `GRAPH_CLIENT_CERT_THUMBPRINT`) | An app registration that may read and write the SharePoint site, by secret or certificate |
| `SHAREPOINT_SITE_URL` (or `SHAREPOINT_SITE_URLS`), `GEOTECH_SHAREPOINT_ROOT` | The site, or several, and the folder conversations are kept in; the cloud is read off the site's address (`GRAPH_CLOUD` to set it) |
| `GEOTECH_FOUNDRY_MODELS`, `GEOTECH_FOUNDRY_CONVERSATION_DATASET` | On Palantir Foundry (above): the model RIDs (`Label=RID`, comma-separated, the first is the default), and the workspace alias of the dataset conversations are saved to |
| `X-Windows-Auth-Header` (a request header, `WINDOWS_AUTH_HEADER` to rename it) | The signed-in user, as a reverse proxy doing Windows sign-in passes it; each user's conversations are kept apart |
| `DEV_IDENTITY` | A stand-in for that header on a laptop (`DOMAIN\user`) |
| `GEOTECH_WEBAPP_DATA` | The folder conversations are kept in locally |
| `GEOTECH_MARKUP_AUTHOR` | Who markups are signed by (default: `<user> via Document Review (AI draft)`) |
| `GEOTECH_INSTRUCTIONS` | A folder of the deployment's own instructions for the agents: `all.md` for every agent and helper, then one `<role>.md` per role (`main`, `general-purpose` for the reading helper, `figure`). Each role's text goes last in its prompt, under protected rules the text cannot switch off, at most 4,000 characters a role; the connection diagnostics say what each role gets. Details: GeotechStaffEngineer's [`webapp/README.md`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/blob/master/webapp/README.md), "Your deployment's own instructions" |

The settings keep the `GEOTECH_` names the page had before the split, so a
deployment carries on unchanged.

The project's deployment notes are in its repository:
[`tinyapps/TINYAPPS.md`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/blob/master/tinyapps/TINYAPPS.md)
and the wrapper template
[`tinyapps/wrapper_repo/`](https://github.com/soconnell345-geotech/GeotechStaffEngineer/tree/master/tinyapps/wrapper_repo).

## The review suite

The 52 review tasks the page is measured on, with deterministic checks, run
through the core's suite framework on this page:

```python
from aec_document_review.review_eval import score_review_suite
score_review_suite(model, out_dir="review_eval_run", docs_dir="docs",
                   dry_run=True)   # lists the tasks and checks the documents
```

## Licence

MIT. See `LICENSE`.
