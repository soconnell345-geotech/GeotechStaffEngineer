# Field feedback — Nairobi SoE, first attempt (2026-09-09, v5.11.2)

Session `4700a8fc5994493b87bf4de90fc26888`, model `funhouse-gpt-high`,
2 turns spanning **43 seconds** (created 11:27:26.5, updated 11:28:09.5).
Raw export in `raw/` (gitignored — 23 MB project PDF).

Owner's note, verbatim:

> Why did this try to run this search while hunting around for the sharepoint
> doc? Looks like it found the doc, but did some search it didn't have
> credentials for.

## F1 — The 401 did NOT come from SharePoint. It came from the LLM endpoint. **[diagnosis corrected]**

**The owner's read is understandable but the evidence rules it out.**

1. `AuthenticationError` is `openai._exceptions.AuthenticationError` (a subclass
   of `APIStatusError`, raised only when the **model endpoint** answers 401).
   It is not a Graph, MSAL, or SharePoint exception.
2. **No SharePoint tool can raise it.** Every tool in `webapp/sharepoint_tools.py`
   catches `Exception` and returns a *string* —
   `sharepoint_search_files` ends `return f"SharePoint search error: {type(exc).__name__}: {exc}"`
   (`sharepoint_tools.py:218`), and the download/list/upload tools follow the
   same pattern. A SharePoint 401 would have appeared as tool-result text and
   the turn would have continued.
3. **The SharePoint work succeeded.** The transcript's `artifacts` list and the
   screenshot both show the file attached at its true size,
   **23,006,108 bytes**, with working Download and Preview controls.
4. The 401 body is an **IIS HTML error page**, not JSON. A real
   OpenAI-compatible API returns a JSON error; an IIS page means the request
   was rejected at the proxy's front door — the Funhouse LLM proxy — before it
   ever reached a model.

So: the file was found and fetched, and then the agent's own model call was
rejected with 401 and the turn died with no answer text.

**Why it failed: the request was too big for the proxy's NTLM handshake.**
Not the PDF, and not a permissions problem — a SIZE-triggered auth failure.
The chain, from the SDK source in
`Funhouse_for_Reference/funhouse-sdk-python/funhouse/services/prompter/prompter_api.py`:

1. The Prompter backend authenticates with **NTLM on the httpx transport**
   (`ExHttpNtlmAuth`, `prompter_api.py:53`), using Windows credentials. The
   OpenAI `api_key` is the placeholder `"PrompterOpenAI-Databricks"` that the
   service never reads (documented in the SDK's own
   `03. Prompter/09. Alternative Prompter Authentication.py`).
2. NTLM is a challenge-response handshake: send, get 401 + challenge, re-send
   with the answer.
3. **This implementation sends the FULL REQUEST BODY on BOTH legs.** Read
   `auth_flow` at `prompter_api.py:61-79`: it sets the negotiate header,
   `response = yield request`, then sets the authenticate header and yields
   *the same request object* again. It does not set httpx's
   `requires_request_body`, and it never sends the negotiate leg empty — which
   is the standard NTLM pattern precisely because of point 4.
4. IIS buffers at most **`uploadReadAheadSize` — 49,152 bytes (48 KB) by
   default** — while completing an authentication handshake. A body larger
   than that cannot be carried across the challenge, and IIS answers
   **401 "Access is denied due to invalid credentials"**. That is the exact
   page in the transcript, and it explains why it is an IIS page at all: the
   request died at the front door and never reached a model.

The margin is thin, which is why this shows up on document work and nowhere
else. Measured locally: `funhouse_agent.system_prompt.SYSTEM_PROMPT` alone is
**16,818 chars** (REACT_PREFIX another 8,720) before any tool schemas,
conversation history, attachment note or tool results. A bare question stays
well under 48 KB; a turn carrying SharePoint tool definitions, a download
result and an attachment note crosses it.

This also fits every observation: credentials valid seconds later (turn 2),
failure only on the tool-heavy turn, and an IIS page rather than a JSON API
error.

**Confirming test (cheap, on-cluster, no SharePoint involved).** Ask a
trivial question with ~60 KB of pasted filler text in it. If it 401s while
the same question without the filler succeeds, size is the variable and this
is settled. Alternatively have the Funhouse admins read back
`uploadReadAheadSize` on the proxy.

**Disposition: PLANNED, four options, in order of how well they fix it.**

1. **Bearer key instead of NTLM (eliminates the whole class).** The SDK notes
   that newer builds accept `services.prompter.api_key`; set it and Prompter
   uses a bearer key, so there is no handshake and no body-replay limit. Check
   support on the cluster with
   `"services.prompter.api_key" in inspect.getsource(PrompterAPI.__init__)`.
   If the cluster's build has it, this is the fix and everything below is
   unnecessary.
2. **Ask the Funhouse admins to raise `uploadReadAheadSize`** on the proxy.
   One server-side setting; every large-prompt user of Prompter hits this.
3. **Warm the connection (client-side, ours).** NTLM authenticates a
   CONNECTION, not a request. Sending one tiny request first leaves the
   pooled connection authenticated, so the large request that follows skips
   the handshake entirely and the read-ahead limit never applies. This is
   within our control and needs no admin action.
4. **Retry once on `AuthenticationError`** — worth having regardless, but on
   its own it only helps if the retry lands on an already-authenticated
   connection, so pair it with 3. There is currently **no retry anywhere** on
   this path: one 401 ends the turn.

## F2 — A raw IIS HTML page is dumped into the chat, and it misled the diagnosis

`core.friendly_turn_error` (`webapp/core.py:1012`) classifies exactly two
cases — recursion-limit and budget-exceeded — and otherwise returns
`f"{type(exc).__name__}: {exc}"`. For this failure that is ~1.5 KB of IIS
markup (doctype, CSS, `<fieldset>`) in the transcript. It is unreadable, it
buries the one useful phrase, and it is the direct cause of the
misattribution in F1: the words "invalid credentials" next to a SharePoint
request read as a SharePoint permissions problem.

**Disposition: PLANNED (small, safe).** Add an auth case to
`friendly_turn_error`: detect `AuthenticationError`/401, strip an HTML body to
its `<title>`/`<h2>`, and say plainly that **the LLM endpoint** rejected the
request — naming the surface that failed, so a model-proxy problem is never
again mistaken for a SharePoint one. Suggested text: "The AI service rejected
the request (401). This is the Funhouse model proxy, not SharePoint or your
files. Re-ask — if it repeats, the cluster's Funhouse credentials need
re-issuing."

## F3 — The turn produced no partial answer despite completed work

The agent had already fetched the document when the model call failed, but the
turn surfaced only the error. The attachment survived (it is in the artifact
list), so the work was not lost — but the transcript reads as a total failure.
Related, already shipped: `turn_jobs.py` keeps turns alive across websocket
death; this is the same idea one layer up (do not discard a turn's completed
tool work on a transient model error).

**Disposition: PLANNED**, folded into F1 — a successful retry makes this moot.

## F4 — Same file, second failure mode: the 22 MB websocket loop **[FIXED today]**

This 23 MB PDF is the same class of payload that put 5.11.2 into a permanent
~14 s websocket reconnect loop when attached through the browser uploader
(root cause and fix in HANDOFF §0a-current, "THE 22 MB UPLOAD LOOP"). Worth
noting that this session avoided that path entirely by fetching from
SharePoint rather than uploading — which is now the more robust route for
large files on 5.11.2 until the fix ships.

## F5 — `_render_pdf_page` has no output-size cap (observation, not this failure)

`planlens/pdf/vision.py:222` renders at `zoom = dpi/72` with no downscale and
no byte budget. Measured on this document at the agent's 220 dpi: page 0 (A4)
→ 0.36 MB PNG / 0.48 MB base64; page 11 (33.1 × 23.4 in drawing sheet, 37.5
megapixels) → **1.60 MB PNG / 2.13 MB base64**. That is well within a normal
vision request, so it did **not** cause this 401 — measured before being
claimed. But nothing bounds it: a larger sheet or a higher `dpi` argument
(`render_region` defaults to 300) scales quadratically with no guard.

**Disposition: BACKLOG.** A max-pixel clamp with proportional downscale, in
planlens, when the drawing work next opens.

---

### Not a problem, checked

- **The document was not over-ingested.** The owner asked it not to ingest 260
  pages immediately; the failure happened before any page analysis.
- **`analyze_pdf_page` cannot kill a turn** — `vision_tools.py:684-692` catches
  `Exception` and returns a JSON error, so the vision path was not the source.
- **Credentials are not mis-scoped.** Turn 2 succeeded seconds later.
