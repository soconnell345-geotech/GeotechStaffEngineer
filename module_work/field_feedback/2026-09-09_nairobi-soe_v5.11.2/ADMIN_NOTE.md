# Draft note to the Funhouse admins — Prompter 401 on large requests

Derived from F1 in `FINDINGS.md`. Copy-paste as-is; the only things to check
before sending are the timestamp/timezone and who it should go to.

---

**Subject:** Prompter returns 401 on large requests — looks like the IIS upload
read-ahead limit, not credentials

Hi [team],

We're getting a reproducible 401 from the Prompter LLM proxy on our Databricks
cluster. We think it's a gateway configuration issue rather than an account
problem, and there are two quick questions at the bottom.

**Symptom.** A request came back as an IIS error page — *"401 - Unauthorized:
Access is denied due to invalid credentials."* — and killed the request. A
further request on the same credentials, seconds later, succeeded. So the
account itself is fine. Observed 2026-09-09, ~11:27 local, model
`funhouse-gpt-high`. It has not recurred since, which is why we are asking
rather than reporting a hard repro.

**What we have narrowed it to.** It is intermittent, and it is not a
permissions problem — a request on the same credentials seconds later
succeeded, and a much larger request on a later day also succeeded. So
something is failing transiently in the authentication handshake rather than
in the account.

One thing we noticed that may be relevant: Prompter authenticates with NTLM on
the httpx transport (`ExHttpNtlmAuth` in
`funhouse/services/prompter/prompter_api.py`), and that implementation sends
the full request body on **both** legs of the challenge-response handshake
rather than sending the first leg empty. That makes every large request cross
the gateway twice while unauthenticated, which is more exposed to a connection
being recycled mid-handshake — and IIS's `uploadReadAheadSize` limit (49,152
bytes by default) governs how much body it will buffer while authenticating.
We could not reproduce a clean size threshold, so we are NOT claiming that is
the cause — only that it is worth ruling in or out from your side.

**Two asks:**

1. Does our SDK build support `services.prompter.api_key` (bearer-key auth on
   the prompter backend)? If it does, we'll switch to it — that removes the
   handshake and this failure mode along with it.
2. Is `uploadReadAheadSize` on the proxy's IIS front end at its 49,152-byte
   default? If so, raising it would remove one candidate explanation cheaply,
   and would help anyone sending long prompts through Prompter.

**FYI for whoever maintains the SDK:** the conventional client-side pattern is
to send the negotiate leg with an empty body and attach the payload only to the
authenticated leg. That keeps the unauthenticated round-trip small regardless
of prompt size, and is worth doing on its own merits.

Could you check the gateway log for that timestamp and tell us what it
recorded? Happy to run any diagnostic that would help.

Thanks,
Sean
