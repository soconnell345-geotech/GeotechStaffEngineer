# TinyApps pilot — deployment plan and working notes

**Status (2026-09-03): pilot AWARDED.** CfA's Azure Web Apps Pilot (TinyApps)
selected the team; 30 days to a proof of concept once licenses land. The
Funhouse/Databricks app stays alive as the fast tester and backup — the two
deployments share the same PyPI package and webapp code.

Sources: `Data.State Tiny Apps User Guide v1.1` (2026-03-06) and
`TinyApps Pilot Guidelines` (owner uploads, 2026-09-03). Support:
cfatinyappssuport@state.gov; GitHub: CFAGitHubSupport@state.gov.

## Why this changes everything (environment facts)

- **No driver proxy.** Azure App Service behind Entra ID — the Databricks
  websocket saga (control-frame swallowing, flag plumbing) does not apply.
  Standard Streamlit hosting; `python -m streamlit run <file>` in `run.sh`.
- **No notebook tether.** The app is a real service: no 30-min idle kill, no
  ephemeral-env deletion, no zombie ports, no launch ritual.
- **SBU-capable** (OpenNet Data.State ATO boundary; FISMA Moderate).
- Pilot AI: **free Prompter API key** via Azure AI Foundry — $50/month token
  cap, ONE model. (The in-app budget line and funhouse-gpt-medium habits
  carry over directly.)
- **Native Funhouse (Databricks) connectivity** + Bring-Your-Own-Data via the
  API-First team. SharePoint mirroring likely still possible — verify auth
  path (Key Vault-stored credentials vs delegated OAuth).
- Multi-tenant shared VM, one deployment slot: CfA monitors usage; heavy
  fem2d/Monte-Carlo runs may need throttling courtesy.
- Visibility: listed in the TinyApps SharePoint library, viewable
  Department-wide; access list (names + emails) controls who can OPEN it.

## Deployment architecture: THIN WRAPPER repo

The Data.State GitHub Enterprise repo carries only:

```
app.py           # Key Vault secrets -> env, then webapp.tinyapps_entry.main()
packages.txt     # 'geotech-staff-engineer==X.Y.Z' (+ azure Key Vault libs)
run.sh           # CfA template; python -m streamlit run app.py
README.md
```

Everything else arrives from the **Data.State Nexus** mirror as the released
PyPI package — releases keep flowing exactly as today (owner-gated tags →
PyPI → Nexus), and an app update = bump one pin + ask App Services to sync.
Nexus Firewall note: the DT package-violation sweep applies here too — the
groundhog removal (5.11.2) was a prerequisite; watch the pandas-2.x flag.

## Required code work (the actual build list)

1. **`webapp/tinyapps_entry.py`** — mirror of foundry_entry:
   `GEOTECH_DEPLOYMENT=tinyapps`; no ANTHROPIC key reads; engine surface =
   the pilot Prompter key; skip the Databricks launcher entirely.
2. **Engine: key-based Prompter client.** The Funhouse SDK's PrompterAPI
   authenticates with NTLM creds on-cluster; the pilot hands an API KEY over
   SSL instead. Need the key-auth client details (office-hours question #1)
   → likely a small `PrompterKeyEngine` beside PrompterChatModel, or plain
   OpenAI-compatible endpoint config if Foundry exposes one.
3. **Key Vault secrets adapter** — the guide's pattern: outer app.py reads
   secrets (Prompter key, SharePoint creds if any) and sets env vars before
   the app imports. Template code in the guide Fig 3-1.
4. **Storage strategy.** App Service filesystem persistence is not
   guaranteed across restarts — conversations should mirror to SharePoint
   (existing sharepoint_store) or ADLS; decide after auth-path answer.
5. **packages.txt generation** — a small script emitting the pinned list
   from the released version (plus azure-identity/azure-keyvault-secrets).
6. **Scan hygiene** — GitHub Advanced Security must pass on the wrapper
   repo (tiny surface = easy); CfA CI/CD workflow is mandatory.

## Approval path (their process, our prep)

- MOU → GitHub Enterprise license → init repo (blocked on CfA).
- SIA (1–2 wks): business-justification script is in the guide §5.3.1 —
  draft ready-to-send once app name/data connections are fixed.
- DT CAB (1–2 wks) ONLY if connections leave the Data.State boundary —
  SharePoint/OpenNet? Ask. Avoiding out-of-boundary connections avoids CAB.
- PCR: 1–2 days, theirs. New-tech-stack review: we are plain
  Python/Streamlit = the happy path.
- Code must be "100% complete and runs locally" before engaging — the
  funhouse app already proves this.

## Office-hours / App Services question list

1. Prompter API key auth: client library + endpoint (SSL note in guide §3.4
   — "let the team know if SSL to Prompter is required").
2. Which model does the free tier pin, and can we pick (gpt-medium-class)?
3. SharePoint from App Service: delegated OAuth impossible (no interactive
   login server-side) — service principal via Key Vault? Or ADLS instead?
4. Is a PyPI package install from Nexus (our thin-wrapper plan) the
   preferred pattern vs vendoring code into the repo?
5. Websocket behavior of the App Service front end (should be standard, but
   confirm no aggressive idle timeouts; our keepalives now actually work).
6. Resource envelope on the shared VM (fem2d meshes, Monte Carlo runs).
7. Data.State boundary status of usdos.sharepoint.com (CAB trigger?).

## Monthly obligations

AI Strikeforce check-in (performance, friction, feature asks) + cooperate
with CfA telemetry/optimization. Put findings in this file.
