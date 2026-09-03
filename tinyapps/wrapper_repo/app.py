"""GeotechStaffEngineer — TinyApps wrapper entry point.

This is the ONLY substantive file in the Data.State GitHub repo: it loads
secrets from Azure Key Vault into environment variables (the Tiny Apps User
Guide §3.4 required pattern), then hands off to the pip-installed app.
Everything else ships as the ``geotech-staff-engineer`` package from the
Data.State Nexus mirror (see packages.txt).
"""

import os

# ── Key Vault secrets → environment (per Tiny Apps User Guide Fig 3-1) ──────
# Secret names are provisioned per-application by the App Services team.
# Fill KEY_VAULT_URL and the SECRET_MAP once the vault is assigned.
KEY_VAULT_URL = os.environ.get("KEY_VAULT_URL", "")
SECRET_MAP = {
    # "<key-vault-secret-name>": "<env var the app reads>",
    "geotech-prompter-api-key": "GEOTECH_PROMPTER_API_KEY",
    # "geotech-sharepoint-token": "GEOTECH_SHAREPOINT_TOKEN",
}

if KEY_VAULT_URL:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    _client = SecretClient(vault_url=KEY_VAULT_URL,
                           credential=DefaultAzureCredential())
    for secret_name, env_var in SECRET_MAP.items():
        try:
            os.environ.setdefault(env_var,
                                  _client.get_secret(secret_name).value)
        except Exception as exc:                     # surface, don't crash
            print(f"[keyvault] could not load {secret_name}: {exc}")

# ── Hand off to the packaged app ────────────────────────────────────────────
from webapp.tinyapps_entry import main

main()
