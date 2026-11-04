"""GeotechStaffEngineer — Tiny Apps wrapper entry point.

This is the wrapper's only code file of its own. The app itself arrives by
one of the three routes ``run.sh`` picks (README, "Choosing how the code
arrives"): our packages in the ``app_src/`` folder beside this file, our
wheels in ``wheels/``, or the released ``geotech-staff-engineer`` package
from the Data.State Nexus mirror (see packages.txt). ``run.sh`` starts
Streamlit on this file.

Configuration follows CfA's ``settings.py`` pattern, built into the package
(``webapp.tinyapps_settings``): a real environment variable wins; on the App
Service the Key Vault named by the ``KV_NAME`` app setting is read with the
managed identity; locally (dosdev, a laptop) the ``.env`` file beside this
script is read instead. So there is nothing to load here — the packaged app
reads ``PROMPTER_URL`` / ``PROMPTER_MODEL`` / ``PROMPTER_API_KEY`` /
``PROMPTER_CA_BUNDLE`` and ``GRAPH_TENANT_ID`` / ``GRAPH_CLIENT_ID`` /
``GRAPH_CLIENT_SECRET`` / ``SHAREPOINT_SITE_URL`` itself, lazily, on first use.
Copy ``.env.example`` to ``.env`` for a local run; never commit ``.env``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Route "source": our packages sit in app_src/. run.sh already puts it first
# on PYTHONPATH; this covers a plain `streamlit run app.py` on dosdev too.
# Skipped when APP_SOURCE names another route.
_SRC = os.path.join(_HERE, "app_src")
if (os.environ.get("APP_SOURCE", "source") == "source"
        and os.path.isdir(_SRC) and _SRC not in sys.path):
    sys.path.insert(0, _SRC)

# Local runs: the .env next to this file is the configuration source.
os.environ.setdefault("GEOTECH_DOTENV", os.path.join(_HERE, ".env"))

# Where conversations and working files live on this host. App Service keeps
# /home across restarts; anywhere else the package's default (~/.geotech_webapp)
# is used. Set the app setting GEOTECH_WEBAPP_DATA to override.
if os.path.isdir("/home") and "GEOTECH_WEBAPP_DATA" not in os.environ:
    os.environ["GEOTECH_WEBAPP_DATA"] = "/home/data/geotech_webapp"

from webapp.tinyapps_entry import main  # noqa: E402  (after the env is set)

main()
