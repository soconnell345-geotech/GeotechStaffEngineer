"""Document Review: Tiny Apps wrapper entry point.

This is the wrapper's only code file of its own. The app itself arrives by
one of the three routes ``run.sh`` picks (README, "Choosing how the code
arrives"): the released ``aec-document-review`` package from the Nexus
mirror (packages.txt; the default), its wheels in ``wheels/``, or its source
in the ``app_src/`` folder beside this file. ``run.sh`` starts Streamlit on
this file.

Configuration is read by the package itself (``aec_document_review.
tinyapps_settings``), lazily, on first use: a real environment variable
wins; on the App Service the Key Vault named by the ``KV_NAME`` app setting
is read with the managed identity; locally (the test machine, a laptop) the
``.env`` file beside this script is read instead. The settings are listed in
``.env.example`` and the README. Copy ``.env.example`` to ``.env`` for a
local run; never commit ``.env``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Route "source": our packages sit in app_src/. run.sh already puts it first
# on PYTHONPATH; this covers a plain `streamlit run app.py` too. Skipped when
# APP_SOURCE names another route.
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

from aec_document_review.tinyapps_entry import main  # noqa: E402  (after the env is set)

main()
