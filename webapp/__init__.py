"""Streamlit web chat app over the GeotechStaffEngineer deep agent.

Two pages over one shell, declared in :mod:`webapp.pages_entry`: Document
Review and GeotechStaffEngineer (``streamlit run webapp/app.py`` runs the
geotech page alone). It runs locally, on Databricks and on Foundry. The
"Tiny Apps" host runs Document Review alone, from its own package
``aec-document-review`` (``tinyapps/TINYAPPS.md``); :mod:`webapp.tinyapps_entry`
stays here, frozen, until the app split's step 15.

All logic lives in :mod:`webapp.core` and :mod:`webapp.engine_config` (both
import-testable without streamlit); ``webapp/app.py`` is a thin view over them.
This package imports NOTHING at module load beyond the standard library and
the shared core's alias helper (itself standard library only), so
``import webapp`` never requires streamlit or a live model.

The model connections moved into the shared core (BACKLOG INF-32, the app
split's step 11 (a)); their old names here are aliases, so
``webapp.engine_config`` IS ``aec_agent_core.engine_config``.
"""

from aec_agent_core import install_aliases as _install_aliases

#: Old name -> the core's name (lazy; see ``aec_agent_core.install_aliases``).
CORE_ALIASES = {
    # Step 11 (c): the run record, which both pages and the suite write.
    "webapp.activity_log": "aec_agent_core.activity_log",
    "webapp.engine_config": "aec_agent_core.engine_config",
    "webapp.http_trace": "aec_agent_core.http_trace",
    "webapp.palantir_sdk_engine": "aec_agent_core.palantir_sdk_engine",
    "webapp.tinyapps_engine": "aec_agent_core.tinyapps_engine",
}
_install_aliases(CORE_ALIASES)

# The page fills the Prompter connection's hooks (aec_agent_core.hooks): its
# settings are read through Key Vault, and whose key serves a caller is
# decided by the sign-in name (webapp.tinyapps_keys). Strings, imported only
# when the connection first asks. Document Review's page fills them from
# step 13 of the split.
from aec_agent_core.hooks import set_hook as _set_hook

_set_hook("prompter.get_setting", "webapp.tinyapps_settings:get_setting")
_set_hook("prompter.resolve_for", "webapp.tinyapps_keys:resolve_for")
_set_hook("prompter.tester_keys_possible",
          "webapp.tinyapps_keys:tester_keys_possible")

# Which application runs on the core (INF-33 step 13; aec_agent_core.
# app_details): the run record's app version, the default Word and markup
# authors and the runtime messages name GeotechStaffEngineer, as before the
# core named no application. The same values are handed in by
# funhouse_agent, webapp and calc_package, whichever loads first
# (webapp/tests/test_app_details.py checks they agree).
from aec_agent_core.app_details import hand_in as _hand_in_app

_hand_in_app(name="GeotechStaffEngineer", dist="geotech-staff-engineer",
             package="funhouse_agent.deep")

__all__ = ["core", "engine_config"]
