"""Streamlit web chat app over the GeotechStaffEngineer deep agent.

Runs standalone locally (``streamlit run webapp/app.py``) and is the submission
target for the "Tiny Apps" hosting environment (Streamlit / Python).

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
    "webapp.engine_config": "aec_agent_core.engine_config",
    "webapp.http_trace": "aec_agent_core.http_trace",
    "webapp.palantir_sdk_engine": "aec_agent_core.palantir_sdk_engine",
    "webapp.tinyapps_engine": "aec_agent_core.tinyapps_engine",
}
_install_aliases(CORE_ALIASES)

__all__ = ["core", "engine_config"]
