"""Streamlit web chat app over the GeotechStaffEngineer deep agent.

Runs standalone locally (``streamlit run webapp/app.py``) and is the submission
target for the State Dept "TinyApp" hosting environment (Streamlit / Python).

All logic lives in :mod:`webapp.core` and :mod:`webapp.engine_config` (both
import-testable without streamlit); ``webapp/app.py`` is a thin view over them.
This package imports NOTHING at module load beyond the standard library, so
``import webapp`` never requires streamlit or a live model.
"""

__all__ = ["core", "engine_config"]
