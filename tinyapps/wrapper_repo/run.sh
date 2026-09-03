#!/bin/bash
# TinyApps launch script (Tiny Apps User Guide §4.1 template).
# The App Services team adjusts the final lines to the assigned project
# name/path during onboarding — expect them to edit this file.

python -m streamlit run app.py --server.port "${PORT:-8000}" \
    --server.address 0.0.0.0 --server.headless true \
    --browser.gatherUsageStats false
