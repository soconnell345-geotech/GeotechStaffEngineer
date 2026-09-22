#!/bin/bash
# App Service startup (CfA Streamlit starter template, 2026-09-17): install
# packages from the internal feed, then run the app. The App Services team
# fills in the published origin below at deployment.
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Installing Python packages from packages.txt..."
if [ -n "$PIP_INDEX_URL" ]; then
    python -m pip install -r "$APP_DIR/packages.txt" --index-url "$PIP_INDEX_URL"
else
    echo "WARNING: PIP_INDEX_URL not set; using default package source"
    python -m pip install -r "$APP_DIR/packages.txt"
fi

echo "Starting Streamlit..."
# --server.port must follow $PORT: App Service probes port 8000 and recycles
# the container (503) if nothing answers there; Streamlit's own default is 8501.
#
# --server.corsAllowedOrigins must list every URL the browser SHOWS for this
# app (the IIS-published ones, one flag per origin). Through IIS/ARR the
# browser's Origin is the published URL while Streamlit sees the internal App
# Service hostname, and without this flag Streamlit rejects the WebSocket
# ("Rejecting WebSocket connection with disallowed Origin or Host header") —
# the app looks started but the page never loads. This keeps CORS and XSRF
# protection ON; do not fall back to disabling them.
#
# Health check path for the App Service: /_stcore/health (Streamlit serves no
# /healthz). If IIS publishes the app under a sub-path, set the app setting
# STREAMLIT_SERVER_BASE_URL_PATH to match.
exec python -m streamlit run "$APP_DIR/app.py" \
    --server.port "${PORT:-8000}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.corsAllowedOrigins "https://CHANGE-ME.cfaapp.cfa.state.sbu"
    # --server.corsAllowedOrigins "https://CHANGE-ME.app.data.state.sbu"  # repeat per additional published origin
