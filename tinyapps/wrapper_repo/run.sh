#!/bin/bash
# App Service startup (CfA Streamlit starter template, 2026-09-17): install
# packages from the internal feed, then run the app. The App Services team
# fills in the published origin below at deployment.
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# How our code arrives (README, "Choosing how the code arrives"). The app
# setting APP_SOURCE names the route; unset, the first one present wins:
#   source  app_src/ holds our importable packages and goes first on
#           PYTHONPATH; requirements-thirdparty.txt (and app_src/
#           requirements.txt, if present) installs from the feed
#   wheel   wheels/*.whl install with requirements-thirdparty.txt; their
#           own dependencies resolve from the feed
#   nexus   packages.txt (our package + requirements-thirdparty.txt) from
#           the feed
# APP_DRY_RUN=1 prints the route and the commands, then exits before
# installing or starting anything.
SRC_DIR="$APP_DIR/app_src"
shopt -s nullglob
WHEELS=("$APP_DIR"/wheels/*.whl)
shopt -u nullglob

fail() { echo "ERROR: $*" >&2; exit 1; }

HOW="set by APP_SOURCE"
case "${APP_SOURCE:-}" in
    source) [ -d "$SRC_DIR" ] || fail "APP_SOURCE=source but $SRC_DIR/ is missing" ;;
    wheel)  [ ${#WHEELS[@]} -gt 0 ] || fail "APP_SOURCE=wheel but $APP_DIR/wheels/ holds no .whl file" ;;
    nexus)  [ -f "$APP_DIR/packages.txt" ] || fail "APP_SOURCE=nexus but $APP_DIR/packages.txt is missing" ;;
    "")
        HOW="chosen by what the repo holds"
        if [ -d "$SRC_DIR" ]; then APP_SOURCE=source
        elif [ ${#WHEELS[@]} -gt 0 ]; then APP_SOURCE=wheel
        else APP_SOURCE=nexus
        fi ;;
    *) fail "APP_SOURCE=$APP_SOURCE is not one of: source, wheel, nexus" ;;
esac
export APP_SOURCE   # app.py reads it

pip_install() {
    if [ -n "$PIP_INDEX_URL" ]; then
        set -- "$@" --index-url "$PIP_INDEX_URL"
    else
        echo "WARNING: PIP_INDEX_URL not set; using default package source"
    fi
    if [ "${APP_DRY_RUN:-}" = "1" ]; then
        echo "DRY RUN: python -m pip install $*"
    else
        python -m pip install "$@"
    fi
}

case "$APP_SOURCE" in
    source)
        echo "Code route: source ($SRC_DIR first on PYTHONPATH; $HOW)"
        REQS=(-r "$APP_DIR/requirements-thirdparty.txt")
        [ -f "$SRC_DIR/requirements.txt" ] && REQS+=(-r "$SRC_DIR/requirements.txt")
        echo "Installing Python packages: third-party pins (+ app_src/requirements.txt)..."
        pip_install "${REQS[@]}"
        export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
        ;;
    wheel)
        echo "Code route: wheel (${WHEELS[*]##*/}; $HOW)"
        echo "Installing Python packages: our wheels + third-party pins..."
        pip_install "${WHEELS[@]}" -r "$APP_DIR/requirements-thirdparty.txt"
        ;;
    nexus)
        echo "Code route: nexus (packages.txt from the feed; $HOW)"
        echo "Installing Python packages from packages.txt..."
        pip_install -r "$APP_DIR/packages.txt"
        ;;
esac

if [ "${APP_DRY_RUN:-}" = "1" ]; then
    echo "DRY RUN: PYTHONPATH=${PYTHONPATH:-} python -m streamlit run $APP_DIR/app.py --server.port ${PORT:-8000} ..."
    exit 0
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
#
# --client.toolbarMode viewer hides Streamlit's own "Deploy" button and
# developer menu (they link to Streamlit's public hosting, meaningless here);
# the Databricks launcher sets the same. --server.fileWatcherType none stops
# Streamlit watching the installed package for edits.
exec python -m streamlit run "$APP_DIR/app.py" \
    --server.port "${PORT:-8000}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --client.toolbarMode viewer \
    --server.fileWatcherType none \
    --server.corsAllowedOrigins "https://CHANGE-ME"   # the published address CfA gives you
    # --server.corsAllowedOrigins "https://CHANGE-ME-TOO"  # repeat per additional published origin
