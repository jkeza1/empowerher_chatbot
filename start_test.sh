#!/bin/bash

# Test version startup script
echo "🧪 Starting EmpowerHer Chatbot - TEST MODE..."

# Set environment variables for clean deployment
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Clean any existing cache
rm -rf ~/.streamlit/static/ 2>/dev/null || true

# Start Streamlit with test app
streamlit run app/test_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
