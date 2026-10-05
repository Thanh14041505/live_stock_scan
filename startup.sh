#!/bin/sh
set -eu
cd /workspace
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/_stcore/health; then
  exit 0
fi
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/; then
  exit 0
fi
export STREAMLIT_SERVER_HEADLESS=true
python3 -m streamlit run /workspace/app.py \
  --server.port 8080 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false \
  >>/tmp/app-startup.log 2>&1 &
