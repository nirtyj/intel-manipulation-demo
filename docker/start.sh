#!/bin/bash
# Start the cell inside the container: simulator (8080 API, 9090 viewer), perception (8100), console (8000).
# Stops everything when any of them exits.
set -e
cd /app
SCENE=${SCENE:-scenes/can_flip/scene.xml}

PHYSICALAI_DEPTH_CAMERAS=${PHYSICALAI_DEPTH_CAMERAS:-wrist,overview} \
PHYSICALAI_DEPTH_PERIOD_S=${PHYSICALAI_DEPTH_PERIOD_S:-0.5} \
  /opt/venv/cell/bin/python scripts/sim_start.py start --profile franka_panda --model "$SCENE" \
  --viewer-theme compact --status-json &
/opt/venv/perception/bin/python -m uvicorn v2.perception.service:app --host 0.0.0.0 --port 8100 &
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
/opt/venv/cell/bin/python -m uvicorn v2.cell_api:app --host 0.0.0.0 --port 8000 &

echo "console: http://localhost:8000"
wait -n
kill $(jobs -p) 2>/dev/null
