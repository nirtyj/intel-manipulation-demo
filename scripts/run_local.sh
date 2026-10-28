#!/bin/bash
# Run the simulator and the orchestrator on this machine. Console: http://localhost:8000
# Needs Intel's physicalai checkout at $PHYSICALAI (default ~/physicalai) with
# physicalai-instruction-endpoint.patch applied and `uv sync --package physicalai-mujoco-plugin` done.
set -e
PHYSICALAI=${PHYSICALAI:-$HOME/physicalai}
HERE=$(cd "$(dirname "$0")/.." && pwd)
"$PHYSICALAI/.venv/bin/physicalai-mujoco" start --profile franka_fr3 --scene bucket_sort --seed 42 --viewer-theme compact &
SIM=$!
trap 'kill $SIM' EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
cd "$HERE" && "$PHYSICALAI/.venv/bin/uvicorn" orchestrator.app:app --host 127.0.0.1 --port 8000
