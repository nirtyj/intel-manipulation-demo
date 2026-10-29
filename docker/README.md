# Running the cell in Docker

One CPU-only image holds the whole demo:

- Intel's MuJoCo simulator (`openvinotoolkit/physicalai` at `ec92b29` plus `patches/physicalai-mujoco-plugin.patch`)
- the OpenVINO 2026.4.1 perception service: Grounding DINO, SAM-6D with its pointnet2 ops built for OpenVINO,
  and CPU-only torch
- the console and cell API
- the ACT policy runtime (physicalai `RobotRuntime` + `InferenceModel`)

Training is not in the image; see `v3/` and `docs/results.md`.

```bash
docker compose -f docker/compose.yaml up --build             # the L-part
CELL_OBJECT=soup_can docker compose -f docker/compose.yaml up # the YCB tomato soup can
```

Open http://localhost:8000. The image is about 5 GB; the perception service needs about 6.5 GB of RAM.

## Weights and assets (mounted, not baked in)

| Host path | What | How to get it |
|---|---|---|
| `models/grounding_dino_tiny.{xml,bin}` | Grounding DINO tiny, OpenVINO IR | `python -m v2.perception.convert_dino` (perception env) |
| `models/sam6d_ir/ov_{fe,pem}_model_cpu.{xml,bin}` | SAM-6D pose model, OpenVINO IR | OpenVINO-SAM-6D's conversion script, from its checkpoints |
| `models/l_part.ply`, `models/soup_can/` | CAD models (in git) | `python scripts/prepare_ycb.py` re-fetches the YCB scan |
| `models/*/templates/` | SAM-6D templates | `python -m v2.perception.templates --cad <model> --out models/<name>` |
| `exports/act_part_flip/` | ACT policy, OpenVINO IR (in git) | `python -m v3.export_policy --ckpt <checkpoint>` |

## Running parts of it

```bash
docker compose -f docker/compose.yaml run --rm cell /opt/venv/cell/bin/python -m v2.run_v2 --trials 5   # needs the services up
docker compose -f docker/compose.yaml exec cell /opt/venv/cell/bin/python -m v3.run_policy --trials 5
```

## Building behind a TLS-inspecting proxy

Put the proxy's CA certificate in `docker/certs/` (ignored by git), and pass the proxy to the build:

```bash
docker build --network host --build-arg HTTPS_PROXY=$HTTPS_PROXY -f docker/Dockerfile -t intel-manip-cell .
```

If Docker Hub rate-limits you, add `--build-arg BASE_IMAGE=mirror.gcr.io/library/ubuntu:24.04`.
