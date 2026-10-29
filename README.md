# Talk to the Cell — Intel open-stack manipulation demo

A Franka Panda in Intel's MuJoCo simulator (`openvinotoolkit/physicalai`) picks up an L-shaped part that
lies upside down on a green pad, turns it upright and places it on a blue pad on the other side of a wall.
The same cell runs it two ways, and the console switches between them:

| | **v2 — traditional manipulation** | **v3 — Physical AI policy (ACT)** |
|---|---|---|
| Sensing | Two simulated Intel RealSense RGB-D cameras (scene D435, wrist D405) with a RealSense depth-noise model | The same two cameras, colour only |
| Perception | Grounding DINO → depth mask → SAM-6D 6-DoF pose, multi-view verified (OpenVINO 2026.4.1) | — (end to end) |
| World model | Voxel map fused from both cameras (nvblox-style, CPU), robot self-filtered | — |
| Decision | Grasp library + Pinocchio IK + RRT-Connect over Coal collision checks | ACT (Action Chunking Transformer), trained with `physicalai-train` on the v2 runs |
| Execution | 50 Hz joint streaming through physicalai's `SharedRobot` | physicalai `RobotRuntime` + `PolicySource(InferenceModel)`, OpenVINO on CPU |
| Hardware | Intel Xeon CPU only | Trained on an Intel Xeon (AMX bf16), runs on an Intel Xeon |

No step reads the simulator's object state. The part's pose comes only from RGB-D; the simulator pose is
read after a run, to score it, and in opt-in diagnostics (`CELL_DIAG=1`).

## The pipeline (v2)

1. **Scene camera → coarse location.** Grounding DINO tiny (OpenVINO IR, `"orange and blue block."`) finds
   candidate boxes; a depth mask with table removal keeps each box's 3D points; support-plane reasoning
   rejects flat look-alikes (the blue pad) and anything larger than the part (the wall).
2. **Wrist camera, two viewpoints (±35° around the part).** Each view gives SAM-6D hypotheses (its pose
   estimation model on OpenVINO, with five missing pointnet2 custom ops registered —
   `patches/openvino-sam6d-register-pointnet2-ops.patch`).
3. **Multi-view verification.** Every hypothesis and its half-turns is ICP-refined on the points of both
   views and scored in each view as *surface fit − 2 × free-space violation* (model surface where the
   camera sees through to something farther away). The robot acts only if the views agree
   (score ≥ 0.93); else it looks from a third side, and otherwise refuses. Seen from the robot's side alone,
   the part's foot hides behind its column and SAM-6D proposes the flipped pose; two views settle it.
4. **Map.** Scene and wrist depth fused into 15 mm voxels; the robot's own body filtered out by its
   collision meshes; the part cut out at its estimated pose (the planner models it exactly); the robot's
   current body carved free before planning.
5. **Plan.** Side grasps at the column's centre (turning the part over mirrors the grasp height, and the
   hand needs 50 mm above the table at both ends); collision-free IK, RRT-Connect + shortcutting,
   straight-line approach / lift / lower. Checks: robot vs map and table, robot vs itself, the held part vs
   the map and the arm; 15 mm extra room around the part on free-space paths; the placed part is an
   obstacle on the way home.
6. **Execute and record.** 50 Hz joint targets through `SharedRobot`; at 10 Hz the recorder stores joint
   state, the commanded action and both camera frames.

## The policy (v3)

* **Data.** v2 runs recorded from home to home (the perception look-around is not part of the episode),
  start pose randomised by ±2 cm and ±30° yaw, failures dropped. `v3/convert_lerobot.py` → LeRobot dataset.
* **Training.** `physicalai fit --config v3/act_cpu.yaml` (physicalai-train, ACT, two 120×160 camera inputs,
  2 s action chunks re-planned every second), bf16 on an Intel Xeon with AMX.
* **Export.** `policy.export(..., backend="openvino")` → `exports/act_part_flip`.
* **Deployment.** `v3/run_policy.py`: physicalai `RobotRuntime` at 10 Hz with `IPCamera` on the
  simulator's MJPEG streams, the cell's `SharedRobot`, and `PolicySource(InferenceModel(export))`.

## Run it

```bash
# 1. Simulator (Intel physicalai + patches/physicalai-mujoco-plugin.patch), with RGB-D on both cameras
MUJOCO_GL=osmesa PHYSICALAI_DEPTH_CAMERAS=wrist,overview \
  python scripts/sim_start.py start --profile franka_panda --model scenes/can_flip/scene.xml
# 2. Perception service (perception env: OpenVINO 2026.4.1, open3d, SAM-6D checkout)
uvicorn v2.perception.service:app --port 8100
# 3a. Traditional pipeline, 10 randomised trials
python -m v2.run_v2 --trials 10 --jitter 0.02 --yaw-jitter 30
# 3b. Record training data
python -m v2.run_v2 --trials 70 --jitter 0.02 --yaw-jitter 30 --record runs/dataset
# 4. Train and export (train env: physicalai-train)
python -m v3.convert_lerobot runs/dataset --out datasets/panda_part_flip --repo-id local/panda_part_flip
physicalai fit --config v3/act_cpu.yaml
python -m v3.export_policy --ckpt <checkpoint>
# 5. Policy in the cell
python -m v3.run_policy --trials 10
# 6. Console: both modes, one switch
uvicorn v2.cell_api:app --port 8000      # http://localhost:8000
```

## Results

See [docs/results.md](docs/results.md): every number measured, every fix traced to a logged failure. Demo video of v2: `media/v2_traditional_flip.mp4`.

## Layout

```
scenes/can_flip/scene.xml    the cell: table, pads, wall, L-part (0.30 kg), RealSense D435 scene camera
v2/perception/               sensors (RGB-D + noise), detect (DINO), pose (SAM-6D), pipeline, service
v2/world_map.py              fused voxel map → Coal octree
v2/planner.py                grasp library, collision world, IK, RRT-Connect, flip plan
v2/run_v2.py                 the cell: perceive → map → plan → execute → score; --record
v2/recorder.py, v2/video.py  episodes for training; replay video
v2/cell_api.py, console/     the console (traditional / ACT switch)
v3/                          LeRobot conversion, ACT config, export, policy runner
patches/                     changes to Intel's physicalai plugin and to OpenVINO-SAM-6D
orchestrator/, console/index.html   v1: voice/LLM orchestration over the bucket_sort scene
```
