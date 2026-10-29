# Demo runbook

## Bring the cell up (one machine, CPU only)

```bash
# 1. Simulator: Intel's MuJoCo plugin, this cell, RGB-D on both cameras, compact embedded viewer
cd talk-to-the-cell
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PHYSICALAI_DEPTH_CAMERAS=wrist,overview PHYSICALAI_DEPTH_PERIOD_S=0.5 \
  ~/physicalai/.venv/bin/python scripts/sim_start.py start --profile franka_panda \
  --model scenes/can_flip/scene.xml --viewer-theme compact --status-json
# 2. Perception service (OpenVINO 2026.4.1: Grounding DINO + SAM-6D)
~/envs/perception/bin/python -m uvicorn v2.perception.service:app --port 8100
# 3. Console + cell API
~/physicalai/.venv/bin/python -m uvicorn v2.cell_api:app --port 8000
```

Open `http://localhost:8000`. The left pane is Intel's live 3D viewer, and you can drag the part there. On
the right are the mode switch, Run / STOP / Reset, the stage rail, the perception images and the results.

## Run order (about 5 minutes)

1. **Reset part.** It goes back upside down on the green pad at a random offset (±2 cm, ±30°). The banner
   shows the offset; neither mode is told it.
2. **Traditional → Run flip** (about 80 s). Point at the scene detection, then the two wrist views and the
   6-DoF pose with both views agreeing. Then the plan: grasp from the library, clearance, RRT time. Then
   the flip around the wall.
3. **Reset part**, then **ACT → Run flip** (about 55 s; it succeeds about half the time, so have the video ready). Point out that nothing runs between the pixels and
   the joints except one OpenVINO network, about 12 ms per step on the CPU.
4. Results table: runs, successes and median time per mode, scored by the simulator afterwards.

Backup if live runs misbehave: `media/v2_traditional_flip.mp4` and `media/v3_act_flip.mp4`.

## Talk track (about 90 s)

- "Same cell, same task, two ways of solving it. Everything runs on Intel's open stack and CPUs."
- **Traditional.** "Simulated RealSense RGB-D, Grounding DINO and SAM-6D on OpenVINO, a fused voxel map,
  Pinocchio IK, and an RRT planner checked with Coal. No ground truth anywhere in the loop. It's precise
  and explainable, but every stage is engineered, and it takes about 80 s."
- **Physical AI.** "The traditional cell recorded its own successful runs: 168 demonstrations over three
  collections, with no teleoperation. The deployed policy learned from 30 of them. physicalai-train trained ACT on them on a Xeon with AMX, and it was
  exported to OpenVINO. It runs in Intel's RobotRuntime on the same cell: images and joint angles in,
  joint targets out."
- **The point.** "The classical stack becomes the data engine for the learned one, and the whole loop
  (sense, plan, record, train, deploy) stays on Intel hardware and Intel's open-source runtime."

## Questions to expect, with answers backed by the measurements

**Why SAM-6D, not FoundationPose?** It's a CAD-based, zero-shot model with an OpenVINO port, so it runs on
CPU. FoundationPose wants CUDA for its refiner. The cost: SAM-6D's answers vary between samplings, and
from one view it proposed flipped poses. So the cell scores several hypotheses and verifies them across
two views.

**How do you know the pose is right without ground truth?** Geometric consistency. For each candidate:
what share of each view's depth points lies on the posed model, and how much of the model sits where a
camera sees through to something farther away. Right poses score 0.95–1.00, half-turned ones 0.75–0.87.
Below 0.93 the cell looks from a third side or refuses. In 70 runs it acted 61 times, and was right all
61 times (rotation error ≤ 6.4°).

**Why does the traditional cell still fail 1 in 4?** Half of its failures are refusals: the views
disagree, so the arm doesn't move. The other half were grasps that slipped during fast wrist turns.
That's fixed with a payload speed limit; see `docs/results.md`.

**Why ACT?** It learns from about 50 demonstrations, it's small enough to train on CPUs, and
physicalai-train exports it straight to OpenVINO. The alternatives:

- Diffusion Policy: better with multimodal demos, but slower to train and to sample.
- SmolVLA / Pi0.5: language-conditioned and they generalise more, but they need GPU-class compute to
  fine-tune.
- Plain BC: no action chunking, so errors compound.

**Why train on CPU?** That was the point of the exercise: stay on Intel. One bf16 step at batch 32 takes
about 0.5 s on a 16-vCPU Sapphire Rapids (AMX), so ~10k steps take ~1.5 h. A training GPU has ~100× the
matrix throughput and ~10× the memory bandwidth. Intel's own answers at scale are Gaudi (HPU) and its
GPUs (`xpu`); physicalai-train ships an XPU config. Inference is where Xeon + OpenVINO is clearly at home:
12 ms per policy step.

**What surprised you?**
- The simulator's cameras were updating once a second during the first data collection. Shadow maps in
  software rendering cost about 500 ms per frame. The first policy learned from stale images and missed
  its grasps.
- Camera pose and depth image were captured 50 ms apart while the arm settled, which put right poses just
  under the verification bar.

Both were found from logs and fixed in the plugin patch. This is the kind of sim-to-pipeline issue that
silently caps a learned policy.

**What next?**
- Language on top (the v1 "talk to the cell" layer).
- More demonstrations with recovery behaviour (a deliberately perturbed grasp).
- Domain randomisation of lighting and textures before going to real hardware.
- An XPU or Gaudi run of the same config.
