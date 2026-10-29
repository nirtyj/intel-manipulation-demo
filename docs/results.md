# Measured results

Every number here comes from the simulator's ground truth, read **after** each run to score it. Nothing
in the loop reads it. Trials start with the part upside down on the green pad, offset at random by up to
±2 cm in x and y and ±30° in yaw.

## v2: traditional pipeline (no ground truth in the loop)

### Perception: why two views

The first version estimated the pose from a single wrist view. It got 3/10 poses within ±2 cm / ±30°, and
every miss was a large rotation error. Scoring every candidate offline against the truth showed why. Seen
from the robot's side, the part's foot hides behind its column, and SAM-6D never proposes the right pose.
From 35° to the side it nearly always does, and the right pose then scores fit ≈ 1.00 with no free-space
violation.

The cell now looks from −35° and +35°. Every view's hypotheses (and their half-turns) are refined on the
fused points and scored in both views. The robot acts only when the combined score is ≥ 0.93. Right poses
score 0.95–1.00, while half-turned ones score 0.75–0.87.

### 70 randomised runs, recorded as training data

| | |
|---|---|
| Successful flips (upright, ≤ 40 mm from the pad centre) | **52 / 70 (74%)** |
| Runs where the cell acted on a pose | 61. Rotation error median 0.0°, max 6.4°; position error median 0.5 mm, max 7.3 mm |
| Perception refusals (the views did not agree, so the arm did not move) | 9 |
| Part lost during the transfer | 9 |
| Placement error on success | median 4.7 mm, p90 17 mm |
| Time per run (two wrist views, map, plan, execute) | median 80 s; planning median 1.7 s |

Failure analysis of the losses: the fingers closed to 47.4 mm on the 50 mm column (49.8 mm in good grasps),
which means edge contact. The first fast wrist turn of the transfer then pulled the part out. The fix limits
joint speed to 0.5 rad/s while holding and gives the grip 1.5 s to settle.

Fixes found along the way, each from a logged failure:

| Symptom | Cause | Fix |
|---|---|---|
| Hand blocked at the end of the approach | the part's own depth points outside the coarse exclusion sphere | cut the part out of the map at its estimated pose |
| No path from the start pose | a leaked robot self-point on link 1 | carve the robot's current body out of the map before planning |
| Part knocked during the move to pre-grasp | arm tracking lag at speed against a zero-margin plan | 15 mm extra room around the part on free-space paths |
| Placed part knocked on the way home | the placed part was no longer an obstacle | keep the placed part as an obstacle for the retreat and the home path |
| Fit < 0.7 on correct poses | table and pad pixels in the mask when the part fills the box | cut mask points within 6 mm of the table |

## v3: ACT policy (physicalai-train, Intel Xeon only)

### Training on Intel Xeon

| Run | Data | Hardware | Steps | Time | Final loss |
|---|---|---|---|---|---|
| Round 1 | 25 demos, 7,536 frames | AWS c7i.4xlarge, Xeon Platinum 8488C (AMX), bf16 | 8,935 (batch 32) | 80 min | 0.063 |
| Round 2 | 52 demos, 15,179 frames | same | (running) | | |

Step time on a 4-vCPU Xeon slice, with the simulator paused: batch 8 takes 0.50 s in fp32 and 0.45 s in
bf16 (AMX); batch 32 takes 1.6 s in fp32 and 1.3 s in bf16. With the simulator running, its software camera
rendering took all four cores and steps were 10–18× slower. The training box has no simulator on it.

### Deployment

The exported OpenVINO model takes 11–14 ms per inference on CPU. It runs through physicalai's
`RobotRuntime` + `PolicySource` at 10 Hz.

Round 1: 1 successful flip out of 3, then 0 out of 8 in later runs, with and without temporal ensembling.
The policy reaches the part but its final approach is imprecise: the fingers clip the part and close on
air, and since no demonstration ever missed a grasp, it does not recover.

Root cause found while rendering the replay video: the simulator's cameras updated about once a second
during the runs. The scene's 8192-px shadow maps cost ~500 ms per 640×480 frame in software rendering, so
the training images lagged the joint states by up to seconds. With shadows off, both cameras run at ~11 fps
(v2 perception is unaffected: 2/2 flips, 0.4 mm pose error). The dataset is being re-recorded and the policy
retrained on it.
