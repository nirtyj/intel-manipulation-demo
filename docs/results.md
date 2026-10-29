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

### Second collection: cameras at ~11 fps, RGB-D timing fixed, payload speed limit

| | |
|---|---|
| Successful flips | **53 / 75 (71%)** |
| Runs where the cell acted on a pose | 62. Rotation error median 0.0°, max 6.4°; position error median 0.5 mm, max 8.2 mm |
| Perception refusals | 13 (12 runs before the RGB-D timing fix accounted for 3 of them) |
| Part lost or knocked over | 8 |
| Placement error on success | median 4.2 mm, p90 18 mm |
| Time per run | median 86 s |

Two simulator-side issues surfaced only because the second collection was examined frame by frame:

1. **Stale cameras.** The scene's 8192-px shadow maps made every 640×480 colour frame cost ~500 ms in
   software rendering. While the arm moved, the cameras updated about once a second, so the first dataset's
   images lagged its joint states. Without shadow maps both cameras run at ~11 fps.
2. **Depth and pose out of step.** The plugin read the camera pose *after* the ~50 ms depth render while
   physics kept stepping. With a settling arm, image and pose were a few mm apart. The free-space check
   then flagged ~10% of a correct pose, pushing it under the 0.93 bar. The pose is now read when the scene
   is captured, and perception waits for a capture newer than its request.

### Third collection: the same transfer every time

The transfer goes straight up to a height read from the fused map, makes one move across with the
half-turn, and comes straight down. Dense waypoints execute as one smooth curve, and the grasp dwell is
0.8 s.

| | |
|---|---|
| Successful flips | **63 / 80 (79%)** |
| Runs where the cell acted on a pose | 67. Rotation error median 0.0°, max 5.8°; position error median 0.5 mm, max 4.6 mm |
| Perception refusals | 13 |
| Part lost or knocked over | 3 (down from 8–9 per collection) |
| Placement error on success | median 7.6 mm, p90 17 mm |
| Time per run | median 83 s; planning median 1.4 s |

Across all three collections, the cell acted on 190 poses, and none of them was wrong.

## v3: ACT policy (physicalai-train, Intel Xeon only)

Every policy is trained with `physicalai fit` (ACT, two 120×160 camera inputs, 2 s action chunks) in bf16
on AWS c7i Xeon Platinum 8488C boxes (AMX), and exported to OpenVINO. In the cell it runs through
physicalai's `RobotRuntime` + `PolicySource(InferenceModel)` at 10 Hz, with ACT's temporal ensembling as a
chunk smoother. It is scored by the simulator afterwards.

| Round | Demonstrations | Training | In the cell | What it does |
|---|---|---|---|---|
| 1 | 25, cameras ~1 fps, RRT transfers | 8.9k steps, 39 epochs, 16 vCPU, 80 min | 1/11 | reaches the part, closes on air |
| 3a | 27, cameras 11 fps, RRT transfers | 11.6k steps, 36 epochs, 2 h | 1/6, then 1/1 from the console | grasps; then stalls or wanders in the transfer |
| 3b | 53, cameras 11 fps, RRT transfers | 11.2k steps, 18 epochs, 8 vCPU, 3 h | 0/6 | hovers at the pre-grasp |
| 4a | 30, cameras 11 fps, same transfer every time | 10.8k steps, 34 epochs, 16 vCPU, 2 h, loss 0.037 | 4/8 at first; **44/100** over 100 randomised runs, placed a median 12 mm from centre, ~54 s per run | grasps, carries over the wall, places upright; every miss leaves the part not upright |
| 4b | 63, cameras 11 fps, same transfer every time | 10.3k steps, 16 epochs, 8 vCPU, 3 h, loss 0.045 | 1/20 | mostly turns over in mid-air or drops at the grasp |
| 4c | 63, cameras 11 fps, same transfer every time | 20k steps, 30 epochs, 16 vCPU, 3.5 h, loss 0.035 | 7/20 (3/8 on 4a's 5/8 seed) | more passes over the larger set recovered most of 4b's loss, but did not beat 4a |

Open-loop check (round 3b on its own training episodes, frames fed exactly as the runtime feeds them):

- The first predicted action is within 1.5° per joint of the demonstration on average (max 9°).
- Blanking the images moves the output by 9°.

So the export and the runtime pipeline match training, and the policy uses vision. What fails is closed-loop
precision: 1.5° at the shoulder is about 1.5 cm at the fingers, against 15 mm of clearance around the
column.

What moved the policy, in order:

1. **Fresh camera images.** Round 1 trained on images that lagged the joints by up to seconds and grasped
   air. With 11 fps cameras, round 3a closes on the part.
2. **Consistent demonstrations.** Round 3a grasped, then stalled or wandered, because the RRT transfers took
   a new route every run and the policy averaged them. With the same amount of data but the fixed route,
   round 4a went from 1/6 to 4/8.
3. **Epochs over data.** Twice the data at half the epochs lost both times: 3b (53 demos, 18 epochs) scored 0/6 against 3a's 1/6, and 4b (63 demos, 16 epochs) scored 1/20 against 4a's 44/100. At a fixed CPU training budget, more passes over a smaller consistent set beat one pass over more. Round 4c then gave the 63-demo set 30 epochs (20k steps): it scored 7/20, close to but not above 4a, so the deployed policy is still 4a.

### Deployment speed

The exported OpenVINO model takes 11–14 ms per inference on CPU.

The simulator renders its cameras in software, so an MJPEG reader blocked each tick and gave ~1 Hz control.
`v3/sim_camera.py` is a physicalai `Camera` on the latest frame, prepared the way the recorder prepared
it, which gives 10 Hz control.
