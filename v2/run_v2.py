"""v2: flip the L-part from perception alone. Scene + wrist RealSense RGB-D -> Grounding DINO -> SAM-6D ->
fused map -> grasp library + IK + RRT-Connect -> execution through physicalai's runtime -> re-perception.

The simulator's object pose is read only after each trial, to score it.

    python -m v2.run_v2 [--trials N] [--jitter 0.02] [--yaw-jitter 30]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pinocchio as pin

from v2.arm import CONTROL_HZ, GRIP_OPEN, HOME_DEG, V_MAX, V_MAX_HELD, PandaKinematics, Segment, interpolate, polyline
from v2.look import solve_look_pose, tcp_to_camera
from v2.perception.sensors import fetch_rgbd
from v2.objects import OBJECT
from v2.planner import PART_BOXES, CollisionWorld, FlipPlan, plan_flip
from v2.world_map import ROBOT_MARGIN, WorldMap, robot_surface_samples

SIM = "http://127.0.0.1:8080"
PERCEPTION = "http://127.0.0.1:8100"
ROBOT = "mujoco-franka_panda-follow"
PART_START = OBJECT.start  # upside down on the green pad
PLACE_XY = np.array([0.48, -0.24])
RUNS = Path(__file__).resolve().parents[1] / "runs" / "v2"
DIAG = os.environ.get("CELL_DIAG") == "1"
DENSE_STEP = 0.25  # rad: path steps shorter than this are points on one curve, not separate moves
MIN_FIT = 0.93  # mean over views of (surface fit - 2 x free-space violation) before the robot acts on a pose
# (measured: right poses score 0.95-1.00, half-turned ones 0.75-0.87)
VIEWPOINTS = ((35.0, 0.28), (-35.0, 0.28), (0.0, 0.30))  # wrist look-at viewpoints: azimuth (deg), distance (m)


def post(url: str, body: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers={"content-type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def truth() -> np.ndarray:
    """Simulator ground truth: scoring only."""
    o = next(o for o in json.load(urllib.request.urlopen(SIM + "/objects")) if o["joint"] == "part:joint")
    w, x, y, z = o["wxyz"]
    T = np.eye(4)
    T[:3, :3] = pin.Quaternion(w, x, y, z).toRotationMatrix()
    T[:3, 3] = o["position"]
    return T


def reset_part(rng: np.random.Generator | None, jitter: float, yaw_jitter_deg: float) -> tuple[float, float, float]:
    """Put the part back upside down on the green pad, offset at random; returns (dx m, dy m, yaw deg)."""
    pose = json.loads(json.dumps(PART_START))
    dx = dy = yaw_deg = 0.0
    if rng is not None:
        dx, dy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
        yaw_deg = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
        pose["position"][0] += dx
        pose["position"][1] += dy
        q = pin.Quaternion(pin.utils.rotate("z", np.radians(yaw_deg)) @ pin.Quaternion(0.0, 1.0, 0.0, 0.0).toRotationMatrix())
        pose["wxyz"] = [q.w, q.x, q.y, q.z]
    post(SIM + "/objects/part:joint/pose", pose)
    time.sleep(1.2)
    return dx, dy, yaw_deg


@dataclass
class Trial:
    events: list = field(default_factory=list)

    def log(self, stage: str, **kw) -> None:
        self.events.append({"t": round(time.time(), 3), "stage": stage, **kw})
        print(f"  {stage:22s} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


class Cell:
    def __init__(self, robot, kin: PandaKinematics) -> None:
        self.robot, self.kin = robot, kin
        self.world = CollisionWorld(kin)
        self.grip = GRIP_OPEN
        self.last_action: np.ndarray | None = None  # what the recorder logs as the action
        self.recorder = None
        self.record_mode = "policy"  # "policy": record only the flip, from home; "full": also the look-around
        self._perceiving = False

    def q(self) -> np.ndarray:
        return np.radians(np.asarray(self.robot.get_observation().joint_positions, float)[:7])

    def stream(self, traj: list[np.ndarray]) -> None:
        period = 1.0 / CONTROL_HZ
        tick = time.perf_counter()
        for qk in traj:
            self.last_action = np.r_[np.degrees(qk), self.grip]
            self.robot.send_action(self.last_action, goal_time=2 * period)
            tick += period
            time.sleep(max(0.0, tick - time.perf_counter()))

    def move_to(self, q_goal: np.ndarray, duration: float = 2.0, settle: float = 0.5) -> np.ndarray:
        traj = interpolate(self.kin, Segment("move", "joint", duration, q_goal=q_goal), self.q())
        self.stream(traj + [traj[-1]] * int(settle * CONTROL_HZ))
        return traj[-1]

    def mark(self, label: str, info: dict | None = None, cameras: list[str] = ()) -> None:
        if self.recorder is None:
            return
        images = {}
        for cam in cameras:
            try:
                images[cam] = urllib.request.urlopen(f"{PERCEPTION}/annotated/{cam}.png", timeout=5).read()
            except OSError:
                pass
        self.recorder.mark(label, info, images)

    def pause_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.pause()

    def resume_recording(self) -> None:
        if self.recorder is not None and not (self._perceiving and self.record_mode == "policy"):
            self.recorder.resume()

    def execute(self, plan: FlipPlan, trial: Trial) -> None:
        q = self.q()
        holding = False
        for step in plan.steps:
            t0 = time.perf_counter()
            self.mark(step.name)
            if step.kind == "grip":
                self.grip = step.grip
                holding = step.grip < GRIP_OPEN
                self.stream([q] * int(step.duration * CONTROL_HZ))
            elif step.kind == "path":
                # Runs of short steps (a straight tool line solved point by point) move as one smooth curve;
                # each long step (an RRT edge) gets its own start and stop.
                v = V_MAX_HELD if holding else V_MAX
                group = [q]
                for wp in step.waypoints[1:]:
                    if np.max(np.abs(wp - group[-1])) > DENSE_STEP and len(group) > 1:
                        traj = polyline(group, v)
                        self.stream(traj)
                        q = traj[-1]
                        group = [q]
                    group.append(wp)
                    if np.max(np.abs(wp - group[-2])) > DENSE_STEP:
                        traj = polyline(group, v)
                        self.stream(traj)
                        q = traj[-1]
                        group = [q]
                if len(group) > 1:
                    traj = polyline(group, v)
                    self.stream(traj)
                    q = traj[-1]
            else:
                traj = interpolate(self.kin, Segment(step.name, "cart", step.duration, goal=step.goal), q)
                self.stream(traj)
                q = traj[-1]
            extra = {}
            if DIAG:  # diagnostics only: where the part really is after each step (never used for control)
                gt = truth()
                q_meas = self.q()
                tcp = self.kin.fk(q_meas)
                rel = tcp.inverse() * pin.SE3(gt[:3, :3], gt[:3, 3])
                extra = {"diag_part_xyz": np.round(gt[:3, 3], 3).tolist(), "diag_part_up": round(float(gt[2, 2]), 2),
                         "diag_part_in_tcp_mm": np.round(rel.translation * 1000, 1).tolist(),
                         "diag_cmd_vs_meas_mm": round(float(np.linalg.norm(self.kin.fk(q).translation - tcp.translation) * 1000), 1)}
            trial.log("execute", step=step.name.replace(" ", "_"), ms=round((time.perf_counter() - t0) * 1000), **extra)


def perceive(cell: Cell, trial: Trial, wmap: WorldMap) -> np.ndarray | None:
    kin = cell.kin
    q = cell.move_to(np.radians(HOME_DEG), duration=2.0)
    cell.pause_recording()
    c = post(PERCEPTION + "/coarse")
    trial.log("scene_detect", score=c["detection_score"], ms=c["timings_ms"].get("total"))
    cell.mark("scene detection", {"score": c["detection_score"], "ms": c["timings_ms"].get("total")}, ["overview"])
    if c["position"] is None:
        return None
    coarse = np.array(c["position"])
    scene = fetch_rgbd("overview")
    wrist0 = fetch_rgbd("wrist", noisy=False)
    T_tcp_cam = tcp_to_camera(kin, q, wrist0.T_world_cam)
    exclude = [(coarse, 0.10)]
    wmap.integrate(scene, robot_surface_samples(cell.world.robot_geom, cell.world.model, cell.world.data,
                                                pin.GeometryData(cell.world.robot_geom), q), exclude)
    # Look at the part from two sides of the robot's reach. Each wrist view gives SAM-6D hypotheses; the
    # pose is the one both views' depth agrees with. A third view only if they still disagree.
    post(PERCEPTION + "/views/reset")
    r, added = None, 0
    for az, dist in VIEWPOINTS:
        look = solve_look_pose(kin, q, T_tcp_cam, coarse, distance=dist, azimuth_deg=az)
        if look is None:
            trial.log("look", azimuth=az, ok=False)
            continue
        cell.resume_recording()
        q = cell.move_to(look[0], duration=1.8, settle=0.8)
        cell.pause_recording()
        v = post(PERCEPTION + "/views/add")
        trial.log("wrist_view", azimuth=az, added=v["added"], sam6d=v.get("sam6d_score"), ms=v.get("timings_ms", {}).get("total"))
        wrist = fetch_rgbd("wrist")
        wmap.integrate(wrist, robot_surface_samples(cell.world.robot_geom, cell.world.model, cell.world.data,
                                                    pin.GeometryData(cell.world.robot_geom), q), exclude)
        added += int(v["added"])
        if added < 2:
            continue
        r = post(PERCEPTION + "/views/pose")
        trial.log("six_dof", views=r["views"], fit=r["fit"], score=r["pose_score"], ms=r["timings_ms"].get("total"),
                  per_view=r["per_view"])
        cell.mark("6-DoF pose", {"views": r["views"], "fit": r["fit"], "pose_score": r["pose_score"],
                                 **r["timings_ms"]}, ["wrist"])
        if r["T_world_obj"] is not None and r["fit"] >= MIN_FIT:
            break
    if DIAG:  # diagnostics only: keep the views, and the simulator's pose next to them, for offline study
        stamp = int(time.time())
        post(PERCEPTION + f"/views/save?path={RUNS.resolve()}/views_{stamp}.npz")
        np.save(RUNS / f"views_{stamp}_truth.npy", truth())
    if r is None or r["T_world_obj"] is None or r["fit"] < MIN_FIT:
        return None  # the views do not agree on a pose: never guess
    T = np.array(r["T_world_obj"])
    cut = wmap.cut_object(T, PART_BOXES)
    trial.log("map", voxels=len(wmap.counts), cut=cut, table_z=round(float(wmap.table_z), 4), frames=wmap.frames)
    return T


def run_trial(cell: Cell, i: int, rng, args) -> dict:
    reset_part(rng, args.jitter, args.yaw_jitter)
    return {"trial": i, **flip(cell, Trial())}


def flip(cell: Cell, trial: Trial) -> dict:
    """One flip from wherever the part is: perceive, map, plan, execute, score."""
    t_start = time.perf_counter()
    if cell.recorder is not None:
        cell.recorder.start(cell.robot, lambda: cell.last_action, paused=cell.record_mode == "policy")
    wmap = WorldMap()
    cell._perceiving = True
    try:
        T = perceive(cell, trial, wmap)
    finally:
        cell._perceiving = False
    result = {"success": False}
    gt0 = truth()
    if T is None:
        result["failure"] = "perception"
        return {**result, "events": trial.events}
    result["pose_error_mm"] = round(1000 * float(np.linalg.norm(T[:3, 3] - gt0[:3, 3])), 1)
    result["pose_error_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ gt0[:3, :3]) - 1) / 2, -1, 1)))), 1)
    np.savez(RUNS / f"planning_input_{int(time.time())}.npz", T=T, occupied=wmap.occupied(), table_z=wmap.table_z, q=cell.q(),
             counts_keys=np.array(list(wmap.counts.keys())), counts_vals=np.array(list(wmap.counts.values())))
    if cell.recorder is not None and cell.record_mode == "policy":
        # Training episodes start at home: the policy learns the flip from its cameras, not the look-around.
        cell.move_to(np.radians(HOME_DEG), duration=2.0)
    # The robot's own body, where it stands now, is free space by definition.
    carved = wmap.carve(robot_surface_samples(cell.world.robot_geom, cell.world.model, cell.world.data,
                                              pin.GeometryData(cell.world.robot_geom), cell.q()), ROBOT_MARGIN)
    if carved:
        trial.log("map", carved_robot_voxels=carved)
    try:
        plan = plan_flip(cell.kin, cell.world, wmap, cell.q(), T, PLACE_XY)
    except RuntimeError as exc:
        trial.log("plan", ok=False, why=str(exc)[:160])
        result["failure"] = "planning"
        return {**result, "events": trial.events}
    trial.log("plan", grasp=plan.grasp, ms=round(plan.plan_ms), clearance_mm=round(1000 * plan.clearance_m))
    cell.mark("plan", {"grasp": plan.grasp, "plan_ms": round(plan.plan_ms), "clearance_mm": round(1000 * plan.clearance_m),
                       "voxels": len(wmap.counts)})
    cell.resume_recording()
    cell.execute(plan, trial)
    time.sleep(1.0)
    gt = truth()
    upright = float(gt[2, 2])
    on_pad = float(np.linalg.norm(gt[:2, 3] - PLACE_XY))
    result.update(success=bool(upright > 0.95 and on_pad < 0.04), upright_cos=round(upright, 3), on_pad_mm=round(1000 * on_pad, 1),
                  grasp=plan.grasp, clearance_mm=round(1000 * plan.clearance_m), plan_ms=round(plan.plan_ms),
                  total_s=round(time.perf_counter() - t_start, 1))
    return {**result, "events": trial.events}


def finish(cell: Cell, result: dict) -> dict:
    if cell.recorder is not None:
        meta = {k: v for k, v in result.items() if k != "events"}
        result["episode_dir"] = str(cell.recorder.stop(meta))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--yaw-jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--record", type=Path, default=None, help="record episodes for policy training into this folder")
    ap.add_argument("--record-mode", choices=["policy", "full"], default="policy",
                    help="policy: only the flip, from home (training data); full: the whole run (replay video)")
    args = ap.parse_args()
    from physicalai.robot.transport import SharedRobot

    RUNS.mkdir(parents=True, exist_ok=True)
    kin = PandaKinematics()
    robot = SharedRobot.attach(ROBOT, connect_timeout=30)
    robot.connect()
    cell = Cell(robot, kin)
    if args.record is not None:
        from v2.recorder import EpisodeRecorder
        cell.recorder = EpisodeRecorder(args.record)
        cell.record_mode = args.record_mode
    rng = np.random.default_rng(args.seed) if (args.jitter or args.yaw_jitter) else None
    results = []
    try:
        for i in range(args.trials):
            print(f"--- trial {i + 1}/{args.trials}", flush=True)
            r = finish(cell, run_trial(cell, i, rng, args))
            print("  RESULT " + json.dumps({k: v for k, v in r.items() if k != "events"}), flush=True)
            results.append(r)
            (RUNS / f"trial_{int(time.time())}_{i}.json").write_text(json.dumps(r, indent=1))
    finally:
        robot.disconnect()
    ok = sum(r["success"] for r in results)
    print(f"SUMMARY {ok}/{len(results)} successful flips (no ground truth in the loop)")


if __name__ == "__main__":
    main()
