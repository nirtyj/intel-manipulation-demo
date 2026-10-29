"""Run one can flip on the simulated Panda: read the can's pose, plan, stream joint targets, check the result.

    python -m v2.run_flip [--trials N]

The can pose comes from the simulator (/objects) for now; SAM-6D replaces it in the perception phase.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import numpy as np

from v2.arm import CONTROL_HZ, GRIP_OPEN, PandaKinematics, interpolate, plan_can_flip

SIM = "http://127.0.0.1:8080"
ROBOT = "mujoco-franka_panda-follow"
CAN_START = {"position": [0.48, 0.24, 0.0505], "wxyz": [0.0, 1.0, 0.0, 0.0]}  # bottom-up on the green pad
PLACE_XY = np.array([0.48, -0.24])
VIA_XY = np.array([0.36, 0.0])


def http(path: str, body: dict | None = None) -> object:
    req = urllib.request.Request(SIM + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    return json.load(urllib.request.urlopen(req, timeout=5))


def can_pose() -> tuple[np.ndarray, np.ndarray]:
    o = next(o for o in http("/objects") if o["joint"] == "can:joint")
    w, x, y, z = o["wxyz"]
    up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])  # body z axis in world
    return np.array(o["position"]), up


def reset_can(jitter_m: float = 0.0, rng: np.random.Generator | None = None) -> None:
    pose = dict(CAN_START)
    if jitter_m and rng is not None:
        x, y, z = pose["position"]
        pose["position"] = [x + rng.uniform(-jitter_m, jitter_m), y + rng.uniform(-jitter_m, jitter_m), z]
    http("/objects/can:joint/pose", pose)
    time.sleep(1.0)


def run_once(robot, kin: PandaKinematics, log=print) -> dict:
    from v2.arm import HOME_DEG

    obs = robot.get_observation()
    q_deg = np.asarray(obs.joint_positions, float)
    q = np.radians(q_deg[:7])
    pos, up = can_pose()
    upside_down = bool(up[2] < 0)
    t0 = time.perf_counter()
    plan = plan_can_flip(kin, q, pos, upside_down, PLACE_XY, VIA_XY)
    log(f"plan {1000 * (time.perf_counter() - t0):.0f} ms | can at {np.round(pos, 3)} "
        f"{'bottom-up' if upside_down else 'upright'} | {'; '.join(plan.notes)}")
    grip = float(q_deg[7]) if len(q_deg) > 7 else GRIP_OPEN
    period = 1.0 / CONTROL_HZ
    tick = time.perf_counter()
    for seg in plan.segments:
        if seg.grip is not None:
            grip = seg.grip
        traj = interpolate(kin, seg, q)
        for qk in traj:
            robot.send_action(np.r_[np.degrees(qk), grip], goal_time=2 * period)
            tick += period
            time.sleep(max(0.0, tick - time.perf_counter()))
        q = traj[-1]
        p, u = can_pose()
        log(f"  {seg.name:16s} can z {p[2]:.3f} m, up-axis z {u[2]:+.2f}")
    time.sleep(1.0)
    pos, up = can_pose()
    on_pad = float(np.linalg.norm(pos[:2] - PLACE_XY))
    result = {"upright": bool(up[2] > 0.95), "on_pad_m": round(on_pad, 3), "success": bool(up[2] > 0.95 and on_pad < 0.05),
              "final_pos": np.round(pos, 3).tolist(), "tilt_deg": round(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))), 1)}
    log(f"result: {result}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.0, help="random start offset of the can, metres")
    args = ap.parse_args()
    from physicalai.robot.transport import SharedRobot

    kin = PandaKinematics()
    robot = SharedRobot.attach(ROBOT, connect_timeout=30)
    robot.connect()
    results, rng = [], np.random.default_rng(42)
    try:
        for i in range(args.trials):
            reset_can(args.jitter, rng)
            print(f"--- trial {i + 1}/{args.trials}")
            results.append(run_once(robot, kin, log=lambda m: print(m) if not m.startswith("  ") else None))
    finally:
        robot.disconnect()
    ok = sum(r["success"] for r in results)
    print(f"SUMMARY {ok}/{len(results)} successful flips")


if __name__ == "__main__":
    main()
