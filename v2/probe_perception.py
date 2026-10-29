"""Probe the perception chain on the live sim: scene camera -> coarse location -> wrist look -> 6-DoF.

The simulator's object pose is read only to score the estimate, after it is made.

    python -m v2.probe_perception [--trials N]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from v2.arm import GRIP_OPEN, HOME_DEG, PandaKinematics
from v2.look import move_joints, solve_look_pose, tcp_to_camera
from v2.perception.sensors import fetch_rgbd

SIM = "http://127.0.0.1:8080"
PERCEPTION = "http://127.0.0.1:8100"


def call(path: str) -> dict:
    req = urllib.request.Request(PERCEPTION + path, data=b"{}", headers={"content-type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))


def save_annotated(camera: str, path: Path) -> None:
    path.write_bytes(urllib.request.urlopen(f"{PERCEPTION}/annotated/{camera}.png", timeout=10).read())
OUT = Path(__file__).resolve().parents[1] / "runs" / "probe"


def truth() -> np.ndarray:
    """Simulator ground truth, for scoring only."""
    o = next(o for o in json.load(urllib.request.urlopen(SIM + "/objects")) if o["joint"] == "part:joint")
    w, x, y, z = o["wxyz"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, o["position"]
    return T


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    a = ap.parse_args()
    from physicalai.robot.transport import SharedRobot

    OUT.mkdir(parents=True, exist_ok=True)
    kin = PandaKinematics()
    robot = SharedRobot.attach("mujoco-franka_panda-follow", connect_timeout=30)
    robot.connect()
    try:
        for trial in range(a.trials):
            q = np.radians(np.asarray(robot.get_observation().joint_positions, float)[:7])
            q = move_joints(robot, kin, q, np.radians(HOME_DEG), GRIP_OPEN, duration=2.0)
            c = call("/coarse")
            save_annotated("overview", OUT / f"t{trial}_scene.png")
            gt = truth()
            if c["position"] is None:
                print("scene camera: no detection")
                continue
            coarse = np.array(c["position"])
            print(f"[{trial}] scene cam: DINO {c['detection_score']} ({c['timings_ms']}); coarse {np.round(coarse, 3)} "
                  f"| error to truth {1000 * np.linalg.norm(coarse[:2] - gt[:2, 3]):.0f} mm (xy)")
            wrist0 = fetch_rgbd("wrist", noisy=False)
            T_tcp_cam = tcp_to_camera(kin, q, wrist0.T_world_cam)
            look = solve_look_pose(kin, q, T_tcp_cam, coarse)
            if look is None:
                print("no reachable look pose")
                continue
            q = move_joints(robot, kin, q, look[0], GRIP_OPEN, duration=2.0, settle=0.8)
            r = call("/six_dof")
            save_annotated("wrist", OUT / f"t{trial}_wrist.png")
            if r["T_world_obj"] is None:
                print("wrist cam: no pose")
                continue
            T = np.array(r["T_world_obj"])
            gt = truth()
            dp = 1000 * np.linalg.norm(T[:3, 3] - gt[:3, 3])
            dr = rot_err_deg(T[:3, :3], gt[:3, :3])
            up_ok = (T[2, 2] > 0) == (gt[2, 2] > 0)
            print(f"[{trial}] wrist cam: DINO {r['detection_score']}, SAM-6D score {r['pose_score']} ({r['timings_ms']}) "
                  f"| position error {dp:.1f} mm, rotation error {dr:.1f} deg, upside-down {'correct' if up_ok else 'WRONG'}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
