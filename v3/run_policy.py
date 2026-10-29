"""v3: run the trained ACT policy in the same cell, through Intel's physicalai runtime.

``RobotRuntime`` ticks at the rate the demonstrations were recorded at. Each tick it reads the Panda's
joints (``SharedRobot``) and both cameras (the simulator's MJPEG streams through
``physicalai.capture.IPCamera``), the exported OpenVINO policy (``physicalai.inference.InferenceModel``
inside a ``PolicySource``) returns the next action, and the runtime sends it. No detector, pose model,
map or planner: the policy has only the two camera images and the joint state.

    python -m v3.run_policy --trials 5 --jitter 0.02 --yaw-jitter 30      (physicalai env, sim running)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from v2 import run_v2
from v2.arm import HOME_DEG

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "exports" / "act_part_flip"
SIM = "http://127.0.0.1:8080"
CAMERAS = ("wrist", "overview")
TASK = "Pick up the orange and blue part, turn it upright and place it on the blue pad."


class _Borrowed:
    """The cell's connected robot, lent to a runtime that must not connect or disconnect it."""

    def __init__(self, robot) -> None:
        self._robot = robot

    def connect(self, *args, **kwargs) -> None:
        pass

    def disconnect(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, name):
        return getattr(self._robot, name)


def _settings(export_dir: Path) -> dict:
    """Rate, image size and episode length the policy was trained with (written next to the export)."""
    path = export_dir / "cell.json"
    out = {"fps": 10.0, "size": [320, 240], "episode_s": 40.0}
    if path.exists():
        out.update(json.loads(path.read_text()))
    return out


def run_policy_episode(cell: run_v2.Cell, emit=lambda *a, **k: None, stop_event: threading.Event | None = None,
                       export_dir: Path = EXPORT) -> dict:
    """One flip by the learned policy, from home. Scored afterwards against the simulator (scoring only)."""
    from physicalai.capture import IPCamera
    from physicalai.inference import InferenceModel
    from physicalai.runtime import PolicySource, RobotRuntime

    cfg = _settings(export_dir)
    t0 = time.perf_counter()
    cell.grip = run_v2.GRIP_OPEN
    cell.move_to(np.radians(HOME_DEG), duration=2.0)  # where every demonstration started
    emit("stage", stage="policy_load", export=str(export_dir))
    model = InferenceModel(export_dir, backend="openvino", device="CPU")
    w, h = cfg["size"]
    cameras = {name: IPCamera(url=f"{SIM}/cameras/{name}/mjpeg", fps=int(cfg["fps"]), width=w, height=h)
               for name in CAMERAS}
    runtime = RobotRuntime(robot=_Borrowed(cell.robot), action_source=PolicySource(model=model, task=TASK),
                           fps=cfg["fps"], cameras=cameras)
    stop = stop_event or threading.Event()
    emit("stage", stage="policy_run", fps=cfg["fps"], seconds=cfg["episode_s"])
    with runtime:
        ticks = runtime.run(duration_s=cfg["episode_s"], stop_event=stop)
    time.sleep(1.0)
    gt = run_v2.truth()  # scoring only
    upright = float(gt[2, 2])
    on_pad = float(np.linalg.norm(gt[:2, 3] - run_v2.PLACE_XY))
    result = {"mode": "act", "success": bool(upright > 0.95 and on_pad < 0.04), "upright_cos": round(upright, 3),
              "on_pad_mm": round(1000 * on_pad, 1), "ticks": ticks, "total_s": round(time.perf_counter() - t0, 1)}
    emit("stage", stage="policy_done", **result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--yaw-jitter", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--export", type=Path, default=EXPORT)
    a = ap.parse_args()
    from physicalai.robot.transport import SharedRobot

    from v2.arm import PandaKinematics

    robot = SharedRobot.attach(run_v2.ROBOT, connect_timeout=30)
    robot.connect()
    cell = run_v2.Cell(robot, PandaKinematics())
    rng = np.random.default_rng(a.seed)
    results = []
    try:
        for i in range(a.trials):
            run_v2.reset_part(rng, a.jitter, a.yaw_jitter)
            r = {"trial": i, **run_policy_episode(cell, export_dir=a.export)}
            print("  RESULT " + json.dumps(r), flush=True)
            results.append(r)
    finally:
        robot.disconnect()
    print(f"SUMMARY {sum(r['success'] for r in results)}/{len(results)} successful flips by the ACT policy")


if __name__ == "__main__":
    main()
