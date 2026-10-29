"""Record v2 runs as training episodes for a visuomotor policy (ACT).

At ``fps`` a thread stores the robot's observed state (physicalai runtime), the action the executor last
commanded, and both camera frames (downscaled JPEG). It pauses while perception or planning computes, so
the policy never learns to sit still waiting for a model.

Raw layout (converted to a LeRobot dataset in the training environment by v3/convert_lerobot.py):
    <root>/episode_000123/
        data.npz      state (T, 8), action (T, 8), timestamp (T,)
        wrist/000000.jpg ...  overview/000000.jpg ...
        meta.json     success, task text, scoring metrics
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from v2.objects import OBJECT

SIM = "http://127.0.0.1:8080"
CAMERAS = ("wrist", "overview")
TASK = OBJECT.task


def _frame(camera: str, size: tuple[int, int]) -> bytes | None:
    try:
        raw = urllib.request.urlopen(f"{SIM}/cameras/{camera}/frame.jpg", timeout=2).read()
    except OSError:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    ok, jpg = cv2.imencode(".jpg", cv2.resize(img, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes() if ok else None


def _viz_part_pose() -> list[float]:
    """The part's pose in the simulator, kept only to draw the replay video (not training data)."""
    try:
        o = next(o for o in json.load(urllib.request.urlopen(f"{SIM}/objects", timeout=2)) if o["joint"] == "part:joint")
        return [*o["position"], *o["wxyz"]]
    except (OSError, StopIteration):
        return [np.nan] * 7


class EpisodeRecorder:
    def __init__(self, root: Path, fps: float = 10.0, size: tuple[int, int] = (320, 240)) -> None:
        self.root, self.fps, self.size = Path(root), fps, size
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()

    def _next_dir(self) -> Path:
        n = len([p for p in self.root.glob("episode_*") if p.is_dir()])
        return self.root / f"episode_{n:06d}"

    def start(self, robot, get_action, paused: bool = False) -> None:
        self.dir = self._next_dir()
        for cam in CAMERAS:
            (self.dir / cam).mkdir(parents=True, exist_ok=True)
        self.state, self.action, self.stamp = [], [], []
        self.viz_part, self.marks = [], []  # for the replay video only; never part of the training data
        self._stop.clear()
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
        self._thread = threading.Thread(target=self._run, args=(robot, get_action), daemon=True)
        self._thread.start()

    def mark(self, label: str, info: dict | None = None, images: dict[str, bytes] | None = None) -> None:
        """A stage marker at the current frame, with optional annotated images, for the replay video."""
        k = len(self.marks)
        names = {}
        for name, png in (images or {}).items():
            path = self.dir / "marks" / f"{k:03d}_{name}.png"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(png)
            names[name] = path.name
        self.marks.append({"frame": len(self.state), "label": label, "info": info or {}, "images": names,
                           "wall_time": time.time()})

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def _run(self, robot, get_action) -> None:
        period = 1.0 / self.fps
        tick = time.perf_counter()
        while not self._stop.is_set():
            tick += period
            if not self._paused.is_set():
                state = np.asarray(robot.get_observation().joint_positions, np.float32)
                action = get_action()
                frames = [_frame(cam, self.size) for cam in CAMERAS]
                if all(f is not None for f in frames):
                    i = len(self.state)
                    for cam, jpg in zip(CAMERAS, frames):
                        (self.dir / cam / f"{i:06d}.jpg").write_bytes(jpg)
                    self.state.append(state)
                    self.action.append(state.copy() if action is None else np.asarray(action, np.float32))
                    self.stamp.append(i / self.fps)  # paused stretches are cut out: time is frame index / fps
                    self.viz_part.append(_viz_part_pose())
            time.sleep(max(0.0, tick - time.perf_counter()))
            if time.perf_counter() - tick > 2 * period:
                tick = time.perf_counter()  # fell behind (slow frame fetch): restart the grid

    def stop(self, meta: dict) -> Path:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        np.savez(self.dir / "data.npz", state=np.array(self.state), action=np.array(self.action),
                 timestamp=np.array(self.stamp, np.float32))
        np.savez(self.dir / "viz.npz", part=np.array(self.viz_part))  # replay video only
        (self.dir / "marks.json").write_text(json.dumps(self.marks, indent=1))
        (self.dir / "meta.json").write_text(json.dumps({"task": TASK, "fps": self.fps, "frames": len(self.state),
                                                         "cameras": list(CAMERAS), "size": self.size, **meta}, indent=1))
        return self.dir
