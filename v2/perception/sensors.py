"""Simulated Intel RealSense RGB-D: fetch a capture from the simulator and apply a depth noise model.

numpy only, so both the robot side (physicalai env) and the perception service can import it.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import numpy as np

SIM = "http://127.0.0.1:8080"
# RealSense-like depth noise: sigma = k * z^2 (D435 scene camera ~2 mm at 0.9 m; D405 wrist well under 1 mm at 0.3 m).
NOISE = {"overview": 0.0025, "wrist": 0.0025}


@dataclass
class RGBD:
    name: str
    rgb: np.ndarray
    depth: np.ndarray          # metres, 0 = no return
    K: np.ndarray
    T_world_cam: np.ndarray    # OpenCV convention
    t: float


def fetch_rgbd(name: str, rng: np.random.Generator | None = None, noisy: bool = True) -> RGBD:
    raw = urllib.request.urlopen(f"{SIM}/cameras/{name}/rgbd", timeout=10).read()
    z = np.load(io.BytesIO(raw))
    depth = z["depth"].astype(np.float32)
    depth[depth > 5.0] = 0.0  # far plane: no return
    if noisy:
        rng = rng or np.random.default_rng()
        k = NOISE.get(name, 0.0025)
        depth = depth + rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * k * depth ** 2
        # stereo sensors lose depth at sharp edges: drop half the pixels with a large local depth jump
        gy, gx = np.gradient(depth)
        edges = np.hypot(gx, gy) > 0.02
        depth[edges & (rng.random(depth.shape) < 0.5)] = 0.0
    return RGBD(name, z["rgb"], depth, z["K"], z["T_world_cam"], float(z["t"]))
