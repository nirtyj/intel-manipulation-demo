"""Render-and-compare for axially symmetric parts: which end is up, and the spin about the axis.

Depth fits a cylinder equally well either way up and at any spin, and SAM-6D's match of a bare cylinder
always came back "top up". The label does not: render the textured CAD model at the estimated pose, both
ways up and every 10 degrees of spin, and keep the one whose image correlates best with what each wrist
view saw (zero-mean normalised cross-correlation of grey levels inside the part's mask, which is blind to
the simulator's lighting gain and offset).

The rendering runs in a worker process that is replaced after every call: pyrender under OSMesa leaks
~2.4 MB per textured render however it is driven (13 GB over 14 trials in one long-lived service).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

CROP = 160  # rendered and compared at this size (pixels), around the part's mask


def _crop(K: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Square crop around the mask, as (intrinsics of the crop at CROP px, (x0, y0, side))."""
    v, u = np.nonzero(mask)
    side = int(1.3 * max(u.max() - u.min(), v.max() - v.min())) + 8
    cx, cy = (u.max() + u.min()) // 2, (v.max() + v.min()) // 2
    x0, y0 = cx - side // 2, cy - side // 2
    s = CROP / side
    Kc = np.array([[K[0, 0] * s, 0, (K[0, 2] - x0) * s], [0, K[1, 1] * s, (K[1, 2] - y0) * s], [0, 0, 1.0]])
    return Kc, (x0, y0, side)


def _prepare(view) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(T_world_cam, crop intrinsics, grey crop, mask crop) of one wrist view."""
    Kc, (x0, y0, side) = _crop(view.frame.K, view.mask)
    pad = cv2.copyMakeBorder(view.frame.rgb, side, side, side, side, cv2.BORDER_CONSTANT)
    mpad = cv2.copyMakeBorder(view.mask.astype(np.uint8), side, side, side, side, cv2.BORDER_CONSTANT)
    crop = pad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    mcrop = mpad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
    gray = cv2.cvtColor(cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    mask = cv2.resize(mcrop, (CROP, CROP), interpolation=cv2.INTER_NEAREST) > 0
    return view.frame.T_world_cam, Kc, gray.astype(np.float32), mask


def _turn(T_world_obj: np.ndarray, flip_up: bool, spin: float) -> np.ndarray:
    Rx = np.diag([1.0, -1.0, -1.0]) if flip_up else np.eye(3)
    c, s = np.cos(spin), np.sin(spin)
    T = T_world_obj.copy()
    T[:3, :3] = T_world_obj[:3, :3] @ Rx @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return T


def _search(cad_mm: str, views: list, T_world_obj: np.ndarray, step_deg: float) -> dict:
    """Worker: best NCC and spin for each end up. Runs in its own short-lived process."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    import pyrender
    import trimesh

    mesh = trimesh.load(cad_mm, force="mesh")
    mesh.apply_scale(0.001)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera -> OpenGL
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.6, 0.6, 0.6])
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = pyrender.IntrinsicsCamera(fx=1.0, fy=1.0, cx=0.0, cy=0.0, znear=0.01, zfar=5.0)
    scene.add(camera, pose=flip)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=flip)
    renderer = pyrender.OffscreenRenderer(CROP, CROP)

    def score(T: np.ndarray) -> float:
        out = []
        for T_world_cam, Kc, gray, mask in views:
            camera.fx, camera.fy, camera.cx, camera.cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
            scene.set_pose(node, np.linalg.inv(T_world_cam) @ T)
            colour, depth = renderer.render(scene)
            both = (depth > 0) & mask
            if both.sum() < 50:
                continue
            a = cv2.cvtColor(colour, cv2.COLOR_RGB2GRAY).astype(np.float32)[both]
            b = gray[both]
            a, b = a - a.mean(), b - b.mean()
            out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)))
        return float(np.mean(out)) if out else -1.0

    best = {}
    for flip_up in (False, True):
        for spin in np.radians(np.arange(0.0, 360.0, step_deg)):
            sc = score(_turn(T_world_obj, flip_up, spin))
            if flip_up not in best or sc > best[flip_up][0]:
                best[flip_up] = (sc, float(spin))
    renderer.delete()
    return best


class RenderCompare:
    def __init__(self, cad_mm) -> None:
        self.cad_mm = str(cad_mm)
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), max_tasks_per_child=1)

    def resolve(self, views, T_world_obj: np.ndarray, step_deg: float = 10.0) -> tuple[np.ndarray, float, float]:
        """The pose with the up/down and spin that best match the images; returns (pose, best NCC, margin of
        the chosen end over the other)."""
        best = self.pool.submit(_search, self.cad_mm, [_prepare(v) for v in views], T_world_obj, step_deg).result()
        flip_up = best[True][0] > best[False][0]
        sc, spin = best[flip_up]
        return _turn(T_world_obj, flip_up, spin), sc, abs(best[True][0] - best[False][0])
