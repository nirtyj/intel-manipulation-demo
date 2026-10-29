"""Render-and-compare for axially symmetric parts: which end is up, and the spin about the axis.

Depth fits a cylinder equally well either way up and at any spin, and SAM-6D's match of a bare cylinder
always came back "top up". The label does not: render the textured CAD model at the estimated pose, both
ways up and every 10 degrees of spin, and keep the one whose image correlates best with what each wrist
view saw (zero-mean normalised cross-correlation of grey levels inside the part's mask, which is blind to
the simulator's lighting gain and offset).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
import pyrender
import trimesh

CROP = 160  # rendered and compared at this size (pixels), around the part's mask


class RenderCompare:
    def __init__(self, cad_mm) -> None:
        mesh = trimesh.load(cad_mm, force="mesh")
        mesh.apply_scale(0.001)
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.6, 0.6, 0.6])
        self.node = self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        self.cam_node = None
        self.light = self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0))
        self.renderer = pyrender.OffscreenRenderer(CROP, CROP)

    def _render(self, K: np.ndarray, T_cam_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=5.0)
        if self.cam_node is not None:
            self.scene.remove_node(self.cam_node)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera -> OpenGL
        self.cam_node = self.scene.add(cam, pose=flip)
        self.scene.set_pose(self.light, flip)
        self.scene.set_pose(self.node, T_cam_obj)
        colour, depth = self.renderer.render(self.scene)
        return cv2.cvtColor(colour, cv2.COLOR_RGB2GRAY).astype(np.float32), depth > 0

    @staticmethod
    def _crop(K: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Square crop around the mask, as (intrinsics of the crop at CROP px, (x0, y0, side))."""
        v, u = np.nonzero(mask)
        side = int(1.3 * max(u.max() - u.min(), v.max() - v.min())) + 8
        cx, cy = (u.max() + u.min()) // 2, (v.max() + v.min()) // 2
        x0, y0 = cx - side // 2, cy - side // 2
        s = CROP / side
        Kc = np.array([[K[0, 0] * s, 0, (K[0, 2] - x0) * s], [0, K[1, 1] * s, (K[1, 2] - y0) * s], [0, 0, 1.0]])
        return Kc, (x0, y0, side)

    def score(self, views, T_world_obj: np.ndarray, flip_up: bool, spin_rad: float) -> float:
        """Mean over views of the grey-level NCC between the rendered and the observed part."""
        Rx = np.diag([1.0, -1.0, -1.0]) if flip_up else np.eye(3)
        c, s = np.cos(spin_rad), np.sin(spin_rad)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        T = T_world_obj.copy()
        T[:3, :3] = T_world_obj[:3, :3] @ Rx @ Rz
        out = []
        for view, (Kc, gray, mask) in views:
            T_cam_obj = np.linalg.inv(view.frame.T_world_cam) @ T
            rend, rmask = self._render(Kc, T_cam_obj)
            both = rmask & mask
            if both.sum() < 50:
                continue
            a, b = rend[both], gray[both]
            a, b = a - a.mean(), b - b.mean()
            out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)))
        return float(np.mean(out)) if out else -1.0

    def resolve(self, views, T_world_obj: np.ndarray, step_deg: float = 10.0) -> tuple[np.ndarray, float, float]:
        """The pose with the up/down and spin that best match the images; returns (pose, best NCC, margin of
        the chosen end over the other)."""
        prepared = []
        for v in views:
            Kc, (x0, y0, side) = self._crop(v.frame.K, v.mask)
            pad = cv2.copyMakeBorder(v.frame.rgb, side, side, side, side, cv2.BORDER_CONSTANT)
            mpad = cv2.copyMakeBorder(v.mask.astype(np.uint8), side, side, side, side, cv2.BORDER_CONSTANT)
            crop = pad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
            mcrop = mpad[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side]
            gray = cv2.cvtColor(cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            mask = cv2.resize(mcrop, (CROP, CROP), interpolation=cv2.INTER_NEAREST) > 0
            prepared.append((v, (Kc, gray.astype(np.float32), mask)))
        best = {}
        for flip_up in (False, True):
            for spin in np.radians(np.arange(0.0, 360.0, step_deg)):
                sc = self.score(prepared, T_world_obj, flip_up, spin)
                if flip_up not in best or sc > best[flip_up][0]:
                    best[flip_up] = (sc, spin)
        flip_up = best[True][0] > best[False][0]
        sc, spin = best[flip_up]
        Rx = np.diag([1.0, -1.0, -1.0]) if flip_up else np.eye(3)
        c, s = np.cos(spin), np.sin(spin)
        T = T_world_obj.copy()
        T[:3, :3] = T_world_obj[:3, :3] @ Rx @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return T, sc, abs(best[True][0] - best[False][0])
