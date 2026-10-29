"""Render SAM-6D templates for a CAD model without Blender.

Matches SAM-6D's render_custom_templates.py (BlenderProc) output:
    templates/rgb_{i}.png   colour view, BGR as written by cv2
    templates/mask_{i}.png  object mask, 0 / 255
    templates/xyz_{i}.npy   per-pixel object coordinates in the CAD's units (mm), float16
for the 42 predefined views in cam_poses_level0.npy (OpenCV camera-to-object poses, translation in mm),
with the object normalised to radius 0.5 and the camera translation scaled by 0.002, as BlenderProc does.

    PYOPENGL_PLATFORM=osmesa python -m v2.perception.templates --cad can.ply --out outputs/can
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
import pyrender
import trimesh

SAM6D_ROOT = Path(os.environ.get("SAM6D_ROOT", "/home/user/OpenVINO-SAM-6D/SAM-6D"))
CAM_POSES = SAM6D_ROOT / "Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"
SIZE = 512                       # BlenderProc's default resolution
FOV_X = 2 * np.arctan(18 / 35)   # BlenderProc's default 35 mm lens on a 36 mm sensor


def norm_scale(mesh: trimesh.Trimesh) -> float:
    """SAM-6D's get_norm_info: 1 / (2 * radius) from 1024 surface samples (CAD units)."""
    pts = trimesh.sample.sample_surface(mesh, 1024)[0]
    radius = max(np.linalg.norm(pts.max(0)), np.linalg.norm(pts.min(0)))
    return 1.0 / (2.0 * radius)


def render_templates(cad: Path, out_dir: Path) -> int:
    mesh = trimesh.load(cad, force="mesh")
    scale = norm_scale(mesh)
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.Mesh.from_trimesh(scaled, smooth=False))
    f = 0.5 * SIZE / np.tan(FOV_X / 2)
    camera = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=SIZE / 2, cy=SIZE / 2, znear=0.01, zfar=20.0)
    cam_node = scene.add(camera)
    light_node = scene.add(pyrender.PointLight(color=np.ones(3), intensity=8.0))
    renderer = pyrender.OffscreenRenderer(SIZE, SIZE)
    tdir = out_dir / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    poses = np.load(CAM_POSES)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes -> OpenGL (pyrender)
    for i, pose_cv in enumerate(poses):
        pose_cv = pose_cv.copy()
        pose_cv[:3, 3] *= 0.001 * 2  # as BlenderProc: mm -> scene units of the normalised object
        scene.set_pose(cam_node, pose_cv @ flip)
        light_pose = np.eye(4)
        light_pose[:3, 3] = 2.5 * pose_cv[:3, 3]
        scene.set_pose(light_node, light_pose)
        color, depth = renderer.render(scene)
        mask = depth > 0
        v, u = np.nonzero(mask)
        z = depth[v, u]
        cam_pts = np.stack([(u + 0.5 - SIZE / 2) * z / f, (v + 0.5 - SIZE / 2) * z / f, z], axis=1)
        obj_pts = cam_pts @ pose_cv[:3, :3].T + pose_cv[:3, 3]
        xyz = np.zeros((SIZE, SIZE, 3), np.float32)
        xyz[v, u] = obj_pts / scale  # back to CAD units (mm)
        cv2.imwrite(str(tdir / f"rgb_{i}.png"), color[:, :, ::-1])
        cv2.imwrite(str(tdir / f"mask_{i}.png"), (mask * 255).astype(np.uint8))
        np.save(tdir / f"xyz_{i}.npy", xyz.astype(np.float16))
    renderer.delete()
    return len(poses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = render_templates(a.cad, a.out)
    print(f"rendered {n} templates into {a.out / 'templates'}")


if __name__ == "__main__":
    main()
