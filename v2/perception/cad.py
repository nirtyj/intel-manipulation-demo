"""Export the scene's objects as coloured CAD models in millimetres (SAM-6D's convention).

The L-shaped part is the demo object; the soup can was the first version.

The geometry and colours are the same as the can in scenes/can_flip/scene.xml: a 66 mm x 101 mm
cylinder, red over the top half, white over the bottom half, metal lid and base. Object frame: origin
at the can's centre, +z through the lid (so +z is "up" when the can stands the right way round).

    python -m v2.perception.cad --out models/soup_can.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

RADIUS_M, HALF_HEIGHT_M = 0.033, 0.0505
RED, WHITE, METAL = (199, 26, 26), (237, 235, 224), (179, 179, 184)


def _band(z0: float, z1: float, radius: float, rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=radius, segment=[[0, 0, z0], [0, 0, z1]], sections=96)
    m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1))
    return m


def soup_can_mesh() -> trimesh.Trimesh:
    h = HALF_HEIGHT_M
    parts = [
        _band(0.0, h, RADIUS_M, RED),
        _band(-h, 0.0, RADIUS_M, WHITE),
        _band(h, h + 0.0003, RADIUS_M - 0.002, METAL),  # just proud of the band, no z-fighting
        _band(-h - 0.0003, -h, RADIUS_M - 0.002, METAL),
    ]
    can = trimesh.util.concatenate(parts)
    can.apply_scale(1000.0)  # metres -> millimetres
    return can


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/l_part.ply"))
    ap.add_argument("--object", choices=["l_part", "soup_can"], default="l_part")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mesh = l_part_mesh() if a.object == "l_part" else soup_can_mesh()
    mesh.export(a.out)
    print(f"wrote {a.out}: {len(mesh.vertices)} vertices, extents {np.round(mesh.extents, 1).tolist()} mm")

ORANGE, BLUE = (242, 128, 20), (31, 77, 191)


def _box(half: tuple[float, float, float], centre: tuple[float, float, float], rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[2 * h for h in half])
    m.apply_translation(centre)
    m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1))
    return m


def l_part_mesh() -> trimesh.Trimesh:
    """The scene's L-shaped part (scenes/can_flip/scene.xml), in millimetres, same frame as the body."""
    parts = [
        _box((0.025, 0.025, 0.0275), (0, 0, 0.0275), ORANGE),
        _box((0.025, 0.025, 0.0275), (0, 0, -0.0275), BLUE),
        _box((0.02, 0.025, 0.015), (0.045, 0, -0.04), BLUE),
    ]
    part = trimesh.util.concatenate(parts)
    part.apply_scale(1000.0)
    return part


if __name__ == "__main__":
    main()
