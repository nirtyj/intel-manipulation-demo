"""The object the cell turns over, and everything about it the pipeline needs. ``CELL_OBJECT`` picks it:

- ``l_part`` (default): the asymmetric L-shaped part. No rotation looks the same, so its full 6-DoF pose
  comes from geometry alone.
- ``soup_can``: the YCB 005 tomato soup can (Calli et al., the Google 16k scan). It is axially symmetric:
  geometry gives its position and axis, not its spin, and not which end is up. As in the Isaac reference
  cell, it starts bottom-up and is grasped round the middle, so the grasp is the same at both ends.

Every frame here is the object's own: origin at its centre, +z through its top.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    scene: Path
    prompt: str               # Grounding DINO text prompt
    label: str                # short name on annotated images
    task: str                 # the instruction recorded with each episode
    cad_mm: Path              # SAM-6D model, millimetres, object frame
    templates: Path           # SAM-6D templates rendered from cad_mm
    boxes: list               # collision shapes as (half extents, centre) boxes, object frame
    cylinder: tuple | None    # (radius, half height) when the object is a cylinder about z
    upright_z: float          # object centre above the table when it stands upright
    radius_m: float           # bounding radius, for depth masks
    half_width_m: float       # visible surface to the object's axis, for the coarse centre
    axial: bool               # symmetric about z: spin is meaningless and up/down is not in the geometry
    approaches_deg: tuple = (0.0, 180.0, 90.0, -90.0)  # side grasp directions, about the object's z
    grasp_heights: tuple = (0.0, 0.004, -0.004)
    extra: dict = field(default_factory=dict)

    @property
    def start(self) -> dict:
        """Upside down on the green pad."""
        return {"position": [0.48, 0.24, self.upright_z], "wxyz": [0.0, 1.0, 0.0, 0.0]}


L_PART = ObjectSpec(
    name="l_part",
    scene=ROOT / "scenes/can_flip/scene.xml",
    prompt="orange and blue block.",
    label="block",
    task="Pick up the orange and blue part, turn it upright and place it on the blue pad.",
    cad_mm=ROOT / "models/l_part.ply",
    templates=ROOT / "models/l_part",
    boxes=[((0.025, 0.025, 0.055), (0.0, 0.0, 0.0)),      # column
           ((0.02, 0.025, 0.015), (0.045, 0.0, -0.04))],   # foot
    cylinder=None,
    upright_z=0.055,
    radius_m=0.09,
    half_width_m=0.025,
    axial=False,
    approaches_deg=(180.0, 0.0, -90.0, 90.0),
)

SOUP_CAN_R, SOUP_CAN_H = 0.0339, 0.1019  # the YCB scan's extents (67.8 mm across, 101.9 mm tall)
SOUP_CAN = ObjectSpec(
    name="soup_can",
    scene=ROOT / "scenes/soup_can/scene.xml",
    prompt="red and white soup can.",
    label="soup can",
    task="Pick up the tomato soup can, turn it upright and place it on the blue pad.",
    cad_mm=ROOT / "models/soup_can/soup_can.obj",
    templates=ROOT / "models/soup_can",
    boxes=[((SOUP_CAN_R, SOUP_CAN_R, SOUP_CAN_H / 2), (0.0, 0.0, 0.0))],
    cylinder=(SOUP_CAN_R, SOUP_CAN_H / 2),
    upright_z=SOUP_CAN_H / 2,
    radius_m=0.075,
    half_width_m=SOUP_CAN_R,
    axial=True,
    approaches_deg=tuple(float(a) for a in range(0, 360, 30)),
    grasp_heights=(0.0, 0.003),  # round the middle: the hand needs ~48 mm above the table at both ends
)

OBJECTS = {o.name: o for o in (L_PART, SOUP_CAN)}
OBJECT = OBJECTS[os.environ.get("CELL_OBJECT", "l_part")]


def canonical_axial(T_world_obj: np.ndarray, up_sign: float) -> np.ndarray:
    """An axially symmetric object's pose with its spin fixed (x pointing away from the robot's base, as
    seen from above) and its axis pointing ``up_sign`` (+1 top up, -1 top down)."""
    z = T_world_obj[:3, 2] * (1.0 if T_world_obj[2, 2] * up_sign >= 0 else -1.0)
    radial = np.r_[T_world_obj[:2, 3], 0.0]
    x = radial - (radial @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    out = T_world_obj.copy()
    out[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return out
