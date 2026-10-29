"""Offline planning check for the current CELL_OBJECT, without the simulator running.

Renders the scene camera's depth of the cell with MuJoCo (no robot), fuses it into the voxel map as the
cell does, places the part bottom-up at a random offset and spin, and plans the flip from home.

    CELL_OBJECT=soup_can MUJOCO_GL=osmesa python -m v2.offline_plan_check 5      (physicalai env)
"""

from __future__ import annotations

import sys
import time

import mujoco
import numpy as np
import pinocchio as pin

from v2.arm import HOME_DEG, PandaKinematics
from v2.objects import OBJECT
from v2.perception.sensors import RGBD
from v2.planner import PART_BOXES, CollisionWorld, plan_flip
from v2.run_v2 import PLACE_XY
from v2.world_map import WorldMap

W, H = 640, 480


def overview_frame(m: mujoco.MjModel, d: mujoco.MjData, r: mujoco.Renderer) -> RGBD:
    cam = m.cam("overview")
    f = 0.5 * H / np.tan(np.radians(m.cam_fovy[cam.id]) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[cam.id].reshape(3, 3) @ np.diag([1, -1, -1])  # OpenGL -> OpenCV camera axes
    T[:3, 3] = d.cam_xpos[cam.id]
    r.enable_depth_rendering()
    r.update_scene(d, camera="overview")
    depth = r.render().copy().astype(np.float32)
    r.disable_depth_rendering()
    return RGBD("overview", np.zeros((H, W, 3), np.uint8), depth, K, T, time.time())


def main(n: int) -> None:
    xml = open(OBJECT.scene).read().replace('file="assets/', f'file="{OBJECT.scene.parent}/assets/')
    m = mujoco.MjModel.from_xml_string(xml)
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    kin = PandaKinematics()
    world = CollisionWorld(kin)
    q_home = np.radians(HOME_DEG)
    rng = np.random.default_rng(3)
    ok = 0
    for i in range(n):
        dx, dy, yaw = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-np.pi, np.pi)
        R = pin.utils.rotate("z", yaw) @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0.48 + dx, 0.24 + dy, OBJECT.upright_z]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        d.qpos[:3], d.qpos[3:7] = T[:3, 3], q
        mujoco.mj_forward(m, d)
        wmap = WorldMap()
        wmap.integrate(overview_frame(m, d, r), np.zeros((0, 3)), [(T[:3, 3], 0.10)])
        if OBJECT.axial:
            from v2.objects import canonical_axial
            T = canonical_axial(T, -1.0)
        wmap.cut_object(T, PART_BOXES)
        t0 = time.perf_counter()
        try:
            plan = plan_flip(kin, world, wmap, q_home, T, PLACE_XY)
            ok += 1
            print(f"{i}: {plan.grasp} in {(time.perf_counter() - t0):.1f}s, clearance {1000 * plan.clearance_m:.0f} mm; "
                  + "; ".join(plan.notes[:3]), flush=True)
        except RuntimeError as e:
            print(f"{i}: FAILED after {(time.perf_counter() - t0):.1f}s: {str(e)[:300]}", flush=True)
    print(f"planned {ok}/{n}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
