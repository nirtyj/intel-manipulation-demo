"""Offline check of the soup-can perception: MuJoCo-rendered RGB-D views of the bottom-up can (no robot),
through DINO + SAM-6D + two-view verification + the up/down vote. Compares with the known pose.

    CELL_OBJECT=soup_can PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa perception-python offline_perception.py N
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user/talk-to-the-cell")
from v2.objects import OBJECT  # noqa: E402
from v2.perception.sensors import RGBD  # noqa: E402
OUT = "/tmp/claude-0/-home-user/df5ef3a4-2c06-527f-a397-7d40bc547731/scratchpad/ycb/views.npz"

W, H, FOVY = 640, 480, 58.0


def look_camera(target, azimuth_deg, distance, elevation_deg=45.0):
    """OpenCV camera pose looking at target from the robot's side (-radial), rotated by azimuth."""
    radial = np.r_[target[:2], 0.0] / np.linalg.norm(target[:2])
    a = np.radians(azimuth_deg)
    back = -radial
    back = np.array([np.cos(a) * back[0] - np.sin(a) * back[1], np.sin(a) * back[0] + np.cos(a) * back[1], 0.0])
    e = np.radians(elevation_deg)
    pos = target + distance * (np.cos(e) * back + np.array([0, 0, np.sin(e)]))
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = pos
    return T


def render(m, d, r, T_world_cam):
    import mujoco
    f = 0.5 * H / np.tan(np.radians(FOVY) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    cam = m.cam("probe")
    R_gl = T_world_cam[:3, :3] @ np.diag([1, -1, -1])  # OpenCV -> OpenGL camera axes
    m.cam_pos[cam.id] = T_world_cam[:3, 3]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_gl.flatten())
    m.cam_quat[cam.id] = q
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera="probe")
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(d, camera="probe")
    depth = r.render().copy()
    r.disable_depth_rendering()
    depth = depth + np.random.default_rng(0).normal(0, 1, depth.shape) * 0.0025 * depth ** 2
    return RGBD("wrist", rgb, depth.astype(np.float32), K, T_world_cam, time.time())


def make(n: int) -> None:
    import mujoco
    xml = open(OBJECT.scene).read().replace(
        "</worldbody>", f'<camera name="probe" pos="0 0 1" fovy="{FOVY}"/></worldbody>')
    xml = xml.replace('file="assets/', f'file="{OBJECT.scene.parent}/assets/')
    m = mujoco.MjModel.from_xml_string(xml)
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    rng = np.random.default_rng(int(os.environ.get("SEED", "7")))
    out = {}
    for i in range(n):
        dx, dy, yaw = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-np.pi, np.pi)
        pos = np.array([0.48 + dx, 0.24 + dy, OBJECT.upright_z])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R = Rz @ (np.diag([1.0, -1.0, -1.0]) if (i % 2 == 0 or os.environ.get("ALL_DOWN")) else np.eye(3))  # bottom-up / upright
        qq = np.zeros(4)
        mujoco.mju_mat2Quat(qq, R.flatten())
        d.qpos[:3], d.qpos[3:7] = pos, qq
        mujoco.mj_forward(m, d)
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, pos
        out[f"truth_{i}"] = T
        for j, az in enumerate((35.0, -35.0)):
            f = render(m, d, r, look_camera(pos, az, 0.28))
            out[f"rgb_{i}_{j}"], out[f"depth_{i}_{j}"], out[f"K_{i}_{j}"], out[f"T_{i}_{j}"] = f.rgb, f.depth, f.K, f.T_world_cam
    np.savez_compressed(OUT, n=n, **out)
    print("wrote", OUT)


def estimate() -> None:
    from v2.perception.pipeline import PartPerception
    z = np.load(OUT)
    per = PartPerception()
    for i in range(int(z["n"])):
        R = z[f"truth_{i}"][:3, :3]; pos = z[f"truth_{i}"][:3, 3]
        t0 = time.perf_counter()
        views = []
        for j in range(2):
            v = per.observe(RGBD("wrist", z[f"rgb_{i}_{j}"], z[f"depth_{i}_{j}"], z[f"K_{i}_{j}"], z[f"T_{i}_{j}"], 0.0))
            if v is not None:
                views.append(v)
        obs = per.fuse(views) if views else None
        dt = time.perf_counter() - t0
        if obs is None or obs.T_world_obj is None:
            print(f"{i}: no pose ({len(views)} views) {dt:.1f}s", flush=True)
            continue
        T = obs.T_world_obj
        pos_err = 1000 * np.linalg.norm(T[:3, 3] - pos)
        axis_err = np.degrees(np.arccos(np.clip(abs(T[:3, 2] @ R[:, 2]), -1, 1)))
        up_ok = bool(np.sign(T[2, 2]) == np.sign(R[2, 2]))
        print(f"{i}: views {len(views)} fit {obs.fitness:.3f} pos_err {pos_err:.1f} mm axis_err {axis_err:.1f} deg "
              f"up/down {'right' if up_ok else 'WRONG'} (margin {obs.up_vote:+.3f}, ncc {obs.appearance_ncc:.2f}, {obs.timings_ms.get('appearance', 0):.0f} ms) {dt:.1f}s", flush=True)


if __name__ == "__main__":
    make(int(sys.argv[2])) if sys.argv[1] == "make" else estimate()
