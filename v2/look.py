"""Point the wrist camera at a target, and move the arm there through physicalai's runtime."""

from __future__ import annotations

import time

import numpy as np
import pinocchio as pin

from v2.arm import CONTROL_HZ, PandaKinematics, Segment, interpolate


def se3_to_mat(T: pin.SE3) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = T.rotation, T.translation
    return M


def tcp_to_camera(kin: PandaKinematics, q: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """The camera's mounting on the tool, measured: FK of the tool vs the camera pose the sim reports."""
    return np.linalg.inv(se3_to_mat(kin.fk(q))) @ T_world_cam


def camera_look_at(cam_pos: np.ndarray, target: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """OpenCV camera pose (x right, y down, z forward) at cam_pos looking at target."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, -1.0])
    y = down - z * (down @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z]) @ pin.utils.rotate("z", roll)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam_pos
    return T


def solve_look_pose(kin: PandaKinematics, q_now: np.ndarray, T_tcp_cam: np.ndarray, target: np.ndarray,
                    distance: float = 0.30, azimuth_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Joint configuration that puts the wrist camera ``distance`` from ``target``, looking at it.

    ``azimuth_deg`` swings the viewpoint around the target (0 = from the robot's side).
    """
    inward = -target[:2] / np.linalg.norm(target[:2])  # from the target back toward the robot base
    c, s_ = np.cos(np.radians(azimuth_deg)), np.sin(np.radians(azimuth_deg))
    inward = np.array([c * inward[0] - s_ * inward[1], s_ * inward[0] + c * inward[1]])
    for elevation in np.radians([55, 65, 45, 75]):
        for roll in np.radians([0, 90, -90, 180]):
            d = np.r_[np.cos(elevation) * inward, np.sin(elevation)]
            T_cam = camera_look_at(target + distance * d, target, roll)
            T_tcp = T_cam @ np.linalg.inv(T_tcp_cam)
            q, err = kin.ik(pin.SE3(T_tcp[:3, :3], T_tcp[:3, 3]), q_now)
            if err < 1e-3:
                return q, T_cam
    return None


def move_joints(robot, kin: PandaKinematics, q_from: np.ndarray, q_to: np.ndarray, grip: float,
                duration: float = 2.0, settle: float = 0.5) -> np.ndarray:
    period = 1.0 / CONTROL_HZ
    tick = time.perf_counter()
    for qk in interpolate(kin, Segment("move", "joint", duration, q_goal=q_to), q_from):
        robot.send_action(np.r_[np.degrees(qk), grip], goal_time=2 * period)
        tick += period
        time.sleep(max(0.0, tick - time.perf_counter()))
    time.sleep(settle)
    return q_to
