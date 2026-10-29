"""Perception for the part flip: scene camera -> coarse location; wrist camera -> 6-DoF pose.

Nothing here reads the simulator's object state. Inputs are RGB-D images and camera poses from the
simulated RealSense cameras, passed through a RealSense-like depth noise model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from v2.perception.detect import Detection, GroundingDino, backproject, mask_from_box
from v2.perception.pose import SAM6DPose
from v2.perception.sensors import RGBD, fetch_rgbd

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "orange and blue block."
OBJ_RADIUS_M = 0.09  # bounding radius of the L-shaped part (for masks)
HALF_WIDTH_M = 0.025  # visible face to column axis, for the coarse centre
TABLE_Z, MIN_HEIGHT_M = 0.0, 0.03  # the cell layout's table height; anything flatter is not the part
SUPPORT_CUT_M = 0.006             # depth points this close to the table (or a pad on it) are not the part
MAX_SIZE_M = 0.16                  # the part is 90 x 50 x 110 mm; anything bigger is not it


@dataclass
class Observation:
    camera: str
    detection: Detection | None
    mask: np.ndarray | None
    T_world_obj: np.ndarray | None = None
    score: float = 0.0
    fitness: float = 0.0       # mean over views of surface fit minus 2 x free-space violation (1.0 = perfect)
    rmse_mm: float = 0.0
    timings_ms: dict = field(default_factory=dict)
    per_view: list = field(default_factory=list)


@dataclass
class View:
    """One wrist capture with its detection and SAM-6D hypotheses (world frame)."""
    frame: RGBD
    detection: Detection
    mask: np.ndarray
    cloud_world: np.ndarray
    hypotheses: list
    sam6d_score: float
    ms: dict


class PartPerception:
    def __init__(self) -> None:
        self.dino = GroundingDino()
        self.pose = SAM6DPose(ROOT / "models/l_part.ply", ROOT / "models/l_part")
        self.rng = np.random.default_rng(0)
        import trimesh
        mesh = trimesh.load(ROOT / "models/l_part.ply", force="mesh")
        self.model_surface = (trimesh.sample.sample_surface(mesh, 6000)[0] / 1000.0).astype(np.float64)

    def detect(self, frame: RGBD) -> tuple[Detection | None, np.ndarray | None]:
        """Best detection that is a real 3D object: its masked points must stand > 3 cm above the table.

        Open-vocabulary matches flat things that share the prompt's words (a blue pad for "blue block");
        support-plane reasoning rejects them.
        """
        for d in self.dino.detect(frame.rgb, PROMPT)[:5]:
            m = mask_from_box(frame.depth, frame.K, d.box, OBJ_RADIUS_M)
            if m.sum() < 20:
                continue
            v, u = np.nonzero(m)
            pts = backproject(frame.depth, frame.K, v, u) @ frame.T_world_cam[:3, :3].T + frame.T_world_cam[:3, 3]
            # The box's RANSAC plane cut misses the table when the part fills the box; cut by height too.
            on_table = pts[:, 2] < TABLE_Z + SUPPORT_CUT_M
            m[v[on_table], u[on_table]] = False
            pts = pts[~on_table]
            if len(pts) < 20:
                continue
            height = np.percentile(pts[:, 2], 95) - TABLE_Z
            spread = np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)
            if MIN_HEIGHT_M < height < MAX_SIZE_M and np.all(spread < MAX_SIZE_M):  # part-sized, not a pad or the wall
                return d, m
        return None, None

    def coarse_location(self, frame: RGBD) -> tuple[np.ndarray | None, Observation]:
        """World position of the can's visible surface centroid, pushed back by its radius."""
        d, m = self.detect(frame)
        obs = Observation(frame.name, d, m)
        if d is None or m is None or m.sum() < 20:
            return None, obs
        v, u = np.nonzero(m)
        pts = backproject(frame.depth, frame.K, v, u)
        c_cam = np.median(pts, axis=0)
        c_cam = c_cam + HALF_WIDTH_M * c_cam / np.linalg.norm(c_cam)  # surface -> axis, along the view ray
        c_world = frame.T_world_cam[:3, :3] @ c_cam + frame.T_world_cam[:3, 3]
        obs.timings_ms["dino"] = d.ms
        return c_world, obs

    def observe(self, frame: RGBD, hypotheses: int = 3) -> View | None:
        """One wrist view: detect the part, mask it, and collect SAM-6D's pose hypotheses (world frame)."""
        d, m = self.detect(frame)
        if d is None or m is None or m.sum() < 40:
            return None
        p = self.pose.estimate(frame.rgb, frame.depth, frame.K, m, d.score, hypotheses=hypotheses)
        if p is None:
            return None
        hyps = []
        for R, t, _ in self.pose.last_hypotheses:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            hyps.append(frame.T_world_cam @ T)
        cloud = self.pose.last_cloud @ frame.T_world_cam[:3, :3].T + frame.T_world_cam[:3, 3]
        return View(frame, d, m, cloud, hyps, p.score, {"dino": d.ms, "sam6d": p.ms})

    def fuse(self, views: list[View]) -> Observation:
        """Pick the pose that every view agrees with.

        Candidates: each view's SAM-6D hypotheses and their half-turns about the part's own axes, refined by
        ICP against the points of all views together. Each candidate is then scored in every view, as the
        share of that view's points on the posed model surface minus twice the share of the posed model
        sitting where that camera sees through to something farther away (free space). One view can be
        fooled when the foot is hidden behind the column; two views from different sides rarely are.
        """
        last = views[-1]
        obs = Observation(last.frame.name, last.detection, last.mask)
        t0 = time.perf_counter()
        best, per_view = select_pose(views, self.model_surface)
        obs.timings_ms = {"dino": sum(v.ms["dino"] for v in views), "sam6d": sum(v.ms["sam6d"] for v in views),
                          "verify": (time.perf_counter() - t0) * 1000}
        if best is None:
            return obs
        T_world_obj, score, rmse = best
        obs.T_world_obj = gravity_snap(T_world_obj)
        obs.score = max(v.sam6d_score for v in views)
        obs.fitness, obs.rmse_mm, obs.per_view = score, rmse * 1000, per_view
        return obs

    def six_dof(self, frame: RGBD, hypotheses: int = 3) -> Observation:
        view = self.observe(frame, hypotheses)
        if view is None:
            d, m = self.detect(frame)
            return Observation(frame.name, d, m)
        return self.fuse([view])


def _flips() -> list[np.ndarray]:
    out = [np.eye(3)]
    for axis in "xyz":
        out.append(pin_rotate(axis, np.pi))
    return out


def pin_rotate(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def free_space_violation(T_cam_obj: np.ndarray, model_m: np.ndarray, depth: np.ndarray, K: np.ndarray,
                         margin: float = 0.012) -> float:
    """Share of the posed model that sits where the camera sees through to something farther away."""
    pts = model_m @ T_cam_obj[:3, :3].T + T_cam_obj[:3, 3]
    z = pts[:, 2]
    ok = z > 0.05
    u = np.round(pts[ok, 0] * K[0, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(pts[ok, 1] * K[1, 1] / z[ok] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    if inside.sum() == 0:
        return 1.0
    d = depth[v[inside], u[inside]]
    valid = d > 0
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(d[valid] > z[ok][inside][valid] + margin))


def view_score(T_world_obj: np.ndarray, view: View, model_m: np.ndarray, max_dist: float = 0.004) -> tuple[float, float, float]:
    """(fit, free-space violation, inlier rmse) of a world pose in one view."""
    from scipy.spatial import cKDTree

    posed = model_m @ T_world_obj[:3, :3].T + T_world_obj[:3, 3]
    dist, _ = cKDTree(posed).query(view.cloud_world, distance_upper_bound=max_dist)
    inl = np.isfinite(dist)
    fit = float(inl.mean())
    rmse = float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else np.inf
    T_cam_obj = np.linalg.inv(view.frame.T_world_cam) @ T_world_obj
    return fit, free_space_violation(T_cam_obj, model_m, view.frame.depth, view.frame.K), rmse


def select_pose(views: list[View], model_m: np.ndarray):
    """Best world pose over all views' hypotheses x half-turns, ICP-refined on the fused cloud."""
    import open3d as o3d

    fused = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([v.cloud_world for v in views])))
    best, best_views = None, []
    for view in views:
        for T0 in view.hypotheses:
            for F in _flips():
                T = T0.copy()
                T[:3, :3] = T0[:3, :3] @ F
                model = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_m @ T[:3, :3].T + T[:3, 3]))
                reg = o3d.pipelines.registration.registration_icp(
                    fused, model, 0.015, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                T_ref = np.linalg.inv(reg.transformation) @ T
                scores = [view_score(T_ref, v, model_m) for v in views]
                score = float(np.mean([fit - 2.0 * free for fit, free, _ in scores]))
                rmse = float(np.mean([r for _, _, r in scores if np.isfinite(r)] or [np.inf]))
                if best is None or score > best[1] + 1e-9 or (abs(score - best[1]) < 1e-9 and rmse < best[2]):
                    best = (T_ref, score, rmse)
                    best_views = [{"camera_pos": np.round(v.frame.T_world_cam[:3, 3], 3).tolist(), "fit": round(f, 3),
                                   "free_space": round(fr, 3)} for v, (f, fr, _) in zip(views, scores)]
    return best, best_views


def gravity_snap(T_world_obj: np.ndarray, max_tilt_deg: float = 20.0) -> np.ndarray:
    """A part resting on the table has its own z axis vertical: snap near-vertical estimates exactly."""
    z = T_world_obj[:3, 2]
    target = np.array([0.0, 0.0, 1.0 if z[2] > 0 else -1.0])
    if np.degrees(np.arccos(np.clip(abs(z[2]), -1, 1))) > max_tilt_deg:
        return T_world_obj
    axis = np.cross(z, target)
    s, c = np.linalg.norm(axis), float(z @ target)
    if s < 1e-9:
        return T_world_obj
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R_fix = np.eye(3) + s * K + (1 - c) * K @ K
    out = T_world_obj.copy()
    out[:3, :3] = R_fix @ T_world_obj[:3, :3]
    return out


def annotate(frame: RGBD, obs: Observation, label: str = "") -> np.ndarray:
    """RGB with the detection box, the mask and the estimated object axes, for the console and video."""
    img = frame.rgb.copy()
    if obs.mask is not None:  # outline, so the object keeps its own colours on screen
        contours, _ = cv2.findContours(obs.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (80, 255, 120), 2, cv2.LINE_AA)
    if obs.detection is not None:
        x1, y1, x2, y2 = np.round(obs.detection.box).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 210, 0), 2)
        cv2.putText(img, f"DINO block {obs.detection.score:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 210, 0), 1, cv2.LINE_AA)
    if obs.T_world_obj is not None:
        T_cam_obj = np.linalg.inv(frame.T_world_cam) @ obs.T_world_obj
        o = T_cam_obj[:3, 3]
        for axis, colour in ((0, (255, 60, 60)), (1, (60, 255, 60)), (2, (80, 140, 255))):
            tip = o + T_cam_obj[:3, axis] * 0.06
            pa, pb = (frame.K @ o)[:2] / o[2], (frame.K @ tip)[:2] / tip[2]
            cv2.line(img, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"SAM-6D 6-DoF pose {obs.score:.2f}", (img.shape[1] - 230, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def to_json(T: np.ndarray | None) -> list | None:
    return None if T is None else np.round(T, 5).tolist()


__all__ = ["PartPerception", "RGBD", "Observation", "View", "fetch_rgbd", "annotate", "to_json", "json"]
