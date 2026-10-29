"""A fused obstacle map from the wrist and scene RGB-D cameras (nvblox-style, on the CPU).

Each depth frame is back-projected into the world, the robot's own body is filtered out (its collision
meshes posed by forward kinematics), the target part is cut out at its estimated pose, and the rest is
accumulated in a voxel grid. The table is found as the dominant horizontal plane and kept as a
half-space; everything above it becomes a Coal octree the planner checks against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import coal
import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree

from v2.perception.sensors import RGBD

WORKSPACE_MIN = np.array([-0.2, -0.7, -0.05])
WORKSPACE_MAX = np.array([1.0, 0.7, 0.9])
VOXEL = 0.015
ROBOT_MARGIN = 0.035      # depth points this close to the robot's surface are the robot itself
MIN_HITS = 3              # points per voxel per frame to count as occupied
PLANNING_PAD_VOXELS = 1   # obstacles are grown by one voxel (15 mm) before planning


def _mesh_surface(geo, n: int, rng: np.random.Generator) -> np.ndarray:
    """Points spread over a BVH mesh's triangles (vertices alone leave big gaps on coarse meshes)."""
    v = np.asarray(geo.vertices())
    try:
        tris = np.array([[t[0], t[1], t[2]] for t in (geo.tri_indices(i) for i in range(geo.num_tris))])
    except Exception:  # no triangle access: fall back to vertices
        return v[rng.choice(len(v), min(n, len(v)), replace=False)]
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pick = rng.choice(len(tris), n, p=area / area.sum())
    r1, r2 = rng.random((2, n, 1))
    s = np.sqrt(r1)
    return (1 - s) * a[pick] + s * (1 - r2) * b[pick] + s * r2 * c[pick]


_SURFACE_CACHE: dict[int, np.ndarray] = {}


def robot_surface_samples(robot_geom: pin.GeometryModel, model: pin.Model, data: pin.Data,
                          gdata: pin.GeometryData, q: np.ndarray, per_geom: int = 600) -> np.ndarray:
    """Points on the robot's collision geometry at configuration q, in the world frame."""
    pin.updateGeometryPlacements(model, data, robot_geom, gdata, q)
    pts = []
    rng = np.random.default_rng(0)
    for i, g in enumerate(robot_geom.geometryObjects):
        geo, M = g.geometry, gdata.oMg[i]
        key = (i, per_geom)
        if key not in _SURFACE_CACHE:
            if hasattr(geo, "vertices"):
                _SURFACE_CACHE[key] = _mesh_surface(geo, per_geom, rng)
            elif isinstance(geo, coal.Box):
                h = np.asarray(geo.halfSide)
                _SURFACE_CACHE[key] = rng.uniform(-h, h, (80, 3))
            else:
                _SURFACE_CACHE[key] = np.zeros((0, 3))
        local = _SURFACE_CACHE[key]
        if len(local):
            pts.append(local @ M.rotation.T + M.translation)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


@dataclass
class WorldMap:
    counts: dict = field(default_factory=dict)  # voxel index -> frames that saw it occupied
    table_z: float | None = None
    frames: int = 0

    def integrate(self, frame: RGBD, robot_pts: np.ndarray, exclude: list[tuple[np.ndarray, float]] = (),
                  stride: int = 2) -> int:
        """Add one RGB-D frame. ``exclude`` lists (centre, radius) spheres to cut out (the target part)."""
        d = frame.depth[::stride, ::stride]
        v, u = np.nonzero(d > 0)
        z = d[v, u]
        v, u = v * stride, u * stride
        K = frame.K
        cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
        world = cam @ frame.T_world_cam[:3, :3].T + frame.T_world_cam[:3, 3]
        world = world[np.all((world > WORKSPACE_MIN) & (world < WORKSPACE_MAX), axis=1)]
        if len(robot_pts):
            near_robot, _ = cKDTree(robot_pts).query(world, distance_upper_bound=ROBOT_MARGIN)
            world = world[~np.isfinite(near_robot)]
        for centre, radius in exclude:
            world = world[np.linalg.norm(world - centre, axis=1) > radius]
        # table: the dominant horizontal layer near the bottom of the cloud
        hist, edges = np.histogram(world[:, 2], bins=np.arange(-0.05, 0.3, 0.004))
        z_table = edges[np.argmax(hist)] + 0.002
        self.table_z = z_table if self.table_z is None else 0.5 * (self.table_z + z_table)
        above = world[world[:, 2] > self.table_z + 0.012]
        keys, n = np.unique(np.floor(above / VOXEL).astype(np.int32), axis=0, return_counts=True)
        for k in map(tuple, keys[n >= MIN_HITS]):
            self.counts[k] = self.counts.get(k, 0) + 1
        self.frames += 1
        return int((n >= MIN_HITS).sum())

    def cut_object(self, T_world_obj: np.ndarray, boxes: list, margin: float = 0.02) -> int:
        """Remove the voxels the known object occupies (its boxes at its estimated pose, grown by ``margin``).

        The planner models the object itself exactly; its depth points in the map would only block the
        fingers from reaching it. Returns how many voxels were removed.
        """
        if not self.counts:
            return 0
        keys = list(self.counts.keys())
        centres = (np.array(keys) + 0.5) * VOXEL
        local = (centres - T_world_obj[:3, 3]) @ T_world_obj[:3, :3]
        inside = np.zeros(len(keys), bool)
        for half, centre in boxes:
            inside |= np.all(np.abs(local - np.array(centre)) <= np.array(half) + margin + 0.5 * VOXEL, axis=1)
        for k, hit in zip(keys, inside):
            if hit:
                del self.counts[k]
        return int(inside.sum())

    def carve(self, points: np.ndarray, radius: float) -> int:
        """Free the voxels within ``radius`` of ``points`` (e.g. the robot's body where it stands now:
        whatever the cameras saw there was the robot itself). Returns how many voxels were removed."""
        if not self.counts or not len(points):
            return 0
        keys = list(self.counts.keys())
        centres = (np.array(keys) + 0.5) * VOXEL
        d, _ = cKDTree(points).query(centres, distance_upper_bound=radius + 0.5 * VOXEL)
        hit = np.isfinite(d)
        for k, h in zip(keys, hit):
            if h:
                del self.counts[k]
        return int(hit.sum())

    def occupied(self) -> np.ndarray:
        if not self.counts:
            return np.zeros((0, 3))
        return (np.array(list(self.counts.keys())) + 0.5) * VOXEL

    def obstacles(self) -> list[tuple[str, coal.CollisionGeometry, pin.SE3]]:
        """Collision objects for the planner: the table half-space and an octree of everything above it."""
        out = [("table", coal.Box(3.0, 3.0, 0.1), pin.SE3(np.eye(3), np.array([0.3, 0.0, self.table_z - 0.05])))]
        occ = self.inflated(PLANNING_PAD_VOXELS)
        if len(occ):
            out.append(("fused_octree", coal.makeOctree(occ, VOXEL), pin.SE3.Identity()))
        return out

    def inflated(self, pad: int) -> np.ndarray:
        """Occupied voxel centres grown by ``pad`` voxels in every direction (a safety margin for planning)."""
        if not self.counts or pad <= 0:
            return self.occupied()
        keys = np.array(list(self.counts.keys()))
        r = np.arange(-pad, pad + 1)
        offsets = np.array(np.meshgrid(r, r, r)).reshape(3, -1).T
        grown = np.unique((keys[:, None, :] + offsets[None]).reshape(-1, 3), axis=0)
        return (grown + 0.5) * VOXEL
