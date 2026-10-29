"""Collision-aware grasp and motion planning for the part flip (Pinocchio + Coal, CPU only).

- Collision model: the Panda's collision meshes (from the same Menagerie MJCF), the held part attached to
  the hand, the part on the table, and the fused obstacle map (sensed table plane + octree).
- Grasps: a pre-authored library in the part's frame. A grasp fixes the part in the hand, so the place pose
  of the hand is the target part pose times the same grasp; turning the part over falls out of that.
- Motion: RRT-Connect in joint space with shortcutting for free moves; straight tool lines for approach,
  lift, lower and retreat, all collision checked (contact with the part allowed only where it must be).
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field

import coal
import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree

from v2.arm import GRIP_CLOSED, GRIP_OPEN, HOME_DEG, PandaKinematics
from v2.world_map import WorldMap

PART_BOXES = [  # (half extents, centre) in the part frame: column and foot (scene.xml)
    ((0.025, 0.025, 0.055), (0.0, 0.0, 0.0)),
    ((0.02, 0.025, 0.015), (0.045, 0.0, -0.04)),
]
PART_UPRIGHT_Z = 0.055  # column centre above the table when it stands on its base


# ------------------------------------------------------------------ grasp library
def grasp_library() -> list[tuple[str, pin.SE3]]:
    """Side grasps on the column, as tool poses in the part frame (tool z = approach, y = finger closing)."""
    grasps = []
    for approach_name, approach in (("-x", np.array([1.0, 0, 0])), ("+x", np.array([-1.0, 0, 0])),
                                    ("-y", np.array([0, 1.0, 0])), ("+y", np.array([0, -1.0, 0]))):
        for roll_name, x_axis in (("up", np.array([0, 0, 1.0])), ("down", np.array([0, 0, -1.0]))):
            z = approach                       # tool z points into the part
            x = x_axis                         # tool x along the column
            y = np.cross(z, x)                 # fingers close across the 50 mm column
            # At the column's centre: turning the part over mirrors the grasp height about the centre (55 mm),
            # and a level Panda hand needs >= 50 mm above the table, so only near-centre grasps work at both ends.
            # The foot starts 25 mm below the centre, well clear of the fingers.
            for h in (0.0, 0.004, -0.004):
                R = np.column_stack([x, y, z])
                grasps.append((f"approach{approach_name}-x{roll_name}-h{int(h * 1000)}",
                               pin.SE3(R, np.array([0.0, 0.0, h]))))
    return grasps


# ------------------------------------------------------------------ collision world
class CollisionWorld:
    def __init__(self, kin: PandaKinematics) -> None:
        xml = glob.glob("/root/.cache/mujoco_menagerie/models/franka_emika_panda-*/panda.xml")[0]
        full, _, full_geom, _ = pin.buildModelsFromMJCF(xml)
        q_ref = pin.neutral(full)
        q_ref[full.joints[full.getJointId("finger_joint1")].idx_q] = GRIP_OPEN
        q_ref[full.joints[full.getJointId("finger_joint2")].idx_q] = GRIP_OPEN
        lock = [full.getJointId(n) for n in ("finger_joint1", "finger_joint2")]
        self.model, (self.robot_geom,) = pin.buildReducedModel(full, [full_geom], lock, q_ref)
        self.data = self.model.createData()
        self.kin = kin
        self.n_robot = self.robot_geom.ngeoms
        self.hand_joint = self.model.getJointId("joint7")
        self.reset()

    def reset(self) -> None:
        self.geom = self.robot_geom.copy()
        self.obstacle_ids: dict[str, list[int]] = {}
        self.held_ids: list[int] = []
        self._refresh()

    def _add(self, name: str, geometry: coal.CollisionGeometry, placement: pin.SE3, joint: int = 0) -> int:
        return self.geom.addGeometryObject(pin.GeometryObject(name, joint, placement, geometry))

    def set_obstacles(self, world: WorldMap, part_pose: np.ndarray | None) -> None:
        """Static obstacles: sensed table + octree, and the part where it was estimated to be."""
        self.reset()
        for name, g, M in world.obstacles():
            self.obstacle_ids.setdefault(name, []).append(self._add(name, g, M))
        if part_pose is not None:
            T = pin.SE3(part_pose[:3, :3], part_pose[:3, 3])
            for i, (half, centre) in enumerate(PART_BOXES):
                gid = self._add(f"part_{i}", coal.Box(*(2 * np.array(half))), T * pin.SE3(np.eye(3), np.array(centre)))
                self.obstacle_ids.setdefault("part", []).append(gid)
        self._refresh()

    def attach_part(self, T_tcp_part: pin.SE3) -> None:
        """The part moves with the hand: attach its boxes to joint 7. Call after set_obstacles(map, None)."""
        T_j7_tcp = self.kin.model.frames[self.kin.tcp].placement  # tcp frame, relative to joint 7
        for i, (half, centre) in enumerate(PART_BOXES):
            M = T_j7_tcp * T_tcp_part * pin.SE3(np.eye(3), np.array(centre))
            self.held_ids.append(self._add(f"held_{i}", coal.Box(*(2 * np.array(half))), M, self.hand_joint))
        self._refresh()

    def _refresh(self) -> None:
        self.geom.removeAllCollisionPairs()
        obstacles = [g for ids in self.obstacle_ids.values() for g in ids]
        movers = list(range(self.n_robot)) + self.held_ids
        for m in movers:
            for o in obstacles:
                name_m = self.geom.geometryObjects[m].name
                name_o = self.geom.geometryObjects[o].name
                if name_m.startswith("link0"):
                    continue  # the base is bolted to the table: whatever is there is the robot itself
                self.geom.addCollisionPair(pin.CollisionPair(m, o))
        self.gdata = pin.GeometryData(self.geom)
        self._ignore: set[int] = set()

    def ignore(self, names: tuple[str, ...]) -> None:
        """Ignore collisions with these obstacle groups (e.g. the part during the final approach)."""
        ids = {g for n in names for g in self.obstacle_ids.get(n, [])}
        for k, cp in enumerate(self.geom.collisionPairs):
            if cp.first in ids or cp.second in ids:
                self.gdata.deactivateCollisionPair(k)

    def ignore_held_vs(self, names: tuple[str, ...]) -> None:
        """Let the held part touch these groups (the table, at the end of the lower)."""
        ids = {g for n in names for g in self.obstacle_ids.get(n, [])}
        held = set(self.held_ids)
        for k, cp in enumerate(self.geom.collisionPairs):
            if (cp.first in held and cp.second in ids) or (cp.second in held and cp.first in ids):
                self.gdata.deactivateCollisionPair(k)

    def free(self, q: np.ndarray) -> bool:
        return not pin.computeCollisions(self.model, self.data, self.geom, self.gdata, q, True)

    def edge_free(self, qa: np.ndarray, qb: np.ndarray, res: float = 0.03) -> bool:
        n = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / res)))
        return all(self.free(qa + (qb - qa) * s) for s in np.linspace(0, 1, n)[1:])


def ik_free(kin: PandaKinematics, world: CollisionWorld, goal: pin.SE3, seed: np.ndarray,
            tries: int = 30, rng_seed: int = 11) -> np.ndarray | None:
    """An IK solution that is also collision-free: the seed's solution first, then others."""
    q, e = kin.ik(goal, seed)
    if e < 1e-3 and world.free(q):
        return q
    rng = np.random.default_rng(rng_seed)
    best, best_d = None, np.inf
    for _ in range(tries):
        start = np.clip(seed + rng.normal(0, 0.8, seed.shape), kin.lo, kin.hi) if rng.random() < 0.5 else rng.uniform(kin.lo, kin.hi)
        qk, ek = kin.ik(goal, start, restarts=0)
        if ek < 1e-3 and world.free(qk):
            d = float(np.linalg.norm(qk - seed))
            if d < best_d:
                best, best_d = qk, d
    return best


# ------------------------------------------------------------------ RRT-Connect
def rrt_connect(world: CollisionWorld, qa: np.ndarray, qb: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                step: float = 0.2, iters: int = 4000, seed: int = 0) -> list[np.ndarray] | None:
    if world.edge_free(qa, qb):
        return [qa, qb]
    rng = np.random.default_rng(seed)
    trees = [([qa], [-1]), ([qb], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        i = int(np.argmin([np.linalg.norm(n - target) for n in nodes]))
        d = target - nodes[i]
        dist = np.linalg.norm(d)
        new = target if dist <= step else nodes[i] + d / dist * step
        if world.edge_free(nodes[i], new):
            nodes.append(new)
            parents.append(i)
            return len(nodes) - 1
        return None

    for _ in range(iters):
        sample = rng.uniform(lo, hi)
        a, b = trees
        ia = extend(a, sample)
        if ia is not None:
            target = a[0][ia]
            while True:  # connect: grow b toward the new node until blocked or joined
                ib = extend(b, target)
                if ib is None:
                    break
                if np.linalg.norm(b[0][ib] - target) < 1e-9:
                    path_a, k = [], ia
                    while k != -1:
                        path_a.append(a[0][k]); k = a[1][k]
                    path_b, k = [], ib
                    while k != -1:
                        path_b.append(b[0][k]); k = b[1][k]
                    path = path_a[::-1] + path_b[1:]  # root of a ... root of b
                    if not np.allclose(path[0], qa):
                        path = path[::-1]
                    return shortcut(world, path, rng)
        trees = [trees[1], trees[0]]
    return None


def shortcut(world: CollisionWorld, path: list[np.ndarray], rng: np.random.Generator, tries: int = 120) -> list[np.ndarray]:
    path = list(path)
    for _ in range(tries):
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and world.edge_free(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path


# ------------------------------------------------------------------ the flip plan
@dataclass
class Step:
    name: str
    kind: str                     # "path" (joint waypoints), "cart" (tool line) or "grip"
    waypoints: list[np.ndarray] = field(default_factory=list)
    goal: pin.SE3 | None = None
    grip: float | None = None
    duration: float = 1.5


@dataclass
class FlipPlan:
    grasp: str
    steps: list[Step]
    T_tcp_part: pin.SE3
    clearance_m: float
    plan_ms: float
    notes: list[str] = field(default_factory=list)


def cart_free(world: CollisionWorld, kin: PandaKinematics, q0: np.ndarray, goal: pin.SE3, n: int = 20):
    """Solve a straight tool line and check it; returns the end configuration or None."""
    start = kin.fk(q0)
    q = q0.copy()
    for s in np.linspace(0, 1, n + 1)[1:]:
        q, e = kin.ik(pin.SE3.Interpolate(start, goal, s), q, iters=60, restarts=0)
        if e > 2e-3 or not world.free(q):
            return None
    return q


def min_clearance(world: CollisionWorld, wmap: WorldMap, configs: list[np.ndarray]) -> float:
    """Smallest distance between the robot's surface and the fused obstacle voxels along a path (metres)."""
    from v2.world_map import robot_surface_samples

    occ = wmap.occupied()
    if not len(occ):
        return float("inf")
    tree = cKDTree(occ)
    best = np.inf
    for q in configs[:: max(1, len(configs) // 25)]:
        pts = robot_surface_samples(world.robot_geom, world.model, world.data, pin.GeometryData(world.robot_geom), q, 150)
        d, _ = tree.query(pts)
        best = min(best, float(d.min()))
    return max(0.0, best - 0.5 * 0.015)


def plan_flip(kin: PandaKinematics, world: CollisionWorld, wmap: WorldMap, q_now: np.ndarray,
              T_world_part: np.ndarray, place_xy: np.ndarray) -> FlipPlan:
    t0 = time.perf_counter()
    lo, hi = kin.lo + 0.03, kin.hi - 0.03
    q_home = np.radians(HOME_DEG)
    Twp = pin.SE3(T_world_part[:3, :3], T_world_part[:3, 3])
    radial = Twp.translation[:2] / np.linalg.norm(Twp.translation[:2])
    table = wmap.table_z if wmap.table_z is not None else 0.0
    candidates = []
    for name, g in grasp_library():
        pick = Twp * g
        z_tool = pick.rotation[:, 2]
        if abs(z_tool[2]) > 0.3:          # approach must be roughly horizontal
            continue
        if pick.translation[2] < table + 0.048:  # a level hand lower than this meets the table
            continue
        align = float(z_tool[:2] @ radial)  # prefer approaching from the robot's side
        candidates.append((-align, name, g, pick))
    candidates.sort(key=lambda c: c[0])
    tried = []
    for _, name, g, pick in candidates:
        world.set_obstacles(wmap, T_world_part)
        pre = pick * pin.SE3(np.eye(3), np.array([0, 0, -0.10]))
        q_pre = ik_free(kin, world, pre, q_now)
        if q_pre is None:
            tried.append(f"{name}: no collision-free pre-grasp"); continue
        path_pre = rrt_connect(world, q_now, q_pre, lo, hi)
        if path_pre is None:
            tried.append(f"{name}: no free path to pre-grasp"); continue
        q_grasp = cart_free(world, kin, q_pre, pick)  # the part stays an obstacle: open fingers straddle the column
        if q_grasp is None:
            tried.append(f"{name}: approach blocked"); continue
        world.set_obstacles(wmap, None)
        world.attach_part(g.inverse())
        world.ignore_held_vs(("table",))
        lift = pin.SE3(pick.rotation, pick.translation + np.array([0, 0, 0.12]))
        q_lift = cart_free(world, kin, q_grasp, lift)
        if q_lift is None:
            tried.append(f"{name}: lift blocked"); continue
        world.set_obstacles(wmap, None)
        world.attach_part(g.inverse())
        placed = None
        for yaw_off in np.radians([0, 90, -90, 180]):
            yaw = np.arctan2(place_xy[1], place_xy[0]) + yaw_off
            target = pin.SE3(pin.utils.rotate("z", yaw), np.array([place_xy[0], place_xy[1], table + PART_UPRIGHT_Z + 0.004]))
            place = target * g
            if abs(place.rotation[2, 2]) > 0.3:
                continue
            above = pin.SE3(place.rotation, place.translation + np.array([0, 0, 0.12]))
            q_above = ik_free(kin, world, above, q_lift)
            if q_above is None:
                continue
            path_transfer = rrt_connect(world, q_lift, q_above, lo, hi, seed=1)
            if path_transfer is None:
                continue
            world.ignore_held_vs(("table",))
            q_place = cart_free(world, kin, q_above, place)
            if q_place is None:
                continue
            retreat = place * pin.SE3(np.eye(3), np.array([0, 0, -0.10]))
            placed = (target, place, above, q_above, path_transfer, q_place, retreat)
            break
        if placed is None:
            tried.append(f"{name}: no reachable, collision-free placement"); continue
        target, place, above, q_above, path_transfer, q_place, retreat = placed
        world.set_obstacles(wmap, None)
        q_retreat = cart_free(world, kin, q_place, retreat)
        if q_retreat is None:
            q_retreat = kin.ik(retreat, q_place)[0]
        path_home = rrt_connect(world, q_retreat, q_home, lo, hi, seed=2) or [q_retreat, q_home]
        steps = [
            Step("open", "grip", grip=GRIP_OPEN, duration=0.5),
            Step("to pre-grasp", "path", waypoints=path_pre),
            Step("approach", "cart", goal=pick, duration=1.5),
            Step("close", "grip", grip=GRIP_CLOSED, duration=1.0),
            Step("lift", "cart", goal=lift, duration=1.5),
            Step("transfer and turn over", "path", waypoints=path_transfer),
            Step("lower", "cart", goal=place, duration=1.6),
            Step("release", "grip", grip=GRIP_OPEN, duration=0.8),
            Step("retreat", "cart", goal=retreat, duration=1.2),
            Step("home", "path", waypoints=path_home),
        ]
        clearance = min_clearance(world, wmap, path_pre + path_transfer)
        notes = [f"grasp {name}", f"placement yaw {np.degrees(np.arctan2(target.rotation[1, 0], target.rotation[0, 0])):.0f} deg",
                 f"{len(tried)} grasps rejected before this one"]
        return FlipPlan(name, steps, g.inverse(), clearance, (time.perf_counter() - t0) * 1000, notes + tried[:6])
    raise RuntimeError("No feasible grasp: " + "; ".join(tried[:8]))
