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

from v2.objects import OBJECT
from v2.arm import GRIP_CLOSED, GRIP_OPEN, HOME_DEG, PandaKinematics
from v2.world_map import WorldMap

PART_BOXES = OBJECT.boxes  # (half extents, centre) in the part frame (v2/objects.py)
PART_UPRIGHT_Z = OBJECT.upright_z  # part centre above the table when it stands on its base


# ------------------------------------------------------------------ grasp library
FREE_MOTION_PART_MARGIN = 0.015  # extra room around the part on free-space paths (tracking lag at speed)


def grasp_library() -> list[tuple[str, pin.SE3]]:
    """Side grasps round the part's middle, as tool poses in the part frame (tool z = approach, y = finger
    closing). The L-part: its four column faces. The soup can: every 30 degrees round it."""
    legacy = {180.0: "-x", 0.0: "+x", -90.0: "-y", 90.0: "+y"}  # the L-part's grasp names
    grasps = []
    for a in OBJECT.approaches_deg:
        side = np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0])
        approach_name = legacy[a] if not OBJECT.axial else f"{a:+.0f}"
        for roll_name, x_axis in (("up", np.array([0, 0, 1.0])), ("down", np.array([0, 0, -1.0]))):
            z = -np.round(side, 12)            # tool z points into the part, from that side
            x = x_axis                         # tool x along the part's axis
            y = np.cross(z, x)                 # fingers close across the column / the can
            # Round the centre: turning the part over mirrors the grasp height about the centre, and a level
            # Panda hand needs >= 48 mm above the table, so only near-centre grasps work at both ends.
            # (The L-part's foot starts 25 mm below the centre, well clear of the fingers.)
            for h in OBJECT.grasp_heights:
                R = np.column_stack([x, y, z])
                grasps.append((f"approach{approach_name}-x{roll_name}-h{int(h * 1000)}",
                               pin.SE3(R, np.array([0.0, 0.0, h]))))
    return grasps


def part_geometries(margin: float = 0.0) -> list[tuple[coal.CollisionGeometry, np.ndarray]]:
    """The part's collision shapes (grown by ``margin``) with their centres in the part frame."""
    if OBJECT.cylinder is not None:
        r, half_h = OBJECT.cylinder
        return [(coal.Cylinder(r + margin, 2 * (half_h + margin)), np.zeros(3))]
    return [(coal.Box(*(2 * (np.array(half) + margin))), np.array(centre)) for half, centre in PART_BOXES]


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
        self.self_pairs = self._self_pairs()
        self.reset()

    def _self_pairs(self) -> list[tuple[int, int]]:
        """Robot link pairs that can meet: two or more joints apart and clear of each other at home."""
        geoms = self.robot_geom.geometryObjects
        cand = [(a, b) for a in range(self.n_robot) for b in range(a + 1, self.n_robot)
                if abs(int(geoms[a].parentJoint) - int(geoms[b].parentJoint)) >= 2]
        g = self.robot_geom.copy()
        for a, b in cand:
            g.addCollisionPair(pin.CollisionPair(a, b))
        gd = pin.GeometryData(g)
        pin.computeCollisions(self.model, self.data, g, gd, np.radians(HOME_DEG), False)
        return [pair for k, pair in enumerate(cand) if not gd.collisionResults[k].isCollision()]

    def reset(self) -> None:
        self.geom = self.robot_geom.copy()
        self.obstacle_ids: dict[str, list[int]] = {}
        self.held_ids: list[int] = []
        self._refresh()

    def _add(self, name: str, geometry: coal.CollisionGeometry, placement: pin.SE3, joint: int = 0) -> int:
        return self.geom.addGeometryObject(pin.GeometryObject(name, joint, placement, geometry))

    def set_obstacles(self, world: WorldMap, part_pose: np.ndarray | None, part_margin: float = 0.0) -> None:
        """Static obstacles: sensed table + octree, and the part where it was estimated to be (grown by
        ``part_margin``: free-space moves keep that much extra room, since the arm lags its command)."""
        self.reset()
        for name, g, M in world.obstacles():
            self.obstacle_ids.setdefault(name, []).append(self._add(name, g, M))
        if part_pose is not None:
            T = pin.SE3(part_pose[:3, :3], part_pose[:3, 3])
            for i, (g, centre) in enumerate(part_geometries(part_margin)):
                gid = self._add(f"part_{i}", g, T * pin.SE3(np.eye(3), centre))
                self.obstacle_ids.setdefault("part", []).append(gid)
        self._refresh()

    def attach_part(self, T_tcp_part: pin.SE3) -> None:
        """The part moves with the hand: attach its boxes to joint 7. Call after set_obstacles(map, None)."""
        T_j7_tcp = self.kin.model.frames[self.kin.tcp].placement  # tcp frame, relative to joint 7
        for i, (g, centre) in enumerate(part_geometries()):
            M = T_j7_tcp * T_tcp_part * pin.SE3(np.eye(3), centre)
            self.held_ids.append(self._add(f"held_{i}", g, M, self.hand_joint))
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
        for a, b in self.self_pairs:  # the arm against itself
            self.geom.addCollisionPair(pin.CollisionPair(a, b))
        arm = [i for i in range(self.n_robot) if int(self.geom.geometryObjects[i].parentJoint) <= 5]
        for h in self.held_ids:  # the held part against the arm (the hand and wrist hold it, so not those)
            for i in arm:
                self.geom.addCollisionPair(pin.CollisionPair(i, h))
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

    def blocking(self, q: np.ndarray) -> list[str]:
        """Names of the colliding (robot, obstacle) pairs at q, for logs."""
        pin.computeCollisions(self.model, self.data, self.geom, self.gdata, q, False)
        out = []
        for k, cp in enumerate(self.geom.collisionPairs):
            if self.gdata.activeCollisionPairs[k] and self.gdata.collisionResults[k].isCollision():
                out.append(f"{self.geom.geometryObjects[cp.first].name}~{self.geom.geometryObjects[cp.second].name}")
        return out

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


MIRROR_Y = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # Panda joints mirrored across the y = 0 plane


def above_config(kin: PandaKinematics, world: CollisionWorld, above: pin.SE3, q_lift: np.ndarray) -> np.ndarray | None:
    """IK for the place approach, seeded so the arm looks like the pick arm mirrored across the wall.

    The pads mirror each other across y = 0; the mirrored pick configuration (plus or minus the half-turn
    of joint 7 that sets the part upright) lands on a similar, well-conditioned arm, from which the
    transfer is a short, repeatable move. Falls back to seeding from the pick configuration.
    """
    mirrored = q_lift * MIRROR_Y
    for dq7 in (0.0, np.pi, -np.pi):
        seed = np.clip(mirrored + np.r_[np.zeros(6), dq7], kin.lo, kin.hi)
        q, e = kin.ik(above, seed, restarts=0)
        margin = float(np.min(np.minimum(q - kin.lo, kin.hi - q)))
        if e < 1e-3 and margin > 0.005 and world.free(q):
            return q
    return ik_free(kin, world, above, q_lift)


def cart_path(world: CollisionWorld, kin: PandaKinematics, q0: np.ndarray, goal: pin.SE3,
              n: int = 12) -> list[np.ndarray] | None:
    """Configurations along a straight tool line from q0 to goal (excluding q0), all collision-free."""
    start, q, out = kin.fk(q0), q0.copy(), []
    for s in np.linspace(0, 1, n + 1)[1:]:
        q, e = kin.ik(pin.SE3.Interpolate(start, goal, s), q, iters=60, restarts=0)
        if e > 2e-3 or not world.free(q):
            return None
        out.append(q.copy())
    return out


def over_the_top(world: CollisionWorld, kin: PandaKinematics, wmap: WorldMap, q_lift: np.ndarray,
                 q_above: np.ndarray, part_drop: float = 0.075, margin: float = 0.04) -> list[np.ndarray] | None:
    """The same transfer every time: straight up, one joint move across at a safe height, straight down.

    The height clears the tallest mapped obstacle between the two pads by the held part's reach below the
    tool (``part_drop``) and a margin. Demonstrations recorded with it all take the same route, which a
    learned policy can imitate; a randomised planner's routes it can only average.
    """
    T_lift, T_above = kin.fk(q_lift), kin.fk(q_above)
    occ = wmap.inflated(1)
    lo_xy = np.minimum(T_lift.translation[:2], T_above.translation[:2]) - 0.08
    hi_xy = np.maximum(T_lift.translation[:2], T_above.translation[:2]) + 0.08
    between = occ[np.all((occ[:, :2] > lo_xy) & (occ[:, :2] < hi_xy), axis=1)] if len(occ) else occ
    top = float(between[:, 2].max()) if len(between) else wmap.table_z
    z = max(T_lift.translation[2], T_above.translation[2], top + part_drop + margin)
    rise = cart_path(world, kin, q_lift, pin.SE3(T_lift.rotation, np.r_[T_lift.translation[:2], z]))
    # The descent, built upward from the place configuration so both ends share an arm branch.
    descent = cart_path(world, kin, q_above, pin.SE3(T_above.rotation, np.r_[T_above.translation[:2], z]))
    if rise is None or descent is None or not world.edge_free(rise[-1], descent[-1]):
        return None
    return [q_lift, *rise, *descent[::-1], q_above]


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


def cart_free(world: CollisionWorld, kin: PandaKinematics, q0: np.ndarray, goal: pin.SE3, n: int = 20,
              why: list | None = None):
    """Solve a straight tool line and check it; returns the end configuration or None (reason in ``why``)."""
    start = kin.fk(q0)
    q = q0.copy()
    for s in np.linspace(0, 1, n + 1)[1:]:
        q, e = kin.ik(pin.SE3.Interpolate(start, goal, s), q, iters=60, restarts=0)
        if e > 2e-3:
            if why is not None:
                why.append(f"IK at {s:.2f}")
            return None
        if not world.free(q):
            if why is not None:
                why.append(f"at {s:.2f}: " + ",".join(sorted(set(world.blocking(q)))[:3]))
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
        world.set_obstacles(wmap, T_world_part, part_margin=FREE_MOTION_PART_MARGIN)
        pre = pick * pin.SE3(np.eye(3), np.array([0, 0, -0.10]))
        q_pre = ik_free(kin, world, pre, q_now)
        if q_pre is None:
            tried.append(f"{name}: no collision-free pre-grasp"); continue
        path_pre = rrt_connect(world, q_now, q_pre, lo, hi)
        if path_pre is None:
            tried.append(f"{name}: no free path to pre-grasp"); continue
        why = []
        world.set_obstacles(wmap, T_world_part)  # exact boxes: the open fingers straddle the column
        q_grasp = cart_free(world, kin, q_pre, pick, why=why)  # the part stays an obstacle: open fingers straddle the column
        if q_grasp is None:
            tried.append(f"{name}: approach blocked ({why[0] if why else ''})"); continue
        world.set_obstacles(wmap, None)
        world.attach_part(g.inverse())
        world.ignore_held_vs(("table",))
        lift = pin.SE3(pick.rotation, pick.translation + np.array([0, 0, 0.12]))
        why = []
        q_lift = cart_free(world, kin, q_grasp, lift, why=why)
        if q_lift is None:
            tried.append(f"{name}: lift blocked ({why[0] if why else ''})"); continue
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
            q_above = above_config(kin, world, above, q_lift)
            if q_above is None:
                continue
            path_transfer = (over_the_top(world, kin, wmap, q_lift, q_above)
                             or rrt_connect(world, q_lift, q_above, lo, hi, seed=1))
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
        placed_pose = target.homogeneous  # the part now stands on the pad: keep clear of it on the way out
        world.set_obstacles(wmap, placed_pose)
        q_retreat = cart_free(world, kin, q_place, retreat)
        if q_retreat is None:
            q_retreat = kin.ik(retreat, q_place)[0]
        world.set_obstacles(wmap, placed_pose, part_margin=FREE_MOTION_PART_MARGIN)
        path_home = rrt_connect(world, q_retreat, q_home, lo, hi, seed=2) or [q_retreat, q_home]
        steps = [
            Step("open", "grip", grip=GRIP_OPEN, duration=0.5),
            Step("to pre-grasp", "path", waypoints=path_pre),
            Step("approach", "cart", goal=pick, duration=1.5),
            Step("close", "grip", grip=GRIP_CLOSED, duration=0.8),  # fingers close in ~0.3 s; keep the dwell short
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
