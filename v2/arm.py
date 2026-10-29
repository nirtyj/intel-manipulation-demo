"""Panda kinematics, a can-flip grasp plan and a 50 Hz executor over Intel's robot runtime.

Kinematics: Pinocchio on the same Menagerie MJCF the simulator loads (FK matches MuJoCo to 1e-15 m).
Execution: physicalai's SharedRobot attaches to the simulator's owner and streams absolute joint targets.
No GPU pieces: IK, planning and collision checks run on the CPU.
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

MENAGERIE_GLOB = "/root/.cache/mujoco_menagerie/models/franka_emika_panda-*/panda.xml"
TCP_OFFSET = 0.110  # the plugin's end effector: 11 cm along the hand's z axis
GRIP_OPEN, GRIP_CLOSED = 0.04, 0.0  # actuator8: half the finger separation, metres
CONTROL_HZ = 50.0
V_MAX = 1.0  # rad/s peak joint speed in joint-space moves (Panda limits: 2.2-2.6 rad/s)
HOME_DEG = np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])  # a common Panda ready pose


def panda_xml() -> str:
    paths = glob.glob(MENAGERIE_GLOB)
    if not paths:
        raise FileNotFoundError("Menagerie franka_emika_panda not cached; start the simulator once first")
    return paths[0]


class PandaKinematics:
    """Seven arm joints (fingers locked), with a ``tcp`` frame at the plugin's end-effector point."""

    def __init__(self, xml: str | None = None) -> None:
        full = pin.buildModelFromMJCF(xml or panda_xml())
        lock = [full.getJointId(n) for n in ("finger_joint1", "finger_joint2")]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        hand = self.model.getFrameId("hand")
        self.model.addFrame(pin.Frame("tcp", self.model.frames[hand].parentJoint, hand,
                                      self.model.frames[hand].placement * pin.SE3(np.eye(3), np.array([0, 0, TCP_OFFSET])),
                                      pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.tcp = self.model.getFrameId("tcp")
        self.lo, self.hi = self.model.lowerPositionLimit.copy(), self.model.upperPositionLimit.copy()

    def fk(self, q: np.ndarray) -> pin.SE3:
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.tcp].copy()

    def _dls(self, goal: pin.SE3, q: np.ndarray, iters: int, tol: float, lo: np.ndarray, hi: np.ndarray):
        err = np.full(6, np.inf)
        for _ in range(iters):
            pin.framesForwardKinematics(self.model, self.data, q)
            err = pin.log6(self.data.oMf[self.tcp].actInv(goal)).vector
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp, pin.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(pin.integrate(self.model, q, np.clip(dq, -0.3, 0.3)), lo, hi)
        return q, float(np.linalg.norm(err))

    def ik(self, goal: pin.SE3, seed: np.ndarray, *, iters: int = 200, tol: float = 1e-4,
           margin: float = 0.02, restarts: int = 24, rng_seed: int = 7) -> tuple[np.ndarray, float]:
        """Damped least squares in the tool frame, inside the joint limits by ``margin`` rad.

        If the seed gets stuck (usually against a limit), retry from perturbed seeds and keep the
        converged solution closest to the seed, so consecutive keyframes stay continuous.
        """
        lo, hi = self.lo + margin, self.hi - margin
        q, e = self._dls(goal, np.clip(seed, lo, hi), iters, tol, lo, hi)
        if e < tol or restarts == 0:
            return q, e
        rng = np.random.default_rng(rng_seed)
        best, best_e, best_d = q, e, np.inf
        for k in range(restarts * 3):
            if k < restarts:  # perturbed seeds first, then anywhere in the joint box
                start = np.clip(seed + rng.normal(0.0, 0.3 + 1.2 * k / restarts, seed.shape), lo, hi)
            else:
                start = rng.uniform(lo, hi)
            qk, ek = self._dls(goal, start, iters, tol, lo, hi)
            dk = float(np.linalg.norm(qk - seed))
            if ek < tol and dk < best_d:
                best, best_e, best_d = qk, ek, dk
            elif best_d == np.inf and ek < best_e:
                best, best_e = qk, ek
        return best, best_e


# ------------------------------------------------------------------ grasp geometry
def side_grasp_pose(center: np.ndarray, approach_xy: np.ndarray, height: float) -> pin.SE3:
    """Tool z along the horizontal approach, fingers closing horizontally, tool x pointing down."""
    z = np.array([approach_xy[0], approach_xy[1], 0.0])
    z /= np.linalg.norm(z)
    x = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return pin.SE3(R, np.array([center[0], center[1], height]))


def rot_about_tool_z(pose: pin.SE3, angle: float) -> pin.SE3:
    return pose * pin.SE3(pin.utils.rotate("z", angle), np.zeros(3))


def offset(pose: pin.SE3, dx: float = 0.0, dz: float = 0.0, along_tool_z: float = 0.0) -> pin.SE3:
    p = pose.copy()
    p.translation = p.translation + np.array([dx, 0.0, dz]) + along_tool_z * p.rotation[:, 2]
    return p


@dataclass
class Segment:
    name: str
    kind: str  # "cart" (straight tool line), "joint" (joint-space blend) or "grip" (gripper only)
    duration: float
    goal: pin.SE3 | None = None
    q_goal: np.ndarray | None = None
    grip: float | None = None
    dq7: float = 0.0  # pure joint-7 turn (the tool point is on joint 7's axis, so only the roll changes)


@dataclass
class Plan:
    segments: list[Segment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def plan_can_flip(kin: PandaKinematics, q_now: np.ndarray, can_pos: np.ndarray, can_upside_down: bool,
                  place_xy: np.ndarray, via: np.ndarray, can_height: float = 0.101, via_z: float = 0.13) -> Plan:
    """Side-grasp the can, lift, turn it upright with a 180 degree tool roll, go around the wall, place, retreat.

    All poses are solved here, before anything moves, so an unreachable plan fails before motion.
    """
    plan = Plan()
    approach = can_pos[:2] / np.linalg.norm(can_pos[:2])  # radial approach from the base
    grip_h = can_pos[2] + 0.010  # a little above the can's centre
    grasp = side_grasp_pose(can_pos, approach, grip_h)
    # The fingers slide along the tool's y axis, which side_grasp_pose keeps horizontal: only a roll of
    # 0 or 180 degrees keeps them closing across the can's 66 mm diameter (its 101 mm height would not
    # fit the 80 mm hand). The flip is a half turn of joint 7, either way round; pick what fits its range.
    best, flip_margin = None, 0.15  # keep ~9 degrees clear of joint 7's stops after the half turn
    for roll in (0.0, np.pi):
        g = rot_about_tool_z(grasp, roll)
        for q7_seed in (None, 1.2, -1.2, 0.6, -0.6, 2.0, -2.0):
            seed = q_now.copy() if q7_seed is None else np.r_[q_now[:6], q7_seed]
            q_g, e1 = kin.ik(g, seed)
            if e1 > 1e-3:
                continue
            q_l, e2 = kin.ik(offset(g, dz=0.12), q_g)
            if e2 > 1e-3:
                continue
            turns = [np.pi, -np.pi] if can_upside_down else [0.0]
            ok = [t for t in turns if kin.lo[6] + flip_margin < q_l[6] + t < kin.hi[6] - flip_margin]
            if ok:
                best = (roll, g, q_g, ok[0])
                break
        if best:
            break
    if best is None:
        raise RuntimeError("No grasp keeps the flip inside joint 7's range")
    roll, grasp, q_grasp, turn = best
    plan.notes.append(f"grasp roll {np.degrees(roll):.0f} deg, flip {np.degrees(turn):+.0f} deg on joint 7")
    pre = offset(grasp, along_tool_z=-0.12)
    lift = offset(grasp, dz=0.12)  # a level tool at this reach tops out near z = 0.18 m (wrist range)
    # Pass in front of the wall with the tool pointing radially there, keeping the can upright.
    via_pose = rot_about_tool_z(side_grasp_pose(np.r_[via, 0.0], via / np.linalg.norm(via), via_z),
                                roll + turn)
    # Place: same grasp height above the table, mirrored if the can was turned over.
    place_h = (can_height - (grip_h - (can_pos[2] - can_height / 2))) + 0.004 if can_upside_down else grip_h + 0.004
    place_dir = place_xy / np.linalg.norm(place_xy)
    place = side_grasp_pose(np.r_[place_xy, 0.0], place_dir, place_h)
    place = rot_about_tool_z(place, roll + turn)
    above_place = offset(place, dz=0.12)
    retreat = offset(place, along_tool_z=-0.12)

    # Solve every keyframe once, in order, each seeded from the one before, and check them all
    # before anything moves.
    q_pre, e_pre = kin.ik(pre, q_grasp)
    q_lift, e_lift = kin.ik(lift, q_grasp)
    q_flip = q_lift + np.r_[np.zeros(6), turn]
    q_via, e_via = kin.ik(via_pose, q_flip)
    q_above, e_above = kin.ik(above_place, q_via)
    q_place, e_place = kin.ik(place, q_above)
    q_retreat, e_ret = kin.ik(retreat, q_place)
    keys = {"pre-grasp": (q_pre, e_pre), "lift": (q_lift, e_lift), "turn upright": (q_flip, 0.0),
            "around the wall": (q_via, e_via), "over the pad": (q_above, e_above), "place": (q_place, e_place),
            "retreat": (q_retreat, e_ret)}
    for name, (qk, e) in keys.items():
        margin = float(min(np.min(qk - kin.lo), np.min(kin.hi - qk)))
        if e > 1e-3 or margin < 0.0:
            raise RuntimeError(f"Keyframe '{name}' is not reachable (pose error {e:.3f}, limit margin {np.degrees(margin):.1f} deg)")
    plan.notes.append(f"7 keyframes solved; q7 {np.degrees(q_grasp[6]):.0f} -> {np.degrees(q_flip[6]):.0f} deg for the flip")

    plan.segments += [
        Segment("open", "grip", 0.6, grip=GRIP_OPEN),
        Segment("pre-grasp", "joint", 2.5, q_goal=q_pre),
        Segment("approach", "cart", 1.5, goal=grasp),
        Segment("close", "grip", 1.0, grip=GRIP_CLOSED),
        Segment("lift", "cart", 1.5, goal=lift),
    ]
    if can_upside_down:
        plan.segments.append(Segment("turn upright", "joint", 2.0, q_goal=q_flip))
    plan.segments += [
        Segment("around the wall", "joint", 2.0, q_goal=q_via),
        Segment("over the pad", "joint", 2.0, q_goal=q_above),
        Segment("lower", "cart", 1.5, goal=place),
        Segment("release", "grip", 0.8, grip=GRIP_OPEN),
        Segment("retreat", "cart", 1.2, goal=retreat),
        Segment("home", "joint", 2.5, q_goal=np.radians(HOME_DEG)),
    ]
    return plan


def smoothstep(s: float) -> float:
    return s * s * s * (10 + s * (-15 + 6 * s))  # quintic: zero velocity and acceleration at both ends


def interpolate(kin: PandaKinematics, seg: Segment, q0: np.ndarray, rate: float = CONTROL_HZ) -> list[np.ndarray]:
    """Joint targets for one segment at ``rate`` Hz, starting from ``q0``."""
    n = max(1, int(seg.duration * rate))
    if seg.kind == "grip":
        return [q0.copy() for _ in range(n)]
    if seg.kind == "joint":
        if False:
            q1 = q0 + np.r_[np.zeros(6), seg.dq7]
        else:
            q1 = seg.q_goal if seg.q_goal is not None else kin.ik(seg.goal, q0)[0]
        # Quintic peak speed is 1.875x the average: stretch the segment so no joint exceeds V_MAX.
        n = max(n, int(np.ceil(1.875 * np.max(np.abs(q1 - q0)) / V_MAX * rate)))
        return [q0 + smoothstep((i + 1) / n) * (q1 - q0) for i in range(n)]
    start = kin.fk(q0)
    out, q = [], q0.copy()
    for i in range(n):
        s = smoothstep((i + 1) / n)
        target = pin.SE3.Interpolate(start, seg.goal, s)
        q, _ = kin.ik(target, q, iters=30, restarts=0)
        out.append(q.copy())
    return out
