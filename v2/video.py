"""Render a recorded episode as a 1920x1080 demo video.

The main view re-renders Intel's MuJoCo model (the plugin's own scene composition) at 1280x720 from a
presentation camera, driven by the recorded joint states; the part is drawn from the replay-only log.
The right column shows the recorded wrist and scene camera frames; the bottom strip shows the stage and
the perception results. Perception and planning time is held on screen with the real CPU timings.

    MUJOCO_GL=osmesa python -m v2.video runs/dataset/episode_000003 --out media/v2_flip.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from physicalai_mujoco_plugin.compose import load_scene_model
from physicalai_mujoco_plugin.profiles.registry import get_profile

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes/can_flip/scene.xml"
W, H, FPS, UPSAMPLE = 1920, 1080, 30, 3
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, PANEL, INK, MUTED, ACCENT, OK = (12, 18, 21), (18, 26, 30), (226, 234, 236), (141, 159, 166), (90, 168, 255), (63, 208, 140)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


class Replay:
    def __init__(self) -> None:
        self.model = load_scene_model(SCENE, get_profile("franka_panda"))
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1920, 1080
        self.data = mujoco.MjData(self.model)
        names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.arm = [self.model.jnt_qposadr[names.index(n)] for n in
                    [next(x for x in names if x.endswith(f"joint{i}") and "finger" not in x) for i in range(1, 8)]]
        self.fingers = [self.model.jnt_qposadr[names.index(n)] for n in names if "finger_joint" in n]
        self.part = self.model.jnt_qposadr[names.index("part:joint")]
        self.renderer = mujoco.Renderer(self.model, 720, 1280)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.40, 0.0, 0.22]  # high enough to keep the transfer arc (up to ~0.85 m) in frame
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.75, 212.0, -22.0

    def render(self, state: np.ndarray, part: np.ndarray) -> np.ndarray:
        q = self.data.qpos
        q[self.arm] = np.radians(state[:7])
        for a in self.fingers:
            q[a] = float(state[7])
        if np.all(np.isfinite(part)):
            q[self.part:self.part + 7] = part
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def load_jpg(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path))
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)[:, :, ::-1] if img is not None else np.zeros((size[1], size[0], 3), np.uint8)


TITLES = {
    "traditional": "Talk to the Cell v2 · traditional manipulation · real physics, no ground truth in the loop",
    "act": "Talk to the Cell v3 · ACT policy, trained and run on Intel Xeon · no detector, no planner",
}


def compose(main: np.ndarray, wrist: np.ndarray, scene: np.ndarray, stage: str, lines: list[str],
            thumb: np.ndarray | None, overlay: tuple[np.ndarray, str, str] | None, t_label: str,
            title: str = TITLES["traditional"]) -> np.ndarray:
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(Image.fromarray(main), (0, 0))
    canvas.paste(Image.fromarray(wrist), (1280, 0))
    canvas.paste(Image.fromarray(scene), (1280, 480))
    d = ImageDraw.Draw(canvas)
    d.rectangle([1280, 960, W, H], fill=PANEL)
    d.text((1296, 975), "Simulated Intel RealSense", font=font(FONT_B, 22), fill=INK)
    d.text((1296, 1008), "D405 on the wrist (top) · D435 on a stand (middle)", font=font(FONT, 18), fill=MUTED)
    d.text((1296, 1036), "RGB-D with a RealSense-like depth noise model", font=font(FONT, 18), fill=MUTED)
    for (x, y, label) in ((1292, 10, "wrist D405"), (1292, 490, "scene D435")):
        d.rectangle([x - 4, y - 2, x + 128, y + 26], fill=(0, 0, 0))
        d.text((x, y), label, font=font(MONO, 18), fill=INK)
    # title band over the main view
    d.rectangle([0, 0, 1280, 48], fill=(0, 0, 0))
    d.text((16, 10), title, font=font(FONT_B, 20), fill=INK)
    d.text((1180, 12), t_label, font=font(MONO, 18), fill=MUTED)
    # bottom strip
    d.rectangle([0, 720, 1280, H], fill=PANEL)
    d.text((24, 740), stage, font=font(FONT_B, 40), fill=ACCENT)
    for i, line in enumerate(lines[:7]):
        d.text((24, 800 + 36 * i), line, font=font(FONT, 24), fill=INK if i == 0 else MUTED)
    if thumb is not None:
        canvas.paste(Image.fromarray(cv2.resize(thumb, (440, 330))), (820, 735))
    if overlay is not None:
        img, title, caption = overlay
        shade = Image.new("RGBA", (1280, 720), (0, 0, 0, 170))
        canvas.paste(shade, (0, 0), shade)
        im = Image.fromarray(cv2.resize(img, (800, 600)))
        canvas.paste(im, (240, 70))
        d.rectangle([240, 670, 1040, 714], fill=(0, 0, 0))
        d.text((256, 678), f"{title} — {caption}", font=font(FONT_B, 22), fill=OK)
    return np.asarray(canvas)


STAGE_TEXT = {
    "scene detection": "Find the part (scene camera)",
    "6-DoF pose": "Estimate its 6-DoF pose (wrist camera)",
    "plan": "Plan: grasp, IK, collision-free path",
    "to pre-grasp": "Move to the grasp",
    "approach": "Approach",
    "close": "Grasp (finger contact, 0.3 kg)",
    "lift": "Lift",
    "transfer and turn over": "Turn it over, around the wall",
    "lower": "Place upright",
    "release": "Release",
    "retreat": "Retreat",
    "home": "Done",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "media/v2_flip.mp4")
    ap.add_argument("--hold", type=float, default=2.5, help="seconds to show each perception result")
    ap.add_argument("--mode", choices=["traditional", "act"], default="traditional")
    ap.add_argument("--policy", type=Path, default=ROOT / "exports/act_part_flip", help="act mode: the export shown")
    a = ap.parse_args()
    title = TITLES[a.mode]
    ep = a.episode
    meta = json.loads((ep / "meta.json").read_text())
    data = np.load(ep / "data.npz")
    viz = np.load(ep / "viz.npz")["part"]
    marks = json.loads((ep / "marks.json").read_text())
    state = data["state"]
    n = len(state)
    rep = Replay()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                           "-preset", "medium", str(a.out)], stdin=subprocess.PIPE)
    by_frame: dict[int, list[dict]] = {}
    for m in marks:
        by_frame.setdefault(m["frame"], []).append(m)
    stage, lines, thumb = "Start", [meta.get("task", "")], None
    if a.mode == "act":
        cell = json.loads((a.policy / "cell.json").read_text()) if (a.policy / "cell.json").exists() else {}
        stage = "ACT policy acting"
        lines = ["Wrist + scene camera images and the joint state in, joint targets out, 10 times a second",
                 f"Trained with physicalai-train on {cell.get('demonstrations', '?')} demonstrations recorded from v2",
                 "bf16 on an Intel Xeon (AMX); exported to OpenVINO",
                 "Runs through physicalai's RobotRuntime + InferenceModel on the CPU (~12 ms per step)"]
    plan_info = {}
    written = 0
    for i in range(n):
        for m in by_frame.get(i, []):
            stage = STAGE_TEXT.get(m["label"], m["label"].capitalize())
            info = m["info"]
            if m["label"] == "scene detection":
                lines = [f"Grounding DINO on OpenVINO: \"orange and blue block\"  score {info.get('score')}",
                         f"{info.get('ms', 0) / 1000:.1f} s on CPU (detection + depth mask + height check)"]
            elif m["label"] == "6-DoF pose":
                lines = [f"SAM-6D on OpenVINO 2026.4: pose score {info.get('pose_score')}",
                         f"Grounding DINO {info.get('dino')} · SAM-6D {info.get('sam6d', 0) / 1000:.1f} s on CPU",
                         "Template features cached once per object"]
            elif m["label"] == "plan":
                plan_info = info
                lines = [f"Grasp {info.get('grasp')} from a pre-authored library",
                         f"Pinocchio IK + Coal collisions + RRT-Connect: {info.get('plan_ms')} ms",
                         f"Fused obstacle map: {info.get('voxels')} voxels (wrist + scene depth)",
                         f"Minimum clearance to obstacles: {info.get('clearance_mm')} mm"]
            elif plan_info:
                lines = [f"Executing through physicalai's robot runtime at 50 Hz",
                         f"Grasp {plan_info.get('grasp')} · clearance {plan_info.get('clearance_mm')} mm"]
            for cam, fname in m["images"].items():
                img = np.asarray(Image.open(ep / "marks" / fname).convert("RGB"))
                thumb = img
                title = "Grounding DINO + depth mask" if cam == "overview" else "Grounding DINO + SAM-6D 6-DoF"
                ms = info.get("ms") or info.get("total")
                caption = f"{(ms or 0) / 1000:.1f} s on CPU, shown in {a.hold:.1f} s"
                main = rep.render(state[i], viz[i])
                wrist = load_jpg(ep / "wrist" / f"{i:06d}.jpg", (640, 480))
                scene = load_jpg(ep / "overview" / f"{i:06d}.jpg", (640, 480))
                frame = compose(main, wrist, scene, stage, lines, thumb, (img, title, caption), f"t={i / 10:5.1f}s",
                                title=TITLES[a.mode])
                for _ in range(int(a.hold * FPS)):
                    ff.stdin.write(frame.tobytes())
                    written += 1
        wrist = load_jpg(ep / "wrist" / f"{i:06d}.jpg", (640, 480))
        scene = load_jpg(ep / "overview" / f"{i:06d}.jpg", (640, 480))
        j = min(i + 1, n - 1)
        for k in range(UPSAMPLE):
            s = k / UPSAMPLE
            st = state[i] + s * (state[j] - state[i])
            part = viz[i].copy()
            if np.all(np.isfinite(viz[[i, j]])):
                part[:3] = viz[i, :3] + s * (viz[j, :3] - viz[i, :3])
                part[3:] = slerp(viz[i, 3:], viz[j, 3:], s)
            frame = compose(rep.render(st, part), wrist, scene, stage, lines, thumb, None, f"t={(i + s) / 10:5.1f}s",
                            title=title)
            ff.stdin.write(frame.tobytes())
            written += 1
    # end card
    ok = meta.get("success")
    stage = "Done: part upright on the pad" if ok else "Episode ended"
    lines = [f"Result scored against the simulator afterwards: upright cos {meta.get('upright_cos')}, "
             f"{meta.get('on_pad_mm')} mm from the pad centre"]
    if a.mode == "act":
        lines += [f"{meta.get('ticks')} policy steps in {meta.get('total_s')} s, from pixels and joint angles only"]
    else:
        lines += [f"Perception error vs truth: {meta.get('pose_error_mm')} mm, {meta.get('pose_error_deg')} deg",
                  f"Total {meta.get('total_s')} s on a 4-vCPU container (perception dominates)"]
    frame = compose(rep.render(state[-1], viz[-1]), load_jpg(ep / "wrist" / f"{n - 1:06d}.jpg", (640, 480)),
                    load_jpg(ep / "overview" / f"{n - 1:06d}.jpg", (640, 480)), stage, lines, thumb, None, "", title=title)
    for _ in range(3 * FPS):
        ff.stdin.write(frame.tobytes())
        written += 1
    ff.stdin.close()
    ff.wait()
    print(f"wrote {a.out}: {written} frames, {written / FPS:.1f} s")


if __name__ == "__main__":
    main()
