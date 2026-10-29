"""Convert recorded v2 episodes into a LeRobot dataset for physicalai-train.

Only successful episodes are kept: the policy should imitate flips that worked.

    python -m v3.convert_lerobot runs/dataset --out datasets/panda_part_flip --repo-id local/panda_part_flip
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

JOINTS = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
CAMERAS = ("wrist", "overview")


def episodes(raw: Path, include_failed: bool = False) -> list[Path]:
    out = []
    for ep in sorted(raw.glob("episode_*")):
        meta_path, data_path = ep / "meta.json", ep / "data.npz"
        if not (meta_path.exists() and data_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        if (meta.get("success") or include_failed) and meta.get("frames", 0) > 20:
            out.append(ep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default="local/panda_part_flip")
    ap.add_argument("--videos", action="store_true", help="store camera streams as videos (smaller) instead of images")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = episodes(a.raw)
    if not eps:
        raise SystemExit(f"No successful episodes under {a.raw}")
    first = json.loads((eps[0] / "meta.json").read_text())
    w, h = first["size"]
    dtype = "video" if a.videos else "image"
    features = {f"observation.images.{cam}": {"dtype": dtype, "shape": (h, w, 3), "names": ["height", "width", "channels"]}
                for cam in CAMERAS}
    features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": JOINTS}
    features["action"] = {"dtype": "float32", "shape": (8,), "names": JOINTS}
    if a.out.exists():
        shutil.rmtree(a.out)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=int(first["fps"]), features=features, root=a.out,
                               robot_type="franka_panda", use_videos=a.videos)
    frames = 0
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "data.npz")
        for i in range(len(d["state"])):
            frame = {f"observation.images.{cam}": np.asarray(Image.open(ep / cam / f"{i:06d}.jpg").convert("RGB"))
                     for cam in CAMERAS}
            frame["observation.state"] = d["state"][i].astype(np.float32)
            frame["action"] = d["action"][i].astype(np.float32)
            frame["task"] = meta["task"]
            ds.add_frame(frame)
            frames += 1
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"wrote {a.out}: {len(eps)} episodes, {frames} frames at {first['fps']} fps")


if __name__ == "__main__":
    main()
