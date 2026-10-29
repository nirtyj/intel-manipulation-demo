"""Fetch the YCB 005 tomato soup can (Google 16k scan, CC BY 4.0) and write the two copies the cell uses.

    python scripts/prepare_ycb.py            (perception env: trimesh, pillow)

- scenes/soup_can/assets/soup_can.{obj,png}   metres, for MuJoCo (visual mesh; collision is a cylinder)
- models/soup_can/soup_can.{obj,mtl,png}      millimetres, for SAM-6D (model points and templates)

Both are recentred on the can's middle with +z through its top, the frame v2/objects.py assumes. The
texture is downsampled to 1024 px: the cameras see the can at ~100-300 px across.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
URL = "https://ycb-benchmarks.s3.amazonaws.com/data/google/005_tomato_soup_can_google_16k.tgz"
MEMBER = "005_tomato_soup_can/google_16k/"


def fetch() -> tuple[trimesh.Trimesh, Image.Image]:
    data = urllib.request.urlopen(URL, timeout=120).read()
    out = ROOT / "runs" / "ycb_005"
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(out, members=[m for m in tf.getmembers() if m.name.startswith(MEMBER)], filter="data")
    src = out / MEMBER
    mesh = trimesh.load(src / "textured.obj", force="mesh", process=False)
    return mesh, Image.open(src / "texture_map.png").convert("RGB")


def write(mesh: trimesh.Trimesh, texture: Image.Image, out_dir: Path, name: str, scale: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = mesh.copy()
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2)  # the can's middle
    m.apply_scale(scale)
    m.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=texture)
    text, _ = trimesh.exchange.obj.export_obj(m, include_texture=True, mtl_name=f"{name}.mtl", return_texture=True)
    (out_dir / f"{name}.obj").write_text(text)
    material = next(l.split()[1] for l in text.splitlines() if l.startswith("usemtl"))
    (out_dir / f"{name}.mtl").write_text(f"newmtl {material}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {name}.png\n")
    texture.save(out_dir / f"{name}.png")
    print(f"wrote {out_dir / name}.obj: extents {np.round(m.extents, 4).tolist()}")


def main() -> None:
    mesh, texture = fetch()
    texture = texture.resize((1024, 1024), Image.LANCZOS)
    write(mesh, texture, ROOT / "scenes/soup_can/assets", "soup_can", 1.0)
    write(mesh, texture, ROOT / "models/soup_can", "soup_can", 1000.0)


if __name__ == "__main__":
    main()
