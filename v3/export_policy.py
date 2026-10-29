"""Export the trained ACT checkpoint to OpenVINO for the cell, with the settings the runner needs.

    python -m v3.export_policy --ckpt runs/act/lightning_logs/version_0/checkpoints/last.ckpt   (train env)

Writes exports/act_part_flip/{act.xml, act.bin, manifest.json} (physicalai's export) and cell.json: the
control rate and camera size of the demonstrations, and how long to let the policy run (the longest
successful demonstration plus a margin).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=ROOT / "runs" / "dataset", help="the recorded episodes")
    ap.add_argument("--out", type=Path, default=ROOT / "exports" / "act_part_flip")
    a = ap.parse_args()
    from physicalai.policies import ACT

    from v3.convert_lerobot import episodes

    policy = ACT.load_from_checkpoint(str(a.ckpt), map_location="cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    policy.export(str(a.out), backend="openvino")
    metas = [json.loads((ep / "meta.json").read_text()) for ep in episodes(a.raw)]
    longest = max(m["frames"] / m["fps"] for m in metas)
    cell = {"fps": metas[0]["fps"], "size": metas[0]["size"], "episode_s": round(longest + 3.0, 1),
            "demonstrations": len(metas), "checkpoint": str(a.ckpt)}
    (a.out / "cell.json").write_text(json.dumps(cell, indent=1))
    print(f"exported {a.ckpt} -> {a.out}: {cell}")


if __name__ == "__main__":
    main()
