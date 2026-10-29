"""Open-vocabulary detection (Grounding DINO on OpenVINO) and a depth-based object mask.

The mask replaces SAM-6D's own instance-segmentation stage: Grounding DINO finds the box from a text
prompt, then the depth inside the box is cleaned of the supporting surface (RANSAC plane fit) and of
background, leaving the object's pixels.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import open3d as o3d
import openvino as ov
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_IR = Path(__file__).resolve().parents[2] / "models" / "grounding_dino_tiny.xml"
INPUTS = ["pixel_values", "input_ids", "token_type_ids", "attention_mask", "pixel_mask"]


@dataclass
class Detection:
    label: str
    score: float
    box: np.ndarray  # x1, y1, x2, y2 in pixels
    ms: float


class GroundingDino:
    def __init__(self, ir: Path = DEFAULT_IR, device: str = "CPU", shortest_edge: int = 800) -> None:
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.proc.image_processor.size = {"shortest_edge": shortest_edge, "longest_edge": 1333}
        self.model = ov.Core().compile_model(str(ir), device)

    def detect(self, rgb: np.ndarray, prompt: str, box_threshold: float = 0.25,
               text_threshold: float = 0.2) -> list[Detection]:
        import torch  # only for the processor's post-processing helpers

        img = Image.fromarray(rgb)
        inputs = self.proc(images=img, text=prompt, return_tensors="pt")
        t0 = time.perf_counter()
        out = self.model({k: inputs[k].numpy() for k in INPUTS})
        ms = (time.perf_counter() - t0) * 1000

        class _Out:
            logits = torch.from_numpy(out[0])
            pred_boxes = torch.from_numpy(out[1])

        res = self.proc.post_process_grounded_object_detection(
            _Out, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]])[0]
        dets = [Detection(str(lbl), float(s), b.numpy(), ms)
                for lbl, s, b in zip(res["labels"], res["scores"], res["boxes"])]
        return sorted(dets, key=lambda d: -d.score)


def backproject(depth_m: np.ndarray, K: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    z = depth_m[v, u]
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)


def mask_from_box(depth_m: np.ndarray, K: np.ndarray, box: np.ndarray, object_radius_m: float,
                  plane_dist_m: float = 0.006) -> np.ndarray:
    """Object pixels inside ``box``: drop the dominant plane (table) and anything far from the object."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = np.clip(np.round(box).astype(int), 0, [w - 1, h - 1, w - 1, h - 1])
    vv, uu = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    v, u = vv.ravel(), uu.ravel()
    valid = (depth_m[v, u] > 0.05) & (depth_m[v, u] < 5.0)
    v, u = v[valid], u[valid]
    pts = backproject(depth_m, K, v, u)
    keep = np.ones(len(pts), bool)
    if len(pts) > 200:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        _, inliers = pcd.segment_plane(plane_dist_m, 3, 200)
        if len(inliers) > 0.25 * len(pts):  # a real support surface, not a face of the object
            keep[inliers] = False
    if keep.sum() < 20:
        return np.zeros_like(depth_m, bool)
    # The object is the nearest compact cluster left: start at the median of the closer half.
    near = pts[keep][np.argsort(pts[keep][:, 2])[: max(20, keep.sum() // 2)]]
    centre = np.median(near, axis=0)
    keep &= np.linalg.norm(pts - centre, axis=1) < 1.3 * object_radius_m
    mask = np.zeros_like(depth_m, bool)
    mask[v[keep], u[keep]] = True
    return mask


def write_sam6d_detection(path: Path, mask: np.ndarray, det: Detection) -> None:
    """SAM-6D's detection_ism.json: one instance with an RLE mask, as its ISM stage writes it."""
    from pycocotools import mask as cocomask

    rle = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    x1, y1, x2, y2 = det.box.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"scene_id": 0, "image_id": 0, "category_id": 1, "score": det.score,
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "time": det.ms / 1000,
                                  "segmentation": rle}]))
