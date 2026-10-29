"""SAM-6D pose estimation (PEM) on OpenVINO, from in-memory RGB-D and an object mask.

Same pre-processing as SAM-6D's run_inference_custom_openvino_cpu.py, but the template features are
computed once per object and the observation comes from arrays, not files.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import trimesh

SAM6D_PEM = Path(os.environ.get("SAM6D_ROOT", "/home/user/OpenVINO-SAM-6D/SAM-6D")) / "Pose_Estimation_Model"
sys.path.insert(0, str(SAM6D_PEM / "utils"))
from data_utils import get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose  # noqa: E402

IMG_SIZE, N_OBS, N_MODEL, N_TEM, N_VIEWS = 224, 2048, 1024, 5000, 42  # SAM-6D test_dataset config
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def _normalize(rgb: np.ndarray) -> np.ndarray:
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


def _template(tdir: Path, i: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(tdir / f"rgb_{i}.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)[:, :, :3]
    xyz = np.load(tdir / f"xyz_{i}.npy").astype(np.float32) / 1000.0
    mask = cv2.imread(str(tdir / f"mask_{i}.png"), cv2.IMREAD_GRAYSCALE) == 255
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    mask = mask[y1:y2, x1:x2]
    rgb = rgb[:, :, ::-1][y1:y2, x1:x2] * mask[:, :, None]
    xyz = xyz[y1:y2, x1:x2].reshape(-1, 3)
    choose = mask.astype(np.float32).flatten().nonzero()[0]
    idx = rng.choice(len(choose), N_TEM, replace=len(choose) <= N_TEM)
    choose = choose[idx]
    h, w = y2 - y1, x2 - x1
    # As the reference script: index from the unfloored row, then truncate the sum.
    rgb_choose = ((choose // w).astype(np.float32) * (IMG_SIZE / h) * IMG_SIZE
                  + (choose % w).astype(np.float32) * (IMG_SIZE / w)).astype(np.int32)
    return _normalize(np.ascontiguousarray(rgb)), rgb_choose.astype(np.int32), xyz[choose]


@dataclass
class Pose:
    R: np.ndarray        # object -> camera rotation
    t: np.ndarray        # object origin in the camera frame, metres
    score: float
    ms: float


class SAM6DPose:
    def __init__(self, cad_mm: Path, template_dir: Path, device: str = "CPU", seed: int = 0,
                 good_enough: float = 0.5) -> None:
        core = ov.Core()
        core.add_extension(str(SAM6D_PEM / "model/ov_pointnet2_op/build/libopenvino_operation_extension.so"))
        fe = core.compile_model(str(SAM6D_PEM / "model_save/ov_fe_model_cpu.xml"), device)
        self.pem = core.compile_model(str(SAM6D_PEM / "model_save/ov_pem_model_cpu.xml"), device)
        self.rng = np.random.default_rng(seed)
        self.good_enough = good_enough
        mesh = trimesh.load(cad_mm, force="mesh")
        self.model_points = mesh.sample(N_MODEL).astype(np.float32) / 1000.0
        self.radius = float(np.max(np.linalg.norm(self.model_points, axis=1)))
        tems = [_template(Path(template_dir) / "templates", i, self.rng) for i in range(N_VIEWS)]
        t0 = time.perf_counter()
        out = list(fe({"rgb_input": np.concatenate([t[0][None] for t in tems], axis=0)[None].reshape(1, -1, IMG_SIZE, IMG_SIZE),
                       "pts_input": np.concatenate([t[2][None] for t in tems], axis=1),
                       "choose_input": np.concatenate([t[1][None] for t in tems], axis=1)}).values())
        self.tem_pts, self.tem_feat = out[0], out[1]
        self.template_ms = (time.perf_counter() - t0) * 1000

    def estimate(self, rgb: np.ndarray, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray,
                 det_score: float = 1.0, hypotheses: int = 6) -> Pose | None:
        """rgb: HxWx3 uint8 RGB; depth_m: HxW metres; mask: HxW bool (object pixels).

        SAM-6D's answer depends on which observed points it samples, and its pose score tells a good
        answer from a flipped one. Run ``hypotheses`` samplings as one batch and keep the best-scoring.
        """
        mask = mask & (depth_m > 0)
        if mask.sum() <= 32:
            return None
        whole_pts = get_point_cloud_from_depth(depth_m.astype(np.float32), K)
        y1, y2, x1, x2 = get_bbox(mask)
        m = mask[y1:y2, x1:x2]
        choose_all = m.astype(np.float32).flatten().nonzero()[0]
        cloud_all = whole_pts[y1:y2, x1:x2].reshape(-1, 3)[choose_all]
        keep = np.linalg.norm(cloud_all - cloud_all.mean(0), axis=1) < self.radius * 1.2
        if keep.sum() < 4:
            return None
        choose_all, cloud_all = choose_all[keep], cloud_all[keep]
        # The reference flips the observed crop's channels after loading RGB; the network was trained so.
        crop = _normalize(np.ascontiguousarray(rgb[y1:y2, x1:x2][:, :, ::-1] * (m[:, :, None] > 0)))
        pts, chooses = [], []
        for _ in range(hypotheses):
            idx = self.rng.choice(len(choose_all), N_OBS, replace=len(choose_all) <= N_OBS)
            pts.append(cloud_all[idx])
            chooses.append(get_resize_rgb_choose(choose_all[idx], [y1, y2, x1, x2], IMG_SIZE))
        # One hypothesis per call: batching them multiplies the network's attention memory.
        best, t0 = None, time.perf_counter()
        self.last_hypotheses = []
        self.last_cloud = cloud_all
        for p, c in zip(pts, chooses):
            out = list(self.pem({
                "pts": p[None].astype(np.float32), "rgb": crop[None], "rgb_choose": c[None].astype(np.int32),
                "model": self.model_points[None], "dense_po": self.tem_pts, "dense_fo": self.tem_feat,
            }).values())
            score = float(np.asarray(out[2]).reshape(-1)[0])
            self.last_hypotheses.append((np.asarray(out[0][0]), np.asarray(out[1][0]), score))
            if best is None or score > best[2]:
                best = (np.asarray(out[0][0]), np.asarray(out[1][0]), score)
            if best[2] >= self.good_enough:
                break  # a confident answer; more samplings only cost time
        ms = (time.perf_counter() - t0) * 1000
        return Pose(R=best[0], t=best[1], score=best[2] * det_score, ms=ms)
