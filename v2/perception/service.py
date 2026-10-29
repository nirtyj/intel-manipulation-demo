"""Perception service: Grounding DINO + SAM-6D on OpenVINO, behind HTTP.

    uvicorn v2.perception.service:app --port 8100      (in the perception env)

POST /coarse   scene camera -> coarse world position of the part
POST /six_dof  wrist camera -> 6-DoF world pose of the part, from this one view
POST /views/reset, /views/add, /views/pose   multi-view: capture the wrist camera from several viewpoints,
               then pick the pose all of them agree with
GET  /annotated/{camera}.png  the last annotated image of that camera (box, mask, pose axes)
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from v2.perception.pipeline import PartPerception, annotate, to_json
from v2.perception.sensors import fetch_rgbd

app = FastAPI(title="Talk to the Cell perception")
_lock = threading.Lock()
_state: dict = {"perception": None, "images": {}, "views": []}


def perception() -> PartPerception:
    if _state["perception"] is None:
        _state["perception"] = PartPerception()
    return _state["perception"]


@app.on_event("startup")
def _warm() -> None:
    perception()


@app.get("/health")
def health() -> dict:
    p = _state["perception"]
    return {"ready": p is not None, "template_ms": None if p is None else round(p.pose.template_ms)}


@app.post("/coarse")
def coarse() -> dict:
    with _lock:
        t0 = time.perf_counter()
        frame = fetch_rgbd("overview")
        pos, obs = perception().coarse_location(frame)
        _state["images"]["overview"] = annotate(frame, obs, "scene D435")
        return {"camera": "overview", "position": None if pos is None else np.round(pos, 4).tolist(),
                "detection_score": None if obs.detection is None else round(obs.detection.score, 3),
                "timings_ms": {**obs.timings_ms, "total": round((time.perf_counter() - t0) * 1000)}}


@app.post("/six_dof")
def six_dof(hypotheses: int = 3) -> dict:
    with _lock:
        t0 = time.perf_counter()
        frame = fetch_rgbd("wrist")
        obs = perception().six_dof(frame, hypotheses=hypotheses)
        _state["images"]["wrist"] = annotate(frame, obs, "wrist D405")
        return {"camera": "wrist", "T_world_obj": to_json(obs.T_world_obj), "pose_score": round(obs.score, 3),
                "fit": round(obs.fitness, 3), "fit_rmse_mm": round(obs.rmse_mm, 2),
                "detection_score": None if obs.detection is None else round(obs.detection.score, 3),
                "timings_ms": {**{k: round(v) for k, v in obs.timings_ms.items()},
                               "total": round((time.perf_counter() - t0) * 1000)}}


def _pose_response(obs, t0: float, views: int) -> dict:
    return {"camera": "wrist", "views": views, "T_world_obj": to_json(obs.T_world_obj), "pose_score": round(obs.score, 3),
            "fit": round(obs.fitness, 3), "fit_rmse_mm": round(obs.rmse_mm, 2), "per_view": obs.per_view,
            "up_vote": None if obs.up_vote is None else round(obs.up_vote, 3),
            "detection_score": None if obs.detection is None else round(obs.detection.score, 3),
            "timings_ms": {**{k: round(v) for k, v in obs.timings_ms.items()},
                           "total": round((time.perf_counter() - t0) * 1000)}}


@app.post("/views/reset")
def views_reset() -> dict:
    with _lock:
        _state["views"].clear()
        return {"views": 0}


@app.post("/views/add")
def views_add(hypotheses: int = 3) -> dict:
    """Capture the wrist camera now and keep its detection and SAM-6D hypotheses."""
    with _lock:
        t0 = time.perf_counter()
        frame = fetch_rgbd("wrist")
        view = perception().observe(frame, hypotheses=hypotheses)
        if view is None:
            d, m = perception().detect(frame)
            from v2.perception.pipeline import Observation

            _state["images"]["wrist"] = annotate(frame, Observation(frame.name, d, m), "wrist D405")
            return {"added": False, "views": len(_state["views"])}
        _state["views"].append(view)
        return {"added": True, "views": len(_state["views"]), "sam6d_score": round(view.sam6d_score, 3),
                "detection_score": round(view.detection.score, 3),
                "timings_ms": {**{k: round(v) for k, v in view.ms.items()}, "total": round((time.perf_counter() - t0) * 1000)}}


@app.post("/views/pose")
def views_pose() -> dict:
    """The pose that best explains every captured view."""
    with _lock:
        t0 = time.perf_counter()
        views = _state["views"]
        if not views:
            raise HTTPException(409, "No views captured")
        obs = perception().fuse(views)
        _state["images"]["wrist"] = annotate(views[-1].frame, obs, f"wrist D405, {len(views)} views")
        return _pose_response(obs, t0, len(views))


@app.post("/views/save")
def views_save(path: str) -> dict:
    """Write the captured views (RGB-D, mask, SAM-6D hypotheses) to an .npz, to study a refusal offline."""
    with _lock:
        views = _state["views"]
        out = {}
        for i, v in enumerate(views):
            out.update({f"v{i}_rgb": v.frame.rgb, f"v{i}_depth": v.frame.depth, f"v{i}_K": v.frame.K,
                        f"v{i}_T_world_cam": v.frame.T_world_cam, f"v{i}_mask": v.mask,
                        f"v{i}_cloud": v.cloud_world, f"v{i}_hyps": np.array(v.hypotheses)})
        np.savez_compressed(path, n=len(views), **out)
        return {"saved": path, "views": len(views)}


@app.get("/annotated/{camera}.png")
def annotated(camera: str) -> Response:
    img = _state["images"].get(camera)
    if img is None:
        raise HTTPException(404, f"No annotated image yet for {camera}")
    ok, png = cv2.imencode(".png", img[:, :, ::-1])
    return Response(png.tobytes(), media_type="image/png")
