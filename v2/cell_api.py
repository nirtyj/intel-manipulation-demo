"""Cell API for the v2/v3 console: run the part flip as traditional manipulation or as a learned policy.

    uvicorn v2.cell_api:app --port 8000      (physicalai env; needs the sim and the perception service)

POST /run {"mode": "traditional" | "act"}   one flip, streamed as events
POST /reset                                  put the part back upside down on the green pad (operator action)
POST /stop                                   stop the current run; the arm holds
GET  /events                                 WebSocket event stream
GET  /perception/{camera}.png                latest annotated perception image (proxied)
GET  /                                       the console page
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from v2 import run_v2
from v2.objects import OBJECT
from v2.arm import PandaKinematics

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Talk to the Cell v2/v3")
_clients: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None
_state = {"cell": None, "busy": False, "stop": threading.Event(), "stats": {}}


def emit(type_: str, **payload) -> None:
    env = {"t": time.time(), "type": type_, "payload": payload}
    if _loop is None:
        return
    for q in list(_clients):
        _loop.call_soon_threadsafe(q.put_nowait, env)


class StopRun(Exception):
    pass


class EventTrial(run_v2.Trial):
    """run_v2's trial log, mirrored onto the event stream."""

    def log(self, stage: str, **kw) -> None:
        super().log(stage, **kw)
        emit("stage", stage=stage, **{k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()})
        if _state["stop"].is_set():
            raise StopRun()


def cell() -> run_v2.Cell:
    if _state["cell"] is None:
        from physicalai.robot.transport import SharedRobot

        robot = SharedRobot.attach(run_v2.ROBOT, connect_timeout=30)
        robot.connect()
        _state["cell"] = run_v2.Cell(robot, PandaKinematics())
    return _state["cell"]


def _run(mode: str) -> None:
    c = cell()
    stats = _state["stats"].setdefault(mode, {"runs": 0, "success": 0, "times_s": []})
    t0 = time.time()
    try:
        emit("run", mode=mode, state="started")
        if mode == "traditional":
            result = run_v2.flip(c, EventTrial())
        else:
            from v3.run_policy import run_policy_episode

            result = run_policy_episode(c, emit, _state["stop"])
        stats["runs"] += 1
        stats["success"] += int(bool(result.get("success")))
        stats["times_s"].append(round(time.time() - t0, 1))
        emit("run", mode=mode, state="done", result={k: v for k, v in result.items() if k != "events"}, stats=_state["stats"])
    except StopRun:
        emit("run", mode=mode, state="stopped")
    except Exception as exc:  # keep the API alive whatever happens
        emit("run", mode=mode, state="error", error=str(exc)[:300])
    finally:
        _state["busy"] = False


class RunBody(BaseModel):
    mode: str = "traditional"


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()


@app.get("/")
def console() -> FileResponse:
    return FileResponse(ROOT / "console" / "v2.html")


@app.post("/run")
def run(body: RunBody) -> dict:
    if body.mode not in ("traditional", "act"):
        raise HTTPException(422, "mode must be 'traditional' or 'act'")
    if _state["busy"]:
        raise HTTPException(409, "A run is in progress")
    _state["busy"] = True
    _state["stop"].clear()
    threading.Thread(target=_run, args=(body.mode,), daemon=True).start()
    return {"started": body.mode}


@app.post("/reset")
def reset() -> dict:
    if _state["busy"]:
        raise HTTPException(409, "A run is in progress")
    # Same randomisation as the demonstrations: up to 2 cm and 30 degrees, never shown to either mode.
    dx, dy, yaw = run_v2.reset_part(np.random.default_rng(), 0.02, 30.0)
    emit("reset", dx_mm=round(dx * 1000), dy_mm=round(dy * 1000), yaw_deg=round(yaw))
    return {"ok": True}


@app.post("/stop")
def stop() -> dict:
    _state["stop"].set()
    emit("stop")
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    return _state["stats"]


@app.get("/perception/{camera}.png")
def perception_image(camera: str) -> Response:
    try:
        png = urllib.request.urlopen(f"{run_v2.PERCEPTION}/annotated/{camera}.png", timeout=5).read()
    except OSError:
        raise HTTPException(404, "No image yet")
    return Response(png, media_type="image/png")


@app.websocket("/events")
async def events(ws: WebSocket) -> None:
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    _clients.add(q)
    try:
        await ws.send_json({"type": "hello", "payload": {"busy": _state["busy"], "stats": _state["stats"],
                                                         "object": OBJECT.title}})
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(q)


__all__ = ["app", "emit", "json", "np"]
