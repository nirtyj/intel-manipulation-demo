"""Orchestrator: commands in, a verified robot task out, every state change on one event stream.

    uvicorn orchestrator.app:app --port 8000

Talks to Intel's MuJoCo plugin over REST (SIM_URL, default http://127.0.0.1:8080) and serves the
console at / with the live event stream at /events.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from orchestrator.intent import parse
from orchestrator.sim import SIM_URL, Sim
from orchestrator.worldline_lite import ENTITIES, LABEL, OBJECTS, Clock, Value, Worldline

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = Path(os.environ.get("EVENT_LOG", ROOT / "events.jsonl"))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", "9090"))
TASK_TIMEOUT_S = 45.0
HOLDING_PHASES = {"close", "lift", "carry", "lower", "release"}
MOVING_PHASES = {"approach", "descend", "close", "lift", "carry", "lower", "release", "retreat"}

clock = Clock()
world = Worldline(clock)
sim = Sim()


# ---------------------------------------------------------------- event bus
class Bus:
    def __init__(self) -> None:
        self.clients: set[asyncio.Queue] = set()
        self.log = LOG_PATH.open("a", encoding="utf-8")

    def emit(self, type_: str, payload: dict, *, log: bool = True) -> dict:
        env = {"t": clock.iso(), "type": type_, "payload": payload}
        if log:
            self.log.write(json.dumps(env) + "\n")
            self.log.flush()
        for q in list(self.clients):
            with suppress(asyncio.QueueFull):
                q.put_nowait(env)
        return env


bus = Bus()


def say(text: str) -> None:
    bus.emit("transcript", {"speaker": "cell", "text": text})


stats = {"commands": 0, "understood": 0, "tasks": 0, "failed": 0, "blocked": 0,
         "command_to_motion_ms": [], "stop_to_hold_ms": []}


def emit_stats() -> None:
    def p50(a: list) -> float | None:
        return sorted(a)[(len(a) - 1) // 2] if a else None

    s = {k: v for k, v in stats.items() if not isinstance(v, list)}
    for k in ("command_to_motion_ms", "stop_to_hold_ms"):
        s[k] = {"last": stats[k][-1] if stats[k] else None, "p50": p50(stats[k]), "n": len(stats[k])}
    bus.emit("metric", s)


# ---------------------------------------------------------------- perception (LOOK)
async def look(source: str = "sim_overview") -> list[str]:
    """Read poses from the simulator (labelled as simulated perception) and update beliefs."""
    h, objs = await asyncio.gather(sim.health(), sim.objects())
    ep = h["episode"]
    pos = {o["joint"].split(":")[0]: o["position"] for o in objs}
    revised = []
    for e in ENTITIES:
        if e.startswith("bucket_"):
            p = ep["buckets"][e[7:]]["position"]
            v = Value(p[0], p[1])
        else:
            p = pos[e]
            v = Value(p[0], p[1], ep["in_bucket"].get(e))
        if world.observe(e, v, source):
            revised.append(e)
        bus.emit("belief", world.envelope(e))
    bus.emit("look", {"source": source, "revised": revised})
    return revised


async def observe_from_robot(obj: str, bucket: str) -> None:
    h, objs = await asyncio.gather(sim.health(), sim.objects())
    p = next(o["position"] for o in objs if o["joint"] == f"{obj}:joint")
    world.observe(obj, Value(p[0], p[1], h["episode"]["in_bucket"].get(obj)), "robot")
    b = h["episode"]["buckets"][bucket]["position"]
    world.observe(f"bucket_{bucket}", Value(b[0], b[1]), "robot")
    for e in (obj, f"bucket_{bucket}"):
        bus.emit("belief", world.envelope(e))


# ---------------------------------------------------------------- task director
class Director:
    def __init__(self) -> None:
        self.state = "IDLE"
        self.task_text = "No task"
        self.phase: str | None = None
        self.stop_latched = False
        self.pending: tuple[str, str] | None = None
        self.job: asyncio.Task | None = None
        self.pending_clarify: str | None = None
        self.phase_at_stop: str | None = None

    @property
    def busy(self) -> bool:
        return self.job is not None and not self.job.done()

    def set(self, state: str, text: str | None = None) -> None:
        self.state = state
        if text is not None:
            self.task_text = text
        bus.emit("task_state", self.snapshot())

    def snapshot(self) -> dict:
        return {"state": self.state, "task": self.task_text, "robot_phase": self.phase,
                "stop_latched": self.stop_latched}

    def start(self, obj: str, bucket: str, t_cmd: float | None, *, verify: bool = False) -> None:
        self.pending = (obj, bucket)
        self.job = asyncio.create_task(self._place(obj, bucket, t_cmd, verify))

    async def _place(self, obj: str, bucket: str, t_cmd: float | None, force_verify: bool) -> None:
        be = f"bucket_{bucket}"
        name = LABEL[obj]
        try:
            self.phase = None
            self.set("RESOLVE", f"{name} → {bucket} bin")
            await asyncio.sleep(0.3)
            doubt = [obj, be] if force_verify else [e for e in (obj, be) if world.doubtful(e)]
            if doubt:
                self.set("VERIFY")
                lines = [f"{cap(LABEL[e])}: last seen {fmt_age(world.age(e))}, verifying" for e in doubt]
                bus.emit("banner", {"kind": "verify", "text": " · ".join(lines)})
                say(". ".join(lines) + ".")
                await asyncio.sleep(0.8)
                revised = await look()
                if revised:
                    say(f"Revised: the {' and '.join(LABEL[e] for e in revised)} moved. Using the new position.")
                await asyncio.sleep(0.4)
            inside = world.beliefs[obj].value.inside
            if inside:
                say(f"The {name} is already in the {inside} bin. I only pick from the table.")
                self.set("FAILED")
                return
            self.set("EXECUTE")
            h = await sim.health()
            prev_episode = h["autopilot"].get("episode")
            await sim.instruct(obj, bucket)
            if h["autopilot"].get("mode") != "drive":
                await sim.autopilot("drive")
            moved_logged, deadline = False, time.monotonic() + TASK_TIMEOUT_S
            while time.monotonic() < deadline:
                await asyncio.sleep(0.25)
                a = (await sim.health())["autopilot"]
                if a.get("episode") == prev_episode:
                    continue  # the new episode has not started yet
                if a.get("phase") != self.phase:
                    self.phase = a.get("phase")
                    bus.emit("task_state", self.snapshot(), log=False)
                if t_cmd is not None and not moved_logged and self.phase in MOVING_PHASES:
                    moved_logged = True
                    ms = round((time.monotonic() - t_cmd) * 1000)
                    stats["command_to_motion_ms"].append(ms)
                    bus.emit("metric", {"name": "command_to_motion_ms", "value": ms})
                if a.get("outcome") in {"success", "failed"}:
                    break
            outcome = a.get("outcome")
            self.set("CONFIRM")
            await observe_from_robot(obj, bucket)
            if outcome == "success":
                stats["tasks"] += 1
                self.pending = None
                self.set("DONE")
                say(f"Done. The {name} is in the {bucket} bin.")
            else:
                stats["failed"] += 1
                self.set("FAILED")
                why = a.get("failure") or ("timed out" if outcome != "failed" else "the attempt failed")
                say(f"That didn't work: {why}. Say “look” and try again.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the console alive whatever the sim does
            self.set("FAILED")
            say(f"Something went wrong talking to the simulator: {exc}")
        finally:
            emit_stats()

    async def stop(self, source: str) -> None:
        t0 = time.monotonic()
        self.stop_latched = True
        self.phase_at_stop = self.phase if self.busy else None
        if self.busy:
            self.job.cancel()
        await sim.autopilot("off")
        for _ in range(40):
            if (await sim.health())["autopilot"].get("mode") == "off":
                break
            await asyncio.sleep(0.02)
        ms = round((time.monotonic() - t0) * 1000)
        stats["stop_to_hold_ms"].append(ms)
        if self.state not in {"IDLE", "DONE", "FAILED"}:
            self.set("STOPPED")
        else:
            bus.emit("task_state", self.snapshot())
        bus.emit("stop", {"source": source, "latched": True, "stop_to_hold_ms": ms})
        bus.emit("banner", {"kind": "stop", "text": "STOP latched. Autopilot off, arm holding.", "sticky": True})
        say("Stopped. Holding position. Say “resume” to continue.")
        emit_stats()

    async def resume(self) -> None:
        if not self.stop_latched:
            say("Nothing to resume.")
            return
        self.stop_latched = False
        bus.emit("stop", {"latched": False})
        bus.emit("banner", {"kind": "clear"})
        if self.pending:
            if self.phase_at_stop in HOLDING_PHASES:
                # The demonstrator plans from a resting object; put the arm home so the object lands first.
                say("Resuming. Homing the arm first, then checking the world.")
                await sim.post("/home")
                await asyncio.sleep(2.5)
            else:
                say("Resuming. Checking the world first.")
            self.start(*self.pending, None, verify=True)
        else:
            self.set("IDLE", "No task")
            say("Resumed.")


director = Director()


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def fmt_age(s: float) -> str:
    return f"{round(s)} s ago" if s < 90 else f"{round(s / 60)} min ago"


# ---------------------------------------------------------------- commands
async def handle_command(text: str, source: str = "typed", extra: dict | None = None) -> dict:
    t_cmd = time.monotonic()
    stats["commands"] += 1
    bus.emit("transcript", {"speaker": "operator", "text": text, "source": source, **(extra or {})})
    it = parse(text, director.pending_clarify)
    bus.emit("intent", it.as_dict())
    if it.intent != "none":
        stats["understood"] += 1
    if it.intent == "stop":
        await director.stop(source)
    elif it.intent == "resume":
        await director.resume()
    elif it.intent == "clarify":
        director.pending_clarify = it.object
        say(it.question or "Can you say that again?")
    elif it.intent == "where_is":
        e = it.object
        st = world.state(e)
        b = world.beliefs[e]
        where = "Somewhere new: the operator said it moved and I haven't looked yet" if b.note else (
            f"In the {b.value.inside} bin" if b.value.inside else f"On the table, {b.value.text().split(', ', 1)[1]}")
        msg = f"{where}. Seen {fmt_age(world.age(e))}, {round(world.confidence(e) * 100)}% confident."
        if st == "stale":
            msg += " That's old. Say “look” and I'll check."
        say(msg)
    elif it.intent == "look":
        if director.busy:
            say("One task at a time. I'll look when this one finishes.")
        else:
            rev = await look()
            say(f"Looked. Revised: {', '.join(LABEL[e] for e in rev)}." if rev else "Looked. Everything is where I thought.")
    elif it.intent == "observe":
        world.testimony(it.entity)
        bus.emit("belief", world.envelope(it.entity))
        say(f"Noted. I'll check the {LABEL[it.entity]} before I use it.")
    elif it.intent == "place":
        director.pending_clarify = None
        if director.stop_latched:
            say("STOP is latched. Say “resume” first.")
        elif director.busy:
            say(f"One task at a time. I'm still on: {director.task_text}.")
        else:
            director.start(it.object, it.bucket, t_cmd)
    else:
        say("I didn't catch an object and a bin. Try “put the battery in the red bin”.")
    emit_stats()
    return it.as_dict()


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    for _ in range(60):
        try:
            await sim.health()
            break
        except httpx.HTTPError:
            await asyncio.sleep(2)
    await sim.auto_reset(False)  # the director owns the episode; no surprise re-layouts
    await sim.autopilot("off")
    await look()
    say("Ready. Three objects, three bins, all seen just now.")
    ticker = asyncio.create_task(belief_ticker())
    yield
    ticker.cancel()
    await sim.close()


async def belief_ticker() -> None:
    """Ages and confidences change with time; push them every 2 s (not written to the log)."""
    while True:
        await asyncio.sleep(2)
        bus.emit("beliefs", {"beliefs": world.all()}, log=False)


app = FastAPI(title="Talk to the Cell orchestrator", lifespan=lifespan)


class CommandBody(BaseModel):
    text: str


class ObserveBody(BaseModel):
    entity: str


class ClockBody(BaseModel):
    seconds: float = 120.0


class BinBody(BaseModel):
    color: str


@app.get("/")
def console() -> FileResponse:
    return FileResponse(ROOT / "console" / "index.html")


@app.get("/config")
def config() -> dict:
    return {"viewer_port": VIEWER_PORT, "sim_url": SIM_URL}


@app.post("/command")
async def command(body: CommandBody) -> dict:
    return await handle_command(body.text)


@app.post("/voice")
async def voice() -> dict:
    raise HTTPException(501, "Voice is the next step: WhisperPipeline on OpenVINO GenAI")


@app.get("/beliefs")
def beliefs() -> list[dict]:
    return world.all()


@app.get("/beliefs/{entity}/why")
def why(entity: str) -> dict:
    if entity not in world.beliefs:
        raise HTTPException(404, f"Unknown entity {entity}")
    return world.why(entity)


@app.post("/observe")
def observe(body: ObserveBody) -> dict:
    if body.entity not in world.beliefs:
        raise HTTPException(404, f"Unknown entity {body.entity}")
    world.testimony(body.entity)
    return bus.emit("belief", world.envelope(body.entity))


@app.post("/look")
async def look_route() -> dict:
    if director.busy:
        raise HTTPException(409, "A task is running")
    return {"revised": await look()}


@app.post("/stop")
async def stop() -> dict:
    await director.stop("button")
    return director.snapshot()


@app.post("/resume")
async def resume() -> dict:
    await director.resume()
    return director.snapshot()


# --- demo controls: things the operator does to the world, or to the clock, for a take
@app.post("/demo/advance-clock")
def advance_clock(body: ClockBody) -> dict:
    clock.offset += body.seconds
    bus.emit("demo", {"action": "advance_clock", "seconds": body.seconds})
    bus.emit("beliefs", {"beliefs": world.all()}, log=False)
    return {"offset_s": clock.offset}


SPOTS = [(0.56, 0.27), (0.68, 0.12), (0.62, -0.04), (0.68, -0.20), (0.56, -0.33), (0.70, 0.30)]


@app.post("/demo/move-bin")
async def move_bin(body: BinBody) -> dict:
    """Move a bin in the simulator without telling Worldline (what dragging it in the viewer does)."""
    if body.color not in ("red", "blue", "black"):
        raise HTTPException(422, "color must be red, blue or black")
    h, objs = await asyncio.gather(sim.health(), sim.objects())
    if any(c == body.color for c in h["episode"]["in_bucket"].values()):
        raise HTTPException(409, f"The {body.color} bin has an object in it; move an empty bin")
    here = h["episode"]["buckets"][body.color]["position"]
    others = [b["position"] for c, b in h["episode"]["buckets"].items() if c != body.color]
    others += [o["position"] for o in objs if not o["joint"].startswith("bucket_")]

    def clearance(s: tuple[float, float]) -> float:
        return min(math.hypot(s[0] - p[0], s[1] - p[1]) for p in others)

    spots = [s for s in SPOTS if math.hypot(s[0] - here[0], s[1] - here[1]) > 0.1]
    x, y = max(spots, key=clearance)
    await sim.move_object(f"bucket_{body.color}:joint", x, y)
    bus.emit("demo", {"action": "move_bin", "color": body.color, "to": [x, y]})
    return {"color": body.color, "to": [x, y]}


@app.post("/demo/reset")
async def demo_reset() -> dict:
    if director.busy:
        director.job.cancel()
    await sim.autopilot("off")
    await sim.post("/reset")
    await asyncio.sleep(1.0)
    director.pending = None
    director.stop_latched = False
    director.set("IDLE", "No task")
    bus.emit("banner", {"kind": "clear"})
    await look()
    say("Scene reset. Fresh batch on the table.")
    return director.snapshot()


@app.post("/demo/reset-counters")
def reset_counters() -> dict:
    for k, v in stats.items():
        stats[k] = [] if isinstance(v, list) else 0
    emit_stats()
    return stats


# --- live streams
@app.get("/cam/{name}")
async def camera(name: str) -> StreamingResponse:
    """Proxy the simulator's MJPEG stream so the console needs one origin for video."""
    client = httpx.AsyncClient(base_url=SIM_URL, timeout=None)
    req = client.build_request("GET", f"/cameras/{name}/mjpeg")
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        await resp.aclose()
        await client.aclose()
        raise HTTPException(resp.status_code, f"No camera {name}")

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(body(), media_type=resp.headers.get("content-type", "multipart/x-mixed-replace"))


@app.websocket("/events")
async def events(ws: WebSocket) -> None:
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    bus.clients.add(q)
    try:
        await ws.send_json({"t": clock.iso(), "type": "snapshot", "payload": {
            "beliefs": world.all(), "task": director.snapshot(), "viewer_port": VIEWER_PORT}})
        emit_stats()
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        bus.clients.discard(q)


__all__ = ["app", "OBJECTS"]
