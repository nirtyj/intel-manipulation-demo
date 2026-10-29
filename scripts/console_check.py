"""Rehearse the demo through the console API, exactly as the buttons do: Reset, then Run flip, per mode.

    python scripts/console_check.py traditional act traditional act      (cell running on localhost:8000)

Prints each run's scored result (the simulator's ground truth, read after the run).
"""

import asyncio
import json
import sys
import time
import urllib.request

import websockets

API = "127.0.0.1:8000"


def post(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(f"http://{API}{path}", data=json.dumps(body or {}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=120))


async def run(mode: str) -> dict:
    async with websockets.connect(f"ws://{API}/events", max_size=None) as ws:
        post("/reset")
        post("/run", {"mode": mode})
        t0 = time.time()
        while True:
            ev = json.loads(await ws.recv())
            p = ev.get("payload", {})
            if ev.get("type") == "run" and p.get("state") in ("done", "error", "stopped"):
                r = p.get("result", {})
                out = {"mode": mode, "state": p["state"], "success": r.get("success"), "upright_cos": r.get("upright_cos"),
                       "on_pad_mm": r.get("on_pad_mm"), "seconds": round(time.time() - t0, 1), "error": p.get("error")}
                print(json.dumps(out), flush=True)
                return out


if __name__ == "__main__":
    results = [asyncio.run(run(m)) for m in (sys.argv[1:] or ["traditional", "act"])]
    for mode in sorted({r["mode"] for r in results}):
        rs = [r for r in results if r["mode"] == mode]
        print(f"{mode}: {sum(bool(r['success']) for r in rs)}/{len(rs)} successful")
