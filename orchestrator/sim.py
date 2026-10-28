"""Thin async client for Intel's MuJoCo plugin REST API (plus the /episode/instruction patch)."""

from __future__ import annotations

import os

import httpx

SIM_URL = os.environ.get("SIM_URL", "http://127.0.0.1:8080")


class Sim:
    def __init__(self, base: str = SIM_URL) -> None:
        self.base = base
        self.http = httpx.AsyncClient(base_url=base, timeout=5.0)

    async def health(self) -> dict:
        r = await self.http.get("/health")
        r.raise_for_status()
        return r.json()

    async def objects(self) -> list[dict]:
        r = await self.http.get("/objects")
        r.raise_for_status()
        return r.json()

    async def post(self, path: str, body: dict | None = None) -> dict:
        r = await self.http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    async def instruct(self, obj: str, bucket: str) -> dict:
        return await self.post("/episode/instruction", {"object": obj, "bucket": bucket})

    async def autopilot(self, mode: str) -> dict:
        return await self.post("/autopilot", {"mode": mode})

    async def auto_reset(self, enabled: bool) -> dict:
        return await self.post("/episode/auto-reset", {"enabled": enabled})

    async def move_object(self, joint: str, x: float, y: float, z: float = 0.0) -> dict:
        return await self.post(f"/objects/{joint}/pose", {"position": [x, y, z]})

    async def close(self) -> None:
        await self.http.aclose()
