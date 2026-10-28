"""Worldline, demo subset: what the system believes about each entity, where it came from, how stale it is.

Beliefs change only on a LOOK, after a robot action, or from operator testimony. Confidence decays as
c0 * exp(-age / tau). The demo clock can be advanced so staleness shows on camera.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

TAU_S = 120.0
FRESH_S, AGING_S, REVISED_SHOW_S = 45.0, 100.0, 25.0
MOVED_M = 0.03  # a pose change bigger than this contradicts the belief
C0 = {"sim_overview": 0.95, "robot": 0.9, "operator": 0.6}

OBJECTS = ("battery", "cpu", "resistor")
ENTITIES = (*OBJECTS, "bucket_red", "bucket_blue", "bucket_black")
LABEL = {
    "battery": "battery", "cpu": "CPU chip", "resistor": "resistor",
    "bucket_red": "red bin", "bucket_blue": "blue bin", "bucket_black": "black bin",
}


class Clock:
    def __init__(self) -> None:
        self.offset = 0.0

    def now(self) -> float:
        return time.time() + self.offset

    def iso(self, t: float | None = None) -> str:
        return datetime.fromtimestamp(self.now() if t is None else t, UTC).isoformat(timespec="milliseconds")


def region(x: float, y: float) -> str:
    """World frame: x forward from the robot base, y to its left."""
    lat = "left" if y > 0.12 else "right" if y < -0.12 else "center"
    dep = "near" if x < 0.45 else "far" if x > 0.62 else ""
    if not dep:
        return lat if lat == "center" else f"{lat} side"
    return f"{dep} {lat}"


@dataclass
class Value:
    x: float
    y: float
    inside: str | None = None  # bucket colour for objects

    def text(self) -> str:
        return f"in the {self.inside} bin" if self.inside else f"table, {region(self.x, self.y)}"

    def as_dict(self) -> dict:
        d = {"position": [round(self.x, 3), round(self.y, 3)], "region": region(self.x, self.y)}
        if self.inside:
            d["in"] = self.inside
        return d


@dataclass
class Belief:
    entity: str
    value: Value
    source: str
    at: float
    c0: float
    history: list[dict] = field(default_factory=list)
    revised_at: float | None = None
    note: str | None = None  # operator testimony: position unknown


class Worldline:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.beliefs: dict[str, Belief] = {}

    # ---- reading ----
    def age(self, e: str) -> float:
        return self.clock.now() - self.beliefs[e].at

    def confidence(self, e: str) -> float:
        b = self.beliefs[e]
        return b.c0 * math.exp(-self.age(e) / TAU_S)

    def state(self, e: str) -> str:
        b = self.beliefs[e]
        if b.note:
            return "stale"
        if b.revised_at is not None and self.clock.now() - b.revised_at < REVISED_SHOW_S:
            return "revised"
        a = self.age(e)
        return "fresh" if a < FRESH_S else "aging" if a < AGING_S else "stale"

    def envelope(self, e: str) -> dict:
        b = self.beliefs[e]
        return {
            "entity": e,
            "label": LABEL[e],
            "attribute": "pose" if e.startswith("bucket_") else "location",
            "value": b.value.as_dict(),
            "text": "moved by operator, position unknown" if b.note else b.value.text(),
            "source": b.source,
            "observed_at": self.clock.iso(b.at),
            "age_s": round(self.age(e), 1),
            "confidence": round(self.confidence(e), 2),
            "state": self.state(e),
        }

    def why(self, e: str) -> dict:
        b = self.beliefs[e]
        rows = [{"when": "now", "text": b.note or b.value.text(), **b.value.as_dict(), "source": b.source,
                 "observed_at": self.clock.iso(b.at)}]
        rows += b.history
        return {"entity": e, "history": rows}

    def all(self) -> list[dict]:
        return [self.envelope(e) for e in ENTITIES if e in self.beliefs]

    # ---- writing ----
    def observe(self, e: str, value: Value, source: str) -> bool:
        """Record an observation; return True when it contradicts the old belief (revised)."""
        now = self.clock.now()
        b = self.beliefs.get(e)
        if b is None:
            self.beliefs[e] = Belief(e, value, source, now, C0[source])
            return False
        moved = math.hypot(value.x - b.value.x, value.y - b.value.y) > MOVED_M or value.inside != b.value.inside
        revised = moved and source == "sim_overview"  # the robot's own moves are expected, not contradictions
        b.history.insert(0, {"when": "before", "text": b.note or b.value.text(), **b.value.as_dict(),
                             "source": b.source, "observed_at": self.clock.iso(b.at)})
        del b.history[6:]
        b.value, b.source, b.at, b.c0, b.note = value, source, now, C0[source], None
        b.revised_at = now if revised else None
        return revised

    def testimony(self, e: str, note: str = "operator says it moved") -> None:
        b = self.beliefs[e]
        b.history.insert(0, {"when": "before", "text": b.value.text(), **b.value.as_dict(),
                             "source": b.source, "observed_at": self.clock.iso(b.at)})
        b.source, b.at, b.c0, b.note, b.revised_at = "operator", self.clock.now(), C0["operator"], note, None

    def doubtful(self, e: str) -> bool:
        return self.state(e) in {"stale", "revised"}
