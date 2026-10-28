"""Text -> intent. Keyword rules plus fuzzy matching against the scene's vocabulary.

Returns the same JSON shape the LLM path will return later, so the LLM can be bolted on in front
of this parser without changing the director.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass

OBJECTS = ("resistor", "battery", "cpu")
BUCKETS = ("red", "blue", "black")

# Scene phrase bank plus the words Whisper tends to produce for them.
OBJECT_WORDS = {
    "battery": ("battery", "batteries", "9v", "nine-volt", "cell", "batt", "matter", "battle"),
    "cpu": ("cpu", "chip", "processor", "microchip", "chipset", "tip", "ship", "tube"),
    "resistor": ("resistor", "resistors", "resister", "register"),
}
BUCKET_WORDS = {"red": ("red", "read"), "blue": ("blue", "blew"), "black": ("black", "block")}
CONTAINER_WORDS = ("bin", "bucket", "box", "tub", "container", "one")

STOP_RE = re.compile(r"\b(stop|halt|freeze|abort)\b")
RESUME_RE = re.compile(r"\b(resume|continue|carry on|go ahead)\b")
WHERE_RE = re.compile(r"\b(where|find|locate)\b")
LOOK_RE = re.compile(r"^\s*(look|check|scan|look around)\b")
PLACE_RE = re.compile(r"\b(put|place|toss|drop|move|throw|sort|stick|pop)\b")
TESTIMONY_RE = re.compile(r"\bi (moved|shifted|bumped|swapped|nudged)\b")


@dataclass
class Intent:
    intent: str  # place | where_is | look | stop | resume | observe | clarify | none
    object: str | None = None
    bucket: str | None = None
    entity: str | None = None
    question: str | None = None
    confidence: float = 1.0
    utterance: str = ""
    parser: str = "keyword"

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9-]+", text.lower())


def _match(tokens: list[str], table: dict[str, tuple[str, ...]]) -> tuple[str | None, float]:
    """Exact synonym hit first, then the closest fuzzy match above 0.8."""
    for key, words in table.items():
        if any(t in words for t in tokens):
            return key, 1.0
    best, score = None, 0.0
    for key, words in table.items():
        for t in tokens:
            if len(t) < 4:
                continue
            r = max(difflib.SequenceMatcher(None, t, w).ratio() for w in words)
            if r > score:
                best, score = key, r
    return (best, score) if score >= 0.8 else (None, 0.0)


def parse(text: str, pending_object: str | None = None) -> Intent:
    t = text.lower().strip()
    toks = _tokens(t)
    if STOP_RE.search(t):
        return Intent("stop", utterance=text)
    if RESUME_RE.search(t):
        return Intent("resume", utterance=text)
    obj, so = _match(toks, OBJECT_WORDS)
    bucket, sb = _match(toks, BUCKET_WORDS)
    conf = round(min(so or 1.0, sb or 1.0), 2)
    if TESTIMONY_RE.search(t):
        if bucket:
            return Intent("observe", entity=f"bucket_{bucket}", confidence=conf, utterance=text)
        if obj:
            return Intent("observe", entity=obj, confidence=conf, utterance=text)
    if WHERE_RE.search(t):
        if obj:
            return Intent("where_is", object=obj, confidence=conf, utterance=text)
        if bucket:
            return Intent("where_is", object=f"bucket_{bucket}", confidence=conf, utterance=text)
        return Intent("clarify", question="Which object: the battery, the CPU chip or the resistor?", utterance=text)
    if not obj and bucket and pending_object:
        return Intent("place", object=pending_object, bucket=bucket, confidence=conf, utterance=text)
    if LOOK_RE.search(t) and not obj:
        return Intent("look", utterance=text)
    if PLACE_RE.search(t) or (obj and bucket):
        if not obj:
            return Intent("clarify", question="Which object: the battery, the CPU chip or the resistor?", utterance=text)
        if not bucket:
            name = "CPU chip" if obj == "cpu" else obj
            return Intent("clarify", object=obj, question=f"Which bin for the {name}: red, blue or black?", utterance=text)
        return Intent("place", object=obj, bucket=bucket, confidence=conf, utterance=text)
    return Intent("none", utterance=text, confidence=0.0)
