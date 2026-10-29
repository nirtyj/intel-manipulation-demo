"""A physicalai camera on the simulator's latest frame, prepared the way the demonstrations were recorded.

The recorder (v2/recorder.py) took ``GET /cameras/{name}/frame.jpg`` at 10 Hz, downscaled it to 320x240
(INTER_AREA) and stored it as JPEG (quality 90). The policy must see its cameras the same way, so this
camera does exactly that, without blocking: the simulator renders its cameras in software at a few
frames per second, and a stream reader would wait for each new frame and stall the control loop.
"""

from __future__ import annotations

import time
import urllib.request

import cv2
import numpy as np
from physicalai.capture.camera import Camera, ColorMode
from physicalai.capture.errors import CaptureError, NotConnectedError
from physicalai.capture.frame import Frame

SIM = "http://127.0.0.1:8080"


class SimSnapshotCamera(Camera):
    def __init__(self, *, name: str, base_url: str = SIM, size: tuple[int, int] = (320, 240), quality: int = 90) -> None:
        super().__init__(color_mode=ColorMode.RGB)
        self._url = f"{base_url}/cameras/{name}/frame.jpg"
        self._name, self._size, self._quality = name, tuple(size), quality
        self._connected = False
        self._sequence = 0

    def connect(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._fetch(timeout=max(0.5, deadline - time.monotonic()))
                break
            except CaptureError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        self._connected = True

    def _do_disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_id(self) -> str:
        return f"sim:{self._name}"

    def _fetch(self, timeout: float = 2.0) -> Frame:
        try:
            raw = urllib.request.urlopen(self._url, timeout=timeout).read()
        except OSError as exc:
            raise CaptureError(f"{self._url}: {exc}") from exc
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        small = cv2.resize(bgr, self._size, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR) if ok else small
        self._sequence += 1
        return Frame(data=np.ascontiguousarray(bgr[:, :, ::-1]), timestamp=time.monotonic(), sequence=self._sequence)

    def read(self, timeout: float = 2.0) -> Frame:
        if not self._connected:
            raise NotConnectedError(f"{self.device_id} is not connected")
        return self._fetch(timeout)

    def read_latest(self) -> Frame:
        return self.read()
