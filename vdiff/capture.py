"""Camera capture module — grabs snapshots from various camera sources."""

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera source."""

    name: str
    type: str  # "http", "rtsp", "local"
    url: str = ""
    username: str = ""
    password: str = ""
    auth: str = "none"  # "digest", "basic", "none"
    interval: float = 10.0
    device_id: int = 0  # for local cameras


class CameraSource:
    """Base camera source with common interface."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.name = config.name
        self._last_capture_time: float = 0

    def capture(self) -> Optional[Image.Image]:
        """Capture a single frame. Returns PIL Image or None on failure."""
        raise NotImplementedError

    @property
    def seconds_until_next(self) -> float:
        """Seconds remaining until next capture is due."""
        elapsed = time.monotonic() - self._last_capture_time
        return max(0, self.config.interval - elapsed)

    @property
    def is_due(self) -> bool:
        """Whether it's time to capture the next frame."""
        return self.seconds_until_next <= 0

    def _mark_captured(self):
        self._last_capture_time = time.monotonic()


class HTTPCamera(CameraSource):
    """Captures snapshots via HTTP/HTTPS GET (e.g., Hikvision ISAPI)."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._session = requests.Session()

        # Set up authentication
        if config.auth == "digest":
            self._session.auth = HTTPDigestAuth(config.username, config.password)
        elif config.auth == "basic":
            self._session.auth = HTTPBasicAuth(config.username, config.password)

        # Disable SSL warnings for self-signed certs (common on NVRs)
        self._session.verify = False

    def capture(self) -> Optional[Image.Image]:
        try:
            resp = self._session.get(self.config.url, timeout=10, stream=True)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            self._mark_captured()
            logger.debug(
                f"[{self.name}] captured {image.size[0]}x{image.size[1]} image"
            )
            return image.convert("RGB")
        except requests.RequestException as e:
            logger.error(f"[{self.name}] HTTP capture failed: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self.name}] image decode failed: {e}")
            return None


class RTSPCamera(CameraSource):
    """Captures a single frame from an RTSP stream via OpenCV."""

    def capture(self) -> Optional[Image.Image]:
        cap = None
        try:
            cap = cv2.VideoCapture(self.config.url)
            if not cap.isOpened():
                logger.error(
                    f"[{self.name}] failed to open RTSP stream: {self.config.url}"
                )
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"[{self.name}] failed to grab frame from RTSP stream")
                return None
            # OpenCV returns BGR, convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self._mark_captured()
            logger.debug(
                f"[{self.name}] captured {image.size[0]}x{image.size[1]} from RTSP"
            )
            return image
        except Exception as e:
            logger.error(f"[{self.name}] RTSP capture error: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()


class LocalCamera(CameraSource):
    """Captures from a local USB/CSI camera via OpenCV."""

    def capture(self) -> Optional[Image.Image]:
        cap = None
        try:
            cap = cv2.VideoCapture(self.config.device_id)
            if not cap.isOpened():
                logger.error(
                    f"[{self.name}] failed to open local camera {self.config.device_id}"
                )
                return None
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error(f"[{self.name}] failed to grab frame from local camera")
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            self._mark_captured()
            logger.debug(
                f"[{self.name}] captured {image.size[0]}x{image.size[1]} from local cam"
            )
            return image
        except Exception as e:
            logger.error(f"[{self.name}] local capture error: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()


def create_camera(cam_cfg: dict) -> CameraSource:
    """Factory: create the appropriate CameraSource from a config dict."""
    config = CameraConfig(
        name=cam_cfg.get("name", "unnamed"),
        type=cam_cfg.get("type", "http"),
        url=cam_cfg.get("url", ""),
        username=cam_cfg.get("username", ""),
        password=cam_cfg.get("password", ""),
        auth=cam_cfg.get("auth", "none"),
        interval=cam_cfg.get("interval", 10),
        device_id=cam_cfg.get("device_id", 0),
    )

    camera_types = {
        "http": HTTPCamera,
        "https": HTTPCamera,
        "rtsp": RTSPCamera,
        "local": LocalCamera,
    }
    cls = camera_types.get(config.type, HTTPCamera)
    logger.info(f"Initialized camera '{config.name}' (type={config.type})")
    return cls(config)
