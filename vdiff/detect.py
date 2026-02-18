"""YOLO object detection with frame-to-frame tracking."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float
    x1: int  # bbox top-left x
    y1: int  # bbox top-left y
    x2: int  # bbox bottom-right x
    y2: int  # bbox bottom-right y

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def __str__(self) -> str:
        return f"{self.class_name}({self.confidence:.0%} @ {self.center})"


@dataclass
class TrackedChange:
    """Describes how an object changed between frames."""

    change_type: str  # "appeared", "disappeared", "moved", "still"
    detection: Detection
    prev_detection: Optional[Detection] = None
    distance_moved: float = 0.0

    def __str__(self) -> str:
        if self.change_type == "appeared":
            return f"{self.detection.class_name} appeared ({self.detection.confidence:.0%})"
        elif self.change_type == "disappeared":
            return f"{self.detection.class_name} disappeared"
        elif self.change_type == "moved":
            return (
                f"{self.detection.class_name} moved {self.distance_moved:.0f}px "
                f"({self.detection.confidence:.0%})"
            )
        else:
            return f"{self.detection.class_name} still present ({self.detection.confidence:.0%})"


@dataclass
class DetectionResult:
    """Result of running detection + tracking on a frame."""

    detections: list[Detection] = field(default_factory=list)
    changes: list[TrackedChange] = field(default_factory=list)
    has_changes: bool = False

    def summary(self) -> str:
        if not self.changes:
            return f"{len(self.detections)} object(s) detected, no changes."
        parts = [str(c) for c in self.changes if c.change_type != "still"]
        if not parts:
            return f"{len(self.detections)} object(s), all stationary."
        return "; ".join(parts)

    def changed_objects(self) -> list[TrackedChange]:
        """Return only objects that appeared, disappeared, or moved."""
        return [c for c in self.changes if c.change_type != "still"]

    def filter_by_mask(
        self, mask: Image.Image, orig_size: tuple[int, int], threshold: int = 50
    ):
        """Filter out changes that don't have corresponding pixel variance in the mask."""
        if not self.changes:
            return

        valid_changes = []
        mask_w, mask_h = mask.size
        orig_w, orig_h = orig_size

        for change in self.changes:
            # Always keep 'still' objects
            if change.change_type == "still":
                valid_changes.append(change)
                continue

            # Scale bbox to mask coords
            x1, y1, x2, y2 = change.detection.bbox
            sx1 = int(x1 * mask_w / orig_w)
            sy1 = int(y1 * mask_h / orig_h)
            sx2 = int(x2 * mask_w / orig_w)
            sy2 = int(y2 * mask_h / orig_h)

            # Clamp
            sx1 = max(0, min(sx1, mask_w - 1))
            sy1 = max(0, min(sy1, mask_h - 1))
            sx2 = max(sx1 + 1, min(sx2, mask_w))
            sy2 = max(sy1 + 1, min(sy2, mask_h))

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            # Check if mask has any bright pixels in this region
            crop = mask.crop((sx1, sy1, sx2, sy2))
            extrema = crop.getextrema()  # (min, max)

            if extrema and extrema[1] >= threshold:
                valid_changes.append(change)
            # else: ghost detection, dropped.

        self.changes = valid_changes
        self.has_changes = any(c.change_type != "still" for c in self.changes)


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection over Union between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class ObjectDetector:
    """YOLO-based object detection with frame-to-frame tracking."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.model_name = config.get("model", "yolov8n.pt")
        self.confidence = config.get("confidence", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.3)
        self.move_threshold = config.get("move_threshold", 15)
        self.classes = config.get("classes", None)  # None = all classes
        self.img_size = config.get("img_size", 640)
        self._model = None
        self._prev_detections: list[Detection] = []

    def _get_model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self._model = YOLO(self.model_name)
            # Suppress ultralytics logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)

            # Resolve class names to IDs if provided as strings
            if self.classes:
                resolved = []
                names_map = {v.lower(): k for k, v in self._model.names.items()}
                for c in self.classes:
                    if isinstance(c, str):
                        c_lower = c.lower()
                        if c_lower in names_map:
                            resolved.append(names_map[c_lower])
                        else:
                            logger.warning(f"Unknown class name: {c}")
                    else:
                        resolved.append(c)
                self.classes = resolved

        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Run detection on an image and track changes from previous frame.
        Returns DetectionResult with detections and tracked changes.
        """
        if not self.enabled:
            return DetectionResult()

        model = self._get_model()

        # Run inference
        results = model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            classes=self.classes,
            verbose=False,
        )

        # Parse detections
        detections = []
        if results and len(results) > 0:
            result = results[0]
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                    )
                )

        # Track changes
        changes = self._track_changes(detections)
        has_changes = any(
            c.change_type in ("appeared", "disappeared", "moved") for c in changes
        )

        # Store for next frame
        self._prev_detections = detections

        det_result = DetectionResult(
            detections=detections,
            changes=changes,
            has_changes=has_changes,
        )

        if detections:
            logger.info(
                f"YOLO: {len(detections)} object(s): "
                f"{', '.join(str(d) for d in detections)}"
            )
        if has_changes:
            logger.info(f"YOLO: {det_result.summary()}")

        return det_result

    def _track_changes(self, curr: list[Detection]) -> list[TrackedChange]:
        """Compare current detections to previous frame's detections."""
        prev = self._prev_detections
        if not prev:
            # First frame — everything is "appeared"
            return [TrackedChange(change_type="appeared", detection=d) for d in curr]

        changes = []
        matched_prev = set()
        matched_curr = set()

        # Match current detections to previous using IoU + same class
        for ci, c_det in enumerate(curr):
            best_iou = 0.0
            best_pi = -1
            for pi, p_det in enumerate(prev):
                if pi in matched_prev:
                    continue
                if c_det.class_id != p_det.class_id:
                    continue
                iou = _iou(c_det.bbox, p_det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi

            if best_pi >= 0 and best_iou > 0.1:
                # Matched — check if it moved
                matched_prev.add(best_pi)
                matched_curr.add(ci)
                p_det = prev[best_pi]
                dist = (
                    (c_det.center[0] - p_det.center[0]) ** 2
                    + (c_det.center[1] - p_det.center[1]) ** 2
                ) ** 0.5

                if dist > self.move_threshold:
                    changes.append(
                        TrackedChange(
                            change_type="moved",
                            detection=c_det,
                            prev_detection=p_det,
                            distance_moved=dist,
                        )
                    )
                else:
                    changes.append(
                        TrackedChange(
                            change_type="still",
                            detection=c_det,
                            prev_detection=p_det,
                        )
                    )

        # Unmatched current = appeared
        for ci, c_det in enumerate(curr):
            if ci not in matched_curr:
                changes.append(
                    TrackedChange(
                        change_type="appeared",
                        detection=c_det,
                    )
                )

        # Unmatched previous = disappeared
        for pi, p_det in enumerate(prev):
            if pi not in matched_prev:
                changes.append(
                    TrackedChange(
                        change_type="disappeared",
                        detection=p_det,
                    )
                )

        return changes

    def crop_detection(
        self, image: Image.Image, detection: Detection, padding: int = 20
    ) -> Image.Image:
        """Crop a detected object from the image with padding for LLM analysis."""
        w, h = image.size
        x1 = max(0, detection.x1 - padding)
        y1 = max(0, detection.y1 - padding)
        x2 = min(w, detection.x2 + padding)
        y2 = min(h, detection.y2 + padding)
        return image.crop((x1, y1, x2, y2))
