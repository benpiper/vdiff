"""Zone (region of interest) support for monitoring specific image areas."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A rectangular region of interest, defined as percentages (0-100)."""

    name: str
    x_pct: float  # left edge, 0-100
    y_pct: float  # top edge, 0-100
    w_pct: float  # width, 0-100
    h_pct: float  # height, 0-100

    def to_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert percentage coords to pixel coords (x1, y1, x2, y2)."""
        x1 = int(img_w * self.x_pct / 100)
        y1 = int(img_h * self.y_pct / 100)
        x2 = int(img_w * (self.x_pct + self.w_pct) / 100)
        y2 = int(img_h * (self.y_pct + self.h_pct) / 100)
        return (
            max(0, x1),
            max(0, y1),
            min(img_w, x2),
            min(img_h, y2),
        )

    def contains_point(self, px: float, py: float, img_w: int, img_h: int) -> bool:
        """Check if a pixel coordinate falls within this zone."""
        x1, y1, x2, y2 = self.to_pixels(img_w, img_h)
        return x1 <= px <= x2 and y1 <= py <= y2

    def contains_bbox(
        self,
        bx1: int,
        by1: int,
        bx2: int,
        by2: int,
        img_w: int,
        img_h: int,
        min_overlap: float = 0.3,
    ) -> bool:
        """Check if a bounding box overlaps this zone by at least min_overlap."""
        zx1, zy1, zx2, zy2 = self.to_pixels(img_w, img_h)

        # Intersection
        ix1 = max(zx1, bx1)
        iy1 = max(zy1, by1)
        ix2 = min(zx2, bx2)
        iy2 = min(zy2, by2)

        if ix1 >= ix2 or iy1 >= iy2:
            return False

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = (bx2 - bx1) * (by2 - by1)

        if bbox_area == 0:
            return False

        return (inter_area / bbox_area) >= min_overlap


def parse_zones(camera_config: dict) -> list[Zone]:
    """Parse zones from camera config. Returns empty list if no zones defined."""
    zones_cfg = camera_config.get("zones", [])
    zones = []
    for z in zones_cfg:
        zones.append(
            Zone(
                name=z.get("name", "unnamed"),
                x_pct=z.get("x", 0),
                y_pct=z.get("y", 0),
                w_pct=z.get("w", 100),
                h_pct=z.get("h", 100),
            )
        )
    if zones:
        names = ", ".join(z.name for z in zones)
        logger.info(f"Loaded {len(zones)} zone(s): {names}")
    return zones


def build_zone_mask(zones: list[Zone], img_w: int, img_h: int) -> np.ndarray:
    """
    Build a binary mask where zone regions are 255 (active) and everything else is 0.
    Returns a single-channel uint8 array of shape (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for zone in zones:
        x1, y1, x2, y2 = zone.to_pixels(img_w, img_h)
        mask[y1:y2, x1:x2] = 255
    return mask


def apply_zone_mask(gray_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply zone mask to a grayscale array — zero out pixels outside zones."""
    return cv2.bitwise_and(gray_array.astype(np.uint8), mask).astype(np.float32)


def filter_detections_by_zones(
    detections: list, zones: list[Zone], img_w: int, img_h: int
) -> list:
    """Filter YOLO detections to only those overlapping configured zones."""
    if not zones:
        return detections
    filtered = []
    for det in detections:
        for zone in zones:
            if zone.contains_bbox(det.x1, det.y1, det.x2, det.y2, img_w, img_h):
                filtered.append(det)
                break
    return filtered


def draw_zones(image, zones: list[Zone]):
    """Draw zones on a copy of the image for debugging. Returns PIL Image."""
    from PIL import ImageDraw, Image

    debug_img = image.copy().convert("RGBA")
    w, h = image.size

    # 1. Dim the excluded areas
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 100))  # Black 40% opacity
    draw_overlay = ImageDraw.Draw(overlay)

    # Clear (make transparent) the zone areas so they show through brightly
    for zone in zones:
        x1, y1, x2, y2 = zone.to_pixels(w, h)
        draw_overlay.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 0))  # Transparent

    # Composite overlay onto debug image
    debug_img = Image.alpha_composite(debug_img, overlay)

    # 2. Draw red outlines and labels
    draw = ImageDraw.Draw(debug_img)
    for zone in zones:
        x1, y1, x2, y2 = zone.to_pixels(w, h)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 5, y1 + 5), zone.name, fill="red")

    return debug_img.convert("RGB")
