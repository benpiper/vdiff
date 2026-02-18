"""Two-stage image diff engine — fast pixel diff, then SSIM for detail."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box for a changed region."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def __str__(self) -> str:
        return f"({self.x},{self.y} {self.w}x{self.h})"


@dataclass
class DiffResult:
    """Result of comparing two images."""

    changed: bool = False
    pixel_diff_score: float = 0.0  # mean absolute diff (0-255)
    changed_pct: float = 0.0  # percentage of pixels that changed
    ssim_score: float = 1.0  # structural similarity (0-1, 1=identical)
    diff_mask: Optional[Image.Image] = None  # visual diff mask
    bounding_boxes: list = field(default_factory=list)
    region_description: str = ""  # fallback text description of where changes are

    def summary(self) -> str:
        if not self.changed:
            return "No significant change detected."
        parts = [
            f"Change detected: {self.changed_pct:.1f}% of pixels changed",
            f"(pixel_diff={self.pixel_diff_score:.1f}, ssim={self.ssim_score:.3f})",
        ]
        if self.region_description:
            parts.append(f"Regions: {self.region_description}")
        return " | ".join(parts)


class DiffEngine:
    """Fast two-stage image comparison engine."""

    def __init__(self, config: dict):
        self.pixel_threshold = config.get("pixel_threshold", 12)
        self.min_changed_pct = config.get("min_changed_pct", 2.0)
        self.ssim_threshold = config.get("ssim_threshold", 0.92)
        self.resize_width = config.get("resize_width", 640)

    def _resize(self, img: Image.Image) -> Image.Image:
        """Resize image for faster processing while keeping aspect ratio."""
        if self.resize_width and img.width > self.resize_width:
            ratio = self.resize_width / img.width
            new_h = int(img.height * ratio)
            return img.resize((self.resize_width, new_h), Image.LANCZOS)
        return img

    def _to_gray_array(self, img: Image.Image) -> np.ndarray:
        """Convert PIL Image to grayscale numpy array."""
        return np.array(img.convert("L"), dtype=np.float32)

    def compare(
        self,
        prev: Image.Image,
        curr: Image.Image,
        zone_mask: "np.ndarray | None" = None,
    ) -> DiffResult:
        """Compare two images. Optionally restrict to zone_mask regions."""
        result = DiffResult()

        # Resize for speed
        prev_r = self._resize(prev)
        curr_r = self._resize(curr)

        # Ensure same dimensions
        if prev_r.size != curr_r.size:
            curr_r = curr_r.resize(prev_r.size, Image.LANCZOS)

        prev_gray = self._to_gray_array(prev_r)
        curr_gray = self._to_gray_array(curr_r)

        # Resize zone mask to match if provided
        if zone_mask is not None:
            mask_h, mask_w = zone_mask.shape[:2]
            img_h, img_w = prev_gray.shape[:2]
            if (mask_w, mask_h) != (img_w, img_h):
                zone_mask = cv2.resize(
                    zone_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )

        # --- Stage 1: Fast pixel diff ---
        abs_diff = np.abs(prev_gray - curr_gray)

        if zone_mask is not None:
            # Only count pixels within zones
            in_zone = zone_mask > 0
            zone_diff = abs_diff[in_zone]
            total_pixels = zone_diff.size
            if total_pixels == 0:
                return result
            result.pixel_diff_score = float(np.mean(zone_diff))
            changed_pixels = np.sum(zone_diff > self.pixel_threshold)
        else:
            result.pixel_diff_score = float(np.mean(abs_diff))
            changed_pixels = np.sum(abs_diff > self.pixel_threshold)
            total_pixels = abs_diff.size

        result.changed_pct = float(changed_pixels / total_pixels * 100)

        if result.changed_pct < self.min_changed_pct:
            logger.debug(
                f"Stage 1 pass: only {result.changed_pct:.2f}% pixels changed "
                f"(threshold: {self.min_changed_pct}%)"
            )
            return result

        # --- Stage 2: SSIM for structural comparison ---
        score, diff_map = ssim(
            prev_gray,
            curr_gray,
            full=True,
            data_range=255.0,
        )
        result.ssim_score = float(score)

        if result.ssim_score > self.ssim_threshold:
            logger.debug(
                f"Stage 2 pass: SSIM={result.ssim_score:.4f} "
                f"(threshold: {self.ssim_threshold})"
            )
            return result

        # Change confirmed — generate diff mask and bounding boxes
        result.changed = True

        # Create visual diff mask
        diff_img = ((1.0 - diff_map) * 255).astype(np.uint8)
        result.diff_mask = Image.fromarray(diff_img)

        # Find contours of changed regions
        thresh = cv2.threshold(diff_img, 50, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter to significant contours and create bounding boxes
        min_area = total_pixels * 0.001  # at least 0.1% of frame
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area:
                result.bounding_boxes.append(BBox(x, y, w, h))

        # Generate region description for non-LLM fallback
        result.region_description = self._describe_regions(
            result.bounding_boxes, prev_r.width, prev_r.height
        )

        logger.info(
            f"Change detected: {result.changed_pct:.1f}% pixels, "
            f"SSIM={result.ssim_score:.3f}, {len(result.bounding_boxes)} regions"
        )
        return result

    @staticmethod
    def _describe_regions(boxes: list, img_w: int, img_h: int) -> str:
        """Generate a human-readable description of where changes occurred."""
        if not boxes:
            return "scattered small changes"

        descriptions = []
        for box in boxes[:5]:  # max 5 regions
            cx = (box.x + box.w / 2) / img_w
            cy = (box.y + box.h / 2) / img_h
            h_pos = "left" if cx < 0.33 else ("center" if cx < 0.67 else "right")
            v_pos = "upper" if cy < 0.33 else ("middle" if cy < 0.67 else "lower")
            descriptions.append(f"{v_pos}-{h_pos}")

        return ", ".join(descriptions)
