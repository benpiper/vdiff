#!/usr/bin/env python3
"""
Calibration utility for vdiff.
Compares a 'background' image and an 'object' image to recommend config settings.
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path to import vdiff
sys.path.append(str(Path(__file__).parent.parent))

from vdiff.diff import DiffEngine
from vdiff.detect import ObjectDetector
from vdiff.zones import parse_zones, build_zone_mask

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calibrate")


def load_config(path: Path) -> dict:
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Black out areas of a PIL Image that are outside the provided numpy mask."""
    if mask is None:
        return image
    np_img = np.array(image.convert("RGB"))
    mask_3d = np.stack([mask] * 3, axis=-1)
    np_img = np.where(mask_3d == 0, 0, np_img)
    return Image.fromarray(np_img.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate vdiff config using two images."
    )
    parser.add_argument(
        "background", type=Path, help="Path to background image (empty)"
    )
    parser.add_argument(
        "object", type=Path, help="Path to image with object of interest"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # 1. Load Config & Zones
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    zones = []
    if config.get("cameras"):
        zones = parse_zones(config["cameras"][0])
        print(
            f"Loaded {len(zones)} zone(s) from first camera config: {', '.join(z.name for z in zones)}"
        )
    else:
        print("No cameras/zones found in config. Analysis will be full-frame.")

    # 2. Load Images
    try:
        img_bg = Image.open(args.background).convert("RGB")
        img_obj = Image.open(args.object).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open images: {e}")
        sys.exit(1)

    if img_bg.size != img_obj.size:
        logger.warning("Image sizes differ! Resizing object image to match background.")
        img_obj = img_obj.resize(img_bg.size)

    # 3. Apply Masking
    mask = None
    if zones:
        w, h = img_bg.size
        mask = build_zone_mask(zones, w, h)
        img_bg = apply_mask_to_image(img_bg, mask)
        img_obj = apply_mask_to_image(img_obj, mask)
        print("Applied strict zone masking to both images.")

    # 4. Initialize Engines
    diff_engine = DiffEngine(config.get("diff", {}))
    detector = ObjectDetector(config.get("detection", {}))

    # 5. Run Motion Analysis
    print("\n" + "=" * 40)
    print(" MOTION ANALYSIS (Background vs Object)")
    print("=" * 40)

    # We use the diff engine to compare
    # Note: diff_engine implementation might resize internally, but we want raw stats
    diff_result = diff_engine.compare(img_bg, img_obj)

    print(
        f"Raw Pixel Difference: {diff_result.pixel_diff_score:.2f} (avg diff per pixel)"
    )
    print(f"Changed Percentage:   {diff_result.changed_pct:.2f}% of pixels")
    print(f"SSIM Score:           {diff_result.ssim_score:.4f} (1.0 = identical)")

    # Recommendations
    rec_min_pct = max(0.5, diff_result.changed_pct * 0.8)  # 80% of observed
    # If SSIM is high (close to 1), it means images are similar.
    # To detect this change, threshold must be > SSIM score.
    rec_ssim_thresh = min(0.99, diff_result.ssim_score + 0.05)

    print("\n--- Recommended Settings (diff) ---")
    print(
        f"diff:\n  min_changed_pct: {rec_min_pct:.1f}\n  ssim_threshold: {rec_ssim_thresh:.3f}"
    )

    # 6. Run Object Detection
    print("\n" + "=" * 40)
    print(" OBJECT DETECTION (Object Image)")
    print("=" * 40)

    det_result = detector.detect(img_obj)

    if det_result.detections:
        print(f"Found {len(det_result.detections)} object(s):")
        max_conf = 0.0
        for det in det_result.detections:
            print(f" - {det.class_name}: {det.confidence:.1%} confidence")
            max_conf = max(max_conf, det.confidence)

        # Recommendation
        rec_conf = max(0.1, max_conf * 0.8)  # 80% of observed confidence
        print("\n--- Recommended Settings (detection) ---")
        print(f"detection:\n  confidence: {rec_conf:.2f}")
    else:
        print("No objects detected in MASKED image.")
        print("Checking RAW image to see if zones are the problem...")

        # Run on raw object image
        try:
            raw_obj = Image.open(args.object).convert("RGB")
            if raw_obj.size != img_bg.size:
                raw_obj = raw_obj.resize(img_bg.size)

            raw_det = detector.detect(raw_obj)
            if raw_det.detections:
                print(
                    f"\n[WARNING] Found {len(raw_det.detections)} object(s) in RAW image!"
                )
                for det in raw_det.detections:
                    print(
                        f" - {det.class_name}: {det.confidence:.1%} confidence at {det.center}"
                    )
                print(
                    "\nCRITICAL: Your object is visible but OUTSIDE your configured zones."
                )
                print(
                    "ACTION: Adjust pixel coordinates or percentages in config.yaml to include these objects."
                )
            else:
                print("No objects detected in RAW image either.")
                print("Try lowering confidence or using a different model.")
        except Exception as e:
            logger.error(f"Failed to check raw image: {e}")


if __name__ == "__main__":
    main()
