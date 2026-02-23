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


def load_images_from_dir(directory: Path, mask: np.ndarray = None) -> list[Image.Image]:
    """Load all valid images from a directory, optionally applying a mask."""
    images = []
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return images

    for p in sorted(directory.glob("*.jpg")):
        try:
            img = Image.open(p).convert("RGB")
            if mask is not None:
                img = apply_mask_to_image(img, mask)
            images.append(img)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")

    return images


def format_stats(values: list) -> str:
    """Format min/max/avg for display."""
    if not values:
        return "N/A"
    return f"Min: {min(values):.3f} | Max: {max(values):.3f} | Avg: {sum(values) / len(values):.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Robustly calibrate vdiff using directories of absent/present images."
    )
    parser.add_argument(
        "--absent-dir",
        type=Path,
        default=Path(__file__).parent / "absent",
        help="Path to directory containing images with NO object (background/noise floor)",
    )
    parser.add_argument(
        "--present-dir",
        type=Path,
        default=Path(__file__).parent / "present",
        help="Path to directory containing images with the object present",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override YOLO model (e.g., yolov8s.pt)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="Override YOLO classes to detect (e.g., --classes 2 5 7)",
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    zones = []
    if config.get("cameras"):
        zones = parse_zones(config["cameras"][0])
        print(f"Loaded {len(zones)} zone(s) from first camera config.")
    else:
        print("No cameras/zones found in config. Analysis will be full-frame.")

    # We need one image to establish dimensions for the mask
    test_img = next(args.absent_dir.glob("*.jpg"), None)
    if not test_img:
        logger.error(f"No JPG images found in absent dir: {args.absent_dir}")
        sys.exit(1)

    base_img = Image.open(test_img)
    w, h = base_img.size

    mask = None
    if zones:
        mask = build_zone_mask(zones, w, h)
        print("Strict zone masking will be applied to all images.")

    print(f"Loading absent images from {args.absent_dir}...")
    absent_images = load_images_from_dir(args.absent_dir, mask)
    print(f"Loading present images from {args.present_dir}...")
    present_images = load_images_from_dir(args.present_dir, mask)

    if len(absent_images) < 2:
        logger.error(
            "Need at least 2 absent images to establish a noise floor baseline."
        )
        sys.exit(1)
    if not present_images:
        logger.error("Need at least 1 present image to establish signal.")
        sys.exit(1)

    print(
        f"\nLoaded {len(absent_images)} absent images and {len(present_images)} present images."
    )

    # Initialize Engines
    diff_engine = DiffEngine(config.get("diff", {}))

    det_config = config.get("detection", {})
    if args.model:
        det_config["model"] = args.model
        print(f"Overriding YOLO model with: {args.model}")
    if args.classes is not None:
        det_config["classes"] = args.classes
        print(f"Overriding YOLO classes with: {args.classes}")

    detector = ObjectDetector(det_config)

    # --- Section: Noise Floor (Absent vs Absent) ---
    print("\n" + "=" * 50)
    print(" 1. NOISE FLOOR ANALYSIS (Absent vs Absent)")
    print("=" * 50)

    noise_pct = []
    noise_ssim = []

    # Compare consecutive frames to simulate real camera operation
    for i in range(len(absent_images) - 1):
        diff_result = diff_engine.compare(absent_images[i], absent_images[i + 1])
        noise_pct.append(diff_result.changed_pct)
        noise_ssim.append(diff_result.ssim_score)

    print(f"Changed % (Noise):  {format_stats(noise_pct)}")
    print(f"SSIM Score (Noise): {format_stats(noise_ssim)}")
    max_noise_pct = max(noise_pct)
    min_noise_ssim = min(noise_ssim)

    # --- Section: Signal Analysis (Absent vs Present) ---
    print("\n" + "=" * 50)
    print(" 2. SIGNAL ANALYSIS (Absent vs Present)")
    print("=" * 50)

    signal_pct = []
    signal_ssim = []

    # Compare each present image against an absent background
    for p_img in present_images:
        diff_result = diff_engine.compare(absent_images[0], p_img)
        signal_pct.append(diff_result.changed_pct)
        signal_ssim.append(diff_result.ssim_score)

    print(f"Changed % (Signal): {format_stats(signal_pct)}")
    print(f"SSIM Score (Signal): {format_stats(signal_ssim)}")
    min_signal_pct = min(signal_pct)
    max_signal_ssim = max(signal_ssim)

    if max_noise_pct >= min_signal_pct:
        print(
            "\n[WARNING] Noise floor overlaps signal! Your camera shift/lighting is changing pixels more than your object."
        )
        print(
            f"  Max Noise %: {max_noise_pct:.2f}%  >=  Min Signal %: {min_signal_pct:.2f}%"
        )

    if min_noise_ssim <= max_signal_ssim:
        print(
            "\n[WARNING] SSIM ranges overlap! Consider relying on YOLO or improving your camera stability."
        )

    # Recommendations for Diff
    # Place it squarely in the middle of max noise and min signal
    diff_range_pct = min_signal_pct - max_noise_pct
    rec_min_pct = (
        max_noise_pct + (diff_range_pct * 0.4)
        if diff_range_pct > 0
        else max(2.0, max_noise_pct * 1.5)
    )

    # SSIM (closer to 1 = more similar). We want threshold lower than noise (so noise passes to stage 2)
    # but higher than signal (so signal breaks threshold and triggers alert).
    diff_range_ssim = min_noise_ssim - max_signal_ssim
    rec_ssim_thresh = (
        max_signal_ssim + (diff_range_ssim * 0.6)
        if diff_range_ssim > 0
        else max(0.9, min_noise_ssim - 0.05)
    )

    # --- Section: YOLO Confidence Detection ---
    print("\n" + "=" * 50)
    print(" 3. YOLO OBJECT DETECTION ANALYSIS")
    print("=" * 50)

    # 1. Stationary Objects (Absent Images)
    # These are not strictly "false positives" because the pixel diff engine will ignore them
    # if they aren't moving, but it's good to know what YOLO sees in the background.
    stat_confs = []
    stat_classes = set()
    for img in absent_images:
        res = detector.detect(img)
        for det in res.detections:
            stat_confs.append(det.confidence)
            stat_classes.add(det.class_name)

    max_stat_conf = max(stat_confs) if stat_confs else 0.0
    print(f"Stationary Objects in Absent: {len(stat_confs)}")
    if stat_confs:
        print(
            f"Highest Stationary Confidence: {max_stat_conf:.1%} (Classes seen: {', '.join(stat_classes)})"
        )
        print(
            "  (Note: Stationary objects are safely ignored by the pixel diff engine, so these do not limit your threshold.)"
        )

    # 2. True Positives (Present Images)
    tp_confs = []
    tp_classes = set()
    for img in present_images:
        res = detector.detect(img)
        if res.detections:
            tp_confs.extend([d.confidence for d in res.detections])
            tp_classes.update([d.class_name for d in res.detections])

    min_tp_conf = min(tp_confs) if tp_confs else 0.0
    print(f"\nMoving Objects in Present: {len(tp_confs)}")
    if tp_confs:
        print(
            f"Lowest True Positive Confidence: {min_tp_conf:.1%} (Classes seen: {', '.join(tp_classes)})"
        )
    else:
        print(
            "[WARNING] YOLO completely missed the object in all 'present' images. Check your zones or lower conf."
        )

    # Recommendation for YOLO Confidence
    # Base this solely on the signal, since stationary noise is filtered by the diff engine.
    if tp_confs:
        rec_conf = max(0.01, min_tp_conf * 0.8)
    else:
        rec_conf = 0.05

    # --- Section: Final Config Recommendation ---
    print("\n" + "=" * 50)
    print(" RECOMMENDED CONFIGURATION (`config.yaml`)")
    print("=" * 50)

    print("diff:")
    print(
        f"  min_changed_pct: {rec_min_pct:.2f}     # Was: {config.get('diff', {}).get('min_changed_pct', 0)}"
    )
    print(
        f"  ssim_threshold: {rec_ssim_thresh:.3f}     # Was: {config.get('diff', {}).get('ssim_threshold', 0)}"
    )
    print("\ndetection:")
    print(
        f"  confidence: {rec_conf:.3f}         # Was: {config.get('detection', {}).get('confidence', 0)}"
    )
    print(
        "\nNote: These values are automatically scaled between your observed NOISE and SIGNAL bounds."
    )


if __name__ == "__main__":
    main()
