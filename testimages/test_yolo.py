import sys
import yaml
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import logging

# Add parent directory to path to import vdiff
sys.path.append(str(Path(__file__).parent.parent))

from vdiff.detect import ObjectDetector
from vdiff.zones import parse_zones, build_zone_mask

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("yolo_test")


def apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Black out areas of a PIL Image that are outside the provided numpy mask."""
    if mask is None:
        return image
    np_img = np.array(image)
    if len(np_img.shape) == 3:
        mask_3d = np.stack([mask] * 3, axis=-1)
        np_img = np.where(mask_3d == 0, 0, np_img)
    else:
        np_img = np.where(mask == 0, 0, np_img)
    return Image.fromarray(np_img.astype(np.uint8))


def annotate_image(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and labels on a copy of the image."""
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    for det in detections:
        # Draw thick red box
        draw.rectangle(det.bbox, outline="red", width=4)
        # Draw label background
        label = f"{det.class_name} {det.confidence:.1%}"
        draw.text((det.x1 + 5, det.y1 + 5), label, fill="red")
    return draw_img


def run_test():
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config.yaml"

    # Create output directory
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to {output_dir.relative_to(base_dir)}")

    # Load config for context
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            print(f"Loaded config from {config_path}")

    # Parse zones if any
    zones = []
    if config.get("cameras"):
        # Just use the first camera's zones for testing
        zones = parse_zones(config["cameras"][0])

    image_dir = Path(__file__).parent
    images = list(image_dir.glob("*.jpg"))
    if not images:
        print(f"No .jpg images found in {image_dir}")
        return

    # Use model from config if possible
    default_model = config.get("detection", {}).get("model", "yolov8s.pt")
    default_size = config.get("detection", {}).get("img_size", 640)

    # Test configurations
    configs = [
        {
            "name": "Current Config",
            "conf": config.get("detection", {}).get("confidence", 0.55),
            "size": default_size,
        },
        # {"name": "Standard Sensitivity", "conf": 0.25, "size": default_size},
        # {"name": "High Res (Strict)", "conf": 0.55, "size": 1280},
        {"name": "High Res (Sensitive)", "conf": 0.15, "size": 1280},
    ]

    for img_path in sorted(images):
        print("\n" + "=" * 60)
        print(f"TESTING IMAGE: {img_path.name}")
        print("=" * 60)

        try:
            img_raw = Image.open(img_path)
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        # Prepare masked version if zones exist
        mask = None
        if zones:
            w, h = img_raw.size
            mask = build_zone_mask(zones, w, h)
            print(f"Zonal masking ENABLED ({len(zones)} zone(s))")

        for cfg in configs:
            print(
                f"\n--- Strategy: {cfg['name']} (conf={cfg['conf']}, size={cfg['size']}) ---"
            )

            # Initialize detector with these specific settings
            detector = ObjectDetector(
                {
                    "enabled": True,
                    "confidence": cfg["conf"],
                    "img_size": cfg["size"],
                    "model": default_model,
                    "classes": ["car"],
                }
            )

            # Test both raw and masked if zones exist
            test_versions = [("Raw", img_raw)]
            if mask is not None:
                test_versions.append(("Masked", apply_mask(img_raw, mask)))

            for label, test_img in test_versions:
                result = detector.detect(test_img)
                prefix = f"  [{label}]"
                if result.detections:
                    for det in result.detections:
                        print(
                            f"{prefix} FOUND: {det.class_name} ({det.confidence:.1%}) at {det.center}"
                        )

                    # Save annotated image
                    strategy_slug = cfg["name"].lower().replace(" ", "_")
                    annotated = annotate_image(test_img, result.detections)
                    output_path = (
                        output_dir
                        / f"{img_path.stem}_{strategy_slug}_{label.lower()}.jpg"
                    )
                    annotated.save(output_path)
                    print(f"{prefix} SAVED: {output_path.relative_to(base_dir)}")
                else:
                    print(f"{prefix} NOTHING FOUND")


if __name__ == "__main__":
    run_test()
