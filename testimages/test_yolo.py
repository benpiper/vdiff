import sys
from pathlib import Path
from PIL import Image
import logging

# Add parent directory to path to import vdiff
sys.path.append(str(Path(__file__).parent.parent))

from vdiff.detect import ObjectDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("yolo_test")


def run_test():
    image_dir = Path(__file__).parent
    images = list(image_dir.glob("*.jpg"))
    if not images:
        print(f"No .jpg images found in {image_dir}")
        return

    # Test configurations
    configs = [
        {"name": "Current (strict)", "conf": 0.55, "size": 640},
        {"name": "More sensitive", "conf": 0.35, "size": 640},
        {"name": "High Res", "conf": 0.55, "size": 1280},
        {"name": "Sensitive + High Res", "conf": 0.25, "size": 1280},
    ]

    for img_path in sorted(images):
        print("\n" + "=" * 50)
        print(f"TESTING IMAGE: {img_path.name}")
        print("=" * 50)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

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
                    "model": "yolov8s.pt",
                }
            )

            result = detector.detect(img)

            if result.detections:
                for det in result.detections:
                    print(
                        f"  [FOUND] {det.class_name} ({det.confidence:.1%}) at {det.center}"
                    )
            else:
                print("  [NOTHING FOUND]")


if __name__ == "__main__":
    run_test()
