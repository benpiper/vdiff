"""Main orchestrator — capture loop with YOLO detection + diff pipeline."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from vdiff.alerts import AlertDispatcher, AlertEvent
from vdiff.capture import CameraSource, create_camera
from vdiff.describe import ChangeDescriber
from vdiff.detect import ObjectDetector
from vdiff.diff import DiffEngine
from vdiff.rules import RuleEngine
from vdiff.zones import Zone, parse_zones, build_zone_mask, filter_detections_by_zones

logger = logging.getLogger("vdiff")


class CameraState:
    """Tracks per-camera state: previous image, history ring buffer."""

    def __init__(
        self,
        camera: CameraSource,
        history_count: int = 20,
        image_dir: str = "./captures",
        zones: list = None,
    ):
        self.camera = camera
        self.prev_image: Optional[Image.Image] = None
        self.history: list[Path] = []
        self.history_count = history_count
        self.image_dir = Path(image_dir) / self._safe_name(camera.name)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.capture_count = 0
        self.change_count = 0
        self.zones: list[Zone] = zones or []
        self.zone_mask = None  # built on first capture (needs image dimensions)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def save_image(self, img: Image.Image, suffix: str = "") -> Path:
        """Save image to history directory, maintaining ring buffer."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}{suffix}.jpg"
        path = self.image_dir / filename
        img.save(path, "JPEG", quality=85)
        self.history.append(path)

        # Trim ring buffer
        while len(self.history) > self.history_count:
            old = self.history.pop(0)
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass

        return path

    def cleanup_stale_images(self):
        """Remove leftover images from previous runs (not tracked in history)."""
        if not self.image_dir.exists():
            return
        count = 0
        for f in sorted(self.image_dir.glob("*.jpg")):
            if f not in self.history:
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
        if count:
            logger.info(f"[{self.camera.name}] cleaned up {count} stale image(s)")

    def cleanup_all_images(self):
        """Remove all captured images (called on shutdown)."""
        count = 0
        for f in self.history:
            try:
                f.unlink(missing_ok=True)
                count += 1
            except OSError:
                pass
        if self.image_dir.exists():
            for f in self.image_dir.glob("*.jpg"):
                try:
                    f.unlink()
                    count += 1
                except OSError:
                    pass
            try:
                self.image_dir.rmdir()
            except OSError:
                pass
        self.history.clear()
        if count:
            logger.info(f"[{self.camera.name}] removed {count} image(s) on shutdown")


class VDiffApp:
    """Main application class orchestrating the capture-detect-diff-alert pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._running = True

        # Initialize components
        self.cameras: list[CameraState] = []
        self.diff_engine = DiffEngine(self.config.get("diff", {}))
        self.detector = ObjectDetector(self.config.get("detection", {}))
        self.describer = ChangeDescriber(self.config.get("llm", {}))
        self.rule_engine = RuleEngine(
            self.config.get("rules", []),
            self.config.get("llm", {}),
        )
        self.alert_dispatcher = AlertDispatcher(self.config.get("alerts", {}))

        # Storage config
        storage = self.config.get("storage", {})
        history_count = storage.get("history_count", 20)
        image_dir = storage.get("image_dir", "./captures")

        # Initialize cameras
        for cam_cfg in self.config.get("cameras", []):
            camera = create_camera(cam_cfg)
            zones = parse_zones(cam_cfg)
            state = CameraState(camera, history_count, image_dir, zones=zones)
            state.cleanup_stale_images()
            self.cameras.append(state)

        det_status = "enabled" if self.detector.enabled else "disabled"
        logger.info(
            f"vdiff initialized: {len(self.cameras)} camera(s), "
            f"{len(self.rule_engine.rules)} rule(s), "
            f"YOLO {det_status} ({self.detector.model_name})"
        )

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    @staticmethod
    def _load_config(path: str) -> dict:
        config_path = Path(path)
        if not config_path.exists():
            logger.error(f"Config file not found: {path}")
            sys.exit(1)
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        level_str = self.config.get("logging", {}).get("level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Suppress noisy libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received, exiting gracefully...")
        self._running = False

    def run(self):
        """Main loop — poll cameras, detect changes, alert."""
        logger.info("Starting vdiff main loop (Ctrl+C to stop)")

        while self._running:
            any_due = False

            for state in self.cameras:
                if not state.camera.is_due:
                    continue

                any_due = True
                self._process_camera(state)

            if not any_due:
                # Sleep until the next camera is due
                min_wait = min(s.camera.seconds_until_next for s in self.cameras)
                sleep_time = min(max(min_wait, 0.1), 1.0)
                time.sleep(sleep_time)

        logger.info("vdiff stopped.")
        self._print_stats()
        self._cleanup()

    def _process_camera(self, state: CameraState):
        """Run the full pipeline for one camera capture cycle."""
        cam_name = state.camera.name
        logger.info(f"[{cam_name}] capturing frame #{state.capture_count + 1}...")

        # 1. Capture
        t0 = time.monotonic()
        image = state.camera.capture()
        if image is None:
            logger.warning(f"[{cam_name}] capture returned None, skipping cycle")
            return
        t_capture = time.monotonic() - t0

        state.capture_count += 1

        # 2. YOLO detection (runs every frame, fast ~50ms)
        t0 = time.monotonic()
        det_result = self.detector.detect(image)
        t_detect = time.monotonic() - t0

        # Filter YOLO detections to configured zones
        if state.zones:
            img_w, img_h = image.size
            det_result.detections = filter_detections_by_zones(
                det_result.detections, state.zones, img_w, img_h
            )
            # Recompute tracking changes with filtered detections
            det_result.has_changes = any(
                c.change_type in ("appeared", "disappeared", "moved")
                for c in det_result.changes
                if any(
                    z.contains_bbox(
                        c.detection.x1,
                        c.detection.y1,
                        c.detection.x2,
                        c.detection.y2,
                        img_w,
                        img_h,
                    )
                    for z in state.zones
                )
            )

        # Build zone mask on first capture (needs image dimensions)
        if state.zones and state.zone_mask is None:
            img_w, img_h = image.size
            state.zone_mask = build_zone_mask(state.zones, img_w, img_h)
            zone_names = ", ".join(z.name for z in state.zones)
            logger.info(f"[{cam_name}] zone mask built for: {zone_names}")

            # Save debug image showing zones
            try:
                from vdiff.zones import draw_zones

                debug_img = draw_zones(image, state.zones)
                state.save_image(debug_img, "_zones_debug")
                logger.info(f"[{cam_name}] saved zone debug image to captures/")
            except Exception as e:
                logger.warning(f"Failed to save zone debug image: {e}")

        # First image — nothing to compare yet
        if state.prev_image is None:
            state.prev_image = image
            state.save_image(image, "_initial")
            det_summary = (
                f", {len(det_result.detections)} objects"
                if det_result.detections
                else ""
            )
            logger.info(
                f"[{cam_name}] first image captured "
                f"({t_capture * 1000:.0f}ms capture, {t_detect * 1000:.0f}ms detect{det_summary})"
            )
            return

        # 3. Pixel diff (cheap gate) — restricted to zones if configured
        diff_result = self.diff_engine.compare(
            state.prev_image, image, zone_mask=state.zone_mask
        )

        # Decide if anything interesting happened
        yolo_changed = det_result.has_changes
        pixel_changed = diff_result.changed

        # Filter ghost YOLO detections using pixel mask (if available)
        if yolo_changed:
            if diff_result.diff_mask:
                # Verify YOLO changes overlap with actual pixel changes
                det_result.filter_by_mask(diff_result.diff_mask, image.size)
            elif not pixel_changed:
                # YOLO sees change, but pixels did not change (below threshold).
                # Likely ghost/jitter or very small object. Filter it out.
                det_result.changes = [
                    c for c in det_result.changes if c.change_type == "still"
                ]
                det_result.has_changes = False

            yolo_changed = det_result.has_changes

        # 4. Build description from YOLO tracking (fast, no LLM needed)
        description = det_result.summary()

        # 5. For autonomous/moving detections, ask LLM to clarify (if enabled)
        # Apply strict zone masking to ensure LLM only sees content within monitoring zones
        masked_image = self._apply_zone_mask(image, state.zone_mask)
        prev_masked_image = (
            self._apply_zone_mask(state.prev_image, state.zone_mask)
            if state.prev_image
            else None
        )

        objects_to_clarify = [
            c
            for c in det_result.changed_objects()
            if c.change_type in ("appeared", "moved")
        ]
        if objects_to_clarify and self.describer:
            llm_parts = []
            for change in objects_to_clarify[:3]:  # max 3 LLM calls per frame
                cropped = self.detector.crop_detection(masked_image, change.detection)
                llm_desc = self.describer.describe_cropped(
                    cropped, change.detection.class_name, change.detection.confidence
                )
                if llm_desc:
                    llm_parts.append(llm_desc)
            if llm_parts:
                description += " | LLM detail: " + "; ".join(llm_parts)
            elif pixel_changed and not det_result.detections:
                # YOLO found nothing, but pixels changed significantly.
                # Use LLM as a fallback "second eyes" on the full scene.
                fallback_desc = self.describer.describe(
                    prev_masked_image, masked_image, diff_result.region_description
                )
                if fallback_desc:
                    description = f"YOLO missed detection fallback: {fallback_desc}"

        # 6. Evaluate rules against the structured detection data
        matches = self.rule_engine.evaluate(
            description,
            det_result,
            masked_image,
            self.detector,
            prev_image=prev_masked_image,
        )

        # Structured Decision Matrix (INFO level)
        # This helps users troubleshoot why alerts were triggered or suppressed.
        if (
            yolo_changed
            or pixel_changed
            or any(c.change_type != "still" for c in det_result.changes)
        ):
            ghost_status = (
                "PASS"
                if yolo_changed
                else "REJECT"
                if any(c.change_type != "still" for c in det_result.changes)
                else "SKIP"
            )
            matrix = [
                f"\n--- [{cam_name}] Decision Matrix ---",
                f"1. Pixel Diff:  {'PASS' if pixel_changed else 'SKIP'} ({diff_result.changed_pct:.1f}%)",
                f"2. YOLO Detect: {len(det_result.detections)} objects",
                f"3. Ghost Filter: {ghost_status}",
                f"4. Rule Eval:   {len(matches)} matches",
            ]

            for m in matches:
                matrix.append(f"   - Match: {m.rule.name} ({m.rule.severity.upper()})")
                if m.detail:
                    matrix.append("     LLM Reasoning:")
                    for line in m.detail.splitlines()[:3]:
                        matrix.append(f"       {line}")
                    if len(m.detail.splitlines()) > 3:
                        matrix.append("       ...")

            matrix.append("-" * (len(matrix[0]) - 1))
            logger.info("\n".join(matrix))

        if not yolo_changed and not pixel_changed:
            state.prev_image = image
            return

        # Change detected!
        state.change_count += 1
        state.save_image(image, "_change")

        # Build a structured description from YOLO detections
        change_source = []
        if yolo_changed:
            change_source.append("YOLO")
        if pixel_changed:
            change_source.append("pixel-diff")

        # 7. Alert

        event = AlertEvent(
            camera_name=cam_name,
            description=description,
            matched_rules=matches,
            current_image=image,
            diff_mask=diff_result.diff_mask,
            changed_pct=diff_result.changed_pct,
            ssim_score=diff_result.ssim_score,
            detections=det_result.detections,
            detection_changes=det_result.changed_objects(),
        )

        if matches:
            self.alert_dispatcher.dispatch(event)
        else:
            logger.info(
                f"[{cam_name}] change detected but no rules matched: {description}"
            )

        # Update previous image
        state.prev_image = image

    def _apply_zone_mask(self, image: Image.Image, mask: Optional[any]) -> Image.Image:
        """Black out areas of a PIL Image that are outside the provided numpy mask."""
        if mask is None:
            return image

        import numpy as np

        # Convert PIL to numpy
        np_img = np.array(image)
        # Apply mask (mask has 255 for active zones, 0 for outside)
        # Broadcast mask to 3 channels for RGB image
        if len(np_img.shape) == 3:
            mask_3d = mask[:, :, np.newaxis] == 0
            np_img[mask_3d.repeat(3, axis=2)] = 0
        else:
            np_img[mask == 0] = 0

        return Image.fromarray(np_img)

    def _print_stats(self):
        """Print summary stats on shutdown."""
        for state in self.cameras:
            logger.info(
                f"[{state.camera.name}] stats: "
                f"{state.capture_count} captures, "
                f"{state.change_count} changes detected"
            )

    def _cleanup(self):
        """Remove all captured images on shutdown."""
        for state in self.cameras:
            state.cleanup_all_images()
        captures_dir = Path(
            self.config.get("storage", {}).get("image_dir", "./captures")
        )
        try:
            captures_dir.rmdir()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="vdiff — lightweight camera change detection & alerts"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    app = VDiffApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
