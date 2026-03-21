import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import yaml
import sys
from pathlib import Path

# Add parent directory to path to import vdiff
sys.path.append(str(Path(__file__).parent.parent))

from vdiff.main import VDiffApp, CameraState

class TestPerCameraConfig(unittest.TestCase):
    def setUp(self):
        self.config_yaml = """
cameras:
  - name: "SensitiveCam"
    type: local
    url: "0"
    interval: 10
    detection:
      confidence: 0.1
    diff:
      min_changed_pct: 0.1
  - name: "StrictCam"
    type: local
    url: "1"
    interval: 10
    detection:
      confidence: 0.9
    diff:
      min_changed_pct: 10.0

diff:
  pixel_threshold: 12
  min_changed_pct: 2.0
  ssim_threshold: 0.92

detection:
  enabled: true
  model: "yolov8n.pt"
  confidence: 0.5
  iou_threshold: 0.3
  move_threshold: 15
"""
        self.config = yaml.safe_load(self.config_yaml)

    @patch("vdiff.main.create_camera")
    @patch("vdiff.main.DiffEngine")
    @patch("vdiff.main.ObjectDetector")
    @patch("vdiff.main.AlertDispatcher")
    @patch("vdiff.main.RuleEngine")
    def test_camera_config_merging(self, mock_rule, mock_alert, mock_detector, mock_diff, mock_create_cam):
        # Mock dependencies to avoid loading models/starting threads
        with patch("vdiff.main.VDiffApp._load_config", return_value=self.config):
            app = VDiffApp("mock_config.yaml")
            
            self.assertEqual(len(app.cameras), 2)
            
            cam_sens = app.cameras[0]
            cam_strict = app.cameras[1]
            
            # Verify SensitiveCam config
            self.assertEqual(cam_sens.detection_config["confidence"], 0.1)
            self.assertEqual(cam_sens.detection_config["iou_threshold"], 0.3) # from global
            self.assertEqual(cam_sens.diff_config["min_changed_pct"], 0.1)
            self.assertEqual(cam_sens.diff_config["pixel_threshold"], 12) # from global
            
            # Verify StrictCam config
            self.assertEqual(cam_strict.detection_config["confidence"], 0.9)
            self.assertEqual(cam_strict.diff_config["min_changed_pct"], 10.0)

    @patch("vdiff.main.create_camera")
    @patch("vdiff.main.DiffEngine")
    @patch("vdiff.main.ObjectDetector")
    @patch("vdiff.main.AlertDispatcher")
    @patch("vdiff.main.RuleEngine")
    def test_process_camera_uses_correct_config(self, mock_rule, mock_alert, mock_detector, mock_diff, mock_create_cam):
        with patch("vdiff.main.VDiffApp._load_config", return_value=self.config):
            app = VDiffApp("mock_config.yaml")
            
            # Mock camera capture and engine results
            img = Image.new("RGB", (100, 100))
            for cam_state in app.cameras:
                cam_state.camera.capture = MagicMock(return_value=img)
                cam_state.prev_image = img # Skip first image return path
                
            # Setup mock return values for engines
            mock_diff.return_value.compare.return_value.changed_pct = 5.0
            mock_diff.return_value.compare.return_value.ssim_score = 0.95
            mock_diff.return_value.compare.return_value.changed = True
            mock_diff.return_value.compare.return_value.diff_mask = None
            mock_detector.return_value.detect.return_value.has_changes = True
            mock_detector.return_value.detect.return_value.detections = []
            mock_detector.return_value.detect.return_value.changed_objects.return_value = []
            mock_detector.return_value.detect.return_value.summary.return_value = "Mock summary"
            
            # Test SensitiveCam processing
            app._process_camera(app.cameras[0])
            
            # Check if detector.detect was called with SensitiveCam's config
            mock_detector.return_value.detect.assert_called()
            args, kwargs = mock_detector.return_value.detect.call_args
            self.assertEqual(kwargs["config"]["confidence"], 0.1)
            
            # Check if diff_engine.compare was called with SensitiveCam's config
            mock_diff.return_value.compare.assert_called()
            args, kwargs = mock_diff.return_value.compare.call_args
            self.assertEqual(kwargs["config"]["min_changed_pct"], 0.1)
            
            # Reset mocks for next camera
            mock_detector.return_value.detect.reset_mock()
            mock_diff.return_value.compare.reset_mock()
            
            # Test StrictCam processing
            app._process_camera(app.cameras[1])
            
            # Check if detector.detect was called with StrictCam's config
            args, kwargs = mock_detector.return_value.detect.call_args
            self.assertEqual(kwargs["config"]["confidence"], 0.9)
            
            # Check if diff_engine.compare was called with StrictCam's config
            args, kwargs = mock_diff.return_value.compare.call_args
            self.assertEqual(kwargs["config"]["min_changed_pct"], 10.0)

if __name__ == "__main__":
    unittest.main()
