import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import io
from vdiff.alerts import EmailAlert, AlertEvent
from vdiff.detect import Detection


class TestEmailAttachment(unittest.TestCase):
    def setUp(self):
        self.config = {
            "enabled": True,
            "smtp_host": "127.0.0.1",
            "smtp_port": 1025,
            "use_tls": False,
            "attach_image": True,
            "from_addr": "test@example.com",
            "to_addrs": ["recipient@example.com"],
            "min_severity": "low",
        }
        self.email_alert = EmailAlert(self.config)

    @patch("smtplib.SMTP")
    def test_send_email_with_attachment(self, mock_smtp):
        # Create a dummy image
        img = Image.new("RGB", (100, 100), color="red")

        # Create an alert event with a detection
        detection = Detection(
            class_id=0, class_name="person", confidence=0.9, x1=10, y1=10, x2=50, y2=50
        )

        event = AlertEvent(
            camera_name="TestCam",
            description="Test alert",
            matched_rules=[],
            current_image=img,
            detections=[detection],
        )

        # Mock draw_detections to ensure it's called
        with patch("vdiff.alerts.draw_detections", return_value=img) as mock_draw:
            self.email_alert.send(event)

            # Verify SMTP interactions
            mock_smtp.assert_called_with("127.0.0.1", 1025)
            instance = mock_smtp.return_value.__enter__.return_value
            self.assertTrue(instance.sendmail.called)

            # Verify draw_detections was called
            mock_draw.assert_called_once()

            # Verify attachment in the sent message
            args, kwargs = instance.sendmail.call_args
            msg_string = args[2]
            self.assertIn("Content-Type: image/jpeg", msg_string)
            self.assertIn(
                'Content-Disposition: attachment; filename="capture.jpg"', msg_string
            )


if __name__ == "__main__":
    unittest.main()
