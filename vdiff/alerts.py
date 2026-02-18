"""Alert dispatchers — console logging and email notifications."""

import io
import logging
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class AlertEvent:
    """Encapsulates everything about an alert-worthy event."""

    def __init__(
        self,
        camera_name: str,
        description: str,
        matched_rules: list,
        current_image: Optional[Image.Image] = None,
        diff_mask: Optional[Image.Image] = None,
        changed_pct: float = 0.0,
        ssim_score: float = 1.0,
        detections: list = None,
        detection_changes: list = None,
    ):
        self.timestamp = datetime.now()
        self.camera_name = camera_name
        self.description = description
        self.matched_rules = matched_rules
        self.current_image = current_image
        self.diff_mask = diff_mask
        self.changed_pct = changed_pct
        self.ssim_score = ssim_score
        self.detections = detections or []
        self.detection_changes = detection_changes or []

    @property
    def max_severity(self) -> str:
        if not self.matched_rules:
            return "low"
        levels = {"low": 0, "medium": 1, "high": 2}
        max_lvl = max(levels.get(m.rule.severity, 0) for m in self.matched_rules)
        return {0: "low", 1: "medium", 2: "high"}[max_lvl]

    @property
    def severity_level(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(self.max_severity, 0)

    def rule_names(self) -> list[str]:
        return [m.rule.name for m in self.matched_rules]

    def format_text(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"[{self.max_severity.upper()}] vdiff Alert — {self.camera_name}",
            f"Time: {ts}",
            f"Description: {self.description}",
            f"Change: {self.changed_pct:.1f}% pixels, SSIM={self.ssim_score:.3f}",
        ]
        if self.detections:
            det_str = ", ".join(
                f"{d.class_name}({d.confidence:.0%})" for d in self.detections
            )
            lines.append(f"Objects detected: {det_str}")
        if self.detection_changes:
            changes_str = ", ".join(str(c) for c in self.detection_changes)
            lines.append(f"Changes: {changes_str}")
        if self.matched_rules:
            lines.append("Matched rules:")
            for m in self.matched_rules:
                lines.append(f"  • {m.rule.name} ({m.rule.severity.upper()})")
                if m.reason:
                    lines.append(f"    Reason: {m.reason}")
                if m.detail:
                    lines.append("    LLM Reasoning:")
                    # Indent and truncate LLM reasoning
                    reasoning_lines = m.detail.splitlines()
                    for line in reasoning_lines[:10]:
                        lines.append(f"      {line.strip()}")
                    if len(reasoning_lines) > 10:
                        lines.append("      ...")

        return "\n".join(lines)


class ConsoleAlert:
    """Logs alerts to the console with structured formatting."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.level = config.get("level", "info")

    def send(self, event: AlertEvent):
        if not self.enabled:
            return

        log_fn = {
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }.get(self.level, logger.info)

        log_fn(f"\n{'=' * 60}\n{event.format_text()}\n{'=' * 60}")


class EmailAlert:
    """Sends alert emails with optional image attachments."""

    SEVERITY_LEVELS = {"low": 0, "medium": 1, "high": 2}

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.smtp_host = config.get("smtp_host", "")
        self.smtp_port = config.get("smtp_port", 587)
        self.use_tls = config.get("use_tls", True)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_addr = config.get("from_addr", "")
        self.to_addrs = config.get("to_addrs", [])
        min_sev = config.get("min_severity", "medium")
        self.min_severity_level = self.SEVERITY_LEVELS.get(min_sev, 1)

    def send(self, event: AlertEvent):
        if not self.enabled:
            return

        if event.severity_level < self.min_severity_level:
            logger.debug(
                f"Email skipped: severity {event.max_severity} below minimum threshold"
            )
            return

        try:
            self._send_email(event)
            logger.info(f"Email alert sent to {', '.join(self.to_addrs)}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_email(self, event: AlertEvent):
        msg = MIMEMultipart()
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)

        # Build subject detail
        details = []
        if event.matched_rules:
            details.extend(event.rule_names())

        # Add specific changes (e.g. "car appeared")
        if event.detection_changes:
            # Prioritize active changes for subject line
            active_changes = [
                c for c in event.detection_changes if c.change_type != "still"
            ]
            display_changes = (
                active_changes if active_changes else event.detection_changes
            )

            # Limit to first 2 changes to keep subject brief
            for c in display_changes[:2]:
                details.append(f"{c.detection.class_name} {c.change_type}")
            if len(display_changes) > 2:
                details.append(f"+{len(display_changes) - 2} more")

        subject_str = ", ".join(details)
        if not subject_str:
            subject_str = "Change detected"

        msg["Subject"] = (
            f"[vdiff {event.max_severity.upper()}] {event.camera_name} — {subject_str}"
        )

        # Text body
        body = event.format_text()
        msg.attach(MIMEText(body, "plain"))

        # Attach current image
        if event.current_image:
            img_data = self._image_bytes(event.current_image)
            img_attachment = MIMEImage(img_data, _subtype="jpeg")
            img_attachment.add_header(
                "Content-Disposition", "attachment", filename="capture.jpg"
            )
            msg.attach(img_attachment)

        # Attach diff mask
        if event.diff_mask:
            diff_data = self._image_bytes(event.diff_mask)
            diff_attachment = MIMEImage(diff_data, _subtype="jpeg")
            diff_attachment.add_header(
                "Content-Disposition", "attachment", filename="diff.jpg"
            )
            msg.attach(diff_attachment)

        # Send
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

    @staticmethod
    def _image_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


class AlertDispatcher:
    """Routes alert events to all configured alert channels."""

    def __init__(self, alerts_config: dict):
        self.channels = []

        console_cfg = alerts_config.get("console", {})
        if console_cfg.get("enabled", True):
            self.channels.append(ConsoleAlert(console_cfg))

        email_cfg = alerts_config.get("email", {})
        if email_cfg.get("enabled", False):
            self.channels.append(EmailAlert(email_cfg))

    def dispatch(self, event: AlertEvent):
        """Send an alert event to all enabled channels."""
        for channel in self.channels:
            try:
                channel.send(event)
            except Exception as e:
                logger.error(f"Alert channel {type(channel).__name__} failed: {e}")
