"""Rule engine — evaluates alert conditions against detections and descriptions."""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single alert rule."""

    name: str
    condition: str
    severity: str = "medium"  # low | medium | high
    # Optional: match specific YOLO classes directly (fast, no LLM)
    detect_classes: list = None  # e.g., ["person", "car"]
    detect_change: str = ""  # "appeared", "disappeared", "moved", or "" for any

    def __post_init__(self):
        if self.detect_classes is None:
            self.detect_classes = []

    @property
    def severity_level(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(self.severity, 1)


@dataclass
class RuleMatch:
    """A matched rule with context."""

    rule: Rule
    confidence: str = ""
    reason: str = ""
    detail: str = ""


class RuleEngine:
    """Evaluates user-defined rules against YOLO detections and descriptions."""

    def __init__(self, rules_config: list, llm_config: dict):
        self.rules = []
        for r in rules_config:
            self.rules.append(
                Rule(
                    name=r.get("name", "unnamed"),
                    condition=r.get("condition", ""),
                    severity=r.get("severity", "medium"),
                    detect_classes=r.get("detect_classes", []),
                    detect_change=r.get("detect_change", ""),
                )
            )
        self.llm_config = llm_config
        self._client = None
        # Separate timeout for rule eval (text-only, should be fast)
        self._eval_timeout = llm_config.get("eval_timeout", 10)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(
                host=self.llm_config.get("host", "http://localhost:11434"),
                timeout=self._eval_timeout,
            )
        return self._client

    def evaluate(
        self,
        description: str,
        det_result=None,
        image: Optional[Image.Image] = None,
        detector=None,
        prev_image: Optional[Image.Image] = None,
    ) -> list[RuleMatch]:
        """
        Evaluate all rules.
        """
        if not self.rules:
            return []

        matches = []
        llm_rules = []

        for rule in self.rules:
            # Strategy 1 & 2: YOLO class matching (with optional LLM hybrid verification)
            if rule.detect_classes:
                if det_result is not None:
                    match = self._match_by_class(
                        rule, det_result, image, detector, prev_image
                    )
                    if match:
                        matches.append(match)
                # If rule specifies classes, do NOT fallback to generic LLM description matching.
                # This prevents FPs where "0 objects" is matched against description by confused LLM.
                continue

            # Strategy 3: Pure LLM evaluation (only if NO classes specified)
            if rule.condition and description:
                llm_rules.append(rule)

        # Strategy 2: Non-blocking LLM evaluation for text-condition rules
        if llm_rules and description:
            llm_matches = self._evaluate_nonblocking(description, llm_rules)
            matches.extend(llm_matches)

        return matches

    def _match_by_class(
        self, rule, det_result, image=None, detector=None, prev_image=None
    ) -> Optional[RuleMatch]:
        """
        Match by YOLO class. If rule ALSO has a condition, verify it against the crop.
        """
        for change in det_result.changes:
            class_name = change.detection.class_name.lower()

            # Check if this detection matches any of the rule's target classes
            class_match = any(
                target.lower() in class_name or class_name in target.lower()
                for target in rule.detect_classes
            )
            if not class_match:
                continue

            # Check change type filter
            if rule.detect_change and change.change_type != rule.detect_change:
                continue

            # If rule has a condition (e.g. "white SUV"), verify it on the crop
            if rule.condition:
                target_img = image
                if change.change_type == "disappeared":
                    target_img = prev_image

                if target_img is None or detector is None:
                    # Can't verify condition without image/detector
                    continue

                success, reasoning = self._verify_crop_condition(
                    target_img, change.detection, rule.condition, detector
                )
                if not success:
                    logger.info(
                        f"Rule {rule.name}: matched class {class_name} but failed condition '{rule.condition}'"
                    )
                    continue

            # Match!
            conf_str = f"({change.detection.confidence:.0%})"
            if change.change_type == "disappeared" and rule.condition:
                reason = f"{change.detection.class_name} disappeared (verified as '{rule.condition}' in previous frame)"
            else:
                reason = (
                    f"{change.detection.class_name} {change.change_type} {conf_str}"
                )
                if rule.condition:
                    reason += f" verified: '{rule.condition}'"

            detail = reasoning if rule.condition else ""

            logger.info(f"Rule matched (Hybrid/YOLO): {rule.name} — {reason}")
            return RuleMatch(
                rule=rule, confidence="yolo-hybrid", reason=reason, detail=detail
            )

        return None

    def _verify_crop_condition(
        self, image, detection, condition, detector
    ) -> tuple[bool, str]:
        """Ask LLM if the cropped object matches the condition."""
        try:
            crop = detector.crop_detection(image, detection)
            import io

            buf = io.BytesIO()
            crop.save(buf, format="JPEG")
            img_bytes = buf.getvalue()

            client = self._get_client()
            # Use vision model for this check
            model = self.llm_config.get("model", "llava")

            prompt = (
                f"Task: Verify if the image contains '{condition}'.\n"
                f"1. Describe the main object in the image.\n"
                f"2. Compare it to the description '{condition}'.\n"
                f"3. Conclude with exactly 'VERDICT: YES' or 'VERDICT: NO'."
            )

            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_bytes],
                    }
                ],
            )
            content = response["message"]["content"].strip()
            # Log reasoning for debugging
            logger.info(f"LLM Verification Reasoning ({condition}):\n{content}")

            success = "VERDICT: YES" in content.upper()
            return success, content

        except Exception as e:
            msg = f"Hybrid rule verification failed: {e}"
            logger.warning(msg)
            return False, msg

    def _evaluate_nonblocking(
        self, description: str, rules: list[Rule]
    ) -> list[RuleMatch]:
        """Run LLM evaluation in a thread with a timeout, fall back to keywords."""
        try:
            future = self._executor.submit(self._evaluate_with_llm, description, rules)
            return future.result(timeout=self._eval_timeout)
        except TimeoutError:
            logger.warning(
                f"LLM rule eval timed out after {self._eval_timeout}s, using keyword fallback"
            )
            return self._evaluate_keywords(description, rules)
        except Exception as e:
            logger.warning(f"LLM rule eval failed, using keyword fallback: {e}")
            return self._evaluate_keywords(description, rules)

    def _evaluate_with_llm(
        self, description: str, rules: list[Rule]
    ) -> list[RuleMatch]:
        """Batch-evaluate rules in a single LLM call."""
        client = self._get_client()
        model = self.llm_config.get("eval_model", "llama3.2")

        rules_text = "\n".join(
            f'{i + 1}. "{rule.name}": {rule.condition}' for i, rule in enumerate(rules)
        )

        prompt = (
            f"You are a security alert evaluator. A camera detected this change:\n"
            f'"{description}"\n\n'
            f"Evaluate each rule below. For EACH rule, respond with ONLY "
            f'"YES" or "NO" on a separate line. No explanations.\n\n'
            f"Rules:\n{rules_text}\n\n"
            f"Responses (one per line):"
        )

        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response["message"]["content"].strip()
        lines = [line.strip().upper() for line in answer.split("\n") if line.strip()]

        matches = []
        for i, rule in enumerate(rules):
            if i < len(lines) and lines[i].startswith("YES"):
                matches.append(RuleMatch(rule=rule, confidence="llm"))
                logger.info(
                    f"Rule matched (LLM): {rule.name} (severity={rule.severity})"
                )

        return matches

    def _evaluate_keywords(
        self, description: str, rules: list[Rule]
    ) -> list[RuleMatch]:
        """Simple keyword-based fallback when LLM is unavailable."""
        desc_lower = description.lower()
        matches = []

        for rule in rules:
            if not rule.condition:
                continue
            condition_words = rule.condition.lower().split()
            keywords = [w for w in condition_words if len(w) > 3]
            matched_words = [w for w in keywords if w in desc_lower]

            if len(matched_words) >= 2 or (
                len(keywords) <= 2 and len(matched_words) >= 1
            ):
                matches.append(
                    RuleMatch(
                        rule=rule,
                        confidence="keyword-match",
                        reason=f"Matched keywords: {', '.join(matched_words)}",
                    )
                )
                logger.info(f"Rule keyword match: {rule.name} ({matched_words})")

        return matches
