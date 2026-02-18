# Agent Instructions for vdiff

This project is a hybrid security camera monitor that combines fast local YOLO detection with expensive Vision LLM verification.

## Core Architectural Patterns

1.  **Strict Zonal Analysis (Privacy-First Architecture)**:
    - The camera frame is masked using `VDiffApp._apply_zone_mask(image, state.zone_mask)` **immediately** after capture.
    - This means YOLO, Pixel Diff, and LLM verification **never see** any pixels outside the monitoring zones.
    - This is the project's most critical high-level pattern.

2.  **Ghost Filtering**:
    - YOLO tracks objects, but it can be noisy or "phantom."
    - The pipeline cross-verifies YOLO's `has_changes` against the `pixel_diff` mask.
    - If YOLO thinks an object moved but there is no corresponding pixel movement in that bounding box, it is filtered as a "ghost."

3.  **Vision Fallback**:
    - If Pixel Diff triggers (significant motion) but YOLO finds nothing, the system falls back to asking the LLM to inspect the full (masked) frame.
    - This is implemented in `vdiff/main.py`.

4.  **Hybrid Rule Verification**:
    - High-reliability rules should use `detect_classes` (YOLO) combined with `condition` (LLM).
    - The `RuleEngine` uses Chain-of-Thought prompting to verify crops.

## Development & Troubleshooting

- **Decision Matrix**: Always refer to the "Decision Matrix" in the console logs. It tracks exactly why a frame passed or failed each gate (Pixel Diff, YOLO, Ghost, Rules).
- **Tuning**: If an object is missed, refer the user to the `testimages/test_yolo.py` diagnostic tool.
- **Model Selection**: `yolov8n.pt` (Nano) is the default, but `yolov8s.pt` (Small) is recommended for better reliability on similar hardware.

## File Map
- `vdiff/main.py`: The orchestrator and main loop. Central logging lives here.
- `vdiff/detect.py`: YOLO inference and object tracking.
- `vdiff/diff.py`: Fast grayscale pixel difference and SSIM structural similarity.
- `vdiff/zones.py`: Coordinate parsing and image masking logic.
- `vdiff/rules.py`: Rule matching logic and LLM verification prompts.
