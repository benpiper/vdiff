# vdiff

A hybrid YOLO + Vision AI security camera monitor that eliminates false alarms through intelligent detection, zonal masking, and automated scene verification.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy example config (contains all defaults and rule examples)
cp example.config.yaml config.yaml

# Edit config with your camera URL, credentials, LLM endpoint, etc.
nano config.yaml

# Run (YOLO model auto-downloads on first run, ~6MB)
python3 run.py
# or with custom config path:
python3 run.py -c /path/to/config.yaml
```

## How It Works

vdiff runs a loop for each camera: **capture → YOLO detect → pixel diff → cross-verify → describe → rules → alert**.

YOLO runs every frame (~50ms on CPU). The pixel diff acts as a secondary gate. **Ghost Filtering** cross-references YOLO motion with the pixel difference mask to filter out tracker jitter. If YOLO misses an object but motion is significant, the system triggers a **Vision Fallback** where the LLM inspects the full frame as a "second set of eyes."

### The Pipeline

```
  Capture image from camera
         │
         ▼
  ┌─── Zone Masking ──────────┐
  │  Apply black mask to any  │
  │  area outside monitored    │
  │  zones. Privacy first.     │
  └────────────┬───────────────┘
               │
               ▼
  ┌─── YOLO Detection ────────┐
  │  Run YOLOv8 on masked frame│
  │  Compare to previous stats:│
  │  • appeared / moved        │
  │  • disappeared / still     │
  └────────────┬───────────────┘
               │
               ▼
  ┌─── Pixel Diff ────────────┐
  │  Stage 1: Count changed %  │
  │  Stage 2: SSIM structural  │
  │  comparison.               │
  └────────────┬───────────────┘
               │
               ▼
  ┌── Verify (Ghost Filter) ──┐
  │  Cross-check YOLO changes  │
  │  against pixel diff mask.  │
  │  Drops detections if no    │
  │  corresponding pixel change│
  └────────────┬───────────────┘
               │
               ▼
  ┌─ Proceed? (Decision Matrix)┐
  │  YOLO changes OR pixel motion│
  │  Log summary & percentages. │
  │  Exit early if no change.  │
  └────────────┬───────────────┘
               │
               ▼
  ┌─── LLM Clarify ───────────┐
  │  1. Autonomous: Sends a    │
  │     CROP of YOLO detections│
  │  2. Fallback: If YOLO misses│
  │     but motion is high,    │
  │     sends the FULL image.  │
  │                            │
  │  **Strict Masking**: The   │
  │  LLM is "blinded" to areas │
  │  outside monitoring zones. │
  └────────────┬───────────────┘
               │
               ▼
  ┌─── Rule Evaluation ────────┐
  │  1. YOLO Class Match       │
  │  2. Hybrid: Class + LLM    │
  │  3. LLM Text Match         │
  └────────────┬───────────────┘
               │ (any rule matched?)
               ▼
  ┌─── Alert ──────────────────┐
  │  Console structured log.   │
  │  Email with attachment.    │
  └────────────────────────────┘
```

## Configuration Reference

### `cameras`

```yaml
cameras:
  - name: "Front Door"
    type: http                     # http | rtsp | local
    url: "http://192.168.88.77/ISAPI/Streaming/channels/101/picture"
    username: "admin"
    password: "password"
    auth: digest                   # digest | basic | none
    interval: 10                   # seconds between captures
```

**Camera types:**
- `http` — fetches a JPEG/PNG from a URL (Hikvision ISAPI, or any snapshot URL)
- `rtsp` — grabs a single frame from an RTSP stream
- `local` — USB/CSI camera via OpenCV (use `device_id: 0` for default)

#### Zones (Region of Interest)

Only monitor specific areas of the frame. Omit `zones` to monitor the entire image.

```yaml
cameras:
  - name: "Front Door"
    # ... camera settings ...
    zones:
      - name: "driveway"
        x: 25         # left edge at 25% from left
        y: 50         # top edge at 50% from top
        w: 50         # 50% of image width
        h: 50         # 50% of image height
```

> [!NOTE]
> On startup, vdiff saves a `_zones_debug.jpg` to your captures folder. Check this image to visually verify your zone boundaries (active zones are bright, masked areas are dimmed).

Coordinates are **percentages** (0–100) of the image dimensions:

```
 0%                    50%                  100%
  ┌──────────────────────────────────────────┐ 0%
  │                                          │
  │   (x=25, y=10)                           │
  │     ┌─────────────────┐                  │
  │     │                 │                  │ 25%
  │     │   "driveway"    │ h=30             │
  │     │                 │                  │
  │     └─────────────────┘                  │ 40%
  │           w=50                           │
  │                                          │ 50%
  │                                          │
  │                                          │
  └──────────────────────────────────────────┘ 100%
```

| Property | Meaning                      | Edge calculation      |
| -------- | ---------------------------- | --------------------- |
| `x`      | Left edge, % from left side  | Right edge = `x + w`  |
| `y`      | Top edge, % from top         | Bottom edge = `y + h` |
| `w`      | Width of zone as % of image  |                       |
| `h`      | Height of zone as % of image |                       |

**Common zone examples:**

| Region              | x   | y   | w   | h   |
| ------------------- | --- | --- | --- | --- |
| Full frame          | 0   | 0   | 100 | 100 |
| Left half           | 0   | 0   | 50  | 100 |
| Right half          | 50  | 0   | 50  | 100 |
| Top half            | 0   | 0   | 100 | 50  |
| Bottom half         | 0   | 50  | 100 | 50  |
| Center quarter      | 25  | 25  | 50  | 50  |
| Bottom-right corner | 70  | 70  | 30  | 30  |

When zones are active:
- **Pixel diff** only counts changes within zone pixels
- **YOLO** runs on the full image but **filters out** detections that don't overlap a zone (≥30% overlap required)
- Multiple zones can be defined — they act as a combined mask

---

### `detection` (YOLO)

```yaml
detection:
  enabled: true
  model: "yolov8n.pt"         # nano (~6MB), also: yolov8s.pt, yolov8m.pt
  confidence: 0.25            # min confidence to keep a detection
  iou_threshold: 0.3          # non-max-suppression overlap threshold
  move_threshold: 15          # pixels moved to count as "moved"
  img_size: 640               # inference resolution
  # classes: [0, 2]           # filter to specific COCO class IDs
```

| Setting          | What it does                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `model`          | YOLO model size. `yolov8n.pt` = fastest/smallest. `yolov8s.pt` = more accurate. Auto-downloads.                  |
| `confidence`     | Detections below this confidence are discarded. Lower = more detections (more false positives).                  |
| `iou_threshold`  | Controls non-max-suppression. Lower = more aggressive deduplication of overlapping boxes.                        |
| `move_threshold` | Minimum pixel distance an object's center must shift between frames to count as "moved".                         |
| `img_size`       | Image is resized to this for inference. Higher = more accurate for small objects, slower.                        |
| `classes`        | Optional COCO class ID filter. Common: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 15=cat, 16=dog. |

> [!TIP]
> Set `enabled: false` to disable YOLO and fall back to pure pixel-diff + LLM mode.

---

### `diff` (Pixel Diff)

```yaml
diff:
  pixel_threshold: 12         # per-pixel noise gate (0-255)
  min_changed_pct: 2.0        # % of pixels that must exceed threshold
  ssim_threshold: 0.97        # structural similarity gate (0-1)
  resize_width: 640           # resize before diffing
```

The diff engine acts as a **secondary gate** alongside YOLO. A frame triggers an alert if **either** YOLO tracking detects a change (appeared/disappeared/moved) **or** the pixel diff confirms structural change.

**Stage 1: Pixel Diff** — For each pixel, compute `|prev - curr|`. Count pixels exceeding `pixel_threshold`. If fewer than `min_changed_pct` percent changed → no change.

**Stage 2: SSIM** — Only runs if Stage 1 triggers. Structural Similarity Index (0–1, 1=identical). If score is above `ssim_threshold` → no change (just noise/lighting).

| Scenario         | pxdiff | SSIM | Result (defaults) |
| ---------------- | ------ | ---- | ----------------- |
| No change        | 0.5%   | 0.99 | ❌ Stage 1 rejects |
| Lighting shift   | 3.5%   | 0.95 | ❌ Stage 2 rejects |
| Person walks by  | 8.0%   | 0.85 | ✅ Change          |
| Car enters frame | 15.0%  | 0.75 | ✅ Change          |

> [!TIP]
> The `pxdiff` value in logs is the Stage 1 percentage. If you see `pxdiff=3.5%` with `min_changed_pct: 2.0` but "no change", it means Stage 2 (SSIM) rejected it as structurally similar.

---

### `llm`

```yaml
llm:
  provider: ollama
  host: "http://192.168.88.86:11434"
  model: "llava"                   # vision model (must support images)
  eval_model: "llama2-uncensored"  # text model for rule evaluation
  timeout: 60                      # seconds per LLM call
```

The **vision model** (`model`) is used only for ambiguous YOLO detections (cropped images, <50% confidence). The **eval model** (`eval_model`) evaluates text-based rules. Both are optional — YOLO class-based rules don't need any LLM.

---

### `rules`

Rules support two matching strategies:

#### 1. YOLO Class Matching (Fast)

```yaml
- name: "Person detected"
  detect_classes: ["person"]       # COCO class names
  detect_change: "appeared"        # appeared | disappeared | moved | (empty = any)
  severity: high
```

Matches instantly by comparing the YOLO detection class name. No LLM call needed.

#### 2. Hybrid: YOLO + LLM Verification (Precise)

```yaml
- name: "White SUV"
  detect_classes: ["car"]          # 1. Fast pre-filter (YOLO)
  detect_change: "appeared"        # (Optional) remove to detect departure too
  condition: "a white SUV"         # 2. Precise verification (LLM Vision)
  severity: high
```

**Features:**
- **Arrival & Departure**: If `detect_change` is omitted, the rule monitors both. For disappearance events, it automatically checks the **previous frame** to verify the object that left.
- **Reliability**: Uses **Chain-of-Thought prompt engineering**. The LLM is forced to describe the object and reason about its identity before issuing a verdict, significantly reducing false positives.
- **Ghost Protection**: YOLO detections are only considered if they overlap with actual pixel changes in the frame.

#### 3. LLM Condition Matching (Flexible, Slower)

```yaml
- name: "Door opened"
  condition: "a door appears to have been opened or is now ajar"
  severity: medium
```

The description text is sent to the eval LLM which answers YES/NO. Falls back to keyword matching if LLM is unavailable.

#### Common COCO Classes for `detect_classes`

`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `cat`, `dog`, `bird`, `backpack`, `umbrella`, `handbag`, `suitcase`, `bottle`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `cell phone`, `laptop`

---

### `alerts`

```yaml
alerts:
  console:
    enabled: true
  email:
    enabled: true
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "you@gmail.com"
    password: "app-password"
    from_addr: "you@gmail.com"
    to_addrs: ["you@gmail.com"]
    min_severity: medium
```

---

### `storage`

```yaml
storage:
  history_count: 20
  image_dir: "./captures"
```

Images are saved during operation and **automatically cleaned up on shutdown**.

---

## Troubleshooting & Tuning

vdiff provides a **Decision Matrix** in the console for every frame that triggers the initial pixel-diff. Use this to understand why alerts are firing or being suppressed.

### Understanding the Decision Matrix

```text
--- [Front Door] Decision Matrix ---
1. Pixel Diff:  PASS (15.2%)       # Stage 1 gate passed
2. YOLO Detect: 1 objects          # YOLO found a "person"
3. Ghost Filter: REJECT            # Pixel mask didn't match YOLO motion
4. Rule Eval:   0 matches          # No alerts sent
------------------------------------
```

### Tuning Guide

| Symptom                                | Stage        | Likely Cause               | Tuning Action                                     |
| :------------------------------------- | :----------- | :------------------------- | :------------------------------------------------ |
| **Alerts on shadows/clouds**           | Pixel Diff   | `ssim_threshold` too low   | Increase `ssim_threshold` (e.g., 0.98)            |
| **No alerts on small objects**         | Pixel Diff   | `min_changed_pct` too high | Decrease `min_changed_pct` (e.g., 1.0)            |
| **Ghost Filter REJECT on real motion** | Ghost Filter | Threshold mismatch         | Decrease `pixel_threshold` in `diff` config       |
| **LLM says NO to a real object**       | Rule Eval    | Crop is too small/blurry   | Increase `img_size` in `detection` config         |
| **Too many alerts on "car"**           | Rule Eval    | Rule is too broad          | Use a **Hybrid Rule** with a specific `condition` |

> [!TIP]
> If you are unsure why an object is being rejected by the LLM, check the console output. The Decision Matrix now includes the **LLM Reasoning** (Chain-of-Thought) when a Hybrid Rule is evaluated.

### Diagnostic Tuning Tool

If YOLO is failing to identify an object in your environment, use the provided diagnostic script to find the optimal settings:

1. Save a sample image with the object to `testimages/`
2. Run the diagnostic: `python3 testimages/test_yolo.py`
3. The script will test several strategies (varying `confidence` and `img_size`) and tell you which one caught the object.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
