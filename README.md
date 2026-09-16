# Road Crack Detection Engine

Road Crack Detection Engine analyzes road-survey video with YOLO segmentation,
custom crack merging, branch analysis, and pixel measurements. Camera calibration
supports physical measurements, and SuperPoint + LightGlue feature matching
supports tracking defects across observations.

[RoadVision](https://infradrone.vercel.app) is the companion web interface for
exploring survey footage, maps, and damage findings. This repository contains the
Python analysis engine; video processing runs independently of the legacy frame queue.

## Video pipeline

```text
Video → timestamped frames → segmentation → crack merging and measurement → Damage records
```

```python
from pathlib import Path
from src.engine.video import Video
from src.engine.video_pipeline import VideoPipeline

video = Video(footage_path=Path("survey.mp4"), route_id="route-1")
pipeline = VideoPipeline(
    model_path=Path("src/ml/models/weights/segmentation/best.pt"),
    device="cpu",  # Use "cuda:0" on a GPU host.
)
manifest = pipeline.write(video, Path("outputs/survey-1"), sample_fps=2)
```

The output directory must be new. It contains `manifest.json`, `damages.jsonl`, and
compressed `masks/<damage-id>.npz` files with `mask` and `skeleton` arrays. Mask paths
in JSON are relative to the output directory. A complete manifest is written only
after all sampled frames have been processed; failures are recorded and re-raised.
For in-memory use, iterate `pipeline.process(video, sample_fps=2)` to receive
`Damage` objects without writing files. The model loads once per pipeline instance.

PyAV decodes formats/codecs supported by its FFmpeg build, including common MP4,
MOV, AVI, and MKV files. Sampling uses presentation timestamps, not nominal FPS.
Frames keep encoded orientation; phone rotation metadata is not applied yet.

Optional `video.gps` samples are matched using `video.metadata.recorded_at` (the
first frame's timezone-aware recording time) plus the frame time and
`gps_offset_seconds`. Gaps longer than `max_gps_gap_seconds` (default 5) and times
outside the track have no location. GPS represents the phone's position.

Masks and bounding boxes use original-frame pixels. The original dimensions
remain in pixels and square pixels; optional metric estimates are separate fields. Length is the existing endpoint-to-endpoint estimate,
not a traced curved-crack length. Severity remains unset. Crack subtype orientation
assumes the road runs vertically in the image. Observations are not deduplicated
across frames. GPS file parsing, camera calibration, and Modal deployment are separate work.

Run the video tests with:

```sh
uv run python -m unittest discover -s src/tests -p 'test_video*.py'
```

## Physical dimensions and two-view reconstruction

Both methods use `video.calibration`. Supply calibration for the exact encoded
camera view; the pipeline does not infer S25+ camera parameters from a model name
or assume a normal MP4 contains intrinsics or motion sensors.

```python
import json
from src.engine.calibration import CameraCalibration

video.calibration = CameraCalibration(**json.loads(Path("calibration.json").read_text()))
pipeline = VideoPipeline(
    Path("src/ml/models/weights/segmentation/best.pt"),
    device="cpu",
    triangulation=True,
    motion_scale_source="camera_height",  # Or explicitly "gps".
)
pipeline.write(video, Path("outputs/calibrated-survey"), sample_fps=2)
```

Calibration fields:

- `image_width`, `image_height`: encoded resolution. Mismatches are rejected.
- `fx`, `fy`, `cx`, `cy`: camera intrinsics in encoded-image pixels, accounting for
  crop/zoom and any rotation or stabilization performed before processing.
- `camera_height_m`: perpendicular camera-to-road distance, not GPS altitude.
- `pitch_degrees`: downward camera tilt from the road (0 horizontal, 90 straight down).
- `roll_degrees`: clockwise rotation of the road normal in image coordinates (default 0).
- `distortion`: OpenCV `[k1, k2, p1, p2, k3]`. Default zeros assume the calibrated
  image already has no distortion; this is not an automatic device correction.
- `road_roi`: `[left, top, right, bottom]` pixel rectangle containing only road.
  Required for triangulation; also bounds single-frame measurements when supplied.
- `max_range_m`: maximum camera-to-point range (default 30 meters).

**Camera geometry:** project each foreground pixel cell onto the road plane using
intrinsics, mounting height, and orientation. Sum projected cell areas and metric
skeleton-edge lengths. `metric_dimensions` reports `area_m2`, `length_m`, and
`mean_width_m` (area / skeleton length). This width is an average estimate; it is
not a local minimum/maximum crack width. Unlike the existing pixel length, metric
length follows the full skeleton rather than its endpoint-to-endpoint chord.
Masks crossing the horizon, range limit, or ROI receive no metric estimate.

**Two-view geometry:** track ORB features inside the road ROI, fit a homography
with RANSAC, recover candidate camera poses, and select a pose consistent with the
mounting normal. A homography is used because the road is approximately planar.
Triangulate matched points and require positive depth, sufficient parallax,
reprojection agreement, and enough inliers before accepting the current road plane.
Metric measurements are restricted to the convex hull of the matched features.
Only the previous sampled frame's features are retained; this is a local two-view
reconstruction, not SLAM, a full 3D road map, or cross-survey defect matching.

`motion_scale_source="camera_height"` scales translation using the known height.
This shares the first method's height assumption, so agreement cannot independently
validate the mounting height. `"gps"` instead uses displacement between synchronized
phone locations only when both report accuracy and displacement exceeds three times
the summed accuracy radii. This is a heuristic, not a confidence bound. Interpolated
GPS fixes currently have unknown accuracy and cannot supply scale. GPS failure does
not silently switch to camera-height scaling.

Exports include original `dimensions`, `metric_dimensions`, and
`triangulated_dimensions`, their availability statuses, and `motion` diagnostics
(reference frame, scale source, baseline, inlier count, parallax, reprojection error,
road plane, and feature support). The first frame, weak/ambiguous motion, or missing
scale produces an unavailable triangulated estimate. Camera-based estimates remain
independent. The manifest records the calibration and selected motion settings.

Both methods assume a locally flat road. They do not measure pothole depth, infer
individual crack GPS coordinates, or establish millimeter accuracy. Fixed mount,
lens, crop, and recording geometry are required; dynamic stabilization needs its
own per-frame correction. Validate on measured pavement examples before interpreting
metric values as physical ground truth. Synthetic tests check the math and pipeline,
not accuracy on actual S25+ recordings.

Run all tests with `uv run python -m unittest discover -s src/tests -p 'test_*.py'`.

## Temporal matching

`VideoPipeline.process_frames()` exposes each sampled frame, synchronized phone
location, and its damage list (including empty lists). `process()` is the flat
measurement iterator. Temporal matching uses pretrained SuperPoint + LightGlue with the shared homography
fitting code, without requiring triangulation or motion scale. Models load once per
engine instance, using CUDA when available and CPU otherwise. Pass `device="cpu"`
to select CPU explicitly. The first initialization downloads pretrained weights.

```python
from contextlib import closing
from src.engine.temporal import TemporalEngine
from src.engine.temporal_store import TemporalStore

# video must have a stable route_id and calibration with a road-only road_roi.
with closing(TemporalStore("damage.sqlite")) as store:
    temporal = TemporalEngine(store)
    with closing(pipeline.process_frames(video, sample_fps=2)) as frames:
        for frame in frames:
            associations = temporal.process(video, frame)
            for association in associations:
                print(association.defect_id, association.status)
```

SQLite stores reference features once per frame, calibration and video provenance,
stable defect IDs, original masks/skeletons, measurement payloads, and associations.
Feature records include original image dimensions; rebuild older reference databases
that lack this field. Use a separate database for each feature extractor.
It requires no database service. Frame writes are atomic; replaying a
video ID/frame index returns the saved associations. Use a new video ID when
reprocessing with changed settings or models. Existing JSONL/NPZ exports remain
independent of the temporal database.

GPS selects nearby reference frames; it is never treated as the crack position.
Unknown GPS keeps route candidates eligible. Each camera's intrinsics and distortion
are applied before comparing masks in reference-image coordinates. Comparisons are
restricted to the intersection of both matched-feature hulls. Existing defects may
match on partial coverage: at least 25% of each original mask and 64 foreground
pixels must be supported. Both masks and skeletons are clipped to this shared area
before comparison. New IDs still require full coverage of the current detection.
The inlier hull establishes geometric support, not an occlusion/vehicle detector.
Statuses are:

- `baseline`: a trackable frame establishes IDs in a neighborhood without references.
- `matched`: one unambiguous existing defect ID matches the observation.
- `new`: no overlapping defect was found and the current mask is fully supported.
- `unresolved`: alignment, coverage, or identity is insufficient or ambiguous.

`new` means a newly cataloged defect, not proof that damage appeared between surveys.
No detection does not imply repair. The initial overlap rule uses three reference
pixels of tolerance and a 0.6 symmetric overlap threshold; intermediate overlap and
competing IDs remain unresolved. These are heuristics requiring validation on real
pavement, not calibrated confidence estimates. This local implementation scans the
route's stored references and retains every processed frame; long-route performance,
reference selection, growth inference, and a review UI are outside this increment.

### Reproducible matcher evaluation

SuperPoint + LightGlue is the temporal engine default, implemented in
`src/engine/learned_features.py`. It filters keypoints to the road ROI, preserves
original pixel coordinates, and passes matches into the shared RANSAC and coverage
code. `uv sync` installs the pinned dependency. Rebuild ORB reference databases with
SuperPoint features before using them with the default matcher.

Run both against identical seeded samples and transformations, using fresh output
folders (the first learned run downloads pretrained weights):

```sh
python -m src.tests.benchmarks.temporal_dataset --count 100 --matcher orb --output reports/orb-evaluation
python -m src.tests.benchmarks.temporal_dataset --count 100 --matcher lightglue --device cpu --output reports/lightglue-evaluation
```

Each run saves the source manifest, a reference database, transformed images,
per-case JSON/JSONL, aggregate results, and visual examples. Match scores are
heuristics, not calibrated identity probabilities. The benchmark transforms full
ground-truth masks and does not test detector errors or genuine seasonal revisits.

## Modal GPU execution

`modal_app.py` wraps the existing recording loader and video pipeline in an L4
worker. SuperPoint and LightGlue load once per container; segmentation weights load
for each job. The worker has a 20-minute execution timeout, no automatic retries,
and one shared container processing one survey at a time. All submissions call the
deployed worker, including the local CLI, so different settings share one writer.
Each submission uses a new UUID job directory.

Temporal surveys require a stable `route_id`, camera calibration with `road_roi`,
and a unique video `id` in metadata (generated if omitted). Supply synchronized GPS
for candidate filtering; missing GPS retains all candidates on the route.
The worker extracts SuperPoint features once per sampled frame, runs LightGlue only
against GPS candidates, and exports a `temporal`
object on each damage row: `observation_id`, `defect_id`, `status`, and `score`.
No prior coverage establishes baseline IDs; failed registration stays unresolved.
Earlier frames in the current survey can preserve an existing damage ID, but only
previous surveys can establish that damage is new. Growth is not measured.

SQLite runs on a private local copy. Only successful surveys publish the closed
database back to the temporal volume. Failed or interrupted processing leaves the
previous database intact. Drain active jobs before redeploying; do not run another
writer against this volume. In-progress frames are visible only inside their own job.

From the engine directory, after authenticating Modal:

```sh
uv run python -m modal deploy modal_app.py
uv run python -m modal run --detach modal_app.py::main \
  --recording recordings/survey-1 \
  --weights src/ml/models/weights/segmentation/best.pt \
  --output-dir outputs \
  --sample-fps 2
```

The command validates the recording locally, uploads the selected checkpoint and
recording files, runs inference on `cuda:0`, and streams results back to
`outputs/<job-id>/`. It prints the job ID before uploading. `--detach` lets the
remote app continue if the local client disconnects. Input files are limited to
`video.*`, `metadata.json`, `gps.csv`, and `calibration.json`; datasets and unrelated
recording-directory files are not uploaded.

Persistent Modal volumes in the selected workspace:

- `road-survey-models`: checkpoints keyed by SHA-256, verified before loading.
- `road-survey-recordings`: input directories keyed by job ID.
- `road-survey-temporal`: `surveys.sqlite`, containing completed reference surveys.
- `road-survey-results`: `job.json` with status/GPU details and `output/` containing
  the pipeline manifest, damage JSONL, and compressed masks.

Volumes retain inputs and results until explicitly removed. Only the engine source
is included in the image; inference dependency versions match the current lockfile.
Update the image's version pins when upgrading those dependencies.

Optional flags are `--confidence`, `--triangulation`, and
`--motion-scale-source camera_height|gps`. Triangulation requires calibration with
a road ROI. The GPU worker persists failures during analysis; a failure before
model initialization completes may not create a result directory. Timeout or
container termination can leave a job marked running; this first version has no
background status reconciliation service. Inspect the Modal run in that case.

Retrieve completed or failed job artifacts without rerunning inference:

```sh
uv run python -m modal run modal_app.py::fetch --job-id YOUR_JOB_ID --output-dir outputs
```

Downloads are staged and then moved into place; existing destination directories
are never overwritten. Fetch refuses a job that is still marked running.

The integration passes 49 local tests, including failed-survey database isolation.
A deployed L4 temporal smoke test on 2026-09-10 processed two two-frame synthetic
surveys. All four observations shared one defect ID: baseline, matched, matched,
matched. Jobs: `b51123e8-d7e5-4424-8a1a-1105991bb84b` and
`8f224114-72c9-4ba0-8958-7847defa0831`. This verifies integration and persistence,
not seasonal robustness.

The command and transfer paths are covered by local tests with mocked Modal
services. A cloud smoke test completed on 2026-09-10 in the `lukepitstick06`
workspace using the custom segmentation checkpoint and an NVIDIA L4. The generated
two-second clip was sampled at 2 FPS; the job and pipeline manifests reported
`complete`, and the results downloaded successfully with zero detections. See
[the Modal run](https://modal.com/apps/lukepitstick06/main/ap-qbHEbFXQGRMyHTvGQNWdRv).
This verifies image build, upload, GPU execution, and result retrieval. Detection
quality, GPS alignment, and metric measurements still require real survey footage.

## Android capture decision and next steps

Use Android Camera2 for the Galaxy S25+ recorder. Camera2 availability is assumed;
usable intrinsic-calibration fields are checked at runtime for the active camera.
ARCore is not required for this capture path. The Python export importer is
implemented; the Android recorder is still planned work.

The importer accepts one recording directory:

```text
survey/
  video.mp4          # Exactly one video.* file; other supported extensions work.
  gps.csv            # Optional.
  metadata.json      # Optional without GPS.
  calibration.json   # Optional; must be complete when supplied.
```

- `gps.csv`: UTC timestamp, latitude, longitude, and optional horizontal accuracy
  in meters. Missing accuracy stays unknown.
- `metadata.json`: survey/video ID, optional route ID, video properties, UTC time
  of the first encoded frame, GPS synchronization offset, camera identity, and
  calibration availability/reason. Record the clock mapping used to synchronize
  camera frame timestamps and location samples; do not use the record-button
  press time as the first frame's timestamp.
- `calibration.json`: the fields accepted by `CameraCalibration`, including
  intrinsics expressed in the encoded video's coordinate system and the supplied
  mounting height, road-relative pitch/roll, and road ROI.

The recorder must account for sensor crop, scaling, orientation, distortion
correction, and zoom when deriving calibration. Raw Camera2 values must not be
copied into the OpenCV calibration without coordinate/convention conversion.
Start with one fixed lens, zoom, resolution, and controlled stabilization settings;
do not silently treat changing image geometry as one fixed calibration.
Mounting height and road-relative angle remain setup inputs, not Camera2 intrinsics.

If intrinsics or another required calibration input are unavailable, still export
the recording and GPS. The importer leaves `Video.calibration` unset, processes
pixel measurements, and disables metric reconstruction explicitly. Do not fabricate
intrinsics or create a partially valid calibration file.

### Importing a recording

```python
from pathlib import Path
from src.engine.recording_io import load_video
from src.engine.video_pipeline import VideoPipeline

video = load_video(Path("recordings/survey-1"))
pipeline = VideoPipeline(Path("src/ml/models/weights/segmentation/best.pt"))
pipeline.write(video, Path("outputs/survey-1"), sample_fps=2)
```

For the first footage test, a folder containing only `video.mp4` is enough. The
loader reads actual video properties, creates an ID, and leaves GPS/calibration
unset. It decodes only the first frame during import; subsequent decoding errors
are handled by the pipeline. No model is loaded by the importer.

`metadata.json` is a flat object, for example (illustrative times, not calibration):

```json
{
  "id": "survey-1",
  "route_id": "route-a",
  "recorded_at": "2026-09-08T18:00:00Z",
  "gps_offset_seconds": 0.0,
  "camera_id": "0",
  "calibration_status": "unavailable",
  "calibration_reason": "Intrinsics have not been captured"
}
```

Optional video properties are `width`, `height`, `fps`, and `duration_seconds`.
`clock_mapping` may contain the recorder's clock-mapping object; it is preserved
for inspection, not used to derive UTC automatically. The recorder must already
supply `recorded_at` and GPS timestamps in a common time system. The importer
normalizes timezone-aware ISO 8601 timestamps to UTC and retains the GPS offset.
Supplied width/height must match the decoded image; other video properties are
populated from the stream. Unknown metadata fields are rejected.

The exact CSV header is `timestamp,latitude,longitude,accuracy_m` (`accuracy_m`
is optional; empty cells mean unknown):

```csv
timestamp,latitude,longitude,accuracy_m
2026-09-08T18:00:00Z,40.0000,-105.0000,5.0
2026-09-08T18:00:01Z,40.0001,-105.0000,
```

GPS samples are sorted by time. Duplicate timestamps, invalid coordinates,
nonpositive reported accuracy, malformed rows, and missing timezones are errors.
A nonempty GPS track requires `recorded_at`. A missing GPS file or header-only
track is allowed. Invalid calibration is rejected rather than silently ignored;
its image dimensions must match actual footage. When supplied, `calibration_status`
must agree with whether `calibration.json` exists. Without calibration, use the
pipeline's default `triangulation=False`; explicit triangulation requires valid
calibration and a road ROI.

Next steps:

1. Run the first real phone clip through this importer and pipeline and inspect
   detections and mask alignment. GPS and metric evaluation can follow when captured.
2. Build a minimal Camera2 recorder: preview, start/stop, synchronized GPS and
   camera metadata collection, mounting setup, and directory export.
3. Validate GPS timing and physical dimensions against measured pavement features.
4. Wrap the validated pipeline in a Modal GPU worker with persistent weights and
   result artifacts; add asynchronous API submission and React afterward.

## Image models and custom post-processing

### 1. Preprocessing

Images are enhanced before inference to handle outdoor lighting, noise, and varying resolution:

- **Resize** to the model input size
- **CLAHE** (contrast-limited adaptive histogram equalization) to bring out surface detail
- **Denoising** to reduce compression and sensor noise

This pipeline lives in `src/engine/preprocessing.py` and is shared between training and inference.

### 2. Detection model (YOLO26s)

The detection model finds **bounding boxes** for road defects. It is trained on the [RDD2022](https://github.com/seunghoonpark/RDD2022)-style dataset with two classes:

| Class ID | Label   |
|----------|---------|
| 0        | crack   |
| 1        | pothole |

Training config: `src/ml/configs/train.yaml` → `detection`  
Training script: `src/ml/scripts/detection/train_model.py`  
Dataset: `datasets/detection/RD2022/`

The `DetectionEngine` in `src/engine/detection.py` runs: **preprocess → infer → post-process**.

### 3. Segmentation model (YOLO26s-seg)

The segmentation model produces **pixel-level crack masks**, which are used for finer measurements (length, width, orientation, branching) that bounding boxes alone cannot capture.

Training config: `src/ml/configs/train.yaml` → `segmentation`  
Training script: `src/ml/scripts/segmentation/train_model.py`  
Dataset: `datasets/segmentation/crack_segmentation_dataset/`

Mask annotations are generated from binary masks via `src/ml/scripts/segmentation/convert_images_masks.py`.

The `SegmentationEngine` in `src/engine/segmentation.py` returns measured `Damage` records for each detected region.

### 4. Domain model

Raw model outputs are mapped into structured `Damage` objects defined in `src/engine/models.py`:

- **Type** — crack or pothole
- **Subtype** — e.g. longitudinal / transverse / alligator crack, or pothole size
- **Dimensions** — width and length with unit conversion (cm / inch)
- **Confidence** — model score
- **Stress range** — road class (residential → freeway), used by the fatigue model
- **Severity** — currently a placeholder value; severity scoring is still being built

Constants and enums live in `src/engine/constants.py`.

### 5. Physical simulation (Paris' Law)

Paris' Law models **fatigue crack growth** under repeated stress:

```
da/dN = C · (ΔK)^m
```

Where crack length grows per load cycle as a function of stress intensity. The engine is designed to use this to estimate how an observed defect may worsen over time given the road's traffic/stress category (`StressRange` in `constants.py`).

> **Note:** The top-level `Engine` class wires detection and segmentation together. Paris' Law integration, severity scoring, backend persistence, and full prediction output are still being built out.

## Pipeline Results

The screenshots below show intermediate outputs from the segmentation post-processing pipeline. They focus on the crack-analysis stage after a detected damage region has been segmented.

### Skeletonized mask

![Skeletonized crack mask](screenshots/skeletonized.png)

The segmentation mask is reduced to a one-pixel-wide skeleton with `skimage.morphology.skeletonize`. This preserves the crack's centerline while stripping away mask thickness, which gives the rest of the pipeline a stable graph-like shape to analyze. The engine then uses convolution-based neighbor counts to find endpoints and junction pixels directly on the skeleton.

### Split branches

![Skeletonized branches split into separate masks](screenshots/skeletonized_branches.png)

After skeletonization, the engine separates the crack network into individual branches. Junction pixels are detected as skeleton pixels with three or more 8-connected neighbors, then a small dilated junction zone is removed so connected-component labeling can isolate each branch segment. This makes each branch easier to measure and classify independently before nearby fragments are merged back together.

### Colored branch labels

![Colored crack branches](screenshots/output.png)

Each detected branch is assigned a separate color after the branch-culling pass. The culling algorithm removes tiny skeleton components below the configured minimum branch length, which filters out specks, jagged junction leftovers, and short mask artifacts that should not become reportable crack branches. This view is useful for checking whether the culling threshold is too aggressive, too permissive, or correctly preserving meaningful crack geometry.

### Branch angle estimates

![Branch angle estimates](screenshots/anglecalcs.png)

The red guide lines show each branch's estimated axis angle. For whole-branch orientation, the engine uses the farthest pair of skeleton endpoints as the branch axis; for endpoint-level merge checks, it estimates local tangent direction with PCA over nearby skeleton pixels. Angles are treated as undirected axes in the `[0, 180)` range so a line pointing left-to-right and right-to-left is considered the same crack direction. These angles help classify cracks as longitudinal or transverse based on their orientation in the image.

### Combined branch results

![Combined branch results](screenshots/branchescomplete.png)

Nearby and similarly aligned branches can be merged into larger crack paths. The engine compares the closest endpoint pair between candidate branches, rejects pairs that are too far apart, then rejects pairs whose local endpoint angles differ too much. Among the remaining candidates, it greedily merges the lowest-score pair, draws a bridge between endpoints, recomputes the skeleton/endpoints, and repeats until no eligible merge remains. The numbered labels show the final branch groups that are passed into measurement and subtype classification.

## Project layout

```
src/
├── engine/          # Inference pipeline, preprocessing, domain types
│   ├── base.py
│   ├── config.yaml
│   ├── constants.py
│   ├── detection.py
│   ├── engine.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── utils.py
└── ml/
    ├── configs/     # Shared train.yaml for both models
    ├── scripts/
    │   ├── detection/
    │   └── segmentation/
    ├── models/      # YOLO architecture configs and weights
    └── runs/        # Training outputs (Ultralytics + W&B)

datasets/
├── detection/RD2022/
└── segmentation/crack_segmentation_dataset/
```

## Training With Config

The training scripts are configured from `src/ml/configs/train.yaml`. Edit that
file first, then run the detection or segmentation trainer from the project
root.

Install dependencies and sync the environment:

```bash
uv sync
```

The shared config has three top-level sections:

- `project`: shared project metadata and the weights directory.
- `detection`: dataset, output directory, base model, image size, epochs, batch size, workers, and augmentation settings for bounding-box training.
- `segmentation`: dataset, output directory, base model, image size, epochs, batch size, workers, and checkpoint cadence for mask training.

Current defaults:

```yaml
project:
  name: road-damage
  weights_dir: src/ml/models/weights

detection:
  dataset: datasets/detection/RD2022/converted.yaml
  runs_dir: src/ml/runs/detection
  experiment_name: yolo26s-rd2022-with-preprocessing
  model: yolo26s.pt
  imgsz: 960
  epochs: 100
  batch: 16
  workers: 8

segmentation:
  dataset: datasets/segmentation/crack_segmentation_dataset/yolo.yaml
  runs_dir: src/ml/runs/segmentation
  experiment_name: yolo26s-crack-segmentation
  model: yolo26s-seg.pt
  imgsz: 448
  epochs: 100
  batch: 16
  workers: 8
  save_period: 10
```

Before training, make sure the configured dataset YAML exists and that the
configured base weights are available under `project.weights_dir`. For example,
the default detection config expects:

```text
datasets/detection/RD2022/converted.yaml
src/ml/models/weights/yolo26s.pt
```

The default segmentation config expects:

```text
datasets/segmentation/crack_segmentation_dataset/yolo.yaml
src/ml/models/weights/yolo26s-seg.pt
```

Train the detection model:

```bash
uv run python src/ml/scripts/detection/train_model.py
```

Train the segmentation model:

```bash
uv run python src/ml/scripts/segmentation/train_model.py
```

Both scripts automatically select the best available device in this order:
CUDA, Apple MPS, then CPU. They also log runs to Weights & Biases using
`project.name` as the W&B project and the model section's `experiment_name` as
the W&B run name. Training outputs are written to the configured `runs_dir`.

To change a run, edit `src/ml/configs/train.yaml` rather than passing command
line flags. Common edits are:

- Change `epochs`, `batch`, `imgsz`, or `workers` for compute budget.
- Change `experiment_name` to keep runs separate.
- Change `dataset` to point at a different YOLO dataset YAML.
- Change `model` to use a different weights file in `src/ml/models/weights`.
- Tune `detection.augment` values for detection-specific augmentation.

## Models

| Model            | Task          | Base weights     | Classes / output   |
|------------------|---------------|------------------|--------------------|
| YOLO26s          | Detection     | `yolo26s.pt`     | crack, pothole     |
| YOLO26s-seg      | Segmentation  | `yolo26s-seg.pt` | crack pixel masks  |

Planned simulation: **Paris' Law** (fatigue crack propagation).
