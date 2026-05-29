# Video2Splat Input-Type Technical Routes

Date: 2026-05-19

## Core Conclusion

There should not be one universal reconstruction route for every input type.

The product should first classify the input, then dispatch to the matching route:

1. Image sequence route.
2. Continuous video route.
3. Panorama route.
4. Drone route.

The Huahuati experiment shows the main failure mode clearly: forcing a long continuous handheld video into one fused PLY creates global drift and noisy Gaussian overlap. For long continuous capture, the best result is a segmented scene graph, not a single monolithic splat.

## Code Organization Target

Production code should be separated from experiment code.

Recommended structure:

```text
apps/api/video2splat/
  input_router.py          # classify source type and select route
  image_sequence.py        # normal photo sequence preparation
  video_sequence.py        # continuous video candidate extraction and segmentation
  panorama.py              # equirectangular/cube/tangent crop preparation
  drone.py                 # drone frame extraction, GPS/EXIF, tiling
  quality.py               # shared quality gates
  run_registry.py          # run/model manifest registry

scripts/
  video2splat_prepare_inputs.py
  video2splat_run_image_sequence_job.py
  video2splat_run_video_job.py
  video2splat_run_panorama_job.py
  video2splat_run_drone_job.py

experiments/
  huahuati_20260519/
  instantsplatpp_comparison/
```

Current Huahuati-specific scripts should eventually move under `experiments/huahuati_20260519/` after the production route modules exist:

- `scripts/huahuati_continuous_reconstruction.py`
- `scripts/huahuati_fuse_segments.py`

## Asset Organization Target

Raw inputs should be immutable. Every generated stage should have a manifest.

Recommended structure:

```text
F:\video2splat\datasets\<dataset_id>\
  raw\
    images\
    videos\
    panoramas\
    drone\
  prepared\
    candidates\
    segments\
    keyframes\
    masks\
    poses\
  notes\

F:\video2splat\runs\<run_id>\
  00_source\
  01_candidates\
  02_segments\
  03_keyframes\
  04_models\
  05_spark\
  reports\
  logs\
  scripts\

F:\video2splat\models\<model_id>\
  splat.ply
  viewer-camera.json
  manifest.json
```

## Route 1: Normal Image Sequence

Best for:

- A set of ordinary photos around one object or one scene.
- Photos with enough parallax and overlap.
- Small to medium scene scale.

Preferred route:

1. Normalize EXIF orientation and image size.
2. Quality filter: blur, exposure, duplicates, tiny baseline removal.
3. Prefer COLMAP/SfM when overlap and texture are sufficient.
4. Use MASt3R/DUSt3R/VGGT as fallback when COLMAP is weak or unordered sparse photos need learned geometry priors.
5. Train 3DGS with the best available poses.
6. Publish one model when the scene is compact; publish chunks when the scene is large.

Best output:

- Usually one Spark-loadable `splat.ply`.

Avoid:

- Feeding too many near-duplicate images.
- Treating weak-baseline photos as better just because there are more of them.

## Route 2: Continuous Video

Best for:

- Phone walkthroughs.
- Handheld现场视频.
- Long continuous capture with turns, occlusion, changing scale, and motion blur.

Preferred route:

1. Decode to a candidate pool, not directly to training frames.
2. Use quality scores and temporal spacing to remove blurred and redundant frames.
3. Split by temporal continuity and motion discontinuity.
4. Build local MASt3R/InstantSplat++ models per segment.
5. Generate a global low-quality navigation model only as an overview.
6. Connect segments through a scene graph: time order, bridge transforms, and viewer transitions.
7. In Spark, load/switch segments instead of forcing all splats into one PLY.

Best output:

- Multi-model segmented scene.
- Optional weak global overview.

Huahuati final decision:

- Best visual result: 21 MASt3R segment models.
- Best single-file candidate: strict pruned fused PLY, but not good enough as final quality.

Avoid:

- One-pass global reconstruction for a several-minute handheld video.
- PLY-level fusion as the final quality path. It cannot repair global pose drift.

## Route 3: Panorama Photos

Best for:

- 360 equirectangular panoramas.
- Indoor capture points.
- Multiple panorama stations.

Preferred route:

1. Keep the source panorama as the canonical raw input.
2. Convert each panorama to cube or tangent perspective crops.
3. Preserve rig constraints: crops from one panorama share the same camera center and have known relative rotations.
4. Match across panorama stations, not only within one panorama.
5. Solve poses with panorama-aware constraints where possible.
6. Train 3DGS from constrained perspective crops.
7. For a single panorama, treat the result as view synthesis/depth approximation, not true complete 3D reconstruction.

Best output:

- Multiple panorama stations: one constrained 3DGS scene.
- Single panorama: spherical viewer or approximate depth/splat scene with clear quality warning.

Avoid:

- Cropping a panorama into normal images and pretending each crop is an independent photo.
- Expecting a single panorama to recover occluded geometry behind the camera center.

## Route 4: Drone

Best for:

- Outdoor aerial capture.
- Building/site/terrain scans.
- Nadir and oblique image sets.
- Inputs with GPS/RTK/IMU metadata.

Preferred route:

1. Extract frames or ingest photos with EXIF/GPS intact.
2. Separate nadir, oblique, and orbit passes.
3. Use GPS/IMU as pose priors when available.
4. Prefer photogrammetry/SfM first: COLMAP, OpenDroneMap, Metashape-like route.
5. Train 3DGS from stable camera poses.
6. For large sites, tile the scene spatially and publish LOD/chunks.
7. Mask sky, propellers, people, vehicles, water glare, and moving objects.

Best output:

- Georeferenced tiled splat scene.
- Optional mesh/point cloud side products for measurement workflows.

Avoid:

- Using handheld-video segmentation logic directly on drone data.
- Ignoring GPS/IMU priors.
- Building one giant splat for a large site when tiled LOD is needed.

## Decision Matrix

| Input | Best route | Final form | Main risk |
| --- | --- | --- | --- |
| Normal photos | SfM/COLMAP first, learned prior fallback | Usually one PLY | Weak overlap / low texture |
| Long handheld video | Candidate pool -> temporal segments -> local MASt3R models | Multi-model scene graph | Global drift |
| Panorama | Pano-to-crops with rig constraints | Constrained scene or spherical/depth viewer | False parallax from bad crop handling |
| Drone | GPS/IMU/SfM -> tiled 3DGS | Georeferenced tiled scene | Scale and LOD |

## Product-Level Best Route

Build an input router and route-specific pipelines.

Default routing:

```text
if input is ordinary image set:
  image sequence route
elif input is long handheld video:
  continuous video segmented route
elif input is panorama:
  panorama constrained-crop route
elif input is drone:
  drone photogrammetry route
else:
  run quality diagnosis and request/manual classify
```

The viewer should support both single-model and multi-model scenes. For long video and large drone scans, multi-model loading is a product requirement, not an optimization.
