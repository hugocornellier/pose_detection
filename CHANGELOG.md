## 3.6.0

* iOS now runs the YOLO person detector on the Metal GPU on the default
  CompiledModel path, cutting live-camera latency roughly 4x on device
  (measured on iPhone: ~100ms to ~22ms per frame, with detection keeping
  pace with the 24 fps camera stream). The CPU pin dated from v2.0.6, when
  the old TFLite Metal GPU delegate produced inconsistent detection counts
  (10 vs 2 on the same image); the LiteRT Next Metal accelerator used by
  the CompiledModel path shows stable counts at fp16 on device. The
  interpreter path keeps its XNNPACK override on iOS because it still uses
  the old delegate.
* CompiledModel GPU-to-CPU fallbacks are no longer silent: the YOLO and
  landmark models log the fallback error and the compiled accelerator set
  in debug builds (`[pose-accel]` lines).
* Fix (example): `PoseSmoother` rebuilt each `Pose` to write filtered landmark
  coordinates but omitted `segmentationMask` from the constructor call, so
  enabling smoothing silently dropped masks. The example app never enables
  segmentation so nothing surfaced it, but the smoother reads as
  general-purpose and example code exists to be copied. Adds `example/test`
  with a regression test that fails without the fix, plus coverage for the
  null-mask case, the unfiltered scalar fields, and the disabled no-op path.
* Deprecate `CameraPoseOverlayPainter` in favour of `CameraPoseOverlay`. The
  painter scales x and y independently, so it only maps correctly when the
  paint box already matches the detection image's aspect ratio; over a
  cover-fitted preview it stretches the overlay. The widget performs the same
  job through a cover fit. Deprecated rather than changed, since altering its
  mapping would move overlays for downstream users who sized their box to match.
* Adopt the shared `flutter_litert` 3.6.0 helpers in place of local copies:
  `aggregateActiveAccelerator` for the multi-runner accelerator aggregation
  (this package already had the correct any-WebGPU behaviour, so this is
  de-duplication rather than a fix), `compiledModelFromBufferAuto` for the
  `{gpu, cpu}` branch at both CompiledModel call sites, and `iouLTRB` for
  track matching.
* Remove the `web_image_utils` re-export shim and import from `flutter_litert`
  directly.
* Update flutter_litert -> 3.6.0.
* Expand the README live camera section with the full production pipeline
  (frame throttling, orientation handling, cover-fit overlay mapping).

## 3.5.3

* Fix landmark confidence decoding for the lite, full, and heavy BlazePose
  models. Their overall score tensor is already produced by a TFLite
  `LOGISTIC` op, so it is now used directly instead of passing through a
  second sigmoid. The default `minLandmarkScore: 0.5` gate can once again
  reject blank or low-confidence crops.
* Return the bundled YOLOv8 person's actual class probability as `Pose.score`.
  The detector now uses flutter_litert's explicit probability mode rather than
  compensating for its legacy logit decoder by transforming the threshold.
  Filtering is unchanged, while reported confidence is calibrated correctly.
* Bump the pose pipeline cache key because identical inputs can now produce
  different landmark inclusion and confidence output.
* Update flutter_litert -> 3.5.1.

## 3.5.2

* Update flutter_litert -> 3.5.0

## 3.5.1

* Web: the `auto` accelerator now resolves through flutter_litert's capability probe (`resolveWebAccelerator`): WebGPU is selected only on Chromium with a real hardware adapter, and everything else starts on WASM. Fixes Firefox 152, whose WebGPU compiles and runs cleanly but far slower than WASM SIMD, so the error-driven fallback could never catch it.
* Web: after an `auto` init lands on WebGPU, a timed warmup on the YOLO stage (`WebGpuFallback.maybeSwapIfWebGpuSlow`) swaps all runners to WASM when the median run exceeds the budget.
* Web: Safari initializes again; flutter_litert now serves LiteRT.js from its wasm directory default, so Safari receives the compat build instead of failing to parse the relaxed-SIMD build (`relaxed simd instructions not supported`).
* Update flutter_litert -> 3.4.1 (also brings the web `CompiledModel` WebGPU compile watchdog: a compile attempt that never settles falls back to WASM instead of hanging).
* example_web: mirror backend + fps into the page title and add `?screen=camera|video|still` deep links for automated validation.

## 3.5.0

* Add opt-in person **segmentation masks**. Initialize with `enableSegmentation: true` (under `PoseMode.boxesAndLandmarks`) to receive a coarse BlazePose person mask per detected pose via `Pose.segmentationMask`. The landmark model already computes this mask on every inference, so returning it adds no inference cost. New `SegmentationMask` type exposes `confidenceAt(x, y)` sampling and `toRgbaBytes()` for rendering. Off by default; populated on native platforms only (web accepts the flag for API parity but leaves the mask `null`).
* Fix the YOLOv8 person detector emitting phantom detections. The bundled model outputs class probabilities, not raw logits, but the shared decoder applies a sigmoid to every score, so background anchors near 0 cleared the confidence gate as `sigmoid(0) = 0.5`. That produced thousands of low-confidence boxes that NMS then collapsed into a random one-to-four detections. The detector now cancels the extra activation so the threshold compares against the model's actual probability.

## 3.4.0

* Update flutter_litert -> 3.2.0
* Import native-only flutter_litert APIs via `package:flutter_litert/native.dart` so they resolve under static analysis (flutter_litert 3.2.0 moved `InterpreterFactory`, `InterpreterPool`, and `IsolateWorkerBase` behind the native conditional export). No runtime or API change.

## 3.3.0

* Update flutter_litert -> 3.1.1
* Add optional LiteRT Next `CompiledModel` inference (GPU with automatic CPU fallback); enable with `useCompiledModel: true`. The default engine remains the Interpreter, so existing code is unchanged.
* Add a flat YOLO decoder and adopt the shared flutter_litert helpers (flat YOLO decode, GPU fallback, camera frame decode plan).

## 3.2.1

* Update flutter_litert -> 2.8.3

## 3.2.0

* Update flutter_litert -> 2.8.0
* Complete Swift Package Manager migration: example apps build via SPM without CocoaPods

## 3.1.5

* Remove unused Darwin podspecs for Dart-only iOS/macOS plugin registration.

## 3.1.4

* Update flutter_litert -> 2.5.8

## 3.1.3

* Update flutter_litert -> 2.5.5

## 3.1.2

* Update flutter_litert to 2.5.3 and camera_desktop to 1.1.4

## 3.1.1

* Update documentation and dartdocs

## 3.1.0

**Web:** LiteRT.js (WebGPU + WASM fallback) is now the default web runtime.
  * `useLiteRt` now defaults to `true` so no opt-in is required. Pass `useLiteRt: false` to use the legacy tflite-js path. `liteRtAccelerator` (`String`, default `'auto'`) controls the backend: `'auto'` and `'webgpu'` request WebGPU with WASM fallback when WebGPU compile fails, while `'wasm'` opts out of GPU. On browsers without WebGPU support, `'auto'` falls back to WASM transparently. Both runtimes load from CDN on first use with no `web/index.html` changes required; see `configureLiteRtLoader` in `flutter_litert` for self-hosting options.
* Update flutter_litert -> 2.5.2

## 3.0.2

* Update flutter_litert -> 2.5.0

## 3.0.1

* Update flutter_litert -> 2.4.1

## 3.0.0

**Breaking:**
* `PoseDetector` configuration moves from the constructor to `initialize()`. `PoseDetector({mode: ..., landmarkModel: ..., ...})` → `PoseDetector()` + `await detector.initialize(mode: ..., landmarkModel: ..., ...)`. Matches `FaceDetector`'s shape. `PoseDetector.create({...})` continues to accept the same named params unchanged.
* `PoseDetector.detectFromMat` signature changed. `detectFromMat(cv.Mat, {required int imageWidth, required int imageHeight})` is now `detectFromMat(cv.Mat)`. Dimensions are read from the Mat directly. Existing callers must drop the `imageWidth` and `imageHeight` named arguments.
* `detect(...)` no longer swallows exceptions. Undecodable native image bytes now throw `FormatException` (matching `FaceDetector` and `HandDetector` behaviour) rather than silently returning an empty list. Wrap `detect(...)` in a `try/catch` if your callsite depended on the previous silent-failure behaviour. On web, decode failure still returns an empty list because browser image decode failure does not throw through this API.

* On native platforms, inference now runs in a dedicated background isolate, keeping the UI thread free. Previously, native inference ran on the calling thread.
* Add `PoseDetector.create({...})` one-step factory (mirrors `FaceDetector.create` and `HandDetector.create`).
* Add `detectFromFilepath(String path)`: reads the file and delegates to `detect`.
* Add `detectFromMatBytes(Uint8List, {required int width, required int height, int matType})` zero-copy fast path via `TransferableTypedData`.
* Add `initializeFromBuffers({required Uint8List yoloBytes, required Uint8List landmarkBytes})` for callers that load model bytes independently of Flutter's asset system.
* Add `isReady` getter as an alias for `isInitialized`.
* Add a top-level Live Camera Detection section to the README, modelled on `face_detection_tflite`'s `packYuv420` + native `cv.cvtColor` pattern, and remove the orphan `assets/models/pose_detection.tflite` left over from the pre-YOLOv8n scaffold.
* Expand `flutter_litert` re-exports through the `pose_detection` barrel to match `face_detection_tflite`: tensor helpers (`createNHWCTensor4D`, `fillNHWC4D`, `allocTensorShape`, `flattenDynamicTensor`), math helpers (`sigmoid`, `sigmoidClipped`, `clamp01`, `clip`), letterbox helpers (`computeLetterboxParams`, `LetterboxParams`), BGR→RGB byte helpers (`bgrBytesToRgbFloat32`, `bgrBytesToSignedFloat32`), and `PerformanceMode`. Consumers no longer need a direct `flutter_litert` import for these.

## 2.1.1

* Add public `PoseDetector.modelVersion` and `PoseDetector.modelVersionFor(...)` APIs for downstream cache invalidation.

## 2.1.0

* Fix live camera in the example app on Android (previously detections were sideways and unreliable):
  * Apply rotation to raw landscape camera frames before detection so the pose detector sees upright people. `_rotationFlagForFrame` handles all four device orientations (portrait up/down, landscape left/right) via a combined `sensorOrientation` + `DeviceOrientation` formula.
  * Mirror the overlay on Android front camera to match `CameraPreview`'s auto-mirrored preview texture.
  * Replace the per-pixel Dart YUV loop with `flutter_litert`'s shared `packYuv420` helper + native `cv.cvtColor` on mobile (iOS NV12, Android NV21 / I420).
  * Replace the per-pixel Dart BGRA→BGR / RGBA→BGR loop with native `cv.cvtColor` on desktop (macOS / Linux).
* Align example app live-camera layout with `face_detection_tflite`: Material+Row top bar (replaces AppBar), flip-camera button, FPS + detection-time display, rotating top bar in landscape with safe-area padding, and a settings popup housing pose-specific controls (landmark-model chips: Lite / Full / Heavy).
* Re-export `packYuv420`, `YuvPlane`, `YuvLayout`, and `PackedYuv` from `flutter_litert` through the `pose_detection` barrel.
* Update `flutter_litert` to `^2.2.0`.

## 2.0.10

* Update flutter_litert -> 2.1.0

## 2.0.9

* Update flutter_litert to 2.0.13

## 2.0.8

* Update flutter_litert -> 2.0.12

## 2.0.7

* Update flutter_litert 2.0.10 -> 2.0.11

## 2.0.6

* Fixed Metal GPU delegate producing inconsistent detection counts on iOS

## 2.0.5

* Update documentation

## 2.0.4

* Update flutter_litert 2.0.8 -> 2.0.10

## 2.0.3

* Enable auto hardware acceleration by default (XNNPACK on all native platforms, Metal GPU on iOS)
* Update flutter_litert 2.0.6 -> 2.0.8

## 2.0.2

* Update flutter_litert 2.0.5 -> 2.0.6 

## 2.0.1

* Fix Xcode build warnings by declaring PrivacyInfo.xcprivacy as a resource bundle in iOS and macOS podspecs

## 2.0.0

**Breaking:** `Point` now uses `double` coordinates. `BoundingBox` is now a 4-corner Point-based type.

* Use shared `Point` and `BoundingBox` from `flutter_litert` 2.0.0
* `toPixel()` now returns full-precision `double` coordinates (was truncating to `int`)
* Extract `PersonDetectorBase` shared between native and web detectors
* Simplify model classes and detector implementations
* Remove integration tests from unit test suite
* Remove dead test helpers (`test_config.dart`)

## 1.0.7

* Update `camera_desktop` 1.0.1 -> 1.0.3

## 1.0.6

* Update `flutter_litert` -> 1.2.0
* Refactor to use `flutter_litert` shared utilities (`InterpreterFactory`, `InterpreterPool`, `PerformanceConfig`)

## 1.0.5

* Update `opencv_dart` 2.1.0 -> 2.2.1
* Update `flutter_litert` 1.0.2 -> 1.0.3

## 1.0.4

* Update `flutter_litert` 1.0.1 -> 1.0.2

## 1.0.3

* Update documentation

## 1.0.2

* Update `flutter_litert` to 1.0.1, `camera` to 0.12.0

## 1.0.1

* Update `flutter_litert` to 0.2.2

## 1.0.0

* Initial release
* Person detection using YOLOv8
* Pose landmark detection with MediaPipe Pose
* Support for lite, full, and heavy models
* Box-only and full landmark detection modes
