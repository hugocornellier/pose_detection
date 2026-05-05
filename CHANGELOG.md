## 3.1.0

**Web:** LiteRT.js (WebGPU + WASM fallback) is now the default web runtime. 
  * `useLiteRt` now defaults to `true` so no opt-in is required. Pass `useLiteRt: false` to use the legacy tflite-js path. `liteRtAccelerator` (`String`, default `'auto'`) controls the backend: `'auto'` prefers WebGPU with automatic WASM fallback, `'webgpu'` / `'wasm'` force a specific backend. On browsers without WebGPU support, `'auto'` falls back to WASM transparently. Both runtimes load from CDN on first use with no `web/index.html` changes required; see `configureLiteRtLoader` in `flutter_litert` for self-hosting options.
* Update flutter_litert -> 2.5.2

## 3.0.2

* Update flutter_litert -> 2.5.0

## 3.0.1

* Update flutter_litert -> 2.4.1

## 3.0.0

**Breaking:**
* `PoseDetector` configuration moves from the constructor to `initialize()`. `PoseDetector({mode: ..., landmarkModel: ..., ...})` → `PoseDetector()` + `await detector.initialize(mode: ..., landmarkModel: ..., ...)`. Matches `FaceDetector`'s shape. `PoseDetector.create({...})` continues to accept the same named params unchanged.
* `PoseDetector.detectFromMat` signature changed. `detectFromMat(cv.Mat, {required int imageWidth, required int imageHeight})` is now `detectFromMat(cv.Mat)`. Dimensions are read from the Mat directly. Existing callers must drop the `imageWidth` and `imageHeight` named arguments.
* `detect(...)` no longer swallows exceptions. Undecodable image bytes now propagate as an error (matching `FaceDetector` and `HandDetector` behaviour) rather than silently returning an empty list. Wrap `detect(...)` in a `try/catch` if your callsite depended on the previous silent-failure behaviour. On web, decode failure still returns an empty list because browser HTMLImageElement decode does not throw.

* All inference now runs in a dedicated background isolate, keeping the UI thread free. Previously, inference ran on the calling thread.
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
