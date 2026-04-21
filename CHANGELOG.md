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
