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
