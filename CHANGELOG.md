## 1.1.1

- Cache yolo input buffer and pass flat tensor to avoid reshape overhead
- Update `tflite_flutter_custom` to 1.1.0
- 100% test coverage

## 1.1.0

- Parallel landmark detection in multi person photos, faster processing
- Improve error handling
- Add integration tests

## 1.0.1

- Improved Dartdoc coverage

## 1.0.0

**This version contains breaking changes:**

- Breaking API changes: `PoseDetector` uses constructor params (no `PoseOptions`), returns `Pose`/`BoundingBox`, and detection methods are async
- YOLO and BlazePose run via isolates to avoid blocking
- Expanded DartDoc coverage
- `poseLandmarkConnections` for drawing skeletons.
- Example app improvements, sample images added.

## 0.0.3

* Removed unused dependency `path_provider`

## 0.0.2

* Improved documentation

## 0.0.1

* Initial release
* Person detection using YOLOv8
* Pose landmark detection with MediaPipe Pose
* Support for lite, full, and heavy models
* Box-only and full landmark detection modes
