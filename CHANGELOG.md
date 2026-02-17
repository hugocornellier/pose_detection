## 1.3.2

- Windows: remove bundled .dll files, as they are no longer needed as of `flutter_litert` 0.1.4

## 1.3.1

- Swift Package Manager support

## 1.3.0

- Deprecate all `img.Image`-based APIs in favor of `cv.Mat` alternatives
- `PoseDetector.detect()` and `detectOnImage()` → use `detectOnMat()`
- `PoseLandmarkModelRunner.run()` → use `runOnMat()`
- `YoloV8PersonDetector.detectOnImage()` → use `detectOnMat()`
- `ImageUtils.letterbox()`, `letterbox256()`, `imageToNHWC4D()` → use `NativeImageUtils` equivalents
- `NativeImageUtils.imageToMat()` → use `cv.imdecode` or pass `cv.Mat` directly
- The `image` package dependency will be removed in 2.0.0

## 1.2.2

- Fix Android build: bump tflite_flutter_custom to 1.2.5 (fixes undefined symbol TfLiteIntArrayCreate linker error)
- Add missing MainActivity.kt for Android example app

## 1.2.1

- Improved auto .dylib bundling on MacOS 

## 1.2.0

- Add native OpenCV preprocessing for 5-15x faster image processing (SIMD-accelerated)
- New `useNativePreprocessing` parameter on `PoseDetector` (default: true)
- New `detectOnMat()` method on `YoloV8PersonDetector` for direct cv.Mat input
- New `runOnMat()` method on `PoseLandmarkModelRunner` for direct cv.Mat input
- Add `opencv_dart` dependency for native image operations
- Refactored detection code with shared post-processing helpers

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
