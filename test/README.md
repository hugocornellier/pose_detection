# Testing Guide

This package has two test layers: fast host tests in `test/`, and model-backed
integration tests in `example/integration_test/`.

## Host Tests

The root `test/` directory currently contains:

- `types_test.dart`: 60 tests for public value types, enums, helpers, and
  skeleton topology.
- `additional_coverage_test.dart`: 3 tests for registration and test-visible
  model helper behavior.

Run them from the package root:

```bash
flutter test
```

Or run the files explicitly:

```bash
flutter test test/types_test.dart test/additional_coverage_test.dart
```

These tests are intentionally limited to APIs that can run in the Flutter test
environment without loading the pose models for inference.

## Integration Tests

Full inference coverage lives under the example app:

- `example/integration_test/pose_detector_integration_test.dart`: 38 tests for
  initialization, disposal, error handling, real image detection, model
  variants, boxes-only mode, `detectFromMat`, `detectFromMatBytes`, result
  consistency, and invalid image bytes.
- `example/integration_test/pose_detector_benchmark_test.dart`: native
  benchmark harness.
- `example/integration_test/pose_detector_web_benchmark_test.dart`: web
  benchmark harness.

Run integration tests from `example/` on a supported device or desktop target:

```bash
cd example
flutter test integration_test/
```

For a specific connected device:

```bash
cd example
flutter devices
flutter test integration_test/ --device-id=<device-id>
```

For a focused native integration test:

```bash
cd example
flutter test integration_test/pose_detector_integration_test.dart
```

## Native Library Notes

The root host tests do not require TensorFlow Lite inference. Tests that
initialize `PoseDetector` or call inference APIs load native TFLite through
`flutter_litert` and should be run as integration tests in `example/`.

If you are deliberately running a host-side native inference test and need to
point `flutter_litert` at a local TensorFlow Lite C library, use
`TFLITE_LIB_PATH`:

```bash
TFLITE_LIB_PATH=/path/to/libtensorflowlite_c-mac.dylib flutter test <test-file>
```

## Sample Images

The integration tests use the example app's sample images in
`example/assets/samples/pose1.jpg` through `pose7.jpg`. Keep `example/pubspec.yaml`
asset entries in sync if images are added, renamed, or moved.

## Adding Tests

When adding inference tests, prefer `example/integration_test/` and clean up
detectors with `await detector.dispose()`. Use root `test/` for pure Dart or
Flutter-test-safe API behavior only.
