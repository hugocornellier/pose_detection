# Testing Guide

This directory contains tests for the `pose_detection_tflite` package.

## Test Structure

### `/test` - Unit Tests (Limited)
Contains basic unit tests that can run in a pure Dart VM environment:
- ✅ Error handling (StateError when not initialized)
- ✅ API structure validation

**Note:** Most functionality requires TensorFlow Lite native libraries, so comprehensive testing must be done via integration tests.

### `/integration_test` - Integration Tests (Comprehensive)
Contains full integration tests that run in an actual app environment with TFLite support:
- ✅ Initialization and disposal
- ✅ Error handling
- ✅ Detection with real sample images (pose1-7.jpg)
- ✅ `detect()` and `detectOnImage()` methods
- ✅ Different model variants (lite, full, heavy)
- ✅ Different modes (boxes, boxesAndLandmarks)
- ✅ Landmark and bounding box access (all 33 BlazePose landmarks)
- ✅ Configuration parameters
- ✅ Edge cases

**Total Test Cases: 30**

## Running Tests

### ⚠️ Important: TensorFlow Lite Requirement

The standard `flutter test` command runs tests in a Dart VM which **does not** have access to native libraries. Since this package uses TensorFlow Lite (a native library), most tests will fail with:

```
Failed to load dynamic library 'libtensorflowlite_c-mac.dylib'
```

**This is expected!** You must run integration tests instead.

### Integration Tests (Recommended)

Integration tests run in an actual app environment where native libraries are available.

#### Using the Example App (Easiest)

```bash
cd example
flutter test integration_test/
```

This will run the integration tests within the example app on a connected device or simulator.

#### On iOS Simulator

```bash
# List available simulators
flutter devices

# Run tests on a specific simulator
cd example
flutter test integration_test/ --device-id=<simulator-id>
```

#### On Android Emulator

```bash
# Start an emulator first
flutter emulators --launch <emulator-name>

# Run tests
cd example
flutter test integration_test/ --device-id=<emulator-id>
```

#### On Physical Device

```bash
# Connect device via USB/WiFi
cd example
flutter test integration_test/
```

### Quick Validation (Limited)

To run the limited unit tests (only error handling):

```bash
flutter test test/pose_detector_test.dart
```

**Expected result:** 2 passed (StateError tests), 28 failed (require TFLite) - this is normal!

## Test Coverage

The test suite covers:

1. **Initialization**
   - Default configuration
   - Custom parameters
   - Re-initialization
   - Multiple dispose calls

2. **Error Handling**
   - StateError when not initialized
   - Invalid image bytes
   - Empty images

3. **Real Image Detection**
   - 7 sample images (pose1.jpg - pose7.jpg)
   - Multiple people per image
   - Different image sizes

4. **API Methods**
   - `detect(Uint8List)` with byte arrays
   - `detectOnImage(img.Image)` with pre-decoded images
   - Results consistency between both methods

5. **Model Variants**
   - PoseLandmarkModel.lite
   - PoseLandmarkModel.full
   - PoseLandmarkModel.heavy

6. **Detection Modes**
   - PoseMode.boxesAndLandmarks (full pipeline)
   - PoseMode.boxes (fast, no landmarks)

7. **Data Access**
   - 33 BlazePose landmarks
   - Bounding box coordinates
   - Normalized coordinates
   - Visibility scores
   - Depth (z) coordinates

8. **Configuration**
   - detectorConf threshold
   - detectorIou threshold
   - maxDetections limit
   - minLandmarkScore threshold

## Sample Images

The tests use real pose images from `assets/samples/`:
- pose1.jpg through pose7.jpg
- Images contain people in various poses
- Different lighting conditions and backgrounds
- Multiple people in some images

## Expected Test Results

When running in a proper environment (device or platform-specific tests):
- ✅ All 25 tests should pass
- Detection should find people in all sample images
- Landmarks should have valid coordinates within image bounds
- Visibility scores should be between 0.0 and 1.0

## Troubleshooting

### "Failed to load dynamic library" error
This error occurs when running `flutter test` without a proper platform environment. The TensorFlow Lite native library is not available to pure Dart tests.

**Solutions:**
- Preferred: Run tests on a device or use platform-specific test commands.
- Local macOS-only fallback: `lib/src/pose_landmark_model.dart` now checks `POSE_TFLITE_LIB` and the repo path `macos/Frameworks/libtensorflowlite_c-mac.dylib`. On macOS you can point directly to a dylib for host tests:
  ```bash
  POSE_TFLITE_LIB=$PWD/macos/Frameworks/libtensorflowlite_c-mac.dylib flutter test test/pose_detector_test.dart
  ```

### Tests timing out
Some tests process multiple images and may take longer on slower devices.

**Solution:** Increase the test timeout or use the lite model for faster processing.

### No people detected in images
If tests fail because no people are detected, verify:
1. Sample images are properly bundled (check pubspec.yaml)
2. Model files are accessible
3. Configuration thresholds aren't too strict

## Adding New Tests

When adding new tests:

1. Use real sample images when possible
2. Test both `detect()` and `detectOnImage()` paths
3. Verify results are within expected bounds
4. Test edge cases and error conditions
5. Clean up resources with `await detector.dispose()`

## CI/CD Integration

For CI/CD pipelines, consider:
- Running tests on emulators/simulators
- Using integration_test package for full E2E tests
- Splitting unit tests (pure Dart) from integration tests (require TFLite)
