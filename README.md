# pose_detection_tflite

A pure Dart/Flutter implementation of Google's MediaPipe pose detection and facial landmark models using TensorFlow Lite. 
This package provides on-device, multi-person pose detection with minimal dependencies, just TensorFlow Lite and image.

![Example Screenshot](assets/screenshots/ex1.png)

## Quick Start

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

Future main() async {
  // 1. initialize
  final PoseDetector detector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
  );
  await detector.initialize();

  // 2. detect
  final Uint8List imageBytes = await File('image.jpg').readAsBytes();
  final List<Pose> results = await detector.detect(imageBytes);

  // 3. access results
  for (final Pose pose in results) {
    final RectPx bbox = pose.bboxPx;
    print('Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (pose.hasLandmarks) {
      // iterate through landmarks
      for (final PoseLandmark lm in pose.landmarks) {
        print('${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) vis=${lm.visibility.toStringAsFixed(2)}');
      }

      // access individual landmarks
      // see "Pose Landmark Types" section in README for full list of landmarks
      final PoseLandmark? leftKnee = pose.getLandmark(PoseLandmarkType.leftKnee);
      if (leftKnee != null) {
        print('Left knee visibility: ${leftKnee.visibility.toStringAsFixed(2)}');
      }
    }
  }

  // 4. clean-up
  await detector.dispose();
}
```

Refer to the [sample code](https://pub.dev/packages/pose_detection_tflite/example) on the pub.dev example tab for a more in-depth example.

## Pose Detection Modes

This package supports two operation modes that determine what data is returned:

| Mode                            | Description                                 | Output                        |
| ------------------------------- | ------------------------------------------- | ----------------------------- |
| **boxesAndLandmarks** (default) | Full two-stage detection (YOLO + BlazePose) | Bounding boxes + 33 landmarks |
| **boxes**                       | Fast YOLO-only detection                    | Bounding boxes only           |

### Using boxes-only mode for faster detection

When you only need to detect where people are (without body landmarks), use `PoseMode.boxes` for better performance:

```dart
final PoseDetector detector = PoseDetector(
  mode: PoseMode.boxes,  // Skip landmark detection
);
await detector.initialize();

final List<Pose> results = await detector.detect(imageBytes);
for (final Pose pose in results) {
  print('Person detected at: ${pose.bboxPx}');
  print('Detection confidence: ${pose.score.toStringAsFixed(2)}');
  // pose.hasLandmarks will be false
}
```

## Pose Landmark Models

Choose the model that fits your performance needs:

| Model | Speed | Accuracy |
|-------|-------|----------|
| **lite** | Fastest | Good |
| **full** | Balanced | Better |
| **heavy** | Slowest | Best |

## Configuration Options

Fine-tune detection behavior by passing parameters to the constructor:

```dart
final PoseDetector detector = PoseDetector(
  mode: PoseMode.boxesAndLandmarks,
  landmarkModel: PoseLandmarkModel.full,

  // Person detection thresholds
  detectorConf: 0.6,        // Min confidence for person detection (0.0-1.0)
  detectorIou: 0.4,         // IoU threshold for NMS (0.0-1.0)
  maxDetections: 10,        // Max number of people to detect

  // Landmark quality threshold
  minLandmarkScore: 0.5,    // Min score to include landmark results
);
await detector.initialize();
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `boxesAndLandmarks` | Detection mode (boxes only or boxes + landmarks) |
| `landmarkModel` | `heavy` | Model variant (lite/full/heavy) |
| `detectorConf` | `0.5` | Minimum confidence threshold for person detection |
| `detectorIou` | `0.45` | IoU threshold for Non-Maximum Suppression |
| `maxDetections` | `10` | Maximum number of people to detect in one image |
| `minLandmarkScore` | `0.5` | Minimum quality score to accept landmark results |

## Pose Landmark Types

Every pose contains up to 33 landmarks that align with the BlazePose specification:

- nose
- leftEyeInner
- leftEye
- leftEyeOuter
- rightEyeInner
- rightEye
- rightEyeOuter
- leftEar
- rightEar
- mouthLeft
- mouthRight
- leftShoulder
- rightShoulder
- leftElbow
- rightElbow
- leftWrist
- rightWrist
- leftPinky
- rightPinky
- leftIndex
- rightIndex
- leftThumb
- rightThumb
- leftHip
- rightHip
- leftKnee
- rightKnee
- leftAnkle
- rightAnkle
- leftHeel
- rightHeel
- leftFootIndex
- rightFootIndex

### Working with landmarks

Landmarks provide pixel coordinates (x, y), depth (z), and visibility scores:

```dart
for (final Pose pose in results) {
  // Check pose quality
  print('Pose confidence: ${pose.score.toStringAsFixed(2)}');

  if (pose.hasLandmarks) {
    // Iterate all landmarks
    for (final PoseLandmark lm in pose.landmarks) {
      print('${lm.type}: (${lm.x}, ${lm.y}) vis=${lm.visibility}');
    }

    // Access specific landmarks
    final PoseLandmark? leftHip = pose.getLandmark(PoseLandmarkType.leftHip);
    if (leftHip != null && leftHip.visibility > 0.5) {
      // Pixel coordinates in original image space
      print('Left hip position: (${leftHip.x}, ${leftHip.y})');

      // Depth information (relative z-coordinate)
      print('Left hip depth: ${leftHip.z}');

      // Normalized coordinates (0.0-1.0)
      final double xNorm = leftHip.xNorm(pose.imageWidth);
      final double yNorm = leftHip.yNorm(pose.imageHeight);
      print('Normalized: ($xNorm, $yNorm)');
    }
  }
}
```

**Landmark properties:**
- `x`, `y`: Pixel coordinates in the original image
- `z`: Depth coordinate (roughly same scale as x, smaller is closer to camera)
- `visibility`: Confidence score (0.0-1.0) that the landmark is visible
- `type`: The landmark type (nose, leftKnee, etc.)

**Helper methods:**
- `xNorm(imageWidth)`: Get x coordinate normalized to 0.0-1.0
- `yNorm(imageHeight)`: Get y coordinate normalized to 0.0-1.0
- `toPixel(imageWidth, imageHeight)`: Convert to integer Point

## Advanced Usage

### Processing pre-decoded images

If you already have an `Image` object from the `image` package, use `detectOnImage()` to skip re-decoding:

```dart
import 'package:image/image.dart' as img;

final img.Image image = img.decodeImage(imageBytes)!;
final List<Pose> results = await detector.detectOnImage(image);
```

### Multi-person detection

The detector automatically handles multiple people in a single image:

```dart
final List<Pose> results = await detector.detect(imageBytes);
print('Detected ${results.length} people');

for (int i = 0; i < results.length; i++) {
  final Pose pose = results[i];
  print('Person ${i + 1}:');
  print('  Bounding box: ${pose.bboxPx}');
  print('  Confidence: ${pose.score.toStringAsFixed(2)}');
  print('  Landmarks: ${pose.landmarks.length}');
}
```

### Camera/video stream processing

For real-time camera processing, reuse the same detector instance:

```dart
// Initialize once
final detector = PoseDetector(
  landmarkModel: PoseLandmarkModel.lite,  // Use lite for better FPS
  detectorConf: 0.6,
);
await detector.initialize();

// Process each frame
void processFrame(Uint8List frameBytes) async {
  final results = await detector.detect(frameBytes);
  // Update UI with results
}

// Clean up when done
await detector.dispose();
```

**Performance tip**: The detector internally reuses buffers to minimize memory allocations during video processing.

## Error Handling

### Initialization errors

Always await initialization and handle potential errors:

```dart
final detector = PoseDetector(
  landmarkModel: PoseLandmarkModel.heavy,
);

try {
  await detector.initialize();
} catch (e) {
  print('Failed to initialize detector: $e');
  // Handle initialization failure (missing model files, etc.)
}
```

### Detection errors

The detector throws `StateError` if used before initialization:

```dart
final detector = PoseDetector();

// This will throw StateError
// final results = await detector.detect(imageBytes);

// Always initialize first
await detector.initialize();
final results = await detector.detect(imageBytes);
```

### Handling empty results

Not finding people in an image is normal, not an error:

```dart
final List<Pose> results = await detector.detect(imageBytes);

if (results.isEmpty) {
  print('No people detected in this image');
} else {
  print('Found ${results.length} person(s)');
}
```

### Low-quality landmarks

Check the landmark quality score before use:

```dart
for (final Pose pose in results) {
  if (pose.score < 0.5) {
    print('Low confidence detection, may be inaccurate');
  }

  for (final PoseLandmark lm in pose.landmarks) {
    if (lm.visibility < 0.5) {
      // This landmark may be occluded or out of frame
      print('${lm.type} has low visibility');
    }
  }
}
```

## Best Practices

1. **Reuse detector instances**: Create one detector and reuse it for multiple images/frames
2. **Choose the right model**: Use `lite` for video, `heavy` for high-quality single-image analysis
3. **Filter by visibility**: Always check `visibility > 0.5` before using landmark coordinates
4. **Dispose properly**: Call `dispose()` when done to free resources
5. **Handle re-initialization**: The detector automatically disposes previous state when re-initialized

## Platform Support

- **Android** ✓
- **iOS** ✓
- **macOS** ✓
- **Windows** ✓
- **Linux** ✓
- **Web** ✗ (TensorFlow Lite C library not available)

All inference runs in pure Dart using TensorFlow Lite. No platform-specific code required.
