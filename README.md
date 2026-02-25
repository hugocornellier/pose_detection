# pose_detection_tflite
 
[![pub points](https://img.shields.io/pub/points/pose_detection_tflite?color=2E8B57&label=pub%20points)](https://pub.dev/packages/pose_detection_tflite/score)
[![pub package](https://img.shields.io/pub/v/pose_detection_tflite.svg)](https://pub.dev/packages/pose_detection_tflite)

Flutter implementation of Google's [Pose Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) with bounding boxes. 
Provides on-device, multi-person pose and landmark detection using TensorFlow Lite.


![Example Screenshot](assets/screenshots/ex1.png)

## Quick Start

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

Future main() async {
  // Set mode/model then initialize
  final PoseDetector detector = PoseDetector(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
  );
  await detector.initialize();

  // Load and detect from image bytes
  final Uint8List imageBytes = await File('image.jpg').readAsBytes();
  final List<Pose> results = await detector.detect(imageBytes);

  // Access results
  for (final Pose pose in results) {
    final BoundingBox bbox = pose.boundingBox;
    print('Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (pose.hasLandmarks) {
      // Iterate over landmarks
      for (final PoseLandmark lm in pose.landmarks) {
        print('${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) vis=${lm.visibility.toStringAsFixed(2)}');
      }

      // Access landmarks individually
      // See "Pose Landmark Types" section in README for full list of landmarks
      final PoseLandmark? leftKnee = pose.getLandmark(PoseLandmarkType.leftKnee);
      if (leftKnee != null) {
        print('Left knee visibility: ${leftKnee.visibility.toStringAsFixed(2)}');
      }
    }
  }

  // Clean up
  await detector.dispose();
}
```

Refer to the [sample code](https://pub.dev/packages/pose_detection_tflite/example) on the pub.dev example tab for a more in-depth example.

## Web (Flutter Web)

This package supports Flutter Web using the same package import:

```dart
import 'package:pose_detection_tflite/pose_detection_tflite.dart';
```

The main difference is how you load images:

- The Quick Start example above uses `dart:io` (`File(...)`), which is not available on web.
- On web, load an image as `Uint8List` (for example from a file picker, drag-and-drop, or network response) and call `detect(imageBytes)`.
- `detectFromMat(...)` (OpenCV `cv.Mat`) is native-only and is not available on web.
- `interpreterPoolSize`, `performanceConfig`, and `useNativePreprocessing` are accepted for API compatibility but are ignored on web (web runs CPU/WASM).

```dart
final detector = PoseDetector(
  mode: PoseMode.boxesAndLandmarks,
  landmarkModel: PoseLandmarkModel.heavy,
);
await detector.initialize(); // Also initializes the web TFLite/WASM runtime

final List<Pose> poses = await detector.detect(imageBytes);

await detector.dispose();
```

### Separate `example_web` app

The repository keeps the browser demo in `example_web/` (separate from `example/`) because the web sample uses browser-specific APIs (HTML file picker + canvas overlay) and UI flow.

Run the web demo locally:

```bash
cd example_web
flutter pub get
flutter run -d chrome
```

Build for web:

```bash
cd example_web
flutter build web
```

Note: `example_web/pubspec.yaml` includes a local `dependency_overrides` entry for `flutter_litert` (`../../flutter_litert`) for repo development. Update or remove it if your local folder layout is different.

## Pose Detection Modes

This package supports two operation modes that determine what data is returned:

| Mode                            | Description                                 | Output                        |
| ------------------------------- | ------------------------------------------- | ----------------------------- |
| **boxesAndLandmarks** (default) | Full two-stage detection (YOLO + BlazePose) | Bounding boxes + 33 landmarks |
| **boxes**                       | Fast YOLO-only detection                    | Bounding boxes only           |

### Use boxes-only mode for faster detection

When you only need to detect where people are (without body landmarks), use `PoseMode.boxes` for better performance:

```dart
final PoseDetector detector = PoseDetector(
  mode: PoseMode.boxes,  // Skip landmark detection
);
await detector.initialize();

final List<Pose> results = await detector.detect(imageBytes);
for (final Pose pose in results) {
  print('Person detected at: ${pose.boundingBox}');
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

```dart
// Example - how to access specific landmarks
// PoseLandmarkType can be any of the 33 landmarks listed above.
final PoseLandmark? leftHip = pose.getLandmark(PoseLandmarkType.leftHip);
if (leftHip != null && leftHip.visibility > 0.5) {
    // Pixel coordinates in original image space
    print('Left hip position: (${leftHip.x}, ${leftHip.y})');

    // Depth information (relative z-coordinate)
    print('Left hip depth: ${leftHip.z}');
}
```

### Drawing Skeleton Connections

The package provides `poseLandmarkConnections`, a predefined list of landmark pairs that form the body skeleton. Use this to draw skeleton overlays:

```dart
import 'package:flutter/material.dart';
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

class PoseOverlayPainter extends CustomPainter {
  final Pose pose;

  PoseOverlayPainter(this.pose);

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Draw all skeleton connections
    for (final connection in poseLandmarkConnections) {
      final PoseLandmark? start = pose.getLandmark(connection[0]);
      final PoseLandmark? end = pose.getLandmark(connection[1]);

      // Only draw if both landmarks are visible
      if (start != null && end != null &&
          start.visibility > 0.5 && end.visibility > 0.5) {
        canvas.drawLine(
          Offset(start.x, start.y),
          Offset(end.x, end.y),
          paint,
        );
      }
    }

    // Draw landmark points
    for (final landmark in pose.landmarks) {
      if (landmark.visibility > 0.5) {
        canvas.drawCircle(
          Offset(landmark.x, landmark.y),
          5,
          Paint()..color = Colors.red,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
```

The `poseLandmarkConnections` constant contains 27 connections organized by body region:
- **Face**: Eyes to nose, eyes to ears, mouth
- **Torso**: Shoulders and hips forming the core
- **Arms**: Shoulders → elbows → wrists → fingers (left and right)
- **Legs**: Hips → knees → ankles → feet (left and right)

## Advanced Usage

### Multi-person detection

The detector automatically handles multiple people in a single image with parallel landmark extraction:

```dart
final List<Pose> results = await detector.detect(imageBytes);
print('Detected ${results.length} people');

for (int i = 0; i < results.length; i++) {
  final Pose pose = results[i];
  print('Person ${i + 1}:');
  print('Bounding box: ${pose.boundingBox}');
  print('Confidence: ${pose.score.toStringAsFixed(2)}');
  print('Landmarks: ${pose.landmarks.length}');
}
```

**Performance Optimization:** The detector uses an interpreter pool to extract landmarks in parallel when multiple people are detected. Configure the pool size based on your use case:

```dart
final detector = PoseDetector(
  interpreterPoolSize: 5,  // Up to 5 concurrent landmark extractions
);
```

- **Pool size 1**: Sequential processing (lowest memory, ~50ms per person)
- **Pool size 3-5**: Recommended for 2-5 people (balanced performance/memory)
- **Pool size 5-10**: For crowded scenes with many people (~10MB memory per interpreter)
- **Default pool size**: 5 (balanced performance vs. memory for most cases)

**Example speedup** with 5 people detected:
- Pool size 1: ~250ms total (sequential)
- Pool size 5: ~50ms total (all parallel) = **5x faster**

### Camera/video stream processing

For real-time camera processing, reuse the same detector instance:

```dart
final detector = PoseDetector(
  landmarkModel: PoseLandmarkModel.lite,  // Use lite for better FPS
  detectorConf: 0.6,
);
await detector.initialize();

// Process each frame (convert camera frame to cv.Mat first)
void processFrame(cv.Mat mat) async {
  final results = await detector.detectFromMat(
    mat,
    imageWidth: mat.cols,
    imageHeight: mat.rows,
  );
  // Update UI with results
}

await detector.dispose();
```
