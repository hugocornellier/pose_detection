<h1 align="center">pose_detection</h1>
 
<p align="center">
<a href="https://flutter.dev"><img src="https://img.shields.io/badge/Platform-Flutter-02569B?logo=flutter" alt="Platform"></a>
<a href="https://dart.dev"><img src="https://img.shields.io/badge/language-Dart-blue" alt="Language: Dart"></a>
<br>
<a href="https://pub.dev/packages/pose_detection"><img src="https://img.shields.io/pub/v/pose_detection?label=pub.dev&labelColor=333940&logo=dart" alt="Pub Version"></a>
<a href="https://pub.dev/packages/pose_detection/score"><img src="https://img.shields.io/pub/points/pose_detection?color=2E8B57&label=pub%20points" alt="pub points"></a>
<a href="https://github.com/hugocornellier/pose_detection/actions/workflows/build.yml"><img src="https://github.com/hugocornellier/pose_detection/actions/workflows/build.yml/badge.svg" alt="CI"></a>
<a href="https://github.com/hugocornellier/pose_detection/actions/workflows/integration.yml"><img src="https://github.com/hugocornellier/pose_detection/actions/workflows/integration.yml/badge.svg" alt="Tests"></a>
<a href="https://github.com/hugocornellier/pose_detection/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-007A88.svg?logo=apache" alt="License"></a>
</p>

Flutter plugin for on-device, multi-person pose detection and landmark estimation using TensorFlow Lite. Uses YOLOv8n for person detection and Google's [BlazePose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) for 33-keypoint landmark extraction.


![Example Screenshot](assets/screenshots/ex1.png)

## Quick Start

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose_detection/pose_detection.dart';

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
    print('Size: ${bbox.width} x ${bbox.height}, center: (${bbox.center.x}, ${bbox.center.y})');

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

Refer to the [sample code](https://pub.dev/packages/pose_detection/example) on the pub.dev example tab for a more in-depth example.

## Web (Flutter Web)

This package supports Flutter Web using the same package import:

```dart
import 'package:pose_detection/pose_detection.dart';
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
import 'package:pose_detection/pose_detection.dart';

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

The detector automatically handles multiple people in a single image:

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

**Interpreter Pool:** The detector maintains a pool of TensorFlow Lite interpreter instances for landmark extraction. Each interpreter adds ~10MB memory overhead.

```dart
final detector = PoseDetector(
  interpreterPoolSize: 3,  // Number of interpreter instances
);
```

- **Default pool size**: 1
- When XNNPACK is enabled (via `performanceConfig`), pool size is automatically forced to 1 to prevent thread contention

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
