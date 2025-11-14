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
  final PoseDetector detector = PoseDetector();
  await detector.initialize(
    options: const PoseOptions(
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: PoseLandmarkModel.heavy,
    ),
  );

  // 2. detect
  final Uint8List imageBytes = await File('path/to/image.jpg').readAsBytes();
  final List<Pose> results = await detector.detect(imageBytes);

  // 3. access results
  for (final Pose pose in results) {
    final RectPx bbox = pose.bboxPx;
    print('Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (pose.hasLandmarks) {
      for (final PoseLandmark lm in pose.landmarks) {
        print('${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) vis=${lm.visibility.toStringAsFixed(2)}');
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

## Pose Landmark Models

Choose the model that fits your performance needs:

| Model | Speed | Accuracy |
|-------|-------|----------|
| **lite** | Fastest | Good |
| **full** | Balanced | Better |
| **heavy** | Slowest | Best |