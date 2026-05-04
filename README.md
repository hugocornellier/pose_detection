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

<p align="center">
  <img src="assets/screenshots/demo.gif" alt="Pose Detection Demo" width="600">
</p>

## Quick Start

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:pose_detection/pose_detection.dart';

Future main() async {
  // One-step construction and initialization
  final PoseDetector detector = await PoseDetector.create(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: PoseLandmarkModel.heavy,
  );

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

Alternatively, construct and initialize separately if you need to configure between steps:

```dart
final PoseDetector detector = PoseDetector();
await detector.initialize(
  mode: PoseMode.boxesAndLandmarks,
  landmarkModel: PoseLandmarkModel.heavy,
);
```

Refer to the [sample code](https://pub.dev/packages/pose_detection/example) on the pub.dev example tab for a more in-depth example.

## Pose Detection Modes

This package supports two operation modes that determine what data is returned:

| Mode                            | Description                                 | Output                        |
| ------------------------------- | ------------------------------------------- | ----------------------------- |
| **boxesAndLandmarks** (default) | Full two-stage detection (YOLO + BlazePose) | Bounding boxes + 33 landmarks |
| **boxes**                       | Fast YOLO-only detection                    | Bounding boxes only           |

### Use boxes-only mode for faster detection

When you only need to detect where people are (without body landmarks), use `PoseMode.boxes` for better performance:

```dart
final PoseDetector detector = PoseDetector();
await detector.initialize(
  mode: PoseMode.boxes,  // Skip landmark detection
);

final List<Pose> results = await detector.detect(imageBytes);
for (final Pose pose in results) {
  print('Person detected at: ${pose.boundingBox}');
  print('Detection confidence: ${pose.score.toStringAsFixed(2)}');
  // pose.hasLandmarks will be false
}
```

## Bounding Boxes

The boundingBox property returns a BoundingBox object representing the pose bounding box in
absolute pixel coordinates. The BoundingBox provides convenient access to corner points,
dimensions (width and height), and the center point.

### Accessing Corners

```dart
final BoundingBox boundingBox = pose.boundingBox;

// Access individual corners by name (each is a Point with x and y)
final Point topLeft     = boundingBox.topLeft;       // Top-left corner
final Point topRight    = boundingBox.topRight;      // Top-right corner
final Point bottomRight = boundingBox.bottomRight;   // Bottom-right corner
final Point bottomLeft  = boundingBox.bottomLeft;    // Bottom-left corner

// Access coordinates
print('Top-left: (${topLeft.x}, ${topLeft.y})');
```

### Additional Bounding Box Parameters

```dart
final BoundingBox boundingBox = pose.boundingBox;

// Access dimensions and center
final double width  = boundingBox.width;     // Width in pixels
final double height = boundingBox.height;    // Height in pixels
final Point center = boundingBox.center;  // Center point

// Access coordinates
print('Size: ${width} x ${height}');
print('Center: (${center.x}, ${center.y})');

// Access all corners as a list (order: top-left, top-right, bottom-right, bottom-left)
final List<Point> allCorners = boundingBox.corners;
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
// Example: how to access specific landmarks
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

### Built-in Overlay Painters

The package ships two ready-to-use `CustomPainter` implementations:

| Class | Use case |
|---|---|
| `MultiOverlayPainter` | Still images: scales detection coordinates to fit the widget |
| `CameraPoseOverlayPainter` | Live camera preview: handles coordinate mapping and optional front-camera horizontal mirroring |

```dart
// Still image overlay
CustomPaint(
  foregroundPainter: MultiOverlayPainter(results: poses),
  child: Image.memory(imageBytes),
)

// Live camera overlay (front camera, mirrored)
CustomPaint(
  foregroundPainter: CameraPoseOverlayPainter(
    poses: poses,
    cameraSize: Size(cameraWidth.toDouble(), cameraHeight.toDouble()),
    mirrorHorizontally: isFrontCamera,
  ),
  child: CameraPreview(controller),
)
```

## Live Camera Detection

For real-time pose detection with a camera feed, use `detectFromCameraImage`. It auto-detects YUV420 (NV12 / NV21 / I420) and desktop BGRA/RGBA layouts, and the `cvtColor`, optional `rotate`, and `maxDim` downscale all run inside the detector's existing isolate: the UI thread is never blocked by OpenCV work.

```dart
import 'package:camera/camera.dart';
import 'package:pose_detection/pose_detection.dart';

final detector = await PoseDetector.create(
  landmarkModel: PoseLandmarkModel.lite, // lite model for higher FPS
);

final cameras = await availableCameras();
final camera = CameraController(
  cameras.first,
  ResolutionPreset.medium,
  enableAudio: false,
  imageFormatGroup: ImageFormatGroup.yuv420,
);
await camera.initialize();

camera.startImageStream((CameraImage image) async {
  final poses = await detector.detectFromCameraImage(
    image,
    // rotation: CameraFrameRotation.cw90, // based on device orientation
    maxDim: 640, // optional in-isolate downscale before inference
  );
  // Process poses...
});
```

**Tips for camera detection:**
- `detectFromCameraImage` replaces the old `packYuv420` + manual `cv.cvtColor` + `cv.rotate` dance in one call; no `cv.Mat` on the UI thread.
- Pass `rotation:` so the detector sees upright frames (Android back/front + device orientation logic); on iOS the camera plugin pre-rotates so this is often null.
- Pass `maxDim:` (e.g. 640) to downscale in-isolate; the detection model internally resizes to 256px, so full-res frames just waste IPC bandwidth.
- Use `PoseLandmarkModel.lite` for fastest real-time performance.
- Mirror the overlay on the front camera to match `CameraPreview`'s auto-mirrored texture.
- For advanced use (e.g. reusing a frame across multiple detectors), `prepareCameraFrame(...)` + `detectFromCameraFrame(...)` is the underlying two-step API.

See the full [example app](https://pub.dev/packages/pose_detection/example) for a production implementation including orientation handling, mirror handling, and frame throttling.

## Video Detection

In addition to still images and live camera feeds, `pose_detection` supports frame-by-frame inference on video files. The example app includes a fully working `VideoFileScreen` that shows the end-to-end flow:

1. **Open the video** with `cv.VideoCapture.fromFile(path)` (powered by [opencv_dart](https://pub.dev/packages/opencv_dart)).
2. **Read frames in a loop** with `cap.read()`, passing each `cv.Mat` directly to `detector.detectFromMat(frame)`.
3. **Draw results** onto the same `Mat` (bounding boxes + skeleton overlay).
4. **Write the annotated frame** to an output file with `cv.VideoWriter`, preserving the original FPS and resolution.
5. **Play back the result** in-app with the `video_player` package.

```dart
final cap = cv.VideoCapture.fromFile(path);
final fps = cap.get(cv.CAP_PROP_FPS);
final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();

final writer = cv.VideoWriter.fromFile(outPath, 'avc1', fps, (width, height));

cv.Mat? frame;
while (true) {
  final (ok, mat) = cap.read(m: frame);
  frame = mat;
  if (!ok || frame.isEmpty) break;

  final List<Pose> poses = await detector.detectFromMat(frame);
  // draw poses on frame...
  writer.write(frame);
}

cap.release();
writer.release();
```

The output is a standard H.264 MP4 with the pose overlay baked in. See `VideoFileScreen` in the [example app](https://pub.dev/packages/pose_detection/example) for the full implementation including progress tracking, cancellation, temporal smoothing, and playback.

**Notes:**
- Video processing is CPU-bound and runs off the UI thread via the detector's isolate. The UI stays responsive.
- Use `PoseLandmarkModel.lite` or `PoseLandmarkModel.full` for a better speed/accuracy tradeoff when processing long videos.
- On Linux, GStreamer plugins are required to open MP4 files: `sudo apt install gstreamer1.0-libav gstreamer1.0-plugins-good gstreamer1.0-plugins-bad`.

## Background Processing

All inference runs automatically in a background isolate: the UI thread is never blocked during detection or landmark extraction. No special configuration is needed; `PoseDetector` handles isolate management internally.

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
final detector = PoseDetector();
await detector.initialize(
  interpreterPoolSize: 3,  // Number of interpreter instances
);
```

- **Default pool size**: 1
- When any hardware acceleration is active (auto, XNNPACK, or GPU), pool size is automatically forced to 1 to prevent thread contention

### Detect from a file path

`detectFromFilepath` reads the file and delegates to `detect`. Native-only (uses `dart:io`).

```dart
final List<Pose> poses = await detector.detectFromFilepath('/path/to/image.jpg');
```

### Detect from raw pixel bytes (zero-copy)

`detectFromMatBytes` accepts raw pixel data without constructing a `cv.Mat` first. Bytes are transferred to the background isolate via `TransferableTypedData` with no copy. Useful when you already have decoded pixel data from another source.

```dart
final List<Pose> poses = await detector.detectFromMatBytes(
  pixelBytes,          // Raw BGR pixel data
  width: imageWidth,
  height: imageHeight,
  matType: 16,         // CV_8UC3 (default)
);
```

## Web (Flutter Web)

This package supports Flutter Web using the same package import:

```dart
import 'package:pose_detection/pose_detection.dart';
```

Two web runtimes are available, selectable per `PoseDetector`:

1. **LiteRT.js with WebGPU delegate (default).** Google's official web runtime via `flutter_litert ≥ 2.5.0`. ~18× faster in real measurements (446 ms → 25 ms / call on the heavy BlazePose model with mixed single/multi-person images). Auto-loaded from CDN on first use, no `web/index.html` changes required. Prefers WebGPU; falls back to WASM automatically on unsupported browsers.
2. **`tflite-js` (CPU/WASM, legacy).** Pass `useLiteRt: false` to opt into the previous default. No additional CDN scripts beyond those already loaded.

The main difference from native is how you load images:

- The Quick Start example above uses `dart:io` (`File(...)`), which is not available on web.
- On web, load an image as `Uint8List` (for example from a file picker, drag-and-drop, or network response) and call `detect(imageBytes)`.
- `detectFromMat(...)` (OpenCV `cv.Mat`) is native-only and is not available on web.
- `interpreterPoolSize` and `performanceConfig` are accepted for API compatibility but are ignored on web.

```dart
final detector = await PoseDetector.create(
  mode: PoseMode.boxesAndLandmarks,
  landmarkModel: PoseLandmarkModel.heavy,
);

final List<Pose> poses = await detector.detect(imageBytes);

await detector.dispose();
```

### Web (LiteRT.js + WebGPU, default)

No extra configuration needed. LiteRT.js is the default runtime:

```dart
final detector = await PoseDetector.create(
  mode: PoseMode.boxesAndLandmarks,
  landmarkModel: PoseLandmarkModel.heavy,
  // liteRtAccelerator defaults to 'auto': prefers WebGPU, falls back to WASM.
);
```

`liteRtAccelerator` accepts:

| Value | Behavior |
|---|---|
| `'auto'` (default) | Try WebGPU; if compile fails (no `navigator.gpu`, or unsupported ops) fall back to WASM. |
| `'webgpu'` | Force WebGPU; same compile-time fallback to WASM if anything fails. |
| `'wasm'` | Force SIMD-optimized WASM. Use this to opt out of GPU even when available. |

The WASM fallback is still substantially faster than the legacy `tflite-js` path because LiteRT.js's WASM is SIMD-optimized.

To opt into the legacy tflite-js path, pass `useLiteRt: false`.

If you need to self-host the runtime (offline, strict CSP, or to pin a specific build), call `flutter_litert`'s `configureLiteRtLoader(moduleUrl: ..., wasmUrl: ...)` before any `PoseDetector.create`, or set `autoLoad: false` and load it from your own `<script>` tag instead.

### Benchmarks

Heavy BlazePose model on macOS Chrome 147, 5 images, 10 timed iterations each, averaged over 2 runs (see `runWebBenchmark.sh`):

| Image | Detections | Default (tflite-js) | LiteRT.js webgpu | Speedup |
|---|---|---|---|---|
| pose1 | 1 | 357 ms | 20 ms | 17.8× |
| pose2 | 1 | 357 ms | 18 ms | 19.9× |
| pose3 | 2 | 430 ms | 23 ms | 18.7× |
| pose4 | 6 | 726 ms | 46 ms | 15.9× |
| pose5 | 1 | 360 ms | 17 ms | 20.7× |
| **mean** | | **446 ms** | **25 ms** | **~18×** |

Detection counts are identical between the two runtimes on every image.

### Separate `example_web` app

The repository keeps the browser demo in `example_web/` (separate from `example/`) because the web sample uses browser-specific APIs (HTML file picker + canvas overlay) and UI flow. The demo uses the default `'auto'` accelerator (WebGPU with WASM fallback). Copy from <a href="https://github.com/hugocornellier/pose_detection/blob/main/example_web/lib/main.dart" target="_blank">example_web/lib/main.dart</a> as a starting point.

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

## Performance

### Hardware Acceleration

The package automatically selects the best acceleration strategy for each platform:

| Platform | Default Delegate | Speedup | Notes |
|----------|-----------------|---------|-------|
| **macOS** | XNNPACK | 2-5x | SIMD vectorization (NEON on ARM, AVX on x86) |
| **Linux** | XNNPACK | 2-5x | SIMD vectorization |
| **iOS** | Metal GPU | 2-4x | Hardware GPU acceleration |
| **Android** | XNNPACK | 2-5x | ARM NEON SIMD acceleration |
| **Windows** | XNNPACK | 2-5x | SIMD vectorization (AVX on x86) |

No configuration needed, just call `initialize()` and you get the optimal performance for your platform.

### Advanced Performance Configuration

```dart
// Auto mode (default), optimal for each platform
await detector.initialize();

// Force XNNPACK (all native platforms)
final detector = await PoseDetector.create(
  performanceConfig: PerformanceConfig.xnnpack(numThreads: 4),
);

// Force GPU delegate
final detector = await PoseDetector.create(
  performanceConfig: PerformanceConfig.gpu(),
);

// CPU-only (maximum compatibility)
final detector = await PoseDetector.create(
  performanceConfig: PerformanceConfig.disabled,
);
```
## Migration Guide

### 3.0.0: `detectFromMat` signature change

The `imageWidth` and `imageHeight` named arguments have been removed. Dimensions are now read directly from the Mat.

```dart
// Before (2.x)
final poses = await detector.detectFromMat(
  mat,
  imageWidth: mat.cols,
  imageHeight: mat.rows,
);

// After (3.0)
final poses = await detector.detectFromMat(mat);
```