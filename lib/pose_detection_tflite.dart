/// On-device pose detection and landmark estimation using TensorFlow Lite.
///
/// This library provides a Flutter plugin for real-time human pose detection
/// using Google's MediaPipe BlazePose models. It detects persons in images
/// and extracts 33 body landmarks (keypoints) for each detected person.
///
/// **Quick Start:**
/// ```dart
/// import 'package:pose_detection_tflite/pose_detection_tflite.dart';
///
/// final detector = PoseDetector();
/// await detector.initialize();
///
/// final poses = await detector.detect(imageBytes);
/// for (final pose in poses) {
///   print('Person detected at ${pose.boundingBox}');
///   if (pose.hasLandmarks) {
///     final nose = pose.getLandmark(PoseLandmarkType.nose);
///     print('Nose position: (${nose?.x}, ${nose?.y})');
///   }
/// }
///
/// await detector.dispose();
/// ```
///
/// **Main Classes:**
/// - [PoseDetector]: Main API for pose detection
/// - [Pose]: Detected person with bounding box and optional 33 landmarks
/// - [PoseLandmark]: Single body keypoint with 3D coordinates and visibility
/// - [PoseLandmarkType]: Enum of 33 body parts (nose, shoulders, knees, etc.)
/// - [BoundingBox]: Axis-aligned rectangle for person location
///
/// **Detection Modes:**
/// - [PoseMode.boxes]: Fast detection returning only bounding boxes
/// - [PoseMode.boxesAndLandmarks]: Full pipeline with 33 landmarks per person
///
/// **Model Variants:**
/// - [PoseLandmarkModel.lite]: Fastest, good accuracy
/// - [PoseLandmarkModel.full]: Balanced speed/accuracy (default)
/// - [PoseLandmarkModel.heavy]: Slowest, best accuracy
library;

export 'src/types.dart';
export 'src/pose_detector.dart' show PoseDetector;
export 'src/dart_registration.dart';
