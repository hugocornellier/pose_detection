// Comprehensive integration tests for PoseDetector.
//
// These tests cover:
// - Initialization and disposal
// - Error handling (works in standard test environment)
// - Detection with real sample images (requires device/platform-specific testing)
// - detect() and detectOnImage() methods
// - Different model variants (lite, full, heavy)
// - Different modes (boxes, boxesAndLandmarks)
// - Landmark and bounding box access
// - Configuration parameters
// - Edge cases
//
// NOTE: Most tests require TensorFlow Lite native libraries which are not
// available in the standard `flutter test` environment. To run all tests:
//
// - macOS: flutter test --platform=macos test/pose_detector_test.dart
// - Device: Run as integration tests on a physical device or emulator
//
// Tests that work in standard environment (no TFLite required):
// - StateError when not initialized
// - Parameter validation
//

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:image/image.dart' as img;
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

/// Test helper to create a minimal 1x1 PNG image
class TestUtils {
  static Uint8List createDummyImageBytes() {
    return Uint8List.fromList([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
      0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // Width: 1, Height: 1
      0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, // Bit depth, color type
      0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
      0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
      0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, // Image data
      0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
      0x42, 0x60, 0x82
    ]);
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('PoseDetector - Initialization and Disposal', () {
    test('should initialize successfully with default options', () async {
      final detector = PoseDetector();
      expect(detector.isInitialized, false);

      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
      expect(detector.isInitialized, false);
    });

    test('should initialize with custom configuration', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxes,
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.6,
        detectorIou: 0.4,
        maxDetections: 5,
        minLandmarkScore: 0.7,
      );

      await detector.initialize();
      expect(detector.isInitialized, true);
      expect(detector.mode, PoseMode.boxes);
      expect(detector.landmarkModel, PoseLandmarkModel.lite);
      expect(detector.detectorConf, 0.6);
      expect(detector.detectorIou, 0.4);
      expect(detector.maxDetections, 5);
      expect(detector.minLandmarkScore, 0.7);

      await detector.dispose();
    });

    test('should allow re-initialization', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();
      expect(detector.isInitialized, true);

      // Re-initialize should work
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    test('should handle multiple dispose calls', () async {
      final detector = PoseDetector();
      await detector.initialize();
      await detector.dispose();
      expect(detector.isInitialized, false);

      // Second dispose should not throw
      await detector.dispose();
      expect(detector.isInitialized, false);
    });
  });

  group('PoseDetector - Error Handling', () {
    test('should throw StateError when detect() called before initialize',
        () async {
      final detector = PoseDetector();
      final bytes = TestUtils.createDummyImageBytes();

      expect(
        () => detector.detect(bytes),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test(
        'should throw StateError when detectOnImage() called before initialize',
        () async {
      final detector = PoseDetector();
      final image = img.Image(width: 100, height: 100);

      expect(
        () => detector.detectOnImage(image),
        throwsA(isA<StateError>().having(
          (e) => e.message,
          'message',
          contains('not initialized'),
        )),
      );
    });

    test('should return empty list for invalid image bytes', () async {
      final detector = PoseDetector();
      await detector.initialize();

      final invalidBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
      final results = await detector.detect(invalidBytes);

      expect(results, isEmpty);
      await detector.dispose();
    });
  });

  group('PoseDetector - detect() with real images', () {
    test('should detect people in pose1.jpg with boxesAndLandmarks mode',
        () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final pose in results) {
        // Verify bounding box
        expect(pose.boundingBox, isNotNull);
        expect(pose.boundingBox.left, greaterThanOrEqualTo(0));
        expect(pose.boundingBox.top, greaterThanOrEqualTo(0));
        expect(pose.boundingBox.right, greaterThan(pose.boundingBox.left));
        expect(pose.boundingBox.bottom, greaterThan(pose.boundingBox.top));

        // Verify score
        expect(pose.score, greaterThan(0));
        expect(pose.score, lessThanOrEqualTo(1.0));

        // Verify landmarks
        expect(pose.hasLandmarks, true);
        expect(pose.landmarks.length, 33); // BlazePose has 33 landmarks

        // Check image dimensions
        expect(pose.imageWidth, greaterThan(0));
        expect(pose.imageHeight, greaterThan(0));
      }

      await detector.dispose();
    });

    test('should detect people in pose2.jpg', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose2.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      await detector.dispose();
    });

    test('should detect people in pose3.jpg', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose3.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      await detector.dispose();
    });

    test('should detect people with boxes-only mode', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxes,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      for (final pose in results) {
        // Should have bounding box
        expect(pose.boundingBox, isNotNull);
        expect(pose.score, greaterThan(0));

        // Should NOT have landmarks in boxes-only mode
        expect(pose.hasLandmarks, false);
        expect(pose.landmarks, isEmpty);
      }

      await detector.dispose();
    });
  });

  group('PoseDetector - detectOnImage() method', () {
    test('should work with pre-decoded image', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      // Load and decode image manually
      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final img.Image? image = img.decodeImage(bytes);

      expect(image, isNotNull);

      // Use detectOnImage instead of detect
      final List<Pose> results = await detector.detectOnImage(image!);

      expect(results, isNotEmpty);

      for (final pose in results) {
        expect(pose.boundingBox, isNotNull);
        expect(pose.hasLandmarks, true);
        expect(pose.landmarks.length, 33);

        // Verify image dimensions match the decoded image
        expect(pose.imageWidth, image.width);
        expect(pose.imageHeight, image.height);
      }

      await detector.dispose();
    });

    test('detectOnImage() should give same results as detect()', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.5,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      // Test with detect()
      final List<Pose> results1 = await detector.detect(bytes);

      // Test with detectOnImage()
      final img.Image? image = img.decodeImage(bytes);
      final List<Pose> results2 = await detector.detectOnImage(image!);

      // Should detect same number of people
      expect(results1.length, results2.length);

      // Scores should be identical (or very close)
      for (int i = 0; i < results1.length; i++) {
        expect(
          (results1[i].score - results2[i].score).abs(),
          lessThan(0.01),
        );
      }

      await detector.dispose();
    });
  });

  group('PoseDetector - Different Model Variants', () {
    test('should work with lite model', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    test('should work with full model', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.full);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    test('should work with heavy model', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.heavy);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });
  });

  group('PoseDetector - Landmark and BoundingBox Access', () {
    late PoseDetector detector;
    late List<Pose> poses;

    setUpAll(() async {
      detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      poses = await detector.detect(bytes);
    });

    tearDownAll(() async {
      await detector.dispose();
    });

    test('should access specific landmarks by type', () {
      expect(poses, isNotEmpty);
      final pose = poses.first;

      // Test accessing different landmark types
      final nose = pose.getLandmark(PoseLandmarkType.nose);
      expect(nose, isNotNull);
      expect(nose!.type, PoseLandmarkType.nose);

      final leftKnee = pose.getLandmark(PoseLandmarkType.leftKnee);
      expect(leftKnee, isNotNull);
      expect(leftKnee!.type, PoseLandmarkType.leftKnee);

      final rightShoulder = pose.getLandmark(PoseLandmarkType.rightShoulder);
      expect(rightShoulder, isNotNull);
      expect(rightShoulder!.type, PoseLandmarkType.rightShoulder);
    });

    test('should have valid landmark coordinates', () {
      final pose = poses.first;

      for (final landmark in pose.landmarks) {
        // Coordinates should be within image bounds
        expect(landmark.x, greaterThanOrEqualTo(0));
        expect(landmark.x, lessThanOrEqualTo(pose.imageWidth.toDouble()));
        expect(landmark.y, greaterThanOrEqualTo(0));
        expect(landmark.y, lessThanOrEqualTo(pose.imageHeight.toDouble()));

        // Visibility should be 0-1
        expect(landmark.visibility, greaterThanOrEqualTo(0));
        expect(landmark.visibility, lessThanOrEqualTo(1.0));

        // Z coordinate exists
        expect(landmark.z, isNotNull);
      }
    });

    test('should calculate normalized coordinates correctly', () {
      final pose = poses.first;
      final landmark = pose.landmarks.first;

      final xNorm = landmark.xNorm(pose.imageWidth);
      final yNorm = landmark.yNorm(pose.imageHeight);

      expect(xNorm, greaterThanOrEqualTo(0));
      expect(xNorm, lessThanOrEqualTo(1.0));
      expect(yNorm, greaterThanOrEqualTo(0));
      expect(yNorm, lessThanOrEqualTo(1.0));

      // Verify calculation
      expect(
        (xNorm - landmark.x / pose.imageWidth).abs(),
        lessThan(0.0001),
      );
      expect(
        (yNorm - landmark.y / pose.imageHeight).abs(),
        lessThan(0.0001),
      );
    });

    test('should convert landmark to pixel Point', () {
      final pose = poses.first;
      final landmark = pose.landmarks.first;

      final point = landmark.toPixel(pose.imageWidth, pose.imageHeight);

      expect(point.x, equals(landmark.x.toInt()));
      expect(point.y, equals(landmark.y.toInt()));
    });

    test('should access bounding box properties', () {
      final pose = poses.first;
      final bbox = pose.boundingBox;

      expect(bbox.left, greaterThanOrEqualTo(0));
      expect(bbox.top, greaterThanOrEqualTo(0));
      expect(bbox.right, greaterThan(bbox.left));
      expect(bbox.bottom, greaterThan(bbox.top));

      // Bounding box should be within image
      expect(bbox.left, lessThanOrEqualTo(pose.imageWidth.toDouble()));
      expect(bbox.top, lessThanOrEqualTo(pose.imageHeight.toDouble()));
      expect(bbox.right, lessThanOrEqualTo(pose.imageWidth.toDouble()));
      expect(bbox.bottom, lessThanOrEqualTo(pose.imageHeight.toDouble()));
    });

    test('should have all 33 BlazePose landmarks', () {
      final pose = poses.first;
      expect(pose.landmarks.length, 33);

      // Verify we can access all landmark types
      final landmarkTypes = [
        PoseLandmarkType.nose,
        PoseLandmarkType.leftEyeInner,
        PoseLandmarkType.leftEye,
        PoseLandmarkType.leftEyeOuter,
        PoseLandmarkType.rightEyeInner,
        PoseLandmarkType.rightEye,
        PoseLandmarkType.rightEyeOuter,
        PoseLandmarkType.leftEar,
        PoseLandmarkType.rightEar,
        PoseLandmarkType.mouthLeft,
        PoseLandmarkType.mouthRight,
        PoseLandmarkType.leftShoulder,
        PoseLandmarkType.rightShoulder,
        PoseLandmarkType.leftElbow,
        PoseLandmarkType.rightElbow,
        PoseLandmarkType.leftWrist,
        PoseLandmarkType.rightWrist,
        PoseLandmarkType.leftPinky,
        PoseLandmarkType.rightPinky,
        PoseLandmarkType.leftIndex,
        PoseLandmarkType.rightIndex,
        PoseLandmarkType.leftThumb,
        PoseLandmarkType.rightThumb,
        PoseLandmarkType.leftHip,
        PoseLandmarkType.rightHip,
        PoseLandmarkType.leftKnee,
        PoseLandmarkType.rightKnee,
        PoseLandmarkType.leftAnkle,
        PoseLandmarkType.rightAnkle,
        PoseLandmarkType.leftHeel,
        PoseLandmarkType.rightHeel,
        PoseLandmarkType.leftFootIndex,
        PoseLandmarkType.rightFootIndex,
      ];

      for (final type in landmarkTypes) {
        final landmark = pose.getLandmark(type);
        expect(landmark, isNotNull, reason: 'Missing landmark: $type');
        expect(landmark!.type, type);
      }
    });
  });

  group('PoseDetector - Configuration Parameters', () {
    test('should respect detectorConf threshold', () async {
      // High confidence threshold
      final strictDetector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.9,
      );
      await strictDetector.initialize();

      // Low confidence threshold
      final lenientDetector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.3,
      );
      await lenientDetector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();

      final strictResults = await strictDetector.detect(bytes);
      final lenientResults = await lenientDetector.detect(bytes);

      // Lenient should detect same or more people
      expect(lenientResults.length, greaterThanOrEqualTo(strictResults.length));

      await strictDetector.dispose();
      await lenientDetector.dispose();
    });

    test('should respect maxDetections parameter', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        maxDetections: 1,
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      // Should not detect more than maxDetections
      expect(results.length, lessThanOrEqualTo(1));

      await detector.dispose();
    });

    test('should respect minLandmarkScore parameter', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        minLandmarkScore: 0.9, // Very high threshold
      );
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      // With high landmark score threshold, might get fewer results
      // or results without landmarks
      if (results.isNotEmpty) {
        for (final pose in results) {
          if (pose.hasLandmarks) {
            // If landmarks exist, they passed the quality threshold
            expect(pose.landmarks.length, 33);
          }
        }
      }

      await detector.dispose();
    });
  });

  group('PoseDetector - Multiple Images', () {
    test('should process multiple images sequentially', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final images = [
        'assets/samples/pose1.jpg',
        'assets/samples/pose2.jpg',
        'assets/samples/pose3.jpg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Pose> results = await detector.detect(bytes);

        expect(results, isNotEmpty, reason: 'Failed to detect in $imagePath');
      }

      await detector.dispose();
    });

    test('should handle different image sizes', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final images = [
        'assets/samples/pose4.jpg',
        'assets/samples/pose5.jpg',
        'assets/samples/pose6.jpg',
        'assets/samples/pose7.jpg',
      ];

      for (final imagePath in images) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();
        final List<Pose> results = await detector.detect(bytes);

        // Should work regardless of image size
        if (results.isNotEmpty) {
          for (final pose in results) {
            expect(pose.imageWidth, greaterThan(0));
            expect(pose.imageHeight, greaterThan(0));
          }
        }
      }

      await detector.dispose();
    });
  });

  group('PoseDetector - Edge Cases', () {
    test('should handle empty landmarks list in boxes mode', () async {
      final detector = PoseDetector(mode: PoseMode.boxes);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      for (final pose in results) {
        expect(pose.landmarks, isEmpty);
        expect(pose.hasLandmarks, false);

        // getLandmark should return null for any type
        expect(pose.getLandmark(PoseLandmarkType.nose), isNull);
      }

      await detector.dispose();
    });

    test('should handle 1x1 image', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final bytes = TestUtils.createDummyImageBytes();
      final List<Pose> results = await detector.detect(bytes);

      // Should not crash, but probably won't detect anything
      expect(results, isNotNull);

      await detector.dispose();
    });

    test('Pose.toString() should not crash', () async {
      final detector = PoseDetector(landmarkModel: PoseLandmarkModel.lite);
      await detector.initialize();

      final ByteData data = await rootBundle.load('assets/samples/pose1.jpg');
      final Uint8List bytes = data.buffer.asUint8List();
      final List<Pose> results = await detector.detect(bytes);

      expect(results, isNotEmpty);

      final poseString = results.first.toString();
      expect(poseString, isNotEmpty);
      expect(poseString, contains('Pose('));
      expect(poseString, contains('score='));
      expect(poseString, contains('landmarks='));

      await detector.dispose();
    });
  });
}
