// Comprehensive unit tests for PoseDetector.
//
// These tests cover:
// - Initialization and disposal
// - Error handling
// - Detection with real sample images via detect()
// - Different model variants (lite, full, heavy)
// - Different modes (boxes, boxesAndLandmarks)
// - Landmark and bounding box access
// - Configuration parameters
// - Edge cases
//
// NOTE: Most tests require TensorFlow Lite and OpenCV native libraries which
// are not available in the standard `flutter test` environment. To run all tests:
//
// - macOS: flutter test --platform=macos test/pose_detector_test.dart
// - Device: Run as integration tests on a physical device or emulator
//
// Tests that work in standard environment (no native libs required):
// - Initialization and disposal
// - Parameter validation

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

/// Helper to load an image asset and run detection, returning results.
Future<List<Pose>> _detectAsset(PoseDetector detector, String path) async {
  final ByteData data = await rootBundle.load(path);
  final cv.Mat mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);
  final results = await detector.detectFromMat(
    mat,
    imageWidth: mat.cols,
    imageHeight: mat.rows,
  );
  mat.dispose();
  return results;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('PoseDetector - Initialization and Disposal', () {
    test('should initialize successfully with default options', () async {
      final detector = PoseDetector(useNativePreprocessing: false);
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
        useNativePreprocessing: false,
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
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();
      expect(detector.isInitialized, true);

      // Re-initialize should work
      await detector.initialize();
      expect(detector.isInitialized, true);

      await detector.dispose();
    });

    test('should handle multiple dispose calls', () async {
      final detector = PoseDetector(useNativePreprocessing: false);
      await detector.initialize();
      await detector.dispose();
      expect(detector.isInitialized, false);

      // Second dispose should not throw
      await detector.dispose();
      expect(detector.isInitialized, false);
    });
  });

  group('PoseDetector - Error Handling', () {
    test(
      'should throw StateError when detect() called before initialize',
      () async {
        final detector = PoseDetector(useNativePreprocessing: false);
        final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

        expect(
          () => detector.detectFromMat(mat, imageWidth: 100, imageHeight: 100),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('not initialized'),
            ),
          ),
        );
        mat.dispose();
      },
    );
  });

  group('PoseDetector - detect() with real images', () {
    test(
      'should detect people in pose1.jpg with boxesAndLandmarks mode',
      () async {
        final detector = PoseDetector(
          mode: PoseMode.boxesAndLandmarks,
          landmarkModel: PoseLandmarkModel.lite,
          useNativePreprocessing: false,
        );
        await detector.initialize();

        final List<Pose> results =
            await _detectAsset(detector, 'assets/samples/pose1.jpg');

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
      },
    );

    test('should detect people in pose2.jpg', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose2.jpg');

      expect(results, isNotEmpty);
      await detector.dispose();
    });

    test('should detect people in pose3.jpg', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose3.jpg');

      expect(results, isNotEmpty);
      await detector.dispose();
    });

    test('should detect people with boxes-only mode', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxes,
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

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

  group('PoseDetector - Different Model Variants', () {
    test('should work with lite model', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    test('should work with full model', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.full,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });

    test('should work with heavy model', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.heavy,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      expect(results, isNotEmpty);
      expect(results.first.hasLandmarks, true);

      await detector.dispose();
    });
  });

  group('PoseDetector - Landmark and BoundingBox Access', () {
    late PoseDetector detector;
    late List<Pose> poses;

    setUpAll(() async {
      detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      poses = await _detectAsset(detector, 'assets/samples/pose1.jpg');
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
      expect((xNorm - landmark.x / pose.imageWidth).abs(), lessThan(0.0001));
      expect((yNorm - landmark.y / pose.imageHeight).abs(), lessThan(0.0001));
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
        useNativePreprocessing: false,
      );
      await strictDetector.initialize();

      // Low confidence threshold
      final lenientDetector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        detectorConf: 0.3,
        useNativePreprocessing: false,
      );
      await lenientDetector.initialize();

      final strictResults =
          await _detectAsset(strictDetector, 'assets/samples/pose1.jpg');
      final lenientResults =
          await _detectAsset(lenientDetector, 'assets/samples/pose1.jpg');

      // Lenient should detect same or more people
      expect(lenientResults.length, greaterThanOrEqualTo(strictResults.length));

      await strictDetector.dispose();
      await lenientDetector.dispose();
    });

    test('should respect maxDetections parameter', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        maxDetections: 1,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      // Should not detect more than maxDetections
      expect(results.length, lessThanOrEqualTo(1));

      await detector.dispose();
    });

    test('should respect minLandmarkScore parameter', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        minLandmarkScore: 0.9, // Very high threshold
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      // With high landmark score threshold, might get fewer results
      if (results.isNotEmpty) {
        for (final pose in results) {
          if (pose.hasLandmarks) {
            expect(pose.landmarks.length, 33);
          }
        }
      }

      await detector.dispose();
    });
  });

  group('PoseDetector - Multiple Images', () {
    test('should process multiple images sequentially', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final images = [
        'assets/samples/pose1.jpg',
        'assets/samples/pose2.jpg',
        'assets/samples/pose3.jpg',
      ];

      for (final imagePath in images) {
        final List<Pose> results = await _detectAsset(detector, imagePath);
        expect(results, isNotEmpty, reason: 'Failed to detect in $imagePath');
      }

      await detector.dispose();
    });

    test('should handle different image sizes', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final images = [
        'assets/samples/pose4.jpg',
        'assets/samples/pose5.jpg',
        'assets/samples/pose6.jpg',
        'assets/samples/pose7.jpg',
      ];

      for (final imagePath in images) {
        final List<Pose> results = await _detectAsset(detector, imagePath);

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

  group('PoseDetector - Sample Expected Counts', () {
    late PoseDetector detector;

    setUp(() async {
      detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.heavy,
        mode: PoseMode.boxesAndLandmarks,
        detectorConf: 0.6,
        detectorIou: 0.4,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        useNativePreprocessing: false,
      );
      await detector.initialize();
    });

    tearDown(() async {
      await detector.dispose();
    });

    test('sample images yield expected pose counts', () async {
      final expectedCounts = <String, int>{
        'assets/samples/multi1.jpg': 3,
        'assets/samples/pose1.jpg': 1,
        'assets/samples/pose2.jpg': 1,
        'assets/samples/pose3.jpg': 1,
        'assets/samples/pose4.jpg': 1,
        'assets/samples/pose5.jpg': 1,
        'assets/samples/pose6.jpg': 1,
        'assets/samples/pose7.jpg': 1,
      };

      for (final entry in expectedCounts.entries) {
        final results = await _detectAsset(detector, entry.key);

        expect(
          results.length,
          entry.value,
          reason: 'Unexpected pose count for ${entry.key}',
        );
      }
    });
  });

  group('PoseDetector - Edge Cases', () {
    test('should handle empty landmarks list in boxes mode', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxes,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

      for (final pose in results) {
        expect(pose.landmarks, isEmpty);
        expect(pose.hasLandmarks, false);

        // getLandmark should return null for any type
        expect(pose.getLandmark(PoseLandmarkType.nose), isNull);
      }

      await detector.dispose();
    });

    test('Pose.toString() should not crash', () async {
      final detector = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        useNativePreprocessing: false,
      );
      await detector.initialize();

      final List<Pose> results =
          await _detectAsset(detector, 'assets/samples/pose1.jpg');

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
