import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';
// In-package import of a non-exported helper, purely to unit-test the raw
// tensor -> quantized mask decode in isolation.
import 'package:pose_detection/src/util/pose_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('SegmentationMask', () {
    // 4x4 mask: left half background (0), right half person (255), covering the
    // image rectangle (10, 20) .. (10+40, 20+40).
    SegmentationMask makeMask() {
      final Uint8List c = Uint8List(4 * 4);
      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
          c[y * 4 + x] = x >= 2 ? 255 : 0;
        }
      }
      return SegmentationMask(
        width: 4,
        height: 4,
        confidences: c,
        imageLeft: 10,
        imageTop: 20,
        imageWidth: 40,
        imageHeight: 40,
      );
    }

    test('confidenceAt maps image space to mask pixels', () {
      final m = makeMask();
      // Left region of the crop -> background column.
      expect(m.confidenceAt(15, 40), 0.0);
      // Right region of the crop -> person column.
      expect(m.confidenceAt(45, 40), 1.0);
      // The 2/4 split falls at imageLeft + 0.5*imageWidth = 30.
      expect(m.confidenceAt(29, 40), 0.0);
      expect(m.confidenceAt(31, 40), 1.0);
    });

    test('confidenceAt returns 0 outside the mask region', () {
      final m = makeMask();
      expect(m.confidenceAt(9, 40), 0.0); // left of region
      expect(m.confidenceAt(51, 40), 0.0); // right of region
      expect(m.confidenceAt(30, 19), 0.0); // above region
      expect(m.confidenceAt(30, 61), 0.0); // below region
    });

    test('toRgbaBytes tints color and uses confidence as alpha', () {
      final m = makeMask();
      final rgba = m.toRgbaBytes(r: 10, g: 20, b: 30);
      expect(rgba.length, 4 * 4 * 4);
      // Pixel 0 is background (alpha 0); pixel 2 is person (alpha 255).
      expect(rgba.sublist(0, 4), [10, 20, 30, 0]);
      expect(rgba.sublist(2 * 4, 2 * 4 + 4), [10, 20, 30, 255]);
    });

    test('toMap/fromMap round-trips buffer and geometry', () {
      final m = makeMask();
      final restored = SegmentationMask.fromMap(
        Map<String, dynamic>.from(m.toMap()),
      );
      expect(restored.width, m.width);
      expect(restored.height, m.height);
      expect(restored.confidences, m.confidences);
      expect(restored.imageLeft, m.imageLeft);
      expect(restored.imageTop, m.imageTop);
      expect(restored.imageWidth, m.imageWidth);
      expect(restored.imageHeight, m.imageHeight);
    });

    test('Pose.toMap/fromMap preserves an attached mask', () {
      final pose = Pose(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.9,
        landmarks: const [],
        imageWidth: 100,
        imageHeight: 100,
        segmentationMask: makeMask(),
      );
      final restored = Pose.fromMap(Map<String, dynamic>.from(pose.toMap()));
      expect(restored.segmentationMask, isNotNull);
      expect(restored.segmentationMask!.confidences, makeMask().confidences);
    });

    test('Pose without a mask serializes to null', () {
      final pose = Pose(
        boundingBox: BoundingBox.ltrb(0, 0, 1, 1),
        score: 0.5,
        landmarks: const [],
        imageWidth: 1,
        imageHeight: 1,
      );
      final restored = Pose.fromMap(Map<String, dynamic>.from(pose.toMap()));
      expect(restored.segmentationMask, isNull);
    });
  });

  group('decodeSegmentationMask', () {
    test('applies sigmoid and quantizes logits to 0-255', () {
      final logits = Float32List.fromList([100.0, -100.0, 0.0, 100.0]);
      final mask = decodeSegmentationMask(logits, width: 2, height: 2);
      expect(mask.length, 4);
      expect(mask[0], 255); // sigmoid(+inf) -> 1
      expect(mask[1], 0); // sigmoid(-inf) -> 0
      expect(mask[2], 128); // sigmoid(0) = 0.5 -> round(127.5)
      expect(mask[3], 255);
    });
  });

  group('SegmentationMask end-to-end', () {
    final String root = Directory.current.path;
    Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

    test(
      'enableSegmentation populates Pose.segmentationMask on a real image',
      () async {
        final detector = PoseDetector();
        await detector.initializeFromBuffers(
          yoloBytes: load('$root/assets/models/yolov8n_float32.tflite'),
          landmarkBytes: load('$root/assets/models/pose_landmark_lite.tflite'),
          landmarkModel: PoseLandmarkModel.lite,
          enableSegmentation: true,
        );

        final poses = await detector.detect(
          load('$root/example_web/assets/samples/pose1.jpg'),
        );

        expect(poses, isNotEmpty);
        final withLandmarks = poses.where((p) => p.hasLandmarks).toList();
        expect(withLandmarks, isNotEmpty);

        final mask = withLandmarks.first.segmentationMask;
        expect(mask, isNotNull, reason: 'mask should be present when enabled');
        expect(mask!.width, 256);
        expect(mask.height, 256);
        expect(mask.confidences.length, 256 * 256);
        // A real person crop should contain some person pixels and some
        // background, i.e. the mask is not uniformly one value.
        final distinct = mask.confidences.toSet();
        expect(distinct.length, greaterThan(1));

        await detector.dispose();
      },
    );

    test('segmentation is off by default (mask stays null)', () async {
      final detector = PoseDetector();
      await detector.initializeFromBuffers(
        yoloBytes: load('$root/assets/models/yolov8n_float32.tflite'),
        landmarkBytes: load('$root/assets/models/pose_landmark_lite.tflite'),
        landmarkModel: PoseLandmarkModel.lite,
      );

      final poses = await detector.detect(
        load('$root/example_web/assets/samples/pose1.jpg'),
      );
      expect(poses.every((p) => p.segmentationMask == null), isTrue);

      await detector.dispose();
    });
  });
}
