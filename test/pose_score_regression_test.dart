import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';
import 'package:pose_detection/src/models/pose_landmark_model_native.dart';
import 'package:pose_detection/src/util/pose_helpers.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('pose parser preserves already-activated model scores', () {
    final landmarks = Float32List(33 * 5);

    for (final score in <double>[0.0, 0.35, 0.9]) {
      final parsed = parsePoseLandmarksFlat(
        landmarks,
        Float32List.fromList(<double>[score]),
      );

      expect(parsed.score, closeTo(score, 1e-6));
    }
  });

  const variants = <PoseLandmarkModel, String>{
    PoseLandmarkModel.lite: 'assets/models/pose_landmark_lite.tflite',
    PoseLandmarkModel.full: 'assets/models/pose_landmark_full.tflite',
    PoseLandmarkModel.heavy: 'assets/models/pose_landmark_heavy.tflite',
  };

  for (final entry in variants.entries) {
    test(
      '${entry.key.name} exposes its blank-input probability directly',
      () async {
        final runner = PoseLandmarkModelRunner();
        final blank = cv.Mat.zeros(256, 256, cv.MatType.CV_8UC3);

        try {
          await runner.initializeFromBuffer(
            await File(entry.value).readAsBytes(),
            performanceConfig: const PerformanceConfig(),
          );
          final result = await runner.run(blank);

          expect(result.score, inInclusiveRange(0.0, 1.0));
          expect(
            result.score,
            lessThan(0.5),
            reason:
                'The blank crop should fail the default landmark gate. A '
                'second sigmoid maps every non-negative model probability to '
                '0.5 or higher.',
          );
        } finally {
          blank.dispose();
          await runner.dispose();
        }
      },
    );
  }
}
