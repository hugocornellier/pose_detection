import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';
import 'package:pose_detection/src/models/person_detector.dart';
import 'package:pose_detection/src/models/pose_landmark_model.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('Dart registration is callable', () {
    final instance = PoseDetectionDart();
    expect(instance, isA<PoseDetectionDart>());
    expect(() => PoseDetectionDart.registerWith(), returnsNormally);
  });

  group('PoseLandmarkModelRunner', () {
    test('reports pool configuration', () {
      final runner = PoseLandmarkModelRunner(poolSize: 7);
      expect(runner.poolSize, 7);
      expect(runner.isInitialized, isFalse);
    });
  });

  group('YoloV8PersonDetector', () {
    test('decodeOutputsForTest rejects outputs with too few channels', () {
      final detector = YoloV8PersonDetector();
      final List<dynamic> outputs = <dynamic>[
        <dynamic>[
          <List<double>>[List<double>.filled(10, 0.0)],
        ],
      ];

      expect(
        () => detector.decodeOutputsForTest(outputs),
        throwsA(isA<StateError>()),
      );
    });
  });
}
