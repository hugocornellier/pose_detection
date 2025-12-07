import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pose_detection_tflite/pose_detection_tflite.dart';
import 'package:pose_detection_tflite/src/image_utils.dart';
import 'package:pose_detection_tflite/src/person_detector.dart';
import 'package:pose_detection_tflite/src/pose_landmark_model.dart';
import 'package:tflite_flutter_custom/tflite_flutter.dart';

class _StubTensor implements Tensor {
  _StubTensor(this.shapeValues);

  final List<int> shapeValues;

  @override
  List<int> get shape => shapeValues;

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _StubInterpreter implements Interpreter {
  _StubInterpreter({required this.inputTensors});

  final List<Tensor> inputTensors;
  List<Tensor> outputTensors = const <Tensor>[];
  void Function(List<Object> inputs, Map<int, Object> outputs)? onRun;

  @override
  List<Tensor> getInputTensors() => inputTensors;

  @override
  Tensor getInputTensor(int index) => inputTensors[index];

  @override
  List<Tensor> getOutputTensors() => outputTensors;

  @override
  void runForMultipleInputs(
    List<Object> inputs,
    Map<int, Object> outputs,
  ) {
    final callback = onRun;
    if (callback != null) {
      callback(inputs, outputs);
    }
  }

  @override
  void close() {}

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

List<double> _buildYoloRow() {
  final List<double> row = List<double>.filled(85, 0.0);
  row[0] = 0.5; // cx
  row[1] = 0.5; // cy
  row[2] = 0.4; // w
  row[3] = 0.4; // h
  row[4] = 4.0; // objectness logit
  row[5] = 2.0; // class-0 logit stays the best
  return row;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('PoseDetector returns empty when image decoding yields null', () async {
    final detector = PoseDetector();
    PoseDetector.imageDecoderOverride = (_) => null;

    await detector.initialize();
    final List<Pose> results = await detector.detect(const <int>[1, 2, 3]);

    try {
      expect(results, isEmpty);
    } finally {
      PoseDetector.imageDecoderOverride = null;
      await detector.dispose();
    }
  });

  test('letterbox throws when reuseCanvas dimensions are wrong', () {
    final img.Image src = img.Image(width: 2, height: 2);
    final img.Image reuse = img.Image(width: 1, height: 1);

    expect(
      () => ImageUtils.letterbox(src, 4, 4, <double>[], <int>[],
          reuseCanvas: reuse),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('Dart registration is callable', () {
    final instance = PoseDetectionTfliteDart();
    expect(instance, isA<PoseDetectionTfliteDart>());
    expect(() => PoseDetectionTfliteDart.registerWith(), returnsNormally);
  });

  group('PoseLandmarkModelRunner', () {
    test('reports pool configuration and guard rails', () async {
      final runner = PoseLandmarkModelRunner(poolSize: 7);
      expect(runner.poolSize, 7);
      expect(runner.isInitialized, isFalse);

      expect(
        () => runner.run(img.Image(width: 1, height: 1)),
        throwsA(isA<StateError>()),
      );
    });

    test('ensureTFLiteLoaded honors env override', () async {
      PoseLandmarkModelRunner.resetNativeLibForTest();
      await PoseLandmarkModelRunner.ensureTFLiteLoaded(
        env: <String, String>{'POSE_TFLITE_LIB': '/usr/lib/libSystem.B.dylib'},
        platformOverride: 'macos',
      );
      expect(PoseLandmarkModelRunner.nativeLibForTest(), isNotNull);
    });

    test('ensureTFLiteLoaded falls back for other platforms', () async {
      PoseLandmarkModelRunner.resetNativeLibForTest();
      await PoseLandmarkModelRunner.ensureTFLiteLoaded(
        env: const <String, String>{},
        platformOverride: 'other',
      );
      expect(PoseLandmarkModelRunner.nativeLibForTest(), isNotNull);
    });

    test('ensureTFLiteLoaded builds candidate lists for Windows/Linux',
        () async {
      PoseLandmarkModelRunner.resetNativeLibForTest();
      await PoseLandmarkModelRunner.ensureTFLiteLoaded(
        env: const <String, String>{},
        platformOverride: 'windows',
      );
      expect(PoseLandmarkModelRunner.nativeLibForTest(), isNull);

      PoseLandmarkModelRunner.resetNativeLibForTest();
      await PoseLandmarkModelRunner.ensureTFLiteLoaded(
        env: const <String, String>{},
        platformOverride: 'linux',
      );
      expect(PoseLandmarkModelRunner.nativeLibForTest(), isNull);
    });
  });

  group('YoloV8PersonDetector', () {
    test('detectOnImage throws before initialization', () {
      final detector = YoloV8PersonDetector();
      expect(detector.isInitialized, isFalse);

      expect(
        () => detector.detectOnImage(img.Image(width: 1, height: 1)),
        throwsA(isA<StateError>()),
      );
    });

    test('runs through interpreter branch and resizes input buffer', () async {
      final _StubInterpreter interpreter = _StubInterpreter(
        inputTensors: <Tensor>[
          _StubTensor(<int>[1, 2, 2, 3])
        ],
      );
      final detector = YoloV8PersonDetector();
      detector.debugConfigureForTest(
        inputWidth: 2,
        inputHeight: 2,
        outputShapes: const <List<int>>[
          <int>[1, 1, 85],
        ],
        interpreter: interpreter,
        inputBuffer: Float32List(1), // force resize path
      );
      expect(detector.isInitialized, isTrue);

      interpreter.onRun = (_, Map<int, Object> outputs) {
        final List<List<List<double>>> out0 =
            outputs[0] as List<List<List<double>>>;
        out0[0][0] = _buildYoloRow();
      };

      final List<YoloDetection> dets =
          await detector.detectOnImage(img.Image(width: 2, height: 2));

      expect(dets, hasLength(1));
      expect(dets.first.score, greaterThan(0.4));
      expect(dets.first.cls, equals(YoloV8PersonDetector.cocoPersonClassId));
    });

    test('fails fast on undersized output tensors', () {
      final _StubInterpreter interpreter = _StubInterpreter(
        inputTensors: <Tensor>[
          _StubTensor(<int>[1, 1, 1, 3])
        ],
      );
      final detector = YoloV8PersonDetector();
      detector.debugConfigureForTest(
        inputWidth: 1,
        inputHeight: 1,
        outputShapes: const <List<int>>[
          <int>[1, 2], // triggers 2D allocation branch
          <int>[1], // triggers flat allocation branch
        ],
        interpreter: interpreter,
        inputBuffer: Float32List(1),
      );

      expect(
        () => detector.detectOnImage(img.Image(width: 1, height: 1)),
        throwsA(isA<TypeError>()),
      );
    });

    test('decodeOutputsForTest rejects outputs with too few channels', () {
      final detector = YoloV8PersonDetector();
      final List<dynamic> outputs = <dynamic>[
        <dynamic>[
          <List<double>>[
            List<double>.filled(10, 0.0),
          ],
        ]
      ];

      expect(
        () => detector.decodeOutputsForTest(outputs),
        throwsA(isA<StateError>()),
      );
    });
  });
}
