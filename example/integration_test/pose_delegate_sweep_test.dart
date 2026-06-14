// ignore_for_file: avoid_print

// Delegate-sweep benchmark for PoseDetector.
//
// Sweeps the four PerformanceMode delegates (disabled / xnnpack / gpu / coreml)
// across a light vs heavy landmark model and reports steady-state p50 latency.
//
// Runs on -d macos, which exercises the *real* Metal GPU and CoreML/ANE
// (unlike the iOS Simulator, where CoreML falls back to CPU and there is no
// Neural Engine). This is the apples-to-apples test that issue #11 lacked.
//
//   flutter test integration_test/pose_delegate_sweep_test.dart -d macos

import 'dart:io' show Platform;
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';

const int iterations = 20;
const int warmupIterations = 5;
const List<String> sampleImages = [
  'assets/samples/pose1.jpg',
  'assets/samples/pose3.jpg',
  'assets/samples/pose5.jpg',
];

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _p95(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.95).floor()].toDouble();
}

double _mean(List<int> t) => t.reduce((a, b) => a + b) / t.length;

double _std(List<int> t) {
  final m = _mean(t);
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

Future<List<int>> _benchModel({
  required PoseLandmarkModel model,
  required PerformanceConfig config,
}) async {
  final detector = PoseDetector();
  await detector.initialize(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: model,
    performanceConfig: config,
    // This sweep measures the interpreter-path PerformanceMode delegates, so
    // it must opt out of the now-default CompiledModel path.
    useCompiledModel: false,
  );

  final all = <int>[];
  for (final imagePath in sampleImages) {
    final data = await rootBundle.load(imagePath);
    final mat = cv.imdecode(data.buffer.asUint8List(), cv.IMREAD_COLOR);

    for (int i = 0; i < warmupIterations; i++) {
      await detector.detectFromMat(mat);
    }
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      await detector.detectFromMat(mat);
      sw.stop();
      all.add(sw.elapsedMilliseconds);
    }
    mat.dispose();
  }

  await detector.dispose();
  return all;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // Use dart:io Platform rather than defaultTargetPlatform: under
  // `flutter test`, defaultTargetPlatform is forced to android in debug
  // builds, which would skip the sweep even on a real macOS run.
  final isMacOS = Platform.isMacOS;

  const modes = <String, PerformanceConfig>{
    'disabled': PerformanceConfig(mode: PerformanceMode.disabled),
    'xnnpack': PerformanceConfig.xnnpack(),
    'gpu': PerformanceConfig.gpu(),
    'coreml': PerformanceConfig.coreml(),
  };

  const models = <String, PoseLandmarkModel>{
    'lite': PoseLandmarkModel.lite,
    'full': PoseLandmarkModel.full,
    'heavy': PoseLandmarkModel.heavy,
  };

  final summary = <String>[];

  group('PoseDetector - Delegate sweep (macOS real GPU/CoreML)', () {
    models.forEach((modelName, model) {
      modes.forEach((modeName, config) {
        test(
          '$modelName / $modeName',
          timeout: const Timeout(Duration(minutes: 10)),
          skip: isMacOS ? false : 'macOS-only delegate sweep',
          () async {
            final t = await _benchModel(model: model, config: config);
            final line =
                '${modelName.padRight(6)} ${modeName.padRight(9)} '
                'p50=${_p50(t).toStringAsFixed(1).padLeft(6)}ms  '
                'p95=${_p95(t).toStringAsFixed(1).padLeft(6)}ms  '
                'mean=${_mean(t).toStringAsFixed(1).padLeft(6)}ms  '
                'std=${_std(t).toStringAsFixed(1).padLeft(5)}ms  '
                '(n=${t.length})';
            summary.add(line);
            print('\n>>> $line\n');
          },
        );
      });
    });

    tearDownAll(() {
      print('\n${'=' * 78}');
      print('DELEGATE SWEEP SUMMARY (pose_detection, -d macos)');
      print('=' * 78);
      for (final l in summary) {
        print(l);
      }
      print('=' * 78);
    });
  });
}
