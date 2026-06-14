// ignore_for_file: avoid_print

// A/B benchmark: interpreter (XNNPACK) vs CompiledModel (GPU|CPU) for the full
// PoseDetector pipeline, across all three landmark variants.
//
// Answers the headline question: does useCompiledModel speed up the end-to-end
// detect() pipeline, and for which models?
//
// Runs on -d macos for the real Metal GPU (the iOS Simulator has no GPU/ANE):
//   flutter test integration_test/pose_compiledmodel_ab_test.dart -d macos

import 'dart:convert';
import 'dart:io' show Platform;
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';

const int iterations = 30;
const int warmupIterations = 8;
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

Future<List<int>> _bench({
  required PoseLandmarkModel model,
  required bool useCompiledModel,
  required PerformanceConfig performanceConfig,
  required List<cv.Mat> mats,
}) async {
  final detector = PoseDetector();
  await detector.initialize(
    mode: PoseMode.boxesAndLandmarks,
    landmarkModel: model,
    performanceConfig: performanceConfig,
    useCompiledModel: useCompiledModel,
  );

  final all = <int>[];
  for (final mat in mats) {
    for (int i = 0; i < warmupIterations; i++) {
      await detector.detectFromMat(mat);
    }
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      await detector.detectFromMat(mat);
      sw.stop();
      all.add(sw.elapsedMicroseconds);
    }
  }

  await detector.dispose();
  return all;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final isMacOS = Platform.isMacOS;

  const models = <String, PoseLandmarkModel>{
    'lite': PoseLandmarkModel.lite,
    'full': PoseLandmarkModel.full,
    'heavy': PoseLandmarkModel.heavy,
  };

  final summary = <String>[];
  final jsonRows = <Map<String, dynamic>>[];

  group('PoseDetector - CompiledModel A/B (full pipeline, -d macos)', () {
    models.forEach((name, model) {
      test(
        '$name : interpreter(xnnpack) vs compiled(gpu|cpu)',
        timeout: const Timeout(Duration(minutes: 15)),
        skip: isMacOS ? false : 'macOS-only (real Metal GPU)',
        () async {
          final mats = <cv.Mat>[];
          for (final p in sampleImages) {
            final d = await rootBundle.load(p);
            mats.add(cv.imdecode(d.buffer.asUint8List(), cv.IMREAD_COLOR));
          }

          final interp = await _bench(
            model: model,
            useCompiledModel: false,
            performanceConfig: const PerformanceConfig.xnnpack(),
            mats: mats,
          );
          final compiled = await _bench(
            model: model,
            useCompiledModel: true,
            performanceConfig: const PerformanceConfig(),
            mats: mats,
          );

          for (final m in mats) {
            m.dispose();
          }

          final double iMs = _p50(interp) / 1000.0;
          final double cMs = _p50(compiled) / 1000.0;
          final double speedup = iMs / cMs;
          final line =
              '${name.padRight(6)} '
              'interp(xnn) p50=${iMs.toStringAsFixed(2).padLeft(7)}ms   '
              'compiled(gpu|cpu) p50=${cMs.toStringAsFixed(2).padLeft(7)}ms   '
              'speedup=${speedup.toStringAsFixed(2)}x';
          summary.add(line);
          print('\n>>> $line\n');

          jsonRows.add({
            'model': name,
            'interpreter_xnnpack': {
              'p50_ms': iMs,
              'p95_ms': _p95(interp) / 1000.0,
              'mean_ms': _mean(interp) / 1000.0,
              'std_ms': _std(interp) / 1000.0,
            },
            'compiled_gpu_cpu': {
              'p50_ms': cMs,
              'p95_ms': _p95(compiled) / 1000.0,
              'mean_ms': _mean(compiled) / 1000.0,
              'std_ms': _std(compiled) / 1000.0,
            },
            'speedup_x': speedup,
            'n_per_config': interp.length,
          });
        },
      );
    });

    tearDownAll(() {
      print('\n${'=' * 78}');
      print('COMPILEDMODEL A/B SUMMARY (pose_detection, -d macos)');
      print('=' * 78);
      for (final l in summary) {
        print(l);
      }
      print('=' * 78);

      final ts = DateTime.now().toIso8601String().replaceAll(':', '-');
      final file = 'compiledmodel_ab_$ts.json';
      print('\n BENCHMARK_JSON_START:$file');
      print(
        const JsonEncoder.withIndent(
          '  ',
        ).convert({'test': 'compiledmodel_ab', 'rows': jsonRows}),
      );
      print(' BENCHMARK_JSON_END:$file');
    });
  });
}
