// ignore_for_file: avoid_print

// Performance benchmark tests for PoseDetector on Web/Chrome.
//
// Mirrors pose_detector_benchmark_test.dart but uses detector.detect(bytes)
// instead of cv.imdecode + detectFromMat, since opencv_dart is FFI-only and
// cannot run on web.
//
// Run with:
//   flutter drive \
//     --driver=test_driver/integration_test.dart \
//     --target=integration_test/pose_detector_web_benchmark_test.dart \
//     -d chrome

import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:pose_detection/pose_detection.dart';

const int iterations = 10;
const int warmupIterations = 2;
const List<String> sampleImages = [
  'assets/samples/pose1.jpg',
  'assets/samples/pose2.jpg',
  'assets/samples/pose3.jpg',
  'assets/samples/pose4.jpg',
  'assets/samples/pose5.jpg',
];

class StageAvgs {
  double decode = 0;
  double yoloPre = 0;
  double yoloInfer = 0;
  double lmPre = 0;
  double lmInfer = 0;
  double total = 0;
  int samples = 0;

  void add(Map<String, int> us) {
    decode += us['decode_us'] ?? 0;
    yoloPre += us['yolo_pre_us'] ?? 0;
    yoloInfer += us['yolo_infer_us'] ?? 0;
    lmPre += us['lm_pre_us'] ?? 0;
    lmInfer += us['lm_infer_us'] ?? 0;
    total += us['total_us'] ?? 0;
    samples++;
  }

  Map<String, double> toMs() {
    final n = samples == 0 ? 1 : samples;
    return {
      'decode_ms': double.parse((decode / n / 1000).toStringAsFixed(2)),
      'yolo_pre_ms': double.parse((yoloPre / n / 1000).toStringAsFixed(2)),
      'yolo_infer_ms': double.parse((yoloInfer / n / 1000).toStringAsFixed(2)),
      'lm_pre_ms': double.parse((lmPre / n / 1000).toStringAsFixed(2)),
      'lm_infer_ms': double.parse((lmInfer / n / 1000).toStringAsFixed(2)),
      'total_ms': double.parse((total / n / 1000).toStringAsFixed(2)),
    };
  }
}

class BenchmarkStats {
  final String imagePath;
  final List<int> timings;
  final int imageSize;
  final int detectionCount;
  final StageAvgs stages;

  BenchmarkStats({
    required this.imagePath,
    required this.timings,
    required this.imageSize,
    required this.detectionCount,
    required this.stages,
  });

  double get average => timings.reduce((a, b) => a + b) / timings.length;
  int get min => timings.reduce((a, b) => a < b ? a : b);
  int get max => timings.reduce((a, b) => a > b ? a : b);

  double get standardDeviation {
    final mean = average;
    final variance =
        timings.map((t) => pow(t - mean, 2)).reduce((a, b) => a + b) /
        timings.length;
    return sqrt(variance);
  }

  double _percentile(double p) {
    final sorted = List<int>.from(timings)..sort();
    final index = ((sorted.length - 1) * p).floor();
    return sorted[index].toDouble();
  }

  double get p50 => _percentile(0.50);
  double get p95 => _percentile(0.95);
  double get p99 => _percentile(0.99);

  void printResults(String testName) {
    print('\n=== $testName ===');
    print('Iterations: ${timings.length}');
    print('Image size: ${(imageSize / 1024).toStringAsFixed(1)} KB');
    print('Detections per frame: $detectionCount');
    print('Average: ${average.toStringAsFixed(2)} ms');
    print('Min: $min ms');
    print('Max: $max ms');
    print('P50: ${p50.toStringAsFixed(2)} ms');
    print('P95: ${p95.toStringAsFixed(2)} ms');
    print('P99: ${p99.toStringAsFixed(2)} ms');
    print('Std Dev: ${standardDeviation.toStringAsFixed(2)} ms');
    print('All times (ms): $timings');
  }

  Map<String, dynamic> toJson() => {
    'image_path': imagePath,
    'iterations': timings.length,
    'image_size_bytes': imageSize,
    'detections_per_frame': detectionCount,
    'average_ms': double.parse(average.toStringAsFixed(2)),
    'min_ms': min,
    'max_ms': max,
    'p50_ms': double.parse(p50.toStringAsFixed(2)),
    'p95_ms': double.parse(p95.toStringAsFixed(2)),
    'p99_ms': double.parse(p99.toStringAsFixed(2)),
    'std_dev_ms': double.parse(standardDeviation.toStringAsFixed(2)),
    'all_timings_ms': timings,
    'stage_avg': stages.toMs(),
  };
}

class BenchmarkResults {
  final String timestamp;
  final String testName;
  final Map<String, dynamic> configuration;
  final List<BenchmarkStats> results;

  BenchmarkResults({
    required this.timestamp,
    required this.testName,
    required this.configuration,
    required this.results,
  });

  Map<String, dynamic> toJson() => {
    'timestamp': timestamp,
    'test_name': testName,
    'configuration': configuration,
    'results': results.map((r) => r.toJson()).toList(),
  };

  void printJson(String filename) {
    print('\n BENCHMARK_JSON_START:$filename');
    print(const JsonEncoder.withIndent('  ').convert(toJson()));
    print(' BENCHMARK_JSON_END:$filename');
  }
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('PoseDetector Web - Performance Benchmarks', () {
    test(
      'Benchmark heavy model on Chrome',
      timeout: const Timeout(Duration(minutes: 20)),
      () async {
        // Toggle via env: BENCH_USE_LITERT=1 BENCH_LITERT_ACCEL=webgpu
        const bool useLiteRt = bool.fromEnvironment(
          'BENCH_USE_LITERT',
          defaultValue: false,
        );
        const String liteRtAccel = String.fromEnvironment(
          'BENCH_LITERT_ACCEL',
          defaultValue: 'wasm',
        );

        final detector = await PoseDetector.create(
          mode: PoseMode.boxesAndLandmarks,
          landmarkModel: PoseLandmarkModel.heavy,
          useLiteRt: useLiteRt,
          liteRtAccelerator: liteRtAccel,
        );
        // Enable per-stage timings (web build only).
        try {
          (detector as dynamic).debugTimings = true;
        } catch (_) {}

        print('\n${'=' * 60}');
        print('BENCHMARK: Heavy Model (Web/Chrome, detect(bytes))');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;
          final stages = StageAvgs();

          for (int i = 0; i < warmupIterations; i++) {
            final results = await detector.detect(bytes);
            if (i == 0) detectionCount = results.length;
          }

          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await detector.detect(bytes);
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
            try {
              final dynamic last = (detector as dynamic).lastTimings;
              if (last != null) {
                final Map<String, int> us = Map<String, int>.from(
                  last.toJsonUs() as Map,
                );
                stages.add(us);
              }
            } catch (_) {}
          }

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: detectionCount,
            stages: stages,
          );
          stats.printResults(imagePath);
          allStats.add(stats);
        }

        await detector.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Heavy Model (Web/Chrome)',
          configuration: {
            'platform': 'web',
            'model': 'heavy',
            'mode': 'boxesAndLandmarks',
            'warmup_iterations': warmupIterations,
            'timed_iterations': iterations,
            'detect_path': 'detect(Uint8List)',
            'use_litert': useLiteRt,
            'litert_accelerator': liteRtAccel,
          },
          results: allStats,
        );
        benchmarkResults.printJson('benchmark_web_$timestamp.json');
        binding.reportData = {
          'benchmark_web_$timestamp.json': benchmarkResults.toJson(),
        };
      },
    );
  });
}
