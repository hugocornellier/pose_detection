// ignore_for_file: avoid_print

// Performance benchmark tests for PoseDetector.
//
// This test suite measures inference performance by running multiple iterations
// on each sample image and logging timing statistics.
//
// Run with:
// flutter test integration_test/pose_detector_benchmark_test.dart

import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';

const int iterations = 20;
const int warmupIterations = 3;
const List<String> sampleImages = [
  'assets/samples/pose1.jpg',
  'assets/samples/pose2.jpg',
  'assets/samples/pose3.jpg',
  'assets/samples/pose4.jpg',
  'assets/samples/pose5.jpg',
  'assets/samples/pose6.jpg',
  'assets/samples/pose7.jpg',
];

class BenchmarkStats {
  final String imagePath;
  final List<int> timings;
  final int imageSize;
  final int detectionCount;

  BenchmarkStats({
    required this.imagePath,
    required this.timings,
    required this.imageSize,
    required this.detectionCount,
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
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('PoseDetector - Performance Benchmarks', () {
    test(
      'Benchmark heavy model with XNNPACK',
      timeout: const Timeout(Duration(minutes: 10)),
      () async {
        final detector = PoseDetector(
          mode: PoseMode.boxesAndLandmarks,
          landmarkModel: PoseLandmarkModel.heavy,
          performanceConfig: const PerformanceConfig.xnnpack(),
        );
        await detector.initialize();

        print('\n${'=' * 60}');
        print('BENCHMARK: Heavy Model (XNNPACK, pool=1 forced)');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          final int imageWidth = mat.cols;
          final int imageHeight = mat.rows;

          final List<int> timings = [];
          int detectionCount = 0;

          // Warmup
          for (int i = 0; i < warmupIterations; i++) {
            final results = await detector.detectFromMat(
              mat,
              imageWidth: imageWidth,
              imageHeight: imageHeight,
            );
            if (i == 0) detectionCount = results.length;
          }

          // Timed iterations
          for (int i = 0; i < iterations; i++) {
            final stopwatch = Stopwatch()..start();
            await detector.detectFromMat(
              mat,
              imageWidth: imageWidth,
              imageHeight: imageHeight,
            );
            stopwatch.stop();
            timings.add(stopwatch.elapsedMilliseconds);
          }

          mat.dispose();

          final stats = BenchmarkStats(
            imagePath: imagePath,
            timings: timings,
            imageSize: bytes.length,
            detectionCount: detectionCount,
          );
          stats.printResults(imagePath);
          allStats.add(stats);
        }

        await detector.dispose();

        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Heavy Model (XNNPACK)',
          configuration: {
            'model': 'heavy',
            'mode': 'boxesAndLandmarks',
            'warmup_iterations': warmupIterations,
            'timed_iterations': iterations,
            'interpreter_pool_size': 1,
            'xnnpack_threads': 'auto',
          },
          results: allStats,
        );
        benchmarkResults.printJson('benchmark_xnnpack_$timestamp.json');
      },
    );
  });
}
