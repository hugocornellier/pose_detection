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
import 'package:image/image.dart' as img;
import 'package:pose_detection_tflite/pose_detection_tflite.dart';

const int ITERATIONS = 20;
const List<String> SAMPLE_IMAGES = [
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

  void printResults(String testName) {
    print('\n=== $testName ===');
    print('Iterations: ${timings.length}');
    print('Image size: ${(imageSize / 1024).toStringAsFixed(1)} KB');
    print('Detections per frame: $detectionCount');
    print('Average: ${average.toStringAsFixed(2)} ms');
    print('Min: $min ms');
    print('Max: $max ms');
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
    print('\n📊 BENCHMARK_JSON_START:$filename');
    print(const JsonEncoder.withIndent('  ').convert(toJson()));
    print('📊 BENCHMARK_JSON_END:$filename');
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('PoseDetector - Performance Benchmarks', () {
    test('Benchmark heavy model with boxes and landmarks', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.heavy,
        performanceConfig: const PerformanceConfig.xnnpack(), // Enable XNNPACK
      );
      await detector.initialize();

      print('\n${'=' * 60}');
      print('BENCHMARK: Heavy Model (boxesAndLandmarks) with XNNPACK');
      print('=' * 60);

      final allStats = <BenchmarkStats>[];

      for (final imagePath in SAMPLE_IMAGES) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final List<int> timings = [];
        int detectionCount = 0;

        // Run iterations
        for (int i = 0; i < ITERATIONS; i++) {
          final stopwatch = Stopwatch()..start();
          final results = await detector.detect(bytes);
          stopwatch.stop();

          timings.add(stopwatch.elapsedMilliseconds);
          if (i == 0) detectionCount = results.length;
        }

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

      // Write results to file
      final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
      final benchmarkResults = BenchmarkResults(
        timestamp: timestamp,
        testName: 'Heavy Model (boxesAndLandmarks) with XNNPACK',
        configuration: {
          'model': 'heavy',
          'mode': 'boxesAndLandmarks',
          'iterations': ITERATIONS,
          'performance_config': 'xnnpack',
        },
        results: allStats,
      );
      benchmarkResults.printJson('benchmark_$timestamp.json');
    });
  });

  group('XNNPACK Delegate - Performance Comparison', () {
    test('Benchmark: Default (no delegate) vs XNNPACK', () async {
      print('\n${'=' * 60}');
      print('XNNPACK DELEGATE PERFORMANCE COMPARISON');
      print('=' * 60);

      // Use a single test image for comparison
      const testImage = 'assets/samples/pose1.jpg';
      final ByteData data = await rootBundle.load(testImage);
      final Uint8List bytes = data.buffer.asUint8List();

      // Create detector WITHOUT XNNPACK (baseline)
      final detectorDefault = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        interpreterPoolSize: 1,
        // performanceConfig: default (disabled)
      );
      await detectorDefault.initialize();

      // Create detector WITH XNNPACK
      final detectorXNNPack = PoseDetector(
        landmarkModel: PoseLandmarkModel.lite,
        interpreterPoolSize: 1,
        performanceConfig: const PerformanceConfig.xnnpack(), // AUTO THREADS
      );
      await detectorXNNPack.initialize();

      const int iterations = 15;

      // Warmup both detectors
      await detectorDefault.detect(bytes);
      await detectorXNNPack.detect(bytes);

      // Benchmark DEFAULT
      final timingsDefault = <int>[];
      for (int i = 0; i < iterations; i++) {
        final sw = Stopwatch()..start();
        await detectorDefault.detect(bytes);
        sw.stop();
        timingsDefault.add(sw.elapsedMilliseconds);
      }

      // Benchmark XNNPACK
      final timingsXNNPack = <int>[];
      for (int i = 0; i < iterations; i++) {
        final sw = Stopwatch()..start();
        await detectorXNNPack.detect(bytes);
        sw.stop();
        timingsXNNPack.add(sw.elapsedMilliseconds);
      }

      await detectorDefault.dispose();
      await detectorXNNPack.dispose();

      // Calculate statistics
      final avgDefault =
          timingsDefault.reduce((a, b) => a + b) / timingsDefault.length;
      final avgXNNPack =
          timingsXNNPack.reduce((a, b) => a + b) / timingsXNNPack.length;
      final speedup = avgDefault / avgXNNPack;

      // Print results
      print('\n═══════════════════════════════════════════');
      print('XNNPACK Delegate Benchmark Results');
      print('═══════════════════════════════════════════');
      print('Model: BlazePose Lite');
      print('Iterations: $iterations');
      print('Test image: $testImage');
      print('');
      print('Default (no delegate):');
      print('  Average: ${avgDefault.toStringAsFixed(1)} ms/frame');
      print('  Min: ${timingsDefault.reduce((a, b) => a < b ? a : b)} ms');
      print('  Max: ${timingsDefault.reduce((a, b) => a > b ? a : b)} ms');
      print('');
      print('XNNPACK (auto threads):');
      print('  Average: ${avgXNNPack.toStringAsFixed(1)} ms/frame');
      print('  Min: ${timingsXNNPack.reduce((a, b) => a < b ? a : b)} ms');
      print('  Max: ${timingsXNNPack.reduce((a, b) => a > b ? a : b)} ms');
      print('');
      print('Speedup: ${speedup.toStringAsFixed(2)}x faster with XNNPACK');
      print('═══════════════════════════════════════════');

      // XNNPACK should provide some speedup (platform-dependent, typically 1.5-5x)
      // On some platforms the speedup may be modest, so we use a conservative threshold
      expect(speedup, greaterThan(0.95),
          reason:
              'XNNPACK should not be slower than default (got ${speedup.toStringAsFixed(2)}x)');

      // Log a warning if speedup is less than expected
      if (speedup < 1.5) {
        print('⚠️  Note: XNNPACK speedup (${speedup.toStringAsFixed(2)}x) is less than typical 1.5-5x.');
        print('   This may be normal on some platforms or with integration test overhead.');
      }
    });

    test('Benchmark: XNNPACK with different thread counts', () async {
      print('\n${'=' * 60}');
      print('XNNPACK THREAD SCALING ANALYSIS');
      print('=' * 60);

      const testImage = 'assets/samples/pose1.jpg';
      final ByteData data = await rootBundle.load(testImage);
      final Uint8List bytes = data.buffer.asUint8List();

      final configs = [
        (1, const PerformanceConfig.xnnpack(numThreads: 1)),
        (2, const PerformanceConfig.xnnpack(numThreads: 2)),
        (4, const PerformanceConfig.xnnpack(numThreads: 4)),
      ];

      print('\n═══════════════════════════════════════════');
      print('XNNPACK Thread Scaling (BlazePose Lite)');
      print('═══════════════════════════════════════════');

      for (final (threads, config) in configs) {
        final detector = PoseDetector(
          landmarkModel: PoseLandmarkModel.lite,
          interpreterPoolSize: 1,
          performanceConfig: config,
        );
        await detector.initialize();

        // Warmup
        await detector.detect(bytes);

        // Benchmark
        const int iters = 10;
        final timings = <int>[];
        for (int i = 0; i < iters; i++) {
          final sw = Stopwatch()..start();
          await detector.detect(bytes);
          sw.stop();
          timings.add(sw.elapsedMilliseconds);
        }

        final avg = timings.reduce((a, b) => a + b) / timings.length;
        final minTime = timings.reduce((a, b) => a < b ? a : b);
        final maxTime = timings.reduce((a, b) => a > b ? a : b);

        print('Threads=$threads: avg=${avg.toStringAsFixed(1)}ms, '
            'min=${minTime}ms, max=${maxTime}ms');

        await detector.dispose();
      }
      print('═══════════════════════════════════════════');
    });

    test('Benchmark: XNNPACK across all model variants', () async {
      print('\n${'=' * 60}');
      print('XNNPACK PERFORMANCE ACROSS MODEL VARIANTS');
      print('=' * 60);

      const testImage = 'assets/samples/pose1.jpg';
      final ByteData data = await rootBundle.load(testImage);
      final Uint8List bytes = data.buffer.asUint8List();

      final models = [
        PoseLandmarkModel.lite,
        PoseLandmarkModel.full,
        PoseLandmarkModel.heavy,
      ];

      print('\n═══════════════════════════════════════════');
      print('Comparing Models with XNNPACK');
      print('═══════════════════════════════════════════');

      for (final model in models) {
        // Test with XNNPACK
        final detector = PoseDetector(
          landmarkModel: model,
          interpreterPoolSize: 1,
          performanceConfig: const PerformanceConfig.xnnpack(),
        );
        await detector.initialize();

        // Warmup
        await detector.detect(bytes);

        // Benchmark
        const int iters = 10;
        final timings = <int>[];
        for (int i = 0; i < iters; i++) {
          final sw = Stopwatch()..start();
          await detector.detect(bytes);
          sw.stop();
          timings.add(sw.elapsedMilliseconds);
        }

        final avg = timings.reduce((a, b) => a + b) / timings.length;

        print('${model.name.padRight(6)}: ${avg.toStringAsFixed(1)} ms/frame');

        await detector.dispose();
      }
      print('═══════════════════════════════════════════');
    });
  });
}
