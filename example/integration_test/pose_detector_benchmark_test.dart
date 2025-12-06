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
    final variance = timings.map((t) => pow(t - mean, 2)).reduce((a, b) => a + b) / timings.length;
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
    test('Benchmark all sample images with lite model', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await detector.initialize();

      print('\n${'=' * 60}');
      print('BENCHMARK: Lite Model (boxesAndLandmarks)');
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
        testName: 'Lite Model (boxesAndLandmarks)',
        configuration: {
          'model': 'lite',
          'mode': 'boxesAndLandmarks',
          'iterations': ITERATIONS,
        },
        results: allStats,
      );
      benchmarkResults.printJson('benchmark_lite_model_$timestamp.json');
    });

    test('Benchmark all sample images with full model', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.full,
      );
      await detector.initialize();

      print('\n${'=' * 60}');
      print('BENCHMARK: Full Model (boxesAndLandmarks)');
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
        testName: 'Full Model (boxesAndLandmarks)',
        configuration: {
          'model': 'full',
          'mode': 'boxesAndLandmarks',
          'iterations': ITERATIONS,
        },
        results: allStats,
      );
      benchmarkResults.printJson('benchmark_full_model_$timestamp.json');
    });

    test('Benchmark all sample images with heavy model', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.heavy,
      );
      await detector.initialize();

      print('\n${'=' * 60}');
      print('BENCHMARK: Heavy Model (boxesAndLandmarks)');
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
        testName: 'Heavy Model (boxesAndLandmarks)',
        configuration: {
          'model': 'heavy',
          'mode': 'boxesAndLandmarks',
          'iterations': ITERATIONS,
        },
        results: allStats,
      );
      benchmarkResults.printJson('benchmark_heavy_model_$timestamp.json');
    });

    test('Benchmark boxes-only mode vs full mode (lite model)', () async {
      print('\n${'=' * 60}');
      print('BENCHMARK: Boxes-Only vs BoxesAndLandmarks (Lite Model)');
      print('=' * 60);

      final boxesStats = <BenchmarkStats>[];
      final landmarksStats = <BenchmarkStats>[];

      // Test boxes-only mode
      final boxesDetector = PoseDetector(
        mode: PoseMode.boxes,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await boxesDetector.initialize();

      print('\n--- BOXES-ONLY MODE ---');
      for (final imagePath in SAMPLE_IMAGES) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final List<int> timings = [];
        int detectionCount = 0;

        for (int i = 0; i < ITERATIONS; i++) {
          final stopwatch = Stopwatch()..start();
          final results = await boxesDetector.detect(bytes);
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
        boxesStats.add(stats);
      }

      await boxesDetector.dispose();

      // Test boxes + landmarks mode
      final fullDetector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await fullDetector.initialize();

      print('\n--- BOXES-AND-LANDMARKS MODE ---');
      for (final imagePath in SAMPLE_IMAGES) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        final List<int> timings = [];
        int detectionCount = 0;

        for (int i = 0; i < ITERATIONS; i++) {
          final stopwatch = Stopwatch()..start();
          final results = await fullDetector.detect(bytes);
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
        landmarksStats.add(stats);
      }

      await fullDetector.dispose();

      // Write results to files
      final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');

      final boxesResults = BenchmarkResults(
        timestamp: timestamp,
        testName: 'Boxes-Only Mode (Lite Model)',
        configuration: {
          'model': 'lite',
          'mode': 'boxes',
          'iterations': ITERATIONS,
        },
        results: boxesStats,
      );
      boxesResults.printJson('benchmark_boxes_only_$timestamp.json');

      final landmarksResults = BenchmarkResults(
        timestamp: timestamp,
        testName: 'BoxesAndLandmarks Mode (Lite Model)',
        configuration: {
          'model': 'lite',
          'mode': 'boxesAndLandmarks',
          'iterations': ITERATIONS,
        },
        results: landmarksStats,
      );
      landmarksResults.printJson('benchmark_boxes_landmarks_$timestamp.json');
    });

    test('Benchmark different interpreter pool sizes (lite model)', () async {
      final poolSizes = [1, 3, 5];

      print('\n${'=' * 60}');
      print('BENCHMARK: Different Interpreter Pool Sizes (Lite Model)');
      print('=' * 60);

      final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');

      for (final poolSize in poolSizes) {
        final detector = PoseDetector(
          mode: PoseMode.boxesAndLandmarks,
          landmarkModel: PoseLandmarkModel.lite,
          interpreterPoolSize: poolSize,
        );
        await detector.initialize();

        print('\n--- POOL SIZE: $poolSize ---');

        final allStats = <BenchmarkStats>[];

        for (final imagePath in SAMPLE_IMAGES) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;

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

        // Write results for this pool size
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Pool Size $poolSize (Lite Model)',
          configuration: {
            'model': 'lite',
            'mode': 'boxesAndLandmarks',
            'pool_size': poolSize,
            'iterations': ITERATIONS,
          },
          results: allStats,
        );
        benchmarkResults.printJson('benchmark_pool_size_${poolSize}_$timestamp.json');
      }
    });

    test('Benchmark detect() vs detectOnImage() (lite model)', () async {
      final detector = PoseDetector(
        mode: PoseMode.boxesAndLandmarks,
        landmarkModel: PoseLandmarkModel.lite,
      );
      await detector.initialize();

      print('\n${'=' * 60}');
      print('BENCHMARK: detect() vs detectOnImage() (Lite Model)');
      print('=' * 60);

      final detectStats = <BenchmarkStats>[];
      final detectOnImageStats = <BenchmarkStats>[];

      for (final imagePath in SAMPLE_IMAGES) {
        final ByteData data = await rootBundle.load(imagePath);
        final Uint8List bytes = data.buffer.asUint8List();

        // Benchmark detect() - includes decoding overhead
        print('\n--- detect() method ---');
        final detectTimings = <int>[];
        int detectionCount = 0;

        for (int i = 0; i < ITERATIONS; i++) {
          final stopwatch = Stopwatch()..start();
          final results = await detector.detect(bytes);
          stopwatch.stop();

          detectTimings.add(stopwatch.elapsedMilliseconds);
          if (i == 0) detectionCount = results.length;
        }

        final detectStat = BenchmarkStats(
          imagePath: imagePath,
          timings: detectTimings,
          imageSize: bytes.length,
          detectionCount: detectionCount,
        );
        detectStat.printResults('$imagePath (detect)');
        detectStats.add(detectStat);

        // Pre-decode image once
        final image = img.decodeImage(bytes);
        expect(image, isNotNull);

        // Benchmark detectOnImage() - no decoding overhead
        print('\n--- detectOnImage() method ---');
        final detectOnImageTimings = <int>[];

        for (int i = 0; i < ITERATIONS; i++) {
          final stopwatch = Stopwatch()..start();
          final results = await detector.detectOnImage(image!);
          stopwatch.stop();

          detectOnImageTimings.add(stopwatch.elapsedMilliseconds);
        }

        final detectOnImageStat = BenchmarkStats(
          imagePath: imagePath,
          timings: detectOnImageTimings,
          imageSize: bytes.length,
          detectionCount: detectionCount,
        );
        detectOnImageStat.printResults('$imagePath (detectOnImage)');
        detectOnImageStats.add(detectOnImageStat);

        // Show overhead comparison
        final overhead = detectStat.average - detectOnImageStat.average;
        print('\nDecoding overhead: ${overhead.toStringAsFixed(2)} ms (${(overhead / detectStat.average * 100).toStringAsFixed(1)}%)');
      }

      await detector.dispose();

      // Write results to files
      final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');

      final detectResults = BenchmarkResults(
        timestamp: timestamp,
        testName: 'detect() method (Lite Model)',
        configuration: {
          'model': 'lite',
          'mode': 'boxesAndLandmarks',
          'method': 'detect',
          'iterations': ITERATIONS,
        },
        results: detectStats,
      );
      detectResults.printJson('benchmark_detect_method_$timestamp.json');

      final detectOnImageResults = BenchmarkResults(
        timestamp: timestamp,
        testName: 'detectOnImage() method (Lite Model)',
        configuration: {
          'model': 'lite',
          'mode': 'boxesAndLandmarks',
          'method': 'detectOnImage',
          'iterations': ITERATIONS,
        },
        results: detectOnImageStats,
      );
      detectOnImageResults.printJson('benchmark_detectOnImage_method_$timestamp.json');
    });
  });
}
