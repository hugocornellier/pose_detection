// ignore_for_file: avoid_print

// Inference-only micro-benchmark: CompiledModel {cpu} vs {gpu,cpu} for every
// pose model (YOLOv8 person detector + lite/full/heavy BlazePose landmark).
//
// Isolates raw CompiledModel.runAsync latency from the rest of the pipeline so
// we can see, per model, whether GPU beats CPU and whether GPU compilation even
// succeeds (it prints CompiledModel.accelerators, the set that actually
// compiled - if a GPU request silently fell back to CPU, both columns match).
//
//   flutter test integration_test/pose_compiledmodel_inference_bench_test.dart -d macos

import 'dart:convert';
import 'dart:io' show Platform;
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const int iterations = 50;
const int warmupIterations = 15;

const Map<String, String> modelAssets = {
  'yolo': 'packages/pose_detection/assets/models/yolov8n_float32.tflite',
  'lite': 'packages/pose_detection/assets/models/pose_landmark_lite.tflite',
  'full': 'packages/pose_detection/assets/models/pose_landmark_full.tflite',
  'heavy': 'packages/pose_detection/assets/models/pose_landmark_heavy.tflite',
};

double _p50(List<int> t) {
  final s = List<int>.from(t)..sort();
  return s[((s.length - 1) * 0.50).floor()].toDouble();
}

double _mean(List<int> t) => t.reduce((a, b) => a + b) / t.length;

double _std(List<int> t) {
  final m = _mean(t);
  return sqrt(t.map((v) => pow(v - m, 2)).reduce((a, b) => a + b) / t.length);
}

({List<int> timings, Set<Accelerator> accelerators}) _benchCompiled(
  Uint8List bytes,
  Set<Accelerator> request, {
  Precision precision = Precision.fp32,
}) {
  final cm = CompiledModel.fromBuffer(
    bytes,
    accelerators: request,
    precision: precision,
  );
  try {
    final inputs = <Float32List>[
      for (final bs in cm.inputByteSizes)
        Float32List(bs ~/ 4)..fillRange(0, bs ~/ 4, 0.5),
    ];
    for (int i = 0; i < warmupIterations; i++) {
      cm.run(inputs);
    }
    final timings = <int>[];
    for (int i = 0; i < iterations; i++) {
      final sw = Stopwatch()..start();
      cm.run(inputs);
      sw.stop();
      timings.add(sw.elapsedMicroseconds);
    }
    return (timings: timings, accelerators: cm.accelerators);
  } finally {
    cm.close();
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  // macOS exercises the real Metal GPU; Android and iOS devices exercise
  // their real mobile GPU (OpenCL on Android, Metal on iOS). Everything else
  // (simulators, desktop CI without a GPU) stays skipped.
  final hasRealGpu = Platform.isMacOS || Platform.isAndroid || Platform.isIOS;

  final summary = <String>[];
  final jsonRows = <Map<String, dynamic>>[];

  group('CompiledModel inference-only (cpu vs gpu, -d macos)', () {
    modelAssets.forEach((name, asset) {
      test(
        '$name : cpu vs gpu|cpu',
        timeout: const Timeout(Duration(minutes: 10)),
        skip: hasRealGpu ? false : 'needs a real GPU (macOS/Android/iOS)',
        () async {
          final data = await rootBundle.load(asset);
          final bytes = data.buffer.asUint8List();

          final cpu = _benchCompiled(bytes, {Accelerator.cpu});
          ({List<int> timings, Set<Accelerator> accelerators})? gpu;
          String gpuNote = '';
          try {
            gpu = _benchCompiled(bytes, {Accelerator.gpu, Accelerator.cpu});
          } catch (e) {
            gpuNote = 'gpu compile threw: $e';
          }

          final cpuMs = _p50(cpu.timings) / 1000.0;
          final gpuMs = gpu == null ? double.nan : _p50(gpu.timings) / 1000.0;
          final speedup = gpu == null ? double.nan : cpuMs / gpuMs;
          final accStr = gpu == null
              ? 'n/a'
              : gpu.accelerators.map((a) => a.name).join('+');

          final line =
              '${name.padRight(6)} '
              'cpu p50=${cpuMs.toStringAsFixed(2).padLeft(7)}ms   '
              'gpu|cpu p50=${gpuMs.toStringAsFixed(2).padLeft(7)}ms   '
              'speedup=${speedup.toStringAsFixed(2)}x   '
              'compiled=[$accStr] $gpuNote';
          summary.add(line);
          print('\n>>> $line\n');

          jsonRows.add({
            'model': name,
            'cpu': {
              'p50_ms': cpuMs,
              'mean_ms': _mean(cpu.timings) / 1000.0,
              'std_ms': _std(cpu.timings) / 1000.0,
              'accelerators': cpu.accelerators.map((a) => a.name).toList(),
            },
            'gpu_cpu': gpu == null
                ? {'error': gpuNote}
                : {
                    'p50_ms': gpuMs,
                    'mean_ms': _mean(gpu.timings) / 1000.0,
                    'std_ms': _std(gpu.timings) / 1000.0,
                    'accelerators': gpu.accelerators
                        .map((a) => a.name)
                        .toList(),
                  },
            'cpu_over_gpu_speedup_x': speedup,
          });
        },
      );
    });

    tearDownAll(() {
      print('\n${'=' * 78}');
      print('COMPILEDMODEL INFERENCE-ONLY SUMMARY (cpu vs gpu|cpu, -d macos)');
      print('=' * 78);
      for (final l in summary) {
        print(l);
      }
      print('=' * 78);

      final ts = DateTime.now().toIso8601String().replaceAll(':', '-');
      final file = 'compiledmodel_inference_$ts.json';
      print('\n BENCHMARK_JSON_START:$file');
      print(
        const JsonEncoder.withIndent(
          '  ',
        ).convert({'test': 'compiledmodel_inference', 'rows': jsonRows}),
      );
      print(' BENCHMARK_JSON_END:$file');
    });
  });
}
