import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';

/// Pre-allocated inference buffers for one pool slot.
///
/// Avoids GC pressure by reusing the same buffers across invocations.
/// Each [InterpreterPool] slot has its own [_PoseBuffers] instance.
class _PoseBuffers {
  final Float32List flatInputBuffer;
  final List<List<double>> outputLandmarks;
  final List<List<double>> outputScore;
  final List<List<List<List<double>>>> outputMask;
  final List<List<List<List<double>>>> outputHeatmap;
  final List<List<double>> outputWorld;

  _PoseBuffers({
    required this.flatInputBuffer,
    required this.outputLandmarks,
    required this.outputScore,
    required this.outputMask,
    required this.outputHeatmap,
    required this.outputWorld,
  });
}

/// BlazePose landmark extraction model runner for Stage 2 of the pose detection pipeline.
///
/// Extracts 33 body landmarks from person crops using the BlazePose model.
/// Supports three model variants (lite, full, heavy) with different accuracy/performance trade-offs.
///
/// **Interpreter Pool Architecture:**
/// To enable parallel processing of multiple people, this runner maintains a pool of
/// TensorFlow Lite interpreter instances using a **round-robin selection pattern**
/// (identical to face_detection_tflite). Each interpreter uses serialization locks
/// to prevent concurrent inference, avoiding XNNPACK thread contention.
///
/// The pool size determines the maximum number of interpreter instances:
/// - Pool size 1: Sequential processing (stable, predictable performance)
/// - Pool size 3: Good balance for typical multi-person scenarios (default)
/// - Pool size 5+: Best for crowded scenes (6+ people)
///
/// **How it works:**
/// - Interpreters are selected using round-robin (person 1 → interpreter 0,
///   person 2 → interpreter 1, person 3 → interpreter 2, person 4 → interpreter 0, etc.)
/// - Each interpreter is locked during inference to prevent thread contention
/// - Multiple people can be processed in parallel using different interpreters
/// - Simple, predictable behavior with stable performance (no queue complexity)
class PoseLandmarkModelRunner {
  final InterpreterPool _pool;

  /// Per-slot pre-allocated buffers, keyed by interpreter identity.
  final Map<Interpreter, _PoseBuffers> _buffers = {};

  bool _isInitialized = false;

  /// Creates a landmark model runner with the specified pool size.
  ///
  /// Parameters:
  /// - [poolSize]: Number of interpreter instances to create (1-10 recommended).
  ///   Higher values enable more parallelism but consume more memory.
  ///   Default is 1 for backward compatibility.
  PoseLandmarkModelRunner({int poolSize = 1})
    : _pool = InterpreterPool(poolSize: poolSize);

  /// Initializes the BlazePose landmark model with the specified variant.
  ///
  /// Creates a pool of interpreter instances based on the configured [poolSize].
  /// Each interpreter is loaded independently, allowing for parallel inference execution.
  /// All interpreters in the pool will use the same performance configuration.
  ///
  /// Parameters:
  /// - [model]: Which BlazePose variant to use (lite, full, or heavy)
  /// - [performanceConfig]: Optional performance configuration for TFLite delegates.
  ///   Defaults to no delegates (backward compatible). Use [PerformanceConfig.xnnpack()]
  ///   for CPU optimization.
  ///
  /// If already initialized, this will dispose all previous instances first.
  ///
  /// **Memory usage:** Approximately 10MB per interpreter instance.
  /// For example, a pool size of 5 will consume ~50MB for the model pool.
  ///
  /// Throws an exception if the model fails to load or TFLite library is unavailable.
  Future<void> initialize(
    PoseLandmarkModel model, {
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    final String path = _getModelPath(model);

    await _pool.initialize((options, _) async {
      final interpreter = await Interpreter.fromAsset(path, options: options);
      interpreter.resizeInputTensor(0, [1, 256, 256, 3]);
      interpreter.allocateTensors();
      return interpreter;
    }, performanceConfig: performanceConfig);

    _buffers.clear();
    for (final interp in _pool.interpreters) {
      _buffers[interp] = _PoseBuffers(
        flatInputBuffer: Float32List(256 * 256 * 3),
        outputLandmarks: [List<double>.filled(195, 0.0, growable: false)],
        outputScore: [List<double>.filled(1, 0.0, growable: false)],
        outputMask: _createTensor4D(1, 256, 256, 1),
        outputHeatmap: _createTensor4D(1, 64, 64, 39),
        outputWorld: [List<double>.filled(117, 0.0, growable: false)],
      );
    }

    _isInitialized = true;
  }

  /// Creates a pre-allocated 4D tensor with the specified dimensions.
  static List<List<List<List<double>>>> _createTensor4D(
    int dim1,
    int dim2,
    int dim3,
    int dim4,
  ) {
    return List.generate(
      dim1,
      (_) => List.generate(
        dim2,
        (_) => List.generate(
          dim3,
          (_) => List<double>.filled(dim4, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  String _getModelPath(PoseLandmarkModel model) {
    switch (model) {
      case PoseLandmarkModel.lite:
        return 'packages/pose_detection/assets/models/pose_landmark_lite.tflite';
      case PoseLandmarkModel.full:
        return 'packages/pose_detection/assets/models/pose_landmark_full.tflite';
      case PoseLandmarkModel.heavy:
        return 'packages/pose_detection/assets/models/pose_landmark_heavy.tflite';
    }
  }

  /// Returns true if the model runner has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns the configured pool size.
  int get poolSize => _pool.poolSize;

  /// Disposes the model runner and releases all resources.
  ///
  /// Closes all interpreter instances in the pool and clears all state.
  /// After disposal, [initialize] must be called again before using the runner.
  Future<void> dispose() async {
    await _pool.dispose();
    _buffers.clear();
    _isInitialized = false;
  }

  /// Runs landmark extraction on a pre-letterboxed 256x256 cv.Mat.
  ///
  /// This method expects the input to already be letterboxed to 256x256.
  /// The caller (typically [PoseDetector.detect]) handles cropping and
  /// letterboxing, storing the transformation parameters for coordinate mapping.
  ///
  /// **Thread-safety:** Safe to call concurrently. Uses round-robin interpreter
  /// selection with per-interpreter serialization locks.
  ///
  /// Parameters:
  /// - [mat]: Letterboxed 256x256 cv.Mat in BGR format
  ///
  /// Returns [PoseLandmarks] containing 33 landmarks with normalized coordinates.
  ///
  /// Throws [StateError] if the model is not initialized.
  Future<PoseLandmarks> run(cv.Mat mat) async {
    if (!_isInitialized) {
      throw StateError(
        'PoseLandmarkModelRunner not initialized. Call initialize() first.',
      );
    }

    return _pool.withInterpreter((interp, iso) async {
      final buf = _buffers[interp]!;
      _matToInputBuffer(mat, buf.flatInputBuffer);

      final outputs = {
        0: buf.outputLandmarks,
        1: buf.outputScore,
        2: buf.outputMask,
        3: buf.outputHeatmap,
        4: buf.outputWorld,
      };

      if (iso != null) {
        await iso.runForMultipleInputs([buf.flatInputBuffer.buffer], outputs);
      } else {
        interp.runForMultipleInputs([buf.flatInputBuffer.buffer], outputs);
      }

      return _parseLandmarks(
        buf.outputLandmarks,
        buf.outputScore,
        buf.outputWorld,
      );
    });
  }

  /// Converts a cv.Mat to a Float32List tensor for BlazePose input.
  ///
  /// Normalizes pixel values to [0.0, 1.0] range (BlazePose uses 0-1, not -1 to 1).
  /// Handles BGR to RGB conversion.
  void _matToInputBuffer(cv.Mat mat, Float32List buffer) {
    final int h = mat.rows;
    final int w = mat.cols;
    final int totalPixels = h * w;
    final Uint8List data = mat.data;

    const double scale = 1.0 / 255.0;
    for (int i = 0, j = 0; i < totalPixels * 3; i += 3, j += 3) {
      buffer[j] = data[i + 2] * scale;
      buffer[j + 1] = data[i + 1] * scale;
      buffer[j + 2] = data[i] * scale;
    }
  }

  PoseLandmarks _parseLandmarks(
    List<dynamic> landmarksData,
    List<dynamic> scoreData,
    List<dynamic> worldData,
  ) {
    double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    double clamp01(double v) => v.isNaN
        ? 0.0
        : v < 0.0
        ? 0.0
        : (v > 1.0 ? 1.0 : v);

    final double score = sigmoid(scoreData[0][0] as double);
    final List<dynamic> raw = landmarksData[0] as List<dynamic>;
    final List<PoseLandmark> lm = <PoseLandmark>[];

    for (int i = 0; i < 33; i++) {
      final int base = i * 5;
      final double x = clamp01((raw[base + 0] as double) / 256.0);
      final double y = clamp01((raw[base + 1] as double) / 256.0);
      final double z = raw[base + 2] as double;
      final double visibility = sigmoid(raw[base + 3] as double);
      final double presence = sigmoid(raw[base + 4] as double);
      final double vis = (visibility * presence).clamp(0.0, 1.0);

      lm.add(
        PoseLandmark(
          type: PoseLandmarkType.values[i],
          x: x,
          y: y,
          z: z,
          visibility: vis,
        ),
      );
    }

    return PoseLandmarks(landmarks: lm, score: score);
  }
}
