import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';
import '../util/pose_helpers.dart';

/// Pre-allocated inference buffers for one pool slot.
///
/// Avoids GC pressure by reusing the same buffers across invocations.
/// Each [InterpreterPool] slot has its own [_PoseBuffers] instance.
///
/// Outputs are stored as flat [Float32List]s and passed to TFLite as their
/// underlying [ByteBuffer], which avoids the boxed-double allocation that
/// `Tensor.copyTo` performs when handed a nested `List<List<double>>` dst.
class _PoseBuffers {
  final Float32List flatInputBuffer;
  final Float32List outputLandmarks;
  final Float32List outputScore;
  final Float32List outputMask;
  final Float32List outputHeatmap;
  final Float32List outputWorld;

  _PoseBuffers({
    required this.flatInputBuffer,
    required this.outputLandmarks,
    required this.outputScore,
    required this.outputMask,
    required this.outputHeatmap,
    required this.outputWorld,
  });
}

/// One CompiledModel pool slot: a [CompiledModel] plus a reusable input buffer.
///
/// A single [CompiledModel] is not thread-safe, so each slot serializes its own
/// calls through an [AsyncLock]. The pool selects slots round-robin, so
/// concurrent [PoseLandmarkModelRunner.run] calls (one per detected person,
/// dispatched via `Future.wait`) land on different slots and overlap, while
/// calls colliding on the same slot run back-to-back.
class _CompiledSlot {
  final CompiledModel model;
  final Float32List input;
  final AsyncLock lock = AsyncLock();

  _CompiledSlot(this.model, this.input);
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
/// - Pool size 1: Sequential processing (stable, predictable default)
/// - Pool size 3: Good balance for typical multi-person scenarios when
///   hardware acceleration is disabled and the caller opts in
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

  /// CompiledModel pool slots, used instead of [_pool] when initialized with
  /// `useCompiledModel: true`. Empty in interpreter mode.
  final List<_CompiledSlot> _compiledSlots = [];
  int _nextCompiled = 0;

  /// Output tensor indices for the CompiledModel path, resolved at init from
  /// output byte sizes (landmarks = floats divisible by 5 and >= 165; score =
  /// a single float).
  int _landmarksIdx = 0;
  int _scoreIdx = 1;

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
    final String path = poseLandmarkModelPath(model);
    await _initWith(
      (options) => Interpreter.fromAsset(path, options: options),
      performanceConfig: performanceConfig,
      useIsolateInterpreter: true,
    );
  }

  /// Initializes from pre-loaded model bytes (used inside a background isolate
  /// where Flutter asset loading is not available).
  ///
  /// When [useCompiledModel] is true, the runner builds a pool of LiteRT Next
  /// [CompiledModel] instances ([poolSize] of them) instead of an
  /// [InterpreterPool]. [compiledForceCpu] pins each CompiledModel to CPU.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
    bool useCompiledModel = false,
    bool compiledForceCpu = false,
  }) async {
    if (_isInitialized) await dispose();
    if (useCompiledModel) {
      await _initCompiled(modelBytes, forceCpu: compiledForceCpu);
      return;
    }
    await _initWith(
      (options) async {
        final interpreter = Interpreter.fromBuffer(
          modelBytes,
          options: options,
        );
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: false,
    );
  }

  /// Builds the CompiledModel pool. One CompiledModel per [poolSize] slot, each
  /// with its own reusable input buffer, so multiple landmark inferences can be
  /// in flight on distinct models.
  Future<void> _initCompiled(
    Uint8List modelBytes, {
    required bool forceCpu,
  }) async {
    _compiledSlots.clear();
    _nextCompiled = 0;
    final int slots = poolSize;
    for (int i = 0; i < slots; i++) {
      final CompiledModel model = CompiledModel.fromBufferWithGpuFallback(
        modelBytes,
        forceCpu: forceCpu,
      );
      if (i == 0) _identifyCompiledOutputs(model);
      _compiledSlots.add(_CompiledSlot(model, Float32List(256 * 256 * 3)));
    }
    _isInitialized = true;
  }

  /// Resolves landmark/score output indices from CompiledModel byte sizes.
  /// BlazePose outputs are {landmarks(195), score(1), mask, heatmap, world}.
  void _identifyCompiledOutputs(CompiledModel model) {
    final List<int> sizes = model.outputByteSizes;
    for (int i = 0; i < sizes.length; i++) {
      final int floatCount = sizes[i] ~/ 4;
      if (floatCount == 1) {
        _scoreIdx = i;
      } else if (floatCount >= 165 && floatCount % 5 == 0) {
        _landmarksIdx = i;
      }
    }
  }

  Future<void> _initWith(
    Future<Interpreter> Function(InterpreterOptions) loader, {
    PerformanceConfig? performanceConfig,
    bool useIsolateInterpreter = true,
  }) async {
    await _pool.initialize(
      (options, _) async {
        final interpreter = await loader(options);
        interpreter.resizeInputTensor(0, [1, 256, 256, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );

    _buffers.clear();
    for (final interp in _pool.interpreters) {
      _buffers[interp] = _PoseBuffers(
        flatInputBuffer: Float32List(256 * 256 * 3),
        outputLandmarks: Float32List(195),
        outputScore: Float32List(1),
        outputMask: Float32List(256 * 256 * 1),
        outputHeatmap: Float32List(64 * 64 * 39),
        outputWorld: Float32List(117),
      );
    }

    _isInitialized = true;
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
    for (final _CompiledSlot slot in _compiledSlots) {
      slot.model.close();
    }
    _compiledSlots.clear();
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

    if (_compiledSlots.isNotEmpty) {
      final _CompiledSlot slot = _compiledSlots[_nextCompiled];
      _nextCompiled = (_nextCompiled + 1) % _compiledSlots.length;
      return slot.lock.run(() async {
        bgrBytesToRgbFloat32(
          bytes: mat.data,
          totalPixels: mat.rows * mat.cols,
          buffer: slot.input,
        );
        final List<Float32List> outs = await slot.model.runAsync([slot.input]);
        return parsePoseLandmarksFlat(outs[_landmarksIdx], outs[_scoreIdx]);
      });
    }

    return _pool.withInterpreter((interp, iso) async {
      final buf = _buffers[interp]!;
      bgrBytesToRgbFloat32(
        bytes: mat.data,
        totalPixels: mat.rows * mat.cols,
        buffer: buf.flatInputBuffer,
      );

      final Map<int, Object> outputs = <int, Object>{
        0: buf.outputLandmarks.buffer,
        1: buf.outputScore.buffer,
        2: buf.outputMask.buffer,
        3: buf.outputHeatmap.buffer,
        4: buf.outputWorld.buffer,
      };

      if (iso != null) {
        await iso.runForMultipleInputs([buf.flatInputBuffer.buffer], outputs);
      } else {
        interp.runForMultipleInputs([buf.flatInputBuffer.buffer], outputs);
      }

      return parsePoseLandmarksFlat(buf.outputLandmarks, buf.outputScore);
    });
  }
}
