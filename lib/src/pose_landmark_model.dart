import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:meta/meta.dart';
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';
import 'types.dart';

/// A single interpreter instance with its associated resources.
///
/// Encapsulates a TensorFlow Lite interpreter and its isolate wrapper,
/// allowing for clean resource management in the interpreter pool.
/// Also holds pre-allocated input/output buffers to avoid GC pressure.
class _InterpreterInstance {
  final Interpreter interpreter;
  final IsolateInterpreter isolateInterpreter;

  // Pre-allocated input buffer [1, 256, 256, 3] - reused across calls
  final List<List<List<List<double>>>> inputBuffer;

  // Pre-allocated output buffers - reused across calls
  final List<List<double>> outputLandmarks; // [1, 195]
  final List<List<double>> outputScore; // [1, 1]
  final List<List<List<List<double>>>> outputMask; // [1, 256, 256, 1]
  final List<List<List<List<double>>>> outputHeatmap; // [1, 64, 64, 39]
  final List<List<double>> outputWorld; // [1, 117]

  _InterpreterInstance({
    required this.interpreter,
    required this.isolateInterpreter,
    required this.inputBuffer,
    required this.outputLandmarks,
    required this.outputScore,
    required this.outputMask,
    required this.outputHeatmap,
    required this.outputWorld,
  });

  /// Disposes interpreter and isolate wrapper.
  Future<void> dispose() async {
    isolateInterpreter.close();
    interpreter.close();
  }
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
  /// Pool of interpreter instances for parallel processing.
  final List<_InterpreterInstance> _interpreterPool = [];

  /// Maximum number of concurrent inferences.
  final int _poolSize;

  /// Delegate instances - one per interpreter (XNNPACK is NOT thread-safe for sharing).
  final List<Delegate> _delegates = [];

  /// Serialization locks to prevent concurrent inference on the same interpreter.
  /// Each interpreter has its own lock to avoid XNNPACK thread contention.
  /// This mirrors the _meshInferenceLocks pattern from face_detection_tflite.
  final List<Future<void>> _interpreterLocks = [];

  /// Round-robin counter for interpreter selection.
  /// Distributes load evenly across the interpreter pool.
  int _poolCounter = 0;

  bool _isInitialized = false;
  static ffi.DynamicLibrary? _tfliteLib;

  /// Creates a landmark model runner with the specified pool size.
  ///
  /// Parameters:
  /// - [poolSize]: Number of interpreter instances to create (1-10 recommended).
  ///   Higher values enable more parallelism but consume more memory.
  ///   Default is 1 for backward compatibility.
  PoseLandmarkModelRunner({int poolSize = 1})
      : _poolSize = poolSize.clamp(1, 10);

  /// Ensures TensorFlow Lite native library is loaded for desktop platforms.
  ///
  /// On Windows, Linux, and macOS, loads the platform-specific TensorFlow Lite C library
  /// from bundled assets. On mobile platforms (iOS/Android), uses the system library.
  ///
  /// This method is idempotent - subsequent calls do nothing if already loaded.
  ///
  /// Throws an exception if the library cannot be found or loaded.
  static Future<void> ensureTFLiteLoaded({
    Map<String, String>? env,
    String? platformOverride,
  }) async {
    if (_tfliteLib != null) return;

    final Map<String, String> environment = env ?? Platform.environment;
    final String platform = platformOverride ?? _platformString();

    // Optional override for local testing: set POSE_TFLITE_LIB to an absolute path.
    final envLibPath = environment['POSE_TFLITE_LIB'];
    if (envLibPath != null && envLibPath.isNotEmpty) {
      _tfliteLib = ffi.DynamicLibrary.open(envLibPath);
      return;
    }

    final exe = File(Platform.resolvedExecutable);
    final exeDir = exe.parent;

    late final List<String> candidates;

    if (platform == 'windows') {
      candidates = [
        p.join(exeDir.path, 'libtensorflowlite_c-win.dll'),
        'libtensorflowlite_c-win.dll',
      ];
    } else if (platform == 'linux') {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
    } else if (platform == 'macos') {
      final contents = exeDir.parent;
      candidates = [
        p.join(contents.path, 'Resources', 'libtensorflowlite_c-mac.dylib'),
        // When running `flutter test`, the dylib is not copied into the engine
        // cache; fall back to the repo / package checkout location.
        p.join(Directory.current.path, 'macos', 'Frameworks',
            'libtensorflowlite_c-mac.dylib'),
      ];
    } else {
      _tfliteLib = ffi.DynamicLibrary.process();
      return;
    }

    for (final String c in candidates) {
      try {
        if (c.contains(p.separator)) {
          if (!File(c).existsSync()) continue;
        }
        _tfliteLib = ffi.DynamicLibrary.open(c);
        return;
      } catch (_) {}
    }
  }

  static String _platformString() {
    if (Platform.isWindows) return 'windows';
    if (Platform.isLinux) return 'linux';
    if (Platform.isMacOS) return 'macos';
    return 'other';
  }

  /// Resets the TFLite native library cache for testing.
  ///
  /// Clears the cached [_tfliteLib] reference, allowing tests to verify
  /// library loading behavior across different platform configurations.
  @visibleForTesting
  static void resetNativeLibForTest() {
    _tfliteLib = null;
  }

  /// Returns the cached TFLite native library instance for testing.
  ///
  /// Provides test access to verify that the correct native library
  /// was loaded after calling [ensureTFLiteLoaded].
  @visibleForTesting
  static ffi.DynamicLibrary? nativeLibForTest() => _tfliteLib;

  /// Initializes the BlazePose landmark model with the specified variant.
  ///
  /// Creates a pool of interpreter instances based on the configured [_poolSize].
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
    await ensureTFLiteLoaded();

    final String path = _getModelPath(model);

    // Create pool of interpreter instances - each with its OWN delegate
    // XNNPACK delegates are NOT thread-safe for sharing across interpreters
    for (int i = 0; i < _poolSize; i++) {
      final (options, delegate) = _createInterpreterOptions(performanceConfig);
      if (delegate != null) {
        _delegates.add(delegate);
      }

      final interpreter = await Interpreter.fromAsset(path, options: options);
      interpreter.resizeInputTensor(0, [1, 256, 256, 3]);
      interpreter.allocateTensors();

      final isolateInterpreter =
          await IsolateInterpreter.create(address: interpreter.address);

      // Pre-allocate input buffer [1, 256, 256, 3]
      final inputBuffer = List.generate(
        1,
        (_) => List.generate(
          256,
          (_) => List.generate(
            256,
            (_) => List<double>.filled(3, 0.0, growable: false),
            growable: false,
          ),
          growable: false,
        ),
        growable: false,
      );

      // Pre-allocate output buffers
      final outputLandmarks = [List<double>.filled(195, 0.0, growable: false)];
      final outputScore = [List<double>.filled(1, 0.0, growable: false)];
      final outputMask = _createTensor4D(1, 256, 256, 1);
      final outputHeatmap = _createTensor4D(1, 64, 64, 39);
      final outputWorld = [List<double>.filled(117, 0.0, growable: false)];

      _interpreterPool.add(_InterpreterInstance(
        interpreter: interpreter,
        isolateInterpreter: isolateInterpreter,
        inputBuffer: inputBuffer,
        outputLandmarks: outputLandmarks,
        outputScore: outputScore,
        outputMask: outputMask,
        outputHeatmap: outputHeatmap,
        outputWorld: outputWorld,
      ));

      // Initialize serialization lock for this interpreter
      _interpreterLocks.add(Future.value());
    }

    _isInitialized = true;
  }

  /// Creates interpreter options with delegates based on performance configuration.
  ///
  /// Returns a tuple of (options, delegate) where delegate may be null.
  /// Each call creates a NEW delegate instance - do NOT share delegates across interpreters.
  (InterpreterOptions, Delegate?) _createInterpreterOptions(
      PerformanceConfig? config) {
    final options = InterpreterOptions();

    // If no config or disabled mode, return default options (backward compatible)
    if (config == null || config.mode == PerformanceMode.disabled) {
      return (options, null);
    }

    // Get effective thread count
    final threadCount = config.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);

    // Set CPU threads
    options.threads = threadCount;

    // Add XNNPACK delegate (for xnnpack or auto mode)
    if (config.mode == PerformanceMode.xnnpack ||
        config.mode == PerformanceMode.auto) {
      try {
        final xnnpackDelegate = XNNPackDelegate(
          options: XNNPackDelegateOptions(numThreads: threadCount),
        );
        options.addDelegate(xnnpackDelegate);
        return (options, xnnpackDelegate);
      } catch (e) {
        // Graceful fallback: if delegate creation fails, continue with CPU
        // ignore: avoid_print
        print('[BlazePose] Warning: Failed to create XNNPACK delegate: $e');
        // ignore: avoid_print
        print('[BlazePose] Falling back to default CPU execution');
      }
    }

    return (options, null);
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
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_lite.tflite';
      case PoseLandmarkModel.full:
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_full.tflite';
      case PoseLandmarkModel.heavy:
        return 'packages/pose_detection_tflite/assets/models/pose_landmark_heavy.tflite';
    }
  }

  /// Returns true if the model runner has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns the configured pool size.
  int get poolSize => _poolSize;

  /// Disposes the model runner and releases all resources.
  ///
  /// Closes all interpreter instances in the pool and clears all state.
  /// After disposal, [initialize] must be called again before using the runner.
  Future<void> dispose() async {
    // Dispose all interpreter instances
    for (final instance in _interpreterPool) {
      await instance.dispose();
    }
    _interpreterPool.clear();

    // Clean up all delegates (one per interpreter)
    for (final delegate in _delegates) {
      delegate.delete();
    }
    _delegates.clear();

    // Clear serialization locks
    _interpreterLocks.clear();

    _isInitialized = false;
  }

  /// Serializes inference calls on a specific interpreter to prevent race conditions.
  ///
  /// This method ensures only one inference runs at a time per interpreter instance,
  /// preventing XNNPACK thread contention when multiple people are being processed.
  /// It chains futures similar to FaceDetector's _withMeshLock pattern.
  ///
  /// Uses round-robin selection to distribute load evenly across the pool.
  ///
  /// Parameters:
  /// - [fn]: The function to execute with exclusive access to an interpreter
  ///
  /// Returns the result of [fn]
  Future<T> _withInterpreterLock<T>(
      Future<T> Function(_InterpreterInstance) fn) async {
    if (_interpreterPool.isEmpty) {
      throw StateError('Interpreter pool is empty. Call initialize() first.');
    }

    // Round-robin selection to distribute load evenly
    final int poolIndex = _poolCounter % _interpreterPool.length;
    _poolCounter = (_poolCounter + 1) % _interpreterPool.length;

    final previous = _interpreterLocks[poolIndex];
    final completer = Completer<void>();
    _interpreterLocks[poolIndex] = completer.future;

    try {
      await previous;
      return await fn(_interpreterPool[poolIndex]);
    } finally {
      completer.complete();
    }
  }

  /// Runs landmark extraction on a person crop image.
  ///
  /// Extracts 33 body landmarks from the input person crop using the BlazePose model.
  /// The input image should be a cropped person region, ideally from the YOLOv8 detector.
  ///
  /// **Thread-safety:** This method is safe to call concurrently. It uses round-robin
  /// selection to distribute load across the interpreter pool, with per-interpreter
  /// serialization locks to prevent XNNPACK thread contention. Multiple people can be
  /// processed in parallel using different interpreters, but each interpreter only
  /// runs one inference at a time.
  ///
  /// The method performs:
  /// 1. Selects an interpreter using round-robin (distributes load evenly)
  /// 2. Serializes access to the interpreter (waits for previous inference to complete)
  /// 3. Converts image to tensor (NHWC format, normalized 0-1)
  /// 4. Runs model inference via IsolateInterpreter
  /// 5. Post-processes results: sigmoid activation, coordinate normalization
  ///
  /// Parameters:
  /// - [roiImage]: Cropped person image (will be resized to 256x256 internally)
  ///
  /// Returns [PoseLandmarks] containing 33 landmarks with normalized coordinates (0-1 range)
  /// and a confidence score. Landmarks are in the 256x256 model output space.
  ///
  /// Throws [StateError] if the model is not initialized.
  Future<PoseLandmarks> run(img.Image roiImage) async {
    if (!_isInitialized) {
      throw StateError(
          'PoseLandmarkModelRunner not initialized. Call initialize() first.');
    }

    // Use round-robin selection with serialization lock (mirrors face detection pattern)
    return await _withInterpreterLock((instance) async {
      // Reuse pre-allocated input buffer (ImageUtils.imageToNHWC4D fills it in-place)
      ImageUtils.imageToNHWC4D(roiImage, 256, 256, reuse: instance.inputBuffer);

      // Run inference using pre-allocated output buffers
      // Note: TFLite overwrites the buffer contents, no need to zero first
      await instance.isolateInterpreter.runForMultipleInputs(
        [instance.inputBuffer],
        {
          0: instance.outputLandmarks,
          1: instance.outputScore,
          2: instance.outputMask,
          3: instance.outputHeatmap,
          4: instance.outputWorld,
        },
      );

      return _parseLandmarks(
        instance.outputLandmarks,
        instance.outputScore,
        instance.outputWorld,
      );
    });
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
