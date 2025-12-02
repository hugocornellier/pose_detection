import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';
import 'types.dart';

/// A single interpreter instance with its associated resources.
///
/// Encapsulates a TensorFlow Lite interpreter and its isolate wrapper,
/// allowing for clean resource management in the interpreter pool.
class _InterpreterInstance {
  final Interpreter interpreter;
  final IsolateInterpreter isolateInterpreter;

  _InterpreterInstance({
    required this.interpreter,
    required this.isolateInterpreter,
  });

  /// Disposes both the interpreter and its isolate wrapper.
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
/// TensorFlow Lite interpreter instances. Each interpreter can run inference independently,
/// allowing concurrent landmark extraction for different detected persons.
///
/// The pool size determines the maximum number of concurrent inferences:
/// - Pool size 1: Sequential processing (safest, lowest memory)
/// - Pool size 3-5: Good balance for multi-person scenarios
/// - Pool size > 5: Diminishing returns, high memory usage (~10MB per interpreter)
///
/// Thread-safe semaphore-based resource management ensures interpreters are properly
/// acquired and released, preventing race conditions.
class PoseLandmarkModelRunner {
  /// Pool of interpreter instances for parallel processing.
  final List<_InterpreterInstance> _interpreterPool = [];

  /// Queue of available interpreters (indices into _interpreterPool).
  final List<int> _availableInterpreters = [];

  /// Pending requests waiting for an available interpreter.
  final List<Completer<int>> _waitQueue = [];

  /// Maximum number of concurrent inferences.
  final int _poolSize;

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
  static Future<void> ensureTFLiteLoaded() async {
    if (_tfliteLib != null) return;

    // Optional override for local testing: set POSE_TFLITE_LIB to an absolute path.
    final envLibPath = Platform.environment['POSE_TFLITE_LIB'];
    if (envLibPath != null && envLibPath.isNotEmpty) {
      _tfliteLib = ffi.DynamicLibrary.open(envLibPath);
      return;
    }

    final exe = File(Platform.resolvedExecutable);
    final exeDir = exe.parent;

    late final List<String> candidates;

    if (Platform.isWindows) {
      candidates = [
        p.join(exeDir.path, 'libtensorflowlite_c-win.dll'),
        'libtensorflowlite_c-win.dll',
      ];
    } else if (Platform.isLinux) {
      candidates = [
        p.join(exeDir.path, 'lib', 'libtensorflowlite_c-linux.so'),
        'libtensorflowlite_c-linux.so',
      ];
    } else if (Platform.isMacOS) {
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

  /// Initializes the BlazePose landmark model with the specified variant.
  ///
  /// Creates a pool of interpreter instances based on the configured [_poolSize].
  /// Each interpreter is loaded independently, allowing for parallel inference execution.
  ///
  /// Parameters:
  /// - [model]: Which BlazePose variant to use (lite, full, or heavy)
  ///
  /// If already initialized, this will dispose all previous instances first.
  ///
  /// **Memory usage:** Approximately 10MB per interpreter instance.
  /// For example, a pool size of 5 will consume ~50MB for the model pool.
  ///
  /// Throws an exception if the model fails to load or TFLite library is unavailable.
  Future<void> initialize(PoseLandmarkModel model) async {
    if (_isInitialized) await dispose();
    await ensureTFLiteLoaded();

    final String path = _getModelPath(model);

    // Create pool of interpreter instances
    for (int i = 0; i < _poolSize; i++) {
      final interpreter = await Interpreter.fromAsset(path);
      interpreter.resizeInputTensor(0, [1, 256, 256, 3]);
      interpreter.allocateTensors();

      final isolateInterpreter =
          await IsolateInterpreter.create(address: interpreter.address);

      _interpreterPool.add(_InterpreterInstance(
        interpreter: interpreter,
        isolateInterpreter: isolateInterpreter,
      ));

      // Mark interpreter as available
      _availableInterpreters.add(i);
    }

    _isInitialized = true;
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
  /// Any pending wait queue requests will be cancelled.
  /// After disposal, [initialize] must be called again before using the runner.
  Future<void> dispose() async {
    // Cancel any pending requests
    for (final completer in _waitQueue) {
      if (!completer.isCompleted) {
        completer.completeError(
          StateError(
              'PoseLandmarkModelRunner disposed while waiting for interpreter'),
        );
      }
    }
    _waitQueue.clear();

    // Dispose all interpreter instances
    for (final instance in _interpreterPool) {
      await instance.dispose();
    }
    _interpreterPool.clear();
    _availableInterpreters.clear();

    _isInitialized = false;
  }

  /// Acquires an available interpreter from the pool.
  ///
  /// If all interpreters are currently in use, this method will wait until
  /// one becomes available. Uses a FIFO queue to ensure fair resource allocation.
  ///
  /// Returns the index of the acquired interpreter in [_interpreterPool].
  ///
  /// This is an internal method - callers should use [run] which handles
  /// acquisition and release automatically.
  Future<int> _acquireInterpreter() async {
    // Fast path: interpreter immediately available
    if (_availableInterpreters.isNotEmpty) {
      return _availableInterpreters.removeLast();
    }

    // Slow path: must wait for an interpreter to be released
    final completer = Completer<int>();
    _waitQueue.add(completer);
    return completer.future;
  }

  /// Releases an interpreter back to the pool.
  ///
  /// If there are pending requests in the wait queue, the interpreter is
  /// immediately assigned to the next waiting request. Otherwise, it's
  /// returned to the available pool.
  ///
  /// Parameters:
  /// - [index]: The interpreter index to release (obtained from [_acquireInterpreter])
  void _releaseInterpreter(int index) {
    // If someone is waiting, give them this interpreter immediately
    if (_waitQueue.isNotEmpty) {
      final completer = _waitQueue.removeAt(0);
      if (!completer.isCompleted) {
        completer.complete(index);
      }
    } else {
      // Return to available pool
      _availableInterpreters.add(index);
    }
  }

  /// Runs landmark extraction on a person crop image.
  ///
  /// Extracts 33 body landmarks from the input person crop using the BlazePose model.
  /// The input image should be a cropped person region, ideally from the YOLOv8 detector.
  ///
  /// **Thread-safety:** This method is safe to call concurrently. It automatically
  /// acquires an available interpreter from the pool, runs inference, and releases
  /// the interpreter back to the pool. If all interpreters are busy, the call will
  /// wait until one becomes available.
  ///
  /// The method performs:
  /// 1. Acquires an interpreter from the pool (waits if all are busy)
  /// 2. Converts image to tensor (NHWC format, normalized 0-1)
  /// 3. Runs model inference via IsolateInterpreter
  /// 4. Post-processes results: sigmoid activation, coordinate normalization
  /// 5. Releases interpreter back to pool
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

    // Acquire an interpreter from the pool
    final interpreterIndex = await _acquireInterpreter();

    try {
      // Allocate fresh buffers for this inference call
      final inputBuffer = ImageUtils.imageToNHWC4D(roiImage, 256, 256);

      final outputLandmarks = [List.filled(195, 0.0)];
      final outputScore = [
        [0.0]
      ];
      final outputMask = ImageUtils.reshapeToTensor4D(
        List.filled(1 * 256 * 256 * 1, 0.0),
        1,
        256,
        256,
        1,
      );
      final outputHeatmap = ImageUtils.reshapeToTensor4D(
        List.filled(1 * 64 * 64 * 39, 0.0),
        1,
        64,
        64,
        39,
      );
      final outputWorld = [List.filled(117, 0.0)];

      // Run inference on the acquired interpreter
      final instance = _interpreterPool[interpreterIndex];
      await instance.isolateInterpreter.runForMultipleInputs(
        [inputBuffer],
        {
          0: outputLandmarks,
          1: outputScore,
          2: outputMask,
          3: outputHeatmap,
          4: outputWorld,
        },
      );

      return _parseLandmarks(outputLandmarks, outputScore, outputWorld);
    } finally {
      // Always release the interpreter, even if inference fails
      _releaseInterpreter(interpreterIndex);
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
