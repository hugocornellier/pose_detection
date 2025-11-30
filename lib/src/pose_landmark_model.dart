import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';
import 'types.dart';

/// BlazePose landmark extraction model runner for Stage 2 of the pose detection pipeline.
///
/// Extracts 33 body landmarks from person crops using the BlazePose model.
/// Supports three model variants (lite, full, heavy) with different accuracy/performance trade-offs.
/// Runs synchronously since it processes one detection at a time.
class PoseLandmarkModelRunner {
  Interpreter? _interpreter;
  IsolateInterpreter? _iso;
  bool _isInitialized = false;
  static ffi.DynamicLibrary? _tfliteLib;
  List<List<List<List<double>>>>? _inputBuffer;
  List<List<double>>? _outputLandmarks;
  List<List<double>>? _outputScore;
  List<List<List<List<double>>>>? _outputMask;
  List<List<List<List<double>>>>? _outputHeatmap;
  List<List<double>>? _outputWorld;

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
  /// Loads the selected model (lite/full/heavy) from assets, resizes input tensor
  /// to 256x256x3, and creates an IsolateInterpreter for async inference.
  ///
  /// Parameters:
  /// - [model]: Which BlazePose variant to use (lite, full, or heavy)
  ///
  /// If already initialized, this will dispose the previous instance first.
  ///
  /// Throws an exception if the model fails to load or TFLite library is unavailable.
  Future<void> initialize(PoseLandmarkModel model) async {
    if (_isInitialized) await dispose();
    await ensureTFLiteLoaded();

    final String path = _getModelPath(model);
    _interpreter = await Interpreter.fromAsset(path);
    _interpreter!.resizeInputTensor(0, [1, 256, 256, 3]);
    _interpreter!.allocateTensors();

    _iso = await IsolateInterpreter.create(address: _interpreter!.address);

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

  /// Disposes the model runner and releases all resources.
  ///
  /// Closes the interpreter, isolate interpreter, and clears all tensor buffers.
  /// After disposal, [initialize] must be called again before using the runner.
  Future<void> dispose() async {
    _iso?.close();
    _iso = null;
    _interpreter?.close();
    _interpreter = null;
    _inputBuffer = null;
    _outputLandmarks = null;
    _outputScore = null;
    _outputMask = null;
    _outputHeatmap = null;
    _outputWorld = null;
    _isInitialized = false;
  }

  /// Runs landmark extraction on a person crop image.
  ///
  /// Extracts 33 body landmarks from the input person crop using the BlazePose model.
  /// The input image should be a cropped person region, ideally from the YOLOv8 detector.
  ///
  /// The method performs:
  /// 1. Image to tensor conversion (NHWC format, normalized 0-1)
  /// 2. Model inference via IsolateInterpreter
  /// 3. Post-processing: sigmoid activation, coordinate normalization
  ///
  /// Parameters:
  /// - [roiImage]: Cropped person image (will be resized to 256x256 internally)
  ///
  /// Returns [PoseLandmarks] containing 33 landmarks with normalized coordinates (0-1 range)
  /// and a confidence score. Landmarks are in the 256x256 model output space.
  ///
  /// Throws [StateError] if the model is not initialized.
  Future<PoseLandmarks> run(img.Image roiImage) async {
    _inputBuffer =
        ImageUtils.imageToNHWC4D(roiImage, 256, 256, reuse: _inputBuffer);

    _outputLandmarks ??= [List.filled(195, 0.0)];
    _outputScore ??= [
      [0.0]
    ];
    _outputMask ??= ImageUtils.reshapeToTensor4D(
      List.filled(1 * 256 * 256 * 1, 0.0),
      1,
      256,
      256,
      1,
    );
    _outputHeatmap ??= ImageUtils.reshapeToTensor4D(
      List.filled(1 * 64 * 64 * 39, 0.0),
      1,
      64,
      64,
      39,
    );
    _outputWorld ??= [List.filled(117, 0.0)];

    if (_iso != null) {
      await _iso!.runForMultipleInputs(
        [_inputBuffer!],
        {
          0: _outputLandmarks!,
          1: _outputScore!,
          2: _outputMask!,
          3: _outputHeatmap!,
          4: _outputWorld!,
        },
      );
    } else {
      _interpreter!.runForMultipleInputs(
        [_inputBuffer!],
        {
          0: _outputLandmarks!,
          1: _outputScore!,
          2: _outputMask!,
          3: _outputHeatmap!,
          4: _outputWorld!,
        },
      );
    }

    return _parseLandmarks(_outputLandmarks!, _outputScore!, _outputWorld!);
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
