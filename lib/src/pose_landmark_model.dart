import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';
import 'types.dart';

class PoseLandmarkModelRunner {
  Interpreter? _interpreter;
  bool _isInitialized = false;

  static ffi.DynamicLibrary? _tfliteLib;

  List<List<List<List<double>>>>? _inputBuffer;
  List<List<double>>? _outputLandmarks;
  List<List<double>>? _outputScore;
  List<List<List<List<double>>>>? _outputMask;
  List<List<List<List<double>>>>? _outputHeatmap;
  List<List<double>>? _outputWorld;

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

    for (final c in candidates) {
      try {
        if (c.contains(p.separator)) {
          if (!File(c).existsSync()) continue;
        }
        _tfliteLib = ffi.DynamicLibrary.open(c);
        return;
      } catch (_) {}
    }
  }

  Future<void> initialize(PoseLandmarkModel model) async {
    if (_isInitialized) await dispose();
    await ensureTFLiteLoaded();

    final path = _getModelPath(model);
    _interpreter = await Interpreter.fromAsset(path);
    _interpreter!.resizeInputTensor(0, [1, 256, 256, 3]);
    _interpreter!.allocateTensors();

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

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
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

  PoseLandmarks run(img.Image roiImage) {
    _inputBuffer = ImageUtils.imageToNHWC4D(roiImage, 256, 256, reuse: _inputBuffer);

    _outputLandmarks ??= [List.filled(195, 0.0)];
    _outputScore ??= [[0.0]];
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

    return _parseLandmarks(_outputLandmarks!, _outputScore!, _outputWorld!);
  }

  PoseLandmarks _parseLandmarks(
    List<dynamic> landmarksData,
    List<dynamic> scoreData,
    List<dynamic> worldData,
  ) {
    double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    double clamp01(double v) => v.isNaN ? 0.0 : v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);

    final score = sigmoid(scoreData[0][0] as double);
    final raw = landmarksData[0] as List<dynamic>;
    final lm = <PoseLandmark>[];

    for (int i = 0; i < 33; i++) {
      final base = i * 5;
      final x = clamp01((raw[base + 0] as double) / 256.0);
      final y = clamp01((raw[base + 1] as double) / 256.0);
      final z = raw[base + 2] as double;
      final visibility = sigmoid(raw[base + 3] as double);
      final presence = sigmoid(raw[base + 4] as double);
      final vis = (visibility * presence).clamp(0.0, 1.0);

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
