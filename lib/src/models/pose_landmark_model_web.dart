// ignore_for_file: implementation_imports

import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/src/web/js_interop/tfjs_tensor.dart';
import 'package:flutter_litert/src/web/model.dart' as litert_web;
import '../types.dart';
import '../util/pose_helpers.dart';
import '../util/web_image_utils.dart';

/// Web implementation of BlazePose landmark extraction model runner.
///
/// Extracts 33 body landmarks from person crops using the BlazePose model.
/// Supports three model variants (lite, full, heavy) with different
/// accuracy/performance trade-offs.
///
/// Key differences from native:
/// - No opencv_dart (no cv.Mat) -- input is RGBA Uint8List from Canvas
/// - No dart:io
/// - No IsolateInterpreter (uses a single web model instance)
/// - No delegates (always CPU/WASM on web)
/// - Single model instance (no pool -- JS is single-threaded)
class PoseLandmarkModelRunner {
  litert_web.Model? _model;
  bool _isInitialized = false;
  Float32List? _inputBufferFlat;

  /// Creates a landmark model runner.
  ///
  /// [poolSize] is accepted for API compatibility but ignored on web
  /// (JS is single-threaded, so a single model instance is used).
  PoseLandmarkModelRunner({int poolSize = 1});

  /// Initializes the BlazePose landmark model with the specified variant.
  ///
  /// Parameters:
  /// - [model]: Which BlazePose variant to use (lite, full, or heavy)
  /// - [performanceConfig]: Accepted for API compatibility but ignored on web
  ///   (web always uses CPU/WASM execution via TFLite.js)
  ///
  /// If already initialized, this will dispose the previous instance first.
  Future<void> initialize(
    PoseLandmarkModel model, {
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    final path = poseLandmarkModelPath(model);
    final ByteData rawAssetFile = await rootBundle.load(path);
    final litert_web.Model loadedModel = await litert_web.Model.fromBuffer(
      rawAssetFile.buffer.asUint8List(),
    );
    _model = loadedModel;
    _inputBufferFlat = Float32List(256 * 256 * 3);
    _isInitialized = true;
  }

  /// Returns true if the model runner has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Always returns 1 on web (single-threaded).
  int get poolSize => 1;

  /// Disposes the model runner and releases all resources.
  ///
  /// After disposal, [initialize] must be called again before using the runner.
  Future<void> dispose() async {
    _model?.delete();
    _model = null;
    _inputBufferFlat = null;
    _isInitialized = false;
  }

  /// Runs landmark extraction from RGBA pixel data (256x256).
  ///
  /// On web, input is RGBA bytes from Canvas getImageData instead of cv.Mat.
  /// The RGBA data is converted to normalized RGB float32 tensor for the model.
  ///
  /// Parameters:
  /// - [rgbaData]: 256x256 RGBA pixel data (length must be 256*256*4 = 262144)
  ///
  /// Returns [PoseLandmarks] containing 33 landmarks with normalized coordinates.
  ///
  /// Throws [StateError] if the model is not initialized.
  Future<PoseLandmarks> runFromRgba(Uint8List rgbaData) async {
    if (!_isInitialized || _model == null) {
      throw StateError(
        'PoseLandmarkModelRunner not initialized. Call initialize() first.',
      );
    }

    // Convert RGBA to normalized RGB float32
    final inputFlat = _inputBufferFlat!;
    rgbaToRgbFloat32(rgbaData, inputFlat);

    final JSTensor inputTensor = JSTensor(
      inputFlat,
      shape: <int>[1, 256, 256, 3],
    );
    dynamic rawOutput;
    try {
      rawOutput = _model!.base.predict<dynamic>(inputTensor);
    } finally {
      inputTensor.dispose();
    }

    final List<dynamic> outputs = _extractOutputTensors(rawOutput);
    final dynamic landmarksData = outputs[0];
    final dynamic scoreData = outputs[1];
    final dynamic worldData = outputs[4];

    if (landmarksData == null || scoreData == null || worldData == null) {
      _disposeOutputTensors(rawOutput);
      throw StateError(
        'BlazePose web model did not return expected outputs (landmarks/score/world).',
      );
    }

    try {
      return parsePoseLandmarks(
        landmarksData as List<dynamic>,
        scoreData as List<dynamic>,
      );
    } finally {
      _disposeOutputTensors(rawOutput);
    }
  }

  /// Extracts BlazePose outputs from either a JS array or a named JS output map.
  ///
  /// TFLite.js often returns multi-output models as a named JS object
  /// (`Identity`, `Identity_1`, ...). The current web interpreter path in
  /// `flutter_litert` does not populate preallocated outputs in that case, so
  /// we read and convert the returned tensors directly here.
  List<dynamic> _extractOutputTensors(dynamic rawOutput) {
    final List<dynamic> outputs = List<dynamic>.filled(
      5,
      null,
      growable: false,
    );

    if (rawOutput is List) {
      for (int i = 0; i < rawOutput.length && i < outputs.length; i++) {
        outputs[i] = _toDartOutput(rawOutput[i], i);
      }
      return outputs;
    }

    final JSObject jsObj = rawOutput as JSObject;
    for (int i = 0; i < outputs.length; i++) {
      final String key = i == 0 ? 'Identity' : 'Identity_$i';
      final dynamic tensor = jsObj[key];
      if (tensor != null) {
        outputs[i] = _toDartOutput(tensor, i);
      }
    }
    return outputs;
  }

  /// Converts a JS output tensor to the nested list shape expected by `_parseLandmarks`.
  dynamic _toDartOutput(dynamic tensor, int outputIndex) {
    final JSTensor jsTensor;
    try {
      jsTensor = tensor as JSTensor;
    } catch (_) {
      return tensor;
    }

    final List<double> flat = (jsTensor.dataSync<List<double>>())
        .cast<double>();
    switch (outputIndex) {
      case 0:
      case 4:
        return <List<double>>[flat];
      case 1:
        return <List<double>>[
          <double>[flat.isEmpty ? 0.0 : flat.first],
        ];
      default:
        return flat;
    }
  }

  /// Disposes output tensors returned by TFLite.js (both JS array and named map forms).
  void _disposeOutputTensors(dynamic rawOutput) {
    if (rawOutput is List) {
      for (final dynamic tensor in rawOutput) {
        try {
          (tensor as JSTensor).dispose();
        } catch (_) {}
      }
      return;
    }

    try {
      final JSObject jsObj = rawOutput as JSObject;
      for (int i = 0; i < 5; i++) {
        final String key = i == 0 ? 'Identity' : 'Identity_$i';
        final dynamic tensor = jsObj[key];
        try {
          (tensor as JSTensor).dispose();
        } catch (_) {}
      }
    } catch (_) {}
  }
}
