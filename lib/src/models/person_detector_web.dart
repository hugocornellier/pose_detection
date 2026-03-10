import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:web/web.dart' as web;
import 'package:flutter_litert/flutter_litert.dart';

/// Web implementation of YOLOv8n person detector.
///
/// Uses Canvas API for image preprocessing instead of OpenCV.
/// Runs on CPU/WASM (no GPU delegates on web).
/// Uses [Interpreter] directly (no [IsolateInterpreter] on web).
class YoloV8PersonDetector {
  Interpreter? _interpreter;
  bool _isInitialized = false;
  late int _inW;
  late int _inH;
  final _outShapes = <List<int>>[];
  Float32List? _inputBuffer;
  web.HTMLCanvasElement? _canvasElement;
  web.CanvasRenderingContext2D? _canvasCtx;
  Map<int, Object>? _cachedOutputs;

  /// COCO dataset class ID for the "person" class.
  static const int cocoPersonClassId = 0;

  /// Initializes the YOLOv8 person detector by loading the model.
  ///
  /// Parameters:
  /// - [performanceConfig]: Accepted for API compatibility but ignored on web
  ///   (always runs CPU/WASM).
  ///
  /// Loads the YOLOv8n model from assets, allocates tensors, and creates
  /// a canvas element for letterbox preprocessing. Must be called before [detect].
  ///
  /// If already initialized, this will dispose the previous instance first.
  ///
  /// Throws an exception if the model fails to load.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/pose_detection/assets/models/yolov8n_float32.tflite';
    if (_isInitialized) await dispose();

    final options = InterpreterOptions();
    final interpreter = await Interpreter.fromAsset(
      assetPath,
      options: options,
    );
    _interpreter = interpreter;
    interpreter.allocateTensors();

    final Tensor inTensor = interpreter.getInputTensor(0);
    final List<int> inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    _outShapes.clear();
    final List<Tensor> outs = interpreter.getOutputTensors();
    for (final Tensor t in outs) {
      _outShapes.add(List<int>.from(t.shape));
    }

    _inputBuffer = Float32List(_inH * _inW * 3);

    // Create canvas for letterbox preprocessing
    _canvasElement = web.HTMLCanvasElement();
    _canvasElement!.width = _inW;
    _canvasElement!.height = _inH;
    _canvasCtx =
        _canvasElement!.getContext('2d') as web.CanvasRenderingContext2D;

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases all resources.
  ///
  /// Closes the interpreter, clears canvas buffer, and releases output buffers.
  /// After disposal, [initialize] must be called again before using the detector.
  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
    _canvasElement = null;
    _canvasCtx = null;
    _inputBuffer = null;
    _cachedOutputs = null;
    _isInitialized = false;
  }

  /// Detects persons in an HTML image element using Canvas API preprocessing.
  ///
  /// Uses GPU-accelerated Canvas drawImage for letterboxing and getImageData
  /// for pixel extraction.
  ///
  /// Parameters:
  /// - [htmlImage]: Input HTML image element
  /// - [imageWidth]: Original image width for coordinate scaling
  /// - [imageHeight]: Original image height for coordinate scaling
  /// - [confThres]: Confidence threshold for detections (default: 0.35)
  /// - [iouThres]: IoU threshold for Non-Maximum Suppression (default: 0.4)
  /// - [maxDet]: Maximum detections to return after NMS (default: 10)
  /// - [personOnly]: If true, only returns person class detections (default: true)
  ///
  /// Returns a list of [Detection] objects with bounding boxes in original
  /// image coordinates.
  Future<List<Detection>> detect(
    web.HTMLImageElement htmlImage, {
    required int imageWidth,
    required int imageHeight,
    double confThres = 0.35,
    double iouThres = 0.4,
    int maxDet = 10,
    bool personOnly = true,
  }) async {
    if (!_isInitialized || _interpreter == null) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    // GPU-accelerated letterbox via Canvas drawImage
    final double r = math.min(_inH / imageHeight, _inW / imageWidth);
    final int nw = (imageWidth * r).round();
    final int nh = (imageHeight * r).round();
    final int dw = (_inW - nw) ~/ 2;
    final int dh = (_inH - nh) ~/ 2;

    final web.CanvasRenderingContext2D ctx = _canvasCtx!;
    ctx.fillStyle = 'rgb(114,114,114)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    ctx.drawImage(htmlImage, 0, 0, imageWidth, imageHeight, dw, dh, nw, nh);

    // Extract pixel data
    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;

    // Normalize RGBA -> RGB [0,1]
    final Float32List inputFlat = _inputBuffer!;
    const double norm = 1.0 / 255.0;
    int dst = 0;
    for (int src = 0; src < rgba.length; src += 4) {
      inputFlat[dst++] = rgba[src] * norm; // R
      inputFlat[dst++] = rgba[src + 1] * norm; // G
      inputFlat[dst++] = rgba[src + 2] * norm; // B
    }

    // Run inference
    _cachedOutputs ??= _createOutputBuffers();
    _zeroOutputBuffers(_cachedOutputs!);

    _interpreter!.runForMultipleInputs([inputFlat.buffer], _cachedOutputs!);

    return postProcessDetections(
      outputs: _cachedOutputs!.values.toList(),
      inputWidth: _inW,
      inputHeight: _inH,
      r: r,
      dw: dw,
      dh: dh,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: confThres,
      iouThres: iouThres,
      topkPreNms: 0,
      maxDet: maxDet,
      filterClassId: personOnly ? cocoPersonClassId : null,
    );
  }

  // ---------------------------------------------------------------------------
  // Private helper methods (pure Dart math, no platform dependencies)
  // ---------------------------------------------------------------------------

  /// Creates pre-allocated output buffers based on cached output shapes.
  Map<int, Object> _createOutputBuffers() => createOutputBuffers(_outShapes);

  /// Zeros out pre-allocated output buffers for reuse.
  void _zeroOutputBuffers(Map<int, Object> outputs) =>
      zeroOutputBuffers(outputs, _outShapes);

  /// Exposes detection output decoding for tests.
  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) {
    return decodeAndSplitOutputs(outputs);
  }
}
