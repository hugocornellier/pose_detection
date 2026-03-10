import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/native_image_utils.dart';

/// YOLOv8n-based person detector for Stage 1 of the pose detection pipeline.
///
/// Detects persons in images and returns bounding boxes. Uses the YOLOv8n model
/// trained on COCO dataset with 640x640 input resolution. Runs asynchronously
/// via IsolateInterpreter for better performance.
class YoloV8PersonDetector {
  IsolateInterpreter? _iso;
  Interpreter? _interpreter;
  bool _isInitialized = false;
  late int _inW;
  late int _inH;

  /// COCO dataset class ID for the "person" class.
  static const int cocoPersonClassId = 0;
  final _outShapes = <List<int>>[];
  Float32List? _inputBuffer;
  Delegate? _delegate;
  Map<int, Object>? _cachedOutputs;

  /// Initializes the YOLOv8 person detector by loading the model.
  ///
  /// Parameters:
  /// - [performanceConfig]: Optional performance configuration for TFLite delegates.
  ///   Defaults to no delegates (backward compatible). Use [PerformanceConfig.xnnpack()]
  ///   for CPU optimization.
  ///
  /// Loads the YOLOv8n model from assets, allocates tensors, and creates
  /// an IsolateInterpreter for async inference. Must be called before [detect].
  ///
  /// If already initialized, this will dispose the previous instance first.
  ///
  /// Throws an exception if the model fails to load.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/pose_detection/assets/models/yolov8n_float32.tflite';
    if (_isInitialized) await dispose();

    final (options, newDelegate) = InterpreterFactory.create(performanceConfig);
    _delegate = newDelegate;

    final Interpreter itp = await Interpreter.fromAsset(
      assetPath,
      options: options,
    );
    _interpreter = itp;
    itp.allocateTensors();

    final Tensor inTensor = itp.getInputTensor(0);
    final List<int> inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    _outShapes.clear();
    final List<Tensor> outs = itp.getOutputTensors();
    for (final Tensor t in outs) {
      _outShapes.add(List<int>.from(t.shape));
    }

    _iso = await InterpreterFactory.createIsolateIfNeeded(itp, _delegate);

    _isInitialized = true;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases all resources.
  ///
  /// Closes the interpreter, isolate interpreter, clears canvas buffer,
  /// and deletes any allocated delegates.
  /// After disposal, [initialize] must be called again before using the detector.
  Future<void> dispose() async {
    _iso?.close();
    _iso = null;
    _interpreter?.close();
    _interpreter = null;
    _delegate?.delete();
    _delegate = null;
    _cachedOutputs = null;
    _isInitialized = false;
  }

  /// Creates pre-allocated output buffers based on cached output shapes.
  Map<int, Object> _createOutputBuffers() => createOutputBuffers(_outShapes);

  /// Zeros out pre-allocated output buffers for reuse.
  void _zeroOutputBuffers(Map<int, Object> outputs) =>
      zeroOutputBuffers(outputs, _outShapes);

  /// Exposes detection output decoding for tests.
  @visibleForTesting
  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) {
    return decodeAndSplitOutputs(outputs);
  }

  /// Configures internal state for unit tests without loading native assets.
  @visibleForTesting
  void debugConfigureForTest({
    required int inputWidth,
    required int inputHeight,
    required List<List<int>> outputShapes,
    Interpreter? interpreter,
    IsolateInterpreter? isolate,
    Float32List? inputBuffer,
    bool initialized = true,
  }) {
    _inW = inputWidth;
    _inH = inputHeight;
    _outShapes
      ..clear()
      ..addAll(outputShapes.map((s) => List<int>.from(s)));
    _interpreter = interpreter;
    _iso = isolate;
    _inputBuffer = inputBuffer;
    _isInitialized = initialized;
  }

  /// Detects persons in a cv.Mat using native OpenCV preprocessing.
  ///
  /// Uses SIMD-accelerated OpenCV operations for preprocessing which is
  /// 5-15x faster than pure Dart preprocessing.
  ///
  /// Parameters:
  /// - [mat]: Input cv.Mat image in BGR format
  /// - [imageWidth]: Original image width for coordinate scaling
  /// - [imageHeight]: Original image height for coordinate scaling
  /// - [confThres]: Confidence threshold for detections (default: 0.35)
  /// - [iouThres]: IoU threshold for Non-Maximum Suppression (default: 0.4)
  /// - [maxDet]: Maximum detections to return after NMS (default: 10)
  /// - [personOnly]: If true, only returns person class detections (default: true)
  ///
  /// Returns a list of [Detection] objects with bounding boxes in original image coordinates.
  Future<List<Detection>> detect(
    cv.Mat mat, {
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

    final (cv.Mat letter, double r, int dw, int dh) =
        NativeImageUtils.letterbox(mat, _inW, _inH);

    final int inputSize = _inH * _inW * 3;
    _inputBuffer ??= Float32List(inputSize);
    if (_inputBuffer!.length != inputSize) {
      _inputBuffer = Float32List(inputSize);
    }
    NativeImageUtils.matToTensorYolo(letter, buffer: _inputBuffer);
    letter.dispose();

    final int inputCount = _interpreter!.getInputTensors().length;
    final List<Object> inputs = List<Object>.filled(
      inputCount,
      _inputBuffer!.buffer,
      growable: false,
    );

    _cachedOutputs ??= _createOutputBuffers();
    _zeroOutputBuffers(_cachedOutputs!);

    if (_iso != null) {
      await _iso!.runForMultipleInputs(inputs, _cachedOutputs!);
    } else {
      _interpreter!.runForMultipleInputs(inputs, _cachedOutputs!);
    }

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
}
