import 'dart:typed_data';
import 'package:flutter/foundation.dart' show setEquals;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/native_image_utils.dart';
import 'person_detector_base.dart';

/// YOLOv8n-based person detector for Stage 1 of the pose detection pipeline.
///
/// Detects persons in images and returns bounding boxes. Uses the YOLOv8n model
/// trained on COCO dataset with 640x640 input resolution. Runs asynchronously
/// via IsolateInterpreter for better performance.
class YoloV8PersonDetector extends PersonDetectorBase {
  IsolateInterpreter? _iso;
  Delegate? _delegate;

  /// LiteRT Next CompiledModel, used instead of [interpreter]/[_iso] when the
  /// detector is initialized with `useCompiledModel: true`. Created from bytes
  /// inside the background isolate.
  CompiledModel? _compiled;

  /// YOLO output-head layout, resolved once at init from the output tensor
  /// shape. Used by the flat decoder ([postProcessDetectionsFlat]) to index without
  /// rebuilding the dense tensor as nested lists.
  int _yoloChannels = 0;
  int _yoloAnchors = 0;
  bool _yoloChannelMajor = true;

  /// Flat output buffer reused by the interpreter path (the CompiledModel path
  /// reads the flat list returned by runAsync directly).
  Float32List? _flatOut;

  /// COCO dataset class ID for the "person" class.
  static const int cocoPersonClassId = 0;

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
    if (isInitializedFlag) await dispose();
    await _initWith(
      (options) => Interpreter.fromAsset(assetPath, options: options),
      performanceConfig,
      useIsolateInterpreter: true,
    );
  }

  /// Initializes from pre-loaded model bytes (used inside a background isolate
  /// where Flutter asset loading is not available).
  ///
  /// When [useCompiledModel] is true, a LiteRT Next [CompiledModel] backs
  /// inference instead of an [Interpreter]. [compiledForceCpu] pins the
  /// CompiledModel to CPU (no GPU attempt); the caller uses this on iOS to
  /// avoid Metal floating-point variance changing detection counts.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
    bool useCompiledModel = false,
    bool compiledForceCpu = false,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (isInitializedFlag) await dispose();
    if (useCompiledModel) {
      await _initCompiled(
        modelBytes,
        forceCpu: compiledForceCpu,
        accelerators: accelerators,
        precision: precision,
      );
      return;
    }
    await _initWith(
      (options) async => Interpreter.fromBuffer(modelBytes, options: options),
      performanceConfig,
      useIsolateInterpreter: false,
    );
  }

  /// Builds the CompiledModel path. Input/output shapes are read once from a
  /// plain throwaway [Interpreter] because [CompiledModel] exposes only byte
  /// sizes, not tensor shapes, and the flat YOLO decoder needs the output
  /// shape to index the flat CompiledModel output.
  Future<void> _initCompiled(
    Uint8List modelBytes, {
    required bool forceCpu,
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    final Interpreter probe = Interpreter.fromBuffer(modelBytes);
    try {
      probe.allocateTensors();
      final List<int> inShape = probe.getInputTensor(0).shape;
      inH = inShape[1];
      inW = inShape[2];
      outShapes.clear();
      for (final Tensor t in probe.getOutputTensors()) {
        outShapes.add(List<int>.from(t.shape));
      }
    } finally {
      probe.close();
    }
    _computeYoloLayout();

    final effectiveAccelerators = forceCpu
        ? const {Accelerator.cpu}
        : accelerators;
    _compiled =
        setEquals(effectiveAccelerators, const {
          Accelerator.gpu,
          Accelerator.cpu,
        })
        ? CompiledModel.fromBufferWithGpuFallback(
            modelBytes,
            precision: precision,
          )
        : CompiledModel.fromBuffer(
            modelBytes,
            accelerators: effectiveAccelerators,
            precision: precision,
          );
    isInitializedFlag = true;
  }

  /// Resolves [_yoloChannels]/[_yoloAnchors]/[_yoloChannelMajor] from the
  /// single output tensor's shape `[1, d1, d2]`, matching the layout heuristic
  /// in flutter_litert's decodeDetectionOutputs (channel-major when the smaller
  /// dim is 84/85). Also allocates the reusable flat output buffer.
  void _computeYoloLayout() {
    final List<int> shape = outShapes[0];
    final int d1 = shape[shape.length - 2];
    final int d2 = shape[shape.length - 1];
    if (d1 < d2 && (d1 == 84 || d1 == 85)) {
      _yoloChannels = d1;
      _yoloAnchors = d2;
      _yoloChannelMajor = true;
    } else {
      _yoloChannels = d2;
      _yoloAnchors = d1;
      _yoloChannelMajor = false;
    }
    _flatOut = Float32List(_yoloChannels * _yoloAnchors);
  }

  Future<void> _initWith(
    Future<Interpreter> Function(InterpreterOptions) loader,
    PerformanceConfig? performanceConfig, {
    required bool useIsolateInterpreter,
  }) async {
    final (options, newDelegate) = InterpreterFactory.create(performanceConfig);
    _delegate = newDelegate;

    final Interpreter itp = await loader(options);
    interpreter = itp;
    itp.allocateTensors();

    final Tensor inTensor = itp.getInputTensor(0);
    final List<int> inShape = inTensor.shape;
    inH = inShape[1];
    inW = inShape[2];

    outShapes.clear();
    final List<Tensor> outs = itp.getOutputTensors();
    for (final Tensor t in outs) {
      outShapes.add(List<int>.from(t.shape));
    }
    _computeYoloLayout();

    if (useIsolateInterpreter) {
      _iso = await InterpreterFactory.createIsolateIfNeeded(itp, _delegate);
    }

    isInitializedFlag = true;
  }

  /// Disposes the detector and releases all resources.
  ///
  /// Closes the interpreter, isolate interpreter, clears canvas buffer,
  /// and deletes any allocated delegates.
  /// After disposal, [initialize] must be called again before using the detector.
  Future<void> dispose() async {
    _iso?.close();
    _iso = null;
    _delegate?.delete();
    _delegate = null;
    _compiled?.close();
    _compiled = null;
    disposeBase();
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
    if (!isInitializedFlag || (interpreter == null && _compiled == null)) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    final (cv.Mat letter, double r, int dw, int dh) =
        NativeImageUtils.letterbox(mat, inW, inH);

    final int inputSize = inH * inW * 3;
    inputBuffer ??= Float32List(inputSize);
    NativeImageUtils.matToTensorYolo(letter, buffer: inputBuffer);
    letter.dispose();

    final Float32List out0;
    if (_compiled != null) {
      final List<Float32List> flat = await _compiled!.runAsync([inputBuffer!]);
      out0 = flat[0];
    } else {
      final int inputCount = interpreter!.getInputTensors().length;
      final List<Object> inputs = List<Object>.filled(
        inputCount,
        inputBuffer!.buffer,
        growable: false,
      );
      // Write the detection head into a flat reusable buffer (raw float bytes),
      // so it can be decoded by index without a nested-list rebuild.
      final Map<int, Object> outputs = <int, Object>{0: _flatOut!.buffer};
      if (_iso != null) {
        await _iso!.runForMultipleInputs(inputs, outputs);
      } else {
        interpreter!.runForMultipleInputs(inputs, outputs);
      }
      out0 = _flatOut!;
    }

    return postProcessDetectionsFlat(
      out0,
      channels: _yoloChannels,
      anchors: _yoloAnchors,
      channelMajor: _yoloChannelMajor,
      inputWidth: inW,
      inputHeight: inH,
      r: r,
      dw: dw,
      dh: dh,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: confThres,
      iouThres: iouThres,
      maxDet: maxDet,
      filterClassId: personOnly ? cocoPersonClassId : null,
    );
  }
}
