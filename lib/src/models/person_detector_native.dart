import 'dart:typed_data';
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

    final (options, newDelegate) = InterpreterFactory.create(performanceConfig);
    _delegate = newDelegate;

    final Interpreter itp = await Interpreter.fromAsset(
      assetPath,
      options: options,
    );
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

    _iso = await InterpreterFactory.createIsolateIfNeeded(itp, _delegate);

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
    if (!isInitializedFlag || interpreter == null) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    final (cv.Mat letter, double r, int dw, int dh) =
        NativeImageUtils.letterbox(mat, inW, inH);

    final int inputSize = inH * inW * 3;
    inputBuffer ??= Float32List(inputSize);
    NativeImageUtils.matToTensorYolo(letter, buffer: inputBuffer);
    letter.dispose();

    final int inputCount = interpreter!.getInputTensors().length;
    final List<Object> inputs = List<Object>.filled(
      inputCount,
      inputBuffer!.buffer,
      growable: false,
    );

    cachedOutputs ??= createOutputBuffers(outShapes);
    zeroOutputBuffers(cachedOutputs!, outShapes);

    if (_iso != null) {
      await _iso!.runForMultipleInputs(inputs, cachedOutputs!);
    } else {
      interpreter!.runForMultipleInputs(inputs, cachedOutputs!);
    }

    return postProcessDetections(
      outputs: cachedOutputs!.values.toList(),
      inputWidth: inW,
      inputHeight: inH,
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
