// ignore_for_file: implementation_imports

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;
import 'package:flutter_litert/flutter_litert.dart';

import 'person_detector_base.dart';

/// Web implementation of YOLOv8n person detector.
///
/// Uses Canvas API for image preprocessing instead of OpenCV.
/// On the LiteRT.js path, runs on WebGPU when available and falls back to
/// SIMD-WASM otherwise. The legacy tflite-js path is CPU/WASM only.
/// Uses browser runtimes directly (no [IsolateInterpreter] on web).
class YoloV8PersonDetector extends PersonDetectorBase {
  web.HTMLCanvasElement? _canvasElement;
  web.CanvasRenderingContext2D? _canvasCtx;

  /// COCO dataset class ID for the "person" class.
  static const int cocoPersonClassId = 0;

  /// Optional LiteRT.js-backed interpreter; non-null when [initialize] was
  /// called with [useLiteRt: true].
  LiteRtInterpreter? _liteRtItp;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`), or
  /// null if running on the legacy tflite-js path.
  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null if running on the legacy tflite-js path.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  /// Reusable Float32List output buffer for the LiteRT.js path (avoids
  /// allocating a 705k-float buffer on every detect call).
  Float32List? _liteRtOutputFlat;

  /// Initializes the YOLOv8 person detector by loading the model.
  ///
  /// Parameters:
  /// - [performanceConfig]: Accepted for API compatibility. Web runtime
  ///   selection is controlled by [useLiteRt] and [liteRtAccelerator].
  ///
  /// Loads the YOLOv8n model from assets, allocates tensors, and creates
  /// a canvas element for letterbox preprocessing. Must be called before [detect].
  ///
  /// If already initialized, this will dispose the previous instance first.
  ///
  /// Throws an exception if the model fails to load.
  Future<void> initialize({
    PerformanceConfig? performanceConfig,
    bool useLiteRt = true,
    String liteRtAccelerator = 'auto',
  }) async {
    const String assetPath =
        'packages/pose_detection/assets/models/yolov8n_float32.tflite';
    if (isInitializedFlag) await dispose();

    if (useLiteRt) {
      final ByteData raw = await rootBundle.load(assetPath);
      // 'auto' picks WebGPU only on Chromium with a hardware adapter (see
      // resolveWebAccelerator); flutter_litert still falls back to WASM when
      // the WebGPU compile fails (e.g., unsupported ops).
      final String resolved = await resolveWebAccelerator(liteRtAccelerator);
      final lrt = await LiteRtInterpreter.fromBytes(
        raw.buffer.asUint8List(),
        accelerator: resolved,
      );
      _liteRtItp = lrt;
      _activeAccelerator = lrt.activeAccelerator;

      final inT = lrt.getInputTensor(0);
      inH = inT.shape[1];
      inW = inT.shape[2];

      outShapes.clear();
      for (final t in lrt.getOutputTensors()) {
        outShapes.add(List<int>.from(t.shape));
      }
      // Pre-size the flat output buffer for the single-output YOLO model.
      int outElems = 1;
      for (final d in outShapes[0]) {
        outElems *= d;
      }
      _liteRtOutputFlat = Float32List(outElems);
    } else {
      final options = InterpreterOptions();
      final itp = await Interpreter.fromAsset(assetPath, options: options);
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
    }

    inputBuffer = Float32List(inH * inW * 3);

    // Create canvas for letterbox preprocessing
    _canvasElement = web.HTMLCanvasElement();
    _canvasElement!.width = inW;
    _canvasElement!.height = inH;
    _canvasCtx =
        _canvasElement!.getContext('2d') as web.CanvasRenderingContext2D;

    isInitializedFlag = true;
  }

  /// Disposes the detector and releases all resources.
  ///
  /// Closes the interpreter, clears canvas buffer, and releases output buffers.
  /// After disposal, [initialize] must be called again before using the detector.
  Future<void> dispose() async {
    _canvasElement = null;
    _canvasCtx = null;
    inputBuffer = null;
    _liteRtItp?.close();
    _liteRtItp = null;
    _activeAccelerator = null;
    _liteRtOutputFlat = null;
    disposeBase();
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
    JSObject htmlImage, {
    required int imageWidth,
    required int imageHeight,
    double confThres = 0.35,
    double iouThres = 0.4,
    int maxDet = 10,
    bool personOnly = true,
    void Function(int us)? timingPreUs,
    void Function(int us)? timingInferUs,
  }) async {
    if (!isInitializedFlag || (interpreter == null && _liteRtItp == null)) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    final Stopwatch sw = (timingPreUs != null || timingInferUs != null)
        ? (Stopwatch()..start())
        : Stopwatch();

    // GPU-accelerated letterbox via Canvas drawImage
    final lb = computeLetterboxParams(
      srcWidth: imageWidth,
      srcHeight: imageHeight,
      targetWidth: inW,
      targetHeight: inH,
    );
    final int dw = lb.padLeft;
    final int dh = lb.padTop;
    final int nw = lb.newWidth;
    final int nh = lb.newHeight;

    final web.CanvasRenderingContext2D ctx = _canvasCtx!;
    ctx.fillStyle = 'rgb(114,114,114)'.toJS;
    ctx.fillRect(0, 0, inW, inH);
    ctx.drawImage(htmlImage, 0, 0, imageWidth, imageHeight, dw, dh, nw, nh);

    // Extract pixel data
    final web.ImageData imageData = ctx.getImageData(0, 0, inW, inH);
    final rgba = imageData.data.toDart;

    // Normalize RGBA -> RGB [0,1]
    final Float32List inputFlat = inputBuffer!;
    rgbaToRgbFloat32(Uint8List.view(rgba.buffer), inputFlat);

    if (timingPreUs != null) {
      sw.stop();
      timingPreUs(sw.elapsedMicroseconds);
      sw.reset();
      sw.start();
    }

    // Run inference
    if (_liteRtItp != null) {
      final out = _liteRtOutputFlat!;
      await _liteRtItp!.runForMultipleInputs(
        <Object>[inputFlat],
        <int, Object>{0: out},
      );
      if (timingInferUs != null) {
        sw.stop();
        timingInferUs(sw.elapsedMicroseconds);
      }
      return postProcessDetectionsFlat(
        out,
        channels: outShapes[0][1],
        anchors: outShapes[0][2],
        channelMajor: true,
        inputWidth: inW,
        inputHeight: inH,
        r: lb.scale,
        dw: dw,
        dh: dh,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        confThres: confThres,
        scoresAreProbabilities: true,
        iouThres: iouThres,
        maxDet: maxDet,
        filterClassId: personOnly ? cocoPersonClassId : null,
      );
    }

    cachedOutputs ??= createOutputBuffers(outShapes);
    zeroOutputBuffers(cachedOutputs!, outShapes);
    interpreter!.runForMultipleInputs([inputFlat.buffer], cachedOutputs!);

    if (timingInferUs != null) {
      sw.stop();
      timingInferUs(sw.elapsedMicroseconds);
    }

    return postProcessDetections(
      outputs: cachedOutputs!.values.toList(),
      inputWidth: inW,
      inputHeight: inH,
      r: lb.scale,
      dw: dw,
      dh: dh,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: confThres,
      scoresAreProbabilities: true,
      iouThres: iouThres,
      topkPreNms: 0,
      maxDet: maxDet,
      filterClassId: personOnly ? cocoPersonClassId : null,
    );
  }
}
