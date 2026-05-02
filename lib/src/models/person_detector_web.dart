// ignore_for_file: implementation_imports

import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import '../util/web_image_utils.dart';
import 'person_detector_base.dart';

/// Web implementation of YOLOv8n person detector.
///
/// Uses Canvas API for image preprocessing instead of OpenCV.
/// On the LiteRT.js path, runs on WebGPU when available and falls back to
/// SIMD-WASM otherwise. The legacy tflite-js path is CPU/WASM only.
/// Uses [Interpreter] directly (no [IsolateInterpreter] on web).
class YoloV8PersonDetector extends PersonDetectorBase {
  web.HTMLCanvasElement? _canvasElement;
  web.CanvasRenderingContext2D? _canvasCtx;

  /// COCO dataset class ID for the "person" class.
  static const int cocoPersonClassId = 0;

  /// Optional LiteRT.js-backed interpreter; non-null when [initialize] was
  /// called with [useLiteRt: true].
  LiteRtInterpreter? _liteRtItp;

  /// The accelerator passed to [LiteRtInterpreter.fromBytes] (`'webgpu'` /
  /// `'wasm'`), or null if running on the legacy tflite-js path.
  String? _activeAccelerator;

  /// The accelerator that compiled this model (`'webgpu'` / `'wasm'`),
  /// or null if running on the legacy tflite-js path.
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  /// Reusable Float32List output buffer for the LiteRT.js path (avoids
  /// allocating a 705k-float buffer on every detect call).
  Float32List? _liteRtOutputFlat;

  /// Cached result of the "is the YOLO export's xywh in normalized space?"
  /// probe. Computed on the first inference call when the answer is
  /// available; afterwards the per-call ws.sort()+median check is skipped.
  /// null = not yet probed; true = normalized (rescale to pixel space);
  /// false = already in pixel space.
  static bool? _xywhIsNormalized;

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
      // 'auto' prefers WebGPU; flutter_litert falls back to WASM if compile
      // fails (e.g., no navigator.gpu, or unsupported ops).
      final String resolved = liteRtAccelerator == 'auto'
          ? 'webgpu'
          : liteRtAccelerator;
      final lrt = await LiteRtInterpreter.fromBytes(
        raw.buffer.asUint8List(),
        accelerator: resolved,
      );
      _liteRtItp = lrt;
      _activeAccelerator = resolved;

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
      return _postProcessFlat(
        flat: out,
        outShape: outShapes[0],
        inputWidth: inW,
        inputHeight: inH,
        r: lb.scale,
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
      iouThres: iouThres,
      topkPreNms: 0,
      maxDet: maxDet,
      filterClassId: personOnly ? cocoPersonClassId : null,
    );
  }

  /// Fast YOLOv8 postProcess that operates directly on the flat Float32List
  /// output, without materializing nested `List<List<double>>` structures.
  ///
  /// Expected `outShape` is `[1, C, A]` (e.g. `[1, 84, 8400]`): C channels
  /// (4 bbox + 80 class scores) over A anchors. Reads anchor `a`, channel `c`
  /// at index `c*A + a`.
  static List<Detection> _postProcessFlat({
    required Float32List flat,
    required List<int> outShape,
    required int inputWidth,
    required int inputHeight,
    required double r,
    required int dw,
    required int dh,
    required int imageWidth,
    required int imageHeight,
    required double confThres,
    required double iouThres,
    required int maxDet,
    int? filterClassId,
  }) {
    final int channels = outShape[1];
    final int anchors = outShape[2];
    final int numClasses = channels - 4;

    // Pre-size pessimistic candidate buffers (still bounded above by anchors).
    final List<double> candScores = <double>[];
    final List<int> candCls = <int>[];
    final List<List<double>> candXywh = <List<double>>[];

    // Match the generic path: argmax over all classes per anchor, score
    // threshold, and (later) filterClassId. Class filter is applied AFTER
    // top-K so we keep the same anchors regardless of `filterClassId`.
    for (int a = 0; a < anchors; a++) {
      double bestLogit = -1e30;
      int bestCls = 0;
      for (int c = 0; c < numClasses; c++) {
        final double v = flat[(4 + c) * anchors + a];
        if (v > bestLogit) {
          bestLogit = v;
          bestCls = c;
        }
      }
      // sigmoid is monotonic: argmax of logits == argmax of sigmoids.
      final double s = 1.0 / (1.0 + math.exp(-bestLogit));
      if (s < confThres) continue;
      candScores.add(s);
      candCls.add(bestCls);
      candXywh.add(<double>[
        flat[0 * anchors + a],
        flat[1 * anchors + a],
        flat[2 * anchors + a],
        flat[3 * anchors + a],
      ]);
    }
    if (candScores.isEmpty) return <Detection>[];

    // YOLOv8 ultralytics export usually has xywh in pixel space already.
    // The output convention is fixed at compile time, so we only run the
    // "median width <= 2.0" probe on the very first inference and cache.
    bool? norm = _xywhIsNormalized;
    if (norm == null && candXywh.length > 1) {
      final List<double> ws = <double>[for (final v in candXywh) v[2]];
      ws.sort();
      final double med = ws[ws.length ~/ 2];
      norm = med <= 2.0;
      _xywhIsNormalized = norm;
    }
    if (norm == true) {
      final double iw = inputWidth.toDouble();
      final double ih = inputHeight.toDouble();
      for (final v in candXywh) {
        v[0] *= iw;
        v[1] *= ih;
        v[2] *= iw;
        v[3] *= ih;
      }
    }

    // xywh -> xyxy in letterbox space, then map back to original image.
    final List<List<double>> boxes = <List<double>>[];
    final double iw = imageWidth.toDouble();
    final double ih = imageHeight.toDouble();
    for (final List<double> v in candXywh) {
      final double cx = v[0];
      final double cy = v[1];
      final double w = v[2];
      final double h = v[3];
      double x1 = cx - w * 0.5;
      double y1 = cy - h * 0.5;
      double x2 = cx + w * 0.5;
      double y2 = cy + h * 0.5;
      x1 = ((x1 - dw) / r).clamp(0.0, iw);
      y1 = ((y1 - dh) / r).clamp(0.0, ih);
      x2 = ((x2 - dw) / r).clamp(0.0, iw);
      y2 = ((y2 - dh) / r).clamp(0.0, ih);
      boxes.add(<double>[x1, y1, x2, y2]);
    }

    // Top-k pre-NMS.
    int effectiveTopk;
    const int basePixels = 640 * 640;
    const int baseCandidates = 100;
    final int imagePixels = imageWidth * imageHeight;
    final double scale = imagePixels / basePixels;
    effectiveTopk = (baseCandidates * scale).round().clamp(20, 200);
    if (boxes.length > effectiveTopk) {
      final List<int> ord = _argSortDescIndices(
        candScores,
      ).take(effectiveTopk).toList();
      final List<List<double>> sb = <List<double>>[];
      final List<double> ss = <double>[];
      final List<int> sc = <int>[];
      for (final int i in ord) {
        sb.add(boxes[i]);
        ss.add(candScores[i]);
        sc.add(candCls[i]);
      }
      boxes
        ..clear()
        ..addAll(sb);
      candScores
        ..clear()
        ..addAll(ss);
      candCls
        ..clear()
        ..addAll(sc);
    }

    // Class filter (matches the generic path's order: between top-K and NMS).
    if (filterClassId != null) {
      final List<List<double>> fb = <List<double>>[];
      final List<double> fs = <double>[];
      final List<int> fc = <int>[];
      for (int i = 0; i < candCls.length; i++) {
        if (candCls[i] == filterClassId) {
          fb.add(boxes[i]);
          fs.add(candScores[i]);
          fc.add(candCls[i]);
        }
      }
      boxes
        ..clear()
        ..addAll(fb);
      candScores
        ..clear()
        ..addAll(fs);
      candCls
        ..clear()
        ..addAll(fc);
    }

    final List<int> keep = nms(
      boxes,
      candScores,
      iouThres: iouThres,
      maxDet: maxDet,
    );
    final List<Detection> out = <Detection>[];
    for (final int i in keep) {
      out.add(
        Detection(cls: candCls[i], score: candScores[i], bboxXYXY: boxes[i]),
      );
    }
    return out;
  }

  static List<int> _argSortDescIndices(List<double> a) {
    final List<int> ix = List<int>.generate(a.length, (i) => i);
    ix.sort((i, j) => a[j].compareTo(a[i]));
    return ix;
  }
}
