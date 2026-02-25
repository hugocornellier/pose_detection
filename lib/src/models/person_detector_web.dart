import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:web/web.dart' as web;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import '../types.dart';

/// A single object detection result from YOLOv8.
///
/// Contains the detected class ID, confidence score, and bounding box coordinates.
class YoloDetection {
  /// Detected class ID (0 = person in COCO dataset).
  final int cls;

  /// Confidence score for the detection (0.0 to 1.0).
  final double score;

  /// Bounding box in XYXY format [x1, y1, x2, y2] in pixel coordinates.
  final List<double> bboxXYXY;

  /// Creates a YOLO detection result.
  YoloDetection({
    required this.cls,
    required this.score,
    required this.bboxXYXY,
  });
}

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
  /// Returns a list of [YoloDetection] objects with bounding boxes in original
  /// image coordinates.
  Future<List<YoloDetection>> detect(
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

    return _postProcessDetections(
      outputs: _cachedOutputs!.values.toList(),
      r: r,
      dw: dw,
      dh: dh,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      confThres: confThres,
      iouThres: iouThres,
      topkPreNms: 0,
      maxDet: maxDet,
      personOnly: personOnly,
    );
  }

  // ---------------------------------------------------------------------------
  // Private helper methods (pure Dart math, no platform dependencies)
  // ---------------------------------------------------------------------------

  /// Creates pre-allocated output buffers based on cached output shapes.
  Map<int, Object> _createOutputBuffers() {
    final Map<int, Object> outputs = <int, Object>{};
    for (int i = 0; i < _outShapes.length; i++) {
      final List<int> shape = _outShapes[i];
      Object buf;
      if (shape.length == 3) {
        buf = List.generate(
          shape[0],
          (_) => List.generate(
            shape[1],
            (_) => List<double>.filled(shape[2], 0.0, growable: false),
            growable: false,
          ),
          growable: false,
        );
      } else if (shape.length == 2) {
        buf = List.generate(
          shape[0],
          (_) => List<double>.filled(shape[1], 0.0, growable: false),
          growable: false,
        );
      } else {
        buf = List<double>.filled(
          shape.reduce((a, b) => a * b),
          0.0,
          growable: false,
        );
      }
      outputs[i] = buf;
    }
    return outputs;
  }

  /// Zeros out pre-allocated output buffers for reuse.
  void _zeroOutputBuffers(Map<int, Object> outputs) {
    for (int i = 0; i < _outShapes.length; i++) {
      final List<int> shape = _outShapes[i];
      final Object buf = outputs[i]!;
      if (shape.length == 3) {
        final list3d = buf as List<List<List<double>>>;
        for (int j = 0; j < shape[0]; j++) {
          for (int k = 0; k < shape[1]; k++) {
            list3d[j][k].fillRange(0, shape[2], 0.0);
          }
        }
      } else if (shape.length == 2) {
        final list2d = buf as List<List<double>>;
        for (int j = 0; j < shape[0]; j++) {
          list2d[j].fillRange(0, shape[1], 0.0);
        }
      } else {
        final list1d = buf as List<double>;
        list1d.fillRange(0, list1d.length, 0.0);
      }
    }
  }

  static double _sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));

  static List<int> _argSortDesc(List<double> a) {
    final List<int> idx = List<int>.generate(a.length, (i) => i);
    idx.sort((i, j) => a[j].compareTo(a[i]));
    return idx;
  }

  static List<int> _nms(
    List<List<double>> boxes,
    List<double> scores, {
    double iouThres = 0.45,
    int maxDet = 100,
  }) {
    if (boxes.isEmpty) return <int>[];
    final List<int> order = _argSortDesc(scores);
    final List<int> keep = <int>[];

    double interArea(List<double> a, List<double> b) {
      final double xx1 = math.max(a[0], b[0]);
      final double yy1 = math.max(a[1], b[1]);
      final double xx2 = math.min(a[2], b[2]);
      final double yy2 = math.min(a[3], b[3]);
      final double w = math.max(0.0, xx2 - xx1);
      final double h = math.max(0.0, yy2 - yy1);
      return w * h;
    }

    double area(List<double> b) =>
        math.max(0.0, b[2] - b[0]) * math.max(0.0, b[3] - b[1]);

    final List<double> areas = boxes.map(area).toList();

    final List<bool> suppressed = List<bool>.filled(order.length, false);
    for (int m = 0; m < order.length; m++) {
      if (suppressed[m]) continue;
      final int i = order[m];
      keep.add(i);
      if (keep.length >= maxDet) break;
      for (int n = m + 1; n < order.length; n++) {
        if (suppressed[n]) continue;
        final int j = order[n];
        final double inter = interArea(boxes[i], boxes[j]);
        final double u = areas[i] + areas[j] - inter + 1e-7;
        final double iou = inter / u;
        if (iou > iouThres) suppressed[n] = true;
      }
    }
    return keep;
  }

  static List<List<double>> _transpose2D(List<List<double>> a) {
    if (a.isEmpty) return <List<double>>[];
    final int rows = a.length, cols = a[0].length;
    final List<List<double>> out = List.generate(
      cols,
      (_) => List<double>.filled(rows, 0.0),
    );
    for (int r = 0; r < rows; r++) {
      final List<double> row = a[r];
      for (int c = 0; c < cols; c++) {
        out[c][r] = row[c];
      }
    }
    return out;
  }

  static List<List<double>> _concat0(List<List<List<double>>> parts) {
    final List<List<double>> out = <List<double>>[];
    for (final List<List<double>> p in parts) {
      out.addAll(p);
    }
    return out;
  }

  static List<List<double>> _ensure2D(List<dynamic> raw) {
    return raw
        .map<List<double>>(
          (e) => (e as List).map((v) => (v as num).toDouble()).toList(),
        )
        .toList();
  }

  List<Map<String, dynamic>> _decodeAnyYoloOutputs(List<dynamic> outputs) {
    final List<List<List<double>>> parts = <List<List<double>>>[];
    for (final raw in outputs) {
      final List<dynamic> t3d = raw as List;
      if (t3d.length != 1) throw StateError('Unexpected YOLO output rank');

      final List<List<double>> out2d = _ensure2D(t3d[0]);
      if (out2d.isEmpty) continue;

      final int rows = out2d.length;
      final int cols = out2d[0].length;
      if (rows < cols && (rows == 84 || rows == 85)) {
        parts.add(_transpose2D(out2d));
      } else {
        parts.add(out2d);
      }
    }

    final List<List<double>> out = _concat0(parts);
    if (out.isEmpty || out[0].length < 84) {
      throw StateError('Expected channels >=84');
    }

    final int channels = out[0].length;
    return out
        .map(
          (row) => {
            'xywh': row.sublist(0, 4),
            'rest': row.sublist(4),
            'C': channels,
          },
        )
        .toList();
  }

  /// Exposes YOLO output decoding for tests.
  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) {
    return _decodeAnyYoloOutputs(outputs);
  }

  static List<double> _xywhToXyxy(List<double> xywh) {
    final double cx = xywh[0], cy = xywh[1], w = xywh[2], h = xywh[3];
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
  }

  static double _median(List<double> a) {
    if (a.isEmpty) return double.nan;

    final List<double> b = List<double>.from(a)..sort();
    final int n = b.length;
    if (n.isOdd) return b[n ~/ 2];

    return 0.5 * (b[n ~/ 2 - 1] + b[n ~/ 2]);
  }

  /// Shared post-processing for YOLO detection outputs.
  ///
  /// Decodes model outputs, applies confidence filtering, NMS, and coordinate
  /// transformation from letterbox space to original image coordinates.
  List<YoloDetection> _postProcessDetections({
    required List<dynamic> outputs,
    required double r,
    required int dw,
    required int dh,
    required int imageWidth,
    required int imageHeight,
    required double confThres,
    required double iouThres,
    required int topkPreNms,
    required int maxDet,
    required bool personOnly,
  }) {
    final List<Map<String, dynamic>> decoded = _decodeAnyYoloOutputs(outputs);
    final List<int> clsIds = <int>[];
    final List<double> scores = <double>[];
    final List<List<double>> xywhs = <List<double>>[];

    for (final Map<String, dynamic> row in decoded) {
      final int C = row['C'] as int;
      final List<double> xywh = (row['xywh'] as List)
          .map((v) => (v as num).toDouble())
          .toList();
      final List<double> rest = (row['rest'] as List)
          .map((v) => (v as num).toDouble())
          .toList();

      if (C == 84) {
        int argMax = 0;
        double best = -1e9;
        for (int i = 0; i < rest.length; i++) {
          final double s = _sigmoid(rest[i]);
          if (s > best) {
            best = s;
            argMax = i;
          }
        }
        scores.add(best);
        clsIds.add(argMax);
        xywhs.add(xywh);
      } else {
        final double obj = _sigmoid(rest[0]);
        final List<double> clsLogits = rest.sublist(1, 81);
        int argMax = 0;
        double best = -1e9;
        for (int i = 0; i < clsLogits.length; i++) {
          final double s = _sigmoid(clsLogits[i]);
          if (s > best) {
            best = s;
            argMax = i;
          }
        }
        scores.add(obj * best);
        clsIds.add(argMax);
        xywhs.add(xywh);
      }
    }

    final List<int> keep0 = <int>[];
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] >= confThres) keep0.add(i);
    }
    if (keep0.isEmpty) return <YoloDetection>[];

    final List<List<double>> keptXywh = [for (final int i in keep0) xywhs[i]];
    final List<int> keptCls = [for (final int i in keep0) clsIds[i]];
    final List<double> keptScore = [for (final int i in keep0) scores[i]];

    if (keptXywh.isNotEmpty &&
        _median([for (final v in keptXywh) v[2]]) <= 2.0) {
      for (final List<double> v in keptXywh) {
        v[0] *= _inW.toDouble();
        v[1] *= _inH.toDouble();
        v[2] *= _inW.toDouble();
        v[3] *= _inH.toDouble();
      }
    }

    final List<List<double>> boxesLtr = [
      for (final List<double> v in keptXywh) _xywhToXyxy(v),
    ];
    final List<List<double>> boxes = <List<double>>[];
    for (final List<double> b in boxesLtr) {
      boxes.add(ImageUtils.scaleFromLetterbox(b, r, dw, dh));
    }
    final double iw = imageWidth.toDouble();
    final double ih = imageHeight.toDouble();
    for (final List<double> b in boxes) {
      b[0] = b[0].clamp(0.0, iw);
      b[2] = b[2].clamp(0.0, iw);
      b[1] = b[1].clamp(0.0, ih);
      b[3] = b[3].clamp(0.0, ih);
    }

    final int effectiveTopk;
    if (topkPreNms > 0) {
      effectiveTopk = topkPreNms;
    } else {
      const int basePixels = 640 * 640;
      const int baseCandidates = 100;
      final int imagePixels = imageWidth * imageHeight;
      final double scale = imagePixels / basePixels;
      effectiveTopk = (baseCandidates * scale).round().clamp(20, 200);
    }

    if (effectiveTopk > 0 && keptScore.length > effectiveTopk) {
      final List<int> ord = _argSortDesc(
        keptScore,
      ).take(effectiveTopk).toList();
      final List<List<double>> sortedBoxes = <List<double>>[];
      final List<double> sortedScores = <double>[];
      final List<int> sortedCls = <int>[];
      for (final int i in ord) {
        sortedBoxes.add(boxes[i]);
        sortedScores.add(keptScore[i]);
        sortedCls.add(keptCls[i]);
      }
      boxes
        ..clear()
        ..addAll(sortedBoxes);
      keptScore
        ..clear()
        ..addAll(sortedScores);
      keptCls
        ..clear()
        ..addAll(sortedCls);
    }

    if (personOnly) {
      final List<List<double>> fBoxes = <List<double>>[];
      final List<double> fScores = <double>[];
      final List<int> fCls = <int>[];
      for (int i = 0; i < keptCls.length; i++) {
        if (keptCls[i] == cocoPersonClassId) {
          fBoxes.add(boxes[i]);
          fScores.add(keptScore[i]);
          fCls.add(keptCls[i]);
        }
      }
      boxes
        ..clear()
        ..addAll(fBoxes);
      keptScore
        ..clear()
        ..addAll(fScores);
      keptCls
        ..clear()
        ..addAll(fCls);
    }

    final List<int> keep = _nms(
      boxes,
      keptScore,
      iouThres: iouThres,
      maxDet: maxDet,
    );
    final List<YoloDetection> out = <YoloDetection>[];
    for (final int i in keep) {
      out.add(
        YoloDetection(cls: keptCls[i], score: keptScore[i], bboxXYXY: boxes[i]),
      );
    }
    return out;
  }
}
