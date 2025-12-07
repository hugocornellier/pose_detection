import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';
import 'types.dart';

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
  img.Image? _canvasBuffer;
  Float32List? _inputBuffer;
  Delegate? _delegate; // Store delegate reference for cleanup
  Map<int, Object>? _cachedOutputs; // Pre-allocated output buffers

  /// Initializes the YOLOv8 person detector by loading the model.
  ///
  /// Parameters:
  /// - [performanceConfig]: Optional performance configuration for TFLite delegates.
  ///   Defaults to no delegates (backward compatible). Use [PerformanceConfig.xnnpack()]
  ///   for CPU optimization.
  ///
  /// Loads the YOLOv8n model from assets, allocates tensors, and creates
  /// an IsolateInterpreter for async inference. Must be called before [detectOnImage].
  ///
  /// If already initialized, this will dispose the previous instance first.
  ///
  /// Throws an exception if the model fails to load.
  Future<void> initialize({
    PerformanceConfig? performanceConfig,
  }) async {
    const String assetPath =
        'packages/pose_detection_tflite/assets/models/yolov8n_float32.tflite';
    if (_isInitialized) await dispose();

    // Create interpreter options with optional XNNPACK delegate
    final InterpreterOptions options =
        _createInterpreterOptions(performanceConfig);

    final Interpreter itp = await Interpreter.fromAsset(assetPath, options: options);
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

    _iso = await IsolateInterpreter.create(address: itp.address);

    _isInitialized = true;
  }

  /// Creates interpreter options with delegates based on performance configuration.
  InterpreterOptions _createInterpreterOptions(PerformanceConfig? config) {
    final options = InterpreterOptions();

    // Clean up any existing delegate before creating a new one
    _delegate?.delete();
    _delegate = null;

    // If no config or disabled mode, return default options (backward compatible)
    if (config == null || config.mode == PerformanceMode.disabled) {
      return options;
    }

    // Get effective thread count
    final threadCount = config.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);

    // Set CPU threads
    options.threads = threadCount;

    // Add XNNPACK delegate (for xnnpack or auto mode)
    if (config.mode == PerformanceMode.xnnpack ||
        config.mode == PerformanceMode.auto) {
      try {
        final xnnpackDelegate = XNNPackDelegate(
          options: XNNPackDelegateOptions(numThreads: threadCount),
        );
        options.addDelegate(xnnpackDelegate);
        _delegate = xnnpackDelegate; // Store for cleanup
      } catch (e) {
        // Graceful fallback: if delegate creation fails, continue with CPU
        // ignore: avoid_print
        print('[YOLO] Warning: Failed to create XNNPACK delegate: $e');
        // ignore: avoid_print
        print('[YOLO] Falling back to default CPU execution');
      }
    }

    return options;
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
    _canvasBuffer = null;
    _cachedOutputs = null;
    _isInitialized = false;
  }

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

  /// Exposes YOLO output decoding for tests.
  @visibleForTesting
  List<Map<String, dynamic>> decodeOutputsForTest(List<dynamic> outputs) {
    return _decodeAnyYoloOutputs(outputs);
  }

  /// Configures internal state for unit tests without loading native assets.
  @visibleForTesting
  void debugConfigureForTest({
    required int inputWidth,
    required int inputHeight,
    required List<List<int>> outputShapes,
    Interpreter? interpreter,
    IsolateInterpreter? isolate,
    img.Image? canvasBuffer,
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
    _canvasBuffer = canvasBuffer;
    _inputBuffer = inputBuffer;
    _isInitialized = initialized;
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

  /// Detects persons in the given image using YOLOv8.
  ///
  /// Performs the following steps:
  /// 1. Letterbox preprocessing to 640x640
  /// 2. Model inference via IsolateInterpreter
  /// 3. Post-processing: confidence filtering, NMS, coordinate transformation
  ///
  /// Parameters:
  /// - [image]: Input image to detect persons in
  /// - [confThres]: Confidence threshold for detections (default: 0.35)
  /// - [iouThres]: IoU threshold for Non-Maximum Suppression (default: 0.4)
  /// - [topkPreNms]: Maximum detections to keep before NMS. If <= 0, uses dynamic
  ///   scaling based on image size (default: 0 for dynamic)
  /// - [maxDet]: Maximum detections to return after NMS (default: 10)
  /// - [personOnly]: If true, only returns person class (class 0) detections (default: true)
  ///
  /// Returns a list of [YoloDetection] objects with bounding boxes in original image coordinates.
  ///
  /// Throws [StateError] if the detector is not initialized.
  Future<List<YoloDetection>> detectOnImage(
    img.Image image, {
    double confThres = 0.35,
    double iouThres = 0.4,
    int topkPreNms = 0,
    int maxDet = 10,
    bool personOnly = true,
  }) async {
    if (!_isInitialized || _interpreter == null) {
      throw StateError('YoloV8PersonDetector not initialized.');
    }

    final List<double> ratio = <double>[];
    final List<int> dwdh = <int>[];
    _canvasBuffer ??= img.Image(width: _inW, height: _inH);
    final img.Image letter = ImageUtils.letterbox(
      image,
      _inW,
      _inH,
      ratio,
      dwdh,
      reuseCanvas: _canvasBuffer,
    );
    final double r = ratio.first;
    final int dw = dwdh[0], dh = dwdh[1];

    final int inputSize = _inH * _inW * 3;
    _inputBuffer ??= Float32List(inputSize);
    if (_inputBuffer!.length != inputSize) {
      _inputBuffer = Float32List(inputSize);
    }
    final Float32List flatInput = _inputBuffer!;

    // Direct buffer access is ~10-50x faster than getPixel() which creates
    // a new Pixel object and performs bounds checking on every call.
    final bytes = letter.buffer.asUint8List();
    final int numChannels = letter.numChannels;
    const double scale = 1.0 / 255.0;
    int k = 0;
    int byteIndex = 0;
    for (int y = 0; y < _inH; y++) {
      for (int x = 0; x < _inW; x++) {
        flatInput[k++] = bytes[byteIndex] * scale;
        flatInput[k++] = bytes[byteIndex + 1] * scale;
        flatInput[k++] = bytes[byteIndex + 2] * scale;
        byteIndex += numChannels;
      }
    }

    final int inputCount = _interpreter!.getInputTensors().length;
    final List<Object> inputs = List<Object>.filled(
      inputCount,
      // Pass the raw buffer so tflite_flutter_custom does not try to
      // auto-resize the input tensor to a 1D shape.
      flatInput.buffer,
      growable: false,
    );

    // Lazy-initialize and reuse output buffers to reduce GC pressure
    _cachedOutputs ??= _createOutputBuffers();
    _zeroOutputBuffers(_cachedOutputs!);

    if (_iso != null) {
      await _iso!.runForMultipleInputs(inputs, _cachedOutputs!);
    } else {
      _interpreter!.runForMultipleInputs(inputs, _cachedOutputs!);
    }

    final List<Map<String, dynamic>> decoded =
        _decodeAnyYoloOutputs(_cachedOutputs!.values.toList());
    final List<int> clsIds = <int>[];
    final List<double> scores = <double>[];
    final List<List<double>> xywhs = <List<double>>[];

    for (final Map<String, dynamic> row in decoded) {
      final int C = row['C'] as int;
      final List<double> xywh =
          (row['xywh'] as List).map((v) => (v as num).toDouble()).toList();
      final List<double> rest =
          (row['rest'] as List).map((v) => (v as num).toDouble()).toList();

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
      for (final List<double> v in keptXywh) _xywhToXyxy(v)
    ];
    final List<List<double>> boxes = <List<double>>[];
    for (final List<double> b in boxesLtr) {
      boxes.add(ImageUtils.scaleFromLetterbox(b, r, dw, dh));
    }
    final double iw = image.width.toDouble();
    final double ih = image.height.toDouble();
    for (final List<double> b in boxes) {
      b[0] = b[0].clamp(0.0, iw);
      b[2] = b[2].clamp(0.0, iw);
      b[1] = b[1].clamp(0.0, ih);
      b[3] = b[3].clamp(0.0, ih);
    }

    // Dynamic topkPreNms: scale based on image area relative to 640x640 baseline
    // Smaller images need fewer candidates, larger images may need more
    final int effectiveTopk;
    if (topkPreNms > 0) {
      effectiveTopk = topkPreNms;
    } else {
      // Scale: 100 candidates for 640x640 (409600 pixels)
      // Minimum 20, maximum 200
      const int basePixels = 640 * 640;
      const int baseCandidates = 100;
      final int imagePixels = image.width * image.height;
      final double scale = imagePixels / basePixels;
      effectiveTopk = (baseCandidates * scale).round().clamp(20, 200);
    }

    if (effectiveTopk > 0 && keptScore.length > effectiveTopk) {
      final List<int> ord = _argSortDesc(keptScore).take(effectiveTopk).toList();
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

    final List<int> keep =
        _nms(boxes, keptScore, iouThres: iouThres, maxDet: maxDet);
    final List<YoloDetection> out = <YoloDetection>[];
    for (final int i in keep) {
      out.add(
        YoloDetection(
          cls: keptCls[i],
          score: keptScore[i],
          bboxXYXY: boxes[i],
        ),
      );
    }
    return out;
  }
}
