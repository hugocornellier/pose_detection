import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter_custom/tflite_flutter.dart';
import 'image_utils.dart';


class YoloDetection {
  final int cls;
  final double score;
  final List<double> bboxXYXY;

  YoloDetection({
    required this.cls,
    required this.score,
    required this.bboxXYXY,
  });
}

class YoloV8PersonDetector {
  IsolateInterpreter? _iso;
  Interpreter? _interpreter;
  bool _isInitialized = false;
  late int _inW;
  late int _inH;
  static const int cocoPersonClassId = 0;
  final _outShapes = <List<int>>[];
  img.Image? _canvasBuffer;

  Future<void> initialize() async {
    const String assetPath = 'packages/pose_detection_tflite/assets/models/yolov8n_float32.tflite';
    if (_isInitialized) await dispose();
    final Interpreter itp = await Interpreter.fromAsset(assetPath, options: InterpreterOptions());
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

  bool get isInitialized => _isInitialized;

  Future<void> dispose() async {
    _iso?.close();
    _iso = null;
    _interpreter?.close();
    _interpreter = null;
    _canvasBuffer = null;
    _isInitialized = false;
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

  static List<List<List<List<double>>>> _asNHWC4D(
      Float32List flat,
      int h,
      int w,
      ) {
    final out = List<List<List<List<double>>>>.filled(
      1,
      List.generate(
        h,
            (_) => List.generate(
          w,
              (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );

    int k = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final List<double> px = out[0][y][x];
        px[0] = flat[k++];
        px[1] = flat[k++];
        px[2] = flat[k++];
      }
    }
    return out;
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

  Future<List<YoloDetection>> detectOnImage(
      img.Image image, {
        double confThres = 0.35,
        double iouThres = 0.4,
        int topkPreNms = 100,
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
    final Float32List flatInput = Float32List(inputSize);
    int k = 0;
    for (int y = 0; y < _inH; y++) {
      for (int x = 0; x < _inW; x++) {
        final img.Pixel px = letter.getPixel(x, y);
        flatInput[k++] = px.r / 255.0;
        flatInput[k++] = px.g / 255.0;
        flatInput[k++] = px.b / 255.0;
      }
    }

    final List<List<List<List<double>>>> input4d =
    _asNHWC4D(flatInput, _inH, _inW);

    final int inputCount = _interpreter!.getInputTensors().length;
    final List<Object> inputs = List<Object>.filled(
      inputCount,
      input4d,
      growable: false,
    );
    inputs[0] = input4d;

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
        buf = List.filled(
          shape.reduce((a, b) => a * b),
          0.0,
          growable: false,
        );
      }
      outputs[i] = buf;
    }

    if (_iso != null) {
      await _iso!.runForMultipleInputs(inputs, outputs);
    } else {
      _interpreter!.runForMultipleInputs(inputs, outputs);
    }

    final List<Map<String, dynamic>> decoded =
    _decodeAnyYoloOutputs(outputs.values.toList());
    final List<double> scores = <double>[];
    final List<int> clsIds = <int>[];
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

    if (topkPreNms > 0 && keptScore.length > topkPreNms) {
      final List<int> ord = _argSortDesc(keptScore).take(topkPreNms).toList();
      final List<List<double>> _boxes = <List<double>>[];
      final List<double> _scores = <double>[];
      final List<int> _cls = <int>[];
      for (final int i in ord) {
        _boxes.add(boxes[i]);
        _scores.add(keptScore[i]);
        _cls.add(keptCls[i]);
      }
      boxes
        ..clear()
        ..addAll(_boxes);
      keptScore
        ..clear()
        ..addAll(_scores);
      keptCls
        ..clear()
        ..addAll(_cls);
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
