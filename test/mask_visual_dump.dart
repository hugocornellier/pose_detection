// Manual visual + statistical harness for the segmentation mask feature.
// Not a CI test (no `_test.dart` suffix). Run explicitly:
//   flutter test test/mask_visual_dump.dart
//
// Reads people photos from $WORK/src, runs detection with enableSegmentation,
// and for each detected person writes:
//   - out/mask_<img>_<i>.png     raw 256x256 model mask (grayscale)
//   - out/overlay_<img>.png      original + green mask tint + yellow bbox
// plus out/stats.json and greppable MASKSTAT log lines. The key correctness
// metrics are bimodality (is the mask a crisp silhouette or a washed-out band,
// which would indicate a double-sigmoid like the YOLO bug) and lmOnMask (what
// fraction of visible landmarks land on mask>0.5 pixels = spatial alignment).
// ignore_for_file: avoid_print
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';

const String work =
    '/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-pose-detection/f02f6ab0-f311-4751-9b12-a2052e8aea45/scratchpad/maskwork';
const String src = '$work/src';
const String out = '$work/out';

class MaskStats {
  int min = 255, max = 0;
  double mean = 0;
  double pLow = 0, pHigh = 0, pMid = 0;
  List<int> hist = List<int>.filled(8, 0);
  int lmTotal = 0, lmOnMask = 0;
  double centroidDistNorm = -1;
}

MaskStats computeStats(SegmentationMask m, Pose p) {
  final s = MaskStats();
  final Uint8List c = m.confidences;
  final int n = c.length;
  double sum = 0;
  int nLow = 0, nHigh = 0, nMid = 0;
  double cxAcc = 0, cyAcc = 0;
  int cCount = 0;
  for (int i = 0; i < n; i++) {
    final int v = c[i];
    if (v < s.min) s.min = v;
    if (v > s.max) s.max = v;
    sum += v;
    if (v < 32) nLow++;
    if (v > 224) nHigh++;
    if (v >= 64 && v <= 192) nMid++;
    s.hist[v >> 5]++;
    if (v > 128) {
      final int row = i ~/ m.width;
      final int col = i % m.width;
      cxAcc += m.imageLeft + (col + 0.5) / m.width * m.imageWidth;
      cyAcc += m.imageTop + (row + 0.5) / m.height * m.imageHeight;
      cCount++;
    }
  }
  s.mean = sum / n;
  s.pLow = nLow / n;
  s.pHigh = nHigh / n;
  s.pMid = nMid / n;

  // Alignment: fraction of visible landmarks that sit on person pixels.
  double lxAcc = 0, lyAcc = 0;
  int lCount = 0;
  for (final lm in p.landmarks) {
    if (lm.visibility > 0.5) {
      s.lmTotal++;
      if (m.confidenceAt(lm.x, lm.y) > 0.5) s.lmOnMask++;
      lxAcc += lm.x;
      lyAcc += lm.y;
      lCount++;
    }
  }
  if (cCount > 0 && lCount > 0) {
    final double mcx = cxAcc / cCount, mcy = cyAcc / cCount;
    final double lcx = lxAcc / lCount, lcy = lyAcc / lCount;
    final double diag = _dist(
      0,
      0,
      p.imageWidth.toDouble(),
      p.imageHeight.toDouble(),
    );
    s.centroidDistNorm = _dist(mcx, mcy, lcx, lcy) / diag;
  }
  return s;
}

double _dist(double ax, double ay, double bx, double by) {
  final dx = ax - bx, dy = ay - by;
  return math.sqrt(dx * dx + dy * dy);
}

void tintAndBox(
  Uint8List ob,
  int W,
  int H,
  SegmentationMask m,
  cv.Mat resized,
) {
  final Uint8List rb = resized.data;
  final int side = resized.cols;
  final int left = m.imageLeft.round(), top = m.imageTop.round();
  for (int dy = 0; dy < side; dy++) {
    final int iy = top + dy;
    if (iy < 0 || iy >= H) continue;
    for (int dx = 0; dx < side; dx++) {
      final int ix = left + dx;
      if (ix < 0 || ix >= W) continue;
      if (rb[dy * side + dx] > 128) {
        final int o = (iy * W + ix) * 3;
        ob[o] = (ob[o] * 0.45).round(); // B
        ob[o + 1] = (ob[o + 1] * 0.45 + 255 * 0.55).round(); // G
        ob[o + 2] = (ob[o + 2] * 0.45).round(); // R
      }
    }
  }
}

void drawBox(Uint8List ob, int W, int H, BoundingBox b) {
  final int l = b.left.round().clamp(0, W - 1);
  final int r = b.right.round().clamp(0, W - 1);
  final int t = b.top.round().clamp(0, H - 1);
  final int bo = b.bottom.round().clamp(0, H - 1);
  void px(int x, int y) {
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    final int o = (y * W + x) * 3;
    ob[o] = 0;
    ob[o + 1] = 255;
    ob[o + 2] = 255; // yellow BGR
  }

  for (int x = l; x <= r; x++) {
    for (int w = 0; w < 2; w++) {
      px(x, t + w);
      px(x, bo - w);
    }
  }
  for (int y = t; y <= bo; y++) {
    for (int w = 0; w < 2; w++) {
      px(l + w, y);
      px(r - w, y);
    }
  }
}

Future<List<Map<String, dynamic>>> variantSweep(
  String root,
  String imgPath,
) async {
  final results = <Map<String, dynamic>>[];
  for (final model in PoseLandmarkModel.values) {
    final d = PoseDetector();
    try {
      await d.initializeFromBuffers(
        yoloBytes: File(
          '$root/assets/models/yolov8n_float32.tflite',
        ).readAsBytesSync(),
        landmarkBytes: File(
          '$root/assets/models/pose_landmark_${model.name}.tflite',
        ).readAsBytesSync(),
        landmarkModel: model,
        enableSegmentation: true,
      );
      final poses = await d.detect(File(imgPath).readAsBytesSync());
      final withMask = poses.where((p) => p.segmentationMask != null).toList();
      if (withMask.isNotEmpty) {
        final s = computeStats(
          withMask.first.segmentationMask!,
          withMask.first,
        );
        results.add({
          'model': model.name,
          'pLow': s.pLow,
          'pHigh': s.pHigh,
          'pMid': s.pMid,
          'lmOnMask': s.lmTotal == 0 ? 0.0 : s.lmOnMask / s.lmTotal,
          'min': s.min,
          'max': s.max,
        });
      }
    } catch (e) {
      results.add({'model': model.name, 'error': '$e'});
    } finally {
      await d.dispose();
    }
  }
  return results;
}

Future<String> compiledPathCheck(String root, String imgPath) async {
  final d = PoseDetector();
  try {
    await d.initializeFromBuffers(
      yoloBytes: File(
        '$root/assets/models/yolov8n_float32.tflite',
      ).readAsBytesSync(),
      landmarkBytes: File(
        '$root/assets/models/pose_landmark_full.tflite',
      ).readAsBytesSync(),
      landmarkModel: PoseLandmarkModel.full,
      enableSegmentation: true,
      useCompiledModel: true,
    );
    final poses = await d.detect(File(imgPath).readAsBytesSync());
    final n = poses.where((p) => p.segmentationMask != null).length;
    return 'ok: $n/${poses.length} poses carried a mask via CompiledModel path';
  } catch (e) {
    return 'not exercised on host: $e';
  } finally {
    await d.dispose();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;

  test('mask visual + stats dump', () async {
    Directory(out).createSync(recursive: true);
    final images = <String, String>{
      'dancer': '$src/dancer.jpg',
      'yoga': '$src/yoga.jpg',
      'sprinter': '$src/sprinter.jpg',
      'walk': '$src/walk.jpg',
      'basketball': '$src/basketball.jpg',
      'group': '$src/group.jpg',
    };

    final detector = PoseDetector();
    await detector.initializeFromBuffers(
      yoloBytes: File(
        '$root/assets/models/yolov8n_float32.tflite',
      ).readAsBytesSync(),
      landmarkBytes: File(
        '$root/assets/models/pose_landmark_heavy.tflite',
      ).readAsBytesSync(),
      landmarkModel: PoseLandmarkModel.heavy,
      enableSegmentation: true,
      maxDetections: 12,
      detectorConf: 0.2, // lower gate so clean single-subject shots register
    );

    final report = <String, dynamic>{'model': 'heavy', 'images': <dynamic>[]};

    for (final e in images.entries) {
      final f = File(e.value);
      if (!f.existsSync()) {
        print('SKIP ${e.key}: missing');
        continue;
      }
      final bytes = f.readAsBytesSync();
      final poses = await detector.detect(bytes);

      final decoded = cv.imdecode(bytes, cv.IMREAD_COLOR);
      final cv.Mat orig = decoded.isContinuous ? decoded : decoded.clone();
      final int W = orig.cols, H = orig.rows;
      final Uint8List ob = orig.data;

      // Pristine "before" image (same dimensions as the masked "after") for the
      // README before/after pair.
      if (e.key == 'dancer' || e.key == 'group') {
        cv.imwrite('$out/before_${e.key}.jpg', orig);
      }

      final posesOut = <Map<String, dynamic>>[];
      int pi = 0;
      int maskCount = 0;
      for (final p in poses) {
        final m = p.segmentationMask;
        final entry = <String, dynamic>{
          'i': pi,
          'score': p.score,
          'hasMask': m != null,
        };
        if (m != null) {
          maskCount++;
          final s = computeStats(m, p);

          final cv.Mat maskMat = cv.Mat.create(
            rows: m.height,
            cols: m.width,
            type: cv.MatType.CV_8UC1,
          );
          maskMat.data.setAll(0, m.confidences);
          final String maskPath = '$out/mask_${e.key}_$pi.png';
          cv.imwrite(maskPath, maskMat);

          final int side = m.imageWidth.round();
          if (side > 0) {
            final cv.Mat resized = cv.resize(maskMat, (side, side));
            tintAndBox(ob, W, H, m, resized);
            resized.dispose();
          }
          maskMat.dispose();

          entry.addAll({
            'maskPng': 'mask_${e.key}_$pi.png',
            'min': s.min,
            'max': s.max,
            'mean': double.parse(s.mean.toStringAsFixed(1)),
            'pLow': double.parse(s.pLow.toStringAsFixed(3)),
            'pHigh': double.parse(s.pHigh.toStringAsFixed(3)),
            'pMid': double.parse(s.pMid.toStringAsFixed(3)),
            'hist': s.hist,
            'lmOnMask': s.lmOnMask,
            'lmTotal': s.lmTotal,
            'centroidDistNorm': double.parse(
              s.centroidDistNorm.toStringAsFixed(3),
            ),
            'maskRect': {
              'left': m.imageLeft,
              'top': m.imageTop,
              'w': m.imageWidth,
              'h': m.imageHeight,
            },
          });
          print(
            'MASKSTAT img=${e.key} pose=$pi score=${p.score.toStringAsFixed(2)} '
            'side=$side min=${s.min} max=${s.max} mean=${s.mean.toStringAsFixed(1)} '
            'pLow=${s.pLow.toStringAsFixed(3)} pHigh=${s.pHigh.toStringAsFixed(3)} '
            'pMid=${s.pMid.toStringAsFixed(3)} lmOnMask=${s.lmOnMask}/${s.lmTotal} '
            'centroidDistNorm=${s.centroidDistNorm.toStringAsFixed(3)}',
          );
        }
        posesOut.add(entry);
        pi++;
      }

      // Clean README demo: mask tint only, no bounding boxes.
      if (e.key == 'dancer' || e.key == 'group') {
        cv.imwrite('$out/demo_${e.key}.jpg', orig);
      }

      for (final p in poses) {
        drawBox(ob, W, H, p.boundingBox);
      }
      final String overlayPath = '$out/overlay_${e.key}.png';
      cv.imwrite(overlayPath, orig);
      orig.dispose();
      if (!identical(orig, decoded)) decoded.dispose();

      print('IMG ${e.key} ${W}x$H poses=${poses.length} withMask=$maskCount');
      (report['images'] as List).add({
        'img': e.key,
        'srcName': '${e.key}.jpg',
        'width': W,
        'height': H,
        'overlay': 'overlay_${e.key}.png',
        'poseCount': poses.length,
        'maskCount': maskCount,
        'poses': posesOut,
      });
    }

    await detector.dispose();

    report['variantSweepSprinter'] = await variantSweep(
      root,
      '$src/sprinter.jpg',
    );
    report['compiledPath'] = await compiledPathCheck(root, '$src/sprinter.jpg');

    File(
      '$out/stats.json',
    ).writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
    print('DUMP_DONE images=${(report['images'] as List).length} out=$out');
  }, timeout: const Timeout(Duration(minutes: 8)));
}
