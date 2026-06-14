// ignore_for_file: avoid_print

// Correctness gate for the CompiledModel path.
//
// For every landmark variant and every sample image it checks that the
// CompiledModel pipeline (a) produces structurally valid poses (finite,
// in-bounds landmarks) and (b) agrees with the interpreter pipeline on pose
// count and landmark positions within a lenient tolerance. The count check is
// the important one for the YOLO person detector, where GPU floating-point
// variance could otherwise change the number of detections.
//
//   flutter test integration_test/pose_compiledmodel_correctness_test.dart -d macos

import 'dart:io' show Platform;
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:pose_detection/pose_detection.dart';

const List<String> sampleImages = [
  'assets/samples/pose1.jpg',
  'assets/samples/pose2.jpg',
  'assets/samples/pose3.jpg',
  'assets/samples/pose4.jpg',
  'assets/samples/pose5.jpg',
  'assets/samples/pose6.jpg',
  'assets/samples/pose7.jpg',
];

// Mean per-landmark distance (normalized by image diagonal) allowed between the
// interpreter and CompiledModel paths. Generous, to absorb GPU fp differences.
const double kMaxMeanNormalizedDistance = 0.05;

double _meanNormalizedLandmarkDistance(Pose a, Pose b) {
  if (a.landmarks.isEmpty || b.landmarks.isEmpty) return 0.0;
  final int n = min(a.landmarks.length, b.landmarks.length);
  if (n == 0) return 0.0;
  final double diag = sqrt(
    (a.imageWidth * a.imageWidth + a.imageHeight * a.imageHeight).toDouble(),
  );
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    final dx = a.landmarks[i].x - b.landmarks[i].x;
    final dy = a.landmarks[i].y - b.landmarks[i].y;
    sum += sqrt(dx * dx + dy * dy);
  }
  return (sum / n) / diag;
}

List<Pose> _sortedByBox(List<Pose> poses) {
  final copy = List<Pose>.from(poses);
  copy.sort((p, q) => p.boundingBox.left.compareTo(q.boundingBox.left));
  return copy;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final isMacOS = Platform.isMacOS;

  const models = <String, PoseLandmarkModel>{
    'lite': PoseLandmarkModel.lite,
    'full': PoseLandmarkModel.full,
    'heavy': PoseLandmarkModel.heavy,
  };

  group('PoseDetector - CompiledModel correctness', () {
    models.forEach((name, model) {
      test(
        '$name : compiled is valid and matches interpreter',
        timeout: const Timeout(Duration(minutes: 15)),
        skip: isMacOS ? false : 'macOS-only (real Metal GPU)',
        () async {
          final interpD = PoseDetector();
          await interpD.initialize(
            mode: PoseMode.boxesAndLandmarks,
            landmarkModel: model,
            performanceConfig: const PerformanceConfig.xnnpack(),
            useCompiledModel: false,
          );
          final compiledD = PoseDetector();
          await compiledD.initialize(
            mode: PoseMode.boxesAndLandmarks,
            landmarkModel: model,
            useCompiledModel: true,
          );

          try {
            for (final p in sampleImages) {
              final d = await rootBundle.load(p);
              final mat = cv.imdecode(d.buffer.asUint8List(), cv.IMREAD_COLOR);
              final List<Pose> ip = await interpD.detectFromMat(mat);
              final List<Pose> cp = await compiledD.detectFromMat(mat);
              mat.dispose();

              // (a) structural validity of the compiled output
              for (final pose in cp) {
                for (final lm in pose.landmarks) {
                  expect(
                    lm.x.isFinite && lm.y.isFinite,
                    isTrue,
                    reason: '$name/$p produced a non-finite landmark',
                  );
                  expect(
                    lm.x >= -1 && lm.x <= pose.imageWidth + 1,
                    isTrue,
                    reason: '$name/$p landmark x out of bounds: ${lm.x}',
                  );
                  expect(
                    lm.y >= -1 && lm.y <= pose.imageHeight + 1,
                    isTrue,
                    reason: '$name/$p landmark y out of bounds: ${lm.y}',
                  );
                }
              }

              // (b) agreement with the interpreter path
              expect(
                cp.length,
                ip.length,
                reason:
                    '$name/$p pose count differs (interpreter=${ip.length}, '
                    'compiled=${cp.length}). If this fails on the YOLO stage, '
                    'pin the YOLO CompiledModel to CPU on this platform too.',
              );

              if (cp.length == ip.length) {
                final si = _sortedByBox(ip);
                final sc = _sortedByBox(cp);
                for (int i = 0; i < si.length; i++) {
                  final dist = _meanNormalizedLandmarkDistance(si[i], sc[i]);
                  expect(
                    dist,
                    lessThan(kMaxMeanNormalizedDistance),
                    reason:
                        '$name/$p pose #$i landmark drift too large: '
                        '${dist.toStringAsFixed(4)}',
                  );
                }
              }
            }
          } finally {
            await interpD.dispose();
            await compiledD.dispose();
          }
        },
      );
    });
  });
}
