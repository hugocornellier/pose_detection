// Headless driver: runs the real pose detector over sample_videos/dancing_10s.mp4
// exactly as VideoFileScreen does (boxesAndLandmarks + heavy + XNNPACK + the
// One-Euro PoseSmoother) and dumps per-frame boxes + skeleton segments + joints
// to JSON, for regenerating the README demo with custom overlay styling.
//
// Run on macOS desktop:
//   cd example && flutter test integration_test/gen_demo_dump_test.dart -d macos
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv.dart' as cv;
import 'package:pose_detection/pose_detection.dart';
import 'package:pose_detection_example/main.dart' show PoseSmoother;

const String kVideoPath =
    '/Users/hugocornellier/IdeaProjects/pose_detection/sample_videos/dancing_10s.mp4';
const String kOutDir =
    '/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-flutter-litert/c0042546-26a9-49af-a45c-ef2d760c1514/scratchpad/pose_dump';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('dump pose detections for demo regeneration', (tester) async {
    Directory(kOutDir).createSync(recursive: true);

    final detector = await PoseDetector.create(
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: PoseLandmarkModel.heavy,
      performanceConfig: const PerformanceConfig.xnnpack(),
    );

    final cap = cv.VideoCapture.fromFile(kVideoPath);
    expect(cap.isOpened, true, reason: 'could not open $kVideoPath');
    final double fps = cap.get(cv.CAP_PROP_FPS);
    final int vw = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final int vh = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();

    final smoother = PoseSmoother(enabled: true)..reset();

    final frames = <Map<String, dynamic>>[];
    cv.Mat? frame;
    int idx = 0;
    while (true) {
      final res = cap.read(m: frame);
      final ok = res.$1;
      frame = res.$2;
      if (!ok || frame.isEmpty) break;

      final List<Pose> raw = await detector.detectFromMat(frame);
      final double tSec = fps > 0 ? idx / fps : idx / 30.0;
      final List<Pose> poses = smoother.apply(raw, tSec);

      final posesJson = <Map<String, dynamic>>[];
      for (final p in poses) {
        final segs = <List<double>>[];
        for (final c in poseLandmarkConnections) {
          final a = p.getLandmark(c[0]);
          final b = p.getLandmark(c[1]);
          if (a != null &&
              b != null &&
              a.visibility > 0.5 &&
              b.visibility > 0.5) {
            segs.add([a.x, a.y, b.x, b.y]);
          }
        }
        final pts = <List<double>>[];
        for (final l in p.landmarks) {
          if (l.visibility > 0.5) pts.add([l.x, l.y]);
        }
        posesJson.add({
          'score': p.score,
          'box': [
            p.boundingBox.left,
            p.boundingBox.top,
            p.boundingBox.right,
            p.boundingBox.bottom,
          ],
          'segments': segs,
          'points': pts,
        });
      }
      frames.add({'i': idx, 'poses': posesJson});
      idx++;
    }

    frame.dispose();
    cap.release();
    await detector.dispose();

    final out = File('$kOutDir/dancing_poses.json');
    out.writeAsStringSync(
      jsonEncode({'fps': fps, 'w': vw, 'h': vh, 'frames': frames}),
    );
    // ignore: avoid_print
    print(
      'POSE_DUMP_DONE frames=${frames.length} size=${vw}x$vh -> ${out.path}',
    );
    expect(frames.length, greaterThan(0));
  });
}
