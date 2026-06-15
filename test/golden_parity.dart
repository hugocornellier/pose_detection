import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';

/// Manual golden harness (untracked; persists across `git checkout`). Runs the
/// full e2e detect on the deterministic Interpreter path and dumps every
/// detection (toMap) as JSON to $GOLDEN_OUT, for before/after result-parity.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('golden parity dump', () async {
    final root = Directory.current.path;
    Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());
    final detector = PoseDetector();
    await detector.initializeFromBuffers(
      yoloBytes: load('$root/assets/models/yolov8n_float32.tflite'),
      landmarkBytes: load('$root/assets/models/pose_landmark_lite.tflite'),
      landmarkModel: PoseLandmarkModel.lite,
      useCompiledModel: false,
    );
    final results = await detector.detect(
      load('$root/example_web/assets/samples/pose1.jpg'),
    );
    final out = Platform.environment['GOLDEN_OUT'] ?? '/tmp/golden_pose.json';
    File(
      out,
    ).writeAsStringSync(jsonEncode(results.map((p) => p.toMap()).toList()));
    stderr.writeln('[golden] pose: ${results.length} detections -> $out');
    await detector.dispose();
  });
}
