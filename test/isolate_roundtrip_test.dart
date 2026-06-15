import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:pose_detection/pose_detection.dart';

/// End-to-end test of the pose background-isolate RPC path now running on the
/// shared `serveIsolateRpc` server. The critical case is the decode-failure
/// protocol: an undecodable image must still surface as a [FormatException] on
/// the main side (preserved via `IsolateRpcExactError` + the
/// `_decodeFailureErrorPrefix` `startsWith` contract).
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  test('pose isolate round-trip: decode failure -> FormatException; valid '
      'detects; graceful dispose', () async {
    final detector = PoseDetector();
    await detector.initializeFromBuffers(
      yoloBytes: load('$root/assets/models/yolov8n_float32.tflite'),
      landmarkBytes: load('$root/assets/models/pose_landmark_lite.tflite'),
      landmarkModel: PoseLandmarkModel.lite,
      useCompiledModel: false,
    );
    expect(detector.isReady, isTrue);

    // Decode-failure path through serveIsolateRpc + IsolateRpcExactError.
    await expectLater(
      detector.detect(Uint8List.fromList([1, 2, 3, 4])),
      throwsA(isA<FormatException>()),
    );

    // Success path through the new 'detect' handler over the real isolate.
    final poses = await detector.detect(
      load('$root/example_web/assets/samples/pose1.jpg'),
    );
    expect(poses, isA<List<Pose>>());

    // Graceful dispose -> serveIsolateRpc dispose ack.
    await detector.dispose();
    expect(detector.isReady, isFalse);
  });
}
