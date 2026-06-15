import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart' show PerformanceConfig;
import 'package:pose_detection/src/isolate/pose_detector_core.dart';
import 'package:pose_detection/pose_detection.dart'
    show PoseMode, PoseLandmarkModel;

/// Manual e2e benchmark (run explicitly): full pose detect latency in-process.
/// Toggle the conversion by swapping NativeImageUtils.matToRgbFloat32Simd's body
/// (SIMD vs pure-Dart) and re-run for an A/B.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  test('pose e2e detect latency', () async {
    final core = PoseDetectorCore();
    await core.initializeFromBuffers(
      yoloBytes: load('$root/assets/models/yolov8n_float32.tflite'),
      landmarkBytes: load('$root/assets/models/pose_landmark_lite.tflite'),
      mode: PoseMode.boxesAndLandmarks,
      landmarkModel: PoseLandmarkModel.lite,
      detectorConf: 0.5,
      detectorIou: 0.45,
      maxDetections: 10,
      minLandmarkScore: 0.5,
      interpreterPoolSize: 1,
      performanceConfig: const PerformanceConfig(),
      useCompiledModel: false,
    );
    final cv.Mat img = cv.imdecode(
      load('$root/example_web/assets/samples/pose1.jpg'),
      cv.IMREAD_COLOR,
    );

    int poses = 0;
    for (int i = 0; i < 8; i++) {
      poses = (await core.detectDirect(
        img,
        imageWidth: img.cols,
        imageHeight: img.rows,
      )).length;
    }

    const int iters = 40;
    final List<int> us = [];
    for (int i = 0; i < iters; i++) {
      final sw = Stopwatch()..start();
      await core.detectDirect(img, imageWidth: img.cols, imageHeight: img.rows);
      sw.stop();
      us.add(sw.elapsedMicroseconds);
    }
    us.sort();
    double ms(int u) => u / 1000.0;
    final double mean = us.reduce((a, b) => a + b) / iters;
    // ignore: avoid_print
    print(
      '[pose-e2e] poses=$poses iters=$iters '
      'mean=${ms(mean.round()).toStringAsFixed(2)}ms '
      'median=${ms(us[iters ~/ 2]).toStringAsFixed(2)}ms '
      'p10=${ms(us[iters ~/ 10]).toStringAsFixed(2)}ms '
      'p90=${ms(us[(iters * 9) ~/ 10]).toStringAsFixed(2)}ms',
    );

    img.dispose();
    await core.dispose();
  }, timeout: const Timeout(Duration(minutes: 4)));
}
