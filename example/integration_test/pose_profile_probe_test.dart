// ignore_for_file: avoid_print, implementation_imports

// Controlled compute probe to compare AOT (profile) vs JIT (debug). No camera.
//
// DEBUG/JIT:
//   flutter test integration_test/pose_profile_probe_test.dart -d macos
//
// PROFILE/AOT (flutter test has no --profile; use flutter drive). On Apple
// Silicon the release/profile native-assets build defaults to universal and the
// opencv (dartcv4) x86_64 slice fails to link on recent SDKs, so force arm64:
//   FLUTTER_XCODE_ONLY_ACTIVE_ARCH=YES FLUTTER_XCODE_EXCLUDED_ARCHS=x86_64 \
//     flutter drive --driver=test_driver/integration_test.dart \
//     --target=integration_test/pose_profile_probe_test.dart --profile -d macos
//
// Finding: detect/detectFromMat are within noise debug-vs-profile -- the
// pipeline is native-bound (Metal GPU + opencv), AOT does not speed it up.

import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kProfileMode, kReleaseMode;
import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import 'package:pose_detection/pose_detection.dart';
import 'package:pose_detection/src/models/person_detector_native.dart';
import 'package:pose_detection/src/util/native_image_utils.dart';

const int iterations = 60;
const int warmup = 10;

double _p50(List<int> us) {
  final s = List<int>.from(us)..sort();
  return s[s.length ~/ 2] / 1000.0;
}

Future<double> _bench(Future<void> Function() once) async {
  for (int i = 0; i < warmup; i++) {
    await once();
  }
  final us = <int>[];
  for (int i = 0; i < iterations; i++) {
    final sw = Stopwatch()..start();
    await once();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  return _p50(us);
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  test(
    'compute probe (profile vs debug)',
    () async {
      final mode = kReleaseMode
          ? 'RELEASE'
          : kProfileMode
          ? 'PROFILE/AOT'
          : 'DEBUG/JIT';

      Future<Uint8List> load(String f) async => (await rootBundle.load(
        'packages/pose_detection/assets/models/$f',
      )).buffer.asUint8List();
      final yoloBytes = await load('yolov8n_float32.tflite');

      final mat = cv.imdecode(
        (await rootBundle.load(
          'assets/samples/pose1.jpg',
        )).buffer.asUint8List(),
        cv.IMREAD_COLOR,
      );
      final iw = mat.cols, ih = mat.rows;

      // YOLO detect split (PRE/INF/POST) on the main isolate.
      final yolo = YoloV8PersonDetector();
      await yolo.initializeFromBuffer(yoloBytes, useCompiledModel: true);
      final probe = Interpreter.fromBuffer(yoloBytes)..allocateTensors();
      final os = probe.getOutputTensor(0).shape;
      final ins = probe.getInputTensor(0).shape;
      probe.close();
      final inW = ins[2], inH = ins[1];
      final d1 = os[os.length - 2], d2 = os[os.length - 1];
      final channelMajor = d1 < d2 && (d1 == 84 || d1 == 85);
      final channels = channelMajor ? d1 : d2, anchors = channelMajor ? d2 : d1;

      final (l0, r, dw, dh) = NativeImageUtils.letterbox(mat, inW, inH);
      l0.dispose();

      final rows = <String>[];
      void rep(String k, double ms) =>
          rows.add('${k.padRight(38)} ${ms.toStringAsFixed(3)} ms');

      rep(
        'yolo detect (pre+inf+post)',
        await _bench(
          () => yolo.detect(
            mat,
            imageWidth: iw,
            imageHeight: ih,
            personOnly: true,
          ),
        ),
      );

      // Decode-only on a zero buffer: no candidate passes conf, so this exercises
      // the full 8400x80 per-anchor argmax scan (the dominant, GPU-free scalar
      // cost) in isolation -- the part JIT punishes most vs AOT.
      rep(
        'yolo decode-only (scalar argmax)',
        _benchDecode(
          inW,
          inH,
          iw,
          ih,
          channels,
          anchors,
          channelMajor,
          r,
          dw,
          dh,
        ),
      );

      await yolo.dispose();

      // Full isolate round-trip.
      final det = await PoseDetector.create(
        landmarkModel: PoseLandmarkModel.heavy,
        useCompiledModel: true,
      );
      await det.detectFromMat(mat);
      rep(
        'FULL detectFromMat (isolate round-trip)',
        await _bench(() => det.detectFromMat(mat)),
      );

      // Same detectFromCameraImage code path the live screen uses, but driven in a
      // tight loop with a synthetic BGRA frame (no live camera stream / event-loop
      // flood). If this is ~detectFromMat but the live app is ~3x slower, the gap
      // is the live environment (camera-stream event-loop congestion), not code.
      final frame = _FakeCameraImage.bgra(640, 480);
      await det.detectFromCameraImage(frame, isBgra: true, maxDim: 640);
      rep(
        'detectFromCameraImage (synthetic 640x480 BGRA loop)',
        await _bench(
          () => det.detectFromCameraImage(frame, isBgra: true, maxDim: 640),
        ),
      );
      await det.dispose();

      mat.dispose();
      print('\nCOMPUTE PROBE [$mode] (p50, pose1 ${iw}x$ih, GPU, heavy)');
      for (final r in rows) {
        print(r);
      }
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );
}

// Minimal CameraImage-shaped object for the synthetic detectFromCameraImage
// path. prepareCameraFrameFromImage duck-types width/height/planes(bytes/
// bytesPerRow/bytesPerPixel).
class _FakePlane {
  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
  _FakePlane(this.bytes, this.bytesPerRow, this.bytesPerPixel);
}

class _FakeCameraImage {
  final int width;
  final int height;
  final List<_FakePlane> planes;
  _FakeCameraImage(this.width, this.height, this.planes);

  factory _FakeCameraImage.bgra(int w, int h) {
    final bytes = Uint8List(w * h * 4);
    for (int i = 0; i < bytes.length; i++) {
      bytes[i] = (i * 37) & 0xff; // non-uniform so YOLO has real work
    }
    return _FakeCameraImage(w, h, [_FakePlane(bytes, w * 4, 4)]);
  }
}

double _benchDecode(
  int inW,
  int inH,
  int iw,
  int ih,
  int channels,
  int anchors,
  bool channelMajor,
  double r,
  int dw,
  int dh,
) {
  final flat = Float32List(channels * anchors);
  void run() => postProcessDetectionsFlat(
    flat,
    channels: channels,
    anchors: anchors,
    channelMajor: channelMajor,
    inputWidth: inW,
    inputHeight: inH,
    r: r,
    dw: dw,
    dh: dh,
    imageWidth: iw,
    imageHeight: ih,
    confThres: 0.5,
    iouThres: 0.45,
    maxDet: 10,
    filterClassId: 0,
  );
  for (int i = 0; i < 10; i++) {
    run();
  }
  final us = <int>[];
  for (int i = 0; i < 60; i++) {
    final sw = Stopwatch()..start();
    run();
    sw.stop();
    us.add(sw.elapsedMicroseconds);
  }
  final s = List<int>.from(us)..sort();
  return s[s.length ~/ 2] / 1000.0;
}
