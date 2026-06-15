import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart' show bgrBytesToRgbFloat32;
import 'package:pose_detection/src/util/native_image_utils.dart';

/// Manual A/B benchmark (run explicitly): pure-Dart vs SIMD BGR->Float32 tensor
/// conversion at the two pose hot-path sizes. Verifies output equivalence too.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  void bench(int size) {
    final cv.Mat mat = cv.Mat.zeros(size, size, cv.MatType.CV_8UC3);
    // Fill with varied data so the equivalence check and timing are realistic.
    final Uint8List d = mat.data;
    for (int i = 0; i < d.length; i++) {
      d[i] = (i * 37 + 11) & 0xFF;
    }
    final int tp = size * size;
    final Float32List bufA = Float32List(tp * 3);
    final Float32List bufB = Float32List(tp * 3);

    final a = bgrBytesToRgbFloat32(
      bytes: mat.data,
      totalPixels: tp,
      buffer: bufA,
    );
    final b = NativeImageUtils.matToRgbFloat32Simd(mat, buffer: bufB);
    double maxDiff = 0;
    for (int i = 0; i < a.length; i++) {
      final double dd = (a[i] - b[i]).abs();
      if (dd > maxDiff) maxDiff = dd;
    }

    for (int i = 0; i < 15; i++) {
      bgrBytesToRgbFloat32(bytes: mat.data, totalPixels: tp, buffer: bufA);
      NativeImageUtils.matToRgbFloat32Simd(mat, buffer: bufB);
    }
    const int n = 200;
    final swP = Stopwatch()..start();
    for (int i = 0; i < n; i++) {
      bgrBytesToRgbFloat32(bytes: mat.data, totalPixels: tp, buffer: bufA);
    }
    swP.stop();
    final swS = Stopwatch()..start();
    for (int i = 0; i < n; i++) {
      NativeImageUtils.matToRgbFloat32Simd(mat, buffer: bufB);
    }
    swS.stop();

    final double pureUs = swP.elapsedMicroseconds / n;
    final double simdUs = swS.elapsedMicroseconds / n;
    // ignore: avoid_print
    print(
      '[pose-simd] ${size}x$size  pureDart=${pureUs.toStringAsFixed(1)}us  '
      'simd=${simdUs.toStringAsFixed(1)}us  '
      'speedup=${(pureUs / simdUs).toStringAsFixed(2)}x  maxDiff=$maxDiff',
    );
    mat.dispose();
  }

  test('pose BGR->Float32 conversion: SIMD vs pure-Dart', () {
    bench(640); // YOLO person-detector input
    bench(256); // landmark input
  });
}
