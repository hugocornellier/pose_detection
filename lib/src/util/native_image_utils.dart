import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';

/// Native image processing utilities using OpenCV.
///
/// Provides SIMD-accelerated image operations (10-50x faster than pure Dart)
/// for preprocessing images before TensorFlow Lite inference.
class NativeImageUtils {
  /// Applies letterbox preprocessing to fit a cv.Mat into target dimensions.
  ///
  /// Uses native OpenCV resize and copyMakeBorder for SIMD acceleration.
  /// Scales the source image to fit within [tw]x[th] while maintaining aspect ratio,
  /// then pads with gray (114, 114, 114) to fill the target dimensions.
  ///
  /// Parameters:
  /// - [src]: Source cv.Mat image (BGR format)
  /// - [tw]: Target width in pixels
  /// - [th]: Target height in pixels
  ///
  /// Returns a tuple of (letterboxed Mat, scale ratio, padLeft, padTop).
  /// Caller must dispose the returned Mat.
  static (cv.Mat, double, int, int) letterbox(cv.Mat src, int tw, int th) {
    final p = computeLetterboxParams(
      srcWidth: src.cols,
      srcHeight: src.rows,
      targetWidth: tw,
      targetHeight: th,
    );

    final cv.Mat resized = cv.resize(src, (
      p.newWidth,
      p.newHeight,
    ), interpolation: cv.INTER_LINEAR);

    final cv.Mat padded = cv.copyMakeBorder(
      resized,
      p.padTop,
      p.padBottom,
      p.padLeft,
      p.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    resized.dispose();

    return (padded, p.scale, p.padLeft, p.padTop);
  }

  /// Converts a letterboxed cv.Mat to a normalized Float32List tensor.
  ///
  /// Normalizes pixel values to [0.0, 1.0] range for YOLO models.
  /// Handles BGR to RGB conversion.
  ///
  /// Parameters:
  /// - [mat]: Letterboxed cv.Mat image (BGR format)
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns Float32List tensor in NHWC format (flattened).
  static Float32List matToTensorYolo(cv.Mat mat, {Float32List? buffer}) {
    return bgrBytesToRgbFloat32(
      bytes: mat.data,
      totalPixels: mat.rows * mat.cols,
      buffer: buffer,
    );
  }

  /// Extracts a rotated square region from a cv.Mat using warpAffine.
  ///
  /// Uses SIMD-accelerated warpAffine which is 10-50x faster than pure Dart
  /// bilinear interpolation.
  ///
  /// Parameters:
  /// - [src]: Source cv.Mat image
  /// - [cx]: Center X coordinate in pixels
  /// - [cy]: Center Y coordinate in pixels
  /// - [size]: Output square size in pixels
  /// - [theta]: Rotation angle in radians (positive = counter-clockwise)
  ///
  /// Returns cropped and rotated cv.Mat. Caller must dispose.
  /// Returns null if size is invalid.
  static cv.Mat? extractAlignedSquare(
    cv.Mat src,
    double cx,
    double cy,
    double size,
    double theta,
  ) {
    final int sizeInt = size.round();
    if (sizeInt <= 0) return null;

    final double angleDegrees = -theta * 180.0 / math.pi;

    final cv.Mat rotMat = cv.getRotationMatrix2D(
      cv.Point2f(cx, cy),
      angleDegrees,
      1.0,
    );

    final double outCenter = sizeInt / 2.0;
    final double tx = rotMat.at<double>(0, 2) + outCenter - cx;
    final double ty = rotMat.at<double>(1, 2) + outCenter - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    final cv.Mat output = cv.warpAffine(
      src,
      rotMat,
      (sizeInt, sizeInt),
      borderMode: cv.BORDER_CONSTANT,
      borderValue: cv.Scalar(114, 114, 114, 0),
    );

    rotMat.dispose();
    return output;
  }
}
