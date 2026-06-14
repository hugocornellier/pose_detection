import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';

/// Native image processing utilities using OpenCV.
///
/// Provides SIMD-accelerated image operations (10-50x faster than pure Dart)
/// for preprocessing images before TensorFlow Lite inference.
class NativeImageUtils {
  /// Rebuilds a tightly packed [cv.Mat] from raw [bytes] with a single copy.
  ///
  /// `cv.Mat.fromList` walks a lazy `cast<int>` view byte-by-byte, which is
  /// slow for camera frames. Allocating the Mat first and copying into its
  /// native buffer keeps reconstruction near memcpy speed. Only use this for
  /// tightly packed, continuous pixel data.
  static cv.Mat matFromPackedBytes(
    int rows,
    int cols,
    cv.MatType type,
    Uint8List bytes,
  ) {
    final cv.Mat mat = cv.Mat.create(rows: rows, cols: cols, type: type);
    final Uint8List dst = mat.data;
    if (dst.length != bytes.length) {
      final int expected = dst.length;
      mat.dispose();
      throw ArgumentError(
        'bytes.length ${bytes.length} does not match a '
        '$rows x $cols Mat of type $type ($expected bytes)',
      );
    }
    dst.setAll(0, bytes);
    return mat;
  }

  /// Rebuilds a [cv.Mat] from a backend-neutral packed image [layout].
  static cv.Mat matFromPackedLayout(
    PackedImageLayout layout,
    Uint8List bytes,
    cv.MatType type,
  ) {
    final cv.Mat mat = cv.Mat.create(
      rows: layout.rows,
      cols: layout.cols,
      type: type,
    );
    try {
      layout.copyTo(mat.data, bytes);
      return mat;
    } catch (_) {
      mat.dispose();
      rethrow;
    }
  }

  /// Decodes a [CameraFrame] into a 3-channel BGR [cv.Mat].
  ///
  /// The layout and safe operation order come from `flutter_litert`; this
  /// method only maps that backend-neutral plan onto OpenCV primitives.
  static cv.Mat cameraFrameToBgrMat(CameraFrame frame, {int? maxDim}) {
    final CameraFrameDecodePlan plan = frame.decodePlan();
    final cv.Mat source = matFromPackedLayout(
      plan.sourceLayout,
      frame.bytes,
      plan.sourceLayout.channels == 4 ? cv.MatType.CV_8UC4 : cv.MatType.CV_8UC1,
    );

    cv.Mat maybeResize(cv.Mat m) {
      if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
      final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
      final cv.Mat resized = cv.resize(m, (
        (m.cols * scale).toInt(),
        (m.rows * scale).toInt(),
      ), interpolation: cv.INTER_LINEAR);
      m.dispose();
      return resized;
    }

    int? rotateFlag() {
      return switch (plan.rotation) {
        CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
        CameraFrameRotation.cw180 => cv.ROTATE_180,
        CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
        null => null,
      };
    }

    cv.Mat maybeRotate(cv.Mat m) {
      final int? flag = rotateFlag();
      if (flag == null) return m;
      final cv.Mat rotated = cv.rotate(m, flag);
      m.dispose();
      return rotated;
    }

    final int cvtCode = switch (plan.conversion) {
      CameraFrameConversion.bgra2bgr => cv.COLOR_BGRA2BGR,
      CameraFrameConversion.rgba2bgr => cv.COLOR_RGBA2BGR,
      CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
      CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
      CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
    };

    switch (plan.order) {
      case CameraFrameDecodeOrder.resizeRotateThenColorConvert:
        cv.Mat current = plan.hasStridePadding
            ? source.region(
                cv.Rect(0, 0, plan.visibleWidth, plan.visibleHeight),
              )
            : source;

        if (maxDim != null &&
            (current.cols > maxDim || current.rows > maxDim)) {
          final double scale =
              maxDim /
              (current.cols > current.rows ? current.cols : current.rows);
          final cv.Mat resized = cv.resize(current, (
            (current.cols * scale).toInt(),
            (current.rows * scale).toInt(),
          ), interpolation: cv.INTER_LINEAR);
          if (!identical(current, source)) current.dispose();
          current = resized;
        }

        final int? flag = rotateFlag();
        if (flag != null) {
          final cv.Mat rotated = cv.rotate(current, flag);
          if (!identical(current, source)) current.dispose();
          current = rotated;
        }

        final cv.Mat bgr = cv.cvtColor(current, cvtCode);
        if (!identical(current, source)) current.dispose();
        source.dispose();
        return bgr;

      case CameraFrameDecodeOrder.colorConvertThenResizeRotate:
        cv.Mat current = cv.cvtColor(source, cvtCode);
        source.dispose();
        current = maybeResize(current);
        current = maybeRotate(current);
        return current;
    }
  }

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
  /// - [size]: Source square side in pixels (the region sampled from [src])
  /// - [theta]: Rotation angle in radians (positive = counter-clockwise)
  /// - [outSize]: Optional output square side in pixels. When provided, the
  ///   crop is resampled straight to [outSize]x[outSize] in a single
  ///   warpAffine (scale folded into the rotation matrix), avoiding a separate
  ///   [cv.resize]. When null, output side equals the rounded [size].
  ///
  /// Returns cropped and rotated cv.Mat. Caller must dispose.
  /// Returns null if size is invalid.
  static cv.Mat? extractAlignedSquare(
    cv.Mat src,
    double cx,
    double cy,
    double size,
    double theta, {
    int? outSize,
  }) {
    final int sizeInt = size.round();
    if (sizeInt <= 0) return null;
    final int outDim = outSize ?? sizeInt;
    if (outDim <= 0) return null;

    final double angleDegrees = -theta * 180.0 / math.pi;

    // Fold the crop->output scale into the rotation matrix so a single
    // warpAffine performs rotate + crop + resize. scale == 1 when outSize is
    // omitted, preserving the original behaviour.
    final double scale = outDim / size;
    final cv.Mat rotMat = cv.getRotationMatrix2D(
      cv.Point2f(cx, cy),
      angleDegrees,
      scale,
    );

    final double outCenter = outDim / 2.0;
    final double tx = rotMat.at<double>(0, 2) + outCenter - cx;
    final double ty = rotMat.at<double>(1, 2) + outCenter - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    final cv.Mat output = cv.warpAffine(
      src,
      rotMat,
      (outDim, outDim),
      borderMode: cv.BORDER_CONSTANT,
      borderValue: cv.Scalar(114, 114, 114, 0),
    );

    rotMat.dispose();
    return output;
  }
}
