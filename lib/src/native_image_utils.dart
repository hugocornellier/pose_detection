import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv_dart.dart' as cv;

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
    final int w = src.cols;
    final int h = src.rows;

    // Calculate scale to fit within target while preserving aspect ratio
    final double r = (tw / w) < (th / h) ? (tw / w) : (th / h);
    final int nw = (w * r).round();
    final int nh = (h * r).round();

    // NATIVE: SIMD-accelerated resize
    final cv.Mat resized =
        cv.resize(src, (nw, nh), interpolation: cv.INTER_LINEAR);

    // Calculate padding
    final int dw = (tw - nw) ~/ 2;
    final int dh = (th - nh) ~/ 2;
    final int padRight = tw - nw - dw;
    final int padBottom = th - nh - dh;

    // NATIVE: Efficient border padding with gray (114, 114, 114)
    final cv.Mat padded = cv.copyMakeBorder(
      resized,
      dh,
      padBottom,
      dw,
      padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    resized.dispose();

    return (padded, r, dw, dh);
  }

  /// Applies letterbox preprocessing to 256x256 dimensions.
  ///
  /// Convenience method for BlazePose landmark model preprocessing.
  static (cv.Mat, double, int, int) letterbox256(cv.Mat src) {
    return letterbox(src, 256, 256);
  }

  /// Resizes a cv.Mat directly to 256x256 without letterboxing.
  ///
  /// This is used for BlazePose landmark model which expects the person
  /// to fill the entire 256x256 input (no padding).
  ///
  /// Parameters:
  /// - [src]: Source cv.Mat image (any size)
  ///
  /// Returns a tuple of (resized Mat, scaleX, scaleY) where scales are
  /// the ratios from original to 256x256.
  /// Caller must dispose the returned Mat.
  static (cv.Mat, double, double) resize256(cv.Mat src) {
    final int w = src.cols;
    final int h = src.rows;
    final double scaleX = 256.0 / w;
    final double scaleY = 256.0 / h;
    final cv.Mat resized =
        cv.resize(src, (256, 256), interpolation: cv.INTER_LINEAR);
    return (resized, scaleX, scaleY);
  }

  /// Applies letterbox preprocessing to 640x640 dimensions.
  ///
  /// Convenience method for YOLO detection model preprocessing.
  static (cv.Mat, double, int, int) letterbox640(cv.Mat src) {
    return letterbox(src, 640, 640);
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
    final int h = mat.rows;
    final int w = mat.cols;
    final int totalPixels = h * w;

    final Float32List tensor = buffer ?? Float32List(totalPixels * 3);
    final Uint8List data = mat.data;

    // BGR to RGB conversion + normalize to [0, 1]
    const double scale = 1.0 / 255.0;
    for (int i = 0, j = 0; i < totalPixels * 3; i += 3, j += 3) {
      tensor[j] = data[i + 2] * scale; // B -> R
      tensor[j + 1] = data[i + 1] * scale; // G -> G
      tensor[j + 2] = data[i] * scale; // R -> B
    }

    return tensor;
  }

  /// Crops a rectangular region from a cv.Mat.
  ///
  /// Parameters:
  /// - [src]: Source cv.Mat image
  /// - [x]: Left coordinate in pixels
  /// - [y]: Top coordinate in pixels
  /// - [width]: Width of crop in pixels
  /// - [height]: Height of crop in pixels
  ///
  /// Returns cropped cv.Mat. Caller must dispose.
  static cv.Mat crop(cv.Mat src, int x, int y, int width, int height) {
    // Clamp to valid bounds
    final int x1 = x.clamp(0, src.cols - 1);
    final int y1 = y.clamp(0, src.rows - 1);
    final int x2 = (x + width).clamp(x1 + 1, src.cols);
    final int y2 = (y + height).clamp(y1 + 1, src.rows);

    final cv.Rect rect = cv.Rect(x1, y1, x2 - x1, y2 - y1);
    return src.region(rect);
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

    // Convert radians to degrees, negate for correct direction
    final double angleDegrees = -theta * 180.0 / 3.141592653589793;

    // Get rotation matrix centered at the crop center
    final cv.Mat rotMat = cv.getRotationMatrix2D(
      cv.Point2f(cx, cy),
      angleDegrees,
      1.0,
    );

    // Adjust translation to crop around the output center
    final double outCenter = sizeInt / 2.0;
    final double tx = rotMat.at<double>(0, 2) + outCenter - cx;
    final double ty = rotMat.at<double>(1, 2) + outCenter - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    // NATIVE: SIMD-optimized warpAffine (10-50x faster)
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

  /// Decodes image bytes to cv.Mat.
  ///
  /// Uses native OpenCV imdecode for SIMD-accelerated decoding.
  ///
  /// Parameters:
  /// - [bytes]: Encoded image data (JPEG, PNG, etc.)
  ///
  /// Returns decoded cv.Mat in BGR format, or null if decoding fails.
  static cv.Mat? decodeImage(Uint8List bytes) {
    try {
      return cv.imdecode(bytes, cv.IMREAD_COLOR);
    } catch (_) {
      return null;
    }
  }

  /// Converts an img.Image to cv.Mat.
  ///
  /// Parameters:
  /// - [image]: Source image from the `image` package
  ///
  /// Returns cv.Mat in BGR format. Caller must dispose.
  @Deprecated(
      'Will be removed in 2.0.0. Use cv.imdecode or pass cv.Mat directly.')
  static cv.Mat imageToMat(img.Image image) {
    final int w = image.width;
    final int h = image.height;

    // Create BGR buffer directly from image data
    final Uint8List bgr = Uint8List(w * h * 3);
    final imageBytes = image.buffer.asUint8List();
    final int numChannels = image.numChannels;

    int bgrIdx = 0;
    int srcIdx = 0;
    for (int i = 0; i < w * h; i++) {
      // RGBA to BGR conversion: source is RGBA, output is BGR
      bgr[bgrIdx] = imageBytes[srcIdx + 2]; // B from source offset 2
      bgr[bgrIdx + 1] = imageBytes[srcIdx + 1]; // G from source offset 1
      bgr[bgrIdx + 2] = imageBytes[srcIdx]; // R from source offset 0
      bgrIdx += 3;
      srcIdx += numChannels;
    }

    return cv.Mat.fromList(h, w, cv.MatType.CV_8UC3, bgr);
  }
}
