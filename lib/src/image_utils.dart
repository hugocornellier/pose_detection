import 'dart:math' as math;
import 'package:image/image.dart' as img;

/// Utility functions for image preprocessing and transformations.
///
/// Provides letterbox preprocessing, coordinate transformations, and tensor
/// conversion utilities used internally by the pose detection pipeline.
class ImageUtils {
  /// Applies letterbox preprocessing to fit an image into target dimensions.
  ///
  /// Scales the source image to fit within [tw]x[th] while maintaining aspect ratio,
  /// then pads with gray (114, 114, 114) to fill the target dimensions.
  ///
  /// This is critical for YOLO-style object detection models that expect fixed input sizes.
  ///
  /// Parameters:
  /// - [src]: Source image to preprocess
  /// - [tw]: Target width in pixels
  /// - [th]: Target height in pixels
  /// - [ratioOut]: Output parameter that receives the scale ratio used
  /// - [dwdhOut]: Output parameter that receives padding [dw, dh] values
  /// - [reuseCanvas]: Optional canvas to reuse (must be exactly [tw]x[th])
  ///
  /// Returns the letterboxed image with dimensions [tw]x[th].
  static img.Image letterbox(
    img.Image src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut, {
    img.Image? reuseCanvas,
  }) {
    final int w = src.width;
    final int h = src.height;
    final double r = math.min(th / h, tw / w);
    final int nw = (w * r).round();
    final int nh = (h * r).round();
    final int dw = (tw - nw) ~/ 2;
    final int dh = (th - nh) ~/ 2;

    final img.Image resized = img.copyResize(
      src,
      width: nw,
      height: nh,
      interpolation: img.Interpolation.linear,
    );
    final img.Image canvas = reuseCanvas ?? img.Image(width: tw, height: th);
    if (canvas.width != tw || canvas.height != th) {
      final String reuseDims = '${canvas.width}x${canvas.height}';
      final String targetDims = '${tw}x$th';
      throw ArgumentError(
        'Reuse canvas dimensions ($reuseDims) '
        'do not match target dimensions ($targetDims)',
      );
    }
    img.fill(canvas, color: img.ColorRgb8(114, 114, 114));
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    ratioOut
      ..clear()
      ..add(r);
    dwdhOut
      ..clear()
      ..addAll([dw, dh]);
    return canvas;
  }

  /// Applies letterbox preprocessing to 256x256 dimensions.
  ///
  /// Convenience method that calls [letterbox] with fixed 256x256 target size.
  /// Used for BlazePose landmark model preprocessing.
  ///
  /// Parameters:
  /// - [src]: Source image to preprocess
  /// - [ratioOut]: Output parameter that receives the scale ratio used
  /// - [dwdhOut]: Output parameter that receives padding [dw, dh] values
  /// - [reuseCanvas]: Optional 256x256 canvas to reuse
  ///
  /// Returns the letterboxed image with dimensions 256x256.
  static img.Image letterbox256(
    img.Image src,
    List<double> ratioOut,
    List<int> dwdhOut, {
    img.Image? reuseCanvas,
  }) {
    return letterbox(src, 256, 256, ratioOut, dwdhOut,
        reuseCanvas: reuseCanvas);
  }

  /// Transforms bounding box coordinates from letterbox space back to original image space.
  ///
  /// Reverses the letterbox transformation by removing padding and unscaling coordinates.
  ///
  /// Parameters:
  /// - [xyxy]: Bounding box in letterbox space as [x1, y1, x2, y2]
  /// - [ratio]: Scale ratio from letterbox preprocessing
  /// - [dw]: Horizontal padding from letterbox preprocessing
  /// - [dh]: Vertical padding from letterbox preprocessing
  ///
  /// Returns the bounding box in original image space as [x1, y1, x2, y2].
  static List<double> scaleFromLetterbox(
    List<double> xyxy,
    double ratio,
    int dw,
    int dh,
  ) {
    final double x1 = (xyxy[0] - dw) / ratio;
    final double y1 = (xyxy[1] - dh) / ratio;
    final double x2 = (xyxy[2] - dw) / ratio;
    final double y2 = (xyxy[3] - dh) / ratio;
    return [x1, y1, x2, y2];
  }

  /// Converts an image to a 4D tensor in NHWC format for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// The output format is [batch, height, width, channels] where batch=1 and channels=3 (RGB).
  ///
  /// Parameters:
  /// - [image]: Source image to convert
  /// - [width]: Target width (must match image width)
  /// - [height]: Target height (must match image height)
  /// - [reuse]: Optional tensor buffer to reuse (must match dimensions)
  ///
  /// Returns a 4D tensor [1, height, width, 3] with normalized pixel values.
  static List<List<List<List<double>>>> imageToNHWC4D(
    img.Image image,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final List<List<List<List<double>>>> out = reuse ??
        List.generate(
          1,
          (_) => List.generate(
            height,
            (_) => List.generate(
              width,
              (_) => List<double>.filled(3, 0.0),
              growable: false,
            ),
            growable: false,
          ),
          growable: false,
        );

    // Direct buffer access is ~10-50x faster than getPixel() which creates
    // a new Pixel object and performs bounds checking on every call.
    final bytes = image.buffer.asUint8List();
    final int numChannels = image.numChannels;
    const double scale = 1.0 / 255.0;
    int byteIndex = 0;
    for (int y = 0; y < height; y++) {
      final List<List<double>> row = out[0][y];
      for (int x = 0; x < width; x++) {
        final List<double> pixel = row[x];
        pixel[0] = bytes[byteIndex] * scale;
        pixel[1] = bytes[byteIndex + 1] * scale;
        pixel[2] = bytes[byteIndex + 2] * scale;
        byteIndex += numChannels;
      }
    }
    return out;
  }

  /// Reshapes a flat array into a 4D tensor.
  ///
  /// Converts a 1D flattened array into a 4D nested list structure with the
  /// specified dimensions. Used for converting TensorFlow Lite output buffers
  /// into the expected tensor shape.
  ///
  /// Parameters:
  /// - [flat]: Flat array of values (length must equal dim1 * dim2 * dim3 * dim4)
  /// - [dim1]: First dimension size (batch)
  /// - [dim2]: Second dimension size (height/rows)
  /// - [dim3]: Third dimension size (width/columns)
  /// - [dim4]: Fourth dimension size (channels/features)
  ///
  /// Returns a 4D tensor [dim1, dim2, dim3, dim4] populated from [flat] in row-major order.
  static List<List<List<List<double>>>> reshapeToTensor4D(
    List<double> flat,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
  ) {
    final List<List<List<List<double>>>> result = List.generate(
      dim1,
      (_) => List.generate(
        dim2,
        (_) => List.generate(
          dim3,
          (_) => List<double>.filled(dim4, 0.0),
        ),
      ),
    );

    int index = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          for (int l = 0; l < dim4; l++) {
            result[i][j][k][l] = flat[index++];
          }
        }
      }
    }

    return result;
  }
}
